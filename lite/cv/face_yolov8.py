import cv2 as cv
import numpy as np
import onnxruntime as ort

from enum import Enum
from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import (
    BBox,
    Keypoint,
    hard_nms,
    blending_nms,
    offset_nms,
    draw_boxes,
    draw_keypoint,
)


class Yolov8FaceScaleParams:
    def __init__(self, r, dw, dh, new_unpad_w, new_unpad_h, flag):
        self.r = r
        self.dw = dw
        self.dh = dh
        self.new_unpad_w = new_unpad_w
        self.new_unpad_h = new_unpad_h
        self.flag = flag


class NMS(Enum):
    HARD = 0
    BLEND = 1
    OFFSET = 2


class FaceYolov8(BasicOrtHandler):
    """Yolov8 face detection: https://github.com/derronqi/yolov8-face"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean = -127.5 / 128.0
        self.scale_value = 1 / 128.0
        self.ratio_width = 1.0
        self.ratio_height = 1.0

    def transform(self, mat):
        mat = cv.cvtColor(mat, cv.COLOR_BGR2RGB)
        mat = normalize(mat, self.mean, self.scale_value)
        input_tensor = create_tensor(mat, self.input_node_dims, DataFormat.CHW)
        return input_tensor

    def resize_unscale(self, mat, target_height=640, target_width=640):
        """等比例缩放图像,使用114填充其余四周"""
        if mat.size == 0:
            return
        img_height, img_width = mat.shape[:2]
        mat_rs = np.full((target_height, target_width, 3), 114, dtype=np.uint8)

        w_r = target_width / img_width
        h_r = target_height / img_height
        r = min(w_r, h_r)

        new_unpad_w = int(img_width * r)
        new_unpad_h = int(img_height * r)

        pad_w = target_width - new_unpad_w
        pad_h = target_height - new_unpad_h

        dw = pad_w // 2
        dh = pad_h // 2

        new_unpad_mat = cv.resize(mat, (new_unpad_w, new_unpad_h))
        mat_rs[dh : dh + new_unpad_h, dw : dw + new_unpad_w] = new_unpad_mat

        sacle_params = Yolov8FaceScaleParams(r, dw, dh, new_unpad_w, new_unpad_h, True)
        return mat_rs, sacle_params

    def nms(self, input_bboxs, iou_threshold, topk, nms_type):
        if nms_type == NMS.BLEND:
            output_bboxs = blending_nms(input_bboxs, iou_threshold, topk)
        elif nms_type == NMS.HARD:
            output_bboxs = hard_nms(input_bboxs, iou_threshold, topk)
        elif nms_type == NMS.OFFSET:
            output_bboxs = offset_nms(input_bboxs, iou_threshold, topk)
        else:
            raise ValueError(f"未实现的NMS方法: {nms_type}")
        return output_bboxs

    def generate_bboxes(
        self, scale_params, output_tensors, score_threashold, img_height, img_width
    ):
        pred = output_tensors[0]
        num_boxs = self.output_node_dims[0][2]

        r_ = scale_params.r
        dw_ = scale_params.dw
        dh_ = scale_params.dh

        bbox_collection = []
        keypoints = []
        for i in range(num_boxs):
            score = pred[0, 4, i]
            if score < score_threashold:
                continue

            dx = pred[0, 0, i]
            dy = pred[0, 1, i]
            dw = pred[0, 2, i]
            dh = pred[0, 3, i]

            cx = dx
            cy = dy
            w = dw
            h = dh
            x1 = ((cx - w / 2.0) - float(dw_)) / r_
            y1 = ((cy - h / 2.0) - float(dh_)) / r_
            x2 = ((cx + w / 2.0) - float(dw_)) / r_
            y2 = ((cy + h / 2.0) - float(dh_)) / r_

            bbox = BBox(
                max(0.0, x1),
                max(0.0, y1),
                min(x2, img_width - 1.0),
                min(y2, img_height - 1),
            )
            bbox.score = score
            bbox.label = 0
            bbox.label_txt = "face"
            bbox.flag = True
            bbox_collection.append(bbox)

            face_keypoint = []
            for j in range(5):
                point_x = pred[0, 4 + j * 3, i]
                point_y = pred[0, 5 + j * 3, i]
                score = pred[0, 6 + j * 3, i]
                point_x = (point_x - float(dw_)) / r_
                point_y = (point_y - float(dw_)) / r_
                point = Keypoint(point_x, point_y, score, True)
                face_keypoint.append(point)
            keypoints.append(face_keypoint)

        return bbox_collection, keypoints

    def detect(
        self,
        mat,
        score_threashold=0.25,
        iou_threshold=0.45,
        topk=100,
        nms_type=NMS.OFFSET,
    ):
        if mat.size == 0:
            return
        input_height = self.input_node_dims[2]
        input_width = self.input_node_dims[3]
        img_height, img_width = mat.shape[:2]
        mat_rs, scale_params = self.resize_unscale(mat, input_height, input_width)
        input_tensor = self.transform(mat_rs)
        output_tensors = self.ort_session.run(
            self.output_node_names,
            {self.input_name: input_tensor},
        )
        bbox_collection, keypoints = self.generate_bboxes(
            scale_params, output_tensors, score_threashold, img_height, img_width
        )
        # TODO: 这里的keypoints也需要随着bboxs做NMS来删减
        detected_boxes = self.nms(bbox_collection, iou_threshold, topk, nms_type)
        return detected_boxes


if __name__ == "__main__":
    onnx_file = "lite/hub/ort/yoloface_8n.onnx"
    img_path = "resources/test_lite_face_detector_3.jpg"
    yolov8 = FaceYolov8(onnx_file)
    img = cv.imread(img_path)
    results = yolov8.detect(img)
    draw_boxes(img, results)
    cv.imwrite("./test.jpg", img)
