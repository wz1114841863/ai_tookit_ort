import cv2 as cv
import numpy as np
import onnxruntime as ort

from enum import Enum
from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import (
    BBox,
    YOLOV5_CLASSES,
    hard_nms,
    blending_nms,
    offset_nms,
    draw_boxes,
)


class Yolov5ScaleParams:
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


class Yolov5(BasicOrtHandler):
    """Yolov5: https://github.com/ultralytics/yolov5"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean = 0.0
        self.scale_value = 1.0 / 255.0
        self.max_nms = 30000
        self.classes = YOLOV5_CLASSES

    def transform(self, mat):
        mat = cv.cvtColor(mat, cv.COLOR_BGR2RGB)
        mat = normalize(mat, self.mean, self.scale_value)
        input_tensor = create_tensor(mat, self.input_node_dims, DataFormat.CHW)
        return input_tensor

    def resize_unscale(self, mat, target_height, target_width):
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

        sacle_params = Yolov5ScaleParams(r, dw, dh, new_unpad_w, new_unpad_h, True)
        return mat_rs, sacle_params

    def generate_bboxes(
        self,
        scale_params,
        output_tensors,
        score_threshold,
        img_height,
        img_width,
    ):
        pred = output_tensors[0]
        # print(f"type of pred: {type(pred)}, shape of pred: {pred.shape}")
        pred_dims = self.output_node_dims[0]
        num_anchores = pred_dims[1]
        num_classes = pred_dims[2] - 5

        r_ = scale_params.r
        dw_ = scale_params.dw
        dh_ = scale_params.dh

        bbox_collection = []
        count = 0
        for i in range(num_anchores):
            obj_conf = pred[0, i, 4]
            if obj_conf < score_threshold:
                continue

            cls_conf = pred[0, i, 5]
            label = 0
            for j in range(num_classes):
                tmp_conf = pred[0, i, j + 5]
                if tmp_conf > cls_conf:
                    cls_conf = tmp_conf
                    label = j
            conf = obj_conf * cls_conf
            if conf < score_threshold:
                continue

            dx = pred[0, i, 0]
            dy = pred[0, i, 1]
            dw = pred[0, i, 2]
            dh = pred[0, i, 3]

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
            bbox.score = conf
            bbox.label = label
            bbox.label_txt = YOLOV5_CLASSES[label]
            bbox.flag = True
            bbox_collection.append(bbox)

            count += 1
            if count > self.max_nms:
                break

        return bbox_collection

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
        bbox_collection = self.generate_bboxes(
            scale_params, output_tensors, score_threashold, img_height, img_width
        )
        detected_boxes = self.nms(bbox_collection, iou_threshold, topk, nms_type)
        return detected_boxes


if __name__ == "__main__":
    onnx_file = "lite/hub/ort/yolov5s.onnx"
    img_path = "resources/test_lite_yolox_1.jpg"
    yolov5 = Yolov5(onnx_file)
    img = cv.imread(img_path)
    results = yolov5.detect(img)
    draw_boxes(img, results)
    cv.imwrite("./test.jpg", img)
