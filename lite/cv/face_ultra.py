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


class NMS(Enum):
    HARD = 0
    BLEND = 1
    OFFSET = 2


class UltraFace(BasicOrtHandler):
    """Ultra Face"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean_val = 127.0
        self.scale_val = 1.0 / 128.0
        self.max_nms = 30000

    def transform(self, mat):
        """将图像转为模型输入"""
        canvas = cv.resize(mat, (self.input_node_dims[3], self.input_node_dims[2]))
        canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
        canvas = normalize(canvas, self.mean_val, self.scale_val)
        input_tensor = create_tensor(canvas, self.input_node_dims, DataFormat.CHW)
        return input_tensor

    def generate_bboxes(self, output_tensors, score_threshold, img_height, img_width):
        scores = output_tensors[0]  # (1, 4420, 2)
        bboxes = output_tensors[1]  # (1, 4420, 4)
        num_anchors = scores.shape[1]
        bbox_collection = []
        count = 0
        for i in range(num_anchors):
            score = scores[0, i, 1]
            if score < score_threshold:
                continue
            x1 = bboxes[0, i, 0] * img_width
            y1 = bboxes[0, i, 1] * img_height
            x2 = bboxes[0, i, 2] * img_width
            y2 = bboxes[0, i, 3] * img_height
            bbox = BBox(x1, y1, x2, y2)
            bbox.score = score
            bbox.flag = True
            bbox.label = 1
            bbox.label_txt = "face"
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
        score_threashold=0.5,
        iou_threshold=0.5,
        topk=100,
        nms_type=NMS.OFFSET,
    ):
        if mat.size == 0:
            return

        img_height, img_width = mat.shape[:2]
        input_tensor = self.transform(mat)
        output_tensors = self.ort_session.run(
            self.output_node_names,
            {self.input_name: input_tensor},
        )

        bbox_collection = self.generate_bboxes(
            output_tensors, score_threashold, img_height, img_width
        )
        detected_boxes = self.nms(bbox_collection, iou_threshold, topk, nms_type)
        return detected_boxes


if __name__ == "__main__":
    onnx_file = "lite/hub/ort/ultraface-slim-320.onnx"
    img_path = "resources/test_lite_face_detector_3.jpg"
    ultraface = UltraFace(onnx_file)
    img = cv.imread(img_path)
    results = ultraface.detect(img)
    draw_boxes(img, results)
    cv.imwrite("./test.jpg", img)
