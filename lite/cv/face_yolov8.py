import cv2 as cv
import numpy as np
import onnxruntime as ort

from enum import Enum
from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import (
    BBox,
    hard_nms,
    blending_nms,
    offset_nms,
    draw_boxes,
)


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

    def geterate_box(self, outputs, conf_threshold, iou_threshold):
        pass

    def detect(
        self,
        mat,
        score_threashold=0.25,
        iou_threshold=0.45,
        nms_type=NMS.OFFSET,
    ):
        if mat.size == 0:
            return
        input_tensor = self.transform(mat)
        output_tensors = self.ort_session.run(
            self.output_node_names,
            {self.input_name: input_tensor},
        )
        bbox_collection = self.generate_bboxes(
            scale_params, output_tensors, score_threashold, img_height, img_width
        )
        detected_boxes = self.nms(bbox_collection, iou_threshold, nms_type)
        return detected_boxes
