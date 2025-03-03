import cv2 as cv
import numpy as np
import onnxruntime as ort

from enum import Enum
from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import ClassificationContent, softmax, YOLOX_CLASSES


class YoloXAnchor:
    def __init__(self, grid0, grid1, stride):
        self.grid0 = grid0
        self.grid1 = grid1
        self.stride = stride


class YoloXScaleParams:
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


class YoloX(BasicOrtHandler):
    """YoloX:"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)

        self.mean = [255.0 * 0.485, 255.0 * 0.456, 255.0 * 0.406]
        self.scale_value = [
            1 / (255.0 * 0.229),
            1 / (255.0 * 0.224),
            1 / (255.0 * 0.225),
        ]
        self.max_nms = 30000

    def transform(self, mat):
        mat = cv.cvtColor(mat, cv.COLOR_BGR2RGB)
        mat = normalize(mat, self.mean, self.scale_value)
        input_tensor = create_tensor(mat, self.input_node_dims, DataFormat.CHW)
        return input_tensor

    def resize_unscale(self, mat, target_height, target_width):
        """等比例缩放图像,使用114填充其余四周"""
        if mat.empty():
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

        sacle_params = YoloXScaleParams(r, dw, dh, new_unpad_w, new_unpad_h, True)
        return mat_rs, sacle_params

    def generate_anchors(self, target_height, target_width, strides, anchors):
        pass

    def generate_bboxes(
        self,
        scale_params,
        output_tensors,
        score_threshold,
        img_height,
        img_width,
    ):
        pass

    def nms(data_in, data_output, iou_threshold, topk, nms_type):
        pass

    def detect(
        self,
        mat,
        score_threashold=0.25,
        iou_threshold=0.45,
        topk=100,
        nms_type=NMS.OFFSET,
    ):
        if mat.empty():
            return
        input_height = self.input_node_dims[2]
        input_width = self.input_node_dims[3]
        img_height, img_width = mat.shape[:2]
        mat_rs, scale_params = self.resize_unscale(mat, input_height, input_width)

        input_tensor = self.transform(mat_rs)
        output_tensors = self.ort_sessionrun(
            self.output_node_names,
            {self.input_name: input_tensor},
        )
        bbox_collection = self.generate_bboxes(
            scale_params, output_tensors, score_threashold, img_height, img_width
        )
        detected_boxes = self.nms(bbox_collection, iou_threshold, topk, nms_type)
        return bbox_collection
