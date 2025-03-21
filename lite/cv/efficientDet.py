import math
import cv2 as cv
import numpy as np
import onnxruntime as ort

from enum import Enum
from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import Gender, draw_gender, softmax, EFFICIENTDET_CLASSES


class EfficientDetAnchor:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __repr__(self):
        return f"EfficientDetAnchor(x={self.x}, y={self.y}, w={self.w}, h={self.h})"


class NMS(Enum):
    HARD = 0
    BLEND = 1
    OFFSET = 2


class GlintArcFace(BasicOrtHandler):
    """"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean_val = [0.406, 0.456, 0.486]
        self.scale_val = [1.0 / 0.225, 1.0 / 0.224, 1.0 / 0.229]
        self.class_names = EFFICIENTDET_CLASSES
        self.max_nms = 30000
        self.anchor_scale = 4.0
        self.anchors_buffer = []
        self.pyramid_levels = [3, 4, 5, 6, 7]

        # 定义strides
        self.strides = [
            math.pow(2.0, 3.0),
            math.pow(2.0, 4.0),
            math.pow(2.0, 5.0),
            math.pow(2.0, 6.0),
            math.pow(2.0, 7.0),
        ]

        # 定义scales
        self.scales = [
            math.pow(2.0, 0.0),
            math.pow(2.0, 1.0 / 3.0),
            math.pow(2.0, 2.0 / 3.0),
        ]

        # 定义ratios
        self.ratios = [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]]

    def transform(self, mat):
        """将图像转为模型输入"""
        canvas = cv.resize(mat, (self.input_node_dims[3], self.input_node_dims[2]))
        canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
        input_tensor = create_tensor(canvas, self.input_node_dims, DataFormat.CHW)
        return input_tensor

    def detect(self, mat):
        if mat.size == 0:
            print("Warning: input is empty.")
            return
        input_tensor = self.transform(mat)
        output_tensors = self.ort_session.run(
            self.output_node_names,
            {self.input_name: input_tensor},
        )

        pred_tensor = output_tensors[0]
        pred_dims = self.output_node_dims[0]
        rows, cols = pred_dims[2], pred_dims[3]


if __name__ == "__main__":
    onnx_file = "lite/hub/ort/vgg16_gender.onnx"
    img_path = "resources/test_lite_age_googlenet.jpg"

    model = GlintArcFace(onnx_file)
    img = cv.imread(img_path)
    results = model.detect(img)
    draw_gender(img, results)
    cv.imwrite("./test.jpg", img)
