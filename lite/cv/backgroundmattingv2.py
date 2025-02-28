import cv2 as cv
import numpy as np
import onnxruntime as ort

from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import Age, softmax


class BackgroundMattingV2(BasicOrtHandler):
    """ """

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean_vale = 0.0
        self.scale_val = 1.0 / 255.0

    def transform(self, mat):
        return super().transform(mat)

    def generate_matting(self):
        pass
