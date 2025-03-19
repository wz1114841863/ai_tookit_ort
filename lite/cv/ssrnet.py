import cv2 as cv
import numpy as np
import onnxruntime as ort

from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import Age, draw_age


class SSRNet(BasicOrtHandler):
    """https://github.com/oukohou/SSR_Net_Pytorch"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean_val = [0.485, 0.456, 0.406]
        self.scale_val = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]

    def transform(self, mat):
        """将图像转为模型输入"""
        canvas = cv.resize(mat, (self.input_node_dims[3], self.input_node_dims[2]))
        canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
        canvas = normalize(canvas, self.mean_val, self.scale_val)
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

        age_tensor = output_tensors[0]
        pred_age = age_tensor[0]
        interval_min = int(pred_age if pred_age - 2.0 > 0.0 else 0.0)
        interval_max = int(pred_age if pred_age + 3.0 < 100.0 else 100.0)

        age = Age()
        age.age = pred_age
        age.age_interval = [interval_min, interval_max]
        age.interval_prob = 1.0
        age.flag = True
        return age


if __name__ == "__main__":
    onnx_path = "./lite/hub/ort/ssrnet.onnx"
    img_path = "./resources/test_lite_ssrnet.jpg"

    net = SSRNet(onnx_path)
    img = cv.imread(img_path)
    age = net.detect(img)
    draw_age(img, age)
    cv.imwrite("./test.jpg", img)
