import cv2 as cv
import numpy as np
import onnxruntime as ort

from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import Age, softmax


class AgeGoogleNet(BasicOrtHandler):
    """年龄推断网络 https://github.com/onnx/models/tree/main/validated/vision/body_analysis/age_gender/models"""

    def __init__(self, onnx_path: str, num_threads: int = 1, print_info: bool = None):
        super().__init__(onnx_path, num_threads)
        self.mean_val = np.array([104.0, 117.0, 123.0], dtype=np.float32)
        self.scale_val = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.age_intervals = [
            (0, 2),
            (4, 6),
            (8, 12),
            (15, 20),
            (25, 32),
            (38, 43),
            (48, 53),
            (60, 100),
        ]
        if print_info:
            self.print_debug_string()

    def transform(self, mat):
        """将图像转为模型输入"""
        canvas = cv.resize(mat, (self.input_node_dims[3], self.input_node_dims[2]))
        canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
        canvas = normalize(canvas, self.mean_val, self.scale_val)
        input_tensor = create_tensor(canvas, self.input_node_dims, DataFormat.CHW)
        return input_tensor

    def detect(self, mat: np.ndarray):
        if mat.size == 0:
            print("Warning: input is empty.")
            return
        input_tensor = self.transform(mat)
        output_tensors = self.ort_session.run(
            self.output_node_names,
            {self.input_name: input_tensor},
        )

        age_logits = output_tensors[0]
        softmax_probs, max_id = softmax(age_logits[0])
        pred_age = (self.age_intervals[max_id][0] + self.age_intervals[max_id][1]) / 2.0

        age = Age()
        age.age = pred_age
        age.age_interval = self.age_intervals[max_id]
        age.interval_prob = softmax_probs[max_id]
        age.flag = True
        return age


if __name__ == "__main__":
    onnx_path = "./lite/hub/ort/age_googlenet.onnx"
    net = AgeGoogleNet(onnx_path, 2)
    img_path = "./resources/test_lite_age_googlenet.jpg"
    img = cv.imread(img_path)
    age = net.detect(img)
    print(age.age)
