import cv2 as cv
import numpy as np
import onnxruntime as ort

from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import Gender, draw_gender, softmax


class GenderGoogleNet(BasicOrtHandler):
    """gender googlenet"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean_val = [104.0, 117.0, 123.0]
        self.scale_val = [1.0, 1.0, 1.0]
        self.gender_texts = ["female", "male"]

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

        gender_logits = output_tensors[0]
        pred_logits = gender_logits[0]

        softmax_probs, pred_label = softmax(pred_logits)
        gender = Gender(
            softmax_probs[pred_label], pred_label, self.gender_texts[pred_label], True
        )
        return gender


if __name__ == "__main__":
    onnx_file = "lite/hub/ort/gender_googlenet.onnx"
    img_path = "resources/test_lite_pfld.jpg"

    model = GenderGoogleNet(onnx_file)
    img = cv.imread(img_path)
    results = model.detect(img)
    draw_gender(img, results)
    cv.imwrite("./test.jpg", img)
