import cv2 as cv
import numpy as np
import onnxruntime as ort

from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import Landmarks, draw_landmarks


class PFLD106(BasicOrtHandler):
    """人脸检测网络"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean_val = 0.0
        self.scale_val = 1.0 / 255.0

    def transform(self, mat):
        """将图像转为模型输入"""
        canvas = cv.resize(mat, (self.input_node_dims[3], self.input_node_dims[2]))
        canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
        canvas = normalize(canvas, self.mean_val, self.scale_val)
        input_tensor = create_tensor(canvas, self.input_node_dims, DataFormat.CHW)
        return input_tensor

    def detect(self, mat):
        if mat.size == 0:
            return

        img_height, img_width = mat.shape[:2]
        input_tensor = self.transform(mat)
        output_tensors = self.ort_session.run(
            self.output_node_names,
            {self.input_name: input_tensor},
        )

        landmarks_norm = output_tensors[1]  # 假设第二个输出是landmarks
        num_landmarks = landmarks_norm.shape[1]
        landmarks = Landmarks()
        for i in range(0, num_landmarks, 2):
            x = landmarks_norm[0, i]
            y = landmarks_norm[0, i + 1]
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)

            landmarks.points.append((x * img_width, y * img_height))
        landmarks.flag = True
        return landmarks


if __name__ == "__main__":
    onnx_file = "lite/hub/ort/pfld-106-v2.onnx"
    onnx_file = "lite/hub/ort/pfld-106-v3.onnx"
    onnx_file = "lite/hub/ort/pfld-106-lite.onnx"
    img_path = "resources/test_lite_pfld_3.png"
    model = PFLD106(onnx_file)
    img = cv.imread(img_path)
    results = model.detect(img)
    draw_landmarks(img, results)
    cv.imwrite("./test.jpg", img)
