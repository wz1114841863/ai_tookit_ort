import cv2 as cv
import numpy as np
import onnxruntime as ort

from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import Age, softmax, draw_age


class AgeVGG16(BasicOrtHandler):
    """年龄推断网络：
    https://github.com/onnx/models/tree/main/validated/vision/body_analysis/age_gender
    """

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)

    def transform(self, mat):
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

        age_probs = output_tensors[0]
        age_dims = self.output_node_dims[0]
        num_intervals = age_dims[1]

        pred_age = 0.0
        pred_probs = np.zeros(num_intervals, dtype=np.float32)
        for i in range(num_intervals):
            cur_prob = age_probs[0, i]
            pred_age += cur_prob * float(i)
            pred_probs[i] = cur_prob

        # 计算前 10 个最高概率的总和
        top10_pred_prob = np.sum(np.sort(pred_probs)[::-1][:10])

        # 计算年龄区间
        interval_min = max(int(pred_age - 2), 0)
        interval_max = min(int(pred_age + 3), 100)

        age = Age()
        age.age = int(pred_age)
        age.age_interval = [interval_min, interval_max]
        age.interval_prob = top10_pred_prob
        age.flag = True
        return age


if __name__ == "__main__":
    onnx_path = "./lite/hub/ort/vgg16_age.onnx"
    net = AgeVGG16(onnx_path)
    img_path = "./resources/test_lite_age_googlenet.jpg"
    img = cv.imread(img_path)
    age = net.detect(img)
    draw_age(img, age)
    cv.imwrite("./test.jpg", img)
