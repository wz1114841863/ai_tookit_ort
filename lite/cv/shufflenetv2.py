import cv2 as cv
import numpy as np
import onnxruntime as ort

from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import ClassificationContent, softmax, SHUFFLENET_CLASSES, draw_label


class ShuffleNetV2(BasicOrtHandler):
    """ShuffleNetV2"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean_val = np.array([0.485, 0.456, 0.406])
        self.scale_val = np.array([1 / 0.229, 1 / 0.224, 1 / 0.225])
        self.classes = SHUFFLENET_CLASSES

    def transform(self, mat):
        canvas = cv.resize(mat, (self.input_node_dims[3], self.input_node_dims[2]))
        canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
        canvas = canvas.astype(np.float32) / 255.0
        canvas = normalize(canvas, self.mean_val, self.scale_val)
        input_tensor = create_tensor(canvas, self.input_node_dims, DataFormat.CHW)
        return input_tensor

    def detect(self, mat, top_k: int = 5):
        if mat.size == 0:
            print("Warning: input is empty.")
            return

        input_tensor = self.transform(mat)
        output_tensors = self.ort_session.run(
            self.output_node_names,
            {self.input_name: input_tensor},
        )
        tensor_logits = output_tensors[0]
        softmax_probs, max_id = softmax(tensor_logits[0])
        # print(max_id)
        sorted_indics = np.argsort(-softmax_probs)
        content = ClassificationContent()
        for i in range(top_k):
            content.labels.append(sorted_indics[i])
            content.scores.append(softmax_probs[sorted_indics[i]])
            content.texts.append(self.classes[sorted_indics[i]])
        content.flag = True
        return content


if __name__ == "__main__":
    onnx_path = "lite/hub/ort/shufflenet-v2-10.onnx"
    img_path = "./resources/test_lite_ssd_mobilenetv1.png"

    net = ShuffleNetV2(onnx_path, 2)
    img = cv.imread(img_path)
    content = net.detect(img, 5)
    draw_label(img, content)
    cv.imwrite("./test.jpg", img)
