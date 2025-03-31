import cv2 as cv
import numpy as np
import onnxruntime as ort

from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import draw_label, softmax, RESNET_CLASSES, ClassificationContent


class Resnet(BasicOrtHandler):
    """Resnet"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean_val = [0.485, 0.456, 0.406]
        self.scale_val = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]
        self.classes = RESNET_CLASSES

    def transform(self, mat):
        """将图像转为模型输入"""
        canvas = cv.resize(mat, (self.input_node_dims[3], self.input_node_dims[2]))
        canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
        canvas = canvas.astype(np.float32) / 255.0
        input_tensor = create_tensor(canvas, self.input_node_dims, DataFormat.CHW)
        return input_tensor

    def detect(self, mat, top_k=10):
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
        # print(content)
        return content


if __name__ == "__main__":
    onnx_file = "lite/hub/ort/resnet18.onnx"
    img_path = "resources/cat.jpg"

    model = Resnet(onnx_file)
    img = cv.imread(img_path)
    results = model.detect(img)
    draw_label(img, results)
    cv.imwrite("./test.jpg", img)
