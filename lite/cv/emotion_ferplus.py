import cv2 as cv
import numpy as np
import onnxruntime as ort

from enum import Enum
from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import Emotions, softmax, draw_emotion


class EmotionFerPlus(BasicOrtHandler):
    """ """

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.emotion_texts = [
            "neutral",
            "happiness",
            "surprise",
            "sadness",
            "anger",
            "disgust",
            "fear",
            "contempt",
        ]

    def transform(self, mat):
        """将图像转为模型输入"""
        canvas = cv.resize(mat, (self.input_node_dims[3], self.input_node_dims[2]))
        canvas = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
        canvas = np.expand_dims(canvas, axis=-1)
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

        emotion_logits = output_tensors[0]
        emotion_dims = emotion_logits.shape  # 获取张量的维度
        num_emotions = emotion_dims[1]  # 情绪类别数量，这里是8

        pred_logits = emotion_logits[0]  # 取第一行，形状为(8,)

        softmax_probs, pred_label = softmax(pred_logits)
        emotion = Emotions(
            softmax_probs[pred_label], pred_label, self.emotion_texts[pred_label], True
        )
        return emotion


if __name__ == "__main__":
    onnx_file = "lite/hub/ort/emotion-ferplus-8.onnx"
    onnx_file = "lite/hub/ort/emotion-ferplus-7.onnx"
    img_path = "resources/test_lite_emotion_ferplus.jpg"
    ultraface = EmotionFerPlus(onnx_file)
    img = cv.imread(img_path)
    results = ultraface.detect(img)
    draw_emotion(img, results)
    cv.imwrite("./test.jpg", img)
