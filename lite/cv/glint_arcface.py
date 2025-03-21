import cv2 as cv
import numpy as np
import onnxruntime as ort

from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import FaceContent, cosine_similarity


class GlintArcFace(BasicOrtHandler):
    """https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean_val = 127.5
        self.scale_val = 1.0 / 127.5

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

        embedding = output_tensors[0]
        embedding_dims = self.output_node_dims[0]
        hidden_dim = embedding_dims[1]

        embedding_values = embedding.reshape(-1)

        embedding_norm = cv.normalize(
            embedding_values, None, norm_type=cv.NORM_L2
        ).reshape(-1)
        face_content = FaceContent(embedding_norm, hidden_dim, True)
        return face_content


if __name__ == "__main__":
    onnx_file = "lite/hub/ort/ms1mv3_arcface_r50.onnx"
    img_path_0 = "resources/test_lite_faceid_0.png"
    img_path_1 = "resources/test_lite_faceid_1.png"

    model = GlintArcFace(onnx_file)
    img_0 = cv.imread(img_path_0)
    results_0 = model.detect(img_0)

    img_1 = cv.imread(img_path_1)
    results_1 = model.detect(img_1)

    similarity = cosine_similarity(results_1.embedding, results_0.embedding)
    print(f"Default Version Detected Sim: {similarity:.4}")
