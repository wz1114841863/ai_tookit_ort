import cv2 as cv
import numpy as np
import onnxruntime as ort

from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import StyleContent


class FastStyleTransfer(BasicOrtHandler):
    """"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)

    def transform(self, mat):
        """将图像转为模型输入"""
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

        pred_tensor = output_tensors[0]
        pred_dims = self.output_node_dims[0]
        rows, cols = pred_dims[2], pred_dims[3]

        style_content_mat = np.zeros((rows, cols, 3), dtype=np.uint8)
        style_content_mat[:, :, 0] = np.uint8(np.clip(pred_tensor[0, 0, :, :], 0, 255))
        style_content_mat[:, :, 1] = np.uint8(np.clip(pred_tensor[0, 1, :, :], 0, 255))
        style_content_mat[:, :, 2] = np.uint8(np.clip(pred_tensor[0, 2, :, :], 0, 255))

        style_content_mat = cv.cvtColor(style_content_mat, cv.COLOR_RGB2BGR)
        style_content = StyleContent(style_content_mat, True)
        return style_content


if __name__ == "__main__":
    candy_onnx_file = "lite/hub/ort/style-candy-8.onnx"
    mosaic_onnx_file = "lite/hub/ort/style-mosaic-8.onnx"
    pointilism_onnx_file = "lite/hub/ort/style-pointilism-8.onnx"
    rain_princess_onnx_file = "lite/hub/ort/style-rain-princess-8.onnx"
    udnie_onnx_file = "lite/hub/ort/style-udnie-8.onnx"

    img_path = "resources/test_lite_fast_style_transfer.jpg"

    model = FastStyleTransfer(udnie_onnx_file)
    img = cv.imread(img_path)
    results = model.detect(img)
    results = results.mat
    cv.imwrite("./test.jpg", results)
