import cv2 as cv
import numpy as np
import onnxruntime as ort

from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import PixelContent


class SubPixelCNN(BasicOrtHandler):
    """超分辨率网络: https://github.com/niazwazir/SUB_PIXEL_CNN"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)

    def transform(self, mat):
        canvas = cv.resize(mat, (self.input_node_dims[3], self.input_node_dims[2]))
        canvas = cv.cvtColor(canvas, cv.COLOR_BGR2YCrCb)
        canvas_y, canvas_cr, canvas_cb = cv.split(canvas)
        canvas_y = canvas_y.astype(np.float32) / 255.0
        canvas_y = np.expand_dims(canvas_y, axis=-1)
        input_tensor = create_tensor(canvas_y, self.input_node_dims, DataFormat.CHW)
        return input_tensor, canvas_cr, canvas_cb

    def detect(self, mat):
        if mat.size == 0:
            print("Warning: input is empty.")
            return

        input_tensor, mat_cr, mat_cb = self.transform(mat)
        output_tensors = self.ort_session.run(
            self.output_node_names,
            {self.input_name: input_tensor},
        )

        pred_tensor = output_tensors[0]
        rows, cols = self.output_node_dims[0][2], self.output_node_dims[0][3]

        mat_y = np.clip(pred_tensor[0, 0] * 255.0, 0, 255).astype(np.uint8)

        mat_cr = cv.resize(mat_cr, (cols, rows))
        mat_cb = cv.resize(mat_cb, (cols, rows))
        out_mats = [mat_y, mat_cr, mat_cb]
        super_resolution_mat = cv.merge(out_mats)

        super_resolution_mat = cv.cvtColor(super_resolution_mat, cv.COLOR_YCrCb2BGR)
        result = PixelContent(super_resolution_mat, True)
        return result


if __name__ == "__main__":
    onnx_file = "lite/hub/ort/subpixel-cnn.onnx"
    img_path = "resources/test_lite_subpixel_cnn.jpg"

    model = SubPixelCNN(onnx_file)
    img = cv.imread(img_path)
    results = model.detect(img)
    cv.imwrite("./test.jpg", results.mat)
