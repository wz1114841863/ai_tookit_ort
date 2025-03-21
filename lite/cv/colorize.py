import cv2 as cv
import numpy as np
import onnxruntime as ort

from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import ColorizeContent


class Colorize(BasicOrtHandler):
    """为图像添加色彩"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)

    def transform(self, mat):
        """将图像转为模型输入"""
        mat_l = np.expand_dims(mat, axis=-1)
        input_tensor = create_tensor(mat_l, self.input_node_dims, DataFormat.CHW)
        return input_tensor

    def detect(self, mat):
        if mat.size == 0:
            print("Warning: input is empty.")
            return
        height, width = mat.shape[:2]

        mat_rs = cv.resize(mat, (self.input_node_dims[3], self.input_node_dims[2]))

        # 归一化图像
        mat_rs_norm = mat_rs.astype(np.float32) / 255.0  # (0., 1.) BGR
        mat_orig_norm = mat.astype(np.float32) / 255.0  # (0., 1.) BGR

        # 转换为Lab色彩空间
        mat_lab_rs = cv.cvtColor(mat_rs_norm, cv.COLOR_BGR2Lab)
        mat_lab_orig = cv.cvtColor(mat_orig_norm, cv.COLOR_BGR2Lab)

        # 提取L通道
        mat_rs_l = mat_lab_rs[:, :, 0]  # (256, 256)
        mat_orig_l = mat_lab_orig[:, :, 0]  # (H, W)

        input_tensor = self.transform(mat_rs_l)
        output_tensors = self.ort_session.run(
            self.output_node_names,
            {self.input_name: input_tensor},
        )

        pred_ab_tensor = output_tensors[0]  # (1, 2, 256, 256)

        # 提取预测的a和b通道
        out_a_orig = pred_ab_tensor[0, 0, :, :]  # (256, 256)
        out_b_orig = pred_ab_tensor[0, 1, :, :]  # (256, 256)]

        # 如果原始图像尺寸与模型输出尺寸不一致，调整大小
        if out_a_orig.shape[0] != height or out_a_orig.shape[1] != width:
            out_a_orig = cv.resize(out_a_orig, (width, height))
            out_b_orig = cv.resize(out_b_orig, (width, height))

        # 合并L、a、b通道
        out_mats_lab = [mat_orig_l, out_a_orig, out_b_orig]
        merge_mat_lab = cv.merge(out_mats_lab)  # (H, W, 3)

        # 转换回BGR色彩空间
        mat_bgr_norm = cv.cvtColor(merge_mat_lab, cv.COLOR_Lab2BGR)  # (H, W, 3)
        mat_bgr_norm = (mat_bgr_norm * 255).astype(np.uint8)  # 转换为uint8

        content = ColorizeContent(mat_bgr_norm, True)
        return content


if __name__ == "__main__":
    onnx_file = "lite/hub/ort/siggraph17-colorizer.onnx"
    img_path1 = "resources/test_lite_colorizer_1.jpg"
    img_path2 = "resources/test_lite_colorizer_2.jpg"
    img_path3 = "resources/test_lite_colorizer_3.jpg"

    model = Colorize(onnx_file)

    img = cv.imread(img_path3)
    result = model.detect(img)
    cv.imwrite("./test.jpg", result.mat)
