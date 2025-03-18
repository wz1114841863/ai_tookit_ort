import cv2 as cv
import numpy as np
import onnxruntime as ort

from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import EulerAngles, draw_axis


class FSANet(BasicOrtHandler):
    """https://github.com/omasaht/headpose-fsanet-pytorch"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.pad = 0.3
        self.mean_val = 127.5
        self.scale_val = 1 / 127.5

    def transform(self, mat):
        h, w = mat.shape[:2]
        nh = int(h + self.pad * h)  # 计算填充后的高度
        nw = int(w + self.pad * w)  # 计算填充后的宽度

        nx1 = max(0, (nw - w) // 2)  # 水平填充起始点
        ny1 = max(0, (nh - h) // 2)  # 垂直填充起始点

        # 创建填充后的画布
        canvas = np.zeros((nh, nw, 3), dtype=np.uint8)
        canvas[ny1 : ny1 + h, nx1 : nx1 + w] = mat  # 将原图像复制到画布中心

        # 调整图像大小到模型输入尺寸
        canvas = cv.resize(mat, (self.input_node_dims[3], self.input_node_dims[2]))
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

        angle = output_tensors[0][0]
        euler_angle = EulerAngles(angle[0], angle[1], angle[2], True)
        return euler_angle


if __name__ == "__main__":
    var_onnx_path = "lite/hub/ort/fsanet-var.onnx"
    conv_onnx_path = "lite/hub/ort/fsanet-1x1.onnx"
    img_path = "resources/test_lite_fsanet.jpg"
    img = cv.imread(img_path)

    var_model = FSANet(var_onnx_path)
    var_results = var_model.detect(img)

    conv_model = FSANet(conv_onnx_path)
    conv_results = conv_model.detect(img)

    yaw = (var_results.yaw + conv_results.yaw) / 2.0
    pitch = (var_results.pitch + conv_results.pitch) / 2.0
    roll = (var_results.roll + conv_results.roll) / 2.0
    flag = var_results.flag and conv_results.flag

    angle = EulerAngles(yaw, pitch, roll, flag)

    draw_axis(img, angle)
    cv.imwrite("./test.jpg", img)
