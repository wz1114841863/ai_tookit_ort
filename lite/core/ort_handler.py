import cv2 as cv
import numpy as np
import onnxruntime as ort


class BasicOrtHandler():
    """ 基类 """

    def __init__(self, onnx_path, num_threads=1):
        self.onnx_path = onnx_path
        self.num_threads = num_threads
        self.ort_session = None

        self.input_name = None
        self.input_node_names = []
        self.input_node_dims = []
        self.input_tensor_size = 1  # 仅支持单输入

        self.output_node_names = []
        self.output_node_dims = []
        self.num_outputs = 1

        self.initialize_handler()

    def __del__(self):
        if self.ort_session is not None:
            del self.ort_session

    def initialize_handler(self):
        """ 初始化ONNX模型, 获取输入输出信息. """
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.num_threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.log_severity_level = 4  # 设置日志级别为4(ERROR)

        self.ort_session = ort.InferenceSession(
            self.onnx_path, session_options)

        self.input_name = self.ort_session.get_inputs()[0].name
        self.input_node_names = [self.input_name]
        self.input_node_dims = self.ort_session.get_inputs()[0].shape
        self.input_tensor_size = np.prod(self.input_node_dims)

        self.num_outputs = len(self.ort_session.get_outputs())
        self.output_node_names = [
            output.name for output in self.ort_session.get_outputs()]
        self.output_node_dims = [
            output.shape for output in self.ort_session.get_outputs()]

        self.print_debug_string()

    # @abstractmethod
    def transform(self, mat):
        """ 抽象基类, 需要由子类实现 """
        pass

    def print_debug_string(self):
        """ 打印调试信息 """
        print(f"LITEORT_DEBUG LogId: {self.onnx_path}")
        print("=============== Input-Dims ==============")
        print(f"Name: {self.input_node_names[0]}")
        for i, dim in enumerate(self.input_node_dims):
            if dim == -1:
                print(f"Dims: dynamic")
            else:
                print(f"Dims: {dim}")

        print("=============== Output-Dims ==============")
        for i, output_dims in enumerate(self.output_node_dims):
            for j, dim in enumerate(output_dims):
                if dim == -1:
                    print(
                        f"Output: {i} Name: {self.output_node_names[i]} Dim: {j} : dynamic")
                else:
                    print(
                        f"Output: {i} Name: {self.output_node_names[i]} Dim: {j} : {dim}")
        print("========================================")


if __name__ == "__main__":
    onnx_path = "./lite/hub/ort/age_googlenet.onnx"
    handler = BasicOrtHandler(onnx_path, num_threads=2)
