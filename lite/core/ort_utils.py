import cv2 as cv
import onnxruntime as ort
import numpy as np
from enum import Enum


class DataFormat(Enum):
    CHW = 0
    HWC = 1


def create_tensor(mat, tensor_dims, data_format):
    """
    将 OpenCV 的 Mat 转换为 ONNX Runtime 的张量.

    参数:
        mat (np.ndarray): 输入的 OpenCV 图像(HWC 格式).
        tensor_dims (list): 目标张量的维度(例如 [1, 3, 224, 224]).
        memory_info_handler: ONNX Runtime 的内存信息处理器.
        data_format (str): 数据格式,"CHW" 或 "HWC".

    返回:
        ort.OrtValue: 转换后的张量.
    """
    rows, cols, channels = mat.shape

    if len(tensor_dims) != 4:
        raise ValueError("Dims mismatch.")
    elif tensor_dims[0] != 1:
        raise ValueError("batch != 1")
    elif tensor_dims[1] != channels:
        raise ValueError("Channel mismatch.")

    target_height, target_width = tensor_dims[2], tensor_dims[3]
    if target_height != rows or target_width != cols:
        mat = cv.resize(mat, (target_width, target_height))

    mat = mat.astype(np.float32)

    # 根据数据格式进行转换
    if data_format == DataFormat.CHW:
        mat = np.transpose(mat, (2, 0, 1))
    elif data_format == DataFormat.HWC:
        pass
    else:
        raise ValueError("Invalid data format.")

    tensor_value_handler = mat.flatten()

    return ort.OrtValue.ortvalue_from_numpy(
        tensor_value_handler.reshape(tensor_dims)
    )


def normalize(mat, mean, scale):
    """
    使用单个均值和缩放因子对图像进行归一化.

    参数:
        mat (np.ndarray): 输入的 OpenCV 图像(BGR 格式).
        mean (float): 归一化的均值.
        scale (float): 归一化的缩放因子.

    返回:
        np.ndarray: 归一化后的图像.
    """
    # 检查输入图像
    if mat is None or mat.size == 0:
        raise ValueError("输入图像为空或无效.")

    mean = np.array(mean, dtype=np.float32)
    scale = np.array(scale, dtype=np.float32)

    if mat.dtype != np.float32:
        mat = mat.astype(np.float32)
    return (mat - mean) * scale



if __name__ == "__main__":
    img_path = "./resources/test_lite_age_googlenet.jpg"
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError("Image not found.")

    tensor_dims = [1, 3, 224, 224]
    tensor = create_tensor(img, tensor_dims, data_format=DataFormat.CHW)

    print("Tensor shape:", tensor.shape())
    print("Tensor data:", tensor.numpy())
    print("Tensor data type", type(tensor))

    # 测试mean 和 scale为单个数值
    mean, scale = 125, 1.0
    mat = normalize(img, mean, scale)
    # print(f"Mat: {mat}")

    # 测试mean 和 scale为长度为3的数组
    mean = [104.0, 117.0, 123.0]
    scale = [1.0, 1.0, 1.0]
    mat = normalize(img, mean, mat)
    # print(f"Mat: {mat}")
