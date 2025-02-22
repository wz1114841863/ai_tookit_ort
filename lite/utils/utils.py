import onnx
import numpy as np


def check_onnxfile(onnx_path):
    """ 检查onnx文件的正确性 """
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


def softmax(logits):
    """
    计算Softmax概率并返回最大概率的索引.
    :param logits: 输入的logits数组.
    :return: 包含Softmax概率和最大概率索引的元组.
    """
    if len(logits) == 0:
        return np.array([]), -1
    logits = np.array(logits)
    exp_logits = np.exp(logits - np.max(logits))
    softmax_probs = exp_logits / np.sum(exp_logits)

    max_id = np.argmax(softmax_probs)
    return softmax_probs, max_id


if __name__ == "__main__":
    onnx_path = "./lite/hub/ort/age_googlenet.onnx"
    check_onnxfile(onnx_path)

    logits = np.array([2.0, 1.0, 0.1])
    probs, max_id = softmax(logits)
    print("Softmax Probabilities:", probs)
    print("Max Probability Index:", max_id)
