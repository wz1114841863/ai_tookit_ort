import math
import cv2 as cv
import numpy as np
import onnxruntime as ort

from enum import Enum
from lite.core import BasicOrtHandler
from lite.core import DataFormat, create_tensor, normalize
from lite.utils import (
    BBox,
    draw_boxes,
    softmax,
    EFFICIENTDET_CLASSES,
    blending_nms,
    hard_nms,
    offset_nms,
)


class NMS(Enum):
    HARD = 0
    BLEND = 1
    OFFSET = 2


class EfficientDetAnchor(BasicOrtHandler):
    """"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean_val = [0.406, 0.456, 0.486]
        self.scale_val = [1.0 / 0.225, 1.0 / 0.224, 1.0 / 0.229]
        self.class_names = EFFICIENTDET_CLASSES
        self.max_nms = 30000
        self.anchor_scale = 4.0
        self.anchors_buffer = []
        self.pyramid_levels = [3, 4, 5, 6, 7]

        # 定义strides
        self.strides = [
            math.pow(2.0, 3.0),
            math.pow(2.0, 4.0),
            math.pow(2.0, 5.0),
            math.pow(2.0, 6.0),
            math.pow(2.0, 7.0),
        ]

        # 定义scales
        self.scales = [
            math.pow(2.0, 0.0),
            math.pow(2.0, 1.0 / 3.0),
            math.pow(2.0, 2.0 / 3.0),
        ]

        # 定义ratios
        self.ratios = [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]]

    def generate_anchors(self, target_height: float, target_width: float):
        """生成锚点"""
        if self.anchors_buffer:
            return
        for stride in self.strides:
            # 生成网格点: yv从stride/2开始，步长stride，直到超过target_height
            for yv in np.arange(stride / 2.0, target_height, stride):
                # 生成网格点: xv从stride/2开始，步长stride，直到超过target_width
                for xv in np.arange(stride / 2.0, target_width, stride):
                    # 遍历所有scale和ratio组合
                    for scale in self.scales:
                        for ratio in self.ratios:
                            base_size = self.anchor_scale * stride * scale
                            # 计算锚点尺寸
                            w = base_size * ratio[0]
                            h = base_size * ratio[1]
                            # 计算锚点坐标 (y1, x1, y2, x2)
                            x1 = xv - w / 2.0
                            y1 = yv - h / 2.0
                            x2 = xv + w / 2.0
                            y2 = yv + h / 2.0
                            self.anchors_buffer.append((y1, x1, y2, x2))

    def generate_bboxes(self, output_tensors, score_threshold, img_height, img_width):
        regression = output_tensors[0]  # (1, n, 4) (dy, dx, dh, dw)
        classification = output_tensors[1]  # (1, n, 90) 90 classes
        reg_dims = regression.shape  # (1, n, 4)
        cls_dims = classification.shape  # (1, n, 90)
        num_anchors = reg_dims[1]  # n
        num_classes = cls_dims[2]  # 90
        input_height = self.input_node_dims[2]  # e.g 512
        input_width = self.input_node_dims[3]  # e.g 512
        scale_height = img_height / input_height
        scale_width = img_width / input_width

        # 生成锚点
        self.generate_anchors(input_height, input_width)

        if len(self.anchors_buffer) != num_anchors:
            raise ValueError(
                f"mismatch size for anchors_buffer and num_anchor. {len(self.anchors_buffer)} != {num_anchors}"
            )

        bbox_collection = []
        count = 0

        for i in range(num_anchors):
            # 获取分类得分
            cls_conf = classification[0, i, 0]
            label = 0
            for j in range(num_classes):
                tmp_conf = classification[0, i, j]
                if tmp_conf > cls_conf:
                    cls_conf = tmp_conf
                    label = j

            # 过滤低分框
            if cls_conf < score_threshold:
                continue

            # 获取锚点坐标
            ay1, ax1, ay2, ax2 = self.anchors_buffer[i]
            cya = (ay1 + ay2) / 2.0  # 中心点 y
            cxa = (ax1 + ax2) / 2.0  # 中心点 x
            ha = ay2 - ay1  # 高度
            wa = ax2 - ax1  # 宽度

            # 获取回归值
            dy = regression[0, i, 0]
            dx = regression[0, i, 1]
            dh = regression[0, i, 2]
            dw = regression[0, i, 3]

            # 计算边界框坐标
            cx = dx * wa + cxa
            cy = dy * ha + cya
            w = math.exp(dw) * wa
            h = math.exp(dh) * ha

            # 缩放边界框到原始图像尺寸
            x1 = (cx - w / 2.0) * scale_width
            y1 = (cy - h / 2.0) * scale_height
            x2 = (cx + w / 2.0) * scale_width
            y2 = (cy + h / 2.0) * scale_height

            # 创建边界框对象
            box = BBox(x1, y1, x2, y2)
            box.score = cls_conf
            box.label = label
            box.label_txt = self.class_names[label]
            box.flag = True
            bbox_collection.append(box)
            count += 1

            # 限制边界框数量
            if count > self.max_nms:
                break

        return bbox_collection

    def transform(self, mat):
        """将图像转为模型输入"""
        canvas = cv.resize(mat, (self.input_node_dims[3], self.input_node_dims[2]))
        canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
        canvas = canvas.astype(np.float32) / 255.0
        canvas = normalize(canvas, self.mean_val, self.scale_val)
        input_tensor = create_tensor(canvas, self.input_node_dims, DataFormat.CHW)
        return input_tensor

    def detect(
        self,
        mat,
        score_threashold=0.25,
        iou_threshold=0.45,
        topk=100,
        nms_type=NMS.OFFSET,
    ):
        if mat.size == 0:
            return
        img_height, img_width = mat.shape[:2]

        input_tensor = self.transform(mat)
        output_tensors = self.ort_session.run(
            self.output_node_names,
            {self.input_name: input_tensor},
        )
        bbox_collection = self.generate_bboxes(
            output_tensors, score_threashold, img_height, img_width
        )
        detected_boxes = self.nms(bbox_collection, iou_threshold, topk, nms_type)
        return detected_boxes

    def nms(self, input_bboxs, iou_threshold, topk, nms_type):
        if nms_type == NMS.BLEND:
            output_bboxs = blending_nms(input_bboxs, iou_threshold, topk)
        elif nms_type == NMS.HARD:
            output_bboxs = hard_nms(input_bboxs, iou_threshold, topk)
        elif nms_type == NMS.OFFSET:
            output_bboxs = offset_nms(input_bboxs, iou_threshold, topk)
        else:
            raise ValueError(f"未实现的NMS方法: {nms_type}")
        return output_bboxs


if __name__ == "__main__":
    onnx_file = "lite/hub/ort/efficientdet-d7.onnx"
    img_path = "resources/cat.jpg"

    model = EfficientDetAnchor(onnx_file)
    img = cv.imread(img_path)
    results = model.detect(img)
    draw_boxes(img, results)
    cv.imwrite("./test.jpg", img)
