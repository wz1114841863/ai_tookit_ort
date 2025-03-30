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
    SSD_CLASSES,
    blending_nms,
    hard_nms,
    offset_nms,
)


class NMS(Enum):
    HARD = 0
    BLEND = 1
    OFFSET = 2


class SSD(BasicOrtHandler):
    """ssd net"""

    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean_val = [0.485, 0.456, 0.406]
        self.scale_val = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]
        self.class_names = SSD_CLASSES
        self.max_nms = 30000

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

    def generate_bboxes(self, output_tensors, score_threshold, img_height, img_width):
        bboxes = output_tensors[0]
        labels = output_tensors[1]
        scores = output_tensors[2]

        bboxes_dims = bboxes.shape
        num_anchors = bboxes_dims[1]
        bbox_collections = []
        count = 0
        for i in range(num_anchors):
            conf = scores[0][i]
            if conf < score_threshold:
                continue

            label = labels[0][i] - 1

            x1 = float(bboxes[0][i][0]) * float(img_width)
            y1 = float(bboxes[0][i][1]) * float(img_height)
            x2 = float(bboxes[0][i][2]) * float(img_width)
            y2 = float(bboxes[0][i][3]) * float(img_height)

            bbox = BBox(x1, y1, x2, y2)
            bbox.score = conf
            bbox.label = label
            bbox.label_txt = self.class_names[label]
            bbox.flag = True
            bbox_collections.append(bbox)

            count += 1
            if count > self.max_nms:
                break

        return bbox_collections

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
    onnx_file = "lite/hub/ort/ssd-10.onnx"
    img_path = "resources/cat.jpg"

    model = SSD(onnx_file)
    img = cv.imread(img_path)
    results = model.detect(img)
    draw_boxes(img, results)
    cv.imwrite("./test.jpg", img)
