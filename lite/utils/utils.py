import onnx
import numpy as np
import cv2 as cv
from lite.utils import BBox, Keypoint


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


def hard_nms(input_boxes, iou_threshold, topk):
    """Hard-NMS, 输入的BBox输入tupes.BBox类型"""
    if not input_boxes:
        return []

    input_boxes.sort(key=lambda x: x.score, reverse=True)
    box_num = len(input_boxes)
    merged = [False] * box_num
    output = []

    count = 0
    for i in range(box_num):
        if merged[i]:
            continue

        buf = [input_boxes[i]]
        merged[i] = True

        for j in range(i + 1, box_num):
            if merged[j]:
                continue

            iou = input_boxes[i].iou_of(input_boxes[j])
            if iou > iou_threshold:
                merged[j] = 1
                buf.append(input_boxes[j])

        output.append(buf[0])

        # Keep top k
        count += 1
        if count >= topk:
            break

    return output


def blending_nms(input_boxes, iou_threshold, topk):
    """加权平均NMS"""
    if not input_boxes:
        return []

    input_boxes.sort(key=lambda x: x.score, reverse=True)
    box_num = len(input_boxes)
    merged = [False] * box_num
    output = []
    count = 0

    for i in range(box_num):
        if merged[i]:
            continue
        buf = [input_boxes[i]]
        merged[i] = 1

        for j in range(i + 1, box_num):
            if merged[j]:
                continue
            iou = input_boxes[i].iou_of(input_boxes[j])
            if iou > iou_threshold:
                merged[j] = 1
                buf.append(input_boxes[j])

        # 计算加权平均
        total = sum(np.exp(box.score) for box in buf)
        bbox = BBox(0.0, 0.0, 0.0, 0.0)
        bbox.score = 0.0
        bbox.flag = True

        for box in buf:
            rate = np.exp(box.score) / total
            bbox.x1 += box.x1 * rate
            bbox.y1 += box.y1 * rate
            bbox.x2 += box.x2 * rate
            bbox.y2 += box.y2 * rate
            bbox.score += box.score * rate

        output.append(bbox)
        count += 1
        if count >= topk:
            break

    return output


def offset_nms(input_boxes, iou_threshold, topk):
    """Offset-NMS"""
    if not input_boxes:
        return []

    input_boxes.sort(key=lambda x: x.score, reverse=True)
    box_num = len(input_boxes)
    merged = [False] * box_num

    offset = 4096.0
    for box in input_boxes:
        offset_ = float(box.label * offset)
        box.x1 += offset_
        box.y1 += offset_
        box.x2 += offset_
        box.y2 += offset_

    output = []
    count = 0
    for i in range(box_num):
        if merged[i]:
            continue
        buf = [input_boxes[i]]
        merged[i] = 1

        for j in range(i + 1, box_num):
            if merged[j]:
                continue

            iou = input_boxes[i].iou_of(input_boxes[j])
            if iou > iou_threshold:
                merged[j] = 1
                buf.append(input_boxes[j])

        output.append(buf[0])
        count += 1
        if count >= topk:
            break

    # Subtract offset
    for box in output:
        fset_ = float(box.label * offset)
        box.x1 -= offset_
        box.y1 -= offset_
        box.x2 -= offset_
        box.y2 -= offset_

    return output


def draw_boxes(mat, boxes):
    if len(boxes) == 0:
        return
    for box in boxes:
        if box.flag:
            cv.rectangle(mat, box.tl(), box.rb(), (255, 255, 0), 2)
            if box.label_txt:
                label_txt = f"{box.label_txt}: {box.score:.4f}"
                cv.putText(
                    mat,
                    label_txt,
                    box.tl(),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )


def draw_age(mat, age):
    if not age.flag:
        return
    offset = int(0.1 * mat.shape[0])
    age_text = f"Age: {age.age}"
    interval_text = f"Interval {age.age_interval[0]}-{age.age_interval[1]}"
    prob = f"Prob: {age.interval_prob:.4f}"
    cv.putText(
        mat, age_text, (10, offset), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )
    cv.putText(
        mat,
        interval_text,
        (10, offset * 2),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )
    cv.putText(
        mat, prob, (10, offset * 3), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    )


def draw_keypoint(mat, KeypointCoordinate):
    if not KeypointCoordinate.flag:
        return
    cv.circle(
        mat,
        (KeypointCoordinate.x, KeypointCoordinate.y),
        radius=1,
        color=(0, 0, 255),
        thickness=-1,
    )


if __name__ == "__main__":
    onnx_path = "./lite/hub/ort/age_googlenet.onnx"

    logits = np.array([2.0, 1.0, 0.1])
    probs, max_id = softmax(logits)
    print("Softmax Probabilities:", probs)
    print("Max Probability Index:", max_id)
