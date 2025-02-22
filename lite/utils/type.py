import cv2 as cv


class BoundingBoxType:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = 0.
        self.label_txt = ""
        self.label = 0
        self.flag = False

    def iou_of(self, other):
        """ calculate iou """
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        self_area = self.width() * self.height()
        other_area = other.width() * other.height()

        return inter_area / (self_area + other_area - inter_area)

    def width(self):
        pass

    def height(self):
        pass

    def area(self):
        pass

    def rect(self):
        pass

    def tl(self):
        pass

    def rb(self):
        pass


class LandmarksType:
    def __init__(self):
        self.points: list[cv.Point2f] = []
        self.flag: bool = False


Landmarks = LandmarksType
Landmarks2D = LandmarksType


class Landmarks3DType:
    def __init__(self):
        self.points: list[cv.Point3f] = []
        self.flag: bool = False


Landmarks3D = Landmarks3DType


class AgeType:
    def __init__(self):
        self.age: float = 0.0
        self.age_interval: list[int] = [0, 0]
        self.interval_prob: float = 0.0
        self.flag: bool = False


Age = AgeType
