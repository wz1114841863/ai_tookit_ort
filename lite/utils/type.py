import cv2 as cv


class BoundingBoxType:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = 0.0
        self.label_txt = ""
        self.label = 0
        self.flag = False

    def iou_of(self, other):
        """calculate iou"""
        inner_x1 = max(self.x1, other.x1)
        inner_y1 = max(self.y1, other.y1)
        inner_x2 = min(self.x2, other.x2)
        inner_y2 = min(self.y2, other.y2)

        inter_area = max(0, inner_x2 - inner_x1) * max(0, inner_y2 - inner_y1)
        self_area = self.width() * self.height()
        other_area = other.width() * other.height()

        return inter_area / (self_area + other_area - inter_area)

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def area(self):
        return abs(self.width() * self.height())

    def rect(self):
        """返回OpenCV格式的矩形区域"""
        return (self.x1, self.y1, self.width(), self.height())

    def tl(self):
        return (int(self.x1), int(self.y1))

    def rb(self):
        return (int(self.x2), int(self.y2))


BBox = BoundingBoxType


class LandmarksType:
    """used for save coordinate"""

    def __init__(self):
        self.points = []
        self.flag = False


Landmarks = LandmarksType
Landmarks2D = LandmarksType


class Landmarks3DType:
    """used for save 3D coordinate"""

    def __init__(self):
        self.points = []
        self.flag = False


Landmarks3D = Landmarks3DType


class AgeType:
    """used for AgeGoogleNet"""

    def __init__(self):
        self.age: float = 0.0
        self.age_interval: list[int] = [0, 0]
        self.interval_prob: float = 0.0
        self.flag: bool = False

    def __str__(self):
        return f"""
            Age.age          : {self.age},
            Age.age_interval : {self.age_interval},
            Age.interval_prob: {self.interval_prob}
        """


Age = AgeType


class ImageContentType:
    """used for object classification"""

    def __init__(self):
        self.scores = []  # sorted
        self.texts = []
        self.labels = []
        self.flag = False

    def __str__(self):
        return f"""
            Content.scores: {self.scores},
            Content.texts : {self.texts},
            Content.labels: {self.labels}
        """


ClassificationContent = ImageContentType


class KeypointCoordinateType:
    """used for 2D Keypoint Coordinate"""

    def __init__(self, x=0.0, y=0.0, score=0.0, flag=False):
        self.x = x
        self.y = y
        self.score = score
        self.flag = flag

    def __str__(self):
        return f"""
            Keypoint.coordinate: {self.x, self.y},
            Keypoint.score: {self.score}
        """


Keypoint = KeypointCoordinateType


class EmotionsType:
    """used for Emotions"""

    def __init__(self, score, label, text, flag):
        self.score = score
        self.label = label
        self.text = text
        self.flag = flag


Emotions = EmotionsType


class EulerAnglesType:
    """used for head engle"""

    def __init__(self, yaw, pitch, roll, flag):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.flag = flag


EulerAngles = EulerAnglesType


class GenderType:
    """used for genders"""

    def __init__(self, score, label, text, flag):
        self.score = score
        self.label = label
        self.text = text
        self.flag = flag


Gender = GenderType
