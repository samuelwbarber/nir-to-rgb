from .base import BaseEvaluator
from .mediapipe_models import (
    MediaPipeHandsEvaluator,
    MediaPipeFaceEvaluator,
    MediaPipePoseEvaluator,
    MediaPipeSelfieEvaluator,
)
from .torchvision_models import (
    DeepLabV3Evaluator,
    MaskRCNNEvaluator,
    ResNet50Evaluator,
)
from .yolo import YOLOEvaluator
from .midas import MiDaSEvaluator


REGISTRY = {
    "mp_hands": MediaPipeHandsEvaluator,
    "mp_face": MediaPipeFaceEvaluator,
    "mp_pose": MediaPipePoseEvaluator,
    "mp_selfie": MediaPipeSelfieEvaluator,
    "yolo": YOLOEvaluator,
    "deeplab": DeepLabV3Evaluator,
    "maskrcnn": MaskRCNNEvaluator,
    "resnet50": ResNet50Evaluator,
    "midas": MiDaSEvaluator,
}


SKIN_HEAVY = ["mp_hands", "mp_face", "mp_pose", "mp_selfie"]
COLOR_HEAVY = ["yolo", "deeplab", "maskrcnn", "resnet50"]
GEOMETRIC = ["midas"]
ALL = list(REGISTRY.keys())
