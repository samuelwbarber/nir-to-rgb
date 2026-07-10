"""MediaPipe evaluators — skin-dependent tasks expected to break worst on NIR.

These models are trained on natural RGB. Skin in NIR appears unusually pale
(high reflectance at 800-900nm); eyes, lips, and other facial features
also look very different. Hand/face/body detection should fail dramatically
on NIR direct.

Supports both the legacy mp.solutions API (mediapipe < 0.10.15) and the
new Task API (mediapipe >= 0.10.15) which dropped mp.solutions.
"""

import math
import urllib.request
import os
import tempfile

import numpy as np

from .base import BaseEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _landmark_distance(pred, ref):
    """Mean Euclidean distance between paired landmark lists in normalized coords."""
    if not pred or not ref or len(pred) != len(ref):
        return float("nan")
    return _mean([_l2(p, r) for p, r in zip(pred, ref)])


def _has_solutions():
    """True if the installed mediapipe version still ships mp.solutions."""
    try:
        import mediapipe as mp
        return hasattr(mp, "solutions")
    except ImportError:
        return False


# Model asset URLs for the new Task API
_TASK_MODEL_URLS = {
    "hand_landmarker": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "image_segmenter": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}

_TASK_MODEL_CACHE = {}


def _get_task_model(key):
    """Download (once) and cache a mediapipe task model file path."""
    if key in _TASK_MODEL_CACHE:
        return _TASK_MODEL_CACHE[key]
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe_tasks")
    os.makedirs(cache_dir, exist_ok=True)
    filename = _TASK_MODEL_URLS[key].split("/")[-1]
    path = os.path.join(cache_dir, filename)
    if not os.path.exists(path):
        print(f"  [mediapipe] downloading {filename}…")
        urllib.request.urlretrieve(_TASK_MODEL_URLS[key], path)
    _TASK_MODEL_CACHE[key] = path
    return path


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class _MediaPipeBase(BaseEvaluator):
    """Shared lazy-import boilerplate; adapts to legacy or Task API."""

    def setup(self, device):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError("mediapipe not installed. pip install mediapipe") from e
        self._mp = mp
        self._use_legacy = _has_solutions()
        self._init_solution()

    def _init_solution(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Hands
# ---------------------------------------------------------------------------

class MediaPipeHandsEvaluator(_MediaPipeBase):
    name = "mp_hands"
    requires = ("mediapipe",)
    primary_metric = "detection_rate"

    def _init_solution(self):
        if self._use_legacy:
            self.solution = self._mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5,
            )
        else:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            model_path = _get_task_model("hand_landmarker")
            base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base_opts,
                num_hands=2,
                min_hand_detection_confidence=0.5,
            )
            self.solution = mp_vision.HandLandmarker.create_from_options(opts)

    def predict(self, rgb_uint8):
        if self._use_legacy:
            res = self.solution.process(rgb_uint8)
            hands = []
            if res.multi_hand_landmarks:
                for hand in res.multi_hand_landmarks:
                    hands.append([(lm.x, lm.y, lm.z) for lm in hand.landmark])
            return {"hands": hands}
        else:
            mp = self._mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_uint8)
            res = self.solution.detect(img)
            hands = []
            for hand in res.hand_landmarks:
                hands.append([(lm.x, lm.y, lm.z) for lm in hand])
            return {"hands": hands}

    def per_condition_stats(self, p):
        return {
            "detection_rate": float(len(p["hands"]) > 0),
            "n_hands": float(len(p["hands"])),
        }

    def vs_reference(self, pred, ref):
        if not pred["hands"] or not ref["hands"]:
            return {"landmark_l2": float("nan"), "detection_match": float(bool(pred["hands"]) == bool(ref["hands"]))}
        used = set()
        dists = []
        for r in ref["hands"]:
            r_wrist = r[0]
            best = None
            best_d = float("inf")
            for i, p in enumerate(pred["hands"]):
                if i in used:
                    continue
                d = _l2(p[0], r_wrist)
                if d < best_d:
                    best_d = d
                    best = i
            if best is not None:
                used.add(best)
                dists.append(_landmark_distance(pred["hands"][best], r))
        return {"landmark_l2": _mean(dists), "detection_match": 1.0}


# ---------------------------------------------------------------------------
# Face
# ---------------------------------------------------------------------------

class MediaPipeFaceEvaluator(_MediaPipeBase):
    name = "mp_face"
    requires = ("mediapipe",)
    primary_metric = "detection_rate"

    def _init_solution(self):
        if self._use_legacy:
            self.solution = self._mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        else:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            model_path = _get_task_model("face_landmarker")
            base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                num_faces=1,
                min_face_detection_confidence=0.5,
            )
            self.solution = mp_vision.FaceLandmarker.create_from_options(opts)

    def predict(self, rgb_uint8):
        if self._use_legacy:
            res = self.solution.process(rgb_uint8)
            faces = []
            if res.multi_face_landmarks:
                for face in res.multi_face_landmarks:
                    faces.append([(lm.x, lm.y, lm.z) for lm in face.landmark])
            return {"faces": faces}
        else:
            mp = self._mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_uint8)
            res = self.solution.detect(img)
            faces = []
            for face in res.face_landmarks:
                faces.append([(lm.x, lm.y, lm.z) for lm in face])
            return {"faces": faces}

    def per_condition_stats(self, p):
        return {"detection_rate": float(len(p["faces"]) > 0)}

    def vs_reference(self, pred, ref):
        if not pred["faces"] or not ref["faces"]:
            return {"landmark_l2": float("nan")}
        return {"landmark_l2": _landmark_distance(pred["faces"][0], ref["faces"][0])}


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

class MediaPipePoseEvaluator(_MediaPipeBase):
    name = "mp_pose"
    requires = ("mediapipe",)
    primary_metric = "detection_rate"

    def _init_solution(self):
        if self._use_legacy:
            self.solution = self._mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
            )
        else:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            model_path = _get_task_model("pose_landmarker")
            base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.PoseLandmarkerOptions(
                base_options=base_opts,
                min_pose_detection_confidence=0.5,
            )
            self.solution = mp_vision.PoseLandmarker.create_from_options(opts)

    def predict(self, rgb_uint8):
        if self._use_legacy:
            res = self.solution.process(rgb_uint8)
            if res.pose_landmarks is None:
                return {"landmarks": None, "visibilities": None}
            landmarks = [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark]
            vis = [lm.visibility for lm in res.pose_landmarks.landmark]
            return {"landmarks": landmarks, "visibilities": vis}
        else:
            mp = self._mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_uint8)
            res = self.solution.detect(img)
            if not res.pose_landmarks:
                return {"landmarks": None, "visibilities": None}
            lms = res.pose_landmarks[0]
            landmarks = [(lm.x, lm.y, lm.z) for lm in lms]
            vis = [lm.visibility for lm in lms]
            return {"landmarks": landmarks, "visibilities": vis}

    def per_condition_stats(self, p):
        if p["landmarks"] is None:
            return {"detection_rate": 0.0, "mean_visibility": float("nan")}
        return {
            "detection_rate": 1.0,
            "mean_visibility": float(np.mean(p["visibilities"])),
        }

    def vs_reference(self, pred, ref):
        if pred["landmarks"] is None or ref["landmarks"] is None:
            return {"landmark_l2": float("nan")}
        return {"landmark_l2": _landmark_distance(pred["landmarks"], ref["landmarks"])}


# ---------------------------------------------------------------------------
# Selfie segmentation
# ---------------------------------------------------------------------------

class MediaPipeSelfieEvaluator(_MediaPipeBase):
    name = "mp_selfie"
    requires = ("mediapipe",)
    primary_metric = "iou_vs_rgb"

    def _init_solution(self):
        if self._use_legacy:
            self.solution = self._mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1,
            )
        else:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            model_path = _get_task_model("image_segmenter")
            base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.ImageSegmenterOptions(
                base_options=base_opts,
                output_category_mask=True,
            )
            self.solution = mp_vision.ImageSegmenter.create_from_options(opts)

    def predict(self, rgb_uint8):
        if self._use_legacy:
            res = self.solution.process(rgb_uint8)
            if res.segmentation_mask is None:
                return {"mask": None}
            return {"mask": (res.segmentation_mask > 0.5).astype(np.uint8)}
        else:
            mp = self._mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_uint8)
            res = self.solution.segment(img)
            if not res.category_mask:
                return {"mask": None}
            mask = np.array(res.category_mask.numpy_view())
            return {"mask": (mask > 0).astype(np.uint8)}

    def per_condition_stats(self, p):
        if p["mask"] is None:
            return {"detection_rate": 0.0, "person_pixel_frac": float("nan")}
        frac = float(p["mask"].mean())
        return {"detection_rate": float(frac > 0.001), "person_pixel_frac": frac}

    def vs_reference(self, pred, ref):
        if pred["mask"] is None or ref["mask"] is None:
            return {"iou_vs_rgb": float("nan")}
        a = pred["mask"].astype(bool)
        b = ref["mask"].astype(bool)
        union = (a | b).sum()
        if union == 0:
            return {"iou_vs_rgb": 1.0}
        return {"iou_vs_rgb": float((a & b).sum()) / float(union)}
