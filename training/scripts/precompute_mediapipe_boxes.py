"""Pre-compute MediaPipe smart-crop and region-mask boxes from full-res RGB.

For each RGB image we run face + pose + hands landmarkers and store:
  - crop_bbox   = union(pose, face), expanded to a square + margin. Used by the
                  dataset to crop both NIR and RGB to the subject before the
                  256x256 resize, so subjects fill more of the frame.
  - region_bbox = union(face, hands). Used by the trainer to upweight L1 loss
                  on the regions MediaPipe downstream eval cares about.

Run once before training:
    python scripts/precompute_mediapipe_boxes.py \
        --rgb-dir data/rgb --out data/mp_cache.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.downstream.mediapipe_models import _get_task_model  # noqa: E402

import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_tasks  # noqa: E402
from mediapipe.tasks.python import vision as mp_vision  # noqa: E402


def _bbox_from_landmarks(landmarks, w, h):
    if not landmarks:
        return None
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1 = max(0, int(round(min(xs))))
    y1 = max(0, int(round(min(ys))))
    x2 = min(w, int(round(max(xs))) + 1)
    y2 = min(h, int(round(max(ys))) + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


# MediaPipe Pose Landmarker landmark indices
POSE_FACE_IDX = list(range(0, 11))   # nose, eyes, ears, mouth corners
POSE_HAND_IDX = list(range(15, 23))  # wrists, pinky/index/thumb on both hands


def _bbox_from_pose_subset(pose_lms, indices, w, h, min_vis=0.5, pad_frac=0.10):
    """Build a bbox from a subset of pose landmarks, filtered by visibility.

    Adds small padding because pose landmarks are points, not extents
    (a "hand" landmark is a fingertip/wrist, not the hand silhouette).
    """
    if not pose_lms:
        return None
    pts = [pose_lms[i] for i in indices if i < len(pose_lms)]
    pts = [p for p in pts if getattr(p, "visibility", 1.0) >= min_vis]
    if len(pts) < 2:
        return None
    xs = [p.x * w for p in pts]
    ys = [p.y * h for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad = pad_frac * max(x2 - x1, y2 - y1, 1.0)
    x1 = max(0, int(round(x1 - pad)))
    y1 = max(0, int(round(y1 - pad)))
    x2 = min(w, int(round(x2 + pad)) + 1)
    y2 = min(h, int(round(y2 + pad)) + 1)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return [x1, y1, x2, y2]


def _union(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _expand_to_square(box, w, h, margin):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    half = side / 2.0
    nx1 = max(0, int(round(cx - half)))
    ny1 = max(0, int(round(cy - half)))
    nx2 = min(w, int(round(cx + half)))
    ny2 = min(h, int(round(cy + half)))
    if nx2 - nx1 < 16 or ny2 - ny1 < 16:
        return None
    return [nx1, ny1, nx2, ny2]


def _build_landmarkers():
    pose = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_get_task_model("pose_landmarker")),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
        )
    )
    face = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_get_task_model("face_landmarker")),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
    )
    hands = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_get_task_model("hand_landmarker")),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
        )
    )
    return pose, face, hands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", nargs="+", default=[".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    ap.add_argument("--crop-margin", type=float, default=0.20,
                    help="fraction to expand the crop bbox before squaring (default 0.20)")
    ap.add_argument("--limit", type=int, default=None, help="for smoke testing")
    args = ap.parse_args()

    rgb_dir = Path(args.rgb_dir)
    out_path = Path(args.out)
    exts = {e.lower() for e in args.ext}

    files = sorted(p for p in rgb_dir.rglob("*") if p.suffix.lower() in exts)
    if args.limit:
        files = files[: args.limit]
    print(f"found {len(files)} images in {rgb_dir}")

    pose, face, hands = _build_landmarkers()

    items = {}
    n_crop = n_region = n_face = n_pose = n_hand = 0
    n_face_fb = n_hand_fb = 0  # fallback counters
    try:
        for path in tqdm(files, desc="mediapipe"):
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))

            pose_res = pose.detect(mp_img)
            face_res = face.detect(mp_img)
            hand_res = hands.detect(mp_img)

            pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
            face_lms = face_res.face_landmarks[0] if face_res.face_landmarks else None
            hand_lms_list = hand_res.hand_landmarks if hand_res.hand_landmarks else []

            pose_box = _bbox_from_landmarks(pose_lms, w, h) if pose_lms else None
            face_box = _bbox_from_landmarks(face_lms, w, h) if face_lms else None
            hand_boxes = [b for b in (_bbox_from_landmarks(lms, w, h) for lms in hand_lms_list) if b]

            n_pose += int(pose_box is not None)
            n_face += int(face_box is not None)
            n_hand += int(bool(hand_boxes))

            # Pose-landmark fallback for region (face∪hands) when dedicated
            # face/hand detectors miss but pose is detected.
            face_for_region = face_box
            hands_for_region = list(hand_boxes)
            if face_for_region is None and pose_lms:
                fb = _bbox_from_pose_subset(pose_lms, POSE_FACE_IDX, w, h)
                if fb:
                    face_for_region = fb
                    n_face_fb += 1
            if not hands_for_region and pose_lms:
                hb = _bbox_from_pose_subset(pose_lms, POSE_HAND_IDX, w, h)
                if hb:
                    hands_for_region = [hb]
                    n_hand_fb += 1

            crop_raw = _union([pose_box, face_box])
            crop_box = _expand_to_square(crop_raw, w, h, margin=args.crop_margin) if crop_raw else None
            region_box = _union([face_for_region] + hands_for_region)

            n_crop += int(crop_box is not None)
            n_region += int(region_box is not None)

            items[path.stem] = {
                "image_wh": [w, h],
                "crop_bbox": crop_box,
                "region_bbox": region_box,
            }
    finally:
        pose.close()
        face.close()
        hands.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"items": items}))
    n = max(1, len(items))
    print(f"wrote {out_path}: {len(items)} items")
    print(f"  pose: {n_pose} ({100*n_pose/n:.1f}%)  face: {n_face} ({100*n_face/n:.1f}%)  "
          f"hands: {n_hand} ({100*n_hand/n:.1f}%)")
    print(f"  pose-landmark fallback used: face={n_face_fb} ({100*n_face_fb/n:.1f}%) "
          f"hands={n_hand_fb} ({100*n_hand_fb/n:.1f}%)")
    print(f"  crop_bbox set: {n_crop} ({100*n_crop/n:.1f}%)  "
          f"region_bbox set: {n_region} ({100*n_region/n:.1f}%)")


if __name__ == "__main__":
    main()
