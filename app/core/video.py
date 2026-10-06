"""Visual pipeline: frame sampling, face detection and per-frame scoring."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .config import DEVICE, VIDEO_FRAME_COUNT
from .models import VIDEO_TRANSFORM, ModelBundle


def get_sample_indices(total_frames: int, count: int = VIDEO_FRAME_COUNT) -> list[int]:
    """Spread ``count`` samples evenly across the clip."""
    if total_frames <= 0:
        return []

    if total_frames <= count:
        return list(range(total_frames))

    return np.linspace(0, total_frames - 1, num=count, dtype=int).tolist()


def detect_face_crop(frame, face_model):
    """Return the detected face crop, or ``None`` when there is no face."""
    if face_model is None:
        return None

    results = face_model(frame, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return None

    x1, y1, x2, y2 = (int(v) for v in boxes[0].xyxy[0])
    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return face


def _score_frames(video_path: Path, indices, models: ModelBundle):
    """Score every sampled frame individually."""
    face_model = models.face_model

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")

    fake_scores: list[float] = []
    frame_details: list[dict] = []
    face_crops: list[tuple[int, np.ndarray]] = []
    faces_detected = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        region = detect_face_crop(frame, face_model)
        cropped = region is not None

        if region is None:
            region = frame
        else:
            faces_detected += 1

        # cv2 decodes frames in BGR order, but the model was trained on
        # RGB images, so the channels must be swapped back first.
        rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        tensor = VIDEO_TRANSFORM(rgb).unsqueeze(0).to(DEVICE)

        # Keep a downscaled copy for the evidence gallery in the UI.
        if cropped:
            gallery = cv2.resize(rgb, (160, 160)) if rgb.shape[0] > 4 else rgb
            face_crops.append((int(idx), gallery))

        logits = models.video_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

        fake_index = int(models.video_classes.get("fake", 0))
        real_index = int(models.video_classes.get("real", 1))

        fake_prob = float(probs[fake_index].item())
        real_prob = float(probs[real_index].item())

        frame_details.append(
            {
                "frame": int(idx),
                "prediction": "FAKE" if fake_prob >= real_prob else "REAL",
                "real": real_prob,
                "fake": fake_prob,
                "confidence": max(fake_prob, real_prob),
            }
        )
        fake_scores.append(fake_prob)

    cap.release()

    return fake_scores, frame_details, faces_detected, face_crops


@torch.inference_mode()
def predict_video(video_path: Path, models: ModelBundle) -> dict[str, Any]:
    """Run the full visual branch over one video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    if total_frames <= 0:
        raise RuntimeError("Video contains no readable frames.")

    indices = get_sample_indices(total_frames)
    fake_scores, frame_details, faces_detected, face_crops = _score_frames(
        video_path, indices, models
    )

    if not fake_scores:
        raise RuntimeError("No valid frames could be analyzed.")

    fake_probability = float(np.mean(fake_scores))
    real_probability = 1.0 - fake_probability
    prediction = "FAKE" if fake_probability >= 0.5 else "REAL"

    return {
        "prediction": prediction,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "confidence": max(real_probability, fake_probability),
        "frames_analyzed": len(frame_details),
        "faces_detected": faces_detected,
        "face_crop_used": faces_detected > 0,
        "frame_details": frame_details,
        # UI-only: RGB face crops keyed by frame index.
        "face_crops": face_crops,
        "total_frames": total_frames,
        "fps": fps,
        "duration": (total_frames / fps) if fps > 0 else None,
    }
