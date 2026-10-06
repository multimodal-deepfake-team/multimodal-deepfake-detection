"""Model architectures and checkpoint loading.

The two architectures are defined here (they are intentionally tiny) and are
loaded from the checkpoints shipped in ``saved_models/``. Loading is kept free
of Streamlit so the same code works from scripts, tests and the web UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from transformers import Wav2Vec2Model
from ultralytics import YOLO

from .config import (
    AUDIO_CHECKPOINT,
    AUDIO_CLASSES,
    AUDIO_SR,
    AUDIO_SECONDS,
    DEVICE,
    FACE_WEIGHTS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VIDEO_CHECKPOINT,
    VIDEO_CLASSES,
    VIDEO_INPUT_SIZE,
    WAV2VEC_MODEL,
)


# ============================================================
# ARCHITECTURES
# ============================================================


class VideoBinaryModel(nn.Module):
    """ResNet18 backbone with a 2-way head (REAL / FAKE)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class AudioBinaryClassifier(nn.Module):
    """Small MLP head placed on top of the frozen Wav2Vec2 embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


VIDEO_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(VIDEO_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# ============================================================
# BUNDLE
# ============================================================


@dataclass
class ModelBundle:
    """Everything the pipeline needs at inference time."""

    video_model: nn.Module
    video_classes: dict[str, int]
    video_val: float
    wav2vec: Any
    audio_model: nn.Module
    audio_classes: dict[str, int]
    audio_val: float
    face_model: Any

    @property
    def device(self) -> torch.device:
        return DEVICE


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required model file is missing: {path}\n"
            "See README.md for how to obtain or rebuild it."
        )
    return path


def load_models(
    video_checkpoint: Path | None = None,
    audio_checkpoint: Path | None = None,
    face_weights: Path | None = None,
) -> ModelBundle:
    """Load the video model, audio model and face detector.

    Raises ``FileNotFoundError`` with an actionable message when a checkpoint
    is absent.
    """
    video_checkpoint = _require(video_checkpoint or VIDEO_CHECKPOINT)
    audio_checkpoint = _require(audio_checkpoint or AUDIO_CHECKPOINT)
    face_weights = _require(face_weights or FACE_WEIGHTS)

    # ---------- VIDEO ----------
    video_ckpt = torch.load(
        video_checkpoint, map_location=DEVICE, weights_only=False
    )

    video_model = VideoBinaryModel().to(DEVICE)
    video_model.backbone.load_state_dict(video_ckpt["model_state_dict"])
    video_model.eval()

    video_classes = video_ckpt.get("class_to_idx", dict(VIDEO_CLASSES))

    # ---------- AUDIO ----------
    audio_ckpt = torch.load(
        audio_checkpoint, map_location=DEVICE, weights_only=False
    )

    wav2vec = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL).to(DEVICE)
    for param in wav2vec.parameters():
        param.requires_grad = False
    wav2vec.eval()

    audio_model = AudioBinaryClassifier().to(DEVICE)
    audio_model.classifier.load_state_dict(audio_ckpt["classifier"])
    audio_model.eval()

    audio_classes = audio_ckpt.get("classes", dict(AUDIO_CLASSES))

    # ---------- FACE DETECTOR ----------
    # The video checkpoint was trained on face crops, so the same detector
    # used to build the training set is required at inference.
    face_model = YOLO(str(face_weights))

    return ModelBundle(
        video_model=video_model,
        video_classes=video_classes,
        video_val=float(video_ckpt.get("best_val_accuracy", 0.0)),
        wav2vec=wav2vec,
        audio_model=audio_model,
        audio_classes=audio_classes,
        audio_val=float(audio_ckpt.get("best_val_accuracy", 0.0)),
        face_model=face_model,
    )


def checkpoint_metadata() -> dict[str, Any]:
    """Cheap checkpoint info for the sidebar / metrics page (no weights loaded)."""
    info: dict[str, Any] = {
        "device": DEVICE.type,
        "video_checkpoint": VIDEO_CHECKPOINT.name,
        "audio_checkpoint": AUDIO_CHECKPOINT.name,
        "face_detector": FACE_WEIGHTS.name,
        "audio_window_seconds": AUDIO_SECONDS,
        "audio_sample_rate": AUDIO_SR,
        "exists": {},
    }

    for label, path in (
        ("video", VIDEO_CHECKPOINT),
        ("audio", AUDIO_CHECKPOINT),
        ("face", FACE_WEIGHTS),
    ):
        info["exists"][label] = path.exists()
        if path.exists():
            stat = path.stat()
            info[f"{label}_size_mb"] = round(stat.st_size / (1024 * 1024), 1)

    return info
