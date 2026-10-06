"""Central configuration for the multimodal deepfake detection pipeline.

Everything that is a path, a constant or an environment override lives here so
the rest of the package never hard-codes a location.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

# ============================================================
# PATHS
# ============================================================

# app/core/config.py -> app/ -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

APP_DIR = PROJECT_ROOT / "app"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TRAINING_DIR = PROJECT_ROOT / "training"
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
UTILS_DIR = PROJECT_ROOT / "utils"

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DIR = PROJECT_ROOT / "processed"
DATASET_DIR = PROJECT_ROOT / "dataset"

# Checkpoints used by the dashboard
VIDEO_CHECKPOINT = SAVED_MODELS_DIR / "video_binary_small.pth"
AUDIO_CHECKPOINT = SAVED_MODELS_DIR / "audio_binary_small.pth"
FACE_WEIGHTS = WEIGHTS_DIR / "yolov8n-face.pt"

# Evaluation artefacts consumed by the Model Metrics page
VIDEO_EVALUATION = RESULTS_DIR / "video_evaluation.txt"
CONFUSION_MATRIX = RESULTS_DIR / "6class_video_confusion_matrix.txt"
FUSION_EVALUATION = PROCESSED_DIR / "fusion_evaluation.txt"

# ============================================================
# ANALYSIS PARAMETERS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIDEO_FRAME_COUNT = 10
VIDEO_INPUT_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

AUDIO_SR = 16_000
AUDIO_SECONDS = 4
AUDIO_SAMPLES = AUDIO_SR * AUDIO_SECONDS
AUDIO_WINDOW_OVERLAP = 0.5

VIDEO_CLASSES = {"fake": 0, "real": 1}
AUDIO_CLASSES = {"real": 0, "fake": 1}

WAV2VEC_MODEL = "facebook/wav2vec2-base"

SUPPORTED_VIDEO_EXTENSIONS = ("mp4", "mov", "avi", "mkv")
MAX_BATCH_FILES = 40

# ============================================================
# FFMPEG DISCOVERY
# ============================================================

_KNOWN_FFMPEG_DIRS = (
    Path(r"D:\ffmpeg\ffmpeg-9.0.2-full_build-shared\bin"),
    Path(r"D:\ffmpeg\ffmpeg-2026-08-06-git-95c43d7df7-full_build\bin"),
    Path(r"C:\ffmpeg\bin"),
)


def find_ffmpeg() -> Path | None:
    """Locate the ffmpeg executable.

    Resolution order:
      1. ``FFMPEG_BINARY`` environment variable
      2. ``ffmpeg`` on PATH
      3. the known local install directories
    """
    override = os.environ.get("FFMPEG_BINARY")
    if override:
        candidate = Path(override)
        if candidate.exists():
            return candidate

    on_path = shutil.which("ffmpeg")
    if on_path:
        return Path(on_path)

    for directory in _KNOWN_FFMPEG_DIRS:
        candidate = directory / "ffmpeg.exe"
        if candidate.exists():
            return candidate

    return None


FFMPEG_EXE = find_ffmpeg()


def ffmpeg_status() -> str:
    """Human readable description of ffmpeg availability."""
    return str(FFMPEG_EXE) if FFMPEG_EXE else "not found"
