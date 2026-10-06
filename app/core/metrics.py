"""Parsers for the evaluation artefacts checked into ``results/``.

Nothing here touches a model - the page only reads the text reports that the
training/evaluation scripts already produced, so the metrics page is instant
and never needs a GPU.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import CONFUSION_MATRIX, FUSION_EVALUATION, VIDEO_EVALUATION

# ------------------------------------------------------------
# Reference: 6-class validation report.
# Source of truth is PROJECT_DOCUMENTATION.md, section 5.
# ------------------------------------------------------------
SIX_CLASS_VALIDATION: dict[str, Any] = {
    "source": "PROJECT_DOCUMENTATION.md §5 (18,586-image dataset)",
    "accuracy": 0.95,
    "samples": 3717,
    "macro_avg": {"precision": 0.87, "recall": 0.93, "f1": 0.89},
    "weighted_avg": {"precision": 0.95, "recall": 0.95, "f1": 0.95},
    "per_class": [
        {"class": "ai_video", "precision": 0.99, "recall": 0.99, "f1": 0.99, "support": 1571},
        {"class": "audio_video_fake", "precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 118},
        {"class": "edited", "precision": 0.87, "recall": 0.80, "f1": 0.83, "support": 313},
        {"class": "faceswap", "precision": 0.95, "recall": 0.97, "f1": 0.96, "support": 460},
        {"class": "real", "precision": 0.94, "recall": 0.92, "f1": 0.93, "support": 1202},
        {"class": "voiceclone", "precision": 0.49, "recall": 0.89, "f1": 0.63, "support": 53},
    ],
}

_CLASS_ROW = re.compile(
    r"^\s*([A-Za-z_][A-Za-z_ ]*?)\s+"
    r"(\d\.\d+|\d+)\s+(\d\.\d+|\d+)\s+(\d\.\d+|\d+)\s+(\d+)\s*$"
)
_SCALAR = re.compile(
    r"(?im)^(samples evaluated|accuracy|macro f1|weighted f1)\s*:\s*([0-9]*\.?[0-9]+)"
)
_MATRIX_BLOCK = re.compile(r"\[\s*\[.*?\]\s*\]", re.S)


def _read(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_report_rows(text: str) -> list[dict[str, Any]]:
    """Extract the classification-report rows from a report body."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _CLASS_ROW.match(line)
        if not match:
            continue
        name = match.group(1).strip()
        if name.lower() in {"precision", "accuracy"}:
            continue
        rows.append(
            {
                "class": name,
                "precision": float(match.group(2)),
                "recall": float(match.group(3)),
                "f1": float(match.group(4)),
                "support": int(match.group(5)),
            }
        )
    return rows


def _parse_scalars(text: str) -> dict[str, float]:
    return {key.lower(): float(value) for key, value in _SCALAR.findall(text)}


def _parse_matrix(text: str) -> list[list[int]] | None:
    block = _MATRIX_BLOCK.search(text)
    if not block:
        return None
    numbers = [int(n) for n in re.findall(r"-?\d+", block.group(0))]
    size = int(len(numbers) ** 0.5)
    if size * size != len(numbers) or size == 0:
        return None
    return [numbers[i * size : (i + 1) * size] for i in range(size)]


def load_confusion_matrix_6class() -> dict[str, Any] | None:
    """6-class confusion matrix written by the video evaluation run."""
    text = _read(CONFUSION_MATRIX)
    if text is None:
        return None

    matrix = _parse_matrix(text)
    labels = re.findall(r"^\s*(\d+)\s*:\s*([A-Za-z_]+)\s*$", text, re.M)
    names = [name for _, name in sorted(((int(i), n) for i, n in labels))]

    if matrix is None:
        return None

    if not names:
        names = [str(i) for i in range(len(matrix))]

    return {"labels": names, "matrix": matrix}


def load_video_evaluation() -> dict[str, Any] | None:
    """6-class / fusion-label video-only evaluation."""
    text = _read(VIDEO_EVALUATION)
    if text is None:
        return None

    return {
        "title": "Video-only ResNet18",
        "path": str(VIDEO_EVALUATION.relative_to(VIDEO_EVALUATION.parents[2])),
        "scalars": _parse_scalars(text),
        "rows": _parse_report_rows(text),
    }


def load_fusion_evaluation() -> dict[str, Any] | None:
    """Late-fusion video + audio evaluation."""
    text = _read(FUSION_EVALUATION)
    if text is None:
        return None

    return {
        "title": "Multimodal fusion",
        "path": str(FUSION_EVALUATION.relative_to(FUSION_EVALUATION.parents[2])),
        "scalars": _parse_scalars(text),
        "rows": _parse_report_rows(text),
    }


def load_all_metrics() -> dict[str, Any]:
    """Everything the metrics page needs, in one call."""
    return {
        "six_class_validation": SIX_CLASS_VALIDATION,
        "confusion_matrix": load_confusion_matrix_6class(),
        "video_evaluation": load_video_evaluation(),
        "fusion_evaluation": load_fusion_evaluation(),
        "available_files": {
            "confusion_matrix": CONFUSION_MATRIX.exists(),
            "video_evaluation": VIDEO_EVALUATION.exists(),
            "fusion_evaluation": FUSION_EVALUATION.exists(),
        },
    }
