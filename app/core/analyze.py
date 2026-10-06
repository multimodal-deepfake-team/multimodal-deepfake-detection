"""End-to-end orchestration: one video in, one full analysis dict out.

The UI layer never talks to the models directly - it calls
:func:`analyze_video` and renders the returned structure.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from .audio import extract_audio_from_video, predict_audio
from .models import ModelBundle
from .verdict import build_multimodal_status, create_text_report
from .video import predict_video


def analyze_video(
    video_path: Path,
    models: ModelBundle,
    work_dir: Path | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """Run both branches over ``video_path`` and return a full result dict.

    The returned mapping is JSON-serialisable apart from ``work_dir`` so it can
    be cached, serialised for batch runs, or rendered by any UI.
    """
    started = time.perf_counter()

    video_path = Path(video_path)
    display_name = filename or video_path.name

    owns_dir = work_dir is None
    work_dir = Path(work_dir) if work_dir else Path(
        tempfile.mkdtemp(prefix="deepfake_")
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    wav_path = work_dir / f"{video_path.stem}.wav"

    try:
        video_result = predict_video(video_path, models)

        audio_ok, audio_message = extract_audio_from_video(video_path, wav_path)
        audio_result = predict_audio(wav_path, models) if audio_ok else None

        status, conclusion, risk = build_multimodal_status(
            video_result["prediction"],
            audio_result["prediction"] if audio_result else None,
        )

        report = create_text_report(
            display_name, video_result, audio_result, status, conclusion, risk
        )

        # Read the extracted audio into memory so the result survives the
        # temporary work directory being cleaned up.
        wav_bytes = None
        if audio_ok and wav_path.exists():
            wav_bytes = wav_path.read_bytes()

        return {
            "filename": display_name,
            "video": video_result,
            "audio": audio_result,
            "audio_ok": audio_ok,
            "audio_message": audio_message,
            "status": status,
            "conclusion": conclusion,
            "risk": risk,
            "report": report,
            "wav_bytes": wav_bytes,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
    finally:
        if owns_dir:
            # Keep nothing behind when the caller did not ask for a work dir.
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten one analysis result into a single row (used by batch mode)."""
    video = result["video"]
    audio = result.get("audio") or {}

    return {
        "file": result["filename"],
        "status": result["status"],
        "risk": result["risk"],
        "video_prediction": video["prediction"],
        "video_fake_pct": round(video["fake_probability"] * 100, 2),
        "audio_prediction": audio.get("prediction", "UNAVAILABLE"),
        "audio_fake_pct": (
            round(audio["fake_probability"] * 100, 2) if audio else None
        ),
        "frames": video["frames_analyzed"],
        "faces": video["faces_detected"],
        "seconds": result["elapsed_seconds"],
    }
