"""Verdict logic and forensic report generation.

No weighted 60/40 score is used: the video and audio branches are reported
independently and only then combined into a status.
"""

from __future__ import annotations

from typing import Any

from .config import AUDIO_SECONDS


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_multimodal_status(
    video_prediction: str,
    audio_prediction: str | None,
) -> tuple[str, str, str]:
    """Return ``(status, conclusion, risk)`` for the two branch predictions."""
    if audio_prediction is None:
        return (
            "VIDEO-ONLY",
            "Audio evidence unavailable; the report is based on visual analysis only.",
            "REVIEW REQUIRED",
        )

    if video_prediction == "FAKE" and audio_prediction == "FAKE":
        return (
            "CONSISTENT FAKE",
            "Video and audio independently indicate manipulation.",
            "HIGH",
        )

    if video_prediction == "REAL" and audio_prediction == "REAL":
        return (
            "CONSISTENT REAL",
            "Video and audio independently indicate authentic content.",
            "LOW",
        )

    return (
        "INCONCLUSIVE",
        "The video and audio models gave different results. "
        f"The video appears {video_prediction}, while the audio appears "
        f"{audio_prediction}. The combined result is therefore INCONCLUSIVE.",
        "REVIEW REQUIRED",
    )


def create_text_report(
    filename: str,
    video_result: dict[str, Any],
    audio_result: dict[str, Any] | None,
    status: str,
    conclusion: str,
    risk: str,
) -> str:
    """Build the downloadable plain-text forensic report."""
    lines = [
        "========================================",
        "MULTIMODAL DEEPFAKE FORENSIC REPORT",
        "========================================",
        "",
        f"File: {filename}",
        "",
        "VIDEO ANALYSIS",
        "-" * 40,
        f"Frames analyzed : {video_result['frames_analyzed']}",
        f"Faces detected  : {video_result['faces_detected']}",
        f"Prediction      : {video_result['prediction']}",
        f"Real probability: {pct(video_result['real_probability'])}",
        f"Fake probability: {pct(video_result['fake_probability'])}",
        "",
    ]

    if audio_result is not None:
        lines += [
            "AUDIO ANALYSIS",
            "-" * 40,
            f"Prediction      : {audio_result['prediction']}",
            f"Windows averaged: {audio_result['windows']}"
            f" x {AUDIO_SECONDS}s",
            f"Real probability: {pct(audio_result['real_probability'])}",
            f"Fake probability: {pct(audio_result['fake_probability'])}",
            "",
        ]
    else:
        lines += [
            "AUDIO ANALYSIS",
            "-" * 40,
            "Audio unavailable / no audio stream detected.",
            "",
        ]

    lines += [
        "MULTIMODAL ASSESSMENT",
        "-" * 40,
        f"Status          : {status}",
        f"Risk level      : {risk}",
        f"Conclusion      : {conclusion}",
        "",
        "NOTE",
        "-" * 40,
        "No weighted score is used for the final status.",
        "Video and audio probabilities are shown independently.",
        "",
        "========================================",
    ]

    return "\n".join(lines)
