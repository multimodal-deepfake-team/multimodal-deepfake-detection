"""Audio pipeline: ffmpeg extraction, windowing and per-window scoring."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch

from .config import AUDIO_SAMPLES, AUDIO_SR, DEVICE
from .models import ModelBundle


def extract_audio_from_video(video_path: Path, wav_path: Path) -> tuple[bool, str]:
    """Extract a 16 kHz mono WAV track with ffmpeg.

    Returns ``(ok, message)`` so callers can surface the reason on failure.
    """
    from .config import FFMPEG_EXE

    if FFMPEG_EXE is None:
        return False, (
            "FFmpeg was not found on this machine. Install it and/or set the "
            "FFMPEG_BINARY environment variable to the full path of the ffmpeg "
            "binary (on Streamlit Community Cloud it comes from packages.txt)."
        )

    command = [
        str(FFMPEG_EXE),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SR),
        "-y",
        str(wav_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        message = result.stderr.strip() or "No audio stream could be extracted."
        return False, message

    if not wav_path.exists() or wav_path.stat().st_size == 0:
        return False, "No audio stream found in this video."

    return True, "Audio extracted successfully."


def prepare_audio(wav_path: Path) -> torch.Tensor:
    """16 kHz / mono / mean-centred / peak-normalised waveform."""
    waveform, sr = sf.read(str(wav_path), always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if sr != AUDIO_SR:
        waveform = librosa.resample(
            waveform, orig_sr=sr, target_sr=AUDIO_SR
        ).astype(np.float32)

    waveform = waveform - float(np.mean(waveform))

    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform = waveform / peak

    return torch.from_numpy(waveform.astype(np.float32))


def split_audio_windows(waveform: torch.Tensor) -> list[torch.Tensor]:
    """Cut the track into 4-second windows with 50% overlap.

    One centre crop can land on silence or background music, so the audio
    verdict is averaged over every window instead.
    """
    length = int(waveform.shape[0])

    if length <= AUDIO_SAMPLES:
        padded = torch.zeros(AUDIO_SAMPLES, dtype=torch.float32)
        padded[:length] = waveform
        return [padded]

    stride = int(AUDIO_SAMPLES * 0.5)
    starts = range(0, length - AUDIO_SAMPLES + 1, stride)

    return [waveform[start : start + AUDIO_SAMPLES] for start in starts]


@torch.inference_mode()
def predict_audio(wav_path: Path, models: ModelBundle) -> dict[str, Any]:
    """Score every window of one WAV file and average the probabilities."""
    waveform = prepare_audio(wav_path)
    windows = split_audio_windows(waveform)

    input_values = torch.stack(windows).to(DEVICE)
    outputs = models.wav2vec(input_values=input_values)
    embedding = outputs.last_hidden_state.mean(dim=1)

    logits = models.audio_model(embedding)
    probs = torch.softmax(logits, dim=1).mean(dim=0)

    real_index = int(models.audio_classes.get("real", 0))
    fake_index = int(models.audio_classes.get("fake", 1))

    real_probability = float(probs[real_index].item())
    fake_probability = float(probs[fake_index].item())

    return {
        "prediction": "FAKE" if fake_probability >= real_probability else "REAL",
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "confidence": max(real_probability, fake_probability),
        "windows": len(windows),
    }
