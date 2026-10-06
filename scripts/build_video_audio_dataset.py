"""Build a labelled audio dataset from the project's own videos.

Why: audio_binary_small.pth was trained on ASVspoof2019 LA (clean studio
speech) but the app scores audio extracted from videos, so it was badly
out of domain -- it labelled every FakeAVCeleb clip as FAKE.

Ground truth comes from the FakeAVCeleb folder names (audio-video pairs):

    RaV  real audio, real video   dataset/real/fakeav_rarv              -> real
    RaFv real audio, fake video   dataset/video_fake_audio_real         -> real
    FaV  fake audio, real video   dataset/voiceclone/fakeav_farv        -> fake
    FfV  fake audio, fake video   dataset/audio_video_fake/fakeav_fafv  -> fake

FFPP / Celeb-DF videos carry no audio stream at all, so they are skipped.

Splits are grouped by source video id (the leading number of the file
name) so that 00001.mp4 and 00001_fake.mp4 can never land in different
splits -- otherwise the model would leak speakers across the split.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "processed" / "audio_v2"

FFMPEG_EXE = Path(
    r"D:\ffmpeg\ffmpeg-9.0.2-full_build-shared\bin\ffmpeg.exe"
)

SOURCES = {
    "real": [
        ROOT / "dataset" / "real" / "fakeav_rarv",
        ROOT / "dataset" / "video_fake_audio_real" / "fakeav_rafv",
    ],
    "fake": [
        ROOT / "dataset" / "voiceclone" / "fakeav_farv",
        ROOT / "dataset" / "audio_video_fake" / "fakeav_fafv",
    ],
}

SAMPLE_RATE = 16000
TEST_FRACTION = 0.25
SEED = 42


def group_key(path: Path) -> str:
    """Source-video id, so the same video never straddles train/test."""
    stem = path.stem
    m = re.match(r"^(\d+)", stem)
    return m.group(1) if m else stem


def split_of(key: str) -> str:
    digest = hashlib.md5(f"{SEED}:{key}".encode()).hexdigest()
    return "test" if (int(digest, 16) % 100) < TEST_FRACTION * 100 else "train"


def extract(video: Path, wav: Path) -> bool:
    cmd = [
        str(FFMPEG_EXE), "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-vn", "-ac", "1",
        "-ar", str(SAMPLE_RATE), "-y", str(wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return wav.exists() and wav.stat().st_size > 0


def main() -> int:
    if not FFMPEG_EXE.exists():
        print(f"ERROR: ffmpeg not found at {FFMPEG_EXE}")
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)

    for split in ("train", "test"):
        for label in ("real", "fake"):
            (OUT / split / label).mkdir(parents=True, exist_ok=True)

    manifest = []
    failures = 0

    for label, dirs in SOURCES.items():
        for folder in dirs:
            videos = sorted(
                p for p in folder.rglob("*")
                if p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov")
            )
            for video in videos:
                key = group_key(video)
                split = split_of(key)
                dest = OUT / split / label / f"{key}_{video.stem}.wav"
                if extract(video, dest):
                    manifest.append(
                        {
                            "file": str(dest.relative_to(ROOT)),
                            "label": label,
                            "split": split,
                            "group": key,
                            "source": str(video.relative_to(ROOT)),
                        }
                    )
                else:
                    failures += 1

    print("=" * 60)
    print("VIDEO AUDIO DATASET")
    print("=" * 60)

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    for split in ("train", "test"):
        for label in ("real", "fake"):
            n = sum(1 for m in manifest
                    if m["split"] == split and m["label"] == label)
            groups = len({m["group"] for m in manifest
                          if m["split"] == split and m["label"] == label})
            print(f"{split:5s} {label:5s}: {n:3d} clips "
                  f"({groups} source videos)")

    leak_train_test = (
        {m["group"] for m in manifest if m["split"] == "train"} &
        {m["group"] for m in manifest if m["split"] == "test"}
    )
    print(f"\nextract failures: {failures}")
    print(f"groups leaking across train/test: {len(leak_train_test)}")

    print("\nNOTE: processed/audio/ is contaminated -- 50 files in")
    print("processed/audio/audio_video_fake are byte-identical copies of")
    print("processed/audio/video_fake_audio_real (i.e. REAL audio).")
    print("This script re-extracts from dataset/ to avoid those labels.")

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
