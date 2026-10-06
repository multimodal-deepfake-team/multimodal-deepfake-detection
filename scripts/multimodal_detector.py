import os
import sys
from pathlib import Path

# --------------------------------------------------
# FFmpeg shared DLL path for Windows / TorchCodec
# --------------------------------------------------

if sys.platform == "win32":
    FFMPEG_DLL_DIR = Path(
        r"D:\ffmpeg\ffmpeg-9.0.2-full_build-shared\bin"
    )

    if FFMPEG_DLL_DIR.exists():
        os.add_dll_directory(str(FFMPEG_DLL_DIR))


import subprocess
import tempfile

from scripts.video_detector import VideoDetector
from scripts.audio_detector import AudioDetector


VIDEO_WEIGHT = 0.6
AUDIO_WEIGHT = 0.4
THRESHOLD = 0.5


def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        audio_path,
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main(video_path):

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    print()
    print("======================================")
    print("MULTIMODAL DEEPFAKE DETECTION")
    print("======================================")

    # --------------------------------------------------
    # VIDEO
    # --------------------------------------------------

    print()
    print("[1/3] Analyzing video...")

    video_detector = VideoDetector()
    video_result = video_detector.predict(video_path)

    video_fake_score = video_result["fake_probability"]

    print(
        f"Video class       : "
        f"{video_result['predicted_class']}"
    )

    print(
        f"Video fake score  : "
        f"{video_fake_score:.4f}"
    )

    # --------------------------------------------------
    # AUDIO EXTRACTION
    # --------------------------------------------------

    print()
    print("[2/3] Extracting audio...")

    with tempfile.TemporaryDirectory() as temp_dir:

        audio_path = os.path.join(
            temp_dir,
            "extracted_audio.wav"
        )

        extract_audio(
            video_path,
            audio_path
        )

        # --------------------------------------------------
        # AUDIO
        # --------------------------------------------------

        print("[2/3] Analyzing audio...")

        audio_detector = AudioDetector()
        audio_result = audio_detector.predict(
            audio_path
        )

    audio_fake_score = audio_result["fake_probability"]

    print(
        f"Audio class       : "
        f"{audio_result['predicted_class']}"
    )

    print(
        f"Audio fake score  : "
        f"{audio_fake_score:.4f}"
    )

    # --------------------------------------------------
    # LATE FUSION
    # --------------------------------------------------

    print()
    print("[3/3] Performing late fusion...")

    combined_fake_score = (
        VIDEO_WEIGHT * video_fake_score
        + AUDIO_WEIGHT * audio_fake_score
    )

    combined_real_score = (
        1.0 - combined_fake_score
    )

    if combined_fake_score >= THRESHOLD:
        final_result = "DEEPFAKE"
    else:
        final_result = "REAL"

    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------

    print()
    print("======================================")
    print("FINAL MULTIMODAL RESULT")
    print("======================================")

    print(
        f"Video fake score   : "
        f"{video_fake_score:.4f}"
    )

    print(
        f"Audio fake score   : "
        f"{audio_fake_score:.4f}"
    )

    print(
        f"Video weight       : "
        f"{VIDEO_WEIGHT:.2f}"
    )

    print(
        f"Audio weight       : "
        f"{AUDIO_WEIGHT:.2f}"
    )

    print()

    print(
        f"Combined fake score: "
        f"{combined_fake_score:.4f}"
    )

    print(
        f"Combined real score: "
        f"{combined_real_score:.4f}"
    )

    print()

    print(
        f"FINAL RESULT       : "
        f"{final_result}"
    )

    print("======================================")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: "
            "python -m scripts.multimodal_detector "
            "<video>"
        )
        raise SystemExit(1)

    main(sys.argv[1])