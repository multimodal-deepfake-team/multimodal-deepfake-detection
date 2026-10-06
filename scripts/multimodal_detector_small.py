from pathlib import Path
import sys
import subprocess
import tempfile

import cv2
import torch

from video_detector_small import VideoDetectorSmall
from audio_detector_small import AudioDetectorSmall


ROOT = Path(__file__).resolve().parents[1]

VIDEO_WEIGHT = 0.60
AUDIO_WEIGHT = 0.40
THRESHOLD = 0.50

FFMPEG = Path(r"D:\ffmpeg\ffmpeg-9.0.2-full_build-shared\bin\ffmpeg.exe")


class MultimodalDetector:

    def __init__(self):
        print("=" * 70)
        print("FINAL MULTIMODAL DEEPFAKE DETECTOR")
        print("=" * 70)

        print()
        print("Loading video detector...")
        self.video_detector = VideoDetectorSmall()

        print()
        print("Loading audio detector...")
        self.audio_detector = AudioDetectorSmall()

        print()
        print("Fusion weights:")
        print(f"  Video: {VIDEO_WEIGHT:.0%}")
        print(f"  Audio: {AUDIO_WEIGHT:.0%}")
        print(f"  Threshold: {THRESHOLD:.2f}")

    def extract_video_frames(self, video_path, max_frames=10):

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25.0

        duration = total_frames / fps if total_frames > 0 else 0

        print()
        print("Video information:")
        print("  Total frames:", total_frames)
        print(f"  FPS: {fps:.2f}")
        print(f"  Duration: {duration:.2f} sec")

        if total_frames <= 0:
            cap.release()
            raise RuntimeError("Video contains no readable frames.")

        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            indices = [
                int(x)
                for x in torch.linspace(
                    0,
                    total_frames - 1,
                    max_frames
                ).tolist()
            ]

        frames = []

        for index in indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                index
            )

            success, frame = cap.read()

            if not success:
                continue

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(
                (index, frame_rgb)
            )

        cap.release()

        if not frames:
            raise RuntimeError(
                "Could not extract any frames from video."
            )

        return frames

    def predict_video(self, video_path):

        frames = self.extract_video_frames(
            video_path,
            max_frames=10
        )

        print()
        print(
            f"Running video detector on {len(frames)} frames..."
        )

        fake_scores = []

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_dir = Path(temp_dir)

            for i, (frame_index, frame) in enumerate(frames):

                image_path = temp_dir / f"frame_{i:03d}.jpg"

                frame_bgr = cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR
                )

                cv2.imwrite(
                    str(image_path),
                    frame_bgr
                )

                result = self.video_detector.predict(
                    image_path
                )

                fake_scores.append(
                    result["fake_probability"]
                )

                print(
                    f"  Frame {frame_index:6d} | "
                    f"Fake: {result['fake_probability']:.4f}"
                )

        video_fake = sum(fake_scores) / len(fake_scores)

        video_real = 1.0 - video_fake

        return {
            "fake_probability": video_fake,
            "real_probability": video_real,
            "frames_used": len(fake_scores)
        }

    def extract_audio(self, video_path, output_wav):

        if not FFMPEG.exists():
            raise FileNotFoundError(
                f"FFmpeg not found:\n{FFMPEG}"
            )

        command = [
            str(FFMPEG),
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_wav)
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg audio extraction failed:\n"
                + result.stderr[-3000:]
            )

        if not output_wav.exists():
            raise RuntimeError(
                "FFmpeg completed but no audio file was created."
            )

    def predict_audio(self, video_path):

        print()
        print("Extracting audio from video...")

        with tempfile.TemporaryDirectory() as temp_dir:

            audio_path = Path(temp_dir) / "audio.wav"

            self.extract_audio(
                video_path,
                audio_path
            )

            result = self.audio_detector.predict(
                audio_path
            )

        return result

    def predict(self, video_path):

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found:\n{video_path}"
            )

        print()
        print("=" * 70)
        print("PROCESSING VIDEO")
        print("=" * 70)
        print("File:", video_path)

        video_result = self.predict_video(
            video_path
        )

        try:
            audio_result = self.predict_audio(
                video_path
            )

            audio_available = True

        except Exception as e:

            print()
            print("WARNING: Audio processing failed.")
            print(str(e))

            audio_result = {
                "fake_probability": 0.0,
                "real_probability": 1.0
            }

            audio_available = False

        video_fake = video_result["fake_probability"]

        audio_fake = audio_result["fake_probability"]

        if audio_available:

            final_score = (
                VIDEO_WEIGHT * video_fake
                +
                AUDIO_WEIGHT * audio_fake
            )

        else:

            final_score = video_fake

        final_label = (
            "FAKE"
            if final_score >= THRESHOLD
            else "REAL"
        )

        return {
            "video": video_result,
            "audio": audio_result,
            "audio_available": audio_available,
            "final_score": final_score,
            "final_label": final_label
        }


def print_report(video_path, result):

    print()
    print("=" * 70)
    print("FINAL MULTIMODAL FORENSIC RESULT")
    print("=" * 70)

    print("File:")
    print(video_path)

    print()
    print("VIDEO ANALYSIS")
    print("-" * 70)
    print(
        f"Frames used: "
        f"{result['video']['frames_used']}"
    )
    print(
        f"Real probability: "
        f"{result['video']['real_probability']:.4f}"
    )
    print(
        f"Fake probability: "
        f"{result['video']['fake_probability']:.4f}"
    )

    print()
    print("AUDIO ANALYSIS")
    print("-" * 70)

    if result["audio_available"]:

        print(
            f"Real probability: "
            f"{result['audio']['real_probability']:.4f}"
        )
        print(
            f"Fake probability: "
            f"{result['audio']['fake_probability']:.4f}"
        )

    else:

        print("Audio unavailable.")

    print()
    print("LATE FUSION")
    print("-" * 70)

    print(
        f"Video contribution: "
        f"{VIDEO_WEIGHT:.0%} × "
        f"{result['video']['fake_probability']:.4f}"
    )

    if result["audio_available"]:

        print(
            f"Audio contribution: "
            f"{AUDIO_WEIGHT:.0%} × "
            f"{result['audio']['fake_probability']:.4f}"
        )

    print(
        f"Final fake score: "
        f"{result['final_score']:.4f}"
    )

    print()
    print("=" * 70)
    print(
        f"FINAL PREDICTION: "
        f"{result['final_label']}"
    )
    print("=" * 70)


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print(
            "python scripts/multimodal_detector_small.py "
            "<video_file>"
        )
        print()

        sys.exit(1)

    video_path = Path(sys.argv[1])

    detector = MultimodalDetector()

    result = detector.predict(
        video_path
    )

    print_report(
        video_path,
        result
    )
