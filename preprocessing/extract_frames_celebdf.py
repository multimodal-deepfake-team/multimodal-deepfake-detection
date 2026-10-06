from pathlib import Path
import cv2

ROOT = Path("dataset/celebdf")
OUT = Path("processed/celebdf_frames")

CLASSES = {
    "Celeb-real": "real",
    "Celeb-synthesis": "fake",
}

FRAME_STEP = 10


def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return 0

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % FRAME_STEP == 0:
            output_name = f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_dir / output_name), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for source_class, target_class in CLASSES.items():
        source_dir = ROOT / source_class
        output_dir = OUT / target_class
        output_dir.mkdir(parents=True, exist_ok=True)

        videos = list(source_dir.rglob("*.mp4"))

        print(f"\n{source_class} -> {target_class}")
        print(f"Videos: {len(videos)}")

        total_frames = 0

        for i, video in enumerate(videos, 1):
            saved = extract_frames(video, output_dir)
            total_frames += saved

            if i % 50 == 0 or i == len(videos):
                print(
                    f"Processed {i}/{len(videos)} videos | "
                    f"Frames: {total_frames}"
                )

        print(f"Total frames: {total_frames}")


if __name__ == "__main__":
    main()