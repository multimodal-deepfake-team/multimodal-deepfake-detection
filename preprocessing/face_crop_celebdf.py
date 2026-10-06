from pathlib import Path
from ultralytics import YOLO
import cv2

INPUT_ROOT = Path("processed/celebdf_frames")
OUTPUT_ROOT = Path("processed/celebdf_faces")

MODEL_PATH = Path("weights/yolov8n-face.pt")

model = YOLO(str(MODEL_PATH))


def process_class(class_name):
    input_dir = INPUT_ROOT / class_name
    output_dir = OUTPUT_ROOT / class_name

    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(input_dir.glob("*.jpg"))

    print(f"\n{class_name}")
    print(f"Images: {len(images)}")

    saved = 0

    for i, image_path in enumerate(images, 1):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        results = model(
            image,
            verbose=False
        )

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            continue

        # Select the largest detected face
        best_box = None
        best_area = 0

        for box in boxes.xyxy:

            x1, y1, x2, y2 = map(
                int,
                box.tolist()
            )

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            area = width * height

            if area > best_area:
                best_area = area
                best_box = (
                    x1,
                    y1,
                    x2,
                    y2
                )

        if best_box is None:
            continue

        x1, y1, x2, y2 = best_box

        h, w = image.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        face = image[y1:y2, x1:x2]

        if face.size == 0:
            continue

        output_path = (
            output_dir /
            image_path.name
        )

        cv2.imwrite(
            str(output_path),
            face
        )

        saved += 1

        if i % 1000 == 0 or i == len(images):
            print(
                f"Processed {i}/{len(images)} | "
                f"Faces saved: {saved}"
            )

    print(
        f"{class_name} complete: "
        f"{saved} faces"
    )


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    process_class("real")
    process_class("fake")

    print("\nCeleb-DF face extraction complete.")


if __name__ == "__main__":
    main()