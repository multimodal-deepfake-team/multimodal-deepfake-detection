import os
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.video_model import DeepfakeResNet18
from models.classes import CLASSES


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = os.path.join(
    "saved_models",
    "social_media_detector.pth"
)

NUM_FRAMES = 8


class VideoDetector:
    def __init__(self):
        self.device = DEVICE

        self.model = DeepfakeResNet18(num_classes=6)

        state_dict = torch.load(
            CHECKPOINT,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _read_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            cap.release()
            raise RuntimeError(
                "Video contains no readable frames."
            )

        indices = torch.linspace(
            0,
            total_frames - 1,
            steps=min(NUM_FRAMES, total_frames)
        ).long().tolist()

        frames = []

        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)

            ok, frame = cap.read()

            if not ok:
                continue

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(
                Image.fromarray(frame)
            )

        cap.release()

        if not frames:
            raise RuntimeError(
                "Could not extract any video frames."
            )

        return frames

    @torch.no_grad()
    def predict(self, video_path):
        frames = self._read_frames(video_path)

        batch = torch.stack([
            self.transform(frame)
            for frame in frames
        ]).to(self.device)

        logits = self.model(batch)

        probabilities = F.softmax(
            logits,
            dim=1
        )

        mean_probabilities = probabilities.mean(
            dim=0
        )

        predicted_index = int(
            torch.argmax(mean_probabilities).item()
        )

        predicted_class = CLASSES[predicted_index]

        real_probability = float(
            mean_probabilities[0].item()
        )

        visual_fake_score = 1.0 - real_probability

        class_probabilities = {
            CLASSES[i]: float(
                mean_probabilities[i].item()
            )
            for i in range(len(CLASSES))
        }

        return {
            "predicted_class": predicted_class,
            "real_probability": real_probability,
            "fake_probability": visual_fake_score,
            "class_probabilities": class_probabilities,
            "frames_analyzed": len(frames)
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/video_detector.py <video>"
        )
        raise SystemExit(1)

    detector = VideoDetector()

    result = detector.predict(sys.argv[1])

    print("\n==============================")
    print("VIDEO DETECTION RESULT")
    print("==============================")

    print(
        f"Predicted class : "
        f"{result['predicted_class']}"
    )

    print(
        f"Real probability: "
        f"{result['real_probability']:.4f}"
    )

    print(
        f"Fake probability: "
        f"{result['fake_probability']:.4f}"
    )

    print(
        f"Frames analyzed : "
        f"{result['frames_analyzed']}"
    )

    print("\nClass probabilities:")

    for name, probability in result[
        "class_probabilities"
    ].items():
        print(
            f"  {name:20s}: "
            f"{probability:.4f}"
        )