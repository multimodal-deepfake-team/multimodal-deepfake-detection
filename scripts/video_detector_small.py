from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "saved_models" / "video_binary_small.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VideoDetectorSmall:
    def __init__(self):
        print("=" * 60)
        print("FINAL VIDEO DETECTOR")
        print("=" * 60)
        print("Device:", DEVICE)

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=False
        )

        self.classes = checkpoint["class_to_idx"]

        print("Classes:", self.classes)
        print(
            "Best validation accuracy:",
            f"{checkpoint['best_val_accuracy'] * 100:.2f}%"
        )

        self.model = resnet18(weights=None)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(DEVICE)
        self.model.eval()

        input_size = checkpoint.get("input_size", 224)

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("Video detector loaded successfully.")

    @torch.no_grad()
    def predict(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)
        image = image.unsqueeze(0).to(DEVICE)

        logits = self.model(image)

        probabilities = torch.softmax(logits, dim=1)[0]

        fake_index = self.classes["fake"]
        real_index = self.classes["real"]

        fake_prob = probabilities[fake_index].item()
        real_prob = probabilities[real_index].item()

        label = "FAKE" if fake_prob >= 0.5 else "REAL"

        return {
            "label": label,
            "fake_probability": fake_prob,
            "real_probability": real_prob
        }


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print()
        print("Usage:")
        print("python scripts/video_detector_small.py <image_file>")
        print()
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print("ERROR: Image file not found:")
        print(image_path)
        sys.exit(1)

    detector = VideoDetectorSmall()

    result = detector.predict(image_path)

    print()
    print("=" * 60)
    print("VIDEO RESULT")
    print("=" * 60)
    print("File:", image_path)
    print("Prediction:", result["label"])
    print(f"Real probability: {result['real_probability']:.4f}")
    print(f"Fake probability: {result['fake_probability']:.4f}")
    print("=" * 60)
