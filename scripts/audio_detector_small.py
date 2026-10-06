from pathlib import Path
import torch
import torch.nn as nn
import soundfile as sf
from transformers import Wav2Vec2Model


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "saved_models" / "audio_binary_small.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
AUDIO_LENGTH = 64000


class AudioDetectorSmall:
    def __init__(self):
        print("=" * 60)
        print("FINAL AUDIO DETECTOR")
        print("=" * 60)
        print("Device:", DEVICE)

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=False
        )

        self.classes = checkpoint["classes"]

        print("Classes:", self.classes)
        print("Best validation accuracy:",
              f"{checkpoint['best_val_accuracy'] * 100:.2f}%")

        print("Loading Wav2Vec2...")

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        ).to(DEVICE)

        for param in self.wav2vec.parameters():
            param.requires_grad = False

        self.wav2vec.eval()

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

        self.classifier.load_state_dict(checkpoint["classifier"])
        self.classifier = self.classifier.to(DEVICE)
        self.classifier.eval()

        print("Audio detector loaded successfully.")

    def load_audio(self, path):
        audio, sr = sf.read(str(path), dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != SR:
            raise ValueError(
                f"Expected {SR} Hz audio, but got {sr} Hz: {path}"
            )

        audio = torch.from_numpy(audio)

        audio = audio - audio.mean()

        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak

        if len(audio) >= AUDIO_LENGTH:
            start = (len(audio) - AUDIO_LENGTH) // 2
            audio = audio[start:start + AUDIO_LENGTH]
        else:
            padded = torch.zeros(AUDIO_LENGTH)
            padded[:len(audio)] = audio
            audio = padded

        return audio

    @torch.no_grad()
    def predict(self, path):
        audio = self.load_audio(path)

        audio = audio.unsqueeze(0).to(DEVICE)

        features = self.wav2vec(
            audio
        ).last_hidden_state.mean(dim=1)

        logits = self.classifier(features)

        probabilities = torch.softmax(logits, dim=1)[0]

        real_prob = probabilities[self.classes["real"]].item()
        fake_prob = probabilities[self.classes["fake"]].item()

        label = "FAKE" if fake_prob >= 0.5 else "REAL"

        return {
            "label": label,
            "fake_probability": fake_prob,
            "real_probability": real_prob
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print()
        print("Usage:")
        print("python scripts/audio_detector_small.py <audio_file>")
        print()
        sys.exit(1)

    audio_path = Path(sys.argv[1])

    if not audio_path.exists():
        print("ERROR: Audio file not found:")
        print(audio_path)
        sys.exit(1)

    detector = AudioDetectorSmall()

    result = detector.predict(audio_path)

    print()
    print("=" * 60)
    print("AUDIO RESULT")
    print("=" * 60)
    print("File:", audio_path)
    print("Prediction:", result["label"])
    print(f"Real probability: {result['real_probability']:.4f}")
    print(f"Fake probability: {result['fake_probability']:.4f}")
    print("=" * 60)
