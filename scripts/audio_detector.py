import os
import sys
from pathlib import Path

if sys.platform == "win32":
    FFMPEG_DLL_DIR = Path(
        r"D:\ffmpeg\ffmpeg-9.0.2-full_build-shared\bin"
    )

    if FFMPEG_DLL_DIR.exists():
        os.add_dll_directory(str(FFMPEG_DLL_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from models.audio_extractor import AudioExtractor


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = os.path.join(
    "saved_models",
    "audio_classifier.pth"
)

SAMPLE_RATE = 16000
AUDIO_LENGTH = 16000

CLASSES = {
    0: "real",
    1: "voiceclone",
    2: "audio_video_fake"
}


class AudioDetector:

    def __init__(self):

        self.device = DEVICE

        # Same architecture used during training
        self.extractor = AudioExtractor()

        self.classifier = nn.Linear(
            256,
            3
        )

        checkpoint = torch.load(
            CHECKPOINT,
            map_location=self.device,
            weights_only=True
        )

        self.extractor.load_state_dict(
            checkpoint["extractor"]
        )

        self.classifier.load_state_dict(
            checkpoint["classifier"]
        )

        self.extractor.to(self.device)
        self.classifier.to(self.device)

        self.extractor.eval()
        self.classifier.eval()

    def load_audio(self, audio_path):

        waveform, sample_rate = torchaudio.load(
            audio_path
        )

        # Convert stereo/multi-channel to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True
            )

        # Resample if necessary
        if sample_rate != SAMPLE_RATE:

            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=SAMPLE_RATE
            )

            waveform = resampler(waveform)

        # Same preprocessing used by training:
        # exactly 16000 samples
        waveform = waveform[:, :AUDIO_LENGTH]

        if waveform.shape[1] < AUDIO_LENGTH:

            padding = AUDIO_LENGTH - waveform.shape[1]

            waveform = F.pad(
                waveform,
                (0, padding)
            )

        # Remove channel dimension
        waveform = waveform.squeeze(0)

        return waveform

    @torch.no_grad()
    def predict(self, audio_path):

        waveform = self.load_audio(
            audio_path
        )

        waveform = waveform.unsqueeze(0)

        waveform = waveform.to(
            self.device
        )

        features = self.extractor(
            waveform
        )

        logits = self.classifier(
            features
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

        predicted_class = CLASSES[
            predicted_index
        ]

        class_probabilities = {
            CLASSES[i]: float(
                probabilities[i].item()
            )
            for i in range(3)
        }

        # Binary audio fake score.
        # Both voiceclone and audio_video_fake
        # are treated as synthetic/fake audio.
        real_probability = class_probabilities[
            "real"
        ]

        audio_fake_score = (
            1.0 - real_probability
        )

        return {
            "predicted_class": predicted_class,
            "real_probability": real_probability,
            "fake_probability": audio_fake_score,
            "class_probabilities": class_probabilities
        }


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python -m scripts.audio_detector <audio>"
        )

        raise SystemExit(1)

    audio_path = sys.argv[1]

    if not os.path.isfile(audio_path):

        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    detector = AudioDetector()

    result = detector.predict(
        audio_path
    )

    print()
    print("==============================")
    print("AUDIO DETECTION RESULT")
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

    print()
    print("Class probabilities:")

    for name, probability in result[
        "class_probabilities"
    ].items():

        print(
            f"  {name:20s}: "
            f"{probability:.4f}"
        )