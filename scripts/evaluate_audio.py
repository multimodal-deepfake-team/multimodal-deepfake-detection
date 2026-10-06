import sys
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn

from transformers import Wav2Vec2Model
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\Final\multimodal_deepfake_detection")

DATA = ROOT / "dataset" / "audio_subset" / "val"
MODEL_PATH = ROOT / "saved_models" / "audio_classifier.pth"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "audio_evaluation.txt"


# ============================================================
# CONFIG
# ============================================================

SR = 16000
AUDIO_LENGTH = 16000

CLASS_NAMES = [
    "real",
    "tts",
    "vc",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 2


# ============================================================
# MODEL
# ============================================================

class AudioExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.proj = nn.Linear(
            768,
            256
        )

    def forward(self, audio):

        out = self.wav2vec(
            audio
        )

        feat = out.last_hidden_state.mean(
            dim=1
        )

        return self.proj(feat)


class AudioClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.extractor = AudioExtractor()

        self.classifier = nn.Linear(
            256,
            3
        )

    def forward(self, audio):

        feat = self.extractor(
            audio
        )

        return self.classifier(
            feat
        )


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path):

    audio, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    audio = audio.astype(
        np.float32
    )

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Peak normalization
    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:
        audio = audio / peak

    # Fixed 1-second input
    if len(audio) < AUDIO_LENGTH:

        audio = np.pad(
            audio,
            (
                0,
                AUDIO_LENGTH - len(audio)
            )
        )

    else:

        audio = audio[
            :AUDIO_LENGTH
        ]

    return torch.tensor(
        audio,
        dtype=torch.float32
    )


# ============================================================
# COLLECT VALIDATION FILES
# ============================================================

audio_files = []

for label, class_name in enumerate(CLASS_NAMES):

    class_dir = DATA / class_name

    if not class_dir.exists():

        raise FileNotFoundError(
            f"Missing directory: {class_dir}"
        )

    files = sorted(
        class_dir.rglob("*.wav")
    )

    print(
        f"{class_name.upper():<8}: "
        f"{len(files)} samples"
    )

    for file in files:

        audio_files.append(
            (
                file,
                label
            )
        )


print()
print("======================================")
print("AUDIO MODEL EVALUATION")
print("======================================")

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(
    f"Loaded {len(audio_files)} validation samples"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("Loading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

if "extractor" not in checkpoint:

    raise RuntimeError(
        "Checkpoint does not contain 'extractor'"
    )

if "classifier" not in checkpoint:

    raise RuntimeError(
        "Checkpoint does not contain 'classifier'"
    )


model = AudioClassifier().to(
    DEVICE
)

model.extractor.load_state_dict(
    checkpoint["extractor"]
)

model.classifier.load_state_dict(
    checkpoint["classifier"]
)

model.eval()

print(
    "Checkpoint loaded successfully."
)


# ============================================================
# EVALUATION
# ============================================================

all_labels = []
all_preds = []
all_probs = []
all_paths = []


print()
print("Running inference...")


with torch.inference_mode():

    for start in range(
        0,
        len(audio_files),
        BATCH_SIZE
    ):

        batch_items = audio_files[
            start:start + BATCH_SIZE
        ]

        audios = []
        labels = []
        paths = []

        for path, label in batch_items:

            audio = load_audio(
                path
            )

            audios.append(
                audio
            )

            labels.append(
                label
            )

            paths.append(
                path
            )

        audio_batch = torch.stack(
            audios
        ).to(DEVICE)

        labels_tensor = torch.tensor(
            labels,
            dtype=torch.long,
            device=DEVICE
        )

        logits = model(
            audio_batch
        )

        probs = torch.softmax(
            logits,
            dim=1
        )

        preds = torch.argmax(
            probs,
            dim=1
        )

        all_labels.extend(
            labels_tensor.cpu().numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

        all_paths.extend(
            [str(p) for p in paths]
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_preds
)

precision, recall, macro_f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )
)

_, _, weighted_f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )
)

report = classification_report(
    all_labels,
    all_preds,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_preds
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("======================================")
print("FINAL AUDIO EVALUATION")
print("======================================")

print(
    f"Samples evaluated : {len(all_labels)}"
)

print(
    f"Accuracy          : {accuracy:.4f}"
)

print(
    f"Macro F1          : {macro_f1:.4f}"
)

print(
    f"Weighted F1       : {weighted_f1:.4f}"
)

print()
print("Classification Report:")
print(
    report
)

print(
    "Confusion Matrix:"
)

print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "AUDIO MODEL EVALUATION\n"
    )

    f.write(
        "======================\n\n"
    )

    f.write(
        f"Samples evaluated : "
        f"{len(all_labels)}\n"
    )

    f.write(
        f"Accuracy          : "
        f"{accuracy:.4f}\n"
    )

    f.write(
        f"Macro F1          : "
        f"{macro_f1:.4f}\n"
    )

    f.write(
        f"Weighted F1       : "
        f"{weighted_f1:.4f}\n\n"
    )

    f.write(
        "Classification Report:\n"
    )

    f.write(
        report
    )

    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\nPer-sample predictions:\n"
    )

    for path, true_label, pred, probs in zip(
        all_paths,
        all_labels,
        all_preds,
        all_probs
    ):

        confidence = float(
            probs[pred]
        )

        f.write(
            f"{path} | "
            f"true={CLASS_NAMES[true_label]} | "
            f"pred={CLASS_NAMES[pred]} | "
            f"confidence={confidence:.4f}\n"
        )


print()
print(
    f"Results saved to: {OUTPUT_FILE}"
)

print()
print("AUDIO EVALUATION COMPLETE.")