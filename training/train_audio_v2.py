from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

from models.audio_extractor import AudioExtractor


# ============================================================
# CONFIG
# ============================================================

CLASSES = {
    "real": 0,
    "voiceclone": 1,
    "audio_video_fake": 2
}

CLASS_NAMES = [
    "real",
    "voiceclone",
    "audio_video_fake"
]

DATASET_ROOT = Path("dataset/audio_subset")

OUTPUT_PATH = Path("saved_models/audio_classifier_v2.pth")

SAMPLE_RATE = 16000

# Use 4 seconds instead of the old 1-second input.
AUDIO_LENGTH = SAMPLE_RATE * 4

BATCH_SIZE = 4

EPOCHS = 8

LEARNING_RATE = 1e-4

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class AudioDataset(Dataset):

    def __init__(self, root, split):

        self.samples = []

        root = Path(root)
        split_root = root / split

        for cls_name, label in CLASSES.items():

            folder = split_root / cls_name

            if not folder.exists():
                continue

            for wav in sorted(folder.glob("*.wav")):
                self.samples.append((wav, label))

        print(
            f"{split.upper()} samples: {len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        wav_path, label = self.samples[idx]

        waveform, sr = torchaudio.load(wav_path)

        # Convert stereo -> mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != SAMPLE_RATE:

            waveform = torchaudio.functional.resample(
                waveform,
                sr,
                SAMPLE_RATE
            )

        waveform = waveform.squeeze(0)

        # ----------------------------------------------------
        # Select a 4-second segment
        # ----------------------------------------------------

        if waveform.shape[0] > AUDIO_LENGTH:

            # Random crop during training
            max_start = waveform.shape[0] - AUDIO_LENGTH

            start = random.randint(0, max_start)

            waveform = waveform[
                start:start + AUDIO_LENGTH
            ]

        elif waveform.shape[0] < AUDIO_LENGTH:

            padding = AUDIO_LENGTH - waveform.shape[0]

            waveform = F.pad(
                waveform,
                (0, padding)
            )

        # ----------------------------------------------------
        # Normalize waveform
        # ----------------------------------------------------

        waveform = waveform - waveform.mean()

        peak = waveform.abs().max()

        if peak > 0:
            waveform = waveform / peak

        return waveform, label


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("======================================")
print("AUDIO MODEL V2 TRAINING")
print("======================================")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# DATA
# ============================================================

train_dataset = AudioDataset(
    DATASET_ROOT,
    "train"
)

val_dataset = AudioDataset(
    DATASET_ROOT,
    "validation"
)

if len(train_dataset) == 0:
    raise RuntimeError(
        f"No training audio found in {DATASET_ROOT / 'train'}"
    )

if len(val_dataset) == 0:
    raise RuntimeError(
        f"No validation audio found in {DATASET_ROOT / 'validation'}"
    )


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = AudioExtractor().to(device)

# IMPORTANT:
# Keep pretrained Wav2Vec2 frozen.
for parameter in model.wav2vec.parameters():
    parameter.requires_grad = False

# Projection remains trainable.
for parameter in model.proj.parameters():
    parameter.requires_grad = True


classifier = nn.Linear(
    256,
    3
).to(device)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    list(model.proj.parameters()) +
    list(classifier.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    # Keep Wav2Vec2 in evaluation mode because it is frozen.
    model.wav2vec.eval()

    classifier.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for audio, labels in train_loader:

        audio = audio.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        # No gradient through frozen Wav2Vec2.
        with torch.no_grad():

            wav_features = model.wav2vec(
                audio
            ).last_hidden_state

            wav_features = wav_features.mean(
                dim=1
            )

        features = model.proj(
            wav_features
        )

        outputs = classifier(
            features
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    train_loss /= len(train_loader)

    train_accuracy = (
        train_correct / train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()
    classifier.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    class_correct = [0] * 3
    class_total = [0] * 3

    with torch.no_grad():

        for audio, labels in val_loader:

            audio = audio.to(device)
            labels = labels.to(device)

            wav_features = model.wav2vec(
                audio
            ).last_hidden_state

            wav_features = wav_features.mean(
                dim=1
            )

            features = model.proj(
                wav_features
            )

            outputs = classifier(
                features
            )

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

            for class_index in range(3):

                mask = labels == class_index

                class_total[class_index] += (
                    mask.sum().item()
                )

                class_correct[class_index] += (
                    (
                        predictions[mask]
                        == labels[mask]
                    ).sum().item()
                )


    val_loss /= len(val_loader)

    val_accuracy = (
        val_correct / val_total
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

    for i, name in enumerate(CLASS_NAMES):

        if class_total[i] > 0:

            accuracy = (
                class_correct[i]
                / class_total[i]
            )

            print(
                f"  {name:20s}: "
                f"{accuracy:.4f} "
                f"({class_correct[i]}/"
                f"{class_total[i]})"
            )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "extractor": model.state_dict(),
                "classifier": classifier.state_dict(),
                "sample_rate": SAMPLE_RATE,
                "audio_length": AUDIO_LENGTH,
                "classes": CLASSES,
                "best_val_accuracy": best_val_accuracy
            },
            OUTPUT_PATH
        )

        print(
            f"Saved best model -> {OUTPUT_PATH}"
        )


print()
print("======================================")
print("TRAINING COMPLETE")
print("======================================")
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)
print(
    f"Model saved: {OUTPUT_PATH}"
)