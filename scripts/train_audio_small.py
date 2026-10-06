from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import soundfile as sf
from transformers import Wav2Vec2Model


DATASET = Path("dataset/audio_small")
OUTPUT = Path("saved_models/audio_binary_small.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAMPLE_RATE = 16000
AUDIO_LENGTH = 64000

BATCH_SIZE = 4
EPOCHS = 8
LR = 1e-4

CLASS_TO_IDX = {
    "real": 0,
    "fake": 1
}


print("=" * 60)
print("SMALL BINARY AUDIO MODEL")
print("=" * 60)
print("Device:", DEVICE)


class AudioDataset(Dataset):

    def __init__(self, root, split, training=False):

        self.samples = []
        self.training = training

        root = Path(root) / split

        for class_name, label in CLASS_TO_IDX.items():

            folder = root / class_name

            for audio_file in sorted(folder.glob("*.flac")):
                self.samples.append(
                    (audio_file, label)
                )

        random.Random(42).shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        waveform, sr = sf.read(
            str(path),
            dtype="float32"
        )

        # Stereo -> mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        waveform = torch.from_numpy(waveform)

        # Resample if necessary
        if sr != SAMPLE_RATE:

            import torchaudio

            waveform = torchaudio.functional.resample(
                waveform,
                sr,
                SAMPLE_RATE
            )

        # Remove DC offset
        waveform = waveform - waveform.mean()

        # Peak normalization
        peak = waveform.abs().max()

        if peak > 0:
            waveform = waveform / peak

        length = waveform.shape[0]

        if length > AUDIO_LENGTH:

            if self.training:

                start = random.randint(
                    0,
                    length - AUDIO_LENGTH
                )

            else:

                start = (
                    length - AUDIO_LENGTH
                ) // 2

            waveform = waveform[
                start:start + AUDIO_LENGTH
            ]

        elif length < AUDIO_LENGTH:

            waveform = F.pad(
                waveform,
                (0, AUDIO_LENGTH - length)
            )

        return waveform, torch.tensor(
            label,
            dtype=torch.long
        )


train_dataset = AudioDataset(
    DATASET,
    "train",
    training=True
)

val_dataset = AudioDataset(
    DATASET,
    "validation",
    training=False
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("Loading fresh Wav2Vec2...")

wav2vec = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
)

wav2vec = wav2vec.to(DEVICE)

# Freeze pretrained backbone
for parameter in wav2vec.parameters():
    parameter.requires_grad = False

wav2vec.eval()


classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 2)
).to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=LR,
    weight_decay=1e-4
)


best_accuracy = 0.0


for epoch in range(EPOCHS):

    classifier.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for waveform, labels in train_loader:

        waveform = waveform.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.no_grad():

            features = wav2vec(
                waveform
            ).last_hidden_state

            features = features.mean(dim=1)

        logits = classifier(features)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()
        optimizer.step()

        train_loss += (
            loss.item() * labels.size(0)
        )

        predictions = logits.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    train_loss /= train_total

    train_accuracy = (
        train_correct / train_total
    )


    classifier.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for waveform, labels in val_loader:

            waveform = waveform.to(DEVICE)
            labels = labels.to(DEVICE)

            features = wav2vec(
                waveform
            ).last_hidden_state

            features = features.mean(dim=1)

            logits = classifier(features)

            loss = criterion(
                logits,
                labels
            )

            val_loss += (
                loss.item() * labels.size(0)
            )

            predictions = logits.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    val_loss /= val_total

    val_accuracy = (
        val_correct / val_total
    )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )


    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            {
                "classifier": classifier.state_dict(),
                "best_val_accuracy": best_accuracy,
                "sample_rate": SAMPLE_RATE,
                "audio_length": AUDIO_LENGTH,
                "classes": CLASS_TO_IDX
            },
            OUTPUT
        )

        print(
            f"  -> Saved best model: "
            f"{best_accuracy * 100:.2f}%"
        )


print("=" * 60)
print("AUDIO TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(f"Checkpoint: {OUTPUT}")
