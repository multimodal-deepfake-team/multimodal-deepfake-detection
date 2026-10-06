from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

from models.video_model import DeepfakeResNet18


# ============================
# CONFIG
# ============================

DATA_DIR = "processed/faces"
CHECKPOINT = "saved_models/social_media_detector.pth"

BATCH_SIZE = 16
VAL_SPLIT = 0.2
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# TRANSFORM
# ============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================
# DATASET
# ============================

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform,
    allow_empty=True
)

print()
print("======================================")
print("VIDEO CHECKPOINT EVALUATION")
print("======================================")

print(f"Device: {DEVICE}")
print(f"Total images: {len(dataset)}")
print(f"Classes: {dataset.classes}")
print(f"Class mapping: {dataset.class_to_idx}")

print()
print("Samples per class:")

counts = Counter(dataset.targets)

for index, name in enumerate(dataset.classes):
    print(f"  {name:20s}: {counts[index]}")


# ============================
# SAME 80/20 SPLIT
# ============================

num_val = int(len(dataset) * VAL_SPLIT)
num_train = len(dataset) - num_val

generator = torch.Generator().manual_seed(SEED)

_, val_dataset = random_split(
    dataset,
    [num_train, num_val],
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print()
print(f"Validation images: {len(val_dataset)}")


# ============================
# MODEL
# ============================

model = DeepfakeResNet18(
    num_classes=len(dataset.classes)
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)

model.eval()


# ============================
# EVALUATION
# ============================

all_preds = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_preds.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


accuracy = 100.0 * correct / total


# ============================
# RESULTS
# ============================

print()
print("======================================")
print("RESULT")
print("======================================")

print(f"Validation accuracy: {accuracy:.2f}%")

print()
print("Confusion Matrix")
print("(rows = true, columns = predicted)")
print()

cm = confusion_matrix(
    all_labels,
    all_preds,
    labels=list(range(len(dataset.classes)))
)

print(cm)

print()
print("Classification Report")
print()

print(
    classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(dataset.classes))),
        target_names=dataset.classes,
        zero_division=0
    )
)

print("======================================")