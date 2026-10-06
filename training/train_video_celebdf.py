from pathlib import Path
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

from models.video_model import DeepfakeResNet18


# ============================================================
# CONFIG
# ============================================================

ORIGINAL_DATA_DIR = "processed/faces"
CELEBDF_DATA_DIR = "processed/celebdf_subset"

OLD_CHECKPOINT = "saved_models/social_media_detector.pth"
SAVE_PATH = "saved_models/social_media_detector_celebdf.pth"

BATCH_SIZE_ORIGINAL = 8
BATCH_SIZE_CELEBDF = 8

EPOCHS = 5
LR = 2e-5

# Weight of Celeb-DF binary real/fake auxiliary loss
AUX_WEIGHT = 0.5

VAL_SPLIT = 0.2
SEED = 42


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("VIDEO TRAINING + CELEB-DF AUXILIARY TRAINING")
print("=" * 60)
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# ORIGINAL 6-CLASS DATASET
# ============================================================

print("\nLoading original 6-class dataset...")

base_dataset = datasets.ImageFolder(
    ORIGINAL_DATA_DIR,
    transform=val_transform,
    allow_empty=True
)

print(f"Total original images: {len(base_dataset)}")
print(f"Classes: {base_dataset.classes}")
print(f"Class mapping: {base_dataset.class_to_idx}")

expected_classes = {
    "ai_video",
    "audio_video_fake",
    "edited",
    "faceswap",
    "real",
    "voiceclone"
}

if set(base_dataset.classes) != expected_classes:
    raise RuntimeError(
        f"Unexpected classes found: {base_dataset.classes}"
    )

# IMPORTANT:
# ImageFolder determines the actual checkpoint output order.
real_idx = base_dataset.class_to_idx["real"]

print(f"REAL class index: {real_idx}")

targets = base_dataset.targets
print(
    "Original samples per class:",
    dict(sorted(Counter(targets).items()))
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

num_val = int(len(base_dataset) * VAL_SPLIT)
num_train = len(base_dataset) - num_val

generator = torch.Generator().manual_seed(SEED)

train_subset, val_subset = random_split(
    base_dataset,
    [num_train, num_val],
    generator=generator
)


class TransformedSubset(torch.utils.data.Dataset):

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

        self.samples = [
            base_dataset.samples[i]
            for i in subset.indices
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = base_dataset.loader(path)
        img = self.transform(img)

        return img, label


original_train_dataset = TransformedSubset(
    train_subset,
    train_transform
)

original_val_dataset = TransformedSubset(
    val_subset,
    val_transform
)


# ============================================================
# BALANCED SAMPLING FOR ORIGINAL DATA
# ============================================================

train_targets = [
    label for _, label in original_train_dataset.samples
]

class_sample_counts = Counter(train_targets)

print(
    "\nOriginal training samples:",
    dict(sorted(class_sample_counts.items()))
)

num_classes = len(base_dataset.classes)

class_weights = [0.0] * num_classes

for cls_idx, count in class_sample_counts.items():
    class_weights[cls_idx] = 1.0 / count

sample_weights = [
    class_weights[label]
    for label in train_targets
]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ============================================================
# ORIGINAL LOADERS
# ============================================================

pin_memory = torch.cuda.is_available()

original_train_loader = DataLoader(
    original_train_dataset,
    batch_size=BATCH_SIZE_ORIGINAL,
    sampler=sampler,
    num_workers=0,
    pin_memory=pin_memory
)

original_val_loader = DataLoader(
    original_val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory
)


# ============================================================
# CELEB-DF DATASET
# ============================================================

print("\nLoading Celeb-DF subset...")

celebdf_raw = datasets.ImageFolder(
    CELEBDF_DATA_DIR,
    transform=train_transform,
    allow_empty=True
)

print(f"Celeb-DF images: {len(celebdf_raw)}")
print(f"Celeb-DF mapping: {celebdf_raw.class_to_idx}")

if "fake" not in celebdf_raw.class_to_idx:
    raise RuntimeError("Celeb-DF fake folder not found.")

if "real" not in celebdf_raw.class_to_idx:
    raise RuntimeError("Celeb-DF real folder not found.")


class CelebDFBinaryDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.fake_idx = dataset.class_to_idx["fake"]
        self.real_idx = dataset.class_to_idx["real"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image, raw_label = self.dataset[idx]

        # Binary auxiliary target:
        # REAL = 0
        # FAKE = 1

        if raw_label == self.fake_idx:
            binary_label = 1.0
        else:
            binary_label = 0.0

        return image, torch.tensor(
            binary_label,
            dtype=torch.float32
        )


celebdf_dataset = CelebDFBinaryDataset(celebdf_raw)

celebdf_loader = DataLoader(
    celebdf_dataset,
    batch_size=BATCH_SIZE_CELEBDF,
    shuffle=True,
    num_workers=0,
    pin_memory=pin_memory
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading existing video checkpoint...")

model = DeepfakeResNet18(
    num_classes=num_classes
).to(device)

checkpoint = torch.load(
    OLD_CHECKPOINT,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)

print(f"Loaded checkpoint: {OLD_CHECKPOINT}")
print("Checkpoint loaded successfully.")


# ============================================================
# LOSSES
# ============================================================

classification_loss = nn.CrossEntropyLoss()

auxiliary_loss = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


# ============================================================
# VALIDATION
# ============================================================

def run_validation(model, loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = classification_loss(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            preds = outputs.argmax(dim=1)

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    return (
        total_loss / total,
        100.0 * correct / total,
        all_preds,
        all_labels
    )


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

Path("saved_models").mkdir(
    exist_ok=True
)

print("\nStarting training...")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LR}")
print(f"Celeb-DF auxiliary weight: {AUX_WEIGHT}")
print(f"New checkpoint: {SAVE_PATH}")


for epoch in range(EPOCHS):

    model.train()

    running_original_loss = 0.0
    running_celebdf_loss = 0.0

    original_correct = 0
    original_total = 0

    # Create fresh iterators.
    original_iter = iter(original_train_loader)
    celebdf_iter = iter(celebdf_loader)

    steps = max(
        len(original_train_loader),
        len(celebdf_loader)
    )

    for step in range(steps):

        # ----------------------------------------------------
        # ORIGINAL 6-CLASS BATCH
        # ----------------------------------------------------

        try:
            original_images, original_labels = next(
                original_iter
            )
        except StopIteration:
            original_iter = iter(original_train_loader)

            original_images, original_labels = next(
                original_iter
            )

        # ----------------------------------------------------
        # CELEB-DF BATCH
        # ----------------------------------------------------

        try:
            celeb_images, celeb_labels = next(
                celebdf_iter
            )
        except StopIteration:
            celebdf_iter = iter(celebdf_loader)

            celeb_images, celeb_labels = next(
                celebdf_iter
            )

        original_images = original_images.to(
            device,
            non_blocking=True
        )

        original_labels = original_labels.to(
            device,
            non_blocking=True
        )

        celeb_images = celeb_images.to(
            device,
            non_blocking=True
        )

        celeb_labels = celeb_labels.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        optimizer.zero_grad()

        original_outputs = model(
            original_images
        )

        celeb_outputs = model(
            celeb_images
        )

        # ----------------------------------------------------
        # ORIGINAL 6-CLASS LOSS
        # ----------------------------------------------------

        loss_original = classification_loss(
            original_outputs,
            original_labels
        )

        # ----------------------------------------------------
        # CELEB-DF BINARY LOSS
        #
        # Model still has 6 classes.
        #
        # P(fake) =
        # 1 - P(real)
        #
        # We calculate the equivalent fake-vs-real logit:
        #
        # log(sum(non-real exp(logits))) - real_logit
        # ----------------------------------------------------

        real_logit = celeb_outputs[
            :,
            real_idx
        ]

        nonreal_mask = torch.ones(
            num_classes,
            dtype=torch.bool,
            device=device
        )

        nonreal_mask[real_idx] = False

        nonreal_logits = celeb_outputs[
            :,
            nonreal_mask
        ]

        fake_logit = (
            torch.logsumexp(
                nonreal_logits,
                dim=1
            )
            - real_logit
        )

        loss_celebdf = auxiliary_loss(
            fake_logit,
            celeb_labels
        )

        # ----------------------------------------------------
        # COMBINED LOSS
        # ----------------------------------------------------

        total_loss = (
            loss_original
            + AUX_WEIGHT * loss_celebdf
        )

        total_loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        running_original_loss += (
            loss_original.item()
        )

        running_celebdf_loss += (
            loss_celebdf.item()
        )

        original_preds = (
            original_outputs.argmax(dim=1)
        )

        original_correct += (
            original_preds == original_labels
        ).sum().item()

        original_total += (
            original_labels.size(0)
        )

        if step % 50 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"| Step {step}/{steps} "
                f"| Original Loss: "
                f"{loss_original.item():.4f} "
                f"| CelebDF Loss: "
                f"{loss_celebdf.item():.4f}",
                flush=True
            )

    # --------------------------------------------------------
    # EPOCH METRICS
    # --------------------------------------------------------

    avg_original_loss = (
        running_original_loss / steps
    )

    avg_celebdf_loss = (
        running_celebdf_loss / steps
    )

    original_train_acc = (
        100.0
        * original_correct
        / original_total
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_loss, val_acc, _, _ = run_validation(
        model,
        original_val_loader
    )

    print("\n" + "-" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Original Train Loss: "
        f"{avg_original_loss:.4f}"
    )

    print(
        f"Celeb-DF Auxiliary Loss: "
        f"{avg_celebdf_loss:.4f}"
    )

    print(
        f"Original Train Accuracy: "
        f"{original_train_acc:.2f}%"
    )

    print(
        f"Original Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Original Validation Accuracy: "
        f"{val_acc:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            SAVE_PATH
        )

        print(
            f"BEST MODEL SAVED -> {SAVE_PATH}"
        )

    print("-" * 60)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\nTraining complete.")

print("\nLoading best Celeb-DF adapted model...")

model.load_state_dict(
    torch.load(
        SAVE_PATH,
        map_location=device,
        weights_only=True
    )
)

_, _, all_preds, all_labels = run_validation(
    model,
    original_val_loader
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("FINAL ORIGINAL VALIDATION EVALUATION")
print("=" * 60)

print("\nClass order:")
print(base_dataset.classes)

cm = confusion_matrix(
    all_labels,
    all_preds,
    labels=list(range(num_classes))
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        target_names=base_dataset.classes,
        zero_division=0
    )
)


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(
    f"New model saved at:\n{SAVE_PATH}"
)

print(
    "\nIMPORTANT:"
)

print(
    "Old model was NOT overwritten:"
)

print(
    OLD_CHECKPOINT
)