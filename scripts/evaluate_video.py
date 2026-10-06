from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"D:\Final\multimodal_deepfake_detection"
)

CSV_PATH = (
    ROOT
    / "processed"
    / "fusion_val.csv"
)

MODEL_PATH = (
    ROOT
    / "saved_models"
    / "social_media_detector.pth"
)

RESULTS_DIR = (
    ROOT
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "video_evaluation.txt"
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_CLASSES = 6

MAX_FRAMES_PER_VIDEO = 4


# ============================================================
# ORIGINAL RESNET18 CLASS ORDER
# ============================================================
#
# IMPORTANT:
# These are the raw output indices of the trained
# ResNet18 checkpoint.
#
# Raw model output:
#
# 0 -> ai_video
# 1 -> audio_video_fake
# 2 -> edited
# 3 -> faceswap
# 4 -> real
# 5 -> voiceclone
#
# ============================================================

CLASS_NAMES = [
    "ai_video",
    "audio_video_fake",
    "edited",
    "faceswap",
    "real",
    "voiceclone",
]


# ============================================================
# FUSION VALIDATION CLASS SPACE
# ============================================================
#
# fusion_val.csv uses:
#
# 0 -> real
# 4 -> voiceclone
# 5 -> audio_video_fake
#
# We evaluate only these three classes because the current
# fusion validation set contains these three paired classes.
#
# ============================================================

EVALUATED_CLASSES = [
    "real",
    "voiceclone",
    "audio_video_fake",
]

EVALUATED_LABELS = [
    0,
    4,
    5,
]


# ============================================================
# RAW RESNET LABEL -> FUSION LABEL
# ============================================================
#
# Raw ResNet:
#
# 1 = audio_video_fake
# 4 = real
# 5 = voiceclone
#
# Fusion manifest:
#
# 0 = real
# 4 = voiceclone
# 5 = audio_video_fake
#
# Therefore:
#
# raw 1 -> fusion 5
# raw 4 -> fusion 0
# raw 5 -> fusion 4
#
# ============================================================

PRED_LABEL_MAP = {
    1: 5,   # raw audio_video_fake -> fusion audio_video_fake
    4: 0,   # raw real             -> fusion real
    5: 4,   # raw voiceclone       -> fusion voiceclone
}


# ============================================================
# MODEL
# ============================================================

class DeepfakeResNet18(nn.Module):

    def __init__(
        self,
        num_classes=6
    ):
        super().__init__()

        self.backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

weights = ResNet18_Weights.DEFAULT

transform = weights.transforms()


def load_frame(path):
    """
    Load one image frame and apply the official
    ResNet18 inference preprocessing.
    """

    image = Image.open(
        path
    ).convert("RGB")

    return transform(
        image
    )


# ============================================================
# CHECK FILES
# ============================================================

if not CSV_PATH.exists():

    raise FileNotFoundError(
        f"\nValidation CSV not found:\n"
        f"{CSV_PATH}\n"
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel checkpoint not found:\n"
        f"{MODEL_PATH}\n"
    )


# ============================================================
# LOAD VALIDATION CSV
# ============================================================

df = pd.read_csv(
    CSV_PATH
)


# ============================================================
# START INFORMATION
# ============================================================

print()

print(
    "======================================"
)

print(
    "VIDEO-ONLY RESNET18 EVALUATION"
)

print(
    "======================================"
)

print(
    f"Device: {DEVICE}"
)


if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


print(
    f"Validation samples: {len(df)}"
)


# ============================================================
# VALIDATION CLASS DISTRIBUTION
# ============================================================

print()

print(
    "Validation class distribution:"
)

print(
    df["class_name"]
    .value_counts()
    .to_string()
)


# ============================================================
# LOAD MODEL
# ============================================================

print()

print(
    "Loading ResNet18 checkpoint..."
)


model = DeepfakeResNet18(
    num_classes=NUM_CLASSES
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


# ============================================================
# HANDLE CHECKPOINT FORMAT
# ============================================================

if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

    else:

        state_dict = checkpoint

else:

    raise RuntimeError(
        "Unsupported checkpoint format."
    )


# ============================================================
# REMOVE DATAPARALLEL PREFIX
# ============================================================

clean_state_dict = {}


for key, value in state_dict.items():

    if key.startswith(
        "module."
    ):

        key = key[7:]

    clean_state_dict[
        key
    ] = value


# ============================================================
# LOAD STATE DICT
# ============================================================

try:

    model.load_state_dict(
        clean_state_dict,
        strict=True
    )

except RuntimeError:

    print()

    print(
        "Direct checkpoint loading failed."
    )

    print(
        "Trying backbone-prefixed checkpoint..."
    )

    backbone_state = {}

    for key, value in (
        clean_state_dict.items()
    ):

        if key.startswith(
            "backbone."
        ):

            backbone_state[
                key
            ] = value

    if len(backbone_state) == 0:

        raise

    model.load_state_dict(
        backbone_state,
        strict=True
    )


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(
    DEVICE
)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()


print(
    "Checkpoint loaded successfully."
)


# ============================================================
# INFERENCE STORAGE
# ============================================================

all_true = []

all_pred = []

all_mapped_pred = []

all_confidence = []

all_paths = []


# ============================================================
# RUN INFERENCE
# ============================================================

print()

print(
    "Running video-only inference..."
)


with torch.inference_mode():

    for index, row in df.iterrows():

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        class_name = (
            str(row["class_name"])
        )

        true_label = int(
            row["label"]
        )


        # ----------------------------------------------------
        # Frame paths
        # ----------------------------------------------------

        frame_paths = str(
            row["frame_paths"]
        ).split("|")


        # ----------------------------------------------------
        # Keep only existing frames
        # ----------------------------------------------------

        frame_paths = [

            Path(path)

            for path in frame_paths

            if Path(path).exists()

        ]


        # ----------------------------------------------------
        # No frames
        # ----------------------------------------------------

        if len(frame_paths) == 0:

            print(
                f"WARNING: "
                f"No valid frames for "
                f"{row['video_id']}"
            )

            continue


        # ----------------------------------------------------
        # Same frame policy as fusion training
        # ----------------------------------------------------

        frame_paths = (
            frame_paths[
                :MAX_FRAMES_PER_VIDEO
            ]
        )


        # ----------------------------------------------------
        # Load frames
        # ----------------------------------------------------

        frame_tensors = []


        for frame_path in frame_paths:

            try:

                frame = load_frame(
                    frame_path
                )

                frame_tensors.append(
                    frame
                )

            except Exception as e:

                print(
                    f"WARNING: "
                    f"Could not load "
                    f"{frame_path}: "
                    f"{e}"
                )


        # ----------------------------------------------------
        # No successfully loaded frames
        # ----------------------------------------------------

        if len(frame_tensors) == 0:

            print(
                f"WARNING: "
                f"No readable frames for "
                f"{row['video_id']}"
            )

            continue


        # ----------------------------------------------------
        # Repeat final frame if fewer than 4
        # ----------------------------------------------------

        while (
            len(frame_tensors)
            < MAX_FRAMES_PER_VIDEO
        ):

            frame_tensors.append(
                frame_tensors[-1].clone()
            )


        # ----------------------------------------------------
        # Create batch
        # ----------------------------------------------------

        batch = torch.stack(
            frame_tensors
        ).to(
            DEVICE
        )


        # ----------------------------------------------------
        # Frame-level logits
        # ----------------------------------------------------

        logits = model(
            batch
        )


        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probs = torch.softmax(
            logits,
            dim=1
        )


        # ----------------------------------------------------
        # Average frame probabilities
        # ----------------------------------------------------
        #
        # Shape:
        #
        # probs
        # [num_frames, 6]
        #
        # mean(dim=0)
        # [6]
        #
        # ----------------------------------------------------

        video_probs = probs.mean(
            dim=0
        )


        # ----------------------------------------------------
        # Raw ResNet prediction
        # ----------------------------------------------------

        predicted_label = int(
            torch.argmax(
                video_probs
            ).item()
        )


        # ----------------------------------------------------
        # Raw confidence
        # ----------------------------------------------------

        confidence = float(
            video_probs[
                predicted_label
            ].item()
        )


        # ----------------------------------------------------
        # Convert raw prediction into
        # fusion validation label space
        # ----------------------------------------------------

        mapped_label = (
            PRED_LABEL_MAP.get(
                predicted_label,
                -1
            )
        )


        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if mapped_label == -1:

            print()

            print(
                "WARNING: Raw prediction "
                f"{predicted_label} "
                f"({CLASS_NAMES[predicted_label]}) "
                "is not part of the evaluated "
                "fusion label mapping."
            )

            print(
                f"Video ID: "
                f"{row['video_id']}"
            )

            continue


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        all_true.append(
            true_label
        )

        all_pred.append(
            predicted_label
        )

        all_mapped_pred.append(
            mapped_label
        )

        all_confidence.append(
            confidence
        )

        all_paths.append(
            str(row["video_id"])
        )


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        mapped_class_name = (
            EVALUATED_CLASSES[
                EVALUATED_LABELS.index(
                    mapped_label
                )
            ]
        )


        print(
            f"[{index + 1:02d}/{len(df)}] "
            f"{row['video_id']} | "
            f"true={class_name} | "
            f"raw_pred="
            f"{CLASS_NAMES[predicted_label]} | "
            f"mapped_pred="
            f"{mapped_class_name} | "
            f"confidence="
            f"{confidence:.4f}"
        )


# ============================================================
# SAFETY CHECK
# ============================================================

if len(all_true) == 0:

    raise RuntimeError(
        "No validation samples were "
        "successfully evaluated."
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

all_true_np = np.array(
    all_true
)

all_pred_np = np.array(
    all_pred
)

all_mapped_pred_np = np.array(
    all_mapped_pred
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_true_np,
    all_mapped_pred_np
)


# ------------------------------------------------------------
# Macro metrics
# ------------------------------------------------------------

precision, recall, macro_f1, _ = (
    precision_recall_fscore_support(
        all_true_np,
        all_mapped_pred_np,
        labels=EVALUATED_LABELS,
        average="macro",
        zero_division=0
    )
)


# ------------------------------------------------------------
# Weighted F1
# ------------------------------------------------------------

_, _, weighted_f1, _ = (
    precision_recall_fscore_support(
        all_true_np,
        all_mapped_pred_np,
        labels=EVALUATED_LABELS,
        average="weighted",
        zero_division=0
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_true_np,
    all_mapped_pred_np,
    labels=EVALUATED_LABELS,
    target_names=EVALUATED_CLASSES,
    digits=4,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================
#
# Rows    = actual
# Columns = predicted
#
# Order:
#
# 0 -> real
# 4 -> voiceclone
# 5 -> audio_video_fake
#
# ============================================================

cm = confusion_matrix(
    all_true_np,
    all_mapped_pred_np,
    labels=EVALUATED_LABELS
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()

print(
    "======================================"
)

print(
    "FINAL VIDEO-ONLY EVALUATION"
)

print(
    "======================================"
)

print(
    f"Samples evaluated : "
    f"{len(all_true)}"
)

print(
    f"Accuracy          : "
    f"{accuracy:.4f}"
)

print(
    f"Macro F1          : "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1       : "
    f"{weighted_f1:.4f}"
)


print()

print(
    "Classification Report:"
)

print(
    report
)


print(
    "Confusion Matrix:"
)

print(
    cm
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "VIDEO-ONLY RESNET18 EVALUATION\n"
    )

    f.write(
        "==============================\n\n"
    )


    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    f.write(
        f"Device: {DEVICE}\n"
    )

    if torch.cuda.is_available():

        f.write(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}\n"
        )

    f.write(
        f"Validation CSV: "
        f"{CSV_PATH}\n"
    )

    f.write(
        f"Model checkpoint: "
        f"{MODEL_PATH}\n"
    )

    f.write(
        f"Max frames per video: "
        f"{MAX_FRAMES_PER_VIDEO}\n\n"
    )


    # --------------------------------------------------------
    # Evaluation summary
    # --------------------------------------------------------

    f.write(
        f"Samples evaluated : "
        f"{len(all_true)}\n"
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


    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    f.write(
        "Evaluated Classes:\n"
    )

    for label, name in zip(
        EVALUATED_LABELS,
        EVALUATED_CLASSES
    ):

        f.write(
            f"{label} -> {name}\n"
        )


    f.write(
        "\n"
    )


    # --------------------------------------------------------
    # Prediction mapping
    # --------------------------------------------------------

    f.write(
        "Raw ResNet -> Fusion Label Mapping:\n"
    )

    for raw_label, fusion_label in (
        PRED_LABEL_MAP.items()
    ):

        f.write(
            f"{raw_label} "
            f"({CLASS_NAMES[raw_label]}) "
            f"-> "
            f"{fusion_label}\n"
        )


    f.write(
        "\n"
    )


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    f.write(
        "Classification Report:\n"
    )

    f.write(
        report
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )


    # --------------------------------------------------------
    # Per-video predictions
    # --------------------------------------------------------

    f.write(
        "\n\nPer-video predictions:\n"
    )


    for (
        video_id,
        true_label,
        raw_pred_label,
        mapped_pred_label,
        confidence
    ) in zip(
        all_paths,
        all_true,
        all_pred,
        all_mapped_pred,
        all_confidence
    ):

        # True class
        if true_label in EVALUATED_LABELS:

            true_class = (
                EVALUATED_CLASSES[
                    EVALUATED_LABELS.index(
                        true_label
                    )
                ]
            )

        else:

            true_class = (
                f"unknown_{true_label}"
            )


        # Mapped predicted class
        if (
            mapped_pred_label
            in EVALUATED_LABELS
        ):

            mapped_pred_class = (
                EVALUATED_CLASSES[
                    EVALUATED_LABELS.index(
                        mapped_pred_label
                    )
                ]
            )

        else:

            mapped_pred_class = (
                f"unknown_{mapped_pred_label}"
            )


        f.write(
            f"{video_id} | "
            f"true={true_class} | "
            f"raw_pred="
            f"{CLASS_NAMES[raw_pred_label]} | "
            f"mapped_pred="
            f"{mapped_pred_class} | "
            f"confidence="
            f"{confidence:.4f}\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print()

print(
    f"Results saved to:\n"
    f"{OUTPUT_FILE}"
)

print()

print(
    "VIDEO-ONLY EVALUATION COMPLETE."
)