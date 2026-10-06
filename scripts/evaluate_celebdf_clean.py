from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torchvision import transforms

from models.video_model import DeepfakeResNet18


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(".")

REAL_DIR = ROOT / "dataset" / "celebdf" / "Celeb-real"
FAKE_DIR = ROOT / "dataset" / "celebdf" / "Celeb-synthesis"

SUBSET_DIR = ROOT / "processed" / "celebdf_subset"

OLD_CHECKPOINT = ROOT / "saved_models" / "social_media_detector.pth"
NEW_CHECKPOINT = ROOT / "saved_models" / "social_media_detector_celebdf.pth"

NUM_REAL = 50
NUM_FAKE = 50
NUM_FRAMES = 8
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMPORTANT:
# This is the actual ImageFolder class order used during training.
CLASS_NAMES = [
    "ai_video",
    "audio_video_fake",
    "edited",
    "faceswap",
    "real",
    "voiceclone",
]

REAL_IDX = 4


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# FIND SOURCE VIDEO STEMS USED IN CELEB-DF TRAINING SUBSET
# ============================================================

def get_used_source_stems():
    used = set()

    for image_path in SUBSET_DIR.rglob("*.jpg"):
        # Example:
        # id0_0000_frame_000010.jpg
        #
        # -> id0_0000
        stem = image_path.stem.rsplit("_frame_", 1)[0]
        used.add(stem)

    return used


# ============================================================
# SELECT CLEAN UNSEEN VIDEOS
# ============================================================

def select_clean_videos(directory, used_stems, count):
    all_videos = sorted(directory.glob("*.mp4"))

    clean_videos = [
        video for video in all_videos
        if video.stem not in used_stems
    ]

    rng = random.Random(SEED)
    rng.shuffle(clean_videos)

    return sorted(clean_videos[:count])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint):
    model = DeepfakeResNet18(num_classes=6)

    checkpoint_data = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=True
    )

    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
    else:
        state_dict = checkpoint_data

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# EXTRACT EVENLY SPACED FRAMES
# ============================================================

def read_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(
        0,
        total_frames - 1,
        num=min(num_frames, total_frames),
        dtype=int,
    )

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    return frames


# ============================================================
# PREDICT ONE VIDEO
# ============================================================

@torch.no_grad()
def predict_video(model, video_path):
    frames = read_frames(video_path, NUM_FRAMES)

    if not frames:
        return None

    tensors = torch.stack([
        transform(frame)
        for frame in frames
    ]).to(DEVICE)

    outputs = model(tensors)

    probabilities = torch.softmax(outputs, dim=1)

    # Average frame-level probabilities -> video-level probability
    avg_probs = probabilities.mean(dim=0)

    real_probability = float(avg_probs[REAL_IDX].item())
    fake_probability = 1.0 - real_probability

    predicted_binary = "FAKE" if fake_probability >= 0.5 else "REAL"

    predicted_class_idx = int(torch.argmax(avg_probs).item())
    predicted_class = CLASS_NAMES[predicted_class_idx]

    return {
        "binary": predicted_binary,
        "fake_score": fake_probability,
        "real_probability": real_probability,
        "class": predicted_class,
        "frames": len(frames),
    }


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model_name, checkpoint, videos):
    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)
    print("Checkpoint:", checkpoint)
    print()

    model = load_model(checkpoint)

    results = []

    for i, (video_path, true_label) in enumerate(videos, start=1):

        prediction = predict_video(model, video_path)

        if prediction is None:
            print(f"[{i}/{len(videos)}] ERROR: {video_path.name}")
            continue

        correct = prediction["binary"] == true_label

        results.append({
            "path": video_path,
            "true": true_label,
            "pred": prediction["binary"],
            "fake_score": prediction["fake_score"],
            "class": prediction["class"],
            "correct": correct,
        })

        print(
            f"[{i}/{len(videos)}] "
            f"{video_path.name} | "
            f"true={true_label:<4} "
            f"pred={prediction['binary']:<4} "
            f"fake_score={prediction['fake_score']:.4f} "
            f"class={prediction['class']}"
        )

    real_results = [
        r for r in results
        if r["true"] == "REAL"
    ]

    fake_results = [
        r for r in results
        if r["true"] == "FAKE"
    ]

    total_correct = sum(r["correct"] for r in results)

    overall_accuracy = (
        total_correct / len(results)
        if results else 0
    )

    real_accuracy = (
        sum(r["correct"] for r in real_results)
        / len(real_results)
        if real_results else 0
    )

    fake_accuracy = (
        sum(r["correct"] for r in fake_results)
        / len(fake_results)
        if fake_results else 0
    )

    avg_real_fake_score = (
        np.mean([r["fake_score"] for r in real_results])
        if real_results else 0
    )

    avg_fake_fake_score = (
        np.mean([r["fake_score"] for r in fake_results])
        if fake_results else 0
    )

    print()
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)

    print(
        f"REAL accuracy : "
        f"{sum(r['correct'] for r in real_results)}/{len(real_results)} "
        f"= {real_accuracy * 100:.2f}%"
    )

    print(
        f"FAKE accuracy : "
        f"{sum(r['correct'] for r in fake_results)}/{len(fake_results)} "
        f"= {fake_accuracy * 100:.2f}%"
    )

    print(
        f"Overall       : "
        f"{total_correct}/{len(results)} "
        f"= {overall_accuracy * 100:.2f}%"
    )

    print(
        f"Avg REAL fake score : "
        f"{avg_real_fake_score:.4f}"
    )

    print(
        f"Avg FAKE fake score : "
        f"{avg_fake_fake_score:.4f}"
    )

    return {
        "model": model_name,
        "real_correct": sum(r["correct"] for r in real_results),
        "real_total": len(real_results),
        "fake_correct": sum(r["correct"] for r in fake_results),
        "fake_total": len(fake_results),
        "total_correct": total_correct,
        "total": len(results),
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "overall_accuracy": overall_accuracy,
        "avg_real_fake_score": avg_real_fake_score,
        "avg_fake_fake_score": avg_fake_fake_score,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CLEAN CELEB-DF CROSS-DATASET EVALUATION")
    print("=" * 70)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("Real videos :", REAL_DIR)
    print("Fake videos :", FAKE_DIR)
    print("Subset      :", SUBSET_DIR)

    # --------------------------------------------------------
    # Find videos used by the Celeb-DF training subset
    # --------------------------------------------------------

    used_stems = get_used_source_stems()

    print()
    print("Source videos represented in training subset:", len(used_stems))

    # --------------------------------------------------------
    # Select unseen videos
    # --------------------------------------------------------

    clean_real = select_clean_videos(
        REAL_DIR,
        used_stems,
        NUM_REAL
    )

    clean_fake = select_clean_videos(
        FAKE_DIR,
        used_stems,
        NUM_FAKE
    )

    print()
    print("Clean unseen REAL videos:", len(clean_real))
    print("Clean unseen FAKE videos:", len(clean_fake))

    if len(clean_real) < NUM_REAL:
        print("WARNING: fewer clean REAL videos than requested.")

    if len(clean_fake) < NUM_FAKE:
        print("WARNING: fewer clean FAKE videos than requested.")

    # Same exact evaluation set for both models
    videos = (
        [(p, "REAL") for p in clean_real] +
        [(p, "FAKE") for p in clean_fake]
    )

    # --------------------------------------------------------
    # Evaluate OLD model
    # --------------------------------------------------------

    old_result = evaluate_model(
        "OLD MODEL",
        OLD_CHECKPOINT,
        videos
    )

    # --------------------------------------------------------
    # Evaluate NEW model
    # --------------------------------------------------------

    new_result = evaluate_model(
        "NEW CELEB-DF ADAPTED MODEL",
        NEW_CHECKPOINT,
        videos
    )

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OLD vs NEW — CLEAN CELEB-DF COMPARISON")
    print("=" * 70)

    print()
    print(
        f"{'Metric':<28}"
        f"{'OLD MODEL':>15}"
        f"{'NEW MODEL':>15}"
    )

    print("-" * 58)

    print(
        f"{'Real accuracy':<28}"
        f"{old_result['real_accuracy'] * 100:>14.2f}%"
        f"{new_result['real_accuracy'] * 100:>14.2f}%"
    )

    print(
        f"{'Fake accuracy':<28}"
        f"{old_result['fake_accuracy'] * 100:>14.2f}%"
        f"{new_result['fake_accuracy'] * 100:>14.2f}%"
    )

    print(
        f"{'Overall accuracy':<28}"
        f"{old_result['overall_accuracy'] * 100:>14.2f}%"
        f"{new_result['overall_accuracy'] * 100:>14.2f}%"
    )

    print(
        f"{'Avg REAL fake score':<28}"
        f"{old_result['avg_real_fake_score']:>15.4f}"
        f"{new_result['avg_real_fake_score']:>15.4f}"
    )

    print(
        f"{'Avg FAKE fake score':<28}"
        f"{old_result['avg_fake_fake_score']:>15.4f}"
        f"{new_result['avg_fake_fake_score']:>15.4f}"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()