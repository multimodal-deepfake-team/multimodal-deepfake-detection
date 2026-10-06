import os
import sys
from pathlib import Path

import torch

from scripts.video_detector import VideoDetector


REAL_DIR = Path("dataset/celebdf/Celeb-real")
FAKE_DIR = Path("dataset/celebdf/Celeb-synthesis")

N_VIDEOS = 50


def evaluate_folder(detector, folder, true_label):
    videos = sorted(folder.glob("*.mp4"))[:N_VIDEOS]

    correct = 0
    fake_scores = []
    results = []

    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")

        result = detector.predict(str(video))

        predicted_fake = result["fake_probability"] >= 0.5

        predicted_label = "FAKE" if predicted_fake else "REAL"

        is_correct = (
            predicted_label == true_label
        )

        if is_correct:
            correct += 1

        fake_scores.append(
            result["fake_probability"]
        )

        results.append({
            "file": video.name,
            "predicted": predicted_label,
            "fake_score": result["fake_probability"],
            "class": result["predicted_class"],
        })

        print(
            f"    predicted={predicted_label} "
            f"fake_score={result['fake_probability']:.4f} "
            f"class={result['predicted_class']}"
        )

    accuracy = (
        correct / len(videos)
        if videos
        else 0
    )

    average_fake_score = (
        sum(fake_scores) / len(fake_scores)
        if fake_scores
        else 0
    )

    return {
        "total": len(videos),
        "correct": correct,
        "accuracy": accuracy,
        "average_fake_score": average_fake_score,
        "results": results,
    }


def main():

    print()
    print("======================================")
    print("CELEB-DF CROSS-DATASET EVALUATION")
    print("======================================")

    print(f"Real videos : {REAL_DIR}")
    print(f"Fake videos : {FAKE_DIR}")
    print(f"Videos/class: {N_VIDEOS}")

    if not REAL_DIR.exists():
        raise FileNotFoundError(
            f"Real directory not found: {REAL_DIR}"
        )

    if not FAKE_DIR.exists():
        raise FileNotFoundError(
            f"Fake directory not found: {FAKE_DIR}"
        )

    print()
    print("Loading video detector...")

    detector = VideoDetector()

    print()
    print("======================================")
    print("REAL VIDEOS")
    print("======================================")

    real_result = evaluate_folder(
        detector,
        REAL_DIR,
        "REAL"
    )

    print()
    print("======================================")
    print("FAKE VIDEOS")
    print("======================================")

    fake_result = evaluate_folder(
        detector,
        FAKE_DIR,
        "FAKE"
    )

    total = (
        real_result["total"]
        + fake_result["total"]
    )

    total_correct = (
        real_result["correct"]
        + fake_result["correct"]
    )

    overall_accuracy = (
        total_correct / total
        if total
        else 0
    )

    print()
    print("======================================")
    print("CELEB-DF RESULTS")
    print("======================================")

    print()
    print("REAL:")
    print(
        f"  Correct: "
        f"{real_result['correct']}/"
        f"{real_result['total']}"
    )

    print(
        f"  Accuracy: "
        f"{real_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"  Average fake score: "
        f"{real_result['average_fake_score']:.4f}"
    )

    print()
    print("FAKE:")

    print(
        f"  Correct: "
        f"{fake_result['correct']}/"
        f"{fake_result['total']}"
    )

    print(
        f"  Accuracy: "
        f"{fake_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"  Average fake score: "
        f"{fake_result['average_fake_score']:.4f}"
    )

    print()
    print("--------------------------------------")

    print(
        f"Overall accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )

    print(
        f"Total correct: "
        f"{total_correct}/{total}"
    )

    print("======================================")


if __name__ == "__main__":
    main()