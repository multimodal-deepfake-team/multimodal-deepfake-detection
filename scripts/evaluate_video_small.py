from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_detector_small import VideoDetectorSmall


ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "dataset" / "video_small" / "validation" / "real"
FAKE_DIR = ROOT / "dataset" / "video_small" / "validation" / "fake"


detector = VideoDetectorSmall()

results = []

for label, folder in [("REAL", REAL_DIR), ("FAKE", FAKE_DIR)]:

    files = sorted(folder.iterdir())

    correct = 0

    for path in files:

        if not path.is_file():
            continue

        result = detector.predict(path)

        predicted = result["label"]

        if predicted == label:
            correct += 1

        results.append((label, predicted))

    print()
    print(f"{label}: {correct}/{len(files)} correct")


total_correct = sum(
    actual == predicted
    for actual, predicted in results
)

print()
print("=" * 60)
print("VIDEO VALIDATION RESULTS")
print("=" * 60)
print("Total samples:", len(results))
print("Correct:", total_correct)
print(
    "Accuracy:",
    f"{100 * total_correct / len(results):.2f}%"
)

print()
print("Confusion Matrix")
print("Actual REAL -> Predicted REAL:",
      sum(a == "REAL" and p == "REAL" for a, p in results))
print("Actual REAL -> Predicted FAKE:",
      sum(a == "REAL" and p == "FAKE" for a, p in results))
print("Actual FAKE -> Predicted REAL:",
      sum(a == "FAKE" and p == "REAL" for a, p in results))
print("Actual FAKE -> Predicted FAKE:",
      sum(a == "FAKE" and p == "FAKE" for a, p in results))
