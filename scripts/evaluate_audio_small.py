from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audio_detector_small import AudioDetectorSmall


REAL_DIR = ROOT / "dataset" / "audio_small" / "validation" / "real"
FAKE_DIR = ROOT / "dataset" / "audio_small" / "validation" / "fake"

detector = AudioDetectorSmall()

results = []

for label, folder in [("REAL", REAL_DIR), ("FAKE", FAKE_DIR)]:

    files = sorted(folder.glob("*.flac"))
    correct = 0

    for path in files:
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

total = len(results)

print()
print("=" * 60)
print("AUDIO VALIDATION RESULTS")
print("=" * 60)
print("Total samples:", total)
print("Correct:", total_correct)
print(f"Accuracy: {100 * total_correct / total:.2f}%")

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
