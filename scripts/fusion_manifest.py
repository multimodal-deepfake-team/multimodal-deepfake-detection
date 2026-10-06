from pathlib import Path
import csv
import random
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FACES_DIR = ROOT / "processed" / "faces"
AUDIO_DIR = ROOT / "processed" / "audio"

OUTPUT_DIR = ROOT / "processed"

MANIFEST_FILE = OUTPUT_DIR / "fusion_manifest.csv"
TRAIN_FILE = OUTPUT_DIR / "fusion_train.csv"
VAL_FILE = OUTPUT_DIR / "fusion_val.csv"

SEED = 42
VAL_RATIO = 0.20


# ============================================================
# PAIRED CLASSES
# ============================================================

# Existing 6-class visual label mapping:
#
# 0 = real
# 1 = ai_video
# 2 = edited
# 3 = faceswap
# 4 = voiceclone
# 5 = audio_video_fake

CLASS_INFO = {
    "real": {
        "face_dir": "real",
        "audio_dir": "real",
        "label": 0,
    },

    "voiceclone": {
        "face_dir": "voiceclone",
        "audio_dir": "voiceclone",
        "label": 4,
    },

    "audio_video_fake": {
        "face_dir": "audio_video_fake",
        "audio_dir": "audio_video_fake",
        "label": 5,
    },
}


# ============================================================
# HELPERS
# ============================================================

def get_face_video_id(filename):
    """
    Convert frame filename into video/sample ID.

    Examples:

    00001_0000.jpg
        -> 00001

    00001_fake_0000.jpg
        -> 00001_fake

    00010_id00391_TjJgUZaenWI_0000.jpg
        -> 00010_id00391_TjJgUZaenWI
    """

    stem = Path(filename).stem

    parts = stem.rsplit("_", 1)

    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]

    return stem


def collect_face_frames(face_dir):
    """
    Returns:

        {
            video_id: [frame1.jpg, frame2.jpg, ...]
        }
    """

    groups = defaultdict(list)

    if not face_dir.exists():
        return groups

    for file in face_dir.glob("*.jpg"):
        video_id = get_face_video_id(file.name)
        groups[video_id].append(str(file.resolve()))

    for video_id in groups:
        groups[video_id].sort()

    return groups


def collect_audio_files(audio_dir):
    """
    Returns:

        {
            audio_video_id: audio.wav
        }
    """

    audio_files = {}

    if not audio_dir.exists():
        return audio_files

    for file in audio_dir.glob("*.wav"):
        audio_files[file.stem] = str(file.resolve())

    return audio_files


# ============================================================
# CREATE PAIRS
# ============================================================

def make_pairs():
    all_pairs = []

    print("=" * 70)
    print("CREATING REAL MULTIMODAL FUSION MANIFEST")
    print("=" * 70)

    for class_name, info in CLASS_INFO.items():

        face_dir = FACES_DIR / info["face_dir"]
        audio_dir = AUDIO_DIR / info["audio_dir"]

        print()
        print(f"CLASS: {class_name}")
        print(f"Faces : {face_dir}")
        print(f"Audio : {audio_dir}")

        face_groups = collect_face_frames(face_dir)
        audio_files = collect_audio_files(audio_dir)

        face_ids = set(face_groups.keys())
        audio_ids = set(audio_files.keys())

        paired_ids = sorted(face_ids & audio_ids)

        print(f"Face video IDs : {len(face_ids)}")
        print(f"Audio IDs      : {len(audio_ids)}")
        print(f"Paired IDs     : {len(paired_ids)}")

        for video_id in paired_ids:

            all_pairs.append({
                "video_id": video_id,
                "class_name": class_name,
                "label": info["label"],
                "audio_path": audio_files[video_id],
                "frame_paths": "|".join(face_groups[video_id]),
            })

    return all_pairs


# ============================================================
# VALIDATE
# ============================================================

def validate_pairs(pairs):

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    if not pairs:
        raise RuntimeError("No paired samples found.")

    # Check duplicate class + video IDs
    seen = set()

    for item in pairs:

        key = (item["class_name"], item["video_id"])

        if key in seen:
            raise RuntimeError(
                f"Duplicate pair detected: {key}"
            )

        seen.add(key)

    # Check same video ID appearing in multiple classes
    id_to_classes = defaultdict(set)

    for item in pairs:
        id_to_classes[item["video_id"]].add(item["class_name"])

    collisions = {
        video_id: classes
        for video_id, classes in id_to_classes.items()
        if len(classes) > 1
    }

    if collisions:

        print()
        print("WARNING: Cross-class video ID collisions found:")

        for video_id, classes in collisions.items():
            print(f"  {video_id}: {sorted(classes)}")

        raise RuntimeError(
            "Cross-class ID collision detected. "
            "Fix dataset IDs before training."
        )

    print(f"Total paired samples: {len(pairs)}")

    # Class distribution
    counts = defaultdict(int)

    for item in pairs:
        counts[item["class_name"]] += 1

    print()
    print("Class distribution:")

    for class_name in CLASS_INFO:
        print(
            f"  {class_name:20s}: "
            f"{counts[class_name]}"
        )


# ============================================================
# SPLIT BY VIDEO ID
# ============================================================

def split_pairs(pairs):

    random.seed(SEED)

    shuffled = pairs.copy()
    random.shuffle(shuffled)

    val_count = max(
        1,
        round(len(shuffled) * VAL_RATIO)
    )

    val_pairs = shuffled[:val_count]
    train_pairs = shuffled[val_count:]

    train_ids = {
        item["video_id"]
        for item in train_pairs
    }

    val_ids = {
        item["video_id"]
        for item in val_pairs
    }

    overlap = train_ids & val_ids

    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE: train/val overlap detected: {overlap}"
        )

    return train_pairs, val_pairs


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(path, rows):

    fieldnames = [
        "video_id",
        "class_name",
        "label",
        "audio_path",
        "frame_paths",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pairs = make_pairs()

    validate_pairs(pairs)

    train_pairs, val_pairs = split_pairs(pairs)

    save_csv(
        MANIFEST_FILE,
        pairs
    )

    save_csv(
        TRAIN_FILE,
        train_pairs
    )

    save_csv(
        VAL_FILE,
        val_pairs
    )

    print()
    print("=" * 70)
    print("MANIFEST CREATED")
    print("=" * 70)

    print(f"All pairs : {len(pairs)}")
    print(f"Train     : {len(train_pairs)}")
    print(f"Validation: {len(val_pairs)}")

    print()
    print(f"Manifest : {MANIFEST_FILE}")
    print(f"Train    : {TRAIN_FILE}")
    print(f"Val      : {VAL_FILE}")

    print()
    print("Train/Validation video-ID overlap: 0")

    print()
    print("ResNet18 model was NOT modified.")


if __name__ == "__main__":
    main()