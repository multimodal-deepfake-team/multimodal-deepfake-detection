from pathlib import Path
import random
import shutil

# ============================================================
# CONFIG
# ============================================================

ROOT = Path("dataset/LA_rebuild/LA")

TRAIN_PROTOCOL = (
    ROOT
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.train.trn.txt"
)

DEV_PROTOCOL = (
    ROOT
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.dev.trl.txt"
)

TRAIN_AUDIO = ROOT / "ASVspoof2019_LA_train" / "flac"
DEV_AUDIO = ROOT / "ASVspoof2019_LA_dev" / "flac"

OUT = Path("dataset/audio_subset")

SEED = 42

TRAIN_PER_CLASS = 2000
VAL_PER_CLASS = 500


# ============================================================
# IMPORTANT ASVSPOOF ATTACK GROUPING
# ============================================================
#
# ASVspoof 2019 LA:
#
# A01-A06 = TTS
# A07-A19 = Voice Conversion
#
# bonafide = real
#
# ============================================================

TTS_ATTACKS = {
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "A06",
}

VC_ATTACKS = {
    "A07",
    "A08",
    "A09",
    "A10",
    "A11",
    "A12",
    "A13",
    "A14",
    "A15",
    "A16",
    "A17",
    "A18",
    "A19",
}


def read_protocol(path):

    samples = {
        "real": [],
        "voiceclone": [],
        "audio_video_fake": [],
    }

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            speaker = parts[0]
            file_id = parts[1]
            attack = parts[3]
            label = parts[4]

            if label == "bonafide":

                samples["real"].append(
                    (speaker, file_id)
                )

            elif label == "spoof":

                if attack in TTS_ATTACKS:

                    samples["voiceclone"].append(
                        (speaker, file_id)
                    )

                elif attack in VC_ATTACKS:

                    samples["audio_video_fake"].append(
                        (speaker, file_id)
                    )

    return samples


def select_samples(samples, count):

    rng = random.Random(SEED)

    selected = {}

    for cls_name, items in samples.items():

        items = list(items)

        if len(items) < count:

            raise RuntimeError(
                f"{cls_name}: only {len(items)} "
                f"samples available, need {count}"
            )

        rng.shuffle(items)

        selected[cls_name] = items[:count]

    return selected


def copy_flac_to_wav_source(
    selected,
    source_audio,
    output_root,
):

    for cls_name, items in selected.items():

        output_dir = output_root / cls_name

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for speaker, file_id in items:

            src = source_audio / f"{file_id}.flac"

            if not src.exists():

                raise FileNotFoundError(
                    f"Missing audio: {src}"
                )

            dst = output_dir / f"{file_id}.flac"

            shutil.copy2(src, dst)


# ============================================================
# READ PROTOCOLS
# ============================================================

print("Reading ASVspoof protocols...")

train_samples = read_protocol(
    TRAIN_PROTOCOL
)

dev_samples = read_protocol(
    DEV_PROTOCOL
)


# ============================================================
# PRINT AVAILABLE COUNTS
# ============================================================

print()
print("AVAILABLE TRAIN:")

for cls_name, items in train_samples.items():

    print(
        f"  {cls_name:20s}: {len(items)}"
    )

print()
print("AVAILABLE VALIDATION:")

for cls_name, items in dev_samples.items():

    print(
        f"  {cls_name:20s}: {len(items)}"
    )


# ============================================================
# SELECT BALANCED SUBSETS
# ============================================================

train_selected = select_samples(
    train_samples,
    TRAIN_PER_CLASS
)

val_selected = select_samples(
    dev_samples,
    VAL_PER_CLASS
)


# ============================================================
# OUTPUT
# ============================================================

if OUT.exists():

    print()
    print(f"Removing old subset: {OUT}")

    shutil.rmtree(OUT)


# ============================================================
# COPY FLAC FILES
# ============================================================

print()
print("Creating training subset...")

copy_flac_to_wav_source(
    train_selected,
    TRAIN_AUDIO,
    OUT / "train"
)

print("Creating validation subset...")

copy_flac_to_wav_source(
    val_selected,
    DEV_AUDIO,
    OUT / "validation"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("======================================")
print("AUDIO SUBSET CREATED")
print("======================================")

for split in ["train", "validation"]:

    split_root = OUT / split

    print()
    print(split.upper())

    total = 0

    for cls_name in [
        "real",
        "voiceclone",
        "audio_video_fake",
    ]:

        count = len(
            list(
                (split_root / cls_name)
                .glob("*.flac")
            )
        )

        print(
            f"  {cls_name:20s}: {count}"
        )

        total += count

    print(f"  TOTAL:                {total}")