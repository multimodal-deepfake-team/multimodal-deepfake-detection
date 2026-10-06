from pathlib import Path
import shutil
import random

ROOT = Path(r"D:\Final\multimodal_deepfake_detection")
LA = ROOT / "dataset" / "LA" / "LA"

TRAIN_PROTOCOL = LA / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL = LA / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"

TRAIN_AUDIO = LA / "ASVspoof2019_LA_train" / "flac"
DEV_AUDIO = LA / "ASVspoof2019_LA_dev" / "flac"

OUT = ROOT / "dataset" / "audio_subset"

random.seed(42)


def read_protocol(protocol_file):
    groups = {
        "real": [],
        "tts": [],
        "vc": []
    }

    with open(protocol_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            utterance = parts[1]
            attack_id = parts[3]
            label = parts[4]

            # REAL
            if label == "bonafide":
                groups["real"].append(utterance)

            # FAKE
            elif label == "spoof":

                # A01-A04 = TTS
                if attack_id in {"A01", "A02", "A03", "A04"}:
                    groups["tts"].append(utterance)

                # A05-A06 = Voice Conversion
                elif attack_id in {"A05", "A06"}:
                    groups["vc"].append(utterance)

    return groups


def select_random(groups, count):
    if len(groups) < count:
        raise RuntimeError(
            f"Only {len(groups)} files available, requested {count}"
        )

    return random.sample(groups, count)


def copy_files(files, source_dir, destination_dir):
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied = 0

    for utterance in files:
        source = source_dir / f"{utterance}.flac"
        destination = destination_dir / f"{utterance}.flac"

        if source.exists():
            shutil.copy2(source, destination)
            copied += 1

    return copied


# =========================
# TRAIN
# =========================

train = read_protocol(TRAIN_PROTOCOL)

train_real = select_random(train["real"], 2000)
train_tts = select_random(train["tts"], 2000)
train_vc = select_random(train["vc"], 2000)

copy_files(train_real, TRAIN_AUDIO, OUT / "train" / "real")
copy_files(train_tts, TRAIN_AUDIO, OUT / "train" / "tts")
copy_files(train_vc, TRAIN_AUDIO, OUT / "train" / "vc")


# =========================
# VALIDATION
# =========================

dev = read_protocol(DEV_PROTOCOL)

dev_real = select_random(dev["real"], 500)
dev_tts = select_random(dev["tts"], 500)
dev_vc = select_random(dev["vc"], 500)

copy_files(dev_real, DEV_AUDIO, OUT / "val" / "real")
copy_files(dev_tts, DEV_AUDIO, OUT / "val" / "tts")
copy_files(dev_vc, DEV_AUDIO, OUT / "val" / "vc")


print()
print("=" * 50)
print("AUDIO SUBSET CREATED SUCCESSFULLY")
print("=" * 50)

print()
print("TRAIN")
print("REAL :", len(train_real))
print("TTS  :", len(train_tts))
print("VC   :", len(train_vc))
print("TOTAL:", len(train_real) + len(train_tts) + len(train_vc))

print()
print("VALIDATION")
print("REAL :", len(dev_real))
print("TTS  :", len(dev_tts))
print("VC   :", len(dev_vc))
print("TOTAL:", len(dev_real) + len(dev_tts) + len(dev_vc))

print()
print("OUTPUT:")
print(OUT)
