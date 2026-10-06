from pathlib import Path
import random
import shutil

random.seed(42)

ROOT = Path("dataset/LA_rebuild/LA")
PROTOCOLS = ROOT / "ASVspoof2019_LA_cm_protocols"

TRAIN_PROTOCOL = PROTOCOLS / "ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL = PROTOCOLS / "ASVspoof2019.LA.cm.dev.trl.txt"

TRAIN_AUDIO = ROOT / "ASVspoof2019_LA_train" / "flac"
DEV_AUDIO = ROOT / "ASVspoof2019_LA_dev" / "flac"

OUT = Path("dataset/audio_small")

if OUT.exists():
    shutil.rmtree(OUT)

for split in ["train", "validation"]:
    for label in ["real", "fake"]:
        (OUT / split / label).mkdir(parents=True, exist_ok=True)


def read_protocol(path):
    real = []
    fake = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()

            if len(p) < 5:
                continue

            file_id = p[1]
            label = p[4]

            if label == "bonafide":
                real.append(file_id)
            elif label == "spoof":
                fake.append(file_id)

    return real, fake


train_real, train_fake = read_protocol(TRAIN_PROTOCOL)
val_real, val_fake = read_protocol(DEV_PROTOCOL)

random.shuffle(train_real)
random.shuffle(train_fake)
random.shuffle(val_real)
random.shuffle(val_fake)

train_real = train_real[:80]
train_fake = train_fake[:80]
val_real = val_real[:20]
val_fake = val_fake[:20]


def copy_files(ids, source, destination):
    copied = 0

    for file_id in ids:
        src = source / f"{file_id}.flac"
        dst = destination / f"{file_id}.flac"

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1

    return copied


a = copy_files(train_real, TRAIN_AUDIO, OUT / "train" / "real")
b = copy_files(train_fake, TRAIN_AUDIO, OUT / "train" / "fake")
c = copy_files(val_real, DEV_AUDIO, OUT / "validation" / "real")
d = copy_files(val_fake, DEV_AUDIO, OUT / "validation" / "fake")

print("=" * 50)
print("AUDIO SMALL DATASET")
print("=" * 50)
print("TRAIN REAL :", a)
print("TRAIN FAKE :", b)
print("VAL REAL   :", c)
print("VAL FAKE   :", d)
print("TOTAL      :", a + b + c + d)
print("OUTPUT     :", OUT.resolve())
