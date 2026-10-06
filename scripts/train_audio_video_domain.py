"""Retrain the audio head on video-domain audio (same architecture).

The shipped checkpoint (audio_binary_small.pth) learned ASVspoof2019 LA
speech, so on video audio it called every clip FAKE. This script:

  1. cuts every clip in processed/audio_v2 into 4-second windows
     (identical normalisation to app.prepare_audio),
  2. embeds them once with the frozen wav2vec2 backbone,
  3. trains the same 768 -> 256 -> 2 head with a group-wise val split,
  4. saves the result in the exact schema the Streamlit app reads,
  5. prints OLD vs NEW accuracy on the held-out test videos.

Nothing is overwritten: the original checkpoint is copied to
saved_models/audio_binary_small_asvspoof.pth first.
"""
from pathlib import Path
import json
import random
import shutil
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "processed" / "audio_v2"
MANIFEST = DATA / "manifest.json"
EMBED_CACHE = DATA / "embeddings.pt"

CHECKPOINT = ROOT / "saved_models" / "audio_binary_small.pth"
BACKUP = ROOT / "saved_models" / "audio_binary_small_asvspoof.pth"

SAMPLE_RATE = 16000
WINDOW = 64000          # 4 s, same as the app and the original training
STRIDE = 32000          # 50 % overlap
BATCH = 16
EPOCHS = 60
LR = 1e-3
PATIENCE = 10
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = {"real": 0, "fake": 1}


# ------------------------------------------------------------------
# Windows, matching app.prepare_audio exactly
# ------------------------------------------------------------------
def load_track(path: Path) -> np.ndarray:
    waveform, sr = sf.read(str(path), dtype="float32")

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sr != SAMPLE_RATE:
        import librosa
        waveform = librosa.resample(
            waveform, orig_sr=sr, target_sr=SAMPLE_RATE
        ).astype(np.float32)

    waveform = waveform - float(np.mean(waveform))
    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform = waveform / peak

    return waveform


def windows_of(waveform: np.ndarray):
    if len(waveform) < WINDOW:
        out = np.zeros(WINDOW, dtype=np.float32)
        out[: len(waveform)] = waveform
        return [out]

    starts = list(range(0, len(waveform) - WINDOW + 1, STRIDE))
    if not starts:
        starts = [0]
    return [waveform[s : s + WINDOW] for s in starts]


# ------------------------------------------------------------------
# Embeddings (frozen wav2vec2, computed once)
# ------------------------------------------------------------------
def build_embeddings():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if EMBED_CACHE.exists():
        print(f"Loading cached embeddings: {EMBED_CACHE.name}")
        return torch.load(EMBED_CACHE, weights_only=False)

    print(f"Loading Wav2Vec2 on {DEVICE} ...")
    wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    wav2vec = wav2vec.to(DEVICE)
    for p in wav2vec.parameters():
        p.requires_grad = False
    wav2vec.eval()

    rows = []
    with torch.no_grad():
        for i, item in enumerate(manifest, 1):
            path = ROOT / item["file"]
            for w in windows_of(load_track(path)):
                t = torch.from_numpy(w).unsqueeze(0).to(DEVICE)
                emb = wav2vec(t).last_hidden_state.mean(dim=1)
                rows.append(
                    {
                        "emb": emb.squeeze(0).cpu(),
                        "label": CLASSES[item["label"]],
                        "split": item["split"],
                        "group": item["group"],
                        "file": item["file"],
                    }
                )
            if i % 25 == 0:
                print(f"  embedded {i}/{len(manifest)} clips", flush=True)

    torch.save(rows, EMBED_CACHE)
    print(f"Cached {len(rows)} window embeddings -> {EMBED_CACHE.name}")
    return rows


# ------------------------------------------------------------------
# Model (identical to app.AudioBinaryClassifier)
# ------------------------------------------------------------------
def make_head() -> nn.Module:
    head = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2),
    )
    return head.to(DEVICE)


def probs_of(rows):
    embs = torch.stack([r["emb"] for r in rows])
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    return embs, labels


def evaluate(head, rows, threshold_mode="mean"):
    """Video-level accuracy: average window probabilities per source clip."""
    head.eval()
    by_file = {}
    with torch.no_grad():
        for r in rows:
            p = F.softmax(
                head(r["emb"].unsqueeze(0).to(DEVICE)), dim=1
            )[0]
            by_file.setdefault(r["file"], []).append(
                (r["label"], float(p[CLASSES["fake"]]), float(p[CLASSES["real"]]))
            )

    correct = 0
    preds = []
    for f, items in by_file.items():
        fake = float(np.mean([i[1] for i in items]))
        real = float(np.mean([i[2] for i in items]))
        label = items[0][0]
        pred = CLASSES["fake"] if fake >= real else CLASSES["real"]
        correct += int(pred == label)
        preds.append((label, pred, fake))

    return correct, len(by_file), preds


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rows = build_embeddings()

    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    # group-wise val split inside train so no video leaks
    groups = sorted({r["group"] for r in train_rows})
    random.Random(SEED).shuffle(groups)
    val_groups = set(groups[: max(1, len(groups) // 5)])

    fit_rows = [r for r in train_rows if r["group"] not in val_groups]
    val_rows = [r for r in train_rows if r["group"] in val_groups]

    print()
    print("=" * 60)
    print("SPLITS (windows / source videos)")
    for name, rs in (("fit", fit_rows), ("val", val_rows), ("test", test_rows)):
        g = len({r["file"] for r in rs})
        n_real = sum(1 for r in rs if r["label"] == CLASSES["real"])
        n_fake = len(rs) - n_real
        print(f"  {name:5s}: {len(rs):4d} windows / {g:3d} videos "
              f"(real {n_real}, fake {n_fake})")
    print("=" * 60)

    fit_e, fit_l = probs_of(fit_rows)
    counts = torch.bincount(fit_l, minlength=2).float()
    weights = (counts.sum() / (2.0 * counts.clamp(min=1))).to(DEVICE)
    print(f"class weights: {weights.tolist()}")

    head = make_head()
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val = -1.0
    best_state = None
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        head.train()
        order = torch.randperm(len(fit_e))
        total_loss = 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i : i + BATCH]
            xb = fit_e[idx].to(DEVICE)
            yb = fit_l[idx].to(DEVICE)
            opt.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(idx)

        head.eval()
        vc, vn, _ = evaluate(head, val_rows)
        val_acc = vc / max(vn, 1)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:02d}  loss {total_loss / len(fit_e):.4f}  "
                  f"val {vc}/{vn} = {val_acc:.3f}", flush=True)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop at epoch {epoch}")
                break

    if best_state is None:
        best_state = head.state_dict()
    head.load_state_dict(best_state)

    # ---------------- old vs new on the held-out test videos ----------------
    print()
    print("=" * 60)
    print("HELD-OUT TEST VIDEOS (video-level, never seen in training)")
    print("=" * 60)

    new_c, new_n, new_preds = evaluate(head, test_rows)
    print(f"NEW model: {new_c}/{new_n} = {100 * new_c / max(new_n, 1):.1f}%")

    old_model = None
    if CHECKPOINT.exists():
        old_ckpt = torch.load(CHECKPOINT, map_location="cpu",
                              weights_only=False)
        old_model = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2)
        )
        old_model.load_state_dict(old_ckpt["classifier"])
        old_model = old_model.to(DEVICE)

        by_file = {}
        with torch.no_grad():
            for r in test_rows:
                p = F.softmax(
                    old_model(r["emb"].unsqueeze(0).to(DEVICE)), dim=1
                )[0]
                by_file.setdefault(r["file"], []).append(
                    (r["label"], float(p[CLASSES["fake"]]))
                )
        correct = 0
        for f, items in by_file.items():
            fake = float(np.mean([i[1] for i in items]))
            pred = CLASSES["fake"] if fake >= 0.5 else CLASSES["real"]
            correct += int(pred == items[0][0])
        print(f"OLD model: {correct}/{len(by_file)} = "
              f"{100 * correct / max(len(by_file), 1):.1f}%")

    print()
    for label, pred, fake in sorted(new_preds, key=lambda x: -x[2]):
        name = "real" if label == CLASSES["real"] else "fake"
        got = "real" if pred == CLASSES["real"] else "fake"
        flag = "OK " if pred == label else "WRONG"
        print(f"  [{flag}] true={name:4s} pred={got:4s} fake={fake:.3f}")

    # ---------------- save ----------------
    if CHECKPOINT.exists() and not BACKUP.exists():
        shutil.copy2(CHECKPOINT, BACKUP)
        print(f"\nOriginal checkpoint backed up -> {BACKUP.name}")

    torch.save(
        {
            "classifier": {k: v.cpu() for k, v in head.state_dict().items()},
            "best_val_accuracy": float(best_val),
            "sample_rate": SAMPLE_RATE,
            "audio_length": WINDOW,
            "classes": CLASSES,
            "trained_on": "video_audio_v2",
            "train_groups": len({r["group"] for r in fit_rows}),
            "val_groups": len(val_groups),
            "test_accuracy": new_c / max(new_n, 1),
        },
        CHECKPOINT,
    )
    print(f"Saved retrained checkpoint -> {CHECKPOINT.name}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
