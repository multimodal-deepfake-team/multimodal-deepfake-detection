import json, random
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# same order as ImageFolder / your inference code
CLASSES = ["ai_video", "audio_video_fake", "edited", "faceswap", "real", "voiceclone"]
C2I = {c: i for i, c in enumerate(CLASSES)}
BATCH, EPOCHS, LR, SEED = 16, 15, 1e-4, 42
OUT = Path("saved_models"); OUT.mkdir(exist_ok=True)
CKPT = OUT / "video_model_v2_best.pth"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev)

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10), transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])

df = pd.read_csv("splits/v1.csv")

class CropDS(Dataset):
    def __init__(self, d, tf):
        self.d = d.reset_index(drop=True); self.tf = tf
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        r = self.d.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), C2I[r["label"]], i

parts = {s: df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}
train_ds = CropDS(parts["train"], train_tf)
val_ds = CropDS(parts["val"], eval_tf)
test_ds = CropDS(parts["test"], eval_tf)

# sampler weights from TRAIN split only
tr_labels = np.array([C2I[l] for l in parts["train"]["label"]])
counts = np.bincount(tr_labels, minlength=len(CLASSES))
print("train class counts:", dict(zip(CLASSES, counts)))
w = (1.0 / counts)[tr_labels]
sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

train_dl = DataLoader(train_ds, BATCH, sampler=sampler, num_workers=0)
val_dl = DataLoader(val_ds, BATCH, shuffle=False, num_workers=0)
test_dl = DataLoader(test_ds, BATCH, shuffle=False, num_workers=0)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.to(dev)
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)

def run_eval(dl):
    model.eval(); loss_sum, n = 0.0, 0
    probs, ys = [], []
    with torch.no_grad():
        for x, y, _ in dl:
            x, y = x.to(dev), y.to(dev)
            lo = model(x)
            loss_sum += crit(lo, y).item() * len(y); n += len(y)
            probs.append(torch.softmax(lo, 1).cpu().numpy()); ys.append(y.cpu().numpy())
    return loss_sum / n, np.concatenate(probs), np.concatenate(ys)

best = float("inf")
for ep in range(1, EPOCHS + 1):
    model.train(); tl, n = 0.0, 0
    for x, y, _ in train_dl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = crit(model(x), y); loss.backward(); opt.step()
        tl += loss.item() * len(y); n += len(y)
    vl, vp, vy = run_eval(val_dl)
    va = accuracy_score(vy, vp.argmax(1))
    print(f"epoch {ep:02d} train_loss {tl/n:.4f} val_loss {vl:.4f} val_acc {va:.4f}")
    if vl < best:
        best = vl; torch.save(model.state_dict(), CKPT); print("  saved best")

# ---- TEST: run once, with the best checkpoint ----
model.load_state_dict(torch.load(CKPT, map_location=dev))
_, tp, ty = run_eval(test_dl)
res = {}

pred = tp.argmax(1)
print("\n=== CROP-LEVEL TEST ===")
print(classification_report(ty, pred, target_names=CLASSES, digits=3, zero_division=0))
res["crop_acc"] = accuracy_score(ty, pred)
res["crop_macro_f1"] = f1_score(ty, pred, average="macro")

# video-level: average probabilities per video_key
t = parts["test"].copy()
for k, c in enumerate(CLASSES): t[f"p{k}"] = tp[:, k]
t["y"] = ty
g = t.groupby("video_key")
vprob = g[[f"p{k}" for k in range(len(CLASSES))]].mean().values
vy_ = g["y"].first().values
vpred = vprob.argmax(1)
print("=== VIDEO-LEVEL TEST (%d videos) ===" % len(vy_))
print(classification_report(vy_, vpred, target_names=CLASSES, digits=3, zero_division=0))
print(confusion_matrix(vy_, vpred))
res["video_acc"] = accuracy_score(vy_, vpred)
res["video_macro_f1"] = f1_score(vy_, vpred, average="macro")
# binary real vs fake at video level
is_fake = vy_ != C2I["real"]; pred_fake = vpred != C2I["real"]
res["video_binary_acc"] = float((is_fake == pred_fake).mean())

json.dump(res, open("results/v2_test_metrics.json", "w"), indent=2)
print(res)