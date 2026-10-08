import io, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score

CLASSES = ["ai_video", "audio_video_fake", "edited", "faceswap", "real"]
C2I = {c: i for i, c in enumerate(CLASSES)}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("saved_models/video_model_v3_best.pth", map_location=dev))
model.to(dev).eval()

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])

df = pd.read_csv("splits/v2.csv")
df = df[df.split == "test"].reset_index(drop=True)

def degrade(img, q, scale):
    if scale < 1.0:
        w, h = img.size
        img = img.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR)
    if q is not None:
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=q); buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img

conds = [("clean", None, 1.0), ("jpeg90", 90, 1.0), ("jpeg70", 70, 1.0),
         ("jpeg50", 50, 1.0), ("jpeg30", 30, 1.0), ("down0.5+jpeg50", 50, 0.5)]

results = {}
for name, q, sc in conds:
    probs = []
    with torch.no_grad():
        for i in range(0, len(df), 64):
            batch = [tf(degrade(Image.open(p).convert("RGB"), q, sc)) for p in df["path"][i:i+64]]
            x = torch.stack(batch).to(dev)
            probs.append(torch.softmax(model(x), 1).cpu().numpy())
    P = np.concatenate(probs)
    t = df[["video_key", "label"]].copy()
    for k in range(len(CLASSES)): t[f"p{k}"] = P[:, k]
    t["y"] = t["label"].map(C2I)
    g = t.groupby("video_key")
    vp = g[[f"p{k}" for k in range(len(CLASSES))]].mean().values.argmax(1)
    vy = g["y"].first().values
    results[name] = {"video_acc": round(accuracy_score(vy, vp), 3),
                     "macro_f1": round(f1_score(vy, vp, average="macro"), 3),
                     "real_recall": round(float(((vp == 4) & (vy == 4)).sum() / (vy == 4).sum()), 3)}
    print(name, results[name])

json.dump(results, open("results/v3_compression_sweep.json", "w"), indent=2)
