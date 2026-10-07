import pandas as pd
from PIL import Image

CLASSES = ["ai_video", "audio_video_fake", "edited", "faceswap", "real"]
P = [f"p{k}" for k in range(5)]
t = pd.read_csv("results/v3_test_predictions.csv")
df = pd.read_csv("splits/v2.csv").set_index("path")

# crop size per test crop (use the path column from the split)
paths = df[df.split == "test"].reset_index()
t["path"] = paths["path"].values[:len(t)] if "path" not in t else t["path"]
t["side"] = [min(Image.open(p).size) for p in t["path"]]

g = t.groupby("video_key")
v = g[P].mean()
v["y"] = g["y"].first(); v["side"] = g["side"].median()
v["pred"] = v[P].values.argmax(1)
v["ok"] = v["y"] == v["pred"]
v["cls"] = v["y"].map(dict(enumerate(CLASSES)))
print(v.groupby(["cls", "ok"])["side"].agg(["median", "count"]).round(0))