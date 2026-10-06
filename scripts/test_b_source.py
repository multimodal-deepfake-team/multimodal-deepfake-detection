import pandas as pd, numpy as np

CLASSES = ["ai_video", "audio_video_fake", "edited", "faceswap", "real"]
t = pd.read_csv("results/v3_test_predictions.csv")
t["pred"] = t[[f"p{k}" for k in range(5)]].values.argmax(1)  # crop-level

def src(row):
    s = row["video_key"]
    if row["y"] == 1: return "fakeavceleb"
    if s.startswith("real_fav::"): return "fakeavceleb"
    stem = s.split("::")[-1]
    if stem.startswith("id") and "_id" in stem: return "celebdf"
    return "ffpp_or_other"

t["source"] = t.apply(src, axis=1)

g = t.groupby("video_key")
v = g[[f"p{k}" for k in range(5)]].mean()
v["y"] = g["y"].first(); v["source"] = g["source"].first()
v["pred"] = v[[f"p{k}" for k in range(5)]].values.argmax(1)
v["correct"] = v["y"] == v["pred"]
v["label"] = v["y"].map(dict(enumerate(CLASSES)))

print(v.groupby(["label", "source"])["correct"].agg(["mean", "count"]).round(3))