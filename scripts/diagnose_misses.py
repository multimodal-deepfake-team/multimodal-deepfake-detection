import pandas as pd

CLASSES = ["ai_video", "audio_video_fake", "edited", "faceswap", "real"]
t = pd.read_csv("results/v3_test_predictions.csv")
P = [f"p{k}" for k in range(5)]
g = t.groupby("video_key")
v = g[P].mean()
v["y"] = g["y"].first()
v["pred"] = v[P].values.argmax(1)
v["conf"] = v[P].max(axis=1)
v["p_true"] = [row[P[int(row["y"])]] for _, row in v.iterrows()]

for cls in (2, 3):
    m = v[(v.y == cls) & (v.pred == 4)]
    print(f"\n{CLASSES[cls]} -> predicted real: {len(m)} videos")
    print(m[["conf", "p_true"]].describe().loc[["mean", "min", "max"]].round(3))
    print("misses:", list(m.index)[:12])