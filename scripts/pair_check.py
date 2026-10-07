import pandas as pd

CLASSES = ["ai_video", "audio_video_fake", "edited", "faceswap", "real"]
P = [f"p{k}" for k in range(5)]
t = pd.read_csv("results/v3_test_predictions.csv")
g = t.groupby("video_key")
v = g[P].mean()
v["y"] = g["y"].first()
v["pred"] = v[P].values.argmax(1)
v["true"] = v["y"].map(dict(enumerate(CLASSES)))
v["pred_cls"] = v["pred"].map(dict(enumerate(CLASSES)))
v["p_real"] = v["p4"]
v["n_crops"] = g.size()

keys = ["055", "147", "953", "974", "266", "252", "556", "588"]
m = v[[any(k in idx for k in keys) for idx in v.index]]
print(m[["true", "pred_cls", "p_real", "n_crops"]].round(2).sort_index().to_string())