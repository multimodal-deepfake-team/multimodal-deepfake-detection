import pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, f1_score

CLASSES = ["ai_video", "audio_video_fake", "edited", "faceswap", "real"]
t = pd.read_csv("results/v3_test_predictions.csv")
P = [f"p{k}" for k in range(5)]

def topk_mean(df, k=0.5):
    # video prob = mean over the top fraction of crops, ranked by max non-real prob
    s = df[P[:4]].max(axis=1)
    n = max(1, int(np.ceil(len(df) * k)))
    return df.loc[s.nlargest(n).index, P].mean()

res = {}
for name, fn in [("mean", lambda d: d[P].mean()),
                 ("topk50", lambda d: topk_mean(d, 0.5)),
                 ("topk25", lambda d: topk_mean(d, 0.25))]:
    rows, ys = [], []
    for _, d in t.groupby("video_key"):
        rows.append(fn(d).values); ys.append(d["y"].iloc[0])
    pred = np.array(rows).argmax(1); ys = np.array(ys)
    print(f"{name:7s} acc {accuracy_score(ys, pred):.3f}  macroF1 {f1_score(ys, pred, average='macro'):.3f}")