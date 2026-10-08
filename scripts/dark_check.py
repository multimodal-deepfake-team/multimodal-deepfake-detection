import pandas as pd, numpy as np
from PIL import Image
df = pd.read_csv("splits/v2.csv")
df = df[df.split == "test"]
out = {}
for lab, d in df.groupby("label"):
    d = d.sample(min(150, len(d)), random_state=0)
    vals = []
    for p in d["path"]:
        a = np.asarray(Image.open(p).convert("L"), dtype=float)
        vals.append(((a < 25).mean(), a.mean()))
    v = np.array(vals)
    out[lab] = (v[:, 0].mean().round(3), v[:, 1].mean().round(1))
for k, (dark, mean) in out.items():
    print(f"{k:18s} dark_pixel_frac {dark}  mean_brightness {mean}")
