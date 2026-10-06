import re
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold

FRAME_SUFFIX = re.compile(r"(_frame)?_\d{4}$")
rows = []

# 1) old crops (v1 manifest), minus voiceclone
old = pd.read_csv("splits/manifest.csv")
old = old[old["label"] != "voiceclone"]
for _, r in old.iterrows():
    toks = r["tokens"]
    if r["label"] == "audio_video_fake":
        m = re.search(r"(id\d{5})", r["path"])
        if m:
            toks = f"fav::{m[1]}"
    rows.append([r["path"], r["label"], r["video_key"], toks, "old"])

# 2) new FakeAVCeleb real crops
for p in sorted(Path("processed/faces_v2/real_fakeavceleb").glob("*.jpg")):
    vid = FRAME_SUFFIX.sub("", p.stem)           # id00076_00109
    ident = vid.split("_")[0]                    # id00076
    rows.append([str(p), "real", f"real_fav::{vid}", f"fav::{ident}", "fav_real"])

df = pd.DataFrame(rows, columns=["path", "label", "video_key", "tokens", "origin"])

# union-find over tokens
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
for t in df["tokens"]:
    toks = t.split("|")
    for x in toks[1:]: union(toks[0], x)
df["group"] = df["tokens"].map(lambda s: find(s.split("|")[0]))

g = df.groupby("group").size().sort_values(ascending=False)
print("groups:", len(g), "| largest share: %.1f%%" % (100 * g.iloc[0] / len(df)))

sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
fold = pd.Series(-1, index=df.index)
for i, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["group"])):
    fold.iloc[idx] = i
df["split"] = "train"
df.loc[fold == 0, "split"] = "val"
df.loc[fold == 1, "split"] = "test"

df.to_csv("splits/v2.csv", index=False)
print(pd.crosstab(df["label"], df["split"]))
print(df.groupby(["label", "split"])["video_key"].nunique().unstack(fill_value=0))

# leakage check
for col in ["video_key", "group"]:
    s = {k: set(df[df.split == k][col]) for k in ["train", "val", "test"]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (s[a] & s[b]), f"LEAK in {col} {a}/{b}"
print("No video or group leakage.")