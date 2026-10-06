import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("splits/manifest.csv")

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
    if ra != rb:
        parent[ra] = rb

for t in df["tokens"]:
    toks = t.split("|")
    for x in toks[1:]:
        union(toks[0], x)

df["group"] = df["tokens"].map(lambda s: find(s.split("|")[0]))

# group size report
g = df.groupby("group").size().sort_values(ascending=False)
print("number of groups:", len(g))
print("largest 5 groups (crops):", g.head(5).tolist())
print("largest group share: %.1f%%" % (100 * g.iloc[0] / len(df)))

# stratified group split: ~71/14/14 via 7 folds
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
fold = pd.Series(-1, index=df.index)
for i, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["group"])):
    fold.iloc[idx] = i

df["split"] = "train"
df.loc[fold == 0, "split"] = "val"
df.loc[fold == 1, "split"] = "test"

df.to_csv("splits/v1.csv", index=False)
print()
print(pd.crosstab(df["label"], df["split"]))
print("\nvideos per split and class:")
print(df.groupby(["label", "split"])["video_key"].nunique().unstack(fill_value=0))
