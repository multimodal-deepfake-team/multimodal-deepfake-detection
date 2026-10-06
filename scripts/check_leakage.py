import pandas as pd
df = pd.read_csv("splits/v1.csv")

for col in ["video_key", "group"]:
    s = {k: set(df[df.split == k][col]) for k in ["train", "val", "test"]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        n = len(s[a] & s[b])
        print(f"{col:10s} {a}/{b} overlap: {n}")
        assert n == 0, f"LEAK in {col}"
print("No video or group leakage.")