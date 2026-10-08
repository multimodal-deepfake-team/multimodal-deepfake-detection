import re
import pandas as pd
from pathlib import Path

df = pd.read_csv("splits/v1.csv")

def source(row):
    stem = Path(row["path"]).stem
    if row["label"] in ("audio_video_fake", "voiceclone"):
        return "fakeavceleb"
    if re.match(r"id\d+_id\d+_", stem) or re.match(r"id\d+_\d+", stem):
        return "celebdf"
    return "ffpp_or_other"

df["source"] = df.apply(source, axis=1)
print(pd.crosstab(df["label"], df["source"]))
df.to_csv("splits/v1_src.csv", index=False)