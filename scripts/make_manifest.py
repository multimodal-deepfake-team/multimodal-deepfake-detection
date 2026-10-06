import re, csv
from pathlib import Path
from collections import Counter

ROOT = Path("processed/faces")
OUT = Path("splits/manifest.csv")
OUT.parent.mkdir(exist_ok=True)

FRAME_SUFFIX = re.compile(r"(_frame)?_\d{4}$")   # strips _0002 or _frame_0000

def video_id(stem: str) -> str:
    return FRAME_SUFFIX.sub("", stem)

def source_tokens(cls: str, vid: str):
    """Identity/source tokens used to link related videos."""
    # FF++ fakes: 008_990  -> both source clips
    m = re.fullmatch(r"(\d{3})_(\d{3})", vid)
    if m:
        return [f"ff{m[1]}", f"ff{m[2]}"]
    # FF++ real: 3-digit id
    if cls == "real" and re.fullmatch(r"\d{3}", vid):
        return [f"ff{vid}"]
    # everything else is its own unit
    return [f"{cls}::{vid}"]

rows = []
for p in sorted(ROOT.rglob("*.jpg")):
    cls = p.parent.name
    vid = video_id(p.stem)
    toks = source_tokens(cls, vid)
    rows.append([str(p), cls, f"{cls}::{vid}", "|".join(toks)])

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path", "label", "video_key", "tokens"])
    w.writerows(rows)

print(f"{len(rows)} crops")
print("videos per class:")
vids = {(r[1], r[2]) for r in rows}
for c, n in sorted(Counter(c for c, _ in vids).items()):
    print(f"  {c:18s} {n}")