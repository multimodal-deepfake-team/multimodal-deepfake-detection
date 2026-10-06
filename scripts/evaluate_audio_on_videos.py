"""Measure app audio predictions on LABELED VIDEO AUDIO.

Label is the audio ground truth of the source video:
  REAL audio : FFPP/Celeb-DF videos only alter faces -> audio is original
               FakeAVCeleb RaV  (real audio, real video)
  FAKE audio : FakeAVCeleb RfV  (fake audio, real video)  = voiceclone
               FakeAVCeleb FfV  (fake audio, fake video)  = audio_video_fake
"""
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import streamlit_app as app  # noqa: E402

info = app.load_models()

REAL_AUDIO = [
    ROOT / "dataset" / "real" / "fakeav_rarv",
    ROOT / "dataset" / "real" / "ffpp_real",
    ROOT / "dataset" / "real" / "celebdf_real",
    ROOT / "dataset" / "ai_video" / "ffpp_fake",
    ROOT / "dataset" / "edited" / "ffpp_face2face",
]
FAKE_AUDIO = [
    ROOT / "dataset" / "voiceclone" / "fakeav_farv",
    ROOT / "dataset" / "audio_video_fake" / "fakeav_fafv",
]

N_PER_GROUP = 15
random.seed(7)

work = ROOT / "_diag_audio_tmp"
work.mkdir(exist_ok=True)

rows = []
for label, dirs in (("REAL", REAL_AUDIO), ("FAKE", FAKE_AUDIO)):
    picked = []
    per = max(1, N_PER_GROUP // len(dirs))
    for d in dirs:
        if not d.exists():
            continue
        files = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov"))
        random.shuffle(files)
        picked += [(label, p) for p in files[:per]]
    for lab, path in picked:
        wav = work / "clip.wav"
        if wav.exists():
            wav.unlink()
        ok, msg = app.extract_audio_from_video(path, wav)
        if not ok:
            print(f"{lab}  EXTRACT-FAIL {path.parent.name}/{path.name}: {msg}")
            continue
        res = app.predict_audio(wav, info)
        rows.append({
            "label": lab,
            "pred": res["prediction"],
            "fake": res["fake_probability"],
            "real": res["real_probability"],
            "path": f"{path.parent.name}/{path.name}",
        })
        print(f"{lab}  -> {res['prediction']:4s} fake={res['fake_probability']:.3f} "
              f"{path.parent.name}/{path.name}", flush=True)

print()
print("=" * 70)
n = len(rows)
correct = sum(1 for r in rows if r["pred"] == r["label"])
print(f"OVERALL: {correct}/{n} = {100 * correct / max(n, 1):.1f}%")

for lab in ("REAL", "FAKE"):
    sub = [r for r in rows if r["label"] == lab]
    if not sub:
        continue
    c = sum(1 for r in sub if r["pred"] == lab)
    mean_fake = sum(r["fake"] for r in sub) / len(sub)
    print(f"  actual {lab}: {c}/{len(sub)} correct "
          f"(mean fake prob = {mean_fake:.3f})")

wrong = [r for r in rows if r["pred"] != r["label"]]
print(f"\nmisclassified: {len(wrong)}")
for r in sorted(wrong, key=lambda x: -abs(x["fake"] - 0.5)):
    print(f"  {r['label']} -> {r['pred']}  fake={r['fake']:.3f}  {r['path']}")
