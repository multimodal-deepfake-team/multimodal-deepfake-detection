"""End-to-end verification of the analysis pipeline.

Exercises ``app.core`` directly (no Streamlit), which is the same code path
the dashboard and the CLI use.

Run::

    python -m scripts.verify_app
"""

from __future__ import annotations

import dataclasses
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.analyze import analyze_video, summarize  # noqa: E402
from app.core.audio import extract_audio_from_video, predict_audio  # noqa: E402
from app.core.config import DEVICE  # noqa: E402
from app.core.models import AudioBinaryClassifier, load_models  # noqa: E402
from app.core.verdict import build_multimodal_status, create_text_report  # noqa: E402
from app.core.video import predict_video  # noqa: E402

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")

REAL_DIRS = [
    ROOT / "dataset" / "real" / "celebdf_real",
    ROOT / "dataset" / "real" / "ffpp_real",
    ROOT / "dataset" / "real" / "fakeav_rarv",
]
FAKE_DIRS = [
    ROOT / "dataset" / "ai_video" / "celebdf_fake",
    ROOT / "dataset" / "ai_video" / "ffpp_fake",
    ROOT / "dataset" / "edited" / "ffpp_face2face",
    ROOT / "dataset" / "edited" / "ffpp_neuraltextures",
]


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    # ------------------------------------------------------------
    banner("VERIFY 1: build_multimodal_status for every combination")
    # ------------------------------------------------------------
    for video, audio in (
        ("REAL", "REAL"),
        ("FAKE", "FAKE"),
        ("FAKE", "REAL"),
        ("REAL", "FAKE"),
        ("REAL", None),
    ):
        status, conclusion, risk = build_multimodal_status(video, audio)
        print(f"video={video:4s} audio={str(audio):4s} -> {status:18s} risk={risk}")
        print(f"     {conclusion}")

    # ------------------------------------------------------------
    banner("VERIFY 2: video predictions on labeled videos")
    # ------------------------------------------------------------
    models = load_models()
    print(f"device: {DEVICE}")

    random.seed(0)
    sample: list[tuple[str, Path]] = []
    for label, directories in (("REAL", REAL_DIRS), ("FAKE", FAKE_DIRS)):
        files: list[Path] = []
        for directory in directories:
            if directory.exists():
                files += sorted(
                    p for p in directory.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS
                )
        random.shuffle(files)
        sample += [(label, p) for p in files[:12]]

    if not sample:
        print("no labeled videos found under dataset/ -- skipping")
        correct = total = 0
    else:
        correct = total = 0
        face_frames = analyzed_frames = 0
        for label, path in sample:
            result = predict_video(path, models)
            ok = result["prediction"] == label
            correct += ok
            total += 1
            face_frames += result["faces_detected"]
            analyzed_frames += result["frames_analyzed"]
            print(
                f"{label:4s} -> {result['prediction']:4s} "
                f"fake={result['fake_probability']:.2f} "
                f"faces={result['faces_detected']}/{result['frames_analyzed']} "
                f"{'OK' if ok else 'WRONG'}  {path.parent.name}/{path.name}"
            )
        print(f"\nVIDEO ACCURACY: {correct}/{total} = {100 * correct / max(total, 1):.1f}%")
        print(f"faces detected: {face_frames}/{analyzed_frames} sampled frames")

    # ------------------------------------------------------------
    banner("VERIFY 3: audio on HELD-OUT VIDEO audio")
    # ------------------------------------------------------------
    import torch

    cases: list[tuple[str, Path]] = []
    for cls in ("real", "fake"):
        folder = ROOT / "processed" / "audio_v2" / "test" / cls
        cases += [(cls, p) for p in sorted(folder.glob("*.wav"))]

    if not cases:
        print("no processed/audio_v2/test clips -- run build_video_audio_dataset.py")
        return

    # The original ASVspoof checkpoint, for comparison.
    old_models = dataclasses.replace(models)
    old_ckpt = torch.load(
        ROOT / "saved_models" / "audio_binary_small_asvspoof.pth",
        map_location=DEVICE,
        weights_only=False,
    )
    old_head = AudioBinaryClassifier().to(DEVICE)
    old_head.classifier.load_state_dict(old_ckpt["classifier"])
    old_head.eval()
    old_models.audio_model = old_head
    old_models.audio_classes = old_ckpt.get("classes", {"real": 0, "fake": 1})

    new_ok = old_ok = 0
    windows_seen: list[int] = []

    for label, path in cases:
        new_result = predict_audio(path, models)
        old_result = predict_audio(path, old_models)

        ok_new = new_result["prediction"] == label.upper()
        ok_old = old_result["prediction"] == label.upper()
        new_ok += ok_new
        old_ok += ok_old
        windows_seen.append(new_result["windows"])

        print(
            f"{label:4s} new={new_result['prediction']:4s}"
            f"({new_result['fake_probability']:.2f}) "
            f"old={old_result['prediction']:4s} "
            f"{'OK' if ok_new else 'WRONG'}  {path.name}"
        )

    count = len(cases)
    print(f"\nOLD audio model: {old_ok}/{count} = {100 * old_ok / count:.1f}%")
    print(f"NEW audio model: {new_ok}/{count} = {100 * new_ok / count:.1f}%")
    print(
        f"windows per clip: min {min(windows_seen)}, max {max(windows_seen)}"
    )

    # ------------------------------------------------------------
    banner("VERIFY 4: full pipeline on one video (video + audio + report)")
    # ------------------------------------------------------------
    work = ROOT / "_verify_tmp"
    work.mkdir(exist_ok=True)

    try:
        # Prefer a video that actually carries an audio stream.
        path = None
        for audio_dir in (
            ROOT / "dataset" / "video_fake_audio_real",
            ROOT / "dataset" / "final",
        ):
            if path:
                break
            for candidate in sorted(audio_dir.rglob("*")):
                if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                probe = work / "probe.mp4"
                probe.write_bytes(candidate.read_bytes())
                ok, _ = extract_audio_from_video(probe, work / "probe.wav")
                if ok:
                    path = candidate
                    break

        if path is None:
            raise SystemExit("no video with an audio stream was found")

        print("test video:", path.relative_to(ROOT))
        result = analyze_video(path, models, work_dir=work)

        print(f"video : {result['video']['prediction']} "
              f"({result['video']['fake_probability']:.4f} fake, "
              f"{result['video']['faces_detected']}/{result['video']['frames_analyzed']} faces)")
        if result["audio"]:
            print(f"audio : {result['audio']['prediction']} "
                  f"({result['audio']['fake_probability']:.4f} fake, "
                  f"{result['audio']['windows']} windows)")
        else:
            print(f"audio : UNAVAILABLE - {result['audio_message']}")
        print(f"status: {result['status']} | risk: {result['risk']}")
        print(f"summary row: {summarize(result)}")
        print()
        print(result["report"])
    finally:
        shutil.rmtree(work, ignore_errors=True)

    banner("VERIFY COMPLETE")


if __name__ == "__main__":
    main()
