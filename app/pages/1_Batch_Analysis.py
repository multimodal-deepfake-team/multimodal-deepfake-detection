"""Batch analysis: score many videos and export the results as CSV."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import csv  # noqa: E402
import io  # noqa: E402

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Batch Analysis · Deepfake Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.core.analyze import analyze_video, summarize  # noqa: E402
from app.core.config import (  # noqa: E402
    MAX_BATCH_FILES,
    SUPPORTED_VIDEO_EXTENSIONS,
    VIDEO_FRAME_COUNT,
)
from app.core.verdict import pct  # noqa: E402
from app.ui import charts, inject_theme  # noqa: E402
from app.ui.sidebar import render_sidebar  # noqa: E402
from app.ui.state import get_models, missing_checkpoints  # noqa: E402
from app.ui.theme import kpi, kpi_row  # noqa: E402

inject_theme()
render_sidebar("Batch Analysis")

st.markdown(
    '<div class="df-hero">'
    "<h1>Batch Analysis</h1>"
    "<p>Score up to "
    f"{MAX_BATCH_FILES} videos in one run, compare them side by side and "
    "export the results as CSV.</p>"
    "</div>",
    unsafe_allow_html=True,
)

missing = missing_checkpoints()
if missing:
    st.error(
        "Batch analysis is disabled because "
        + ", ".join(f"`{path}`" for path in missing)
        + " is/are missing. These checkpoints are committed to the repository, "
        "so a fresh clone or a deployed app already has them - if they are "
        "not there, pull the latest `main` branch (see README.md)."
    )
    st.stop()


# ============================================================
# INPUT
# ============================================================
st.markdown("##### 1 · Choose videos")

mode = st.radio(
    "Input mode",
    ["Upload files", "Server folder"],
    horizontal=True,
    help="Server folder reads from a directory on the machine running Streamlit.",
)

paths: list[Path] = []
folder_input = None

if mode == "Upload files":
    uploads = st.file_uploader(
        f"Select up to {MAX_BATCH_FILES} videos",
        type=list(SUPPORTED_VIDEO_EXTENSIONS),
        accept_multiple_files=True,
    )
    if uploads:
        uploads = sorted(uploads, key=lambda f: f.name)[:MAX_BATCH_FILES]

        if "_batch_dir" not in st.session_state:
            import tempfile

            st.session_state["_batch_dir"] = tempfile.mkdtemp(
                prefix="deepfake_batch_"
            )
        tmp_dir = Path(st.session_state["_batch_dir"])
        tmp_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for handle in uploads:
            target = tmp_dir / handle.name
            target.write_bytes(handle.getbuffer())
            paths.append(target)
else:
    folder_input = st.text_input(
        "Absolute folder path",
        placeholder=r"D:\Final\multimodal_deepfake_detection\dataset\sample_videos",
    )
    if folder_input:
        folder = Path(folder_input)
        if folder.is_dir():
            paths = sorted(
                p
                for p in folder.rglob("*")
                if p.suffix.lower().lstrip(".") in SUPPORTED_VIDEO_EXTENSIONS
            )[:MAX_BATCH_FILES]
        else:
            st.error(f"Not a directory: {folder}")

if paths:
    st.caption(f"{len(paths)} video(s) queued.")

st.markdown("##### 2 · Run")
run = st.button(
    "Run batch analysis",
    type="primary",
    disabled=not paths,
    use_container_width=False,
)

if run:
    try:
        bundle = get_models()
    except Exception as exc:  # noqa: BLE001
        st.error("Could not load the models.")
        st.code(str(exc))
        st.stop()

    import shutil
    import tempfile

    work_dir = Path(tempfile.mkdtemp(prefix="deepfake_batch_run_"))
    rows: list[dict] = []
    errors: list[tuple[str, str]] = []
    progress = st.progress(0.0, text="Starting…")

    try:
        for index, path in enumerate(paths, start=1):
            label = f"[{index}/{len(paths)}] {path.name}"
            progress.progress(
                index / len(paths),
                text=f"Analyzing {label} "
                f"({VIDEO_FRAME_COUNT} frames + audio windows)",
            )
            try:
                result = analyze_video(path, bundle, work_dir=work_dir)
                rows.append(summarize(result))
            except Exception as exc:  # noqa: BLE001
                errors.append((path.name, str(exc)))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        progress.empty()

    st.session_state["df_batch_rows"] = rows
    st.session_state["df_batch_errors"] = errors

    if errors:
        st.warning(f"{len(errors)} file(s) could not be analyzed.")


# ============================================================
# RESULTS
# ============================================================
rows = st.session_state.get("df_batch_rows", [])
errors = st.session_state.get("df_batch_errors", [])

if not rows:
    st.info(" Choose videos and run the batch to see results here.")
    st.stop()

st.divider()
st.markdown("##### Results")

fake = sum(1 for r in rows if r["status"] == "CONSISTENT FAKE")
real = sum(1 for r in rows if r["status"] == "CONSISTENT REAL")
other = len(rows) - fake - real

kpi_row(
    [
        ("Videos analyzed", str(len(rows)), "in this batch"),
        (
            "Flagged fake",
            str(fake),
            f"{fake / len(rows) * 100:.0f}% of the batch",
            "danger" if fake else "",
        ),
        (
            "Consistent real",
            str(real),
            f"{real / len(rows) * 100:.0f}% of the batch",
            "success" if real else "",
        ),
        (
            "Needs review",
            str(other),
            "inconclusive or video-only",
            "warning" if other else "",
        ),
    ],
    st.columns(4),
)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(charts.batch_status_breakdown(rows), theme=None)
with c2:
    st.plotly_chart(charts.batch_scores(rows), theme=None)

st.markdown("##### Detail")
table = st.dataframe(
    [
        {
            "File": r["file"],
            "Status": r["status"],
            "Risk": r["risk"],
            "Video": r["video_prediction"],
            "Video fake %": r["video_fake_pct"],
            "Audio": r["audio_prediction"],
            "Audio fake %": r["audio_fake_pct"],
            "Frames": r["frames"],
            "Faces": r["faces"],
            "Sec": r["seconds"],
        }
        for r in rows
    ],
    hide_index=True,
    height=360,
)

# ------------------------------------------------------------
# CSV export
# ------------------------------------------------------------
def to_csv(data: list[dict]) -> str:
    if not data:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(data[0]))
    writer.writeheader()
    writer.writerows(data)
    return buffer.getvalue()


c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    st.download_button(
        "⬇ Export results as CSV",
        data=to_csv(rows),
        file_name="batch_analysis_results.csv",
        mime="text/csv",
        type="primary",
    )
with c2:
    st.download_button(
        "⬇ Export as report (txt)",
        data="\n\n".join(
            f"{r['file']}\n  status={r['status']} risk={r['risk']}\n"
            f"  video={r['video_prediction']} ({pct((r['video_fake_pct'] or 0) / 100)} fake)\n"
            f"  audio={r['audio_prediction']}"
            + (
                f" ({pct((r['audio_fake_pct'] or 0) / 100)} fake)"
                if r["audio_fake_pct"] is not None
                else ""
            )
            for r in rows
        ),
        file_name="batch_analysis_report.txt",
        mime="text/plain",
    )

if errors:
    with st.expander(f"Errors ({len(errors)})"):
        st.table([{"File": f, "Error": e[:300]} for f, e in errors])
