"""Multimodal Deepfake Detection - main analysis page.

Run with::

    streamlit run app/streamlit_app.py

The heavy lifting lives in ``app.core`` (Streamlit-free) and the presentation
helpers live in ``app.ui``, so this file is only layout and orchestration.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable no matter how Streamlit was launched.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Multimodal Deepfake Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.core.analyze import analyze_video  # noqa: E402
from app.core.config import (  # noqa: E402
    SUPPORTED_VIDEO_EXTENSIONS,
    VIDEO_FRAME_COUNT,
)
from app.ui import inject_theme  # noqa: E402
from app.ui.components import render_result  # noqa: E402
from app.ui.sidebar import render_sidebar  # noqa: E402
from app.ui.state import get_models, missing_checkpoints  # noqa: E402
from app.ui.theme import kpi  # noqa: E402

inject_theme()


# ============================================================
# PAGE
# ============================================================
render_sidebar("Analyze")

st.markdown(
    '<div class="df-hero">'
    "<h1>Multimodal Deepfake Detection</h1>"
    "<p>Independent visual + audio forensics on a single video, "
    "combined into one evidence-backed verdict.</p>"
    "</div>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload a video to analyze",
    type=list(SUPPORTED_VIDEO_EXTENSIONS),
    help="mp4, mov, avi or mkv - the file is processed in memory and never stored.",
)

if uploaded is None:
    st.info(
        " Upload a video above to get started, or open **Batch Analysis**, "
        "**Model Metrics** or **About** from the sidebar."
    )
    col1, col2, col3 = st.columns(3)
    col1.markdown(kpi("Video model", "ResNet18", "face-cropped 10-frame sampling"), unsafe_allow_html=True)
    col2.markdown(kpi("Audio model", "Wav2Vec2", "4 s windows, 50% overlap"), unsafe_allow_html=True)
    col3.markdown(kpi("Fusion", "Independent", "no weighted blend", "warning"), unsafe_allow_html=True)
    st.stop()


missing = missing_checkpoints()
if missing:
    st.error(
        "Analysis is disabled because "
        + ", ".join(f"`{path}`" for path in missing)
        + " is/are missing. These checkpoints are committed to the repository, "
        "so a fresh clone or a deployed app already has them - if they are "
        "not there, pull the latest `main` branch (see README.md)."
    )
    st.info(
        "**Model Metrics** and **About** still work without the weights - "
        "open them from the sidebar."
    )
    st.stop()

work_dir = Path(tempfile.mkdtemp(prefix="deepfake_"))
video_path = work_dir / f"input_{uploaded.name}"

# Keep the last verdict on screen across reruns (tab clicks, downloads, ...)
# as long as the same file is still uploaded.
stored = st.session_state.get("df_last_result")
if stored and stored.get("filename") != uploaded.name:
    st.session_state.pop("df_last_result", None)
    stored = None

try:
    video_path.write_bytes(uploaded.getbuffer())

    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        st.markdown("##### Preview")
        st.video(uploaded)
    with c2:
        st.markdown("##### Run")
        st.caption(
            f"{VIDEO_FRAME_COUNT} frames sampled across the clip, each "
            "face-detected and cropped before scoring."
        )
        run = st.button(" Run analysis", type="primary", use_container_width=True)
        if stored:
            st.caption(f" Last analyzed in {stored['elapsed_seconds']} s.")

    if run:
        try:
            with st.spinner("Loading models (first run only)..."):
                bundle = get_models()
        except Exception as exc:  # noqa: BLE001
            st.error("Could not load the models.")
            st.code(str(exc))
            st.stop()

        try:
            with st.spinner("Analyzing video frames, audio and building the report..."):
                stored = analyze_video(
                    video_path,
                    bundle,
                    work_dir=work_dir,
                    filename=uploaded.name,
                )
            st.session_state["df_last_result"] = stored
        except Exception as exc:  # noqa: BLE001
            st.error("Analysis failed. Details below.")
            st.exception(exc)
            stored = None

    if stored:
        render_result(stored)

finally:
    shutil.rmtree(work_dir, ignore_errors=True)

st.divider()
st.caption(
    "Deepfake analysis is probabilistic. Treat a REAL verdict as "
    '"not detected", not as proof of authenticity.'
)
