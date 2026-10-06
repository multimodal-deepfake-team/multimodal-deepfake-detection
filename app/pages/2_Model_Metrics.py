"""Model metrics: evaluation reports, confusion matrix and per-class scores.

Everything on this page is read from the text reports already checked into
``results/`` and ``processed/`` - no model is loaded, so it renders instantly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Model Metrics · Deepfake Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.core.metrics import load_all_metrics  # noqa: E402
from app.ui import charts, inject_theme  # noqa: E402
from app.ui.sidebar import render_sidebar  # noqa: E402
from app.ui.theme import kpi, kpi_row  # noqa: E402

inject_theme()
render_sidebar("Model Metrics")

st.markdown(
    '<div class="df-hero">'
    "<h1>Model Metrics</h1>"
    "<p>Validation and evaluation reports for every component of the "
    "pipeline, read straight from the checked-in artefacts.</p>"
    "</div>",
    unsafe_allow_html=True,
)

metrics = load_all_metrics()
validation = metrics["six_class_validation"]
matrix = metrics["confusion_matrix"]
video_eval = metrics["video_evaluation"]
fusion_eval = metrics["fusion_evaluation"]

# ============================================================
# HEADLINE NUMBERS
# ============================================================
st.markdown("##### Headline")

cards = [
    (
        "6-class accuracy",
        f"{validation['accuracy'] * 100:.0f}%",
        f"{validation['samples']:,} validation images",
        "success",
    )
]

if video_eval and video_eval["scalars"].get("accuracy") is not None:
    cards.append(
        (
            "Video-only (fusion labels)",
            f"{video_eval['scalars']['accuracy'] * 100:.1f}%",
            f"n = {int(video_eval['scalars'].get('samples evaluated', 0))}",
        )
    )

if fusion_eval and fusion_eval["scalars"].get("accuracy") is not None:
    cards.append(
        (
            "Video + audio fusion",
            f"{fusion_eval['scalars']['accuracy'] * 100:.1f}%",
            f"n = {int(fusion_eval['scalars'].get('samples evaluated', 0))}",
            "warning",
        )
    )

if matrix:
    total = sum(sum(row) for row in matrix["matrix"])
    cards.append(("Confusion matrix", f"{total:,}", "samples scored", ""))

while len(cards) < 4:
    cards.append(("Pending", "—", "no report found", "warning"))

kpi_row(cards[:4], st.columns(4))

st.caption(validation["source"])

st.divider()

# ============================================================
# CONFUSION MATRIX + PER-CLASS
# ============================================================
st.markdown("##### 6-class video model")

if matrix:
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.plotly_chart(
            charts.confusion_matrix_heatmap(matrix["labels"], matrix["matrix"]),
            theme=None,
        )
    with c2:
        st.plotly_chart(
            charts.per_class_bar(
                validation["per_class"],
                title="Validation precision / recall / F1",
            ),
            theme=None,
        )
else:
    st.info(
        "No confusion matrix found - expected `results/6class_video_confusion_matrix.txt`."
    )

c1, c2 = st.columns([2, 3], gap="large")
with c1:
    st.plotly_chart(charts.support_bar(validation["per_class"]), theme=None)
with c2:
    st.markdown("###### Validation classification report")
    st.dataframe(validation["per_class"], hide_index=True, height=320)

st.caption(
    "voiceclone is the weakest class (49% precision) because only 253 "
    "voice-clone faces were available - a tracked limitation, not a defect."
)

st.divider()

# ============================================================
# EVALUATION RUNS
# ============================================================
st.markdown("##### Evaluation runs")

comparison: dict[str, dict[str, float]] = {}
if video_eval:
    comparison["Video-only"] = {
        k: v for k, v in video_eval["scalars"].items() if k in ("accuracy", "macro f1", "weighted f1")
    }
if fusion_eval:
    comparison["Fusion"] = {
        k: v for k, v in fusion_eval["scalars"].items() if k in ("accuracy", "macro f1", "weighted f1")
    }
comparison["6-class (val)"] = {
    "accuracy": validation["accuracy"],
    "macro f1": validation["macro_avg"]["f1"],
    "weighted f1": validation["weighted_avg"]["f1"],
}

if comparison:
    st.plotly_chart(charts.comparison_grouped_bars(comparison), theme=None)

tab_a, tab_b, tab_c = st.tabs(
    [" Video-only report", " Fusion report", " Raw artefacts"]
)

with tab_a:
    if video_eval and video_eval["rows"]:
        st.dataframe(video_eval["rows"], hide_index=True, height=300)
        st.caption(f"Source: `{video_eval['path']}`")
    else:
        st.info("No video-only evaluation report found.")

with tab_b:
    if fusion_eval and fusion_eval["rows"]:
        st.dataframe(fusion_eval["rows"], hide_index=True, height=300)
        st.caption(f"Source: `{fusion_eval['path']}`")
    else:
        st.info("No fusion evaluation report found.")

with tab_c:
    for label, path in (
        ("results/6class_video_confusion_matrix.txt", "confusion_matrix"),
        ("results/video_evaluation.txt", "video_evaluation"),
        ("processed/fusion_evaluation.txt", "fusion_evaluation"),
    ):
        available = metrics["available_files"][path]
        st.write(f"{'✅' if available else '❌'} `{label}`")
        if available:
            with st.expander(f"Contents of {Path(label).name}"):
                full = _PROJECT_ROOT / label
                st.text(full.read_text(encoding="utf-8", errors="replace")[:6000])
