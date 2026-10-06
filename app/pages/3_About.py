"""About: architecture, models, dataset, project layout and limitations."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="About · Deepfake Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.core.config import (  # noqa: E402
    AUDIO_SECONDS,
    DEVICE,
    VIDEO_FRAME_COUNT,
    WAV2VEC_MODEL,
)
from app.ui import inject_theme  # noqa: E402
from app.ui.sidebar import render_sidebar  # noqa: E402

inject_theme()
render_sidebar("About")

st.markdown(
    '<div class="df-hero">'
    "<h1>About this project</h1>"
    "<p>An explainable, multimodal deepfake detector that classifies social "
    "media video and reports the evidence behind every verdict.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ============================================================
tab_pipeline, tab_models, tab_data, tab_layout, tab_limits = st.tabs(
    [" Pipeline", " Models", " Dataset", " Project layout", " Limitations"]
)

# ------------------------------------------------------------
with tab_pipeline:
    st.markdown("#### How a video is scored")
    st.code(
        """
Input video
   │
   ├─► visual branch ─────────────────────────────┐
   │     10 frames sampled evenly                │
   │     YOLOv8-face detect + crop               │
   │     ResNet18 → per-frame P(fake)            │
   │     mean over frames                        │
   │                                             ├─► verdict
   ├─► audio branch ─────────────────────────────┤     CONSISTENT FAKE
   │     ffmpeg → 16 kHz mono                    │     CONSISTENT REAL
   │     {sec}s windows, 50% overlap             │     INCONCLUSIVE
   │     Wav2Vec2 → P(fake) per window           │     VIDEO-ONLY
   │     mean over windows                       │
   │                                             │
   └─────────────────────────────────────────────┘
        """.replace("{sec}", str(AUDIO_SECONDS)),
        language="text",
    )
    st.info(
        "The two branches are **never** blended into a single weighted score. "
        "Each reports an independent REAL/FAKE probability; only the final "
        "status combines them."
    )

    st.markdown("#### Verdict rules")
    st.table(
        [
            {"Video": "FAKE", "Audio": "FAKE", "Status": "CONSISTENT FAKE", "Risk": "HIGH"},
            {"Video": "REAL", "Audio": "REAL", "Status": "CONSISTENT REAL", "Risk": "LOW"},
            {"Video": "FAKE", "Audio": "REAL", "Status": "INCONCLUSIVE", "Risk": "REVIEW REQUIRED"},
            {"Video": "REAL", "Audio": "FAKE", "Status": "INCONCLUSIVE", "Risk": "REVIEW REQUIRED"},
            {"Video": "any", "Audio": "missing", "Status": "VIDEO-ONLY", "Risk": "REVIEW REQUIRED"},
        ]
    )

# ------------------------------------------------------------
with tab_models:
    st.markdown("#### Components")
    st.table(
        [
            {
                "Component": "Face detector",
                "Architecture": "YOLOv8n-face",
                "Weights": "weights/yolov8n-face.pt",
                "Role": "crop the face before visual scoring",
            },
            {
                "Component": "Video branch",
                "Architecture": "ResNet18 (2-way head)",
                "Weights": "saved_models/video_binary_small.pth",
                "Role": f"{VIDEO_FRAME_COUNT} frames → mean P(fake)",
            },
            {
                "Component": "Audio branch",
                "Architecture": f"{WAV2VEC_MODEL} + MLP",
                "Weights": "saved_models/audio_binary_small.pth",
                "Role": f"{AUDIO_SECONDS}s windows → mean P(fake)",
            },
            {
                "Component": "6-class CLI",
                "Architecture": "ResNet18 (6-way head)",
                "Weights": "saved_models/social_media_detector.pth",
                "Role": "categorical report via detect_social_video.py",
            },
        ]
    )

    st.markdown("#### Runtime")
    st.code(
        f"device       : {DEVICE.type.upper()}\n"
        f"frame count  : {VIDEO_FRAME_COUNT}\n"
        f"audio window : {AUDIO_SECONDS}s @ 50% overlap\n"
        f"sample rate  : 16000 Hz",
        language="text",
    )

    st.caption(
        "Fusion weights exist (saved_models/fusion_model_best.pth) but the "
        "dashboard deliberately shows the branches independently."
    )

# ------------------------------------------------------------
with tab_data:
    st.markdown("#### Datasets used for training")
    st.table(
        [
            {"Dataset": "FaceForensics++", "Supplies": "faceswap, edited (F2F/NeuralTextures), real", "Role": "primary"},
            {"Dataset": "Celeb-DF v2", "Supplies": "deepfake, real", "Role": "primary + cross-check"},
            {"Dataset": "FakeAVCeleb", "Supplies": "voiceclone, audio_video_fake", "Role": "audio classes"},
            {"Dataset": "ASVspoof2019 LA", "Supplies": "spoof / bona fide audio", "Role": "audio pretraining"},
        ]
    )

    st.markdown("#### Layout")
    st.code(
        """
dataset/      raw source videos and archives   (gitignored)
processed/    extracted frames, face crops, audio features (gitignored)
saved_models/ trained checkpoints              (gitignored)
weights/      YOLO face detector               (gitignored)
results/      evaluation reports checked into git
        """.strip(),
        language="text",
    )
    st.caption("Datasets and weights are excluded from git because of size/licensing.")

# ------------------------------------------------------------
with tab_layout:
    st.markdown("#### Repository layout")
    st.code(
        """
multimodal_deepfake_detection/
├── .streamlit/config.toml     dark theme + server config
├── detect_social_video.py     CLI entry point (6-class)
├── app/
│   ├── streamlit_app.py       Analyze page (entry point)
│   ├── pages/                 Batch Analysis · Model Metrics · About
│   ├── core/                  Streamlit-free pipeline
│   │   ├── config.py          paths, constants, ffmpeg discovery
│   │   ├── models.py          architectures + checkpoint loading
│   │   ├── video.py           frame sampling, face crop, scoring
│   │   ├── audio.py           extraction, windowing, scoring
│   │   ├── verdict.py         status rules + report builder
│   │   ├── analyze.py         end-to-end orchestrator
│   │   └── metrics.py         evaluation-report parsers
│   └── ui/                    theme, charts, sidebar, shared state
├── models/                    architecture definitions
├── preprocessing/             frame / face / audio extraction
├── training/                  training + evaluation entry points
├── scripts/                   dataset build, checkpoints, verification
├── utils/                     dataset summary helper
├── results/                   evaluation reports (committed)
├── saved_models/              checkpoints                (gitignored)
├── weights/                   YOLO weights               (gitignored)
├── dataset/  processed/       data                       (gitignored)
├── README.md
└── PROJECT_DOCUMENTATION.md   full technical report
        """.strip(),
        language="text",
    )

    st.markdown("#### Commands")
    st.code(
        """
# dashboard
streamlit run app/streamlit_app.py

# CLI inference (6-class)
python detect_social_video.py <video.mp4>

# training
python -m training.train_video
python -m training.train_audio_v2

# verification
python -m scripts.verify_app
        """.strip(),
        language="bash",
    )

# ------------------------------------------------------------
with tab_limits:
    st.markdown("#### Known limitation: cross-dataset generalization")
    st.markdown(
        "The 6-class model was validated at **95% accuracy on held-out data "
        "from the same distribution it was trained on** (FaceForensics++, "
        "Celeb-DF, FakeAVCeleb - landscape, medium-shot, c40-compressed "
        "footage)."
    )
    st.markdown(
        "On two independently sourced, in-the-wild vertical mobile videos "
        "(one AI-generated, one WhatsApp-manipulated) the model labelled both "
        "`real` with 80-88% confidence. This is a **distribution shift**, not "
        "a code defect."
    )
    st.markdown(
        "**Planned fix:** add DFDC in-the-wild footage, rebuild the frame → "
        "face-crop pipeline, retrain, and re-evaluate on the same test videos."
    )

    st.warning(
        "A REAL verdict means \"not detected\" - never proof of authenticity. "
        "Always pair the score with human review for consequential decisions."
    )

st.divider()
st.caption("See PROJECT_DOCUMENTATION.md for the full technical report.")
