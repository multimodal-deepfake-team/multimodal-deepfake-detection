"""Shared sidebar rendered by every page of the dashboard."""

from __future__ import annotations

import streamlit as st

from app.core.config import (
    DEVICE,
    FFMPEG_EXE,
    PROJECT_ROOT,
    VIDEO_FRAME_COUNT,
)
from app.core.verdict import pct
from app.ui.state import get_model_info


def render_sidebar(page: str = "Analyze") -> dict:
    """Draw the branding, navigation hint, environment and checkpoint panels."""
    info = get_model_info()

    with st.sidebar:
        st.markdown(
            '<div class="df-hero" style="padding:.9rem 1rem;margin-bottom:.8rem">'
            '<h1 style="font-size:1.15rem"> Forensics Suite</h1>'
            f'<p style="font-size:.8rem">Multimodal deepfake detection · {page}</p>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Environment")
        st.code(
            f"device      : {DEVICE.type.upper()}\n"
            f"ffmpeg      : {'ok' if FFMPEG_EXE else 'missing'}\n"
            f"frames/video: {VIDEO_FRAME_COUNT}\n"
            f"project     : {PROJECT_ROOT.name}",
            language="text",
        )

        st.markdown("#### Checkpoints")
        exists = info.get("exists", {})
        st.write(f"{'✅' if exists.get('video') else '❌'} `{info['video_checkpoint']}`")
        st.write(f"{'✅' if exists.get('audio') else '❌'} `{info['audio_checkpoint']}`")
        st.write(f"{'✅' if exists.get('face') else '❌'} `{info['face_detector']}`")

        sizes = [
            f"{info[k]} MB"
            for k in ("video_size_mb", "audio_size_mb", "face_size_mb")
            if k in info
        ]
        if sizes:
            st.caption(" + ".join(sizes) + " on disk")
        st.caption("Weights are loaded once, on the first analysis.")

        st.divider()
        st.markdown("#### Pages")
        st.caption(
            " **Analyze** — single video verdict\n\n"
            " **Batch Analysis** — score many files, export CSV\n\n"
            " **Model Metrics** — evaluation reports\n\n"
            " **About** — pipeline and limitations"
        )

    return info
