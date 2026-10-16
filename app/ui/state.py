"""Shared, process-wide Streamlit state.

Keeping the caches here means every page resolves the same model bundle
instead of reloading checkpoints once per page.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.core.config import PROJECT_ROOT, SAVED_MODELS_DIR, WEIGHTS_DIR
from app.core.models import ModelBundle, checkpoint_metadata, load_models


@st.cache_resource(show_spinner=False)
def get_models() -> ModelBundle:
    """Load (and cache) the video, audio and face models.

    Called on the first analysis; every later page or rerun reuses the same
    bundle.
    """
    return load_models()


@st.cache_data(show_spinner=False)
def get_model_info() -> dict[str, Any]:
    """Cheap checkpoint metadata (existence + size) for the sidebar.

    Deliberately does **not** load any weights.
    """
    return checkpoint_metadata()


@st.cache_data(show_spinner=False)
def models_ready() -> bool:
    """True when every checkpoint needed by the dashboard is present."""
    return all(checkpoint_metadata()["exists"].values())


@st.cache_data(show_spinner=False)
def missing_checkpoints() -> tuple[str, ...]:
    """Repo-relative paths of every checkpoint the dashboard cannot find.

    Empty in a healthy deployment: the three files are committed to the
    repository, so a fresh clone - and a Streamlit Community Cloud build -
    already has them. When the list is not empty the app explains exactly
    which file is missing instead of failing later inside the pipeline.
    """
    info = checkpoint_metadata()
    locations = {
        "video": (info["video_checkpoint"], SAVED_MODELS_DIR),
        "audio": (info["audio_checkpoint"], SAVED_MODELS_DIR),
        "face": (info["face_detector"], WEIGHTS_DIR),
    }

    return tuple(
        (directory / name).relative_to(PROJECT_ROOT).as_posix()
        for key, (name, directory) in locations.items()
        if not info["exists"][key]
    )
