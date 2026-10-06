"""Components that render one analysis result.

Kept out of the page script so they can be exercised directly (and reused by
any future page that shows a verdict).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.core.verdict import pct
from app.ui import charts
from app.ui.theme import kpi_row, verdict_banner


def render_evidence(result: dict) -> None:
    """Overview / video / audio / report tabs for a single result."""
    video = result["video"]
    audio = result.get("audio")

    tab_overview, tab_video, tab_audio, tab_report = st.tabs(
        [" Overview", " Video evidence", " Audio evidence", " Report"]
    )

    # ---------------- overview ----------------
    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                charts.fake_probability_gauge(video["fake_probability"], "Video branch"),
                theme=None,
            )
        with c2:
            if audio:
                st.plotly_chart(
                    charts.fake_probability_gauge(audio["fake_probability"], "Audio branch"),
                    theme=None,
                )
            else:
                st.plotly_chart(
                    charts.fake_probability_gauge(
                        video["fake_probability"], "Audio branch (unavailable)"
                    ),
                    theme=None,
                )
                st.info("No usable audio track - only the visual branch was scored.")

        st.markdown(
            '<p class="df-caption" style="text-align:center">'
            "Both branches are reported independently - no weighted blend.</p>",
            unsafe_allow_html=True,
        )

    # ---------------- video ----------------
    with tab_video:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prediction", video["prediction"])
        c2.metric("Real probability", pct(video["real_probability"]))
        c3.metric("Fake probability", pct(video["fake_probability"]))
        c4.metric(
            "Faces detected", f"{video['faces_detected']}/{video['frames_analyzed']}"
        )

        st.plotly_chart(charts.frame_probabilities(video["frame_details"]), theme=None)

        if not video["face_crop_used"]:
            st.warning(
                "No face could be detected in the sampled frames, so full frames "
                "were analyzed. Expect lower reliability."
            )

        crops = video.get("face_crops") or []
        if crops:
            st.markdown("##### Face crops fed to the classifier")
            columns = st.columns(min(len(crops), 5))
            for column, (frame_index, crop) in zip(columns, crops):
                with column:
                    st.image(crop, caption=f"frame {frame_index}", width=160)

        st.markdown("##### Per-frame scores")
        st.dataframe(
            [
                {
                    "Frame": d["frame"],
                    "Prediction": d["prediction"],
                    "Real": pct(d["real"]),
                    "Fake": pct(d["fake"]),
                    "Confidence": pct(d["confidence"]),
                }
                for d in video["frame_details"]
            ],
            hide_index=True,
        )

        st.caption(
            "The video branch outputs REAL/FAKE only - it does not localise the "
            "manipulated region."
        )

    # ---------------- audio ----------------
    with tab_audio:
        if audio:
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", audio["prediction"])
            c2.metric("Real probability", pct(audio["real_probability"]))
            c3.metric("Fake probability", pct(audio["fake_probability"]))

            st.plotly_chart(
                charts.audio_window_bars(audio["real_probability"], audio["fake_probability"]),
                theme=None,
            )
            st.plotly_chart(
                charts.real_fake_donut(
                    audio["real_probability"], audio["fake_probability"], "Audio branch"
                ),
                theme=None,
            )

            st.markdown(
                "-  Track converted to 16 kHz mono\n"
                f"-  **{audio['windows']}** overlapping 4-second windows scored and averaged\n"
                "-  Wav2Vec2 embedding + MLP head"
            )

            wav_bytes = result.get("wav_bytes")
            if wav_bytes:
                st.audio(wav_bytes, format="audio/wav")
        else:
            st.info(
                "Audio evidence unavailable: "
                + result.get("audio_message", "unknown reason")
            )

    # ---------------- report ----------------
    with tab_report:
        st.markdown("##### Final summary")
        st.write(f"**File:** {result['filename']}")
        st.write(f"**Video:** {video['prediction']} ({pct(video['fake_probability'])} fake)")
        st.write(
            "**Audio:** "
            + (
                f"{audio['prediction']} ({pct(audio['fake_probability'])} fake)"
                if audio
                else "UNAVAILABLE"
            )
        )
        st.write(f"**Status:** {result['status']}")
        st.write(f"**Risk level:** {result['risk']}")
        st.write(f"**Assessment:** {result['conclusion']}")

        st.download_button(
            "⬇ Download forensic report",
            data=result["report"],
            file_name=f"forensic_report_{Path(result['filename']).stem}.txt",
            mime="text/plain",
            type="primary",
        )

        with st.expander("Plain-text preview"):
            st.code(result["report"], language="text")


def render_result(result: dict) -> None:
    """Verdict banner + KPI row + evidence tabs for one result."""
    video = result["video"]
    audio = result.get("audio")

    st.divider()
    st.markdown("##### Result")
    verdict_banner(result["status"], result["conclusion"], result["risk"])

    kpi_row(
        [
            (
                "Status",
                result["status"],
                f"risk: {result['risk']}",
                "danger"
                if result["status"] == "CONSISTENT FAKE"
                else ("success" if result["status"] == "CONSISTENT REAL" else "warning"),
            ),
            (
                "Video branch",
                video["prediction"],
                f"{pct(video['confidence'])} confidence",
                "danger" if video["prediction"] == "FAKE" else "success",
            ),
            (
                "Audio branch",
                audio["prediction"] if audio else "UNAVAILABLE",
                f"{pct(audio['confidence'])} confidence" if audio else "no audio track",
                ("danger" if audio["prediction"] == "FAKE" else "success") if audio else "",
            ),
            (
                "Analyzed in",
                f"{result['elapsed_seconds']} s",
                f"{video['frames_analyzed']} frames",
            ),
        ],
        st.columns(4),
    )

    render_evidence(result)
