"""Dashboard theme: custom CSS layered on top of ``.streamlit/config.toml``.

``inject_theme`` is idempotent - it only writes the ``<style>`` block once per
browser session, so every page can call it safely.
"""

from __future__ import annotations

import streamlit as st

ACCENT = "#6366F1"
SUCCESS = "#22C55E"
DANGER = "#EF4444"
WARNING = "#F59E0B"
PANEL = "#131C31"
TEXT = "#E2E8F0"
MUTED = "#94A3B8"

_CSS = f"""
<style>
/* ---------- layout ---------- */
.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    max-width: 1280px;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {PANEL} 0%, #0B1120 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}}

/* ---------- hero ---------- */
.df-hero {{
    background: linear-gradient(120deg, rgba(99,102,241,.20) 0%, rgba(14,165,233,.12) 55%, rgba(34,197,94,.10) 100%);
    border: 1px solid rgba(99,102,241,.35);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}}
.df-hero h1 {{
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 .25rem 0;
    color: {TEXT};
}}
.df-hero p {{
    margin: 0;
    color: {MUTED};
    font-size: .95rem;
}}

/* ---------- KPI cards ---------- */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: .9rem;
    margin: .4rem 0 1.1rem 0;
}}
.kpi {{
    background: {PANEL};
    border: 1px solid rgba(148,163,184,.14);
    border-left: 3px solid {ACCENT};
    border-radius: 12px;
    padding: .85rem 1rem;
}}
.kpi .kpi-label {{
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .09em;
    color: {MUTED};
    margin-bottom: .3rem;
}}
.kpi .kpi-value {{
    font-size: 1.45rem;
    font-weight: 700;
    color: {TEXT};
    line-height: 1.15;
}}
.kpi .kpi-sub {{
    font-size: .78rem;
    color: {MUTED};
    margin-top: .2rem;
}}
.kpi.kpi-success {{ border-left-color: {SUCCESS}; }}
.kpi.kpi-danger  {{ border-left-color: {DANGER}; }}
.kpi.kpi-warning {{ border-left-color: {WARNING}; }}

/* ---------- verdict banner ---------- */
.verdict {{
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: .3rem 0 1.1rem 0;
    border: 1px solid transparent;
}}
.verdict .verdict-title {{
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: .02em;
}}
.verdict .verdict-sub {{
    font-size: .88rem;
    opacity: .85;
    margin-top: .2rem;
}}
.verdict-danger  {{ background: rgba(239,68,68,.14);  border-color: rgba(239,68,68,.45);  color: #FCA5A5; }}
.verdict-success {{ background: rgba(34,197,94,.14);  border-color: rgba(34,197,94,.45);  color: #86EFAC; }}
.verdict-warning {{ background: rgba(245,158,11,.15); border-color: rgba(245,158,11,.45); color: #FCD34D; }}
.verdict-info    {{ background: rgba(99,102,241,.15); border-color: rgba(99,102,241,.45); color: #C7D2FE; }}

/* ---------- tables ---------- */
[data-testid="stDataFrame"] {{
    border: 1px solid rgba(148,163,184,.14);
    border-radius: 10px;
}}

/* ---------- tabs ---------- */
[data-baseweb="tab-list"] {{
    gap: .4rem;
}}
[data-baseweb="tab"] {{
    border-radius: 8px 8px 0 0;
    padding: .35rem .9rem;
}}

/* ---------- captions & footers ---------- */
small, .df-caption {{
    color: {MUTED};
}}
hr {{ border-color: rgba(148,163,184,.16); }}

/* ---------- hide the default empty padding ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 12px;
}}
</style>
"""


def inject_theme() -> None:
    """Write the stylesheet once per browser session."""
    if st.session_state.get("_df_theme_loaded"):
        return
    st.markdown(_CSS, unsafe_allow_html=True)
    st.session_state["_df_theme_loaded"] = True


def kpi(
    label: str,
    value: str,
    sub: str = "",
    tone: str = "",
) -> str:
    """Render one KPI card as HTML (returns the markup for composition)."""
    classes = "kpi" + (f" kpi-{tone}" if tone else "")
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="{classes}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{sub_html}"
        f"</div>"
    )


def kpi_row(cards: list[tuple], columns) -> None:
    """Render a list of KPI tuples across ``st.columns``.

    Each tuple is ``(label, value, sub, tone)``; ``tone`` may be empty,
    ``success``, ``danger`` or ``warning``.
    """
    for column, card in zip(columns, cards):
        label, value = card[0], card[1]
        sub = card[2] if len(card) > 2 else ""
        tone = card[3] if len(card) > 3 else ""
        column.markdown(kpi(label, value, sub, tone), unsafe_allow_html=True)


def verdict_banner(status: str, conclusion: str, risk: str) -> None:
    """Render the combined multimodal verdict."""
    tone = {
        "CONSISTENT FAKE": "danger",
        "CONSISTENT REAL": "success",
        "INCONCLUSIVE": "warning",
    }.get(status, "info")

    st.markdown(
        f'<div class="verdict verdict-{tone}">'
        f'<div class="verdict-title">{status} &nbsp;·&nbsp; risk {risk}</div>'
        f'<div class="verdict-sub">{conclusion}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
