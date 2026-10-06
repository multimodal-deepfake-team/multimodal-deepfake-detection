"""Plotly figure builders.

Every helper returns a ``plotly.graph_objects.Figure`` pre-styled for the dark
dashboard theme, so the pages stay declarative.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

ACCENT = "#6366F1"
SUCCESS = "#22C55E"
DANGER = "#EF4444"
WARNING = "#F59E0B"
INFO = "#38BDF8"
MUTED = "#94A3B8"
GRID = "rgba(148,163,184,.16)"

_PALETTE = [ACCENT, INFO, SUCCESS, WARNING, DANGER, "#A78BFA"]


def _base(height: int | None = None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=13),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def _tone_for(fake_probability: float) -> str:
    if fake_probability >= 0.66:
        return DANGER
    if fake_probability >= 0.34:
        return WARNING
    return SUCCESS


# ============================================================
# SINGLE-VIDEO CHARTS
# ============================================================


def fake_probability_gauge(fake_probability: float, title: str = "Fake probability") -> go.Figure:
    """Semi-circular gauge for a 0..1 fake probability."""
    percent = round(fake_probability * 100, 2)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percent,
            number={"suffix": "%", "font": {"size": 34, "color": _tone_for(fake_probability)}},
            title={
                "text": title,
                "font": {"size": 14, "color": MUTED},
            },
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED, "tickwidth": 1},
                "bar": {"color": _tone_for(fake_probability), "thickness": 0.35},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 34], "color": "rgba(34,197,94,.18)"},
                    {"range": [34, 66], "color": "rgba(245,158,11,.18)"},
                    {"range": [66, 100], "color": "rgba(239,68,68,.18)"},
                ],
                "threshold": {
                    "line": {"color": "#E2E8F0", "width": 3},
                    "thickness": 0.85,
                    "value": percent,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED),
        height=280,
        margin=dict(l=30, r=30, t=60, b=10),
    )
    return fig


def real_fake_donut(real_probability: float, fake_probability: float, title: str) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=["Real", "Fake"],
            values=[real_probability * 100, fake_probability * 100],
            hole=0.62,
            marker=dict(colors=[SUCCESS, DANGER]),
            textinfo="label+percent",
            textfont=dict(size=13),
            hovertemplate="%{label}: %{value:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=13),
        title={"text": title, "font": {"size": 14, "color": MUTED}},
        showlegend=True,
        height=290,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", y=-0.05, x=0.35),
    )
    return fig


def frame_probabilities(frame_details: list[dict[str, Any]]) -> go.Figure:
    """Fake probability for every sampled frame, with the 0.5 decision line."""
    frames = [d["frame"] for d in frame_details]
    fake = [d["fake"] * 100 for d in frame_details]
    colors = [DANGER if value >= 50 else SUCCESS for value in fake]

    fig = _base(height=300)
    fig.add_trace(
        go.Bar(
            x=frames,
            y=fake,
            marker=dict(color=colors, line=dict(width=0)),
            name="Fake %",
            hovertemplate="Frame %{x}<br>Fake %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=50, line_dash="dash", line_color=MUTED, annotation_text="decision threshold")
    fig.update_layout(
        title={"text": "Per-frame fake probability", "font": {"size": 14, "color": MUTED}},
        showlegend=False,
        xaxis_title="Frame index",
        yaxis_title="Fake %",
        yaxis=dict(range=[0, 100]),
    )
    return fig


def audio_window_bars(real_probability: float, fake_probability: float) -> go.Figure:
    """Audio branch split as a horizontal real/fake bar."""
    fig = _base(height=150)
    fig.add_trace(
        go.Bar(
            y=["Audio"],
            x=[real_probability * 100],
            orientation="h",
            name="Real",
            marker=dict(color=SUCCESS),
            hovertemplate="Real %{x:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Audio"],
            x=[fake_probability * 100],
            orientation="h",
            name="Fake",
            marker=dict(color=DANGER),
            hovertemplate="Fake %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="stack",
        title={"text": "Audio branch (averaged over windows)", "font": {"size": 14, "color": MUTED}},
        xaxis=dict(range=[0, 100], title="Probability %"),
        height=190,
    )
    return fig


# ============================================================
# METRICS PAGE
# ============================================================


def confusion_matrix_heatmap(labels: list[str], matrix: list[list[int]]) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "#0B1120"],
                [0.25, "#1E3A8A"],
                [0.6, ACCENT],
                [1.0, "#F472B6"],
            ],
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
            colorbar=dict(title="count", thickness=14),
            showscale=True,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=13),
        title={"text": "Confusion matrix (rows = actual)", "font": {"size": 15, "color": MUTED}},
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=460,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(side="top", tickangle=-30)
    return fig


def per_class_bar(rows: list[dict[str, Any]], metric: str = "f1", title: str = "") -> go.Figure:
    """Grouped precision / recall / F1 bars for every class."""
    names = [r["class"] for r in rows]
    fig = _base(height=340)
    for key, color in (("precision", INFO), ("recall", ACCENT), ("f1", SUCCESS)):
        fig.add_trace(
            go.Bar(
                x=names,
                y=[r[key] for r in rows],
                name=key.capitalize(),
                marker=dict(color=color),
                text=[f"{r[key]:.2f}" for r in rows],
                textposition="outside",
                textfont=dict(size=10),
                hovertemplate=f"{key} %{{x}}: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        title={"text": title or "Per-class scores", "font": {"size": 15, "color": MUTED}},
        yaxis=dict(range=[0, 1.12], title="Score"),
        xaxis_title="Class",
        legend=dict(orientation="h", y=1.14, x=0),
    )
    return fig


def support_bar(rows: list[dict[str, Any]]) -> go.Figure:
    fig = _base(height=250)
    fig.add_trace(
        go.Bar(
            x=[r["class"] for r in rows],
            y=[r["support"] for r in rows],
            marker=dict(color=_PALETTE[0]),
            text=[str(r["support"]) for r in rows],
            textposition="outside",
        )
    )
    fig.update_layout(
        title={"text": "Samples per class", "font": {"size": 14, "color": MUTED}},
        yaxis_title="Support",
        showlegend=False,
        xaxis_title="Class",
    )
    return fig


# ============================================================
# BATCH PAGE
# ============================================================


def batch_status_breakdown(rows: list[dict[str, Any]]) -> go.Figure:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    tone = {
        "CONSISTENT FAKE": DANGER,
        "CONSISTENT REAL": SUCCESS,
        "INCONCLUSIVE": WARNING,
        "VIDEO-ONLY": INFO,
    }
    labels = list(counts)
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=[counts[k] for k in labels],
            hole=0.58,
            marker=dict(colors=[tone.get(l, ACCENT) for l in labels]),
            textinfo="label+value",
            hovertemplate="%{label}: %{value} video(s)<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=13),
        title={"text": "Verdict breakdown", "font": {"size": 15, "color": MUTED}},
        height=330,
        margin=dict(l=10, r=10, t=45, b=10),
        annotations=[
            dict(
                text=str(len(rows)),
                x=0.5,
                y=0.5,
                font=dict(size=26, color="#E2E8F0"),
                showarrow=False,
            )
        ],
    )
    return fig


def batch_scores(rows: list[dict[str, Any]]) -> go.Figure:
    """Horizontal fake-score bars, sorted worst-first."""
    ordered = sorted(rows, key=lambda r: r["video_fake_pct"])
    colors = [
        DANGER if r["video_fake_pct"] >= 66 else (WARNING if r["video_fake_pct"] >= 34 else SUCCESS)
        for r in ordered
    ]

    fig = _base(height=max(220, 34 * len(ordered)))
    fig.add_trace(
        go.Bar(
            x=[r["video_fake_pct"] for r in ordered],
            y=[r["file"] for r in ordered],
            orientation="h",
            marker=dict(color=colors),
            text=[f'{r["video_fake_pct"]:.1f}%' for r in ordered],
            textposition="outside",
            hovertemplate="%{y}<br>Fake %{x:.2f}%<extra></extra>",
        )
    )
    fig.add_vline(x=50, line_dash="dash", line_color=MUTED)
    fig.update_layout(
        title={"text": "Video fake score by file", "font": {"size": 15, "color": MUTED}},
        xaxis=dict(range=[0, 108], title="Fake %"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def comparison_grouped_bars(metrics: dict[str, dict[str, float]]) -> go.Figure:
    """Compare headline metrics across evaluation runs.

    ``metrics`` maps run name -> {"accuracy": x, "macro f1": y, "weighted f1": z}
    """
    keys = ["accuracy", "macro f1", "weighted f1"]
    fig = _base(height=320)
    for index, (run, values) in enumerate(metrics.items()):
        fig.add_trace(
            go.Bar(
                x=[k.title() for k in keys],
                y=[values.get(k) for k in keys],
                name=run,
                marker=dict(color=_PALETTE[index % len(_PALETTE)]),
                text=[f"{(values.get(k) or 0):.3f}" for k in keys],
                textposition="outside",
                textfont=dict(size=11),
            )
        )
    fig.update_layout(
        barmode="group",
        title={"text": "Headline metrics by evaluation run", "font": {"size": 15, "color": MUTED}},
        yaxis=dict(range=[0, 1.12], title="Score"),
        legend=dict(orientation="h", y=1.14, x=0),
    )
    return fig
