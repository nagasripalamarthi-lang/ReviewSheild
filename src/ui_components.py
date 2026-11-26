"""Reusable Streamlit UI components, custom styling, KPI cards, and charts."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import LABEL_GENUINE, LABEL_SUSPICIOUS


def inject_custom_css() -> None:
    """Inject modern, responsive SaaS-inspired CSS styling into Streamlit."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1240px;
        }
        .rs-hero {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 65%, #2563EB 100%);
            color: #FFFFFF;
            padding: 1.4rem 1.75rem;
            border-radius: 16px;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 25px -10px rgba(30, 58, 138, 0.45);
        }
        .rs-hero h1 {
            margin: 0;
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #FFFFFF;
        }
        .rs-hero p {
            margin: 0.4rem 0 0 0;
            color: #DBEAFE;
            font-size: 0.98rem;
        }
        .rs-kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1rem 1.15rem;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
            margin-bottom: 0.5rem;
        }
        .rs-kpi-label {
            font-size: 0.80rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748B;
            margin-bottom: 0.25rem;
        }
        .rs-kpi-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.15;
        }
        .rs-kpi-sub {
            font-size: 0.80rem;
            color: #475569;
            margin-top: 0.25rem;
        }
        .rs-badge-genuine {
            display: inline-block;
            background: #DCFCE7;
            color: #166534;
            border: 1px solid #86EFAC;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.92rem;
        }
        .rs-badge-suspicious {
            display: inline-block;
            background: #FEE2E2;
            color: #991B1B;
            border: 1px solid #FCA5A5;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.92rem;
        }
        .rs-signal-card {
            border-radius: 10px;
            padding: 0.75rem 0.95rem;
            margin-bottom: 0.55rem;
            border-left: 4px solid #94A3B8;
            background: #F8FAFC;
        }
        .rs-signal-warning {
            border-left-color: #DC2626;
            background: #FEF2F2;
        }
        .rs-signal-positive {
            border-left-color: #16A34A;
            background: #F0FDF4;
        }
        .rs-signal-neutral {
            border-left-color: #2563EB;
            background: #EFF6FF;
        }
        .rs-disclaimer {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            border-left: 4px solid #D97706;
            color: #78350F;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            font-size: 0.88rem;
            margin: 0.75rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str) -> None:
    """Render top page hero banner."""
    st.markdown(
        f"""
        <div class="rs-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer_banner() -> None:
    """Display clear ethical & ML interpretation boundary notice."""
    st.markdown(
        """
        <div class="rs-disclaimer">
            <strong>Model Prediction Notice:</strong> ReviewShield outputs <em>statistical likelihoods</em>
            (<strong>Potentially Genuine</strong> vs. <strong>Potentially Suspicious</strong>) trained on a
            <strong>synthetic &amp; curated educational benchmark dataset</strong>. Predictions indicate linguistic
            and pattern-based risk signals — they are <strong>not verified proof</strong> of real-world fake reviews
            or real e-commerce fraud.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtext: str = "", accent: str = "#0F172A") -> None:
    """Render a clean KPI metric card."""
    st.markdown(
        f"""
        <div class="rs-kpi-card">
            <div class="rs-kpi-label">{label}</div>
            <div class="rs-kpi-value" style="color: {accent};">{value}</div>
            <div class="rs-kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_gauge(suspicious_prob: float, prediction: str) -> go.Figure:
    """Create a sleek Plotly gauge chart showing suspiciousness probability."""
    pct = round(suspicious_prob * 100.0, 1)
    bar_color = "#DC2626" if prediction == LABEL_SUSPICIOUS else "#16A34A"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 30, "color": "#0F172A"}},
            title={"text": "Suspicion Score", "font": {"size": 14, "color": "#475569"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#F1F5F9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#DCFCE7"},
                    {"range": [40, 60], "color": "#FEF3C7"},
                    {"range": [60, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#0F172A", "width": 2.5},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_feature_drivers_chart(
    suspicious_drivers: list[dict[str, Any]],
    genuine_drivers: list[dict[str, Any]],
) -> go.Figure | None:
    """Visualize top linear feature contributions pushing toward Genuine vs Suspicious."""
    rows = []
    for d in suspicious_drivers:
        rows.append({"Feature": d["feature"], "Impact": d["contribution"], "Direction": LABEL_SUSPICIOUS})
    for d in genuine_drivers:
        rows.append({"Feature": d["feature"], "Impact": d["contribution"], "Direction": LABEL_GENUINE})

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("Impact")
    fig = px.bar(
        df,
        x="Impact",
        y="Feature",
        color="Direction",
        orientation="h",
        color_discrete_map={
            LABEL_SUSPICIOUS: "#EF4444",
            LABEL_GENUINE: "#22C55E",
        },
        title="Model Feature Contributions (Log-Odds Impact)",
    )
    fig.update_layout(
        height=max(250, len(df) * 34 + 80),
        margin=dict(l=10, r=15, t=45, b=20),
        yaxis_title="",
        xaxis_title="← Pushes Genuine | Pushes Suspicious →",
        legend_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FAFC",
    )
    fig.add_vline(x=0, line_width=1.2, line_color="#475569")
    return fig


def render_distribution_donut(genuine_count: int, suspicious_count: int) -> go.Figure:
    """Create a Plotly donut chart comparing Potentially Genuine vs Potentially Suspicious."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[LABEL_GENUINE, LABEL_SUSPICIOUS],
                values=[genuine_count, suspicious_count],
                hole=0.58,
                marker=dict(colors=["#22C55E", "#EF4444"]),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value} reviews (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Classification Breakdown",
        height=340,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_confidence_histogram(df: pd.DataFrame) -> go.Figure:
    """Create a Plotly histogram of suspicion probabilities across analyzed reviews."""
    fig = px.histogram(
        df,
        x="suspicious_probability",
        color="prediction",
        nbins=20,
        range_x=[0, 1],
        color_discrete_map={
            LABEL_GENUINE: "#22C55E",
            LABEL_SUSPICIOUS: "#EF4444",
        },
        title="Suspicion Score Distribution",
        labels={"suspicious_probability": "Predicted Suspicion Probability", "prediction": "Prediction"},
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=45, b=20),
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FAFC",
    )
    fig.update_traces(opacity=0.78)
    fig.add_vline(x=0.5, line_dash="dash", line_color="#0F172A")
    return fig


def render_confusion_matrix_figure(cm: list[list[int]]) -> plt.Figure:
    """Render a clean Matplotlib confusion matrix for the ML Model Card page."""
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Genuine", "Suspicious"]
    ax.set(
        xticks=np.arange(arr.shape[1]),
        yticks=np.arange(arr.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Benchmark Label",
        xlabel="Model Prediction",
        title="Held-Out Test Confusion Matrix",
    )
    thresh = arr.max() / 2.0 if arr.size else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j,
                i,
                format(arr[i, j], "d"),
                ha="center",
                va="center",
                color="white" if arr[i, j] > thresh else "#0F172A",
                fontsize=12,
                fontweight="bold",
            )
    fig.tight_layout()
    return fig
