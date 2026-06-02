"""Visualisation utilities for LogisChain AI — Plotly + Matplotlib."""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_supply_chain_network(
    node_features: pd.DataFrame,
    title: str = "Supply Chain Network Risk Map",
    size_col: str = "weighted_out_degree",
    color_col: str = "betweenness_centrality",
) -> Optional[object]:
    if not PLOTLY_AVAILABLE or node_features.empty:
        return None
    fig = px.scatter(
        node_features,
        x="hub_score",
        y="authority_score",
        size=size_col if size_col in node_features.columns else None,
        color=color_col if color_col in node_features.columns else None,
        hover_data=["node"],
        title=title,
        color_continuous_scale="RdYlGn_r",
        labels={"hub_score": "Hub Score", "authority_score": "Authority Score"},
    )
    fig.update_layout(height=500, template="plotly_white")
    return fig


def plot_risk_score_distribution(
    risk_scores: pd.Series,
    title: str = "LogisChain Risk Score Distribution",
) -> Optional[object]:
    if not PLOTLY_AVAILABLE:
        return None
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Distribution", "Box Plot"])
    fig.add_trace(
        go.Histogram(x=risk_scores, nbinsx=50, name="Risk Scores",
                     marker_color="steelblue", opacity=0.7),
        row=1, col=1,
    )
    fig.add_trace(
        go.Box(y=risk_scores, name="Risk Score", marker_color="steelblue",
               boxmean="sd"),
        row=1, col=2,
    )
    # Add tier lines
    for threshold, label, color in [(0.25, "LOW/MED", "green"), (0.50, "MED/HIGH", "orange"), (0.75, "HIGH/CRIT", "red")]:
        fig.add_vline(x=threshold, line_dash="dash", line_color=color,
                      annotation_text=label, row=1, col=1)
    fig.update_layout(title=title, height=400, template="plotly_white")
    return fig


def plot_ccc_decomposition(
    financial_df: pd.DataFrame,
) -> Optional[object]:
    if not PLOTLY_AVAILABLE:
        return None
    cols = ["days_sales_outstanding", "days_inventory_outstanding", "days_payable_outstanding"]
    available = [c for c in cols if c in financial_df.columns]
    if not available:
        return None
    fig = go.Figure()
    labels = {"days_sales_outstanding": "DSO", "days_inventory_outstanding": "DIO",
               "days_payable_outstanding": "DPO (negative)"}
    colors = {"days_sales_outstanding": "#e74c3c", "days_inventory_outstanding": "#f39c12",
               "days_payable_outstanding": "#2ecc71"}
    for col in available:
        vals = financial_df[col].values
        if col == "days_payable_outstanding":
            vals = -vals
        fig.add_trace(go.Box(y=vals, name=labels[col], marker_color=colors.get(col, "blue")))
    fig.update_layout(
        title="Cash Conversion Cycle Components",
        yaxis_title="Days",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_forecast(
    historical: np.ndarray,
    forecast: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Demand Forecast",
) -> Optional[object]:
    if not PLOTLY_AVAILABLE:
        return None
    n_hist = len(historical)
    n_fore = len(forecast)
    if dates is not None:
        hist_x = dates[:n_hist]
        fore_x = pd.date_range(start=hist_x[-1], periods=n_fore + 1, freq="D")[1:]
    else:
        hist_x = list(range(n_hist))
        fore_x = list(range(n_hist, n_hist + n_fore))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_x, y=historical, name="Historical",
                             line=dict(color="steelblue", width=2)))
    fig.add_trace(go.Scatter(x=fore_x, y=forecast, name="Forecast",
                             line=dict(color="crimson", dash="dash", width=2)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Value",
                      template="plotly_white", height=400)
    return fig


def plot_scenario_impact(
    scenario_name: str,
    before_state: dict,
    after_state: dict,
) -> Optional[object]:
    if not PLOTLY_AVAILABLE:
        return None
    metrics = ["cash_usd", "trade_finance_exposure_usd", "inventory_value_usd",
               "accounts_receivable_usd", "credit_reserves_usd"]
    labels = ["Cash", "TF Exposure", "Inventory", "AR", "Credit Reserve"]
    before_vals = [before_state.get(m, 0) / 1e6 for m in metrics]
    after_vals = [after_state.get(m, 0) / 1e6 for m in metrics]
    fig = go.Figure(data=[
        go.Bar(name="Before Disruption", x=labels, y=before_vals, marker_color="#2ecc71"),
        go.Bar(name="After Disruption", x=labels, y=after_vals, marker_color="#e74c3c"),
    ])
    fig.update_layout(
        barmode="group",
        title=f"Portfolio Impact: {scenario_name}",
        yaxis_title="Value ($M)",
        template="plotly_white",
        height=450,
    )
    return fig


def plot_shap_waterfall(
    local_explanation: pd.DataFrame,
    entity_id: str = "Entity",
    base_value: float = 0.3,
) -> Optional[object]:
    if not PLOTLY_AVAILABLE or local_explanation.empty:
        return None
    top_n = local_explanation.head(10)
    features = top_n["feature"].tolist()
    shap_vals = top_n["shap_value"].tolist()
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_vals]
    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="h",
        measure=["relative"] * len(shap_vals) + ["total"],
        y=features + ["Final Score"],
        x=shap_vals + [sum(shap_vals)],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#e74c3c"}},
        decreasing={"marker": {"color": "#2ecc71"}},
    ))
    fig.update_layout(
        title=f"SHAP Explanation — {entity_id}",
        xaxis_title="SHAP Value (risk contribution)",
        template="plotly_white",
        height=450,
    )
    return fig


def plot_survival_curve(
    kmf_timeline: np.ndarray,
    kmf_survival: np.ndarray,
    kmf_ci_lower: Optional[np.ndarray] = None,
    kmf_ci_upper: Optional[np.ndarray] = None,
    title: str = "Carrier Survival Function",
) -> Optional[object]:
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=kmf_timeline, y=kmf_survival,
        mode="lines", name="Survival Probability",
        line=dict(color="steelblue", width=2),
    ))
    if kmf_ci_lower is not None and kmf_ci_upper is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([kmf_timeline, kmf_timeline[::-1]]),
            y=np.concatenate([kmf_ci_upper, kmf_ci_lower[::-1]]),
            fill="toself", fillcolor="rgba(70,130,180,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="Median Lifetime")
    fig.update_layout(
        title=title,
        xaxis_title="Time (Days)",
        yaxis_title="P(Survival)",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=400,
    )
    return fig


def plot_simulation_timeline(history_df: pd.DataFrame) -> Optional[object]:
    if not PLOTLY_AVAILABLE or history_df.empty:
        return None
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Cash Position ($M)", "Cash Conversion Cycle (Days)", "Cumulative Score"],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )
    periods = history_df["period"]
    fig.add_trace(go.Scatter(x=periods, y=history_df["cash_usd"] / 1e6,
                             mode="lines+markers", name="Cash ($M)",
                             line=dict(color="#2ecc71")), row=1, col=1)
    if "net_working_capital" in history_df.columns:
        fig.add_trace(go.Scatter(x=periods, y=history_df["net_working_capital"] / 1e6,
                                 mode="lines", name="NWC ($M)",
                                 line=dict(color="#3498db", dash="dot")), row=1, col=1)
    if "liquidity_ratio" in history_df.columns:
        pass  # add to cash panel if needed
    fig.add_trace(go.Bar(x=periods, y=history_df.get("ccc_change_days", [0]*len(periods)),
                         name="CCC Change", marker_color="#e74c3c"), row=2, col=1)
    fig.add_trace(go.Scatter(x=periods, y=history_df["cumulative_score"],
                             mode="lines+markers", name="Score",
                             line=dict(color="#9b59b6")), row=3, col=1)

    # Mark disruption periods
    for _, row in history_df.iterrows():
        if row["scenario"] != "None":
            fig.add_vline(x=row["period"], line_dash="dot", line_color="orange",
                          annotation_text=row["scenario"][:15])

    fig.update_layout(height=700, title="Simulation Timeline", template="plotly_white")
    return fig
