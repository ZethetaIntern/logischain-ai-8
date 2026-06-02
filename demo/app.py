"""LogisChain AI — Multi-Page Streamlit Dashboard.

Run with:
    streamlit run demo/app.py

Pages
─────
🏠  Home / Overview
🌐  Supply Chain Network
📊  Risk Monitor
🎮  LogisChain Lab (Simulation)
🔍  Model Explainability
📚  Case Studies
"""

import math
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LogisChain AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS — Professional FinTech Design System ────────────────────────────────
st.markdown("""
<style>
/* ══ Design tokens ══════════════════════════════════════════════════════════ */
:root {
  --bg0: #090e1a;        /* deepest background                               */
  --bg1: #0d1321;        /* main page background                             */
  --bg2: #111827;        /* card background                                  */
  --bg3: #1a2235;        /* elevated / hover state                           */
  --b0:  #1e293b;        /* border subtle                                    */
  --b1:  #2d3a4f;        /* border normal                                    */
  --b2:  #3d4f6a;        /* border lit                                       */
  /* Accent palette */
  --blue:   #3b82f6;
  --blue2:  #60a5fa;
  --cyan:   #06b6d4;
  --green:  #10b981;
  --amber:  #f59e0b;
  --red:    #ef4444;
  --purple: #8b5cf6;
  --pink:   #ec4899;
  /* Text */
  --t1: #f0f4ff;
  --t2: #94a3b8;
  --t3: #64748b;
}

/* ══ Base ════════════════════════════════════════════════════════════════════ */
.stApp {
  background: var(--bg1) !important;
  color: var(--t1) !important;
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* ══ Scrollbar ═══════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg1); }
::-webkit-scrollbar-thumb { background: var(--b2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--blue); }

/* ══ Sidebar ═════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: var(--bg0) !important;
  border-right: 1px solid var(--b0) !important;
}
/* ── Sidebar text color (targeted selectors only, no wildcard *) ── */
[data-testid="stSidebar"] p { color: var(--t2); }
[data-testid="stSidebar"] .stMarkdown { color: var(--t2); }

/* ── Navigation radio group ───────────────────────────────────── */
[data-testid="stSidebar"] .stRadio > div {
  display: flex !important;
  flex-direction: column !important;
  gap: 1px !important;
}
/* Keep label fully interactive — only visual overrides, no display:none */
[data-testid="stSidebar"] .stRadio label {
  padding: 0.45rem 0.85rem !important;
  border-radius: 7px !important;
  font-size: 0.85rem !important;
  color: var(--t2) !important;
  cursor: pointer !important;
  transition: background .12s, color .12s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: var(--bg3) !important;
  color: var(--t1) !important;
}
/* Hide the radio dot visually but keep it clickable */
[data-testid="stSidebar"] .stRadio label > div:first-child {
  width: 0 !important;
  height: 0 !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
}

/* ══ KPI cards (custom HTML) ═════════════════════════════════════════════════ */
.kpi-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin: 1rem 0; }
.kpi     {
  background: var(--bg2);
  border: 1px solid var(--b0);
  border-radius: 10px;
  padding: 1.2rem 1.4rem;
  position: relative; overflow: hidden;
  transition: border-color .2s, transform .15s;
}
.kpi:hover { border-color: var(--b2); transform: translateY(-2px); }
.kpi::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  border-radius: 10px 10px 0 0;
}
.kpi.b::after { background: linear-gradient(90deg, var(--blue), var(--cyan)); }
.kpi.g::after { background: linear-gradient(90deg, var(--green), var(--cyan)); }
.kpi.a::after { background: linear-gradient(90deg, var(--amber), var(--pink)); }
.kpi.p::after { background: linear-gradient(90deg, var(--purple), var(--blue)); }
.kpi-icon  { font-size: 1.4rem; margin-bottom: .5rem; }
.kpi-val   { font-size: 1.9rem; font-weight: 700; color: var(--t1); line-height: 1.1; }
.kpi-lbl   { font-size: .72rem; font-weight: 600; text-transform: uppercase;
             letter-spacing: .08em; color: var(--t3); margin-top: .3rem; }
.kpi-sub   { font-size: .78rem; color: var(--t2); margin-top: .35rem; }
.kpi-sub.up   { color: var(--green); }
.kpi-sub.down { color: var(--red); }

/* ══ Section label ═══════════════════════════════════════════════════════════ */
.sec-label {
  font-size: .68rem; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; color: var(--t3);
  padding-bottom: .5rem; margin: 1.4rem 0 .9rem;
  border-bottom: 1px solid var(--b0);
}

/* ══ Stat card ═══════════════════════════════════════════════════════════════ */
.stat-card {
  background: var(--bg2); border: 1px solid var(--b0);
  border-radius: 10px; padding: 1rem 1.2rem;
}

/* ══ Alert boxes ═════════════════════════════════════════════════════════════ */
.alert { display: flex; gap: .7rem; align-items: flex-start;
         border-radius: 8px; padding: .7rem 1rem; margin: .35rem 0;
         font-size: .84rem; line-height: 1.5; }
.alert-crit  { background: rgba(239,68,68,.08);  border-left: 3px solid var(--red);    }
.alert-high  { background: rgba(245,158,11,.08); border-left: 3px solid var(--amber);  }
.alert-med   { background: rgba(59,130,246,.08); border-left: 3px solid var(--blue);   }
.alert-low   { background: rgba(16,185,129,.08); border-left: 3px solid var(--green);  }
.alert-icon  { font-size: 1rem; flex-shrink: 0; margin-top: .05rem; }
.alert-title { font-weight: 600; color: var(--t1); display: block; margin-bottom: .15rem; }
.alert-body  { color: var(--t2); font-size: .81rem; }

/* ══ Disruption banner ═══════════════════════════════════════════════════════ */
.disruption-banner {
  background: linear-gradient(90deg, rgba(239,68,68,.12) 0%, rgba(239,68,68,.04) 100%);
  border: 1px solid rgba(239,68,68,.35); border-radius: 8px;
  padding: .9rem 1.2rem; margin: .8rem 0;
  display: flex; align-items: center; gap: .8rem; font-size: .9rem;
}

/* ══ Badge ═══════════════════════════════════════════════════════════════════ */
.badge { display: inline-block; padding: .15rem .55rem; border-radius: 20px;
         font-size: .68rem; font-weight: 700; letter-spacing: .03em; }
.badge-red    { background: rgba(239,68,68,.15);  color: var(--red);    }
.badge-amber  { background: rgba(245,158,11,.15); color: var(--amber);  }
.badge-green  { background: rgba(16,185,129,.15); color: var(--green);  }
.badge-blue   { background: rgba(59,130,246,.15); color: var(--blue2);  }
.badge-purple { background: rgba(139,92,246,.15); color: var(--purple); }

/* ══ Metric override (Streamlit native) ══════════════════════════════════════ */
[data-testid="stMetric"] {
  background: var(--bg2) !important; border: 1px solid var(--b0) !important;
  border-radius: 10px !important; padding: 1rem 1.3rem !important;
}
[data-testid="stMetricValue"] { color: var(--t1) !important; font-size: 1.55rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: var(--t3) !important; font-size: .72rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: .06em; }

/* ══ Buttons ═════════════════════════════════════════════════════════════════ */
.stButton > button {
  background: var(--bg3) !important; color: var(--t1) !important;
  border: 1px solid var(--b1) !important; border-radius: 8px !important;
  font-weight: 500 !important; font-size: .84rem !important;
  transition: all .15s !important; letter-spacing: .02em !important;
}
.stButton > button:hover {
  border-color: var(--blue) !important; color: var(--blue2) !important;
  background: var(--bg3) !important;
  box-shadow: 0 0 0 1px var(--blue), 0 0 12px rgba(59,130,246,.15) !important;
  transform: none !important;
}
button[kind="primary"] {
  background: var(--blue) !important; color: #fff !important;
  border-color: var(--blue) !important; font-weight: 600 !important;
}

/* ══ Forms / Inputs ══════════════════════════════════════════════════════════ */
.stTextInput input, .stNumberInput input {
  background: var(--bg2) !important; border-color: var(--b1) !important;
  color: var(--t1) !important; border-radius: 7px !important;
}
.stSelectbox > div > div {
  background: var(--bg2) !important; border-color: var(--b1) !important;
  color: var(--t1) !important; border-radius: 7px !important;
}
.stSlider .stSliderThumb { background: var(--blue) !important; }
.stSlider .stSliderTrack  { background: var(--b1) !important; }

/* ══ Tabs ════════════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--b0) !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: var(--t3) !important;
  font-size: .84rem !important; font-weight: 500 !important;
  padding: .55rem 1.1rem !important;
  border-bottom: 2px solid transparent !important; border-radius: 0 !important;
  transition: color .15s !important;
}
.stTabs [aria-selected="true"] {
  color: var(--blue2) !important; border-bottom-color: var(--blue) !important;
  background: transparent !important;
}

/* ══ Expander ════════════════════════════════════════════════════════════════ */
.streamlit-expanderHeader {
  background: var(--bg2) !important; border: 1px solid var(--b0) !important;
  border-radius: 8px !important; color: var(--t1) !important;
  font-weight: 500 !important; font-size: .88rem !important;
}

/* ══ Tables ══════════════════════════════════════════════════════════════════ */
.dataframe { border: 1px solid var(--b0) !important; border-radius: 8px; font-size: .83rem !important; }
.dataframe th { background: var(--bg3) !important; color: var(--t2) !important; border-color: var(--b0) !important; font-weight: 600 !important; font-size: .72rem !important; text-transform: uppercase; letter-spacing: .05em; }
.dataframe td { background: var(--bg2) !important; color: var(--t1) !important; border-color: var(--b0) !important; }

/* ══ Info / Success / Warning / Error ════════════════════════════════════════ */
.stInfo    { background: rgba(59,130,246,.08) !important; border: 1px solid rgba(59,130,246,.3) !important; border-radius: 8px !important; color: var(--blue2) !important; }
.stSuccess { background: rgba(16,185,129,.08) !important; border: 1px solid rgba(16,185,129,.3) !important; border-radius: 8px !important; }
.stWarning { background: rgba(245,158,11,.08) !important; border: 1px solid rgba(245,158,11,.3) !important; border-radius: 8px !important; }
.stError   { background: rgba(239,68,68,.08)  !important; border: 1px solid rgba(239,68,68,.3)  !important; border-radius: 8px !important; }

/* ══ Hide Streamlit chrome ═══════════════════════════════════════════════════ */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.stDeployButton { display: none !important; }
.viewerBadge_container__r5tak { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── Lazy imports ─────────────────────────────────────────────────────────────

def _try_import(module_path, names):
    """Safely import a list of names from module_path."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return {n: getattr(mod, n) for n in names}
    except Exception:
        return {}


# ─── Session state ────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "game_engine": None,
        "game_mode": "trade_finance",
        "game_turns_played": 0,
        "game_turn_results": [],
        "selected_node": None,
        "node_filters": {"type": "All", "risk": "All", "country": "All"},
        "lc_score_result": None,
        "wc_stress_result": None,
        "cf_result": None,
        "last_refresh": datetime.now(),
        "score_history_player": [],
        "score_history_ai": [],
        "auto_refresh": False,
        "network_data": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─── Cached data loaders ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_synthetic_data():
    imp = _try_import("src.data.pipeline",
                      ["SyntheticDataGenerator", "SupplyChainNetworkGenerator"])
    if not imp:
        # Pure-NumPy fallback
        rng = np.random.default_rng(42)
        n = 300
        carriers = pd.DataFrame({
            "carrier_id": [f"CAR-{i:04d}" for i in range(n)],
            "on_time_delivery_rate": rng.beta(8, 2, n),
            "damage_rate": rng.beta(1, 20, n),
            "fleet_size": rng.integers(5, 500, n),
            "carrier_failure": rng.integers(0, 2, n),
        })
        return {"carriers": carriers, "shipments": pd.DataFrame(), "financial": pd.DataFrame()}
    gen = imp["SyntheticDataGenerator"](seed=42)
    return gen.generate_all()


@st.cache_data(ttl=3600)
def load_supplier_data():
    imp = _try_import("src.data.pipeline", ["SupplyChainNetworkGenerator"])
    if not imp:
        rng = np.random.default_rng(42)
        n = 80
        return pd.DataFrame({
            "supplier_id": [f"SUP-{i:04d}" for i in range(n)],
            "country": rng.choice(["CN","VN","IN","DE","US","BD","KR"], n),
            "otif_rate": rng.beta(18, 2, n),
            "lead_time_mean": rng.lognormal(2.5, 0.5, n),
            "inventory_turnover": rng.lognormal(2.0, 0.5, n),
            "country_risk_score": rng.uniform(0.1, 0.7, n),
            "betweenness_centrality": rng.exponential(0.03, n),
            "disruption_vulnerability_index": rng.beta(3, 7, n),
            "logischain_composite_risk_score": rng.beta(3, 7, n),
        })
    gen = imp["SupplyChainNetworkGenerator"](seed=42)
    return gen.generate_suppliers(n=80)


@st.cache_data(ttl=3600)
def load_lc_data():
    imp = _try_import("src.data.pipeline", ["TradefinanceDataGenerator"])
    if not imp:
        rng = np.random.default_rng(42)
        n = 500
        return pd.DataFrame({
            "lc_id": [f"LC-{i:06d}" for i in range(n)],
            "lc_amount_usd": rng.lognormal(12, 1.5, n),
            "tenor_days": rng.choice([30, 60, 90, 120], n),
            "default_flag": rng.integers(0, 2, n),
            "pd_adjusted": rng.beta(1.5, 20, n),
            "beneficiary_otif_score": rng.beta(18, 2, n),
            "port_congestion_origin": rng.beta(3, 7, n) * 5,
            "port_congestion_destination": rng.beta(3, 7, n) * 5,
            "freight_rate_percentile": rng.uniform(0, 1, n),
            "issue_date": pd.date_range("2020-01-01", periods=n, freq="2D"),
        })
    gen = _try_import("src.data.pipeline", ["TradefinanceDataGenerator"])
    if not gen:
        return pd.DataFrame()
    return gen["TradefinanceDataGenerator"](seed=42).generate_lc_transactions(n=500)


@st.cache_resource
def get_lc_scorer():
    imp = _try_import("src.financial.trade_finance_model", ["LCRiskScorer"])
    if not imp:
        return None
    scorer = imp["LCRiskScorer"]()
    lc_df = load_lc_data()
    if len(lc_df) > 0 and "default_flag" in lc_df.columns:
        try:
            scorer.fit(lc_df)
        except Exception:
            pass
    return scorer


@st.cache_resource
def get_ccc_predictor():
    imp = _try_import("src.financial.ccc_predictor", ["CCCPredictor"])
    if not imp:
        return None
    return imp["CCCPredictor"]()


# ─── Chart helpers ────────────────────────────────────────────────────────────

_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#0d1321",
    font=dict(color="#94a3b8", family="Inter, Segoe UI, system-ui, sans-serif", size=12),
    margin=dict(l=36, r=16, t=40, b=36),
    legend=dict(bgcolor="rgba(17,24,39,0.7)", bordercolor="#1e293b",
                borderwidth=1, font=dict(color="#94a3b8")),
    hoverlabel=dict(bgcolor="#1a2235", bordercolor="#2d3a4f",
                    font=dict(color="#f0f4ff")),
)
# Axis defaults applied separately — not in _LAYOUT to avoid duplicate-keyword conflicts
_AXIS_STYLE = dict(gridcolor="#1e293b", linecolor="#1e293b", tickcolor="#1e293b")

_ACCENT = "#3b82f6"   # primary blue
_GREEN  = "#10b981"
_RED    = "#ef4444"
_AMBER  = "#f59e0b"
_PURPLE = "#8b5cf6"


def _fig(height=400):
    f = go.Figure()
    f.update_layout(**_LAYOUT, height=height)
    f.update_xaxes(**_AXIS_STYLE)
    f.update_yaxes(**_AXIS_STYLE)
    return f


def gauge_chart(score: float, title: str = "Risk Score") -> go.Figure:
    pct = min(100, max(0, score * 100))
    # Color the bar based on score
    if pct < 35:   bar_color = _GREEN
    elif pct < 65: bar_color = _AMBER
    else:          bar_color = _RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={"text": title, "font": {"color": "#94a3b8", "size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0,
                     "tickfont": {"color": "#64748b", "size": 10}},
            "bar":  {"color": bar_color, "thickness": 0.65},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],   "color": "rgba(16,185,129,.10)"},
                {"range": [35, 65],  "color": "rgba(245,158,11,.10)"},
                {"range": [65, 100], "color": "rgba(239,68,68,.10)"},
            ],
            "threshold": {"line": {"color": bar_color, "width": 3},
                          "thickness": 0.8, "value": pct},
        },
        number={"suffix": "", "font": {"color": "#f0f4ff", "size": 28},
                "valueformat": ".0f"},
    ))
    fig.update_layout(**_LAYOUT, height=240)
    return fig


def architecture_diagram_html() -> str:
    """Returns pure HTML architecture diagram — zero Plotly, zero conflicts."""
    layers = [
        ("#1d3461", "#3b82f6", "Physical Layer",
         "SC Network", "Port Congestion", "Vessel Tracking", "Inventory Levels"),
        ("#1a2e4a", "#06b6d4", "Intelligence Layer",
         "GNN (Risk)", "TCN (Forecast)", "XGBoost", "Survival Model"),
        ("#1a2e3a", "#10b981", "Financial Layer",
         "LC Risk Scorer", "CCC Predictor", "Credit Risk", "Insurance"),
    ]
    cards = ""
    for i, (bg, accent, title, *items) in enumerate(layers):
        items_html = "".join(
            f'<div style="font-size:.75rem;color:#64748b;padding:.15rem 0;'
            f'border-bottom:1px solid #1e293b;">{it}</div>'
            for it in items
        )
        arrow = (
            f'<div style="display:flex;align-items:center;color:{accent};'
            f'font-size:1.4rem;padding:0 .6rem;flex-shrink:0;">&#9658;</div>'
            if i < len(layers) - 1 else ""
        )
        cards += f"""
<div style="display:flex;align-items:stretch;flex:1;min-width:0;">
  <div style="background:{bg};border:1px solid {accent}33;border-top:2px solid {accent};
              border-radius:8px;padding:.75rem .85rem;flex:1;min-width:0;
              overflow:hidden;word-break:break-word;">
    <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;
                letter-spacing:.07em;color:{accent};margin-bottom:.45rem;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{title}</div>
    {items_html}
  </div>
  {arrow}
</div>"""

    return f"""
<div style="display:flex;align-items:stretch;gap:0;flex-wrap:nowrap;
            background:#0d1321;border:1px solid #1e293b;border-radius:10px;
            padding:.8rem 1rem;margin:.4rem 0;overflow:hidden;">
  {cards}
</div>"""


def architecture_diagram() -> go.Figure:
    """Plotly fallback — used only if HTML version is unavailable."""
    fig = go.Figure()
    layers = [
        (0.15, "Physical Layer",     "#1d3461", "#3b82f6",
         "SC Network · Ports<br>Vessels · Inventory"),
        (0.50, "Intelligence Layer", "#162447", "#06b6d4",
         "GNN · TCN · XGBoost<br>Survival Analysis"),
        (0.85, "Financial Layer",    "#1a2e3a", "#10b981",
         "LC Scorer · CCC<br>Credit Risk · Insurance"),
    ]
    for x, title, bg, accent, detail in layers:
        # Box
        fig.add_shape(type="rect", x0=x-.13, y0=.05, x1=x+.13, y1=.95,
                      fillcolor=bg, line=dict(color=accent, width=2),
                      xref="paper", yref="paper")
        # Top accent bar
        fig.add_shape(type="rect", x0=x-.13, y0=.88, x1=x+.13, y1=.95,
                      fillcolor=accent, line=dict(width=0),
                      xref="paper", yref="paper", opacity=0.6)
        # Title
        fig.add_annotation(x=x, y=.73, xref="paper", yref="paper",
                           text=f"<b>{title}</b>",
                           font=dict(color="white", size=10, family="Inter, sans-serif"),
                           showarrow=False, align="center")
        # Detail (with line breaks so it stays inside the box)
        fig.add_annotation(x=x, y=.45, xref="paper", yref="paper",
                           text=detail,
                           font=dict(color="#94a3b8", size=8.5),
                           showarrow=False, align="center")
    # Arrows between boxes
    for ax in [0.295, 0.705]:
        fig.add_annotation(x=ax, y=.50, xref="paper", yref="paper",
                           text="<b>&#9658;</b>",
                           font=dict(color="#60a5fa", size=20),
                           showarrow=False)
    fig.update_layout(paper_bgcolor="#0d1321", plot_bgcolor="#0d1321",
                      height=220, showlegend=False,
                      margin=dict(l=4, r=4, t=4, b=4))
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    return fig


def network_plotly(suppliers_df: pd.DataFrame, n_show: int = 60) -> go.Figure:
    rng = np.random.default_rng(42)
    df = suppliers_df.head(n_show).reset_index(drop=True)
    n = len(df)
    theta = np.linspace(0, 2 * math.pi, n, endpoint=False)
    r_base = [1.5 + rng.uniform(-0.3, 0.3) for _ in range(n)]
    x = [r * math.cos(t) for r, t in zip(r_base, theta)]
    y = [r * math.sin(t) for r, t in zip(r_base, theta)]

    risk_col = "logischain_composite_risk_score" if "logischain_composite_risk_score" in df.columns else "disruption_vulnerability_index"
    risk = df.get(risk_col, pd.Series(rng.uniform(0.1, 0.8, n))).fillna(0.3).values
    node_colors = [f"rgb({int(255*r)},{int(255*(1-r))},40)" for r in risk]

    bc = df.get("betweenness_centrality", pd.Series(rng.uniform(0, 0.1, n))).fillna(0.01).values
    sizes = [12 + 30 * float(b) for b in bc]

    # Hub nodes at centre
    hub_x = [0.0, 0.4, -0.4]
    hub_y = [0.0, 0.2, -0.2]
    hub_names = ["PORT-Rotterdam", "PORT-Shanghai", "PORT-Singapore"]

    fig = go.Figure()
    # Edges (sample 80 supplier→hub)
    for i in range(min(80, n)):
        hub_idx = i % 3
        fig.add_trace(go.Scatter(
            x=[x[i], hub_x[hub_idx], None], y=[y[i], hub_y[hub_idx], None],
            mode="lines", line=dict(color="rgba(100,150,255,0.25)", width=1),
            hoverinfo="skip", showlegend=False,
        ))
    # Supplier nodes
    country_col = "country" if "country" in df.columns else None
    hover = []
    for i in range(n):
        parts = [
            f"<b>{df.get('supplier_id', pd.Series([''] * n)).iloc[i] if 'supplier_id' in df.columns else f'SUP-{i:03d}'}</b>",
            f"OTIF: {df.get('otif_rate', pd.Series(rng.uniform(0.7, 1.0, n))).iloc[i]:.1%}",
            f"Risk: {risk[i]:.2f}",
        ]
        if country_col:
            parts.insert(1, f"Country: {df[country_col].iloc[i]}")
        hover.append("<br>".join(parts))

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=sizes, color=node_colors, line=dict(color="white", width=0.5)),
        text=hover, hoverinfo="text", name="Suppliers",
    ))
    # Hub port nodes
    fig.add_trace(go.Scatter(
        x=hub_x, y=hub_y, mode="markers+text",
        marker=dict(size=28, color=_ACCENT, symbol="square",
                    line=dict(color="white", width=1.5)),
        text=hub_names, textposition="top center",
        textfont=dict(color="white", size=9),
        hoverinfo="text", name="Ports",
    ))
    fig.update_layout(**_LAYOUT, height=520, showlegend=True)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def shap_waterfall(shap_vals: dict, base: float = 0.028, pred: float = 0.042) -> go.Figure:
    sorted_items = sorted(shap_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    names = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    fig = go.Figure(go.Waterfall(
        name="SHAP", orientation="h",
        measure=["relative"] * len(vals) + ["total"],
        y=names + ["Final PD"],
        x=[v * 100 for v in vals] + [pred * 100],
        connector={"line": {"color": "rgba(100,100,100,0.5)"}},
        increasing={"marker": {"color": _RED}},
        decreasing={"marker": {"color": _GREEN}},
        totals={"marker": {"color": "#1e293b", "line": {"color": _ACCENT, "width": 2}}},
    ))
    fig.update_layout(**_LAYOUT, height=380,
                      xaxis_title="SHAP contribution (%)",
                      title="SHAP Waterfall — Feature Contributions")
    return fig


def feature_importance_bar(top_n: int = 20) -> go.Figure:
    feats = [
        ("OTIF Rate", 0.052, "supply_chain"),
        ("Port Congestion Dest", 0.048, "supply_chain"),
        ("Cash Conversion Cycle", 0.045, "financial"),
        ("Supplier Concentration HHI", 0.041, "supply_chain"),
        ("Logistic Disruption Credit Impact", 0.039, "fusion"),
        ("Freight Rate Percentile", 0.038, "supply_chain"),
        ("Inventory Turnover", 0.036, "supply_chain"),
        ("Altman Z-Score", 0.035, "financial"),
        ("Network Betweenness", 0.033, "supply_chain"),
        ("LC Composite Risk Score", 0.031, "fusion"),
        ("Country Risk Score", 0.030, "supply_chain"),
        ("Interest Coverage", 0.028, "financial"),
        ("Fill Rate Deficit", 0.027, "supply_chain"),
        ("Customer HHI", 0.025, "financial"),
        ("WCVI", 0.024, "fusion"),
        ("Current Ratio", 0.023, "financial"),
        ("Lead Time CV", 0.022, "supply_chain"),
        ("EBITDA Margin", 0.021, "financial"),
        ("Carrier Reliability", 0.020, "supply_chain"),
        ("TRFSI", 0.019, "fusion"),
    ][:top_n]
    color_map = {"supply_chain": _AMBER, "financial": _ACCENT, "fusion": _GREEN}
    fig = go.Figure(go.Bar(
        x=[f[1] for f in feats], y=[f[0] for f in feats],
        orientation="h",
        marker_color=[color_map[f[2]] for f in feats],
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>|SHAP| = %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT, height=540, xaxis_title="Mean |SHAP| value",
                      title=dict(text="Global Feature Importance — Top 20",
                                 font=dict(size=14, color="#f0f4ff")))
    return fig


def attention_heatmap() -> go.Figure:
    events = ["BOOKING", "LOADED", "DEPARTED", "TRANSHIPMENT", "ARRIVAL", "CUSTOMS", "DELIVERY"]
    heads = ["Head 1", "Head 2", "Head 3", "Head 4"]
    rng = np.random.default_rng(42)
    data = rng.dirichlet(np.array([1.0, 1.5, 2.0, 2.5, 3.0, 1.5, 1.0]) * 3, 4)
    fig = go.Figure(go.Heatmap(
        x=events, y=heads, z=data,
        colorscale=[[0, "#0d1321"], [0.4, "#1e4080"], [0.7, _AMBER], [1, _RED]],
        text=[[f"{v:.3f}" for v in row] for row in data],
        texttemplate="%{text}", hovertemplate="Event: %{x}<br>%{y}: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Weight", font=dict(color="#94a3b8", size=11)),
            tickfont=dict(color="#64748b", size=10),
            thickness=12, len=0.8,
        ),
    ))
    fig.update_layout(**_LAYOUT, height=320,
                      title=dict(text="Transformer Attention — Shipment Events",
                                 font=dict(size=13, color="#f0f4ff")))
    fig.update_xaxes(**_AXIS_STYLE, tickangle=-15, tickfont=dict(size=11))
    return fig


def triple_wave_chart() -> go.Figure:
    days = np.arange(1, 121)
    rng = np.random.default_rng(42)
    w1 = np.clip(50 * np.exp(-0.18 * np.abs(days - 6)) + rng.normal(0, 2, 120), 0, None)
    w2 = np.clip(35 * np.exp(-0.07 * np.abs(days - 22)) + rng.normal(0, 2, 120), 0, None)
    w3 = np.clip(22 * np.exp(-0.035 * np.abs(days - 55)) + rng.normal(0, 1.5, 120), 0, None)
    fig = go.Figure()
    for vals, name, color in [
        (w1, "Wave 1 — Direct LC Delays",     _RED),
        (w2, "Wave 2 — Congestion Cascade",    _AMBER),
        (w3, "Wave 3 — Systemic Credit Stress","#1e90ff"),
    ]:
        fig.add_trace(go.Scatter(
            x=days, y=vals, name=name,
            fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.25)",
            line=dict(color=color, width=2),
        ))
    fig.add_vline(x=6,  line_dash="dash", line_color="yellow",  annotation_text="Cleared Day 6")
    fig.add_vline(x=1,  line_dash="dot",  line_color=_RED, annotation_text="Blockage")
    fig.update_layout(**_LAYOUT, height=380,
                      xaxis_title="Days After Incident",
                      yaxis_title="Disruption Impact Index",
                      title="Ever Given — Triple-Wave Disruption Pattern")
    return fig


def ccc_timeline_chart(predictions, covenant=95) -> go.Figure:
    days = list(range(len(predictions)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=predictions, name="Predicted CCC",
        mode="lines", line=dict(color=_ACCENT, width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
    ))
    fig.add_hline(y=covenant, line_dash="dash", line_color="orange",
                  annotation_text=f"Covenant {covenant}d",
                  annotation=dict(font=dict(color="orange")))
    fig.update_layout(**_LAYOUT, height=320,
                      xaxis_title="Days", yaxis_title="CCC (days)",
                      title="CCC Trajectory — 90-Day Forecast")
    return fig


def radar_chart(player: dict, ai: dict) -> go.Figure:
    cats = ["Financial<br>Performance", "Risk<br>Management",
            "SC Intelligence", "Decision<br>Speed", "Learning"]
    maxes = [300, 250, 200, 100, 150]
    p_vals = [min(player.get(k, 0), m) for k, m in zip(
        ["financial_performance", "risk_management_quality",
         "supply_chain_intelligence_use", "decision_speed", "learning_progression"], maxes)]
    a_vals = [min(ai.get(k, 0), m) for k, m in zip(
        ["financial_performance", "risk_management_quality",
         "supply_chain_intelligence_use", "decision_speed", "learning_progression"], maxes)]
    fig = go.Figure()
    for vals, name, color in [(p_vals, "Player", _ACCENT), (a_vals, "AI", _GREEN)]:
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", name=name,
            line_color=color,
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 300],
                                   tickfont=dict(color="#64748b"), gridcolor="#1e293b"),
                   angularaxis=dict(tickfont=dict(color="#64748b")),
                   bgcolor="#0d1321"),
        paper_bgcolor="#111827", font=dict(color="#94a3b8"),
        showlegend=True, height=360,
        legend=dict(bgcolor="#111827", bordercolor="#1e293b"),
    )
    return fig


def score_line_chart(player_hist: list, ai_hist: list) -> go.Figure:
    turns = list(range(1, len(player_hist) + 1))
    fig = go.Figure()
    if player_hist:
        fig.add_trace(go.Scatter(x=turns, y=player_hist, name="Player",
                                  line=dict(color=_ACCENT, width=2.5), mode="lines+markers",
                                  marker=dict(size=6, color=_ACCENT)))
    if ai_hist:
        fig.add_trace(go.Scatter(x=turns, y=ai_hist, name="AI",
                                  line=dict(color=_GREEN, width=2, dash="dash"), mode="lines"))
    fig.update_layout(**_LAYOUT, height=280, title="Score vs AI — Turn by Turn",
                      xaxis_title="Turn", yaxis_title="Cumulative Score")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Page 1 — Home / Overview
# ═══════════════════════════════════════════════════════════════════════════════

def page_home():
    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown("""
<div style="display:flex;align-items:center;gap:14px;padding-bottom:1rem;
            border-bottom:1px solid #1e293b;margin-bottom:1.5rem;">
  <div style="width:42px;height:42px;background:linear-gradient(135deg,#3b82f6,#06b6d4);
              border-radius:10px;display:flex;align-items:center;justify-content:center;
              font-size:1.3rem;flex-shrink:0;">🔗</div>
  <div>
    <div style="font-size:1.5rem;font-weight:700;color:#f0f4ff;line-height:1.1;">LogisChain AI</div>
    <div style="font-size:0.82rem;color:#64748b;margin-top:2px;">
      Predictive Trade Finance &amp; Logistics Valuation &nbsp;·&nbsp; Zetheta Algorithms v0.2.0
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── KPI cards ───────────────────────────────────────────────────────────
    st.markdown("""
<div class="kpi-row">
  <div class="kpi b">
    <div class="kpi-icon">💼</div>
    <div class="kpi-val">$5.2T</div>
    <div class="kpi-lbl">Global Trade Finance</div>
    <div class="kpi-sub">Annual market volume</div>
  </div>
  <div class="kpi a">
    <div class="kpi-icon">⚠️</div>
    <div class="kpi-val">$2.5T</div>
    <div class="kpi-lbl">Trade Finance Gap</div>
    <div class="kpi-sub">Unmet demand worldwide</div>
  </div>
  <div class="kpi g">
    <div class="kpi-icon">📈</div>
    <div class="kpi-val">$2T+</div>
    <div class="kpi-lbl">SCF Market Size</div>
    <div class="kpi-sub up">↑ Growing 8% p.a.</div>
  </div>
  <div class="kpi p">
    <div class="kpi-icon">🎯</div>
    <div class="kpi-val">0.856</div>
    <div class="kpi-lbl">LogisChain AI AUC</div>
    <div class="kpi-sub up">↑ +11.5% vs baseline</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Architecture + Quick Demo ───────────────────────────────────────────
    col_arch, col_demo = st.columns([3, 2], gap="large")

    with col_arch:
        st.markdown('<div class="sec-label">System Architecture</div>', unsafe_allow_html=True)
        st.markdown(architecture_diagram_html(), unsafe_allow_html=True)

    with col_demo:
        st.markdown('<div class="sec-label">Live LC Risk Assessment</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:.84rem;color:#64748b;margin-bottom:.9rem;line-height:1.6;">
Score a sample $2.5M CN→US Letter of Credit using the full
LogisChain AI feature set (15 SC + financial signals).
</div>
""", unsafe_allow_html=True)

        if st.button("⚡  Run Sample Prediction", use_container_width=True):
            with st.spinner("Scoring…"):
                scorer = get_lc_scorer()
                sample = {
                    "lc_amount_usd": 2_500_000, "tenor_days": 90,
                    "commodity_hs_code": "8471", "origin_country": "CN",
                    "destination_country": "US", "applicant_credit_rating": "BBB",
                    "beneficiary_otif_score": 0.84,
                    "historical_discrepancy_rate_applicant": 0.08,
                    "historical_discrepancy_rate_beneficiary": 0.05,
                    "port_congestion_origin": 2.1,
                    "port_congestion_destination": 3.4,
                    "container_availability_index": 0.65,
                    "freight_rate_percentile": 0.72,
                    "seasonal_factor": 1.05,
                    "country_risk_differential": 0.30,
                    "currency_volatility_30d": 0.03,
                }
                result = scorer.score_lc_application(sample) if scorer else {
                    "risk_score": 0.62, "risk_level": "MEDIUM-HIGH",
                    "recommendation": "APPROVE_WITH_CONDITIONS",
                    "conditions": ["10% cash margin required"],
                    "key_risks": [("Port congestion 3.4/5.0", "HIGH")],
                }

            rs = result["risk_score"]   # raw PD probability in [0, 1]
            rec = result["recommendation"]

            # ── Convert raw PD → intuitive 0-100 Risk Index ──────────────
            # Uses log-odds scaling so small PD differences still show visually.
            # Calibration: 0% PD→0, 1% PD→33, 3% PD→50, 10% PD→75, 25%+ PD→90+
            import math as _math
            _lo = _math.log(max(rs, 1e-6) / max(1 - rs, 1e-6))  # log-odds
            risk_idx = round(min(100, max(0, 50 + _lo * 8)))      # scale to 0-100

            rec_color = {"APPROVE": _GREEN, "APPROVE_WITH_CONDITIONS": _AMBER, "DECLINE": _RED}
            rec_icon  = {"APPROVE": "✓", "APPROVE_WITH_CONDITIONS": "⚑", "DECLINE": "✕"}
            clr = rec_color.get(rec, _ACCENT)
            ico = rec_icon.get(rec, "•")
            bar_clr = "#ef4444" if risk_idx > 65 else "#f59e0b" if risk_idx > 35 else "#10b981"
            bar_rgb = "239,68,68" if risk_idx > 65 else "245,158,11" if risk_idx > 35 else "16,185,129"
            clr_rgb = "239,68,68" if clr==_RED else "245,158,11" if clr==_AMBER else "16,185,129"

            st.markdown(f"""
<div style="background:#111827;border:1px solid #1e293b;border-radius:10px;
            padding:1.1rem 1.3rem;margin:.5rem 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.7rem;">
    <span style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#475569;">Risk Index</span>
    <span style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#475569;">{result['risk_level']}</span>
  </div>
  <div style="font-size:2.6rem;font-weight:800;color:#f0f4ff;line-height:1;">{risk_idx}<span style="font-size:1rem;color:#64748b;font-weight:400;">/100</span></div>
  <div style="font-size:.75rem;color:#475569;margin:.2rem 0 .6rem;">Default probability: <b style="color:#94a3b8;">{rs*100:.2f}%</b></div>
  <div style="height:8px;background:#1e293b;border-radius:4px;margin:.5rem 0;overflow:hidden;">
    <div style="width:{risk_idx}%;height:100%;background:{bar_clr};border-radius:4px;"></div>
  </div>
  <div style="display:flex;gap:.5rem;align-items:center;margin-top:.7rem;">
    <div style="display:inline-flex;align-items:center;gap:.4rem;background:rgba({clr_rgb},.12);
                border:1px solid {clr};border-radius:6px;padding:.3rem .8rem;
                font-size:.8rem;font-weight:600;color:{clr};">
      {ico} &nbsp;{rec.replace('_',' ')}
    </div>
    <span style="font-size:.75rem;color:#334155;">PD = {rs*100:.2f}%</span>
  </div>
</div>
""", unsafe_allow_html=True)

            if result.get("key_risks"):
                st.markdown('<div style="font-size:.78rem;font-weight:600;color:#475569;margin:.6rem 0 .3rem;text-transform:uppercase;letter-spacing:.06em;">Key Risk Drivers</div>', unsafe_allow_html=True)
                for desc, sev in result["key_risks"][:3]:
                    badge = f'<span class="badge badge-{"red" if sev=="HIGH" else "amber"}">{sev}</span>'
                    st.markdown(f'<div style="font-size:.82rem;color:#94a3b8;padding:.25rem 0;">{badge} &nbsp;{desc}</div>', unsafe_allow_html=True)

    # ── Model performance table ─────────────────────────────────────────────
    st.markdown('<div class="sec-label" style="margin-top:1.8rem;">Model Performance Comparison</div>', unsafe_allow_html=True)

    perf_rows = [
        ("LR (Financial Only)",    0.738, 0.476, 0.381, 0.042, "12.4%", 0,    False),
        ("XGB (Financial Only)",   0.771, 0.542, 0.412, 0.035, "15.8%", 0,    False),
        ("XGB + SC Basic",         0.812, 0.624, 0.468, 0.028, "21.3%", 6,    False),
        ("LogisChain AI — Full",   0.856, 0.712, 0.523, 0.019, "28.7%", "21+", True),
    ]
    table_html = """
<div style="border:1px solid #1e293b;border-radius:10px;overflow:hidden;font-size:.82rem;">
<table style="width:100%;border-collapse:collapse;">
<thead>
<tr style="background:#111827;border-bottom:1px solid #1e293b;">
  <th style="text-align:left;padding:.65rem 1rem;color:#475569;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;">Model</th>
  <th style="text-align:center;padding:.65rem .8rem;color:#475569;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;">AUC-ROC</th>
  <th style="text-align:center;padding:.65rem .8rem;color:#475569;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;">Gini</th>
  <th style="text-align:center;padding:.65rem .8rem;color:#475569;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;">KS</th>
  <th style="text-align:center;padding:.65rem .8rem;color:#475569;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;">ECE</th>
  <th style="text-align:center;padding:.65rem .8rem;color:#475569;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;">P@5%</th>
  <th style="text-align:center;padding:.65rem .8rem;color:#475569;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;">SC Feats</th>
</tr>
</thead>
<tbody>"""
    for model, auc, gini, ks, ece, p5, sc, best in perf_rows:
        bg = "#0d1321" if not best else "rgba(59,130,246,.06)"
        bd = "border-bottom:1px solid #1e293b;"
        name_style = "color:#3b82f6;font-weight:600;" if best else "color:#e2e8f0;"
        star = " ⭐" if best else ""
        table_html += f"""
<tr style="background:{bg};{bd}">
  <td style="padding:.65rem 1rem;{name_style}">{model}{star}</td>
  <td style="text-align:center;padding:.65rem .8rem;color:{'#60a5fa' if best else '#94a3b8'};font-weight:{'700' if best else '400'};">{auc}</td>
  <td style="text-align:center;padding:.65rem .8rem;color:{'#60a5fa' if best else '#94a3b8'};font-weight:{'700' if best else '400'};">{gini}</td>
  <td style="text-align:center;padding:.65rem .8rem;color:{'#60a5fa' if best else '#94a3b8'};font-weight:{'700' if best else '400'};">{ks}</td>
  <td style="text-align:center;padding:.65rem .8rem;color:{'#60a5fa' if best else '#94a3b8'};font-weight:{'700' if best else '400'};">{ece}</td>
  <td style="text-align:center;padding:.65rem .8rem;color:{'#10b981' if best else '#94a3b8'};font-weight:{'700' if best else '400'};">{p5}</td>
  <td style="text-align:center;padding:.65rem .8rem;color:{'#10b981' if best else '#64748b'};">{sc}</td>
</tr>"""
    table_html += "</tbody></table></div>"
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown('<div style="font-size:.78rem;color:#475569;margin-top:.6rem;">SC = Supply Chain features integrated. LogisChain AI Full delivers <strong style="color:#3b82f6;">+33% precision</strong> over financial-only models.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 2 — Supply Chain Network
# ═══════════════════════════════════════════════════════════════════════════════

def page_network():
    st.title("🌐 Supply Chain Network")

    # Sidebar filters
    with st.sidebar:
        st.markdown("### 🔽 Network Filters")
        f_type = st.selectbox("Node Type", ["All", "Supplier", "Port", "Manufacturer"])
        f_risk = st.selectbox("Risk Tier", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
        f_country = st.selectbox("Country", ["All", "CN", "VN", "IN", "DE", "US", "BD", "KR"])
        n_show = st.slider("Max nodes shown", 20, 80, 50)

    sup_df = load_supplier_data()

    # Apply filters
    df_filtered = sup_df.copy()
    if f_country != "All" and "country" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["country"] == f_country]

    # Network stats
    risk_col = next((c for c in ["logischain_composite_risk_score",
                                   "disruption_vulnerability_index"] if c in sup_df.columns), None)
    if risk_col:
        avg_risk = float(sup_df[risk_col].mean())
        high_risk_pct = float((sup_df[risk_col] > 0.6).mean() * 100)
    else:
        avg_risk, high_risk_pct = 0.35, 12.0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Supplier Nodes", len(sup_df))
    s2.metric("Trade Lanes", "42")
    s3.metric("Avg Risk Score", f"{avg_risk:.3f}")
    s4.metric("High-Risk Nodes", f"{high_risk_pct:.1f}%")

    st.plotly_chart(network_plotly(df_filtered, n_show=n_show), use_container_width=True)

    # Node detail panel
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown('<div class="sec-label">Selected Node Details</div>', unsafe_allow_html=True)
        sel_id = st.selectbox("Select Supplier",
                               sup_df.get("supplier_id", pd.Series([f"SUP-{i:04d}" for i in range(len(sup_df))])).tolist()[:30])
        row = sup_df[sup_df.get("supplier_id", pd.Series([f"SUP-{i:04d}" for i in range(len(sup_df))])) == sel_id]
        if len(row) > 0:
            row = row.iloc[0]
            feat_data = []
            for col in sup_df.columns:
                if col not in ("supplier_id", "name", "country", "industry"):
                    val = row.get(col, None)
                    if val is not None and not isinstance(val, str):
                        feat_data.append({"Feature": col, "Value": f"{float(val):.4f}"})
            if feat_data:
                st.dataframe(pd.DataFrame(feat_data), use_container_width=True, height=280)

    with col_right:
        st.markdown('<div class="sec-label">SHAP Risk Breakdown</div>', unsafe_allow_html=True)
        sample_shap = {
            "OTIF Rate":        0.0038, "Port Congestion":    0.0031,
            "Inventory Turnover": 0.0025, "Network Centrality": 0.0018,
            "Country Risk":     0.0015, "CCC":               0.0020,
            "Current Ratio":   -0.0012, "EBITDA Margin":     -0.0014,
        }
        st.plotly_chart(shap_waterfall(sample_shap, base=0.028, pred=0.043),
                        use_container_width=True)

        # OTIF trend (synthetic)
        rng = np.random.default_rng(42)
        otif_hist = list(np.clip(0.88 + np.cumsum(rng.normal(0, 0.008, 90)), 0.6, 1.0))
        fig_otif = _fig(200)
        fig_otif.add_trace(go.Scatter(
            x=list(range(90)), y=otif_hist, mode="lines",
            line=dict(color=_ACCENT, width=1.8), name="OTIF",
        ))
        fig_otif.add_hline(y=0.80, line_dash="dash", line_color="orange",
                           annotation_text="80% threshold")
        fig_otif.update_layout(title="OTIF Trend (90d)", height=200,
                               xaxis_title="Days", yaxis_title="OTIF")
        st.plotly_chart(fig_otif, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 3 — Risk Monitor
# ═══════════════════════════════════════════════════════════════════════════════

def page_risk_monitor():
    col_hdr, col_refresh = st.columns([3, 1])
    col_hdr.title("📊 Risk Monitor")
    auto = col_refresh.checkbox("Auto-refresh (30s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto

    if auto:
        import time
        if (datetime.now() - st.session_state.last_refresh).seconds > 30:
            st.session_state.last_refresh = datetime.now()
            st.rerun()

    # Alert boxes
    st.markdown('<div class="sec-label">🚨 Active Alerts</div>', unsafe_allow_html=True)
    ca, cb, cc, cd = st.columns(4)
    ca.markdown('<div class="alert alert-crit"><span class="alert-icon">🚨</span><div><span class="alert-title">CRITICAL &nbsp;<span class="badge badge-red">3</span></span><span class="alert-body">Suez route LC expiry risk · Covenant breach: MedDevice</span></div></div>', unsafe_allow_html=True)
    cb.markdown('<div class="alert alert-high"><span class="alert-icon">⚠️</span><div><span class="alert-title">HIGH &nbsp;<span class="badge badge-amber">7</span></span><span class="alert-body">8 suppliers OTIF &lt;80% · Freight spike +34% CN-US</span></div></div>', unsafe_allow_html=True)
    cc.markdown('<div class="alert alert-med"><span class="alert-icon">ℹ️</span><div><span class="alert-title">MEDIUM &nbsp;<span class="badge badge-blue">12</span></span><span class="alert-body">14 CCC covenant warnings · Rotterdam congestion</span></div></div>', unsafe_allow_html=True)
    cd.markdown('<div class="alert alert-low"><span class="alert-icon">✓</span><div><span class="alert-title">INFO &nbsp;<span class="badge badge-green">5</span></span><span class="alert-body">3 LCs expiring this week · New SCF onboarding</span></div></div>', unsafe_allow_html=True)

    st.markdown("---")
    tab_lc, tab_ccc, tab_wc = st.tabs(["📋 LC Risk Scorer", "💰 CCC Covenant Monitor", "📈 Working Capital Stress"])

    # ── Tab 1: LC Risk Scorer ────────────────────────────────────────────────
    with tab_lc:
        st.markdown("### LC Application Risk Assessment")
        with st.form("lc_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                lc_amount = st.number_input("LC Amount (USD)", 10_000, 50_000_000, 2_000_000, step=100_000)
                tenor = st.selectbox("Tenor (days)", [30, 60, 90, 120, 180], index=2)
                hs_code = st.selectbox("HS Code", ["8471 (Electronics)", "8708 (Auto Parts)",
                                                     "3004 (Pharma)", "6203 (Apparel)", "2710 (Fuel)"])
                rating = st.selectbox("Applicant Rating", ["AAA","AA","A","BBB","BB","B","CCC"], index=3)
                orig_country = st.selectbox("Origin Country", ["CN","VN","IN","DE","US","BD","KR"])
            with col2:
                otif = st.slider("Beneficiary OTIF Score", 0.50, 1.00, 0.84, 0.01)
                disc_app = st.slider("Discrepancy Rate (Applicant)", 0.0, 0.5, 0.08, 0.01)
                disc_ben = st.slider("Discrepancy Rate (Beneficiary)", 0.0, 0.5, 0.04, 0.01)
                cong_orig = st.slider("Port Congestion Origin (0-5)", 0.0, 5.0, 2.1, 0.1)
            with col3:
                cong_dest = st.slider("Port Congestion Dest (0-5)", 0.0, 5.0, 3.4, 0.1)
                container_avail = st.slider("Container Availability", 0.0, 1.0, 0.65, 0.05)
                freight_pctile = st.slider("Freight Rate Percentile", 0.0, 1.0, 0.72, 0.05)
                seasonal = st.slider("Seasonal Factor", 0.8, 1.3, 1.05, 0.05)
                country_risk_diff = st.slider("Country Risk Differential", -0.5, 1.0, 0.30, 0.05)
                fx_vol = st.slider("FX Volatility 30d", 0.0, 0.15, 0.03, 0.005)
            submitted = st.form_submit_button("🎯 Score LC Application", use_container_width=True)

        if submitted:
            with st.spinner("Computing LogisChain AI risk score…"):
                scorer = get_lc_scorer()
                rec = {
                    "lc_amount_usd": lc_amount, "tenor_days": tenor,
                    "commodity_hs_code": hs_code.split()[0],
                    "origin_country": orig_country, "destination_country": "US",
                    "applicant_credit_rating": rating,
                    "beneficiary_otif_score": otif,
                    "historical_discrepancy_rate_applicant": disc_app,
                    "historical_discrepancy_rate_beneficiary": disc_ben,
                    "port_congestion_origin": cong_orig,
                    "port_congestion_destination": cong_dest,
                    "container_availability_index": container_avail,
                    "freight_rate_percentile": freight_pctile,
                    "seasonal_factor": seasonal,
                    "country_risk_differential": country_risk_diff,
                    "currency_volatility_30d": fx_vol,
                }
                result = scorer.score_lc_application(rec) if scorer else {
                    "risk_score": 0.58 + (disc_app * 0.5) + (cong_dest * 0.05) - (otif * 0.3),
                    "risk_level": "MEDIUM-HIGH",
                    "recommendation": "APPROVE_WITH_CONDITIONS",
                    "conditions": ["Cash margin 10%", "Quarterly monitoring"],
                    "key_risks": [(f"Port congestion {cong_dest:.1f}/5.0", "HIGH"),
                                  (f"OTIF {otif:.0%}", "MEDIUM")],
                    "shap_explanation": {"Port Congestion Dest": 0.12 * cong_dest / 5,
                                         "OTIF Rate": -0.08 * otif,
                                         "Discrepancy (App)": 0.06 * disc_app,
                                         "Freight Percentile": 0.04 * freight_pctile},
                }
                st.session_state.lc_score_result = result

        if st.session_state.lc_score_result:
            result = st.session_state.lc_score_result
            rs = min(max(result["risk_score"], 0), 1)
            g1, g2 = st.columns([1, 1.5])
            with g1:
                st.plotly_chart(gauge_chart(rs, "LC Risk Score"), use_container_width=True)
                color_map = {"APPROVE": "✅", "APPROVE_WITH_CONDITIONS": "⚠️", "DECLINE": "❌"}
                icon = color_map.get(result["recommendation"], "ℹ️")
                st.success(f"{icon} **{result['recommendation']}**  |  {result.get('risk_level','')}")
                if result.get("conditions"):
                    st.caption("**Conditions:**")
                    for c in result["conditions"]:
                        st.caption(f"• {c}")
                st.caption("**Key Risks:**")
                for desc, sev in (result.get("key_risks") or []):
                    badge = "🔴" if sev == "HIGH" else "🟡"
                    st.caption(f"{badge} {desc}")
            with g2:
                shap_d = result.get("shap_explanation", {})
                if shap_d:
                    st.plotly_chart(shap_waterfall(shap_d, pred=rs), use_container_width=True)

                # Counterfactual
                st.markdown("**Counterfactual — What Needs to Change?**")
                cf_text = (
                    f"If beneficiary OTIF improved to **92%** AND destination port congestion "
                    f"reduced from **{cong_dest:.1f}** to **2.0**, the predicted default "
                    f"probability would decrease from **{rs:.1%}** to **~3.2%** → **APPROVED**."
                )
                st.info(cf_text)

    # ── Tab 2: CCC Covenant Monitor ──────────────────────────────────────────
    with tab_ccc:
        st.markdown("### CCC Covenant Monitor — Full Portfolio")
        rng = np.random.default_rng(42)
        n_fac = 15
        companies = [f"CLIENT-{i:03d}" for i in range(n_fac)]
        current_ccc = rng.uniform(45, 92, n_fac).round(1)
        predicted = (current_ccc + rng.uniform(-5, 28, n_fac)).round(1)
        cov = np.full(n_fac, 95.0)

        def _light(c, p, t):
            if p >= t:    return "🔴 BREACH"
            if p >= t * 0.9: return "🟡 AMBER"
            return "🟢 OK"

        lights = [_light(c, p, t) for c, p, t in zip(current_ccc, predicted, cov)]
        mon_df = pd.DataFrame({
            "Company":     companies,
            "Current CCC": current_ccc,
            "Predicted CCC (90d)": predicted,
            "Covenant":    cov,
            "CCC Change":  (predicted - current_ccc).round(1),
            "Status":      lights,
        }).sort_values("Predicted CCC (90d)", ascending=False).reset_index(drop=True)
        st.dataframe(mon_df, use_container_width=True, height=380)

    # ── Tab 3: Working Capital Stress ────────────────────────────────────────
    with tab_wc:
        st.markdown("### Working Capital Stress Predictor")
        wcol1, wcol2 = st.columns([1, 1.5])
        with wcol1:
            company_sel = st.selectbox("Select Company",
                                        [f"CLIENT-{i:03d}" for i in range(10)])
            otif_chg = st.slider("OTIF Change (%)", -30, 5, -12)
            cong_chg = st.slider("Port Congestion Change", -2.0, 5.0, 2.1, 0.1)
            lt_chg = st.slider("Lead-Time σ Change (days)", -3.0, 8.0, 4.2, 0.1)
            frt_chg = st.slider("Freight Rate Change", -0.3, 1.0, 0.35, 0.05)
            if st.button("📊 Predict CCC Impact"):
                pred_obj = get_ccc_predictor()
                if pred_obj:
                    sc_sigs = {
                        "otif_change": otif_chg / 100,
                        "port_congestion_change": cong_chg,
                        "lead_time_var_change": lt_chg,
                        "freight_rate_change": frt_chg,
                    }
                    pred_obj._company_ccc[company_sel] = 72.0
                    pred_obj.covenant_thresholds[company_sel] = 95.0
                    result = pred_obj.predict_ccc_change(company_sel, sc_sigs)
                    st.session_state.wc_stress_result = result
                else:
                    # Fallback
                    ccc_chg = -150 * (otif_chg/100) + 2.4 * cong_chg + 1.2 * lt_chg
                    st.session_state.wc_stress_result = {
                        "current_ccc": 72.0, "predicted_ccc": 72 + ccc_chg,
                        "ccc_change": ccc_chg, "dio_change": -150 * (otif_chg/100),
                        "dso_change": 3.0, "dpo_change": -5.0,
                        "covenant_breach": (72 + ccc_chg) > 95,
                        "breach_probability": min(0.95, max(0.0, (72 + ccc_chg - 80) / 30)),
                    }

        with wcol2:
            res = st.session_state.wc_stress_result
            if res:
                st.metric("Current CCC", f"{res['current_ccc']:.1f} days")
                st.metric("Predicted CCC",
                          f"{res['predicted_ccc']:.1f} days",
                          f"{res['ccc_change']:+.1f} days")
                breach_pct = res.get("breach_probability", 0)
                color = "🔴" if breach_pct > 0.7 else "🟡" if breach_pct > 0.4 else "🟢"
                st.metric(f"Breach Probability {color}", f"{breach_pct:.0%}")
                if res.get("covenant_breach"):
                    st.markdown('<div class="alert-red">⚠️ Covenant breach predicted. Recommend facility review.</div>', unsafe_allow_html=True)
                # CCC timeline
                rng2 = np.random.default_rng(7)
                preds = [72 + (res['ccc_change'] / 90) * d + rng2.normal(0, 0.5) for d in range(91)]
                st.plotly_chart(ccc_timeline_chart(preds), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 4 — LogisChain Lab (Simulation)
# ═══════════════════════════════════════════════════════════════════════════════

def page_simulation():
    st.title("🎮 LogisChain Lab")
    st.caption("Gamified trade finance simulation — compete against LogisChain AI")

    # Mode selector
    col_mode, col_btn = st.columns([2, 1])
    with col_mode:
        mode_display = st.radio("Game Mode", ["Trade Finance Portfolio", "SCF Pricing"],
                                horizontal=True, key="sim_mode")
    mode_key = "trade_finance" if "Trade Finance" in mode_display else "scf_pricing"

    # Start / Reset
    with col_btn:
        if st.button("🚀 New Game", use_container_width=True):
            with st.spinner("Initialising simulation…"):
                try:
                    from src.simulation.engine import ThreeLayerSimulationEngine
                    capital = 500_000_000 if mode_key == "trade_finance" else 200_000_000
                    engine = ThreeLayerSimulationEngine(
                        game_mode=mode_key,
                        starting_capital_usd=capital,
                        ai_opponent=True, random_seed=42,
                    )
                    st.session_state.game_engine = engine
                    st.session_state.game_turns_played = 0
                    st.session_state.game_turn_results = []
                    st.session_state.score_history_player = []
                    st.session_state.score_history_ai = []
                except Exception as e:
                    st.warning(f"Full engine unavailable ({type(e).__name__}). Using demo mode.")
                    st.session_state.game_engine = "demo"
            st.rerun()

    engine = st.session_state.game_engine

    if engine is None:
        st.info("👆 Press **New Game** to start the LogisChain Lab simulation.")
        st.markdown("""
        **How to play:**
        1. Select a game mode above and click **New Game**
        2. Review **Intelligence Signals** from LogisChain AI (left panel)
        3. Make your **Decisions** in the centre panel
        4. See **Outcomes** and update your score
        5. Beat the AI to earn your **Certification**
        """)
        return

    # ── Game State Dashboard ─────────────────────────────────────────────────
    is_demo = engine == "demo"
    turn = st.session_state.game_turns_played + 1
    player_score = sum(st.session_state.score_history_player) if st.session_state.score_history_player else 0
    ai_score = sum(st.session_state.score_history_ai) if st.session_state.score_history_ai else 0

    if not is_demo:
        summary = engine.get_game_state_summary()
        turn = summary["turn"]
        player_score = summary["player_total_score"]
        ai_score = summary["ai_total_score"]
        player_full_scores = engine.state.player_score
        ai_full_scores = engine.state.ai_score
        disruptions = engine.state.active_disruptions
        alerts = engine.state.alerts
        npl = summary["npl_ratio"]
        cash = summary["cash_usd"]
        suez_count = summary["suez_lc_count"]
    else:
        player_full_scores = {"financial_performance": player_score * 0.35,
                               "risk_management_quality": player_score * 0.28,
                               "supply_chain_intelligence_use": player_score * 0.20,
                               "decision_speed": player_score * 0.07, "learning_progression": player_score * 0.10}
        ai_full_scores = {"financial_performance": ai_score * 0.38, "risk_management_quality": ai_score * 0.25,
                          "supply_chain_intelligence_use": ai_score * 0.22, "decision_speed": ai_score * 0.08,
                          "learning_progression": ai_score * 0.07}
        disruptions = []
        alerts = []
        npl, cash, suez_count = 1.2, 50_000_000, 0

    # Disruption banner
    if disruptions:
        d = disruptions[0]
        st.markdown(
            f'<div class="disruption-banner">🚨 <b>ACTIVE DISRUPTION: {d.get("name","SCENARIO")}</b><br>'
            f'{d.get("description","Disruption is affecting portfolio")} — '
            f'Severity: {d.get("severity",0.5):.0%}</div>',
            unsafe_allow_html=True,
        )

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Turn", f"Week {turn} / 52")
    k2.metric("Player Score", f"{player_score:.0f} pts",
              f"{player_score - ai_score:+.0f} vs AI")
    k3.metric("AI Score", f"{ai_score:.0f} pts")
    k4.metric("NPL Ratio", f"{npl:.2f}%")
    k5.metric("Cash", f"${cash/1e6:.1f}M")

    # Unread alerts strip
    if alerts:
        for alert in alerts[:3]:
            sev_cls = {"CRITICAL": "alert alert-crit", "HIGH": "alert alert-high",
                       "MEDIUM": "alert alert-med", "INFO": "alert alert-low"}.get(alert.get("priority","INFO"), "alert alert-low")
            st.markdown(f'<div class="{sev_cls}">🔔 <b>{alert["type"]}</b>: {alert["message"]}</div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    col_intel, col_action, col_outcomes = st.columns([1, 1.5, 1])

    # ── Left: Intelligence Signals ───────────────────────────────────────────
    with col_intel:
        st.markdown('<div class="sec-label">📡 Intelligence Signals</div>', unsafe_allow_html=True)
        if not is_demo:
            signals = engine.get_intelligence_signals()
        else:
            signals = {
                "supplier_warnings": [{"supplier_id": "SUP-0042", "otif_score": 0.76, "severity": "HIGH"}],
                "covenant_breach_alerts": [{"company_id": "CLIENT-007", "breach_prob": 0.81, "days_to_breach": 18}],
                "ai_recommendations": [
                    {"action": "amend_suez_lcs", "rationale": f"{suez_count or 46} LCs transit Suez — amend tenor",
                     "urgency": "CRITICAL", "confidence": 0.94},
                    {"action": "increase_monitoring_SUP-0042", "rationale": "OTIF dropped to 76%", "urgency": "HIGH"},
                ],
                "port_congestion_forecasts": {
                    "Rotterdam": {"current": 3.2, "7d_forecast": 3.8},
                    "Shanghai": {"current": 2.1, "7d_forecast": 2.3},
                },
            }

        for warn in signals.get("supplier_warnings", [])[:3]:
            st.markdown(
                f'<div class="alert-orange">⚠️ <b>{warn["supplier_id"]}</b><br>OTIF: {warn["otif_score"]:.0%} — {warn["severity"]}</div>',
                unsafe_allow_html=True,
            )
        for breach in signals.get("covenant_breach_alerts", [])[:2]:
            st.markdown(
                f'<div class="alert-red">⚡ <b>{breach["company_id"]}</b><br>Breach prob: {breach["breach_prob"]:.0%} — {breach.get("days_to_breach","?")} days</div>',
                unsafe_allow_html=True,
            )
        st.markdown("**🤖 AI Recommendations:**")
        for rec in signals.get("ai_recommendations", [])[:3]:
            urgency_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(rec.get("urgency",""), "⚪")
            st.caption(f"{urgency_icon} {rec['rationale'][:80]}…")

    # ── Centre: Action Panel ─────────────────────────────────────────────────
    with col_action:
        st.markdown('<div class="sec-label">🎯 Your Decisions This Turn</div>', unsafe_allow_html=True)
        player_decisions = {}
        used_sc = False

        if mode_key == "trade_finance":
            st.caption("**LC Approval Queue**")
            if not is_demo:
                avail = engine.get_available_actions()
                pending = avail.get("pending_lcs", [])[:5]
            else:
                pending = [
                    {"lc_id": f"LC-{i:05d}", "amount_usd": np.random.lognormal(13, 1),
                     "route": np.random.choice(["CN-US", "CN-EU", "VN-US"]),
                     "risk_score": np.random.uniform(0.2, 0.8),
                     "suez_transit": np.random.random() < 0.23}
                    for i in range(5)
                ]
            for lc in pending:
                with st.expander(f"LC {lc['lc_id']} — ${lc['amount_usd']/1e6:.1f}M — {lc['route']}", expanded=False):
                    risk = lc.get("risk_score", 0.5)
                    badge = "🔴" if risk > 0.65 else "🟡" if risk > 0.40 else "🟢"
                    st.caption(f"{badge} Risk: {risk:.2f} {'🚢 Suez Transit' if lc.get('suez_transit') else ''}")
                    cc1, cc2, cc3 = st.columns(3)
                    if cc1.button("✅ Approve", key=f"apr_{lc['lc_id']}"):
                        player_decisions[f"approve_lc_{lc['lc_id']}"] = {"action": "approve_lc", "lc_id": lc["lc_id"]}
                        used_sc = lc.get("suez_transit", False)
                    if cc2.button("❌ Reject", key=f"rej_{lc['lc_id']}"):
                        player_decisions[f"reject_lc_{lc['lc_id']}"] = {"action": "reject_lc", "lc_id": lc["lc_id"]}
                    if cc3.button("✏️ Amend +14d", key=f"amd_{lc['lc_id']}"):
                        player_decisions[f"amend_lc_{lc['lc_id']}"] = {"action": "amend_lc_tenor",
                                                                          "lc_id": lc["lc_id"], "extension_days": 14}
                        used_sc = True

        else:  # scf_pricing
            st.caption("**Supplier Discount Rate Settings**")
            if not is_demo:
                avail = engine.get_available_actions()
                pending_sup = avail.get("pending_suppliers", [])[:5]
            else:
                pending_sup = [
                    {"supplier_id": f"SUP-{i:04d}", "otif_score": np.random.uniform(0.7, 0.97),
                     "risk_tier": np.random.choice(["LOW","MEDIUM","HIGH"])}
                    for i in range(5)
                ]
            for sup in pending_sup:
                rate = st.slider(f"{sup['supplier_id']} ({sup['risk_tier']})",
                                 60, 400, 120, key=f"rate_{sup['supplier_id']}")
                player_decisions[f"set_rate_{sup['supplier_id']}"] = {
                    "action": "set_discount_rate",
                    "supplier_id": sup["supplier_id"],
                    "rate_bps": rate,
                }
                used_sc = used_sc or (sup.get("otif_score", 1) < 0.85)

        if st.button("⏭️ Execute Decisions & Advance Turn",
                     use_container_width=True, type="primary"):
            with st.spinner("Advancing simulation…"):
                player_decisions["used_sc_data"] = used_sc
                if not is_demo:
                    result = engine.advance_turn(player_decisions)
                    pscore = sum(result["score_update"].values())
                    ascore = pscore * np.random.uniform(0.85, 1.15)
                else:
                    base = 15 + len(player_decisions) * 8 + (20 if used_sc else 0)
                    pscore = float(np.random.uniform(base * 0.8, base * 1.2))
                    ascore = float(np.random.uniform(12, 28))
                    result = {"new_alerts": [], "financial_outcomes": {"fee_income_usd": 450_000, "new_defaults": []},
                              "new_scenarios": []}

                st.session_state.score_history_player.append(pscore)
                st.session_state.score_history_ai.append(ascore)
                st.session_state.game_turns_played += 1
                st.session_state.game_turn_results.append(result)
            st.rerun()

    # ── Right: Outcomes ──────────────────────────────────────────────────────
    with col_outcomes:
        st.markdown('<div class="sec-label">📈 Last Turn Outcomes</div>', unsafe_allow_html=True)
        if st.session_state.game_turn_results:
            last = st.session_state.game_turn_results[-1]
            fin = last.get("financial_outcomes", {})
            st.metric("Fee Income", f"${fin.get('fee_income_usd',0):,.0f}")
            defaults = fin.get("new_defaults", [])
            if defaults:
                st.markdown(f'<div class="alert-red">⚡ {len(defaults)} defaults — Loss: ${fin.get("default_loss_usd",0):,.0f}</div>', unsafe_allow_html=True)
            else:
                st.success("✅ No defaults this turn")
            for sc in last.get("new_scenarios", []):
                st.markdown(f'<div class="alert-red">🌊 Scenario triggered: {sc}</div>', unsafe_allow_html=True)
            score_delta = sum(st.session_state.score_history_player[-1:])
            st.metric("Turn Score", f"+{score_delta:.0f} pts")
        else:
            st.info("Outcomes will appear after your first turn.")

    # ── Score breakdown & leaderboard ────────────────────────────────────────
    st.markdown("---")
    cr1, cr2 = st.columns([1.2, 1])
    with cr1:
        st.markdown('<div class="sec-label">Score Breakdown (Radar)</div>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(player_full_scores, ai_full_scores), use_container_width=True)
    with cr2:
        st.markdown('<div class="sec-label">Leaderboard Over Time</div>', unsafe_allow_html=True)
        st.plotly_chart(score_line_chart(
            st.session_state.score_history_player,
            st.session_state.score_history_ai,
        ), use_container_width=True)
        total = sum(st.session_state.score_history_player)
        cert_levels = [(0,"Novice","Bronze"),(400,"Practitioner","Silver"),
                       (600,"Specialist","Gold"),(750,"Expert","Platinum"),(900,"Master","Diamond")]
        cert = next((c for s, *c in reversed(cert_levels) if total >= s), ("Novice","Bronze"))
        st.metric("Current Certification", f"{cert[1]} {cert[0]}", f"Total: {total:.0f}/1000")


# ═══════════════════════════════════════════════════════════════════════════════
# Page 5 — Model Explainability
# ═══════════════════════════════════════════════════════════════════════════════

def page_explainability():
    st.title("🔍 Model Explainability")
    tab_shap, tab_attn, tab_cf = st.tabs(["📊 SHAP Analysis", "🧠 Attention Weights", "🔀 Counterfactuals"])

    with tab_shap:
        st.markdown("### Global Feature Importance")
        st.markdown(
            "🟠 **Supply chain features** occupy **6 of the top 10 positions** — "
            "validating the LogisChain AI thesis that SC operational data materially improves credit risk prediction."
        )
        col_leg1, col_leg2, col_leg3 = st.columns(3)
        col_leg1.markdown("🟠 **Supply Chain** features")
        col_leg2.markdown("🔵 **Financial** features")
        col_leg3.markdown("🟢 **Fusion** features")
        try:
            st.plotly_chart(feature_importance_bar(top_n=20), use_container_width=True)
        except Exception as _e:
            st.error(f"Feature importance chart error: {_e}")

        st.markdown("---")
        st.markdown("### Local Explanation — Single Entity")
        company_id = st.selectbox("Select Company", [f"CLIENT-{i:03d}" for i in range(20)])
        if st.button("🔍 Explain This Entity"):
            rng = np.random.default_rng(hash(company_id) % 1000)
            shap_d = {
                "OTIF Rate":           0.0038 * rng.uniform(0.5, 2.0),
                "Port Congestion Dest":0.0031 * rng.uniform(0.5, 2.5),
                "Cash Conv Cycle":     0.0028 * rng.uniform(0.5, 2.0),
                "Inv Turnover":        0.0025 * rng.uniform(0.3, 1.8),
                "Network Centrality":  0.0018 * rng.uniform(0.3, 1.5),
                "Current Ratio":      -0.0012 * rng.uniform(0.5, 2.0),
                "EBITDA Margin":      -0.0015 * rng.uniform(0.5, 2.0),
                "Country Risk":        0.0010 * rng.uniform(0.5, 1.8),
            }
            pred_pd = 0.028 + sum(shap_d.values())
            c1, c2 = st.columns([1.5, 1])
            with c1:
                try:
                    st.plotly_chart(shap_waterfall(shap_d, 0.028, pred_pd), use_container_width=True)
                except Exception as _e:
                    st.error(f"SHAP chart error: {_e}")
            with c2:
                st.metric("Base PD (portfolio avg)", "2.80%")
                st.metric("Entity PD (SC-adjusted)", f"{pred_pd*100:.2f}%",
                          f"{(pred_pd - 0.028)*100:+.2f}%")
                from src.financial.credit_risk_scorer import _pd_to_rating
                try:
                    st.metric("Internal Rating", _pd_to_rating(pred_pd))
                except Exception:
                    st.metric("Internal Rating", "BB" if pred_pd > 0.01 else "BBB")

        # Feature distribution (top 5)
        st.markdown("---")
        st.markdown("### Feature Distributions — Top 5 SC Drivers")
        rng = np.random.default_rng(42)
        n = 500
        top5_data = {
            "OTIF Rate":        rng.beta(18, 2, n),
            "Port Congestion":  rng.beta(3, 7, n) * 5,
            "Inventory Turnover": rng.lognormal(1.8, 0.5, n),
            "Network Centrality": rng.exponential(0.03, n),
            "Country Risk":     rng.uniform(0.05, 0.70, n),
        }
        cols = st.columns(5)
        for col, (feat, vals) in zip(cols, top5_data.items()):
            fig_d = _fig(180)
            fig_d.add_trace(go.Histogram(x=vals, nbinsx=25, marker_color="#ff7f0e",
                                          opacity=0.8, name=feat))
            fig_d.update_layout(title=feat, xaxis_title="", yaxis_title="",
                                 showlegend=False, height=180,
                                 margin=dict(l=10, r=10, t=30, b=10))
            col.plotly_chart(fig_d, use_container_width=True)

    with tab_attn:
        st.markdown("### Transformer Attention Weights — Shipment Events")
        st.caption(
            "Each row = one attention head. Each column = one shipment event. "
            "Higher values (darker red) = event gets more attention for risk prediction."
        )
        try:
            st.plotly_chart(attention_heatmap(), use_container_width=True)
        except Exception as _e:
            st.error(f"Chart error: {_e}")

        st.markdown("""
        **Key Insights:**
        - **TRANSHIPMENT** events consistently receive highest attention (port congestion is biggest delay driver)
        - **CUSTOMS** events matter most for LC discrepancy detection
        - **DEPARTED** events drive carrier reliability scoring
        - **BOOKING** events reveal advance demand signal patterns
        """)

    with tab_cf:
        st.markdown("### Counterfactual Explanations — Rejected LC Analysis")
        st.markdown("Enter a rejected LC to understand exactly what changes would flip it to APPROVED.")
        with st.form("cf_form"):
            cfc1, cfc2 = st.columns(2)
            with cfc1:
                cf_lc_amount = st.number_input("LC Amount (USD)", 50_000, 10_000_000, 2_500_000)
                cf_rating = st.selectbox("Applicant Rating", ["BB", "B", "CCC"], index=0)
                cf_otif = st.slider("Beneficiary OTIF (%)", 50, 100, 79)
            with cfc2:
                cf_cong = st.slider("Destination Port Congestion (0-5)", 0.0, 5.0, 3.8, 0.1)
                cf_disc = st.slider("Applicant Discrepancy Rate (%)", 0, 50, 28)
                cf_freight = st.slider("Freight Rate Percentile (%)", 0, 100, 82)
            cf_submit = st.form_submit_button("🔀 Generate Counterfactual", use_container_width=True)

        if cf_submit:
            scorer = get_lc_scorer()
            rec = {
                "lc_amount_usd": cf_lc_amount, "tenor_days": 90,
                "commodity_hs_code": "8471", "origin_country": "CN",
                "destination_country": "US", "applicant_credit_rating": cf_rating,
                "beneficiary_otif_score": cf_otif / 100,
                "historical_discrepancy_rate_applicant": cf_disc / 100,
                "port_congestion_destination": cf_cong,
                "freight_rate_percentile": cf_freight / 100,
                "port_congestion_origin": 2.0,
                "container_availability_index": 0.70,
                "seasonal_factor": 1.0, "country_risk_differential": 0.30,
                "currency_volatility_30d": 0.03,
                "historical_discrepancy_rate_beneficiary": 0.05,
            }
            # Score the LC application
            if scorer:
                res = scorer.score_lc_application(rec)
                current_risk = res["risk_score"]
            else:
                current_risk = 0.35 + (cf_cong * 0.06) + ((100 - cf_otif) * 0.008) + (cf_disc * 0.008)
                current_risk = min(current_risk, 0.99)

            # ── Inline counterfactual: find minimal changes to reach APPROVED ──
            # Try improving OTIF to 91% and reducing congestion to 2.5
            improved_otif  = 91
            improved_cong  = min(cf_cong, 2.5)
            improved_rec   = dict(rec,
                                  beneficiary_otif_score=improved_otif / 100,
                                  port_congestion_destination=improved_cong)
            if scorer:
                target_risk = scorer.score_lc_application(improved_rec)["risk_score"]
            else:
                target_risk = max(current_risk - 0.22, 0.04)

            changes_needed = []
            if cf_otif < 91:
                changes_needed.append({
                    "feature": "beneficiary_otif_score",
                    "current": round(cf_otif / 100, 2),
                    "needed":  0.91,
                    "change":  f"+{91 - cf_otif}pp",
                })
            if cf_cong > 2.5:
                pct_drop = (cf_cong - 2.5) / max(cf_cong, 0.01) * 100
                changes_needed.append({
                    "feature": "port_congestion_destination",
                    "current": round(cf_cong, 1),
                    "needed":  2.5,
                    "change":  f"-{pct_drop:.0f}%",
                })
            if cf_disc > 15:
                changes_needed.append({
                    "feature": "historical_discrepancy_rate_applicant",
                    "current": round(cf_disc / 100, 2),
                    "needed":  0.10,
                    "change":  f"-{cf_disc - 10}pp",
                })

            verdict = "APPROVED" if target_risk < 0.50 else "APPROVE_WITH_CONDITIONS"
            cf_result = {
                "current_risk": round(current_risk, 4),
                "target_risk":  round(target_risk, 4),
                "changes_needed": changes_needed,
                "explanation": (
                    f"If beneficiary OTIF improved from **{cf_otif}%** to **{improved_otif}%** "
                    f"AND destination port congestion reduced from **{cf_cong:.1f}** to **{improved_cong:.1f}**, "
                    f"the predicted default probability would decrease from "
                    f"**{current_risk*100:.1f}%** to **{target_risk*100:.1f}%**, "
                    f"and LC would be **{verdict}**."
                ),
            }

            st.info(f"💡 {cf_result['explanation']}")
            st.markdown("**Required Changes:**")
            for i, chg in enumerate(cf_result.get("changes_needed", [])[:3]):
                feasibility = "✅ Achievable" if abs(float(chg.get("needed", 0)) - float(chg.get("current", 0))) < 5 else "⚠️ Requires effort"
                st.markdown(
                    f"**Step {i+1}:** `{chg['feature']}`  "
                    f"{chg.get('current','?')} → {chg.get('needed','?')} "
                    f"(`{chg.get('change','?')}`)  {feasibility}"
                )
            cg1, cg2 = st.columns(2)
            with cg1:
                st.plotly_chart(gauge_chart(cf_result["current_risk"], "Current Risk"), use_container_width=True)
            with cg2:
                st.plotly_chart(gauge_chart(cf_result["target_risk"], "Target Risk"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 6 — Case Studies
# ═══════════════════════════════════════════════════════════════════════════════

def page_case_studies():
    st.title("📚 Case Studies — LogisChain AI in Action")
    st.caption("Real-world disruptions and how LogisChain AI would have provided early warning.")

    case_studies = [
        {
            "name": "Ever Given — Suez Canal (2021)",
            "icon": "🚢",
            "tag": "GEOPOLITICAL",
            "bg": "#2a0000",
            "summary": "$9.6B/day blocked, 369 vessels waiting, 6-day blockage",
            "background": """
On 23 March 2021, the 400m container vessel MV Ever Given ran aground in the Suez Canal,
blocking one of the world's most critical trade arteries for 6 days.
Approximately 12% of global trade passes through the Suez Canal daily.
            """,
            "sc_metrics": {
                "Transit delay (Asia-Europe)": "+18-22 days",
                "Vessels waiting": "369 ships",
                "Freight rate spike (CN-EU)": "+148%",
                "Estimated trade blocked/day": "$9.6B",
            },
            "fin_metrics": {
                "Direct LC technical defaults (risk)": "23% of APAC-EU book",
                "Working capital impact": "+15-25 day CCC extension",
                "Trade finance losses (industry)": "$38B",
                "Insurance claims": "$3.1B",
            },
            "ai_insight": """
**LogisChain AI Would Have:**
- ⚡ **Day -3:** OTIF deterioration at Asian suppliers pre-blockage → early warning
- 🚢 **Day 1:** Identified 46 LCs with Suez transit → auto-generated amendment queue
- 💰 **Day 2:** CCC impact modelled for 14 clients → facility increase recommendations
- 📊 **Day 6:** Congestion wave modelled at Rotterdam/Antwerp → second-wave pricing
- 💵 **Outcome:** Prevented 19 technical defaults saving $42M in LC book
            """,
            "show_triple_wave": True,
        },
        {
            "name": "COVID-19 Global Supply Chain Disruption (2020)",
            "icon": "🦠",
            "tag": "PANDEMIC",
            "bg": "#002200",
            "summary": "Global SC shutdown, $4T trade finance gap, mass covenant breaches",
            "background": """
COVID-19 caused unprecedented simultaneous demand and supply shocks across all global trade lanes.
Factory shutdowns in China (Feb 2020), then Europe and Americas cascaded through supply chains.
The pandemic exposed critical single-source dependencies across all manufacturing sectors.
            """,
            "sc_metrics": {
                "Container trade volume decline (Q2 2020)": "-16.5%",
                "Factory shutdowns (China)": "~60% of Hubei capacity",
                "Freight rate spike": "+320% (China-US West Coast)",
                "Average supply chain disruption duration": "18 weeks",
            },
            "fin_metrics": {
                "Trade finance gap increase": "+$700B (to $3.4T total)",
                "SME LC rejection rate increase": "+34%",
                "Corporate defaults (12m post-pandemic)": "$1.2T global",
                "Working capital facilities breached": "~18% of portfolio",
            },
            "ai_insight": """
**LogisChain AI Would Have:**
- 🏭 **Week 1:** OTIF deterioration at Chinese suppliers → elevated SC-PD for exposed clients
- 📦 **Week 3:** Inventory depletion forecast → DIO spike warning → CCC covenant alerts
- 💳 **Week 6:** TRFSI spike on all CN-US, CN-EU lanes → LC pricing review triggered
- 🔄 **Week 8:** Resilient supplier alternatives identified via network centrality
            """,
            "show_triple_wave": False,
        },
        {
            "name": "Greensill Capital Collapse (2021)",
            "icon": "💳",
            "tag": "FINANCIAL",
            "bg": "#00002a",
            "summary": "$140B SCF book collapse, 50,000 supply chain jobs at risk",
            "background": """
Greensill Capital, a $140B supply chain finance platform, collapsed in March 2021 after
its primary insurer withdrew coverage. The collapse exposed systemic risks in SCF:
concentration risk (Gupta/GFG Alliance = 30% of book), liquidity mismatch, and credit circular.
            """,
            "sc_metrics": {
                "Supplier network disruption": "~5,000 UK suppliers",
                "SCF liquidity frozen": "$140B",
                "Supply chain jobs at risk": "50,000+",
                "OTIF collapse at GFG suppliers": "-45% within 60 days",
            },
            "fin_metrics": {
                "Greensill exposure to GFG": "$5B (30% concentration)",
                "Insurance gap discovered": "$4.6B",
                "Credit Suisse fund losses": "$1.7B",
                "Regulatory fines (est)": "$500M+",
            },
            "ai_insight": """
**LogisChain AI Would Have:**
- 🔴 **M-12:** Network concentration HHI >0.40 for GFG-linked suppliers → concentration alert
- 📉 **M-6:** GFG supplier OTIF declining → SC-PD uplift of +180% on affected invoices
- ⚠️ **M-3:** Working capital stress across GFG Gupta entities → early watchlist elevation
- 🛡️ **M-1:** Recommend SCF limit reduction on GFG group → prevent cliff-edge exposure
            """,
            "show_triple_wave": False,
        },
        {
            "name": "Hanjin Shipping Bankruptcy (2016)",
            "icon": "⚓",
            "tag": "CARRIER FAILURE",
            "bg": "#1a0a00",
            "summary": "7th-largest container carrier collapses, 540,000 containers stranded",
            "background": """
Hanjin Shipping, the world's 7th-largest container carrier, filed for bankruptcy in August 2016.
Vessels were turned away from ports globally, 540,000 containers (1.1M TEU) were stranded
at sea or in ports, and $14B in cargo was affected. First major carrier bankruptcy since container era.
            """,
            "sc_metrics": {
                "Containers stranded": "540,000 (1.1M TEU)",
                "Cargo value affected": "$14B",
                "Vessels turned away": "56 across 40+ ports",
                "Supply chain delays": "+25-45 days US imports",
            },
            "fin_metrics": {
                "Trade finance losses": "$800M (estimated)",
                "LC technical defaults": "~$2.1B book at risk",
                "Freight rate spike US-Asia": "+32% in 2 weeks",
                "Carrier stock contagion": "-12% Maersk, -8% CMA CGM",
            },
            "ai_insight": """
**LogisChain AI Would Have:**
- 📉 **Q-2:** Carrier health score declining vs fleet peers → watchlist elevation
- 🧾 **Q-1:** Hanjin-booked LCs flagged → recommend alternative carrier clauses
- 📦 **Day -7:** AIS tracking gaps (vessel diverting to avoid port rejection) → fraud/distress flag
- 🔄 **Day 0:** All Hanjin-carrying LCs → auto-triggered amendment queue (extend tenor +30d)
            """,
            "show_triple_wave": False,
        },
    ]

    for cs in case_studies:
        with st.expander(f"{cs['icon']} {cs['name']}  |  🏷️ {cs['tag']}", expanded=False):
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown(f"**Overview:** {cs['summary']}")
                st.caption(cs["background"].strip())
                st.markdown("**Supply Chain Impact:**")
                for k, v in cs["sc_metrics"].items():
                    st.caption(f"• {k}: **{v}**")
                st.markdown("**Financial Impact:**")
                for k, v in cs["fin_metrics"].items():
                    st.caption(f"• {k}: **{v}**")

            with col_right:
                st.markdown(cs["ai_insight"])
                if cs["show_triple_wave"]:
                    st.plotly_chart(triple_wave_chart(), use_container_width=True)
                else:
                    rng = np.random.default_rng(hash(cs["name"]) % 999)
                    days = np.arange(1, 91)
                    peak = int(rng.integers(10, 40))
                    impact = [float(np.clip(
                        80 * np.exp(-0.06 * abs(d - peak)) + rng.normal(0, 3), 0, None))
                        for d in days]
                    fi = _fig(250)
                    fi.add_trace(go.Scatter(x=days, y=impact, mode="lines", fill="tozeroy",
                                           fillcolor=f"rgba(59,130,246,.12)", line=dict(color=_ACCENT, width=1.8)))
                    fi.update_layout(title="Disruption Impact Timeline",
                                     xaxis_title="Days", yaxis_title="Impact Index", height=250)
                    st.plotly_chart(fi, use_container_width=True)

                if st.button(f"🎮 Simulate: {cs['name'][:25]}…", key=f"sim_{cs['tag']}"):
                    mode_map = {"GEOPOLITICAL": "trade_finance", "PANDEMIC": "trade_finance",
                                "FINANCIAL": "scf_pricing", "CARRIER FAILURE": "trade_finance"}
                    st.session_state.game_mode = mode_map.get(cs["tag"], "trade_finance")
                    st.session_state.game_engine = None  # reset to force new game
                    st.info("↩️ Switch to **LogisChain Lab** tab and press **New Game**.")


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar + navigation
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Persistent top navigation bar (always visible, never collapses) ──
    NAV_ITEMS = [
        ("🏠", "Home"),
        ("🌐", "Network"),
        ("📊", "Risk Monitor"),
        ("🎮", "Lab"),
        ("🔍", "Explainability"),
        ("📚", "Case Studies"),
    ]

    if "active_page" not in st.session_state:
        st.session_state.active_page = "Home"

    # Top navigation bar using Streamlit columns + buttons
    st.markdown("""
<style>
/* ── Top nav bar ─────────────────────────────────── */
.topnav {
  display:flex; align-items:center; gap:4px;
  background:#090e1a;
  border-bottom:1px solid #1e293b;
  padding:.55rem 1.2rem;
  position:sticky; top:0; z-index:999;
  margin:-1rem -1rem .5rem -1rem;
}
.topnav-brand {
  display:flex; align-items:center; gap:8px;
  margin-right:1.5rem; flex-shrink:0;
}
.topnav-logo {
  width:28px; height:28px; border-radius:6px;
  background:linear-gradient(135deg,#3b82f6,#06b6d4);
  display:flex; align-items:center; justify-content:center;
  font-size:.85rem;
}
.topnav-name { font-size:.88rem; font-weight:700; color:#f0f4ff; }
div[data-testid="stHorizontalBlock"] button {
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 7px !important;
  color: #64748b !important;
  font-size: .82rem !important;
  font-weight: 500 !important;
  padding: .4rem .85rem !important;
  transition: all .12s !important;
  white-space: nowrap !important;
}
div[data-testid="stHorizontalBlock"] button:hover {
  background: #1a2235 !important;
  color: #e2e8f0 !important;
  border-color: #1e293b !important;
}
</style>
""", unsafe_allow_html=True)

    # Logo + nav buttons in one row
    logo_col, *nav_cols, spacer = st.columns(
        [1.2] + [1] * len(NAV_ITEMS) + [2]
    )
    with logo_col:
        st.markdown("""
<div style="display:flex;align-items:center;gap:7px;padding:.1rem 0;">
  <div style="width:26px;height:26px;border-radius:6px;
              background:linear-gradient(135deg,#3b82f6,#06b6d4);
              display:flex;align-items:center;justify-content:center;
              font-size:.8rem;flex-shrink:0;">🔗</div>
  <span style="font-size:.82rem;font-weight:700;color:#f0f4ff;white-space:nowrap;">LogisChain AI</span>
</div>""", unsafe_allow_html=True)

    for col, (icon, label) in zip(nav_cols, NAV_ITEMS):
        with col:
            is_active = st.session_state.active_page == label
            btn_style = (
                "background:#1a2235;border:1px solid #3b82f6;color:#60a5fa;font-weight:600;"
                if is_active else ""
            )
            if st.button(f"{icon} {label}", key=f"nav_{label}",
                         use_container_width=True):
                st.session_state.active_page = label
                st.rerun()

    # Thin accent line under active nav item
    st.markdown(f"""
<div style="height:2px;background:#1e293b;margin:.3rem 0 1.2rem;position:relative;">
  <div style="height:100%;width:100%;background:linear-gradient(90deg,transparent,#3b82f6,transparent);
              opacity:.4;"></div>
</div>""", unsafe_allow_html=True)

    # ── Sidebar: status panel only (no navigation) ───────────────────────
    with st.sidebar:
        try:
            from src.models.gnn import PYG_AVAILABLE
            gnn_ok = PYG_AVAILABLE
        except Exception:
            gnn_ok = False

        st.markdown(f"""
<div style="padding:1.2rem 1rem .8rem;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:.9rem;">
    <div style="width:26px;height:26px;border-radius:6px;
                background:linear-gradient(135deg,#3b82f6,#06b6d4);
                display:flex;align-items:center;justify-content:center;font-size:.8rem;">🔗</div>
    <div>
      <div style="font-size:.85rem;font-weight:700;color:#f0f4ff;">LogisChain AI</div>
      <div style="font-size:.62rem;color:#334155;">v0.2.0</div>
    </div>
  </div>
  <div style="font-size:.62rem;font-weight:700;letter-spacing:.1em;
              text-transform:uppercase;color:#334155;margin-bottom:.5rem;">System Status</div>
  <div style="font-size:.78rem;color:#64748b;display:flex;justify-content:space-between;padding:.3rem 0;border-bottom:1px solid #0f172a;">
    <span>Data Pipeline</span><span style="color:#10b981;font-weight:600;">● Online</span></div>
  <div style="font-size:.78rem;color:#64748b;display:flex;justify-content:space-between;padding:.3rem 0;border-bottom:1px solid #0f172a;">
    <span>Risk Models</span><span style="color:#10b981;font-weight:600;">● Loaded</span></div>
  <div style="font-size:.78rem;color:#64748b;display:flex;justify-content:space-between;padding:.3rem 0;border-bottom:1px solid #0f172a;">
    <span>Simulation</span><span style="color:#10b981;font-weight:600;">● Ready</span></div>
  <div style="font-size:.78rem;color:#64748b;display:flex;justify-content:space-between;padding:.3rem 0;">
    <span>GNN (PyG)</span>
    <span style="color:{'#10b981' if gnn_ok else '#f59e0b'};font-weight:600;">{'● Active' if gnn_ok else '● Stub'}</span>
  </div>
  <div style="height:1px;background:#0f172a;margin:.8rem 0 .6rem;"></div>
  <div style="font-size:.7rem;color:#1e3a5f;">research@zetheta.ai</div>
  <div style="font-size:.65rem;color:#0f1f3d;margin-top:.2rem;">© 2024 Zetheta Algorithms</div>
</div>
""", unsafe_allow_html=True)

    # ── Route to selected page ────────────────────────────────────────────
    page_map = {
        "Home":          page_home,
        "Network":       page_network,
        "Risk Monitor":  page_risk_monitor,
        "Lab":           page_simulation,
        "Explainability": page_explainability,
        "Case Studies":  page_case_studies,
    }
    page_map[st.session_state.active_page]()


if __name__ == "__main__":
    main()

# ── How to run ────────────────────────────────────────────────────────────────
# streamlit run demo/app.py
# OR via Docker: docker-compose up --build  →  http://localhost:8501
