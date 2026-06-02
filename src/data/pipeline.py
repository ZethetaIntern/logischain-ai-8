"""Comprehensive synthetic data generation pipeline for LogisChain AI.

Generates statistically realistic supply chain and trade finance datasets
that mirror real-world distributions from global trade data.

Run as a script to produce all raw CSV files:
    python -m src.data.pipeline
"""
import math
import os
import time
import logging
import random
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

try:
    from faker import Faker
    _fake = Faker(["en_US", "zh_CN", "de_DE", "ja_JP", "ko_KR"])
    Faker.seed(42)
except ImportError:
    _fake = None

logger = logging.getLogger(__name__)
random.seed(42)

# ─── Global lookup tables ────────────────────────────────────────────────────

COUNTRIES = [
    "CN", "US", "DE", "JP", "IN", "KR", "VN", "MX",
    "TH", "BR", "TW", "NL", "GB", "FR", "IT", "ID",
    "MY", "SG", "PL", "TR", "BD", "PK", "AU", "CA", "ZA",
]
_CW = [0.25, 0.12, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04,
       0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02,
       0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
COUNTRY_WEIGHTS = [w / sum(_CW) for w in _CW]

COUNTRY_RISK: Dict[str, float] = {
    "CN": 0.45, "US": 0.15, "DE": 0.10, "JP": 0.15, "IN": 0.40,
    "KR": 0.25, "VN": 0.45, "MX": 0.50, "TH": 0.40, "BR": 0.55,
    "TW": 0.30, "NL": 0.08, "GB": 0.12, "FR": 0.12, "IT": 0.18,
    "ID": 0.50, "MY": 0.35, "SG": 0.08, "PL": 0.20, "TR": 0.60,
    "BD": 0.65, "PK": 0.70, "AU": 0.08, "CA": 0.10, "ZA": 0.55,
}
NATURAL_DISASTER_RISK: Dict[str, float] = {
    "JP": 0.85, "ID": 0.80, "BD": 0.75, "TW": 0.70, "IN": 0.65,
    "VN": 0.65, "TH": 0.60, "MX": 0.55, "CN": 0.55, "BR": 0.50,
    "US": 0.45, "AU": 0.45, "TR": 0.40, "MY": 0.40, "KR": 0.30,
    "CA": 0.30, "ZA": 0.35, "SG": 0.20, "NL": 0.20, "IT": 0.35,
    "DE": 0.15, "GB": 0.15, "FR": 0.15, "PL": 0.15, "PK": 0.58,
}
GEOPOLITICAL_RISK: Dict[str, float] = {
    "US": 0.15, "DE": 0.10, "JP": 0.18, "AU": 0.10, "CA": 0.10,
    "GB": 0.14, "NL": 0.10, "SG": 0.10, "KR": 0.28, "FR": 0.14,
    "CN": 0.48, "TW": 0.50, "IN": 0.38, "VN": 0.42, "TH": 0.35,
    "MX": 0.48, "BR": 0.52, "ID": 0.42, "MY": 0.32, "IT": 0.20,
    "PL": 0.30, "TR": 0.62, "ZA": 0.55, "BD": 0.60, "PK": 0.72,
}
PORT_PROXIMITY: Dict[str, float] = {
    "NL": 0.98, "SG": 0.98, "GB": 0.90, "JP": 0.90, "DE": 0.85,
    "KR": 0.88, "TW": 0.85, "MY": 0.85, "FR": 0.75, "IT": 0.75,
    "CN": 0.80, "US": 0.70, "VN": 0.68, "TH": 0.72, "BR": 0.60,
    "MX": 0.65, "ID": 0.70, "TR": 0.65, "AU": 0.62, "IN": 0.60,
    "CA": 0.55, "PL": 0.65, "BD": 0.50, "PK": 0.45, "ZA": 0.60,
}

INDUSTRIES = [
    "Electronics", "Automotive", "Pharmaceuticals", "Apparel",
    "Food & Beverage", "Chemicals", "Machinery", "Metals",
    "Semiconductors", "Consumer Goods",
]
INDUSTRY_WEIGHTS = [0.18, 0.12, 0.10, 0.12, 0.10, 0.08, 0.10, 0.08, 0.07, 0.05]

PORT_META: Dict[str, Dict] = {
    "LA":        {"lat": 33.74, "lon": -118.26, "base_teu": 10_000},
    "Rotterdam": {"lat": 51.92, "lon":    4.48, "base_teu":  8_000},
    "Singapore": {"lat":  1.26, "lon":  103.82, "base_teu":  9_500},
    "Shanghai":  {"lat": 31.24, "lon":  121.50, "base_teu": 14_000},
    "Hamburg":   {"lat": 53.55, "lon":    9.97, "base_teu":  5_500},
    "Busan":     {"lat": 35.10, "lon":  129.04, "base_teu":  7_500},
}

# Waypoints (lat, lon) for major shipping lanes
LANE_WAYPOINTS: Dict[str, List[Tuple[float, float]]] = {
    "Shanghai-LA": [
        (31.24, 121.50), (35.0, 140.0), (42.0, 160.0),
        (45.0, -175.0), (38.0, -155.0), (33.74, -118.26),
    ],
    "Shanghai-Rotterdam": [
        (31.24, 121.50), (22.0, 114.0), (1.26, 103.82),
        (12.5, 44.5), (30.0, 32.5), (36.5, 10.0),
        (43.0, -2.0), (51.92, 4.48),
    ],
    "LA-Rotterdam": [
        (33.74, -118.26), (20.0, -85.0), (9.0, -79.5),
        (20.0, -65.0), (38.0, -40.0), (45.0, -12.0), (51.92, 4.48),
    ],
    "Singapore-Rotterdam": [
        (1.26, 103.82), (5.0, 80.0), (12.5, 44.5),
        (30.0, 32.5), (36.5, 10.0), (51.92, 4.48),
    ],
    "Rotterdam-LA": [
        (51.92, 4.48), (45.0, -12.0), (38.0, -40.0),
        (20.0, -65.0), (9.0, -79.5), (20.0, -85.0), (33.74, -118.26),
    ],
}
LANE_DAYS: Dict[str, int] = {
    "Shanghai-LA": 14, "Shanghai-Rotterdam": 28,
    "LA-Rotterdam": 20, "Singapore-Rotterdam": 22,
    "Rotterdam-LA": 20,
}
DEST_PORTS: Dict[str, str] = {
    "Shanghai-LA": "LA", "Shanghai-Rotterdam": "Rotterdam",
    "LA-Rotterdam": "Rotterdam", "Singapore-Rotterdam": "Rotterdam",
    "Rotterdam-LA": "LA",
}

CREDIT_RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
RATING_WEIGHTS = [0.04, 0.08, 0.15, 0.28, 0.22, 0.15, 0.08]
RATING_RISK: Dict[str, float] = {
    "AAA": 0.05, "AA": 0.08, "A": 0.12, "BBB": 0.20,
    "BB": 0.35, "B": 0.55, "CCC": 0.80,
}

HS_CODES = [
    "8471", "8542", "8708", "8703", "8544", "2710",
    "3004", "6203", "6204", "9403", "8516", "8528",
    "9999", "2601", "7201",
]


# ─── 1. Supply Chain Network Generator ──────────────────────────────────────

class SupplyChainNetworkGenerator:
    """Generates supplier nodes and supply-chain edges with realistic distributions.

    Node features mirror S&P Global Panjiva and D&B supplier risk databases.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def generate_suppliers(self, n: int = 200) -> pd.DataFrame:
        """Generate n supplier nodes with 28 realistic features.

        Key distributions:
        - OTIF:         Beta(18, 2)  → mean 90%, std ~7%
        - Lead time:    LogNormal(2.5, 0.5) → mean ~15 days
        - Revenue:      LogNormal(17, 2) → wide corporate range
        - Current ratio:LogNormal(0.5, 0.3) → mean ~1.7
        - Interest cov: LogNormal(1.5, 0.8) → mean ~5.7
        """
        rng = self.rng
        countries = rng.choice(COUNTRIES, n, p=COUNTRY_WEIGHTS)
        industries = rng.choice(INDUSTRIES, n, p=INDUSTRY_WEIGHTS)

        # Core financial ratios
        revenue = rng.lognormal(17.0, 2.0, n)
        ebitda_margin = rng.beta(3, 7, n) * 0.50        # 0–50 %, mean ~15 %
        current_ratio = np.clip(rng.lognormal(0.50, 0.30, n), 0.3, 6.0)
        quick_ratio = current_ratio * rng.uniform(0.55, 0.85, n)
        debt_equity = np.clip(rng.lognormal(0.30, 0.60, n), 0.0, 12.0)
        interest_coverage = np.clip(rng.lognormal(1.50, 0.80, n), 0.5, 25.0)

        # Working capital – ensure DIO + DSO - DPO ≈ CCC
        dso = np.clip(rng.lognormal(3.6, 0.4, n), 10.0, 120.0)
        dpo = np.clip(rng.lognormal(3.4, 0.4, n), 10.0, 90.0)
        dio = np.clip(rng.lognormal(3.7, 0.5, n), 5.0, 180.0)
        ccc = dso + dio - dpo

        # Operational metrics
        otif_rate = np.clip(rng.beta(18, 2, n), 0.50, 1.00)
        fill_rate = np.clip(rng.beta(20, 2, n), 0.55, 1.00)
        lead_time_mean = np.clip(rng.lognormal(2.5, 0.5, n), 1.0, 90.0)
        lead_time_std = rng.uniform(0.5, lead_time_mean * 0.4)
        inventory_turnover = np.clip(rng.lognormal(2.0, 0.5, n), 1.0, 20.0)
        freight_cost_ratio = np.clip(rng.beta(2, 20, n), 0.005, 0.25)
        capacity_utilization = np.clip(rng.beta(7, 3, n), 0.30, 0.99)

        # HHI (Herfindahl-Hirschman) – 0-10000 scale, >2500 = concentrated
        supplier_hhi = np.clip(rng.beta(2, 5, n) * 10_000, 0, 10_000)
        customer_hhi = np.clip(rng.beta(2, 5, n) * 10_000, 0, 10_000)

        # Country-level risk (lookup + individual noise)
        cr_base = np.array([COUNTRY_RISK[c] for c in countries])
        nd_base = np.array([NATURAL_DISASTER_RISK.get(c, 0.3) for c in countries])
        gr_base = np.array([GEOPOLITICAL_RISK.get(c, 0.3) for c in countries])
        pp_base = np.array([PORT_PROXIMITY.get(c, 0.5) for c in countries])

        country_risk_score = np.clip(cr_base + rng.normal(0, 0.05, n), 0.0, 1.0)
        natural_disaster_exposure = np.clip(nd_base + rng.normal(0, 0.05, n), 0.0, 1.0)
        geopolitical_risk = np.clip(gr_base + rng.normal(0, 0.05, n), 0.0, 1.0)
        port_proximity_score = np.clip(pp_base + rng.normal(0, 0.03, n), 0.0, 1.0)

        # Synthetic network centrality (will be recomputed after graph build)
        pagerank_raw = rng.lognormal(-3, 1.5, n)
        pagerank = pagerank_raw / pagerank_raw.sum()
        betweenness_centrality = np.clip(rng.exponential(0.03, n), 0.0, 1.0)
        clustering_coeff = np.clip(rng.beta(2, 5, n), 0.0, 1.0)

        # Generate names via Faker (or fallback)
        if _fake is not None:
            names = [_fake.company() for _ in range(n)]
        else:
            names = [f"Supplier-{i:04d}" for i in range(n)]

        df = pd.DataFrame(
            {
                "supplier_id":                [f"SUP-{i:04d}" for i in range(n)],
                "name":                        names,
                "country":                     countries,
                "industry":                    industries,
                "revenue_usd":                 revenue,
                "ebitda_margin":               ebitda_margin,
                "current_ratio":               current_ratio,
                "quick_ratio":                 quick_ratio,
                "debt_equity":                 debt_equity,
                "interest_coverage":           interest_coverage,
                "otif_rate":                   otif_rate,
                "lead_time_mean":              lead_time_mean,
                "lead_time_std":               lead_time_std,
                "inventory_turnover":          inventory_turnover,
                "supplier_concentration_hhi":  supplier_hhi,
                "customer_concentration_hhi":  customer_hhi,
                "dso":                         dso,
                "dpo":                         dpo,
                "dio":                         dio,
                "cash_conversion_cycle":       ccc,
                "fill_rate":                   fill_rate,
                "freight_cost_ratio":          freight_cost_ratio,
                "capacity_utilization":        capacity_utilization,
                "betweenness_centrality":      betweenness_centrality,
                "clustering_coeff":            clustering_coeff,
                "pagerank":                    pagerank,
                "country_risk_score":          country_risk_score,
                "natural_disaster_exposure":   natural_disaster_exposure,
                "geopolitical_risk":           geopolitical_risk,
                "port_proximity_score":        port_proximity_score,
            }
        )
        logger.info(f"Generated {n} supplier nodes.")
        return df

    def generate_edges(
        self, suppliers_df: pd.DataFrame, n_edges: int = 2000
    ) -> pd.DataFrame:
        """Generate supply chain edges between supplier nodes.

        Edge types (with probabilities):
        - supplies (60 %): A directly supplies goods to B
        - ships_via (20 %): A ships through logistics hub B
        - finances (15 %): A provides trade finance to B
        - owns (5 %): A is parent entity of B (corporate ownership)

        Modal types: ocean (50 %), road (25 %), air (15 %), rail (10 %)
        """
        rng = self.rng
        ids = suppliers_df["supplier_id"].tolist()
        n_suppliers = len(ids)

        edge_types = rng.choice(
            ["supplies", "ships_via", "finances", "owns"],
            n_edges,
            p=[0.60, 0.20, 0.15, 0.05],
        )
        modal_types = rng.choice(
            ["ocean", "road", "air", "rail"],
            n_edges,
            p=[0.50, 0.25, 0.15, 0.10],
        )
        payment_terms = rng.choice([30, 45, 60, 90, 120], n_edges, p=[0.10, 0.15, 0.35, 0.30, 0.10])

        src_idx = rng.integers(0, n_suppliers, n_edges)
        dst_idx = rng.integers(0, n_suppliers, n_edges)
        # Prevent self-loops
        same = src_idx == dst_idx
        dst_idx[same] = (dst_idx[same] + 1) % n_suppliers

        # Trade volume correlated with source revenue
        src_rev = suppliers_df["revenue_usd"].values[src_idx]
        volume_usd = np.clip(
            rng.lognormal(0, 0.5, n_edges) * src_rev * 0.05,
            1_000, 5e9
        )

        # Transit time depends on modal type
        transit_base = {"ocean": 14, "air": 3, "road": 5, "rail": 10}
        transit_time = np.array([
            max(1, int(rng.lognormal(math.log(transit_base[m]), 0.4)))
            for m in modal_types
        ])

        reliability_score = np.clip(rng.beta(8, 2, n_edges), 0.30, 1.00)

        df = pd.DataFrame(
            {
                "source_id":         [ids[i] for i in src_idx],
                "target_id":         [ids[i] for i in dst_idx],
                "edge_type":         edge_types,
                "volume_usd":        volume_usd,
                "payment_terms_days": payment_terms,
                "reliability_score": reliability_score,
                "transit_time_days": transit_time,
                "modal_type":        modal_types,
            }
        )
        logger.info(f"Generated {n_edges} supply chain edges.")
        return df


# ─── 2. Time Series Generator ────────────────────────────────────────────────

class TimeSeriesGenerator:
    """Generates realistic multi-variate supply chain time series.

    Implements:
    - Ornstein-Uhlenbeck mean-reverting processes for freight rates
    - GARCH-like volatility clustering
    - Seasonal decomposition (trend + Fourier seasonality + noise)
    - Stochastic shock events for port disruptions
    - COVID-19-style demand/freight spike regime (mid-2020 to 2021)
    """

    CNY_WINDOWS = {
        2019: ("2019-02-05", "2019-02-19"),
        2020: ("2020-01-25", "2020-02-08"),
        2021: ("2021-02-12", "2021-02-26"),
        2022: ("2022-02-01", "2022-02-15"),
        2023: ("2023-01-22", "2023-02-05"),
        2024: ("2024-02-10", "2024-02-24"),
    }

    def __init__(self, seed: int = 42, start_date: str = "2020-01-01"):
        self.rng = np.random.default_rng(seed)
        self.start_date = pd.Timestamp(start_date)

    def _date_range(self, days: int) -> pd.DatetimeIndex:
        return pd.date_range(self.start_date, periods=days, freq="D")

    def _seasonal_component(
        self, n: int, amplitude: float = 0.08, phase_shift: float = 0.0
    ) -> np.ndarray:
        """Annual + weekly Fourier seasonality."""
        t = np.arange(n)
        annual = amplitude * np.sin(2 * np.pi * t / 365 + phase_shift)
        weekly = 0.02 * np.sin(2 * np.pi * t / 7)
        return annual + weekly

    def _insert_shocks(
        self,
        series: np.ndarray,
        n_shocks: int,
        magnitude_lo: float = 0.20,
        magnitude_hi: float = 0.45,
        duration_lo: int = 7,
        duration_hi: int = 30,
    ) -> Tuple[np.ndarray, List[dict]]:
        """Inject sudden shock events (sharp declines lasting 7-30 days)."""
        rng = self.rng
        series = series.copy()
        shock_log = []
        n = len(series)
        starts = rng.integers(30, n - duration_hi - 1, n_shocks)
        for s in starts:
            dur = int(rng.integers(duration_lo, duration_hi + 1))
            drop = float(rng.uniform(magnitude_lo, magnitude_hi))
            recovery = np.linspace(1 - drop, 1.0, dur + 1)[1:]
            end = min(s + dur, n)
            actual_dur = end - s
            series[s:end] *= recovery[:actual_dur]
            shock_log.append({"start": int(s), "duration": dur, "magnitude": round(drop, 3)})
        return series, shock_log

    def _ou_process(
        self,
        n: int,
        theta: float,
        mu: float,
        sigma: float,
        x0: Optional[float] = None,
        floor: float = 50.0,
    ) -> np.ndarray:
        """Ornstein-Uhlenbeck mean-reverting process."""
        x0 = x0 if x0 is not None else mu
        x = np.zeros(n)
        x[0] = x0
        eps = self.rng.standard_normal(n)
        for t in range(1, n):
            dx = theta * (mu - x[t - 1]) + sigma * eps[t]
            x[t] = max(x[t - 1] + dx, floor)
        return x

    def generate_port_throughput(
        self,
        ports: Optional[List[str]] = None,
        days: int = 1095,
    ) -> pd.DataFrame:
        """Generate daily TEU throughput per port with trend, seasonality, and disruption shocks.

        Base volumes from real-world estimates:
        - LA: ~10,000 TEU/day   Rotterdam: ~8,000   Singapore: ~9,500
        Trend: +0.4 %/month (global container trade growth ~5 % p.a.)
        Seasonality: peak Aug–Oct (Q4 preparation), trough Jan–Feb.
        Shocks: 3-5 events per year, duration 7-30 days, magnitude 20-45 %.
        """
        rng = self.rng
        ports = ports or ["LA", "Rotterdam", "Singapore"]
        dates = self._date_range(days)
        t = np.arange(days)
        n_shocks_per_year = 4

        records = []
        for port in tqdm(ports, desc="Port throughput", leave=False):
            base = PORT_META.get(port, {}).get("base_teu", 8_000)
            trend = base * (1 + 0.004 / 30) ** t
            seasonal = self._seasonal_component(days, amplitude=0.12, phase_shift=2.0)
            noise = rng.normal(0, 0.04, days)
            raw = trend * (1 + seasonal + noise)
            # Weekends -15 % throughput
            is_weekend = dates.dayofweek >= 5
            raw[is_weekend] *= 0.85

            n_shocks = int(n_shocks_per_year * days / 365)
            raw, shock_log = self._insert_shocks(raw, n_shocks)
            raw = np.clip(raw, 0, None)

            # Build shock flag
            shock_flag = np.zeros(days, dtype=int)
            for sh in shock_log:
                shock_flag[sh["start"]: sh["start"] + sh["duration"]] = 1

            for i, d in enumerate(dates):
                records.append(
                    {
                        "port": port,
                        "date": d.date(),
                        "teu_day": round(float(raw[i]), 1),
                        "trend_component": round(float(trend[i]), 1),
                        "seasonal_component": round(float(seasonal[i]), 4),
                        "is_shock_day": int(shock_flag[i]),
                        "is_weekend": int(is_weekend[i]),
                    }
                )

        df = pd.DataFrame(records)
        logger.info(f"Port throughput: {len(df):,} rows for {len(ports)} ports over {days} days.")
        return df

    def generate_freight_rates(
        self,
        lanes: Optional[List[str]] = None,
        days: int = 1095,
    ) -> pd.DataFrame:
        """Generate SCFI-like freight rate index per lane.

        Process: Ornstein-Uhlenbeck with regime switching + GARCH volatility.
        Regimes (from 2020-01-01):
        - Normal     (days 0–180):    mu=1000,  sigma=50,   theta=0.04
        - Stress     (days 181–360):  mu=2500,  sigma=200,  theta=0.03
        - COVID-peak (days 361–730):  mu=7500,  sigma=500,  theta=0.02
        - Recovery   (days 731–1094): mu=2500,  sigma=150,  theta=0.04
        """
        rng = self.rng
        lanes = lanes or ["Shanghai-LA", "Shanghai-Rotterdam", "LA-Rotterdam"]
        dates = self._date_range(days)

        # Regime schedule (day → regime params)
        def _regime(t: int) -> Tuple[float, float, float]:
            if t < 181:
                return 1000.0, 50.0, 0.04
            elif t < 361:
                return 2500.0, 200.0, 0.03
            elif t < 731:
                return 7500.0, 500.0, 0.02
            else:
                return 2500.0, 150.0, 0.04

        records = []
        for lane in tqdm(lanes, desc="Freight rates", leave=False):
            # GARCH(1,1) volatility
            omega, alpha_g, beta_g = 25.0, 0.08, 0.88
            vol = 50.0
            rate = 1000.0
            eps_arr = rng.standard_normal(days)

            rates, vols = [], []
            for t in range(days):
                mu_r, sigma_base, theta = _regime(t)
                vol = math.sqrt(max(omega + alpha_g * (eps_arr[t - 1] * vol) ** 2 + beta_g * vol ** 2, 1.0))
                vol = min(vol, sigma_base * 3)
                dr = theta * (mu_r - rate) + vol * eps_arr[t]
                rate = max(rate + dr, 50.0)
                rates.append(rate)
                vols.append(vol)

            rates_arr = np.array(rates)
            vols_arr = np.array(vols)
            # Translate index to USD/TEU (×1 for Shanghai-LA, ×0.85 for others)
            lane_factor = 1.0 if "Shanghai-LA" in lane else 0.88
            rate_usd = rates_arr * lane_factor

            # Percentile rank within this lane's history
            percentiles = np.array([np.mean(rates_arr[:max(1, i + 1)] <= rates_arr[i]) for i in range(days)])

            for i, d in enumerate(dates):
                records.append(
                    {
                        "lane": lane,
                        "date": d.date(),
                        "rate_index": round(float(rates_arr[i]), 2),
                        "rate_usd_per_teu": round(float(rate_usd[i]), 2),
                        "daily_volatility": round(float(vols_arr[i]), 2),
                        "rate_percentile_rank": round(float(percentiles[i]), 4),
                        "regime": (
                            "normal" if i < 181
                            else "stress" if i < 361
                            else "covid_peak" if i < 731
                            else "recovery"
                        ),
                    }
                )

        df = pd.DataFrame(records)
        logger.info(f"Freight rates: {len(df):,} rows for {len(lanes)} lanes.")
        return df

    def generate_vessel_positions(
        self, n_vessels: int = 500, days: int = 365
    ) -> pd.DataFrame:
        """Generate daily vessel lat/lon positions along realistic shipping lanes.

        Each vessel is assigned to a lane. Position is interpolated along
        the lane waypoints based on voyage progress + small positional noise.
        """
        rng = self.rng
        dates = self._date_range(days)
        lane_names = list(LANE_WAYPOINTS.keys())

        # Assign each vessel to a lane
        vessel_lanes = rng.choice(lane_names, n_vessels)
        vessel_speeds = np.clip(rng.normal(14.0, 2.0, n_vessels), 8.0, 22.0)
        # Voyage cycle: total_days for this lane
        vessel_cycle = np.array([LANE_DAYS[l] for l in vessel_lanes])
        # Random phase offset (where in the voyage each vessel starts)
        vessel_phase = rng.uniform(0, 1, n_vessels)

        records = []
        for day_idx in tqdm(range(days), desc="Vessel positions", leave=False):
            for v_idx in range(n_vessels):
                lane = vessel_lanes[v_idx]
                waypoints = LANE_WAYPOINTS[lane]
                cycle = vessel_cycle[v_idx]
                voyage_progress = (vessel_phase[v_idx] + day_idx / cycle) % 1.0

                # Interpolate along waypoints
                n_wp = len(waypoints)
                seg = voyage_progress * (n_wp - 1)
                wp_lo = int(seg)
                wp_hi = min(wp_lo + 1, n_wp - 1)
                frac = seg - wp_lo
                lat = waypoints[wp_lo][0] + frac * (waypoints[wp_hi][0] - waypoints[wp_lo][0])
                lon = waypoints[wp_lo][1] + frac * (waypoints[wp_hi][1] - waypoints[wp_lo][1])

                # Add positional noise (±0.5°)
                lat += rng.uniform(-0.5, 0.5)
                lon += rng.uniform(-0.5, 0.5)

                # Cargo load varies through voyage (full departure, lighter near arrival)
                cargo_load = max(0.0, 0.95 - 0.15 * voyage_progress + rng.normal(0, 0.05))
                cargo_load = float(np.clip(cargo_load, 0.0, 1.0))

                speed = float(vessel_speeds[v_idx]) * float(rng.uniform(0.85, 1.10))
                on_time = int(rng.random() < 0.88)

                records.append(
                    {
                        "vessel_id": f"VES-{v_idx:04d}",
                        "date": dates[day_idx].date(),
                        "lane": lane,
                        "latitude": round(lat, 4),
                        "longitude": round(lon, 4),
                        "speed_knots": round(speed, 1),
                        "destination_port": DEST_PORTS[lane],
                        "cargo_load_pct": round(cargo_load * 100, 1),
                        "voyage_progress_pct": round(voyage_progress * 100, 1),
                        "on_time_flag": on_time,
                    }
                )

        df = pd.DataFrame(records)
        logger.info(f"Vessel positions: {len(df):,} rows ({n_vessels} vessels × {days} days).")
        return df

    def generate_inventory_levels(
        self, n_companies: int = 100, days: int = 730
    ) -> pd.DataFrame:
        """Generate daily inventory levels with sawtooth restock cycles, safety stock, and stockouts."""
        rng = self.rng
        dates = self._date_range(days)
        records = []

        for c in tqdm(range(n_companies), desc="Inventory levels", leave=False):
            reorder_point = float(rng.lognormal(5, 0.8))
            safety_stock = reorder_point * float(rng.uniform(0.10, 0.25))
            max_stock = reorder_point * float(rng.uniform(2.5, 5.0))
            order_qty = reorder_point * float(rng.uniform(1.5, 3.0))
            # Seasonal demand pattern
            seasonal_amp = float(rng.uniform(0.05, 0.25))
            daily_demand_base = reorder_point / float(rng.uniform(15, 45))

            inventory = max_stock * float(rng.uniform(0.5, 1.0))
            for d in range(days):
                t = d
                seasonal = 1 + seasonal_amp * math.sin(2 * math.pi * t / 365 + 1.5)
                demand = max(0.0, daily_demand_base * seasonal * float(rng.lognormal(0, 0.15)))
                # Restock event
                if inventory <= reorder_point:
                    lead = int(rng.integers(3, 21))
                    if (d + lead) < days:
                        inventory += order_qty
                inventory -= min(demand, inventory)
                stockout = inventory <= safety_stock

                records.append(
                    {
                        "company_id": f"CO-{c:04d}",
                        "date": dates[d].date(),
                        "inventory_units": round(inventory, 2),
                        "reorder_point": round(reorder_point, 2),
                        "safety_stock": round(safety_stock, 2),
                        "daily_demand_units": round(demand, 2),
                        "stockout_flag": int(stockout),
                    }
                )

        df = pd.DataFrame(records)
        logger.info(f"Inventory levels: {len(df):,} rows.")
        return df

    def generate_demand_signals(
        self, n_skus: int = 1000, days: int = 730
    ) -> pd.DataFrame:
        """Generate SKU-level demand signals with trend, seasonality, bullwhip amplification.

        Bullwhip effect: upstream amplification factor ~1.5–3× vs final consumer demand.
        Promotions: random events with 20-80% demand uplift for 3-10 days.
        """
        rng = self.rng
        dates = self._date_range(days)
        t = np.arange(days)

        records = []
        for s in tqdm(range(n_skus), desc="Demand signals", leave=False):
            base_demand = float(rng.lognormal(4, 1.5))
            trend = float(rng.normal(0.0003, 0.0002))
            seasonal_amp = float(rng.uniform(0.05, 0.35))
            phase = float(rng.uniform(0, 2 * math.pi))
            bullwhip_factor = float(rng.uniform(1.2, 3.0))
            noise_sigma = float(rng.uniform(0.05, 0.20))

            # Promotion schedule (3-8 events)
            n_promo = int(rng.integers(3, 9))
            promo_starts = rng.integers(0, days - 10, n_promo)
            promo_durations = rng.integers(3, 11, n_promo)
            promo_lifts = rng.uniform(0.2, 0.8, n_promo)
            promo_flag = np.zeros(days, dtype=int)
            for ps, pd_, pl in zip(promo_starts, promo_durations, promo_lifts):
                promo_flag[ps: ps + pd_] = 1

            demand_arr = (
                base_demand
                * (1 + trend * t)
                * (1 + seasonal_amp * np.sin(2 * math.pi * t / 365 + phase))
                * (1 + promo_flag * rng.uniform(0.15, 0.80, days))
                * np.exp(rng.normal(0, noise_sigma, days))
            )
            demand_arr = np.clip(demand_arr, 0.0, None)
            upstream_demand = demand_arr * bullwhip_factor

            for d in range(days):
                records.append(
                    {
                        "sku_id": f"SKU-{s:05d}",
                        "date": dates[d].date(),
                        "demand_units": round(float(demand_arr[d]), 2),
                        "upstream_demand_units": round(float(upstream_demand[d]), 2),
                        "trend_component": round(float(base_demand * (1 + trend * d)), 4),
                        "seasonal_component": round(
                            float(seasonal_amp * math.sin(2 * math.pi * d / 365 + phase)), 4
                        ),
                        "promotion_flag": int(promo_flag[d]),
                        "bullwhip_factor": round(bullwhip_factor, 3),
                    }
                )

        df = pd.DataFrame(records)
        logger.info(f"Demand signals: {len(df):,} rows.")
        return df


# ─── 3. Trade Finance Data Generator ─────────────────────────────────────────

class TradefinanceDataGenerator:
    """Generates realistic trade finance instrument datasets.

    Statistical calibration targets:
    - LC default rate: ~1.8 % base (ICC Trade Register 2023)
    - SCF invoice early payment: ~65 % of invoices
    - Covenant breach rate: ~8 % of WC facilities (stressed environment)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._start = pd.Timestamp("2020-01-01")

    def _rand_date(self, n: int, lo: int = 0, hi: int = 1095) -> np.ndarray:
        offsets = self.rng.integers(lo, hi, n)
        return np.array([self._start + timedelta(days=int(o)) for o in offsets])

    def generate_lc_transactions(self, n: int = 25_000) -> pd.DataFrame:
        """Generate n Letter of Credit records with SC-adjusted default probability.

        Default model:
            PD = clip(0.018 × stress_multiplier, 0, 0.30)
        where stress_multiplier = f(port_congestion, OTIF, freight_rate_percentile,
                                    country_risk, credit_score, discrepancy_history)

        days_to_default: Weibull survival time (censored at tenor for non-defaults).
        """
        rng = self.rng

        # ── Applicant & beneficiary IDs ──
        n_applicants = 500
        n_beneficiaries = 800
        applicant_ids = [f"APP-{i:04d}" for i in rng.integers(0, n_applicants, n)]
        beneficiary_ids = [f"BEN-{i:04d}" for i in rng.integers(0, n_beneficiaries, n)]

        # ── Instrument parameters ──
        lc_amount = np.clip(rng.lognormal(12.0, 1.5, n), 5_000, 50_000_000)
        tenor_days = rng.choice([30, 60, 90, 120, 180, 365], n, p=[0.10, 0.20, 0.30, 0.20, 0.15, 0.05])
        hs_codes = rng.choice(HS_CODES, n)
        origin_countries = rng.choice(COUNTRIES, n, p=COUNTRY_WEIGHTS)
        dest_countries = rng.choice(COUNTRIES, n, p=COUNTRY_WEIGHTS)
        issue_dates = self._rand_date(n, 0, 900)
        expiry_dates = np.array([issue_dates[i] + timedelta(days=int(tenor_days[i])) for i in range(n)])

        # Port names
        port_list = list(PORT_META.keys())
        port_origin = rng.choice(port_list, n)
        port_dest = rng.choice(port_list, n)

        # ── Supply chain risk features ──
        applicant_credit_score = rng.integers(300, 850, n).astype(float)
        beneficiary_otif_score = np.clip(rng.beta(18, 2, n), 0.40, 1.00)
        hist_disc_applicant = np.clip(rng.beta(1, 15, n), 0.0, 0.40)
        hist_disc_beneficiary = np.clip(rng.beta(1, 15, n), 0.0, 0.40)
        port_congestion_origin = np.clip(rng.beta(3, 7, n), 0.0, 1.0)
        port_congestion_destination = np.clip(rng.beta(3, 7, n), 0.0, 1.0)
        freight_rate_percentile = rng.uniform(0.0, 1.0, n)
        seasonal_factor = rng.uniform(0.80, 1.30, n)
        currency_volatility = np.clip(rng.beta(2, 8, n), 0.0, 0.60)
        cr_origin = np.array([COUNTRY_RISK[c] for c in origin_countries])
        cr_dest = np.array([COUNTRY_RISK[c] for c in dest_countries])
        country_risk_differential = np.abs(cr_origin - cr_dest)

        # ── SC-adjusted default model ──
        stress = (
            0.50 * (port_congestion_origin + port_congestion_destination)
            + 0.40 * (1.0 - beneficiary_otif_score)
            + 0.30 * freight_rate_percentile
            + 0.50 * country_risk_differential
            + 0.60 * (1.0 - applicant_credit_score / 850.0)
            + 0.80 * hist_disc_applicant
            + 0.30 * currency_volatility
        )
        pd_adjusted = np.clip(0.018 * (1.0 + stress), 0.005, 0.30)
        default_flag = (rng.random(n) < pd_adjusted).astype(int)

        # ── Survival time (days_to_default) ──
        weibull_scale = tenor_days * 0.40
        weibull_shape = 1.5
        raw_ttd = rng.weibull(weibull_shape, n) * weibull_scale + 1.0
        days_to_default = np.where(
            default_flag == 1,
            np.clip(raw_ttd, 1, tenor_days).astype(int),
            tenor_days,
        )

        df = pd.DataFrame(
            {
                "lc_id":                               [f"LC-{i:07d}" for i in range(n)],
                "applicant_id":                        applicant_ids,
                "beneficiary_id":                      beneficiary_ids,
                "lc_amount_usd":                       lc_amount,
                "tenor_days":                          tenor_days,
                "commodity_hs_code":                   hs_codes,
                "origin_country":                      origin_countries,
                "destination_country":                 dest_countries,
                "issue_date":                          [d.date() for d in issue_dates],
                "expiry_date":                         [d.date() for d in expiry_dates],
                "port_origin":                         port_origin,
                "port_destination":                    port_dest,
                "applicant_credit_score":              applicant_credit_score,
                "beneficiary_otif_score":              beneficiary_otif_score,
                "historical_discrepancy_rate_applicant":   hist_disc_applicant,
                "historical_discrepancy_rate_beneficiary": hist_disc_beneficiary,
                "port_congestion_origin":              port_congestion_origin,
                "port_congestion_destination":         port_congestion_destination,
                "freight_rate_percentile":             freight_rate_percentile,
                "seasonal_factor":                     seasonal_factor,
                "currency_volatility":                 currency_volatility,
                "country_risk_differential":           country_risk_differential,
                "pd_adjusted":                         pd_adjusted,
                "default_flag":                        default_flag,
                "days_to_default":                     days_to_default,
            }
        )
        actual_dr = df["default_flag"].mean()
        logger.info(
            f"Generated {n:,} LC transactions. Actual default rate: {actual_dr:.2%}"
        )
        return df

    def generate_scf_invoices(self, n: int = 50_000) -> pd.DataFrame:
        """Generate SCF platform invoice records.

        Models early-payment dynamics: 65 % of invoices paid early via SCF,
        with discount rates reflecting anchor creditworthiness and SC risk.
        """
        rng = self.rng

        n_suppliers = 600
        n_anchors = 80

        supplier_ids = [f"SUP-{i:04d}" for i in rng.integers(0, n_suppliers, n)]
        anchor_ids = [f"ANC-{i:03d}" for i in rng.integers(0, n_anchors, n)]

        invoice_amount = np.clip(rng.lognormal(11.5, 1.2, n), 1_000, 5_000_000)
        invoice_dates = self._rand_date(n, 0, 900)
        payment_terms = rng.choice([30, 45, 60, 90], n, p=[0.15, 0.20, 0.40, 0.25])
        due_dates = np.array([invoice_dates[i] + timedelta(days=int(payment_terms[i])) for i in range(n)])

        anchor_ratings = rng.choice(CREDIT_RATINGS, n, p=RATING_WEIGHTS)
        anchor_risk = np.array([RATING_RISK[r] for r in anchor_ratings])

        sc_risk_score = np.clip(rng.beta(3, 7, n) + 0.1 * anchor_risk, 0, 1)

        # Discount rate: base rate + SC risk premium
        discount_rate_bps = np.clip(
            rng.normal(120, 40, n) + sc_risk_score * 80 + anchor_risk * 60,
            20, 600,
        )
        # Annualized discount to actual discount over early payment days
        days_early_potential = rng.integers(5, payment_terms + 1)
        discount_amount = invoice_amount * (discount_rate_bps / 10_000) * (days_early_potential / 365)

        # Payment outcome
        early_paid = rng.random(n) < (0.65 - 0.20 * sc_risk_score)
        late_paid = (~early_paid) & (rng.random(n) < (0.12 + 0.15 * sc_risk_score))
        defaulted = (~early_paid) & (~late_paid) & (rng.random(n) < (0.02 + 0.08 * anchor_risk))

        payment_status = np.where(
            early_paid, "early_paid",
            np.where(late_paid, "late", np.where(defaulted, "defaulted", "paid_on_time"))
        )

        early_payment_date = np.array([
            invoice_dates[i] + timedelta(days=int(days_early_potential[i])) if early_paid[i]
            else due_dates[i]
            for i in range(n)
        ])

        days_early_actual = np.where(early_paid, days_early_potential, 0).astype(int)
        days_past_expiry = np.where(
            late_paid,
            rng.integers(1, 61, n),
            np.zeros(n, dtype=int)
        )

        df = pd.DataFrame(
            {
                "invoice_id":            [f"INV-{i:08d}" for i in range(n)],
                "supplier_id":           supplier_ids,
                "anchor_id":             anchor_ids,
                "invoice_amount_usd":    invoice_amount,
                "invoice_date":          [d.date() for d in invoice_dates],
                "due_date":              [d.date() for d in due_dates],
                "early_payment_date":    [d.date() for d in early_payment_date],
                "payment_terms_days":    payment_terms,
                "discount_rate_bps":     discount_rate_bps,
                "discount_amount_usd":   discount_amount,
                "payment_status":        payment_status,
                "days_early":            days_early_actual,
                "days_past_expiry":      days_past_expiry,
                "supply_chain_risk_score": sc_risk_score,
                "anchor_credit_rating":  anchor_ratings,
                "financing_cost_bps":    discount_rate_bps,
            }
        )
        early_pct = early_paid.mean()
        logger.info(f"Generated {n:,} SCF invoices. Early payment rate: {early_pct:.1%}")
        return df

    def generate_working_capital_facilities(self, n: int = 500) -> pd.DataFrame:
        """Generate working capital facility records with covenant monitoring.

        Covenant breach rate target: ~8 % (stressed supply chain environment).
        """
        rng = self.rng

        n_companies = 200
        company_ids = [f"CO-{i:04d}" for i in rng.integers(0, n_companies, n)]
        facility_types = rng.choice(
            ["RCF", "Overdraft", "TermLoan", "ABL"],
            n, p=[0.40, 0.20, 0.25, 0.15],
        )

        limit_usd = np.clip(rng.lognormal(14.5, 1.5, n), 100_000, 500_000_000)
        utilization = np.clip(rng.beta(5, 3, n), 0.10, 0.99)
        drawn_usd = limit_usd * utilization
        tenor_months = rng.choice([6, 12, 24, 36, 60], n, p=[0.10, 0.25, 0.30, 0.25, 0.10])
        margin_bps = np.clip(rng.normal(280, 80, n), 75, 600).astype(int)

        # Covenant thresholds
        cov_leverage_thresh = rng.uniform(2.5, 5.0, n)
        cov_icr_thresh = rng.uniform(1.5, 3.0, n)
        cov_cr_thresh = rng.uniform(1.0, 2.0, n)

        # Actual covenant values (with supply chain stress)
        sc_disruption_index = np.clip(rng.beta(2, 6, n), 0.0, 1.0)
        actual_leverage = np.clip(
            rng.lognormal(0.6, 0.6, n) * (1 + 0.5 * sc_disruption_index), 0.1, 15.0
        )
        actual_icr = np.clip(
            rng.lognormal(1.6, 0.7, n) * (1 - 0.3 * sc_disruption_index), 0.3, 25.0
        )
        actual_cr = np.clip(
            rng.lognormal(0.5, 0.3, n) * (1 - 0.2 * sc_disruption_index), 0.2, 6.0
        )

        breach_leverage = (actual_leverage > cov_leverage_thresh).astype(int)
        breach_icr = (actual_icr < cov_icr_thresh).astype(int)
        breach_cr = (actual_cr < cov_cr_thresh).astype(int)
        covenant_breach_flag = ((breach_leverage | breach_icr | breach_cr) == 1).astype(int)

        # Days to breach estimate (survival-like)
        days_to_breach = np.where(
            covenant_breach_flag == 1,
            rng.integers(1, 91, n),
            rng.integers(90, 730, n),
        )

        review_dates = self._rand_date(n, 0, 365)
        next_review_dates = np.array([
            review_dates[i] + timedelta(days=int(rng.choice([30, 60, 90, 180])))
            for i in range(n)
        ])

        df = pd.DataFrame(
            {
                "facility_id":                  [f"FAC-{i:04d}" for i in range(n)],
                "company_id":                   company_ids,
                "facility_type":                facility_types,
                "limit_usd":                    limit_usd,
                "drawn_usd":                    drawn_usd,
                "utilization_pct":              utilization * 100,
                "tenor_months":                 tenor_months,
                "margin_bps":                   margin_bps,
                "covenant_threshold_leverage":  cov_leverage_thresh,
                "covenant_threshold_icr":       cov_icr_thresh,
                "covenant_threshold_current_ratio": cov_cr_thresh,
                "actual_leverage":              actual_leverage,
                "actual_icr":                   actual_icr,
                "actual_current_ratio":         actual_cr,
                "breach_leverage_flag":         breach_leverage,
                "breach_icr_flag":              breach_icr,
                "breach_current_ratio_flag":    breach_cr,
                "covenant_breach_flag":         covenant_breach_flag,
                "sc_disruption_index":          sc_disruption_index,
                "days_to_breach_estimate":      days_to_breach,
                "last_review_date":             [d.date() for d in review_dates],
                "next_review_date":             [d.date() for d in next_review_dates],
            }
        )
        breach_rate = df["covenant_breach_flag"].mean()
        logger.info(f"Generated {n} WC facilities. Covenant breach rate: {breach_rate:.1%}")
        return df


# ─── 4. Comtrade API Fetcher ─────────────────────────────────────────────────

class ComtradeAPIFetcher:
    """Fetches bilateral trade flows from UN Comtrade API v1 with robust retry logic.

    Falls back to a realistic synthetic bilateral trade matrix if the API
    is unavailable or the API key is missing/expired.
    """

    BASE_URL = "https://comtradeapi.un.org/data/v1/get"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("COMTRADE_API_KEY", "")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Ocp-Apim-Subscription-Key": self.api_key})

    def fetch_with_retry(
        self, url: str, params: dict, max_retries: int = 3
    ) -> dict:
        """Fetch with exponential back-off. Returns parsed JSON or {} on failure."""
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {wait}s.")
                time.sleep(wait)
        logger.error(f"All {max_retries} attempts failed for {url}.")
        return {}

    def fetch_bilateral_trade_flows(
        self,
        reporters: List[str] = ("842", "156"),
        partners: List[str] = ("0",),
        years: List[int] = (2022, 2023),
        commodities: List[str] = ("TOTAL",),
    ) -> pd.DataFrame:
        """Fetch annual bilateral trade flows. Falls back to synthetic if API fails."""
        if not self.api_key:
            logger.info("No COMTRADE_API_KEY — generating synthetic bilateral trade matrix.")
            return self._synthetic_bilateral_matrix(list(reporters), list(partners), list(years))

        frames = []
        for reporter in reporters:
            for year in years:
                for commodity in commodities:
                    params = {
                        "reporterCode": reporter,
                        "partnerCode": ",".join(partners),
                        "cmdCode": commodity,
                        "period": str(year),
                        "freq": "A",
                        "flowCode": "X,M",
                        "includeDesc": True,
                    }
                    data = self.fetch_with_retry(self.BASE_URL, params)
                    rows = data.get("data", [])
                    if rows:
                        frames.append(pd.DataFrame(rows))

        if frames:
            df = pd.concat(frames, ignore_index=True)
            logger.info(f"Fetched {len(df):,} Comtrade rows.")
            return df

        logger.warning("No Comtrade data fetched — returning synthetic matrix.")
        return self._synthetic_bilateral_matrix(list(reporters), list(partners), list(years))

    def _synthetic_bilateral_matrix(
        self, reporters: List[str], partners: List[str], years: List[int]
    ) -> pd.DataFrame:
        """Synthetic bilateral trade flows calibrated to real-world country pairs."""
        rng = np.random.default_rng(99)
        country_gdp = {
            "CN": 17_700, "US": 25_400, "DE": 4_100, "JP": 4_200, "IN": 3_400,
            "KR": 1_700, "VN": 408, "MX": 1_270, "BR": 1_920, "NL": 1_000,
        }
        rows = []
        for reporter in reporters:
            for partner in partners:
                if reporter == partner:
                    continue
                for year in years:
                    base_gdp = country_gdp.get(reporter[:2].upper(), 500)
                    export_val = base_gdp * 1e9 * float(rng.lognormal(-2, 0.8))
                    import_val = base_gdp * 1e9 * float(rng.lognormal(-2, 0.8))
                    rows.append(
                        {
                            "reporterCode": reporter,
                            "partnerCode": partner,
                            "period": year,
                            "flowCode": "X",
                            "primaryValue": round(export_val, 0),
                            "cmdCode": "TOTAL",
                        }
                    )
                    rows.append(
                        {
                            "reporterCode": reporter,
                            "partnerCode": partner,
                            "period": year,
                            "flowCode": "M",
                            "primaryValue": round(import_val, 0),
                            "cmdCode": "TOTAL",
                        }
                    )
        return pd.DataFrame(rows)


# ─── 5. Legacy classes (kept for backward compatibility) ─────────────────────

class ComtradeIngester(ComtradeAPIFetcher):
    """Backward-compatible alias for ComtradeAPIFetcher."""

    def fetch(self, reporter="842", partner="156", commodity="TOTAL", period="2023", freq="A"):
        params = {
            "reporterCode": reporter, "partnerCode": partner,
            "cmdCode": commodity, "period": period, "freq": freq,
            "flowCode": "X,M", "includeDesc": True,
        }
        data = self.fetch_with_retry(self.BASE_URL, params)
        return pd.DataFrame(data.get("data", []))


class SyntheticDataGenerator:
    """Lightweight generator kept for backward compatibility.

    For comprehensive generation, use SupplyChainNetworkGenerator,
    TimeSeriesGenerator, and TradefinanceDataGenerator.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._sc_gen = SupplyChainNetworkGenerator(seed)
        self._tf_gen = TradefinanceDataGenerator(seed)

    def generate_carriers(self, n: int = 500) -> pd.DataFrame:
        return self._sc_gen.generate_suppliers(n).rename(
            columns={"supplier_id": "carrier_id", "otif_rate": "on_time_delivery_rate",
                     "debt_equity": "debt_to_equity"}
        ).assign(
            carrier_failure=lambda df: (self.rng.random(len(df)) < 0.05).astype(int),
            carrier_tenure_days=lambda df: self.rng.integers(30, 3650, len(df)),
            carrier_type=lambda df: self.rng.choice(
                ["Air", "Ocean", "Road", "Rail", "Multimodal"], len(df)
            ),
            region=lambda df: self.rng.choice(
                ["APAC", "EMEA", "AMER", "LATAM", "MEA"], len(df)
            ),
            fleet_size=lambda df: self.rng.integers(5, 500, len(df)),
            credit_score=lambda df: self.rng.integers(300, 850, len(df)),
            damage_rate=lambda df: self.rng.beta(1, 20, len(df)),
            cost_per_kg=lambda df: self.rng.lognormal(1.5, 0.5, len(df)),
        )

    def generate_shipments(self, n: int = 50_000, carrier_ids: Optional[list] = None) -> pd.DataFrame:
        rng = self.rng
        carrier_ids = carrier_ids or [f"CAR-{i:04d}" for i in range(500)]
        base = datetime(2020, 1, 1)
        records = []
        for i in range(n):
            orig = str(rng.choice(["CN", "US", "DE", "JP", "IN", "BR", "KR", "NL"]))
            dest = str(rng.choice(["US", "DE", "GB", "JP", "AU", "CA", "FR", "SG"]))
            if orig == dest:
                dest = "US"
            transit = int(rng.integers(1, 60))
            planned = max(1, int(rng.integers(max(1, transit - 3), transit + 11)))
            ship_date = base + timedelta(days=int(rng.integers(0, 1460)))
            val = float(rng.lognormal(10, 2))
            records.append({
                "shipment_id": f"SHP-{i:06d}",
                "carrier_id": str(rng.choice(carrier_ids)),
                "origin_country": orig,
                "destination_country": dest,
                "ship_date": ship_date.strftime("%Y-%m-%d"),
                "planned_transit_days": planned,
                "actual_transit_days": transit,
                "delay_days": max(0, transit - planned),
                "on_time": int(transit <= planned),
                "weight_kg": float(rng.lognormal(3, 1.5)),
                "volume_cbm": float(rng.lognormal(0.5, 1.0)),
                "value_usd": val,
                "insurance_usd": val * float(rng.uniform(0.001, 0.01)),
                "freight_cost_usd": float(rng.lognormal(6, 1)),
                "port_congestion_days": int(rng.integers(0, 14)),
                "customs_delay_days": int(rng.integers(0, 7)),
                "damage_flag": int(rng.random() < 0.03),
                "commodity_code": str(rng.choice(["84", "85", "87", "30", "10"])),
            })
        return pd.DataFrame(records)

    def generate_financial_data(self, n_companies: int = 200) -> pd.DataFrame:
        rng = self.rng
        industries = ["Manufacturing", "Retail", "Wholesale", "Logistics", "Tech"]
        records = []
        for i in range(n_companies):
            rev = float(rng.lognormal(15, 2))
            cogs = rev * float(rng.uniform(0.4, 0.75))
            records.append({
                "company_id": f"CO-{i:04d}",
                "industry": str(rng.choice(industries)),
                "revenue_usd": rev,
                "cogs_usd": cogs,
                "gross_margin": (rev - cogs) / rev,
                "days_sales_outstanding": float(rng.uniform(20, 90)),
                "days_payable_outstanding": float(rng.uniform(15, 75)),
                "days_inventory_outstanding": float(rng.uniform(10, 120)),
                "current_ratio": float(rng.lognormal(0.4, 0.3)),
                "quick_ratio": float(rng.lognormal(0.1, 0.3)),
                "debt_to_equity": float(rng.lognormal(0.3, 0.6)),
                "interest_coverage": float(rng.lognormal(1.5, 0.8)),
                "altman_z_score": float(rng.normal(3.5, 1.5)),
                "credit_rating": str(rng.choice(CREDIT_RATINGS, p=RATING_WEIGHTS)),
                "default_flag": int(rng.random() < 0.05),
                "lc_utilization_rate": float(rng.beta(3, 5)),
                "payment_terms_days": int(rng.choice([30, 45, 60, 90, 120])),
                "fx_exposure_usd": float(rng.lognormal(12, 2)),
            })
        return pd.DataFrame(records)

    def generate_all(self, save_path: Optional[str] = None) -> dict:
        logger.info("Generating synthetic supply chain data (legacy generator)...")
        carriers = self.generate_carriers(500)
        shipments = self.generate_shipments(50_000, carriers["carrier_id"].tolist())
        financial = self.generate_financial_data(200)
        dataset = {"carriers": carriers, "shipments": shipments, "financial": financial}
        if save_path:
            p = Path(save_path)
            p.mkdir(parents=True, exist_ok=True)
            for name, df in dataset.items():
                df.to_csv(p / f"{name}.csv", index=False)
        return dataset


# ─── 6. Orchestration Pipeline ───────────────────────────────────────────────

class DataPipeline:
    """Orchestrates all data generators and loads/saves datasets."""

    def __init__(self, config: Optional[dict] = None, api_key: Optional[str] = None):
        self.config = config or {}
        self.comtrade = ComtradeAPIFetcher(api_key)
        self.synthetic = SyntheticDataGenerator()
        self.raw_path = Path(os.getenv("RAW_DATA_PATH", "./data/raw"))
        self.processed_path = Path(os.getenv("PROCESSED_DATA_PATH", "./data/processed"))

    def run(self, use_synthetic: bool = True) -> dict:
        logger.info("Starting data pipeline...")
        if use_synthetic:
            data = self.synthetic.generate_all(save_path=str(self.raw_path))
        else:
            data = {name: self._load_or_empty(f"{name}.csv")
                    for name in ["carriers", "shipments", "financial"]}
        logger.info(f"Pipeline complete: {list(data.keys())}")
        return data

    def _load_or_empty(self, filename: str) -> pd.DataFrame:
        path = self.raw_path / filename
        if path.exists():
            return pd.read_csv(path)
        logger.warning(f"{filename} not found.")
        return pd.DataFrame()


# ─── 7. main() — Generate and save all raw datasets ─────────────────────────

def main():
    """Generate all synthetic datasets and save to data/raw/ as CSV files.

    Output files
    ────────────
    suppliers_500.csv               500 supplier nodes (28 features)
    supply_chain_edges_2000.csv     2,000 supply chain edges (8 features)
    port_throughput_3years.csv      Daily TEU for LA, Rotterdam, Singapore (3 yrs)
    freight_rates_3years.csv        SCFI-like index for 3 major lanes (3 yrs)
    vessel_positions_1year.csv      500 vessels × 365 days position tracking
    lc_transactions_25000.csv       25,000 LC records with SC-adjusted default
    scf_invoices_50000.csv          50,000 SCF invoice records
    working_capital_facilities_500.csv  500 WC facility records with covenants
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    RAW_PATH = Path("data/raw")
    RAW_PATH.mkdir(parents=True, exist_ok=True)

    sc_gen = SupplyChainNetworkGenerator(seed=42)
    ts_gen = TimeSeriesGenerator(seed=42, start_date="2020-01-01")
    tf_gen = TradefinanceDataGenerator(seed=42)

    tasks = [
        ("suppliers_500.csv",                  "500 supplier nodes"),
        ("supply_chain_edges_2000.csv",         "2,000 SC edges"),
        ("port_throughput_3years.csv",          "Port throughput 3yr"),
        ("freight_rates_3years.csv",            "Freight rates 3yr"),
        ("vessel_positions_1year.csv",          "Vessel positions 1yr"),
        ("lc_transactions_25000.csv",           "25,000 LC transactions"),
        ("scf_invoices_50000.csv",              "50,000 SCF invoices"),
        ("working_capital_facilities_500.csv",  "500 WC facilities"),
    ]

    with tqdm(total=len(tasks), desc="LogisChain AI — Data Generation", unit="dataset") as pbar:

        # 1. Suppliers
        pbar.set_description(f"[1/8] {tasks[0][1]}")
        suppliers = sc_gen.generate_suppliers(n=500)
        suppliers.to_csv(RAW_PATH / tasks[0][0], index=False)
        tqdm.write(f"  ✓ {tasks[0][0]}  shape={suppliers.shape}")
        pbar.update(1)

        # 2. Edges
        pbar.set_description(f"[2/8] {tasks[1][1]}")
        edges = sc_gen.generate_edges(suppliers, n_edges=2000)
        edges.to_csv(RAW_PATH / tasks[1][0], index=False)
        tqdm.write(f"  ✓ {tasks[1][0]}  shape={edges.shape}")
        pbar.update(1)

        # 3. Port throughput
        pbar.set_description(f"[3/8] {tasks[2][1]}")
        port_df = ts_gen.generate_port_throughput(
            ports=["LA", "Rotterdam", "Singapore"], days=1095
        )
        port_df.to_csv(RAW_PATH / tasks[2][0], index=False)
        tqdm.write(f"  ✓ {tasks[2][0]}  shape={port_df.shape}")
        pbar.update(1)

        # 4. Freight rates
        pbar.set_description(f"[4/8] {tasks[3][1]}")
        rates_df = ts_gen.generate_freight_rates(
            lanes=["Shanghai-LA", "Shanghai-Rotterdam", "LA-Rotterdam"], days=1095
        )
        rates_df.to_csv(RAW_PATH / tasks[3][0], index=False)
        tqdm.write(f"  ✓ {tasks[3][0]}  shape={rates_df.shape}")
        pbar.update(1)

        # 5. Vessel positions
        pbar.set_description(f"[5/8] {tasks[4][1]}")
        vessels_df = ts_gen.generate_vessel_positions(n_vessels=500, days=365)
        vessels_df.to_csv(RAW_PATH / tasks[4][0], index=False)
        tqdm.write(f"  ✓ {tasks[4][0]}  shape={vessels_df.shape}")
        pbar.update(1)

        # 6. LC transactions
        pbar.set_description(f"[6/8] {tasks[5][1]}")
        lc_df = tf_gen.generate_lc_transactions(n=25_000)
        lc_df.to_csv(RAW_PATH / tasks[5][0], index=False)
        tqdm.write(f"  ✓ {tasks[5][0]}  shape={lc_df.shape}  "
                   f"default_rate={lc_df['default_flag'].mean():.2%}")
        pbar.update(1)

        # 7. SCF invoices
        pbar.set_description(f"[7/8] {tasks[6][1]}")
        scf_df = tf_gen.generate_scf_invoices(n=50_000)
        scf_df.to_csv(RAW_PATH / tasks[6][0], index=False)
        tqdm.write(f"  ✓ {tasks[6][0]}  shape={scf_df.shape}")
        pbar.update(1)

        # 8. WC facilities
        pbar.set_description(f"[8/8] {tasks[7][1]}")
        wc_df = tf_gen.generate_working_capital_facilities(n=500)
        wc_df.to_csv(RAW_PATH / tasks[7][0], index=False)
        tqdm.write(f"  ✓ {tasks[7][0]}  shape={wc_df.shape}  "
                   f"breach_rate={wc_df['covenant_breach_flag'].mean():.1%}")
        pbar.update(1)

    total_rows = (
        len(suppliers) + len(edges) + len(port_df) + len(rates_df)
        + len(vessels_df) + len(lc_df) + len(scf_df) + len(wc_df)
    )
    logger.info(
        f"\nAll datasets generated → {RAW_PATH.resolve()}\n"
        f"Total rows across all files: {total_rows:,}"
    )


if __name__ == "__main__":
    main()
