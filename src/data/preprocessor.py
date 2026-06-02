"""Data preprocessing, feature engineering, quality checking, and splitting for LogisChain AI.

Classes
───────
FeatureEngineer        — Computes 21 SC + 21 financial + 3 fusion + temporal + network features
DataQualityChecker     — Completeness, consistency, anomaly detection, quality reports
DataSplitter           — Temporal train/test split and walk-forward CV splits
LogisChainPreprocessor — Imputation, scaling, encoding (backward-compatible)
"""
import logging
import math
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler

logger = logging.getLogger(__name__)

# ── Chinese New Year windows for Fourier holiday features ──
_CNY: Dict[int, Tuple[str, str]] = {
    2019: ("2019-02-05", "2019-02-19"),
    2020: ("2020-01-25", "2020-02-08"),
    2021: ("2021-02-12", "2021-02-26"),
    2022: ("2022-02-01", "2022-02-15"),
    2023: ("2023-01-22", "2023-02-05"),
    2024: ("2024-02-10", "2024-02-24"),
}


# ─── 1. Feature Engineer ─────────────────────────────────────────────────────

class FeatureEngineer:
    """Computes domain features across supply chain, financial, and fusion domains.

    Each compute_* method is idempotent — calling it multiple times on the
    same DataFrame produces the same result, making it safe to cache output.
    """

    # ── Supply chain features ──────────────────────────────────────────────

    def compute_supply_chain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive 21 supply chain features from raw supplier / shipment data.

        Feature catalogue
        ─────────────────
        1.  hhi_supplier_concentration_norm   — Supplier HHI scaled 0-1
        2.  hhi_customer_concentration_norm   — Customer HHI scaled 0-1
        3.  ccc                               — Cash Conversion Cycle (days)
        4.  lead_time_cv                      — Coefficient of variation of lead time
        5.  inventory_days_of_supply          — 365 / inventory_turnover
        6.  capacity_slack                    — 1 - capacity_utilization
        7.  fill_rate_deficit                 — 1 - fill_rate
        8.  otif_deviation                    — 1 - otif_rate
        9.  freight_cost_ratio                — freight cost as % of revenue
        10. port_proximity_score              — 0 (inland) → 1 (major port)
        11. country_risk_score                — 0 (safe) → 1 (high risk)
        12. natural_disaster_exposure         — 0 → 1
        13. geopolitical_risk                 — 0 → 1
        14. route_risk_composite              — weighted combination of 11-13
        15. betweenness_centrality            — network graph metric
        16. clustering_coeff                  — local network clustering
        17. pagerank                          — network importance score
        18. supply_chain_resilience_score     — composite resilience metric
        19. disruption_vulnerability_index    — composite vulnerability metric
        20. operational_risk_score            — operational performance metric
        21. sc_risk_tier                      — categorical: LOW/MEDIUM/HIGH/CRITICAL
        """
        df = df.copy()

        # 1-2: HHI normalisation (0-10000 → 0-1)
        for col, out in [
            ("supplier_concentration_hhi", "hhi_supplier_concentration_norm"),
            ("customer_concentration_hhi", "hhi_customer_concentration_norm"),
        ]:
            if col in df.columns:
                df[out] = df[col] / 10_000.0

        # 3: Cash conversion cycle
        if all(c in df.columns for c in ["dio", "dso", "dpo"]):
            df["ccc"] = df["dio"] + df["dso"] - df["dpo"]
        elif "cash_conversion_cycle" in df.columns:
            df["ccc"] = df["cash_conversion_cycle"]

        # 4: Lead time coefficient of variation
        if "lead_time_mean" in df.columns and "lead_time_std" in df.columns:
            df["lead_time_cv"] = df["lead_time_std"] / df["lead_time_mean"].clip(lower=0.01)

        # 5: Inventory days of supply
        if "inventory_turnover" in df.columns:
            df["inventory_days_of_supply"] = 365.0 / df["inventory_turnover"].clip(lower=0.1)

        # 6-7-8: Deficit / slack / deviation
        for src, out in [
            ("capacity_utilization", "capacity_slack"),
            ("fill_rate", "fill_rate_deficit"),
            ("otif_rate", "otif_deviation"),
        ]:
            if src in df.columns:
                df[out] = 1.0 - df[src].clip(0, 1)

        # 9: Freight cost ratio (already in source or derive)
        if "freight_cost_ratio" not in df.columns:
            if "freight_cost_usd" in df.columns and "revenue_usd" in df.columns:
                df["freight_cost_ratio"] = df["freight_cost_usd"] / df["revenue_usd"].clip(lower=1)

        # 10-13: Country-level risk (pass-through, clip to [0,1])
        for col in ["port_proximity_score", "country_risk_score",
                    "natural_disaster_exposure", "geopolitical_risk"]:
            if col in df.columns:
                df[col] = df[col].clip(0, 1)

        # 14: Route risk composite = 0.4*geo + 0.3*disaster + 0.3*(1-port_proximity)
        gpr = df.get("geopolitical_risk", pd.Series(0.3, index=df.index))
        ndr = df.get("natural_disaster_exposure", pd.Series(0.3, index=df.index))
        pps = df.get("port_proximity_score", pd.Series(0.5, index=df.index))
        df["route_risk_composite"] = (0.40 * gpr + 0.30 * ndr + 0.30 * (1.0 - pps)).clip(0, 1)

        # 15-17: Network centrality (pass-through from network computation or synthetic)
        for col in ["betweenness_centrality", "clustering_coeff", "pagerank"]:
            if col not in df.columns:
                df[col] = 0.0

        # 18: Supply chain resilience score (higher = more resilient)
        otif = df.get("otif_rate", pd.Series(0.85, index=df.index))
        fr = df.get("fill_rate", pd.Series(0.90, index=df.index))
        pp = df.get("port_proximity_score", pd.Series(0.5, index=df.index))
        div = 1.0 - df.get("hhi_supplier_concentration_norm", pd.Series(0.3, index=df.index))
        df["supply_chain_resilience_score"] = (
            0.30 * otif + 0.25 * fr + 0.20 * pp + 0.25 * div
        ).clip(0, 1)

        # 19: Disruption vulnerability index (higher = more vulnerable)
        rrc = df.get("route_risk_composite", pd.Series(0.3, index=df.index))
        hhi = df.get("hhi_supplier_concentration_norm", pd.Series(0.3, index=df.index))
        ltd = df.get("lead_time_cv", pd.Series(0.2, index=df.index))
        df["disruption_vulnerability_index"] = (
            0.40 * rrc + 0.35 * hhi + 0.25 * ltd.clip(0, 2) / 2.0
        ).clip(0, 1)

        # 20: Operational risk score
        otif_dev = df.get("otif_deviation", pd.Series(0.10, index=df.index))
        frd = df.get("fill_rate_deficit", pd.Series(0.08, index=df.index))
        cap_sl = df.get("capacity_slack", pd.Series(0.25, index=df.index))
        df["operational_risk_score"] = (
            0.40 * otif_dev + 0.35 * frd + 0.25 * cap_sl
        ).clip(0, 1)

        # 21: Categorical risk tier
        dvi = df["disruption_vulnerability_index"]
        df["sc_risk_tier"] = pd.cut(
            dvi,
            bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
            labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        ).astype(str)

        logger.debug(f"compute_supply_chain_features: added {21} SC features.")
        return df

    # ── Financial features ─────────────────────────────────────────────────

    def compute_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive 21 financial features from raw financial statement data.

        Feature catalogue
        ─────────────────
        1.  cash_conversion_cycle         — DIO + DSO - DPO
        2.  working_capital_ratio         — current assets / current liabilities
        3.  altman_z_score                — Altman Z (from components or given)
        4.  altman_zone                   — Safe / Grey / Distress
        5.  altman_distress_flag          — 1 if Z < 1.81
        6.  credit_rating_numeric         — AAA→1 … CCC→7
        7.  investment_grade_flag         — 1 if rating ≤ BBB (numeric ≤ 4)
        8.  high_leverage_flag            — 1 if D/E > 3.0
        9.  leverage_tier                 — Conservative / Moderate / Elevated / High / Extreme
        10. coverage_risk_flag            — 1 if interest coverage < 1.5
        11. credit_stress_index           — composite stress 0-1
        12. net_debt_to_ebitda            — leverage via earnings
        13. free_cash_flow_proxy          — ebitda_margin * revenue (simplified FCF)
        14. operating_efficiency_ratio    — revenue / (cogs + freight) proxy
        15. lc_util_tier                  — LOW/MEDIUM/HIGH/CRITICAL (LC utilisation)
        16. payment_terms_bucket          — 30d / 60d / 90d / 120d+
        17. fx_exposure_to_revenue        — FX exposure as % of revenue
        18. financial_health_score        — composite 0-1 (higher = healthier)
        19. pd_from_altman                — logistic PD from Altman Z-score
        20. ccc_bucket                    — Excellent / Good / Fair / Poor / Critical
        21. working_capital_pressure      — liquidity pressure index 0-1
        """
        df = df.copy()
        RATING_MAP = {"AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5, "B": 6, "CCC": 7}

        # 1: CCC
        dso_col = next((c for c in ["days_sales_outstanding", "dso"] if c in df.columns), None)
        dpo_col = next((c for c in ["days_payable_outstanding", "dpo"] if c in df.columns), None)
        dio_col = next((c for c in ["days_inventory_outstanding", "dio"] if c in df.columns), None)
        if all(c is not None for c in [dso_col, dpo_col, dio_col]):
            df["cash_conversion_cycle"] = df[dso_col] + df[dio_col] - df[dpo_col]
        elif "cash_conversion_cycle" not in df.columns:
            df["cash_conversion_cycle"] = np.nan

        # 2: Working capital ratio (proxy via current_ratio)
        if "current_ratio" in df.columns:
            df["working_capital_ratio"] = df["current_ratio"]

        # 3-5: Altman Z
        if "altman_z_score" not in df.columns:
            df["altman_z_score"] = np.nan
        df["altman_zone"] = df["altman_z_score"].apply(
            lambda z: "Safe" if z >= 2.99 else ("Grey" if z >= 1.81 else "Distress")
            if pd.notna(z) else "Unknown"
        )
        df["altman_distress_flag"] = (df["altman_z_score"].fillna(3.0) < 1.81).astype(int)

        # 6-7: Credit rating
        rating_col = next((c for c in ["credit_rating", "anchor_credit_rating"] if c in df.columns), None)
        if rating_col:
            df["credit_rating_numeric"] = (
                df[rating_col].str.upper().map(RATING_MAP).fillna(5.0)
            )
            df["investment_grade_flag"] = (df["credit_rating_numeric"] <= 4).astype(int)

        # 8-9: Leverage
        de_col = next((c for c in ["debt_equity", "debt_to_equity"] if c in df.columns), None)
        if de_col:
            df["high_leverage_flag"] = (df[de_col] > 3.0).astype(int)
            df["leverage_tier"] = pd.cut(
                df[de_col].clip(0, 10),
                bins=[-0.001, 1.0, 2.0, 3.0, 5.0, 10.001],
                labels=["Conservative", "Moderate", "Elevated", "High", "Extreme"],
            ).astype(str)

        # 10: Interest coverage risk
        if "interest_coverage" in df.columns:
            df["coverage_risk_flag"] = (df["interest_coverage"] < 1.5).astype(int)

        # 11: Credit stress index (composite 0-1)
        stress_parts = []
        w_total = 0.0
        if "altman_distress_flag" in df.columns:
            stress_parts.append(df["altman_distress_flag"] * 0.30); w_total += 0.30
        if "high_leverage_flag" in df.columns:
            stress_parts.append(df["high_leverage_flag"] * 0.25); w_total += 0.25
        if "coverage_risk_flag" in df.columns:
            stress_parts.append(df["coverage_risk_flag"] * 0.25); w_total += 0.25
        if "credit_rating_numeric" in df.columns:
            stress_parts.append((df["credit_rating_numeric"] / 7.0) * 0.20); w_total += 0.20
        if stress_parts:
            df["credit_stress_index"] = sum(stress_parts).clip(0, 1)

        # 12: Net debt / EBITDA proxy
        if de_col and "ebitda_margin" in df.columns and "revenue_usd" in df.columns:
            ebitda = df["ebitda_margin"] * df["revenue_usd"]
            df["net_debt_to_ebitda"] = (df[de_col] * df["revenue_usd"] * 0.3) / ebitda.clip(lower=1)

        # 13: FCF proxy
        if "ebitda_margin" in df.columns and "revenue_usd" in df.columns:
            df["free_cash_flow_proxy"] = df["ebitda_margin"] * df["revenue_usd"] * 0.65

        # 14: Operating efficiency (revenue / (COGS + 1))
        rev = df.get("revenue_usd", pd.Series(np.nan, index=df.index))
        cogs = df.get("cogs_usd", pd.Series(np.nan, index=df.index))
        if rev.notna().any() and cogs.notna().any():
            df["operating_efficiency_ratio"] = rev / (cogs.clip(lower=1))

        # 15: LC utilisation tier
        if "lc_utilization_rate" in df.columns:
            df["lc_util_tier"] = pd.cut(
                df["lc_utilization_rate"].clip(0, 1),
                bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
                labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            ).astype(str)

        # 16: Payment terms bucket
        pt_col = next((c for c in ["payment_terms_days", "payment_terms"] if c in df.columns), None)
        if pt_col:
            df["payment_terms_bucket"] = pd.cut(
                df[pt_col].clip(0, 365),
                bins=[-1, 30, 60, 90, 365],
                labels=["30d", "60d", "90d", "120d+"],
            ).astype(str)

        # 17: FX exposure / revenue
        if "fx_exposure_usd" in df.columns and "revenue_usd" in df.columns:
            df["fx_exposure_to_revenue"] = (
                df["fx_exposure_usd"] / df["revenue_usd"].clip(lower=1)
            ).clip(0, 5)

        # 18: Financial health score (higher = healthier)
        health_parts = []
        if "current_ratio" in df.columns:
            health_parts.append(df["current_ratio"].clip(0, 4) / 4.0 * 0.20)
        if "interest_coverage" in df.columns:
            health_parts.append(df["interest_coverage"].clip(0, 10) / 10.0 * 0.25)
        if de_col:
            health_parts.append((1 - df[de_col].clip(0, 5) / 5.0) * 0.25)
        if "ebitda_margin" in df.columns:
            health_parts.append(df["ebitda_margin"].clip(0, 0.5) / 0.5 * 0.30)
        if health_parts:
            df["financial_health_score"] = sum(health_parts).clip(0, 1)

        # 19: PD from Altman (logistic: PD = 1/(1+exp(0.5*(Z-1.81))))
        df["pd_from_altman"] = 1.0 / (
            1.0 + np.exp(0.5 * (df["altman_z_score"].fillna(1.81) - 1.81))
        )

        # 20: CCC bucket
        if "cash_conversion_cycle" in df.columns:
            df["ccc_bucket"] = pd.cut(
                df["cash_conversion_cycle"].clip(-30, 200),
                bins=[-31, 30, 60, 90, 120, 201],
                labels=["Excellent", "Good", "Fair", "Poor", "Critical"],
            ).astype(str)

        # 21: Working capital pressure
        cr = df.get("current_ratio", pd.Series(1.5, index=df.index))
        ccc_vals = df.get("cash_conversion_cycle", pd.Series(60.0, index=df.index))
        df["working_capital_pressure"] = (
            0.50 * (1 - cr.clip(0, 3) / 3.0) + 0.50 * ccc_vals.clip(0, 180) / 180.0
        ).clip(0, 1)

        logger.debug("compute_financial_features: added 21 financial features.")
        return df

    # ── Fusion features ────────────────────────────────────────────────────

    def compute_fusion_features(
        self, sc_df: pd.DataFrame, fin_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute 3 primary cross-domain fusion features (SC-PD, WCVI, TRFSI)
        plus the flagship LogisChain composite score.

        Merges on shared index. If sc_df and fin_df have different lengths,
        aligns on positional index (take min length).

        Fusion Features
        ───────────────
        SC-PD   Supply-chain-adjusted probability of default
                = pd_from_altman × (1 + 0.5×route_risk + 0.4×otif_deviation
                                      + 0.3×hhi_supplier_concentration_norm)

        WCVI    Working Capital Vulnerability Index
                = 0.4×(ccc/180) + 0.3×fill_rate_deficit + 0.3×disruption_vulnerability_index

        TRFSI   Trade Route Financial Stress Index
                = 0.5×route_risk_composite + 0.3×credit_stress_index + 0.2×fill_rate_deficit

        Also recomputes the logischain_composite_risk_score and sc_financial_stress_index.
        """
        n = min(len(sc_df), len(fin_df))
        sc = sc_df.iloc[:n].reset_index(drop=True)
        fin = fin_df.iloc[:n].reset_index(drop=True)
        fused = pd.concat([sc, fin], axis=1)
        fused = fused.loc[:, ~fused.columns.duplicated()]  # drop duplicate cols

        # SC-PD
        base_pd = fused.get("pd_from_altman", pd.Series(0.05, index=fused.index)).fillna(0.05)
        rrc = fused.get("route_risk_composite", pd.Series(0.30, index=fused.index)).fillna(0.30)
        otif_dev = fused.get("otif_deviation", pd.Series(0.10, index=fused.index)).fillna(0.10)
        hhi_sup = fused.get("hhi_supplier_concentration_norm", pd.Series(0.30, index=fused.index)).fillna(0.30)
        fused["sc_pd"] = np.clip(
            base_pd * (1.0 + 0.50 * rrc + 0.40 * otif_dev + 0.30 * hhi_sup), 0, 1
        )

        # WCVI
        ccc_vals = fused.get("cash_conversion_cycle", pd.Series(60.0, index=fused.index)).fillna(60)
        frd = fused.get("fill_rate_deficit", pd.Series(0.08, index=fused.index)).fillna(0.08)
        dvi = fused.get("disruption_vulnerability_index", pd.Series(0.30, index=fused.index)).fillna(0.30)
        fused["wcvi"] = np.clip(
            0.40 * (ccc_vals / 180.0) + 0.30 * frd + 0.30 * dvi, 0, 1
        )

        # TRFSI
        csi = fused.get("credit_stress_index", pd.Series(0.25, index=fused.index)).fillna(0.25)
        fused["trfsi"] = np.clip(
            0.50 * rrc + 0.30 * csi + 0.20 * frd, 0, 1
        )

        # LogisChain composite risk score (weighted fusion of all key signals)
        weights = {
            "sc_pd": 0.25,
            "wcvi": 0.20,
            "trfsi": 0.20,
            "operational_risk_score": 0.15,
            "credit_stress_index": 0.10,
            "route_risk_composite": 0.10,
        }
        composite = pd.Series(0.0, index=fused.index)
        w_used = 0.0
        for feat, w in weights.items():
            if feat in fused.columns:
                composite += fused[feat].fillna(0) * w
                w_used += w
        fused["logischain_composite_risk_score"] = (composite / max(w_used, 1e-8)).clip(0, 1)

        # SC-financial stress index
        stress_feats = [f for f in ["logischain_composite_risk_score", "credit_stress_index",
                                     "altman_distress_flag"] if f in fused.columns]
        if stress_feats:
            fused["sc_financial_stress_index"] = fused[stress_feats].fillna(0).mean(axis=1)

        logger.debug("compute_fusion_features: SC-PD, WCVI, TRFSI, composite risk, stress index added.")
        return fused

    # ── Temporal features ──────────────────────────────────────────────────

    def compute_temporal_features(
        self,
        ts_df: pd.DataFrame,
        date_col: str = "date",
        value_col: str = "value",
    ) -> pd.DataFrame:
        """Derive temporal features from a time-indexed DataFrame.

        Features added
        ──────────────
        Rolling means & stds   : 7, 14, 30, 90-day windows
        Lag features           : t-1, t-7, t-14, t-30
        Fourier seasonality    : annual, semi-annual, quarterly, weekly sin/cos
        Calendar               : month, quarter, year, day_of_week, is_weekend
        Holiday indicators     : chinese_new_year, golden_week_japan, golden_week_china
        Year-over-year         : % change vs same period last year
        """
        df = ts_df.copy()
        if date_col not in df.columns:
            logger.warning(f"date_col '{date_col}' not in DataFrame — temporal features skipped.")
            return df

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)
        t = np.arange(len(df))
        dt = df[date_col]

        if value_col not in df.columns:
            logger.warning(f"value_col '{value_col}' not in DataFrame — rolling/lag features skipped.")
        else:
            # Rolling mean & std
            for w in [7, 14, 30, 90]:
                df[f"{value_col}_roll_mean_{w}d"] = (
                    df[value_col].rolling(w, min_periods=1).mean()
                )
                df[f"{value_col}_roll_std_{w}d"] = (
                    df[value_col].rolling(w, min_periods=1).std().fillna(0)
                )

            # Lag features
            for lag in [1, 7, 14, 30]:
                df[f"{value_col}_lag_{lag}"] = df[value_col].shift(lag)

            # Year-over-year
            df[f"{value_col}_yoy_pct"] = df[value_col].pct_change(periods=365)

        # Fourier seasonality terms
        for k, name in [(1, "annual"), (2, "semi_annual"), (4, "quarterly")]:
            df[f"fourier_sin_{name}"] = np.sin(2 * math.pi * k * t / 365.0)
            df[f"fourier_cos_{name}"] = np.cos(2 * math.pi * k * t / 365.0)
        df["fourier_sin_weekly"] = np.sin(2 * math.pi * t / 7.0)
        df["fourier_cos_weekly"] = np.cos(2 * math.pi * t / 7.0)

        # Calendar
        df["year"]         = dt.dt.year
        df["month"]        = dt.dt.month
        df["quarter"]      = dt.dt.quarter
        df["day_of_week"]  = dt.dt.dayofweek
        df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype(int)
        df["is_month_end"] = dt.dt.is_month_end.astype(int)
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)

        # Chinese New Year holiday indicator
        cny_flag = np.zeros(len(df), dtype=int)
        for year, (lo, hi) in _CNY.items():
            mask = (dt >= pd.Timestamp(lo)) & (dt <= pd.Timestamp(hi))
            cny_flag[mask.values] = 1
        df["chinese_new_year"] = cny_flag

        # Japanese Golden Week (April 29 – May 5)
        df["golden_week_japan"] = (
            (dt.dt.month == 4) & (dt.dt.day >= 29)
            | (dt.dt.month == 5) & (dt.dt.day <= 5)
        ).astype(int)

        # Chinese National Day Golden Week (October 1-7)
        df["golden_week_china"] = (
            (dt.dt.month == 10) & (dt.dt.day <= 7)
        ).astype(int)

        logger.debug(
            f"compute_temporal_features: {df.shape[1] - ts_df.shape[1]} temporal features added."
        )
        return df

    # ── Network features ───────────────────────────────────────────────────

    def compute_network_features(self, graph: nx.Graph) -> pd.DataFrame:
        """Compute graph-theoretic features for all nodes in a NetworkX graph.

        Returns a DataFrame indexed by node with columns:
            degree, in_degree, out_degree,
            betweenness_centrality, closeness_centrality, eigenvector_centrality,
            clustering_coefficient, pagerank, hub_score, authority_score
        """
        if graph.number_of_nodes() == 0:
            return pd.DataFrame()

        is_directed = isinstance(graph, nx.DiGraph)

        # Degree
        degree = dict(graph.degree())
        in_deg = dict(graph.in_degree()) if is_directed else degree
        out_deg = dict(graph.out_degree()) if is_directed else degree

        # Centrality measures
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        closeness = nx.closeness_centrality(graph)

        try:
            eigenvector = nx.eigenvector_centrality(
                graph, max_iter=500, tol=1e-6,
                weight="weight" if nx.is_weighted(graph) else None,
            )
        except nx.PowerIterationFailedConvergence:
            eigenvector = {n: 0.0 for n in graph.nodes()}

        pagerank = nx.pagerank(graph, alpha=0.85, weight="weight" if nx.is_weighted(graph) else None)

        undirected = graph.to_undirected() if is_directed else graph
        clustering = nx.clustering(undirected)

        hub_score = authority_score = {}
        try:
            hub_score, authority_score = nx.hits(graph, max_iter=200, tol=1e-8)
        except Exception:
            hub_score = {n: 0.0 for n in graph.nodes()}
            authority_score = {n: 0.0 for n in graph.nodes()}

        rows = []
        for node in graph.nodes():
            rows.append(
                {
                    "node":                    node,
                    "degree":                  degree.get(node, 0),
                    "in_degree":               in_deg.get(node, 0),
                    "out_degree":              out_deg.get(node, 0),
                    "betweenness_centrality":  betweenness.get(node, 0.0),
                    "closeness_centrality":    closeness.get(node, 0.0),
                    "eigenvector_centrality":  eigenvector.get(node, 0.0),
                    "clustering_coefficient":  clustering.get(node, 0.0),
                    "pagerank":                pagerank.get(node, 0.0),
                    "hub_score":               hub_score.get(node, 0.0),
                    "authority_score":         authority_score.get(node, 0.0),
                }
            )

        df = pd.DataFrame(rows).set_index("node")
        logger.debug(f"compute_network_features: {len(df)} nodes, {df.shape[1]} metrics.")
        return df


# ─── 2. Data Quality Checker ─────────────────────────────────────────────────

class DataQualityChecker:
    """Comprehensive data quality assessment for LogisChain datasets."""

    COMPLETENESS_THRESHOLD = 0.80  # columns below this pct non-null are flagged

    def check_completeness(self, df: pd.DataFrame) -> Dict[str, object]:
        """Report null percentage per column and flag columns below threshold.

        Returns
        ───────
        {
            "total_rows": int,
            "total_cols": int,
            "completeness_pct_per_col": {col: float},
            "columns_below_threshold": [col, ...],
            "overall_completeness_pct": float,
        }
        """
        n = len(df)
        pct = {col: round(float(df[col].notna().mean()) * 100, 2) for col in df.columns}
        below = [col for col, p in pct.items() if p < self.COMPLETENESS_THRESHOLD * 100]
        overall = round(float(df.notna().mean().mean()) * 100, 2)
        return {
            "total_rows": n,
            "total_cols": len(df.columns),
            "completeness_pct_per_col": pct,
            "columns_below_threshold": below,
            "overall_completeness_pct": overall,
        }

    def check_consistency(self, df: pd.DataFrame) -> Dict[str, object]:
        """Cross-validate related fields for internal consistency.

        Checks performed
        ────────────────
        - CCC ≈ DIO + DSO - DPO  (|diff| > 5 days flagged)
        - current_ratio > 0
        - debt_equity ≥ 0
        - lead_time_mean > 0
        - otif_rate ∈ [0, 1]
        - fill_rate ∈ [0, 1]
        - capacity_utilization ∈ [0, 1]
        - interest_coverage > 0 (negative = coverage insufficient)
        """
        issues: Dict[str, List[int]] = {}

        # CCC identity check
        ccc_cols = ("cash_conversion_cycle", "dio", "dso", "dpo")
        if all(c in df.columns for c in ccc_cols):
            computed = df["dio"] + df["dso"] - df["dpo"]
            delta = (df["cash_conversion_cycle"] - computed).abs()
            bad = df.index[delta > 5].tolist()
            if bad:
                issues["ccc_identity_violation"] = bad

        # Bounds checks
        bounds = {
            "current_ratio": (0, np.inf),
            "quick_ratio": (0, np.inf),
            "debt_equity": (0, np.inf),
            "debt_to_equity": (0, np.inf),
            "lead_time_mean": (0.5, np.inf),
            "otif_rate": (0, 1),
            "fill_rate": (0, 1),
            "capacity_utilization": (0, 1),
            "interest_coverage": (0, np.inf),
            "freight_cost_ratio": (0, 1),
            "ebitda_margin": (-1, 1),
        }
        for col, (lo, hi) in bounds.items():
            if col in df.columns:
                bad = df.index[
                    (df[col].notna()) & ((df[col] < lo) | (df[col] > hi))
                ].tolist()
                if bad:
                    issues[f"{col}_out_of_bounds"] = bad

        return {
            "n_issues_found": sum(len(v) for v in issues.values()),
            "issue_types": len(issues),
            "issues": {k: v[:20] for k, v in issues.items()},  # cap at 20 per type
        }

    def detect_anomalies(
        self, df: pd.DataFrame, threshold: float = 3.0
    ) -> Dict[str, List[int]]:
        """IQR-based anomaly detection on numeric columns.

        Flags rows where any value exceeds Q1 - threshold×IQR or Q3 + threshold×IQR.
        Returns dict: {column_name: [anomalous_row_indices]}.
        """
        num_df = df.select_dtypes(include=[np.number])
        anomalies: Dict[str, List[int]] = {}
        for col in num_df.columns:
            series = num_df[col].dropna()
            if len(series) < 10:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr < 1e-10:
                continue
            lo = q1 - threshold * iqr
            hi = q3 + threshold * iqr
            bad = num_df.index[(num_df[col].notna()) & ((num_df[col] < lo) | (num_df[col] > hi))].tolist()
            if bad:
                anomalies[col] = bad
        return anomalies

    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, object]:
        """Run all quality checks and return a unified quality report.

        Returns
        ───────
        {
            "completeness": {...},
            "consistency": {...},
            "anomaly_summary": {col: n_anomalies},
            "quality_score": float (0-100, higher is better),
            "flags": [str, ...],
        }
        """
        completeness = self.check_completeness(df)
        consistency = self.check_consistency(df)
        anomalies = self.detect_anomalies(df)

        anomaly_summary = {col: len(idxs) for col, idxs in anomalies.items()}
        total_anomaly_cells = sum(anomaly_summary.values())

        # Quality score: penalise for missing, inconsistencies, and anomalies
        completeness_score = completeness["overall_completeness_pct"]
        consistency_penalty = min(consistency["n_issues_found"] / max(len(df), 1) * 100, 20)
        anomaly_penalty = min(total_anomaly_cells / max(len(df) * len(df.columns), 1) * 1000, 20)
        quality_score = max(0.0, completeness_score - consistency_penalty - anomaly_penalty)

        flags = []
        if completeness["columns_below_threshold"]:
            flags.append(
                f"Low completeness (<{self.COMPLETENESS_THRESHOLD*100:.0f}%) in: "
                + ", ".join(completeness["columns_below_threshold"][:5])
            )
        if consistency["n_issues_found"] > 0:
            flags.append(f"{consistency['n_issues_found']} consistency violations detected.")
        if total_anomaly_cells > 0:
            flags.append(f"{total_anomaly_cells} anomalous values in {len(anomaly_summary)} columns.")

        return {
            "completeness": completeness,
            "consistency": consistency,
            "anomaly_summary": anomaly_summary,
            "quality_score": round(quality_score, 2),
            "flags": flags,
        }


# ─── 3. Data Splitter ────────────────────────────────────────────────────────

class DataSplitter:
    """Temporal train/test splitting for time series financial data.

    Prevents look-ahead bias by strictly enforcing temporal ordering.
    All splits are non-overlapping with a configurable gap period.
    """

    def temporal_split(
        self,
        df: pd.DataFrame,
        date_col: str,
        train_end: str,
        test_start: str,
        gap_days: int = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split a time-indexed DataFrame into train and test sets.

        Parameters
        ──────────
        date_col  : column name containing dates
        train_end : last date (inclusive) of the training period
        test_start: first date (inclusive) of the test period
        gap_days  : optional buffer between train end and test start

        Returns (train_df, test_df).
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        train_cutoff = pd.Timestamp(train_end)
        test_start_dt = pd.Timestamp(test_start)

        train = df[df[date_col] <= train_cutoff].copy()
        test = df[df[date_col] >= test_start_dt].copy()

        if gap_days > 0:
            gap_end = train_cutoff + pd.Timedelta(days=gap_days)
            test = test[test[date_col] >= gap_end]

        logger.info(
            f"Temporal split: train={len(train):,} rows (≤{train_end}), "
            f"test={len(test):,} rows (≥{test_start})"
        )
        return train, test

    def walk_forward_splits(
        self,
        df: pd.DataFrame,
        date_col: str,
        n_splits: int = 5,
        val_size_days: int = 90,
        min_train_days: int = 180,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """Generate expanding walk-forward (train, validation) pairs.

        Each subsequent fold adds more data to the training set while the
        validation window slides forward. Used for time-series cross-validation.

        Parameters
        ──────────
        n_splits       : number of (train, val) pairs to generate
        val_size_days  : width of each validation window in days
        min_train_days : minimum training period before first split

        Yields
        ──────
        (train_df, val_df) tuples, expanding train, sliding val.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)

        min_date = df[date_col].min()
        max_date = df[date_col].max()
        total_days = (max_date - min_date).days

        # Compute validation start points
        usable_days = total_days - min_train_days
        if usable_days < val_size_days:
            raise ValueError(
                f"Not enough data for {n_splits} splits. "
                f"Need ≥ {min_train_days + val_size_days} days, got {total_days}."
            )

        step = usable_days // n_splits

        for fold in range(n_splits):
            val_start = min_date + pd.Timedelta(days=min_train_days + fold * step)
            val_end = val_start + pd.Timedelta(days=val_size_days)
            train = df[df[date_col] < val_start]
            val = df[(df[date_col] >= val_start) & (df[date_col] < val_end)]
            if len(train) == 0 or len(val) == 0:
                continue
            logger.debug(
                f"Walk-forward fold {fold+1}/{n_splits}: "
                f"train≤{val_start.date()}, val [{val_start.date()}–{val_end.date()}], "
                f"sizes=({len(train)}, {len(val)})"
            )
            yield train.copy(), val.copy()


# ─── 4. LogisChainPreprocessor (backward-compatible) ─────────────────────────

class LogisChainPreprocessor:
    """Cleans, imputes, scales, and encodes raw LogisChain data.

    Backward-compatible class — maintained from v0.1.0.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.numeric_strategy = cfg.get("numeric_imputation", "median")
        self.categorical_strategy = cfg.get("categorical_imputation", "most_frequent")
        self.outlier_threshold = cfg.get("outlier_threshold", 3.0)
        self.lag_windows = cfg.get("lag_windows", [7, 14, 30, 60, 90])
        self.rolling_windows = cfg.get("rolling_windows", [7, 30, 90])

        self.num_imputer = SimpleImputer(strategy=self.numeric_strategy)
        self.cat_imputer = SimpleImputer(strategy=self.categorical_strategy)
        self.scaler = RobustScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._fitted = False

    def _detect_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [c for c in num_cols if not c.endswith("_id") and c != "index"]
        return num_cols, cat_cols

    def _clip_outliers(self, df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - self.outlier_threshold * iqr,
                                   q3 + self.outlier_threshold * iqr)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        num_cols, cat_cols = self._detect_types(df)
        if num_cols:
            df[num_cols] = self.num_imputer.fit_transform(df[num_cols])
        if cat_cols:
            df[cat_cols] = self.cat_imputer.fit_transform(df[cat_cols])
        df = self._clip_outliers(df, num_cols)
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        if num_cols:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        self._fitted = True
        self._num_cols = num_cols
        self._cat_cols = cat_cols
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        df = df.copy()
        if self._num_cols:
            df[self._num_cols] = self.num_imputer.transform(df[self._num_cols])
            df = self._clip_outliers(df, self._num_cols)
        if self._cat_cols:
            df[self._cat_cols] = self.cat_imputer.transform(df[self._cat_cols])
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
                )
        if self._num_cols:
            df[self._num_cols] = self.scaler.transform(df[self._num_cols])
        return df

    def add_datetime_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        if date_col not in df.columns:
            return df
        df = df.copy()
        dt = pd.to_datetime(df[date_col], errors="coerce")
        df[f"{date_col}_year"] = dt.dt.year
        df[f"{date_col}_month"] = dt.dt.month
        df[f"{date_col}_dayofweek"] = dt.dt.dayofweek
        df[f"{date_col}_quarter"] = dt.dt.quarter
        df[f"{date_col}_is_month_end"] = dt.dt.is_month_end.astype(int)
        return df

    def add_lag_features(self, df: pd.DataFrame, col: str, sort_col: str) -> pd.DataFrame:
        df = df.sort_values(sort_col).copy()
        for lag in self.lag_windows:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    def add_rolling_features(self, df: pd.DataFrame, col: str, sort_col: str) -> pd.DataFrame:
        df = df.sort_values(sort_col).copy()
        for w in self.rolling_windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(w, min_periods=1).std().fillna(0)
        return df

    def train_test_split_temporal(
        self, df: pd.DataFrame, date_col: str, cutoff: str, val_days: int = 90
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        dt = pd.to_datetime(df[date_col], errors="coerce")
        cutoff_dt = pd.to_datetime(cutoff)
        val_start = cutoff_dt - pd.Timedelta(days=val_days)
        train = df[dt < val_start]
        val = df[(dt >= val_start) & (dt < cutoff_dt)]
        test = df[dt >= cutoff_dt]
        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test
