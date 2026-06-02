"""Survival analysis for carrier reliability modelling.

Uses lifelines Cox Proportional Hazards and Kaplan-Meier estimators to model
time-to-failure for logistics carriers, translating to credit risk durations.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from lifelines import (
        CoxPHFitter,
        KaplanMeierFitter,
        WeibullAFTFitter,
        LogNormalAFTFitter,
    )
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logger.warning("lifelines not available. Survival module in stub mode.")


class CarrierSurvivalModel:
    """Cox PH + Kaplan-Meier survival analysis on carrier reliability data."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.duration_col = cfg.get("duration_col", "carrier_tenure_days")
        self.event_col = cfg.get("event_col", "carrier_failure")
        self.penalizer = cfg.get("penalizer", 0.1)
        self.l1_ratio = cfg.get("l1_ratio", 0.0)
        self.alpha = cfg.get("alpha", 0.05)
        self.cph: Optional[object] = None
        self.kmf: Optional[object] = None
        self.weibull: Optional[object] = None

    def fit(
        self,
        df: pd.DataFrame,
        covariate_cols: Optional[List[str]] = None,
    ) -> "CarrierSurvivalModel":
        if not LIFELINES_AVAILABLE:
            logger.warning("lifelines unavailable. fit() skipped.")
            return self

        required = {self.duration_col, self.event_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Kaplan-Meier (unconditional)
        self.kmf = KaplanMeierFitter(alpha=self.alpha)
        self.kmf.fit(
            df[self.duration_col],
            event_observed=df[self.event_col],
            label="All Carriers",
        )

        # Cox PH (conditional on covariates)
        cols = [self.duration_col, self.event_col]
        if covariate_cols:
            cols += [c for c in covariate_cols if c in df.columns]
        df_fit = df[cols].dropna()
        self.cph = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.cph.fit(
            df_fit,
            duration_col=self.duration_col,
            event_col=self.event_col,
        )

        # Weibull AFT
        self.weibull = WeibullAFTFitter()
        self.weibull.fit(
            df_fit,
            duration_col=self.duration_col,
            event_col=self.event_col,
        )

        logger.info(
            f"Survival models fitted on {len(df_fit)} carriers. "
            f"Concordance: {self.cph.concordance_index_:.4f}"
        )
        return self

    def predict_survival_at(
        self, df: pd.DataFrame, times: List[int]
    ) -> pd.DataFrame:
        """Predict survival probability at given time points for each row."""
        if not LIFELINES_AVAILABLE or self.cph is None:
            n = len(df)
            return pd.DataFrame(
                {f"S(t={t})": np.ones(n) * 0.5 for t in times}
            )
        sf = self.cph.predict_survival_function(df, times=times)
        return sf.T.reset_index(drop=True)

    def predict_median_lifetime(self, df: pd.DataFrame) -> np.ndarray:
        """Predict median survival time (days) for each carrier."""
        if not LIFELINES_AVAILABLE or self.cph is None:
            return np.full(len(df), 365.0)
        return self.cph.predict_median(df).values

    def predict_cumulative_hazard(
        self, df: pd.DataFrame, times: List[int]
    ) -> pd.DataFrame:
        if not LIFELINES_AVAILABLE or self.cph is None:
            return pd.DataFrame(
                {f"H(t={t})": np.zeros(len(df)) for t in times}
            )
        ch = self.cph.predict_cumulative_hazard(df, times=times)
        return ch.T.reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        if not LIFELINES_AVAILABLE or self.cph is None:
            return pd.DataFrame()
        return self.cph.summary.reset_index()

    def plot_baseline_hazard(self):
        if not LIFELINES_AVAILABLE or self.cph is None:
            return None
        return self.cph.baseline_hazard_

    def group_comparison(
        self,
        df: pd.DataFrame,
        group_col: str,
    ) -> dict:
        """Log-rank test comparing survival curves across groups."""
        if not LIFELINES_AVAILABLE:
            return {}
        groups = df[group_col].unique()
        if len(groups) < 2:
            return {}
        g0 = df[df[group_col] == groups[0]]
        g1 = df[df[group_col] == groups[1]]
        result = logrank_test(
            g0[self.duration_col],
            g1[self.duration_col],
            g0[self.event_col],
            g1[self.event_col],
            alpha=self.alpha,
        )
        return {
            "test_statistic": float(result.test_statistic),
            "p_value": float(result.p_value),
            "significant": bool(result.p_value < self.alpha),
        }

    def carrier_risk_score(self, df: pd.DataFrame, horizon: int = 365) -> np.ndarray:
        """Return [0,1] risk score = 1 - P(survival at horizon)."""
        sf = self.predict_survival_at(df, [horizon])
        return 1.0 - sf.values.flatten()
