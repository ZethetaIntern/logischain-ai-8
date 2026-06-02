"""Financial feature engineering: trade finance, working capital, credit risk."""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TradeFinanceFeatureExtractor:
    """Extracts and derives trade finance instrument features."""

    def extract(self, financial: pd.DataFrame) -> pd.DataFrame:
        df = financial.copy()

        # Letter of Credit utilisation tier
        if "lc_utilization_rate" in df.columns:
            df["lc_util_tier"] = pd.cut(
                df["lc_utilization_rate"],
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=["Low", "Medium", "High", "Critical"],
            ).astype(str)

        # Payment terms bucket
        if "payment_terms_days" in df.columns:
            df["payment_terms_bucket"] = pd.cut(
                df["payment_terms_days"],
                bins=[0, 30, 60, 90, 365],
                labels=["30d", "60d", "90d", "120d+"],
            ).astype(str)

        # FX exposure normalised by revenue
        if "fx_exposure_usd" in df.columns and "revenue_usd" in df.columns:
            df["fx_exposure_to_revenue"] = df["fx_exposure_usd"] / df["revenue_usd"].clip(lower=1)

        # Trade finance cost as % of revenue
        if "cogs_usd" in df.columns and "revenue_usd" in df.columns:
            df["cogs_margin"] = df["cogs_usd"] / df["revenue_usd"].clip(lower=1)

        return df


class WorkingCapitalFeatureExtractor:
    """Derives CCC and working capital efficiency features."""

    def extract(self, financial: pd.DataFrame) -> pd.DataFrame:
        df = financial.copy()

        # Cash Conversion Cycle
        if all(c in df.columns for c in ["days_sales_outstanding", "days_inventory_outstanding", "days_payable_outstanding"]):
            df["cash_conversion_cycle"] = (
                df["days_sales_outstanding"]
                + df["days_inventory_outstanding"]
                - df["days_payable_outstanding"]
            )

        # Working capital ratio
        if "current_ratio" in df.columns:
            df["working_capital_efficiency"] = 1.0 / df["current_ratio"].clip(lower=0.01)

        # CCC buckets
        if "cash_conversion_cycle" in df.columns:
            df["ccc_bucket"] = pd.cut(
                df["cash_conversion_cycle"],
                bins=[-np.inf, 30, 60, 90, 120, np.inf],
                labels=["Excellent", "Good", "Fair", "Poor", "Critical"],
            ).astype(str)

        # DIO / DSO ratio (inventory dominance)
        if "days_inventory_outstanding" in df.columns and "days_sales_outstanding" in df.columns:
            df["inventory_to_receivables_ratio"] = (
                df["days_inventory_outstanding"] / df["days_sales_outstanding"].clip(lower=1)
            )

        # Payables leverage
        if "days_payable_outstanding" in df.columns and "days_sales_outstanding" in df.columns:
            df["payables_leverage"] = (
                df["days_payable_outstanding"] / df["days_sales_outstanding"].clip(lower=1)
            )

        return df


class CreditRiskFeatureExtractor:
    """Extracts and enhances credit risk features."""

    # Altman Z-score thresholds
    SAFE_ZONE = 2.99
    DISTRESS_ZONE = 1.81

    # Rating to numeric score mapping
    RATING_MAP = {
        "AAA": 1, "AA": 2, "A": 3, "BBB": 4,
        "BB": 5, "B": 6, "CCC": 7, "CC": 8, "D": 9,
    }

    def extract(self, financial: pd.DataFrame) -> pd.DataFrame:
        df = financial.copy()

        # Altman Z-score zone classification
        if "altman_z_score" in df.columns:
            df["altman_zone"] = df["altman_z_score"].apply(self._classify_altman)
            df["altman_distress_flag"] = (df["altman_z_score"] < self.DISTRESS_ZONE).astype(int)

        # Numeric credit rating
        if "credit_rating" in df.columns:
            df["credit_rating_numeric"] = (
                df["credit_rating"].str.upper().map(self.RATING_MAP).fillna(9)
            )
            df["investment_grade"] = (df["credit_rating_numeric"] <= 4).astype(int)

        # Leverage risk
        if "debt_to_equity" in df.columns:
            df["high_leverage_flag"] = (df["debt_to_equity"] > 3.0).astype(int)
            df["leverage_tier"] = pd.cut(
                df["debt_to_equity"].clip(upper=10),
                bins=[0, 1, 2, 3, 5, 10],
                labels=["Conservative", "Moderate", "Elevated", "High", "Extreme"],
            ).astype(str)

        # Interest coverage risk
        if "interest_coverage" in df.columns:
            df["coverage_risk_flag"] = (df["interest_coverage"] < 1.5).astype(int)

        # Composite credit stress index
        stress_components = []
        if "altman_distress_flag" in df.columns:
            stress_components.append(df["altman_distress_flag"] * 0.3)
        if "high_leverage_flag" in df.columns:
            stress_components.append(df["high_leverage_flag"] * 0.25)
        if "coverage_risk_flag" in df.columns:
            stress_components.append(df["coverage_risk_flag"] * 0.25)
        if "credit_rating_numeric" in df.columns:
            stress_components.append((df["credit_rating_numeric"] / 9) * 0.2)

        if stress_components:
            df["credit_stress_index"] = sum(stress_components).clip(0, 1)

        return df

    @staticmethod
    def _classify_altman(z: float) -> str:
        if z >= 2.99:
            return "Safe"
        elif z >= 1.81:
            return "Grey"
        else:
            return "Distress"

    def pd_from_z_score(self, z_scores: pd.Series) -> pd.Series:
        """Logistic mapping from Altman Z-score to probability of default."""
        return 1 / (1 + np.exp(0.5 * (z_scores - 1.81)))
