"""SHAP-based explainability for LogisChain AI models."""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed. Explainability module in stub mode.")


class LogisChainExplainer:
    """Unified SHAP explainer for all LogisChain models.

    Supports: TreeExplainer (XGBoost/LightGBM), DeepExplainer (PyTorch),
    KernelExplainer (any black-box model).
    """

    def __init__(self, model, model_type: str = "tree"):
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    def fit(self, X_background: pd.DataFrame, nsamples: int = 100):
        """Initialise the SHAP explainer with background data."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available.")
            return self
        self.feature_names = list(X_background.columns)
        if self.model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == "linear":
            self.explainer = shap.LinearExplainer(self.model, X_background)
        else:
            background = shap.sample(X_background, min(nsamples, len(X_background)))
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba
                if hasattr(self.model, "predict_proba")
                else self.model.predict,
                background,
            )
        return self

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        if not SHAP_AVAILABLE or self.explainer is None:
            return np.zeros((len(X), X.shape[1]))
        self.shap_values = self.explainer.shap_values(X)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # positive class
        return self.shap_values

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        shap_vals = self.explain(X)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        names = self.feature_names or [f"f{i}" for i in range(len(mean_abs))]
        return (
            pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def local_explanation(
        self, X: pd.DataFrame, idx: int
    ) -> pd.DataFrame:
        shap_vals = self.explain(X)
        row_shap = shap_vals[idx]
        names = self.feature_names or [f"f{i}" for i in range(len(row_shap))]
        feature_values = X.iloc[idx].values
        return (
            pd.DataFrame(
                {
                    "feature": names,
                    "shap_value": row_shap,
                    "feature_value": feature_values,
                }
            )
            .sort_values("shap_value", key=abs, ascending=False)
            .reset_index(drop=True)
        )

    def sc_vs_financial_decomposition(
        self, X: pd.DataFrame
    ) -> Dict[str, float]:
        """Decompose SHAP values into supply chain vs financial drivers."""
        shap_vals = self.explain(X)
        names = self.feature_names or []

        sc_keywords = [
            "sc_risk", "logistics", "carrier", "inventory_risk",
            "route_concentration", "demand_vol", "disruption",
            "network_centrality", "on_time", "delay", "port_congestion",
            "shipment", "transit", "supply_chain",
        ]
        fin_keywords = [
            "altman", "debt", "interest", "current_ratio", "quick_ratio",
            "credit", "rating", "cash_conversion", "dso", "dpo", "dio",
            "revenue", "gross_margin", "lc_util", "fx_exposure", "payment_terms",
        ]

        sc_mask = np.array([
            any(kw in name.lower() for kw in sc_keywords) for name in names
        ], dtype=bool)
        fin_mask = ~sc_mask

        mean_abs = np.abs(shap_vals).mean(axis=0)
        total = mean_abs.sum() + 1e-8

        return {
            "supply_chain_shap_pct": float(mean_abs[sc_mask].sum() / total * 100),
            "financial_shap_pct": float(mean_abs[fin_mask].sum() / total * 100),
            "top_sc_driver": names[np.argmax(mean_abs * sc_mask)] if sc_mask.any() else "N/A",
            "top_fin_driver": names[np.argmax(mean_abs * fin_mask)] if fin_mask.any() else "N/A",
        }

    def generate_explanation_text(
        self, local_df: pd.DataFrame, entity_id: str, pd_value: float
    ) -> str:
        top3 = local_df.head(3)
        lines = [
            f"**{entity_id}** — PD: {pd_value*100:.2f}%",
            "",
            "Key risk drivers:",
        ]
        for _, row in top3.iterrows():
            direction = "increases" if row["shap_value"] > 0 else "decreases"
            lines.append(
                f"  • **{row['feature']}** = {row['feature_value']:.3f} "
                f"→ {direction} default risk by {abs(row['shap_value']):.4f} SHAP units"
            )
        return "\n".join(lines)
