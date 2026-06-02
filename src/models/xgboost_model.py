"""XGBoost and LightGBM models for supply-chain-aware financial risk prediction.

v0.2.0 — LogisChainXGB
───────────────────────
Full production class with Optuna hyperparameter optimisation, SHAP explainability,
counterfactual generation, calibration (ECE), and portfolio reporting.

v0.1.0 backward-compat classes are kept at the bottom of the file.

MLflow experiment: logischain_ai / xgb_risk
"""

import logging
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

logger = logging.getLogger(__name__)

# ── Feature category keywords ─────────────────────────────────────────────────

_SC_KW = frozenset([
    "otif", "inventory", "carrier", "port", "freight", "supplier", "betweenness",
    "pagerank", "clustering", "transit", "delay", "network", "route", "congestion",
    "fill_rate", "capacity", "lead_time", "concentration", "disruption", "geopolitical",
    "country_risk", "natural_disaster", "port_proximity", "dio", "dso", "dpo",
])
_FIN_KW = frozenset([
    "debt", "equity", "leverage", "altman", "interest_coverage", "credit", "rating",
    "payment", "revenue", "ebitda", "margin", "current_ratio", "quick_ratio",
    "working_capital", "lc_util", "fx_exposure", "default",
])
_FUSION_KW = frozenset([
    "logischain", "sc_risk", "logistics_disruption", "sc_pd", "wcvi", "trfsi",
    "fusion", "stress_index", "composite_risk",
])


def _feature_category(name: str) -> str:
    n = name.lower()
    if any(k in n for k in _FUSION_KW):
        return "fusion"
    if any(k in n for k in _SC_KW):
        return "supply_chain"
    return "financial"


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (lower = better calibrated)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    df = pd.DataFrame({"s": y_score, "y": y_true}).sort_values("s", ascending=False)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    df["cp"] = (df["y"] == 1).cumsum() / n_pos
    df["cn"] = (df["y"] == 0).cumsum() / n_neg
    return float((df["cp"] - df["cn"]).abs().max())


def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_pct: float = 0.05) -> float:
    k = max(1, int(k_pct * len(y_true)))
    top_idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_idx].mean())


def _gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1


class _noop:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ═══════════════════════════════════════════════════════════════════════════════
# LogisChainXGB
# ═══════════════════════════════════════════════════════════════════════════════

class LogisChainXGB:
    """Production XGBoost risk model with Optuna HPO, SHAP explainability, and
    counterfactual generation.

    Supports
    ────────
    - task='classification'   binary default / carrier failure prediction
    - task='regression'       CCC prediction, spread pricing

    Usage
    ─────
    model = LogisChainXGB(task='classification')
    model.fit(X_train, y_train, optimize=True, n_trials=50)
    probs = model.predict_proba(X_test)
    metrics = model.evaluate(X_test, y_test)
    cf = model.generate_counterfactual(X_instance)
    """

    DEFAULT_PARAMS = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "gamma": 0.05,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, task: str = "classification"):
        assert task in ("classification", "regression"), "task must be 'classification' or 'regression'"
        self.task = task
        self.best_params: Optional[dict] = None
        self.model: Optional[xgb.XGBClassifier] = None
        self.shap_explainer = None
        self._shap_values: Optional[np.ndarray] = None
        self._shap_background: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self._fitted = False

    # ── HPO ──────────────────────────────────────────────────────────────────

    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 100,
    ) -> dict:
        """Bayesian hyperparameter search with Optuna + StratifiedKFold(5).

        Search space
        ────────────
        n_estimators       [200, 1000]
        max_depth          [3, 8]
        learning_rate      [0.005, 0.1] log-uniform
        min_child_weight   [1, 50]
        subsample          [0.6, 1.0]
        colsample_bytree   [0.5, 1.0]
        gamma              [0, 0.5]
        reg_alpha          [0, 1.0]
        reg_lambda         [0.5, 3.0]
        scale_pos_weight   fixed at class ratio

        Returns best parameter dict (also stored in self.best_params).
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("optuna not installed — using default params.")
            self.best_params = deepcopy(self.DEFAULT_PARAMS)
            return self.best_params

        n_pos = int(y_train.sum())
        n_neg = int((1 - y_train).sum())
        class_ratio = max(n_neg / max(n_pos, 1), 1.0)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        def objective(trial):
            params = {
                "n_estimators":       trial.suggest_int("n_estimators", 200, 1000),
                "max_depth":          trial.suggest_int("max_depth", 3, 8),
                "learning_rate":      trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "min_child_weight":   trial.suggest_int("min_child_weight", 1, 50),
                "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma":              trial.suggest_float("gamma", 0, 0.5),
                "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda":         trial.suggest_float("reg_lambda", 0.5, 3.0),
                "scale_pos_weight":   class_ratio,
                "tree_method":        "hist",
                "eval_metric":        "logloss",
                "random_state":       42,
                "n_jobs":             -1,
            }
            aucs = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                m = xgb.XGBClassifier(**params, early_stopping_rounds=50, verbosity=0)
                m.fit(
                    X_train.iloc[train_idx], y_train.iloc[train_idx],
                    eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                    verbose=False,
                )
                probs = m.predict_proba(X_train.iloc[val_idx])[:, 1]
                aucs.append(roc_auc_score(y_train.iloc[val_idx], probs))

            mean_auc = float(np.mean(aucs))
            if _MLFLOW:
                try:
                    mlflow.log_metric("trial_auc", mean_auc, step=trial.number)
                except Exception:
                    pass
            return mean_auc

        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.best_params = study.best_params
        self.best_params.update({
            "scale_pos_weight": class_ratio,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        })
        logger.info(f"Optuna best AUC: {study.best_value:.4f} | params: {self.best_params}")
        return self.best_params

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        optimize: bool = True,
        n_trials: int = 100,
    ) -> "LogisChainXGB":
        """Train LogisChainXGB.

        Parameters
        ──────────
        optimize   : run Optuna HPO before final fit (uses StratifiedKFold(5))
        n_trials   : number of Optuna trials (ignored if optimize=False)
        """
        self.feature_names_ = list(X_train.columns)

        if optimize and self.best_params is None:
            self.optimize_hyperparameters(X_train, y_train, n_trials=n_trials)

        params = deepcopy(self.best_params or self.DEFAULT_PARAMS)
        params.setdefault("eval_metric", "logloss")
        params.setdefault("verbosity", 0)

        eval_set = [(X_val, y_val)] if X_val is not None else None
        es_rounds = 50 if eval_set else None

        if self.task == "classification":
            self.model = xgb.XGBClassifier(**params, early_stopping_rounds=es_rounds)
        else:
            reg_params = {k: v for k, v in params.items()
                          if k not in ("scale_pos_weight",)}
            self.model = xgb.XGBRegressor(**reg_params)

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # SHAP explainer
        if SHAP_AVAILABLE:
            try:
                bg = X_train.sample(min(500, len(X_train)), random_state=42)
                self.shap_explainer = shap.TreeExplainer(self.model)
                self._shap_background = bg
                sv = self.shap_explainer.shap_values(bg)
                self._shap_values = sv[1] if isinstance(sv, list) else sv
            except Exception as exc:
                logger.warning(f"SHAP explainer failed: {exc}")

        self._fitted = True

        # MLflow logging
        if _MLFLOW:
            try:
                with mlflow.start_run(run_name="logischain_xgb", nested=True):
                    mlflow.log_params(params)
                    if eval_set:
                        train_auc = roc_auc_score(y_train, self.predict_proba(X_train))
                        mlflow.log_metric("train_auc", train_auc)
            except Exception:
                pass

        logger.info(
            f"LogisChainXGB fitted: {len(X_train)} samples, "
            f"{len(self.feature_names_)} features."
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        if self.task == "classification":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Comprehensive evaluation with 6 metrics.

        Returns
        ───────
        {auc_roc, gini, ks_stat, precision_at_5pct, brier_score, ece}
        """
        y = np.asarray(y_test)
        p = self.predict_proba(X_test)
        return {
            "auc_roc":           round(float(roc_auc_score(y, p)), 4),
            "gini":              round(_gini(y, p), 4),
            "ks_stat":           round(_ks_statistic(y, p), 4),
            "precision_at_5pct": round(_precision_at_k(y, p, 0.05), 4),
            "brier_score":       round(float(brier_score_loss(y, p)), 4),
            "ece":               round(_ece(y, p), 4),
        }

    # ── SHAP explainability ───────────────────────────────────────────────────

    def explain_global(self) -> Optional[plt.Figure]:
        """SHAP beeswarm plot; supply-chain features in orange, others in blue.

        Returns matplotlib Figure or None if SHAP unavailable.
        """
        if not SHAP_AVAILABLE or self._shap_values is None or self._shap_background is None:
            logger.warning("SHAP not available or model not fitted with SHAP support.")
            return None

        fig, ax = plt.subplots(figsize=(10, 8))
        try:
            shap.summary_plot(
                self._shap_values,
                self._shap_background,
                feature_names=self.feature_names_,
                plot_type="beeswarm",
                show=False,
                max_display=20,
            )
        except Exception:
            # Fallback bar chart
            imp = self.feature_importance_report()
            ax.barh(imp["feature"].head(20), imp["shap_mean"].head(20))
            ax.set_xlabel("Mean |SHAP|")
            ax.set_title("Feature Importance (|SHAP|)")

        plt.tight_layout()
        fig = plt.gcf()
        plt.close("all")
        return fig

    def explain_local(
        self,
        X_instance: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> dict:
        """SHAP waterfall for a single prediction.

        Returns
        ───────
        {
            'base_value'  : float,
            'shap_values' : {feature: shap_value},
            'prediction'  : float,
            'top_factors' : [(label_str, +/-pct_str), ...] top 5
        }
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            p = float(self.predict_proba(X_instance)[0])
            return {"base_value": 0.5, "shap_values": {}, "prediction": p, "top_factors": []}

        sv = self.shap_explainer.shap_values(X_instance)
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv.flatten()
        names = feature_names or self.feature_names_ or [f"f{i}" for i in range(len(sv))]

        base = float(self.shap_explainer.expected_value[1]
                     if isinstance(self.shap_explainer.expected_value, (list, np.ndarray))
                     else self.shap_explainer.expected_value)
        pred = float(self.predict_proba(X_instance)[0])
        shap_dict = {names[i]: float(sv[i]) for i in range(min(len(names), len(sv)))}

        # Format top factors with human-readable labels
        top_k = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_factors = []
        x_vals = X_instance.values.flatten()
        for fname, sval in top_k:
            idx = names.index(fname) if fname in names else -1
            val_raw = float(x_vals[idx]) if 0 <= idx < len(x_vals) else 0.0
            label = self._format_factor_label(fname, val_raw)
            pct_str = f"{sval * 100:+.2f}%"
            top_factors.append((label, pct_str))

        return {
            "base_value":  round(base, 4),
            "shap_values": {k: round(v, 6) for k, v in shap_dict.items()},
            "prediction":  round(pred, 4),
            "top_factors": top_factors,
        }

    @staticmethod
    def _format_factor_label(feature_name: str, value: float) -> str:
        n = feature_name.lower()
        if "otif" in n or "fill_rate" in n or "capacity" in n:
            return f"{feature_name} ({value * 100:.0f}%)"
        if "ccc" in n or "dso" in n or "dpo" in n or "dio" in n or "lead_time" in n or "transit" in n:
            return f"{feature_name} ({value:.0f} days)"
        if "hhi" in n:
            return f"{feature_name} ({value:.0f})"
        if 0 <= value <= 1 and ("rate" in n or "score" in n or "ratio" in n):
            return f"{feature_name} ({value * 100:.0f}%)"
        return f"{feature_name} ({value:.2g})"

    # ── Counterfactual generation ─────────────────────────────────────────────

    def generate_counterfactual(
        self,
        X_instance: pd.DataFrame,
        target_outcome: int = 0,
        max_changes: int = 3,
    ) -> dict:
        """Find minimal feature changes to flip prediction toward target_outcome.

        Algorithm
        ─────────
        1. Get current prediction and SHAP values.
        2. Sort features by risk contribution (positive SHAP for target=0).
        3. Binary-search each feature's required change to cross the threshold.
        4. Stop when target achieved or max_changes reached.

        Returns
        ───────
        {
            'current_risk'  : float,
            'target_risk'   : float,
            'changes_needed': [{'feature', 'current', 'needed', 'change'}, ...],
            'explanation'   : str
        }
        """
        X = np.asarray(X_instance).reshape(1, -1)
        names = self.feature_names_ or [f"f{i}" for i in range(X.shape[1])]
        current_pred = float(self.predict_proba(pd.DataFrame(X, columns=names))[0])
        threshold = 0.5

        # Already at target
        at_target = (target_outcome == 0 and current_pred < threshold) or \
                    (target_outcome == 1 and current_pred >= threshold)
        if at_target:
            return {
                "current_risk": round(current_pred, 4),
                "target_risk": round(current_pred, 4),
                "changes_needed": [],
                "explanation": "Prediction already at target — no changes needed.",
            }

        # Get SHAP contributions
        shap_vals = np.zeros(X.shape[1])
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            try:
                sv = self.shap_explainer.shap_values(
                    pd.DataFrame(X, columns=names)
                )
                shap_vals = (sv[1] if isinstance(sv, list) else sv).flatten()
            except Exception:
                pass

        # For target=0 (reduce risk): attack high positive SHAP features
        # For target=1 (increase risk): attack high negative SHAP features
        if target_outcome == 0:
            sorted_feats = sorted(
                [(i, v) for i, v in enumerate(shap_vals) if v > 0],
                key=lambda x: -x[1]
            )
        else:
            sorted_feats = sorted(
                [(i, v) for i, v in enumerate(shap_vals) if v < 0],
                key=lambda x: x[1]
            )

        # Fallback: use all features sorted by |SHAP|
        if not sorted_feats:
            sorted_feats = sorted(enumerate(shap_vals), key=lambda x: -abs(x[1]))

        changes = []
        X_mod = X.copy()

        for feat_idx, shap_val in sorted_feats[:max_changes * 2]:
            if len(changes) >= max_changes:
                break
            fname = names[feat_idx]
            current_val = float(X[0, feat_idx])

            # Determine change direction and search range
            if target_outcome == 0:
                lo_frac, hi_frac = 0.3, 1.0  # reduce feature value
                best_new = current_val * 0.7
            else:
                lo_frac, hi_frac = 1.0, 2.0  # increase feature value
                best_new = current_val * 1.4

            # Binary search for minimal change
            lo = current_val * lo_frac
            hi = current_val * hi_frac
            for _ in range(25):
                mid = (lo + hi) / 2
                X_test = X_mod.copy()
                X_test[0, feat_idx] = mid
                p = float(self.predict_proba(pd.DataFrame(X_test, columns=names))[0])
                if target_outcome == 0:
                    if p < threshold:
                        hi = mid; best_new = mid
                    else:
                        lo = mid
                else:
                    if p >= threshold:
                        lo = mid; best_new = mid
                    else:
                        hi = mid

            X_mod[0, feat_idx] = best_new
            change_pct = (best_new - current_val) / (abs(current_val) + 1e-9) * 100
            changes.append({
                "feature": fname,
                "current": round(current_val, 4),
                "needed":  round(best_new, 4),
                "change":  f"{change_pct:+.0f}%",
            })

            final_pred = float(self.predict_proba(pd.DataFrame(X_mod, columns=names))[0])
            if target_outcome == 0 and final_pred < threshold:
                break
            if target_outcome == 1 and final_pred >= threshold:
                break

        final_pred = float(self.predict_proba(pd.DataFrame(X_mod, columns=names))[0])
        explanation = self._counterfactual_explanation(changes, current_pred, final_pred)
        return {
            "current_risk":   round(current_pred, 4),
            "target_risk":    round(final_pred, 4),
            "changes_needed": changes,
            "explanation":    explanation,
        }

    @staticmethod
    def _counterfactual_explanation(changes: list, before: float, after: float) -> str:
        if not changes:
            return "No changes identified."
        parts = [f"{c['feature']} → {c['needed']:.4g} (change: {c['change']})"
                 for c in changes]
        verb = "reduced to below approval threshold" if after < 0.5 else "increased above threshold"
        return (
            f"If {' and '.join(parts)}, "
            f"the risk score would be {verb} "
            f"({before:.0%} → {after:.0%})."
        )

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance_report(self) -> pd.DataFrame:
        """DataFrame sorted by mean |SHAP| with feature categories.

        Columns: feature, importance (native), category, shap_mean
        """
        if self.model is None:
            return pd.DataFrame()

        names = self.feature_names_ or []
        imp = self.model.feature_importances_ if hasattr(self.model, "feature_importances_") else []

        rows = []
        for i, name in enumerate(names):
            native_imp = float(imp[i]) if i < len(imp) else 0.0
            shap_m = float(np.mean(np.abs(self._shap_values[:, i]))) \
                if self._shap_values is not None and i < self._shap_values.shape[1] else native_imp
            rows.append({
                "feature":   name,
                "importance": native_imp,
                "category":   _feature_category(name),
                "shap_mean":  shap_m,
            })

        df = pd.DataFrame(rows).sort_values("shap_mean", ascending=False).reset_index(drop=True)

        # Print table
        sc_top10 = sum(1 for r in df.head(10)["category"] if r == "supply_chain")
        logger.info(
            f"Feature importance: SC features occupy {sc_top10}/10 top positions."
        )
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# Backward-compatible v0.1.0 classes
# ═══════════════════════════════════════════════════════════════════════════════

class XGBoostRiskModel:
    """v0.1.0 XGBoost classifier. Use LogisChainXGB for production."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.params = {
            "n_estimators":    cfg.get("n_estimators", 800),
            "max_depth":       cfg.get("max_depth", 6),
            "learning_rate":   cfg.get("learning_rate", 0.02),
            "subsample":       cfg.get("subsample", 0.8),
            "colsample_bytree": cfg.get("colsample_bytree", 0.8),
            "min_child_weight": cfg.get("min_child_weight", 5),
            "reg_alpha":       cfg.get("reg_alpha", 0.1),
            "reg_lambda":      cfg.get("reg_lambda", 1.0),
            "gamma":           cfg.get("gamma", 0.05),
            "eval_metric":     cfg.get("eval_metric", "logloss"),
            "tree_method":     "hist",
            "random_state":    42,
            "n_jobs":          -1,
        }
        self.early_stopping_rounds = cfg.get("early_stopping_rounds", 50)
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: Optional[list] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
            verbose=False,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X, y) -> dict:
        p = self.predict_proba(X)
        preds = (p >= 0.5).astype(int)
        return {
            "roc_auc":       float(roc_auc_score(y, p)),
            "avg_precision": float(average_precision_score(y, p)),
            "f1":            float(f1_score(y, preds, zero_division=0)),
        }

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None or self.feature_names is None:
            return pd.DataFrame()
        return (
            pd.DataFrame({"feature": self.feature_names,
                          "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def cross_validate(self, X, y, n_folds: int = 5) -> dict:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        aucs = []
        for _, (tr, vl) in enumerate(skf.split(X, y)):
            m = xgb.XGBClassifier(**self.params)
            m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
            aucs.append(roc_auc_score(y.iloc[vl], m.predict_proba(X.iloc[vl])[:, 1]))
        return {"cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs))}


class LightGBMRiskModel:
    """v0.1.0 LightGBM classifier. Use LogisChainXGB for production."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.params = {
            "n_estimators":      cfg.get("n_estimators", 800),
            "max_depth":         cfg.get("max_depth", 7),
            "learning_rate":     cfg.get("learning_rate", 0.02),
            "num_leaves":        cfg.get("num_leaves", 63),
            "subsample":         cfg.get("subsample", 0.8),
            "colsample_bytree":  cfg.get("colsample_bytree", 0.8),
            "min_child_samples": cfg.get("min_child_samples", 20),
            "reg_alpha":         cfg.get("reg_alpha", 0.1),
            "reg_lambda":        cfg.get("reg_lambda", 1.0),
            "objective":         "binary",
            "metric":            "auc",
            "random_state":      42,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        self.early_stopping = cfg.get("early_stopping_rounds", 50)
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: Optional[list] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
        callbacks = [lgb.early_stopping(self.early_stopping, verbose=False),
                     lgb.log_evaluation(0)]
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks if eval_set else [lgb.log_evaluation(0)],
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y) -> dict:
        p = self.predict_proba(X)
        preds = (p >= 0.5).astype(int)
        return {
            "roc_auc":       float(roc_auc_score(y, p)),
            "avg_precision": float(average_precision_score(y, p)),
            "f1":            float(f1_score(y, preds, zero_division=0)),
        }

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None or self.feature_names is None:
            return pd.DataFrame()
        return (
            pd.DataFrame({"feature": self.feature_names,
                          "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    from src.data.pipeline import SyntheticDataGenerator
    from src.features.fusion_features import FeaturePipeline

    print("LogisChain AI — LogisChainXGB on synthetic data")
    gen = SyntheticDataGenerator(seed=42)
    data = gen.generate_all()
    fp = FeaturePipeline()
    fused = fp.run(data["carriers"], data["shipments"], data["financial"])

    target = "carrier_failure" if "carrier_failure" in fused.columns else "default_flag"
    drop = [target, "carrier_id", "company_id", "carrier_type", "region", "industry", "name"]
    X = fused.drop(columns=[c for c in drop if c in fused.columns]).select_dtypes(include=np.number).fillna(0)
    y = fused[target].fillna(0)

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = LogisChainXGB(task="classification")
    model.fit(X_tr, y_tr, optimize=False)   # fast run without Optuna

    metrics = model.evaluate(X_te, y_te)
    print("\n── Evaluation ────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.4f}")

    imp = model.feature_importance_report()
    print(f"\n── Top 10 features ────────────────────────────")
    print(imp.head(10)[["feature", "category", "shap_mean"]].to_string(index=False))

    if len(X_te) > 0:
        cf = model.generate_counterfactual(X_te.iloc[:1], target_outcome=0)
        print(f"\n── Counterfactual ─────────────────────────────")
        print(f"  Current risk: {cf['current_risk']:.3f} → Target: {cf['target_risk']:.3f}")
        for c in cf["changes_needed"]:
            print(f"  {c['feature']}: {c['current']} → {c['needed']} ({c['change']})")
        print(f"  {cf['explanation']}")
