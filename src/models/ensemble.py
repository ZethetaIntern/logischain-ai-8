"""Stacking ensemble that fuses GNN, XGBoost, LightGBM, and Survival signals."""
import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow

logger = logging.getLogger(__name__)


class LogisChainEnsemble:
    """Stacking ensemble: base models → meta-learner (logistic regression).

    Base model predictions are used as features for the meta-learner.
    Out-of-fold predictions prevent data leakage.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.weights: Dict[str, float] = cfg.get("weights", {
            "xgboost": 0.35,
            "lightgbm": 0.30,
            "gnn": 0.25,
            "survival": 0.10,
        })
        self.meta_learner = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self.base_models: Dict[str, Any] = {}
        self.n_folds: int = cfg.get("cross_val_folds", 5)
        self._fitted = False

    def register_model(self, name: str, model: Any):
        self.base_models[name] = model
        logger.info(f"Registered base model: {name}")

    def _get_oof_predictions(
        self,
        name: str,
        model_class: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_kwargs: Optional[dict] = None,
    ) -> np.ndarray:
        """Get out-of-fold predictions for stacking."""
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            m = model_class(**(model_kwargs or {}))
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof[val_idx] = m.predict_proba(X.iloc[val_idx])
        return oof

    def fit_from_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        y: np.ndarray,
    ) -> "LogisChainEnsemble":
        """Fit meta-learner from a dict of model_name → prediction arrays."""
        model_names = list(predictions.keys())
        meta_X = np.column_stack([predictions[n] for n in model_names])
        self.meta_learner.fit(meta_X, y)
        self._model_names = model_names
        self._fitted = True
        logger.info(f"Meta-learner fitted on {len(model_names)} base model outputs.")
        return self

    def predict_proba_from_predictions(
        self, predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        if not self._fitted:
            # Fall back to weighted average
            return self.weighted_average(predictions)
        meta_X = np.column_stack([predictions[n] for n in self._model_names])
        return self.meta_learner.predict_proba(meta_X)[:, 1]

    def weighted_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple weighted average of model predictions."""
        total_weight = sum(self.weights.get(n, 1.0) for n in predictions)
        combined = np.zeros(len(next(iter(predictions.values()))))
        for name, preds in predictions.items():
            w = self.weights.get(name, 1.0) / total_weight
            combined += w * np.asarray(preds)
        return combined

    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        y: np.ndarray,
        log_to_mlflow: bool = False,
    ) -> dict:
        ensemble_probs = self.predict_proba_from_predictions(predictions)
        weighted_probs = self.weighted_average(predictions)

        metrics = {
            "ensemble_roc_auc": float(roc_auc_score(y, ensemble_probs)),
            "ensemble_avg_precision": float(average_precision_score(y, ensemble_probs)),
            "weighted_avg_roc_auc": float(roc_auc_score(y, weighted_probs)),
        }
        for name, preds in predictions.items():
            metrics[f"{name}_roc_auc"] = float(roc_auc_score(y, preds))

        if log_to_mlflow:
            try:
                with mlflow.start_run(nested=True, run_name="ensemble_eval"):
                    mlflow.log_metrics(metrics)
            except Exception as e:
                logger.debug(f"MLflow logging skipped: {e}")

        logger.info("Ensemble evaluation: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return metrics

    def get_risk_tier(self, prob: float) -> str:
        if prob >= 0.75:
            return "CRITICAL"
        elif prob >= 0.50:
            return "HIGH"
        elif prob >= 0.25:
            return "MEDIUM"
        else:
            return "LOW"

    def score_portfolio(
        self, predictions: Dict[str, np.ndarray], ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        probs = self.predict_proba_from_predictions(predictions)
        n = len(probs)
        ids = ids or [f"ENTITY-{i:04d}" for i in range(n)]
        tiers = [self.get_risk_tier(p) for p in probs]
        return pd.DataFrame({
            "entity_id": ids,
            "risk_score": probs,
            "risk_tier": tiers,
            **{f"{name}_score": preds for name, preds in predictions.items()},
        }).sort_values("risk_score", ascending=False).reset_index(drop=True)
