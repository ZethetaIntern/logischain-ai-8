"""Custom evaluation metrics for LogisChain AI models."""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    brier_score_loss, log_loss, confusion_matrix,
)


def classification_report_dict(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "avg_precision": float(average_precision_score(y_true, y_pred_proba)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "brier_score": float(brier_score_loss(y_true, y_pred_proba)),
        "log_loss": float(log_loss(y_true, y_pred_proba)),
        "ks_statistic": float(ks_statistic(y_true, y_pred_proba)),
        "gini": float(gini_coefficient(y_true, y_pred_proba)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def regression_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100),
        "smape": float(
            np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8)) * 100
        ),
        "bias": float(np.mean(y_pred - y_true)),
    }


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic — separation between defaulters and non-defaulters."""
    df = pd.DataFrame({"score": y_score, "label": y_true}).sort_values("score", ascending=False)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    df["cum_pos"] = (df["label"] == 1).cumsum() / n_pos
    df["cum_neg"] = (df["label"] == 0).cumsum() / n_neg
    return float((df["cum_pos"] - df["cum_neg"]).abs().max())


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini coefficient = 2 * AUC - 1."""
    return 2 * roc_auc_score(y_true, y_score) - 1


def information_value(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> float:
    """Weight of Evidence / Information Value for scorecard assessment."""
    df = pd.DataFrame({"score": y_score, "label": y_true})
    df["bin"] = pd.qcut(df["score"], n_bins, duplicates="drop", labels=False)
    n_events = y_true.sum()
    n_non_events = len(y_true) - n_events
    iv = 0.0
    for _, grp in df.groupby("bin"):
        pct_e = grp["label"].sum() / max(n_events, 1)
        pct_ne = (len(grp) - grp["label"].sum()) / max(n_non_events, 1)
        if pct_e > 0 and pct_ne > 0:
            woe = np.log(pct_e / pct_ne)
            iv += (pct_e - pct_ne) * woe
    return float(iv)


def portfolio_var(losses: np.ndarray, confidence: float = 0.99) -> float:
    """Value at Risk at given confidence level."""
    return float(np.percentile(losses, confidence * 100))


def portfolio_cvar(losses: np.ndarray, confidence: float = 0.99) -> float:
    """Conditional Value at Risk (Expected Shortfall)."""
    var = portfolio_var(losses, confidence)
    return float(losses[losses >= var].mean())


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.05 / 252) -> float:
    excess = returns - risk_free
    return float(excess.mean() / (excess.std() + 1e-8) * np.sqrt(252))


def supply_chain_disruption_detection_rate(
    y_true_disruptions: np.ndarray,
    y_pred_disruptions: np.ndarray,
    lead_time_days: int = 7,
) -> dict:
    """Measures how early the model detects supply chain disruptions."""
    tp = int(np.sum((y_true_disruptions == 1) & (y_pred_disruptions == 1)))
    fn = int(np.sum((y_true_disruptions == 1) & (y_pred_disruptions == 0)))
    fp = int(np.sum((y_true_disruptions == 0) & (y_pred_disruptions == 1)))
    detection_rate = tp / max(tp + fn, 1)
    false_alarm_rate = fp / max(fp + len(y_true_disruptions) - (tp + fn), 1)
    return {
        "detection_rate": float(detection_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "early_warning_score": float(detection_rate - 0.5 * false_alarm_rate),
    }
