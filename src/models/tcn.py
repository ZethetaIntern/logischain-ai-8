"""Temporal Convolutional Network for multi-horizon supply chain forecasting.

Architecture
────────────
LogisChainTCN: 7 stacked TCNResidualBlocks with dilations [1,2,4,8,16,32,64].
Receptive field ≈ (kernel_size-1) × 2 × Σ(dilations) = 2 × 2 × 127 = 508 steps.
Three quantile output heads (P10 / P50 / P90) per forecast horizon (30d, 60d, 90d).

TemporalFeatureExtractor: 42 temporal features covering rolling stats, EWMA,
lags, calendar, holiday indicators, and Fourier seasonality.

SupplyChainForecaster: end-to-end training, multi-horizon prediction,
walk-forward backtesting, inventory depletion, and payment timing prediction.

MLflow experiment: logischain_ai / tcn_forecaster
"""

import logging
import math
import os
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

logger = logging.getLogger(__name__)

# ── Chinese New Year windows (year → (cny_date, offset_days)) ─────────────────
_CNY_DATES = {
    2019: "2019-02-05", 2020: "2020-01-25",
    2021: "2021-02-12", 2022: "2022-02-01",
    2023: "2023-01-22", 2024: "2024-02-10",
}

# ── Dummy context for optional MLflow ─────────────────────────────────────────
class _noop:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  TCN Building Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class TCNResidualBlock(nn.Module):
    """Single dilated causal convolution residual block.

    Architecture
    ────────────
    Input ──► CausalPad ──► WeightNorm(Conv1d) ──► ReLU ──► Dropout
          ──► CausalPad ──► WeightNorm(Conv1d) ──► ReLU ──► Dropout ──► + ──► ReLU
              Residual (1×1 conv if channels differ) ──────────────────────────►

    All convolutions are dilated and causal (only left-padding, never future leakage).
    Weight normalization accelerates convergence on non-stationary supply chain series.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self._padding = (kernel_size - 1) * dilation
        self.dropout = dropout

        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, dilation=dilation, padding=0)
        )
        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, dilation=dilation, padding=0)
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 1×1 projection if channel dimensions differ
        self.downsample = (
            weight_norm(nn.Conv1d(n_inputs, n_outputs, 1))
            if n_inputs != n_outputs
            else None
        )
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── First dilated causal conv ──
        out = F.pad(x, (self._padding, 0))   # left-only causal pad
        out = self.conv1(out)                 # (B, C_out, T)
        out = self.relu(out)
        out = self.drop1(out)

        # ── Second dilated causal conv ──
        out = F.pad(out, (self._padding, 0))
        out = self.conv2(out)
        out = self.relu(out)
        out = self.drop2(out)

        # ── Residual ──
        res = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + res)


class QuantileLoss(nn.Module):
    """Pinball (quantile) loss for distributional forecasting.

    L_q(y, ŷ) = max(q·(y−ŷ), (q−1)·(y−ŷ))
    """

    def __init__(self, quantiles: List[float] = (0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ──────────
        preds   : (B, n_quantiles)
        targets : (B,)
        """
        total = torch.tensor(0.0, device=preds.device)
        for i, q in enumerate(self.quantiles):
            err = targets - preds[:, i]
            total = total + torch.max(q * err, (q - 1) * err).mean()
        return total / len(self.quantiles)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  LogisChainTCN
# ═══════════════════════════════════════════════════════════════════════════════

class LogisChainTCN(nn.Module):
    """Pure-PyTorch TCN for multi-horizon quantile forecasting.

    Parameters
    ──────────
    input_channels   : number of input features (≥1)
    hidden_channels  : channel width of all residual blocks (default 64)
    kernel_size      : dilated conv kernel size (default 3)
    dilation_base    : exponential dilation base; dilations = [2^i for i in 0..6]
    num_layers       : number of residual blocks (default 7 → RF ≈ 508 steps)
    dropout          : dropout applied inside each block
    forecast_horizons: list of day-ahead prediction targets [30, 60, 90]
    quantiles        : quantile levels to output per horizon [P10, P50, P90]

    Forward input  : (batch, input_channels, time_steps)
    Forward output : dict{'30d': (batch,3), '60d': (batch,3), '90d': (batch,3)}
                     where each tensor = [P10, P50, P90]
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        dilation_base: int = 2,
        num_layers: int = 7,
        dropout: float = 0.2,
        forecast_horizons: List[int] = (30, 60, 90),
        quantiles: List[float] = (0.1, 0.5, 0.9),
    ):
        super().__init__()
        self.forecast_horizons = list(forecast_horizons)
        self.quantiles = list(quantiles)
        self.n_quantiles = len(quantiles)
        self.hidden_channels = hidden_channels

        # Compute dilations
        dilations = [dilation_base ** i for i in range(num_layers)]

        # Stack of residual blocks
        blocks = []
        in_ch = input_channels
        for dil in dilations:
            blocks.append(TCNResidualBlock(in_ch, hidden_channels, kernel_size, dil, dropout))
            in_ch = hidden_channels
        self.network = nn.Sequential(*blocks)

        # One quantile head per forecast horizon
        self.heads = nn.ModuleDict(
            {
                f"{h}d": nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_channels // 2, self.n_quantiles),
                )
                for h in self.forecast_horizons
            }
        )
        self._receptive_field = (
            1 + 2 * (kernel_size - 1) * sum(dilations)
        )
        logger.info(
            f"LogisChainTCN: {num_layers} blocks | "
            f"RF ≈ {self._receptive_field} steps | "
            f"horizons={self.forecast_horizons}d"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x : (B, input_channels, T)
        Returns dict {'{h}d': (B, n_quantiles)} for each horizon.
        """
        h = self.network(x)           # (B, hidden_channels, T)
        last = h[:, :, -1]            # (B, hidden_channels)
        return {f"{h}d": self.heads[f"{h}d"](last) for h in self.forecast_horizons}


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TemporalFeatureExtractor  (42 features)
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalFeatureExtractor:
    """Extracts 42 temporal features from a univariate time series DataFrame.

    Feature groups (42 total)
    ─────────────────────────
    Rolling mean   [7,14,30,90]   4
    Rolling std    [7,14,30,90]   4
    Rolling min    [7,14,30,90]   4
    Rolling max    [7,14,30,90]   4
    EWMA           [7,14,30]      3
    Lag            [1,7,14,30]    4
    Day-of-week one-hot [0..6]    7
    Month sin + cos               2
    Chinese New Year (±14d)       1
    Golden Week (Oct 1-7)         1
    Year-over-year % change       1
    Fourier sin+cos [annual,
     semi-annual, quarterly]      6
    is_weekend                    1
    ─────────────────────────────
    Total                        42
    """

    WINDOWS = [7, 14, 30, 90]
    EWMA_SPANS = [7, 14, 30]
    LAGS = [1, 7, 14, 30]
    FOURIER_PERIODS = [(365.25, "annual"), (182.625, "semi_annual"), (91.3125, "quarterly")]
    N_FEATURES = 42

    def extract_features(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
    ) -> pd.DataFrame:
        """Return DataFrame with 42 temporal feature columns + value_col.

        Parameters
        ──────────
        df        : DataFrame containing at least date_col and value_col
        date_col  : name of the date / timestamp column
        value_col : name of the numeric series column

        Returns
        ───────
        DataFrame with 42 feature columns + value_col (index reset).
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        v: pd.Series = df[value_col].fillna(df[value_col].median())
        dt: pd.Series = df[date_col]

        feats: Dict[str, np.ndarray] = {}

        # ── Rolling statistics ──────────────────────────────────────────────
        for w in self.WINDOWS:
            roll = v.rolling(w, min_periods=1)
            feats[f"roll_mean_{w}d"] = roll.mean().values
            feats[f"roll_std_{w}d"] = roll.std().fillna(0).values
            feats[f"roll_min_{w}d"] = roll.min().values
            feats[f"roll_max_{w}d"] = roll.max().values

        # ── Exponentially weighted moving averages ──────────────────────────
        for s in self.EWMA_SPANS:
            feats[f"ewma_{s}d"] = v.ewm(span=s, adjust=False).mean().values

        # ── Lag features ────────────────────────────────────────────────────
        v_median = float(v.median())
        for lag in self.LAGS:
            feats[f"lag_{lag}d"] = v.shift(lag).fillna(v_median).values

        # ── Day-of-week one-hot encoding ────────────────────────────────────
        dow = dt.dt.dayofweek.values
        for d in range(7):
            feats[f"dow_{d}"] = (dow == d).astype(np.float32)

        # ── Month cyclical ──────────────────────────────────────────────────
        month = dt.dt.month.values.astype(float)
        feats["month_sin"] = np.sin(2 * math.pi * month / 12)
        feats["month_cos"] = np.cos(2 * math.pi * month / 12)

        # ── Chinese New Year ─────────────────────────────────────────────────
        cny_flag = np.zeros(len(df), dtype=np.float32)
        for year, cny_str in _CNY_DATES.items():
            cny_ts = pd.Timestamp(cny_str)
            delta = (dt - cny_ts).dt.days.abs()
            cny_flag = np.maximum(cny_flag, (delta <= 14).astype(np.float32).values)
        feats["chinese_new_year"] = cny_flag

        # ── Golden Week (China National Day, Oct 1-7) ───────────────────────
        feats["golden_week"] = (
            (dt.dt.month == 10) & (dt.dt.day <= 7)
        ).astype(np.float32).values

        # ── Year-over-year % change ──────────────────────────────────────────
        yoy = v.pct_change(periods=365)
        feats["yoy_change"] = yoy.fillna(0).clip(-2, 2).values

        # ── Fourier seasonality terms ────────────────────────────────────────
        t = np.arange(len(df), dtype=float)
        for period, name in self.FOURIER_PERIODS:
            feats[f"fourier_sin_{name}"] = np.sin(2 * math.pi * t / period)
            feats[f"fourier_cos_{name}"] = np.cos(2 * math.pi * t / period)

        # ── is_weekend ───────────────────────────────────────────────────────
        feats["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.float32).values

        assert len(feats) == self.N_FEATURES, (
            f"Expected {self.N_FEATURES} features, got {len(feats)}: {list(feats.keys())}"
        )

        result = pd.DataFrame(feats)
        result[value_col] = v.values
        return result

    @property
    def feature_names(self) -> List[str]:
        """Ordered list of the 42 feature names."""
        names = []
        for w in self.WINDOWS:
            for stat in ["mean", "std", "min", "max"]:
                names.append(f"roll_{stat}_{w}d")
        for s in self.EWMA_SPANS:
            names.append(f"ewma_{s}d")
        for lag in self.LAGS:
            names.append(f"lag_{lag}d")
        for d in range(7):
            names.append(f"dow_{d}")
        names += ["month_sin", "month_cos", "chinese_new_year", "golden_week",
                  "yoy_change"]
        for _, name in self.FOURIER_PERIODS:
            names += [f"fourier_sin_{name}", f"fourier_cos_{name}"]
        names.append("is_weekend")
        assert len(names) == self.N_FEATURES
        return names


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SupplyChainForecaster
# ═══════════════════════════════════════════════════════════════════════════════

class SupplyChainForecaster:
    """End-to-end TCN-based supply chain forecaster.

    Supports
    ────────
    - Multi-horizon quantile forecasting (30d / 60d / 90d, P10/P50/P90)
    - Walk-forward backtesting (MAPE, WQL, bias)
    - Inventory depletion date prediction
    - Payment timing prediction (DSO modelling)
    - Fan-chart visualisation

    Usage
    ─────
    forecaster = SupplyChainForecaster()
    history = forecaster.fit({'port_throughput': df, 'freight_rates': df2})
    result  = forecaster.predict('port_throughput', last_obs, horizons=[30,60,90])
    metrics = forecaster.backtest(df, forecast_horizon=30)
    """

    SEQ_LEN = 128   # input sequence length fed to the TCN
    HORIZONS = [30, 60, 90]
    QUANTILES = [0.1, 0.5, 0.9]

    def __init__(self, hidden_channels: int = 64, num_layers: int = 7):
        self.feature_extractor = TemporalFeatureExtractor()
        self.scalers: Dict[str, RobustScaler] = {}
        self.model: Optional[LogisChainTCN] = None
        self._fitted = False
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self._training_history: dict = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_model(self, n_features: int) -> LogisChainTCN:
        return LogisChainTCN(
            input_channels=n_features,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            forecast_horizons=self.HORIZONS,
            quantiles=self.QUANTILES,
        ).to(self.device)

    def _prepare_series(
        self, df: pd.DataFrame, series_key: str, date_col: str = "date", value_col: str = "value"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features, scale, and return (features_array, values_array)."""
        # Try to infer date/value columns if default names missing
        if date_col not in df.columns:
            date_col = df.select_dtypes(include=["datetime64", "object"]).columns[0]
        if value_col not in df.columns:
            value_col = df.select_dtypes(include=[np.number]).columns[0]

        feat_df = self.feature_extractor.extract_features(df, date_col, value_col)
        feature_cols = self.feature_extractor.feature_names
        available = [c for c in feature_cols if c in feat_df.columns]
        feat_arr = feat_df[available].values.astype(np.float32)  # (T, F)

        scaler = RobustScaler()
        feat_arr = scaler.fit_transform(feat_arr)
        self.scalers[series_key] = scaler

        val_arr = feat_df[value_col].values.astype(np.float32)
        return feat_arr, val_arr

    def _make_sequences(
        self, feat_arr: np.ndarray, val_arr: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Slide a window across the series to create (X, y) pairs.

        X : (N, F, SEQ_LEN)
        y : {'{h}d': (N,)} for each horizon
        """
        T = len(feat_arr)
        max_h = max(self.HORIZONS)
        if T < self.SEQ_LEN + max_h:
            raise ValueError(
                f"Series too short ({T}). Need ≥ {self.SEQ_LEN + max_h} time steps."
            )

        X_list = []
        y_dict: Dict[str, list] = {f"{h}d": [] for h in self.HORIZONS}

        for i in range(self.SEQ_LEN, T - max_h + 1):
            x_seq = feat_arr[i - self.SEQ_LEN: i].T  # (F, SEQ_LEN)
            X_list.append(x_seq)
            for h in self.HORIZONS:
                y_dict[f"{h}d"].append(val_arr[min(i + h - 1, T - 1)])

        X = np.stack(X_list, axis=0).astype(np.float32)  # (N, F, SEQ_LEN)
        y = {k: np.array(v, dtype=np.float32) for k, v in y_dict.items()}
        return X, y

    # ── Public API ─────────────────────────────────────────────────────────

    def fit(
        self,
        time_series_dict: Dict[str, pd.DataFrame],
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 64,
        date_col: str = "date",
        value_col: str = "value",
    ) -> dict:
        """Train the TCN on one or more named time series.

        Parameters
        ──────────
        time_series_dict : {series_name: DataFrame}  each with date_col & value_col
        epochs           : training epochs
        lr               : initial learning rate (ReduceLROnPlateau halves on plateau)

        Returns training history dict.
        """
        # ── Collect all sequences ──
        all_X, all_y = [], {f"{h}d": [] for h in self.HORIZONS}
        n_features = None

        for key, df in time_series_dict.items():
            try:
                feat_arr, val_arr = self._prepare_series(df, key, date_col, value_col)
                X, y = self._make_sequences(feat_arr, val_arr)
                all_X.append(X)
                for h in self.HORIZONS:
                    all_y[f"{h}d"].append(y[f"{h}d"])
                n_features = feat_arr.shape[1]
            except Exception as exc:
                logger.warning(f"Skipping series '{key}': {exc}")

        if not all_X:
            raise RuntimeError("No valid series to train on.")

        X_all = np.concatenate(all_X, axis=0)
        y_all = {k: np.concatenate(v) for k, v in all_y.items()}

        logger.info(f"Training sequences: {X_all.shape} | n_features={n_features}")

        # ── Build model & optimiser ──
        self.model = self._build_model(n_features)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=10, verbose=False
        )
        criterion = QuantileLoss(self.QUANTILES)

        X_t = torch.tensor(X_all, dtype=torch.float32)
        y_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in y_all.items()}
        primary_key = f"{self.HORIZONS[0]}d"
        dataset = TensorDataset(X_t, y_tensors[primary_key],
                                y_tensors[f"{self.HORIZONS[1]}d"],
                                y_tensors[f"{self.HORIZONS[2]}d"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        history = {"train_loss": []}
        best_loss, best_state = float("inf"), None

        run_ctx = mlflow.start_run(run_name="tcn_forecaster", nested=True) if _MLFLOW else _noop()
        with run_ctx:
            if _MLFLOW:
                mlflow.log_params({
                    "epochs": epochs, "lr": lr, "batch_size": batch_size,
                    "hidden_channels": self.hidden_channels, "num_layers": self.num_layers,
                    "n_features": n_features,
                })

            for epoch in range(epochs):
                self.model.train()
                epoch_loss = 0.0
                for xb, yb0, yb1, yb2 in loader:
                    xb = xb.to(self.device)
                    yb = {f"{self.HORIZONS[0]}d": yb0.to(self.device),
                          f"{self.HORIZONS[1]}d": yb1.to(self.device),
                          f"{self.HORIZONS[2]}d": yb2.to(self.device)}
                    optimiser.zero_grad()
                    preds = self.model(xb)
                    loss = sum(criterion(preds[k], yb[k]) for k in preds)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimiser.step()
                    epoch_loss += float(loss.item())

                avg_loss = epoch_loss / len(loader)
                history["train_loss"].append(avg_loss)
                scheduler.step(avg_loss)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_state = deepcopy(self.model.state_dict())

                if epoch % 20 == 0 or epoch == epochs - 1:
                    logger.info(f"Epoch {epoch:04d}/{epochs} | loss={avg_loss:.4f}")
                if _MLFLOW:
                    mlflow.log_metric("train_loss", avg_loss, step=epoch)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._fitted = True
        self._training_history = history
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")
        return history

    @torch.no_grad()
    def predict(
        self,
        series_name: str,
        last_observations: np.ndarray,
        horizons: List[int] = (30, 60, 90),
    ) -> Dict[str, Dict[str, float]]:
        """Predict quantile forecasts for the given recent observations.

        Parameters
        ──────────
        series_name      : key used during fit() for scaler lookup
        last_observations: 1-D array of the most recent ≥ SEQ_LEN raw values
        horizons         : list of day horizons to return

        Returns
        ───────
        {'30d': {'p10': x, 'p50': x, 'p90': x}, '60d': ..., '90d': ...}
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Call fit() first.")

        obs = np.asarray(last_observations, dtype=np.float32)
        if len(obs) < self.SEQ_LEN:
            # Pad with mean of observations
            pad = np.full(self.SEQ_LEN - len(obs), obs.mean(), dtype=np.float32)
            obs = np.concatenate([pad, obs])
        obs = obs[-self.SEQ_LEN:]

        # Build a minimal DataFrame for feature extraction
        dates = pd.date_range(end="2023-12-31", periods=self.SEQ_LEN, freq="D")
        temp_df = pd.DataFrame({"date": dates, "value": obs})
        scaler = self.scalers.get(series_name)
        if scaler is None:
            temp_feats, _ = self._prepare_series(temp_df, series_name)
        else:
            feat_df = self.feature_extractor.extract_features(temp_df, "date", "value")
            available = [c for c in self.feature_extractor.feature_names if c in feat_df.columns]
            temp_feats = scaler.transform(feat_df[available].values.astype(np.float32))

        x = torch.tensor(temp_feats.T[np.newaxis], dtype=torch.float32).to(self.device)
        self.model.eval()
        raw = self.model(x)

        result = {}
        for h in horizons:
            key = f"{h}d"
            if key in raw:
                q = raw[key].squeeze(0).cpu().numpy()
                result[key] = {"p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}
        return result

    def backtest(
        self,
        df: pd.DataFrame,
        start_fraction: float = 0.8,
        forecast_horizon: int = 30,
        date_col: str = "date",
        value_col: str = "value",
    ) -> Dict[str, float]:
        """Walk-forward backtest on a held-out portion of the series.

        Steps
        ─────
        1. Train on first `start_fraction` of the series.
        2. For each step t in the held-out period, predict at `forecast_horizon` ahead.
        3. Compute MAPE, WQL, and bias.

        Returns
        ───────
        {'mape': float, 'wql': float, 'bias': float}
        Target: mape < 12% for 30-day horizon.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        split_idx = int(len(df) * start_fraction)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        if len(train_df) < self.SEQ_LEN + forecast_horizon:
            logger.warning("Training split too short for backtest — returning NaN metrics.")
            return {"mape": float("nan"), "wql": float("nan"), "bias": float("nan")}

        # Quick retrain on train split
        self.fit({f"bt_series": train_df}, epochs=30, date_col=date_col, value_col=value_col)

        actuals, p10s, p50s, p90s = [], [], [], []
        series_vals = df[value_col].values

        for i in range(split_idx, len(df) - forecast_horizon):
            last_obs = series_vals[max(0, i - self.SEQ_LEN): i]
            result = self.predict("bt_series", last_obs, horizons=[forecast_horizon])
            key = f"{forecast_horizon}d"
            if key in result:
                actuals.append(series_vals[i + forecast_horizon - 1])
                p10s.append(result[key]["p10"])
                p50s.append(result[key]["p50"])
                p90s.append(result[key]["p90"])

        if not actuals:
            return {"mape": float("nan"), "wql": float("nan"), "bias": float("nan")}

        actuals = np.array(actuals)
        p50s = np.array(p50s)
        p10s = np.array(p10s)
        p90s = np.array(p90s)

        # MAPE
        mape = float(np.mean(np.abs((actuals - p50s) / (np.abs(actuals) + 1e-8))) * 100)

        # WQL (weighted quantile loss = mean pinball over P10+P50+P90)
        def _pinball(y, q_pred, q_level):
            err = y - q_pred
            return np.mean(np.maximum(q_level * err, (q_level - 1) * err))

        wql = float(
            (_pinball(actuals, p10s, 0.1) + _pinball(actuals, p50s, 0.5) + _pinball(actuals, p90s, 0.9))
            / 3
        )
        bias = float(np.mean(p50s - actuals))
        logger.info(f"Backtest: MAPE={mape:.2f}%, WQL={wql:.4f}, Bias={bias:.4f}")
        return {"mape": mape, "wql": wql, "bias": bias}

    def predict_inventory_depletion(
        self,
        inventory_df: pd.DataFrame,
        consumption_rate: float,
        replenishment_pipeline: List[Tuple[datetime, float]],
        uncertainty_factor: float = 1.15,
        date_col: str = "date",
        value_col: str = "inventory_units",
    ) -> int:
        """Predict days until stockout given consumption and incoming replenishments.

        Parameters
        ──────────
        inventory_df          : DataFrame with date_col + value_col (inventory levels)
        consumption_rate      : mean daily consumption (units/day)
        replenishment_pipeline: [(arrival_date, units), ...] scheduled deliveries
        uncertainty_factor    : multiplier on consumption for pessimistic scenario

        Returns
        ───────
        Predicted days until stockout (int).  Returns 9999 if no stockout within 1 year.
        """
        current_stock = float(inventory_df[value_col].iloc[-1])
        rep_dict: Dict[int, float] = {}
        base_date = pd.Timestamp(inventory_df[date_col].iloc[-1])
        for arr_date, units in replenishment_pipeline:
            delta = (pd.Timestamp(arr_date) - base_date).days
            if delta > 0:
                rep_dict[delta] = rep_dict.get(delta, 0) + units

        # Pessimistic consumption (P90-like)
        p90_consumption = consumption_rate * uncertainty_factor
        stock = current_stock
        for day in range(1, 365):
            stock += rep_dict.get(day, 0)
            stock -= p90_consumption
            if stock <= 0:
                return day
        return 9999

    def predict_payment_timing(
        self,
        invoice_df: pd.DataFrame,
        date_col: str = "invoice_date",
        due_col: str = "due_date",
        amount_col: str = "invoice_amount_usd",
    ) -> pd.DataFrame:
        """Predict actual payment date vs contractual due date (DSO modelling).

        Uses a heuristic (credit-risk-adjusted Weibull delay) calibrated to
        supply chain stress levels.  Replace with fitted model for production use.

        Returns DataFrame with: invoice_id, predicted_payment_date, expected_delay_days.
        """
        df = invoice_df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[due_col] = pd.to_datetime(df[due_col], errors="coerce")

        rng = np.random.default_rng(42)
        results = []
        for _, row in df.iterrows():
            # Weibull delay: shape=1.5, scale calibrated to industry DSO ~5 days late
            base_delay = float(rng.weibull(1.5) * 6.0)

            # SC stress adjustor: higher invoice amount → slightly more delay
            amount = float(row.get(amount_col, 100_000))
            stress_adj = math.log1p(amount / 100_000) * 0.5

            expected_delay = round(base_delay * (1 + stress_adj * 0.1))
            pred_date = row[due_col] + timedelta(days=int(expected_delay))
            results.append(
                {
                    "invoice_id": row.get("invoice_id", f"INV-{_}"),
                    "predicted_payment_date": pred_date,
                    "expected_delay_days": expected_delay,
                }
            )
        return pd.DataFrame(results)

    def plot_forecast(
        self,
        series_name: str,
        forecast_dict: Dict[str, Dict[str, float]],
        history_df: pd.DataFrame,
        date_col: str = "date",
        value_col: str = "value",
        disruption_dates: Optional[List[datetime]] = None,
        save_path: Optional[str] = None,
    ):
        """Fan chart showing P10/P50/P90 forecast bands.

        Parameters
        ──────────
        forecast_dict   : output of predict() e.g. {'30d': {'p10':x,'p50':x,'p90':x}}
        history_df      : historical series DataFrame
        disruption_dates: optional list of disruption event timestamps to mark
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        hist = history_df.copy()
        hist[date_col] = pd.to_datetime(hist[date_col])
        hist = hist.sort_values(date_col).tail(180)
        ax.plot(hist[date_col], hist[value_col], color="steelblue", lw=1.5, label="Historical")

        # Draw fan for each horizon
        colors = ["#2ca02c", "#ff7f0e", "#d62728"]
        last_date = hist[date_col].iloc[-1]
        for (key, q_dict), col in zip(sorted(forecast_dict.items()), colors):
            h_days = int(key.replace("d", ""))
            fore_date = last_date + timedelta(days=h_days)
            p10, p50, p90 = q_dict["p10"], q_dict["p50"], q_dict["p90"]
            ax.fill_between([last_date, fore_date], [hist[value_col].iloc[-1], p10],
                            [hist[value_col].iloc[-1], p90],
                            alpha=0.20, color=col)
            ax.plot([last_date, fore_date], [hist[value_col].iloc[-1], p50],
                    linestyle="--", color=col, lw=1.5, label=f"P50 {key}")
            ax.scatter(fore_date, p50, color=col, zorder=5, s=40)

        # Mark disruptions
        if disruption_dates:
            for d in disruption_dates:
                ax.axvline(pd.Timestamp(d), color="red", linestyle=":", alpha=0.6)

        ax.set_title(f"Supply Chain Forecast — {series_name} (P10/P50/P90)", fontsize=11)
        ax.set_xlabel("Date")
        ax.set_ylabel(value_col)
        ax.legend(fontsize=8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig


# ── Backward-compatible aliases ───────────────────────────────────────────────

class SupplyChainTCN(LogisChainTCN):
    """v0.1.0 compatibility alias for LogisChainTCN."""
    pass


class DemandForecastPipeline(SupplyChainForecaster):
    """v0.1.0 compatibility alias for SupplyChainForecaster."""

    def run(
        self,
        df: pd.DataFrame,
        date_col: str = "ship_date",
        value_col: str = "value_usd",
        forecast_days: int = 30,
    ) -> dict:
        try:
            self.fit({"series": df}, epochs=30, date_col=date_col, value_col=value_col)
            return self.predict("series", df[value_col].values, horizons=[forecast_days])
        except Exception as exc:
            logger.warning(f"DemandForecastPipeline.run() fell back to naive: {exc}")
            mean_val = float(df[value_col].mean())
            return {f"{forecast_days}d": {"p10": mean_val * 0.9, "p50": mean_val, "p90": mean_val * 1.1}}


# ═══════════════════════════════════════════════════════════════════════════════
# __main__ — train on synthetic sinusoidal data and print results
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("LogisChain AI — TCN Supply Chain Forecaster")

    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=800, freq="D")
    t = np.arange(800)
    # Realistic series: trend + seasonality + noise + COVID spike
    values = (
        1000
        + 0.5 * t
        + 120 * np.sin(2 * np.pi * t / 365)
        + 60 * np.sin(2 * np.pi * t / 7)
        + rng.normal(0, 30, 800)
    )
    values[350:450] *= 2.5  # COVID spike

    df = pd.DataFrame({"date": dates, "value": values})
    print(f"Series shape: {df.shape}")

    forecaster = SupplyChainForecaster(hidden_channels=32, num_layers=4)
    history = forecaster.fit({"demo": df}, epochs=30)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")

    preds = forecaster.predict("demo", values[-128:], horizons=[30, 60, 90])
    print("\nForecasts:")
    for k, q in preds.items():
        print(f"  {k}: P10={q['p10']:.1f}  P50={q['p50']:.1f}  P90={q['p90']:.1f}")

    metrics = forecaster.backtest(df, start_fraction=0.85, forecast_horizon=30)
    print(f"\nBacktest: MAPE={metrics['mape']:.2f}%  WQL={metrics['wql']:.4f}  Bias={metrics['bias']:.2f}")

    # Inventory depletion
    inv_df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=30, freq="D"),
        "inventory_units": np.linspace(500, 350, 30),
    })
    days_to_stockout = forecaster.predict_inventory_depletion(
        inv_df, consumption_rate=15, replenishment_pipeline=[]
    )
    print(f"\nInventory depletion in: {days_to_stockout} days")

    # TemporalFeatureExtractor
    fe = TemporalFeatureExtractor()
    feat_df = fe.extract_features(df, "date", "value")
    n_feat = len(fe.feature_names)
    print(f"\nTemporalFeatureExtractor: {n_feat} features extracted, DataFrame shape={feat_df.shape}")
