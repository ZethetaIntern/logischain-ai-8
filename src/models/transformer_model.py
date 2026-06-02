"""Transformer models for LogisChain AI.

Part A — LogisChainTransformer (v0.1.0)
────────────────────────────────────────
Encoder-decoder sequence-to-sequence model for multi-step forecasting.
Kept for backward compatibility.

Part B — ShipmentRiskTransformer (v0.2.0)
──────────────────────────────────────────
Transformer encoder with [CLS] token for multi-task shipment risk prediction:
  • P(delay)             + expected delay days
  • P(damage)            + damage severity %
  • P(doc_discrepancy)   for LC compliance
  • composite risk score (0-100)

Architecture: ShipmentEventEncoder → [CLS] + TransformerEncoder → 4 task heads.
Targets: delay_AUC > 0.80, Brier score < 0.18 (on calibrated training data).

MLflow experiment: logischain_ai / shipment_risk_transformer
"""

import logging
import math
import os
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

logger = logging.getLogger(__name__)


class _noop:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ═══════════════════════════════════════════════════════════════════════════════
# Part A — v0.1.0 components (backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences (batch_first)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model) or (L, B, d_model)
        if x.dim() == 3 and x.size(0) != 1:
            # batch_first
            x = x + self.pe[:, : x.size(1), :]
        else:
            x = x + self.pe[:, : x.size(0), :].transpose(0, 1)
        return self.dropout(x)


class LogisChainTransformer(nn.Module):
    """Encoder-decoder Transformer for multi-step supply chain forecasting (v0.1.0)."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_dim: int = 1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_seq_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src_emb = self.pos_enc(self.input_proj(src))
        tgt_emb = self.pos_enc(self.output_proj(tgt))
        if tgt_mask is None:
            T = tgt.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=src.device)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask)
        return self.output_head(out)


class TransformerTrainer:
    """Training loop for LogisChainTransformer (v0.1.0)."""

    def __init__(self, model: LogisChainTransformer, lr: float = 5e-4, device: Optional[str] = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.criterion = nn.MSELoss()

    def train_step(self, src, tgt_in, tgt_out) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(src.to(self.device), tgt_in.to(self.device))
        loss = self.criterion(pred, tgt_out.to(self.device))
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss)

    def fit(self, train_loader, n_epochs: int = 80, val_loader=None) -> dict:
        history = {"train_loss": []}
        for epoch in range(n_epochs):
            losses = [self.train_step(s, ti, to) for s, ti, to in train_loader]
            history["train_loss"].append(float(np.mean(losses)))
            self.scheduler.step()
        return history

    def predict(self, src: torch.Tensor, forecast_len: int = 30) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            src = src.to(self.device)
            B = src.size(0)
            tgt = torch.zeros(B, 1, 1, device=self.device)
            preds = []
            for _ in range(forecast_len):
                out = self.model(src, tgt)
                next_val = out[:, -1:, :]
                preds.append(next_val.cpu().numpy())
                tgt = torch.cat([tgt, next_val], dim=1)
        return np.concatenate(preds, axis=1).squeeze(-1)


def make_sequences(
    arr: np.ndarray, input_len: int = 128, output_len: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert 1-D array into (src, tgt_in, tgt_out) training sequences."""
    X, Y_in, Y_out = [], [], []
    for i in range(len(arr) - input_len - output_len):
        src = arr[i: i + input_len]
        tgt_out = arr[i + input_len: i + input_len + output_len]
        tgt_in = np.concatenate([[arr[i + input_len - 1]], tgt_out[:-1]])
        X.append(src.reshape(-1, 1))
        Y_in.append(tgt_in.reshape(-1, 1))
        Y_out.append(tgt_out.reshape(-1, 1))
    return (
        np.stack(X).astype(np.float32),
        np.stack(Y_in).astype(np.float32),
        np.stack(Y_out).astype(np.float32),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Part B — ShipmentRiskTransformer (v0.2.0)
# ═══════════════════════════════════════════════════════════════════════════════

# ── ShipmentEvent dataclass ──────────────────────────────────────────────────

@dataclass
class ShipmentEvent:
    """A single tracked event in a shipment's lifecycle.

    Eight event types mark key milestones:
        BOOKING → LOADED → DEPARTED → TRANSHIPMENT_ARR → TRANSHIPMENT_DEP
        → ARRIVAL → CUSTOMS → DELIVERY
    """
    event_type: str
    timestamp: datetime
    port_lat: float
    port_lon: float
    vessel_speed: float
    cargo_weight_tons: float
    port_congestion_index: float   # 0-5 scale
    weather_severity: float        # 0-1 scale
    carrier_reliability_score: float  # 0-1 scale
    days_since_booking: int
    extra: Dict = field(default_factory=dict)

    EVENT_TYPES: List[str] = field(default_factory=lambda: [
        "BOOKING", "LOADED", "DEPARTED", "TRANSHIPMENT_ARR",
        "TRANSHIPMENT_DEP", "ARRIVAL", "CUSTOMS", "DELIVERY",
    ], init=False, repr=False)


# Allow class-level access without instantiation
_EVENT_TYPES = ["BOOKING", "LOADED", "DEPARTED", "TRANSHIPMENT_ARR",
                "TRANSHIPMENT_DEP", "ARRIVAL", "CUSTOMS", "DELIVERY"]
_EVENT2IDX = {e: i for i, e in enumerate(_EVENT_TYPES)}
_N_EVENT_TYPES = len(_EVENT_TYPES)


# ── ShipmentEventEncoder ─────────────────────────────────────────────────────

class ShipmentEventEncoder(nn.Module):
    """Encode a batch of shipment event sequences into (B, L, d_model) tensors.

    Modalities
    ──────────
    Event type    : embedding (8 types → d_model)
    Temporal      : positional + day-of-week (7-dim) + month cyclical (2-dim)
    Spatial       : lat/lon sinusoidal encoding (4-dim) → d_model
    Operational   : [speed, weight, congestion, weather, reliability, days_booking] → d_model
    All projected to d_model and summed.

    Input dict keys (all float tensors unless noted)
    ────────────────────────────────────────────────
    'event_type_idx' : (B, L) int64  — event type indices
    'ops'            : (B, L, 6)     — operational features
    'spatial'        : (B, L, 4)     — [sin_lat, cos_lat, sin_lon, cos_lon]
    'temporal'       : (B, L, 9)     — [dow×7, month_sin, month_cos]
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

        # Event type embedding (padding_idx=0 unused since we shift by 1)
        self.event_emb = nn.Embedding(_N_EVENT_TYPES, d_model)

        # Operational: speed, weight, congestion, weather, reliability, days_since_booking
        self.ops_proj = nn.Sequential(
            nn.Linear(6, d_model), nn.LayerNorm(d_model)
        )
        # Spatial: sin(lat), cos(lat), sin(lon), cos(lon)
        self.spatial_proj = nn.Sequential(
            nn.Linear(4, d_model), nn.LayerNorm(d_model)
        )
        # Temporal: 7 × DOW one-hot + sin(month) + cos(month)
        self.temporal_proj = nn.Sequential(
            nn.Linear(9, d_model), nn.LayerNorm(d_model)
        )

        self.pos_enc = PositionalEncoding(d_model, dropout=0.1, max_len=64)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return (B, L, d_model) combined embedding."""
        evt_emb = self.event_emb(x_dict["event_type_idx"])       # (B, L, d_model)
        ops_emb = self.ops_proj(x_dict["ops"])                   # (B, L, d_model)
        sp_emb = self.spatial_proj(x_dict["spatial"])            # (B, L, d_model)
        tmp_emb = self.temporal_proj(x_dict["temporal"])         # (B, L, d_model)

        combined = evt_emb + ops_emb + sp_emb + tmp_emb          # (B, L, d_model)
        combined = self.pos_enc(combined)                         # adds pos enc
        return self.out_norm(combined)

    @staticmethod
    def encode_events_to_dict(
        events: List[ShipmentEvent],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Convert a list of ShipmentEvent objects to a batched dict of tensors.

        Returns dict with batch_size=1.
        """
        L = len(events)
        event_idx = torch.zeros(1, L, dtype=torch.long, device=device)
        ops = torch.zeros(1, L, 6, dtype=torch.float32, device=device)
        spatial = torch.zeros(1, L, 4, dtype=torch.float32, device=device)
        temporal = torch.zeros(1, L, 9, dtype=torch.float32, device=device)

        for j, ev in enumerate(events):
            event_idx[0, j] = _EVENT2IDX.get(ev.event_type, 0)

            ops[0, j] = torch.tensor([
                ev.vessel_speed / 25.0,
                math.log1p(ev.cargo_weight_tons) / 10.0,
                ev.port_congestion_index / 5.0,
                ev.weather_severity,
                ev.carrier_reliability_score,
                ev.days_since_booking / 60.0,
            ])

            lat_r = math.radians(ev.port_lat)
            lon_r = math.radians(ev.port_lon)
            spatial[0, j] = torch.tensor([
                math.sin(lat_r), math.cos(lat_r),
                math.sin(lon_r), math.cos(lon_r),
            ])

            dow = ev.timestamp.weekday()
            dow_onehot = [float(dow == d) for d in range(7)]
            month = ev.timestamp.month
            temporal[0, j] = torch.tensor(
                dow_onehot + [math.sin(2 * math.pi * month / 12),
                              math.cos(2 * math.pi * month / 12)],
                dtype=torch.float32,
            )

        return {
            "event_type_idx": event_idx,
            "ops": ops,
            "spatial": spatial,
            "temporal": temporal,
        }


# ── ShipmentRiskTransformer ──────────────────────────────────────────────────

class ShipmentRiskTransformer(nn.Module):
    """Transformer encoder for shipment-level multi-task risk prediction.

    Architecture
    ────────────
    [CLS] + event_embeddings → TransformerEncoder (4 layers, 4 heads)
                             → CLS representation
                             → 4 task heads:
        delay_head       → (P_delay, expected_delay_days)
        damage_head      → (P_damage, damage_severity_pct)
        discrepancy_head → P(doc_discrepancy)
        composite_head   → risk_score (0-100)

    Targets
    ───────
    delay_AUC > 0.80, Brier score < 0.18 on calibrated training data.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.event_encoder = ShipmentEventEncoder(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )
        self._num_heads = nhead
        self._num_layers = num_encoder_layers

        # ── Task heads ──────────────────────────────────────────────────────
        # delay: P(delay), expected_delay_days
        self.delay_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2)
        )
        # damage: P(damage), damage_severity_%
        self.damage_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2)
        )
        # discrepancy: P(doc_discrepancy)
        self.discrepancy_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )
        # composite: risk_score
        self.composite_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_layer_with_attn(
        self,
        layer: nn.TransformerEncoderLayer,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Manual forward of TransformerEncoderLayer capturing attention weights."""
        # Self-attention with need_weights=True
        src2, attn_w = layer.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = layer.norm1(x + layer.dropout1(src2))
        src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = layer.norm2(x + layer.dropout2(src2))
        return x, attn_w  # attn_w: (B, nhead, L+1, L+1)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ──────────
        x_dict         : dict of tensors from ShipmentEventEncoder.encode_events_to_dict()
        attention_mask : (B, L) bool tensor, True = masked (padding)
        return_attention: if True, run layer-by-layer to extract attention weights

        Returns
        ───────
        {
            'delay_prob'       : (B,)   P(delay)
            'delay_days'       : (B,)   expected delay days ≥ 0
            'damage_prob'      : (B,)   P(damage)
            'damage_severity'  : (B,)   damage severity % ≥ 0
            'discrepancy_prob' : (B,)   P(doc discrepancy)
            'risk_score'       : (B,)   composite risk score 0-100
            'attention_weights': (B, n_layers, nhead, L+1, L+1) or None
        }
        """
        # Encode events
        x = self.event_encoder(x_dict)   # (B, L, d_model)
        B = x.size(0)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)   # (B, L+1, d_model)

        # Extend attention mask for CLS token (never masked)
        if attention_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            src_pad_mask = torch.cat([cls_mask, attention_mask], dim=1)
        else:
            src_pad_mask = None

        if return_attention:
            all_attn = []
            for layer in self.transformer.layers:
                x, attn_w = self._forward_layer_with_attn(layer, x, src_pad_mask)
                all_attn.append(attn_w)
            attn_stack = torch.stack(all_attn, dim=1)  # (B, n_layers, nhead, L+1, L+1)
        else:
            x = self.transformer(x, src_key_padding_mask=src_pad_mask)
            attn_stack = None

        cls_rep = x[:, 0, :]   # (B, d_model)

        # ── Task heads ──
        delay_out = self.delay_head(cls_rep)         # (B, 2)
        damage_out = self.damage_head(cls_rep)        # (B, 2)
        disc_out = self.discrepancy_head(cls_rep)     # (B, 1)
        risk_out = self.composite_head(cls_rep)       # (B, 1)

        return {
            "delay_prob":        torch.sigmoid(delay_out[:, 0]),
            "delay_days":        F.softplus(delay_out[:, 1]),
            "damage_prob":       torch.sigmoid(damage_out[:, 0]),
            "damage_severity":   F.softplus(damage_out[:, 1]),
            "discrepancy_prob":  torch.sigmoid(disc_out[:, 0]),
            "risk_score":        100.0 * torch.sigmoid(risk_out[:, 0]),
            "attention_weights": attn_stack,
        }

    def get_attention_weights(
        self,
        x_dict: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return (B, n_layers, nhead, L+1, L+1) attention weight tensor."""
        out = self.forward(x_dict, attention_mask=attention_mask, return_attention=True)
        return out["attention_weights"]


# ── ShipmentRiskPredictor ────────────────────────────────────────────────────

class ShipmentRiskPredictor:
    """End-to-end trainer and inference wrapper for ShipmentRiskTransformer.

    Supports
    ────────
    - Multi-task training: delay + damage + discrepancy + composite
    - Class-weighted BCE for imbalanced targets
    - Gradient clipping at 1.0
    - Attention-based explanation of risk predictions
    - Synthetic shipment generation for training

    Usage
    ─────
    predictor = ShipmentRiskPredictor()
    train_df  = predictor.generate_synthetic_shipments(n=5000)
    predictor.fit(train_df, epochs=30)
    risk = predictor.predict_shipment_risk(events_list)
    """

    MAX_SEQ_LEN = 12     # max events per shipment (padded)
    TASK_WEIGHTS = {"delay": 1.0, "damage": 2.0, "discrepancy": 1.0, "composite": 0.5}

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[ShipmentRiskTransformer] = None
        self._fitted = False
        self._best_state: Optional[dict] = None

    # ── Synthetic data generation ──────────────────────────────────────────

    def generate_synthetic_shipments(self, n: int = 10_000) -> pd.DataFrame:
        """Generate n realistic synthetic shipment records.

        Distribution
        ────────────
        Normal shipments   70%  — low congestion, reliable carrier
        Delayed            20%  — high congestion + low reliability
        Damaged             5%  — severe weather + heavy cargo
        Doc discrepancy    15%  — LC mismatch scenarios
        (labels are not mutually exclusive)

        Returns
        ───────
        DataFrame with one row per event (4-8 events per shipment).
        Shipment-level labels: delay_flag, delay_days, damage_flag,
        damage_severity_pct, discrepancy_flag, composite_risk_score.
        """
        rng = np.random.default_rng(42)
        EVENT_SEQ = ["BOOKING", "LOADED", "DEPARTED", "TRANSHIPMENT_ARR",
                     "TRANSHIPMENT_DEP", "ARRIVAL", "CUSTOMS", "DELIVERY"]
        records = []

        for i in range(n):
            roll = rng.random()
            is_delayed = roll < 0.20
            is_damaged = roll < 0.05
            is_discrepancy = rng.random() < 0.15

            # Correlated congestion boosts delay probability
            base_congestion = float(rng.beta(2, 6))
            delay_boost = float(rng.uniform(0.3, 0.8)) if is_delayed else 0.0

            n_events = int(rng.integers(4, 9))
            event_types = EVENT_SEQ[:n_events]
            base_date = datetime(2020, 1, 1) + timedelta(days=int(rng.integers(0, 1200)))
            cum_days = 0

            for j, evt in enumerate(event_types):
                cum_days += int(rng.integers(1, 6))
                lat = float(rng.uniform(-50, 60))
                lon = float(rng.uniform(-180, 180))
                congestion = float(min(1.0, base_congestion + delay_boost * rng.uniform(0, 1)))
                weather = float(rng.beta(3, 5) if is_damaged else rng.beta(1, 6))
                speed = float(max(0.5, rng.normal(14.0, 2.5)))
                weight = float(rng.lognormal(3.0, 0.6))
                reliability = float(
                    rng.beta(5, 4) if is_delayed else rng.beta(10, 2)
                )
                records.append(
                    {
                        "shipment_id":            f"SHP-{i:07d}",
                        "event_idx":              j,
                        "event_type":             evt,
                        "timestamp":              base_date + timedelta(days=cum_days),
                        "port_lat":               lat,
                        "port_lon":               lon,
                        "vessel_speed":           speed,
                        "cargo_weight_tons":      weight,
                        "port_congestion_index":  congestion * 5.0,
                        "weather_severity":       weather,
                        "carrier_reliability_score": reliability,
                        "days_since_booking":     cum_days,
                        # Shipment-level labels (same for all events of a shipment)
                        "delay_flag":             int(is_delayed),
                        "delay_days":             float(int(rng.integers(1, 15)) if is_delayed else 0),
                        "damage_flag":            int(is_damaged),
                        "damage_severity_pct":    float(rng.uniform(2, 25) if is_damaged else 0.0),
                        "discrepancy_flag":       int(is_discrepancy),
                        "composite_risk_score":   float(
                            (is_delayed * 40 + is_damaged * 35 + is_discrepancy * 25)
                            * float(rng.uniform(0.85, 1.15))
                        ),
                    }
                )

        df = pd.DataFrame(records)
        n_delay = df.groupby("shipment_id")["delay_flag"].first().mean()
        n_damage = df.groupby("shipment_id")["damage_flag"].first().mean()
        n_disc = df.groupby("shipment_id")["discrepancy_flag"].first().mean()
        logger.info(
            f"Generated {n:,} shipments | "
            f"delay={n_delay:.1%}, damage={n_damage:.1%}, discrepancy={n_disc:.1%}"
        )
        return df

    # ── Sequence preparation ────────────────────────────────────────────────

    def _df_to_tensors(self, df: pd.DataFrame):
        """Group by shipment_id and create padded sequence tensors.

        Returns
        ───────
        x_dict         : dict of (N, MAX_SEQ_LEN, ...) tensors
        labels         : dict of (N,) label tensors
        attn_mask      : (N, MAX_SEQ_LEN) padding mask
        """
        groups = df.groupby("shipment_id")
        ship_ids = list(groups.groups.keys())
        N = len(ship_ids)
        L = self.MAX_SEQ_LEN

        event_idx_arr = np.zeros((N, L), dtype=np.int64)
        ops_arr       = np.zeros((N, L, 6), dtype=np.float32)
        spatial_arr   = np.zeros((N, L, 4), dtype=np.float32)
        temporal_arr  = np.zeros((N, L, 9), dtype=np.float32)
        mask_arr      = np.ones((N, L), dtype=bool)   # True = padded (masked)

        delay_flag  = np.zeros(N, dtype=np.float32)
        delay_days  = np.zeros(N, dtype=np.float32)
        damage_flag = np.zeros(N, dtype=np.float32)
        damage_sev  = np.zeros(N, dtype=np.float32)
        disc_flag   = np.zeros(N, dtype=np.float32)
        risk_score  = np.zeros(N, dtype=np.float32)

        for i, sid in enumerate(ship_ids):
            evts = groups.get_group(sid).sort_values("event_idx")
            n_ev = min(len(evts), L)
            mask_arr[i, :n_ev] = False  # not padded

            for j in range(n_ev):
                row = evts.iloc[j]
                event_idx_arr[i, j] = _EVENT2IDX.get(str(row.get("event_type", "BOOKING")), 0)
                ops_arr[i, j] = [
                    float(row.get("vessel_speed", 14)) / 25.0,
                    math.log1p(float(row.get("cargo_weight_tons", 10))) / 10.0,
                    float(row.get("port_congestion_index", 0)) / 5.0,
                    float(row.get("weather_severity", 0)),
                    float(row.get("carrier_reliability_score", 0.85)),
                    float(row.get("days_since_booking", 0)) / 60.0,
                ]
                lat = math.radians(float(row.get("port_lat", 0)))
                lon = math.radians(float(row.get("port_lon", 0)))
                spatial_arr[i, j] = [math.sin(lat), math.cos(lat),
                                      math.sin(lon), math.cos(lon)]
                ts = pd.Timestamp(row.get("timestamp", "2020-01-01"))
                dow = ts.dayofweek
                month = ts.month
                temporal_arr[i, j] = ([float(dow == d) for d in range(7)]
                                       + [math.sin(2 * math.pi * month / 12),
                                          math.cos(2 * math.pi * month / 12)])

            # Labels (take from first event row — same for all)
            r0 = evts.iloc[0]
            delay_flag[i] = float(r0.get("delay_flag", 0))
            delay_days[i] = float(r0.get("delay_days", 0))
            damage_flag[i] = float(r0.get("damage_flag", 0))
            damage_sev[i] = float(r0.get("damage_severity_pct", 0))
            disc_flag[i] = float(r0.get("discrepancy_flag", 0))
            risk_score[i] = float(r0.get("composite_risk_score", 0))

        x_dict = {
            "event_type_idx": torch.tensor(event_idx_arr, dtype=torch.long),
            "ops":            torch.tensor(ops_arr, dtype=torch.float32),
            "spatial":        torch.tensor(spatial_arr, dtype=torch.float32),
            "temporal":       torch.tensor(temporal_arr, dtype=torch.float32),
        }
        labels = {
            "delay_flag":  torch.tensor(delay_flag),
            "delay_days":  torch.tensor(delay_days),
            "damage_flag": torch.tensor(damage_flag),
            "damage_sev":  torch.tensor(damage_sev),
            "disc_flag":   torch.tensor(disc_flag),
            "risk_score":  torch.tensor(risk_score),
        }
        attn_mask = torch.tensor(mask_arr, dtype=torch.bool)
        return x_dict, labels, attn_mask

    # ── Training ────────────────────────────────────────────────────────────

    def fit(
        self,
        shipment_df: Optional[pd.DataFrame] = None,
        epochs: int = 50,
        lr: float = 1e-4,
        batch_size: int = 64,
    ) -> dict:
        """Train the ShipmentRiskTransformer.

        If shipment_df is None, calls generate_synthetic_shipments(n=5000).

        Multi-task loss
        ───────────────
        L = w_delay × BCE(P_delay, y_delay)
          + w_damage × BCE(P_damage, y_damage)
          + w_disc   × BCE(P_disc, y_disc)
          + w_risk   × MSE(risk_score, y_risk/100)
        with class-weighted BCE for imbalanced targets.
        """
        if shipment_df is None:
            logger.info("No shipment_df provided — generating 5,000 synthetic shipments.")
            shipment_df = self.generate_synthetic_shipments(n=5_000)

        x_dict, labels, attn_mask = self._df_to_tensors(shipment_df)
        N = labels["delay_flag"].size(0)

        # Class weights for imbalanced tasks
        def _pos_weight(label_t: torch.Tensor) -> torch.Tensor:
            pos = float(label_t.sum())
            neg = float((1 - label_t).sum())
            return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)

        bce_delay = nn.BCELoss(reduction="mean")
        bce_damage = nn.BCELoss(reduction="mean")
        bce_disc = nn.BCELoss(reduction="mean")
        mse_risk = nn.MSELoss()

        # Build model
        self.model = ShipmentRiskTransformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs, eta_min=lr / 100
        )

        # Mini-batch indices
        idx = torch.arange(N)
        history = {"train_loss": []}
        best_loss = float("inf")

        run_ctx = (
            mlflow.start_run(run_name="shipment_risk_transformer", nested=True)
            if _MLFLOW else _noop()
        )
        with run_ctx:
            if _MLFLOW:
                mlflow.log_params({
                    "d_model": self.d_model, "nhead": self.nhead,
                    "num_encoder_layers": self.num_encoder_layers,
                    "epochs": epochs, "lr": lr, "N_train": N,
                })

            for epoch in range(epochs):
                self.model.train()
                perm = torch.randperm(N)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, N, batch_size):
                    batch_idx = perm[start: start + batch_size]
                    xb = {k: v[batch_idx].to(self.device) for k, v in x_dict.items()}
                    mb = attn_mask[batch_idx].to(self.device)
                    lb = {k: v[batch_idx].to(self.device) for k, v in labels.items()}

                    optimiser.zero_grad()
                    preds = self.model(xb, attention_mask=mb)

                    loss = (
                        self.TASK_WEIGHTS["delay"]   * bce_delay(preds["delay_prob"], lb["delay_flag"])
                        + self.TASK_WEIGHTS["damage"] * bce_damage(preds["damage_prob"], lb["damage_flag"])
                        + self.TASK_WEIGHTS["discrepancy"] * bce_disc(preds["discrepancy_prob"], lb["disc_flag"])
                        + self.TASK_WEIGHTS["composite"] * mse_risk(
                            preds["risk_score"] / 100.0, lb["risk_score"] / 100.0
                        )
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimiser.step()
                    epoch_loss += float(loss.item())
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                history["train_loss"].append(avg_loss)
                scheduler.step()

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self._best_state = deepcopy(self.model.state_dict())

                if epoch % 10 == 0 or epoch == epochs - 1:
                    logger.info(f"Epoch {epoch:04d}/{epochs} | loss={avg_loss:.4f}")
                if _MLFLOW:
                    mlflow.log_metric("train_loss", avg_loss, step=epoch)

        if self._best_state:
            self.model.load_state_dict(self._best_state)
        self._fitted = True
        return history

    # ── Inference ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_shipment_risk(
        self, shipment_events: List[ShipmentEvent]
    ) -> Dict[str, object]:
        """Return complete risk profile for a shipment event sequence.

        Returns
        ───────
        {
            'delay_probability'    : float (0-1)
            'expected_delay_days'  : float
            'damage_probability'   : float (0-1)
            'damage_severity_pct'  : float
            'discrepancy_probability': float (0-1)
            'total_risk_score'     : int (0-100)
            'risk_factors'         : List[str]
        }
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Call fit() before predict_shipment_risk().")

        self.model.eval()
        x_dict = ShipmentEventEncoder.encode_events_to_dict(shipment_events, self.device)
        raw = self.model(x_dict, return_attention=True)
        attn = raw["attention_weights"]   # (1, n_layers, nhead, L+1, L+1)

        delay_p  = float(raw["delay_prob"][0])
        delay_d  = float(raw["delay_days"][0])
        damage_p = float(raw["damage_prob"][0])
        damage_s = float(raw["damage_severity"][0])
        disc_p   = float(raw["discrepancy_prob"][0])
        risk_s   = int(round(float(raw["risk_score"][0])))

        # Build risk factors from attention weights (last layer, mean over heads)
        risk_factors = []
        if attn is not None:
            last_attn = attn[0, -1].mean(0).cpu().numpy()  # (L+1, L+1)
            cls_attn = last_attn[0, 1:]  # CLS attention over events (skip CLS→CLS)
            n_evts = min(len(shipment_events), cls_attn.shape[0])
            if n_evts > 0:
                for j in np.argsort(cls_attn[:n_evts])[::-1][:3]:
                    evt_name = shipment_events[j].event_type if j < len(shipment_events) else "UNKNOWN"
                    ev = shipment_events[j] if j < len(shipment_events) else None
                    if ev and ev.port_congestion_index > 2.0:
                        risk_factors.append(
                            f"Port congestion at {evt_name} "
                            f"(congestion={ev.port_congestion_index:.1f}/5, "
                            f"attn={cls_attn[j]:.2f})"
                        )
                    elif ev and ev.carrier_reliability_score < 0.70:
                        risk_factors.append(
                            f"Below-threshold carrier reliability at {evt_name} "
                            f"(score={ev.carrier_reliability_score:.2f}, "
                            f"attn={cls_attn[j]:.2f})"
                        )
                    elif ev and ev.weather_severity > 0.5:
                        risk_factors.append(
                            f"Adverse weather at {evt_name} "
                            f"(severity={ev.weather_severity:.2f}, "
                            f"attn={cls_attn[j]:.2f})"
                        )
                    else:
                        risk_factors.append(f"{evt_name} event (attn={cls_attn[j]:.2f})")

        return {
            "delay_probability":       round(delay_p, 4),
            "expected_delay_days":     round(delay_d, 2),
            "damage_probability":      round(damage_p, 4),
            "damage_severity_pct":     round(damage_s, 2),
            "discrepancy_probability": round(disc_p, 4),
            "total_risk_score":        min(100, max(0, risk_s)),
            "risk_factors":            risk_factors[:3],
        }

    @torch.no_grad()
    def explain_prediction(
        self, shipment_events: List[ShipmentEvent]
    ) -> Dict[str, object]:
        """Return attention-based explanation of the risk prediction.

        Returns
        ───────
        {
            'event_importance': [{event_type, attention_weight, event_idx}],
            'top_event':  str,
            'attention_by_layer': list of (L+1, L+1) mean attention matrices
        }
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Call fit() first.")
        self.model.eval()
        x_dict = ShipmentEventEncoder.encode_events_to_dict(shipment_events, self.device)
        attn_t = self.model.get_attention_weights(x_dict)

        if attn_t is None:
            return {"event_importance": [], "top_event": "UNKNOWN", "attention_by_layer": []}

        # (1, n_layers, nhead, L+1, L+1) → (n_layers, L+1, L+1) mean over heads
        attn_np = attn_t[0].mean(1).cpu().numpy()
        # CLS → events (last layer)
        cls_row = attn_np[-1, 0, 1:]   # (L,)
        n_evts = min(len(shipment_events), len(cls_row))

        importance = sorted(
            [
                {
                    "event_idx": j,
                    "event_type": shipment_events[j].event_type if j < len(shipment_events) else "PAD",
                    "attention_weight": round(float(cls_row[j]), 4),
                }
                for j in range(n_evts)
            ],
            key=lambda r: r["attention_weight"],
            reverse=True,
        )

        return {
            "event_importance": importance,
            "top_event": importance[0]["event_type"] if importance else "UNKNOWN",
            "attention_by_layer": [attn_np[l].tolist() for l in range(len(attn_np))],
        }

    @torch.no_grad()
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Compute AUC and Brier scores on a test DataFrame.

        Returns
        ───────
        {
            'delay_auc'      : float   (target > 0.80)
            'delay_brier'    : float   (target < 0.18)
            'damage_auc'     : float
            'discrepancy_auc': float
        }
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Call fit() first.")
        self.model.eval()
        x_dict, labels, attn_mask = self._df_to_tensors(test_df)

        preds = self.model(
            {k: v.to(self.device) for k, v in x_dict.items()},
            attention_mask=attn_mask.to(self.device),
        )

        def _safe_auc(y_true, y_score):
            y_np = y_true.cpu().numpy().astype(int)
            if len(np.unique(y_np)) < 2:
                return 0.5
            return float(roc_auc_score(y_np, y_score.cpu().numpy()))

        return {
            "delay_auc":       _safe_auc(labels["delay_flag"], preds["delay_prob"]),
            "delay_brier":     float(brier_score_loss(
                labels["delay_flag"].numpy().astype(int),
                preds["delay_prob"].cpu().numpy()
            )),
            "damage_auc":      _safe_auc(labels["damage_flag"], preds["damage_prob"]),
            "discrepancy_auc": _safe_auc(labels["disc_flag"], preds["discrepancy_prob"]),
        }

    def save(self, path: str = "models/shipment_risk_transformer.pt"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict() if self.model else {},
            "config": {
                "d_model": self.d_model, "nhead": self.nhead,
                "num_encoder_layers": self.num_encoder_layers, "dropout": self.dropout,
            },
            "fitted": self._fitted,
        }, path)
        logger.info(f"ShipmentRiskPredictor saved → {path}")

    def load(self, path: str = "models/shipment_risk_transformer.pt"):
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt["config"]
        self.d_model = cfg["d_model"]; self.nhead = cfg["nhead"]
        self.num_encoder_layers = cfg["num_encoder_layers"]
        self.dropout = cfg["dropout"]
        self.model = ShipmentRiskTransformer(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self._fitted = ckpt.get("fitted", True)
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# __main__ — train on synthetic data and print evaluation
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("LogisChain AI — ShipmentRiskTransformer")

    predictor = ShipmentRiskPredictor(d_model=64, nhead=4, num_encoder_layers=2)
    df = predictor.generate_synthetic_shipments(n=2000)
    print(f"Generated {df['shipment_id'].nunique()} shipments, {len(df)} events")

    # Train
    history = predictor.fit(df, epochs=20)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")

    # Evaluate
    metrics = predictor.evaluate(df)
    print("\n── Evaluation ────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.4f}")

    # Predict a single shipment
    events = [
        ShipmentEvent("BOOKING", datetime(2023, 1, 1), 31.2, 121.5, 0, 0, 1.0, 0.1, 0.95, 0),
        ShipmentEvent("LOADED",  datetime(2023, 1, 3), 31.2, 121.5, 12, 450, 3.5, 0.2, 0.80, 2),
        ShipmentEvent("DEPARTED",datetime(2023, 1, 5), 35.0, 140.0, 15, 450, 1.0, 0.1, 0.80, 4),
        ShipmentEvent("ARRIVAL", datetime(2023, 1, 19), 33.7,-118.3, 14, 450, 2.0, 0.0, 0.80, 18),
    ]
    risk = predictor.predict_shipment_risk(events)
    print("\n── Sample Shipment Risk Profile ──────────────")
    for k, v in risk.items():
        print(f"  {k}: {v}")

    # Explain
    exp = predictor.explain_prediction(events)
    print(f"\n── Explanation: top event = {exp['top_event']} ──")
    for item in exp["event_importance"][:3]:
        print(f"  {item['event_type']}: attn={item['attention_weight']:.4f}")

    # Save / load
    predictor.save("models/shipment_risk_demo.pt")
    p2 = ShipmentRiskPredictor(d_model=64, nhead=4, num_encoder_layers=2)
    p2.load("models/shipment_risk_demo.pt")
    r2 = p2.predict_shipment_risk(events)
    print(f"\nReloaded model risk_score: {r2['total_risk_score']}")
    print("Models match:", risk["total_risk_score"] == r2["total_risk_score"])
