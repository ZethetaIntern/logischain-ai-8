"""LogisChain AI — models subpackage.

Graph models (HetGAT)
─────────────────────
HetGAT                      Heterogeneous Graph Attention Network
SupplyChainHeteroGraph       Builds PyG HeteroData from raw DataFrames
GNNRiskPredictor             Train / infer / explain / save-load wrapper
compute_network_features      NetworkX centrality metrics from HeteroData
visualize_attention_weights   GATConv attention heatmap
run_gnn_pipeline              End-to-end convenience pipeline

TCN (time-series)
─────────────────
TCNResidualBlock             Dilated causal convolution residual block
LogisChainTCN                7-layer TCN, 3 horizons × 3 quantiles
QuantileLoss                 Pinball loss for distributional forecasting
TemporalFeatureExtractor     42 temporal features (rolling/EWMA/Fourier/holidays)
SupplyChainForecaster        Fit / predict / backtest / inventory depletion
SupplyChainTCN               v0.1.0 alias → LogisChainTCN
DemandForecastPipeline       v0.1.0 alias → SupplyChainForecaster

Transformer (shipment risk)
───────────────────────────
ShipmentEvent                Dataclass representing one tracked lifecycle event
ShipmentEventEncoder         Multi-modal event → d_model tensor encoder
ShipmentRiskTransformer      [CLS] + TransformerEncoder → 4 multi-task heads
ShipmentRiskPredictor        Fit / predict / explain / evaluate / save-load
LogisChainTransformer        v0.1.0 encoder-decoder Transformer
TransformerTrainer           v0.1.0 training loop

Tabular risk
────────────
XGBoostRiskModel             XGBoost binary classifier
LightGBMRiskModel            LightGBM binary classifier

Survival
────────
CarrierSurvivalModel         Cox PH + Kaplan-Meier time-to-failure

Ensemble
────────
LogisChainEnsemble           Stacking meta-learner over all base models
"""

from src.models.gnn import (
    HetGAT,
    SupplyChainHeteroGraph,
    GNNRiskPredictor,
    compute_network_features,
    visualize_attention_weights,
    run_gnn_pipeline,
    SupplyChainGNN,
    SupplyChainGraphBuilder,
    GNNTrainer,
    PYG_AVAILABLE,
)
from src.models.tcn import (
    TCNResidualBlock,
    LogisChainTCN,
    QuantileLoss,
    TemporalFeatureExtractor,
    SupplyChainForecaster,
    SupplyChainTCN,
    DemandForecastPipeline,
)
from src.models.transformer_model import (
    # v0.2.0 — shipment risk
    ShipmentEvent,
    ShipmentEventEncoder,
    ShipmentRiskTransformer,
    ShipmentRiskPredictor,
    # v0.1.0 — backward compat
    LogisChainTransformer,
    TransformerTrainer,
    PositionalEncoding,
    make_sequences,
)
from src.models.xgboost_model import XGBoostRiskModel, LightGBMRiskModel
from src.models.survival import CarrierSurvivalModel
from src.models.ensemble import LogisChainEnsemble

__all__ = [
    # HetGAT
    "HetGAT", "SupplyChainHeteroGraph", "GNNRiskPredictor",
    "compute_network_features", "visualize_attention_weights", "run_gnn_pipeline",
    "PYG_AVAILABLE", "SupplyChainGNN", "SupplyChainGraphBuilder", "GNNTrainer",
    # TCN
    "TCNResidualBlock", "LogisChainTCN", "QuantileLoss",
    "TemporalFeatureExtractor", "SupplyChainForecaster",
    "SupplyChainTCN", "DemandForecastPipeline",
    # Transformer
    "ShipmentEvent", "ShipmentEventEncoder",
    "ShipmentRiskTransformer", "ShipmentRiskPredictor",
    "LogisChainTransformer", "TransformerTrainer", "PositionalEncoding", "make_sequences",
    # Tabular
    "XGBoostRiskModel", "LightGBMRiskModel",
    # Survival
    "CarrierSurvivalModel",
    # Ensemble
    "LogisChainEnsemble",
]
