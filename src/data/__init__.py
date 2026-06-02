"""LogisChain AI — data subpackage.

Generators
──────────
SupplyChainNetworkGenerator  Supplier nodes + supply chain edges
TimeSeriesGenerator          Port throughput, freight rates, vessel positions, demand
TradefinanceDataGenerator    LC transactions, SCF invoices, WC facilities

Fetchers
────────
ComtradeAPIFetcher    UN Comtrade API v1 with retry + synthetic fallback
ComtradeIngester      Backward-compatible alias

Preprocessors & tools
─────────────────────
FeatureEngineer       21 SC + 21 financial + fusion + temporal + network features
DataQualityChecker    Completeness, consistency, anomaly detection
DataSplitter          Temporal split, walk-forward CV
LogisChainPreprocessor  Impute, scale, encode (backward-compatible)
FeatureStore          Versioned Parquet store with LRU cache + dependency tracking
DataPipeline          Orchestration pipeline
SyntheticDataGenerator  Lightweight backward-compatible generator
"""

from src.data.pipeline import (
    SupplyChainNetworkGenerator,
    TimeSeriesGenerator,
    TradefinanceDataGenerator,
    ComtradeAPIFetcher,
    ComtradeIngester,
    SyntheticDataGenerator,
    DataPipeline,
)
from src.data.preprocessor import (
    FeatureEngineer,
    DataQualityChecker,
    DataSplitter,
    LogisChainPreprocessor,
)
from src.data.feature_store import FeatureStore

__all__ = [
    # Generators
    "SupplyChainNetworkGenerator",
    "TimeSeriesGenerator",
    "TradefinanceDataGenerator",
    # Fetchers
    "ComtradeAPIFetcher",
    "ComtradeIngester",
    # Orchestration
    "SyntheticDataGenerator",
    "DataPipeline",
    # Preprocessors
    "FeatureEngineer",
    "DataQualityChecker",
    "DataSplitter",
    "LogisChainPreprocessor",
    # Store
    "FeatureStore",
]
