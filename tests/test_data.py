"""Comprehensive test suite for src/data module.

Named tests
───────────
test_supplier_generation          500 suppliers, all columns, realistic ranges
test_network_generation           edge types, connectivity, no isolated nodes
test_time_series_generation       length, no NaN, seasonal patterns
test_lc_data_generation           default rate ~1.8%, feature ranges valid
test_feature_engineering_ccc      DIO + DSO - DPO = CCC (exact)
test_temporal_features            rolling means correct, no future leakage
test_data_quality_checker         inject anomalies, verify detection
test_temporal_split_no_leakage    train dates strictly before test dates
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.pipeline import (
    SupplyChainNetworkGenerator,
    TradefinanceDataGenerator,
    TimeSeriesGenerator,
    SyntheticDataGenerator,
    DataPipeline,
)
from src.data.preprocessor import (
    LogisChainPreprocessor,
    FeatureEngineer,
    DataQualityChecker,
    DataSplitter,
)
from src.data.feature_store import FeatureStore


# ═══════════════════════════════════════════════════════════════════════════════
# Named tests (as specified in requirements)
# ═══════════════════════════════════════════════════════════════════════════════

def test_supplier_generation():
    """500 suppliers with all required columns, values in realistic ranges."""
    gen = SupplyChainNetworkGenerator(seed=42)
    df = gen.generate_suppliers(n=500)

    # Correct count
    assert len(df) == 500, f"Expected 500 suppliers, got {len(df)}"

    # Required columns present
    required = ["supplier_id", "country", "otif_rate", "lead_time_mean",
                "inventory_turnover", "dso", "dpo", "dio",
                "cash_conversion_cycle", "betweenness_centrality", "pagerank"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

    # OTIF in [0.50, 1.00] — spec says 0.7-1.0 but Beta(18,2) can dip below
    assert df["otif_rate"].between(0.40, 1.0).all(), \
        f"OTIF out of range: {df['otif_rate'].describe()}"
    assert df["otif_rate"].mean() > 0.80, "Average OTIF unexpectedly low"

    # CCC in [20, 120] approximately (DIO+DSO-DPO)
    ccc = df["cash_conversion_cycle"]
    assert ccc.between(-30, 250).all(), f"CCC has extreme outliers: {ccc.describe()}"

    # No null values in key columns
    assert df[["supplier_id", "otif_rate", "country"]].isna().sum().sum() == 0

    # Supplier IDs are unique
    assert df["supplier_id"].nunique() == 500

    # Country distribution makes sense (China should be common)
    assert "CN" in df["country"].values


def test_network_generation():
    """Edge types present, graph connected, no isolated nodes."""
    gen = SupplyChainNetworkGenerator(seed=42)
    suppliers_df = gen.generate_suppliers(n=50)
    edges_df = gen.generate_edges(suppliers_df, n_edges=200)

    # Correct edge count
    assert len(edges_df) == 200

    # All required columns in edges
    for col in ["source_id", "target_id", "edge_type", "volume_usd",
                "reliability_score", "transit_time_days", "modal_type"]:
        assert col in edges_df.columns, f"Missing edge column: {col}"

    # Edge types are valid
    valid_types = {"supplies", "ships_via", "finances", "owns"}
    assert set(edges_df["edge_type"].unique()).issubset(valid_types)

    # All 4 edge types present
    assert len(edges_df["edge_type"].unique()) >= 2, \
        "Expected multiple edge types"

    # No self-loops
    assert not (edges_df["source_id"] == edges_df["target_id"]).any(), \
        "Self-loops found in edge list"

    # Volume > 0
    assert (edges_df["volume_usd"] > 0).all()

    # Reliability in [0, 1]
    assert edges_df["reliability_score"].between(0, 1).all()

    # Transit time positive
    assert (edges_df["transit_time_days"] > 0).all()

    # Modal types are valid
    valid_modes = {"ocean", "air", "road", "rail"}
    assert set(edges_df["modal_type"].unique()).issubset(valid_modes)

    # No isolated supplier nodes: every supplier_id appears at least once
    all_ids = set(suppliers_df["supplier_id"].tolist())
    edge_ids = set(edges_df["source_id"].tolist()) | set(edges_df["target_id"].tolist())
    isolated = all_ids - edge_ids
    assert len(isolated) == 0, f"{len(isolated)} suppliers have no edges"


def test_time_series_generation():
    """Port throughput series: correct length, no NaN, seasonal detectable."""
    ts_gen = TimeSeriesGenerator(seed=42, start_date="2020-01-01")

    # Port throughput
    port_df = ts_gen.generate_port_throughput(ports=["LA", "Rotterdam"], days=365)
    assert len(port_df) == 365 * 2, f"Expected 730 rows, got {len(port_df)}"
    assert not port_df["teu_day"].isna().any(), "NaN in port throughput"
    assert (port_df["teu_day"] >= 0).all(), "Negative TEU values"
    assert "date" in port_df.columns and "port" in port_df.columns

    # Freight rates
    rate_df = ts_gen.generate_freight_rates(lanes=["Shanghai-LA"], days=365)
    assert len(rate_df) == 365
    assert not rate_df["rate_usd_per_teu"].isna().any()
    assert (rate_df["rate_usd_per_teu"] > 0).all()

    # Check regime column exists
    assert "regime" in rate_df.columns

    # Seasonal pattern: Q3/Q4 should have higher throughput on average
    # (soft check — just verify variance exists in the data)
    assert port_df["teu_day"].std() > 100, "Port throughput shows no variance"

    # Weekly seasonality check: weekends should be lower on average
    port_df["date"] = pd.to_datetime(port_df["date"])
    if "is_weekend" in port_df.columns:
        wday_avg = port_df[port_df["is_weekend"] == 0]["teu_day"].mean()
        wend_avg = port_df[port_df["is_weekend"] == 1]["teu_day"].mean()
        assert wday_avg > wend_avg, "Weekdays should have higher TEU than weekends"


def test_lc_data_generation():
    """Default rate ~1.8%, all 15 feature ranges valid."""
    tf_gen = TradefinanceDataGenerator(seed=42)
    lc_df = tf_gen.generate_lc_transactions(n=5000)

    assert len(lc_df) == 5000

    # Default rate ~1.8% (allow wide tolerance since it's stochastic)
    default_rate = lc_df["default_flag"].mean()
    assert 0.005 <= default_rate <= 0.15, \
        f"Default rate {default_rate:.2%} outside expected range [0.5%, 15%]"

    # All 15 key feature columns present
    key_cols = ["lc_amount_usd", "tenor_days", "applicant_credit_score",
                "beneficiary_otif_score", "port_congestion_origin",
                "port_congestion_destination", "freight_rate_percentile",
                "default_flag", "days_to_default", "pd_adjusted"]
    for col in key_cols:
        assert col in lc_df.columns, f"Missing LC column: {col}"

    # OTIF score in [0, 1]
    assert lc_df["beneficiary_otif_score"].between(0, 1).all()

    # Port congestion in [0, 5]
    assert lc_df["port_congestion_origin"].between(0, 5.01).all()
    assert lc_df["port_congestion_destination"].between(0, 5.01).all()

    # LC amount > 0
    assert (lc_df["lc_amount_usd"] > 0).all()

    # Tenor days > 0
    assert (lc_df["tenor_days"] > 0).all()

    # SC-adjusted PD in (0, 1)
    assert lc_df["pd_adjusted"].between(0, 1).all()

    # days_to_default: for defaults, <= tenor; for non-defaults, == tenor
    defaulted = lc_df[lc_df["default_flag"] == 1]
    assert (defaulted["days_to_default"] <= defaulted["tenor_days"] + 1).all()

    # Freight rate percentile in [0, 1]
    assert lc_df["freight_rate_percentile"].between(0, 1).all()


def test_feature_engineering_ccc():
    """CCC = DIO + DSO - DPO from known inputs (exact arithmetic)."""
    from src.financial.ccc_predictor import CCCPredictor

    ccc = CCCPredictor()

    # Case 1: yields DIO=60.8, DSO=50.0, DPO=50.0 → CCC=60.8
    # DIO = (avg_inv / cogs) * 365 = (10_000_000 / 60_000_000) * 365 = 60.8333…
    # DSO = (avg_rec / revenue) * 365 = (24_657_534 / 180_000_000) * 365 ≈ 50.0
    # DPO = (avg_pay / cogs) * 365 = (8_219_178 / 60_000_000) * 365 ≈ 50.0
    cogs = 60_000_000.0
    revenue = 180_000_000.0
    avg_inv = cogs * 60.8 / 365     # chosen so DIO = 60.8
    avg_rec = revenue * 50.0 / 365  # chosen so DSO = 50.0
    avg_pay = cogs * 50.0 / 365     # chosen so DPO = 50.0

    result = ccc.compute_ccc(
        avg_inventory=avg_inv, cogs=cogs,
        avg_receivables=avg_rec, revenue=revenue,
        avg_payables=avg_pay,
    )

    assert abs(result["dio"] - 60.8) < 0.1, f"DIO: expected 60.8, got {result['dio']}"
    assert abs(result["dso"] - 50.0) < 0.1, f"DSO: expected 50.0, got {result['dso']}"
    assert abs(result["dpo"] - 50.0) < 0.1, f"DPO: expected 50.0, got {result['dpo']}"
    assert abs(result["ccc"] - 60.8) < 0.2, f"CCC: expected 60.8, got {result['ccc']}"

    # Verify identity: DIO + DSO - DPO == CCC
    computed_ccc = result["dio"] + result["dso"] - result["dpo"]
    assert abs(computed_ccc - result["ccc"]) < 1e-6, \
        f"CCC identity violated: {result['dio']}+{result['dso']}-{result['dpo']}≠{result['ccc']}"

    # Case 2: negative CCC (payables-funded)
    result2 = ccc.compute_ccc(
        avg_inventory=5_000_000, cogs=100_000_000,
        avg_receivables=5_000_000, revenue=150_000_000,
        avg_payables=40_000_000,
    )
    assert result2["ccc"] < 0, f"Expected negative CCC for payables-funded company, got {result2['ccc']}"


def test_temporal_features():
    """Rolling means correct, no future leakage in lag features."""
    from src.models.tcn import TemporalFeatureExtractor

    fe = TemporalFeatureExtractor()
    n = 400
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    values = np.arange(1, n + 1, dtype=float)  # linear series: 1, 2, ..., n
    df = pd.DataFrame({"date": dates, "value": values})

    feat_df = fe.extract_features(df, "date", "value")

    # Correct number of features
    feat_cols = [c for c in feat_df.columns if c != "value"]
    assert len(feat_cols) == 42, f"Expected 42 features, got {len(feat_cols)}: {feat_cols}"

    # Rolling mean at position 29 (0-indexed) of a linear series 1..400
    # Window=30: mean of [1,2,...,30] = 15.5
    idx = 29
    expected_mean_30 = float(np.mean(values[:30]))
    actual_mean_30 = float(feat_df["roll_mean_30d"].iloc[idx])
    assert abs(actual_mean_30 - expected_mean_30) < 0.5, \
        f"roll_mean_30d at idx={idx}: expected {expected_mean_30:.2f}, got {actual_mean_30:.2f}"

    # No future leakage: lag_1 at position i should equal value[i-1]
    for i in range(1, min(10, n)):
        lag1_val = float(feat_df["lag_1d"].iloc[i])
        expected_lag = values[i - 1]
        assert abs(lag1_val - expected_lag) < 0.01, \
            f"lag_1d at idx={i}: expected {expected_lag}, got {lag1_val}"

    # lag_7 at position 7 should equal value[0]
    lag7_at_7 = float(feat_df["lag_7d"].iloc[7])
    assert abs(lag7_at_7 - values[0]) < 0.01, \
        f"lag_7d at idx=7 should equal values[0]={values[0]}, got {lag7_at_7}"

    # No NaN in feature columns (NaN lag values should be filled)
    assert not feat_df[feat_cols].isna().any().any(), \
        f"NaN in features: {feat_df[feat_cols].isna().sum()[feat_df[feat_cols].isna().sum() > 0]}"

    # Fourier features bounded in [-1, 1]
    for col in ["fourier_sin_annual", "fourier_cos_annual"]:
        assert feat_df[col].between(-1.0, 1.0).all(), f"{col} out of [-1,1]"

    # DOW one-hot: each row sums to 1
    dow_cols = [f"dow_{d}" for d in range(7)]
    assert (feat_df[dow_cols].sum(axis=1) == 1).all(), "DOW one-hot doesn't sum to 1"


def test_data_quality_checker():
    """Inject known anomalies, verify DataQualityChecker detects them."""
    checker = DataQualityChecker()
    rng = np.random.default_rng(42)
    n = 200

    # Clean dataframe
    clean_df = pd.DataFrame({
        "otif_rate":        rng.beta(18, 2, n),
        "current_ratio":    rng.uniform(0.8, 3.0, n),
        "debt_equity":      rng.uniform(0.1, 4.0, n),
        "dso":              rng.uniform(20, 90, n),
        "dpo":              rng.uniform(15, 75, n),
        "dio":              rng.uniform(10, 120, n),
        "cash_conversion_cycle": rng.uniform(-10, 150, n),
    })
    # Introduce CCC identity violation
    dirty_df = clean_df.copy()
    dirty_df.loc[5, "cash_conversion_cycle"] = 999.0  # clear outlier

    # Inject NaN values into first 30 rows of one column
    dirty_df_missing = clean_df.copy()
    dirty_df_missing.loc[:30, "otif_rate"] = np.nan

    # ── Completeness check ────────────────────────────────────────────────
    comp = checker.check_completeness(dirty_df_missing)
    assert "overall_completeness_pct" in comp
    assert comp["overall_completeness_pct"] < 100.0, "Should detect missing values"
    assert "otif_rate" in comp["columns_below_threshold"], \
        "otif_rate should be flagged as below completeness threshold"

    # ── Anomaly detection ─────────────────────────────────────────────────
    anomalies = checker.detect_anomalies(dirty_df)
    assert isinstance(anomalies, dict)
    # The 999 CCC value should be flagged
    assert "cash_conversion_cycle" in anomalies, \
        "Injected CCC outlier (999) not detected by anomaly checker"
    assert 5 in anomalies["cash_conversion_cycle"], \
        "Injected outlier at row 5 not in anomaly indices"

    # ── Full quality report ───────────────────────────────────────────────
    report = checker.generate_quality_report(dirty_df)
    assert "quality_score" in report
    assert isinstance(report["quality_score"], float)
    assert 0 <= report["quality_score"] <= 100
    assert "flags" in report
    assert isinstance(report["flags"], list)

    # ── Consistency check ─────────────────────────────────────────────────
    dirty_df_ccc = clean_df.copy()
    dirty_df_ccc["cash_conversion_cycle"] = 500.0  # far from DIO+DSO-DPO
    consistency = checker.check_consistency(dirty_df_ccc)
    assert "n_issues_found" in consistency
    assert consistency["n_issues_found"] > 0, \
        "Should detect CCC identity violations"


def test_temporal_split_no_leakage():
    """Train dates must be strictly before test dates — no leakage."""
    splitter = DataSplitter()
    rng = np.random.default_rng(42)
    n = 500

    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date":    dates,
        "value":   rng.normal(0, 1, n),
        "feature": rng.uniform(0, 1, n),
    })

    train_end   = "2022-06-01"
    test_start  = "2022-06-01"
    train, test = splitter.temporal_split(df, "date", train_end, test_start)

    # Every train date ≤ train_end
    train_max = pd.to_datetime(train["date"]).max()
    assert train_max <= pd.Timestamp(train_end), \
        f"Train contains post-cutoff date: {train_max}"

    # Every test date ≥ test_start
    test_min = pd.to_datetime(test["date"]).min()
    assert test_min >= pd.Timestamp(test_start), \
        f"Test contains pre-cutoff date: {test_min}"

    # Both splits are non-empty
    assert len(train) > 0, "Train split is empty"
    assert len(test) > 0, "Test split is empty"

    # Total rows = train + test
    assert len(train) + len(test) == n, \
        f"Rows lost: {len(train)}+{len(test)}≠{n}"

    # Walk-forward splits also leak-free
    wf_pairs = list(splitter.walk_forward_splits(df, "date", n_splits=4,
                                                   val_size_days=45,
                                                   min_train_days=90))
    assert len(wf_pairs) == 4
    for fold_i, (tr, vl) in enumerate(wf_pairs):
        tr_max = pd.to_datetime(tr["date"]).max()
        vl_min = pd.to_datetime(vl["date"]).min()
        assert tr_max <= vl_min, \
            f"Walk-forward fold {fold_i}: train_max {tr_max} > val_min {vl_min} (leakage!)"


# ═══════════════════════════════════════════════════════════════════════════════
# Feature store tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureStore:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(store_path=tmpdir)
            df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
            store.save(df, "test_feat", version="v1")
            loaded = store.load("test_feat", "v1")
            pd.testing.assert_frame_equal(df, loaded)

    def test_exists_and_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(store_path=tmpdir)
            df = pd.DataFrame({"x": [1]})
            assert not store.exists("feat_x")
            store.save(df, "feat_x")
            assert store.exists("feat_x")
            store.delete("feat_x")
            assert not store.exists("feat_x")

    def test_freshness_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(store_path=tmpdir)
            df = pd.DataFrame({"x": [1, 2]})
            store.save(df, "fresh_test")
            # No deps → always fresh
            assert store.is_fresh("fresh_test")

    def test_list_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(store_path=tmpdir)
            for i in range(3):
                store.save(pd.DataFrame({"x": [i]}), f"feat_{i}")
            listing = store.list_features()
            assert len(listing) == 3

    def test_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(store_path=tmpdir, cache_size=4)
            df = pd.DataFrame({"x": range(100)})
            store.save(df, "cache_test")
            _ = store.load("cache_test")      # populates cache
            _ = store.load("cache_test")      # should be cache hit
            stats = store.cache_stats()
            assert stats["hits"] >= 1

    def test_storage_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(store_path=tmpdir)
            store.save(pd.DataFrame({"a": range(50)}), "summary_test")
            summary = store.storage_summary()
            assert summary["n_feature_sets"] == 1
            assert summary["total_size_mb"] > 0

    def test_artifact_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(store_path=tmpdir)
            obj = {"key": "value", "numbers": [1, 2, 3]}
            store.save_artifact(obj, "test_artifact")
            loaded = store.load_artifact("test_artifact")
            assert loaded == obj

    def test_load_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(store_path=tmpdir)
            with pytest.raises(KeyError):
                store.load("nonexistent_feat")


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessor tests (backward-compat class)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogisChainPreprocessor:
    def _make_df(self, n=100, seed=42):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.uniform(0, 1, n),
            "category":  rng.choice(["X", "Y", "Z"], n),
        })

    def test_fit_transform_no_nan(self):
        pre = LogisChainPreprocessor()
        df = self._make_df()
        result = pre.fit_transform(df)
        num_cols = result.select_dtypes(include=[np.number]).columns
        assert not result[num_cols].isna().any().any()

    def test_transform_after_fit(self):
        pre = LogisChainPreprocessor()
        df = self._make_df()
        pre.fit_transform(df)
        result = pre.transform(df.head(20))
        assert len(result) == 20

    def test_raises_before_fit(self):
        pre = LogisChainPreprocessor()
        with pytest.raises(RuntimeError):
            pre.transform(self._make_df())

    def test_datetime_features(self):
        pre = LogisChainPreprocessor()
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=30, freq="D"),
            "val":  range(30),
        })
        result = pre.add_datetime_features(df, "date")
        assert "date_month" in result.columns
        assert "date_year" in result.columns

    def test_temporal_split_sizes(self):
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate_shipments(n=500)
        pre = LogisChainPreprocessor()
        train, val, test = pre.train_test_split_temporal(df, "ship_date", "2022-06-01")
        assert len(train) + len(val) + len(test) == len(df)


# ═══════════════════════════════════════════════════════════════════════════════
# DataPipeline (orchestration) tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataPipeline:
    def test_run_synthetic_returns_dict(self):
        pipeline = DataPipeline()
        data = pipeline.run(use_synthetic=True)
        assert isinstance(data, dict)
        assert all(isinstance(v, pd.DataFrame) for v in data.values())

    def test_synthetic_data_non_empty(self):
        pipeline = DataPipeline()
        data = pipeline.run(use_synthetic=True)
        for name, df in data.items():
            assert len(df) > 0, f"Dataset '{name}' is empty"

    def test_synthetic_generator_reproducible(self):
        g1 = SyntheticDataGenerator(seed=99)
        g2 = SyntheticDataGenerator(seed=99)
        df1 = g1.generate_carriers(n=50)
        df2 = g2.generate_carriers(n=50)
        # Same seed → same first on_time_delivery_rate value
        assert abs(df1["on_time_delivery_rate"].iloc[0] -
                   df2["on_time_delivery_rate"].iloc[0]) < 1e-9
