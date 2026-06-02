"""Tests for src/financial module.

Existing (v0.1.0):
  TestTradeFinanceRiskModel     — v0.1.0 instrument pricing
  TestCCCPredictor              — v0.1.0 sklearn pipeline
  TestSupplyChainCreditScorer   — v0.1.0 credit scorer

New (v0.2.0):
  TestLCRiskScorer              — 15-feature LC scoring, backtest, fraud, pricing
  TestCCCPredictorNew           — compute_ccc, predict_ccc_change, EWS, WCVI, SCF
  TestCreditRiskScorer          — SC-PD, SHAP, TRFSI, insurance, model card
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.pipeline import SyntheticDataGenerator
from src.features.financial_features import WorkingCapitalFeatureExtractor, CreditRiskFeatureExtractor
from src.financial.trade_finance_model import TradeFinanceRiskModel, TradeFinanceInstrument, LCRiskScorer
from src.financial.ccc_predictor import CCCPredictor
from src.financial.credit_risk_scorer import SupplyChainCreditScorer, CreditRiskScorer


@pytest.fixture(scope="module")
def enriched_financial():
    gen = SyntheticDataGenerator(seed=42)
    financial = gen.generate_financial_data(300)
    financial = WorkingCapitalFeatureExtractor().extract(financial)
    financial = CreditRiskFeatureExtractor().extract(financial)
    return financial


class TestTradeFinanceRiskModel:
    def setup_method(self):
        self.model = TradeFinanceRiskModel(risk_free_rate=0.053)

    def _make_instrument(self, **kwargs):
        defaults = dict(
            instrument_id="TEST-001",
            instrument_type="LC",
            face_value_usd=1_000_000,
            tenor_days=90,
            discount_rate=0.05,
            issuer_rating="A",
            counterparty_rating="BBB",
            commodity_code="84",
            disruption_probability=0.10,
            carrier_reliability_score=0.85,
        )
        defaults.update(kwargs)
        return TradeFinanceInstrument(**defaults)

    def test_price_lc(self):
        instr = self._make_instrument()
        result = self.model.price_instrument(instr)
        assert result["spread_bps"] > 0
        assert result["present_value_usd"] < result["face_value_usd"]
        assert 0 < result["pd_estimate"] < 1

    def test_spread_increases_with_disruption(self):
        low_disrupt = self._make_instrument(disruption_probability=0.01)
        high_disrupt = self._make_instrument(disruption_probability=0.50)
        spread_low = self.model.compute_spread(low_disrupt)
        spread_high = self.model.compute_spread(high_disrupt)
        assert spread_high > spread_low

    def test_price_portfolio(self):
        instruments = [
            self._make_instrument(instrument_id=f"INS-{i:03d}", counterparty_rating=r)
            for i, r in enumerate(["AAA", "BBB", "B", "CCC"])
        ]
        df = self.model.price_portfolio(instruments)
        assert len(df) == 4
        assert "expected_loss_usd" in df.columns

    def test_scf_platform_pricing(self):
        result = self.model.scf_platform_pricing(
            anchor_rating="BBB",
            supplier_rating="B",
            invoice_amount=500_000,
        )
        assert result["early_payment_usd"] < result["invoice_amount_usd"]
        assert result["annualised_cost_pct"] > 0

    def test_riskier_rating_higher_pd(self):
        instr_bbb = self._make_instrument(counterparty_rating="BBB")
        instr_b = self._make_instrument(counterparty_rating="B")
        assert self.model._estimate_pd(instr_b) > self.model._estimate_pd(instr_bbb)


class TestCCCPredictor:
    def test_fit_predict(self, enriched_financial):
        model = CCCPredictor()
        model.fit(enriched_financial)
        preds = model.predict(enriched_financial)
        assert len(preds) == len(enriched_financial)
        assert not np.any(np.isnan(preds))

    def test_evaluate_returns_mae(self, enriched_financial):
        model = CCCPredictor()
        model.fit(enriched_financial)
        metrics = model.evaluate(enriched_financial)
        assert "mae" in metrics
        assert metrics["mae"] >= 0

    def test_feature_importance_length(self, enriched_financial):
        model = CCCPredictor()
        model.fit(enriched_financial)
        imp = model.feature_importance()
        assert not imp.empty
        assert len(imp) <= 20

    def test_sc_shock_simulation(self, enriched_financial):
        model = CCCPredictor()
        model.fit(enriched_financial)
        baseline, shocked, delta = model.simulate_sc_shock(
            enriched_financial, delay_increase_days=10.0
        )
        assert len(delta) == len(enriched_financial)
        # Shock should increase CCC on average
        assert np.mean(delta) >= 0


class TestSupplyChainCreditScorer:
    def test_fit_and_score(self, enriched_financial):
        scorer = SupplyChainCreditScorer()
        scorer.fit(enriched_financial)
        pds = scorer.score(enriched_financial)
        assert len(pds) == len(enriched_financial)
        assert (pds >= 0).all() and (pds <= 1).all()

    def test_evaluate_auc_reasonable(self, enriched_financial):
        scorer = SupplyChainCreditScorer()
        scorer.fit(enriched_financial)
        metrics = scorer.evaluate(enriched_financial)
        assert "roc_auc" in metrics
        assert 0.4 <= metrics["roc_auc"] <= 1.0  # sanity check

    def test_score_entities_returns_results(self, enriched_financial):
        scorer = SupplyChainCreditScorer()
        scorer.fit(enriched_financial)
        results = scorer.score_entities(enriched_financial.head(20))
        assert len(results) == 20
        assert all(hasattr(r, "pd_estimate") for r in results)
        assert all(r.risk_tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"] for r in results)

    def test_portfolio_expected_loss(self, enriched_financial):
        scorer = SupplyChainCreditScorer()
        scorer.fit(enriched_financial)
        results = scorer.score_entities(enriched_financial.head(50))
        summary = scorer.portfolio_expected_loss(results)
        assert summary["total_ead_usd"] > 0
        assert "tier_distribution" in summary

    def test_sc_contribution_sums_to_one(self, enriched_financial):
        scorer = SupplyChainCreditScorer()
        scorer.fit(enriched_financial)
        results = scorer.score_entities(enriched_financial.head(10))
        for r in results:
            total = r.sc_disruption_contribution + r.financial_stress_contribution
            assert abs(total - 1.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# v0.2.0 — LCRiskScorer tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_lc_record(**overrides) -> dict:
    base = {
        "lc_amount_usd":         1_000_000,
        "tenor_days":            90,
        "commodity_hs_code":     "8471",
        "origin_country":        "CN",
        "destination_country":   "US",
        "applicant_credit_rating": "BBB",
        "beneficiary_otif_score": 0.88,
        "historical_discrepancy_rate_applicant":   0.06,
        "historical_discrepancy_rate_beneficiary": 0.04,
        "port_congestion_origin":      1.5,
        "port_congestion_destination": 2.0,
        "container_availability_index": 0.72,
        "freight_rate_percentile":      0.60,
        "seasonal_factor":              1.05,
        "country_risk_differential":    0.30,
        "currency_volatility_30d":      0.025,
        "default_flag":                 0,
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def lc_scorer():
    gen = SyntheticDataGenerator(seed=42)
    from src.data.pipeline import TradefinanceDataGenerator
    tf_gen = TradefinanceDataGenerator(seed=42)
    lc_df = tf_gen.generate_lc_transactions(n=500)
    scorer = LCRiskScorer()
    scorer.fit(lc_df)
    return scorer, lc_df


class TestLCRiskScorer:

    def test_compute_lc_features_length(self):
        scorer = LCRiskScorer()
        feats = scorer.compute_lc_features(_make_lc_record())
        assert len(feats) == LCRiskScorer.N_FEATURES == 15

    def test_compute_lc_features_no_nan(self):
        scorer = LCRiskScorer()
        feats = scorer.compute_lc_features(_make_lc_record())
        assert not np.isnan(feats).any(), "NaN in LC features"

    def test_feature_values_bounded(self):
        scorer = LCRiskScorer()
        feats = scorer.compute_lc_features(_make_lc_record())
        # All normalised features should be in a reasonable numeric range
        assert np.isfinite(feats).all()

    def test_high_risk_features_give_higher_score(self):
        scorer = LCRiskScorer()
        low_risk  = _make_lc_record(beneficiary_otif_score=0.98,
                                     port_congestion_destination=0.5,
                                     historical_discrepancy_rate_applicant=0.01)
        high_risk = _make_lc_record(beneficiary_otif_score=0.70,
                                     port_congestion_destination=4.5,
                                     historical_discrepancy_rate_applicant=0.35)
        s_lo = scorer._predict_score(scorer.compute_lc_features(low_risk))
        s_hi = scorer._predict_score(scorer.compute_lc_features(high_risk))
        assert s_hi > s_lo, "High-risk features should produce higher score"

    def test_score_lc_application_keys(self, lc_scorer):
        scorer, lc_df = lc_scorer
        result = scorer.score_lc_application(lc_df.iloc[0].to_dict())
        required = {"risk_score", "risk_level", "recommendation",
                    "conditions", "key_risks", "shap_explanation",
                    "comparable_transactions", "pricing"}
        assert required.issubset(set(result.keys()))

    def test_risk_score_in_range(self, lc_scorer):
        scorer, lc_df = lc_scorer
        for i in range(min(20, len(lc_df))):
            r = scorer.score_lc_application(lc_df.iloc[i].to_dict())
            assert 0.0 <= r["risk_score"] <= 1.0

    def test_risk_level_is_valid(self, lc_scorer):
        scorer, lc_df = lc_scorer
        valid_levels = {"LOW", "MEDIUM-LOW", "MEDIUM", "MEDIUM-HIGH", "HIGH"}
        for i in range(min(10, len(lc_df))):
            r = scorer.score_lc_application(lc_df.iloc[i].to_dict())
            assert r["risk_level"] in valid_levels

    def test_recommendation_is_valid(self, lc_scorer):
        scorer, lc_df = lc_scorer
        valid_recs = {"APPROVE", "APPROVE_WITH_CONDITIONS", "DECLINE"}
        for i in range(min(10, len(lc_df))):
            r = scorer.score_lc_application(lc_df.iloc[i].to_dict())
            assert r["recommendation"] in valid_recs

    def test_comparable_transactions_list(self, lc_scorer):
        scorer, lc_df = lc_scorer
        r = scorer.score_lc_application(lc_df.iloc[0].to_dict())
        assert isinstance(r["comparable_transactions"], list)
        if r["comparable_transactions"]:
            comp = r["comparable_transactions"][0]
            assert "risk_score" in comp and "outcome" in comp

    def test_shap_explanation_is_dict(self, lc_scorer):
        scorer, lc_df = lc_scorer
        r = scorer.score_lc_application(lc_df.iloc[0].to_dict())
        assert isinstance(r["shap_explanation"], dict)
        assert len(r["shap_explanation"]) > 0

    def test_backtest_returns_four_models(self, lc_scorer):
        scorer, lc_df = lc_scorer
        result = scorer.backtest_model(lc_df)
        assert "metrics" in result
        assert len(result["metrics"]) == 4
        model_keys = set(result["metrics"].keys())
        assert "logistic_regression_financial_only" in model_keys
        assert "logischain_ai_full" in model_keys

    def test_backtest_metrics_structure(self, lc_scorer):
        scorer, lc_df = lc_scorer
        result = scorer.backtest_model(lc_df)
        for model_name, m in result["metrics"].items():
            assert "auc" in m and "gini" in m and "ks" in m and "ece" in m
            assert 0 <= m["auc"] <= 1.0, f"AUC out of range for {model_name}"
            assert -1 <= m["gini"] <= 1.0

    def test_detect_phantom_shipment_no_ais(self):
        scorer = LCRiskScorer()
        lc = _make_lc_record()
        result = scorer.detect_phantom_shipment(lc, ais_data=None)
        assert "fraud_probability" in result
        assert "flags" in result
        assert "recommendation" in result
        assert result["fraud_probability"] > 0  # no AIS → some suspicion

    def test_detect_phantom_shipment_with_clean_ais(self):
        scorer = LCRiskScorer()
        lc = _make_lc_record()
        ais = {"vessel_imo": "9876543", "confirmed_at_origin": True,
               "actual_transit_days": 14, "avg_speed_knots": 14.5}
        result = scorer.detect_phantom_shipment(lc, ais_data=ais)
        assert result["fraud_probability"] < 0.5  # clean AIS → lower fraud prob

    def test_detect_phantom_shipment_high_risk_route(self):
        scorer = LCRiskScorer()
        lc = _make_lc_record(origin_country="PK", destination_country="US")
        result = scorer.detect_phantom_shipment(lc)
        assert result["fraud_probability"] > 0.2
        assert result["recommendation"] in {"REFER_TO_FRAUD_TEAM",
                                             "ENHANCED_DUE_DILIGENCE",
                                             "ADDITIONAL_DOCUMENTATION_REQUIRED"}

    def test_detect_phantom_flags_is_list(self):
        scorer = LCRiskScorer()
        result = scorer.detect_phantom_shipment(_make_lc_record())
        assert isinstance(result["flags"], list)

    def test_price_lc_fee_keys(self, lc_scorer):
        scorer, lc_df = lc_scorer
        price = scorer.price_lc_fee(lc_df.iloc[0].to_dict())
        assert "base_fee_pct" in price
        assert "risk_adjustment_pct" in price
        assert "total_fee_pct" in price
        assert "annual_revenue_usd" in price

    def test_price_lc_fee_total_above_base(self, lc_scorer):
        scorer, lc_df = lc_scorer
        price = scorer.price_lc_fee(lc_df.iloc[0].to_dict())
        assert price["total_fee_pct"] >= price["base_fee_pct"]

    def test_price_lc_fee_risk_adj_non_negative(self, lc_scorer):
        scorer, lc_df = lc_scorer
        for i in range(min(10, len(lc_df))):
            price = scorer.price_lc_fee(lc_df.iloc[i].to_dict())
            assert price["risk_adjustment_pct"] >= 0.0

    def test_fit_and_score_consistency(self, lc_scorer):
        scorer, lc_df = lc_scorer
        # Riskier records should generally score higher
        high_risk_lc = _make_lc_record(beneficiary_otif_score=0.60,
                                        port_congestion_destination=4.8,
                                        historical_discrepancy_rate_applicant=0.40)
        low_risk_lc  = _make_lc_record(beneficiary_otif_score=0.97,
                                        port_congestion_destination=0.5,
                                        historical_discrepancy_rate_applicant=0.01)
        hi = scorer.score_lc_application(high_risk_lc)["risk_score"]
        lo = scorer.score_lc_application(low_risk_lc)["risk_score"]
        assert hi > lo


# ═══════════════════════════════════════════════════════════════════════════════
# v0.2.0 — CCCPredictor (new methods) tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCCCPredictorNew:

    def test_compute_ccc_formula(self):
        ccc = CCCPredictor()
        result = ccc.compute_ccc(
            avg_inventory=12_000_000, cogs=85_000_000,
            avg_receivables=22_000_000, revenue=120_000_000,
            avg_payables=9_000_000,
        )
        # DIO = 12/85 × 365 ≈ 51.5 days
        assert 40 < result["dio"] < 65
        # DSO = 22/120 × 365 ≈ 66.9 days
        assert 55 < result["dso"] < 80
        # DPO = 9/85 × 365 ≈ 38.6 days
        assert 30 < result["dpo"] < 50
        # CCC = DIO + DSO - DPO
        expected_ccc = result["dio"] + result["dso"] - result["dpo"]
        assert abs(result["ccc"] - expected_ccc) < 0.01

    def test_compute_ccc_keys(self):
        ccc = CCCPredictor()
        result = ccc.compute_ccc(1_000_000, 5_000_000, 500_000, 6_000_000, 300_000)
        assert set(result.keys()) == {"dio", "dso", "dpo", "ccc"}

    def test_compute_ccc_all_positive(self):
        ccc = CCCPredictor()
        result = ccc.compute_ccc(1e6, 5e6, 5e5, 6e6, 3e5)
        assert result["dio"] > 0 and result["dso"] > 0 and result["dpo"] > 0

    def test_predict_ccc_change_meddevice(self):
        """Reproduce MedDevice Corp: net CCC +26 days."""
        ccc = CCCPredictor()
        ccc._company_ccc["MedDevice"] = 72.0
        ccc.covenant_thresholds["MedDevice"] = 98.0
        signals = {
            "otif_change": -0.12,
            "port_congestion_change": 2.1,
            "lead_time_var_change": 4.2,
            "freight_rate_change": 0.35,
        }
        pred = ccc.predict_ccc_change("MedDevice", signals, horizon_days=90)
        # DIO from OTIF alone: -150 × (-0.12) = +18 days
        assert abs(pred["dio_change"] - 18) < 5, \
            f"OTIF-driven DIO change expected ~18d, got {pred['dio_change']}"
        # Net CCC change ≈ +26 days
        assert 20 < pred["ccc_change"] < 35, \
            f"Net CCC change expected ~26d, got {pred['ccc_change']}"
        # Covenant breach = True (72 + 26 > 98)
        assert pred["covenant_breach"] is True
        assert pred["breach_probability"] > 0.70

    def test_predict_ccc_change_no_signals(self):
        ccc = CCCPredictor()
        pred = ccc.predict_ccc_change("CO-001", {})
        assert pred["ccc_change"] == 0.0
        assert pred["dio_change"] == 0.0
        assert pred["covenant_breach"] is False

    def test_predict_ccc_change_keys(self):
        ccc = CCCPredictor()
        pred = ccc.predict_ccc_change("CO-001", {"otif_change": -0.05})
        required = {"current_ccc", "predicted_ccc", "ccc_change", "dio_change",
                    "dso_change", "dpo_change", "covenant_breach",
                    "breach_probability", "confidence_interval", "key_drivers"}
        assert required.issubset(set(pred.keys()))

    def test_predict_ccc_confidence_interval_order(self):
        ccc = CCCPredictor()
        pred = ccc.predict_ccc_change("CO-001", {"otif_change": -0.10})
        lo, hi = pred["confidence_interval"]
        assert lo <= pred["predicted_ccc"] <= hi

    def test_early_warning_system_returns_dataframe(self):
        ccc = CCCPredictor()
        rng = np.random.default_rng(42)
        n = 8
        portfolio = pd.DataFrame({
            "company_id":             [f"CO-{i:03d}" for i in range(n)],
            "current_ccc":            rng.uniform(50, 90, n),
            "otif_change":            rng.uniform(-0.15, 0.02, n),
            "port_congestion_change": rng.uniform(-1, 3, n),
            "lead_time_var_change":   rng.uniform(-1, 5, n),
            "freight_rate_change":    rng.uniform(0, 0.4, n),
        })
        alerts = ccc.early_warning_system(portfolio)
        assert isinstance(alerts, pd.DataFrame)
        assert len(alerts) == n
        assert "traffic_light" in alerts.columns
        assert set(alerts["traffic_light"].unique()).issubset({"RED", "AMBER", "GREEN"})

    def test_early_warning_system_sorted_by_breach_prob(self):
        ccc = CCCPredictor()
        rng = np.random.default_rng(7)
        n = 6
        portfolio = pd.DataFrame({
            "company_id":             [f"CO-{i}" for i in range(n)],
            "current_ccc":            rng.uniform(50, 90, n),
            "otif_change":            rng.uniform(-0.20, 0.05, n),
            "port_congestion_change": rng.uniform(0, 3, n),
            "lead_time_var_change":   rng.uniform(0, 5, n),
            "freight_rate_change":    rng.uniform(0, 0.5, n),
        })
        alerts = ccc.early_warning_system(portfolio)
        probs = alerts["breach_probability"].values
        assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1)), \
            "EWS should be sorted descending by breach_probability"

    def test_compute_wcvi_returns_float(self):
        ccc = CCCPredictor()
        rng = np.random.default_rng(42)
        n = 15
        df = pd.DataFrame({
            "date":             pd.date_range("2022-01-01", periods=n, freq="MS"),
            "cogs":             rng.normal(100, 5, n),
            "revenue":          rng.normal(120, 6, n),
            "avg_inventory":    rng.normal(20, 2, n),
            "avg_receivables":  rng.normal(30, 3, n),
            "avg_payables":     rng.normal(10, 1, n),
        })
        wcvi = ccc.compute_wcvi(df)
        assert isinstance(wcvi, float)
        assert np.isfinite(wcvi)

    def test_compute_wcvi_short_series(self):
        ccc = CCCPredictor()
        df = pd.DataFrame({"cogs": [100], "revenue": [120],
                            "avg_inventory": [20], "avg_receivables": [30], "avg_payables": [10]})
        wcvi = ccc.compute_wcvi(df)
        assert wcvi == 0.0  # insufficient data

    def test_scf_optimization_dpo_increases(self):
        ccc = CCCPredictor()
        plan = ccc.scf_optimization(current_dpo=50, current_dio=68, current_dso=42,
                                     target_ccc_reduction_days=25)
        assert plan["new_dpo"] > plan["current_dpo"]
        assert plan["ccc_reduction"] >= 0

    def test_scf_optimization_capital_released_positive(self):
        ccc = CCCPredictor()
        plan = ccc.scf_optimization(50, 68, 42, 25, annual_revenue_usd=375_000_000)
        assert plan["capital_released_usd"] > 0
        # DPO 50→75: $375M × 25/365 ≈ $25.7M
        expected = 375_000_000 * 25 / 365
        assert abs(plan["capital_released_usd"] - expected) < 500_000

    def test_scf_optimization_keys(self):
        ccc = CCCPredictor()
        plan = ccc.scf_optimization(50, 68, 42, 25)
        required = {"current_dpo", "new_dpo", "current_ccc", "new_ccc",
                    "ccc_reduction", "capital_released_usd", "narrative"}
        assert required.issubset(set(plan.keys()))

    def test_predict_covenant_breach_timeline_returns_dict(self):
        ccc = CCCPredictor()
        ccc._company_ccc["CO-X"] = 72.0
        ccc.covenant_thresholds["CO-X"] = 95.0
        timeline = ccc.predict_covenant_breach_timeline(
            "CO-X", {"otif_change": -0.12}, forecast_days=90
        )
        assert "timeline" in timeline
        assert "breach_day" in timeline
        assert isinstance(timeline["timeline"], list)
        assert len(timeline["timeline"]) == 91  # days 0..90

    def test_predict_covenant_breach_timeline_values_increasing(self):
        ccc = CCCPredictor()
        ccc._company_ccc["CO-Y"] = 72.0
        ccc.covenant_thresholds["CO-Y"] = 110.0  # covenant well above current
        timeline = ccc.predict_covenant_breach_timeline(
            "CO-Y", {"otif_change": -0.10}, forecast_days=30
        )
        days_cccs = [v for _, v in timeline["timeline"]]
        # CCC should be monotonically increasing (positive shock)
        assert days_cccs[-1] >= days_cccs[0]


# ═══════════════════════════════════════════════════════════════════════════════
# v0.2.0 — CreditRiskScorer tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreditRiskScorer:

    def test_autoparts_corp_sc_pd(self):
        """AutoParts Corp example: PD 2.5% → 3.33%, uplift 33%."""
        scorer = CreditRiskScorer()
        res = scorer.compute_sc_adjusted_pd(
            traditional_pd=0.025,
            sc_metrics={"otif_rate": 0.85, "inventory_turnover": 4.8,
                        "alt_supplier_count": 1, "base_lc_fee_pct": 1.25},
        )
        assert abs(res["sc_pd"] - 0.0333) < 0.002, \
            f"Expected SC-PD ≈ 3.33%, got {res['sc_pd']*100:.3f}%"
        assert abs(res["risk_uplift_pct"] - 33.0) < 2.0, \
            f"Expected uplift ≈ 33%, got {res['risk_uplift_pct']:.1f}%"

    def test_sc_pd_keys(self):
        scorer = CreditRiskScorer()
        res = scorer.compute_sc_adjusted_pd(0.025, {"otif_rate": 0.85,
                                                      "inventory_turnover": 4.8,
                                                      "alt_supplier_count": 1})
        required = {"traditional_pd", "sc_pd", "sc_pd_pct", "risk_uplift_pct",
                    "otif_contribution", "inventory_contribution", "network_contribution",
                    "pricing_impact"}
        assert required.issubset(set(res.keys()))

    def test_sc_pd_no_uplift_for_good_sc(self):
        scorer = CreditRiskScorer()
        res = scorer.compute_sc_adjusted_pd(0.025, {"otif_rate": 0.95,
                                                      "inventory_turnover": 8.0,
                                                      "alt_supplier_count": 5})
        assert res["sc_pd"] == res["traditional_pd"], "No uplift expected for strong SC"
        assert res["risk_uplift_pct"] == 0.0

    def test_sc_pd_larger_than_traditional(self):
        scorer = CreditRiskScorer()
        res = scorer.compute_sc_adjusted_pd(0.03, {"otif_rate": 0.70,
                                                    "inventory_turnover": 2.0,
                                                    "alt_supplier_count": 0})
        assert res["sc_pd"] > res["traditional_pd"]

    def test_pricing_impact_adjusted_above_base(self):
        scorer = CreditRiskScorer()
        res = scorer.compute_sc_adjusted_pd(0.025, {"otif_rate": 0.85,
                                                      "inventory_turnover": 4.8,
                                                      "alt_supplier_count": 1,
                                                      "base_lc_fee_pct": 1.25})
        assert res["pricing_impact"]["adjusted_fee_pct"] >= res["pricing_impact"]["base_fee_pct"]

    def test_shap_explanation_keys(self):
        scorer = CreditRiskScorer()
        res = scorer.compute_shap_explanation("CO-001", {
            "otif_rate": 0.85, "cash_conversion_cycle": 78,
            "inventory_turnover": 4.8, "customer_concentration_hhi": 0.38,
            "current_ratio": 1.41, "ebitda_margin": 0.12,
            "betweenness_centrality": 0.34,
        })
        assert "base_value_pct" in res
        assert "shap_contributions" in res
        assert "final_pd_pct" in res
        assert "rating" in res

    def test_shap_explanation_base_plus_contribs_approx_final(self):
        scorer = CreditRiskScorer()
        res = scorer.compute_shap_explanation("CO-001", {
            "otif_rate": 0.85, "cash_conversion_cycle": 78,
            "inventory_turnover": 4.8, "customer_concentration_hhi": 0.38,
            "current_ratio": 1.41, "ebitda_margin": 0.12,
            "betweenness_centrality": 0.34,
        })
        total = res["base_value_pct"] + sum(res["shap_contributions"].values())
        assert abs(total - res["final_pd_pct"]) < 0.5, \
            f"Base + contribs ≠ final: {total:.3f} vs {res['final_pd_pct']:.3f}"

    def test_trfsi_range(self):
        scorer = CreditRiskScorer()
        t = scorer.compute_trfsi("Shanghai-LA", 0.68, 0.45, 0.22, 0.30)
        assert 0.0 <= t <= 1.0

    def test_trfsi_higher_with_worse_conditions(self):
        scorer = CreditRiskScorer()
        good = scorer.compute_trfsi("Shanghai-LA", 0.10, 0.10, 0.05, 0.10)
        bad  = scorer.compute_trfsi("Shanghai-LA", 0.90, 0.80, 0.70, 0.60)
        assert bad > good

    def test_trfsi_clamps_to_zero_one(self):
        scorer = CreditRiskScorer()
        t = scorer.compute_trfsi("any", 1.5, 2.0, 1.2, 0.5)  # > 1 inputs
        assert 0.0 <= t <= 1.0

    def test_score_borrower_keys(self):
        scorer = CreditRiskScorer()
        res = scorer.score_borrower(
            "CO-001",
            {"credit_rating": "BBB", "altman_z_score": 3.5, "current_ratio": 1.8,
             "cash_conversion_cycle": 60},
            {"otif_rate": 0.88, "inventory_turnover": 5.5, "alt_supplier_count": 2},
        )
        required = {"pd", "lgd", "ead_usd", "expected_loss_usd", "rating",
                    "watch_flags", "sc_risk_factors", "traditional_pd", "risk_uplift_pct"}
        assert required.issubset(set(res.keys()))

    def test_score_borrower_pd_in_range(self):
        scorer = CreditRiskScorer()
        res = scorer.score_borrower("CO-001",
                                     {"credit_rating": "BBB", "altman_z_score": 3.5},
                                     {"otif_rate": 0.88, "inventory_turnover": 5.5, "alt_supplier_count": 2})
        assert 0.0 < res["pd"] < 1.0

    def test_score_borrower_expected_loss_positive(self):
        scorer = CreditRiskScorer()
        res = scorer.score_borrower("CO-001",
                                     {"credit_rating": "B", "revenue_usd": 5_000_000},
                                     {"otif_rate": 0.75, "inventory_turnover": 3.0, "alt_supplier_count": 1})
        assert res["expected_loss_usd"] > 0

    def test_monitor_portfolio_returns_dataframe(self):
        scorer = CreditRiskScorer()
        rng = np.random.default_rng(42)
        n = 8
        portfolio = pd.DataFrame({
            "company_id":        [f"CO-{i:03d}" for i in range(n)],
            "credit_rating":     rng.choice(["BBB", "BB", "B", "A"], n),
            "altman_z_score":    rng.uniform(1.0, 4.0, n),
            "current_ratio":     rng.uniform(0.8, 3.0, n),
            "otif_rate":         rng.uniform(0.65, 0.97, n),
            "inventory_turnover": rng.uniform(2.0, 9.0, n),
            "alt_supplier_count": rng.integers(0, 5, n),
            "revenue_usd":       rng.lognormal(15, 1, n),
        })
        result = scorer.monitor_portfolio(portfolio)
        assert len(result) == n
        assert "traffic_light" in result.columns
        assert set(result["traffic_light"].unique()).issubset({"RED", "AMBER", "GREEN"})

    def test_monitor_portfolio_sorted_by_uplift(self):
        scorer = CreditRiskScorer()
        rng = np.random.default_rng(99)
        n = 6
        portfolio = pd.DataFrame({
            "company_id": [f"CO-{i}" for i in range(n)],
            "credit_rating": rng.choice(["BBB", "B", "CCC"], n),
            "altman_z_score": rng.uniform(0.5, 3.5, n),
            "otif_rate": rng.uniform(0.60, 0.99, n),
            "inventory_turnover": rng.uniform(1.5, 8.0, n),
            "alt_supplier_count": rng.integers(0, 4, n),
        })
        result = scorer.monitor_portfolio(portfolio)
        uplifts = result["risk_uplift_pct"].values
        assert all(uplifts[i] >= uplifts[i + 1] for i in range(len(uplifts) - 1))

    def test_cargo_insurance_mv_pacific_star(self):
        """Reproduce MV Pacific Star: base 0.60%, adjusted 1.14%, premium $28,500."""
        scorer = CreditRiskScorer()
        prem = scorer.compute_dynamic_cargo_insurance_premium({
            "base_rate_pct":             0.60,
            "cyclone_probability":       0.35,
            "weather_severity":          0.43,
            "carrier_reliability_score": 0.78,
            "port_congestion_index":     3.2,
            "cargo_type":                "electronics",
            "cargo_value_usd":           2_500_000,
        })
        assert abs(prem["adjusted_rate_pct"] - 1.14) < 0.05, \
            f"Expected ≈1.14%, got {prem['adjusted_rate_pct']:.3f}%"
        assert abs(prem["adjusted_premium_usd"] - 28_500) < 1_000, \
            f"Expected ≈$28,500, got ${prem['adjusted_premium_usd']:,.0f}"

    def test_cargo_insurance_keys(self):
        scorer = CreditRiskScorer()
        prem = scorer.compute_dynamic_cargo_insurance_premium({
            "cargo_value_usd": 1_000_000, "cargo_type": "general_cargo",
        })
        required = {"base_rate_pct", "weather_uplift_pct", "carrier_uplift_pct",
                    "congestion_uplift_pct", "cargo_sensitivity_multiplier",
                    "adjusted_rate_pct", "adjusted_premium_usd", "standard_premium_usd"}
        assert required.issubset(set(prem.keys()))

    def test_cargo_insurance_adjusted_above_standard(self):
        scorer = CreditRiskScorer()
        prem = scorer.compute_dynamic_cargo_insurance_premium({
            "cargo_value_usd": 500_000,
            "carrier_reliability_score": 0.70,
            "port_congestion_index": 3.0,
            "cargo_type": "electronics",
        })
        assert prem["adjusted_premium_usd"] >= prem["standard_premium_usd"]

    def test_generate_model_card_keys(self):
        scorer = CreditRiskScorer()
        card = scorer.generate_model_card()
        required = {"model_id", "model_name", "purpose", "scope", "model_type",
                    "performance", "model_risk_rating", "monitoring", "compliance",
                    "limitations", "conceptual_soundness"}
        assert required.issubset(set(card.keys()))

    def test_generate_model_card_compliance(self):
        scorer = CreditRiskScorer()
        card = scorer.generate_model_card()
        assert card["compliance"]["sr_11_7"] is True
        assert card["compliance"]["basel_iii"] is True
        assert card["compliance"]["ifrs9"] is True

    def test_generate_model_card_performance_auc(self):
        scorer = CreditRiskScorer()
        card = scorer.generate_model_card()
        assert 0.80 <= card["performance"]["auc_roc"] <= 1.0

    def test_generate_model_card_monitoring_psi(self):
        scorer = CreditRiskScorer()
        card = scorer.generate_model_card()
        assert card["monitoring"]["psi_threshold"] == 0.20

    def test_fit_and_score_after_training(self):
        scorer = CreditRiskScorer()
        gen = SyntheticDataGenerator(seed=42)
        fin = gen.generate_financial_data(300)
        fin = WorkingCapitalFeatureExtractor().extract(fin)
        fin = CreditRiskFeatureExtractor().extract(fin)
        scorer.fit(fin, target_col="default_flag")
        assert scorer._fitted
        # After fitting, score_borrower should still work (analytical fallback)
        res = scorer.score_borrower("CO-001",
                                     {"credit_rating": "BB"},
                                     {"otif_rate": 0.82, "inventory_turnover": 4.0,
                                      "alt_supplier_count": 1})
        assert 0.0 < res["pd"] < 1.0
