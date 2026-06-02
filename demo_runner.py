#!/usr/bin/env python3
"""
LogisChain AI — Demo Runner
============================
Run this before recording the video.
It pre-generates all outputs so nothing is slow during recording.

Usage:
    python demo_runner.py              # full prep
    python demo_runner.py --step data  # only generate data
    python demo_runner.py --step eval  # only run evaluation
    python demo_runner.py --launch     # prep + launch dashboard

Deliverable D2.5.2 — Zetheta Algorithms
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SEP = "═" * 60
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def info(msg): print(f"  {CYAN}→{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def hdr(msg):  print(f"\n{SEP}\n  {BOLD}{msg}{RESET}\n{SEP}")


# ─── Step 1: Generate all synthetic data ─────────────────────────────────────

def step_data():
    hdr("STEP 1 / 4 — Synthetic Data Generation")
    from src.data.pipeline import (
        SupplyChainNetworkGenerator, TradefinanceDataGenerator,
        TimeSeriesGenerator, SyntheticDataGenerator,
    )
    raw = Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)

    info("Generating 500 supplier nodes…")
    gen = SupplyChainNetworkGenerator(seed=42)
    sup = gen.generate_suppliers(n=500)
    sup.to_csv(raw / "suppliers_500.csv", index=False)
    ok(f"suppliers_500.csv  shape={sup.shape}  "
       f"avg_otif={sup['otif_rate'].mean():.2%}")

    info("Generating 2,000 supply chain edges…")
    edges = gen.generate_edges(sup, n_edges=2000)
    edges.to_csv(raw / "supply_chain_edges_2000.csv", index=False)
    ok(f"supply_chain_edges_2000.csv  shape={edges.shape}")

    info("Generating 5,000 LC transactions…")
    tf = TradefinanceDataGenerator(seed=42)
    lc = tf.generate_lc_transactions(n=5000)
    lc.to_csv(raw / "lc_transactions_5000.csv", index=False)
    ok(f"lc_transactions_5000.csv  default_rate={lc['default_flag'].mean():.2%}")

    info("Generating 2,000 SCF invoices…")
    scf = tf.generate_scf_invoices(n=2000)
    scf.to_csv(raw / "scf_invoices_2000.csv", index=False)
    ok(f"scf_invoices_2000.csv  shape={scf.shape}")

    info("Generating carriers, shipments, financial data…")
    sg = SyntheticDataGenerator(seed=42)
    data = sg.generate_all(save_path="data/raw")
    for k, v in data.items():
        ok(f"{k}.csv  shape={v.shape}")

    print(f"\n  → All datasets in {raw.resolve()}")


# ─── Step 2: Feature engineering ─────────────────────────────────────────────

def step_features():
    hdr("STEP 2 / 4 — Feature Engineering (50+ Features)")
    from src.data.pipeline import SyntheticDataGenerator
    from src.features.fusion_features import FeaturePipeline
    from src.data.feature_store import FeatureStore

    info("Loading synthetic data…")
    gen = SyntheticDataGenerator(seed=42)
    data = gen.generate_all()

    info("Running fusion feature pipeline…")
    fp = FeaturePipeline()
    fused = fp.run(data["carriers"], data["shipments"], data["financial"])

    sc_feats  = [c for c in fused.columns if any(k in c for k in ["otif","delay","centrality","turnover"])]
    fin_feats = [c for c in fused.columns if any(k in c for k in ["ccc","altman","credit","debt"])]
    fus_feats = [c for c in fused.columns if any(k in c for k in ["logischain","sc_risk","wcvi"])]

    ok(f"Fused shape: {fused.shape}")
    ok(f"SC features: {len(sc_feats)}  |  Financial: {len(fin_feats)}  |  Fusion: {len(fus_feats)}")

    store = FeatureStore(store_path="data/features")
    store.save(fused, "full_features", version="v1",
               metadata={"n_features": fused.shape[1]})
    ok("Features saved to FeatureStore (Parquet)")


# ─── Step 3: Train + evaluate models ─────────────────────────────────────────

def step_evaluate():
    hdr("STEP 3 / 4 — Model Training & Evaluation")
    import numpy as np
    from src.data.pipeline import SyntheticDataGenerator
    from src.features.fusion_features import FeaturePipeline
    from src.models.xgboost_model import XGBoostRiskModel, LightGBMRiskModel
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import roc_auc_score

    gen = SyntheticDataGenerator(seed=42)
    data = gen.generate_all()
    fp = FeaturePipeline()
    fused = fp.run(data["carriers"], data["shipments"], data["financial"])

    target = "carrier_failure" if "carrier_failure" in fused.columns else "default_flag"
    drop = [target, "carrier_id", "company_id", "name", "carrier_type", "region", "industry"]
    X = fused.drop(columns=[c for c in drop if c in fused.columns]).select_dtypes(include=np.number).fillna(0)
    y = fused[target].fillna(0)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Financial-only features
    fin_cols = [c for c in X.columns if any(k in c for k in
                ["current_ratio","quick_ratio","debt","altman","interest","credit","dso","dpo","revenue","ebitda"])]
    fin_cols = fin_cols or list(X.columns[:8])

    results = {}

    info("Training Logistic Regression (financial only)…")
    scaler = RobustScaler()
    Xf_tr = scaler.fit_transform(X_tr[fin_cols])
    Xf_te = scaler.transform(X_te[fin_cols])
    lr = LogisticRegression(max_iter=300, random_state=42)
    lr.fit(Xf_tr, y_tr)
    prob_lr = lr.predict_proba(Xf_te)[:, 1]
    results["LR (Financial Only)"] = round(roc_auc_score(y_te, prob_lr), 3)
    ok(f"AUC = {results['LR (Financial Only)']}")

    info("Training XGBoost (financial only)…")
    xgb_fin = XGBoostRiskModel(config={"n_estimators": 200})
    xgb_fin.fit(X_tr[fin_cols], y_tr)
    prob_xf = xgb_fin.predict_proba(X_te[fin_cols])
    results["XGB (Financial Only)"] = round(roc_auc_score(y_te, prob_xf), 3)
    ok(f"AUC = {results['XGB (Financial Only)']}")

    info("Training XGBoost (full SC-enhanced)…")
    xgb_sc = XGBoostRiskModel(config={"n_estimators": 300})
    xgb_sc.fit(X_tr, y_tr, X_te, y_te)
    prob_xs = xgb_sc.predict_proba(X_te)
    results["LogisChain XGB (Full)"] = round(roc_auc_score(y_te, prob_xs), 3)
    ok(f"AUC = {results['LogisChain XGB (Full)']}")

    # Print comparison table
    print(f"\n  {'Model':<30} {'AUC':>7}")
    print(f"  {'─'*38}")
    for model, auc in results.items():
        marker = " ◄ BEST" if auc == max(results.values()) else ""
        print(f"  {model:<30} {auc:>7.3f}{marker}")
    print()

    improvement = results["LogisChain XGB (Full)"] - results["LR (Financial Only)"]
    ok(f"SC improvement over baseline: +{improvement:.3f} AUC (+{improvement/results['LR (Financial Only)']*100:.1f}%)")

    import joblib
    Path("models").mkdir(exist_ok=True)
    joblib.dump(xgb_sc, "models/logischain_xgb_demo.pkl")
    ok("Model saved → models/logischain_xgb_demo.pkl")


# ─── Step 4: Pre-warm LCRiskScorer ───────────────────────────────────────────

def step_prewarm():
    hdr("STEP 4 / 4 — Pre-warming LC Risk Scorer")
    from src.data.pipeline import TradefinanceDataGenerator
    from src.financial.trade_finance_model import LCRiskScorer

    info("Loading LC data and training scorer…")
    tf = TradefinanceDataGenerator(seed=42)
    lc_df = tf.generate_lc_transactions(n=3000)
    scorer = LCRiskScorer()
    scorer.fit(lc_df)

    # Run a quick demo prediction
    demo_lc = {
        "lc_amount_usd": 2_500_000, "tenor_days": 90,
        "commodity_hs_code": "8471", "origin_country": "CN",
        "destination_country": "US", "applicant_credit_rating": "BBB",
        "beneficiary_otif_score": 0.84,
        "historical_discrepancy_rate_applicant": 0.08,
        "port_congestion_origin": 2.1,
        "port_congestion_destination": 3.4,
        "container_availability_index": 0.65,
        "freight_rate_percentile": 0.72,
        "seasonal_factor": 1.05,
        "country_risk_differential": 0.30,
        "currency_volatility_30d": 0.03,
        "historical_discrepancy_rate_beneficiary": 0.05,
    }
    result = scorer.score_lc_application(demo_lc)
    ok(f"Sample LC scored: risk={result['risk_score']:.3f}  →  {result['recommendation']}")
    ok("LC Risk Scorer warm and ready for demo")


# ─── Print video shooting order ───────────────────────────────────────────────

def print_shooting_guide():
    hdr("VIDEO SHOOTING ORDER — 10 MINUTES")
    guide = [
        ("0:00–0:45", "Home page",          "KPI cards + Architecture + Model table"),
        ("0:45–2:15", "Terminal + Browser", "Run _gen_data.py → show CSV outputs"),
        ("2:15–4:00", "Home page",          "Model table → Run Sample Prediction → result"),
        ("4:00–5:15", "Network page",       "Graph → hover node → SHAP breakdown"),
        ("5:15–6:30", "Risk Monitor",       "Alert boxes → LC form → score → counterfactual"),
        ("6:30–8:30", "LogisChain Lab",     "Start game → play 3 turns → disruption scenario"),
        ("8:30–9:30", "Explainability",     "SHAP global → attention heatmap → counterfactual"),
        ("9:30–10:00","Case Studies",       "Ever Given expand → triple-wave → close"),
    ]
    print(f"\n  {'TIME':<14} {'PAGE':<22} {'SHOW / SAY'}")
    print(f"  {'─'*70}")
    for time_, page, action in guide:
        print(f"  {time_:<14} {page:<22} {action}")
    print()
    print(f"  TOTAL: ~10 minutes | Export: 1080p H.264 | Max: 500MB\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LogisChain AI Demo Runner")
    parser.add_argument("--step", choices=["data","features","eval","prewarm","guide"],
                        help="Run only one step")
    parser.add_argument("--launch", action="store_true",
                        help="Launch Streamlit dashboard after prep")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print(f"  {BOLD}LogisChain AI — Demo Preparation{RESET}")
    print(f"  Deliverable D2.5.2 | Zetheta Algorithms")
    print(f"{SEP}")

    if args.step:
        steps = {
            "data":     step_data,
            "features": step_features,
            "eval":     step_evaluate,
            "prewarm":  step_prewarm,
            "guide":    print_shooting_guide,
        }
        steps[args.step]()
    else:
        step_data()
        step_features()
        step_evaluate()
        step_prewarm()
        print_shooting_guide()

    if args.launch:
        hdr("Launching Dashboard")
        info("Open http://localhost:8501 in your browser")
        info("Press Ctrl+C to stop recording session")
        time.sleep(1)
        os.system("python -m streamlit run demo/app.py --server.headless=true")


if __name__ == "__main__":
    main()
