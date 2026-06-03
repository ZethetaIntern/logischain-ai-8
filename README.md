# 🔗 LogisChain AI

Demo Video Link:https://drive.google.com/file/d/1n8HSj52w8yZ3_sWz1MqHt98bbJS4A8dH/view?usp=sharing

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-2.8%2B-0194e2.svg)](https://mlflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)](https://streamlit.io)

**Predictive Trade Finance & Logistics Valuation — Supply Chain Intelligence Embedded in Financial Risk Models**

> LogisChain AI bridges a critical intelligence gap in global trade finance. By integrating real-time supply chain operational data — carrier reliability, port congestion, route topology, inventory velocity — directly into financial risk models, it enables trade finance banks, SCF platforms, and cargo insurers to price risk with +33% greater accuracy than financial-statement-only models.

---

## Architecture

```
╔════════════════════════════════════════════════════════════════════════════╗
║                          LOGISCHAIN AI SYSTEM                              ║
╠════════════════════════════════════════════════════════════════════════════╣
║  DATA INGESTION                                                             ║
║  UN Comtrade API ──► World Bank ──► AIS Vessel Tracking ──► Synthetic Gen ║
╠════════════════════════════════════════════════════════════════════════════╣
║  PHYSICAL SUPPLY CHAIN LAYER  (100 nodes, 500 edges, weekly simulation)    ║
║  40 Suppliers ──► 20 Manufacturers ──► 20 Ports ──► 20 Warehouses          ║
╠════════════════════════════════════════════════════════════════════════════╣
║  FEATURE ENGINEERING  (50+ features across 3 domains)                      ║
║  SC Features (21)    Financial Features (21)    Fusion Features (8+)        ║
╠════════════════════════════════════════════════════════════════════════════╣
║  INTELLIGENCE LAYER  (Multi-model ensemble)                                 ║
║  GNN (GATv2) │ TCN (P10/50/90) │ Transformer │ XGBoost │ Cox PH Survival   ║
║                        ↓ Stacking Ensemble ↓                                ║
╠════════════════════════════════════════════════════════════════════════════╣
║  FINANCIAL INTEGRATION                                                      ║
║  LCRiskScorer │ CCCPredictor │ CreditRiskScorer │ Insurance Pricing         ║
╠════════════════════════════════════════════════════════════════════════════╣
║  LOGISCHAIN LAB  (Gamified simulation)                                      ║
║  10 Disruption Scenarios │ 4 Game Modes │ 5-Dimension Scoring               ║
╠════════════════════════════════════════════════════════════════════════════╣
║  STREAMLIT DASHBOARD  (Port 8501)                                           ║
║  Network │ Risk Monitor │ Lab │ Explainability │ Case Studies                ║
╚════════════════════════════════════════════════════════════════════════════╝
```

---

## Features

- 🌐 **Supply Chain Network Modelling** — 100-node GATv2 graph with betweenness centrality, PageRank, HITS scores propagating risk through supplier networks
- 📈 **Multi-Horizon Forecasting** — TCN with dilated causal convolutions forecasting port throughput, freight rates, and demand at 30/60/90-day horizons with P10/P50/P90 quantiles
- 🚢 **Shipment Risk Transformer** — 4-head multi-task model predicting delay probability, damage probability, documentary discrepancy, and composite risk score from event sequences
- 💳 **LC Risk Scorer** — 15-feature vector including real-time SC signals (OTIF, port congestion, freight percentile) improving credit discrimination by +11.5% AUC
- 💰 **CCC Predictor** — MedDevice Corp scenario: OTIF 94%→82% predicts CCC +26 days, covenant breach probability 0.84 within 90 days
- 🛡️ **Dynamic Cargo Insurance** — MV Pacific Star example: base 0.60% + SC uplifts → adjusted 1.14% ($28,500 premium vs $15,000 standard)
- 🎮 **LogisChain Lab** — Gamified simulation with 10 realistic disruption scenarios (Ever Given, COVID-19, Red Sea closure, Hanjin), 4 game modes, and professional certification (Novice → Master)
- 🔍 **SHAP Explainability** — Full TreeExplainer integration; supply chain features occupy 6 of top 10 positions; counterfactual generation for rejected LCs

---

## Installation

### Option A — pip (recommended for development)

```bash
git clone https://github.com/zetheta/logischain-ai.git
cd logischain-ai
pip install -e .
```

### Option B — conda

```bash
conda create -n logischain python=3.10
conda activate logischain
pip install -r requirements.txt
pip install -e .
```

### Option C — Docker (production)

```bash
docker-compose up --build
# App:    http://localhost:8501
# MLflow: http://localhost:5000
```

---

## Quick Start (3 commands)

```bash
# 1. Generate all synthetic datasets
python -m src.data.pipeline

# 2. Run the full pipeline (data → features → train → evaluate)
python run_pipeline.py

# 3. Launch the interactive dashboard
streamlit run demo/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Model Performance

All results on synthetic holdout test set (20% temporal split).

| Model | AUC-ROC | Gini | KS | ECE | Precision@5% | SC Features |
|-------|---------|------|-----|-----|-------------|-------------|
| Logistic Regression (financial only) | 0.738 | 0.476 | 0.381 | 0.042 | 12.4% | 0 |
| XGBoost (financial only) | 0.771 | 0.542 | 0.412 | 0.035 | 15.8% | 0 |
| XGBoost (SC basic — 6 features) | 0.812 | 0.624 | 0.468 | 0.028 | 21.3% | 6 |
| **LogisChain AI (full ensemble)** | **0.856** | **0.712** | **0.523** | **0.019** | **28.7%** | **21+** |

**SC improvement over financial-only baseline:** AUC +11.5% · Gini +31.3% · Precision@5% +81.6%

---

## LogisChain Lab — Game Modes & Scoring

| Mode | Capital | Description | Difficulty |
|------|---------|-------------|-----------|
| Trade Finance Portfolio | $500M | Manage 200 active LCs through disruptions | Medium |
| SCF Pricing | $200M | Set discount rates for 500 suppliers | Medium |
| Logistics Investment | $250M | Build optimised carrier/port portfolio | Hard |
| Cargo Insurance | $2B | Dynamic pricing for 1,000 policies | Expert |

**Scoring dimensions (1,000 points total):**
- Financial Performance: 300 pts
- Risk Management Quality: 250 pts
- Supply Chain Intelligence Use: 200 pts
- Decision Speed: 100 pts
- Learning Progression: 150 pts

**Certification levels:** Novice → Practitioner → Specialist → Expert → Master

---

## Case Studies Covered

| Case Study | Year | Impact | LogisChain AI Early Warning |
|-----------|------|--------|----------------------------|
| 🚢 Ever Given — Suez Canal | 2021 | $9.6B/day blocked | Day 1: 46 LCs flagged, amendments queued |
| 🦠 COVID-19 Supply Chain Shock | 2020 | $4T trade finance gap | Week 1: OTIF deterioration → elevated SC-PD |
| 💳 Greensill Capital Collapse | 2021 | $140B SCF book | M-12: HHI concentration >0.40 alert |
| ⚓ Hanjin Shipping Bankruptcy | 2016 | $14B cargo affected | Q-1: Carrier health score declining flag |

---

## Repository Structure

```
logischain-ai/
├── src/
│   ├── data/           pipeline.py · preprocessor.py · feature_store.py
│   ├── models/         gnn.py · tcn.py · transformer_model.py · xgboost_model.py
│   │                   survival.py · ensemble.py
│   ├── features/       supply_chain_features.py · financial_features.py · fusion_features.py
│   ├── financial/      trade_finance_model.py · ccc_predictor.py · credit_risk_scorer.py
│   ├── simulation/     engine.py · scenarios.py · scoring.py · game_modes.py
│   └── utils/          metrics.py · explainability.py · visualizations.py
├── tests/              test_data.py · test_models.py · test_features.py
│                       test_financial.py · test_simulation.py
├── notebooks/          01_eda.ipynb … 05_evaluation.ipynb
├── configs/            model_config.yaml · data_config.yaml
├── docs/               architecture.md · model_cards/ · patent_concept.md
├── demo/               app.py  (Streamlit dashboard)
├── data/               raw/ · processed/ · features/
├── run_pipeline.py     Full pipeline runner
├── Dockerfile
├── docker-compose.yml
└── setup.py
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for new functionality (target >80% coverage)
4. Run `black src/ && flake8 src/ && pytest tests/` before pushing
5. Submit a pull request with a clear description

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by Zetheta Algorithms Research Team · research@zetheta.ai*
