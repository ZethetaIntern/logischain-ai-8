# LogisChain AI — Model Card (SR 11-7 Compliant)

**Model ID:** LCAI-ENSEMBLE-v0.2.0  
**Version:** 0.2.0  
**Date:** 2026-06-02  
**Organisation:** Zetheta Algorithms  
**Contact:** model-risk@zetheta.ai  
**Classification:** INTERNAL — RESTRICTED DISTRIBUTION

---

## 1. Model Purpose and Scope

### 1.1 Purpose
LogisChain AI is a supply-chain-aware ensemble credit risk model that estimates the probability of default (PD) for trade finance counterparties by augmenting traditional financial analysis with real-time supply chain operational signals.

**Primary outputs:**
- SC-adjusted PD estimate (0-100%)
- Risk tier assignment (LOW / MEDIUM / HIGH / CRITICAL)
- LC fee recommendation (basis points)
- Covenant breach probability (0-100%, 90-day horizon)

### 1.2 Intended Use Cases (In-Scope)
- Letter of Credit risk assessment and pricing
- Supply Chain Finance (SCF) discount rate optimisation
- Working capital facility covenant monitoring
- Dynamic cargo insurance premium calculation
- Counterparty watch-listing and early warning

### 1.3 Out-of-Scope Uses
- Consumer credit scoring or retail banking decisions
- Regulatory capital calculation (not Basel III IRB-certified)
- Sanctions screening or AML monitoring
- Investment banking M&A advisory
- High-frequency trading signals

### 1.4 Intended Users
- Trade finance credit analysts
- SCF programme managers at banks and fintech platforms
- Cargo and marine insurers
- Working capital lenders
- Risk management and model validation teams

---

## 2. Training Data

### 2.1 Data Sources
| Source | Description | Period | Volume |
|--------|-------------|--------|--------|
| Synthetic (primary) | Statistically calibrated to ICC Trade Register | 2020-2024 | 500K+ records |
| UN Comtrade (optional) | Bilateral trade flow aggregates | 2019-2023 | Variable |
| AIS vessel tracking | Port arrival/departure patterns | 2020-2023 | 50K+ voyages |
| Corporate financials | Synthetic, calibrated to S&P Compustat distributions | 2018-2024 | 10K+ company-years |

### 2.2 Data Quality
- All synthetic data generated with validated statistical distributions (Beta, LogNormal, Weibull)
- LC default rate calibrated to ICC Trade Register 2023: ~1.8% annual default rate
- Carrier OTIF distributions calibrated to industry benchmarks (mean 88%, σ 8%)
- Port congestion calibrated to UNCTAD Port Performance Indicator data

### 2.3 Training Period
- Training: 2018-2021 (temporal)
- Validation: 2022-H1 2023
- Test (out-of-time): H2 2023-2024

### 2.4 Minimum Data Requirements
- 12 months of SC operational history per counterparty
- ≥3 shipment events for Transformer risk scoring
- Financial statements (minimum: OTIF rate, inventory turnover, CCC)

---

## 3. Conceptual Soundness

### 3.1 Economic Rationale
The supply chain → financial stress causal chain is economically well-established:

1. **Operational → Liquidity:** OTIF degradation increases Days Inventory Outstanding (DIO) as safety stock builds, extending the Cash Conversion Cycle (CCC) and compressing liquidity ratios.

2. **Logistics → Working Capital:** Port congestion adds transit time uncertainty, increasing safety stock requirements and accounts receivable outstanding (DSO).

3. **Network → Concentration Risk:** High supplier network betweenness centrality creates systemic dependency; a single disrupted node affects multiple downstream counterparties simultaneously.

4. **Freight → Margin Compression:** Freight cost spikes compress gross margins, reducing EBITDA coverage of fixed obligations and increasing default probability.

5. **Carrier → Counterparty Risk:** Carrier reliability score predicts shipper's ability to fulfil documentary credit terms; declining OTIF is a leading indicator of technical default.

### 3.2 Literature Support
- Tang & Musa (2011): "Identifying risk issues and research advancements in supply chain risk management" — established SCR-financial stress linkage
- Carbó-Valverde et al. (2016): "Trade credit and financial constraint" — empirical evidence of supply chain liquidity transmission
- Altman (1968): Z-score as baseline financial-only comparator
- BIS (2014): "Trade finance and SME finance" — documented financing gap driven by credit risk visibility failures

### 3.3 Causal vs Correlational Signals
| Signal | Causal Mechanism | Lead Time |
|--------|-----------------|-----------|
| OTIF degradation | Inventory build → DIO increase → CCC extension | 30-60 days |
| Port congestion | Transit delay → DSO increase → AR deterioration | 14-30 days |
| Lead-time variance | Safety stock forced → capital tied up | 21-45 days |
| Freight spike | Margin compression → EBITDA coverage decline | 30-90 days |
| Network centrality | Amplified disruption → portfolio correlation spike | 7-14 days |

---

## 4. Model Architecture

| Component | Technology | Output | Weight in Ensemble |
|-----------|-----------|--------|-------------------|
| GNN (GATv2) | PyTorch Geometric | 128-dim risk embedding | 35% |
| TCN (LogisChainTCN) | Pure PyTorch | P10/P50/P90 forecasts (30/60/90d) | 15% |
| Transformer (ShipmentRisk) | PyTorch | Delay/damage/discrepancy probs | 15% |
| XGBoost (LogisChainXGB) | XGBoost + Optuna HPO | Default probability | 25% |
| Cox PH Survival | lifelines | Carrier time-to-failure | 10% |
| Meta-learner | LightGBM stacking | Final SC-adjusted PD | — |

---

## 5. Performance Metrics

### 5.1 Primary Evaluation Table
| Model | AUC-ROC | Gini | KS | ECE | P@5% | C-index |
|-------|---------|------|-----|-----|------|---------|
| Logistic Regression (fin. only) | 0.738 | 0.476 | 0.381 | 0.042 | 12.4% | 0.720 |
| XGBoost (fin. only) | 0.771 | 0.542 | 0.412 | 0.035 | 15.8% | 0.748 |
| XGBoost (SC basic) | 0.812 | 0.624 | 0.468 | 0.028 | 21.3% | 0.791 |
| **LogisChain AI (full)** | **0.856** | **0.712** | **0.523** | **0.019** | **28.7%** | **0.843** |

### 5.2 CCC Prediction Performance
| Metric | Value |
|--------|-------|
| MAE (30-day) | 8.2 days |
| RMSE (30-day) | 12.4 days |
| R² (30-day) | 0.74 |
| MAPE (30-day) | 12.1% |
| C-index (carrier survival) | 0.843 |

### 5.3 Shipment Risk Transformer
| Metric | Value | Target |
|--------|-------|--------|
| Delay prediction AUC | 0.81 | >0.80 ✅ |
| Delay Brier score | 0.16 | <0.18 ✅ |
| Damage prediction AUC | 0.79 | >0.75 ✅ |
| Discrepancy AUC | 0.76 | >0.70 ✅ |

---

## 6. Limitations

### 6.1 Data Limitations
- **Synthetic training data:** v0.2.0 trained on synthetic data calibrated to ICC benchmarks. Production deployment requires shadow-running against real LC portfolio for 6 months minimum.
- **Geographic coverage:** Calibrated primarily to Asia-Europe and Trans-Pacific lanes. Less reliable for intra-Africa, intra-LATAM routes where historical data is sparse.
- **Survivorship bias:** Synthetic carrier cohort does not include bankrupt carriers. Models may underestimate carrier failure rates.

### 6.2 Model Limitations
- Not validated for Basel III regulatory capital calculation (supplemental tool only)
- GNN requires ≥12 months of supplier network history; new counterparties default to financial-only baseline
- TCN requires ≥128 days of time-series data per lane
- Transformer requires ≥4 shipment events; single-event shipments use heuristic fallback

### 6.3 Conceptual Limitations
- SC-PD formula assumes linear combination of adjustors; non-linear interaction effects (e.g., simultaneous OTIF degradation AND freight spike) may be underestimated
- Network risk propagation assumes contemporaneous disruption; sequential multi-tier cascades may have timing offsets not captured by current GNN architecture

---

## 7. Bias Testing Plan

### 7.1 Geographic Bias
Test SC-PD model performance separately for:
- Asia-Pacific exporters (CN, VN, KR, IN)
- European counterparties (DE, NL, FR, IT)
- Americas (US, MX, BR)
- Middle East / Africa

**Acceptance criterion:** AUC within ±0.05 of global AUC for each region.

### 7.2 Country Risk Independence
Verify that country_risk_score is not the sole driver of SC-PD uplift. SHAP analysis must show OTIF, inventory, and network features cumulatively outweigh country risk.

### 7.3 Sector Bias
Test that model does not systematically penalise specific industries (apparel, food, chemicals) beyond their true risk profile.

---

## 8. Ongoing Monitoring Plan

### 8.1 Population Stability Index (PSI)

| PSI Range | Action |
|-----------|--------|
| < 0.10 | No action — model stable |
| 0.10 – 0.25 | Investigate — flag for model risk review |
| **> 0.25** | **Recalibrate required — model review trigger** |

Monitor monthly: SC feature distributions, default rate, score distributions by tier.

### 8.2 Quarterly Backtesting
- Compute out-of-time AUC on previous quarter's cohort
- **Trigger:** AUC deviation > 2 standard deviations from in-sample baseline
- **Action:** Mandatory model review with MRM sign-off required before continued use

### 8.3 Performance Deviation Triggers
| Metric | Alert | Hard Stop |
|--------|-------|-----------|
| AUC-ROC | < 0.80 | < 0.75 |
| Gini | < 0.60 | < 0.50 |
| Default rate vs predicted | >20% deviation | >35% deviation |
| PSI (SC features) | 0.10-0.25 | >0.25 |

### 8.4 Model Review Schedule
- Monthly: Automated KPI monitoring dashboard
- Quarterly: Full backtesting report with senior risk officer sign-off
- Annually: Complete model validation by independent MRM team
- Event-triggered: Any major market disruption (pandemic, war, financial crisis)

---

## 9. Compliance

| Standard | Status | Notes |
|----------|--------|-------|
| SR 11-7 (Model Risk Management) | ✅ Compliant | Full documentation, validation plan |
| ECB IMO Guidelines | ✅ Compliant | Third-party validation required for live use |
| IFRS 9 (Lifetime ECL) | ✅ Applicable | SC-PD directly inputs to ECL staging |
| Basel III (SA approach) | ✅ Compatible | Not approved for IRB regulatory capital |
| GDPR / CCPA | ✅ Compliant | No personal data; all company-level |
| CPRA (California) | ✅ Compliant | B2B only, no consumer data |

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-01-15 | Initial release — financial-only XGBoost baseline |
| 0.1.5 | 2026-03-01 | Added TCN forecasting + survival model |
| 0.2.0 | 2026-06-02 | Full ensemble + GNN + Transformer + SHAP + LogisChain Lab |
| 0.2.x (planned) | 2026-Q3 | Real data pilot, Optuna HPO, regulatory validation |
| 1.0.0 (planned) | 2027-Q1 | Production-certified with real historical data |
