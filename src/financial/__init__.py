"""LogisChain AI — financial subpackage.

v0.2.0 classes (production)
────────────────────────────
LCRiskScorer            15-feature LC risk scoring, backtest, fraud detection, pricing
CCCPredictor            CCC computation, SC-signal prediction, EWS, WCVI, SCF optimization
CreditRiskScorer        SC-adjusted PD, TRFSI, dynamic insurance, SR 11-7 model card

v0.1.0 classes (backward-compatible)
─────────────────────────────────────
TradeFinanceRiskModel   Instrument pricing (LC, SCF, Forfeiting)
TradeFinanceInstrument  Dataclass for trade finance instruments
SupplyChainCreditScorer Credit risk scorer with SHAP decomposition
CreditScoreResult       Dataclass for credit score output
"""

from src.financial.trade_finance_model import (
    LCRiskScorer,
    TradeFinanceRiskModel,
    TradeFinanceInstrument,
)
from src.financial.ccc_predictor import CCCPredictor
from src.financial.credit_risk_scorer import (
    CreditRiskScorer,
    SupplyChainCreditScorer,
    CreditScoreResult,
)

__all__ = [
    # v0.2.0 production classes
    "LCRiskScorer",
    "CCCPredictor",
    "CreditRiskScorer",
    # v0.1.0 backward compat
    "TradeFinanceRiskModel",
    "TradeFinanceInstrument",
    "SupplyChainCreditScorer",
    "CreditScoreResult",
]
