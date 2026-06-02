"""LogisChain Lab — gamified supply chain × financial simulation.

v0.2.0 classes (Three-Layer Engine)
─────────────────────────────────────
ThreeLayerSimulationEngine  Full three-layer engine with physical/financial/AI layers
SimulationState             Complete simulation state dataclass (turn, portfolio, scores)
PhysicalSupplyChainLayer    100-node SC graph with weekly event simulation
FinancialLayer              Payment processing, covenant checks, P&L
LogisChainAIAdvisor         AI opponent + intelligence signals
ScenarioEngine              10-scenario catalogue with probabilistic triggering
ScoringEngine               5-dimension scoring + certification levels

v0.1.0 classes (backward-compatible)
──────────────────────────────────────
SimulationEngine            Original discrete-time engine
PortfolioState              v0.1.0 portfolio state
SimulationResult            v0.1.0 turn result
DisruptionScenario          v0.1.0 scenario dataclass
SCENARIO_LIBRARY            v0.1.0 scenario catalogue (8 scenarios)
compute_period_score        v0.1.0 per-period scoring
compute_final_grade         v0.1.0 grade conversion
GameSession / GameConfig / GAME_MODES  v0.1.0 game session management
"""

# ── v0.2.0 ────────────────────────────────────────────────────────────────────
from src.simulation.engine import (
    ThreeLayerSimulationEngine,
    SimulationState,
    PhysicalSupplyChainLayer,
    FinancialLayer,
    LogisChainAIAdvisor,
    # v0.1.0 compat
    SimulationEngine,
    PortfolioState,
    SimulationResult,
)
from src.simulation.scenarios import (
    ScenarioEngine,
    # v0.1.0 compat
    DisruptionScenario,
    SCENARIO_LIBRARY,
    get_scenario,
    list_scenarios,
)
from src.simulation.scoring import (
    ScoringEngine,
    # v0.1.0 compat
    compute_period_score,
    compute_final_grade,
    leaderboard_percentile,
)
from src.simulation.game_modes import (
    GameSession,
    GameConfig,
    GAME_MODES,
)

__all__ = [
    # v0.2.0 engine
    "ThreeLayerSimulationEngine",
    "SimulationState",
    "PhysicalSupplyChainLayer",
    "FinancialLayer",
    "LogisChainAIAdvisor",
    # v0.2.0 scenarios
    "ScenarioEngine",
    # v0.2.0 scoring
    "ScoringEngine",
    # v0.1.0 engine
    "SimulationEngine",
    "PortfolioState",
    "SimulationResult",
    # v0.1.0 scenarios
    "DisruptionScenario",
    "SCENARIO_LIBRARY",
    "get_scenario",
    "list_scenarios",
    # v0.1.0 scoring
    "compute_period_score",
    "compute_final_grade",
    "leaderboard_percentile",
    # v0.1.0 game modes
    "GameSession",
    "GameConfig",
    "GAME_MODES",
]
