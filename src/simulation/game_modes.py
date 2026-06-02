"""LogisChain Lab game modes: Tutorial, Campaign, Crisis, and Sandbox."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.simulation.scenarios import DisruptionScenario, SCENARIO_LIBRARY
from src.simulation.engine import SimulationEngine, PortfolioState


@dataclass
class GameConfig:
    mode: str
    name: str
    description: str
    periods: int
    initial_cash: float
    initial_exposure: float
    scenario_override: Optional[str]   # force a specific scenario each period
    difficulty: str                    # easy / medium / hard / expert
    allowed_actions: List[str]
    objectives: List[str]
    target_score: float = 1000.0
    time_limit_seconds: Optional[int] = None


GAME_MODES: Dict[str, GameConfig] = {
    "tutorial": GameConfig(
        mode="tutorial",
        name="Supply Chain Finance 101",
        description="Learn the basics of trade finance risk management under a guided "
                    "Port Strike scenario with step-by-step hints.",
        periods=4,
        initial_cash=10_000_000,
        initial_exposure=5_000_000,
        scenario_override="port_strike",
        difficulty="easy",
        allowed_actions=["buy_insurance", "hold", "build_credit_reserves"],
        objectives=[
            "Keep liquidity ratio above 1.5x",
            "Limit CCC to under 90 days",
            "Score 300+ points",
        ],
    ),
    "campaign_asia_pacific": GameConfig(
        mode="campaign",
        name="APAC Trade Finance Manager",
        description="Manage an Asia-Pacific trade portfolio over 8 quarters. Face typhoons, "
                    "geopolitical tensions, and semiconductor shortages. Maximize returns.",
        periods=8,
        initial_cash=20_000_000,
        initial_exposure=50_000_000,
        scenario_override=None,
        difficulty="medium",
        allowed_actions=["buy_insurance", "diversify_carriers", "build_credit_reserves",
                         "reduce_lc_exposure", "early_payment_scf", "hold"],
        objectives=[
            "Maintain portfolio expected loss below 2% of EAD",
            "Keep cash conversion cycle below 75 days",
            "Achieve net working capital growth of 15%",
            "Score 2000+ points",
        ],
    ),
    "crisis_response": GameConfig(
        mode="crisis",
        name="Black Swan Crisis Manager",
        description="A pandemic lockdown hits your global supply chain. You have 6 quarters "
                    "to stabilize the portfolio using all available tools. Every decision counts.",
        periods=6,
        initial_cash=8_000_000,
        initial_exposure=40_000_000,
        scenario_override="pandemic_lockdown",
        difficulty="hard",
        allowed_actions=["buy_insurance", "diversify_carriers", "build_credit_reserves",
                         "reduce_lc_exposure", "early_payment_scf", "hold"],
        objectives=[
            "Avoid insolvency (cash > 0)",
            "Limit total losses to under 30% of initial portfolio",
            "Rebuild liquidity ratio to 1.2x by period 6",
        ],
        target_score=800.0,
    ),
    "expert_sandbox": GameConfig(
        mode="sandbox",
        name="Expert Sandbox — Free Play",
        description="Full control. Set your own portfolio, choose scenarios manually, "
                    "and test advanced hedging strategies with no guardrails.",
        periods=12,
        initial_cash=50_000_000,
        initial_exposure=200_000_000,
        scenario_override=None,
        difficulty="expert",
        allowed_actions=["buy_insurance", "diversify_carriers", "build_credit_reserves",
                         "reduce_lc_exposure", "early_payment_scf", "hold"],
        objectives=["No fixed objectives — maximize your LogisChain Score"],
        target_score=5000.0,
    ),
    "scf_platform": GameConfig(
        mode="scf_platform",
        name="SCF Platform Operator",
        description="You run a Supply Chain Finance platform. Onboard suppliers, price "
                    "early payment, manage anchor credit risk, and survive a tariff shock.",
        periods=8,
        initial_cash=15_000_000,
        initial_exposure=80_000_000,
        scenario_override="tariff_shock",
        difficulty="hard",
        allowed_actions=["early_payment_scf", "build_credit_reserves",
                         "reduce_lc_exposure", "hold"],
        objectives=[
            "Maintain platform NIM above 1.5%",
            "Keep default rate below 3%",
            "Grow SCF book by 20%",
        ],
    ),
}


class GameSession:
    """Manages a single player game session."""

    def __init__(self, mode_key: str = "campaign_asia_pacific", seed: int = 42):
        if mode_key not in GAME_MODES:
            raise ValueError(f"Unknown mode '{mode_key}'. Available: {list(GAME_MODES.keys())}")
        self.config = GAME_MODES[mode_key]
        self.engine = SimulationEngine(seed=seed)
        self._setup()

    def _setup(self):
        initial = PortfolioState(
            cash_usd=self.config.initial_cash,
            trade_finance_exposure_usd=self.config.initial_exposure,
        )
        self.engine.reset(initial)

    def validate_action(self, action: str) -> bool:
        return action in self.config.allowed_actions

    def play_period(
        self, actions: Optional[List[Tuple[str, Optional[dict]]]] = None
    ):
        """Play one period. Returns SimulationResult."""
        from src.simulation.engine import SimulationResult
        actions = actions or [("hold", {})]
        valid_actions = [(a, p) for a, p in actions if self.validate_action(a)]
        if not valid_actions:
            valid_actions = [("hold", {})]

        # Override scenario if mode specifies one
        if self.config.scenario_override:
            scenario = SCENARIO_LIBRARY.get(self.config.scenario_override)
            result = self.engine.step(valid_actions)
            # Force scenario narrative (engine will naturally draw)
        else:
            result = self.engine.step(valid_actions)

        return result

    @property
    def is_complete(self) -> bool:
        return self.engine.state.period >= self.config.periods

    @property
    def is_bankrupt(self) -> bool:
        return self.engine.state.cash_usd < 0

    def get_leaderboard_entry(self) -> dict:
        return {
            "mode": self.config.mode,
            "difficulty": self.config.difficulty,
            "periods_played": self.engine.state.period,
            "final_score": round(self.engine.state.score, 2),
            "final_cash_usd": round(self.engine.state.cash_usd, 2),
            "final_ccc": round(self.engine.state.cash_conversion_cycle, 2),
            "final_liquidity": round(self.engine.state.liquidity_ratio, 3),
            "objective_achieved": self.engine.state.score >= self.config.target_score,
        }

    @staticmethod
    def list_modes() -> List[dict]:
        return [
            {
                "key": k,
                "name": cfg.name,
                "difficulty": cfg.difficulty,
                "periods": cfg.periods,
                "description": cfg.description[:80] + "...",
            }
            for k, cfg in GAME_MODES.items()
        ]
