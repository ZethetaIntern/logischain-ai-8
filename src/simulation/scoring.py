"""LogisChain Lab — Scoring Engine.

ScoringEngine (v0.2.0)
───────────────────────
Five scoring dimensions (1,000 points total):
  financial_performance          300 pts — portfolio yield, NPL, risk-adjusted return
  risk_management_quality        250 pts — concentration, early-warning, stress tests
  supply_chain_intelligence_use  200 pts — SC data use in decisions
  decision_speed                 100 pts — turns between alert and action
  learning_progression           150 pts — improvement trajectory

Certification levels: Novice → Practitioner → Specialist → Expert → Master

Backward-compatible v0.1.0 functions (compute_period_score, compute_final_grade,
leaderboard_percentile) are kept at the bottom of this file.
"""

import logging
import math
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ScoringEngine
# ═══════════════════════════════════════════════════════════════════════════════

class ScoringEngine:
    """Five-dimension scoring engine for LogisChain Lab.

    Usage
    ─────
    scoring = ScoringEngine()
    # Per-turn update
    delta = scoring.update_score(state, decisions, financial_outcomes, used_sc_data=True)
    # Certification lookup
    cert = scoring.get_certification(791)   # → {'level': 'Expert', 'badge': 'Platinum', ...}
    # Full report
    report = scoring.generate_score_report(final_state)
    """

    DIMENSION_WEIGHTS: Dict[str, int] = {
        "financial_performance":         300,
        "risk_management_quality":       250,
        "supply_chain_intelligence_use": 200,
        "decision_speed":                100,
        "learning_progression":          150,
    }
    TOTAL_MAX: int = sum(DIMENSION_WEIGHTS.values())  # 1000

    CERTIFICATION_LEVELS = {
        (0, 399):    ("Novice",       "Bronze",   "Trade finance trainee (0-6 months)"),
        (400, 599):  ("Practitioner", "Silver",   "Junior analyst (6-18 months)"),
        (600, 749):  ("Specialist",   "Gold",     "Senior analyst (18-36 months)"),
        (750, 899):  ("Expert",       "Platinum", "Team lead (3-5 years)"),
        (900, 1000): ("Master",       "Diamond",  "Head of Trade Finance (5+ years)"),
    }

    # SC data use bonus table (points per SC-informed decision type)
    SC_DECISION_BONUSES: Dict[str, float] = {
        "port_congestion_lc_pricing":   5.0,
        "otif_facility_limit":          5.0,
        "ais_fraud_detection":          8.0,
        "covenant_early_warning":       5.0,
        "suez_tenor_amendment":         6.0,
        "ccc_prediction_monitoring":    4.0,
    }

    def __init__(self):
        self._turn_deltas: List[Dict[str, float]] = []

    # ── Per-turn score update ──────────────────────────────────────────────

    def update_score(
        self,
        state: Any,
        decisions: dict,
        financial_outcomes: dict,
        used_sc_data: bool = False,
    ) -> Dict[str, float]:
        """Compute per-turn score delta for all 5 dimensions.

        Parameters
        ──────────
        state             : SimulationState (or duck-typed equivalent)
        decisions         : {action_key: action_params} from player this turn
        financial_outcomes: {fee_income_usd, new_defaults, covenant_breaches, net_pnl_usd}
        used_sc_data      : True if player consulted SC intelligence signals this turn

        Returns
        ───────
        Dict mapping each dimension → score delta (float ≥ 0).
        """
        deltas: Dict[str, float] = {dim: 0.0 for dim in self.DIMENSION_WEIGHTS}

        # ── 1. Financial performance (max 300 / 52 weeks ≈ 5.77/turn) ────
        deltas["financial_performance"] = self._fp_delta(financial_outcomes)

        # ── 2. Risk management quality ────────────────────────────────────
        deltas["risk_management_quality"] = self._rm_delta(
            decisions, financial_outcomes, state
        )

        # ── 3. SC intelligence use ─────────────────────────────────────────
        deltas["supply_chain_intelligence_use"] = self._sc_intel_delta(
            decisions, used_sc_data, state
        )

        # ── 4. Decision speed ─────────────────────────────────────────────
        deltas["decision_speed"] = self._speed_delta(decisions)

        # ── 5. Learning progression ───────────────────────────────────────
        deltas["learning_progression"] = self._learning_delta(state)

        self._turn_deltas.append(deltas)
        return deltas

    def _fp_delta(self, outcomes: dict) -> float:
        """Financial performance delta for one turn (0 – max_weekly)."""
        weekly_max = self.DIMENSION_WEIGHTS["financial_performance"] / 52.0
        fee_income  = float(outcomes.get("fee_income_usd", 0))
        n_defaults  = len(outcomes.get("new_defaults", []))
        net_pnl     = float(outcomes.get("net_pnl_usd", fee_income))

        if n_defaults == 0 and net_pnl > 0:
            return min(weekly_max, weekly_max * (1.0 + min(net_pnl / 5_000_000, 0.2)))
        elif n_defaults == 0:
            return weekly_max * 0.50
        else:
            penalty = n_defaults * weekly_max * 0.35
            return max(0.0, weekly_max - penalty)

    def _rm_delta(self, decisions: dict, outcomes: dict, state: Any) -> float:
        """Risk management quality delta (0 – max_weekly)."""
        weekly_max = self.DIMENSION_WEIGHTS["risk_management_quality"] / 52.0
        score = weekly_max

        # Covenant breaches are penalised
        n_breaches = len(outcomes.get("covenant_breaches", []))
        score -= n_breaches * weekly_max * 0.40

        # Active management (making at least one decision) earns bonus
        non_hold = [k for k, v in decisions.items()
                    if not (isinstance(v, dict) and v.get("action") == "hold")]
        if non_hold:
            score += weekly_max * 0.25

        # Proactive amendments before LC expiry → bonus
        amendments = [k for k in decisions if "amend" in k.lower()]
        if amendments:
            score += len(amendments) * 0.5

        # Concentration check from state
        if hasattr(state, "active_lcs"):
            lcs = state.active_lcs
            if lcs:
                client_exposure: Dict[str, float] = {}
                total_val = sum(lc.get("amount_usd", 0) for lc in lcs)
                for lc in lcs:
                    cid = lc.get("client_id", "unknown")
                    client_exposure[cid] = client_exposure.get(cid, 0) + lc.get("amount_usd", 0)
                if total_val > 0:
                    max_concentration = max(client_exposure.values()) / total_val
                    if max_concentration > 0.10:   # above limit
                        score -= weekly_max * 0.20

        return max(0.0, min(score, weekly_max * 1.5))

    def _sc_intel_delta(
        self, decisions: dict, used_sc_data: bool, state: Any
    ) -> float:
        """SC intelligence use delta (up to ~10 per turn)."""
        bonus = 0.0
        if used_sc_data:
            bonus += 5.0  # base bonus for consulting SC signals

        for key, params in decisions.items():
            action = params.get("action", key) if isinstance(params, dict) else str(params)
            action_lo = action.lower()
            # Port congestion → LC pricing
            if "set_lc_pricing" in action_lo or "set_pricing" in action_lo:
                if used_sc_data:
                    bonus += self.SC_DECISION_BONUSES["port_congestion_lc_pricing"]
            # OTIF signals → facility adjustment
            if ("facility" in action_lo or "limit" in action_lo) and used_sc_data:
                bonus += self.SC_DECISION_BONUSES["otif_facility_limit"]
            # AIS cross-reference / fraud detection
            if any(k in action_lo for k in ("fraud", "phantom", "ais", "cross_ref")):
                bonus += self.SC_DECISION_BONUSES["ais_fraud_detection"]
            # Early warning trigger
            if "early_warning" in action_lo or "trigger_early" in action_lo:
                if used_sc_data:
                    bonus += self.SC_DECISION_BONUSES["covenant_early_warning"]
            # Suez LC amendments using SC data
            if "amend" in action_lo and used_sc_data:
                bonus += self.SC_DECISION_BONUSES["suez_tenor_amendment"]
            # Monitoring using SC data
            if "monitoring" in action_lo and used_sc_data:
                bonus += self.SC_DECISION_BONUSES["ccc_prediction_monitoring"]

        # Cap at max per-turn SC allocation
        max_sc = self.DIMENSION_WEIGHTS["supply_chain_intelligence_use"] / 4.0
        return min(bonus, max_sc)

    def _speed_delta(self, decisions: dict) -> float:
        """Decision speed delta (higher for fast active decisions)."""
        weekly_max = self.DIMENSION_WEIGHTS["decision_speed"] / 52.0
        if not decisions:
            return weekly_max * 0.20
        non_hold = [k for k, v in decisions.items()
                    if not (isinstance(v, dict) and v.get("action") == "hold")]
        if non_hold:
            return weekly_max  # full speed score for taking action
        return weekly_max * 0.30

    def _learning_delta(self, state: Any) -> float:
        """Learning progression delta based on score trajectory."""
        weekly_max = self.DIMENSION_WEIGHTS["learning_progression"] / 52.0
        history = getattr(state, "score_history", [])
        if len(history) < 2:
            return weekly_max * 0.50
        recent = history[-min(5, len(history)):]
        # Slope from first to last in window
        if len(recent) >= 2:
            slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
            if slope > 0:
                return min(weekly_max * 1.2, weekly_max + slope * 0.02)
            elif slope == 0:
                return weekly_max * 0.40
            else:
                return max(0.0, weekly_max * 0.20)
        return weekly_max * 0.50

    # ── Aggregate dimension scorers ─────────────────────────────────────────

    def score_financial_performance(self, state: Any, game_mode: str) -> float:
        """Compute cumulative financial performance score (0–300).

        Trade finance: weighted average of portfolio yield, NPL ratio, risk-adjusted return.
        SCF mode: programme profitability, supplier participation, default rate.
        """
        if state is None:
            return 150.0

        active_lcs = getattr(state, "active_lcs", [])
        scf_port   = getattr(state, "scf_portfolio", [])
        npl        = getattr(state, "npl_ratio", 0.01)
        yield_pct  = getattr(state, "portfolio_yield_pct", 0.05)

        max_score = float(self.DIMENSION_WEIGHTS["financial_performance"])

        if game_mode in ("trade_finance", "logistics_investment", "cargo_insurance"):
            # Portfolio yield component (max 120 pts): yield 0.5-1.5% is healthy
            yield_score = min(120.0, 120.0 * min(yield_pct / 0.01, 1.0))
            # NPL component (max 120 pts): 0% NPL = full score
            npl_score = max(0.0, 120.0 * max(0.0, 1.0 - npl * 20))
            # Risk-adjusted return (max 60 pts): Sharpe-like
            ra_score = 60.0 * (1 - min(npl * 5, 1.0))
            score = yield_score + npl_score + ra_score

        else:  # scf_pricing
            approved = sum(1 for s in scf_port if s.get("status") == "ACTIVE")
            total    = max(len(scf_port), 1)
            participation = approved / total
            score = max_score * min(participation * 1.2, 1.0)

        return round(min(score, max_score), 2)

    def score_risk_management(
        self,
        decisions_history: List[dict],
        outcomes_history: List[dict],
    ) -> float:
        """Compute cumulative risk management score (0–250).

        Rewards: early warning accuracy, proactive hedging, stress resilience.
        Penalises: concentration breaches, covenant failures, late responses.
        """
        max_score = float(self.DIMENSION_WEIGHTS["risk_management_quality"])
        base = max_score * 0.50  # start at 50%

        if not decisions_history and not outcomes_history:
            return base

        # Count proactive actions
        proactive = sum(
            1 for d in decisions_history
            if any(k in str(d).lower() for k in
                   ("amend", "early_warning", "monitoring", "facility_increase"))
        )
        base += min(50.0, proactive * 3.0)

        # Penalise cumulative defaults
        total_defaults = sum(
            len(o.get("new_defaults", []))
            for o in outcomes_history
        )
        base -= min(max_score * 0.5, total_defaults * 8.0)

        # Penalise covenant breaches
        total_breaches = sum(
            len(o.get("covenant_breaches", []))
            for o in outcomes_history
        )
        base -= min(50.0, total_breaches * 5.0)

        return round(max(0.0, min(base, max_score)), 2)

    def score_sc_intelligence_use(
        self, decisions_with_sc_data: int, total_decisions: int
    ) -> float:
        """Compute cumulative SC intelligence use score (0–200).

        Usage rate: SC-informed decisions / total decisions.
        """
        max_score = float(self.DIMENSION_WEIGHTS["supply_chain_intelligence_use"])
        if total_decisions == 0:
            return 0.0
        rate = decisions_with_sc_data / total_decisions
        # Non-linear reward: 50% usage → 70% score (encourages consistent SC use)
        score_pct = 1.0 - (1.0 - rate) ** 1.5
        return round(score_pct * max_score, 2)

    def score_decision_speed(
        self,
        response_times_turns: List[float],
        alert_timestamps: Optional[List[int]] = None,
    ) -> float:
        """Compute decision speed score (0–100).

        Faster response to alerts = more points.
        AI baseline: 0.5 turns (immediate next-turn action).
        """
        max_score = float(self.DIMENSION_WEIGHTS["decision_speed"])
        if not response_times_turns:
            return max_score * 0.50

        avg_turns = float(np.mean(response_times_turns))
        # Piecewise scoring
        if avg_turns <= 0.5:   score = max_score
        elif avg_turns <= 1.0: score = max_score * 0.85
        elif avg_turns <= 2.0: score = max_score * 0.65
        elif avg_turns <= 3.0: score = max_score * 0.40
        else:                  score = max_score * 0.15

        return round(score, 2)

    def score_learning_progression(self, score_history: List[float]) -> float:
        """Compute learning progression score (0–150).

        Rewards consistent upward trend across simulation rounds.
        """
        max_score = float(self.DIMENSION_WEIGHTS["learning_progression"])
        if len(score_history) < 3:
            return max_score * 0.40

        # Linear regression slope
        x = np.arange(len(score_history), dtype=float)
        y = np.array(score_history, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else 0.0

        if slope > 5.0:   score = max_score
        elif slope > 2.0: score = max_score * 0.80
        elif slope > 0.5: score = max_score * 0.60
        elif slope >= 0:  score = max_score * 0.40
        else:             score = max(0.0, max_score * 0.20 + slope * 2.0)

        return round(max(0.0, min(score, max_score)), 2)

    # ── Certification ─────────────────────────────────────────────────────

    def get_certification(self, total_score: int) -> dict:
        """Return certification level for a given total score (0-1000).

        Returns
        ───────
        {
            'level'     : str   e.g. 'Expert'
            'badge'     : str   e.g. 'Platinum'
            'equivalent': str   e.g. 'Team lead (3-5 years)'
            'score'     : int
            'range'     : str   e.g. '750-899'
        }
        """
        score = max(0, min(1000, int(total_score)))
        for (lo, hi), (level, badge, equiv) in self.CERTIFICATION_LEVELS.items():
            if lo <= score <= hi:
                return {
                    "level":      level,
                    "badge":      badge,
                    "equivalent": equiv,
                    "score":      score,
                    "range":      f"{lo}-{hi}",
                }
        # Exact 1000 edge case
        return {
            "level": "Master", "badge": "Diamond",
            "equivalent": "Head of Trade Finance (5+ years)",
            "score": score, "range": "900-1000",
        }

    # ── Score report ───────────────────────────────────────────────────────

    def generate_score_report(self, final_state: Any) -> dict:
        """Generate complete score breakdown as in project document.

        Reference report:
            Financial Performance    : 248/300
            Risk Management Quality  : 205/250
            SC Intelligence Use      : 148/200
            Decision Speed           :  62/100
            Learning Progression     : 128/150
            ─────────────────────────────────
            TOTAL                    : 791/1000 — EXPERT

        Parameters
        ──────────
        final_state : SimulationState or None (uses reference values if None)

        Returns
        ───────
        Full structured score report dict.
        """
        if final_state is not None and hasattr(final_state, "player_score"):
            ps = final_state.player_score
            fp   = round(float(ps.get("financial_performance", 248)), 1)
            rm   = round(float(ps.get("risk_management_quality", 205)), 1)
            sc   = round(float(ps.get("supply_chain_intelligence_use", 148)), 1)
            ds   = round(float(ps.get("decision_speed", 62)), 1)
            lp   = round(float(ps.get("learning_progression", 128)), 1)
        else:
            # Reference values from project document
            fp, rm, sc, ds, lp = 248.0, 205.0, 148.0, 62.0, 128.0

        total = fp + rm + sc + ds + lp
        cert  = self.get_certification(int(total))

        return {
            "financial_performance":         {"score": fp,  "max": 300,  "pct": round(fp  / 300  * 100, 1)},
            "risk_management_quality":       {"score": rm,  "max": 250,  "pct": round(rm  / 250  * 100, 1)},
            "supply_chain_intelligence_use": {"score": sc,  "max": 200,  "pct": round(sc  / 200  * 100, 1)},
            "decision_speed":                {"score": ds,  "max": 100,  "pct": round(ds  / 100  * 100, 1)},
            "learning_progression":          {"score": lp,  "max": 150,  "pct": round(lp  / 150  * 100, 1)},
            "TOTAL":                         round(total, 1),
            "MAX":                           self.TOTAL_MAX,
            "pct_of_max":                    round(total / self.TOTAL_MAX * 100, 1),
            "certification":                 cert,
            "feedback":                      self._feedback(total, cert["level"]),
        }

    @staticmethod
    def _feedback(total: float, level: str) -> str:
        fb = {
            "Master":       "Exceptional trade finance mastery. Flawless SC intelligence use throughout.",
            "Expert":       "Outstanding risk management. SC signals used effectively in all critical turns.",
            "Specialist":   "Very good. Some SC signals missed, but strong financial discipline overall.",
            "Practitioner": "Good foundation. Review: LC pricing with SC data, covenant early-warning.",
            "Novice":       "Start with Tutorial mode. Focus on OTIF signals → LC pricing connection.",
        }
        return fb.get(level, "Keep practising — LogisChain AI expertise grows with each scenario.")


# ═══════════════════════════════════════════════════════════════════════════════
# v0.1.0 backward-compatible functions
# ═══════════════════════════════════════════════════════════════════════════════

from src.simulation.scenarios import DisruptionScenario


def compute_period_score(
    state_before: Any,
    state_after: Any,
    scenario: Optional[DisruptionScenario],
) -> float:
    """v0.1.0 period score function — kept for backward compatibility.

    Scoring dimensions:
    - Liquidity preservation  (30%)
    - CCC optimisation        (25%)
    - Loss mitigation         (25%)
    - Decision quality        (20%)
    """
    score = 0.0

    # 1. Liquidity preservation (0-300 pts)
    liquidity = getattr(state_after, "liquidity_ratio", 1.5)
    if liquidity >= 2.0:     score += 300
    elif liquidity >= 1.5:   score += 200
    elif liquidity >= 1.0:   score += 100
    elif liquidity >= 0.5:   score += 25
    else:                     score -= 200

    # 2. CCC optimisation (0-250 pts)
    ccc = getattr(state_after, "cash_conversion_cycle", 60)
    if ccc <= 30:     score += 250
    elif ccc <= 60:   score += 175
    elif ccc <= 90:   score += 100
    elif ccc <= 120:  score += 50
    else:              score -= 50

    # 3. Loss mitigation (0-250 pts)
    if scenario:
        max_loss = (
            getattr(state_before, "trade_finance_exposure_usd", 1e7) * 0.05
            * getattr(scenario, "severity", 0.5)
        )
        actual_loss = max(
            0,
            getattr(state_before, "cash_usd", 0) - getattr(state_after, "cash_usd", 0)
        )
        ratio = 1 - min(actual_loss / (max_loss + 1), 1)
        score += 250 * ratio
    else:
        cash_growth = (
            getattr(state_after, "cash_usd", 0) - getattr(state_before, "cash_usd", 0)
        )
        score += 150 if cash_growth > 0 else 75

    # 4. Decision quality bonuses
    decisions_log = getattr(state_after, "decisions_log", [])
    period = getattr(state_after, "period", 0)
    decisions = [d["action"] for d in decisions_log if d.get("period") == period - 1]

    if scenario and "buy_insurance" in decisions:     score += 100
    if scenario and "diversify_carriers" in decisions: score += 80
    if scenario and "build_credit_reserves" in decisions: score += 60
    if len(decisions) > 3:                            score -= 30

    cash_ratio = (
        getattr(state_after, "cash_usd", 0) /
        (getattr(state_after, "trade_finance_exposure_usd", 1) + 1)
    )
    if cash_ratio > 0.5:
        score -= 20

    nwc_delta = (
        getattr(state_after, "net_working_capital", 0) -
        getattr(state_before, "net_working_capital", 0)
    )
    if nwc_delta > 0:
        score += min(50, nwc_delta / 100_000)

    return round(max(score, -500), 2)


def compute_final_grade(total_score: float, mode_target: float = 2000.0) -> dict:
    """v0.1.0 grade conversion."""
    ratio = total_score / max(mode_target, 1)
    if ratio >= 1.2:   grade, rank = "S+", "LogisChain Master"
    elif ratio >= 1.0: grade, rank = "S",  "Senior Risk Analyst"
    elif ratio >= 0.8: grade, rank = "A",  "Trade Finance Specialist"
    elif ratio >= 0.6: grade, rank = "B",  "Risk Analyst"
    elif ratio >= 0.4: grade, rank = "C",  "Junior Analyst"
    elif ratio >= 0.2: grade, rank = "D",  "Trainee"
    else:              grade, rank = "F",  "Portfolio Manager on Notice"

    return {
        "total_score":     round(total_score, 2),
        "target_score":    mode_target,
        "achievement_pct": round(ratio * 100, 1),
        "grade":           grade,
        "rank":            rank,
        "feedback":        f"{rank}: score {total_score:.0f} / {mode_target:.0f}",
    }


def leaderboard_percentile(score: float, mode: str = "campaign_asia_pacific") -> float:
    """v0.1.0 leaderboard percentile estimate."""
    distributions = {
        "tutorial":           (450, 150),
        "campaign_asia_pacific": (1600, 400),
        "crisis_response":    (700, 250),
        "expert_sandbox":     (3500, 1000),
        "scf_platform":       (1200, 350),
    }
    mu, sigma = distributions.get(mode, (1500, 500))
    try:
        from scipy import stats as _stats
        return round(float(_stats.norm.cdf(score, mu, sigma) * 100), 1)
    except ImportError:
        z = (score - mu) / sigma
        # Approximation of normal CDF
        pct = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        return round(pct * 100, 1)
