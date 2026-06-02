"""LogisChain Lab — Disruption Scenarios.

ScenarioEngine manages all 10 realistic disruption scenarios:

  1.  suez_canal_blockage          Ever Given–style (Suez Canal)
  2.  carrier_bankruptcy           Hanjin-style carrier collapse
  3.  port_congestion_event        LA / Rotterdam mega-congestion surge
  4.  geopolitical_route_closure   Red Sea / Bab el-Mandeb closure
  5.  supplier_quality_failure     Mass recall / safety shutdown
  6.  demand_whiplash              Bullwhip effect — boom/bust demand
  7.  cyber_attack                 Maersk NotPetya–style ransomware
  8.  natural_disaster             Typhoon / earthquake at key port
  9.  commodity_price_shock        Oil / semiconductor / grain spike
  10. pandemic_style_disruption    COVID-19–scale global disruption

Backward-compatible v0.1.0 classes (DisruptionScenario dataclass,
SCENARIO_LIBRARY dict) are kept at the bottom of this file.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Suez-transiting and Red Sea lanes ────────────────────────────────────────
SUEZ_LANES = {"CN-EU", "CN-DE", "CN-NL", "IN-EU", "KR-EU", "JP-EU",
              "APAC-EU", "CN-MED", "SG-EU", "TW-EU"}
RED_SEA_LANES = SUEZ_LANES  # same routes affected

PANAMA_LANES = {"CN-US", "KR-US", "JP-US", "APAC-US_EAST", "LATAM-US"}
TRANS_PACIFIC  = {"CN-US", "VN-US", "KR-US", "JP-US", "TW-US"}
TRANS_ATLANTIC = {"US-EU", "US-DE", "US-NL"}

MAJOR_PORTS = ["Shanghai", "Singapore", "Rotterdam", "LA",
               "Hamburg", "Busan", "Dubai", "Antwerp",
               "Ningbo", "Guangzhou"]


# ═══════════════════════════════════════════════════════════════════════════════
# ScenarioEngine
# ═══════════════════════════════════════════════════════════════════════════════

class ScenarioEngine:
    """Probabilistic disruption scenario engine for LogisChain Lab.

    Usage
    ─────
    engine = ScenarioEngine(seed=42)
    # Check and trigger at each turn
    new_scenarios = engine.check_and_trigger_scenarios(turn=3, state=game_state)
    # Apply effects of a specific scenario
    effects = engine.apply_scenario_effects(scenario_dict, game_state)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.active_scenarios: List[dict] = []
        self.scenario_history: List[dict] = []
        self._active_keys: set = set()
        self.scenario_catalogue: Dict[str, dict] = self._build_catalogue()

    # ── Catalogue construction ─────────────────────────────────────────────

    def _build_catalogue(self) -> Dict[str, dict]:
        """Build the full 10-scenario catalogue."""
        return {
            "suez_canal_blockage":        self.suez_canal_blockage_scenario(),
            "carrier_bankruptcy":          self.carrier_bankruptcy_scenario(),
            "port_congestion_event":       self.port_congestion_event_scenario(),
            "geopolitical_route_closure":  self.geopolitical_route_closure_scenario(),
            "supplier_quality_failure":    self.supplier_quality_failure_scenario(),
            "demand_whiplash":             self.demand_whiplash_scenario(),
            "cyber_attack":                self.cyber_attack_scenario(),
            "natural_disaster":            self.natural_disaster_scenario(),
            "commodity_price_shock":       self.commodity_price_shock_scenario(),
            "pandemic_style_disruption":   self.pandemic_style_disruption_scenario(),
        }

    # ── Trigger logic ──────────────────────────────────────────────────────

    def check_and_trigger_scenarios(self, turn: int, state: Any) -> List[dict]:
        """Probability-based scenario triggering at each simulation turn.

        Scenarios already active are not re-triggered.
        Some scenarios can chain (port congestion → LC expiry → WC stress).

        Returns list of newly triggered scenario dicts.
        """
        newly_triggered: List[dict] = []

        # Expire old scenarios
        self.active_scenarios = [s for s in self.active_scenarios
                                  if s.get("end_turn", 0) > turn]
        self._active_keys = {s["key"] for s in self.active_scenarios}

        for key, scenario in self.scenario_catalogue.items():
            if key in self._active_keys:
                continue  # already active

            prob = scenario.get("trigger_probability_per_turn", 0.02)
            if self.rng.random() < prob:
                sev_lo, sev_hi = scenario.get("severity_range", (2, 5))
                severity = float(self.rng.uniform(sev_lo, sev_hi))
                dur_lo, dur_hi = scenario.get("duration_range", (2, 6))
                duration = int(self.rng.integers(dur_lo, dur_hi + 1))

                active = {
                    **scenario,
                    "current_severity": severity,
                    "duration_weeks": duration,
                    "start_turn": turn,
                    "end_turn": turn + duration,
                    "triggered_at_turn": turn,
                }
                self.active_scenarios.append(active)
                self._active_keys.add(key)
                self.scenario_history.append(active)
                newly_triggered.append(active)
                logger.info(
                    f"Scenario triggered at turn {turn}: {scenario['name']} "
                    f"(severity={severity:.1f}, duration={duration}w)"
                )

        return newly_triggered

    # ── Effect application ─────────────────────────────────────────────────

    def apply_scenario_effects(self, scenario: dict, state: Any) -> dict:
        """Apply a scenario's effects to the simulation state.

        Modifies state in-place (freight_rates, port_congestion_index,
        supplier_health_scores) and generates alert messages.

        Returns
        ───────
        {
            'state_changes'  : {field: new_value},
            'financial_impacts': {metric: value},
            'new_alerts'     : [alert_dict, ...],
            'new_disruptions': [disruption_dict, ...],
        }
        """
        effects: dict = {
            "state_changes":    {},
            "financial_impacts": {},
            "new_alerts":       [],
            "new_disruptions":  [],
        }

        key = scenario.get("key", "")
        severity = float(scenario.get("current_severity", scenario.get("severity", 3.0)))
        phys = scenario.get("physical_impacts", {})
        fin = scenario.get("financial_impacts", {})

        # ── Freight rate impact ────────────────────────────────────────────
        affected_lanes = scenario.get("geographic_scope", [])
        frt_mult_range = phys.get("freight_multiplier_range", (1.0, 1.0))
        if frt_mult_range[1] > 1.0:
            frt_mult = float(self.rng.uniform(*frt_mult_range))
            if hasattr(state, "freight_rates"):
                for lane in affected_lanes:
                    if lane in state.freight_rates:
                        state.freight_rates[lane] *= frt_mult
                        effects["state_changes"][f"freight_{lane}"] = state.freight_rates[lane]
            effects["financial_impacts"]["freight_cost_multiplier"] = round(frt_mult, 3)

        # ── Port congestion impact ─────────────────────────────────────────
        cong_delta = float(phys.get("port_congestion_delta", 0.0)) * (severity / 5.0)
        affected_ports = phys.get("affected_ports", MAJOR_PORTS[:5])
        if hasattr(state, "port_congestion_index") and cong_delta > 0:
            for port in affected_ports:
                if port in state.port_congestion_index:
                    new_cong = min(5.0, state.port_congestion_index[port] + cong_delta)
                    state.port_congestion_index[port] = new_cong
                    effects["state_changes"][f"congestion_{port}"] = new_cong

        # ── Supplier health impact ─────────────────────────────────────────
        otif_drop = float(phys.get("otif_drop", 0.0)) * (severity / 5.0)
        if hasattr(state, "supplier_health_scores") and otif_drop > 0:
            n_affected = max(1, int(len(state.supplier_health_scores) * 0.3))
            affected_sup = list(state.supplier_health_scores.keys())[:n_affected]
            for sup_id in affected_sup:
                new_health = float(np.clip(state.supplier_health_scores[sup_id] - otif_drop, 0.3, 1.0))
                state.supplier_health_scores[sup_id] = new_health

        # ── Financial metrics ──────────────────────────────────────────────
        effects["financial_impacts"].update({
            "lc_default_spread_bps":       fin.get("lc_default_spread_bps", 0),
            "ccc_impact_days":             fin.get("ccc_impact_days", 0),
            "credit_spread_widening_bps":  fin.get("credit_spread_widening_bps", 0),
            "lc_expiry_risk_pct":          fin.get("lc_expiry_risk_pct", 0.0),
        })

        # ── Generate alerts ────────────────────────────────────────────────
        alerts = self._generate_scenario_alerts(scenario, state, severity)
        effects["new_alerts"].extend(alerts)

        # ── Add to active disruptions list ─────────────────────────────────
        effects["new_disruptions"].append({
            "type":        scenario.get("type", key.upper()),
            "key":         key,
            "name":        scenario.get("name", "Unknown Scenario"),
            "severity":    severity,
            "description": scenario.get("description", ""),
        })

        return effects

    def _generate_scenario_alerts(
        self, scenario: dict, state: Any, severity: float
    ) -> List[dict]:
        alerts = []
        key = scenario.get("key", "")
        fin = scenario.get("financial_impacts", {})
        active_lcs = getattr(state, "active_lcs", [])
        n_lcs = len(active_lcs)

        if key == "suez_canal_blockage":
            lc_risk_pct = fin.get("lc_expiry_risk_pct", 0.23)
            n_at_risk = int(n_lcs * lc_risk_pct)
            alerts += [
                {
                    "id": f"SUEZ-BLOCK-CRITICAL",
                    "type": "SCENARIO_TRIGGERED",
                    "severity": "CRITICAL",
                    "message": (
                        f"🚢 SUEZ CANAL BLOCKED — {n_at_risk} LCs at expiry risk "
                        f"({lc_risk_pct:.0%} of book). Amend tenors immediately."
                    ),
                    "priority": "CRITICAL",
                    "action_hint": "amend_lc_tenor +14d for all Suez-transit LCs",
                    "score_opportunity": scenario.get("score_bonus_optimal_play", 85),
                },
                {
                    "id": "SUEZ-FREIGHT-HIGH",
                    "type": "FREIGHT_SPIKE",
                    "severity": "HIGH",
                    "message": "Freight rates Asia-Europe: +120-180%. LC pricing review required.",
                    "priority": "HIGH",
                    "action_hint": "set_lc_pricing for CN-EU, IN-EU lanes",
                },
            ]

        elif key == "carrier_bankruptcy":
            alerts.append({
                "id": "CARRIER-BANKRUPT-CRITICAL",
                "type": "CARRIER_FAILURE",
                "severity": "CRITICAL",
                "message": "Major carrier bankruptcy declared. Vessels detained at multiple ports.",
                "priority": "CRITICAL",
                "action_hint": "identify_carrier_exposure; offer_facility_increase to affected clients",
            })

        elif key == "port_congestion_event":
            alerts.append({
                "id": "PORT-CONGESTION-HIGH",
                "type": "PORT_CONGESTION",
                "severity": "HIGH",
                "message": "Major port congestion surge. Expected +10-15 day transit delays.",
                "priority": "HIGH",
                "action_hint": "amend_lc_tenor for affected lanes",
            })

        elif key == "geopolitical_route_closure":
            alerts.append({
                "id": "ROUTE-CLOSURE-CRITICAL",
                "type": "ROUTE_CLOSURE",
                "severity": "CRITICAL",
                "message": "Red Sea / Bab el-Mandeb closure. All Asia-Europe shipments rerouting.",
                "priority": "CRITICAL",
                "action_hint": "amend_lc_tenor +18d; increase_monitoring for affected LCs",
            })

        elif key == "cyber_attack":
            alerts.append({
                "id": "CYBER-ATTACK-HIGH",
                "type": "CYBER_INCIDENT",
                "severity": "HIGH",
                "message": "Ransomware attack on major freight management system. Booking/tracking disrupted.",
                "priority": "HIGH",
                "action_hint": "trigger_early_warning for documentary credits",
            })

        else:
            alerts.append({
                "id": f"{key.upper()}-ALERT",
                "type": "SCENARIO_TRIGGERED",
                "severity": "HIGH" if severity >= 3 else "MEDIUM",
                "message": f"Disruption event: {scenario.get('name', key)}",
                "priority": "HIGH" if severity >= 3 else "MEDIUM",
            })

        return alerts

    # ── 10 Scenario factory methods ────────────────────────────────────────

    def suez_canal_blockage_scenario(self, severity: int = 4, duration_days: int = 6) -> dict:
        """Ever Given–style Suez Canal blockage.

        Full worked example from LogisChain AI project document (Section B3.2):

        Turn 1 (days 1-7):   Identify all Suez-transit LCs, offer tenor amendments
        Turn 2 (days 8-14):  Assess WC impact, offer facility increases to 14 clients
        Turn 3 (days 15-21): Monitor freight cost pass-through, adjust LC pricing
        Turn 4 (days 22-28): Canal reopens → congestion wave at Rotterdam/Antwerp

        Optimal play: prevent 19 technical defaults ($42M saved), +$2.8M vs passive
        Score bonus:  +85 points for optimal Turn-1 response
        """
        return {
            "key":         "suez_canal_blockage",
            "type":        "SUEZ_CANAL_BLOCKAGE",
            "name":        "Suez Canal Blockage",
            "description": (
                "A large container vessel has run aground in the Suez Canal, blocking "
                "both northbound and southbound traffic. Approximately 23% of active LCs "
                "have Suez transit exposure. Freight rates on Asia-Europe lanes spiking "
                "+120-180%. 14 clients face +15-25 day CCC extension."
            ),
            "category":    "geopolitical",
            "trigger_probability_per_turn": 0.015,
            "severity_range":  (3, 5),
            "duration_range":  (3, 6),
            "geographic_scope": list(SUEZ_LANES),
            "affected_sectors": ["electronics", "automotive", "retail", "pharma"],
            "physical_impacts": {
                "freight_multiplier_range": (2.20, 2.80),   # +120-180%
                "port_congestion_delta":    1.50,
                "otif_drop":                0.08,
                "transit_delay_days":       18,
                "affected_ports":           ["Rotterdam", "Hamburg", "Antwerp", "Singapore"],
            },
            "financial_impacts": {
                "lc_expiry_risk_pct":         0.23,
                "lc_default_spread_bps":      200,
                "ccc_impact_days":            20,
                "credit_spread_widening_bps": 100,
                "n_technical_defaults_risk":  19,
                "default_value_at_risk_usd":  42_000_000,
                "optimal_play_value_usd":     2_800_000,
            },
            "ai_signals_generated": [
                "FREIGHT_RATE_SPIKE", "LC_EXPIRY_RISK",
                "CCC_EXTENSION_WARNING", "FACILITY_INCREASE_OPPORTUNITY",
                "COVENANT_BREACH_RISK",
            ],
            "score_bonus_optimal_play": 85,
            "optimal_actions_turn_1": [
                "amend_lc_tenor for all Suez-transit LCs (+14 days)",
                "set_lc_pricing increase for new CN-EU LCs (+25bps)",
            ],
            "optimal_actions_turn_2": [
                "offer_facility_increase to 14 clients facing CCC extension",
                "trigger_early_warning for covenant breach candidates",
            ],
        }

    def carrier_bankruptcy_scenario(self) -> dict:
        """Hanjin-style major carrier bankruptcy.

        ~540,000 containers stranded, $14B cargo affected.
        """
        return {
            "key":         "carrier_bankruptcy",
            "type":        "CARRIER_BANKRUPTCY",
            "name":        "Major Carrier Bankruptcy",
            "description": (
                "A top-10 global container carrier has filed for bankruptcy protection. "
                "56 vessels are being turned away from ports globally. $14B in cargo affected. "
                "LC clients with this carrier face immediate documentary non-compliance."
            ),
            "category":    "financial",
            "trigger_probability_per_turn": 0.005,
            "severity_range":  (4, 5),
            "duration_range":  (6, 12),
            "geographic_scope": list(TRANS_PACIFIC | SUEZ_LANES),
            "affected_sectors": ["all"],
            "physical_impacts": {
                "freight_multiplier_range": (1.30, 1.50),
                "port_congestion_delta":    1.0,
                "otif_drop":                0.15,
                "transit_delay_days":       25,
                "affected_ports":           MAJOR_PORTS[:8],
            },
            "financial_impacts": {
                "lc_default_spread_bps":      350,
                "ccc_impact_days":            15,
                "credit_spread_widening_bps": 180,
                "lc_expiry_risk_pct":         0.15,
                "cargo_at_risk_usd":          14_000_000_000,
            },
            "ai_signals_generated": [
                "CARRIER_FAILURE_ALERT", "LC_DOCUMENTARY_RISK",
                "CARGO_INSURANCE_CLAIM_SPIKE",
            ],
            "score_bonus_optimal_play": 70,
        }

    def port_congestion_event_scenario(self) -> dict:
        """LA / Rotterdam mega-congestion surge (anchored vessels > 100)."""
        return {
            "key":         "port_congestion_event",
            "type":        "PORT_CONGESTION_SURGE",
            "name":        "Major Port Congestion Event",
            "description": (
                "Severe congestion at LA and Rotterdam. Over 100 vessels anchored, "
                "average dwell time +15 days. LC expiries under threat for Trans-Pacific "
                "and Asia-Europe shipments."
            ),
            "category":    "logistics",
            "trigger_probability_per_turn": 0.030,
            "severity_range":  (2, 4),
            "duration_range":  (2, 8),
            "geographic_scope": list(TRANS_PACIFIC | TRANS_ATLANTIC),
            "affected_sectors": ["retail", "electronics", "consumer_goods"],
            "physical_impacts": {
                "freight_multiplier_range": (1.20, 1.60),
                "port_congestion_delta":    2.0,
                "otif_drop":                0.06,
                "transit_delay_days":       10,
                "affected_ports":           ["LA", "Rotterdam", "Shanghai", "Ningbo"],
            },
            "financial_impacts": {
                "lc_default_spread_bps":      120,
                "ccc_impact_days":            12,
                "credit_spread_widening_bps": 60,
                "lc_expiry_risk_pct":         0.12,
            },
            "ai_signals_generated": [
                "PORT_CONGESTION_SPIKE", "TRANSIT_DELAY_WARNING", "LC_TENOR_RISK",
            ],
            "score_bonus_optimal_play": 50,
        }

    def geopolitical_route_closure_scenario(self) -> dict:
        """Red Sea / Strait of Hormuz closure (Houthi-style attacks)."""
        return {
            "key":         "geopolitical_route_closure",
            "type":        "RED_SEA_CLOSURE",
            "name":        "Red Sea / Geopolitical Route Closure",
            "description": (
                "Escalating geopolitical tensions have forced all major carriers to "
                "suspend Red Sea transits. Vessels rerouting via Cape of Good Hope adds "
                "+10-14 days and +$500-700/TEU in bunker costs."
            ),
            "category":    "geopolitical",
            "trigger_probability_per_turn": 0.010,
            "severity_range":  (3, 5),
            "duration_range":  (4, 26),
            "geographic_scope": list(RED_SEA_LANES),
            "affected_sectors": ["oil", "electronics", "automotive", "retail"],
            "physical_impacts": {
                "freight_multiplier_range": (1.50, 2.20),
                "port_congestion_delta":    0.80,
                "otif_drop":                0.10,
                "transit_delay_days":       12,
                "affected_ports":           ["Dubai", "Jeddah", "Singapore", "Rotterdam"],
            },
            "financial_impacts": {
                "lc_default_spread_bps":      180,
                "ccc_impact_days":            14,
                "credit_spread_widening_bps": 90,
                "lc_expiry_risk_pct":         0.20,
            },
            "ai_signals_generated": [
                "ROUTE_DIVERSION_ALERT", "FREIGHT_SPIKE", "TENOR_EXTENSION_NEEDED",
            ],
            "score_bonus_optimal_play": 75,
        }

    def supplier_quality_failure_scenario(self) -> dict:
        """Mass supplier quality recall / factory safety shutdown."""
        return {
            "key":         "supplier_quality_failure",
            "type":        "SUPPLIER_QUALITY_FAILURE",
            "name":        "Major Supplier Quality Failure",
            "description": (
                "A critical quality failure has forced factory shutdowns at major "
                "electronics suppliers in Vietnam and China. Recall affecting "
                "$8B in downstream goods. Inventory depletion risk for dependent buyers."
            ),
            "category":    "operational",
            "trigger_probability_per_turn": 0.020,
            "severity_range":  (2, 4),
            "duration_range":  (2, 6),
            "geographic_scope": ["VN-US", "CN-US", "VN-EU", "CN-EU"],
            "affected_sectors": ["electronics", "automotive", "consumer_goods"],
            "physical_impacts": {
                "freight_multiplier_range": (1.0, 1.0),   # no freight impact
                "port_congestion_delta":    0.0,
                "otif_drop":                0.20,
                "transit_delay_days":       0,
                "affected_ports":           [],
            },
            "financial_impacts": {
                "lc_default_spread_bps":      80,
                "ccc_impact_days":            25,
                "credit_spread_widening_bps": 40,
                "lc_expiry_risk_pct":         0.08,
                "inventory_write_down_risk":  0.15,
            },
            "ai_signals_generated": [
                "SUPPLIER_OTIF_ALERT", "INVENTORY_DEPLETION_RISK", "CCC_EXTENSION_WARNING",
            ],
            "score_bonus_optimal_play": 45,
        }

    def demand_whiplash_scenario(self) -> dict:
        """Post-COVID bullwhip — demand boom then collapse."""
        return {
            "key":         "demand_whiplash",
            "type":        "DEMAND_WHIPLASH",
            "name":        "Demand Whiplash / Bullwhip Effect",
            "description": (
                "Sudden demand reversal: retailers over-ordered during shortage, "
                "now cancelling. DIO spiking as inventory builds. DSO extending as "
                "buyers request payment delays. CCC pressure across consumer sectors."
            ),
            "category":    "demand",
            "trigger_probability_per_turn": 0.015,
            "severity_range":  (3, 4),
            "duration_range":  (8, 26),
            "geographic_scope": ["CN-US", "CN-EU", "VN-US"],
            "affected_sectors": ["retail", "consumer_goods", "electronics"],
            "physical_impacts": {
                "freight_multiplier_range": (0.65, 0.80),  # freight drops on demand collapse
                "port_congestion_delta":    -0.5,
                "otif_drop":                0.05,
                "transit_delay_days":       0,
                "affected_ports":           [],
            },
            "financial_impacts": {
                "lc_default_spread_bps":      150,
                "ccc_impact_days":            30,
                "credit_spread_widening_bps": 80,
                "lc_expiry_risk_pct":         0.10,
                "inventory_build_dio_days":   25,
            },
            "ai_signals_generated": [
                "DEMAND_COLLAPSE_SIGNAL", "INVENTORY_BUILD_WARNING",
                "CCC_EXTENSION_WARNING", "COVENANT_BREACH_RISK",
            ],
            "score_bonus_optimal_play": 55,
        }

    def cyber_attack_scenario(self) -> dict:
        """NotPetya / Maersk-style ransomware attack on shipping systems."""
        return {
            "key":         "cyber_attack",
            "type":        "CYBER_ATTACK",
            "name":        "Major Cyber Attack on Freight Systems",
            "description": (
                "Ransomware attack has taken down booking, tracking, and customs "
                "clearance systems at a major freight operator. 45,000 containers "
                "temporarily untrackable. AIS data unreliable for affected vessels."
            ),
            "category":    "cyber",
            "trigger_probability_per_turn": 0.010,
            "severity_range":  (3, 5),
            "duration_range":  (1, 3),
            "geographic_scope": ["GLOBAL"],
            "affected_sectors": ["all"],
            "physical_impacts": {
                "freight_multiplier_range": (1.10, 1.25),
                "port_congestion_delta":    0.50,
                "otif_drop":                0.12,
                "transit_delay_days":       7,
                "affected_ports":           MAJOR_PORTS[:6],
            },
            "financial_impacts": {
                "lc_default_spread_bps":      100,
                "ccc_impact_days":            8,
                "credit_spread_widening_bps": 50,
                "lc_expiry_risk_pct":         0.10,
                "phantom_shipment_risk":      True,
            },
            "ai_signals_generated": [
                "AIS_DATA_UNRELIABLE", "PHANTOM_SHIPMENT_RISK",
                "DOCUMENTARY_FRAUD_ALERT",
            ],
            "score_bonus_optimal_play": 60,
        }

    def natural_disaster_scenario(self) -> dict:
        """Super typhoon / earthquake affecting key manufacturing / port hubs."""
        return {
            "key":         "natural_disaster",
            "type":        "NATURAL_DISASTER",
            "name":        "Natural Disaster — Key Port Region",
            "description": (
                "Category 5 typhoon has made landfall near major manufacturing / port "
                "hub. Factory shutdowns and port closure expected for 1-3 weeks. "
                "Insurance claims expected across cargo and trade credit books."
            ),
            "category":    "weather",
            "trigger_probability_per_turn": 0.008,
            "severity_range":  (3, 5),
            "duration_range":  (1, 4),
            "geographic_scope": ["APAC-US", "CN-US", "TW-US", "JP-US"],
            "affected_sectors": ["electronics", "automotive", "semiconductors"],
            "physical_impacts": {
                "freight_multiplier_range": (1.30, 1.70),
                "port_congestion_delta":    1.50,
                "otif_drop":                0.18,
                "transit_delay_days":       14,
                "affected_ports":           ["Shanghai", "Ningbo", "Busan", "Yokohama"],
            },
            "financial_impacts": {
                "lc_default_spread_bps":      160,
                "ccc_impact_days":            18,
                "credit_spread_widening_bps": 75,
                "lc_expiry_risk_pct":         0.15,
                "insurance_claims_trigger":   True,
            },
            "ai_signals_generated": [
                "PORT_CLOSURE_ALERT", "SUPPLY_DISRUPTION", "INSURANCE_CLAIM_OPPORTUNITY",
            ],
            "score_bonus_optimal_play": 65,
        }

    def commodity_price_shock_scenario(self) -> dict:
        """Oil / semiconductor / grain commodity price spike."""
        return {
            "key":         "commodity_price_shock",
            "type":        "COMMODITY_PRICE_SHOCK",
            "name":        "Commodity Price Shock",
            "description": (
                "Oil prices spiked 45% in two weeks following geopolitical tensions. "
                "Bunker costs pushing freight rates +30-50% across all lanes. "
                "Margin compression across energy-intensive manufacturing sectors."
            ),
            "category":    "financial",
            "trigger_probability_per_turn": 0.020,
            "severity_range":  (3, 5),
            "duration_range":  (4, 26),
            "geographic_scope": list(SUEZ_LANES | TRANS_PACIFIC | TRANS_ATLANTIC),
            "affected_sectors": ["chemicals", "plastics", "food", "manufacturing"],
            "physical_impacts": {
                "freight_multiplier_range": (1.30, 1.60),
                "port_congestion_delta":    0.0,
                "otif_drop":                0.03,
                "transit_delay_days":       0,
                "affected_ports":           [],
            },
            "financial_impacts": {
                "lc_default_spread_bps":      120,
                "ccc_impact_days":            10,
                "credit_spread_widening_bps": 70,
                "lc_expiry_risk_pct":         0.08,
                "margin_compression_pct":     0.12,
            },
            "ai_signals_generated": [
                "COMMODITY_PRICE_SPIKE", "MARGIN_COMPRESSION_WARNING",
                "COVENANT_BREACH_RISK",
            ],
            "score_bonus_optimal_play": 40,
        }

    def pandemic_style_disruption_scenario(self) -> dict:
        """COVID-19–scale global supply chain and financial disruption."""
        return {
            "key":         "pandemic_style_disruption",
            "type":        "PANDEMIC_DISRUPTION",
            "name":        "Pandemic-Scale Global Disruption",
            "description": (
                "A new highly transmissible pathogen has triggered simultaneous "
                "supply and demand shocks globally. Factory closures in Asia reducing "
                "supply 30-50%. Consumer demand collapsing in Western markets. "
                "Trade finance gap widening by $700B."
            ),
            "category":    "pandemic",
            "trigger_probability_per_turn": 0.002,
            "severity_range":  (4, 5),
            "duration_range":  (26, 104),
            "geographic_scope": ["GLOBAL"],
            "affected_sectors": ["all"],
            "physical_impacts": {
                "freight_multiplier_range": (2.50, 3.50),
                "port_congestion_delta":    2.0,
                "otif_drop":                0.25,
                "transit_delay_days":       30,
                "affected_ports":           MAJOR_PORTS,
            },
            "financial_impacts": {
                "lc_default_spread_bps":      500,
                "ccc_impact_days":            40,
                "credit_spread_widening_bps": 300,
                "lc_expiry_risk_pct":         0.35,
                "trade_finance_gap_increase_bn": 700,
            },
            "ai_signals_generated": [
                "GLOBAL_SHUTDOWN_WARNING", "MASS_LC_EXPIRY_RISK",
                "CCC_COVENANT_BREACH_WAVE", "PORTFOLIO_STRESS_TEST",
            ],
            "score_bonus_optimal_play": 150,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v0.1.0 backward-compatible classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DisruptionScenario:
    """v0.1.0 disruption scenario dataclass — kept for backward compatibility."""
    name: str
    description: str
    category: str
    severity: float
    duration_days: int
    affected_routes: List[str]
    supply_shock: float
    demand_shock: float
    freight_cost_multiplier: float
    transit_time_multiplier: float
    carrier_failure_prob: float
    lc_default_spread_bps: float
    ccc_impact_days: float
    credit_spread_widening_bps: float
    probability: float
    tags: Dict[str, str] = field(default_factory=dict)


SCENARIO_LIBRARY: Dict[str, DisruptionScenario] = {
    "suez_closure": DisruptionScenario(
        name="Suez Canal Closure",
        description="Major shipping lane blockage forcing rerouting around Cape of Good Hope.",
        category="geopolitical", severity=0.75, duration_days=21,
        affected_routes=["APAC-EMEA", "CN-DE", "CN-NL"],
        supply_shock=-0.25, demand_shock=0.05, freight_cost_multiplier=1.65,
        transit_time_multiplier=1.45, carrier_failure_prob=0.08,
        lc_default_spread_bps=120, ccc_impact_days=14,
        credit_spread_widening_bps=85, probability=0.15,
        tags={"source": "2021-Ever Given"},
    ),
    "pandemic_lockdown": DisruptionScenario(
        name="Pandemic Port Lockdown",
        description="Manufacturing hub lockdown, port congestion, container shortage.",
        category="pandemic", severity=0.90, duration_days=90,
        affected_routes=["CN-US", "CN-EU", "CN-LATAM"],
        supply_shock=-0.45, demand_shock=-0.20, freight_cost_multiplier=2.80,
        transit_time_multiplier=1.80, carrier_failure_prob=0.15,
        lc_default_spread_bps=300, ccc_impact_days=35,
        credit_spread_widening_bps=200, probability=0.05,
        tags={"source": "COVID-19 2020"},
    ),
    "port_strike": DisruptionScenario(
        name="Major Port Strike (West Coast USA)",
        description="Labour dispute halts US West Coast ports.",
        category="geopolitical", severity=0.60, duration_days=30,
        affected_routes=["APAC-US_WEST"],
        supply_shock=-0.35, demand_shock=0.0, freight_cost_multiplier=1.40,
        transit_time_multiplier=1.60, carrier_failure_prob=0.05,
        lc_default_spread_bps=80, ccc_impact_days=20,
        credit_spread_widening_bps=50, probability=0.12,
        tags={"source": "ILWU Strike 2023"},
    ),
    "typhoon_pacific": DisruptionScenario(
        name="Super Typhoon – Pacific Routes",
        description="Category 5 typhoon disrupts APAC-AMER shipping lanes.",
        category="weather", severity=0.55, duration_days=14,
        affected_routes=["APAC-AMER", "TW-US", "JP-US"],
        supply_shock=-0.20, demand_shock=0.0, freight_cost_multiplier=1.25,
        transit_time_multiplier=1.30, carrier_failure_prob=0.04,
        lc_default_spread_bps=50, ccc_impact_days=8,
        credit_spread_widening_bps=30, probability=0.20,
        tags={"region": "APAC"},
    ),
    "tariff_shock": DisruptionScenario(
        name="Tariff Escalation – US-China",
        description="25% tariffs imposed on $300B of trade.",
        category="geopolitical", severity=0.65, duration_days=180,
        affected_routes=["CN-US"],
        supply_shock=-0.15, demand_shock=-0.10, freight_cost_multiplier=1.10,
        transit_time_multiplier=1.05, carrier_failure_prob=0.03,
        lc_default_spread_bps=100, ccc_impact_days=25,
        credit_spread_widening_bps=120, probability=0.08,
        tags={"source": "US-China Trade War"},
    ),
    "cyber_attack_logistics": DisruptionScenario(
        name="Ransomware Attack – Major Freight Platform",
        description="Cyberattack on global freight management system.",
        category="cyber", severity=0.50, duration_days=12,
        affected_routes=["GLOBAL"],
        supply_shock=-0.10, demand_shock=0.0, freight_cost_multiplier=1.15,
        transit_time_multiplier=1.20, carrier_failure_prob=0.06,
        lc_default_spread_bps=60, ccc_impact_days=6,
        credit_spread_widening_bps=40, probability=0.18,
        tags={"source": "Maersk NotPetya 2017"},
    ),
    "financial_crisis": DisruptionScenario(
        name="Global Financial Crisis – Credit Crunch",
        description="Bank failures, LC capacity reduction, trade finance drying up.",
        category="financial", severity=0.95, duration_days=365,
        affected_routes=["GLOBAL"],
        supply_shock=-0.30, demand_shock=-0.40, freight_cost_multiplier=0.80,
        transit_time_multiplier=0.90, carrier_failure_prob=0.20,
        lc_default_spread_bps=500, ccc_impact_days=45,
        credit_spread_widening_bps=450, probability=0.03,
        tags={"source": "GFC 2008"},
    ),
    "semiconductor_shortage": DisruptionScenario(
        name="Semiconductor Supply Shortage",
        description="Global chip shortage halts automotive and electronics manufacturing.",
        category="supply", severity=0.70, duration_days=270,
        affected_routes=["TW-US", "KR-DE"],
        supply_shock=-0.40, demand_shock=0.15, freight_cost_multiplier=1.30,
        transit_time_multiplier=1.15, carrier_failure_prob=0.05,
        lc_default_spread_bps=90, ccc_impact_days=30,
        credit_spread_widening_bps=70, probability=0.10,
        tags={"source": "2021 Chip Crisis"},
    ),
}


def get_scenario(name: str) -> DisruptionScenario:
    if name not in SCENARIO_LIBRARY:
        raise KeyError(f"Scenario '{name}' not found. Available: {list(SCENARIO_LIBRARY.keys())}")
    return SCENARIO_LIBRARY[name]


def list_scenarios() -> list:
    return [
        {
            "name": s.name, "key": k, "category": s.category,
            "severity": s.severity, "probability": s.probability,
            "duration_days": s.duration_days,
        }
        for k, s in SCENARIO_LIBRARY.items()
    ]
