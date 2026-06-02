"""LogisChain Lab — Three-Layer Simulation Engine.

Architecture
────────────
ThreeLayerSimulationEngine orchestrates:
  PhysicalSupplyChainLayer  — 100-node graph, weekly supply chain events
  FinancialLayer            — payment processing, covenant checks, P&L
  LogisChainAIAdvisor       — AI opponent + intelligence signals for player
  ScenarioEngine            — probability-based disruption triggering

Game Modes
──────────
trade_finance        $500M portfolio, 50 clients, 200 active LCs
scf_pricing          $200M SCF programme, 500 suppliers
logistics_investment $250M capital, logistics asset investment
cargo_insurance      $2B premium book, 1000 policies

Backward-compatible v0.1.0 classes (PortfolioState, SimulationEngine, SimulationResult)
are kept at the bottom of this file.
"""

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Country / port constants ───────────────────────────────────────────────────
_COUNTRIES = ["CN", "VN", "BD", "IN", "DE", "US", "MX", "KR", "JP", "TH", "TR"]
_CW = [0.25, 0.10, 0.06, 0.10, 0.10, 0.12, 0.05, 0.06, 0.06, 0.05, 0.05]

_MAJOR_PORTS = [
    "Shanghai", "Singapore", "Rotterdam", "LA", "Hamburg", "Busan",
    "Dubai", "Antwerp", "Ningbo", "Guangzhou", "Qingdao", "Tianjin",
    "Yokohama", "Colombo", "Jeddah", "Felixstowe", "Manila", "Mumbai",
    "Santos", "Vancouver",
]

_LANES = [
    "CN-US", "CN-EU", "CN-LATAM", "VN-US", "IN-EU", "KR-US",
    "JP-US", "TH-EU", "BD-EU", "DE-US", "MX-US", "TR-EU",
]

_TRADE_FINANCE_PRODUCTS = ["LC", "SCF_Invoice", "Forfeiting", "Factoring", "Bank_Guarantee"]
_HS_CODES = ["8471", "8542", "8708", "8703", "3004", "6203", "9403", "2710"]
_RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
_RATING_PD = {"AAA": 0.0001, "AA": 0.0005, "A": 0.001, "BBB": 0.003,
              "BB": 0.012, "B": 0.035, "CCC": 0.12}

_SUEZ_ROUTES = {"CN-EU", "IN-EU", "KR-EU", "JP-EU", "CN-MED", "APAC-EU",
                "SG-EU", "TW-EU", "CN-DE", "CN-NL"}
_RED_SEA_ROUTES = _SUEZ_ROUTES
_TRANS_PACIFIC  = {"CN-US", "VN-US", "KR-US", "JP-US", "TW-US"}


# ═══════════════════════════════════════════════════════════════════════════════
# SimulationState dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationState:
    """Complete snapshot of the simulation at a given turn."""
    turn: int
    year: int
    portfolio_value_usd: float
    cash_usd: float
    active_lcs: List[dict]
    scf_portfolio: List[dict]
    active_facilities: List[dict]
    cargo_policies: List[dict]
    supply_chain_network: nx.DiGraph
    port_congestion_index: Dict[str, float]
    freight_rates: Dict[str, float]
    supplier_health_scores: Dict[str, float]
    active_disruptions: List[dict]
    player_score: Dict[str, float]
    ai_score: Dict[str, float]
    game_mode: str
    alerts: List[dict]
    # Internal tracking
    decisions_history: List[dict] = field(default_factory=list)
    outcomes_history: List[dict] = field(default_factory=list)
    sc_data_usage: List[bool] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    score_history: List[float] = field(default_factory=list)

    @property
    def cumulative_player_score(self) -> float:
        return sum(self.player_score.values())

    @property
    def npl_ratio(self) -> float:
        defaulted = sum(1 for lc in self.active_lcs if lc.get("status") == "DEFAULTED")
        total = max(len(self.active_lcs), 1)
        return defaulted / total

    @property
    def portfolio_yield_pct(self) -> float:
        fees = sum(lc.get("fee_income_usd", 0) for lc in self.active_lcs)
        return fees / max(self.portfolio_value_usd, 1) * 100


# ═══════════════════════════════════════════════════════════════════════════════
# PhysicalSupplyChainLayer
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicalSupplyChainLayer:
    """Simulates one week of supply chain events on a 100-node trade network.

    Network topology
    ────────────────
    40 suppliers      (hub-and-spoke, heavy China/Vietnam weighting)
    20 manufacturers  (mesh edges to suppliers + port connections)
    20 ports          (hub nodes, major world ports)
    20 warehouses     (spoke destinations from ports)
    Total: 500+ directed edges
    """

    NODE_TYPES = {
        "supplier":     (0,   40),
        "manufacturer": (40,  60),
        "port":         (60,  80),
        "warehouse":    (80, 100),
    }

    def __init__(self, n_nodes: int = 100, n_edges: int = 500, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.network = self._generate_network(n_nodes, n_edges)
        self.port_congestion: Dict[str, float] = {
            port: float(self.rng.uniform(0.5, 3.0))
            for port in _MAJOR_PORTS
        }
        self.freight_rates: Dict[str, float] = {
            lane: float(self.rng.lognormal(7.5, 0.4))
            for lane in _LANES
        }
        self.supplier_health: Dict[str, float] = {}
        # Initialise supplier health from network
        for node, attrs in self.network.nodes(data=True):
            if attrs.get("node_type") == "supplier":
                self.supplier_health[node] = float(attrs.get("otif_rate", 0.88))

    def _generate_network(self, n_nodes: int, n_edges: int) -> nx.DiGraph:
        """Build realistic supply chain graph with four node types."""
        G = nx.DiGraph()
        rng = self.rng

        # ── Supplier nodes (0-39) ──────────────────────────────────────────
        for i in range(40):
            country = str(rng.choice(_COUNTRIES, p=[w / sum(_CW) for w in _CW]))
            G.add_node(
                f"SUP-{i:03d}",
                node_type="supplier",
                country=country,
                otif_rate=float(rng.beta(18, 2)),
                lead_time_mean=float(rng.lognormal(2.5, 0.5)),
                lead_time_std=float(rng.uniform(1, 5)),
                inventory_turnover=float(rng.lognormal(2.0, 0.4)),
                capacity_utilization=float(rng.beta(7, 3)),
                freight_cost_ratio=float(rng.beta(2, 20)),
                fill_rate=float(rng.beta(20, 2)),
                country_risk=float(rng.uniform(0.1, 0.7)),
                port_proximity=float(rng.uniform(0.3, 1.0)),
            )

        # ── Manufacturer nodes (40-59) ─────────────────────────────────────
        for i in range(20):
            G.add_node(
                f"MFG-{i:03d}",
                node_type="manufacturer",
                country=str(rng.choice(["DE", "JP", "US", "CN", "KR"])),
                production_capacity=float(rng.lognormal(14, 1)),
                utilization=float(rng.uniform(0.6, 0.95)),
            )

        # ── Port nodes (60-79) ────────────────────────────────────────────
        for i in range(20):
            port_name = _MAJOR_PORTS[i]
            G.add_node(
                f"PORT-{i:03d}",
                node_type="port",
                name=port_name,
                throughput_teu_day=float(rng.lognormal(9, 0.8)),
                congestion=float(rng.beta(3, 7)),
                suez_route=(port_name in ["Singapore", "Colombo", "Jeddah", "Dubai",
                                           "Mumbai", "Rotterdam", "Hamburg",
                                           "Antwerp", "Felixstowe"]),
            )

        # ── Warehouse/retail nodes (80-99) ────────────────────────────────
        for i in range(20):
            G.add_node(
                f"WH-{i:03d}",
                node_type="warehouse",
                region=str(rng.choice(["NA", "EU", "APAC", "LATAM", "MEA"])),
                storage_capacity=float(rng.lognormal(12, 1)),
            )

        # ── Edges ──────────────────────────────────────────────────────────
        edges_added = 0
        # Supplier → Manufacturer
        for sup_i in range(40):
            n_mfg = int(rng.integers(1, 5))
            for mfg_j in rng.choice(20, n_mfg, replace=False):
                G.add_edge(f"SUP-{sup_i:03d}", f"MFG-{mfg_j:03d}",
                           edge_type="supplies", volume_usd=float(rng.lognormal(13, 1.5)),
                           reliability=float(rng.beta(8, 2)))
                edges_added += 1

        # Manufacturer → Port (hub-and-spoke)
        for mfg_i in range(20):
            n_ports = int(rng.integers(1, 4))
            for port_j in rng.choice(20, n_ports, replace=False):
                G.add_edge(f"MFG-{mfg_i:03d}", f"PORT-{port_j:03d}",
                           edge_type="ships_via", transit_days=int(rng.integers(5, 40)),
                           modal="ocean")
                edges_added += 1

        # Port → Warehouse
        for port_i in range(20):
            n_wh = int(rng.integers(2, 6))
            for wh_j in rng.choice(20, n_wh, replace=False):
                G.add_edge(f"PORT-{port_i:03d}", f"WH-{wh_j:03d}",
                           edge_type="distributes", transit_days=int(rng.integers(1, 7)))
                edges_added += 1

        # Supplier → Port (direct export)
        while edges_added < n_edges:
            sup_i = int(rng.integers(0, 40))
            port_j = int(rng.integers(0, 20))
            if not G.has_edge(f"SUP-{sup_i:03d}", f"PORT-{port_j:03d}"):
                G.add_edge(f"SUP-{sup_i:03d}", f"PORT-{port_j:03d}",
                           edge_type="direct_export", volume_usd=float(rng.lognormal(12, 1)))
                edges_added += 1

        logger.info(f"PhysicalLayer network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def simulate_week(self, active_disruptions: List[dict]) -> List[dict]:
        """Simulate one week of supply chain events. Returns list of events."""
        events = []
        rng = self.rng

        # 1. Port congestion (OU mean-reverting process)
        for port in list(self.port_congestion.keys()):
            old = self.port_congestion[port]
            mean_cong = 2.0
            theta, sigma = 0.12, 0.25
            dt = 7 / 365
            new_cong = float(np.clip(
                old + theta * (mean_cong - old) * dt + sigma * rng.normal() * math.sqrt(dt) * 5,
                0.0, 5.0
            ))
            self.port_congestion[port] = new_cong
            if abs(new_cong - old) > 0.5:
                events.append({
                    "type":         "PORT_CONGESTION_CHANGE",
                    "port":         port,
                    "old":          round(old, 2),
                    "new":          round(new_cong, 2),
                    "financial_flag": new_cong > 3.5,
                })

        # 2. Freight rate changes (OU on log-rates)
        for lane in list(self.freight_rates.keys()):
            old_rate = self.freight_rates[lane]
            log_target = 7.5  # ~$1800/TEU
            log_rate = math.log(max(old_rate, 1))
            new_log = log_rate + 0.10 * (log_target - log_rate) + 0.08 * rng.normal()
            new_rate = float(np.clip(math.exp(new_log), 300, 25_000))
            self.freight_rates[lane] = new_rate
            pct_chg = (new_rate - old_rate) / old_rate
            if abs(pct_chg) > 0.10:
                events.append({
                    "type":   "FREIGHT_RATE_CHANGE",
                    "lane":   lane,
                    "old":    round(old_rate, 0),
                    "new":    round(new_rate, 0),
                    "pct":    round(pct_chg * 100, 1),
                })

        # 3. Supplier OTIF updates
        for sup_id, otif in list(self.supplier_health.items()):
            shock = rng.normal(0, 0.015)
            new_otif = float(np.clip(otif + shock, 0.40, 1.00))
            self.supplier_health[sup_id] = new_otif
            if new_otif < 0.80 and otif >= 0.80:
                events.append({
                    "type":       "SUPPLIER_OTIF_ALERT",
                    "supplier_id": sup_id,
                    "old_otif":   round(otif, 3),
                    "new_otif":   round(new_otif, 3),
                    "severity":   "HIGH" if new_otif < 0.70 else "MEDIUM",
                })

        # 4. Random shipment delay events
        n_ships = max(1, int(rng.poisson(8)))
        for _ in range(n_ships):
            delay_days = int(rng.exponential(4))
            if delay_days > 2:
                lane = str(rng.choice(_LANES))
                events.append({
                    "type":       "SHIPMENT_DELAY",
                    "lane":       lane,
                    "delay_days": delay_days,
                    "financial_impact_usd": delay_days * float(rng.lognormal(10, 0.5)),
                })

        # 5. Apply active disruptions
        for disruption in active_disruptions:
            disruption_events = self._apply_disruption_to_week(disruption)
            events.extend(disruption_events)

        return events

    def _apply_disruption_to_week(self, disruption: dict) -> List[dict]:
        events = []
        # Handle both 'type' (v0.1.0) and 'key' (ScenarioEngine v0.2.0) fields
        d_type = disruption.get("type", "")
        if not d_type:
            key = disruption.get("key", "GENERIC")
            d_type = key.upper().replace("-", "_")
        severity = float(disruption.get("current_severity",
                         disruption.get("severity", 3.0))) / 5.0   # normalise 0-5 → 0-1

        # ── Suez / Red Sea / Geopolitical route closure ──────────────────
        if d_type in ("SUEZ_CANAL_BLOCKAGE", "RED_SEA_CLOSURE",
                       "GEOPOLITICAL_ROUTE_CLOSURE"):
            phys = disruption.get("physical_impacts", {})
            frt_lo, frt_hi = phys.get("freight_multiplier_range", (2.0, 2.8))
            for lane in _SUEZ_ROUTES:
                if lane in self.freight_rates:
                    mult = float(self.rng.uniform(frt_lo, frt_hi))
                    self.freight_rates[lane] *= mult
                    events.append({
                        "type":   "FREIGHT_RATE_SPIKE",
                        "lane":   lane,
                        "multiplier": round(mult, 2),
                        "cause":  d_type,
                    })
            # Also spike congestion at key ports
            for port in ["Rotterdam", "Hamburg", "Antwerp", "Singapore"]:
                if port in self.port_congestion:
                    self.port_congestion[port] = min(
                        5.0, self.port_congestion[port] + 1.5 * severity
                    )
                    events.append({
                        "type": "PORT_CONGESTION_SPIKE",
                        "port": port,
                        "new_congestion": self.port_congestion[port],
                    })

        # ── Port congestion surge ─────────────────────────────────────────
        elif d_type == "PORT_CONGESTION_SURGE":
            phys = disruption.get("physical_impacts", {})
            affected = phys.get("affected_ports",
                                disruption.get("affected_ports", _MAJOR_PORTS[:5]))
            cong_delta = phys.get("port_congestion_delta", 1.5) * severity
            for port in affected:
                if port in self.port_congestion:
                    self.port_congestion[port] = min(
                        5.0, self.port_congestion[port] + cong_delta
                    )
                    events.append({
                        "type": "PORT_CONGESTION_SPIKE",
                        "port": port,
                        "new_congestion": self.port_congestion[port],
                    })

        # ── Carrier bankruptcy ────────────────────────────────────────────
        elif d_type == "CARRIER_BANKRUPTCY":
            # Spike freight rates on all trans-pacific lanes
            for lane in list(_SUEZ_ROUTES | _TRANS_PACIFIC):
                if lane in self.freight_rates:
                    mult = float(self.rng.uniform(1.25, 1.50))
                    self.freight_rates[lane] *= mult
                    events.append({
                        "type": "FREIGHT_RATE_SPIKE",
                        "lane": lane, "multiplier": round(mult, 2), "cause": d_type,
                    })
            # Supplier OTIF shock
            for sup_id in list(self.supplier_health.keys())[:20]:
                self.supplier_health[sup_id] = max(
                    0.40, self.supplier_health[sup_id] - 0.10 * severity
                )

        # ── Pandemic / global disruption ─────────────────────────────────
        elif d_type == "PANDEMIC_DISRUPTION":
            phys = disruption.get("physical_impacts", {})
            frt_lo, frt_hi = phys.get("freight_multiplier_range", (2.5, 3.5))
            for lane in self.freight_rates:
                mult = float(self.rng.uniform(frt_lo, frt_hi))
                self.freight_rates[lane] *= mult
                events.append({"type": "FREIGHT_RATE_SPIKE", "lane": lane,
                                "multiplier": round(mult, 2), "cause": d_type})
            for port in _MAJOR_PORTS:
                if port in self.port_congestion:
                    self.port_congestion[port] = min(5.0, self.port_congestion[port] + 2.0)

        # ── Natural disaster / cyber / commodity (generic physical handler) ─
        elif d_type in ("NATURAL_DISASTER", "CYBER_ATTACK", "COMMODITY_PRICE_SHOCK",
                         "DEMAND_WHIPLASH", "SUPPLIER_QUALITY_FAILURE"):
            phys = disruption.get("physical_impacts", {})
            frt_lo, frt_hi = phys.get("freight_multiplier_range", (1.0, 1.0))
            if frt_hi > 1.0:
                affected_lanes = disruption.get("geographic_scope", [])
                for lane in affected_lanes:
                    if lane in self.freight_rates:
                        mult = float(self.rng.uniform(frt_lo, frt_hi))
                        self.freight_rates[lane] *= mult
                        events.append({"type": "FREIGHT_RATE_SPIKE", "lane": lane,
                                        "multiplier": round(mult, 2), "cause": d_type})
            otif_drop = float(phys.get("otif_drop", 0.0)) * severity
            if otif_drop > 0:
                n_affected = max(1, int(len(self.supplier_health) * 0.25))
                for sup_id in list(self.supplier_health.keys())[:n_affected]:
                    self.supplier_health[sup_id] = max(
                        0.40, self.supplier_health[sup_id] - otif_drop
                    )

        return events

    def get_network_stats(self) -> dict:
        G = self.network
        return {
            "n_nodes":       G.number_of_nodes(),
            "n_edges":       G.number_of_edges(),
            "n_suppliers":   sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "supplier"),
            "n_ports":       sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "port"),
            "avg_congestion": round(np.mean(list(self.port_congestion.values())), 3),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FinancialLayer
# ═══════════════════════════════════════════════════════════════════════════════

class FinancialLayer:
    """Processes financial events each turn: payments, covenants, P&L."""

    def __init__(self):
        self.rng = np.random.default_rng(99)

    def process_turn(self, state: "SimulationState") -> dict:
        """Execute all financial processing for one turn. Returns P&L summary."""
        fee_income = 0.0
        defaults = []
        covenant_breaches = []
        payments_received = 0.0

        # Process LC income + defaults
        for lc in state.active_lcs:
            if lc.get("status") == "ACTIVE":
                weekly_fee = lc.get("amount_usd", 0) * lc.get("fee_pct", 0.005) / 52
                fee_income += weekly_fee
                lc["fee_income_usd"] = lc.get("fee_income_usd", 0) + weekly_fee

                # Check default probability
                pd_adj = lc.get("pd_adjusted", 0.018)
                weekly_pd = 1 - (1 - pd_adj) ** (1 / 52)
                if self.rng.random() < weekly_pd:
                    lc["status"] = "DEFAULTED"
                    defaults.append({"lc_id": lc["lc_id"], "amount_usd": lc["amount_usd"]})

        # Check facility covenants
        for fac in state.active_facilities:
            ccc = fac.get("current_ccc", 70)
            threshold = fac.get("ccc_covenant", 95)
            if ccc > threshold:
                covenant_breaches.append({
                    "facility_id": fac["facility_id"],
                    "company_id":  fac["company_id"],
                    "ccc":         ccc,
                    "threshold":   threshold,
                    "breach_pct":  round((ccc / threshold - 1) * 100, 1),
                })
                fac["covenant_status"] = "BREACH"
            else:
                fac["covenant_status"] = "COMPLIANT"

        # SCF portfolio payments
        for scf in state.scf_portfolio:
            if self.rng.random() < 0.65:  # 65% early payment rate
                discount = scf.get("invoice_amount_usd", 0) * scf.get("discount_rate_bps", 120) / 10_000 / 12
                payments_received += discount

        state.cash_usd += fee_income + payments_received - sum(d["amount_usd"] * 0.45 for d in defaults)

        return {
            "fee_income_usd":       round(fee_income, 2),
            "payments_received_usd": round(payments_received, 2),
            "new_defaults":         defaults,
            "default_loss_usd":     round(sum(d["amount_usd"] * 0.45 for d in defaults), 2),
            "covenant_breaches":    covenant_breaches,
            "net_pnl_usd":          round(fee_income + payments_received - sum(d["amount_usd"] * 0.45 for d in defaults), 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LogisChainAIAdvisor
# ═══════════════════════════════════════════════════════════════════════════════

class LogisChainAIAdvisor:
    """AI opponent + intelligence signal generator.

    The AI uses the full LogisChain AI signal set optimally.
    Players can see the same signals — better decisions = higher score.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def get_optimal_decisions(self, state: "SimulationState") -> dict:
        """Return AI's optimal decisions for this turn."""
        decisions = {}
        mode = state.game_mode

        if mode == "trade_finance":
            # Approve all low-risk pending LCs, reject high-risk
            for lc in state.active_lcs:
                if lc.get("status") == "PENDING":
                    risk = lc.get("risk_score", 0.3)
                    if risk < 0.45:
                        decisions[f"approve_lc_{lc['lc_id']}"] = {
                            "action": "approve_lc",
                            "lc_id":  lc["lc_id"],
                            "fee_pct": max(0.005, 0.003 + risk * 0.008),
                        }
                    elif risk > 0.75:
                        decisions[f"reject_lc_{lc['lc_id']}"] = {
                            "action": "reject_lc",
                            "lc_id":  lc["lc_id"],
                            "reason": "Risk score exceeds appetite",
                        }

            # Amend Suez LCs if blockage active
            suez_active = any(d.get("type") == "SUEZ_CANAL_BLOCKAGE"
                               for d in state.active_disruptions)
            if suez_active:
                for lc in state.active_lcs:
                    if lc.get("route", "") in _SUEZ_ROUTES and lc.get("status") == "ACTIVE":
                        decisions[f"amend_lc_{lc['lc_id']}"] = {
                            "action":       "amend_lc_tenor",
                            "lc_id":        lc["lc_id"],
                            "extension_days": 14,
                        }

        elif mode == "scf_pricing":
            # Set optimal discount rates based on anchor credit + OTIF
            for scf in state.scf_portfolio:
                if scf.get("status") == "PENDING":
                    otif = scf.get("otif_score", 0.88)
                    base_bps = 120
                    sc_adj = int((0.90 - max(otif, 0.70)) * 500)
                    decisions[f"set_rate_{scf['supplier_id']}"] = {
                        "action":       "set_discount_rate",
                        "supplier_id":  scf["supplier_id"],
                        "rate_bps":     base_bps + sc_adj,
                    }

        return decisions

    def get_signals_for_player(self, state: "SimulationState") -> dict:
        """Generate LogisChain AI intelligence signals for the player."""
        signals: dict = {
            "shipment_risks":          [],
            "supplier_warnings":       [],
            "port_congestion_forecasts": {},
            "ccc_predictions":         [],
            "covenant_breach_alerts":  [],
            "ai_recommendations":      [],
        }

        # Shipment risks from active LCs
        for lc in state.active_lcs[:20]:  # limit to top 20
            risk = lc.get("risk_score", 0.3)
            if risk > 0.5:
                signals["shipment_risks"].append({
                    "lc_id":        lc["lc_id"],
                    "risk_score":   risk,
                    "route":        lc.get("route", "UNKNOWN"),
                    "risk_factors": lc.get("risk_factors", []),
                })

        # Supplier warnings from health scores
        for sup_id, health in state.supplier_health_scores.items():
            if health < 0.80:
                signals["supplier_warnings"].append({
                    "supplier_id": sup_id,
                    "otif_score":  health,
                    "severity":    "HIGH" if health < 0.70 else "MEDIUM",
                    "recommended_action": "Increase monitoring",
                })

        # Port congestion forecasts
        for port, cong in state.port_congestion_index.items():
            signals["port_congestion_forecasts"][port] = {
                "current":    round(cong, 2),
                "7d_forecast": round(float(np.clip(cong + self.rng.normal(0, 0.3), 0, 5)), 2),
                "14d_forecast": round(float(np.clip(cong + self.rng.normal(0, 0.5), 0, 5)), 2),
            }

        # CCC predictions for facilities
        for fac in state.active_facilities[:10]:
            ccc = fac.get("current_ccc", 70)
            predicted = ccc + self.rng.uniform(-5, 20)
            breach_prob = float(1 / (1 + math.exp(-0.15 * (predicted - fac.get("ccc_covenant", 95)))))
            signals["ccc_predictions"].append({
                "company_id":   fac["company_id"],
                "current_ccc":  round(ccc, 1),
                "predicted_ccc": round(predicted, 1),
                "breach_prob":  round(breach_prob, 3),
            })
            if breach_prob > 0.60:
                signals["covenant_breach_alerts"].append({
                    "company_id":  fac["company_id"],
                    "breach_prob": round(breach_prob, 3),
                    "days_to_breach": max(1, int(45 * (1 - breach_prob))),
                    "recommended_action": "Offer facility amendment",
                })

        # AI recommendations
        suez = any(d.get("type") == "SUEZ_CANAL_BLOCKAGE" for d in state.active_disruptions)
        if suez:
            suez_count = sum(1 for lc in state.active_lcs
                             if lc.get("route", "") in _SUEZ_ROUTES)
            signals["ai_recommendations"].append({
                "action":         "amend_suez_lcs",
                "rationale":      f"{suez_count} active LCs transit Suez. Canal blockage active.",
                "expected_value": f"Prevent {max(1, suez_count // 5)} technical defaults",
                "urgency":        "CRITICAL",
                "confidence":     0.94,
            })

        for w in signals["supplier_warnings"][:3]:
            signals["ai_recommendations"].append({
                "action":     f"increase_monitoring_{w['supplier_id']}",
                "rationale":  f"OTIF {w['otif_score']:.0%} — below 80% threshold",
                "urgency":    w["severity"],
                "confidence": 0.87,
            })

        return signals


# ═══════════════════════════════════════════════════════════════════════════════
# ThreeLayerSimulationEngine
# ═══════════════════════════════════════════════════════════════════════════════

class ThreeLayerSimulationEngine:
    """Orchestrates PhysicalLayer + FinancialLayer + AIAdvisor for LogisChain Lab.

    Game Modes
    ──────────
    trade_finance         $500M portfolio, 50 clients, 200 active LCs
    scf_pricing           $200M SCF programme, 500 suppliers
    logistics_investment  $250M investment capital, logistics assets
    cargo_insurance       $2B premium book, 1000 cargo policies
    """

    MODES = {"trade_finance", "scf_pricing", "logistics_investment", "cargo_insurance"}

    def __init__(
        self,
        game_mode: str,
        starting_capital_usd: float,
        ai_opponent: bool = True,
        random_seed: int = 42,
    ):
        if game_mode not in self.MODES:
            raise ValueError(f"game_mode must be one of {self.MODES}")
        self.game_mode = game_mode
        self.ai_opponent = ai_opponent
        self.seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.physical_layer = PhysicalSupplyChainLayer(n_nodes=100, n_edges=500, seed=random_seed)
        self.financial_layer = FinancialLayer()
        self.intelligence_layer = LogisChainAIAdvisor(seed=random_seed)

        # Import here to avoid circular at module load
        from src.simulation.scenarios import ScenarioEngine
        from src.simulation.scoring import ScoringEngine
        self.scenario_engine = ScenarioEngine(seed=random_seed)
        self.scoring_engine = ScoringEngine()

        self.state = self._initialize_state(game_mode, starting_capital_usd)

    # ── State initialization ───────────────────────────────────────────────

    def _initialize_state(self, mode: str, capital: float) -> SimulationState:
        rng = self.rng

        if mode == "trade_finance":
            active_lcs = self._generate_lcs(200, rng)
            active_facilities = self._generate_facilities(50, rng)
            scf_portfolio, cargo_policies = [], []

        elif mode == "scf_pricing":
            scf_portfolio = self._generate_scf_suppliers(500, rng)
            active_lcs, active_facilities, cargo_policies = [], [], []

        elif mode == "logistics_investment":
            active_lcs, scf_portfolio = [], []
            active_facilities = self._generate_facilities(20, rng)
            cargo_policies = []

        else:  # cargo_insurance
            cargo_policies = self._generate_cargo_policies(1000, rng)
            active_lcs, scf_portfolio, active_facilities = [], [], []

        return SimulationState(
            turn=1,
            year=2024,
            portfolio_value_usd=capital,
            cash_usd=capital * 0.10,
            active_lcs=active_lcs,
            scf_portfolio=scf_portfolio,
            active_facilities=active_facilities,
            cargo_policies=cargo_policies,
            supply_chain_network=self.physical_layer.network,
            port_congestion_index=dict(self.physical_layer.port_congestion),
            freight_rates=dict(self.physical_layer.freight_rates),
            supplier_health_scores=dict(self.physical_layer.supplier_health),
            active_disruptions=[],
            player_score={d: 0.0 for d in ["financial_performance",
                          "risk_management_quality", "supply_chain_intelligence_use",
                          "decision_speed", "learning_progression"]},
            ai_score={d: 0.0 for d in ["financial_performance",
                      "risk_management_quality", "supply_chain_intelligence_use",
                      "decision_speed", "learning_progression"]},
            game_mode=mode,
            alerts=[{
                "id":       "WELCOME",
                "type":     "INFO",
                "message":  f"LogisChain Lab started — {mode.upper()} mode. Capital: ${capital/1e6:.0f}M",
                "turn":     1,
                "priority": "LOW",
            }],
        )

    def _generate_lcs(self, n: int, rng: np.random.Generator) -> List[dict]:
        lcs = []
        for i in range(n):
            amount = float(rng.lognormal(13.5, 1.2))
            route = str(rng.choice(_LANES))
            rating = str(rng.choice(_RATINGS, p=[0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05]))
            pd_adj = float(_RATING_PD[rating] * rng.uniform(0.8, 1.8))
            lcs.append({
                "lc_id":        f"LC-{i:05d}",
                "client_id":    f"CLIENT-{i % 50:03d}",
                "amount_usd":   round(amount, 0),
                "tenor_days":   int(rng.choice([30, 60, 90, 120, 180])),
                "status":       "ACTIVE" if rng.random() < 0.85 else "PENDING",
                "route":        route,
                "suez_transit": route in _SUEZ_ROUTES,
                "commodity_hs": str(rng.choice(_HS_CODES)),
                "risk_score":   round(float(rng.beta(3, 7)), 3),
                "pd_adjusted":  round(pd_adj, 5),
                "fee_pct":      round(float(rng.uniform(0.004, 0.012)), 4),
                "fee_income_usd": 0.0,
                "rating":       rating,
                "risk_factors": [],
            })
        return lcs

    def _generate_facilities(self, n: int, rng: np.random.Generator) -> List[dict]:
        facs = []
        for i in range(n):
            current_ccc = float(rng.uniform(45, 90))
            facs.append({
                "facility_id":    f"FAC-{i:04d}",
                "company_id":     f"CLIENT-{i:03d}",
                "limit_usd":      float(rng.lognormal(14, 1)),
                "drawn_usd":      float(rng.lognormal(13, 1)),
                "current_ccc":    round(current_ccc, 1),
                "ccc_covenant":   95.0,
                "leverage_ratio": float(rng.lognormal(0.4, 0.5)),
                "covenant_status": "COMPLIANT",
            })
        return facs

    def _generate_scf_suppliers(self, n: int, rng: np.random.Generator) -> List[dict]:
        suppliers = []
        for i in range(n):
            suppliers.append({
                "supplier_id":    f"SUP-{i:04d}",
                "anchor_id":      f"ANC-{i % 20:03d}",
                "invoice_amount_usd": float(rng.lognormal(11.5, 1.2)),
                "discount_rate_bps": int(rng.integers(80, 300)),
                "otif_score":     float(rng.beta(18, 2)),
                "status":         "ACTIVE" if rng.random() < 0.80 else "PENDING",
                "risk_tier":      str(rng.choice(["LOW", "MEDIUM", "HIGH"],
                                                  p=[0.50, 0.35, 0.15])),
            })
        return suppliers

    def _generate_cargo_policies(self, n: int, rng: np.random.Generator) -> List[dict]:
        policies = []
        for i in range(n):
            cargo_val = float(rng.lognormal(13, 1.5))
            policies.append({
                "policy_id":         f"POL-{i:05d}",
                "cargo_value_usd":   cargo_val,
                "base_rate_pct":     float(rng.uniform(0.4, 0.9)),
                "adjusted_rate_pct": float(rng.uniform(0.5, 1.5)),
                "premium_usd":       cargo_val * float(rng.uniform(0.005, 0.015)),
                "carrier_id":        f"CAR-{i % 100:04d}",
                "route":             str(rng.choice(_LANES)),
                "status":            "ACTIVE",
                "claims":            0,
            })
        return policies

    # ── Turn advance ────────────────────────────────────────────────────────

    def advance_turn(self, player_decisions: dict) -> dict:
        """Advance simulation by one week (one turn).

        Parameters
        ──────────
        player_decisions : dict mapping action names to parameters
            Example: {'approve_lc_LC-00001': {'action': 'approve_lc', 'lc_id': 'LC-00001'}}

        Returns
        ───────
        {physical_events, financial_outcomes, new_alerts, score_update,
         new_scenarios, turn_summary, decision_results}
        """
        # 1. Execute player decisions
        used_sc_data = player_decisions.pop("used_sc_data", False)
        decision_results = self._execute_decisions(player_decisions, self.state)
        self.state.decisions_history.append(player_decisions)
        self.state.sc_data_usage.append(bool(used_sc_data))

        # 2. Execute AI decisions
        if self.ai_opponent:
            ai_decisions = self.intelligence_layer.get_optimal_decisions(self.state)
            self._execute_decisions(ai_decisions, self.state, player="ai")

        # 3. Advance physical layer
        physical_events = self.physical_layer.simulate_week(self.state.active_disruptions)
        self.state.port_congestion_index = dict(self.physical_layer.port_congestion)
        self.state.freight_rates = dict(self.physical_layer.freight_rates)
        self.state.supplier_health_scores = dict(self.physical_layer.supplier_health)

        # 4. Advance financial layer
        financial_outcomes = self.financial_layer.process_turn(self.state)
        self.state.outcomes_history.append(financial_outcomes)

        # 5. Check and trigger scenarios
        new_scenarios = self.scenario_engine.check_and_trigger_scenarios(
            self.state.turn, self.state
        )
        for sc in new_scenarios:
            effects = self.scenario_engine.apply_scenario_effects(sc, self.state)
            self._apply_effects(effects)

        # 6. Update scores
        player_score_delta = self.scoring_engine.update_score(
            self.state, player_decisions, financial_outcomes, used_sc_data
        )
        for dim, delta in player_score_delta.items():
            self.state.player_score[dim] = self.state.player_score.get(dim, 0) + delta

        if self.ai_opponent:
            ai_delta = self.scoring_engine.update_score(
                self.state, ai_decisions, financial_outcomes, True
            )
            for dim, delta in ai_delta.items():
                self.state.ai_score[dim] = self.state.ai_score.get(dim, 0) + delta

        total = sum(self.state.player_score.values())
        self.state.score_history.append(total)

        # 7. Generate alerts
        new_alerts = self._generate_alerts(physical_events, financial_outcomes, new_scenarios)
        self.state.alerts = new_alerts

        # 8. Advance turn counter
        self.state.turn += 1
        if self.state.turn > 52:
            self.state.turn = 1
            self.state.year += 1

        return {
            "physical_events":   physical_events[:20],  # cap for readability
            "financial_outcomes": financial_outcomes,
            "new_alerts":        new_alerts,
            "score_update":      player_score_delta,
            "new_scenarios":     [s.get("name", "UNKNOWN") for s in new_scenarios],
            "turn_summary":      self.get_game_state_summary(),
            "decision_results":  decision_results,
        }

    def _execute_decisions(
        self, decisions: dict, state: "SimulationState", player: str = "player"
    ) -> dict:
        results = {}
        for key, params in decisions.items():
            action = params.get("action", key.split("_")[0])
            try:
                if action == "approve_lc":
                    results[key] = self._approve_lc(params, state)
                elif action == "reject_lc":
                    results[key] = self._reject_lc(params, state)
                elif action == "amend_lc_tenor":
                    results[key] = self._amend_lc(params, state)
                elif action == "set_lc_pricing":
                    results[key] = self._set_pricing(params, state)
                elif action == "increase_monitoring":
                    results[key] = {"status": "OK", "action": "monitoring_increased"}
                elif action == "offer_facility_increase":
                    results[key] = self._offer_facility_increase(params, state)
                elif action == "set_discount_rate":
                    results[key] = self._set_scf_rate(params, state)
                elif action == "approve_supplier":
                    results[key] = self._approve_supplier(params, state)
                elif action == "reject_supplier":
                    results[key] = self._reject_supplier(params, state)
                else:
                    results[key] = {"status": "UNKNOWN_ACTION"}
            except Exception as e:
                results[key] = {"status": "ERROR", "message": str(e)}
        return results

    def _approve_lc(self, params: dict, state: "SimulationState") -> dict:
        lc_id = params.get("lc_id")
        for lc in state.active_lcs:
            if lc["lc_id"] == lc_id:
                lc["status"] = "ACTIVE"
                if "fee_pct" in params:
                    lc["fee_pct"] = params["fee_pct"]
                return {"status": "APPROVED", "lc_id": lc_id, "fee_pct": lc["fee_pct"]}
        return {"status": "NOT_FOUND", "lc_id": lc_id}

    def _reject_lc(self, params: dict, state: "SimulationState") -> dict:
        lc_id = params.get("lc_id")
        for lc in state.active_lcs:
            if lc["lc_id"] == lc_id:
                lc["status"] = "REJECTED"
                return {"status": "REJECTED", "lc_id": lc_id}
        return {"status": "NOT_FOUND"}

    def _amend_lc(self, params: dict, state: "SimulationState") -> dict:
        lc_id = params.get("lc_id")
        ext = int(params.get("extension_days", 14))
        for lc in state.active_lcs:
            if lc["lc_id"] == lc_id:
                lc["tenor_days"] = lc.get("tenor_days", 90) + ext
                lc["amended"] = True
                return {"status": "AMENDED", "lc_id": lc_id, "new_tenor": lc["tenor_days"]}
        return {"status": "NOT_FOUND"}

    def _set_pricing(self, params: dict, state: "SimulationState") -> dict:
        lc_id = params.get("lc_id")
        for lc in state.active_lcs:
            if lc["lc_id"] == lc_id:
                lc["fee_pct"] = float(params.get("fee_pct", lc["fee_pct"]))
                return {"status": "PRICED", "lc_id": lc_id, "fee_pct": lc["fee_pct"]}
        return {"status": "NOT_FOUND"}

    def _offer_facility_increase(self, params: dict, state: "SimulationState") -> dict:
        fac_id = params.get("facility_id", params.get("company_id"))
        increase = float(params.get("increase_usd", 500_000))
        for fac in state.active_facilities:
            if fac["facility_id"] == fac_id or fac["company_id"] == fac_id:
                fac["limit_usd"] += increase
                return {"status": "INCREASED", "new_limit": fac["limit_usd"]}
        return {"status": "NOT_FOUND"}

    def _set_scf_rate(self, params: dict, state: "SimulationState") -> dict:
        sup_id = params.get("supplier_id")
        rate = int(params.get("rate_bps", 120))
        for scf in state.scf_portfolio:
            if scf["supplier_id"] == sup_id:
                scf["discount_rate_bps"] = rate
                return {"status": "RATE_SET", "supplier_id": sup_id, "rate_bps": rate}
        return {"status": "NOT_FOUND"}

    def _approve_supplier(self, params: dict, state: "SimulationState") -> dict:
        sup_id = params.get("supplier_id")
        for scf in state.scf_portfolio:
            if scf["supplier_id"] == sup_id:
                scf["status"] = "ACTIVE"
                return {"status": "APPROVED", "supplier_id": sup_id}
        return {"status": "NOT_FOUND"}

    def _reject_supplier(self, params: dict, state: "SimulationState") -> dict:
        sup_id = params.get("supplier_id")
        for scf in state.scf_portfolio:
            if scf["supplier_id"] == sup_id:
                scf["status"] = "REJECTED"
                return {"status": "REJECTED", "supplier_id": sup_id}
        return {"status": "NOT_FOUND"}

    def _apply_effects(self, effects: dict):
        sc = effects.get("state_changes", {})
        for key, val in sc.items():
            if hasattr(self.state, key):
                setattr(self.state, key, val)
        for disruption in effects.get("new_disruptions", []):
            self.state.active_disruptions.append(disruption)

    def _generate_alerts(
        self, physical_events: list, financial_outcomes: dict, new_scenarios: list
    ) -> List[dict]:
        alerts = []
        for event in physical_events:
            if event.get("type") == "SUPPLIER_OTIF_ALERT":
                alerts.append({
                    "id":       f"ALERT-OTIF-{event['supplier_id']}",
                    "type":     "SUPPLIER_WARNING",
                    "severity": event["severity"],
                    "message":  f"OTIF dropped to {event['new_otif']:.0%} for {event['supplier_id']}",
                    "turn":     self.state.turn,
                    "priority": "HIGH" if event["severity"] == "HIGH" else "MEDIUM",
                })
            if event.get("type") == "FREIGHT_RATE_SPIKE":
                alerts.append({
                    "id":       f"ALERT-FREIGHT-{event['lane']}",
                    "type":     "FREIGHT_ALERT",
                    "message":  f"Freight rate spike on {event['lane']}: +{event.get('multiplier',1):.0%}x",
                    "turn":     self.state.turn,
                    "priority": "HIGH",
                })
        for sc in new_scenarios:
            alerts.append({
                "id":       f"ALERT-SCENARIO-{sc.get('name','?')}",
                "type":     "SCENARIO_TRIGGERED",
                "message":  f"New disruption: {sc.get('name','Unknown')}",
                "turn":     self.state.turn,
                "priority": "CRITICAL",
            })
        for default in financial_outcomes.get("new_defaults", []):
            alerts.append({
                "id":       f"ALERT-DEFAULT-{default['lc_id']}",
                "type":     "LC_DEFAULT",
                "message":  f"LC {default['lc_id']} defaulted — loss ${default['amount_usd']*0.45:,.0f}",
                "turn":     self.state.turn,
                "priority": "CRITICAL",
            })
        return alerts

    # ── Player interface ────────────────────────────────────────────────────

    def get_available_actions(self) -> dict:
        """All actions available to player this turn."""
        mode = self.state.game_mode
        actions: dict = {}

        if mode == "trade_finance":
            pending = [lc for lc in self.state.active_lcs if lc.get("status") == "PENDING"]
            actions["pending_lcs"] = [
                {"action": f"approve_lc / reject_lc / amend_lc_tenor",
                 "lc_id": lc["lc_id"],
                 "amount_usd": lc["amount_usd"],
                 "route": lc["route"],
                 "risk_score": lc["risk_score"],
                 "suez_transit": lc["suez_transit"]}
                for lc in pending[:20]
            ]
            actions["monitoring_actions"] = [
                {"action": "increase_monitoring", "description": "Flag client for enhanced surveillance"},
                {"action": "trigger_early_warning", "description": "Pre-emptive CCC alert to client"},
                {"action": "offer_facility_increase", "description": "Extend WC facility limit"},
            ]
            actions["pricing_action"] = {"action": "set_lc_pricing", "params": {"fee_pct": "float 0-0.02"}}

        elif mode == "scf_pricing":
            pending = [s for s in self.state.scf_portfolio if s.get("status") == "PENDING"]
            actions["pending_suppliers"] = [
                {"action": "approve_supplier / reject_supplier / set_discount_rate",
                 "supplier_id": s["supplier_id"],
                 "invoice_amount_usd": s["invoice_amount_usd"],
                 "otif_score": s["otif_score"],
                 "risk_tier": s["risk_tier"]}
                for s in pending[:30]
            ]

        return actions

    def get_intelligence_signals(self) -> dict:
        """LogisChain AI signals available to player this turn."""
        return self.intelligence_layer.get_signals_for_player(self.state)

    def get_game_state_summary(self) -> dict:
        """Full dashboard KPIs for current turn."""
        total = sum(self.state.player_score.values())
        ai_total = sum(self.state.ai_score.values())
        active_lcs = [lc for lc in self.state.active_lcs if lc.get("status") == "ACTIVE"]
        return {
            "turn":               self.state.turn,
            "year":               self.state.year,
            "game_mode":          self.state.game_mode,
            "portfolio_value":    round(self.state.portfolio_value_usd, 0),
            "cash_usd":           round(self.state.cash_usd, 0),
            "player_total_score": round(total, 1),
            "ai_total_score":     round(ai_total, 1),
            "score_gap":          round(total - ai_total, 1),
            "score_breakdown":    {k: round(v, 1) for k, v in self.state.player_score.items()},
            "active_lcs":         len(active_lcs),
            "npl_ratio":          round(self.state.npl_ratio * 100, 2),
            "active_disruptions": len(self.state.active_disruptions),
            "unread_alerts":      len(self.state.alerts),
            "portfolio_yield_pct": round(self.state.portfolio_yield_pct, 4),
            "avg_congestion":     round(np.mean(list(self.state.port_congestion_index.values())), 2),
            "suez_lc_count":      sum(1 for lc in self.state.active_lcs
                                       if lc.get("suez_transit") and lc.get("status") == "ACTIVE"),
        }

    @property
    def is_game_over(self) -> bool:
        return self.state.cash_usd < 0 or self.state.turn > 52 * 2


# ═══════════════════════════════════════════════════════════════════════════════
# v0.1.0 backward-compatible classes
# ═══════════════════════════════════════════════════════════════════════════════

import copy as _copy
from dataclasses import dataclass as _dc, field as _field


@_dc
class PortfolioState:
    cash_usd: float = 5_000_000.0
    trade_finance_exposure_usd: float = 10_000_000.0
    inventory_value_usd: float = 2_000_000.0
    accounts_receivable_usd: float = 3_000_000.0
    accounts_payable_usd: float = 1_500_000.0
    carrier_contracts: Dict[str, float] = _field(default_factory=dict)
    route_diversification_score: float = 0.5
    insurance_coverage_pct: float = 0.3
    credit_reserves_usd: float = 500_000.0
    period: int = 0
    score: float = 0.0
    decisions_log: List[dict] = _field(default_factory=list)

    @property
    def cash_conversion_cycle(self) -> float:
        daily = max(self.trade_finance_exposure_usd / 90, 1)
        return (self.accounts_receivable_usd / daily
                + self.inventory_value_usd / daily
                - self.accounts_payable_usd / daily)

    @property
    def net_working_capital(self) -> float:
        return (self.cash_usd + self.accounts_receivable_usd
                + self.inventory_value_usd - self.accounts_payable_usd)

    @property
    def liquidity_ratio(self) -> float:
        cl = max(self.accounts_payable_usd + 500_000, 1)
        ca = self.cash_usd + self.accounts_receivable_usd + self.inventory_value_usd
        return ca / cl


@_dc
class SimulationResult:
    period: int
    scenario_applied: Optional[str]
    state_before: PortfolioState
    state_after: PortfolioState
    financial_impact_usd: float
    ccc_change_days: float
    pd_change: float
    period_score: float
    decisions_made: List[str]
    narrative: str


class SimulationEngine:
    """v0.1.0 engine — kept for backward compatibility."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state = PortfolioState()
        self.history: List[SimulationResult] = []
        self.active_scenario = None

    def reset(self, initial_state: Optional[PortfolioState] = None):
        self.state = initial_state or PortfolioState()
        self.history = []
        self.active_scenario = None

    def _draw_scenario(self) -> Optional[object]:
        from src.simulation.scenarios import SCENARIO_LIBRARY
        for sc in self.rng.permutation(list(SCENARIO_LIBRARY.values())):
            if self.rng.random() < sc.probability / 4:
                return sc
        return None

    def _apply_disruption(self, state: PortfolioState, sc) -> Tuple[PortfolioState, float, float, float]:
        state = _copy.deepcopy(state)
        freight_impact = state.trade_finance_exposure_usd * 0.02 * (sc.freight_cost_multiplier - 1)
        inv_loss = state.inventory_value_usd * abs(sc.supply_shock)
        inv_net = inv_loss * (1 - state.insurance_coverage_pct)
        lc_spread = state.trade_finance_exposure_usd * (sc.lc_default_spread_bps / 10_000) * (sc.duration_days / 365)
        absorbed = min(state.credit_reserves_usd, lc_spread * 0.5)
        state.credit_reserves_usd -= absorbed
        total_loss = (freight_impact + inv_net + lc_spread - absorbed) * (1 - state.route_diversification_score * 0.3)
        state.cash_usd -= total_loss
        state.inventory_value_usd -= inv_net
        state.accounts_receivable_usd *= 1 + (sc.transit_time_multiplier - 1) * 0.1
        ccc_change = sc.ccc_impact_days * (1 - state.route_diversification_score * 0.3)
        pd_change = (sc.credit_spread_widening_bps / 10_000) * 0.5
        return state, total_loss, ccc_change, pd_change

    def _apply_player_action(self, state: PortfolioState, action: str, params: Optional[dict] = None) -> Tuple[PortfolioState, str]:
        state = _copy.deepcopy(state)
        p = params or {}
        if action == "buy_insurance":
            cov = float(p.get("coverage_pct", 0.2))
            cost = state.trade_finance_exposure_usd * cov * 0.005
            state.cash_usd -= cost
            state.insurance_coverage_pct = min(state.insurance_coverage_pct + cov, 1.0)
            return state, f"Purchased {cov*100:.0f}% insurance at ${cost:,.0f} premium."
        elif action == "diversify_carriers":
            inv = float(p.get("investment_usd", 100_000))
            state.cash_usd -= inv
            state.route_diversification_score = min(state.route_diversification_score + inv / 2_000_000, 1.0)
            return state, f"Invested ${inv:,.0f} in carrier diversification."
        elif action == "build_credit_reserves":
            amt = float(p.get("amount_usd", 200_000))
            state.cash_usd -= amt
            state.credit_reserves_usd += amt
            return state, f"Added ${amt:,.0f} to credit reserves."
        elif action == "reduce_lc_exposure":
            pct = float(p.get("reduction_pct", 0.1))
            released = state.trade_finance_exposure_usd * pct
            state.trade_finance_exposure_usd -= released
            state.cash_usd += released * 0.95
            return state, f"Reduced LC exposure by {pct*100:.0f}%."
        elif action == "early_payment_scf":
            amt = float(p.get("amount_usd", 500_000))
            state.accounts_receivable_usd -= amt
            state.cash_usd += amt * 0.98
            return state, f"SCF early payment ${amt:,.0f} (2% discount)."
        return state, "No action taken."

    def step(self, player_actions: Optional[List[Tuple[str, Optional[dict]]]] = None) -> SimulationResult:
        player_actions = player_actions or [("hold", {})]
        state_before = _copy.deepcopy(self.state)
        action_narratives = []
        for action, params in player_actions:
            self.state, narr = self._apply_player_action(self.state, action, params)
            action_narratives.append(narr)
            self.state.decisions_log.append({"period": self.state.period, "action": action})
        sc = self._draw_scenario()
        total_loss, ccc_change, pd_change = 0.0, 0.0, 0.0
        sc_name = None
        if sc:
            self.state, total_loss, ccc_change, pd_change = self._apply_disruption(self.state, sc)
            sc_name = sc.name
        organic = self.state.trade_finance_exposure_usd * 0.03
        self.state.cash_usd += organic
        self.state.period += 1
        from src.simulation.scoring import compute_period_score
        period_score = compute_period_score(state_before, self.state, sc)
        self.state.score += period_score
        result = SimulationResult(
            period=self.state.period, scenario_applied=sc_name,
            state_before=state_before, state_after=_copy.deepcopy(self.state),
            financial_impact_usd=total_loss, ccc_change_days=ccc_change,
            pd_change=pd_change, period_score=period_score,
            decisions_made=action_narratives,
            narrative=f"Period {self.state.period}: " + " | ".join(action_narratives[:2]),
        )
        self.history.append(result)
        return result

    def run_auto(self, periods: int = 8) -> List[SimulationResult]:
        return [self.step([("hold", {})]) for _ in range(periods)]

    def get_history_df(self) -> pd.DataFrame:
        rows = []
        for r in self.history:
            rows.append({"period": r.period, "scenario": r.scenario_applied or "None",
                         "financial_impact_usd": r.financial_impact_usd,
                         "ccc_change_days": r.ccc_change_days,
                         "cash_usd": r.state_after.cash_usd,
                         "net_working_capital": r.state_after.net_working_capital,
                         "liquidity_ratio": r.state_after.liquidity_ratio,
                         "period_score": r.period_score,
                         "cumulative_score": r.state_after.score})
        return pd.DataFrame(rows)
