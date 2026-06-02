"""Production-grade Heterogeneous Graph Attention Network for supply chain risk embedding.

Architecture
────────────
HetGAT uses PyTorch Geometric HeteroConv wrapping GATConv per edge type.

Node types  : supplier, port, customer
Edge types  : (supplier, supplies, port)
              (port, ships_to, customer)
              (supplier, finances, customer)
              (supplier, owns, supplier)

Output      : 128-dim risk embedding per node
Targets     : AUC > 0.75 (link prediction), Accuracy > 0.70 (node classification)

MLflow experiment: logischain_ai / hetgat_training
"""

import copy
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

logger = logging.getLogger(__name__)

# ── PyG availability ──────────────────────────────────────────────────────────
try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, GATConv
    from torch_geometric.utils import negative_sampling
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logger.warning("torch_geometric not installed — HetGAT running in stub mode.")
    HeteroData = dict  # type alias for stubs

# ── Constants ─────────────────────────────────────────────────────────────────
NODE_TYPES: List[str] = ["supplier", "port", "customer"]
EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("supplier", "supplies", "port"),
    ("port", "ships_to", "customer"),
    ("supplier", "finances", "customer"),
    ("supplier", "owns", "supplier"),
]
N_PORTS = 20
N_CUSTOMERS = 50
N_RISK_CLASSES = 3          # 0=LOW, 1=MEDIUM, 2=HIGH
SUPPLIER_NUMERIC_FEATURES = [
    "revenue_usd", "ebitda_margin", "current_ratio", "quick_ratio",
    "debt_equity", "interest_coverage", "otif_rate", "lead_time_mean",
    "lead_time_std", "inventory_turnover", "supplier_concentration_hhi",
    "customer_concentration_hhi", "dso", "dpo", "dio", "cash_conversion_cycle",
    "fill_rate", "freight_cost_ratio", "capacity_utilization",
    "betweenness_centrality", "clustering_coeff", "pagerank",
    "country_risk_score", "natural_disaster_exposure", "geopolitical_risk",
    "port_proximity_score",
]
PORT_FEATURE_DIM = 5          # lat, lon, teu_norm, congestion, reliability
CUSTOMER_FEATURE_DIM = 8      # demand, credit, payment_rel, sector, region, ccc, leverage, rating


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SupplyChainHeteroGraph
# ═══════════════════════════════════════════════════════════════════════════════

class SupplyChainHeteroGraph:
    """Converts raw supplier/edge DataFrames into a PyG HeteroData object.

    Node types created
    ──────────────────
    supplier  — real nodes from suppliers_df with 21+ SC features
    port      — N_PORTS synthetic port nodes (lat/lon/TEU/congestion/reliability)
    customer  — N_CUSTOMERS synthetic downstream buyer nodes

    Edge types created
    ──────────────────
    (supplier, supplies, port)     from 'supplies'  edges in edges_df
    (port, ships_to, customer)     synthetic (each port connects to ~5 customers)
    (supplier, finances, customer) from 'finances'  edges in edges_df
    (supplier, owns, supplier)     from 'owns'      edges in edges_df

    Node labels (risk_tier)
    ───────────────────────
    Tier computed as composite of OTIF, country risk, fill rate, and D/E.
    0 = LOW, 1 = MEDIUM, 2 = HIGH
    """

    def __init__(
        self,
        suppliers_df: pd.DataFrame,
        edges_df: Optional[pd.DataFrame] = None,
        seed: int = 42,
    ):
        self.suppliers_df = suppliers_df.reset_index(drop=True)
        self.edges_df = edges_df if edges_df is not None else pd.DataFrame()
        self.rng = np.random.default_rng(seed)
        self._scaler = StandardScaler()

        # Build supplier-id → integer index mapping
        self._sup_id_to_idx: Dict[str, int] = {
            sid: i for i, sid in enumerate(self.suppliers_df.get("supplier_id", self.suppliers_df.index))
        }
        self.n_suppliers = len(self.suppliers_df)
        self.n_ports = N_PORTS
        self.n_customers = N_CUSTOMERS

    # ── Feature extraction ─────────────────────────────────────────────────

    def _supplier_features(self) -> np.ndarray:
        available = [c for c in SUPPLIER_NUMERIC_FEATURES if c in self.suppliers_df.columns]
        # Fill obvious derived cols if missing
        df = self.suppliers_df.copy()
        if "cash_conversion_cycle" not in df.columns and all(
            c in df.columns for c in ["dio", "dso", "dpo"]
        ):
            df["cash_conversion_cycle"] = df["dio"] + df["dso"] - df["dpo"]
        feats = df[available].fillna(df[available].median()).values.astype(np.float32)
        feats = np.clip(feats, -1e6, 1e6)
        return self._scaler.fit_transform(feats)

    def _port_features(self) -> np.ndarray:
        """Synthetic port nodes: lat, lon, teu_norm, congestion, reliability."""
        lat = self.rng.uniform(-50, 60, self.n_ports)
        lon = self.rng.uniform(-180, 180, self.n_ports)
        teu = self.rng.lognormal(0, 0.5, self.n_ports)
        congestion = self.rng.beta(3, 7, self.n_ports)
        reliability = self.rng.beta(8, 2, self.n_ports)
        return np.stack([lat, lon, teu, congestion, reliability], axis=1).astype(np.float32)

    def _customer_features(self) -> np.ndarray:
        """Synthetic customer nodes: 8 demand/credit features."""
        return self.rng.normal(0, 1, (self.n_customers, CUSTOMER_FEATURE_DIM)).astype(np.float32)

    # ── Risk tier labels ───────────────────────────────────────────────────

    def _compute_risk_tiers(self) -> np.ndarray:
        df = self.suppliers_df
        otif = df.get("otif_rate", pd.Series(0.85, index=df.index)).fillna(0.85).values
        cr   = df.get("country_risk_score", pd.Series(0.30, index=df.index)).fillna(0.30).values
        fill = df.get("fill_rate", pd.Series(0.90, index=df.index)).fillna(0.90).values
        de   = df.get("debt_equity", df.get("debt_to_equity", pd.Series(1.0, index=df.index))).fillna(1.0).values
        ic   = df.get("interest_coverage", pd.Series(5.0, index=df.index)).fillna(5.0).values

        de_norm = np.clip(de, 0, 10) / 10.0
        ic_norm = 1.0 - np.clip(ic, 0, 10) / 10.0
        score = (
            0.30 * (1.0 - np.clip(otif, 0, 1))
            + 0.20 * np.clip(cr, 0, 1)
            + 0.20 * (1.0 - np.clip(fill, 0, 1))
            + 0.15 * de_norm
            + 0.15 * ic_norm
        )
        tiers = np.where(score < 0.30, 0, np.where(score < 0.55, 1, 2)).astype(np.int64)
        return tiers

    # ── Edge index builders ────────────────────────────────────────────────

    def _build_supplier_port_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """supplier → port: map 'supplies' edges from edges_df, or use random fallback."""
        src, dst = [], []
        rel_scores, transit, volumes = [], [], []

        if not self.edges_df.empty and "edge_type" in self.edges_df.columns:
            mask = self.edges_df["edge_type"] == "supplies"
            sub = self.edges_df[mask]
            for _, row in sub.iterrows():
                s_id = str(row.get("source_id", ""))
                s_idx = self._sup_id_to_idx.get(s_id, -1)
                if s_idx == -1:
                    continue
                p_idx = int(self.rng.integers(0, self.n_ports))
                src.append(s_idx)
                dst.append(p_idx)
                rel_scores.append(float(row.get("reliability_score", 0.85)))
                transit.append(float(row.get("transit_time_days", 14)))
                volumes.append(float(row.get("volume_usd", 1e6)))

        # Ensure every supplier has ≥ 1 port connection
        for s_idx in range(self.n_suppliers):
            if s_idx not in src:
                p_idx = int(self.rng.integers(0, self.n_ports))
                src.append(s_idx)
                dst.append(p_idx)
                rel_scores.append(float(self.rng.beta(8, 2)))
                transit.append(float(self.rng.lognormal(2.5, 0.4)))
                volumes.append(float(self.rng.lognormal(13, 1.5)))

        return (
            np.array(src, dtype=np.int64),
            np.array(dst, dtype=np.int64),
            np.column_stack([rel_scores, transit, volumes]).astype(np.float32),
        )

    def _build_port_customer_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """port → customer: each port ships to 3-6 customers."""
        src, dst = [], []
        for p in range(self.n_ports):
            n_links = int(self.rng.integers(3, 7))
            customers = self.rng.choice(self.n_customers, size=n_links, replace=False)
            for c in customers:
                src.append(p)
                dst.append(int(c))
        edge_feats = self.rng.uniform(0, 1, (len(src), 3)).astype(np.float32)
        return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64), edge_feats

    def _build_supplier_customer_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """supplier → customer: 'finances' edges from edges_df or random."""
        src, dst = [], []
        rel_scores, amounts = [], []

        if not self.edges_df.empty and "edge_type" in self.edges_df.columns:
            mask = self.edges_df["edge_type"] == "finances"
            for _, row in self.edges_df[mask].iterrows():
                s_id = str(row.get("source_id", ""))
                s_idx = self._sup_id_to_idx.get(s_id, -1)
                if s_idx == -1:
                    continue
                c_idx = int(self.rng.integers(0, self.n_customers))
                src.append(s_idx)
                dst.append(c_idx)
                rel_scores.append(float(row.get("reliability_score", 0.80)))
                amounts.append(float(row.get("volume_usd", 5e5)))

        # Fallback: each supplier finances ~2 random customers
        if len(src) < self.n_suppliers:
            for s_idx in range(self.n_suppliers):
                for _ in range(2):
                    c_idx = int(self.rng.integers(0, self.n_customers))
                    src.append(s_idx)
                    dst.append(c_idx)
                    rel_scores.append(float(self.rng.beta(7, 3)))
                    amounts.append(float(self.rng.lognormal(12, 1)))

        edge_feats = np.column_stack([rel_scores, amounts]).astype(np.float32)
        return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64), edge_feats

    def _build_supplier_owns_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """supplier → supplier: 'owns' edges from edges_df."""
        src, dst = [], []
        if not self.edges_df.empty and "edge_type" in self.edges_df.columns:
            mask = self.edges_df["edge_type"] == "owns"
            for _, row in self.edges_df[mask].iterrows():
                s_src = str(row.get("source_id", ""))
                s_dst = str(row.get("target_id", ""))
                i = self._sup_id_to_idx.get(s_src, -1)
                j = self._sup_id_to_idx.get(s_dst, -1)
                if i != -1 and j != -1 and i != j:
                    src.append(i)
                    dst.append(j)

        # Fallback synthetic ownership chains
        if len(src) < 10:
            for _ in range(max(10, self.n_suppliers // 10)):
                i = int(self.rng.integers(0, self.n_suppliers))
                j = int(self.rng.integers(0, self.n_suppliers))
                if i != j:
                    src.append(i)
                    dst.append(j)
        return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)

    # ── Main builder ───────────────────────────────────────────────────────

    def build_hetero_data(self) -> "HeteroData":
        """Build a PyG HeteroData object from suppliers_df and edges_df.

        Returns HeteroData with:
        - node features for all 3 node types
        - edge indices and edge features for all 4 edge types
        - supplier node classification labels (risk_tier)
        - negative edges for link prediction (edge_label_index, edge_label)
        """
        if not PYG_AVAILABLE:
            logger.warning("PyG unavailable — returning stub data dict.")
            return self._stub_data()

        data = HeteroData()

        # ── Node features ──
        sup_x = self._supplier_features()
        data["supplier"].x = torch.tensor(sup_x, dtype=torch.float)
        data["supplier"].y = torch.tensor(self._compute_risk_tiers(), dtype=torch.long)
        data["supplier"].node_ids = list(self._sup_id_to_idx.keys())

        port_x = self._port_features()
        data["port"].x = torch.tensor(port_x, dtype=torch.float)

        cust_x = self._customer_features()
        data["customer"].x = torch.tensor(cust_x, dtype=torch.float)

        # ── Edge indices and features ──
        sp_src, sp_dst, sp_ef = self._build_supplier_port_edges()
        data["supplier", "supplies", "port"].edge_index = torch.tensor(
            np.stack([sp_src, sp_dst]), dtype=torch.long
        )
        data["supplier", "supplies", "port"].edge_attr = torch.tensor(sp_ef, dtype=torch.float)

        pc_src, pc_dst, pc_ef = self._build_port_customer_edges()
        data["port", "ships_to", "customer"].edge_index = torch.tensor(
            np.stack([pc_src, pc_dst]), dtype=torch.long
        )
        data["port", "ships_to", "customer"].edge_attr = torch.tensor(pc_ef, dtype=torch.float)

        sc_src, sc_dst, sc_ef = self._build_supplier_customer_edges()
        data["supplier", "finances", "customer"].edge_index = torch.tensor(
            np.stack([sc_src, sc_dst]), dtype=torch.long
        )
        data["supplier", "finances", "customer"].edge_attr = torch.tensor(sc_ef, dtype=torch.float)

        oo_src, oo_dst = self._build_supplier_owns_edges()
        data["supplier", "owns", "supplier"].edge_index = torch.tensor(
            np.stack([oo_src, oo_dst]), dtype=torch.long
        )

        # ── Store dimension metadata ──
        data.supplier_feature_dim = int(sup_x.shape[1])
        data.port_feature_dim = PORT_FEATURE_DIM
        data.customer_feature_dim = CUSTOMER_FEATURE_DIM

        logger.info(
            f"HeteroData built: {self.n_suppliers} suppliers, "
            f"{self.n_ports} ports, {self.n_customers} customers | "
            f"S→P: {sp_src.shape[0]}, P→C: {pc_src.shape[0]}, "
            f"S→C: {sc_src.shape[0]}, S→S: {oo_src.shape[0]} edges"
        )
        return data

    def _stub_data(self) -> dict:
        """Minimal stub when PyG is unavailable — returns plain dict."""
        return {
            "n_supplier": self.n_suppliers,
            "n_port": self.n_ports,
            "n_customer": self.n_customers,
            "supplier_x": self._supplier_features(),
            "port_x": self._port_features(),
            "customer_x": self._customer_features(),
            "supplier_y": self._compute_risk_tiers(),
        }

    def add_synthetic_edges(
        self, data: "HeteroData", n_negative: int = 5000
    ) -> "HeteroData":
        """Add negative edge samples for link prediction training.

        Attaches `edge_label_index` and `edge_label` to the primary link-pred
        edge type ('supplier', 'supplies', 'port').

        Positive edges: existing supply edges (label=1)
        Negative edges: randomly sampled non-existent edges (label=0)
        """
        if not PYG_AVAILABLE:
            return data

        pos_ei = data["supplier", "supplies", "port"].edge_index
        n_pos = pos_ei.size(1)
        n_neg = min(n_negative, self.n_suppliers * self.n_ports - n_pos)

        # Negative sampling: random (supplier, port) pairs not in pos set
        pos_set = set(zip(pos_ei[0].tolist(), pos_ei[1].tolist()))
        neg_src, neg_dst = [], []
        attempts = 0
        while len(neg_src) < n_neg and attempts < n_neg * 10:
            attempts += 1
            s = int(self.rng.integers(0, self.n_suppliers))
            p = int(self.rng.integers(0, self.n_ports))
            if (s, p) not in pos_set:
                neg_src.append(s)
                neg_dst.append(p)

        neg_src_t = torch.tensor(neg_src, dtype=torch.long)
        neg_dst_t = torch.tensor(neg_dst, dtype=torch.long)
        neg_ei = torch.stack([neg_src_t, neg_dst_t])

        label_index = torch.cat([pos_ei, neg_ei], dim=1)
        labels = torch.cat([
            torch.ones(n_pos, dtype=torch.float),
            torch.zeros(neg_ei.size(1), dtype=torch.float),
        ])

        data["supplier", "supplies", "port"].edge_label_index = label_index
        data["supplier", "supplies", "port"].edge_label = labels
        return data

    def visualize_network(
        self,
        data: "HeteroData",
        color_by: str = "risk_score",
        risk_scores: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
    ):
        """Visualise the supply chain graph.

        Node size  ∝ betweenness centrality (supplier) or degree (port/customer)
        Node color = risk score gradient (green=low → red=high) for suppliers
        Edge width ∝ log(volume)
        """
        G = nx.DiGraph()

        # Add supplier nodes
        for i in range(self.n_suppliers):
            label = f"S{i}"
            rs = float(risk_scores[i]) if risk_scores is not None else 0.3
            G.add_node(label, node_type="supplier", risk_score=rs)

        # Add port nodes
        for p in range(self.n_ports):
            G.add_node(f"P{p}", node_type="port", risk_score=0.1)

        # Add customer nodes (small subset for clarity)
        for c in range(min(self.n_customers, 10)):
            G.add_node(f"C{c}", node_type="customer", risk_score=0.2)

        # Add edges if PyG data available
        if PYG_AVAILABLE and isinstance(data, HeteroData):
            ei = data["supplier", "supplies", "port"].edge_index
            for s, p in zip(ei[0].tolist(), ei[1].tolist()):
                G.add_edge(f"S{s}", f"P{p}", weight=1.0, edge_type="supplies")

            ei = data["port", "ships_to", "customer"].edge_index
            for p, c in zip(ei[0].tolist(), ei[1].tolist()):
                if c < 10:
                    G.add_edge(f"P{p}", f"C{c}", weight=1.0, edge_type="ships_to")

        # Layout and drawing
        if G.number_of_nodes() == 0:
            return

        betweenness = nx.betweenness_centrality(G)
        try:
            pos = nx.spring_layout(G, seed=42, k=0.5)
        except Exception:
            pos = nx.random_layout(G, seed=42)

        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap("RdYlGn_r")

        for ntype, marker, size_mult in [
            ("supplier", "o", 800), ("port", "s", 600), ("customer", "^", 400)
        ]:
            nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == ntype]
            if not nodes:
                continue
            node_colors = [cmap(G.nodes[n].get("risk_score", 0.3)) for n in nodes]
            sizes = [size_mult * (1 + 3 * betweenness.get(n, 0)) for n in nodes]
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes, node_color=node_colors,
                node_size=sizes, node_shape=marker, ax=ax, alpha=0.85
            )
            nx.draw_networkx_labels(
                G, pos, labels={n: n for n in nodes[:20]},
                font_size=6, ax=ax
            )

        nx.draw_networkx_edges(
            G, pos, edge_color="gray", arrows=True, arrowsize=10,
            width=0.8, alpha=0.5, ax=ax
        )

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="green", label="Low risk (supplier)"),
            Patch(facecolor="orange", label="Medium risk (supplier)"),
            Patch(facecolor="red", label="High risk (supplier)"),
            Patch(facecolor="steelblue", label="Port ■"),
            Patch(facecolor="gray", label="Customer ▲"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
        ax.set_title(f"Supply Chain Risk Network (color_by={color_by})", fontsize=12)
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.tight_layout()
        plt.close()
        return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  HetGAT Model
# ═══════════════════════════════════════════════════════════════════════════════

class _StubHetGAT(nn.Module):
    """Fallback MLP when PyG is unavailable — mimics HetGAT interface."""

    def __init__(self, in_channels_dict, hidden_channels=128, out_channels=128, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.mlps = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(in_ch, hidden_channels), nn.ReLU(),
                nn.Linear(hidden_channels, out_channels),
            )
            for nt, in_ch in in_channels_dict.items()
        })
        self.clf_head = nn.Sequential(
            nn.Linear(out_channels, 64), nn.ReLU(), nn.Linear(64, N_RISK_CLASSES)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        return {nt: self.mlps[nt](x) for nt, x in x_dict.items() if nt in self.mlps}

    def get_attention_weights(self, x_dict, edge_index_dict):
        return {}


if PYG_AVAILABLE:
    class HetGAT(nn.Module):
        """Heterogeneous Graph Attention Network for supply chain risk embedding.

        Architecture per layer
        ──────────────────────
        1. Input projection: in_ch → hidden_channels  (per node type)
        2. HeteroConv[GATConv × 4 edge types]
        3. Residual skip connection  +  BatchNorm  +  ELU  +  Dropout
        Repeat for num_layers.
        4. Output MLP: hidden_channels → out_channels (128-dim embeddings)
        5. Node classification head: out_channels → 3 risk classes (supplier only)
        """

        def __init__(
            self,
            in_channels_dict: Dict[str, int],
            hidden_channels: int = 128,
            out_channels: int = 128,
            num_heads: int = 4,
            num_layers: int = 3,
            dropout: float = 0.2,
        ):
            super().__init__()
            assert hidden_channels % num_heads == 0, (
                f"hidden_channels ({hidden_channels}) must be divisible by num_heads ({num_heads})"
            )
            self.hidden_channels = hidden_channels
            self.out_channels = out_channels
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.dropout = dropout
            self._in_channels_dict = in_channels_dict
            head_channels = hidden_channels // num_heads

            # ── Input projections (one per node type) ──
            self.input_proj = nn.ModuleDict({
                nt: nn.Sequential(
                    nn.Linear(in_ch, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.ELU(inplace=True),
                )
                for nt, in_ch in in_channels_dict.items()
            })

            # ── L × HeteroConv layers ──
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()

            for layer_idx in range(num_layers):
                conv_dict = {}
                for et in EDGE_TYPES:
                    src_t, _, dst_t = et
                    conv_dict[et] = GATConv(
                        in_channels=(hidden_channels, hidden_channels),
                        out_channels=head_channels,
                        heads=num_heads,
                        dropout=dropout,
                        add_self_loops=(src_t == dst_t),
                        concat=True,   # output dim = num_heads × head_channels = hidden_channels
                        bias=True,
                    )
                self.convs.append(HeteroConv(conv_dict, aggr="sum"))

                norm = nn.ModuleDict({
                    nt: nn.BatchNorm1d(hidden_channels)
                    for nt in NODE_TYPES
                })
                self.norms.append(norm)

            # ── Output MLP ──
            self.out_mlp = nn.ModuleDict({
                nt: nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ELU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_channels, out_channels),
                )
                for nt in NODE_TYPES
            })

            # ── Node classification head (supplier only) ──
            self.clf_head = nn.Sequential(
                nn.Linear(out_channels, 64),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, N_RISK_CLASSES),
            )

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def _encode(
            self,
            x_dict: Dict[str, torch.Tensor],
            edge_index_dict: Dict,
        ) -> Dict[str, torch.Tensor]:
            """Run input projection + GNN layers, returning hidden representations."""
            h = {}
            for nt, x in x_dict.items():
                if nt in self.input_proj:
                    h[nt] = self.input_proj[nt](x)

            for layer_idx, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
                h_prev = {nt: t.clone() for nt, t in h.items()}

                valid_edges = {
                    et: ei
                    for et, ei in edge_index_dict.items()
                    if et in {e: None for e in EDGE_TYPES} and ei.size(1) > 0
                }
                h_new = conv(h, valid_edges)

                for nt in NODE_TYPES:
                    new_t = h_new.get(nt)
                    prev_t = h_prev.get(nt)
                    if new_t is not None:
                        out_t = new_t + prev_t if prev_t is not None else new_t
                        if nt in norm_dict and out_t.size(0) > 1:
                            out_t = norm_dict[nt](out_t)
                        out_t = F.elu(out_t)
                        out_t = F.dropout(out_t, p=self.dropout, training=self.training)
                        h[nt] = out_t

            return h

        def forward(
            self,
            x_dict: Dict[str, torch.Tensor],
            edge_index_dict: Dict,
            edge_attr_dict: Optional[Dict] = None,
        ) -> Dict[str, torch.Tensor]:
            """Forward pass.

            Returns dict {node_type: (N, out_channels)} embeddings.
            """
            h = self._encode(x_dict, edge_index_dict)
            return {nt: self.out_mlp[nt](h_t) for nt, h_t in h.items() if nt in self.out_mlp}

        def get_attention_weights(
            self,
            x_dict: Dict[str, torch.Tensor],
            edge_index_dict: Dict,
        ) -> Dict[str, Dict]:
            """Extract per-layer, per-edge-type attention weights.

            Returns nested dict:
                {
                    "layer_0": {
                        ('supplier','supplies','port'): {
                            "edge_index": Tensor[2, E],
                            "weights":    Tensor[E, num_heads],
                            "mean_attention": float,
                        },
                        ...
                    },
                    ...
                }
            """
            h = {}
            with torch.no_grad():
                for nt, x in x_dict.items():
                    if nt in self.input_proj:
                        h[nt] = self.input_proj[nt](x)

            attention_dict: Dict[str, Dict] = {}

            for layer_idx, hetero_conv in enumerate(self.convs):
                layer_attn: Dict[Tuple, Dict] = {}

                for et in EDGE_TYPES:
                    et_key = "__".join(et)
                    if et_key not in hetero_conv.convs:
                        continue
                    if et not in edge_index_dict or edge_index_dict[et].size(1) == 0:
                        continue

                    src_t, _, dst_t = et
                    src_x = h.get(src_t)
                    dst_x = h.get(dst_t)
                    if src_x is None or dst_x is None:
                        continue

                    gat = hetero_conv.convs[et_key]
                    try:
                        with torch.no_grad():
                            _, (ei, alpha) = gat(
                                (src_x, dst_x),
                                edge_index_dict[et],
                                return_attention_weights=True,
                            )
                        layer_attn[et] = {
                            "edge_index": ei.cpu(),
                            "weights": alpha.cpu(),
                            "mean_attention": float(alpha.mean().item()),
                        }
                    except Exception as exc:
                        logger.debug(f"Attention extraction failed for {et}: {exc}")

                attention_dict[f"layer_{layer_idx}"] = layer_attn

                # Advance hidden state through this conv for next layer
                valid_edges = {et: ei for et, ei in edge_index_dict.items()
                               if et in {e: None for e in EDGE_TYPES} and ei.size(1) > 0}
                h_new = hetero_conv(h, valid_edges)
                for nt in NODE_TYPES:
                    new_t = h_new.get(nt)
                    if new_t is not None:
                        h[nt] = F.elu(new_t + h.get(nt, torch.zeros_like(new_t)))

            return attention_dict

else:
    # Alias when PyG unavailable
    HetGAT = _StubHetGAT  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  GNNRiskPredictor
# ═══════════════════════════════════════════════════════════════════════════════

class GNNRiskPredictor:
    """End-to-end trainer and inference engine for HetGAT supply chain risk scoring.

    Supports two tasks
    ──────────────────
    node_classification  — 3-class risk tier (LOW/MEDIUM/HIGH) for supplier nodes
    link_prediction      — predict missing supply chain connections
    both                 — multi-task with equal weighting

    Usage
    ─────
    predictor = GNNRiskPredictor(hidden_channels=128, num_layers=3)
    history = predictor.fit(hetero_data, epochs=200, task='both')
    scores_df = predictor.predict_risk_scores(hetero_data)
    metrics = predictor.evaluate(hetero_data)
    predictor.save('models/gnn_model.pt')
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        device: Optional[str] = None,
    ):
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[nn.Module] = None
        self._fitted = False
        self._best_state: Optional[dict] = None
        self._in_channels_dict: Optional[Dict[str, int]] = None

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_model(self, data: "HeteroData") -> nn.Module:
        """Infer input dimensions and instantiate HetGAT."""
        if PYG_AVAILABLE and isinstance(data, HeteroData):
            in_dict = {
                nt: int(data[nt].x.size(1))
                for nt in NODE_TYPES
                if nt in data.node_types and data[nt].x is not None
            }
        else:
            # Stub
            in_dict = {
                "supplier": data.get("supplier_x", np.zeros((1, 26))).shape[1],
                "port": PORT_FEATURE_DIM,
                "customer": CUSTOMER_FEATURE_DIM,
            }

        self._in_channels_dict = in_dict
        model = HetGAT(
            in_channels_dict=in_dict,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        logger.info(
            f"HetGAT built: {sum(p.numel() for p in model.parameters()):,} params | "
            f"in_dims={in_dict}"
        )
        return model

    def _data_to_device(self, data: "HeteroData") -> "HeteroData":
        if PYG_AVAILABLE and isinstance(data, HeteroData):
            return data.to(self.device)
        return data

    def _get_tensors(self, data: "HeteroData"):
        """Return (x_dict, edge_index_dict) on self.device."""
        if PYG_AVAILABLE and isinstance(data, HeteroData):
            x_dict = {nt: data[nt].x for nt in NODE_TYPES if nt in data.node_types}
            ei_dict = {
                (et[0], et[1], et[2]): data[et[0], et[1], et[2]].edge_index
                for et in EDGE_TYPES
                if (et[0], et[1], et[2]) in data.edge_types
            }
            return x_dict, ei_dict
        # Stub
        x_dict = {
            "supplier": torch.tensor(data["supplier_x"], dtype=torch.float).to(self.device),
            "port": torch.tensor(data["port_x"], dtype=torch.float).to(self.device),
            "customer": torch.tensor(data["customer_x"], dtype=torch.float).to(self.device),
        }
        return x_dict, {}

    def _node_clf_loss(
        self,
        emb_dict: Dict[str, torch.Tensor],
        data: "HeteroData",
        train_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if "supplier" not in emb_dict:
            return torch.tensor(0.0, device=self.device)

        logits = self.model.clf_head(emb_dict["supplier"])  # type: ignore[union-attr]

        if PYG_AVAILABLE and isinstance(data, HeteroData) and hasattr(data["supplier"], "y"):
            labels = data["supplier"].y.to(self.device)
        elif isinstance(data, dict):
            labels = torch.tensor(data["supplier_y"], dtype=torch.long).to(self.device)
        else:
            return torch.tensor(0.0, device=self.device)

        if train_mask is not None:
            logits = logits[train_mask]
            labels = labels[train_mask]

        return F.cross_entropy(logits, labels)

    def _link_pred_loss(
        self, emb_dict: Dict[str, torch.Tensor], data: "HeteroData"
    ) -> torch.Tensor:
        if not PYG_AVAILABLE or not isinstance(data, HeteroData):
            return torch.tensor(0.0, device=self.device)

        try:
            lp_et = ("supplier", "supplies", "port")
            if not hasattr(data[lp_et[0], lp_et[1], lp_et[2]], "edge_label_index"):
                return torch.tensor(0.0, device=self.device)
            label_index = data[lp_et[0], lp_et[1], lp_et[2]].edge_label_index.to(self.device)
            labels = data[lp_et[0], lp_et[1], lp_et[2]].edge_label.to(self.device)
        except Exception:
            return torch.tensor(0.0, device=self.device)

        sup_emb = emb_dict.get("supplier")
        port_emb = emb_dict.get("port")
        if sup_emb is None or port_emb is None:
            return torch.tensor(0.0, device=self.device)

        src_emb = sup_emb[label_index[0]]
        dst_emb = port_emb[label_index[1]]
        # L2 normalise for stable dot-product scoring
        src_emb = F.normalize(src_emb, p=2, dim=-1)
        dst_emb = F.normalize(dst_emb, p=2, dim=-1)
        scores = (src_emb * dst_emb).sum(dim=-1)
        return F.binary_cross_entropy_with_logits(scores, labels)

    def _train_epoch(
        self,
        data: "HeteroData",
        optimizer: torch.optim.Optimizer,
        task: str,
        train_mask: Optional[torch.Tensor],
    ) -> float:
        self.model.train()  # type: ignore[union-attr]
        optimizer.zero_grad()
        x_dict, ei_dict = self._get_tensors(data)
        emb_dict = self.model(x_dict, ei_dict)  # type: ignore[union-attr]

        loss = torch.tensor(0.0, device=self.device)
        if task in ("node_classification", "both"):
            loss = loss + self._node_clf_loss(emb_dict, data, train_mask)
        if task in ("link_prediction", "both"):
            loss = loss + self._link_pred_loss(emb_dict, data)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # type: ignore
        optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def _eval_epoch(self, data: "HeteroData", task: str) -> dict:
        self.model.eval()  # type: ignore[union-attr]
        x_dict, ei_dict = self._get_tensors(data)
        emb_dict = self.model(x_dict, ei_dict)  # type: ignore[union-attr]

        metrics: dict = {}

        # Node classification accuracy (all suppliers)
        if task in ("node_classification", "both") and "supplier" in emb_dict:
            logits = self.model.clf_head(emb_dict["supplier"])  # type: ignore
            preds = logits.argmax(dim=-1).cpu().numpy()
            if PYG_AVAILABLE and isinstance(data, HeteroData) and hasattr(data["supplier"], "y"):
                true = data["supplier"].y.cpu().numpy()
            elif isinstance(data, dict):
                true = data["supplier_y"]
            else:
                true = preds
            metrics["node_clf_accuracy"] = float(accuracy_score(true, preds))
            metrics["node_clf_f1"] = float(f1_score(true, preds, average="macro", zero_division=0))

        # Link prediction AUC
        if task in ("link_prediction", "both") and PYG_AVAILABLE and isinstance(data, HeteroData):
            try:
                lp_et = ("supplier", "supplies", "port")
                if hasattr(data[lp_et[0], lp_et[1], lp_et[2]], "edge_label"):
                    li = data[lp_et[0], lp_et[1], lp_et[2]].edge_label_index.to(self.device)
                    lab = data[lp_et[0], lp_et[1], lp_et[2]].edge_label.cpu().numpy()
                    s_e = F.normalize(emb_dict["supplier"][li[0]], p=2, dim=-1)
                    p_e = F.normalize(emb_dict["port"][li[1]], p=2, dim=-1)
                    scores = torch.sigmoid((s_e * p_e).sum(-1)).cpu().numpy()
                    if len(np.unique(lab)) > 1:
                        metrics["link_pred_auc"] = float(roc_auc_score(lab, scores))
            except Exception:
                pass

        return metrics

    # ── Public API ─────────────────────────────────────────────────────────

    def fit(
        self,
        hetero_data: "HeteroData",
        epochs: int = 200,
        lr: float = 0.001,
        task: str = "both",
        patience: int = 20,
        val_ratio: float = 0.2,
        verbose_every: int = 20,
    ) -> dict:
        """Train the HetGAT model.

        Parameters
        ──────────
        hetero_data  : HeteroData from SupplyChainHeteroGraph.build_hetero_data()
        epochs       : maximum training epochs
        lr           : initial learning rate (cosine annealed to lr/100)
        task         : 'node_classification' | 'link_prediction' | 'both'
        patience     : early stopping patience (epochs without improvement)
        val_ratio    : fraction of supplier nodes held out for validation
        verbose_every: print interval

        Returns history dict with 'train_loss', 'node_clf_accuracy', 'link_pred_auc'.
        """
        self.model = self._build_model(hetero_data)
        data = self._data_to_device(hetero_data)

        # Build train mask for node classification
        n_sup = (
            data["supplier"].x.size(0) if PYG_AVAILABLE and isinstance(data, HeteroData)
            else len(data.get("supplier_y", []))
        )
        perm = torch.randperm(n_sup)
        n_train = max(1, int(n_sup * (1 - val_ratio)))
        train_idx = perm[:n_train]
        train_mask = torch.zeros(n_sup, dtype=torch.bool, device=self.device)
        train_mask[train_idx] = True

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr / 100
        )

        history: dict = {"train_loss": [], "node_clf_accuracy": [], "link_pred_auc": []}
        best_loss = float("inf")
        patience_counter = 0

        run_ctx = (
            mlflow.start_run(run_name="hetgat_training", nested=True)
            if _MLFLOW else _dummy_ctx()
        )

        with run_ctx:
            if _MLFLOW:
                mlflow.log_params({
                    "hidden_channels": self.hidden_channels,
                    "out_channels": self.out_channels,
                    "num_heads": self.num_heads,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "lr": lr, "epochs": epochs, "task": task, "patience": patience,
                })

            for epoch in range(epochs):
                loss = self._train_epoch(data, optimizer, task, train_mask)
                scheduler.step()

                history["train_loss"].append(loss)

                if epoch % verbose_every == 0 or epoch == epochs - 1:
                    val_metrics = self._eval_epoch(data, task)
                    acc = val_metrics.get("node_clf_accuracy", 0.0)
                    auc = val_metrics.get("link_pred_auc", 0.0)
                    history["node_clf_accuracy"].append(acc)
                    history["link_pred_auc"].append(auc)

                    if _MLFLOW:
                        mlflow.log_metrics(
                            {"train_loss": loss, "node_clf_accuracy": acc, "link_pred_auc": auc},
                            step=epoch,
                        )
                    logger.info(
                        f"Epoch {epoch:04d}/{epochs} | loss={loss:.4f} | "
                        f"acc={acc:.3f} | auc={auc:.3f}"
                    )

                # Early stopping on training loss
                if loss < best_loss - 1e-5:
                    best_loss = loss
                    self._best_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                        break

            # Restore best weights
            if self._best_state is not None:
                self.model.load_state_dict(self._best_state)

            final = self._eval_epoch(data, task)
            if _MLFLOW:
                mlflow.log_metrics({f"final_{k}": v for k, v in final.items()})
            logger.info(f"Training complete. Final metrics: {final}")

        self._fitted = True
        return history

    @torch.no_grad()
    def predict_risk_scores(self, hetero_data: "HeteroData") -> pd.DataFrame:
        """Return a DataFrame with risk scores and tiers for all supplier nodes.

        Columns: supplier_id | risk_score (0-1) | risk_tier | embedding_l2_norm
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Call fit() before predict_risk_scores().")

        self.model.eval()
        data = self._data_to_device(hetero_data)
        x_dict, ei_dict = self._get_tensors(data)
        emb_dict = self.model(x_dict, ei_dict)

        sup_emb = emb_dict.get("supplier")
        if sup_emb is None:
            return pd.DataFrame()

        # Risk score: P(HIGH risk) from classification head
        logits = self.model.clf_head(sup_emb)  # type: ignore
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        risk_scores = probs[:, 2]  # P(HIGH)

        tiers = np.where(risk_scores >= 0.60, "HIGH",
                np.where(risk_scores >= 0.30, "MEDIUM", "LOW"))
        emb_norms = sup_emb.norm(dim=-1).cpu().numpy()

        n = sup_emb.size(0)
        if PYG_AVAILABLE and isinstance(data, HeteroData) and hasattr(data["supplier"], "node_ids"):
            ids = list(data["supplier"].node_ids)[:n]
        else:
            ids = [f"SUP-{i:04d}" for i in range(n)]

        return pd.DataFrame({
            "supplier_id":       ids,
            "risk_score":        risk_scores,
            "risk_tier":         tiers,
            "p_low":             probs[:, 0],
            "p_medium":          probs[:, 1],
            "p_high":            probs[:, 2],
            "embedding_l2_norm": emb_norms,
        })

    @torch.no_grad()
    def get_entity_embeddings(self, hetero_data: "HeteroData") -> Dict[str, np.ndarray]:
        """Return {node_type: (N, 128) embedding matrix} for downstream ensemble use."""
        if not self._fitted or self.model is None:
            raise RuntimeError("Call fit() before get_entity_embeddings().")
        self.model.eval()
        data = self._data_to_device(hetero_data)
        x_dict, ei_dict = self._get_tensors(data)
        emb_dict = self.model(x_dict, ei_dict)
        return {nt: emb.cpu().numpy() for nt, emb in emb_dict.items()}

    @torch.no_grad()
    def evaluate(self, hetero_data: "HeteroData") -> dict:
        """Compute link_pred_auc, node_clf_accuracy, node_clf_f1 on all data."""
        if not self._fitted or self.model is None:
            raise RuntimeError("Call fit() before evaluate().")
        data = self._data_to_device(hetero_data)
        return self._eval_epoch(data, "both")

    @torch.no_grad()
    def explain_risk(self, supplier_id: str, hetero_data: "HeteroData") -> dict:
        """Explain a supplier's risk score using GATConv attention weights.

        Returns
        ───────
        {
            "supplier_id": str,
            "risk_score": float,
            "risk_tier": str,
            "attention_by_layer": {
                "layer_0": {
                    edge_type: {
                        "top_neighbors": [{index, attention_weight}],
                        "mean_attention": float
                    }
                }
            },
            "top_risk_drivers": [str, ...],
        }
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Call fit() before explain_risk().")

        data = self._data_to_device(hetero_data)
        x_dict, ei_dict = self._get_tensors(data)
        x_dict_cpu = {nt: x.cpu() for nt, x in x_dict.items()}
        ei_dict_cpu = {et: ei.cpu() for et, ei in ei_dict.items()}

        # Find supplier index
        sup_idx: Optional[int] = None
        if PYG_AVAILABLE and isinstance(data, HeteroData) and hasattr(data["supplier"], "node_ids"):
            ids = list(data["supplier"].node_ids)
            if supplier_id in ids:
                sup_idx = ids.index(supplier_id)

        # Get risk score
        scores_df = self.predict_risk_scores(hetero_data)
        row = scores_df[scores_df["supplier_id"] == supplier_id]
        risk_score = float(row["risk_score"].iloc[0]) if len(row) else 0.5
        risk_tier = str(row["risk_tier"].iloc[0]) if len(row) else "UNKNOWN"

        # Get attention weights
        attn = self.model.get_attention_weights(x_dict_cpu, ei_dict_cpu)  # type: ignore

        explanation: dict = {
            "supplier_id": supplier_id,
            "risk_score": round(risk_score, 4),
            "risk_tier": risk_tier,
            "attention_by_layer": {},
            "top_risk_drivers": [],
        }

        for layer_name, layer_data in attn.items():
            layer_exp: dict = {}
            for et, attn_data in layer_data.items():
                if sup_idx is None:
                    continue
                ei = attn_data["edge_index"]
                alpha = attn_data["weights"]  # (E, num_heads)
                src_t, rel, dst_t = et

                if src_t != "supplier":
                    continue

                src_mask = (ei[0] == sup_idx)
                if not src_mask.any():
                    continue

                neighbor_idxs = ei[1][src_mask]
                attn_vals = alpha[src_mask].mean(dim=-1)  # mean over heads
                top_k = min(5, len(attn_vals))
                top_idx = attn_vals.argsort(descending=True)[:top_k]

                layer_exp[str(et)] = {
                    "top_neighbors": [
                        {
                            "neighbor_index": int(neighbor_idxs[i]),
                            "neighbor_type": dst_t,
                            "attention_weight": round(float(attn_vals[i]), 4),
                        }
                        for i in top_idx
                    ],
                    "mean_attention": attn_data["mean_attention"],
                }
                if float(attn_vals[0]) > 0.15:
                    explanation["top_risk_drivers"].append(
                        f"{rel} → {dst_t}[{int(neighbor_idxs[top_idx[0]])}] "
                        f"(attn={float(attn_vals[top_idx[0]]):.3f})"
                    )

            explanation["attention_by_layer"][layer_name] = layer_exp

        return explanation

    def save(self, path: str = "models/gnn_model.pt"):
        """Save model weights, config, and fitted state to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": (
                self.model.state_dict() if self.model else {}
            ),
            "model_config": {
                "in_channels_dict": self._in_channels_dict or {},
                "hidden_channels": self.hidden_channels,
                "out_channels": self.out_channels,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            "fitted": self._fitted,
        }
        torch.save(payload, path)
        logger.info(f"Model saved → {path}")

    def load(self, path: str = "models/gnn_model.pt"):
        """Load model from disk, reconstructing architecture from saved config."""
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt["model_config"]
        self._in_channels_dict = cfg["in_channels_dict"]
        self.hidden_channels = cfg["hidden_channels"]
        self.out_channels = cfg["out_channels"]
        self.num_heads = cfg["num_heads"]
        self.num_layers = cfg["num_layers"]
        self.dropout = cfg["dropout"]
        self.model = HetGAT(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self._fitted = ckpt.get("fitted", True)
        logger.info(f"Model loaded ← {path}")
        return self


# ── Dummy context manager for when MLflow is unavailable ──────────────────────

class _dummy_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_network_features(graph_data: "HeteroData") -> pd.DataFrame:
    """Compute NetworkX graph metrics from HeteroData.

    Converts the heterogeneous graph to a unified directed NetworkX graph,
    then computes per-node centrality metrics.

    Returns
    ───────
    DataFrame with columns:
        node_id, node_type, degree, in_degree, out_degree,
        betweenness_centrality, closeness_centrality,
        clustering_coefficient, pagerank, node_criticality_score
    """
    G = nx.DiGraph()

    if PYG_AVAILABLE and isinstance(graph_data, HeteroData):
        # Add supplier nodes
        n_sup = graph_data["supplier"].x.size(0) if hasattr(graph_data["supplier"], "x") else 0
        for i in range(n_sup):
            G.add_node(f"S_{i}", node_type="supplier")
        # Add port nodes
        n_port = graph_data["port"].x.size(0) if hasattr(graph_data["port"], "x") else 0
        for i in range(n_port):
            G.add_node(f"P_{i}", node_type="port")
        # Add customer nodes
        n_cust = graph_data["customer"].x.size(0) if hasattr(graph_data["customer"], "x") else 0
        for i in range(n_cust):
            G.add_node(f"C_{i}", node_type="customer")

        for et in EDGE_TYPES:
            try:
                ei = graph_data[et[0], et[1], et[2]].edge_index
                prefix = {"supplier": "S", "port": "P", "customer": "C"}
                src_p = prefix[et[0]]
                dst_p = prefix[et[2]]
                for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                    weight = 1.0
                    G.add_edge(f"{src_p}_{s}", f"{dst_p}_{d}", weight=weight)
            except Exception:
                pass

    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    betweenness = nx.betweenness_centrality(G, normalized=True, weight="weight")
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G, alpha=0.85, weight="weight")
    undirected = G.to_undirected()
    clustering = nx.clustering(undirected)

    rows = []
    for node, attrs in G.nodes(data=True):
        deg = G.degree(node, weight="weight")
        rows.append({
            "node_id":                 node,
            "node_type":               attrs.get("node_type", "unknown"),
            "degree":                  G.degree(node),
            "in_degree":               G.in_degree(node),
            "out_degree":              G.out_degree(node),
            "betweenness_centrality":  betweenness.get(node, 0.0),
            "closeness_centrality":    closeness.get(node, 0.0),
            "clustering_coefficient":  clustering.get(node, 0.0),
            "pagerank":                pagerank.get(node, 0.0),
            # Node criticality = betweenness × (1 + log(1 + volume))
            "node_criticality_score":  betweenness.get(node, 0.0) * (
                1.0 + np.log1p(float(deg))
            ),
        })

    return pd.DataFrame(rows).sort_values("node_criticality_score", ascending=False)


def visualize_attention_weights(
    attention_dict: dict,
    top_k: int = 10,
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Heatmap of mean attention weights per (layer, edge_type).

    Rows   = GNN layers
    Columns = edge relationship types
    Cell   = mean attention weight across all edges
    """
    if not attention_dict:
        return None

    layer_names = sorted(attention_dict.keys())
    edge_type_labels = [f"{et[1]}" for et in EDGE_TYPES]

    matrix = np.zeros((len(layer_names), len(EDGE_TYPES)))
    for i, layer in enumerate(layer_names):
        for j, et in enumerate(EDGE_TYPES):
            entry = attention_dict[layer].get(et, {})
            matrix[i, j] = entry.get("mean_attention", 0.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(EDGE_TYPES)))
    ax.set_xticklabels(edge_type_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=9)

    for i in range(len(layer_names)):
        for j in range(len(EDGE_TYPES)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, label="Mean Attention Weight")
    ax.set_title("HetGAT Attention Weights by Layer × Edge Type", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def run_gnn_pipeline(
    data_path: str = "data/raw/",
    epochs: int = 100,
    hidden_channels: int = 128,
    num_layers: int = 3,
    save_dir: str = "models/",
) -> GNNRiskPredictor:
    """End-to-end pipeline: load data → build graph → train → evaluate → save.

    Steps
    ─────
    1. Load suppliers_500.csv and supply_chain_edges_2000.csv
    2. Build HeteroData object
    3. Add negative edges for link prediction
    4. Train HetGAT (node_classification + link_prediction)
    5. Evaluate and print metrics table
    6. Save model

    Returns fitted GNNRiskPredictor.
    """
    raw = Path(data_path)
    sup_path = raw / "suppliers_500.csv"
    edge_path = raw / "supply_chain_edges_2000.csv"

    if sup_path.exists():
        suppliers_df = pd.read_csv(sup_path)
        logger.info(f"Loaded {len(suppliers_df)} suppliers from {sup_path}")
    else:
        from src.data.pipeline import SupplyChainNetworkGenerator
        logger.info("suppliers_500.csv not found — generating synthetic data.")
        gen = SupplyChainNetworkGenerator(seed=42)
        suppliers_df = gen.generate_suppliers(n=200)

    edges_df = None
    if edge_path.exists():
        edges_df = pd.read_csv(edge_path)
        logger.info(f"Loaded {len(edges_df)} edges from {edge_path}")

    # Build graph
    builder = SupplyChainHeteroGraph(suppliers_df, edges_df, seed=42)
    data = builder.build_hetero_data()
    data = builder.add_synthetic_edges(data, n_negative=5000)

    # Train
    predictor = GNNRiskPredictor(
        hidden_channels=hidden_channels,
        num_layers=num_layers,
    )
    history = predictor.fit(data, epochs=epochs, task="both")

    # Evaluate
    metrics = predictor.evaluate(data)
    scores_df = predictor.predict_risk_scores(data)
    tier_dist = scores_df["risk_tier"].value_counts().to_dict()

    print("\n" + "═" * 50)
    print("   LogisChain AI — HetGAT Evaluation Results")
    print("═" * 50)
    print(f"  Link Pred AUC      : {metrics.get('link_pred_auc', 0):.4f}")
    print(f"  Node Clf Accuracy  : {metrics.get('node_clf_accuracy', 0):.4f}")
    print(f"  Node Clf F1 (macro): {metrics.get('node_clf_f1', 0):.4f}")
    print(f"  Risk tier breakdown: {tier_dist}")
    print("═" * 50)

    # Save
    save_path = str(Path(save_dir) / "gnn_hetgat.pt")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    predictor.save(save_path)
    print(f"  Model saved → {save_path}")

    return predictor


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Backward-compatible stubs (kept from v0.1.0)
# ═══════════════════════════════════════════════════════════════════════════════

class SupplyChainGNN(_StubHetGAT):
    """v0.1.0 GNN stub — use HetGAT for production."""
    pass


class SupplyChainGraphBuilder:
    """v0.1.0 compatibility class."""

    def __init__(self, node_features=10, edge_features=5):
        self.node_features = node_features
        self.edge_features = edge_features

    def build_graph(self, carriers, shipments):
        if not PYG_AVAILABLE:
            return None
        builder = SupplyChainHeteroGraph(carriers, None, seed=42)
        return builder.build_hetero_data()


class GNNTrainer:
    """v0.1.0 training stub — wraps GNNRiskPredictor."""

    def __init__(self, model, lr=1e-3, weight_decay=1e-4, device=None):
        self.predictor = GNNRiskPredictor(device=device)
        self.predictor.model = model

    def fit(self, data, epochs=100):
        if PYG_AVAILABLE and isinstance(data, HeteroData):
            return self.predictor.fit(data, epochs=epochs)
        return []

    def predict(self, data):
        if PYG_AVAILABLE and isinstance(data, HeteroData):
            return self.predictor.predict_risk_scores(data)["risk_score"].values
        return np.zeros(10)


# ═══════════════════════════════════════════════════════════════════════════════
# __main__ — train on synthetic data and print results
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("LogisChain AI — HetGAT Training on Synthetic Data")
    print(f"PyG available: {PYG_AVAILABLE}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Generate synthetic data
    from src.data.pipeline import SupplyChainNetworkGenerator
    gen = SupplyChainNetworkGenerator(seed=42)
    suppliers_df = gen.generate_suppliers(n=150)
    edges_df = gen.generate_edges(suppliers_df, n_edges=600)

    # Build heterogeneous graph
    builder = SupplyChainHeteroGraph(suppliers_df, edges_df, seed=42)
    data = builder.build_hetero_data()
    data = builder.add_synthetic_edges(data, n_negative=2000)

    # Train model
    predictor = GNNRiskPredictor(
        hidden_channels=64, out_channels=64, num_heads=4, num_layers=3, dropout=0.2
    )
    history = predictor.fit(data, epochs=50, task="both", verbose_every=10)

    # Evaluate
    metrics = predictor.evaluate(data)
    print("\n── Final Evaluation ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<30}: {v:.4f}")

    # Risk scores
    scores = predictor.predict_risk_scores(data)
    print(f"\n── Risk Tier Distribution ────────────────────")
    print(scores["risk_tier"].value_counts().to_string())

    # Explain first supplier
    if len(scores) > 0:
        first_id = scores["supplier_id"].iloc[0]
        exp = predictor.explain_risk(first_id, data)
        print(f"\n── Risk Explanation for {first_id} ────────────")
        print(f"  Score: {exp['risk_score']}  Tier: {exp['risk_tier']}")
        if exp["top_risk_drivers"]:
            print("  Top drivers:")
            for d in exp["top_risk_drivers"][:3]:
                print(f"    • {d}")

    # Compute network features
    net_feats = compute_network_features(data)
    if not net_feats.empty:
        print(f"\n── Network Features (top 5 by criticality) ──")
        print(net_feats.head(5)[["node_id", "node_type", "betweenness_centrality",
                                  "pagerank", "node_criticality_score"]].to_string(index=False))

    # Save model
    predictor.save("models/gnn_hetgat_demo.pt")
    print("\nModel saved to models/gnn_hetgat_demo.pt")
