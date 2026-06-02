"""Supply chain feature engineering: network, shipment, demand, disruption features."""
import logging
from typing import Optional, List

import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


class NetworkFeatureExtractor:
    """Extracts graph-theoretic features from the supply chain network."""

    def __init__(self):
        self.graph: Optional[nx.DiGraph] = None

    def build_graph(self, shipments: pd.DataFrame) -> nx.DiGraph:
        G = nx.DiGraph()
        if "origin_country" not in shipments.columns:
            return G
        for _, row in shipments.iterrows():
            src = row.get("origin_country", "UNK")
            dst = row.get("destination_country", "UNK")
            val = float(row.get("value_usd", 1.0))
            if G.has_edge(src, dst):
                G[src][dst]["weight"] += val
                G[src][dst]["count"] += 1
            else:
                G.add_edge(src, dst, weight=val, count=1)
        self.graph = G
        logger.info(f"Built supply chain graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def extract_node_features(self, G: Optional[nx.DiGraph] = None) -> pd.DataFrame:
        G = G or self.graph
        if G is None or G.number_of_nodes() == 0:
            return pd.DataFrame()

        degree_c = nx.degree_centrality(G)
        between_c = nx.betweenness_centrality(G, weight="weight")
        eigen_c = {}
        try:
            eigen_c = nx.eigenvector_centrality(G, max_iter=500, weight="weight")
        except Exception:
            eigen_c = {n: 0.0 for n in G.nodes()}
        try:
            pagerank = nx.pagerank(G, weight="weight")
        except Exception:
            pagerank = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}

        hub_score, authority_score = {}, {}
        try:
            hub_score, authority_score = nx.hits(G, max_iter=100)
        except Exception:
            hub_score = {n: 0.0 for n in G.nodes()}
            authority_score = {n: 0.0 for n in G.nodes()}

        undirected = G.to_undirected()
        clustering = nx.clustering(undirected)

        records = []
        for node in G.nodes():
            records.append(
                {
                    "node": node,
                    "degree_centrality": degree_c.get(node, 0),
                    "betweenness_centrality": between_c.get(node, 0),
                    "eigenvector_centrality": eigen_c.get(node, 0),
                    "pagerank": pagerank.get(node, 0),
                    "hub_score": hub_score.get(node, 0),
                    "authority_score": authority_score.get(node, 0),
                    "clustering_coefficient": clustering.get(node, 0),
                    "in_degree": G.in_degree(node),
                    "out_degree": G.out_degree(node),
                    "weighted_in_degree": sum(
                        d.get("weight", 0) for _, _, d in G.in_edges(node, data=True)
                    ),
                    "weighted_out_degree": sum(
                        d.get("weight", 0) for _, _, d in G.out_edges(node, data=True)
                    ),
                }
            )
        return pd.DataFrame(records)

    def get_network_stats(self, G: Optional[nx.DiGraph] = None) -> dict:
        G = G or self.graph
        if G is None:
            return {}
        stats = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_dag": nx.is_directed_acyclic_graph(G),
        }
        try:
            stats["avg_shortest_path"] = nx.average_shortest_path_length(G)
        except Exception:
            stats["avg_shortest_path"] = -1
        return stats


class ShipmentFeatureExtractor:
    """Feature engineering from shipment-level records."""

    def extract(self, shipments: pd.DataFrame) -> pd.DataFrame:
        df = shipments.copy()

        # Delay ratio
        if "actual_transit_days" in df.columns and "planned_transit_days" in df.columns:
            df["delay_ratio"] = (
                df["actual_transit_days"] - df["planned_transit_days"]
            ) / (df["planned_transit_days"].clip(lower=1))
            df["delay_ratio"] = df["delay_ratio"].clip(lower=-1, upper=5)

        # Cost efficiency
        if "freight_cost_usd" in df.columns and "weight_kg" in df.columns:
            df["cost_per_kg"] = df["freight_cost_usd"] / df["weight_kg"].clip(lower=0.01)

        # Insurance rate
        if "insurance_usd" in df.columns and "value_usd" in df.columns:
            df["insurance_rate"] = df["insurance_usd"] / df["value_usd"].clip(lower=1)

        # Volume-weight ratio
        if "volume_cbm" in df.columns and "weight_kg" in df.columns:
            df["volume_weight_ratio"] = df["volume_cbm"] / df["weight_kg"].clip(lower=0.01)

        # Port congestion + customs as share of transit
        if "port_congestion_days" in df.columns and "actual_transit_days" in df.columns:
            df["port_congestion_share"] = (
                df["port_congestion_days"] / df["actual_transit_days"].clip(lower=1)
            )

        return df

    def carrier_reliability_stats(self, shipments: pd.DataFrame) -> pd.DataFrame:
        if "carrier_id" not in shipments.columns:
            return pd.DataFrame()
        stats = (
            shipments.groupby("carrier_id")
            .agg(
                total_shipments=("shipment_id", "count"),
                on_time_rate=("on_time", "mean"),
                avg_delay=("delay_days", "mean"),
                damage_rate=("damage_flag", "mean"),
                avg_transit=("actual_transit_days", "mean"),
                avg_cost=("freight_cost_usd", "mean"),
                total_value=("value_usd", "sum"),
            )
            .reset_index()
        )
        # Reliability score: composite
        stats["route_reliability_score"] = (
            0.5 * stats["on_time_rate"]
            + 0.3 * (1 - stats["damage_rate"])
            + 0.2 * (1 / (1 + stats["avg_delay"].clip(lower=0)))
        ).clip(0, 1)
        return stats


class DemandFeatureExtractor:
    """Feature engineering for demand forecasting and inventory signals."""

    def extract(
        self,
        df: pd.DataFrame,
        date_col: str = "ship_date",
        value_col: str = "value_usd",
    ) -> pd.DataFrame:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

        # Rolling demand stats
        for w in [7, 30, 90]:
            df[f"demand_roll_mean_{w}d"] = (
                df[value_col].rolling(w, min_periods=1).mean()
            )
            df[f"demand_roll_std_{w}d"] = (
                df[value_col].rolling(w, min_periods=1).std().fillna(0)
            )

        # Demand volatility
        df["demand_volatility_30d"] = df[value_col].rolling(30, min_periods=5).std().fillna(0)
        df["demand_volatility_90d"] = df[value_col].rolling(90, min_periods=10).std().fillna(0)

        # Trend slope (linear regression over 30-day window)
        def rolling_slope(series, window=30):
            slopes = []
            arr = series.values
            for i in range(len(arr)):
                start = max(0, i - window + 1)
                chunk = arr[start : i + 1]
                if len(chunk) < 3:
                    slopes.append(0.0)
                else:
                    x = np.arange(len(chunk))
                    slope = np.polyfit(x, chunk, 1)[0]
                    slopes.append(float(slope))
            return slopes

        df["demand_trend_slope"] = rolling_slope(df[value_col])

        # Seasonal index (month / global average)
        if date_col in df.columns:
            monthly_avg = df.groupby(df[date_col].dt.month)[value_col].transform("mean")
            global_avg = df[value_col].mean()
            df["seasonal_index"] = monthly_avg / (global_avg + 1e-8)

        return df


class DisruptionFeatureExtractor:
    """Proxy disruption risk features derived from shipment patterns."""

    def extract(self, shipments: pd.DataFrame, carriers: pd.DataFrame) -> pd.DataFrame:
        df = shipments.copy()

        # Supplier concentration per destination
        if "destination_country" in df.columns and "carrier_id" in df.columns:
            n_carriers = (
                df.groupby("destination_country")["carrier_id"]
                .nunique()
                .rename("n_carriers_per_dest")
            )
            df = df.merge(
                n_carriers.reset_index(), on="destination_country", how="left"
            )
            df["supplier_concentration_ratio"] = 1.0 / df["n_carriers_per_dest"].clip(lower=1)

        # Port congestion risk index (normalized)
        if "port_congestion_days" in df.columns:
            max_cong = df["port_congestion_days"].max()
            df["port_congestion_index"] = df["port_congestion_days"] / (max_cong + 1)

        # Route single-source dependency
        if "origin_country" in df.columns and "destination_country" in df.columns:
            route_carrier_count = (
                df.groupby(["origin_country", "destination_country"])["carrier_id"]
                .nunique()
                .rename("route_carrier_count")
            )
            df = df.merge(
                route_carrier_count.reset_index(),
                on=["origin_country", "destination_country"],
                how="left",
            )
            df["single_source_dependency"] = (df["route_carrier_count"] == 1).astype(int)

        return df
