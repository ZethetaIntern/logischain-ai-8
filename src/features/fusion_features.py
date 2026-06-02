"""Cross-domain fusion feature engineering: supply chain × financial signals."""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FusionFeatureEngine:
    """Fuses supply chain risk signals with financial metrics to create cross-domain features.

    These fusion features are the core innovation of LogisChain AI — embedding
    logistics network intelligence directly into financial risk models.
    """

    def fuse(
        self,
        shipments_enriched: pd.DataFrame,
        financial: pd.DataFrame,
        carriers: pd.DataFrame,
        join_key_shipment: str = "carrier_id",
        join_key_financial: str = "company_id",
    ) -> pd.DataFrame:
        """Produces the full fusion feature matrix."""
        carrier_stats = self._aggregate_carrier_financials(carriers, shipments_enriched)
        fused = carrier_stats.copy()

        # Merge financial data if possible
        if join_key_financial in financial.columns and join_key_shipment in fused.columns:
            fin_sample = financial.sample(min(len(financial), len(fused)), random_state=42).copy()
            fin_sample[join_key_shipment] = fused[join_key_shipment].values[:len(fin_sample)]
            fused = fused.merge(fin_sample, on=join_key_shipment, how="left")

        fused = self._compute_cross_domain_features(fused)
        logger.info(f"Fusion complete: {fused.shape[1]} features, {len(fused)} rows")
        return fused

    def _aggregate_carrier_financials(
        self, carriers: pd.DataFrame, shipments: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate shipment features back to carrier level."""
        if "carrier_id" not in shipments.columns:
            return carriers.copy()

        agg = shipments.groupby("carrier_id").agg(
            shipment_count=("shipment_id", "count"),
            avg_delay_days=("delay_days", "mean"),
            on_time_rate=("on_time", "mean"),
            total_value_usd=("value_usd", "sum"),
            avg_freight_cost=("freight_cost_usd", "mean"),
            damage_rate=("damage_flag", "mean"),
            avg_transit_days=("actual_transit_days", "mean"),
            port_congestion_avg=("port_congestion_days", "mean") if "port_congestion_days" in shipments.columns else ("shipment_id", "count"),
        ).reset_index()

        result = carriers.merge(agg, on="carrier_id", how="left")
        return result

    def _compute_cross_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Supply-chain risk-adjusted cost of capital
        # Higher operational disruptions → higher cost of capital
        if "on_time_rate" in df.columns and "debt_to_equity" in df.columns:
            operational_risk = 1 - df["on_time_rate"].fillna(0.8)
            leverage_multiplier = 1 + df["debt_to_equity"].fillna(1.0).clip(upper=5)
            df["sc_risk_adjusted_cost_of_capital"] = (
                operational_risk * leverage_multiplier * 0.1
            ).clip(0, 0.5)

        # 2. Logistics disruption → credit impact score
        if "avg_delay_days" in df.columns and "cash_conversion_cycle" in df.columns:
            df["logistics_disruption_credit_impact"] = (
                df["avg_delay_days"].fillna(0) * 0.01
                + df["cash_conversion_cycle"].fillna(60) * 0.005
            ).clip(0, 1)

        # 3. Carrier reliability × payment risk index
        if "on_time_rate" in df.columns and "credit_stress_index" in df.columns:
            df["carrier_reliability_payment_risk"] = (
                (1 - df["on_time_rate"].fillna(0.8)) * df["credit_stress_index"].fillna(0.3)
            )

        # 4. Inventory risk × working capital multiplier
        if "damage_rate" in df.columns and "current_ratio" in df.columns:
            df["inventory_risk_wc_multiplier"] = (
                df["damage_rate"].fillna(0.02)
                * (2.0 - df["current_ratio"].fillna(1.0).clip(0.5, 3.0))
            ).clip(0, 1)

        # 5. Route concentration × credit exposure
        if "supplier_concentration_ratio" in df.columns and "fx_exposure_to_revenue" in df.columns:
            df["route_concentration_credit_exposure"] = (
                df["supplier_concentration_ratio"].fillna(0.5)
                * df["fx_exposure_to_revenue"].fillna(0.1)
            ).clip(0, 1)

        # 6. Demand volatility → CCC impact
        if "demand_volatility_30d" in df.columns and "cash_conversion_cycle" in df.columns:
            vol_norm = df["demand_volatility_30d"].fillna(0) / (
                df["demand_volatility_30d"].fillna(0).max() + 1e-8
            )
            ccc_norm = df["cash_conversion_cycle"].fillna(60) / 180
            df["demand_vol_ccc_impact"] = (vol_norm * ccc_norm).clip(0, 1)

        # 7. Disruption probability × default correlation
        if "carrier_failure" in df.columns and "default_flag" in df.columns:
            df["disruption_default_correlation"] = (
                df["carrier_failure"].fillna(0) * df["default_flag"].fillna(0)
            )

        # 8. Network centrality × counterparty risk
        if "betweenness_centrality" in df.columns and "credit_rating_numeric" in df.columns:
            df["network_centrality_cp_risk"] = (
                df["betweenness_centrality"].fillna(0)
                * df["credit_rating_numeric"].fillna(5) / 9
            ).clip(0, 1)

        # 9. LogisChain composite risk score (flagship feature)
        components = []
        weights = []
        feature_weight_map = {
            "sc_risk_adjusted_cost_of_capital": 0.20,
            "logistics_disruption_credit_impact": 0.20,
            "carrier_reliability_payment_risk": 0.15,
            "inventory_risk_wc_multiplier": 0.10,
            "route_concentration_credit_exposure": 0.10,
            "demand_vol_ccc_impact": 0.10,
            "network_centrality_cp_risk": 0.15,
        }
        for feat, w in feature_weight_map.items():
            if feat in df.columns:
                components.append(df[feat].fillna(0) * w)
                weights.append(w)

        if components:
            total_w = sum(weights)
            df["logischain_composite_risk_score"] = sum(components) / total_w

        # 10. Supply Chain Financial Stress Index
        stress_features = [c for c in [
            "logischain_composite_risk_score", "credit_stress_index", "altman_distress_flag",
        ] if c in df.columns]
        if stress_features:
            df["sc_financial_stress_index"] = df[stress_features].fillna(0).mean(axis=1)

        return df


class FeaturePipeline:
    """Orchestrates full feature pipeline: supply chain → financial → fusion."""

    def __init__(self):
        from src.features.supply_chain_features import (
            NetworkFeatureExtractor,
            ShipmentFeatureExtractor,
            DemandFeatureExtractor,
            DisruptionFeatureExtractor,
        )
        from src.features.financial_features import (
            TradeFinanceFeatureExtractor,
            WorkingCapitalFeatureExtractor,
            CreditRiskFeatureExtractor,
        )
        self.network_fe = NetworkFeatureExtractor()
        self.shipment_fe = ShipmentFeatureExtractor()
        self.demand_fe = DemandFeatureExtractor()
        self.disruption_fe = DisruptionFeatureExtractor()
        self.tf_fe = TradeFinanceFeatureExtractor()
        self.wc_fe = WorkingCapitalFeatureExtractor()
        self.cr_fe = CreditRiskFeatureExtractor()
        self.fusion = FusionFeatureEngine()

    def run(
        self,
        carriers: pd.DataFrame,
        shipments: pd.DataFrame,
        financial: pd.DataFrame,
    ) -> pd.DataFrame:
        logger.info("Running full feature pipeline...")

        # Supply chain features
        shipments = self.shipment_fe.extract(shipments)
        shipments = self.disruption_fe.extract(shipments, carriers)
        if "ship_date" in shipments.columns and "value_usd" in shipments.columns:
            shipments = self.demand_fe.extract(shipments)

        # Financial features
        financial = self.tf_fe.extract(financial)
        financial = self.wc_fe.extract(financial)
        financial = self.cr_fe.extract(financial)

        # Network features → merge to carriers
        G = self.network_fe.build_graph(shipments)
        node_feats = self.network_fe.extract_node_features(G)
        if not node_feats.empty and "origin_country" in shipments.columns:
            shipments = shipments.merge(
                node_feats.rename(columns={"node": "origin_country"}),
                on="origin_country",
                how="left",
            )

        # Fusion
        fused = self.fusion.fuse(shipments, financial, carriers)
        logger.info(f"Feature pipeline complete: {fused.shape}")
        return fused
