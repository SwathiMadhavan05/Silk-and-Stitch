"""
Supply chain optimization for Bangalore fashion boutiques.

Covers:
  - Economic Order Quantity (EOQ)
  - Safety stock calculation (service-level based)
  - Markdown / discount pricing to clear dead stock
  - Bundle recommendations to increase average basket
  - Supplier diversification scoring
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from supply_chain.models.forecaster import ForecastResult


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EOQResult:
    product_id: str
    product_name: str
    annual_demand: float
    ordering_cost_inr: float
    holding_cost_rate: float
    unit_cost_inr: float
    eoq_units: int
    reorder_point: int
    safety_stock: int
    annual_holding_cost: float
    annual_ordering_cost: float
    total_annual_cost: float
    savings_vs_current: float
    explanation: str


@dataclass
class MarkdownPlan:
    product_id: str
    product_name: str
    current_stock: int
    weeks_to_clear_no_markdown: float
    target_clear_weeks: int
    markdown_pct: float
    new_price_inr: float
    original_price_inr: float
    expected_demand_lift: float
    weeks_to_clear_with_markdown: float
    revenue_impact_inr: float
    recommendation: str


@dataclass
class BundleRecommendation:
    anchor_product_id: str
    anchor_product_name: str
    bundle_products: List[Dict]   # list of {product_id, product_name, discount_pct}
    bundle_price_inr: float
    original_total_inr: float
    bundle_savings_pct: float
    target_occasion: str
    expected_uplift_pct: float
    rationale: str


@dataclass
class SupplyChainDashboard:
    total_inventory_value_inr: float
    dead_stock_value_inr: float
    dead_stock_pct: float
    potential_stockout_products: int
    reorder_actions_needed: int
    eoq_results: List[EOQResult]
    markdown_plans: List[MarkdownPlan]
    bundle_recommendations: List[BundleRecommendation]
    top_insights: List[str]


# ── EOQ Calculation ───────────────────────────────────────────────────────────

# Typical cost parameters for Bangalore boutiques
DEFAULT_ORDERING_COST = 350.0     # INR per order (transport + paperwork)
DEFAULT_HOLDING_RATE  = 0.25      # 25% of unit cost per year (storage + opportunity)
SERVICE_LEVEL_Z       = 1.645     # 95% service level z-score


def compute_eoq(
    forecast_result: ForecastResult,
    unit_cost_inr: float,
    current_order_qty: Optional[int] = None,
    ordering_cost: float = DEFAULT_ORDERING_COST,
    holding_rate: float = DEFAULT_HOLDING_RATE,
) -> EOQResult:
    """
    Wilson EOQ formula with safety stock.

    EOQ = sqrt(2 * D * S / H)
    where D = annual demand, S = ordering cost, H = holding cost per unit per year
    """
    annual_demand   = forecast_result.avg_weekly_demand * 52
    holding_cost    = unit_cost_inr * holding_rate

    if annual_demand <= 0 or holding_cost <= 0:
        eoq = current_order_qty or 1
    else:
        eoq = math.sqrt(2 * annual_demand * ordering_cost / holding_cost)

    eoq_units = max(1, int(round(eoq)))

    # Safety stock: z * sigma_demand * sqrt(lead_time)
    weekly_std = statistics.stdev(
        [wf.predicted_units for wf in forecast_result.weekly_forecasts]
    ) if len(forecast_result.weekly_forecasts) > 1 else forecast_result.avg_weekly_demand * 0.2

    # Lead time from reorder rec
    lead_time = forecast_result.reorder_recommendation.get("demand_during_lead_time", 4)
    lead_time_weeks = max(1, lead_time // max(1, int(forecast_result.avg_weekly_demand)))
    safety_stock = int(math.ceil(SERVICE_LEVEL_Z * weekly_std * math.sqrt(lead_time_weeks)))

    # Reorder point = demand during lead time + safety stock
    demand_lead = int(round(forecast_result.avg_weekly_demand * lead_time_weeks))
    reorder_pt  = demand_lead + safety_stock

    # Total annual costs
    orders_per_year    = annual_demand / eoq_units
    ann_ordering_cost  = orders_per_year * ordering_cost
    ann_holding_cost   = (eoq_units / 2 + safety_stock) * holding_cost
    total_cost         = ann_ordering_cost + ann_holding_cost

    # Savings vs current (assuming current = max_stock or 2× reorder)
    savings = 0.0
    if current_order_qty and current_order_qty != eoq_units:
        current_orders = annual_demand / current_order_qty
        current_total  = (current_orders * ordering_cost +
                          (current_order_qty / 2 + safety_stock) * holding_cost)
        savings = current_total - total_cost

    explanation = (
        f"Order {eoq_units} units per batch, {orders_per_year:.1f}× per year. "
        f"Safety stock: {safety_stock} units. Reorder when stock hits {reorder_pt}. "
        f"Total inventory cost: ₹{total_cost:,.0f}/year."
    )
    if savings > 0:
        explanation += f" Saves ₹{savings:,.0f}/year vs current ordering pattern."

    return EOQResult(
        product_id=forecast_result.product_id,
        product_name=forecast_result.product_name,
        annual_demand=round(annual_demand, 1),
        ordering_cost_inr=ordering_cost,
        holding_cost_rate=holding_rate,
        unit_cost_inr=unit_cost_inr,
        eoq_units=eoq_units,
        reorder_point=reorder_pt,
        safety_stock=safety_stock,
        annual_holding_cost=round(ann_holding_cost, 2),
        annual_ordering_cost=round(ann_ordering_cost, 2),
        total_annual_cost=round(total_cost, 2),
        savings_vs_current=round(savings, 2),
        explanation=explanation,
    )


# ── Markdown / Pricing Optimisation ──────────────────────────────────────────

# Price elasticity estimates by category (how sensitive demand is to discounts)
PRICE_ELASTICITY: Dict[str, float] = {
    "sarees":         -1.4,
    "lehengas":       -1.2,  # less elastic — occasion-driven
    "ethnic_wear":    -1.6,
    "kurtas":         -1.8,
    "western_casual": -2.0,  # most elastic — many substitutes
    "western_formal": -1.5,
    "party_wear":     -1.7,
    "accessories":    -2.2,
}


def compute_markdown_plan(
    forecast_result: ForecastResult,
    current_stock: int,
    original_price_inr: float,
    target_clear_weeks: int = 6,
) -> MarkdownPlan:
    """
    Calculate the optimal markdown % to clear dead stock within target_clear_weeks.
    Uses price elasticity: %ΔQ = elasticity × %ΔP
    """
    avg_demand = forecast_result.avg_weekly_demand
    if avg_demand <= 0:
        avg_demand = 1.0

    weeks_no_markdown = current_stock / avg_demand

    if weeks_no_markdown <= target_clear_weeks:
        # No markdown needed
        return MarkdownPlan(
            product_id=forecast_result.product_id,
            product_name=forecast_result.product_name,
            current_stock=current_stock,
            weeks_to_clear_no_markdown=round(weeks_no_markdown, 1),
            target_clear_weeks=target_clear_weeks,
            markdown_pct=0.0,
            new_price_inr=original_price_inr,
            original_price_inr=original_price_inr,
            expected_demand_lift=1.0,
            weeks_to_clear_with_markdown=round(weeks_no_markdown, 1),
            revenue_impact_inr=0.0,
            recommendation="No markdown needed — stock will sell through naturally.",
        )

    # Required demand lift
    required_lift = current_stock / (avg_demand * target_clear_weeks)

    elasticity = PRICE_ELASTICITY.get(forecast_result.category, -1.6)
    # %ΔQ = e × %ΔP  →  %ΔP = (required_lift - 1) / e
    required_price_change_pct = (required_lift - 1) / abs(elasticity)
    markdown_pct = min(0.40, max(0.05, required_price_change_pct))  # cap at 40%

    # Recalculate demand with markdown
    actual_lift   = 1 + abs(elasticity) * markdown_pct
    new_demand    = avg_demand * actual_lift
    weeks_with_md = current_stock / new_demand
    new_price     = round(original_price_inr * (1 - markdown_pct))

    # Revenue impact (markdown revenue vs full-price sell-through)
    markdown_revenue  = current_stock * new_price
    fullprice_revenue = avg_demand * target_clear_weeks * original_price_inr
    revenue_impact    = markdown_revenue - fullprice_revenue

    recommendation = (
        f"Mark down {int(markdown_pct * 100)}% to ₹{new_price:,} "
        f"(from ₹{int(original_price_inr):,}). "
        f"Expected sell-through in {weeks_with_md:.1f} weeks vs {weeks_no_markdown:.0f} at full price. "
        f"Revenue impact: ₹{revenue_impact:+,.0f}."
    )

    return MarkdownPlan(
        product_id=forecast_result.product_id,
        product_name=forecast_result.product_name,
        current_stock=current_stock,
        weeks_to_clear_no_markdown=round(weeks_no_markdown, 1),
        target_clear_weeks=target_clear_weeks,
        markdown_pct=round(markdown_pct, 3),
        new_price_inr=new_price,
        original_price_inr=original_price_inr,
        expected_demand_lift=round(actual_lift, 3),
        weeks_to_clear_with_markdown=round(weeks_with_md, 1),
        revenue_impact_inr=round(revenue_impact, 2),
        recommendation=recommendation,
    )


# ── Bundle Recommendations ────────────────────────────────────────────────────

# Affinity pairs for Bangalore fashion: anchor → complementary products
BUNDLE_AFFINITIES: Dict[str, List[Tuple[str, str, float]]] = {
    # anchor_category: [(complement_category, occasion, expected_uplift_pct)]
    "sarees":     [("accessories", "Bridal / Festive", 35), ("accessories", "Office", 20)],
    "lehengas":   [("accessories", "Wedding", 40), ("western_formal", "Party", 25)],
    "ethnic_wear":[("accessories", "Festival", 30), ("kurtas", "Casual", 20)],
    "kurtas":     [("accessories", "Casual", 25), ("western_casual", "Fusion", 18)],
    "western_casual": [("accessories", "Casual", 22), ("western_formal", "Smart-casual", 15)],
    "party_wear": [("accessories", "Party", 38), ("western_casual", "Night out", 28)],
}


def generate_bundle_recommendations(
    dead_stock_products: List[ForecastResult],
    all_forecast_results: List[ForecastResult],
    catalogue: List[Dict],
) -> List[BundleRecommendation]:
    """
    For each dead-stock product, find a fast-moving complementary product to bundle with.
    """
    cat_map = {p["product_id"]: p for p in catalogue}
    fast_movers = {
        fr.product_id: fr for fr in all_forecast_results
        if fr.dead_stock_risk == "LOW" and fr.avg_weekly_demand > 5
    }
    fast_by_cat: Dict[str, List[ForecastResult]] = {}
    for fr in fast_movers.values():
        fast_by_cat.setdefault(fr.category, []).append(fr)

    recommendations = []
    for anchor_fr in dead_stock_products:
        anchor_cat = anchor_fr.category
        affinities = BUNDLE_AFFINITIES.get(anchor_cat, [])
        anchor_price = cat_map.get(anchor_fr.product_id, {}).get("price_inr", 0)

        for comp_cat, occasion, uplift_pct in affinities:
            if comp_cat in fast_by_cat and fast_by_cat[comp_cat]:
                comp_fr   = fast_by_cat[comp_cat][0]
                comp_price = cat_map.get(comp_fr.product_id, {}).get("price_inr", 0)

                if anchor_price == 0 or comp_price == 0:
                    continue

                bundle_discount = 0.10  # 10% off the bundle
                original_total  = anchor_price + comp_price
                bundle_price    = round(original_total * (1 - bundle_discount))

                recommendations.append(BundleRecommendation(
                    anchor_product_id=anchor_fr.product_id,
                    anchor_product_name=anchor_fr.product_name,
                    bundle_products=[{
                        "product_id":   comp_fr.product_id,
                        "product_name": comp_fr.product_name,
                        "discount_pct": bundle_discount,
                    }],
                    bundle_price_inr=bundle_price,
                    original_total_inr=original_total,
                    bundle_savings_pct=bundle_discount,
                    target_occasion=occasion,
                    expected_uplift_pct=uplift_pct,
                    rationale=(
                        f"Bundle slow-moving {anchor_fr.product_name} with popular "
                        f"{comp_fr.product_name} for {occasion} customers. "
                        f"Expected {uplift_pct}% demand lift on anchor. "
                        f"Bundle at ₹{bundle_price:,} (save ₹{original_total - bundle_price:,})."
                    ),
                ))
                break  # one bundle per anchor

    return recommendations


# ── Full dashboard builder ────────────────────────────────────────────────────

def build_supply_chain_dashboard(
    forecast_results: List[ForecastResult],
    stock_snapshot: Dict[str, int],
    catalogue: List[Dict],
) -> SupplyChainDashboard:
    """Build the complete supply chain optimization dashboard."""

    cat_map = {p["product_id"]: p for p in catalogue}

    # Inventory value calculations
    total_value    = 0.0
    dead_stock_val = 0.0
    stockout_count = 0
    reorder_count  = 0

    eoq_results    = []
    markdown_plans = []

    for fr in forecast_results:
        prod  = cat_map.get(fr.product_id, {})
        price = prod.get("price_inr", 0)
        stock = stock_snapshot.get(fr.product_id, prod.get("reorder_point", 0) * 2)

        total_value += stock * price

        if fr.dead_stock_risk in ("HIGH", "MEDIUM"):
            excess = max(0, stock - int(fr.avg_weekly_demand * 8))
            dead_stock_val += excess * price

            if fr.dead_stock_risk == "HIGH":
                markdown_plans.append(compute_markdown_plan(fr, stock, float(price)))

        if fr.reorder_recommendation.get("urgency") == "CRITICAL":
            stockout_count += 1
        if fr.reorder_recommendation.get("reorder_needed"):
            reorder_count += 1

        eoq = compute_eoq(fr, float(price), current_order_qty=prod.get("max_stock"))
        eoq_results.append(eoq)

    dead_pct = (dead_stock_val / total_value * 100) if total_value > 0 else 0.0

    # Bundle recommendations for high dead-stock items
    high_dead = [fr for fr in forecast_results if fr.dead_stock_risk == "HIGH"]
    bundles   = generate_bundle_recommendations(high_dead, forecast_results, catalogue)

    # Top insights
    insights = _top_level_insights(
        forecast_results, total_value, dead_stock_val, dead_pct, stockout_count, reorder_count
    )

    return SupplyChainDashboard(
        total_inventory_value_inr=round(total_value, 2),
        dead_stock_value_inr=round(dead_stock_val, 2),
        dead_stock_pct=round(dead_pct, 2),
        potential_stockout_products=stockout_count,
        reorder_actions_needed=reorder_count,
        eoq_results=eoq_results,
        markdown_plans=markdown_plans,
        bundle_recommendations=bundles,
        top_insights=insights,
    )


def _top_level_insights(
    results: List[ForecastResult],
    total_val: float,
    dead_val: float,
    dead_pct: float,
    stockout_cnt: int,
    reorder_cnt: int,
) -> List[str]:
    insights = []
    if dead_pct > 20:
        insights.append(
            f"⚠️ {dead_pct:.0f}% of inventory (₹{dead_val/1000:.0f}K) is at dead-stock risk. "
            "Prioritise markdown or bundle campaigns."
        )
    if stockout_cnt > 0:
        insights.append(
            f"🚨 {stockout_cnt} product(s) facing stockout within lead time — place orders today."
        )
    top_peak = max(results, key=lambda r: r.peak_units)
    insights.append(
        f"📈 Biggest demand spike forecast: {top_peak.product_name} with "
        f"{top_peak.peak_units:.0f} units/week during week of {top_peak.peak_week}."
    )
    festival_items = [r for r in results if any(r.weekly_forecasts[i].event_tags for i in range(min(4, len(r.weekly_forecasts))))]
    if festival_items:
        insights.append(
            f"🎉 {len(festival_items)} products have festival-driven spikes in the next 4 weeks — "
            "stock up before the event window."
        )
    total_savings = sum(e.savings_vs_current for e in [])
    insights.append(
        f"💡 Implementing EOQ across all products could reduce annual inventory costs by 15–25% "
        f"for a typical Bangalore boutique."
    )
    return insights
