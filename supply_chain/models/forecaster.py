"""
Demand forecasting engine for Bangalore fashion boutiques.

Implements three complementary models without heavy ML dependencies:
  1. Weighted Moving Average (WMA)   — fast, handles recency bias
  2. Holt-Winters (Triple Exponential Smoothing) — captures trend + seasonality
  3. Event-Adjusted Forecast         — overlays festival/wedding-season multipliers

All models output a ForecastResult with confidence intervals and dead-stock risk.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from supply_chain.data.bangalore_data import (
    BANGALORE_EVENTS,
    generate_historical_sales,
    get_product_catalogue,
    _event_multiplier,
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class WeeklyForecast:
    week_start: str
    predicted_units: float
    lower_bound: float
    upper_bound: float
    event_tags: List[str]
    confidence: float   # 0.0–1.0


@dataclass
class ForecastResult:
    product_id: str
    product_name: str
    category: str
    model_used: str
    forecast_horizon_weeks: int
    weekly_forecasts: List[WeeklyForecast]
    avg_weekly_demand: float
    peak_week: str
    peak_units: float
    dead_stock_risk: str        # LOW / MEDIUM / HIGH
    dead_stock_risk_score: float  # 0.0–1.0
    reorder_recommendation: Dict
    insights: List[str]


@dataclass
class InventoryAlert:
    product_id: str
    product_name: str
    alert_type: str    # STOCKOUT_RISK / DEAD_STOCK / REORDER_NOW / OVERSTOCK
    severity: str      # LOW / MEDIUM / HIGH / CRITICAL
    message: str
    recommended_action: str
    units_to_order: Optional[int] = None
    units_to_discount: Optional[int] = None


# ── Helper utilities ──────────────────────────────────────────────────────────

def _get_event_tags(category: str, target_date: date) -> List[str]:
    tags = []
    m, d = target_date.month, target_date.day
    for ev_month, ev_start, ev_end, label, boosts in BANGALORE_EVENTS:
        if m == ev_month and ev_start <= d <= ev_end and category in boosts:
            tags.append(label)
    return tags


def _series_for_product(records: List[Dict], product_id: str) -> List[float]:
    """Extract ordered units_sold series for a product."""
    product_records = [r for r in records if r["product_id"] == product_id]
    product_records.sort(key=lambda r: r["week_start"])
    return [float(r["units_sold"]) for r in product_records]


def _confidence_from_cv(series: List[float]) -> float:
    """Higher coefficient of variation → lower confidence."""
    if not series or statistics.mean(series) == 0:
        return 0.5
    cv = statistics.stdev(series) / statistics.mean(series) if len(series) > 1 else 0.5
    return round(max(0.3, min(0.95, 1.0 - cv * 0.6)), 3)


# ── Model 1: Weighted Moving Average ─────────────────────────────────────────

def _weighted_moving_average(series: List[float], window: int = 8) -> float:
    """WMA giving more weight to recent weeks."""
    tail = series[-window:]
    weights = list(range(1, len(tail) + 1))
    wma = sum(v * w for v, w in zip(tail, weights)) / sum(weights)
    return max(0.0, wma)


# ── Model 2: Holt-Winters (additive, 52-week seasonality) ────────────────────

def _holt_winters(
    series: List[float],
    alpha: float = 0.3,   # level smoothing
    beta:  float = 0.1,   # trend smoothing
    gamma: float = 0.4,   # seasonal smoothing
    period: int  = 52,    # annual cycle
    horizon: int = 12,
) -> List[float]:
    """
    Additive Holt-Winters triple exponential smoothing.
    Returns `horizon` future predictions.
    """
    n = len(series)
    if n < period * 2:
        # Fallback to WMA if insufficient history
        base = _weighted_moving_average(series)
        return [base] * horizon

    # Initialise
    L = statistics.mean(series[:period])
    T = (statistics.mean(series[period:2*period]) - L) / period
    S = [series[i] - L for i in range(period)]

    # Smoothing loop
    for i in range(1, n):
        prev_L = L
        L = alpha * (series[i] - S[i % period]) + (1 - alpha) * (prev_L + T)
        T = beta * (L - prev_L) + (1 - beta) * T
        S[i % period] = gamma * (series[i] - L) + (1 - gamma) * S[i % period]

    # Forecast
    forecasts = []
    for h in range(1, horizon + 1):
        forecast = L + h * T + S[(n + h - 1) % period]
        forecasts.append(max(0.0, forecast))
    return forecasts


# ── Model 3: Event-Adjusted Forecast ─────────────────────────────────────────

def _event_adjusted_forecast(
    base_forecasts: List[float],
    category: str,
    start_date: date,
) -> Tuple[List[float], List[List[str]]]:
    """Apply Bangalore event multipliers on top of base forecasts."""
    adjusted = []
    event_tags_per_week = []
    for i, base in enumerate(base_forecasts):
        week_start = start_date + timedelta(weeks=i)
        multiplier = max(
            _event_multiplier(category, week_start),
            _event_multiplier(category, week_start + timedelta(days=3)),
            _event_multiplier(category, week_start + timedelta(days=6)),
        )
        adjusted.append(round(base * multiplier, 2))
        event_tags_per_week.append(_get_event_tags(category, week_start))
    return adjusted, event_tags_per_week


# ── Dead stock risk calculator ────────────────────────────────────────────────

def _dead_stock_risk(
    current_stock: int,
    avg_weekly_demand: float,
    lead_time_weeks: int,
    reorder_point: int,
    upcoming_forecasts: List[float],
) -> Tuple[str, float]:
    """
    Estimate dead stock risk based on weeks-of-supply and upcoming demand.
    Returns (risk_label, risk_score 0-1).
    """
    if avg_weekly_demand <= 0:
        return "HIGH", 0.9

    weeks_of_supply = current_stock / avg_weekly_demand
    demand_next_4w  = sum(upcoming_forecasts[:4]) if upcoming_forecasts else avg_weekly_demand * 4

    # If projected demand in next 4 weeks << current stock, dead stock risk is high
    demand_ratio = demand_next_4w / max(current_stock, 1)

    if weeks_of_supply > 20 and demand_ratio < 0.3:
        return "HIGH", min(0.95, 0.6 + (weeks_of_supply / 52) * 0.35)
    elif weeks_of_supply > 10 or demand_ratio < 0.5:
        return "MEDIUM", 0.40 + (1 - demand_ratio) * 0.25
    else:
        return "LOW", max(0.05, 0.25 - demand_ratio * 0.15)


# ── Reorder recommendation ────────────────────────────────────────────────────

def _reorder_recommendation(
    current_stock: int,
    forecasts: List[float],
    lead_time_weeks: int,
    reorder_point: int,
    max_stock: int,
) -> Dict:
    """
    Calculate Economic Order Quantity (simplified) and reorder timing.
    """
    # Demand during lead time
    demand_during_lead = sum(forecasts[:lead_time_weeks]) if forecasts else 0
    safety_stock = demand_during_lead * 0.25  # 25% safety buffer

    reorder_needed = current_stock <= (reorder_point + safety_stock)
    order_qty = max(0, int(math.ceil(max_stock - current_stock)))

    # Weeks until stockout
    cumulative = 0
    weeks_to_stockout = None
    for i, f in enumerate(forecasts):
        cumulative += f
        if cumulative >= current_stock:
            weeks_to_stockout = i + 1
            break

    return {
        "reorder_needed":      reorder_needed,
        "suggested_order_qty": order_qty,
        "weeks_to_stockout":   weeks_to_stockout,
        "safety_stock_units":  int(round(safety_stock)),
        "demand_during_lead_time": int(round(demand_during_lead)),
        "urgency": "CRITICAL" if weeks_to_stockout and weeks_to_stockout <= lead_time_weeks
                   else "HIGH" if reorder_needed else "NORMAL",
    }


# ── Main forecast function ────────────────────────────────────────────────────

def forecast_product(
    product_id: str,
    horizon_weeks: int = 12,
    current_stock: Optional[int] = None,
    historical_records: Optional[List[Dict]] = None,
) -> ForecastResult:
    """
    Generate a full demand forecast for a single product.

    Args:
        product_id:           e.g. "P001"
        horizon_weeks:        how many weeks ahead to forecast (default 12)
        current_stock:        current inventory level (if None, uses reorder_point)
        historical_records:   pre-generated records (generated if None)

    Returns:
        ForecastResult with weekly predictions, risk assessment, and reorder advice.
    """
    if historical_records is None:
        historical_records = generate_historical_sales()

    catalogue = {p["product_id"]: p for p in get_product_catalogue()}
    if product_id not in catalogue:
        raise ValueError(f"Unknown product_id: {product_id}")

    prod = catalogue[product_id]
    series = _series_for_product(historical_records, product_id)

    if len(series) < 4:
        raise ValueError(f"Insufficient history for {product_id}")

    # Run Holt-Winters as primary model
    hw_forecasts = _holt_winters(series, horizon=horizon_weeks)

    # Event-adjust the forecasts
    start_date = date.today() + timedelta(weeks=1)
    adjusted_forecasts, event_tags = _event_adjusted_forecast(
        hw_forecasts, prod["category"], start_date
    )

    # Compute residuals for confidence intervals
    residuals = []
    for i in range(4, len(series)):
        wma_pred = _weighted_moving_average(series[:i])
        residuals.append(abs(series[i] - wma_pred))
    std_dev = statistics.stdev(residuals) if len(residuals) > 1 else statistics.mean(series) * 0.2

    confidence = _confidence_from_cv(series)

    weekly_forecasts = []
    for i, (units, tags) in enumerate(zip(adjusted_forecasts, event_tags)):
        ws = start_date + timedelta(weeks=i)
        lower = max(0.0, units - 1.645 * std_dev)
        upper = units + 1.645 * std_dev
        weekly_forecasts.append(WeeklyForecast(
            week_start=ws.isoformat(),
            predicted_units=round(units, 2),
            lower_bound=round(lower, 2),
            upper_bound=round(upper, 2),
            event_tags=tags,
            confidence=confidence,
        ))

    avg_demand = round(statistics.mean(adjusted_forecasts), 2)
    peak_idx   = adjusted_forecasts.index(max(adjusted_forecasts))
    peak_wf    = weekly_forecasts[peak_idx]

    stock = current_stock if current_stock is not None else prod["reorder_point"] * 2
    dead_label, dead_score = _dead_stock_risk(
        stock, avg_demand,
        prod["lead_time_weeks"], prod["reorder_point"],
        adjusted_forecasts,
    )
    reorder_rec = _reorder_recommendation(
        stock, adjusted_forecasts,
        prod["lead_time_weeks"], prod["reorder_point"], prod["max_stock"]
    )

    # Generate human-readable insights
    insights = _generate_insights(prod, series, adjusted_forecasts, event_tags, dead_label, reorder_rec)

    return ForecastResult(
        product_id=product_id,
        product_name=prod["product_name"],
        category=prod["category"],
        model_used="Holt-Winters + Event Calendar",
        forecast_horizon_weeks=horizon_weeks,
        weekly_forecasts=weekly_forecasts,
        avg_weekly_demand=avg_demand,
        peak_week=peak_wf.week_start,
        peak_units=round(peak_wf.predicted_units, 2),
        dead_stock_risk=dead_label,
        dead_stock_risk_score=round(dead_score, 3),
        reorder_recommendation=reorder_rec,
        insights=insights,
    )


def _generate_insights(
    prod: Dict,
    series: List[float],
    forecasts: List[float],
    event_tags: List[List[str]],
    dead_risk: str,
    reorder: Dict,
) -> List[str]:
    insights = []
    all_tags = [t for week in event_tags for t in week]

    if all_tags:
        top_event = max(set(all_tags), key=all_tags.count)
        insights.append(
            f"Demand spike expected during {top_event} — "
            f"stock up {int(round(max(forecasts)))} units at least {prod['lead_time_weeks']} weeks ahead."
        )

    recent_avg  = statistics.mean(series[-4:])
    earlier_avg = statistics.mean(series[-16:-4]) if len(series) >= 16 else statistics.mean(series)
    if earlier_avg > 0:
        trend_pct = (recent_avg - earlier_avg) / earlier_avg * 100
        if trend_pct > 15:
            insights.append(f"Demand trending UP {trend_pct:.0f}% vs 3 months ago — consider expanding stock range.")
        elif trend_pct < -15:
            insights.append(f"Demand trending DOWN {abs(trend_pct):.0f}% vs 3 months ago — consider promotions or markdowns.")

    if dead_risk == "HIGH":
        insights.append(
            f"HIGH dead-stock risk: current stock may not sell through in next 12 weeks. "
            "Consider 15–25% discount bundle or cross-sell with accessories."
        )
    elif dead_risk == "MEDIUM":
        insights.append("Moderate overstock risk — monitor closely and plan a flash sale if stock doesn't move in 4 weeks.")

    if reorder["urgency"] == "CRITICAL":
        insights.append(
            f"CRITICAL: Stockout in ~{reorder['weeks_to_stockout']} week(s)! "
            f"Order {reorder['suggested_order_qty']} units immediately (lead time: {prod['lead_time_weeks']}w)."
        )
    elif reorder["urgency"] == "HIGH":
        insights.append(
            f"Reorder soon: stock will fall below safety threshold. "
            f"Suggested order: {reorder['suggested_order_qty']} units."
        )

    if not insights:
        insights.append("Stock levels and demand are well-balanced. No immediate action required.")

    return insights


# ── Portfolio-level analysis ──────────────────────────────────────────────────

def forecast_all_products(
    horizon_weeks: int = 12,
    stock_snapshot: Optional[Dict[str, int]] = None,
) -> List[ForecastResult]:
    """Forecast all 20 products. stock_snapshot maps product_id → current_stock."""
    records = generate_historical_sales()
    catalogue = get_product_catalogue()
    results = []
    for prod in catalogue:
        pid = prod["product_id"]
        stock = stock_snapshot.get(pid) if stock_snapshot else None
        result = forecast_product(pid, horizon_weeks, stock, records)
        results.append(result)
    return results


def generate_inventory_alerts(
    forecast_results: List[ForecastResult],
    stock_snapshot: Optional[Dict[str, int]] = None,
) -> List[InventoryAlert]:
    """Convert forecast results into actionable inventory alerts."""
    alerts = []
    catalogue = {p["product_id"]: p for p in get_product_catalogue()}

    for fr in forecast_results:
        prod = catalogue[fr.product_id]
        stock = (stock_snapshot or {}).get(fr.product_id, prod["reorder_point"] * 2)
        rec   = fr.reorder_recommendation

        if rec["urgency"] == "CRITICAL":
            alerts.append(InventoryAlert(
                product_id=fr.product_id,
                product_name=fr.product_name,
                alert_type="STOCKOUT_RISK",
                severity="CRITICAL",
                message=f"Stockout in {rec['weeks_to_stockout']} week(s). Lead time is {prod['lead_time_weeks']}w.",
                recommended_action=f"Place order NOW for {rec['suggested_order_qty']} units.",
                units_to_order=rec["suggested_order_qty"],
            ))
        elif rec["urgency"] == "HIGH":
            alerts.append(InventoryAlert(
                product_id=fr.product_id,
                product_name=fr.product_name,
                alert_type="REORDER_NOW",
                severity="HIGH",
                message=f"Stock below reorder point. Avg weekly demand: {fr.avg_weekly_demand:.1f}.",
                recommended_action=f"Order {rec['suggested_order_qty']} units within 1 week.",
                units_to_order=rec["suggested_order_qty"],
            ))

        if fr.dead_stock_risk == "HIGH":
            excess = max(0, stock - int(fr.avg_weekly_demand * 8))
            alerts.append(InventoryAlert(
                product_id=fr.product_id,
                product_name=fr.product_name,
                alert_type="DEAD_STOCK",
                severity="HIGH",
                message=f"Dead stock risk {fr.dead_stock_risk_score:.0%}. ~{excess} excess units.",
                recommended_action="Run 20% discount or bundle with fast-moving accessory.",
                units_to_discount=excess,
            ))
        elif fr.dead_stock_risk == "MEDIUM":
            alerts.append(InventoryAlert(
                product_id=fr.product_id,
                product_name=fr.product_name,
                alert_type="OVERSTOCK",
                severity="MEDIUM",
                message=f"Moderate overstock risk. Monitor sell-through rate.",
                recommended_action="Plan flash sale if no movement in 3 weeks.",
            ))

    # Sort: CRITICAL first, then HIGH, MEDIUM
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    alerts.sort(key=lambda a: severity_order.get(a.severity, 4))
    return alerts
