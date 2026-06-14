"""
ml/explainability/shap_explainer.py

SHAP-based explainability for outfit recommendations and demand forecasts.
Explains WHY the system made each recommendation in plain English.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ShapExplanation:
    item_id: str
    item_name: str
    overall_score: float
    feature_contributions: Dict[str, float]
    top_reasons: List[str]
    warning_reasons: List[str]
    plain_english: str


def explain_outfit_item(
    item: Dict,
    customer: Dict,
    current_outfit: List[Dict],
) -> ShapExplanation:
    """
    Explain why an item is (or isn't) a good fit using SHAP-style attribution.
    Each feature gets a contribution score showing its impact on the recommendation.
    """
    contributions = {}
    reasons = []
    warnings = []

    # ── Style match contribution ──────────────────────────────────────────────
    pref_styles = set(customer.get("preferred_styles", []))
    item_styles = set(item.get("style_tags", []))
    style_overlap = len(pref_styles & item_styles) / max(len(pref_styles), 1)
    contributions["style_match"] = round(style_overlap * 0.30, 4)
    if style_overlap > 0.5:
        matching = list(pref_styles & item_styles)[:2]
        reasons.append(f"Matches your {', '.join(matching)} style preference")
    elif style_overlap == 0:
        warnings.append(f"Doesn't match your preferred styles ({', '.join(list(pref_styles)[:2])})")

    # ── Occasion match ────────────────────────────────────────────────────────
    occasion = customer.get("occasion", "casual")
    OCCASION_STYLE_MAP = {
        "office":  ["formal", "classic", "office", "minimalist"],
        "wedding": ["formal", "glamorous", "classic", "feminine"],
        "casual":  ["casual", "boho", "retro", "classic"],
        "party":   ["party", "glamorous", "evening", "edgy"],
        "outdoor": ["casual", "sporty", "boho"],
    }
    occ_styles = set(OCCASION_STYLE_MAP.get(occasion, []))
    occ_match = len(item_styles & occ_styles) / max(len(occ_styles), 1)
    contributions["occasion_fit"] = round(occ_match * 0.20, 4)
    if occ_match > 0.4:
        reasons.append(f"Well suited for {occasion}")
    else:
        warnings.append(f"May not be ideal for {occasion}")

    # ── Color match contribution ──────────────────────────────────────────────
    color_prefs = [c.lower() for c in customer.get("color_preferences", [])]
    item_color  = item.get("color", "").lower()
    color_match = 1.0 if item_color in color_prefs else 0.0

    COLOR_COMPAT = {
        "black": ["white", "grey", "navy", "red", "gold", "silver"],
        "white": ["black", "navy", "beige", "gold", "silver"],
        "navy":  ["white", "beige", "camel", "gold"],
        "beige": ["white", "navy", "brown", "camel"],
    }
    if not color_match and color_prefs:
        for pref in color_prefs:
            if item_color in COLOR_COMPAT.get(pref, []):
                color_match = 0.6
                break
    contributions["color_preference"] = round(color_match * 0.15, 4)
    if color_match >= 1.0:
        reasons.append(f"Exactly matches your color preference ({item_color})")
    elif color_match > 0:
        reasons.append(f"{item_color} complements your preferred colors")
    elif color_prefs:
        warnings.append(f"Color ({item_color}) doesn't match your preferences")

    # ── Season match ──────────────────────────────────────────────────────────
    season = customer.get("season", "spring")
    item_seasons = item.get("season", [])
    season_match = 1.0 if season in item_seasons else 0.0
    contributions["seasonal_fit"] = round(season_match * 0.10, 4)
    if season_match:
        reasons.append(f"Appropriate for {season}")
    else:
        warnings.append(f"Not ideal for {season} — designed for {', '.join(item_seasons)}")

    # ── Budget contribution ───────────────────────────────────────────────────
    budget_remaining = customer.get("budget_remaining",
                       customer.get("budget", 500))
    item_price = item.get("price", 0)
    if item_price <= budget_remaining:
        budget_score = 1.0 - (item_price / max(budget_remaining, 1)) * 0.3
        contributions["within_budget"] = round(budget_score * 0.15, 4)
        reasons.append(f"Within budget (₹{item_price:.0f} / ${item_price:.0f}, ₹{budget_remaining:.0f} remaining)")
    else:
        over = item_price - budget_remaining
        contributions["within_budget"] = round(-0.10, 4)
        warnings.append(f"Over budget by ₹{over:.0f} / ${over:.0f}")

    # ── Outfit harmony contribution ───────────────────────────────────────────
    harmony_score = 0.0
    if current_outfit:
        outfit_styles = set()
        outfit_colors = []
        for existing in current_outfit:
            outfit_styles.update(existing.get("style_tags", []))
            outfit_colors.append(existing.get("color", "").lower())

        style_compat = len(item_styles & outfit_styles) / max(len(outfit_styles), 1)
        harmony_score = style_compat * 0.5

        for ec in outfit_colors:
            if item_color in COLOR_COMPAT.get(ec, [ec]):
                harmony_score += 0.1

        harmony_score = min(harmony_score, 1.0)
        if harmony_score > 0.5:
            reasons.append("Complements the existing outfit well")
        elif harmony_score < 0.2 and current_outfit:
            warnings.append("May clash with existing outfit items")

    contributions["outfit_harmony"] = round(harmony_score * 0.10, 4)

    # ── Disliked items check ──────────────────────────────────────────────────
    disliked = customer.get("disliked_items", [])
    if item.get("name") in disliked:
        contributions["not_disliked"] = -0.50
        warnings.append(f"This item is in your disliked list")
    else:
        contributions["not_disliked"] = 0.0

    overall = sum(contributions.values())
    overall = round(max(0.001, min(overall, 0.999)), 4)

    # Build plain English explanation
    if overall > 0.6:
        sentiment = "Strong recommendation"
    elif overall > 0.4:
        sentiment = "Good option"
    elif overall > 0.2:
        sentiment = "Possible choice"
    else:
        sentiment = "Not recommended"

    plain = f"{sentiment} (score: {overall:.2f}). "
    if reasons:
        plain += "Pros: " + "; ".join(reasons[:3]) + ". "
    if warnings:
        plain += "Concerns: " + "; ".join(warnings[:2]) + "."

    return ShapExplanation(
        item_id=item.get("id", ""),
        item_name=item.get("name", ""),
        overall_score=overall,
        feature_contributions=contributions,
        top_reasons=reasons[:3],
        warning_reasons=warnings[:2],
        plain_english=plain,
    )


def explain_demand_forecast(
    item_id: str,
    item_name: str,
    category: str,
    predictions: List[float],
    festival_weeks: List[int],
    historical_avg: float,
) -> Dict:
    """
    Explain what's driving the demand forecast.
    Shows contribution of each factor to the predicted demand.
    """
    avg_pred = statistics.mean(predictions) if predictions else 0
    trend_contribution    = round((avg_pred - historical_avg) / max(historical_avg, 1) * 100, 1)
    festival_contribution = round(
        sum(predictions[w-1] - historical_avg for w in festival_weeks if w <= len(predictions)) /
        max(len(festival_weeks), 1), 1
    ) if festival_weeks else 0.0

    seasonal_factor = 1.0
    if category in ("saree", "lehenga"):
        seasonal_factor = 1.3
    elif category in ("western",):
        seasonal_factor = 0.9

    explanations = []

    if trend_contribution > 10:
        explanations.append(
            f"Upward trend (+{trend_contribution:.0f}%): demand is growing vs recent history"
        )
    elif trend_contribution < -10:
        explanations.append(
            f"Downward trend ({trend_contribution:.0f}%): demand declining vs recent history"
        )

    if festival_weeks:
        festival_names = {
            1: "Navratri", 2: "Diwali", 3: "Dussehra",
            4: "Christmas", 5: "New Year",
        }
        explanations.append(
            f"Festival boost in weeks {festival_weeks[:3]}: "
            f"{category} sees {festival_contribution:.0f}+ extra units during festive period"
        )

    if seasonal_factor > 1.1:
        explanations.append(
            f"{category.capitalize()} sees strong seasonal demand in India — "
            f"especially during wedding and festival seasons"
        )

    if not explanations:
        explanations.append("Stable demand expected — no major festivals or trend shifts")

    return {
        "item_id":              item_id,
        "item_name":            item_name,
        "avg_prediction":       round(avg_pred, 2),
        "historical_avg":       round(historical_avg, 2),
        "trend_contribution_pct": trend_contribution,
        "festival_contribution":  round(festival_contribution, 2),
        "seasonal_factor":      seasonal_factor,
        "key_drivers":          explanations,
        "plain_english": (
            f"Expected {avg_pred:.0f} units/week on average. "
            + (" | ".join(explanations[:2]))
        ),
    }


def rank_inventory_by_fit(
    inventory: List[Dict],
    customer: Dict,
    current_outfit: List[Dict],
    top_n: int = 10,
) -> List[Dict]:
    """
    Rank all inventory items by fit for the customer.
    Returns top N with SHAP explanations — used by the dashboard.
    """
    ranked = []
    for item in inventory:
        exp = explain_outfit_item(item, customer, current_outfit)
        ranked.append({
            "item":        item,
            "score":       exp.overall_score,
            "explanation": exp,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_n]
