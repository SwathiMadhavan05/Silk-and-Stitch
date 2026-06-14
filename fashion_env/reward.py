"""
Reward function engine for the Fashion Stylist environment.
Provides partial progress rewards across multiple dimensions.
"""
from typing import List, Dict
from fashion_env.models import ClothingItem, CustomerProfile, Reward


# Style compatibility matrix — how well do two style_tags mix?
STYLE_COMPAT: Dict[str, List[str]] = {
    "formal":      ["classic", "office", "minimalist", "feminine"],
    "casual":      ["classic", "boho", "retro", "streetwear", "nautical", "resort"],
    "office":      ["formal", "classic", "minimalist", "feminine"],
    "party":       ["glamorous", "evening", "edgy"],
    "boho":        ["casual", "resort", "feminine", "retro"],
    "edgy":        ["streetwear", "casual", "retro"],
    "glamorous":   ["party", "evening", "classic"],
    "minimalist":  ["classic", "formal", "office"],
    "classic":     ["formal", "casual", "office", "feminine", "minimalist"],
    "streetwear":  ["casual", "edgy", "sporty"],
    "resort":      ["boho", "casual", "summer"],
    "preppy":      ["classic", "casual"],
    "retro":       ["casual", "boho", "edgy"],
}

COLOR_COMPAT: Dict[str, List[str]] = {
    "black":      ["white", "black", "grey", "navy", "red", "gold", "silver", "nude", "camel"],
    "white":      ["black", "white", "navy", "beige", "nude", "gold", "silver", "light blue"],
    "navy":       ["white", "beige", "camel", "gold", "light blue", "grey"],
    "beige":      ["white", "navy", "brown", "camel", "nude", "black"],
    "grey":       ["black", "white", "navy", "pink", "yellow"],
    "camel":      ["white", "beige", "navy", "black", "brown"],
    "gold":       ["black", "white", "navy", "cream", "champagne"],
    "silver":     ["black", "white", "grey", "navy"],
    "nude":       ["black", "white", "navy", "gold", "silver"],
    "dusty rose": ["white", "beige", "camel", "gold", "navy"],
    "champagne":  ["gold", "black", "white", "nude"],
    "cream":      ["camel", "white", "gold", "navy"],
    "tan":        ["white", "navy", "black", "camel", "beige"],
    "brown":      ["camel", "beige", "white", "black"],
    "multicolor": ["black", "white", "navy", "beige"],  # neutrals work with prints
    "ivory":      ["black", "camel", "gold", "nude", "white"],
    "light blue": ["white", "navy", "beige", "camel"],
    "plaid":      ["black", "white", "camel"],
    "medium wash": ["white", "black", "navy", "beige"],
    "tortoise":   ["beige", "white", "camel", "black"],
}

REQUIRED_CATEGORIES = {"top", "bottom", "shoes"}
OPTIONAL_CATEGORIES = {"accessory", "outerwear"}


def _style_score(items: List[ClothingItem], customer: CustomerProfile) -> float:
    """Score how well the outfit's styles match the customer's preferences and occasion."""
    if not items:
        return 0.0

    all_tags = set()
    for item in items:
        all_tags.update(item.style_tags)

    # Match with customer preferred styles
    pref_matches = sum(1 for p in customer.preferred_styles if p in all_tags)
    pref_score = min(pref_matches / max(len(customer.preferred_styles), 1), 1.0)

    # Internal style coherence — check pairwise compatibility
    tag_list = list(all_tags)
    compat_pairs = 0
    total_pairs = 0
    for i in range(len(tag_list)):
        for j in range(i + 1, len(tag_list)):
            total_pairs += 1
            a, b = tag_list[i], tag_list[j]
            if b in STYLE_COMPAT.get(a, []) or a in STYLE_COMPAT.get(b, []):
                compat_pairs += 1

    coherence = compat_pairs / total_pairs if total_pairs > 0 else 1.0

    # Occasion match
    occasion_map = {
        "office": ["formal", "classic", "office", "minimalist"],
        "wedding": ["formal", "glamorous", "classic", "feminine"],
        "casual": ["casual", "boho", "retro", "classic"],
        "party": ["party", "glamorous", "evening", "edgy"],
        "outdoor": ["casual", "sporty", "boho", "resort"],
    }
    occ_tags = set(occasion_map.get(customer.occasion, []))
    occ_matches = len(all_tags & occ_tags)
    occ_score = min(occ_matches / max(len(occ_tags), 1), 1.0)

    return round(0.4 * pref_score + 0.3 * coherence + 0.3 * occ_score, 4)


def _color_score(items: List[ClothingItem]) -> float:
    """Score color harmony across the outfit."""
    if len(items) < 2:
        return 0.5

    colors = [item.color for item in items]
    compat_count = 0
    total = 0
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            total += 1
            c1, c2 = colors[i].lower(), colors[j].lower()
            if c1 == c2 or c2 in COLOR_COMPAT.get(c1, []) or c1 in COLOR_COMPAT.get(c2, []):
                compat_count += 1

    return round(compat_count / total if total > 0 else 1.0, 4)


def _completeness_score(items: List[ClothingItem]) -> float:
    """Partial credit for covering required + optional categories."""
    categories = {item.category for item in items}
    required_covered = len(categories & REQUIRED_CATEGORIES)
    optional_covered = len(categories & OPTIONAL_CATEGORIES)

    required_score = required_covered / len(REQUIRED_CATEGORIES)
    optional_bonus = min(optional_covered * 0.1, 0.2)
    return round(required_score * 0.8 + optional_bonus, 4)


def _budget_score(items: List[ClothingItem], budget: float) -> float:
    """Reward staying within budget; penalise overspend."""
    total_cost = sum(item.price for item in items)
    if budget <= 0:
        return 1.0
    if total_cost <= budget:
        # Slight bonus for using ≥50% of budget (not underdressing)
        utilisation = total_cost / budget
        return round(0.7 + 0.3 * min(utilisation / 0.5, 1.0), 4)
    else:
        overspend_ratio = (total_cost - budget) / budget
        return max(0.0, round(1.0 - overspend_ratio * 2, 4))


def _season_score(items: List[ClothingItem], season: str) -> float:
    """Score how seasonally appropriate each item is."""
    if not items:
        return 0.0
    matches = sum(1 for item in items if season in item.season)
    return round(matches / len(items), 4)


def _color_pref_score(items: List[ClothingItem], customer: CustomerProfile) -> float:
    """Bonus for incorporating customer's color preferences."""
    if not customer.color_preferences:
        return 1.0
    item_colors = {item.color.lower() for item in items}
    matches = sum(1 for c in customer.color_preferences if c.lower() in item_colors)
    return round(min(matches / len(customer.color_preferences), 1.0), 4)


def compute_reward(
    items: List[ClothingItem],
    customer: CustomerProfile,
    previous_score: float,
    is_final: bool,
) -> Reward:
    """
    Compute a rich multi-dimensional reward signal.
    Returns partial progress rewards at every step, not just terminal.
    """
    if not items:
        return Reward(
            value=-0.05,
            breakdown={"empty_outfit": -0.05},
            reason="No items in outfit yet.",
            is_terminal=False,
        )

    style   = _style_score(items, customer)
    color   = _color_score(items)
    complete = _completeness_score(items)
    budget  = _budget_score(items, customer.budget)
    season  = _season_score(items, customer.season)
    color_pref = _color_pref_score(items, customer)

    # Weighted composite
    composite = (
        0.30 * style +
        0.20 * color +
        0.20 * complete +
        0.15 * budget +
        0.10 * season +
        0.05 * color_pref
    )

    # Step delta reward — reward improvement over previous state
    delta = composite - previous_score
    step_reward = round(0.7 * composite + 0.3 * delta, 4)

    # Terminal bonus / penalty
    if is_final:
        if complete >= 0.8 and composite >= 0.7:
            step_reward = min(1.0, step_reward + 0.15)
        elif composite < 0.3:
            step_reward = max(-0.5, step_reward - 0.2)

    # Penalise disliked items
    item_names = [item.name.lower() for item in items]
    disliked_hits = sum(
        1 for d in customer.disliked_items if d.lower() in item_names
    )
    if disliked_hits:
        step_reward = max(-1.0, step_reward - 0.1 * disliked_hits)

    # Bonus: variety of categories (not all same category)
    cat_variety = len({item.category for item in items}) / 5.0
    cat_variety = min(cat_variety, 1.0)

    # Bonus: no duplicate items
    item_ids = [item.id for item in items]
    no_dupes = 1.0 if len(item_ids) == len(set(item_ids)) else 0.5

    # Adjust composite with bonuses
    composite = composite * 0.9 + cat_variety * 0.05 + no_dupes * 0.05
    composite = round(min(composite, 1.0), 4)

    # Recalculate step reward
    delta = composite - previous_score
    step_reward = round(0.7 * composite + 0.3 * delta, 4)

    if is_final:
        if complete >= 0.8 and composite >= 0.7:
            step_reward = min(1.0, step_reward + 0.15)
        elif composite < 0.3:
            step_reward = max(-0.5, step_reward - 0.2)

    if disliked_hits:
        step_reward = max(-1.0, step_reward - 0.1 * disliked_hits)

    breakdown = {
        "style_match":    style,
        "color_harmony":  color,
        "completeness":   complete,
        "budget":         budget,
        "seasonal_fit":   season,
        "color_pref":     color_pref,
        "category_variety": round(cat_variety, 4),
        "no_duplicates":  no_dupes,
        "composite":      round(composite, 4),
        "delta":          round(delta, 4),
    }

    reasons = []
    if style < 0.4:
        reasons.append("Style doesn't match customer preferences or occasion well.")
    if color < 0.5:
        reasons.append("Color clashes detected in the outfit.")
    if complete < 0.8:
        missing = REQUIRED_CATEGORIES - {item.category for item in items}
        reasons.append(f"Missing required categories: {missing}.")
    if budget < 0.5:
        total = sum(item.price for item in items)
        reasons.append(f"Over budget! Total ${total:.2f} vs budget ${customer.budget:.2f}.")
    if season < 0.5:
        reasons.append(f"Some items not suited for {customer.season}.")
    if not reasons:
        reasons.append("Outfit is looking great!")

    return Reward(
        value=round(max(-0.99, min(0.99, step_reward)), 4),
        breakdown=breakdown,
        reason=" | ".join(reasons),
        is_terminal=is_final,
    )
