"""
Task definitions for the Fashion Stylist OpenEnv.

Task 1 (Easy)   — Budget Casual: Build a casual daily outfit under $200.
Task 2 (Medium) — Office Ready: Style a complete office outfit for a specific customer profile.
Task 3 (Hard)   — Event Stylist: Create a gala-worthy outfit with strict budget, colour palette,
                   and occasion constraints; must include ≥4 items.
"""
from typing import List
from fashion_env.models import ClothingItem, CustomerProfile
from fashion_env.reward import (
    REQUIRED_CATEGORIES, _style_score, _color_score,
    _completeness_score, _budget_score, _season_score,
)


def _safe_score(score: float) -> float:
    """Ensure score is strictly between 0 and 1, never 0.0 or 1.0."""
    clamped = max(0.001, min(float(score), 0.999))
    return round(clamped, 4)


# ─────────────────────────── helpers ────────────────────────────

def _get_categories(items: List[ClothingItem]) -> set:
    return {item.category for item in items}


def _total_price(items: List[ClothingItem]) -> float:
    return sum(item.price for item in items)


# ─────────────────────────── Task 1: Easy ───────────────────────

TASK_1_CUSTOMER = CustomerProfile(
    id="C001",
    name="Mia",
    budget=200.0,
    preferred_styles=["casual", "classic"],
    occasion="casual",
    season="summer",
    color_preferences=["white", "navy"],
)

TASK_1 = {
    "id": "task_casual_budget",
    "name": "Budget Casual Summer Look",
    "difficulty": "easy",
    "max_steps": 15,
    "description": (
        "Build a complete casual summer outfit for Mia within a $200 budget. "
        "The outfit must include at least a top, bottom, and shoes. "
        "Prefer casual/classic styles and colours like white and navy."
    ),
    "customer": TASK_1_CUSTOMER,
}


def grade_task_1(items: List[ClothingItem], customer: CustomerProfile) -> float:
    """
    Grader for Task 1 (Easy). Score 0.0–1.0.
    Criteria:
        - Has top, bottom, shoes          → 0.40
        - Within $200 budget              → 0.25
        - At least 2 casual/classic items → 0.20
        - Summer-appropriate              → 0.15
    """
    score = 0.0
    cats = _get_categories(items)

    # Completeness (required categories)
    if REQUIRED_CATEGORIES.issubset(cats):
        score += 0.40
    else:
        covered = len(cats & REQUIRED_CATEGORIES)
        score += covered / len(REQUIRED_CATEGORIES) * 0.40

    # Budget
    total = _total_price(items)
    if total <= 200.0:
        score += 0.25
    elif total <= 240.0:
        score += 0.10  # partial credit for slight over

    # Style
    casual_classic = sum(
        1 for item in items
        if any(t in item.style_tags for t in ["casual", "classic"])
    )
    score += min(casual_classic / 2, 1.0) * 0.20

    # Season
    summer_items = sum(1 for item in items if "summer" in item.season)
    score += (summer_items / max(len(items), 1)) * 0.15

    return _safe_score(score)


# ─────────────────────────── Task 2: Medium ─────────────────────

TASK_2_CUSTOMER = CustomerProfile(
    id="C002",
    name="Priya",
    budget=450.0,
    preferred_styles=["office", "classic", "formal"],
    occasion="office",
    season="fall",
    color_preferences=["navy", "black", "white"],
    disliked_items=["Graphic Band Tee", "Platform Combat Boots"],
)

TASK_2 = {
    "id": "task_office_ready",
    "name": "Office-Ready Professional Look",
    "difficulty": "medium",
    "max_steps": 20,
    "description": (
        "Style Priya for a professional office environment in fall. "
        "Budget: $450. She prefers office/classic/formal styles in navy, black, and white. "
        "She dislikes the Graphic Band Tee and Platform Combat Boots. "
        "Must include top, bottom, shoes, and at least one accessory."
    ),
    "customer": TASK_2_CUSTOMER,
}


def grade_task_2(items: List[ClothingItem], customer: CustomerProfile) -> float:
    """
    Grader for Task 2 (Medium). Score 0.0–1.0.
    Criteria:
        - Has top, bottom, shoes, accessory  → 0.30
        - No disliked items                  → 0.15
        - ≥ 3 items with office/formal tags  → 0.20
        - Within budget                       → 0.20
        - Color palette matches (navy/black/white) → 0.15
    """
    score = 0.0
    cats = _get_categories(items)

    # Completeness incl. accessory
    required_with_acc = REQUIRED_CATEGORIES | {"accessory"}
    covered = len(cats & required_with_acc)
    score += covered / len(required_with_acc) * 0.30

    # No disliked items
    item_names = [item.name for item in items]
    if not any(d in item_names for d in customer.disliked_items):
        score += 0.15

    # Office/formal style count
    pro_items = sum(
        1 for item in items
        if any(t in item.style_tags for t in ["office", "formal", "classic"])
    )
    score += min(pro_items / 3, 1.0) * 0.20

    # Budget
    total = _total_price(items)
    if total <= customer.budget:
        score += 0.20
    elif total <= customer.budget * 1.1:
        score += 0.08

    # Color palette
    palette = {"navy", "black", "white", "charcoal", "grey"}
    palette_items = sum(1 for item in items if item.color.lower() in palette)
    score += min(palette_items / 3, 1.0) * 0.15

    return _safe_score(score)


# ─────────────────────────── Task 3: Hard ───────────────────────

TASK_3_CUSTOMER = CustomerProfile(
    id="C003",
    name="Sophia",
    budget=500.0,
    preferred_styles=["glamorous", "classic", "evening", "party"],
    occasion="wedding",
    season="winter",
    color_preferences=["gold", "champagne", "black"],
    disliked_items=["Graphic Band Tee", "Platform Combat Boots",
                    "Oversized Denim Jacket", "High-Waist Mom Jeans"],
)

TASK_3 = {
    "id": "task_gala_stylist",
    "name": "Winter Gala / Wedding Guest Look",
    "difficulty": "hard",
    "max_steps": 25,
    "description": (
        "Sophia is attending a winter gala / wedding. Budget: $500. "
        "She wants a glamorous evening look in gold, champagne, or black tones. "
        "Must include ≥4 items (top OR dress as 'top', bottom/skirt, shoes, AND ≥1 accessory). "
        "All items must be winter-appropriate. No casual/streetwear styles. "
        "Must avoid her disliked items. Color harmony is critical."
    ),
    "customer": TASK_3_CUSTOMER,
}


def grade_task_3(items: List[ClothingItem], customer: CustomerProfile) -> float:
    """
    Grader for Task 3 (Hard). Score 0.0–1.0.
    Criteria:
        - ≥ 4 items total                                  → 0.15
        - Has top, shoes, accessory (bottom or skirt OK)   → 0.20
        - ≥ 2 items in gold/champagne/black/silver         → 0.15
        - ≥ 3 items with glamorous/evening/party/classic   → 0.15
        - Within $500 budget                               → 0.15
        - All items winter-appropriate                     → 0.10
        - No disliked items                                → 0.10
    """
    score = 0.0
    cats = _get_categories(items)
    item_names = [item.name for item in items]
    item_colors = [item.color.lower() for item in items]

    # 4+ items
    if len(items) >= 4:
        score += 0.15
    elif len(items) == 3:
        score += 0.07

    # Required categories
    needed = REQUIRED_CATEGORIES | {"accessory"}
    covered = len(cats & needed)
    score += (covered / len(needed)) * 0.20

    # Color palette
    event_colors = {"gold", "champagne", "black", "silver", "ivory", "nude", "white"}
    palette_match = sum(1 for c in item_colors if c in event_colors)
    score += min(palette_match / 2, 1.0) * 0.15

    # Glamorous/evening style
    glam_items = sum(
        1 for item in items
        if any(t in item.style_tags for t in ["glamorous", "evening", "party", "classic"])
    )
    score += min(glam_items / 3, 1.0) * 0.15

    # Budget
    total = _total_price(items)
    if total <= customer.budget:
        score += 0.15
    elif total <= customer.budget * 1.05:
        score += 0.05

    # Winter appropriate
    winter_items = sum(1 for item in items if "winter" in item.season)
    score += (winter_items / max(len(items), 1)) * 0.10

    # No disliked items
    if not any(d in item_names for d in customer.disliked_items):
        score += 0.10

    return _safe_score(score)




# ─────────────────────────── Task 4: Expert ─────────────────────────────────

TASK_4_CUSTOMER = CustomerProfile(
    id="C004",
    name="Aria",
    budget=350.0,
    preferred_styles=["edgy", "streetwear", "retro", "casual"],
    occasion="party",
    season="fall",
    color_preferences=["black", "red", "gold"],
    disliked_items=["White Oxford Shirt", "Tailored Trousers", "Nude Ballet Flats"],
)

TASK_4 = {
    "id": "task_street_party",
    "name": "Edgy Street-to-Party Look",
    "difficulty": "expert",
    "max_steps": 20,
    "description": (
        "Style Aria for a rooftop party in fall. Budget: $350. "
        "She wants an edgy, street-inspired party look in black, red, or gold tones. "
        "Must include ≥4 items. No formal or office styles. "
        "Avoid her disliked items. Shoes must be edgy or party style."
    ),
    "customer": TASK_4_CUSTOMER,
}


def grade_task_4(items: List[ClothingItem], customer: CustomerProfile) -> float:
    """
    Grader for Task 4 (Expert). Score 0.0–1.0.
    Criteria:
        - ≥ 4 items total                                  → 0.15
        - Has top, bottom, shoes                           → 0.20
        - ≥ 2 items in black/red/gold palette              → 0.15
        - ≥ 2 items with edgy/streetwear/party/retro tags  → 0.20
        - Within $350 budget                               → 0.15
        - No formal/office style items                     → 0.10
        - No disliked items                                → 0.05
    """
    score = 0.0
    cats = {item.category for item in items}
    item_names = [item.name for item in items]
    item_colors = [item.color.lower() for item in items]

    # ≥4 items
    if len(items) >= 4:
        score += 0.15
    elif len(items) == 3:
        score += 0.07

    # Required categories
    required = {"top", "bottom", "shoes"}
    score += (len(cats & required) / len(required)) * 0.20

    # Color palette
    party_colors = {"black", "red", "gold", "silver", "white"}
    palette_match = sum(1 for c in item_colors if c in party_colors)
    score += min(palette_match / 2, 1.0) * 0.15

    # Edgy/street style
    street_items = sum(
        1 for item in items
        if any(t in item.style_tags for t in ["edgy", "streetwear", "party", "retro"])
    )
    score += min(street_items / 2, 1.0) * 0.20

    # Budget
    total = sum(item.price for item in items)
    if total <= customer.budget:
        score += 0.15
    elif total <= customer.budget * 1.05:
        score += 0.05

    # No formal/office items
    formal_items = sum(
        1 for item in items
        if any(t in item.style_tags for t in ["formal", "office"])
    )
    if formal_items == 0:
        score += 0.10

    # No disliked items
    if not any(d in item_names for d in customer.disliked_items):
        score += 0.05

    return _safe_score(score)


# ─────────────────────────── Registry ───────────────────────────

TASKS = {
    "task_casual_budget": (TASK_1, grade_task_1),
    "task_office_ready":  (TASK_2, grade_task_2),
    "task_gala_stylist":  (TASK_3, grade_task_3),
    "task_street_party":  (TASK_4, grade_task_4),
}
