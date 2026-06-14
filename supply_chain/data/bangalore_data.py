"""
Synthetic historical sales data for Bangalore boutiques.
Captures India-specific seasonality: festivals, wedding season, monsoon, summer.
"""
from __future__ import annotations
import random
import math
from datetime import date, timedelta
from typing import List, Dict

# ── Bangalore-specific seasonal event calendar ───────────────────────────────
# Each event boosts certain categories during its window

BANGALORE_EVENTS = [
    # (month, day_start, day_end, label, category_boosts)
    (1,  14, 16,  "Pongal/Sankranti",      {"ethnic_wear": 2.8, "sarees": 3.2, "accessories": 2.0}),
    (1,  26, 26,  "Republic Day",           {"western_formal": 1.5, "kurtas": 1.4}),
    (2,  14, 14,  "Valentine's Day",        {"western_casual": 1.8, "accessories": 2.2, "party_wear": 2.0}),
    (3,  25, 30,  "Holi/Ugadi",             {"ethnic_wear": 2.5, "kurtas": 2.2, "sarees": 1.8}),
    (4,  14, 14,  "Vishu / Tamil New Year", {"sarees": 3.0, "ethnic_wear": 2.6, "accessories": 1.9}),
    (5,  1,  31,  "Wedding Season (May)",   {"sarees": 3.5, "lehengas": 4.0, "ethnic_wear": 3.0, "western_formal": 2.0}),
    (6,  1,  30,  "Monsoon – slow",         {"western_casual": 0.6, "ethnic_wear": 0.7, "sarees": 0.8}),
    (7,  1,  31,  "Monsoon – slow",         {"western_casual": 0.6, "ethnic_wear": 0.7, "sarees": 0.8}),
    (8,  15, 15,  "Independence Day",       {"kurtas": 1.6, "ethnic_wear": 1.5}),
    (8,  19, 23,  "Onam",                   {"sarees": 3.0, "ethnic_wear": 2.8, "accessories": 2.0}),
    (9,  1,  30,  "Wedding Season (Sep)",   {"sarees": 3.2, "lehengas": 3.5, "ethnic_wear": 2.8}),
    (10, 2,  26,  "Navratri & Dussehra",   {"ethnic_wear": 3.0, "lehengas": 3.8, "sarees": 2.5, "accessories": 2.2}),
    (11, 1,  5,   "Diwali",                 {"ethnic_wear": 4.0, "sarees": 4.2, "lehengas": 3.8, "accessories": 3.0, "party_wear": 2.5}),
    (11, 11, 13,  "Kannada Rajyotsava",    {"ethnic_wear": 2.0, "sarees": 2.2}),
    (11, 25, 30,  "Pre-wedding season",     {"sarees": 2.8, "lehengas": 3.0, "western_formal": 1.8}),
    (12, 1,  25,  "Christmas / Year-end",   {"western_casual": 2.0, "party_wear": 2.8, "accessories": 2.4, "western_formal": 1.6}),
    (12, 26, 31,  "New Year",               {"party_wear": 3.2, "western_casual": 1.8, "accessories": 2.5}),
]

# ── Product catalogue for Bangalore boutiques ─────────────────────────────────
BOUTIQUE_PRODUCTS = [
    # id, name, category, base_price (INR), base_weekly_sales, lead_time_weeks, reorder_point, max_stock
    ("P001", "Kanjivaram Silk Saree",       "sarees",         12000, 3,  8, 6,  20),
    ("P002", "Cotton Casual Saree",          "sarees",          2500, 8,  4, 10, 40),
    ("P003", "Designer Lehenga Choli",       "lehengas",       18000, 2,  6, 4,  12),
    ("P004", "Bridal Lehenga",               "lehengas",       45000, 1,  10,2,  6),
    ("P005", "Anarkali Suit",                "ethnic_wear",     4500, 6,  4, 8,  30),
    ("P006", "Churidar Kurta Set",           "ethnic_wear",     2200, 10, 3, 12, 50),
    ("P007", "Festive Kurta (Women)",        "kurtas",          1800, 12, 3, 15, 60),
    ("P008", "Men's Nehru Jacket",           "kurtas",          3500, 5,  4, 6,  25),
    ("P009", "Western Crop Top",             "western_casual",  1200, 15, 2, 18, 70),
    ("P010", "High-Waist Jeans",             "western_casual",  2800, 12, 3, 14, 55),
    ("P011", "Maxi Dress",                   "western_casual",  2200, 8,  3, 10, 40),
    ("P012", "Blazer (Women)",               "western_formal",  5500, 4,  4, 5,  20),
    ("P013", "Formal Trouser Set",           "western_formal",  4200, 5,  4, 6,  22),
    ("P014", "Party Jumpsuit",               "party_wear",      3800, 4,  3, 5,  18),
    ("P015", "Sequin Kurta (Party)",         "party_wear",      5200, 3,  4, 4,  15),
    ("P016", "Gold Jhumkas",                 "accessories",      800, 20, 2, 25, 100),
    ("P017", "Oxidised Silver Necklace",     "accessories",     1400, 14, 2, 18, 70),
    ("P018", "Embroidered Potli Bag",        "accessories",     2200, 8,  3, 10, 40),
    ("P019", "Kolhapuri Sandals",            "accessories",     1600, 10, 3, 12, 50),
    ("P020", "Designer Dupatta",             "accessories",      900, 18, 2, 22, 80),
]


def _event_multiplier(product_category: str, target_date: date) -> float:
    """Return the demand multiplier for a product on a given date based on events."""
    multiplier = 1.0
    m, d = target_date.month, target_date.day
    for ev_month, ev_start, ev_end, _label, boosts in BANGALORE_EVENTS:
        if m == ev_month and ev_start <= d <= ev_end:
            if product_category in boosts:
                multiplier = max(multiplier, boosts[product_category])
    return multiplier


def _temperature_factor(target_date: date) -> float:
    """Bangalore has mild winters, hot summers, wet monsoon. Affects western vs ethnic."""
    m = target_date.month
    # Bangalore temp: Jan–Feb cool (22°C), Mar–May hot (32°C), Jun–Sep monsoon, Oct–Dec pleasant
    if m in (6, 7, 8):  # monsoon — people shop less
        return 0.75
    if m in (3, 4, 5):  # hot & humid — lighter fabrics sell more
        return 0.90
    return 1.0  # pleasant season — normal shopping


def _weekend_boost(target_date: date) -> float:
    """Weekends have higher footfall in Bangalore boutiques."""
    return 1.35 if target_date.weekday() >= 5 else 1.0


def generate_historical_sales(
    weeks: int = 104,  # 2 years of history
    noise_level: float = 0.18,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate synthetic weekly sales records for each product for `weeks` weeks.
    Returns list of dicts: {product_id, week_start, units_sold, revenue, stock_level, stockout}
    """
    rng = random.Random(seed)
    end_date = date.today()
    records = []

    for product in BOUTIQUE_PRODUCTS:
        pid, name, category, price, base_sales, lead_time, reorder, max_stock = product
        current_stock = rng.randint(reorder, max_stock)
        trend_slope = rng.uniform(-0.005, 0.012)  # slight upward trend overall

        for week_idx in range(weeks, 0, -1):
            week_start = end_date - timedelta(weeks=week_idx)

            # Base demand
            trend_factor = 1.0 + trend_slope * (weeks - week_idx)
            event_factor = max(
                _event_multiplier(category, week_start),
                _event_multiplier(category, week_start + timedelta(days=3)),
                _event_multiplier(category, week_start + timedelta(days=6)),
            )
            temp_factor   = _temperature_factor(week_start)
            weekend_factor = _weekend_boost(week_start)

            # Compute expected demand
            expected = base_sales * trend_factor * event_factor * temp_factor * weekend_factor
            # Add noise
            noise = rng.gauss(0, noise_level * expected)
            demand = max(0, int(round(expected + noise)))

            # Stock simulation
            stockout = False
            if demand > current_stock:
                units_sold = current_stock
                stockout = True
            else:
                units_sold = demand

            revenue = units_sold * price
            current_stock = max(0, current_stock - units_sold)

            # Reorder simulation (instant replenishment for data gen simplicity)
            if current_stock <= reorder:
                restock = rng.randint(reorder, max_stock - current_stock)
                current_stock = min(max_stock, current_stock + restock)

            records.append({
                "product_id":   pid,
                "product_name": name,
                "category":     category,
                "price_inr":    price,
                "week_start":   week_start.isoformat(),
                "units_sold":   units_sold,
                "revenue_inr":  revenue,
                "stock_level":  current_stock,
                "stockout":     stockout,
                "event_factor": round(event_factor, 3),
                "lead_time_weeks": lead_time,
                "reorder_point": reorder,
                "max_stock":    max_stock,
            })

    return records


def get_product_catalogue() -> List[Dict]:
    return [
        {
            "product_id":    p[0],
            "product_name":  p[1],
            "category":      p[2],
            "price_inr":     p[3],
            "base_weekly_sales": p[4],
            "lead_time_weeks":   p[5],
            "reorder_point": p[6],
            "max_stock":     p[7],
        }
        for p in BOUTIQUE_PRODUCTS
    ]
