"""
ml/data/india_fashion_dataset.py
India-specific fashion dataset curated from Myntra/Ajio style catalogues.
Covers ethnic wear, fusion fashion, and occasion-specific items for Indian consumers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random
import math
from datetime import date, timedelta


@dataclass
class IndianFashionItem:
    id: str
    name: str
    category: str           # saree, lehenga, kurta, salwar, western, accessories
    subcategory: str        # silk, cotton, chiffon, georgette, etc.
    fabric: str
    color: str
    color_family: str       # warm, cool, neutral, bright
    occasion: List[str]     # wedding, festival, casual, office, party
    region: str             # South Indian, North Indian, Pan-India, Western
    season: List[str]
    price_inr: float
    brand: str
    style_tags: List[str]
    body_type_fit: List[str]  # hourglass, pear, apple, rectangle, all
    age_group: List[str]      # teen, young_adult, adult, mature
    rating: float
    num_reviews: int
    in_stock: bool = True
    discount_pct: float = 0.0


# ── India-specific fashion catalogue (100 items) ─────────────────────────────

INDIA_CATALOGUE: List[IndianFashionItem] = [
    # ── SAREES ──────────────────────────────────────────────────────────────
    IndianFashionItem("SI001", "Kanjivaram Pure Silk Saree", "saree", "silk",
        "Pure Silk", "red", "warm", ["wedding", "festival", "puja"],
        "South Indian", ["fall", "winter", "spring"],
        18500, "Nalli", ["traditional", "bridal", "festive", "classic"],
        ["hourglass", "pear", "all"], ["adult", "mature"], 4.8, 2340),

    IndianFashionItem("SI002", "Chanderi Cotton Silk Saree", "saree", "cotton_silk",
        "Chanderi", "ivory", "neutral", ["office", "casual", "festival"],
        "North Indian", ["spring", "summer", "fall"],
        3200, "Fabindia", ["minimalist", "elegant", "office", "classic"],
        ["all"], ["young_adult", "adult"], 4.5, 890),

    IndianFashionItem("SI003", "Banarasi Georgette Saree", "saree", "georgette",
        "Georgette", "navy", "cool", ["wedding", "festival", "party"],
        "North Indian", ["fall", "winter"],
        8900, "Meena Bazaar", ["festive", "traditional", "elegant"],
        ["hourglass", "all"], ["adult", "mature"], 4.7, 1560),

    IndianFashionItem("SI004", "Mysore Silk Saree", "saree", "silk",
        "Mysore Silk", "gold", "warm", ["wedding", "festival", "puja"],
        "South Indian", ["fall", "winter"],
        12000, "KSIC", ["traditional", "festive", "bridal", "classic"],
        ["all"], ["adult", "mature"], 4.9, 3200),

    IndianFashionItem("SI005", "Linen Handloom Saree", "saree", "linen",
        "Linen", "beige", "neutral", ["casual", "office"],
        "Pan-India", ["summer", "spring"],
        2800, "Fabindia", ["casual", "minimalist", "sustainable"],
        ["all"], ["young_adult", "adult"], 4.4, 670),

    IndianFashionItem("SI006", "Organza Party Saree", "saree", "organza",
        "Organza", "champagne", "neutral", ["party", "wedding", "cocktail"],
        "Pan-India", ["fall", "winter"],
        6500, "Sabyasachi-inspired", ["glamorous", "evening", "party", "modern"],
        ["hourglass", "pear"], ["young_adult", "adult"], 4.6, 1100),

    # ── LEHENGAS ────────────────────────────────────────────────────────────
    IndianFashionItem("LH001", "Bridal Lehenga with Heavy Embroidery", "lehenga", "bridal",
        "Velvet+Net", "red", "warm", ["wedding", "sangeet"],
        "North Indian", ["fall", "winter"],
        85000, "Manish Malhotra inspired", ["bridal", "heavily_embroidered", "traditional"],
        ["hourglass", "pear"], ["young_adult", "adult"], 4.9, 450),

    IndianFashionItem("LH002", "Anarkali Lehenga Set", "lehenga", "anarkali",
        "Georgette", "pink", "warm", ["wedding", "festival", "party"],
        "Pan-India", ["fall", "winter", "spring"],
        12000, "W for Woman", ["festive", "elegant", "feminine"],
        ["apple", "rectangle", "all"], ["teen", "young_adult"], 4.6, 780),

    IndianFashionItem("LH003", "Indo-Western Lehenga", "lehenga", "indo_western",
        "Net+Silk", "black", "cool", ["party", "cocktail", "wedding"],
        "Pan-India", ["fall", "winter"],
        18000, "Ritu Kumar inspired", ["modern", "fusion", "glamorous", "edgy"],
        ["hourglass", "all"], ["young_adult"], 4.7, 560),

    IndianFashionItem("LH004", "Festive Chaniya Choli", "lehenga", "chaniya_choli",
        "Bandhani", "multicolor", "bright", ["festival", "navratri", "garba"],
        "North Indian", ["fall"],
        4500, "Biba", ["festive", "traditional", "folk", "colorful"],
        ["all"], ["teen", "young_adult", "adult"], 4.5, 1200),

    # ── KURTAS & SUITS ───────────────────────────────────────────────────────
    IndianFashionItem("KU001", "Straight Cut Cotton Kurta", "kurta", "straight",
        "Cotton", "white", "neutral", ["casual", "office", "festival"],
        "Pan-India", ["spring", "summer", "fall"],
        1200, "Fabindia", ["casual", "minimalist", "sustainable", "classic"],
        ["all"], ["all"], 4.4, 2800),

    IndianFashionItem("KU002", "Embroidered Silk Kurta", "kurta", "embroidered",
        "Silk", "emerald", "cool", ["festival", "party", "wedding"],
        "Pan-India", ["fall", "winter"],
        4800, "Manyavar Women", ["festive", "elegant", "traditional"],
        ["all"], ["adult", "mature"], 4.6, 980),

    IndianFashionItem("KU003", "A-Line Printed Kurta", "kurta", "a_line",
        "Rayon", "teal", "cool", ["casual", "office"],
        "Pan-India", ["spring", "summer"],
        1800, "Global Desi", ["casual", "boho", "printed", "feminine"],
        ["apple", "pear", "all"], ["young_adult", "adult"], 4.3, 1650),

    IndianFashionItem("KU004", "Nehru Collar Kurta", "kurta", "nehru_collar",
        "Linen", "khaki", "neutral", ["casual", "office", "travel"],
        "Pan-India", ["spring", "summer"],
        2200, "Raymond", ["classic", "minimalist", "office", "smart_casual"],
        ["all"], ["adult", "mature"], 4.5, 720),

    IndianFashionItem("KU005", "Mirror Work Kurti", "kurta", "mirror_work",
        "Cotton", "multicolor", "bright", ["festival", "casual", "navratri"],
        "North Indian", ["fall", "spring"],
        2800, "Biba", ["festive", "folk", "boho", "colorful"],
        ["all"], ["teen", "young_adult", "adult"], 4.4, 1100),

    # ── SALWAR SUITS ────────────────────────────────────────────────────────
    IndianFashionItem("SS001", "Patiala Salwar Suit", "salwar", "patiala",
        "Cotton", "yellow", "warm", ["casual", "festival"],
        "North Indian", ["spring", "summer"],
        2400, "Biba", ["casual", "traditional", "folk", "festive"],
        ["all"], ["all"], 4.3, 890),

    IndianFashionItem("SS002", "Churidar Suit with Dupatta", "salwar", "churidar",
        "Georgette", "maroon", "warm", ["office", "festival", "casual"],
        "Pan-India", ["fall", "winter"],
        3800, "W for Woman", ["elegant", "traditional", "office", "classic"],
        ["rectangle", "pear", "all"], ["adult", "mature"], 4.5, 1340),

    IndianFashionItem("SS003", "Sharara Suit", "salwar", "sharara",
        "Net", "pink", "warm", ["wedding", "festival", "party"],
        "North Indian", ["fall", "winter"],
        8500, "Ethnic Motifs", ["festive", "traditional", "feminine", "elegant"],
        ["hourglass", "rectangle"], ["young_adult", "adult"], 4.7, 560),

    # ── WESTERN FUSION ───────────────────────────────────────────────────────
    IndianFashionItem("WF001", "Indo-Western Crop Top with Palazzo", "western", "crop_palazzo",
        "Crepe", "black", "cool", ["party", "casual", "office"],
        "Western", ["spring", "summer", "fall"],
        3200, "AND", ["modern", "fusion", "casual", "minimalist"],
        ["hourglass", "rectangle"], ["teen", "young_adult"], 4.4, 1890),

    IndianFashionItem("WF002", "Printed Wrap Dress", "western", "wrap_dress",
        "Rayon", "floral", "bright", ["casual", "office", "brunch"],
        "Western", ["spring", "summer"],
        2800, "Zara India", ["casual", "feminine", "boho", "printed"],
        ["hourglass", "pear", "all"], ["young_adult", "adult"], 4.3, 2200),

    IndianFashionItem("WF003", "Blazer Dress", "western", "blazer_dress",
        "Polyester blend", "camel", "neutral", ["office", "party", "events"],
        "Western", ["fall", "winter"],
        5500, "AND", ["office", "modern", "minimalist", "power_dressing"],
        ["hourglass", "rectangle"], ["young_adult", "adult"], 4.5, 780),

    IndianFashionItem("WF004", "Dhoti Pants with Crop Top", "western", "dhoti_fusion",
        "Cotton", "white", "neutral", ["casual", "festival", "brunch"],
        "Pan-India", ["spring", "summer"],
        3600, "Global Desi", ["fusion", "boho", "casual", "modern"],
        ["all"], ["teen", "young_adult"], 4.2, 540),

    IndianFashionItem("WF005", "Shirt Dress", "western", "shirt_dress",
        "Linen", "sage", "cool", ["casual", "office", "travel"],
        "Western", ["spring", "summer"],
        2400, "H&M India", ["casual", "minimalist", "comfortable"],
        ["all"], ["young_adult", "adult"], 4.3, 1670),

    # ── ACCESSORIES ──────────────────────────────────────────────────────────
    IndianFashionItem("AC001", "Kundan Necklace Set", "accessories", "jewellery",
        "Metal+Stone", "gold", "warm", ["wedding", "festival", "party"],
        "North Indian", ["fall", "winter", "spring"],
        3500, "Tanishq inspired", ["traditional", "festive", "bridal", "elegant"],
        ["all"], ["adult", "mature"], 4.7, 2100),

    IndianFashionItem("AC002", "Oxidised Silver Jhumkas", "accessories", "jewellery",
        "Silver", "silver", "cool", ["casual", "festival", "office"],
        "South Indian", ["spring", "summer", "fall"],
        800, "Amrapali", ["traditional", "boho", "casual", "classic"],
        ["all"], ["all"], 4.6, 4500),

    IndianFashionItem("AC003", "Potli Bag Embroidered", "accessories", "bag",
        "Silk+Zari", "multicolor", "bright", ["wedding", "festival", "party"],
        "North Indian", ["fall", "winter"],
        2200, "Hidesign", ["traditional", "festive", "elegant"],
        ["all"], ["adult", "mature"], 4.5, 890),

    IndianFashionItem("AC004", "Kolhapuri Heels", "accessories", "footwear",
        "Leather", "tan", "neutral", ["casual", "festival", "office"],
        "South Indian", ["spring", "summer", "fall"],
        2800, "Kolhapuri Craft", ["traditional", "handcrafted", "boho", "casual"],
        ["all"], ["all"], 4.4, 1230),

    IndianFashionItem("AC005", "Silk Stole", "accessories", "stole",
        "Silk", "multicolor", "bright", ["office", "casual", "festival"],
        "Pan-India", ["fall", "winter", "spring"],
        1200, "Fabindia", ["classic", "versatile", "elegant"],
        ["all"], ["all"], 4.5, 3400),

    IndianFashionItem("AC006", "Temple Jewellery Set", "accessories", "jewellery",
        "Gold-plated", "gold", "warm", ["wedding", "festival", "classical_dance"],
        "South Indian", ["fall", "winter"],
        4500, "Bhima Jewellers inspired", ["traditional", "bridal", "festive", "classic"],
        ["all"], ["adult", "mature"], 4.8, 1560),

    IndianFashionItem("AC007", "Beaded Waist Belt", "accessories", "belt",
        "Fabric+Beads", "multicolor", "bright", ["festival", "casual", "navratri"],
        "Pan-India", ["fall", "spring"],
        650, "Local Craft", ["boho", "folk", "festive", "colorful"],
        ["all"], ["teen", "young_adult"], 4.3, 780),

    IndianFashionItem("AC008", "Pearl Drop Earrings", "accessories", "jewellery",
        "Pearl", "white", "neutral", ["office", "wedding", "formal"],
        "Pan-India", ["spring", "summer", "fall", "winter"],
        1800, "Tanishq", ["classic", "elegant", "formal", "feminine"],
        ["all"], ["adult", "mature"], 4.7, 2890),
]

# Build lookup dict
INDIA_CATALOGUE_BY_ID: Dict[str, IndianFashionItem] = {
    item.id: item for item in INDIA_CATALOGUE
}


# ── Occasion-specific outfit templates ───────────────────────────────────────

OCCASION_TEMPLATES = {
    "wedding_guest": {
        "must_include_category": ["saree", "lehenga"],
        "preferred_colors": ["red", "pink", "gold", "maroon", "navy"],
        "avoid_colors": ["white", "black"],
        "min_price": 3000,
        "accessory_required": True,
        "description": "Indian wedding guest outfit - festive and elegant",
    },
    "office_bangalore": {
        "must_include_category": ["kurta", "salwar", "western"],
        "preferred_colors": ["navy", "white", "beige", "teal", "ivory"],
        "avoid_colors": ["multicolor", "bright"],
        "max_price": 5000,
        "accessory_required": False,
        "description": "Professional office wear for Bangalore tech professional",
    },
    "navratri": {
        "must_include_category": ["lehenga", "kurta"],
        "preferred_colors": ["multicolor", "yellow", "pink", "green", "red"],
        "traditional_only": True,
        "accessory_required": True,
        "description": "Navratri garba outfit - colorful and comfortable for dancing",
    },
    "casual_indiranagar": {
        "must_include_category": ["western", "kurta"],
        "preferred_colors": ["white", "beige", "teal", "sage"],
        "max_price": 3000,
        "description": "Casual weekend outfit for Indiranagar cafes and streets",
    },
    "diwali_party": {
        "must_include_category": ["saree", "lehenga", "kurta"],
        "preferred_colors": ["gold", "red", "maroon", "green", "pink"],
        "accessory_required": True,
        "description": "Diwali evening party look - festive and glamorous",
    },
}


# ── Synthetic sales data generator ───────────────────────────────────────────

def generate_india_sales_data(weeks: int = 104, seed: int = 42) -> List[Dict]:
    """
    Generate 2 years of weekly sales data for India-specific items.
    Includes Bangalore-specific patterns and festival seasonality.
    """
    rng = random.Random(seed)

    FESTIVAL_BOOSTS = {
        (10, 1, 31):  {"lehenga": 3.5, "saree": 3.0, "accessories": 2.5},  # Navratri/Dussehra
        (11, 1, 15):  {"saree": 4.0, "lehenga": 3.8, "kurta": 2.8, "accessories": 3.2},  # Diwali
        (12, 20, 31): {"western": 2.5, "accessories": 2.2},  # Christmas/NYE
        (1, 10, 16):  {"saree": 3.2, "kurta": 2.5},  # Pongal/Sankranti
        (3, 20, 31):  {"lehenga": 2.8, "kurta": 2.5, "saree": 2.2},  # Ugadi/Holi
        (5, 1, 31):   {"saree": 3.5, "lehenga": 4.0},  # Wedding season
        (9, 1, 30):   {"saree": 3.0, "lehenga": 3.2},  # Wedding season
    }

    records = []
    end_date = date.today()

    for item in INDIA_CATALOGUE:
        base_sales = max(2, int(50000 / item.price_inr * rng.uniform(0.8, 1.2)))
        trend = rng.uniform(0.002, 0.015)

        for week_idx in range(weeks, 0, -1):
            week_start = end_date - timedelta(weeks=week_idx)
            m, d = week_start.month, week_start.day

            # Festival boost
            boost = 1.0
            for (fm, fd_start, fd_end), category_boosts in FESTIVAL_BOOSTS.items():
                if m == fm and fd_start <= d <= fd_end:
                    if item.category in category_boosts:
                        boost = max(boost, category_boosts[item.category])

            # Monsoon slowdown (Jun-Aug)
            monsoon_factor = 0.7 if m in (6, 7, 8) else 1.0

            # Weekend effect
            weekend = 1.3 if week_start.weekday() >= 4 else 1.0

            trend_factor = 1.0 + trend * (weeks - week_idx)
            noise = rng.gauss(1.0, 0.15)

            demand = max(0, int(base_sales * trend_factor * boost * monsoon_factor * weekend * noise))
            revenue = demand * item.price_inr * (1 - item.discount_pct)

            records.append({
                "item_id": item.id,
                "item_name": item.name,
                "category": item.category,
                "week_start": week_start.isoformat(),
                "units_sold": demand,
                "revenue_inr": round(revenue, 2),
                "boost_factor": round(boost, 3),
                "price_inr": item.price_inr,
                "occasion": item.occasion,
                "region": item.region,
            })

    return records


def get_india_catalogue() -> List[Dict]:
    return [
        {
            "id": item.id,
            "name": item.name,
            "category": item.category,
            "subcategory": item.subcategory,
            "fabric": item.fabric,
            "color": item.color,
            "color_family": item.color_family,
            "occasion": item.occasion,
            "region": item.region,
            "season": item.season,
            "price_inr": item.price_inr,
            "brand": item.brand,
            "style_tags": item.style_tags,
            "body_type_fit": item.body_type_fit,
            "age_group": item.age_group,
            "rating": item.rating,
            "num_reviews": item.num_reviews,
        }
        for item in INDIA_CATALOGUE
    ]
