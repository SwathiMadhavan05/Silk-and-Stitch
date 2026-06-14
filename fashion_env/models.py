"""
Typed Pydantic models for the Fashion Stylist OpenEnv environment.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ClothingItem(BaseModel):
    id: str
    name: str
    category: str          # top, bottom, shoes, accessory, outerwear
    color: str
    style_tags: List[str]  # casual, formal, sporty, boho, classic, etc.
    season: List[str]      # spring, summer, fall, winter
    price: float
    in_stock: bool = True
    brand: str = ""
    size_available: List[str] = Field(default_factory=list)


class CustomerProfile(BaseModel):
    id: str
    name: str
    budget: float
    preferred_styles: List[str]
    occasion: str          # office, wedding, casual, party, outdoor
    season: str
    body_type: str = ""
    color_preferences: List[str] = Field(default_factory=list)
    disliked_items: List[str] = Field(default_factory=list)


class Observation(BaseModel):
    """Full environment state observable by the agent."""
    step_number: int
    customer: CustomerProfile
    inventory: List[ClothingItem]
    current_outfit: List[str]          # list of item IDs currently selected
    outfit_score: float                 # running score 0.0–1.0
    budget_remaining: float
    feedback_history: List[str]        # natural language feedback from grader
    task_id: str
    task_description: str
    max_steps: int
    done: bool = False


class Action(BaseModel):
    """Actions the agent can take in the environment."""
    action_type: str = Field(
        ...,
        description=(
            "One of: add_item, remove_item, replace_item, "
            "filter_inventory, finalize_outfit, request_feedback"
        )
    )
    item_id: Optional[str] = None          # target item for add/remove/replace
    replacement_id: Optional[str] = None   # used with replace_item
    filter_criteria: Optional[Dict[str, Any]] = None  # used with filter_inventory
    notes: Optional[str] = None            # agent's reasoning (logged, not scored)


class Reward(BaseModel):
    """Reward signal returned after each step."""
    value: float = Field(..., ge=-1.0, le=1.0)
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension reward components"
    )
    reason: str = ""
    is_terminal: bool = False
