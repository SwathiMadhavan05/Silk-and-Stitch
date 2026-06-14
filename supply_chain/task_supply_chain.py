"""
Task 4 — Supply Chain Optimizer (Hard+)
The agent must analyze inventory, interpret demand forecasts, and take
supply-chain actions (reorder, markdown, bundle) to minimize dead stock
and prevent stockouts across a Bangalore boutique's catalogue.

Actions available in this task:
  reorder_product   — place a purchase order for N units
  apply_markdown    — apply a discount % to a product
  create_bundle     — create a product bundle offer
  request_forecast  — get demand forecast for a product
  check_alerts      — view current inventory alerts
  finalize_plan     — commit the supply chain plan (ends episode)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from supply_chain.data.bangalore_data import (
    generate_historical_sales,
    get_product_catalogue,
    BOUTIQUE_PRODUCTS,
)
from supply_chain.models.forecaster import (
    forecast_product,
    forecast_all_products,
    generate_inventory_alerts,
    ForecastResult,
    InventoryAlert,
)
from supply_chain.models.optimizer import (
    compute_markdown_plan,
    build_supply_chain_dashboard,
    SupplyChainDashboard,
)


# ── Observation / Action models for supply chain task ───────────────────────

class SupplyChainObservation(BaseModel):
    step_number: int
    boutique_name: str
    stock_snapshot: Dict[str, int]          # product_id → current units
    alerts: List[Dict]                       # serialised InventoryAlert list
    recent_forecasts: Dict[str, Dict]        # product_id → forecast summary
    actions_taken: List[Dict]               # history of agent actions this episode
    inventory_value_inr: float
    dead_stock_pct: float
    total_score: float
    max_steps: int
    done: bool
    task_description: str


class SupplyChainAction(BaseModel):
    action_type: str = Field(
        ...,
        description=(
            "One of: reorder_product, apply_markdown, create_bundle, "
            "request_forecast, check_alerts, finalize_plan"
        )
    )
    product_id:     Optional[str]   = None
    units:          Optional[int]   = None     # for reorder
    markdown_pct:   Optional[float] = None     # for apply_markdown (0.0–0.40)
    bundle_with_id: Optional[str]   = None     # for create_bundle
    notes:          Optional[str]   = None


# ── Grader ───────────────────────────────────────────────────────────────────

def grade_supply_chain(
    actions_taken: List[Dict],
    initial_alerts: List[InventoryAlert],
    stock_snapshot: Dict[str, int],
    forecast_cache: Dict[str, ForecastResult],
) -> float:
    """
    Grade the agent's supply chain plan. Score 0.0–1.0.

    Criteria:
        - Addressed all CRITICAL alerts              → 0.30
        - Applied markdown to HIGH dead-stock items  → 0.25
        - At least 2 reorder actions placed          → 0.20
        - At least 1 bundle recommendation made      → 0.10
        - Did not over-order (no reorder > 2× EOQ)  → 0.15
    """
    score = 0.0

    critical_ids = {a.product_id for a in initial_alerts if a.severity == "CRITICAL"}
    dead_high_ids = {a.product_id for a in initial_alerts if a.alert_type == "DEAD_STOCK" and a.severity == "HIGH"}

    reorder_actions   = [a for a in actions_taken if a.get("action_type") == "reorder_product"]
    markdown_actions  = [a for a in actions_taken if a.get("action_type") == "apply_markdown"]
    bundle_actions    = [a for a in actions_taken if a.get("action_type") == "create_bundle"]

    reordered_ids  = {a.get("product_id") for a in reorder_actions}
    markdowned_ids = {a.get("product_id") for a in markdown_actions}

    # Addressed CRITICAL alerts
    if critical_ids:
        addressed = len(critical_ids & reordered_ids) / len(critical_ids)
        score += addressed * 0.30
    else:
        score += 0.30  # no critical alerts = full marks

    # Markdown on dead stock
    if dead_high_ids:
        addressed_md = len(dead_high_ids & markdowned_ids) / len(dead_high_ids)
        score += addressed_md * 0.25
    else:
        score += 0.25

    # Reorder actions count
    if len(reorder_actions) >= 3:
        score += 0.20
    elif len(reorder_actions) >= 1:
        score += 0.10

    # Bundle
    if bundle_actions:
        score += 0.10

    # Over-ordering check
    over_ordered = False
    for act in reorder_actions:
        pid   = act.get("product_id")
        units = act.get("units", 0)
        if pid and pid in forecast_cache:
            fr  = forecast_cache[pid]
            max_reasonable = fr.avg_weekly_demand * 26  # 6 months supply
            if units > max_reasonable:
                over_ordered = True
                break
    if not over_ordered:
        score += 0.15

    return round(max(0.001, min(float(score), 0.999)), 4)


# ── Task config ───────────────────────────────────────────────────────────────

def make_task_4_config() -> Dict:
    """Build the supply chain task config (generated dynamically)."""
    return {
        "id":          "task_supply_chain",
        "name":        "Bangalore Boutique Supply Chain Optimizer",
        "difficulty":  "hard+",
        "max_steps":   30,
        "description": (
            "You are managing inventory for 'Silk & Stitch', a Bangalore boutique specialising "
            "in ethnic wear, sarees, and contemporary fusion fashion. "
            "Analyse the demand forecasts and inventory alerts, then take supply chain actions: "
            "reorder products at risk of stockout, apply markdowns to dead-stock items, and create "
            "bundles to boost slow-moving products. "
            "Goal: address all CRITICAL stockout alerts and reduce dead-stock value. "
            "Budget constraint: total reorder spend ≤ ₹5,00,000."
        ),
        "boutique_name": "Silk & Stitch, Indiranagar, Bangalore",
    }


# ── Supply Chain Environment ──────────────────────────────────────────────────

class SupplyChainEnv:
    """
    OpenEnv-compliant supply chain optimization environment.
    Wraps the Bangalore boutique forecasting and optimization modules.
    """

    TASK_ID = "task_supply_chain"

    def __init__(self):
        self._task_config   = make_task_4_config()
        self._records       = generate_historical_sales()
        self._catalogue     = get_product_catalogue()
        self._cat_map       = {p["product_id"]: p for p in self._catalogue}

        self._step_count    = 0
        self._done          = False
        self._actions_taken : List[Dict] = []
        self._forecast_cache: Dict[str, ForecastResult] = {}
        self._alerts        : List[InventoryAlert] = []
        self._stock_snapshot: Dict[str, int] = {}
        self._spend_so_far  = 0.0

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(self) -> SupplyChainObservation:
        self._step_count    = 0
        self._done          = False
        self._actions_taken = []
        self._spend_so_far  = 0.0

        # Generate stock snapshot (slightly below reorder for dramatic tension)
        import random
        rng = random.Random(99)
        self._stock_snapshot = {
            p["product_id"]: rng.randint(
                max(0, p["reorder_point"] - 2),
                p["reorder_point"] + 5
            )
            for p in self._catalogue
        }

        # Pre-compute forecasts for all products
        self._forecast_cache = {}
        for prod in self._catalogue:
            pid = prod["product_id"]
            self._forecast_cache[pid] = forecast_product(
                pid,
                horizon_weeks=12,
                current_stock=self._stock_snapshot[pid],
                historical_records=self._records,
            )

        # Generate alerts
        self._alerts = generate_inventory_alerts(
            list(self._forecast_cache.values()),
            self._stock_snapshot,
        )

        return self._build_observation()

    def step(self, action: SupplyChainAction) -> Tuple[SupplyChainObservation, float, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode done. Call reset().")

        self._step_count += 1
        info: Dict[str, Any] = {"step": self._step_count, "action": action.action_type}
        reward = 0.0

        if action.action_type == "reorder_product":
            reward, info = self._handle_reorder(action, info)

        elif action.action_type == "apply_markdown":
            reward, info = self._handle_markdown(action, info)

        elif action.action_type == "create_bundle":
            reward, info = self._handle_bundle(action, info)

        elif action.action_type == "request_forecast":
            reward, info = self._handle_forecast_request(action, info)

        elif action.action_type == "check_alerts":
            info["alerts"] = [vars(a) for a in self._alerts]
            reward = 0.02  # small reward for checking

        elif action.action_type == "finalize_plan":
            self._done = True
            info["finalized"] = True
            reward = 0.05

        else:
            info["error"] = f"Unknown action: {action.action_type}"

        self._actions_taken.append({
            "action_type": action.action_type,
            "product_id":  action.product_id,
            "units":       action.units,
            "markdown_pct":action.markdown_pct,
            "bundle_with_id": action.bundle_with_id,
            "step": self._step_count,
            "reward": round(reward, 4),
        })

        if self._step_count >= self._task_config["max_steps"]:
            self._done = True
            info["step_limit_reached"] = True

        obs = self._build_observation()
        return obs, round(reward, 4), self._done, info

    def state(self) -> Dict[str, Any]:
        return {
            "task_id":          self.TASK_ID,
            "step":             self._step_count,
            "done":             self._done,
            "boutique":         self._task_config["boutique_name"],
            "stock_snapshot":   self._stock_snapshot,
            "spend_so_far_inr": self._spend_so_far,
            "alerts_count":     len(self._alerts),
            "critical_alerts":  sum(1 for a in self._alerts if a.severity == "CRITICAL"),
        }

    def grade(self) -> float:
        raw = grade_supply_chain(
            self._actions_taken,
            self._alerts,
            self._stock_snapshot,
            self._forecast_cache,
        )
        return round(max(0.001, min(float(raw), 0.999)), 4)

    # ── Action handlers ───────────────────────────────────────────────────────

    def _handle_reorder(self, action: SupplyChainAction, info: Dict) -> Tuple[float, Dict]:
        pid   = action.product_id
        units = action.units or 0

        if pid not in self._cat_map:
            info["error"] = f"Unknown product {pid}"
            return -0.05, info

        prod  = self._cat_map[pid]
        cost  = prod["price_inr"] * units * 0.55  # wholesale ~55% of retail
        budget_remaining = 500_000 - self._spend_so_far

        if cost > budget_remaining:
            info["error"] = f"Over budget. Remaining: ₹{budget_remaining:,.0f}, order costs ₹{cost:,.0f}"
            return -0.08, info

        self._spend_so_far += cost
        self._stock_snapshot[pid] = min(
            prod["max_stock"],
            self._stock_snapshot.get(pid, 0) + units
        )

        # Remove resolved alerts
        self._alerts = [a for a in self._alerts if not (
            a.product_id == pid and a.alert_type in ("STOCKOUT_RISK", "REORDER_NOW")
        )]

        info["reordered"] = {"product_id": pid, "units": units, "cost_inr": round(cost)}
        # Higher reward if addressing a CRITICAL alert
        was_critical = any(
            a.product_id == pid and a.severity == "CRITICAL"
            for a in generate_inventory_alerts(
                list(self._forecast_cache.values()), self._stock_snapshot
            )
        )
        reward = 0.18 if was_critical else 0.10
        return reward, info

    def _handle_markdown(self, action: SupplyChainAction, info: Dict) -> Tuple[float, Dict]:
        pid = action.product_id
        md  = action.markdown_pct or 0.0

        if pid not in self._cat_map:
            info["error"] = f"Unknown product {pid}"
            return -0.05, info
        if not (0.0 < md <= 0.40):
            info["error"] = "markdown_pct must be between 0.01 and 0.40"
            return -0.03, info

        fr  = self._forecast_cache.get(pid)
        prod = self._cat_map[pid]
        stock = self._stock_snapshot.get(pid, 0)

        if fr:
            plan = compute_markdown_plan(fr, stock, float(prod["price_inr"]))
            info["markdown_plan"] = {
                "new_price_inr": plan.new_price_inr,
                "weeks_to_clear": plan.weeks_to_clear_with_markdown,
                "revenue_impact": plan.revenue_impact_inr,
            }

        # Remove dead-stock alert
        self._alerts = [a for a in self._alerts if not (
            a.product_id == pid and a.alert_type == "DEAD_STOCK"
        )]

        reward = 0.12 if (fr and fr.dead_stock_risk == "HIGH") else 0.06
        return reward, info

    def _handle_bundle(self, action: SupplyChainAction, info: Dict) -> Tuple[float, Dict]:
        pid1 = action.product_id
        pid2 = action.bundle_with_id

        if not pid1 or not pid2:
            info["error"] = "create_bundle requires product_id and bundle_with_id"
            return -0.03, info

        p1 = self._cat_map.get(pid1)
        p2 = self._cat_map.get(pid2)
        if not p1 or not p2:
            info["error"] = "One or both product IDs not found"
            return -0.05, info

        bundle_price = int((p1["price_inr"] + p2["price_inr"]) * 0.90)
        info["bundle_created"] = {
            "products": [pid1, pid2],
            "bundle_price_inr": bundle_price,
            "savings_inr": p1["price_inr"] + p2["price_inr"] - bundle_price,
        }
        return 0.10, info

    def _handle_forecast_request(self, action: SupplyChainAction, info: Dict) -> Tuple[float, Dict]:
        pid = action.product_id
        if pid and pid in self._forecast_cache:
            fr = self._forecast_cache[pid]
            info["forecast"] = {
                "product_id":    fr.product_id,
                "product_name":  fr.product_name,
                "avg_weekly":    fr.avg_weekly_demand,
                "peak_week":     fr.peak_week,
                "peak_units":    fr.peak_units,
                "dead_stock_risk": fr.dead_stock_risk,
                "insights":      fr.insights,
                "next_4_weeks":  [
                    {"week": w.week_start, "units": w.predicted_units, "events": w.event_tags}
                    for w in fr.weekly_forecasts[:4]
                ],
            }
        else:
            # Return all forecasts summary
            info["forecast_summary"] = [
                {
                    "product_id": fr.product_id,
                    "product_name": fr.product_name,
                    "avg_weekly": fr.avg_weekly_demand,
                    "dead_stock_risk": fr.dead_stock_risk,
                    "urgency": fr.reorder_recommendation.get("urgency"),
                }
                for fr in self._forecast_cache.values()
            ]
        return 0.02, info

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_observation(self) -> SupplyChainObservation:
        # Inventory value
        total_val = sum(
            self._stock_snapshot.get(pid, 0) * self._cat_map[pid]["price_inr"]
            for pid in self._cat_map
        )
        dead_val = sum(
            max(0, self._stock_snapshot.get(fr.product_id, 0) - int(fr.avg_weekly_demand * 8))
            * self._cat_map[fr.product_id]["price_inr"]
            for fr in self._forecast_cache.values()
            if fr.dead_stock_risk in ("HIGH", "MEDIUM")
        )
        dead_pct = (dead_val / total_val * 100) if total_val > 0 else 0.0

        recent_forecasts = {
            pid: {
                "avg_weekly":      fr.avg_weekly_demand,
                "peak_week":       fr.peak_week,
                "peak_units":      fr.peak_units,
                "dead_stock_risk": fr.dead_stock_risk,
                "urgency":         fr.reorder_recommendation.get("urgency", "NORMAL"),
                "insights":        fr.insights[:2],
            }
            for pid, fr in list(self._forecast_cache.items())[:8]  # top 8 for brevity
        }

        return SupplyChainObservation(
            step_number=self._step_count,
            boutique_name=self._task_config["boutique_name"],
            stock_snapshot=dict(self._stock_snapshot),
            alerts=[{
                "product_id":   a.product_id,
                "product_name": a.product_name,
                "alert_type":   a.alert_type,
                "severity":     a.severity,
                "message":      a.message,
                "recommended_action": a.recommended_action,
            } for a in self._alerts],
            recent_forecasts=recent_forecasts,
            actions_taken=list(self._actions_taken),
            inventory_value_inr=round(total_val, 2),
            dead_stock_pct=round(dead_pct, 2),
            total_score=self.grade(),
            max_steps=self._task_config["max_steps"],
            done=self._done,
            task_description=self._task_config["description"],
        )
