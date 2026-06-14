"""
FashionStylistEnv — OpenEnv-compliant environment for the Fashion Stylist task.
Implements: step(), reset(), state()
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

from fashion_env.models import Action, Observation, Reward
from fashion_env.inventory import INVENTORY, INVENTORY_BY_ID
from fashion_env.reward import compute_reward
from fashion_env.tasks import TASKS


class FashionStylistEnv:
    """
    An OpenEnv environment that simulates a fashion styling agent.

    The agent selects clothing items from an inventory to build an outfit
    that satisfies a customer profile's preferences, occasion, budget,
    and seasonal requirements.

    Supported action types:
        add_item         — add an item (by id) to the current outfit
        remove_item      — remove an item from the current outfit
        replace_item     — swap one item for another
        filter_inventory — get a filtered view of items (no state change)
        finalize_outfit  — end the episode and receive final score
        request_feedback — receive natural-language reward feedback
    """

    def __init__(self, task_id: str = "task_casual_budget"):
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}"
            )
        self.task_id = task_id
        self._task_config, self._grader = TASKS[task_id]
        self._obs: Optional[Observation] = None
        self._current_outfit_ids: list[str] = []
        self._step_count: int = 0
        self._done: bool = False
        self._last_score: float = 0.0

    # ─────────────────────────── Public API ──────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment to its initial state and return the first observation."""
        self._current_outfit_ids = []
        self._step_count = 0
        self._done = False
        self._last_score = 0.0
        self._obs = self._build_observation()
        return self._obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.

        Returns:
            observation  — updated environment state
            reward       — Reward model with value and breakdown
            done         — whether the episode has ended
            info         — auxiliary diagnostic info
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        if self._obs is None:
            self.reset()

        self._step_count += 1
        info: Dict[str, Any] = {"step": self._step_count, "action": action.action_type}

        # ── Dispatch action ───────────────────────────────────────────────
        if action.action_type == "add_item":
            info = {**info, **self._handle_add(action)}

        elif action.action_type == "remove_item":
            info = {**info, **self._handle_remove(action)}

        elif action.action_type == "replace_item":
            info = {**info, **self._handle_replace(action)}

        elif action.action_type == "filter_inventory":
            # Read-only — returns filtered list in info, no outfit change
            info["filtered"] = self._handle_filter(action)

        elif action.action_type == "finalize_outfit":
            self._done = True
            info["finalized"] = True

        elif action.action_type == "request_feedback":
            info["feedback"] = "Use reward.reason for detailed feedback."

        else:
            info["error"] = f"Unknown action_type: '{action.action_type}'"

        # Check step limit
        max_steps = self._task_config["max_steps"]
        if self._step_count >= max_steps:
            self._done = True
            info["step_limit_reached"] = True

        # ── Compute reward ────────────────────────────────────────────────
        current_items = [INVENTORY_BY_ID[i] for i in self._current_outfit_ids]
        reward = compute_reward(
            items=current_items,
            customer=self._task_config["customer"],
            previous_score=self._last_score,
            is_final=self._done,
        )
        self._last_score = reward.breakdown.get("composite", self._last_score)

        # ── Build observation ─────────────────────────────────────────────
        obs = self._build_observation(
            feedback=reward.reason if action.action_type == "request_feedback" else None
        )
        obs.outfit_score = reward.breakdown.get("composite", 0.0)
        obs.done = self._done
        self._obs = obs

        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return current raw environment state as a plain dict."""
        customer = self._task_config["customer"]
        items = [INVENTORY_BY_ID[i] for i in self._current_outfit_ids]
        return {
            "task_id": self.task_id,
            "step": self._step_count,
            "done": self._done,
            "customer_id": customer.id,
            "customer_name": customer.name,
            "occasion": customer.occasion,
            "season": customer.season,
            "budget": customer.budget,
            "outfit_item_ids": list(self._current_outfit_ids),
            "outfit_total_price": round(sum(i.price for i in items), 2),
            "outfit_score": self._last_score,
        }

    # ─────────────────────────── Helpers ─────────────────────────────────

    def _build_observation(self, feedback: Optional[str] = None) -> Observation:
        customer = self._task_config["customer"]
        budget_used = sum(
            INVENTORY_BY_ID[i].price for i in self._current_outfit_ids
            if i in INVENTORY_BY_ID
        )
        safe_score = round(max(0.001, min(float(self._last_score), 0.999)), 4) if self._last_score > 0 else 0.001
        return Observation(
            step_number=self._step_count,
            customer=customer,
            inventory=copy.deepcopy(INVENTORY),
            current_outfit=list(self._current_outfit_ids),
            outfit_score=safe_score,
            budget_remaining=round(customer.budget - budget_used, 2),
            feedback_history=[feedback] if feedback else [],
            task_id=self.task_id,
            task_description=self._task_config["description"],
            max_steps=self._task_config["max_steps"],
            done=self._done,
        )

    def _handle_add(self, action: Action) -> Dict[str, Any]:
        item_id = action.item_id
        if item_id is None:
            return {"error": "add_item requires item_id"}
        if item_id not in INVENTORY_BY_ID:
            return {"error": f"Item '{item_id}' not found in inventory"}
        if item_id in self._current_outfit_ids:
            return {"warning": f"Item '{item_id}' already in outfit"}
        self._current_outfit_ids.append(item_id)
        return {"added": item_id}

    def _handle_remove(self, action: Action) -> Dict[str, Any]:
        item_id = action.item_id
        if item_id is None:
            return {"error": "remove_item requires item_id"}
        if item_id not in self._current_outfit_ids:
            return {"warning": f"Item '{item_id}' not in current outfit"}
        self._current_outfit_ids.remove(item_id)
        return {"removed": item_id}

    def _handle_replace(self, action: Action) -> Dict[str, Any]:
        old_id = action.item_id
        new_id = action.replacement_id
        if old_id is None or new_id is None:
            return {"error": "replace_item requires item_id and replacement_id"}
        if old_id not in self._current_outfit_ids:
            return {"error": f"'{old_id}' not in current outfit"}
        if new_id not in INVENTORY_BY_ID:
            return {"error": f"'{new_id}' not in inventory"}
        idx = self._current_outfit_ids.index(old_id)
        self._current_outfit_ids[idx] = new_id
        return {"replaced": old_id, "with": new_id}

    def _handle_filter(self, action: Action) -> list:
        criteria = action.filter_criteria or {}
        results = []
        for item in INVENTORY:
            if criteria.get("category") and item.category != criteria["category"]:
                continue
            if criteria.get("max_price") and item.price > criteria["max_price"]:
                continue
            if criteria.get("season") and criteria["season"] not in item.season:
                continue
            if criteria.get("style_tag"):
                if criteria["style_tag"] not in item.style_tags:
                    continue
            if criteria.get("color") and item.color.lower() != criteria["color"].lower():
                continue
            results.append(item.model_dump())
        return results

    def grade(self) -> float:
        """Run the task grader and return a score strictly between 0 and 1."""
        items = [INVENTORY_BY_ID[i] for i in self._current_outfit_ids]
        customer = self._task_config["customer"]
        raw = self._grader(items, customer)
        return round(max(0.001, min(float(raw), 0.999)), 4)
