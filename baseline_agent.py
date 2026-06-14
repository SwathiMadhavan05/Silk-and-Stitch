#!/usr/bin/env python3
"""
baseline_agent.py
─────────────────
Baseline inference script for the Fashion Stylist OpenEnv.
Uses the OpenAI API client (GPT-4o) to run an LLM agent against all 3 tasks
and prints reproducible baseline scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline_agent.py [--base-url http://localhost:7860] [--model gpt-4o]

The agent uses a simple ReAct-style loop:
    1. Observe the current state
    2. Call the LLM with the observation as context
    3. Parse the action from the LLM response
    4. Execute the action via step()
    5. Repeat until done
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import requests

# ── OpenAI import guard ───────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not found. Run: pip install openai")
    sys.exit(1)


BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")
TASK_IDS = ["task_casual_budget", "task_office_ready", "task_gala_stylist"]

SYSTEM_PROMPT = """\
You are a professional fashion stylist agent operating inside a clothing simulation.
Your job is to build a complete outfit for a customer by adding, removing, or replacing
items from a fashion inventory.

At each step you must output ONLY a valid JSON object describing your next action.
Valid action types:
  - add_item         : {"action_type": "add_item", "item_id": "<ID>"}
  - remove_item      : {"action_type": "remove_item", "item_id": "<ID>"}
  - replace_item     : {"action_type": "replace_item", "item_id": "<OLD_ID>", "replacement_id": "<NEW_ID>"}
  - filter_inventory : {"action_type": "filter_inventory", "filter_criteria": {"category": "...", "max_price": ..., "season": "...", "style_tag": "..."}}
  - request_feedback : {"action_type": "request_feedback"}
  - finalize_outfit  : {"action_type": "finalize_outfit"}

Rules:
1. Always read the task_description and customer preferences carefully.
2. You must include a top, bottom, and shoes as a minimum.
3. Stay within the customer budget.
4. Match styles to the customer's occasion and season.
5. When you are satisfied with the outfit, call finalize_outfit.
6. Output ONLY the raw JSON — no markdown, no explanation.
"""


def api_call(endpoint: str, method: str = "GET", body: Optional[dict] = None) -> dict:
    url = f"{BASE_URL}{endpoint}"
    if method == "POST":
        resp = requests.post(url, json=body, timeout=30)
    else:
        resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_action(text: str) -> Optional[dict]:
    """Robustly extract a JSON action from LLM response."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object in the response
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def summarise_obs(obs: dict) -> str:
    """Convert observation dict to a concise string for the LLM."""
    c = obs["customer"]
    outfit_ids = obs["current_outfit"]
    inventory = obs["inventory"]

    # Only pass relevant subset to avoid token overflow
    by_id = {item["id"]: item for item in inventory}
    outfit_items = [
        f"  {iid}: {by_id[iid]['name']} ({by_id[iid]['category']}, "
        f"${by_id[iid]['price']}, {by_id[iid]['color']}, tags={by_id[iid]['style_tags']})"
        for iid in outfit_ids if iid in by_id
    ]

    # Suggest relevant items
    suggestions = [
        f"  {item['id']}: {item['name']} (${item['price']}, {item['color']}, "
        f"cat={item['category']}, tags={item['style_tags']}, season={item['season']})"
        for item in inventory[:25]  # first 25 to keep context manageable
    ]

    return f"""
TASK: {obs['task_description']}
STEP: {obs['step_number']} / {obs['max_steps']}
CUSTOMER: {c['name']} | Occasion: {c['occasion']} | Season: {c['season']}
  Budget remaining: ${obs['budget_remaining']} | Preferred styles: {c['preferred_styles']}
  Color prefs: {c['color_preferences']} | Dislikes: {c.get('disliked_items', [])}
CURRENT OUTFIT SCORE: {obs['outfit_score']:.3f}
CURRENT OUTFIT:
{chr(10).join(outfit_items) if outfit_items else '  (empty)'}
INVENTORY SAMPLE (first 25 items):
{chr(10).join(suggestions)}
"""


def run_task(client: OpenAI, model: str, task_id: str, verbose: bool = True) -> float:
    """Run one task episode and return the final graded score."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Task: {task_id}")
        print(f"{'='*60}")

    # Reset
    resp = api_call("/reset", "POST", {"task_id": task_id})
    obs = resp["observation"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    done = False
    step = 0

    while not done:
        step += 1
        obs_text = summarise_obs(obs)
        messages.append({"role": "user", "content": obs_text})

        # LLM decision
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=256,
        )
        action_text = completion.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": action_text})

        action_dict = parse_action(action_text)
        if action_dict is None:
            if verbose:
                print(f"  Step {step}: Could not parse action — finalizing.")
            action_dict = {"action_type": "finalize_outfit"}

        if verbose:
            print(f"  Step {step}: {action_dict}")

        # Execute
        step_resp = api_call("/step", "POST", {"task_id": task_id, "action": action_dict})
        obs = step_resp["observation"]
        reward = step_resp["reward"]
        done = step_resp["done"]

        if verbose:
            print(f"           reward={reward['value']:.4f} | {reward['reason'][:80]}")

        time.sleep(0.1)  # avoid rate-limiting

    # Grade
    grade_resp = api_call(f"/grade?task_id={task_id}")
    score = grade_resp["grade"]
    if verbose:
        print(f"\n  FINAL GRADE: {score:.4f}")
    return score


def main():
    parser = argparse.ArgumentParser(description="Fashion Stylist Baseline Agent")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.base_url

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    verbose = not args.quiet

    print(f"\nFashion Stylist OpenEnv — Baseline Agent ({args.model})")
    print(f"Server: {BASE_URL}\n")

    # Health check
    try:
        health = api_call("/health")
        print(f"Server health: {health}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {BASE_URL}: {e}")
        sys.exit(1)

    scores = {}
    for task_id in TASK_IDS:
        scores[task_id] = run_task(client, args.model, task_id, verbose=verbose)

    print("\n" + "="*60)
    print("BASELINE SCORES SUMMARY")
    print("="*60)
    for task_id, score in scores.items():
        difficulty = {"task_casual_budget": "Easy", "task_office_ready": "Medium",
                      "task_gala_stylist": "Hard"}.get(task_id, "")
        print(f"  {task_id:<30} [{difficulty:<6}]  {score:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  AVERAGE SCORE: {avg:.4f}")
    print("="*60)

    # Persist scores
    with open("baseline_scores.json", "w") as f:
        json.dump({"model": args.model, "scores": scores, "average": avg}, f, indent=2)
    print("\nScores saved to baseline_scores.json")


if __name__ == "__main__":
    main()
