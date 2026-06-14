"""
inference.py — Fashion Stylist OpenEnv inference script.
Follows exact OpenEnv hackathon output format from guidelines PDF.
"""
import os
import json
import requests
from openai import OpenAI

# ── Required environment variables (with defaults where allowed) ──────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── OpenAI client using HF_TOKEN as api_key, pointed at proxy ────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ── Our environment server on HF Spaces ──────────────────────────────────────
ENV_URL   = "https://swathi-01-fashion-openenv.hf.space"
ENV_NAME  = "fashion-openenv"

TASKS = [
    "task_casual_budget",
    "task_office_ready",
    "task_gala_stylist",
]

SYSTEM_PROMPT = """You are a fashion stylist AI agent building outfits for customers.

At each step output ONLY a single JSON object. Valid actions:
{"action_type": "add_item", "item_id": "<ID>"}
{"action_type": "remove_item", "item_id": "<ID>"}
{"action_type": "finalize_outfit"}

Rules:
- Must include a top, bottom, and shoes at minimum
- Stay within the customer budget
- Match styles to the occasion and season
- Call finalize_outfit when the outfit is complete

Output ONLY raw JSON. No markdown. No explanation."""


def env_post(endpoint, body):
    r = requests.post(f"{ENV_URL}{endpoint}", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def env_get(endpoint):
    r = requests.get(f"{ENV_URL}{endpoint}", timeout=60)
    r.raise_for_status()
    return r.json()


def get_llm_action(obs_text):
    """Call LLM through the proxy and parse action."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": obs_text},
            ],
            max_tokens=100,
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        return json.loads(text), None
    except Exception as e:
        return {"action_type": "finalize_outfit"}, str(e)


def obs_to_text(obs, task_id):
    c      = obs.get("customer", {})
    outfit = obs.get("current_outfit", [])
    budget = obs.get("budget_remaining", 0)
    inv    = obs.get("inventory", [])[:12]
    inv_lines = [
        f"  {i['id']}: {i['name']} (${i['price']}, {i['category']}, tags={i['style_tags']})"
        for i in inv
    ]
    return (
        f"Task: {obs.get('task_description', task_id)}\n"
        f"Customer: {c.get('name')} | Occasion: {c.get('occasion')} | Season: {c.get('season')}\n"
        f"Budget remaining: ${budget}\n"
        f"Current outfit: {outfit if outfit else 'empty'}\n"
        f"Inventory:\n" + "\n".join(inv_lines) +
        "\n\nOutput your next action as JSON:"
    )


def run_task(task_id):
    # ── [START] line — exact format from PDF ─────────────────────────────────
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    rewards    = []
    step       = 0
    last_error = None
    success    = False

    try:
        # Reset
        data = env_post("/reset", {"task_id": task_id})
        obs  = data.get("observation", {})
        max_steps = obs.get("max_steps", 15)
        done = False

        while not done and step < max_steps:
            step += 1

            # LLM decides action
            obs_text      = obs_to_text(obs, task_id)
            action, error = get_llm_action(obs_text)
            last_error    = error

            try:
                result = env_post("/step", {"task_id": task_id, "action": action})
                obs    = result.get("observation", {})
                r      = result.get("reward", {})
                reward = r.get("value", 0.0) if isinstance(r, dict) else float(r or 0)
                done   = result.get("done", False)
                rewards.append(reward)
                error_str = "null" if error is None else error

                # ── [STEP] line — exact format from PDF ───────────────────────
                print(
                    f"[STEP] step={step} action={action.get('action_type')} "
                    f"reward={reward:.2f} done={'true' if done else 'false'} "
                    f"error={error_str}",
                    flush=True
                )

                if done:
                    success = True

            except Exception as e:
                last_error = str(e)
                rewards.append(0.0)
                print(
                    f"[STEP] step={step} action={action.get('action_type','unknown')} "
                    f"reward=0.00 done=false error={last_error}",
                    flush=True
                )
                break

    except Exception as e:
        last_error = str(e)

    # ── [END] line — exact format from PDF ───────────────────────────────────
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step} rewards={rewards_str}",
        flush=True
    )

    return sum(rewards) / len(rewards) if rewards else 0.0


def main():
    for task_id in TASKS:
        try:
            run_task(task_id)
        except Exception as e:
            print(
                f"[END] success=false steps=0 rewards=0.00",
                flush=True
            )


if __name__ == "__main__":
    main()
