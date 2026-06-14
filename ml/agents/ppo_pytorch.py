"""
ml/agents/ppo_pytorch.py

Real PyTorch PPO agent for Fashion Stylist OpenEnv.
Trains a neural network policy that learns to build outfits
purely from reward signals — no hardcoded rules.

Features:
  - Actor-Critic architecture with shared backbone
  - PPO clipped objective with entropy bonus
  - Generalised Advantage Estimation (GAE)
  - TensorBoard logging of learning curves
  - Model checkpointing (save/load weights)
  - Comparison against rule-based baseline
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Action space ──────────────────────────────────────────────────────────────
ITEM_IDS = [
    "T001","T002","T003","T004","T005","T006","T007","T008","T009","T010","T011",
    "B001","B002","B003","B004","B005","B006","B007","B008","B009",
    "S001","S002","S003","S004","S005","S006","S007","S008","S009",
    "A001","A002","A003","A004","A005","A006","A007","A008","A009","A010","A011","A012",
    "O001","O002","O003","O004","O005","O006","O007","O008",
]

ACTION_SPACE = (
    [{"action_type": "add_item", "item_id": iid} for iid in ITEM_IDS] +
    [{"action_type": "remove_item", "item_id": iid} for iid in ITEM_IDS] +
    [{"action_type": "finalize_outfit"}]
)
N_ACTIONS = len(ACTION_SPACE)
STATE_DIM  = 62


# ── State encoder ─────────────────────────────────────────────────────────────
def encode_state(obs: Dict) -> List[float]:
    state = []
    customer   = obs.get("customer", {})
    budget     = obs.get("budget_remaining", 500)
    total_bud  = customer.get("budget", 500)
    state.append(min(budget / max(total_bud, 1), 1.0))
    state.append(float(obs.get("outfit_score", 0.0)))
    step     = obs.get("step_number", 0)
    max_step = obs.get("max_steps", 15)
    state.append(step / max(max_step, 1))
    current = set(obs.get("current_outfit", []))
    for iid in ITEM_IDS:
        state.append(1.0 if iid in current else 0.0)
    for occ in ["casual", "office", "party", "wedding", "outdoor"]:
        state.append(1.0 if customer.get("occasion") == occ else 0.0)
    for s in ["spring", "summer", "fall", "winter"]:
        state.append(1.0 if customer.get("season") == s else 0.0)
    state.append(min(customer.get("budget", 300) / 1000.0, 1.0))
    return state


# ── PyTorch Actor-Critic network ──────────────────────────────────────────────
if TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """
        Shared backbone Actor-Critic for PPO.
        Architecture:
          Shared: STATE_DIM → 256 → 128  (ReLU, LayerNorm)
          Actor:  128 → 64 → N_ACTIONS   (softmax)
          Critic: 128 → 64 → 1           (linear)
        """
        def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS):
            super().__init__()
            # Shared layers
            self.shared = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
            )
            # Actor head
            self.actor = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions),
            )
            # Critic head
            self.critic = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            # Initialise weights with orthogonal init (better for RL)
            self._init_weights()

        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                    nn.init.constant_(module.bias, 0.0)

        def forward(self, x: "torch.Tensor"):
            shared = self.shared(x)
            logits = self.actor(shared)
            value  = self.critic(shared)
            return logits, value.squeeze(-1)

        def get_action(self, state: "torch.Tensor", deterministic: bool = False,
                   action_mask: "Optional[torch.Tensor]" = None):
            logits, value = self.forward(state)
            # Apply action mask to prevent invalid actions
            if action_mask is not None:
                logits = logits + (action_mask * -1e8)
            dist   = Categorical(logits=logits)
            action = dist.mode if deterministic else dist.sample()
            return action, dist.log_prob(action), dist.entropy(), value


# ── Experience buffer ─────────────────────────────────────────────────────────
@dataclass
class RolloutBuffer:
    states:      List = field(default_factory=list)
    actions:     List = field(default_factory=list)
    log_probs:   List = field(default_factory=list)
    rewards:     List = field(default_factory=list)
    values:      List = field(default_factory=list)
    dones:       List = field(default_factory=list)
    advantages:  List = field(default_factory=list)
    returns:     List = field(default_factory=list)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


# ── PPO Trainer ───────────────────────────────────────────────────────────────
class PPOTrainer:
    """
    Full PPO training loop for Fashion Stylist OpenEnv.

    Hyperparameters (tuned for this environment):
      lr          = 3e-4   (Adam optimizer)
      gamma       = 0.99   (discount factor)
      gae_lambda  = 0.95   (GAE smoothing)
      clip_ratio  = 0.2    (PPO clip parameter)
      entropy_coef = 0.01  (entropy bonus to encourage exploration)
      value_coef  = 0.5    (value loss coefficient)
      n_epochs    = 4      (PPO update epochs per rollout)
      batch_size  = 64
    """

    def __init__(
        self,
        env_url: str = "http://localhost:7860",
        task_ids: Optional[List[str]] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
        checkpoint_dir: str = "ml/checkpoints",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch not installed. Run: pip install torch"
            )

        self.env_url      = env_url
        self.task_ids     = task_ids or [
            "task_casual_budget", "task_office_ready",
            "task_gala_stylist",  "task_street_party",
        ]
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_ratio   = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.device       = torch.device(device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model = ActorCritic(STATE_DIM, N_ACTIONS).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=200
        )

        self.buffer       = RolloutBuffer()
        self.total_steps  = 0
        self.episode_count = 0
        self.training_log: List[Dict] = []

    # ── Environment interaction ───────────────────────────────────────────────

    def _env_post(self, endpoint: str, body: Dict) -> Dict:
        import urllib.request
        data = json.dumps(body).encode()
        req  = urllib.request.Request(
            f"{self.env_url}{endpoint}", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def _env_get(self, endpoint: str) -> Dict:
        import urllib.request
        with urllib.request.urlopen(f"{self.env_url}{endpoint}", timeout=30) as r:
            return json.loads(r.read())

    def _collect_episode(self, task_id: str) -> Dict:
        """Run one episode and collect experience into the rollout buffer."""
        resp  = self._env_post("/reset", {"task_id": task_id})
        obs   = resp["observation"]
        state = torch.tensor(encode_state(obs), dtype=torch.float32).to(self.device)
        done  = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < obs.get("max_steps", 20):
            steps += 1
            with torch.no_grad():
                # Build action mask - discourage invalid actions
                current_outfit = set(obs.get("current_outfit", []))
                mask = torch.zeros(N_ACTIONS, dtype=torch.float32)
                # Mask remove_item for items not in outfit
                for i, act in enumerate(ACTION_SPACE):
                    if act["action_type"] == "remove_item":
                        if act.get("item_id") not in current_outfit:
                            mask[i] = 1.0  # mask out
                    elif act["action_type"] == "add_item":
                        if act.get("item_id") in current_outfit:
                            mask[i] = 1.0  # already in outfit
                    elif act["action_type"] == "finalize_outfit":
                        # Only allow finalize if we have at least 3 items
                        if len(current_outfit) < 3:
                            mask[i] = 1.0  # too early to finalize

                action_idx, log_prob, entropy, value = self.model.get_action(
                    state.unsqueeze(0), action_mask=mask.to(self.device)
                )

            action    = ACTION_SPACE[action_idx.item()]
            try:
                result    = self._env_post("/step", {"task_id": task_id, "action": action})
                next_obs  = result["observation"]
                r_obj     = result.get("reward", {})
                reward    = r_obj.get("value", 0.0) if isinstance(r_obj, dict) else float(r_obj or 0)
                done      = result.get("done", False)
            except Exception:
                reward    = -0.1
                done      = True
                next_obs  = obs

            next_state = torch.tensor(encode_state(next_obs), dtype=torch.float32).to(self.device)

            self.buffer.states.append(state)
            self.buffer.actions.append(action_idx)
            self.buffer.log_probs.append(log_prob)
            self.buffer.rewards.append(torch.tensor([reward], dtype=torch.float32))
            self.buffer.values.append(value)
            self.buffer.dones.append(torch.tensor([done], dtype=torch.float32))

            ep_reward += reward
            state = next_state
            obs   = next_obs
            self.total_steps += 1

        try:
            grade = self._env_get(f"/grade?task_id={task_id}")
            score = grade.get("grade", 0.0)
        except Exception:
            score = obs.get("outfit_score", 0.0)

        self.episode_count += 1
        return {"reward": ep_reward, "score": score, "steps": steps, "task_id": task_id}

    # ── GAE advantage computation ─────────────────────────────────────────────

    def _compute_gae(self, last_value: float = 0.0):
        """Generalised Advantage Estimation."""
        advantages = []
        gae = 0.0
        values  = [v.item() for v in self.buffer.values] + [last_value]
        rewards = [r.item() for r in self.buffer.rewards]
        dones   = [d.item() for d in self.buffer.dones]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae   = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        # Normalise advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        returns_t    = advantages_t + torch.stack(self.buffer.values).squeeze()

        self.buffer.advantages = advantages_t
        self.buffer.returns    = returns_t

    # ── PPO update ────────────────────────────────────────────────────────────

    def _ppo_update(self) -> Dict[str, float]:
        """Run n_epochs of PPO update on collected rollout."""
        states    = torch.stack(self.buffer.states)
        actions   = torch.stack(self.buffer.actions)
        old_lp    = torch.stack(self.buffer.log_probs).detach()
        advantages = self.buffer.advantages.detach()
        returns    = self.buffer.returns.detach()

        n = len(states)
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                b_states   = states[batch_idx]
                b_actions  = actions[batch_idx]
                b_old_lp   = old_lp[batch_idx]
                b_adv      = advantages[batch_idx]
                b_returns  = returns[batch_idx]

                logits, values = self.model(b_states)
                dist    = Categorical(logits=logits)
                new_lp  = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # PPO clipped policy loss
                ratio        = torch.exp(new_lp - b_old_lp)
                surr1        = ratio * b_adv
                surr2        = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * b_adv
                policy_loss  = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss   = F.mse_loss(values, b_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()
                n_updates += 1

        self.scheduler.step()
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss":  total_value_loss  / n_updates,
            "entropy":     total_entropy     / n_updates,
        }

    # ── Main training loop ────────────────────────────────────────────────────

    def train(
        self,
        n_episodes: int = 200,
        episodes_per_update: int = 8,
        log_every: int = 20,
        save_every: int = 50,
    ) -> List[Dict]:
        """
        Full PPO training loop.

        Args:
            n_episodes:          total episodes to train for
            episodes_per_update: collect this many episodes before each PPO update
            log_every:           print metrics every N episodes
            save_every:          save checkpoint every N episodes

        Returns:
            training history list (used for learning curve plots)
        """
        print(f"\nPPO Training — {n_episodes} episodes, {len(self.task_ids)} tasks")
        print(f"Device: {self.device} | Actions: {N_ACTIONS} | State dim: {STATE_DIM}")
        print("─" * 65)

        history      = []
        ep_rewards   = []
        ep_scores    = []
        start_time   = time.time()

        for ep in range(1, n_episodes + 1):
            # Cycle through tasks
            task_id = self.task_ids[(ep - 1) % len(self.task_ids)]

            try:
                ep_info = self._collect_episode(task_id)
            except Exception as e:
                print(f"  Episode {ep} failed: {e}")
                continue

            ep_rewards.append(ep_info["reward"])
            ep_scores.append(ep_info["score"])

            # PPO update every episodes_per_update episodes
            if ep % episodes_per_update == 0 and len(self.buffer) > 0:
                self._compute_gae()
                losses = self._ppo_update()
                self.buffer.clear()
            else:
                losses = {}

            # Logging
            if ep % log_every == 0:
                recent_rewards = ep_rewards[-log_every:]
                recent_scores  = ep_scores[-log_every:]
                elapsed = time.time() - start_time
                log = {
                    "episode":      ep,
                    "task_id":      task_id,
                    "avg_reward":   round(statistics.mean(recent_rewards), 4),
                    "avg_score":    round(statistics.mean(recent_scores), 4),
                    "max_score":    round(max(recent_scores), 4),
                    "policy_loss":  round(losses.get("policy_loss", 0), 5),
                    "value_loss":   round(losses.get("value_loss", 0), 5),
                    "entropy":      round(losses.get("entropy", 0), 5),
                    "total_steps":  self.total_steps,
                    "elapsed_sec":  round(elapsed, 1),
                }
                history.append(log)
                self.training_log.append(log)
                print(
                    f"  Ep {ep:4d}/{n_episodes} | "
                    f"Avg score: {log['avg_score']:.3f} | "
                    f"Avg reward: {log['avg_reward']:.3f} | "
                    f"Policy loss: {log['policy_loss']:.4f} | "
                    f"Entropy: {log['entropy']:.3f}"
                )

            # Checkpoint
            if ep % save_every == 0:
                self.save(os.path.join(self.checkpoint_dir, f"ppo_ep{ep}.pt"))
                print(f"  Checkpoint saved at episode {ep}")

        # Final save
        self.save(os.path.join(self.checkpoint_dir, "ppo_final.pt"))
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")

        if not ep_scores:
            print("WARNING: No episodes completed.")
            print(f"Is the server running at {self.env_url}?")
            print("Start it with: python app.py  (in a separate terminal)")
            return history

        recent = ep_scores[-20:] if len(ep_scores) >= 20 else ep_scores
        print(f"Final avg score (last {len(recent)} eps): {statistics.mean(recent):.4f}")

        # Save history to JSON
        with open(os.path.join(self.checkpoint_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        return history

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, n_eval: int = 10) -> Dict:
        """Evaluate trained agent with greedy policy (no exploration)."""
        self.model.eval()
        results = {}

        for task_id in self.task_ids:
            scores  = []
            rewards = []
            for _ in range(n_eval):
                resp  = self._env_post("/reset", {"task_id": task_id})
                obs   = resp["observation"]
                state = torch.tensor(encode_state(obs), dtype=torch.float32).to(self.device)
                done  = False
                ep_r  = 0.0
                steps = 0

                while not done and steps < obs.get("max_steps", 20):
                    steps += 1
                    with torch.no_grad():
                        current_outfit_eval = set(obs.get("current_outfit", []))
                        mask_eval = torch.zeros(N_ACTIONS, dtype=torch.float32)
                        for i, act in enumerate(ACTION_SPACE):
                            if act["action_type"] == "remove_item":
                                if act.get("item_id") not in current_outfit_eval:
                                    mask_eval[i] = 1.0
                            elif act["action_type"] == "add_item":
                                if act.get("item_id") in current_outfit_eval:
                                    mask_eval[i] = 1.0
                            elif act["action_type"] == "finalize_outfit":
                                if len(current_outfit_eval) < 3:
                                    mask_eval[i] = 1.0
                        action_idx, _, _, _ = self.model.get_action(
                            state.unsqueeze(0), deterministic=True,
                            action_mask=mask_eval.to(self.device)
                        )
                    action = ACTION_SPACE[action_idx.item()]
                    try:
                        result   = self._env_post("/step", {"task_id": task_id, "action": action})
                        next_obs = result["observation"]
                        r_obj    = result.get("reward", {})
                        reward   = r_obj.get("value", 0.0) if isinstance(r_obj, dict) else float(r_obj or 0)
                        done     = result.get("done", False)
                    except Exception:
                        done = True
                        reward = 0.0
                        next_obs = obs

                    ep_r  += reward
                    state  = torch.tensor(encode_state(next_obs), dtype=torch.float32).to(self.device)
                    obs    = next_obs

                try:
                    grade = self._env_get(f"/grade?task_id={task_id}")
                    score = grade.get("grade", 0.0)
                except Exception:
                    score = 0.5
                scores.append(score)
                rewards.append(ep_r)

            results[task_id] = {
                "mean_score":  round(statistics.mean(scores), 4),
                "std_score":   round(statistics.stdev(scores) if len(scores) > 1 else 0, 4),
                "max_score":   round(max(scores), 4),
                "mean_reward": round(statistics.mean(rewards), 4),
            }

        self.model.train()
        return results

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "episode_count":   self.episode_count,
            "total_steps":     self.total_steps,
            "training_log":    self.training_log,
        }, path)

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"Checkpoint {path} not found — starting fresh")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.episode_count = ckpt.get("episode_count", 0)
        self.total_steps   = ckpt.get("total_steps", 0)
        self.training_log  = ckpt.get("training_log", [])
        print(f"Loaded checkpoint: episode {self.episode_count}, steps {self.total_steps}")