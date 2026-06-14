"""
train.py — Master training script for Fashion OpenEnv ML models.

Usage:
    # Train both LSTM and PPO (server must be running)
    python train.py

    # Train only LSTM (no server needed)
    python train.py --model lstm

    # Train only PPO (server must be running at localhost:7860)
    python train.py --model ppo

    # Custom hyperparameters
    python train.py --model ppo --episodes 300 --lr 1e-4

    # Use GPU if available
    python train.py --device cuda

After training, checkpoints are saved to ml/checkpoints/
Training curves are saved to ml/checkpoints/training_history.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

def check_pytorch():
    try:
        import torch
        print(f"PyTorch {torch.__version__} found")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("Running on CPU")
        return True
    except ImportError:
        print("ERROR: PyTorch not installed.")
        print("Install it with: pip install torch")
        return False


def train_lstm(args):
    """Train the PyTorch LSTM demand forecasting model."""
    print("\n" + "="*60)
    print("LSTM DEMAND FORECASTING — PyTorch Training")
    print("="*60)

    from ml.forecasting.lstm_pytorch import LSTMTrainer
    from supply_chain.data.bangalore_data import generate_historical_sales, get_product_catalogue

    print("Generating 2 years of Bangalore boutique sales data...")
    records = generate_historical_sales(weeks=104)
    print(f"  {len(records)} weekly records for 20 products")
    print(f"  Sample keys: {list(records[0].keys())[:5]}")

    trainer = LSTMTrainer(
        hidden_size=args.hidden_size,
        lr=args.lr,
        device=args.device,
        checkpoint_dir="ml/checkpoints",
    )

    print(f"\nTraining on {args.lstm_epochs} epochs per item...")
    start = time.time()
    results = trainer.train_all(records, n_epochs=args.lstm_epochs, verbose=True)
    elapsed = time.time() - start

    print(f"\nTraining complete in {elapsed:.1f}s")
    print("\nSample metrics (first 5 items):")
    for r in results[:5]:
        if "error" not in r:
            print(f"  {r['item_id']} | Val loss: {r.get('best_val_loss', 'N/A')}")

    # Model comparison
    print("\nLSTM vs Holt-Winters comparison:")
    catalogue = get_product_catalogue()
    comparisons = []
    for prod in catalogue[:5]:  # compare first 5
        try:
            comp = trainer.compare_with_holt_winters(records, prod["product_id"])
            print(
                f"  {prod['product_id']} | LSTM MAE: {comp['lstm']['mae']:.2f} | "
                f"HW MAE: {comp['holt_winters']['mae']} | Winner: {comp['winner']}"
            )
            comparisons.append(comp)
        except Exception as e:
            print(f"  {prod['product_id']}: comparison failed ({e})")

    os.makedirs("ml/checkpoints", exist_ok=True)
    with open("ml/checkpoints/model_comparison.json", "w") as f:
        json.dump(comparisons, f, indent=2)
    print("\nModel comparison saved to ml/checkpoints/model_comparison.json")


def train_ppo(args):
    """Train the PyTorch PPO styling agent."""
    print("\n" + "="*60)
    print("PPO REINFORCEMENT LEARNING — PyTorch Training")
    print("="*60)

    # Check server is running
    import urllib.request
    try:
        with urllib.request.urlopen(f"{args.env_url}/health", timeout=5) as r:
            health = json.loads(r.read())
            print(f"Server health: {health.get('status')}")
    except Exception:
        print(f"ERROR: Server not reachable at {args.env_url}")
        print("Start the server first with: python app.py")
        return

    from ml.agents.ppo_pytorch import PPOTrainer

    task_ids = [
        "task_casual_budget",
        "task_office_ready",
        "task_gala_stylist",
        "task_street_party",
    ]

    trainer = PPOTrainer(
        env_url=args.env_url,
        task_ids=task_ids,
        lr=args.lr,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        device=args.device,
        checkpoint_dir="ml/checkpoints",
    )

    # Load existing checkpoint if available
    ckpt_path = "ml/checkpoints/ppo_final.pt"
    if os.path.exists(ckpt_path) and not args.fresh:
        trainer.load(ckpt_path)
        print(f"Resuming from episode {trainer.episode_count}")

    # Train
    history = trainer.train(
        n_episodes=args.episodes,
        episodes_per_update=args.episodes_per_update,
        log_every=args.log_every,
        save_every=args.save_every,
    )

    # Evaluate trained agent
    print("\nEvaluating trained PPO agent (10 greedy episodes per task)...")
    eval_results = trainer.evaluate(n_eval=10)

    print("\nEvaluation results:")
    print(f"{'Task':<30} {'Mean Score':>12} {'Std':>8} {'Max':>8}")
    print("─" * 60)
    scores = []
    for task_id, metrics in eval_results.items():
        print(
            f"  {task_id:<28} {metrics['mean_score']:>10.4f} "
            f"{metrics['std_score']:>8.4f} {metrics['max_score']:>8.4f}"
        )
        scores.append(metrics["mean_score"])

    avg = sum(scores) / len(scores)
    print(f"\n  Overall average: {avg:.4f}")
    print(f"\n  vs Rule-based baseline: 0.6250")
    improvement = (avg - 0.625) / 0.625 * 100
    print(f"  Improvement: {improvement:+.1f}%")

    # Save evaluation results
    eval_summary = {
        "ppo_results":    eval_results,
        "avg_score":      round(avg, 4),
        "vs_rule_based":  0.625,
        "improvement_pct": round(improvement, 2),
        "training_episodes": trainer.episode_count,
    }
    with open("ml/checkpoints/eval_results.json", "w") as f:
        json.dump(eval_summary, f, indent=2)
    print("Evaluation saved to ml/checkpoints/eval_results.json")


def main():
    parser = argparse.ArgumentParser(description="Fashion OpenEnv ML Training")
    parser.add_argument("--model", choices=["lstm", "ppo", "both"], default="both")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--episodes", type=int, default=200, help="PPO training episodes")
    parser.add_argument("--lstm-epochs", type=int, default=80, dest="lstm_epochs")
    parser.add_argument("--hidden-size", type=int, default=64, dest="hidden_size")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-ratio", type=float, default=0.2, dest="clip_ratio")
    parser.add_argument("--entropy-coef", type=float, default=0.01, dest="entropy_coef")
    parser.add_argument("--episodes-per-update", type=int, default=8, dest="episodes_per_update")
    parser.add_argument("--log-every", type=int, default=20, dest="log_every")
    parser.add_argument("--save-every", type=int, default=50, dest="save_every")
    parser.add_argument("--env-url", default="http://localhost:7860", dest="env_url")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints")
    args = parser.parse_args()

    if not check_pytorch():
        sys.exit(1)

    os.makedirs("ml/checkpoints", exist_ok=True)

    if args.model in ("lstm", "both"):
        train_lstm(args)

    if args.model in ("ppo", "both"):
        train_ppo(args)

    print("\n✓ Training complete. Checkpoints saved to ml/checkpoints/")
    print("  Start the server and visit localhost:7860/ml/agent/comparison")
    print("  to see live evaluation results.\n")


if __name__ == "__main__":
    main()