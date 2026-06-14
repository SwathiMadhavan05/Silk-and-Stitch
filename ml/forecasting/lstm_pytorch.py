"""
ml/forecasting/lstm_pytorch.py

Real PyTorch LSTM demand forecasting model.
Trained on 2 years of Indian fashion sales data.
Compares against Holt-Winters baseline with proper metrics.

Architecture:
  Input:  [sales, festival_flag, season_flag, trend, category_encoding]
  LSTM:   2 layers × 64 hidden units with dropout
  Output: next-week demand prediction

Features:
  - Multi-variate input (5 features)
  - 2-layer stacked LSTM with dropout
  - Monte Carlo dropout for prediction intervals
  - Festival-aware: learns Diwali/Navratri spike patterns
  - MAE, RMSE, MAPE metrics vs Holt-Winters
  - Model checkpointing
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Dict, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Festival calendar ─────────────────────────────────────────────────────────
FESTIVAL_CALENDAR = {
    (10, 1,  31): 0.9,   # Navratri/Dussehra
    (11, 1,  15): 1.0,   # Diwali
    (12, 20, 31): 0.6,   # Christmas/NYE
    (1,  10, 16): 0.7,   # Pongal
    (3,  20, 31): 0.6,   # Ugadi/Holi
    (5,  1,  31): 0.8,   # Wedding season
    (9,  1,  30): 0.7,   # Wedding season
}

CATEGORIES = ["saree", "lehenga", "kurta", "salwar", "western", "accessories"]
CATEGORY_IDX = {c: i for i, c in enumerate(CATEGORIES)}

INPUT_SIZE = 5  # [sales, festival, season, trend, category_onehot_avg]
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
SEQ_LEN     = 12  # look back 12 weeks


def get_festival_flag(d: date) -> float:
    for (m, ds, de), intensity in FESTIVAL_CALENDAR.items():
        if d.month == m and ds <= d.day <= de:
            return intensity
    return 0.0


def get_season_flag(d: date) -> float:
    m = d.month
    if m in (12, 1, 2):  return 0.0
    if m in (3,  4, 5):  return 0.33
    if m in (6,  7, 8):  return 0.66
    return 1.0


# ── PyTorch LSTM Model ────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """
        2-layer stacked LSTM for demand forecasting.
        Uses MC Dropout for uncertainty estimation.
        """
        def __init__(
            self,
            input_size:  int = INPUT_SIZE,
            hidden_size: int = HIDDEN_SIZE,
            num_layers:  int = NUM_LAYERS,
            dropout:     float = 0.2,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers  = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout  = nn.Dropout(dropout)
            self.fc1      = nn.Linear(hidden_size, 32)
            self.relu     = nn.ReLU()
            self.fc2      = nn.Linear(32, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            lstm_out, _ = self.lstm(x)
            out = lstm_out[:, -1, :]    # take last timestep
            out = self.dropout(out)
            out = self.relu(self.fc1(out))
            out = self.fc2(out)
            return out.squeeze(-1)


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_sequences(
    records: List[Dict],
    item_id: str,
    seq_len: int = SEQ_LEN,
) -> Tuple[List, List, float, float, str]:
    """
    Convert raw sales records into (X, y) sequences for LSTM training.
    Returns normalisation parameters for denormalisation.
    """
    # Support both "item_id" (India dataset) and "product_id" (Bangalore dataset)
    id_key = "item_id" if records and "item_id" in records[0] else "product_id"
    name_key = "item_name" if records and "item_name" in records[0] else "product_name"

    item_records = sorted(
        [r for r in records if r[id_key] == item_id],
        key=lambda r: r["week_start"]
    )
    if len(item_records) < seq_len + 4:
        return [], [], 0.0, 1.0, "unknown"

    category = item_records[0].get("category", "western")
    sales    = [float(r["units_sold"]) for r in item_records]
    dates    = [r["week_start"] for r in item_records]

    # Normalise sales
    mean_s = statistics.mean(sales)
    std_s  = statistics.stdev(sales) if len(sales) > 1 else 1.0
    std_s  = max(std_s, 1e-6)
    norm_s = [(s - mean_s) / std_s for s in sales]

    X, y = [], []
    for i in range(len(norm_s) - seq_len):
        seq_x = []
        for j in range(seq_len):
            d = date.fromisoformat(dates[i + j])
            trend = (i + j) / len(norm_s)
            feast = get_festival_flag(d)
            seas  = get_season_flag(d)
            cat_f = CATEGORY_IDX.get(category, 0) / len(CATEGORIES)
            seq_x.append([norm_s[i + j], feast, seas, trend, cat_f])
        X.append(seq_x)
        y.append(norm_s[i + seq_len])

    return X, y, mean_s, std_s, category


# ── LSTM Trainer ──────────────────────────────────────────────────────────────

class LSTMTrainer:
    """
    Trains and evaluates the PyTorch LSTM demand forecasting model.
    """

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        num_layers:  int = NUM_LAYERS,
        dropout:     float = 0.2,
        lr:          float = 1e-3,
        device:      str = "cpu",
        checkpoint_dir: str = "ml/checkpoints",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Run: pip install torch")

        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model = LSTMModel(INPUT_SIZE, hidden_size, num_layers, dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.criterion = nn.HuberLoss()  # robust to outliers
        self._norm_params: Dict[str, Dict] = {}

    def train_item(
        self,
        records: List[Dict],
        item_id: str,
        n_epochs: int = 100,
        val_split: float = 0.15,
        verbose: bool = False,
    ) -> Dict:
        """Train on one item's historical data."""
        X, y, mean_s, std_s, category = prepare_sequences(records, item_id)
        if not X:
            return {"error": "insufficient data"}

        self._norm_params[item_id] = {"mean": mean_s, "std": std_s}

        # Train/val split
        split     = max(1, int(len(X) * (1 - val_split)))
        X_train   = torch.tensor(X[:split], dtype=torch.float32).to(self.device)
        y_train   = torch.tensor(y[:split], dtype=torch.float32).to(self.device)
        X_val     = torch.tensor(X[split:], dtype=torch.float32).to(self.device)
        y_val     = torch.tensor(y[split:], dtype=torch.float32).to(self.device)

        best_val_loss = float("inf")
        train_losses  = []
        val_losses    = []

        for epoch in range(n_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            pred  = self.model(X_train)
            loss  = self.criterion(pred, y_train)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            train_losses.append(loss.item())

            # Validation
            if len(X_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val)
                    val_loss = self.criterion(val_pred, y_val).item()
                val_losses.append(val_loss)
                self.scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            if verbose and (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Train loss: {train_losses[-1]:.5f} | "
                      f"Val loss: {val_losses[-1] if val_losses else 0:.5f}")

        return {
            "item_id":        item_id,
            "category":       category,
            "epochs":         n_epochs,
            "final_train_loss": round(train_losses[-1], 6),
            "best_val_loss":  round(best_val_loss, 6),
            "norm_mean":      mean_s,
            "norm_std":       std_s,
        }

    def train_all(
        self,
        records: List[Dict],
        n_epochs: int = 80,
        verbose: bool = True,
    ) -> List[Dict]:
        """Train on all items in the dataset."""
        id_key = "item_id" if records and "item_id" in records[0] else "product_id"
        item_ids = list({r[id_key] for r in records})
        results  = []
        print(f"\nTraining LSTM on {len(item_ids)} items...")
        for i, item_id in enumerate(item_ids):
            result = self.train_item(records, item_id, n_epochs, verbose=False)
            results.append(result)
            if verbose:
                status = f"Val loss: {result.get('best_val_loss', 'N/A')}"
                print(f"  [{i+1:2d}/{len(item_ids)}] {item_id} | {status}")

        self.save(os.path.join(self.checkpoint_dir, "lstm_final.pt"))
        return results

    def forecast(
        self,
        records: List[Dict],
        item_id: str,
        horizon: int = 12,
        n_mc_samples: int = 30,
    ) -> Dict:
        """
        Generate probabilistic forecast using MC Dropout.
        Returns mean prediction + 95% confidence intervals.
        """
        X, y, mean_s, std_s, category = prepare_sequences(records, item_id)
        if not X:
            return {"error": "insufficient data for item"}

        params = self._norm_params.get(item_id, {"mean": mean_s, "std": std_s})
        last_seq = torch.tensor([X[-1]], dtype=torch.float32).to(self.device)

        # MC Dropout — keep dropout active for uncertainty
        self.model.train()
        all_preds = []
        start_date = date.today() + timedelta(weeks=1)

        for _ in range(n_mc_samples):
            seq = last_seq.clone()
            sample_preds = []
            with torch.no_grad():
                for fw in range(horizon):
                    pred_norm = self.model(seq).item()
                    pred_raw  = pred_norm * params["std"] + params["mean"]
                    sample_preds.append(max(0.0, pred_raw))

                    # Slide window: drop oldest, add new prediction
                    fw_date = start_date + timedelta(weeks=fw)
                    feast   = get_festival_flag(fw_date)
                    seas    = get_season_flag(fw_date)
                    trend   = (len(X) + fw) / (len(X) + horizon)
                    cat_f   = CATEGORY_IDX.get(category, 0) / len(CATEGORIES)
                    new_step = torch.tensor(
                        [[[pred_norm, feast, seas, trend, cat_f]]],
                        dtype=torch.float32
                    ).to(self.device)
                    seq = torch.cat([seq[:, 1:, :], new_step], dim=1)
            all_preds.append(sample_preds)

        self.model.eval()

        # Aggregate MC samples
        predictions  = []
        lower_bounds = []
        upper_bounds = []
        festival_weeks = []

        for fw in range(horizon):
            week_preds = sorted([all_preds[s][fw] for s in range(n_mc_samples)])
            mean_p = statistics.mean(week_preds)
            # 95% CI from MC samples
            lo = week_preds[int(0.025 * n_mc_samples)]
            hi = week_preds[int(0.975 * n_mc_samples)]
            predictions.append(round(mean_p, 2))
            lower_bounds.append(round(lo, 2))
            upper_bounds.append(round(hi, 2))

            fw_date = start_date + timedelta(weeks=fw)
            if get_festival_flag(fw_date) > 0:
                festival_weeks.append(fw + 1)

        # Compute metrics on held-out data
        mae, rmse, mape = self._compute_metrics(records, item_id, params)

        # Insights
        avg_pred = statistics.mean(predictions)
        insights = []
        if festival_weeks:
            peak = max(predictions[w-1] for w in festival_weeks)
            insights.append(
                f"Festival demand spike in weeks {festival_weeks[:3]}: peak {peak:.0f} units. "
                f"Stock up {len(festival_weeks)} weeks before the event window."
            )
        hist_sales = [float(r["units_sold"]) for r in records if r["item_id"] == item_id]
        if hist_sales:
            trend_pct = (avg_pred - statistics.mean(hist_sales[-8:])) / max(statistics.mean(hist_sales[-8:]), 1) * 100
            if trend_pct > 10:
                insights.append(f"Demand trending UP {trend_pct:.0f}% vs recent history.")
            elif trend_pct < -10:
                insights.append(f"Demand trending DOWN {abs(trend_pct):.0f}% — consider promotions.")

        id_key2   = "item_id" if records and "item_id" in records[0] else "product_id"
        name_key2 = "item_name" if records and "item_name" in records[0] else "product_name"
        item_name = next((r[name_key2] for r in records if r[id_key2] == item_id), item_id)
        return {
            "item_id":       item_id,
            "item_name":     item_name,
            "category":      category,
            "model":         "LSTM-PyTorch",
            "horizon_weeks": horizon,
            "predictions":   predictions,
            "lower_bound":   lower_bounds,
            "upper_bound":   upper_bounds,
            "festival_weeks":festival_weeks,
            "metrics": {"mae": mae, "rmse": rmse, "mape": mape},
            "insights":      insights,
        }

    def _compute_metrics(
        self,
        records: List[Dict],
        item_id: str,
        params: Dict,
    ) -> Tuple[float, float, float]:
        """Compute MAE, RMSE, MAPE on last 8 weeks of history."""
        X, y, _, _, _ = prepare_sequences(records, item_id, SEQ_LEN)
        if len(X) < 8:
            return 0.0, 0.0, 0.0

        test_X = torch.tensor(X[-8:], dtype=torch.float32).to(self.device)
        test_y = [yi * params["std"] + params["mean"] for yi in y[-8:]]

        self.model.eval()
        with torch.no_grad():
            preds_norm = self.model(test_X).tolist()
        preds = [max(0, p * params["std"] + params["mean"]) for p in preds_norm]

        n    = len(preds)
        mae  = sum(abs(preds[i] - test_y[i]) for i in range(n)) / n
        rmse = math.sqrt(sum((preds[i] - test_y[i])**2 for i in range(n)) / n)
        mape = sum(abs(preds[i] - test_y[i]) / max(abs(test_y[i]), 1) for i in range(n)) / n * 100

        return round(mae, 3), round(rmse, 3), round(mape, 2)

    def compare_with_holt_winters(
        self,
        records: List[Dict],
        item_id: str,
    ) -> Dict:
        """Compare LSTM vs Holt-Winters metrics on same item."""
        from supply_chain.models.forecaster import forecast_product

        lstm_result = self.forecast(records, item_id)
        lstm_metrics = lstm_result.get("metrics", {})

        try:
            hw = forecast_product(item_id, historical_records=records)
            hw_preds  = [wf.predicted_units for wf in hw.weekly_forecasts[:8]]
            hw_actuals = [
                float(r["units_sold"])
                for r in sorted(records, key=lambda x: x["week_start"])
                if r["item_id"] == item_id
            ][-8:]
            n  = min(len(hw_preds), len(hw_actuals))
            hw_mae  = sum(abs(hw_preds[i] - hw_actuals[i]) for i in range(n)) / n
            hw_rmse = math.sqrt(sum((hw_preds[i] - hw_actuals[i])**2 for i in range(n)) / n)
        except Exception:
            hw_mae, hw_rmse = None, None

        lstm_mae  = lstm_metrics.get("mae", 0)
        lstm_rmse = lstm_metrics.get("rmse", 0)

        winner = "LSTM" if hw_mae and lstm_mae < hw_mae else "Holt-Winters"
        improvement = round((hw_mae - lstm_mae) / max(hw_mae, 1e-6) * 100, 1) if hw_mae else None

        return {
            "item_id": item_id,
            "lstm": {
                "mae":  lstm_mae,
                "rmse": lstm_rmse,
                "mape": lstm_metrics.get("mape", 0),
            },
            "holt_winters": {
                "mae":  round(hw_mae, 3) if hw_mae else "N/A",
                "rmse": round(hw_rmse, 3) if hw_rmse else "N/A",
            },
            "winner":       winner,
            "improvement_pct": improvement,
            "verdict": (
                f"LSTM reduces MAE by {improvement:.1f}% vs Holt-Winters "
                if improvement and improvement > 0
                else "Models perform similarly on this item"
            ),
        }

    def save(self, path: str):
        torch.save({
            "model_state":  self.model.state_dict(),
            "norm_params":  self._norm_params,
        }, path)
        print(f"  LSTM checkpoint saved: {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self._norm_params = ckpt.get("norm_params", {})
        print(f"  LSTM loaded from: {path}")