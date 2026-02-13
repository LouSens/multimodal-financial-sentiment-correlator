"""
train.py — Production-Ready Training Pipeline

Features
────────
  • Reproducible runs           (full seed locking)
  • Auto device detection       (CUDA → MPS → CPU)
  • Batched DataLoaders         (train / val / test from preprocessor)
  • AdamW + Cosine LR schedule  (with warm restarts)
  • Gradient clipping           (max_norm = 1.0)
  • Early stopping              (patience-based on val loss)
  • Best-model checkpointing    (saves weights + scaler + config)
  • Ticker-specific folders     (checkpoints/AAPL/, checkpoints/TSLA/, etc.)
  • Rich logging & tqdm bars
  • CLI via argparse
"""

import os
import time
import random
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from data_loader import MarketDataLoader
from preprocessor import DataPreprocessor
from model import MultiModalNet

# ───────────────────────────────────────────────────────────────────── #
#  Logging
# ───────────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────── #
#  Paths
# ───────────────────────────────────────────────────────────────────── #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


def get_ticker_dir(ticker: str) -> str:
    """Return the ticker-specific checkpoint folder, e.g. checkpoints/AAPL/"""
    d = os.path.join(CHECKPOINT_DIR, ticker.upper())
    os.makedirs(d, exist_ok=True)
    return d


# ===================================================================== #
#  Utilities
# ===================================================================== #
def set_seed(seed: int = 42):
    """Lock every source of randomness for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"  🔒 Seed locked → {seed}")


def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    logger.info(f"  🖥️  Device → {dev}")
    return dev


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ===================================================================== #
#  Training Loop
# ===================================================================== #
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    """Single training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch).flatten()
        loss = criterion(preds, y_batch)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Validation / test evaluation. Returns (mse_loss, mae_loss)."""
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model(X_batch).flatten()
        total_mse += criterion(preds, y_batch).item()
        total_mae += torch.mean(torch.abs(preds - y_batch)).item()
        n_batches += 1

    avg_mse = total_mse / n_batches if n_batches > 0 else float("nan")
    avg_mae = total_mae / n_batches if n_batches > 0 else float("nan")
    return avg_mse, avg_mae


def save_checkpoint(model, processor, config, path):
    """Bundle model weights + scaler state + config into a single file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, path)
    # Also save the scaler alongside (uses processor._scaler_path)
    processor.save_scaler()
    logger.info(f"  💾 Checkpoint saved → {path}")


# ===================================================================== #
#  Main Training Function
# ===================================================================== #
def train_model(
    ticker: str = "AAPL",
    days: int = 90,
    seq_length: int = 24,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    grad_clip: float = 1.0,
    patience: int = 15,
    seed: int = 42,
):
    """
    End-to-end training pipeline.

    Saves all artifacts to checkpoints/{TICKER}/:
      • best_model.pth   — model weights + config
      • scaler.joblib     — fitted preprocessor scaler
      • model_weights.pth — legacy flat weights

    Returns
    -------
    model : MultiModalNet         (best checkpoint, loaded)
    processor : DataPreprocessor  (fitted scaler)
    history : dict                (train_loss, val_mse, val_mae per epoch)
    """
    # ── Setup ────────────────────────────────────────────────────────────
    set_seed(seed)
    device = get_device()

    # Ticker-specific output folder
    ticker_dir = get_ticker_dir(ticker)
    logger.info(f"  📁 Output folder → {ticker_dir}")

    logger.info("━" * 60)
    logger.info(f"  🏋️  TRAINING  |  ticker={ticker}  epochs={epochs}  "
                f"batch={batch_size}  lr={lr}")
    logger.info("━" * 60)

    config = {
        "ticker": ticker, "days": days, "seq_length": seq_length,
        "batch_size": batch_size, "epochs": epochs, "lr": lr,
        "weight_decay": weight_decay, "hidden_dim": hidden_dim,
        "num_layers": num_layers, "dropout": dropout,
        "grad_clip": grad_clip, "patience": patience, "seed": seed,
    }

    # ── 1. Load Data ─────────────────────────────────────────────────────
    logger.info("  📥 Loading market data...")
    loader = MarketDataLoader(ticker)
    df = loader.get_aligned_data(days=days)
    logger.info(f"  📊 Raw data: {df.shape[0]} rows × {df.shape[1]} cols")

    # ── 2. Preprocess → DataLoaders ──────────────────────────────────────
    processor = DataPreprocessor(seq_length=seq_length)
    # Point scaler persistence to ticker-specific folder
    processor._scaler_path = os.path.join(ticker_dir, "scaler.joblib")

    train_loader, val_loader, test_loader, feature_names = processor.get_dataloaders(
        df, batch_size=batch_size, seq_length=seq_length
    )
    logger.info(f"  🔧 Features ({len(feature_names)}): {feature_names}")

    # ── 3. Initialise Model ──────────────────────────────────────────────
    model = MultiModalNet(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stop = EarlyStopping(patience=patience)

    # ── 4. Training Loop ─────────────────────────────────────────────────
    best_val_mse = float("inf")
    best_path = os.path.join(ticker_dir, "best_model.pth")
    history = {"train_loss": [], "val_mse": [], "val_mae": [], "lr": []}

    logger.info("")
    logger.info("  Epoch │ Train Loss │  Val MSE  │  Val MAE  │   LR      │ Status")
    logger.info("  ──────┼────────────┼───────────┼───────────┼───────────┼────────")

    t_start = time.time()

    # Detect if val/test are empty
    has_val = len(val_loader.dataset) > 0
    has_test = len(test_loader.dataset) > 0
    if not has_val:
        logger.info("  ⚠️  No validation set — using train loss for checkpointing")

    for epoch in range(1, epochs + 1):
        # — Train —
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip
        )

        # — Validate (or use train loss as proxy) —
        if has_val:
            val_mse, val_mae = evaluate(model, val_loader, criterion, device)
        else:
            val_mse, val_mae = train_loss, train_loss  # proxy

        # — Scheduler step (epoch-based for warm restarts) —
        scheduler.step(epoch - 1)
        current_lr = optimizer.param_groups[0]["lr"]

        # — Record history —
        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)
        history["lr"].append(current_lr)

        # — Best model checkpoint —
        status = ""
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            save_checkpoint(model, processor, config, best_path)
            status = "✅ saved"

        # — Logging —
        val_label = "Train*" if not has_val else "Val"
        logger.info(
            f"  {epoch:>5d} │ {train_loss:>10.6f} │ {val_mse:>9.6f} │ "
            f"{val_mae:>9.6f} │ {current_lr:>9.6f} │ {status}"
        )

        # — Early stopping —
        if early_stop.step(val_mse):
            logger.info(f"\n  ⏹️  Early stopping triggered at epoch {epoch} "
                        f"(patience={patience})")
            break

    elapsed = time.time() - t_start
    logger.info(f"\n  ⏱️  Training finished in {elapsed:.1f}s  "
                f"({epoch} epochs)")

    # ── 5. Load Best Model & Test ────────────────────────────────────────
    if os.path.exists(best_path):
        logger.info("\n  📈 Loading best checkpoint for final evaluation...")
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.info("\n  ⚠️  No checkpoint found — using final epoch weights")
        save_checkpoint(model, processor, config, best_path)

    if has_test:
        test_mse, test_mae = evaluate(model, test_loader, criterion, device)
        test_mse_str = f"{test_mse:.6f}"
        test_mae_str = f"{test_mae:.6f}"
    else:
        test_mse_str = "N/A (no test set)"
        test_mae_str = "N/A (no test set)"

    logger.info("  ┌──────────────────────────────────────┐")
    logger.info(f"  │  TEST MSE  : {test_mse_str:<22s} │")
    logger.info(f"  │  TEST MAE  : {test_mae_str:<22s} │")
    logger.info(f"  │  Best Val  : {best_val_mse:.6f}                │")
    logger.info("  └──────────────────────────────────────┘")

    # Also save flat weights in the ticker folder
    legacy_path = os.path.join(ticker_dir, "model_weights.pth")
    torch.save(model.state_dict(), legacy_path)
    logger.info(f"  💾 Legacy weights → {legacy_path}")
    logger.info(f"  📁 All files saved in: {ticker_dir}")

    return model, processor, history


# ===================================================================== #
#  CLI Entry Point
# ===================================================================== #
def parse_args():
    p = argparse.ArgumentParser(
        description="Train the Multi-Modal Sentiment-Price model"
    )
    p.add_argument("--ticker",       type=str,   default="AAPL",  help="Stock ticker")
    p.add_argument("--days",         type=int,   default=90,      help="Days of history")
    p.add_argument("--seq-length",   type=int,   default=24,      help="LSTM sequence length")
    p.add_argument("--batch-size",   type=int,   default=32,      help="Batch size")
    p.add_argument("--epochs",       type=int,   default=100,     help="Max training epochs")
    p.add_argument("--lr",           type=float, default=1e-3,    help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4,    help="AdamW weight decay")
    p.add_argument("--hidden-dim",   type=int,   default=64,      help="Hidden dim for LSTM/GRU")
    p.add_argument("--num-layers",   type=int,   default=2,       help="Stacked recurrent layers")
    p.add_argument("--dropout",      type=float, default=0.3,     help="Dropout rate")
    p.add_argument("--grad-clip",    type=float, default=1.0,     help="Gradient clipping norm")
    p.add_argument("--patience",     type=int,   default=15,      help="Early stopping patience")
    p.add_argument("--seed",         type=int,   default=42,      help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, processor, history = train_model(
        ticker=args.ticker,
        days=args.days,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        patience=args.patience,
        seed=args.seed,
    )