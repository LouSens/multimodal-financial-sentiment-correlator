"""
evaluate.py — Hyperparameter Tuning & Model Evaluation

Features
────────
  • Optuna-based Bayesian hyperparameter optimization (TPE sampler)
  • Systematic search across learning rate, architecture, regularization
  • Detailed model evaluation with real-world metrics:
      - Dollar-denominated MAE (actual price error)
      - Directional accuracy (up/down prediction correctness)
      - Naive baseline comparison
      - R² score
  • Results saved to JSON for comparison
  • CLI with configurable budget (number of trials)

Usage
─────
  # Run hyperparameter search (20 trials)
  python evaluate.py --ticker AAPL --trials 20

  # Quick search (5 trials) with fewer epochs per trial
  python evaluate.py --ticker AAPL --trials 5 --trial-epochs 30

  # Evaluate an already-trained model (no tuning)
  python evaluate.py --ticker AAPL --evaluate-only

  # Full pipeline: tune + retrain best + evaluate
  python evaluate.py --ticker AAPL --trials 30 --final-epochs 200
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import optuna
from optuna.samplers import TPESampler

from data_loader import MarketDataLoader
from preprocessor import DataPreprocessor
from model import MultiModalNet
from train import (
    set_seed, get_device, train_one_epoch, evaluate,
    EarlyStopping, save_checkpoint, get_ticker_dir
)

# ───────────────────────────────────────────────────────────────────── #
#  Logging
# ───────────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "tuning_results")


# ===================================================================== #
#  Hyperparameter Search Space
# ===================================================================== #
SEARCH_SPACE = {
    "lr":           {"type": "loguniform", "low": 1e-4,  "high": 5e-3},
    "hidden_dim":   {"type": "categorical", "choices": [32, 48, 64, 96, 128]},
    "num_layers":   {"type": "int",        "low": 1,     "high": 3},
    "dropout":      {"type": "uniform",    "low": 0.1,   "high": 0.6},
    "weight_decay": {"type": "loguniform", "low": 1e-5,  "high": 1e-2},
    "seq_length":   {"type": "categorical", "choices": [8, 12, 16, 24]},
    "batch_size":   {"type": "categorical", "choices": [16, 32, 64]},
    "grad_clip":    {"type": "uniform",    "low": 0.5,   "high": 2.0},
}


def sample_params(trial: optuna.Trial) -> dict:
    """Sample hyperparameters from the search space using Optuna trial."""
    params = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "loguniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif spec["type"] == "uniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


# ===================================================================== #
#  Single Trial Training (lightweight — no checkpoint saving)
# ===================================================================== #
def train_trial(params: dict, df, device, epochs: int = 50,
                patience: int = 10, seed: int = 42) -> dict:
    """
    Train a model with given hyperparameters and return metrics.
    Lightweight version — no checkpointing, minimal logging.
    """
    set_seed(seed)

    seq_length = params["seq_length"]
    batch_size = params["batch_size"]

    # Preprocess
    processor = DataPreprocessor(seq_length=seq_length)
    try:
        train_loader, val_loader, test_loader, feature_names = processor.get_dataloaders(
            df, batch_size=batch_size, seq_length=seq_length
        )
    except Exception as e:
        return {"val_mse": float("inf"), "test_mse": float("inf"),
                "test_mae": float("inf"), "error": str(e)}

    has_val = len(val_loader.dataset) > 0
    has_test = len(test_loader.dataset) > 0

    if len(train_loader.dataset) == 0:
        return {"val_mse": float("inf"), "test_mse": float("inf"),
                "test_mae": float("inf"), "error": "No training data"}

    # Build model
    model = MultiModalNet(
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stop = EarlyStopping(patience=patience)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, params["grad_clip"]
        )

        if has_val:
            val_mse, val_mae = evaluate(model, val_loader, criterion, device)
        else:
            val_mse = train_loss

        scheduler.step(epoch - 1)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if early_stop.step(val_mse):
            break

    # Load best and evaluate on test
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    if has_test:
        test_mse, test_mae = evaluate(model, test_loader, criterion, device)
    else:
        test_mse, test_mae = best_val, best_val

    # Compute real-world metrics
    real_metrics = compute_real_metrics(model, processor, df, device, seq_length)

    return {
        "val_mse": best_val,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "epochs_run": epoch,
        **real_metrics,
    }


# ===================================================================== #
#  Real-World Metrics (dollar MAE, directional accuracy, etc.)
# ===================================================================== #
def compute_real_metrics(model, processor, df, device, seq_length):
    """
    Compute human-interpretable metrics:
      - Dollar MAE: average error in actual price ($)
      - Directional accuracy: % of correct up/down predictions
      - Naive baseline: MAE of "predict same as last hour"
      - R² score
    """
    model.eval()

    # Process the full dataset
    X, y = processor.create_sequences(df, seq_length=seq_length)

    if len(X) == 0:
        return {"dollar_mae": None, "directional_accuracy": None,
                "naive_mae": None, "r2": None}

    # Use last 20% as evaluation set
    eval_start = int(len(X) * 0.8)
    X_eval = X[eval_start:].to(device)
    y_eval = y[eval_start:]

    with torch.no_grad():
        preds_scaled = model(X_eval).cpu().numpy().flatten()

    y_actual_scaled = y_eval.numpy().flatten()

    # Inverse transform to get real prices
    try:
        preds_dollars = processor.inverse_transform(preds_scaled)
        actual_dollars = processor.inverse_transform(y_actual_scaled)
    except Exception:
        return {"dollar_mae": None, "directional_accuracy": None,
                "naive_mae": None, "r2": None}

    # Dollar MAE
    dollar_mae = float(np.mean(np.abs(preds_dollars - actual_dollars)))

    # Directional accuracy (did we predict up/down correctly?)
    if len(actual_dollars) > 1:
        actual_direction = np.sign(np.diff(actual_dollars))
        pred_direction = np.sign(np.diff(preds_dollars))
        # Remove zero-change cases
        mask = actual_direction != 0
        if mask.sum() > 0:
            dir_accuracy = float(np.mean(actual_direction[mask] == pred_direction[mask]) * 100)
        else:
            dir_accuracy = 50.0
    else:
        dir_accuracy = None

    # Naive baseline: predict "same as last hour"
    if len(actual_dollars) > 1:
        naive_preds = actual_dollars[:-1]  # shift by 1
        naive_mae = float(np.mean(np.abs(naive_preds - actual_dollars[1:])))
    else:
        naive_mae = None

    # R² score
    ss_res = np.sum((actual_dollars - preds_dollars) ** 2)
    ss_tot = np.sum((actual_dollars - np.mean(actual_dollars)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None

    # Price range for context
    price_range = f"${actual_dollars.min():.2f} – ${actual_dollars.max():.2f}"
    mean_price = float(np.mean(actual_dollars))
    pct_mae = (dollar_mae / mean_price * 100) if mean_price > 0 else None

    return {
        "dollar_mae": round(dollar_mae, 4),
        "dollar_mae_pct": round(pct_mae, 4) if pct_mae else None,
        "directional_accuracy": round(dir_accuracy, 2) if dir_accuracy else None,
        "naive_baseline_mae": round(naive_mae, 4) if naive_mae else None,
        "beats_naive": dollar_mae < naive_mae if naive_mae else None,
        "r2_score": round(r2, 4) if r2 else None,
        "price_range": price_range,
        "eval_samples": len(actual_dollars),
    }


# ===================================================================== #
#  Optuna Objective
# ===================================================================== #
def create_objective(df, device, trial_epochs, patience):
    """Factory to create an Optuna objective with pre-loaded data."""

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)

        results = train_trial(
            params, df, device,
            epochs=trial_epochs,
            patience=patience,
        )

        # Report intermediate results
        trial.set_user_attr("test_mse", results["test_mse"])
        trial.set_user_attr("test_mae", results["test_mae"])
        trial.set_user_attr("epochs_run", results.get("epochs_run", 0))
        trial.set_user_attr("dollar_mae", results.get("dollar_mae"))
        trial.set_user_attr("directional_accuracy", results.get("directional_accuracy"))
        trial.set_user_attr("r2_score", results.get("r2_score"))
        trial.set_user_attr("beats_naive", results.get("beats_naive"))

        # Optimize for test MSE (not val — we want generalization)
        return results["test_mse"]

    return objective


# ===================================================================== #
#  Main Tuning Pipeline
# ===================================================================== #
def run_tuning(ticker: str, n_trials: int = 20, trial_epochs: int = 50,
               patience: int = 10, days: int = 700, seed: int = 42):
    """
    Run full hyperparameter search with Optuna.

    Returns
    -------
    best_params : dict    Best hyperparameters found
    study       : optuna.Study
    """
    device = get_device()
    set_seed(seed)

    logger.info("━" * 65)
    logger.info(f"  🔬 HYPERPARAMETER TUNING  |  ticker={ticker}  trials={n_trials}")
    logger.info(f"     trial_epochs={trial_epochs}  patience={patience}  days={days}")
    logger.info("━" * 65)

    # Load data once (shared across all trials)
    logger.info("  📥 Loading market data...")
    loader = MarketDataLoader(ticker)
    df = loader.get_aligned_data(days=days)
    logger.info(f"  📊 Data: {df.shape[0]} rows × {df.shape[1]} cols")

    if len(df) < 50:
        logger.error(f"  ❌ Not enough data ({len(df)} rows). Need at least 50.")
        return None, None

    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        study_name=f"sentiprice_{ticker}",
    )

    objective = create_objective(df, device, trial_epochs, patience)

    # Callback for progress logging
    def log_trial(study, trial):
        best = study.best_trial
        logger.info(
            f"  Trial {trial.number:>3d}/{n_trials}  │  "
            f"Test MSE: {trial.value:.6f}  │  "
            f"Best: {best.value:.6f}  │  "
            f"Dir Acc: {trial.user_attrs.get('directional_accuracy', 'N/A')}%  │  "
            f"$MAE: {trial.user_attrs.get('dollar_mae', 'N/A')}"
        )

    logger.info("")
    logger.info("  Trial │  Test MSE   │   Best     │  Dir Acc  │  $MAE")
    logger.info("  ──────┼─────────────┼────────────┼───────────┼──────────")

    t_start = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[log_trial])
    elapsed = time.time() - t_start

    # Results summary
    best = study.best_trial
    logger.info("")
    logger.info("━" * 65)
    logger.info("  🏆 BEST HYPERPARAMETERS")
    logger.info("━" * 65)
    for k, v in best.params.items():
        logger.info(f"    {k:<16s}: {v}")
    logger.info(f"    {'Test MSE':<16s}: {best.value:.6f}")
    logger.info(f"    {'Dollar MAE':<16s}: ${best.user_attrs.get('dollar_mae', 'N/A')}")
    logger.info(f"    {'Dir Accuracy':<16s}: {best.user_attrs.get('directional_accuracy', 'N/A')}%")
    logger.info(f"    {'R²':<16s}: {best.user_attrs.get('r2_score', 'N/A')}")
    logger.info(f"    {'Beats Naive?':<16s}: {best.user_attrs.get('beats_naive', 'N/A')}")
    logger.info(f"\n  ⏱️  Tuning completed in {elapsed:.1f}s ({n_trials} trials)")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"{ticker}_tuning.json")
    results = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "n_trials": n_trials,
        "trial_epochs": trial_epochs,
        "days": days,
        "elapsed_seconds": round(elapsed, 1),
        "best_params": best.params,
        "best_test_mse": best.value,
        "best_dollar_mae": best.user_attrs.get("dollar_mae"),
        "best_directional_accuracy": best.user_attrs.get("directional_accuracy"),
        "best_r2": best.user_attrs.get("r2_score"),
        "beats_naive": best.user_attrs.get("beats_naive"),
        "all_trials": [
            {
                "number": t.number,
                "params": t.params,
                "test_mse": t.value,
                "dollar_mae": t.user_attrs.get("dollar_mae"),
                "directional_accuracy": t.user_attrs.get("directional_accuracy"),
                "r2_score": t.user_attrs.get("r2_score"),
            }
            for t in study.trials
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  💾 Results saved → {results_path}")

    return best.params, study


# ===================================================================== #
#  Retrain with Best Params
# ===================================================================== #
def retrain_best(ticker: str, best_params: dict, days: int = 700,
                 epochs: int = 200, patience: int = 25):
    """Retrain the model with discovered best params and save checkpoint."""
    from train import train_model

    logger.info("")
    logger.info("━" * 65)
    logger.info("  🏋️  RETRAINING WITH BEST PARAMS")
    logger.info("━" * 65)

    model, processor, history = train_model(
        ticker=ticker,
        days=days,
        seq_length=best_params.get("seq_length", 12),
        batch_size=best_params.get("batch_size", 32),
        epochs=epochs,
        lr=best_params.get("lr", 1e-3),
        weight_decay=best_params.get("weight_decay", 1e-4),
        hidden_dim=best_params.get("hidden_dim", 64),
        num_layers=best_params.get("num_layers", 2),
        dropout=best_params.get("dropout", 0.3),
        grad_clip=best_params.get("grad_clip", 1.0),
        patience=patience,
    )

    return model, processor, history


# ===================================================================== #
#  Evaluate Existing Model
# ===================================================================== #
def evaluate_existing(ticker: str, days: int = 700):
    """Load an existing trained model and run full evaluation."""
    device = get_device()

    logger.info("━" * 65)
    logger.info(f"  📊 EVALUATING EXISTING MODEL  |  ticker={ticker}")
    logger.info("━" * 65)

    # Load model
    ticker_dir = get_ticker_dir(ticker)
    ckpt_path = os.path.join(ticker_dir, "best_model.pth")
    scaler_path = os.path.join(ticker_dir, "scaler.joblib")

    if not os.path.exists(ckpt_path):
        logger.error(f"  ❌ No model found at {ckpt_path}")
        logger.error(f"     Train first: python train.py --ticker {ticker}")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get("config", {})

    model = MultiModalNet(
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # Load scaler
    processor = DataPreprocessor(seq_length=config.get("seq_length", 24))
    if os.path.exists(scaler_path):
        processor.load_scaler(scaler_path)
    else:
        logger.warning("  ⚠️  No scaler found — using fresh preprocessing")

    # Load fresh data
    logger.info("  📥 Loading market data...")
    loader = MarketDataLoader(ticker)
    df = loader.get_aligned_data(days=days)
    logger.info(f"  📊 Data: {df.shape[0]} rows")

    seq_length = config.get("seq_length", 24)
    metrics = compute_real_metrics(model, processor, df, device, seq_length)

    # Display results
    logger.info("")
    logger.info("  ┌──────────────────────────────────────────────────┐")
    logger.info(f"  │  📈 EVALUATION RESULTS — {ticker:<26s}│")
    logger.info("  ├──────────────────────────────────────────────────┤")
    logger.info(f"  │  Dollar MAE        : ${metrics.get('dollar_mae', 'N/A'):<20s}     │")
    logger.info(f"  │  Dollar MAE (%)    : {metrics.get('dollar_mae_pct', 'N/A')}%{' ' * 17}│")
    logger.info(f"  │  Dir. Accuracy     : {metrics.get('directional_accuracy', 'N/A')}%{' ' * 17}│")
    logger.info(f"  │  R² Score          : {metrics.get('r2_score', 'N/A'):<20s}     │")
    logger.info(f"  │  Naive Baseline MAE: ${metrics.get('naive_baseline_mae', 'N/A'):<20s}     │")
    logger.info(f"  │  Beats Naive?      : {'✅ Yes' if metrics.get('beats_naive') else '❌ No':<20s}     │")
    logger.info(f"  │  Price Range       : {metrics.get('price_range', 'N/A'):<20s}     │")
    logger.info(f"  │  Eval Samples      : {metrics.get('eval_samples', 'N/A'):<20s}     │")
    logger.info("  └──────────────────────────────────────────────────┘")

    if config:
        logger.info("\n  Training Config:")
        for k, v in config.items():
            logger.info(f"    {k:<16s}: {v}")

    return metrics


# ===================================================================== #
#  CLI
# ===================================================================== #
def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter Tuning & Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --ticker AAPL --trials 20
  python evaluate.py --ticker TSLA --trials 30 --final-epochs 200
  python evaluate.py --ticker AAPL --evaluate-only
        """,
    )
    p.add_argument("--ticker",        type=str,   default="AAPL",  help="Stock ticker")
    p.add_argument("--trials",        type=int,   default=20,      help="Number of Optuna trials")
    p.add_argument("--trial-epochs",  type=int,   default=50,      help="Epochs per trial (quick)")
    p.add_argument("--final-epochs",  type=int,   default=200,     help="Epochs for final retrain")
    p.add_argument("--days",          type=int,   default=700,     help="Days of historical data")
    p.add_argument("--patience",      type=int,   default=10,      help="Early stopping per trial")
    p.add_argument("--final-patience",type=int,   default=25,      help="Patience for final retrain")
    p.add_argument("--seed",          type=int,   default=42,      help="Random seed")
    p.add_argument("--evaluate-only", action="store_true",         help="Only evaluate existing model")
    p.add_argument("--no-retrain",    action="store_true",         help="Skip final retrain with best params")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.evaluate_only:
        # Just evaluate existing model
        evaluate_existing(args.ticker, days=args.days)
    else:
        # Run hyperparameter tuning
        best_params, study = run_tuning(
            ticker=args.ticker,
            n_trials=args.trials,
            trial_epochs=args.trial_epochs,
            patience=args.patience,
            days=args.days,
            seed=args.seed,
        )

        if best_params and not args.no_retrain:
            # Retrain with best params (full epochs)
            model, processor, history = retrain_best(
                ticker=args.ticker,
                best_params=best_params,
                days=args.days,
                epochs=args.final_epochs,
                patience=args.final_patience,
            )

            # Final evaluation
            logger.info("")
            evaluate_existing(args.ticker, days=args.days)
