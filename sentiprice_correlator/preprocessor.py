"""
preprocessor.py — Production-Ready Data Preprocessing Pipeline

Handles:
  • Cleaning (NaN, Inf, outliers)
  • Feature engineering (returns, volatility, sentiment momentum, etc.)
  • Robust scaling (outlier-resilient)
  • Sequential train / val / test splitting (no data leakage)
  • PyTorch DataLoader creation for batched training
  • Scaler persistence (save / load) for reproducible inference
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

# Default path for persisting the fitted scaler (legacy fallback)
_SCALER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scaler.joblib"
)


class DataPreprocessor:
    """
    End-to-end preprocessor for the Sentiprice pipeline.

    Input :  DataFrame with columns  ['Close', 'Volume', 'Sentiment']
    Output:  PyTorch tensors / DataLoaders ready for LSTM consumption.
    """

    # Column ordering expected from data_loader.py
    RAW_COLS = ["Close", "Volume", "Sentiment"]

    def __init__(self, seq_length: int = 24):
        self.seq_length = seq_length
        # RobustScaler is more resilient to Volume & Price outliers than MinMaxScaler
        self.scaler = RobustScaler()
        self._n_features: int = 0   # set after feature engineering
        self._is_fitted: bool = False
        self._scaler_path: str = _SCALER_PATH  # overridable per-ticker

    # ------------------------------------------------------------------ #
    #  1. CLEANING
    # ------------------------------------------------------------------ #
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove / repair problematic rows before feature engineering."""
        df = df.copy()

        # --- Drop fully-empty rows ---
        df.dropna(how="all", inplace=True)

        # --- Forward-fill small gaps, then back-fill leading edge ---
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # --- Replace any remaining Inf / -Inf with NaN, then drop ---
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # --- Clip Volume outliers (IQR method) ---
        if "Volume" in df.columns:
            q1 = df["Volume"].quantile(0.05)
            q3 = df["Volume"].quantile(0.95)
            iqr = q3 - q1
            lower = max(q1 - 1.5 * iqr, 0)
            upper = q3 + 1.5 * iqr
            df["Volume"] = df["Volume"].clip(lower, upper)

        # --- Clamp Sentiment to [-1, 1] (safety net) ---
        if "Sentiment" in df.columns:
            df["Sentiment"] = df["Sentiment"].clip(-1.0, 1.0)

        logger.info(f"  🧹 Cleaning complete — {len(df)} rows retained.")
        return df

    # ------------------------------------------------------------------ #
    #  2. FEATURE ENGINEERING
    # ------------------------------------------------------------------ #
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives new columns that give the LSTM richer signal:
          • Returns        — hourly % change of Close
          • Volatility     — 6-hour rolling std of Returns
          • Sentiment_MA   — 6-hour rolling mean of Sentiment (momentum)
          • Volume_Ratio   — Volume / 12-hour rolling mean (relative spike)
          • Hour            — hour-of-day (cyclical, sin-encoded)
        """
        df = df.copy()

        # --- Price returns (percentage) ---
        df["Returns"] = df["Close"].pct_change().fillna(0.0)

        # --- Rolling volatility (6-hour window) ---
        df["Volatility"] = (
            df["Returns"]
            .rolling(window=6, min_periods=1)
            .std()
            .fillna(0.0)
        )

        # --- Sentiment momentum (6-hour rolling mean) ---
        df["Sentiment_MA"] = (
            df["Sentiment"]
            .rolling(window=6, min_periods=1)
            .mean()
            .fillna(0.0)
        )

        # --- Relative volume spike ---
        vol_rolling = df["Volume"].rolling(window=12, min_periods=1).mean()
        df["Volume_Ratio"] = (df["Volume"] / vol_rolling.replace(0, 1)).fillna(1.0)

        # --- Hour of day (sin-encoded for cyclical continuity) ---
        if hasattr(df.index, "hour"):
            df["Hour_Sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        else:
            df["Hour_Sin"] = 0.0

        # Drop raw Returns after volatility is computed (Close already carries level info)
        # Keep Returns as an additional signal — the LSTM can learn from it.

        logger.info(
            f"  🔧 Feature engineering complete — "
            f"{len(df.columns)} features: {list(df.columns)}"
        )
        return df

    # ------------------------------------------------------------------ #
    #  3. SCALING
    # ------------------------------------------------------------------ #
    def _scale(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """Apply RobustScaler. Use fit=True only on training data."""
        if fit:
            scaled = self.scaler.fit_transform(data)
            self._is_fitted = True
        else:
            scaled = self.scaler.transform(data)
        return scaled

    # ------------------------------------------------------------------ #
    #  4. SEQUENCE CREATION
    # ------------------------------------------------------------------ #
    def _create_sequences(self, data_scaled: np.ndarray):
        """
        Converts 2D scaled array into 3D sequences for LSTM.
        Input  : (Rows, Features)
        Output : X → (Samples, SeqLen, Features),  y → (Samples,)
        Target : Scaled 'Close' at the next time step (column index 0).
        """
        xs, ys = [], []
        for i in range(len(data_scaled) - self.seq_length):
            xs.append(data_scaled[i : i + self.seq_length])
            ys.append(data_scaled[i + self.seq_length][0])  # 'Close' = col 0
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  5. TRAIN / VAL / TEST SPLIT  (sequential — no leakage)
    # ------------------------------------------------------------------ #
    def _split(self, X: np.ndarray, y: np.ndarray,
               train_ratio=0.70, val_ratio=0.15):
        """
        Chronological split — preserves temporal ordering.
        Default: 70 % train · 15 % val · 15 % test.
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits = {
            "train": (X[:train_end], y[:train_end]),
            "val":   (X[train_end:val_end], y[train_end:val_end]),
            "test":  (X[val_end:], y[val_end:]),
        }
        for name, (sx, sy) in splits.items():
            logger.info(f"    {name:>5s}:  {len(sx)} samples")
        return splits

    # ------------------------------------------------------------------ #
    #  PUBLIC API — full pipeline
    # ------------------------------------------------------------------ #
    def prepare(self, df: pd.DataFrame, batch_size: int = 32):
        """
        Full preprocessing pipeline:  clean → engineer → split → scale → sequence.

        IMPORTANT: The scaler is fitted ONLY on the training portion
        to prevent data leakage from val/test into the training stats.

        Uses adaptive splitting:
          • 3-way (70/15/15) if enough data
          • 2-way (80/20)    if data is tight
          • Train-only       if data is very small (with warning)

        Returns
        -------
        splits : dict   {"train": (X, y), "val": (X, y), "test": (X, y)}
                         All values are torch.FloatTensor.
        feature_names : list[str]
        """
        logger.info("━━━ DataPreprocessor.prepare() ━━━")

        # 1. Clean
        df = self._clean(df)

        # 2. Feature engineering
        df = self._engineer_features(df)

        # Drop metadata columns that are not model features
        if "has_news" in df.columns:
            df = df.drop(columns=["has_news"])

        # Store feature metadata
        self.feature_names = list(df.columns)
        self._n_features = len(self.feature_names)

        # 3. Adaptive split — guarantee viable sizes
        values = df.values.astype(np.float32)
        n = len(values)
        min_split = self.seq_length + 2  # minimum rows needed per split

        if n >= min_split * 4:
            # Enough for 3-way split (70/15/15)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            split_mode = "3-way (70/15/15)"
        elif n >= min_split * 2:
            # Enough for 2-way split (80/20), no separate test
            train_end = int(n * 0.80)
            val_end = n
            split_mode = "2-way (80/20)"
        else:
            # Very small dataset — train only
            train_end = n
            val_end = n
            split_mode = "train-only (⚠ no validation)"

        logger.info(f"  📐 Split mode: {split_mode} ({n} total rows, "
                    f"min per split: {min_split})")

        train_raw = values[:train_end]
        val_raw = values[train_end:val_end] if train_end < val_end else np.empty((0, self._n_features))
        test_raw = values[val_end:] if val_end < n else np.empty((0, self._n_features))

        # 4. Fit scaler ONLY on training data (prevents leakage)
        train_scaled = self._scale(train_raw, fit=True)
        val_scaled = self._scale(val_raw, fit=False) if len(val_raw) > 0 else val_raw
        test_scaled = self._scale(test_raw, fit=False) if len(test_raw) > 0 else test_raw

        # 5. Create sequences per split
        splits = {}
        for name, data_scaled in [("train", train_scaled),
                                   ("val", val_scaled),
                                   ("test", test_scaled)]:
            if len(data_scaled) <= self.seq_length:
                splits[name] = (
                    torch.FloatTensor(np.empty((0, self.seq_length, self._n_features))),
                    torch.FloatTensor(np.empty((0,))),
                )
                logger.info(f"    {name:>5s}:  0 samples (split too small)")
                continue
            X, y = self._create_sequences(data_scaled)
            splits[name] = (torch.FloatTensor(X), torch.FloatTensor(y))
            logger.info(f"    {name:>5s}:  {len(X)} samples")

        logger.info("━━━ Preprocessing complete ━━━\n")
        return splits, self.feature_names

    def create_sequences(self, df: pd.DataFrame, seq_length: int = None):
        """
        Legacy-compatible method (used by train.py for initial data prep).
        Returns (X, y) tensors with cleaning + feature engineering applied.

        If the scaler is already fitted (e.g. loaded from joblib),
        it will use transform-only (no re-fit) to preserve the training scaler.
        """
        if seq_length is not None:
            self.seq_length = seq_length

        df = self._clean(df)
        df = self._engineer_features(df)

        # Drop metadata columns that are not model features
        if "has_news" in df.columns:
            df = df.drop(columns=["has_news"])

        self.feature_names = list(df.columns)
        self._n_features = len(self.feature_names)

        values = df.values.astype(np.float32)
        # Auto-detect: if scaler is already fitted (loaded from file), don't re-fit
        scaled = self._scale(values, fit=not self._is_fitted)
        X, y = self._create_sequences(scaled)

        return torch.FloatTensor(X), torch.FloatTensor(y)

    def get_dataloaders(self, df: pd.DataFrame,
                        batch_size: int = 32,
                        seq_length: int = 24):
        """
        Convenience wrapper that returns PyTorch DataLoaders.

        Returns
        -------
        train_loader, val_loader, test_loader, feature_names
        """
        self.seq_length = seq_length
        splits, feature_names = self.prepare(df, batch_size)

        loaders = {}
        for name, (X, y) in splits.items():
            shuffle = (name == "train")
            ds = TensorDataset(X, y)
            loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        return loaders["train"], loaders["val"], loaders["test"], feature_names

    # ------------------------------------------------------------------ #
    #  INVERSE TRANSFORM
    # ------------------------------------------------------------------ #
    def inverse_transform(self, prediction):
        """
        Converts a scaled 'Close' prediction back to real price.
        Dynamically adapts to the actual number of features.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Run prepare() first.")

        pred = np.array(prediction).flatten()
        # Build a dummy matrix matching the scaler's expected shape
        dummy = np.zeros((len(pred), self._n_features))
        dummy[:, 0] = pred  # 'Close' is always column 0
        return self.scaler.inverse_transform(dummy)[:, 0]

    # ------------------------------------------------------------------ #
    #  SCALER PERSISTENCE (ticker-aware)
    # ------------------------------------------------------------------ #
    def save_scaler(self, path: str = None):
        """Persist fitted scaler + metadata to disk.
        Uses self._scaler_path if no explicit path given.
        """
        path = path or self._scaler_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "scaler": self.scaler,
            "n_features": self._n_features,
            "feature_names": getattr(self, "feature_names", []),
            "seq_length": self.seq_length,
        }
        joblib.dump(state, path)
        logger.info(f"  💾 Scaler saved → {path}")

    def load_scaler(self, path: str = None):
        """Restore a previously saved scaler from disk.
        Uses self._scaler_path if no explicit path given.
        """
        path = path or self._scaler_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found: {path}")
        state = joblib.load(path)
        self.scaler = state["scaler"]
        self._n_features = state["n_features"]
        self.feature_names = state.get("feature_names", [])
        self.seq_length = state.get("seq_length", 24)
        self._is_fitted = True
        logger.info(f"  📦 Scaler loaded ← {path}")


# ===================================================================== #
#  Quick smoke-test
# ===================================================================== #
if __name__ == "__main__":
    # Generate synthetic data that mirrors the real pipeline
    np.random.seed(42)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    fake_df = pd.DataFrame(
        {
            "Close": np.cumsum(np.random.randn(n) * 0.5) + 100,
            "Volume": np.random.randint(1000, 50000, size=n).astype(float),
            "Sentiment": np.random.uniform(-0.6, 0.6, size=n),
        },
        index=idx,
    )

    proc = DataPreprocessor(seq_length=24)
    splits, features = proc.prepare(fake_df)

    for name, (X, y) in splits.items():
        print(f"  {name:>5s}  X={X.shape}  y={y.shape}")
    print(f"  Features: {features}")

    # Test save / load round-trip
    proc.save_scaler()
    proc2 = DataPreprocessor()
    proc2.load_scaler()
    print(f"  Loaded features: {proc2.feature_names}")