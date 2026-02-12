"""
main.py — Production-Ready FastAPI Backend

Endpoints
─────────
  POST  /predict          → next-hour price prediction for a ticker
  GET   /health           → liveness check + model status
  GET   /model/info       → architecture config & parameter count
  GET   /history/{ticker} → historical price + sentiment time-series

Run
───
  uvicorn main:app --reload --port 8000
  python main.py                          # built-in launcher
"""

import os
import time
import asyncio
import logging
import argparse
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import MultiModalNet
from data_loader import MarketDataLoader
from preprocessor import DataPreprocessor

# ───────────────────────────────────────────────────────────────────── #
#  Logging
# ───────────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────── #
#  Paths (ticker-specific)
# ───────────────────────────────────────────────────────────────────── #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# Legacy flat paths (backward compatibility)
LEGACY_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LEGACY_WEIGHTS = os.path.join(BASE_DIR, "model_weights.pth")
LEGACY_SCALER = os.path.join(BASE_DIR, "scaler.joblib")


def get_ticker_paths(ticker: str):
    """Get checkpoint and scaler paths for a specific ticker."""
    ticker_dir = os.path.join(CHECKPOINT_DIR, ticker.upper())
    return {
        "checkpoint": os.path.join(ticker_dir, "best_model.pth"),
        "scaler": os.path.join(ticker_dir, "scaler.joblib"),
        "dir": ticker_dir,
    }


def find_available_models():
    """Scan checkpoints/ for all trained ticker models."""
    models = {}
    if os.path.exists(CHECKPOINT_DIR):
        for name in os.listdir(CHECKPOINT_DIR):
            ticker_dir = os.path.join(CHECKPOINT_DIR, name)
            if os.path.isdir(ticker_dir):
                ckpt = os.path.join(ticker_dir, "best_model.pth")
                if os.path.exists(ckpt):
                    models[name.upper()] = get_ticker_paths(name)
    return models

# ───────────────────────────────────────────────────────────────────── #
#  Concurrency lock (prevent overlapping yfinance calls)
# ───────────────────────────────────────────────────────────────────── #
_fetch_lock = asyncio.Lock()


# ===================================================================== #
#  Device Detection
# ===================================================================== #
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ===================================================================== #
#  Lifespan — load model once at startup
# ===================================================================== #
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model & scaler into app.state on startup; cleanup on shutdown."""
    device = get_device()
    logger.info(f"🚀 Starting up on device: {device}")

    # Find all available trained models
    available = find_available_models()
    if available:
        logger.info(f"  📋 Trained models: {list(available.keys())}")

    # ── Load model ───────────────────────────────────────────────────
    model = MultiModalNet()
    model_loaded = False
    config = {}

    # Determine which ticker model to load
    default_ticker = os.environ.get("SENTIPRICE_TICKER", "AAPL").upper()
    paths = get_ticker_paths(default_ticker)

    if os.path.exists(paths["checkpoint"]):
        logger.info(f"  📦 Loading {default_ticker} model: {paths['checkpoint']}")
        checkpoint = torch.load(paths["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        config = checkpoint.get("config", {})
        model_loaded = True
    elif os.path.exists(LEGACY_CHECKPOINT):
        logger.info(f"  📦 Loading legacy checkpoint: {LEGACY_CHECKPOINT}")
        checkpoint = torch.load(LEGACY_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        config = checkpoint.get("config", {})
        model_loaded = True
    elif os.path.exists(LEGACY_WEIGHTS):
        logger.info(f"  📦 Loading legacy weights: {LEGACY_WEIGHTS}")
        model.load_state_dict(torch.load(LEGACY_WEIGHTS, map_location=device))
        model_loaded = True
    else:
        logger.warning("  ⚠️  No model weights found — /predict will return errors.")

    model.to(device).eval()

    # ── Load scaler ──────────────────────────────────────────────────
    processor = DataPreprocessor()
    scaler_loaded = False
    scaler_path = paths["scaler"] if os.path.exists(paths["scaler"]) else LEGACY_SCALER
    if os.path.exists(scaler_path):
        processor.load_scaler(scaler_path)
        scaler_loaded = True
        logger.info(f"  📦 Scaler loaded ({len(processor.feature_names)} features)")
    else:
        logger.warning("  ⚠️  No scaler found — predictions may be inaccurate.")

    # ── Store in app.state ───────────────────────────────────────────
    application.state.model = model
    application.state.device = device
    application.state.processor = processor
    application.state.config = config
    application.state.model_loaded = model_loaded
    application.state.scaler_loaded = scaler_loaded
    application.state.start_time = datetime.now(timezone.utc)
    application.state.available_models = available

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ✅ Model ready — {total_params:,} parameters")
    logger.info(f"  🌐 API is live\n")

    yield  # ← app runs here

    logger.info("🛑 Shutting down...")


# ===================================================================== #
#  FastAPI App
# ===================================================================== #
app = FastAPI(
    title="Sentiprice Correlator API",
    description="Multi-Modal Sentiment & Price Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow Streamlit dashboard & any frontend) ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================== #
#  Pydantic Schemas
# ===================================================================== #
class PredictionRequest(BaseModel):
    ticker: str = Field(..., example="AAPL", description="Stock ticker symbol")
    days: int = Field(default=7, ge=2, le=90, description="Days of history to fetch")

class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_next_hour: float
    price_change: float
    price_change_pct: float
    sentiment_score: float
    sentiment_label: str
    timestamp: str
    model_device: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    device: str
    uptime_seconds: float
    timestamp: str

class ModelInfoResponse(BaseModel):
    total_parameters: int
    trainable_parameters: int
    training_config: dict
    feature_names: list
    architecture: str

class HistoryPoint(BaseModel):
    timestamp: str
    close: float
    volume: float
    sentiment: float

class HistoryResponse(BaseModel):
    ticker: str
    points: list[HistoryPoint]
    total_points: int


# ===================================================================== #
#  Helper
# ===================================================================== #
def _sentiment_label(score: float) -> str:
    """Map a [-1, 1] sentiment score to a readable label."""
    if score >= 0.3:
        return "Bullish 🟢"
    elif score <= -0.3:
        return "Bearish 🔴"
    return "Neutral 🟡"


# ===================================================================== #
#  Endpoints
# ===================================================================== #

# ── Health ───────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness probe — returns model & system status."""
    now = datetime.now(timezone.utc)
    uptime = (now - app.state.start_time).total_seconds()
    return HealthResponse(
        status="healthy" if app.state.model_loaded else "degraded",
        model_loaded=app.state.model_loaded,
        scaler_loaded=app.state.scaler_loaded,
        device=str(app.state.device),
        uptime_seconds=round(uptime, 1),
        timestamp=now.isoformat(),
    )


# ── Model Info ───────────────────────────────────────────────────────
@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Returns architecture details and training configuration."""
    m = app.state.model
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return ModelInfoResponse(
        total_parameters=total,
        trainable_parameters=trainable,
        training_config=app.state.config,
        feature_names=getattr(app.state.processor, "feature_names", []),
        architecture=m.__class__.__name__,
    )


# ── Predict ──────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(req: PredictionRequest):
    """
    Fetch live market data, run through the model, and return
    the next-hour price prediction with sentiment context.
    """
    if not app.state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first.",
        )

    t0 = time.time()

    try:
        # 1. Fetch live data (with lock to avoid concurrent yfinance issues)
        async with _fetch_lock:
            data_loader = MarketDataLoader(req.ticker)
            df = data_loader.get_aligned_data(days=req.days)

        if df.empty or len(df) < 25:
            raise HTTPException(
                status_code=422,
                detail=f"Not enough data for '{req.ticker}'. "
                       f"Got {len(df)} rows, need ≥25.",
            )

        # 2. Preprocess
        processor = app.state.processor
        X, _ = processor.create_sequences(df)

        if len(X) == 0:
            raise HTTPException(
                status_code=422,
                detail="Could not create sequences from the data.",
            )

        # 3. Predict
        device = app.state.device
        model = app.state.model
        latest_seq = X[-1].unsqueeze(0).to(device)

        with torch.no_grad():
            pred_scaled = model(latest_seq).cpu().numpy()

        # 4. Inverse transform
        price_predicted = float(processor.inverse_transform(pred_scaled)[0])
        current_price = float(df["Close"].iloc[-1])
        sentiment = float(df["Sentiment"].iloc[-1])
        change = price_predicted - current_price
        change_pct = (change / current_price) * 100 if current_price != 0 else 0.0

        elapsed = time.time() - t0
        logger.info(
            f"  🔮 {req.ticker}  ${current_price:.2f} → ${price_predicted:.2f}  "
            f"({change_pct:+.2f}%)  sentiment={sentiment:.3f}  [{elapsed:.2f}s]"
        )

        return PredictionResponse(
            ticker=req.ticker,
            current_price=round(current_price, 2),
            predicted_next_hour=round(price_predicted, 2),
            price_change=round(change, 2),
            price_change_pct=round(change_pct, 4),
            sentiment_score=round(sentiment, 4),
            sentiment_label=_sentiment_label(sentiment),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_device=str(device),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"  ❌ Prediction failed for {req.ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── History ──────────────────────────────────────────────────────────
@app.get("/history/{ticker}", response_model=HistoryResponse, tags=["Data"])
async def get_history(
    ticker: str,
    days: int = Query(default=7, ge=1, le=90, description="Days of history"),
):
    """
    Returns historical price + sentiment time-series for the dashboard.
    """
    try:
        async with _fetch_lock:
            data_loader = MarketDataLoader(ticker)
            df = data_loader.get_aligned_data(days=days)

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for '{ticker}'.",
            )

        points = []
        for ts, row in df.iterrows():
            points.append(HistoryPoint(
                timestamp=str(ts),
                close=round(float(row["Close"]), 2),
                volume=round(float(row["Volume"]), 2),
                sentiment=round(float(row["Sentiment"]), 4),
            ))

        return HistoryResponse(
            ticker=ticker,
            points=points,
            total_points=len(points),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"  ❌ History fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===================================================================== #
#  Built-in Launcher
# ===================================================================== #
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Sentiprice API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )