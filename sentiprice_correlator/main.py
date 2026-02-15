"""
main.py — Production-Ready FastAPI Backend

Endpoints
─────────
  POST  /predict          → next-hour price prediction for a ticker
  GET   /health           → liveness check + model status
  GET   /model/info       → architecture config & parameter count
  GET   /history/{ticker} → historical price + sentiment time-series
  GET   /models           → list all trained ticker models

Dynamic model loading:
  When /predict is called with a specific ticker, the API dynamically
  loads that ticker's model from checkpoints/{TICKER}/. Models are
  cached in memory after first load for fast subsequent requests.

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

# ───────────────────────────────────────────────────────────────────── #
#  Model cache — keeps loaded models in memory per ticker
# ───────────────────────────────────────────────────────────────────── #
_model_cache: dict = {}  # {ticker: {"model": ..., "processor": ..., "config": ...}}


# ===================================================================== #
#  Device Detection
# ===================================================================== #
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_ticker_model(ticker: str, device: torch.device):
    """
    Load a ticker-specific model + scaler from checkpoints/{TICKER}/.
    Returns (model, processor, config) or raises HTTPException.
    """
    ticker = ticker.upper()

    # Return from cache if already loaded
    if ticker in _model_cache:
        entry = _model_cache[ticker]
        return entry["model"], entry["processor"], entry["config"]

    paths = get_ticker_paths(ticker)

    # Check if ticker model exists
    if not os.path.exists(paths["checkpoint"]):
        # Try legacy paths as fallback
        if os.path.exists(LEGACY_CHECKPOINT):
            ckpt_path = LEGACY_CHECKPOINT
            scaler_path = LEGACY_SCALER
            logger.info(f"  � No {ticker} model found, using legacy checkpoint")
        elif os.path.exists(LEGACY_WEIGHTS):
            # Legacy flat weights (no config)
            model = MultiModalNet()
            model.load_state_dict(torch.load(LEGACY_WEIGHTS, map_location=device))
            model.to(device).eval()
            processor = DataPreprocessor()
            if os.path.exists(LEGACY_SCALER):
                processor.load_scaler(LEGACY_SCALER)
            _model_cache[ticker] = {"model": model, "processor": processor, "config": {}}
            return model, processor, {}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for '{ticker}'. "
                       f"Run: python train.py --ticker {ticker}"
            )
    else:
        ckpt_path = paths["checkpoint"]
        scaler_path = paths["scaler"]

    # Load checkpoint and read config FIRST (to get architecture params)
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get("config", {})

    # Build model with MATCHING architecture from training config
    model = MultiModalNet(
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # Load scaler + set matching seq_length from training
    processor = DataPreprocessor(seq_length=config.get("seq_length", 24))
    if os.path.exists(scaler_path):
        processor.load_scaler(scaler_path)
    else:
        logger.warning(f"  ⚠️  No scaler found for {ticker} — predictions may be inaccurate.")

    # Cache for future requests
    _model_cache[ticker] = {"model": model, "processor": processor, "config": config}
    logger.info(f"  ✅ Loaded and cached {ticker} model from {ckpt_path}")

    return model, processor, config


# ===================================================================== #
#  Lifespan — preload available models at startup
# ===================================================================== #
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Discover models and preload default at startup; cleanup on shutdown."""
    device = get_device()
    logger.info(f"� Starting up on device: {device}")

    # Find all available trained models
    available = find_available_models()
    if available:
        logger.info(f"  📋 Trained models: {list(available.keys())}")
    else:
        logger.info("  📋 No trained models found in checkpoints/")

    # Preload default model (AAPL or first available)
    default_ticker = os.environ.get("SENTIPRICE_TICKER", "AAPL").upper()
    model_loaded = False
    config = {}

    try:
        model, processor, config = load_ticker_model(default_ticker, device)
        model_loaded = True
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  ✅ Default model ({default_ticker}) ready — {total_params:,} parameters")
    except HTTPException:
        # Try loading any available model
        if available:
            first_ticker = list(available.keys())[0]
            try:
                model, processor, config = load_ticker_model(first_ticker, device)
                model_loaded = True
                logger.info(f"  ✅ Loaded {first_ticker} as default model")
            except Exception:
                logger.warning("  ⚠️  Could not load any model")
        else:
            logger.warning("  ⚠️  No model weights found — /predict will load on first request")

    # ── Store in app.state ───────────────────────────────────────────
    application.state.device = device
    application.state.model_loaded = model_loaded
    application.state.config = config
    application.state.start_time = datetime.now(timezone.utc)
    application.state.available_models = available

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
    model_ticker: str  # which ticker's model was used
    fundamentals: dict = {}
    has_news: bool = False

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float
    timestamp: str
    available_models: list

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
    has_news: bool = False

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
    available = find_available_models()
    return HealthResponse(
        status="healthy" if app.state.model_loaded else "degraded",
        model_loaded=app.state.model_loaded,
        device=str(app.state.device),
        uptime_seconds=round(uptime, 1),
        timestamp=now.isoformat(),
        available_models=list(available.keys()),
    )


# ── Available Models ─────────────────────────────────────────────────
@app.get("/models", tags=["System"])
async def list_models():
    """List all trained ticker models available for prediction."""
    available = find_available_models()
    cached = list(_model_cache.keys())
    return {
        "available_models": list(available.keys()),
        "cached_models": cached,
        "total": len(available),
    }


# ── Model Info ───────────────────────────────────────────────────────
@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info(ticker: str = Query(default="AAPL", description="Ticker to get model info for")):
    """Returns architecture details and training configuration for a specific ticker model."""
    device = app.state.device
    try:
        model, processor, config = load_ticker_model(ticker, device)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ModelInfoResponse(
        total_parameters=total,
        trainable_parameters=trainable,
        training_config=config,
        feature_names=getattr(processor, "feature_names", []),
        architecture=model.__class__.__name__,
    )


# ── Predict ──────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(req: PredictionRequest):
    """
    Fetch live market data, dynamically load the correct ticker model,
    and return the next-hour price prediction with sentiment context.
    """
    ticker = req.ticker.upper()
    device = app.state.device

    t0 = time.time()

    try:
        # 1. Load the ticker-specific model (cached after first load)
        model, processor, config = load_ticker_model(ticker, device)
        model_ticker = config.get("ticker", ticker)

        # 2. Fetch live data (with lock to avoid concurrent yfinance issues)
        async with _fetch_lock:
            data_loader = MarketDataLoader(req.ticker)
            df = data_loader.get_aligned_data(days=req.days)

        if df.empty or len(df) < 25:
            # Try to fetch more history if possible, otherwise error
            raise HTTPException(
                status_code=422,
                detail=f"Not enough data for '{req.ticker}'. "
                       f"Got {len(df)} rows, need ≥25.",
            )

        # 3. Preprocess
        X, _ = processor.create_sequences(df)

        if len(X) == 0:
            raise HTTPException(
                status_code=422,
                detail="Could not create sequences from the data.",
            )

        # 4. Predict
        latest_seq = X[-1].unsqueeze(0).to(device)

        with torch.no_grad():
            pred_scaled = model(latest_seq).cpu().numpy()


        # 5. Inverse transform
        price_predicted = float(processor.inverse_transform(pred_scaled)[0])
        current_price = float(df["Close"].iloc[-1])
        
        # Handle sentiment - might be 0.0 if no recent news
        sentiment = float(df["Sentiment"].iloc[-1])
        has_news = bool(df["has_news"].iloc[-1]) if "has_news" in df.columns else False
        
        change = price_predicted - current_price
        change_pct = (change / current_price) * 100 if current_price != 0 else 0.0

        # Create fundamental context for the dashboard
        fund = data_loader.get_fundamentals()

        elapsed = time.time() - t0
        logger.info(
            f"  🔮 {req.ticker}  ${current_price:.2f} → ${price_predicted:.2f}  "
            f"({change_pct:+.2f}%)  sentiment={sentiment:.3f} (news={has_news})  "
            f"model={model_ticker}  [{elapsed:.2f}s]"
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
            model_ticker=model_ticker,
            fundamentals=fund,
            has_news=has_news  # New field
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"  ❌ Prediction failed for {req.ticker}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Market Scanner ───────────────────────────────────────────────────
@app.post("/scan_market", tags=["Scanner"])
async def scan_market_endpoint(payload: dict):
    """
    Scans a list of tickers for Sentiment + Fundamentals.
    Payload: {"tickers": ["AAPL", "NVDA", ...]}
    """
    from scanner import MarketScanner
    
    tickers = payload.get("tickers", [])
    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")
        
    try:
        scanner = MarketScanner(tickers)
        df = scanner.scan_market()
        
        if df.empty:
            return {"results": []}
            
        # Nan replace for JSON safety
        df = df.fillna(0.0)
        return {"results": df.to_dict(orient="records")}
        
    except Exception as e:
        logger.error(f"Scanner failed: {e}")
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
            # Handle has_news if it exists, default False
            has_news = bool(row["has_news"]) if "has_news" in row else (row["Sentiment"] != 0.0)

            points.append(HistoryPoint(
                timestamp=str(ts),
                close=round(float(row["Close"]), 2),
                volume=round(float(row["Volume"]), 2),
                sentiment=round(float(row["Sentiment"]), 4),
                has_news=has_news,
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