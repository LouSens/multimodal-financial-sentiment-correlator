# 🧠 Sentiprice Correlator

**Multi-Modal Sentiment & Price Prediction Engine**

A production-ready deep learning system that fuses BERT-powered news sentiment analysis with LSTM time-series forecasting to predict next-hour stock prices. Built with a dual-branch neural architecture, real-time FastAPI backend, and interactive Streamlit dashboard.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

Sentiprice Correlator addresses the challenge of incorporating unstructured news sentiment into quantitative price forecasting. Traditional time-series models treat all market signals equally — this system separates price dynamics and sentiment signals into dedicated neural pathways, then fuses them through a learned cross-modal gating mechanism.

**Key Insight:** Not all sentiment matters equally. During earnings season, a single headline can move a stock 10%. On a quiet Tuesday, the same headline might have zero impact. The cross-modal gate learns _when_ to amplify or suppress the sentiment signal.

### Pipeline

```
Yahoo Finance News Headlines
        │
        ▼
   FinBERT Sentiment Scoring ──────────────────┐
   (per-headline → [-1, +1])                   │
                                                ▼
Yahoo Finance Price Data (Hourly) ───→ Feature Engineering ───→ Preprocessing
   (Close, Volume, OHLCV)               (Returns, Volatility,     (RobustScaler,
                                          Sentiment_MA,             Sequence Creation,
                                          Volume_Ratio,             Adaptive Splitting)
                                          Hour_Sin)                       │
                                                                          ▼
                                                                   MultiModalNet
                                                              ┌─────────┴─────────┐
                                                              │                   │
                                                         Price Branch       Sentiment Branch
                                                        (Bi-LSTM +         (GRU + Temporal
                                                         Residual)          Attention)
                                                              │                   │
                                                              └─────────┬─────────┘
                                                                        │
                                                                 Cross-Modal Gate
                                                                   (Sigmoid)
                                                                        │
                                                                   MLP Head
                                                                        │
                                                                        ▼
                                                              Next-Hour Price
                                                                Prediction
```

---

## Architecture

### Dual-Branch Neural Network (`MultiModalNet`)

| Component | Type | Details |
|---|---|---|
| **Price Branch** | Bidirectional LSTM | 2-layer, 64 hidden units, residual skip connection |
| **Sentiment Branch** | GRU + Temporal Attention | 1-layer GRU with learned attention over time steps |
| **Fusion** | Cross-Modal Gating | Sigmoid gate controls sentiment influence dynamically |
| **Prediction Head** | 2-Layer MLP | GELU activation, LayerNorm, dropout regularization |

### Feature Set (8 engineered features)

| Feature | Source | Description |
|---|---|---|
| `Close` | yfinance | Hourly closing price |
| `Volume` | yfinance | Trading volume (IQR-clipped) |
| `Sentiment` | FinBERT | News headline sentiment score [-1, +1] |
| `Returns` | Derived | Hourly % price change |
| `Volatility` | Derived | 6-hour rolling std of returns |
| `Sentiment_MA` | Derived | 6-hour rolling mean of sentiment |
| `Volume_Ratio` | Derived | Volume relative to 12-hour moving average |
| `Hour_Sin` | Derived | Sin-encoded hour-of-day (cyclical) |

---

## Features

- **🔮 Real-Time Predictions** — Next-hour price forecasting via live market data
- **📰 BERT Sentiment Analysis** — FinBERT scores Yahoo Finance news headlines
- **🧠 Dual-Branch Architecture** — Separate price and sentiment neural pathways
- **🎯 Cross-Modal Gating** — Learned mechanism to weight sentiment influence
- **📊 Interactive Dashboard** — Streamlit UI with glassmorphism design
- **⚡ FastAPI Backend** — Async REST API with Swagger docs
- **🏷️ Multi-Ticker Support** — Train and deploy separate models per stock
- **💾 Ticker-Specific Checkpoints** — Models saved to `checkpoints/{TICKER}/` without overwriting
- **📐 Adaptive Data Splitting** — Automatically adjusts train/val/test ratios for small datasets
- **🔒 Reproducibility** — Full seed locking across NumPy, PyTorch, and CUDA
- **🛡️ Data Leakage Prevention** — Scaler fitted only on training data
- **🖥️ Auto Device Detection** — CUDA → MPS → CPU fallback

---

## Project Structure

```
sentiprice_correlator/
├── main.py              # FastAPI backend — dynamic per-ticker model loading
├── dashboard.py         # Streamlit frontend — glassmorphism dark-themed UI
├── train.py             # Training pipeline — CLI with per-ticker checkpointing
├── model.py             # MultiModalNet — dual-branch LSTM/GRU architecture
├── preprocessor.py      # Data preprocessing — cleaning, features, scaling, splitting
├── data_loader.py       # Yahoo Finance data + BERT sentiment scoring
├── check_gpu.py         # GPU availability checker
├── requirements.txt     # Python dependencies
├── .gitignore           # Git exclusion rules
└── checkpoints/         # Trained model storage
    ├── AAPL/
    │   ├── best_model.pth     # Model weights + training config
    │   ├── scaler.joblib      # Fitted RobustScaler state
    │   └── model_weights.pth  # Legacy flat weights
    ├── TSLA/
    │   ├── best_model.pth
    │   ├── scaler.joblib
    │   └── model_weights.pth
    └── ...
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended)
- CUDA-capable GPU (optional, recommended for training)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/LouSens/multimodal-financial-sentiment-correlator.git
cd multimodal-financial-sentiment-correlator/sentiprice_correlator

# 2. Create conda environment
conda create -n sentiprice python=3.10 -y
conda activate sentiprice

# 3. Install PyTorch (with CUDA support — adjust for your setup)
# For CUDA 12.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# For CPU only:
# pip install torch torchvision torchaudio

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify GPU (optional)
python check_gpu.py
```

---

## Usage

### 1. Train a Model

```bash
# Train on Apple with default settings (100 epochs, 90 days of data)
python train.py --ticker AAPL --epochs 50

# Train on Tesla with custom parameters
python train.py --ticker TSLA --epochs 100 --lr 0.0005 --days 60

# Train on NVIDIA with more history
python train.py --ticker NVDA --epochs 80 --days 90 --patience 20
```

**Output:** Model saved to `checkpoints/{TICKER}/`

Each ticker gets its own isolated folder — training TSLA will **never** overwrite your AAPL model.

### 2. Start the API Server

```bash
# Start FastAPI backend
python main.py

# Or with auto-reload for development
python main.py --reload

# Custom port
python main.py --port 9000
```

The API automatically discovers all trained models in `checkpoints/` and dynamically loads the correct one when a prediction is requested.

### 3. Launch the Dashboard

```bash
# In a separate terminal
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Enter any ticker you've trained a model for, and click **⚡ Analyze**.

### Multi-Ticker Workflow

```bash
# Train multiple tickers
python train.py --ticker AAPL --epochs 50
python train.py --ticker TSLA --epochs 50
python train.py --ticker MSFT --epochs 50

# Start API (all models available simultaneously)
python main.py

# Dashboard can now switch between AAPL, TSLA, MSFT
streamlit run dashboard.py
```

---

## API Reference

Base URL: `http://localhost:8000`

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

### Endpoints

#### `POST /predict` — Price Prediction

Request a next-hour price prediction for any trained ticker.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days": 7}'
```

**Response:**
```json
{
  "ticker": "AAPL",
  "current_price": 185.42,
  "predicted_next_hour": 185.67,
  "price_change": 0.25,
  "price_change_pct": 0.1348,
  "sentiment_score": 0.2341,
  "sentiment_label": "Neutral 🟡",
  "timestamp": "2026-02-13T04:11:20+00:00",
  "model_device": "cuda",
  "model_ticker": "AAPL"
}
```

#### `GET /health` — System Health

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "uptime_seconds": 142.5,
  "timestamp": "2026-02-13T04:11:20+00:00",
  "available_models": ["AAPL", "TSLA", "MSFT"]
}
```

#### `GET /models` — Available Models

```bash
curl http://localhost:8000/models
```

**Response:**
```json
{
  "available_models": ["AAPL", "TSLA"],
  "cached_models": ["AAPL"],
  "total": 2
}
```

#### `GET /model/info?ticker=AAPL` — Model Details

```bash
curl "http://localhost:8000/model/info?ticker=AAPL"
```

#### `GET /history/{ticker}?days=14` — Historical Data

```bash
curl "http://localhost:8000/history/AAPL?days=14"
```

---

## Model Details

### Training Configuration (Defaults)

| Parameter | Default | Description |
|---|---|---|
| `--epochs` | 100 | Max training epochs |
| `--lr` | 0.001 | AdamW learning rate |
| `--batch-size` | 32 | Mini-batch size |
| `--seq-length` | 24 | LSTM input sequence length (hours) |
| `--hidden-dim` | 64 | Hidden units per recurrent layer |
| `--num-layers` | 2 | Stacked LSTM/GRU layers |
| `--dropout` | 0.3 | Dropout probability |
| `--patience` | 15 | Early stopping patience (epochs) |
| `--grad-clip` | 1.0 | Gradient clipping max norm |
| `--weight-decay` | 0.0001 | AdamW L2 regularization |
| `--days` | 90 | Days of historical data to fetch |
| `--seed` | 42 | Random seed for reproducibility |

### Training Features

- **Optimizer:** AdamW with cosine annealing warm restarts
- **Loss:** MSE (Mean Squared Error)
- **Regularization:** Dropout, gradient clipping, weight decay, early stopping
- **Scaling:** RobustScaler (outlier-resilient, fitted only on train split)
- **Splitting:** Adaptive — automatically adjusts from 3-way → 2-way → train-only based on data size

### Checkpointing

Each checkpoint contains:
- `model_state_dict` — Trained model weights
- `config` — Full training hyperparameter configuration
- Companion `scaler.joblib` — Fitted preprocessor state for inference

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SENTIPRICE_TICKER` | `AAPL` | Default ticker model to preload at API startup |

### API Server Arguments

```bash
python main.py --host 0.0.0.0 --port 8000 --reload
```

| Argument | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |
| `--reload` | `false` | Auto-reload on code changes |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | PyTorch (LSTM, GRU, Attention) |
| **NLP** | Hugging Face Transformers (FinBERT) |
| **Data** | yfinance, pandas, NumPy |
| **Scaling** | scikit-learn (RobustScaler) |
| **API** | FastAPI, Uvicorn |
| **Frontend** | Streamlit, Plotly |
| **Serialization** | joblib, torch.save |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Kylo
