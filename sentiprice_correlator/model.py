"""
model.py — Multi-Modal Sentiment-Price Fusion Network

Architecture
────────────
  Branch A  (Price)     : Bi-directional LSTM  →  residual projection
  Branch B  (Sentiment) : GRU  →  temporal attention  →  context vector
  Fusion                : Cross-modal gating (sigmoid gate)
  Head                  : 2-layer MLP  →  next-hour Close prediction

Why this design?
  • Standard LSTMs treat all features equally. This model gives sentiment
    its own dedicated pathway so the network can learn *when* sentiment
    actually matters (earnings day vs. quiet weekend).
  • The cross-modal gate lets the model dynamically suppress or amplify
    the sentiment signal depending on the price context.
"""

import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)

# ===================================================================== #
#  Feature-index constants  (must match preprocessor.py column order)
# ===================================================================== #
#  0: Close   1: Volume   2: Sentiment   3: Returns   4: Volatility
#  5: Sentiment_MA   6: Volume_Ratio   7: Hour_Sin
PRICE_INDICES     = [0, 1, 3, 4, 6, 7]   # 6 features
SENTIMENT_INDICES = [2, 5]                 # 2 features


# ===================================================================== #
#  Temporal Attention (for Sentiment branch)
# ===================================================================== #
class TemporalAttention(nn.Module):
    """
    Learns which time-steps in the sentiment sequence matter most.
    Produces a single context vector from GRU hidden states.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, gru_out: torch.Tensor):
        """
        gru_out : (batch, seq_len, hidden)
        returns : (batch, hidden)  — weighted context vector
        """
        # Score every time-step
        scores = self.attn_score(gru_out).squeeze(-1)        # (B, T)
        weights = torch.softmax(scores, dim=-1).unsqueeze(1)  # (B, 1, T)
        context = torch.bmm(weights, gru_out).squeeze(1)      # (B, H)
        return context


# ===================================================================== #
#  Main Model
# ===================================================================== #
class MultiModalNet(nn.Module):
    """
    Dual-branch network for sentiment-aware price prediction.

    Parameters
    ----------
    price_dim      : int   Number of price/volume features   (default 6)
    sentiment_dim  : int   Number of sentiment features      (default 2)
    hidden_dim     : int   Hidden size for LSTM / GRU         (default 64)
    num_layers     : int   Stacked recurrent layers           (default 2)
    dropout        : float Dropout probability                (default 0.3)
    output_dim     : int   Prediction targets                 (default 1)
    """

    def __init__(
        self,
        input_dim: int = 8,          # kept for backward compat (ignored internally)
        price_dim: int = 6,
        sentiment_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_dim: int = 1,
    ):
        super().__init__()

        self.price_dim = price_dim
        self.sentiment_dim = sentiment_dim
        self.hidden_dim = hidden_dim

        # ── Branch A: Price LSTM (bidirectional) ─────────────────────────
        self.price_lstm = nn.LSTM(
            input_size=price_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        # Project bi-directional output (2 * hidden) back to hidden_dim
        self.price_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.price_norm = nn.LayerNorm(hidden_dim)
        # Residual skip: project raw last-step price features to hidden_dim
        self.price_skip = nn.Linear(price_dim, hidden_dim)

        # ── Branch B: Sentiment GRU + Attention ──────────────────────────
        self.sent_gru = nn.GRU(
            input_size=sentiment_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.sent_attention = TemporalAttention(hidden_dim)
        self.sent_norm = nn.LayerNorm(hidden_dim)

        # ── Cross-Modal Gating ───────────────────────────────────────────
        # Sigmoid gate: decides how much sentiment modulates the price repr
        self.cross_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # ── Prediction Head (MLP) ───────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Weight initialization
        self._init_weights()

        # Log architecture summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"  🧠 MultiModalNet initialised  |  "
            f"params: {total_params:,} total, {trainable:,} trainable  |  "
            f"price_dim={price_dim}  sent_dim={sentiment_dim}  "
            f"hidden={hidden_dim}  layers={num_layers}"
        )

    # ------------------------------------------------------------------ #
    #  Weight Initialisation
    # ------------------------------------------------------------------ #
    def _init_weights(self):
        """Xavier/Kaiming init for faster, more stable convergence."""
        for name, param in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                # Recurrent weights — orthogonal init
                nn.init.orthogonal_(param)
            elif "weight" in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)

    # ------------------------------------------------------------------ #
    #  Forward Pass
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, 8)  — full feature tensor from preprocessor
        returns : (batch, 1)     — predicted next-hour scaled Close
        """
        # ── Split features by modality ───────────────────────────────────
        x_price = x[:, :, PRICE_INDICES]       # (B, T, 6)
        x_sent  = x[:, :, SENTIMENT_INDICES]   # (B, T, 2)

        # ── Branch A: Price ──────────────────────────────────────────────
        lstm_out, _ = self.price_lstm(x_price)  # (B, T, 2*H)
        price_last = lstm_out[:, -1, :]          # (B, 2*H)
        price_h = self.price_proj(price_last)    # (B, H)

        # Residual connection from raw last time-step
        skip = self.price_skip(x_price[:, -1, :])  # (B, H)
        price_h = self.price_norm(price_h + skip)   # (B, H)

        # ── Branch B: Sentiment ──────────────────────────────────────────
        gru_out, _ = self.sent_gru(x_sent)          # (B, T, H)
        sent_h = self.sent_attention(gru_out)        # (B, H)
        sent_h = self.sent_norm(sent_h)              # (B, H)

        # ── Cross-Modal Gating ───────────────────────────────────────────
        # Gate ∈ [0,1] — controls how much sentiment influences the fusion
        combined = torch.cat([price_h, sent_h], dim=-1)  # (B, 2*H)
        gate = self.cross_gate(combined)                   # (B, H)

        # Modulate sentiment with the gate, then fuse
        sent_gated = gate * sent_h                         # (B, H)
        fused = torch.cat([price_h, sent_gated], dim=-1)   # (B, 2*H)

        # ── Prediction ───────────────────────────────────────────────────
        out = self.head(fused)  # (B, 1)
        return out


# ===================================================================== #
#  Smoke Test
# ===================================================================== #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    model = MultiModalNet()
    print(model)

    # Fake batch: 16 samples, 24 time-steps, 8 features
    dummy = torch.randn(16, 24, 8)
    out = model(dummy)
    print(f"\nInput:  {dummy.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (16, 1), f"Expected (16,1), got {out.shape}"
    print("✅ Forward pass OK")