"""
data_loader.py — Unified yfinance Data Pipeline

All data comes from Yahoo Finance:
  • Price & Volume  : yf.download()     — hourly OHLCV
  • News Sentiment  : yf.Ticker().news  — recent headlines scored with BERT

No CSV files, no API keys. One source of truth.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import os
import random
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# Cache file for pre-computed BERT sentiment scores
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_cache.csv")


class MarketDataLoader:
    def __init__(self, ticker):
        self.ticker = ticker
        self._interval = "1h"

        # Load BERT model & tokenizer for sentiment classification
        print("Loading BERT sentiment model...")
        self.tokenizer = BertTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        )
        self.sentiment_model = BertForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        )
        self.sentiment_model.eval()

    # ------------------------------------------------------------------ #
    #  Price Data
    # ------------------------------------------------------------------ #
    def fetch_price_data(self, start_date, end_date, interval="1h"):
        """Fetches OHLCV data from Yahoo Finance with UTC-normalized index."""
        print(f"Fetching price data for {self.ticker} ({interval})...")
        df = yf.download(self.ticker, start=start_date, end=end_date, interval=interval)

        if df.empty:
            raise ValueError(f"No price data found for {self.ticker}.")

        # Flatten MultiIndex columns if present (common yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Timezone Safety: ensure index is UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        self._interval = interval
        print(f"  📊 Price data: {len(df)} rows ({interval})")
        return df[['Close', 'Volume']]

    # ------------------------------------------------------------------ #
    #  yfinance Live News
    # ------------------------------------------------------------------ #
    def _fetch_yfinance_news(self):
        """
        Fetches recent news headlines directly from Yahoo Finance.
        Handles multiple yfinance API versions (field names change across versions).
        Returns a DataFrame with columns: headline, date.
        """
        print(f"  📡 Fetching live news from Yahoo Finance...")
        try:
            ticker_obj = yf.Ticker(self.ticker)
            news_items = ticker_obj.news

            if not news_items:
                print("  ⚠ No live news returned.")
                return None

            records = []
            for item in news_items:
                # --- Extract title (multiple possible field names) ---
                title = (
                    item.get("title")
                    or item.get("headline")
                    or (item.get("content", {}) or {}).get("title")
                    or ""
                )

                # --- Extract timestamp (multiple possible formats) ---
                dt = None

                # Format 1: Unix timestamp (older yfinance)
                pub_time = item.get("providerPublishTime")
                if pub_time and isinstance(pub_time, (int, float)):
                    dt = pd.to_datetime(pub_time, unit="s", utc=True)

                # Format 2: ISO string in 'publish_time' or 'pubDate'
                if dt is None:
                    for key in ["publish_time", "pubDate"]:
                        val = item.get(key)
                        if val:
                            try:
                                dt = pd.to_datetime(val, utc=True)
                            except Exception:
                                pass
                            if dt is not None:
                                break

                # Format 3: Nested in 'content' dict (newest yfinance)
                if dt is None:
                    content = item.get("content", {}) or {}
                    for key in ["pubDate", "publish_time", "providerPublishTime"]:
                        val = content.get(key)
                        if val:
                            try:
                                if isinstance(val, (int, float)):
                                    dt = pd.to_datetime(val, unit="s", utc=True)
                                else:
                                    dt = pd.to_datetime(val, utc=True)
                            except Exception:
                                pass
                            if dt is not None:
                                break

                # Format 4: Use current time as fallback if title exists
                if dt is None and title:
                    dt = pd.Timestamp.now(tz="UTC")

                if not title or dt is None:
                    continue

                records.append({"headline": title.strip(), "date": dt})

            if not records:
                print("  ⚠ No parseable headlines.")
                return None

            df = pd.DataFrame(records)
            print(f"  ✅ {len(df)} headlines retrieved "
                  f"({df['date'].min().date()} → {df['date'].max().date()})")
            return df

        except Exception as e:
            print(f"  ⚠ News fetch failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  BERT Sentiment Scorer
    # ------------------------------------------------------------------ #
    def _score_sentiment(self, text):
        """
        Scores a headline with BERT → returns sentiment in [-1, 1].
        Maps 5-star rating: 1★ = -1.0, 3★ = 0.0, 5★ = +1.0
        """
        try:
            inputs = self.tokenizer(
                str(text), return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze()
            stars = torch.arange(1, 6, dtype=torch.float)
            weighted_score = (probs * stars).sum().item()
            return (weighted_score - 3.0) / 2.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    #  Cache Helpers
    # ------------------------------------------------------------------ #
    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            print("  📦 Loading sentiment cache...")
            cache = pd.read_csv(CACHE_FILE)
            cache["date"] = pd.to_datetime(cache["date"], utc=True)
            return cache
        return None

    def _save_cache(self, df):
        df.to_csv(CACHE_FILE, index=False)
        print(f"  💾 Cache saved ({len(df)} entries)")

    # ------------------------------------------------------------------ #
    #  Sentiment Pipeline
    # ------------------------------------------------------------------ #
    def fetch_news_sentiment(self, start_date, end_date):
        """
        Full pipeline:
          1. Fetch headlines from yfinance
          2. Score with BERT (using cache)
          3. Aggregate to hourly sentiment
          4. Falls back to mock if no news available
        """
        print(f"Computing sentiment for {self.ticker}...")

        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)

        # --- Fetch live news ---
        news_df = self._fetch_yfinance_news()

        if news_df is None or news_df.empty:
            print("  ⚠ No news available. Using mock sentiment.")
            return self._generate_mock_sentiment(start_date, end_date)

        # Filter to date range
        filtered = news_df[
            (news_df["date"] >= start_dt) & (news_df["date"] <= end_dt)
        ].copy()

        if filtered.empty:
            # News exists but outside our range — spread the sentiment
            # across the full range since these are the closest headlines
            print(f"  ℹ Headlines outside date range. Using all {len(news_df)} headlines.")
            filtered = news_df.copy()

        # --- Score with BERT (cached) ---
        cache = self._load_cache()
        if cache is not None:
            cached_set = set(cache["headline"].tolist())
            needs_scoring = filtered[~filtered["headline"].isin(cached_set)]
            already_scored = filtered[filtered["headline"].isin(cached_set)]

            if not already_scored.empty:
                already_scored = already_scored.merge(
                    cache[["headline", "Sentiment"]], on="headline", how="left"
                )

            if not needs_scoring.empty:
                print(f"  Scoring {len(needs_scoring)} new headlines with BERT...")
                tqdm.pandas(desc="  🧠 BERT")
                needs_scoring = needs_scoring.copy()
                needs_scoring["Sentiment"] = needs_scoring["headline"].progress_apply(
                    self._score_sentiment
                )
                new_cache = pd.concat([
                    cache, needs_scoring[["headline", "date", "Sentiment"]]
                ])
                self._save_cache(new_cache)
            else:
                print("  ✅ All headlines cached — skipping BERT.")
                needs_scoring = pd.DataFrame()

            parts = [df for df in [already_scored, needs_scoring] if not df.empty]
            filtered = pd.concat(parts, ignore_index=True) if parts else already_scored
        else:
            print(f"  Scoring {len(filtered)} headlines with BERT (first run)...")
            tqdm.pandas(desc="  🧠 BERT")
            filtered["Sentiment"] = filtered["headline"].progress_apply(
                self._score_sentiment
            )
            self._save_cache(filtered[["headline", "date", "Sentiment"]])

        # --- Aggregate to hourly ---
        filtered["date"] = pd.to_datetime(filtered["date"], utc=True)
        filtered = filtered.set_index("date")
        hourly = filtered["Sentiment"].resample("1h").mean()
        hourly = hourly.to_frame(name="Sentiment")

        scored_count = filtered["Sentiment"].notna().sum()
        avg_sent = filtered["Sentiment"].mean()
        print(f"  📰 Sentiment: {scored_count} scored, avg = {avg_sent:.3f}")
        return hourly

    # ------------------------------------------------------------------ #
    #  Mock Sentiment (Fallback)
    # ------------------------------------------------------------------ #
    def _generate_mock_sentiment(self, start_date, end_date):
        """Generates random sentiment when no news is available."""
        print("  🎲 Mock sentiment mode")
        dates = pd.date_range(start=start_date, end=end_date, freq="1h", tz="UTC")
        sentiments = [random.uniform(-0.8, 0.8) for _ in range(len(dates))]
        return pd.DataFrame({"Sentiment": sentiments}, index=dates)

    # ------------------------------------------------------------------ #
    #  Alignment (main entry point)
    # ------------------------------------------------------------------ #
    def get_aligned_data(self, days=90):
        """
        Fetches price + sentiment and returns an aligned DataFrame.
        Uses yfinance for both price and news → same ticker, same dates.
        """
        end = datetime.now()
        start = end - timedelta(days=days)

        price_df = self.fetch_price_data(start, end, interval="1h")
        sentiment_df = self.fetch_news_sentiment(start, end)

        # Timezone Safety
        if price_df.index.tz is None:
            price_df.index = price_df.index.tz_localize("UTC")
        if sentiment_df.index.tz is None:
            sentiment_df.index = sentiment_df.index.tz_localize("UTC")

        # Left-join: keep ALL price rows, fill missing sentiment with 0.0
        combined = price_df.join(sentiment_df, how="left")
        combined["Sentiment"] = combined["Sentiment"].ffill().fillna(0.0)
        combined = combined.dropna(subset=["Close", "Volume"])

        non_zero = (combined["Sentiment"] != 0.0).sum()
        print(f"\n  ✅ Final dataset: {len(combined)} rows, "
              f"{non_zero} with real sentiment ({non_zero/len(combined)*100:.1f}%)")
        return combined


# Quick Test
if __name__ == "__main__":
    loader = MarketDataLoader("AAPL")
    data = loader.get_aligned_data(days=30)
    print(data.head(10))
    print(f"\nShape: {data.shape}")
    print(f"Sentiment range: [{data['Sentiment'].min():.3f}, {data['Sentiment'].max():.3f}]")
    print(f"Non-zero sentiment: {(data['Sentiment'] != 0).sum()} / {len(data)}")
