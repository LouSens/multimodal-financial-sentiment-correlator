"""
data_loader.py — Unified yfinance Data Pipeline

All data comes from Yahoo Finance:
  • Price & Volume  : yf.download()     — hourly OHLCV
  • News Sentiment  : yf.Ticker().news  — recent headlines scored with BERT

Refactored for strict time-alignment and reduced noise.
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
MAX_NEWS_AGE_DAYS = 7  # yfinance usually only gives ~7 days of news


class MarketDataLoader:
    def __init__(self, ticker):
        self.ticker = ticker
        self._interval = "1h"

        # Load BERT model & tokenizer for sentiment classification
        # Suppress extensive loading logs usually
        print(f"[{self.ticker}] Initializing MarketDataLoader...")
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
        # Simple, clean fetch
        df = yf.download(self.ticker, start=start_date, end=end_date, interval=interval, progress=False)

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
        return df[['Close', 'Volume']]

    # ------------------------------------------------------------------ #
    #  yfinance Live News
    # ------------------------------------------------------------------ #
    def _fetch_yfinance_news(self):
        """
        Fetches recent news headlines directly from Yahoo Finance.
        Returns a DataFrame with columns: headline, summary, provider, url, date.
        """
        try:
            ticker_obj = yf.Ticker(self.ticker)
            news_items = ticker_obj.news
            
            if not news_items:
                return None

            records = []
            for item in news_items:
                # Content extracted from 'content' dictionary or top-level
                content = item.get("content", {})
                
                # Title
                title = item.get("title") or item.get("headline") or content.get("title") or ""
                
                # Summary (Key new feature)
                summary = content.get("summary") or item.get("summary") or ""
                
                # Provider
                provider_data = content.get("provider") or item.get("provider") or {}
                provider = provider_data.get("displayName") or "Yahoo Finance"
                
                # URL
                url = (
                    content.get("clickThroughUrl") 
                    or content.get("canonicalUrl") 
                    or item.get("link") 
                    or f"https://finance.yahoo.com/quote/{self.ticker}"
                )
                
                # Get Date (Robust parsing)
                dt = None
                
                # 1. Unix timestamp
                if "providerPublishTime" in item:
                    dt = pd.to_datetime(item["providerPublishTime"], unit="s", utc=True)
                elif "pubDate" in content:
                     try: dt = pd.to_datetime(content["pubDate"], utc=True)
                     except: pass
                
                # 2. ISO Strings fallback
                if dt is None:
                    for key in ["publish_time", "pubDate"]:
                        val = item.get(key)
                        if val:
                            try:
                                dt = pd.to_datetime(val, utc=True)
                                break
                            except: pass
                
                if title and dt:
                    records.append({
                        "headline": title.strip(),
                        "summary": summary.strip(),
                        "provider": provider,
                        "url": url.get("url") if isinstance(url, dict) else url,
                        "date": dt
                    })

            if not records:
                return None

            df = pd.DataFrame(records)
            return df

        except Exception as e:
            print(f"  Warning: News fetch failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  BERT Sentiment Scorer
    # ------------------------------------------------------------------ #
    def _score_sentiment(self, text):
        """
        Scores text with BERT → returns sentiment in [-1, 1].
        Maps 5-star rating: 1★ = -1.0, 3★ = 0.0, 5★ = +1.0
        """
        try:
            # text is now likely "Headline. Summary"
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
            try:
                cache = pd.read_csv(CACHE_FILE)
                cache["date"] = pd.to_datetime(cache["date"], utc=True)
                return cache
            except:
                return None
        return None

    def _save_cache(self, df):
        df.to_csv(CACHE_FILE, index=False)

    # ------------------------------------------------------------------ #
    #  Sentiment Pipeline
    # ------------------------------------------------------------------ #
    def fetch_news_sentiment(self, start_date, end_date):
        """
        Full pipeline:
          1. Fetch headlines from yfinance
          2. Score with BERT (using cache)
          3. Aggregate to hourly sentiment
        """
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)

        # --- Fetch live news ---
        news_df = self._fetch_yfinance_news()

        # If absolutely no news, return empty with correct index so alignment works (as 0.0)
        if news_df is None or news_df.empty:
            return pd.DataFrame(columns=["Sentiment"])

        # Filter strictly to date range? 
        # Actually, for yfinance news, it only gives RECENT news. 
        # We should NOT filter extensively if the user asks for historical data
        # because we only have the tail end. We will let alignment handle the gaps.
        
        filtered = news_df.copy()

        # --- Score with BERT (cached) ---
        cache = self._load_cache()
        if cache is not None:
            cached_set = set(cache["headline"].tolist())
            needs_scoring = filtered[~filtered["headline"].isin(cached_set)]
            already_scored = filtered[filtered["headline"].isin(cached_set)]
            
            if not already_scored.empty:
                # Merge to get the sentiment score
                already_scored = already_scored.merge(
                    cache[["headline", "Sentiment"]], on="headline", how="left"
                )

            if not needs_scoring.empty:
                # print(f"  Scoring {len(needs_scoring)} new headlines...")
                tqdm.pandas(desc="  BERT Scoring")
                needs_scoring = needs_scoring.copy()
                
                # Combine Headline + Summary for scoring
                needs_scoring["text_to_score"] = needs_scoring["headline"] + ". " + needs_scoring["summary"]
                
                needs_scoring["Sentiment"] = needs_scoring["text_to_score"].progress_apply(
                    self._score_sentiment
                )
                
                # Update Cache
                new_cache = pd.concat([
                    cache, needs_scoring[["headline", "date", "Sentiment"]]
                ])
                self._save_cache(new_cache)
            
            parts = [df for df in [already_scored, needs_scoring] if not df.empty]
            filtered = pd.concat(parts, ignore_index=True) if parts else already_scored
        else:
            # First run
            print(f"  Scoring {len(filtered)} items (Headline + Summary)...")
            tqdm.pandas(desc="  BERT Scoring")
            
            filtered["text_to_score"] = filtered["headline"] + ". " + filtered["summary"]
            filtered["Sentiment"] = filtered["text_to_score"].progress_apply(
                self._score_sentiment
            )
            self._save_cache(filtered[["headline", "date", "Sentiment"]])

        # --- Aggregate to hourly ---
        # Reduce to hourly mean
        filtered["date"] = pd.to_datetime(filtered["date"], utc=True)
        filtered = filtered.set_index("date").sort_index()
        
        # Resample to hourly to match price candles
        hourly = filtered["Sentiment"].resample("1h").mean()
        hourly = hourly.to_frame(name="Sentiment")
        
        return hourly

    # ------------------------------------------------------------------ #
    #  Alignment (main entry point)
    # ------------------------------------------------------------------ #
    def get_aligned_data(self, days=90):
        """
        Fetches price + sentiment and returns an aligned DataFrame.
        Strict alignment: Price at T is matched with News BEFORE T.
        """
        end = datetime.now()
        start = end - timedelta(days=days)

        # 1. Get Price
        try:
            price_df = self.fetch_price_data(start, end, interval="1h")
        except Exception as e:
            print(f"Error fetching price for {self.ticker}: {e}")
            return pd.DataFrame()

        # 2. Get Sentiment
        sentiment_df = self.fetch_news_sentiment(start, end)

        # 3. Align
        # Sort both
        price_df = price_df.sort_index()
        sentiment_df = sentiment_df.sort_index()

        # Merge Logic:
        # We use merge_asof with direction='backward'.
        # For a price candle at 10:00, we look for the most recent sentiment reading
        # at or before 10:00. 
        # Tolerance: 12 hours. If no news in last 12h, sentiment = 0 (Neutral).
        
        if sentiment_df.empty:
            # No news at all -> all neutral
            combined = price_df.copy()
            combined["Sentiment"] = 0.0
        else:
            combined = pd.merge_asof(
                price_df,
                sentiment_df,
                left_index=True,
                right_index=True,
                direction="backward",
                tolerance=pd.Timedelta("12h")
            )
            # Fill NaN sentiment with 0.0 (Neutral)
            combined["Sentiment"] = combined["Sentiment"].fillna(0.0)

        # Drop any rows with missing price data
        combined = combined.dropna(subset=["Close", "Volume"])

        # Statistics
        non_zero = (combined["Sentiment"] != 0.0).sum()
        total = len(combined)
        # print(f"  Aligned {total} rows. Non-zero sentiment: {non_zero} ({non_zero/total:.1%})")
        
        return combined


if __name__ == "__main__":
    # Test
    loader = MarketDataLoader("AAPL")
    df = loader.get_aligned_data(days=10)
    print(df.tail(20))
    print(f"Non-zero sentiment count: {(df['Sentiment'] != 0).sum()}")

