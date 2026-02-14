"""
scanner.py — Multi-Ticker Market Sentiment Scanner

Features:
  • Scans a user-defined watchlist
  • Fetches latest news (headline + summary)
  • Scores sentiment using BERT
  • Ranks tickers by sentiment score
  • Extracts "representative" news for display
"""

import pandas as pd
from tqdm import tqdm
from data_loader import MarketDataLoader

class MarketScanner:
    def __init__(self, tickers: list = None):
        if tickers is None:
            self.tickers = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"]
        else:
            self.tickers = tickers

    def scan_market(self):
        """
        Iterate over tickers, fetch their latest news, and aggregate sentiment.
        Returns a DataFrame sorted by sentiment.
        """
        results = []
        
        # Iterate tickers with progress bar logic (helper method if needed)
        for ticker in self.tickers:
            try:
                loader = MarketDataLoader(ticker)
                
                # We specifically want the raw news DF here, not the aligned time-series
                news_df = loader._fetch_yfinance_news()
                
                if news_df is None or news_df.empty:
                    continue
                    
                # Score them (using loader's text scorer directly or via pipeline)
                # To be efficient, let's reuse loader logic but manually to keep details
                
                # Make sure we have the full text column
                news_df["text_to_score"] = news_df["headline"] + ". " + news_df["summary"]
                
                # Predict
                news_df["sentiment"] = news_df["text_to_score"].apply(loader._score_sentiment)
                
                # Aggregate
                avg_sentiment = news_df["sentiment"].mean()
                num_articles = len(news_df)
                
                # Get the "Most Impactful" story (furthest from neutral)
                # i.e., max absolute sentiment
                news_df["abs_sent"] = news_df["sentiment"].abs()
                best_story_idx = news_df["abs_sent"].idxmax()
                top_story = news_df.loc[best_story_idx]
                
                results.append({
                    "Ticker": ticker,
                    "Sentiment Score": avg_sentiment,
                    "Sentiment Label": self._get_label(avg_sentiment),
                    "Articles": num_articles,
                    "Top Headline": top_story["headline"],
                    "Top Summary": top_story["summary"],
                    "Provider": top_story["provider"],
                    "URL": top_story["url"]
                })
                
            except Exception as e:
                print(f"Failed to scan {ticker}: {e}")
                continue
                
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        df = df.sort_values("Sentiment Score", ascending=False).reset_index(drop=True)
        return df

    def _get_label(self, score):
        if score >= 0.3: return "Bullish 🟢"
        if score <= -0.3: return "Bearish 🔴"
        return "Neutral 🟡"

if __name__ == "__main__":
    scanner = MarketScanner(["AAPL", "TSLA"])
    print(scanner.scan_market())
