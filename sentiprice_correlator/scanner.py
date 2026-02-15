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
        Iterate over tickers, fetch their latest news + fundamentals.
        Returns a DataFrame sorted by composite score (Sentiment + Fundamentals hint).
        """
        results = []
        
        # Iterate tickers with progress bar logic (helper method if needed)
        for ticker in self.tickers:
            try:
                loader = MarketDataLoader(ticker)
                
                # 1. Fetch Fundamentals (Tri-Modal)
                fund = loader.get_fundamentals()
                
                # 2. Fetch News & Score
                news_df = loader._fetch_yfinance_news()
                
                avg_sentiment = 0.0
                num_articles = 0
                top_story = {"headline": "No recent news", "summary": "", "provider": "", "url": ""}
                
                if news_df is not None and not news_df.empty:
                    # Enforce "Content" check:
                    # Filter out items with very short summaries to improve reliability
                    news_df["content_len"] = news_df["summary"].astype(str).apply(len)
                    valid_news = news_df[news_df["content_len"] > 20].copy()
                    
                    if valid_news.empty:
                         # Fallback to all if strict filtering leaves nothing
                         valid_news = news_df.copy()

                    # Score them
                    valid_news["text_to_score"] = valid_news["headline"] + ". " + valid_news["summary"]
                    valid_news["sentiment"] = valid_news["text_to_score"].apply(loader._score_sentiment)
                    
                    # Weighted Average? 
                    # For now, simple arithmetic mean is robust enough
                    avg_sentiment = valid_news["sentiment"].mean()
                    num_articles = len(valid_news)
                    
                    # representative story
                    valid_news["abs_sent"] = valid_news["sentiment"].abs()
                    best_story_idx = valid_news["abs_sent"].idxmax()
                    top_row = valid_news.loc[best_story_idx]
                    top_story = {
                        "headline": top_row["headline"],
                        "summary": top_row["summary"],
                        "provider": top_row["provider"],
                        "url": top_row["url"]
                    }

                # Construct result row with EXTENDED metrics
                results.append({
                    "Ticker": ticker,
                    "Sentiment Score": avg_sentiment,
                    "Sentiment Label": self._get_label(avg_sentiment),
                    "Articles": num_articles,
                    
                    # Top Story
                    "Top Headline": top_story["headline"],
                    "Top Summary": top_story["summary"],
                    "Provider": top_story["provider"],
                    "URL": top_story["url"],
                    
                    # Fundamentals
                    "Market Cap": fund.get("market_cap"),
                    "P/E Ratio": fund.get("trailing_pe"),
                    "Recommendation": fund.get("recommendation_key", "N/A").replace("_", " ").title(),
                    "ROE": fund.get("roe"),
                    "Debt/Eq": fund.get("debt_to_equity"),
                    "Profit Margin": fund.get("profit_margins")
                })
                
            except Exception as e:
                print(f"Failed to scan {ticker}: {e}")
                continue
                
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        # Sort by sentiment magnitude (absolute interest) or just raw score?
        # Let's sort by raw score descending (Bullish first)
        df = df.sort_values("Sentiment Score", ascending=False).reset_index(drop=True)
        return df

    def _get_label(self, score):
        if score >= 0.3: return "Bullish 🟢"
        if score <= -0.3: return "Bearish 🔴"
        return "Neutral 🟡"

if __name__ == "__main__":
    scanner = MarketScanner(["AAPL", "NVDA"])
    df = scanner.scan_market()
    print(df[["Ticker", "Sentiment Score", "P/E Ratio", "Recommendation"]])
