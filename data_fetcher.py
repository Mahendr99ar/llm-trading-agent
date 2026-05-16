"""
DATA FETCHER MODULE
- Crypto/Stock price data from yfinance
- News headlines from RSS feeds
"""

import yfinance as yf
import feedparser
import pandas as pd
from datetime import datetime, timedelta


# ─────────────────────────────────────────
# PRICE DATA
# ─────────────────────────────────────────

def get_mock_price_data(symbol: str = "BTC-USD", days: int = 90) -> pd.DataFrame:
    """Generate realistic mock price data for testing."""
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
    start_price = 65000 if "BTC" in symbol else (3500 if "ETH" in symbol else 180)
    returns = np.random.normal(0.001, 0.025, days)
    prices  = start_price * (1 + returns).cumprod()
    df = pd.DataFrame({
        "open":   prices * (1 + np.random.normal(0, 0.003, days)),
        "high":   prices * (1 + abs(np.random.normal(0, 0.01, days))),
        "low":    prices * (1 - abs(np.random.normal(0, 0.01, days))),
        "close":  prices,
        "volume": np.random.randint(1_000_000, 50_000_000, days).astype(float),
    }, index=dates)
    df["returns"]    = df["close"].pct_change()
    df["sma_7"]      = df["close"].rolling(7).mean()
    df["sma_21"]     = df["close"].rolling(21).mean()
    df["volatility"] = df["returns"].rolling(7).std()
    df["rsi"]        = compute_rsi(df["close"])
    df.dropna(inplace=True)
    print(f"[DataFetcher] {symbol}: {len(df)} rows (mock data)")
    return df


def get_price_data(symbol: str = "BTC-USD", period: str = "60d", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data.
    symbol  : 'BTC-USD', 'ETH-USD', 'AAPL', etc.
    period  : '7d', '30d', '60d', '1y'
    interval: '1d', '1h', '15m'
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]

    # Basic features
    df["returns"]    = df["close"].pct_change()
    df["sma_7"]      = df["close"].rolling(7).mean()
    df["sma_21"]     = df["close"].rolling(21).mean()
    df["volatility"] = df["returns"].rolling(7).std()
    df["rsi"]        = compute_rsi(df["close"])

    df.dropna(inplace=True)
    print(f"[DataFetcher] {symbol}: {len(df)} rows fetched")
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — momentum indicator."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────
# NEWS DATA
# ─────────────────────────────────────────

NEWS_FEEDS = {
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://coindesk.com/arc/outboundfeeds/rss/",
    ],
    "stocks": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSLA&region=US&lang=en-US",
    ],
}


def fetch_news(category: str = "crypto", max_articles: int = 20) -> list[dict]:
    """
    Fetch latest news headlines from RSS feeds.
    Returns list of {title, summary, published} dicts.
    """
    articles = []
    feeds = NEWS_FEEDS.get(category, NEWS_FEEDS["crypto"])

    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_articles]:
                articles.append({
                    "title":     entry.get("title", ""),
                    "summary":   entry.get("summary", "")[:300],
                    "published": entry.get("published", str(datetime.now())),
                    "source":    url,
                })
        except Exception as e:
            print(f"[NewsFetcher] Warning: {url} failed — {e}")

    print(f"[NewsFetcher] Fetched {len(articles)} articles")
    return articles[:max_articles]


# ─────────────────────────────────────────
# MOCK NEWS (fallback if no internet)
# ─────────────────────────────────────────

def get_mock_news() -> list[dict]:
    """Mock news for testing without internet."""
    return [
        {"title": "Bitcoin surges past $70,000 as ETF inflows hit record high",
         "summary": "Institutional demand drives BTC to new highs amid positive macro sentiment.",
         "published": "2024-03-15"},
        {"title": "Fed signals rate cuts — crypto markets rally strongly",
         "summary": "Lower interest rates expected to boost risk assets including crypto.",
         "published": "2024-03-14"},
        {"title": "Major exchange hack causes panic selling in crypto markets",
         "summary": "Security breach leads to $200M loss, Bitcoin drops 8% in hours.",
         "published": "2024-03-13"},
        {"title": "Ethereum upgrade reduces gas fees by 90%, adoption soars",
         "summary": "Technical improvements drive ETH network usage to all-time high.",
         "published": "2024-03-12"},
        {"title": "Crypto regulatory crackdown in Asia — markets uncertain",
         "summary": "New regulations announced causing fear among investors.",
         "published": "2024-03-11"},
    ]
