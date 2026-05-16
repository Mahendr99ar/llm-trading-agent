"""
MAIN — LLM TRADING AGENT
========================================
Run this file to:
  1. Fetch price + news data
  2. Analyze sentiment
  3. Generate trading signals
  4. Ask LLM agent for decision
  5. Backtest the strategy
  6. Print results

Usage:
  python main.py                    # BTC default
  python main.py --symbol ETH-USD   # Ethereum
  python main.py --symbol AAPL      # Apple stock
  python main.py --backtest-only    # Skip LLM, just backtest
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher       import get_price_data, get_mock_price_data, fetch_news, get_mock_news
from sentiment_analyzer import analyze_news_batch
from signal_generator   import generate_signal, generate_signals_batch
from llm_agent          import ask_trading_agent, watchdog_agent
from engine             import run_backtest, print_results


def run_pipeline(symbol: str = "BTC-USD", use_live_news: bool = True, backtest_only: bool = False):

    print("\n" + "="*60)
    print(f"  LLM TRADING AGENT — {symbol}")
    print("="*60)

    # ── STEP 1: Fetch Price Data ──────────────────────────────
    print("\n[Step 1] Fetching price data...")
    try:
        df = get_price_data(symbol=symbol, period="90d", interval="1d")
        if df.empty:
            raise ValueError("Empty dataframe")
    except Exception:
        print("         Live data unavailable — using mock data")
        df = get_mock_price_data(symbol=symbol, days=90)
    print(f"         Latest close: ${df['close'].iloc[-1]:,.2f}")
    print(f"         7d change   : {df['returns'].tail(7).sum()*100:+.2f}%")

    # ── STEP 2: Fetch & Analyze News ─────────────────────────
    print("\n[Step 2] Fetching news & analyzing sentiment...")
    category = "crypto" if "USD" in symbol or "BTC" in symbol or "ETH" in symbol else "stocks"

    if use_live_news:
        try:
            articles = fetch_news(category=category, max_articles=15)
            if not articles:
                raise ValueError("No articles fetched")
        except Exception:
            print("         Live news failed — using mock news")
            articles = get_mock_news()
    else:
        articles = get_mock_news()

    sentiment = analyze_news_batch(articles, use_finbert=False)

    # ── STEP 3: Generate Technical Signal ────────────────────
    print("\n[Step 3] Generating technical signals...")
    latest       = df.iloc[-1]
    tech_signal_obj = generate_signal(df, sentiment)
    tech_action  = tech_signal_obj.action

    # ── STEP 4: Ask LLM Agent ─────────────────────────────────
    if not backtest_only:
        print("\n[Step 4] Consulting LLM Trading Agent...")
        headlines = [a["title"] for a in articles[:5]]
        price_change_7d = df["returns"].tail(7).sum() * 100

        decision = ask_trading_agent(
            symbol          = symbol,
            current_price   = float(latest["close"]),
            price_change_7d = price_change_7d,
            rsi             = float(latest["rsi"]),
            volatility      = float(latest["volatility"]),
            regime          = tech_signal_obj.regime,
            tech_signal     = tech_action,
            sentiment_label = sentiment["label"],
            sentiment_score = sentiment["aggregate_score"],
            top_headlines   = headlines,
        )

        print(f"\n  ┌─ AGENT DECISION ─────────────────────────────")
        print(f"  │  Action      : {decision.action}")
        print(f"  │  Confidence  : {decision.confidence}")
        print(f"  │  Risk Level  : {decision.risk_level}")
        print(f"  │  Position    : {decision.position_size}")
        print(f"  │  Reasoning   : {decision.reasoning}")
        print(f"  └──────────────────────────────────────────────")

    # ── STEP 5: Watchdog Check ────────────────────────────────
    print("\n[Step 5] Watchdog monitoring check...")
    recent_returns = df["returns"].tail(30).tolist()
    watch = watchdog_agent(
        recent_returns   = recent_returns,
        expected_sharpe  = 1.0,
        current_drawdown = float((df["close"].iloc[-1] - df["close"].tail(30).max()) /
                                  df["close"].tail(30).max()),
    )
    print(f"         Status: {watch['status']}")
    if watch["anomalies"]:
        for a in watch["anomalies"]:
            print(f"         ⚠️  {a}")
    print(f"         Recommendation: {watch['recommendation']}")

    # ── STEP 6: Backtest ──────────────────────────────────────
    print("\n[Step 6] Running backtest on 90 days of data...")
    df_signals = generate_signals_batch(df, sentiment_score=sentiment["aggregate_score"])
    result, trades_df = run_backtest(df_signals, initial_capital=100_000)
    print_results(result, symbol=symbol)

    # Save trades
    if not trades_df.empty:
        os.makedirs("results", exist_ok=True)
        trades_df.to_csv(f"results/{symbol.replace('-','_')}_trades.csv", index=False)
        print(f"  Trades saved to results/{symbol.replace('-','_')}_trades.csv")

    return result


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Parse simple args
    args        = sys.argv[1:]
    symbol      = "BTC-USD"
    backtest_only = False

    if "--symbol" in args:
        idx    = args.index("--symbol")
        symbol = args[idx + 1]
    if "--backtest-only" in args:
        backtest_only = True

    run_pipeline(
        symbol        = symbol,
        use_live_news = True,
        backtest_only = backtest_only,
    )
