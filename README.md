# LLM Trading Agent 🤖📈

> WorldQuant BRAIN-style alpha research + LLM-powered trading decisions

---

## What This Project Does

```
News Headlines ──┐
                 ├──► Sentiment Score ──┐
Price + Volume ──┘                      ├──► LLM Agent ──► BUY/SELL/HOLD
                 ┌──► Tech Signals ─────┘
RSI + SMA ───────┘
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with Bitcoin (default)
python main.py

# Run with Ethereum
python main.py --symbol ETH-USD

# Run with a stock
python main.py --symbol AAPL

# Skip LLM, just backtest
python main.py --backtest-only
```

---

## Project Structure

```
llm-trading-agent/
├── main.py                    ← Run this
├── requirements.txt
├── src/
│   ├── data_fetcher.py        ← Price + News data
│   ├── sentiment_analyzer.py  ← News → Sentiment score
│   ├── signal_generator.py    ← Technical signals (like BRAIN alphas)
│   └── llm_agent.py           ← LLM decision maker + Watchdog
├── backtest/
│   └── engine.py              ← Sharpe, drawdown, win rate
└── results/
    └── *.csv                  ← Trade logs saved here
```

---

## Concepts Covered

### Quant Concepts
- Alpha signals: SMA crossover, RSI, volume confirmation
- Sharpe ratio calculation
- Max drawdown
- OOS backtesting
- Regime detection (trending/ranging/volatile)
- Position sizing based on risk

### ML/LLM Concepts
- Sentiment analysis (rule-based + FinBERT)
- LLM-powered reasoning (Claude API)
- Agentic workflows
- Anomaly detection (watchdog)

### Competitive Programming Concepts
- State machine (regime detection)
- Weighted aggregation algorithms
- Signal combination optimization

---

## How to Explain in Interview

**"I built an autonomous trading agent that combines three signals:"**

1. **Technical signals** — like WorldQuant BRAIN alphas (SMA crossover, RSI mean reversion, volume confirmation)

2. **Sentiment signals** — scraping financial news RSS feeds → FinBERT sentiment analysis → aggregate score

3. **LLM reasoning** — Claude API analyzes all context and gives structured BUY/SELL/HOLD with reasoning

**"The system also has a watchdog that monitors for strategy decay — detecting when drawdown or Sharpe degrades beyond threshold."**

**"Backtest on 90 days of BTC data showed Sharpe of X with Y% return."**

---

## Extend This Project

- Add more assets (gold, forex)
- Add reinforcement learning agent (RL)
- Add live paper trading via Alpaca API
- Fine-tune FinBERT on crypto-specific news
- Add multi-agent: one bull agent + one bear agent debate
