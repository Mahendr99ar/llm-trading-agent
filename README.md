# LLM Trading Agent
 
> Combines technical signals, news sentiment, and an LLM reasoning layer to generate trading decisions — then backtests them on real market data.
 
**Mahendra Meena | IIIT Gwalior | B.Tech EEE 2027**
 
---
 
I wanted to see if an LLM could actually reason about market conditions better than a pure rule-based system. Not just "RSI < 30 = buy" — but actually reading news, understanding regime, and making a judgment call. This project is the result of that experiment.
 
Three separate signals feed into the agent: technical indicators (SMA crossover, RSI, volume), news sentiment (rule-based or FinBERT), and then Claude's API makes the final call with reasoning. There's also a watchdog that monitors for strategy decay in real time.
 
---
 
## How it works
 
```
News Headlines ──┐
                 ├──► Sentiment Score ──┐
Price + Volume ──┘                      ├──► LLM Agent ──► BUY/SELL/HOLD
                 ┌──► Tech Signals ─────┘
RSI + SMA ───────┘
```
 
**Step 1 — Price data** via yfinance (90 days OHLCV), falls back to mock data if offline
 
**Step 2 — News sentiment** from CoinTelegraph/CoinDesk RSS feeds, scored by rule-based keyword model or FinBERT if you have transformers installed
 
**Step 3 — Technical signals** — SMA crossover (momentum), RSI (mean reversion), volume spike (confirmation), combined with weighted scoring. Regime detection (TRENDING / RANGING / VOLATILE) adjusts the weights dynamically
 
**Step 4 — LLM decision** — all context bundled into a prompt, Claude returns structured JSON: action, confidence, risk level, position size, and reasoning
 
**Step 5 — Watchdog** — checks for drawdown breach (>10%), Sharpe decay, and 3σ return anomalies. Flags PAUSE if anything looks wrong
 
**Step 6 — Backtest** — runs the signal logic on full 90-day history, reports Sharpe, drawdown, win rate, per-trade P&L
 
---
 
## Quick start
 
```bash
git clone https://github.com/Mahendr99ar/llm-trading-agent.git
cd llm-trading-agent
 
pip install -r requirements.txt
 
# Set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here   # Linux/Mac
set ANTHROPIC_API_KEY=your_key_here      # Windows
 
# Run with Bitcoin (default)
python main.py
 
# Run with Ethereum
python main.py --symbol ETH-USD
 
# Run with a stock
python main.py --symbol AAPL
 
# Skip LLM, just backtest
python main.py --backtest-only
```
 
On Windows there's a `run.bat` — just double-click after setting the API key.
 
---
 
## Project structure
 
```
llm-trading-agent/
├── main.py                 ← entry point, runs the full pipeline
├── data_fetcher.py         ← price data (yfinance) + news (RSS feeds)
├── sentiment_analyzer.py   ← rule-based scorer + FinBERT wrapper
├── signal_generator.py     ← technical signals, regime detection, signal combiner
├── llm_agent.py            ← Claude API call + watchdog logic
├── engine.py               ← backtesting: Sharpe, drawdown, win rate, trade log
├── requirements.txt
├── run.sh / run.bat        ← convenience scripts
└── results/                ← trade CSVs saved here (gitignored)
```
 
---
 
## Backtest results (real runs)
 
| Symbol | Trade | Entry | Exit | Return | Reason |
|--------|-------|-------|------|--------|--------|
| BTC-USD | 1 | $71,123 | $78,268 | +10.05% | Take Profit |
| ETH-USD | 1 | $2,241 | $2,253 | +0.52% | Signal |
| ETH-USD | 2 | $2,306 | $2,257 | -2.14% | Signal |
| AAPL | 1 | $260.57 | $246.40 | -5.44% | Stop Loss |
| AAPL | 2 | $253.55 | $279.88 | +10.38% | Take Profit |
 
Stop loss at 5%, take profit at 10%, 0.1% transaction cost per trade.
 
---
 
## Signals used
 
**Technical (60% weight in normal regime):**
- SMA 7 vs SMA 21 crossover — trend direction
- RSI mean reversion — oversold/overbought
- Volume spike confirmation — signal strength
- 3-day momentum — short-term continuation
**Regime-adjusted weights:**
- TRENDING → technicals 70%, sentiment 30%
- VOLATILE → technicals 40%, sentiment 60% (news matters more in chaos)
- RANGING → default 60/40
**Sentiment (40% weight):**
- Rule-based keyword scorer (always works, no dependencies)
- FinBERT option if you have `transformers` + `torch` installed — noticeably better on ambiguous headlines
---
 
## ⚠️ Known issues / things to fix
 
- **Backtest sentiment is a single fixed value** for the entire historical period — ideally you'd need timestamped historical sentiment, which requires a news archive API
- **FinBERT loads a new pipeline every call** — in production you'd cache the model at startup
- LLM decision isn't fed back into the backtest loop (backtest runs on pure technicals) — the two pipelines are separate
---
 
## What's next
 
- Live paper trading via Alpaca API
- Historical news archive for proper sentiment backtesting
- Multi-agent setup — one bull agent + one bear agent debate before final decision
- RL agent replacing or augmenting the rule-based signal layer
- Fine-tuning FinBERT on crypto-specific news corpus
---
 
**Mahendra Meena** — [LinkedIn](https://www.linkedin.com/in/mahendra-meena-72047b201/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BtZR9MhvxSn%2B%2BMaIaHZNDIw%3D%3D)
