"""
SIGNAL GENERATOR MODULE
Combines:
  1. Technical signals (price-based — like BRAIN alphas)
  2. Sentiment signals (news-based)
  3. Regime detection (market condition)

Output: BUY / SELL / HOLD with confidence score
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class TradingSignal:
    date: str
    action: str          # BUY / SELL / HOLD
    confidence: float    # 0.0 to 1.0
    tech_score: float    # technical signal
    sent_score: float    # sentiment signal
    regime: str          # TRENDING / RANGING / VOLATILE
    reason: str          # human readable explanation


# ─────────────────────────────────────────
# TECHNICAL SIGNALS (Like BRAIN alphas)
# ─────────────────────────────────────────

def compute_technical_score(df: pd.DataFrame) -> pd.Series:
    """
    Combines multiple technical signals into one score.
    Range: -1.0 (strong sell) to +1.0 (strong buy)
    
    Signals used:
    - SMA crossover (momentum)
    - RSI (mean reversion)
    - Volume spike (confirmation)
    - Volatility (risk filter)
    """
    scores = pd.DataFrame(index=df.index)

    # Signal 1: SMA Crossover (Momentum)
    # Like Alpha#101 from the paper — trend following
    sma_signal = np.where(df["sma_7"] > df["sma_21"], 1.0, -1.0)
    scores["sma"] = sma_signal

    # Signal 2: RSI Mean Reversion
    # Oversold (RSI < 30) → BUY, Overbought (RSI > 70) → SELL
    rsi_signal = pd.Series(0.0, index=df.index)
    rsi_signal[df["rsi"] < 30] =  1.0   # oversold → buy
    rsi_signal[df["rsi"] > 70] = -1.0   # overbought → sell
    rsi_signal[(df["rsi"] >= 30) & (df["rsi"] <= 70)] = 0.0
    scores["rsi"] = rsi_signal

    # Signal 3: Volume Confirmation
    # High volume on up day = strong signal
    avg_volume = df["volume"].rolling(20).mean()
    vol_ratio  = df["volume"] / (avg_volume + 1e-9)
    return_sign = np.sign(df["returns"])
    scores["volume"] = (vol_ratio - 1).clip(-1, 1) * return_sign

    # Signal 4: Short-term momentum (returns based)
    # Like -1 * delta(close, 1) but combined
    short_momentum = df["returns"].rolling(3).mean()
    scores["momentum"] = short_momentum.clip(-0.05, 0.05) / 0.05

    # Weighted combination
    weights = {"sma": 0.3, "rsi": 0.25, "volume": 0.2, "momentum": 0.25}
    combined = sum(scores[k] * v for k, v in weights.items())

    return combined.clip(-1.0, 1.0)


# ─────────────────────────────────────────
# REGIME DETECTION (CP concept: state machine)
# ─────────────────────────────────────────

def detect_regime(df: pd.DataFrame) -> str:
    """
    Detect current market regime.
    Uses volatility + trend strength.
    
    TRENDING  → clear directional move
    RANGING   → sideways, mean-reversion works
    VOLATILE  → high uncertainty, reduce position size
    """
    if len(df) < 5:
        return "UNKNOWN"

    recent = df.tail(5)
    vol    = recent["volatility"].mean()
    trend  = abs(recent["sma_7"].iloc[-1] - recent["sma_21"].iloc[-1]) / recent["close"].mean()

    if vol > 0.04:
        return "VOLATILE"
    elif trend > 0.02:
        return "TRENDING"
    else:
        return "RANGING"


# ─────────────────────────────────────────
# MAIN SIGNAL GENERATOR
# ─────────────────────────────────────────

def generate_signal(
    df: pd.DataFrame,
    sentiment: dict,
    tech_weight: float = 0.6,
    sent_weight: float = 0.4,
) -> TradingSignal:
    """
    Generate final trading signal combining technical + sentiment.
    
    tech_weight + sent_weight should = 1.0
    """
    # Get latest price data
    latest = df.iloc[-1]
    date   = str(df.index[-1].date())

    # Technical score
    tech_scores = compute_technical_score(df)
    tech_score  = float(tech_scores.iloc[-1])

    # Sentiment score
    sent_score = sentiment.get("aggregate_score", 0.0)

    # Regime
    regime = detect_regime(df)

    # Adjust weights based on regime
    if regime == "VOLATILE":
        # In volatile market, reduce position — trust sentiment more
        tech_weight, sent_weight = 0.4, 0.6
    elif regime == "TRENDING":
        # In trending market, trust technicals more
        tech_weight, sent_weight = 0.7, 0.3

    # Combined score
    final_score = (tech_score * tech_weight) + (sent_score * sent_weight)

    # Volatility-based confidence penalty
    vol_penalty = min(1.0, 0.03 / (latest["volatility"] + 1e-9))
    confidence  = min(1.0, abs(final_score) * vol_penalty)

    # Decision
    if final_score > 0.15:
        action = "BUY"
    elif final_score < -0.15:
        action = "SELL"
    else:
        action = "HOLD"

    # Human readable reason
    sent_label = sentiment.get("label", "NEUTRAL")
    reason = (
        f"Tech:{tech_score:+.2f} | Sent:{sent_score:+.2f} ({sent_label}) | "
        f"Regime:{regime} | Final:{final_score:+.2f}"
    )

    signal = TradingSignal(
        date=date,
        action=action,
        confidence=round(confidence, 3),
        tech_score=round(tech_score, 4),
        sent_score=round(sent_score, 4),
        regime=regime,
        reason=reason,
    )

    print(f"[Signal] {date} → {action} (conf:{confidence:.2f}) | {reason}")
    return signal


# ─────────────────────────────────────────
# BATCH SIGNAL GENERATION (for backtesting)
# ─────────────────────────────────────────

def generate_signals_batch(df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
    """
    Generate signals for every row — used in backtesting.
    sentiment_score: fixed value for historical backtest.
    """
    tech_scores = compute_technical_score(df)

    signals = []
    for i in range(len(df)):
        if i < 21:  # Need enough history
            signals.append("HOLD")
            continue

        sub_df    = df.iloc[:i+1]
        tech      = float(tech_scores.iloc[i])
        combined  = tech * 0.6 + sentiment_score * 0.4

        if combined > 0.15:
            signals.append("BUY")
        elif combined < -0.15:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    df = df.copy()
    df["signal"] = signals
    return df
