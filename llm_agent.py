"""
LLM TRADING AGENT
The "brain" — uses an LLM to:
1. Analyze market context
2. Reason about news + technicals together
3. Give structured trading decision with explanation

Uses Anthropic Claude API (same model you're chatting with!)
"""

import json
import os
import requests
from dataclasses import dataclass

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


@dataclass
class AgentDecision:
    action: str          # BUY / SELL / HOLD
    confidence: str      # HIGH / MEDIUM / LOW
    reasoning: str       # LLM's explanation
    risk_level: str      # LOW / MEDIUM / HIGH
    position_size: str   # FULL / HALF / QUARTER


# ─────────────────────────────────────────
# LLM AGENT (Anthropic Claude)
# ─────────────────────────────────────────

def ask_trading_agent(
    symbol: str,
    current_price: float,
    price_change_7d: float,
    rsi: float,
    volatility: float,
    regime: str,
    tech_signal: str,
    sentiment_label: str,
    sentiment_score: float,
    top_headlines: list[str],
) -> AgentDecision:
    """
    Ask the LLM to make a trading decision.
    Gives all market context and gets structured response.
    """

    # Build context for LLM
    headlines_text = "\n".join(f"- {h}" for h in top_headlines[:5])

    prompt = f"""You are an expert quantitative trading analyst. Analyze the following market data and make a trading decision.

MARKET DATA FOR {symbol}:
- Current Price: ${current_price:,.2f}
- 7-Day Price Change: {price_change_7d:+.2f}%
- RSI (14): {rsi:.1f} (< 30 oversold, > 70 overbought)
- Volatility (7d): {volatility:.4f}
- Market Regime: {regime}

TECHNICAL SIGNAL: {tech_signal}

NEWS SENTIMENT:
- Overall Sentiment: {sentiment_label} (score: {sentiment_score:+.3f})
- Recent Headlines:
{headlines_text}

Based on this analysis, provide a trading decision in the following JSON format ONLY (no other text):
{{
  "action": "BUY or SELL or HOLD",
  "confidence": "HIGH or MEDIUM or LOW",
  "reasoning": "2-3 sentence explanation",
  "risk_level": "LOW or MEDIUM or HIGH",
  "position_size": "FULL or HALF or QUARTER"
}}

Rules:
- If RSI > 70 and sentiment is BEARISH → strong SELL signal
- If RSI < 30 and sentiment is BULLISH → strong BUY signal
- If regime is VOLATILE → reduce position size
- If conflicting signals → HOLD or reduce size
"""

    try:
        if not ANTHROPIC_API_KEY:
            print("[Agent] ANTHROPIC_API_KEY not set — using rule-based fallback")
            return _fallback_decision(tech_signal, sentiment_label)

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            text = data["content"][0]["text"].strip()

            # Parse JSON response
            # Remove markdown code blocks if present
            text = text.replace("```json", "").replace("```", "").strip()
            decision_data = json.loads(text)

            return AgentDecision(
                action       = decision_data.get("action", "HOLD"),
                confidence   = decision_data.get("confidence", "LOW"),
                reasoning    = decision_data.get("reasoning", "No reasoning provided"),
                risk_level   = decision_data.get("risk_level", "HIGH"),
                position_size= decision_data.get("position_size", "QUARTER"),
            )
        else:
            print(f"[Agent] API error {response.status_code} — using fallback")
            return _fallback_decision(tech_signal, sentiment_label)

    except Exception as e:
        print(f"[Agent] Error: {e} — using rule-based fallback")
        return _fallback_decision(tech_signal, sentiment_label)


def _fallback_decision(tech_signal: str, sentiment_label: str) -> AgentDecision:
    """Rule-based fallback when LLM is unavailable."""
    # Both agree → stronger signal
    if tech_signal == "BUY" and sentiment_label == "BULLISH":
        return AgentDecision("BUY", "MEDIUM", "Technical and sentiment both bullish.", "MEDIUM", "HALF")
    elif tech_signal == "SELL" and sentiment_label == "BEARISH":
        return AgentDecision("SELL", "MEDIUM", "Technical and sentiment both bearish.", "MEDIUM", "HALF")
    else:
        return AgentDecision("HOLD", "LOW", "Conflicting signals — waiting for clarity.", "HIGH", "QUARTER")


# ─────────────────────────────────────────
# WATCHDOG AGENT (Bot monitoring — CoinDCX specific)
# ─────────────────────────────────────────

def watchdog_agent(
    recent_returns: list[float],
    expected_sharpe: float,
    current_drawdown: float,
) -> dict:
    """
    Monitors trading bot performance.
    Detects anomalies / strategy decay.
    Like CoinDCX's 'Bot Monitoring Logic' requirement.
    """
    anomalies = []

    # Check 1: Drawdown too high
    if current_drawdown < -0.10:
        anomalies.append(f"HIGH DRAWDOWN: {current_drawdown*100:.1f}%")

    # Check 2: Recent performance degradation
    if len(recent_returns) >= 10:
        recent_sharpe = _quick_sharpe(recent_returns[-10:])
        if recent_sharpe < expected_sharpe * 0.5:
            anomalies.append(f"PERFORMANCE DECAY: Sharpe dropped to {recent_sharpe:.2f}")

    # Check 3: Unusual return pattern (CP concept: anomaly detection)
    if len(recent_returns) >= 5:
        avg = sum(recent_returns) / len(recent_returns)
        std = (sum((r - avg)**2 for r in recent_returns) / len(recent_returns)) ** 0.5
        latest = recent_returns[-1]
        if std > 0 and abs(latest - avg) > 3 * std:
            anomalies.append(f"RETURN ANOMALY: {latest*100:.2f}% is 3σ outlier")

    status = "ALERT" if anomalies else "HEALTHY"

    return {
        "status": status,
        "anomalies": anomalies,
        "recommendation": "PAUSE TRADING" if anomalies else "CONTINUE",
    }


def _quick_sharpe(returns: list[float]) -> float:
    if not returns:
        return 0.0
    avg = sum(returns) / len(returns)
    std = (sum((r - avg)**2 for r in returns) / len(returns)) ** 0.5
    return (avg / (std + 1e-9)) * (252 ** 0.5)
