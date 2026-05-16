"""
SENTIMENT ANALYZER MODULE
- HuggingFace FinBERT model for financial sentiment
- Rule-based fallback (no API key needed)
- Output: score from -1.0 (bearish) to +1.0 (bullish)
"""

import re
from dataclasses import dataclass


@dataclass
class SentimentResult:
    text: str
    score: float        # -1.0 to +1.0
    label: str          # BULLISH / BEARISH / NEUTRAL
    confidence: float   # 0.0 to 1.0


# ─────────────────────────────────────────
# RULE-BASED SENTIMENT (Always works, no API needed)
# ─────────────────────────────────────────

BULLISH_WORDS = [
    "surge", "rally", "soar", "gain", "rise", "bull", "breakout",
    "record", "high", "growth", "adoption", "positive", "strong",
    "institutional", "etf", "approval", "upgrade", "milestone",
    "pump", "moon", "green", "profit", "buy", "long",
]

BEARISH_WORDS = [
    "crash", "drop", "fall", "decline", "bear", "sell", "loss",
    "hack", "scam", "ban", "regulation", "crackdown", "fear",
    "panic", "uncertainty", "risk", "dump", "red", "short",
    "liquidation", "FUD", "fraud", "investigation",
]

INTENSIFIERS = ["major", "massive", "huge", "significant", "record", "extreme"]


def rule_based_sentiment(text: str) -> SentimentResult:
    """
    Simple keyword-based sentiment scorer.
    Good baseline — works without any model.
    """
    text_lower = text.lower()
    words = re.findall(r'\w+', text_lower)

    bull_count = sum(1 for w in words if w in BULLISH_WORDS)
    bear_count = sum(1 for w in words if w in BEARISH_WORDS)
    intensifier = sum(1 for w in words if w in INTENSIFIERS)

    # Intensifiers boost the dominant signal
    if bull_count > bear_count:
        bull_count += intensifier * 0.5
    elif bear_count > bull_count:
        bear_count += intensifier * 0.5

    total = bull_count + bear_count + 1e-9
    score = (bull_count - bear_count) / total

    # Normalize to [-1, 1]
    score = max(-1.0, min(1.0, score * 2))
    confidence = min(1.0, total / 5)

    if score > 0.1:
        label = "BULLISH"
    elif score < -0.1:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return SentimentResult(text=text[:100], score=score, label=label, confidence=confidence)


# ─────────────────────────────────────────
# FINBERT SENTIMENT (Better accuracy, needs transformers)
# ─────────────────────────────────────────

def finbert_sentiment(text: str) -> SentimentResult:
    """
    FinBERT — financial domain BERT model.
    pip install transformers torch
    """
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,
        )
        results = pipe(text[:512])[0]
        scores = {r["label"]: r["score"] for r in results}

        positive = scores.get("positive", 0)
        negative = scores.get("negative", 0)
        neutral  = scores.get("neutral", 0)

        score = positive - negative  # -1 to +1
        confidence = max(positive, negative, neutral)

        if score > 0.1:
            label = "BULLISH"
        elif score < -0.1:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        return SentimentResult(text=text[:100], score=score, label=label, confidence=confidence)

    except ImportError:
        print("[Sentiment] transformers not installed — using rule-based fallback")
        return rule_based_sentiment(text)
    except Exception as e:
        print(f"[Sentiment] FinBERT error: {e} — using rule-based fallback")
        return rule_based_sentiment(text)


# ─────────────────────────────────────────
# AGGREGATE SENTIMENT FROM MULTIPLE NEWS
# ─────────────────────────────────────────

def analyze_news_batch(articles: list[dict], use_finbert: bool = False) -> dict:
    """
    Analyze multiple news articles and return aggregated sentiment.
    
    Returns:
        {
            "aggregate_score": float,    # -1 to +1
            "label": str,                # BULLISH/BEARISH/NEUTRAL
            "bullish_count": int,
            "bearish_count": int,
            "neutral_count": int,
            "details": list[SentimentResult]
        }
    """
    if not articles:
        return {"aggregate_score": 0.0, "label": "NEUTRAL",
                "bullish_count": 0, "bearish_count": 0, "neutral_count": 0, "details": []}

    analyzer = finbert_sentiment if use_finbert else rule_based_sentiment
    results  = []

    for article in articles:
        text = article.get("title", "") + " " + article.get("summary", "")
        result = analyzer(text)
        results.append(result)

    # Weighted average (higher confidence = more weight)
    total_weight = sum(r.confidence for r in results) + 1e-9
    aggregate    = sum(r.score * r.confidence for r in results) / total_weight

    bullish = sum(1 for r in results if r.label == "BULLISH")
    bearish = sum(1 for r in results if r.label == "BEARISH")
    neutral = sum(1 for r in results if r.label == "NEUTRAL")

    if aggregate > 0.1:
        label = "BULLISH"
    elif aggregate < -0.1:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    print(f"[Sentiment] Score: {aggregate:.3f} | {label} | "
          f"Bull:{bullish} Bear:{bearish} Neutral:{neutral}")

    return {
        "aggregate_score": round(aggregate, 4),
        "label": label,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "details": results,
    }
