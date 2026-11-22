"""
analytics.py
------------
Surface-level text statistics shown to the user alongside the ML
prediction (message length, URL count, uppercase words, etc).

IMPORTANT: these statistics are DISPLAYED as supporting context only.
They are never fed into the model and never used to override or
replace the ML model's decision -- the model always makes the actual
classification. This module exists purely for the "Email Content
Analysis" explainability feature in the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\w+\.(?:com|net|org|info|biz|xyz)\b)", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d")
_SPECIAL_CHAR_RE = re.compile(r"[^a-zA-Z0-9\s]")
_REPEATED_PUNCT_RE = re.compile(r"([!?.])\1{1,}")
_URGENCY_WORDS = {
    "urgent", "immediately", "now", "act", "limited", "offer", "winner",
    "won", "free", "claim", "congratulations", "prize", "cash", "click",
    "verify", "suspended", "expire", "guarantee", "risk-free", "alert",
}


@dataclass
class MessageStats:
    char_count: int
    word_count: int
    sentence_count: int
    url_count: int
    digit_count: int
    special_char_count: int
    uppercase_word_count: int
    repeated_punctuation_count: int
    urgency_word_count: int
    avg_word_length: float

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_message(text: str) -> MessageStats:
    """Compute surface-level statistics for a raw (unprocessed) message."""
    text = text or ""
    words = text.split()
    word_count = len(words)
    uppercase_words = [w for w in words if len(w) > 1 and w.isupper() and w.isalpha()]

    sentence_count = max(1, len(re.findall(r"[.!?]+", text))) if text.strip() else 0
    avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0.0

    return MessageStats(
        char_count=len(text),
        word_count=word_count,
        sentence_count=sentence_count,
        url_count=len(_URL_RE.findall(text)),
        digit_count=len(_DIGIT_RE.findall(text)),
        special_char_count=len(_SPECIAL_CHAR_RE.findall(text)),
        uppercase_word_count=len(uppercase_words),
        repeated_punctuation_count=len(_REPEATED_PUNCT_RE.findall(text)),
        urgency_word_count=sum(1 for w in words if w.strip(".,!?").lower() in _URGENCY_WORDS),
        avg_word_length=round(avg_word_len, 2),
    )


def risk_category(risk_score: float) -> str:
    """Map a 0-100 risk score to a human-readable UI category (interpretation only)."""
    if risk_score <= 30:
        return "Low Risk"
    if risk_score <= 60:
        return "Moderate Risk"
    if risk_score <= 80:
        return "High Risk"
    return "Very High Risk"
