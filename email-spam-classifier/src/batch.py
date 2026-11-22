"""
batch.py
--------
Testable batch-analysis logic (kept out of app.py so it can be unit tested).
"""

from __future__ import annotations

import pandas as pd

from src.predict import predict_batch

MAX_BATCH_ROWS = 5000
LIKELY_TEXT_COLUMNS = ("v2", "text", "message", "messages", "sms", "body", "email", "content")


def detect_text_column(df: pd.DataFrame) -> str:
    """Pick the most likely message column: known names first, else the text column with the longest average value."""
    lowered = {str(c).lower().strip(): c for c in df.columns}
    for name in LIKELY_TEXT_COLUMNS:
        if name in lowered:
            return lowered[name]
    best, best_len = df.columns[0], -1.0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        avg_len = df[col].fillna("").astype(str).str.len().mean()
        if avg_len > best_len:
            best, best_len = col, avg_len
    return best


def analyze_dataframe(df: pd.DataFrame, text_col: str, progress_callback=None, chunk_size: int = 100) -> pd.DataFrame:
    """Classify every row of `df[text_col]`; returns a copy with prediction columns appended."""
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in the uploaded file.")

    work = df.head(MAX_BATCH_ROWS).copy()
    messages = work[text_col].fillna("").astype(str).tolist()
    results = []
    for start in range(0, len(messages), chunk_size):
        results.extend(predict_batch(messages[start:start + chunk_size]))
        if progress_callback:
            progress_callback(min(len(messages), start + chunk_size), len(messages))

    work["prediction"] = [r.label for r in results]
    work["confidence_%"] = [r.confidence for r in results]
    work["spam_risk_score"] = [r.risk_score for r in results]
    work["risk_category"] = [r.risk_category for r in results]
    return work
