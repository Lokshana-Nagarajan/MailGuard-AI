"""
data_loader.py
--------------
Loads and validates the SMS/Email Spam Collection dataset.

The function in this module is deliberately defensive: real datasets
downloaded from Kaggle/UCI are messy (extra unnamed columns, mixed
encodings, duplicate rows, odd label spellings). We never fabricate
data -- if the dataset file is missing or unusable we raise a clear,
actionable error instead of silently generating fake examples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

# Default location the rest of the project expects the dataset at.
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "spam.csv"
)

# Encodings we try, in order, because the public Kaggle CSV is Latin-1,
# not UTF-8, and will raise a UnicodeDecodeError if read naively.
_CANDIDATE_ENCODINGS = ["utf-8", "latin-1", "cp1252"]

# Column name variants we try to auto-detect, covering the common
# re-uploads of this dataset found in the wild.
_LABEL_COLUMN_CANDIDATES = ["v1", "label", "labels", "category", "class", "target"]
_TEXT_COLUMN_CANDIDATES = ["v2", "text", "message", "messages", "sms", "body", "email"]


class DatasetError(Exception):
    """Raised when the spam dataset cannot be found or is structurally invalid."""


@dataclass
class DatasetStats:
    total_messages: int
    spam_count: int
    ham_count: int
    spam_percentage: float
    duplicates_removed: int
    missing_values_removed: int


def _find_column(columns, candidates) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def load_raw_dataset(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the CSV file from disk, trying a few sensible encodings."""
    if not os.path.exists(path):
        raise DatasetError(
            "Dataset not found.\n\n"
            f"Expected the SMS Spam Collection CSV at:\n    {path}\n\n"
            "Download it from Kaggle:\n"
            "    https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset\n"
            "and save it as 'spam.csv' inside the project's 'data/' folder, "
            "then re-run training."
        )

    last_error: Exception | None = None
    for enc in _CANDIDATE_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:  # pragma: no cover
            last_error = exc
            continue

    raise DatasetError(
        f"Could not read '{path}' with any of the supported encodings "
        f"{_CANDIDATE_ENCODINGS}. Original error: {last_error}"
    )


def load_and_clean_dataset(path: str = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, DatasetStats]:
    """
    Load the dataset, standardize columns, clean it, and return a tidy
    DataFrame with exactly two columns: 'label' (0/1) and 'message' (str),
    plus a DatasetStats summary describing what was found/removed.

    Raises DatasetError if the file is missing or the columns cannot be
    identified -- callers should surface this message to the user rather
    than fabricate training data.
    """
    df = load_raw_dataset(path)

    # Drop the unnamed "ghost" columns the Kaggle export is known to include.
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    label_col = _find_column(df.columns, _LABEL_COLUMN_CANDIDATES)
    text_col = _find_column(df.columns, _TEXT_COLUMN_CANDIDATES)

    if label_col is None or text_col is None:
        raise DatasetError(
            "Could not identify the label/message columns in the dataset. "
            f"Columns found: {list(df.columns)}. Expected something like "
            "'v1'/'v2' (Kaggle) or 'label'/'message'."
        )

    df = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "message"})

    missing_before = len(df)
    df = df.dropna(subset=["label", "message"])
    missing_removed = missing_before - len(df)

    # Normalize the message to a plain string and the label to lowercase text
    # before mapping to binary, so "Spam", " spam ", "SPAM" all match.
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() > 0]
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    valid_labels = {"ham", "spam"}
    unknown = set(df["label"].unique()) - valid_labels
    if unknown:
        raise DatasetError(
            f"Dataset contains unexpected label values {unknown}; "
            "expected only 'ham' and 'spam'."
        )

    df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)

    before_dupes = len(df)
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    duplicates_removed = before_dupes - len(df)

    total = len(df)
    spam_count = int((df["label"] == 1).sum())
    ham_count = int((df["label"] == 0).sum())
    stats = DatasetStats(
        total_messages=total,
        spam_count=spam_count,
        ham_count=ham_count,
        spam_percentage=round(100 * spam_count / total, 2) if total else 0.0,
        duplicates_removed=duplicates_removed,
        missing_values_removed=missing_removed,
    )

    return df, stats


if __name__ == "__main__":
    data, info = load_and_clean_dataset()
    print("Dataset loaded successfully.")
    print(data.head())
    print(info)
