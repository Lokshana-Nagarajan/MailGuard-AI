import os

import pandas as pd
import pytest

from src.data_loader import load_and_clean_dataset, load_raw_dataset, DatasetError


def test_dataset_file_exists():
    from src.data_loader import DEFAULT_DATA_PATH
    assert os.path.exists(DEFAULT_DATA_PATH), (
        "data/spam.csv is missing -- see README for download instructions."
    )


def test_load_and_clean_dataset_shape_and_columns():
    df, stats = load_and_clean_dataset()
    assert list(df.columns) == ["label", "message"]
    assert len(df) == stats.total_messages
    assert len(df) > 0


def test_labels_are_binary():
    df, _ = load_and_clean_dataset()
    assert set(df["label"].unique()).issubset({0, 1})


def test_no_duplicate_messages_after_cleaning():
    df, _ = load_and_clean_dataset()
    assert df["message"].duplicated().sum() == 0


def test_no_empty_messages():
    df, _ = load_and_clean_dataset()
    assert (df["message"].str.len() > 0).all()


def test_missing_file_raises_dataset_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(DatasetError):
        load_raw_dataset(str(missing_path))


def test_unknown_label_values_raise_dataset_error(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"v1": ["ham", "maybe"], "v2": ["hi there", "unclear message"]}).to_csv(
        bad_csv, index=False
    )
    with pytest.raises(DatasetError):
        load_and_clean_dataset(str(bad_csv))
