import pandas as pd
import pytest

from src import batch
from src.explainability import WordContribution, highlight_html
from src.predict import is_model_available


def test_highlight_marks_words_and_escapes_html():
    out = highlight_html("Claim your <b>prize</b> now", [WordContribution("prize", 0.5)], [WordContribution("now", -0.2)])
    assert "<mark" in out
    assert "&lt;b&gt;" in out and "<b>" not in out


def test_highlight_without_weights_still_escapes():
    assert highlight_html("<script>x</script>", [], []) == "&lt;script&gt;x&lt;/script&gt;"


def test_detect_text_column_by_name_and_by_length():
    assert batch.detect_text_column(pd.DataFrame({"id": [1], "Message": ["hi"]})) == "Message"
    df = pd.DataFrame({"id": [1, 2], "notes": ["a very long free-text column here", "another long text value"], "tag": ["x", "y"]})
    assert batch.detect_text_column(df) == "notes"


needs_model = pytest.mark.skipif(not is_model_available(), reason="model not trained")


@needs_model
def test_analyze_dataframe_adds_columns_and_keeps_rows():
    df = pd.DataFrame({"text": ["Win a FREE prize now!!! Click here", "See you at lunch", ""]})
    out = batch.analyze_dataframe(df, "text")
    assert len(out) == 3
    for col in ("prediction", "confidence_%", "spam_risk_score", "risk_category"):
        assert col in out.columns
    assert out.loc[2, "prediction"] == "N/A"


@needs_model
def test_analyze_dataframe_bad_column_and_row_cap(monkeypatch):
    df = pd.DataFrame({"text": ["hello"] * 6})
    with pytest.raises(ValueError):
        batch.analyze_dataframe(df, "missing")
    monkeypatch.setattr(batch, "MAX_BATCH_ROWS", 3)
    assert len(batch.analyze_dataframe(df, "text")) == 3
