import pytest

from src.predict import predict_message, predict_batch, is_model_available

pytestmark = pytest.mark.skipif(
    not is_model_available(),
    reason="Trained model not found -- run `python src/train_model.py` first.",
)

SPAM_SAMPLE = "Congratulations! You have won a $1000 prize. Click now to claim your reward!"
HAM_SAMPLE = "Hi, are we still meeting tomorrow at 10 AM? Please confirm."


def test_obvious_spam_is_classified_as_spam():
    result = predict_message(SPAM_SAMPLE)
    assert result.label == "SPAM"
    assert result.is_spam is True
    assert result.confidence > 50


def test_obvious_ham_is_classified_as_ham():
    result = predict_message(HAM_SAMPLE)
    assert result.label == "HAM"
    assert result.is_spam is False
    assert result.confidence > 50


def test_empty_input_does_not_crash_and_is_flagged():
    result = predict_message("")
    assert result.label == "N/A"
    assert result.error is not None


def test_whitespace_only_input_is_flagged():
    result = predict_message("    ")
    assert result.error is not None


def test_very_short_message_returns_a_result():
    result = predict_message("ok")
    assert result.label in ("SPAM", "HAM")


def test_very_long_message_does_not_crash():
    long_text = ("free prize winner click now " * 500).strip()
    result = predict_message(long_text)
    assert result.label in ("SPAM", "HAM")


def test_confidence_and_risk_are_within_0_100():
    result = predict_message(SPAM_SAMPLE)
    assert 0 <= result.confidence <= 100
    assert 0 <= result.risk_score <= 100
    assert 0 <= result.spam_probability <= 100


def test_risk_category_is_one_of_expected_buckets():
    result = predict_message(SPAM_SAMPLE)
    assert result.risk_category in ("Low Risk", "Moderate Risk", "High Risk", "Very High Risk")


def test_batch_prediction_returns_one_result_per_message():
    messages = [SPAM_SAMPLE, HAM_SAMPLE, "ok"]
    results = predict_batch(messages)
    assert len(results) == len(messages)


def test_explanation_text_is_non_empty_for_valid_input():
    result = predict_message(SPAM_SAMPLE)
    assert isinstance(result.explanation_text, str)
    assert len(result.explanation_text) > 0
