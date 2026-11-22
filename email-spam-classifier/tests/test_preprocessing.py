from src.preprocessing import clean_text


def test_lowercases_text():
    assert clean_text("HELLO World") == clean_text("hello world")


def test_removes_stopwords():
    cleaned = clean_text("this is a message about the meeting")
    assert "is" not in cleaned.split()
    assert "the" not in cleaned.split()


def test_keeps_useful_signal_words():
    cleaned = clean_text("This is completely FREE, you WIN now!")
    tokens = cleaned.split()
    assert "free" in tokens
    assert "win" in tokens
    assert "now" in tokens


def test_url_is_tokenized_not_deleted():
    cleaned = clean_text("Click here: http://scam-example.com now")
    assert "urltoken" in cleaned


def test_empty_input_returns_empty_string():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""


def test_output_is_consistent_between_calls():
    text = "Win cash prizes now!!! Call 12345 today."
    assert clean_text(text) == clean_text(text)
