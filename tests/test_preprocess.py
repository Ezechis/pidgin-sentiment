from app.preprocess import normalize_for_model


def test_lowercases_and_strips_punctuation():
    assert normalize_for_model("Sapa dey CHOKE me!!!") == "sapa dey choke me"


def test_removes_urls_handles_and_digits():
    text = "@OPay_NG abeg check https://t.co/xyz my 2k never land"
    assert normalize_for_model(text) == "abeg check my k never land"


def test_keeps_hashtag_word_and_drops_emoji():
    assert normalize_for_model("#EndSARS 😡 e don do") == "endsars e don do"


def test_empty_and_whitespace():
    assert normalize_for_model("   ") == ""
    assert normalize_for_model(None) == ""
