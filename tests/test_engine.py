import pytest

from engine import analyze


def fake_classifier(texts):
    assert isinstance(texts, list), "classifier must receive a list"
    return [[{"label": "negative", "score": 0.8},
             {"label": "positive", "score": 0.15},
             {"label": "neutral", "score": 0.05}] for _ in texts]


def test_combines_model_and_rules():
    result = analyze("Una be thief! Return my money @OPay https://x.co", fake_classifier)
    assert result["sentiment"] == pytest.approx({"negative": 0.8, "neutral": 0.05, "positive": 0.15})
    assert list(result["sentiment"]) == ["negative", "neutral", "positive"]
    assert result["top_label"] == "negative"
    assert result["intent"] == "Payment / Transaction"
    assert result["flagged"] is True and "thief" in result["abuse_terms"]
    assert result["cleaned"] == "una be thief return my money"


def test_clean_message():
    result = analyze("Make una send my token code", fake_classifier)
    assert result["intent"] == "Account / Access"
    assert result["intent_terms"] == ["token", "code"]
    assert result["flagged"] is False


def test_empty_input_never_calls_model():
    def exploding_classifier(texts):
        raise AssertionError("model must not be called")

    assert analyze("  !!! 123 ", exploding_classifier) is None
    assert analyze(None, exploding_classifier) is None
