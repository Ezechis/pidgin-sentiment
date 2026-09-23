import importlib
import sys
from unittest import mock

import pytest


@pytest.fixture
def app_module(monkeypatch):
    # The UI layer isn't under test: stub gradio so importing app.py has no side effects.
    monkeypatch.setitem(sys.modules, "gradio", mock.MagicMock())
    sys.modules.pop("app.app", None)
    module = importlib.import_module("app.app")

    def fake_classifier(texts):
        assert isinstance(texts, list), "classifier must receive a list"
        return [[{"label": "negative", "score": 0.8},
                 {"label": "positive", "score": 0.15},
                 {"label": "neutral", "score": 0.05}] for _ in texts]

    monkeypatch.setattr(module, "_classifier", fake_classifier)
    return module


def test_analyze_combines_model_and_rules(app_module):
    sentiment, intent_md, mod_md, cleaned = app_module.analyze(
        "Una be thief! Return my money @OPay https://x.co"
    )
    assert sentiment == {"🔴 negative": 0.8, "🟢 positive": 0.15, "🟡 neutral": 0.05}
    assert "Payment / Transaction" in intent_md
    assert "Flagged" in mod_md and "`thief`" in mod_md
    assert cleaned == "una be thief return my money"


def test_analyze_clean_message(app_module):
    _, intent_md, mod_md, _ = app_module.analyze("Make una send my token code")
    assert "Account / Access" in intent_md
    assert mod_md.startswith("✅")


def test_analyze_empty_input_skips_model(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "_classifier", None)
    monkeypatch.setattr(app_module, "get_classifier", mock.Mock(side_effect=AssertionError))
    sentiment, intent_md, _, _ = app_module.analyze("  !!! ")
    assert sentiment == {}
    assert intent_md == "Type a sentence first."
