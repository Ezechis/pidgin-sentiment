"""Core analysis: model sentiment + rule-based intent and moderation.

UI-agnostic so it can be unit-tested without Streamlit or a real model.
"""

from preprocess import normalize_for_model
from rules import classify_intent, moderate

LABEL_ORDER = ["negative", "neutral", "positive"]


def analyze(text, classifier):
    """Analyse one message.

    `classifier` is a transformers text-classification pipeline created with
    top_k=None (or anything with the same call signature).
    Returns None when the message has no usable text.
    """
    cleaned = normalize_for_model(text)
    if not cleaned:
        return None

    # A list input always returns one list of {label, score} per text.
    scores = {s["label"]: float(s["score"]) for s in classifier([cleaned])[0]}
    sentiment = {label: scores.get(label, 0.0) for label in LABEL_ORDER}
    intent = classify_intent(text)
    mod = moderate(text)
    return {
        "cleaned": cleaned,
        "sentiment": sentiment,
        "top_label": max(sentiment, key=sentiment.get),
        "intent": intent["intent"],
        "intent_terms": intent["matched"],
        "flagged": mod["flagged"],
        "abuse_terms": mod["matched"],
    }
