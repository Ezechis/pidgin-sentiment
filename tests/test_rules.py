import pytest

from app.rules import classify_intent, moderate


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Abeg track my order, delivery rider dey use me play.", "Delivery / Logistics"),
        ("Where my package? Dispatch never arrive since Monday", "Delivery / Logistics"),
        ("Una debit me twice, return my money abeg", "Payment / Transaction"),
        ("My transfer never reflect and no reversal", "Payment / Transaction"),
        ("I no fit login, OTP no dey come", "Account / Access"),
        ("Make una send my token code, e no dey drop.", "Account / Access"),
        ("This app don crash again, network dey fail", "App / Network Performance"),
        ("Nigeria my country, we go dey alright.", "General Feedback"),
        ("", "General Feedback"),
    ],
)
def test_classify_intent(text, expected):
    assert classify_intent(text)["intent"] == expected


def test_intent_reports_matched_terms():
    result = classify_intent("Abeg track my order")
    assert set(result["matched"]) >= {"track", "order"}


def test_intent_picks_strongest_category():
    # Two payment cues beat one delivery cue.
    text = "The order come but una still debit me twice and no refund"
    assert classify_intent(text)["intent"] == "Payment / Transaction"


@pytest.mark.parametrize(
    "text",
    [
        "Una be big thief, return my double debit money",
        "Thunder fire una for this nonsense",
        "You be mumu for real",
        "Customer care na ode",
        "Oloshi, you no get sense",
    ],
)
def test_moderation_flags_abuse(text):
    assert moderate(text)["flagged"] is True


@pytest.mark.parametrize(
    "text",
    [
        # Substring traps that the old keyword check flagged.
        "Make una send my token code, e no dey drop.",
        "Switch to dark mode abeg",
        "The episode was sweet",
        "Na thieving I dey read about for history",  # 'thief' not a whole word here
        "This na scam alert, make una report am",  # complaint, not abuse
        "",
    ],
)
def test_moderation_ignores_clean_text(text):
    assert moderate(text)["flagged"] is False


def test_moderation_reports_matched_terms():
    result = moderate("Una be thief, thunder fire una")
    assert "thief" in result["matched"]
    assert "thunder fire" in result["matched"]
