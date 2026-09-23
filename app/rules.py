"""Rule-based intent routing and content moderation for Nigerian Pidgin text.

These are deliberately transparent keyword rules, not learned models: no labelled
Pidgin intent or abuse dataset was available in the project window. Every result
reports the exact terms that triggered it so the decision can be audited.
"""

import re

DEFAULT_INTENT = "General Feedback"

# Order matters: it breaks ties when two intents match the same number of terms.
INTENT_KEYWORDS = {
    "Payment / Transaction": [
        "debit", "debited", "double debit", "refund", "reversal", "reverse",
        "transfer", "money", "charge", "charged", "payment", "pay", "paid",
        "withdraw", "withdrawal", "credit", "credited", "wallet",
    ],
    "Account / Access": [
        "login", "log in", "sign in", "password", "otp", "token", "code", "pin",
        "account", "verify", "verification", "bvn", "blocked", "locked",
    ],
    "Delivery / Logistics": [
        "order", "orders", "delivery", "deliver", "rider", "dispatch", "package",
        "parcel", "track", "tracking", "waybill", "arrive", "shipment", "pickup",
    ],
    "App / Network Performance": [
        "app", "crash", "crashed", "network", "slow", "glitch", "bug", "error",
        "hang", "server", "update", "down",
    ],
}

# Insults, curses and threats common in Nigerian online speech.
# Complaints such as "scam" or "fraud" are intentionally NOT flagged:
# an angry customer reporting a scam is not being abusive.
ABUSIVE_TERMS = [
    "thief", "thieves", "ole", "oloshi", "olodo", "mumu", "ode", "werey",
    "idiot", "fool", "foolish", "stupid", "bastard", "ashawo",
    "thunder fire", "i go kill", "kill you",
]


def _compile(terms):
    # Longest first so "double debit" is matched before "debit".
    ordered = sorted(terms, key=len, reverse=True)
    alternation = "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in ordered)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


_INTENT_PATTERNS = {intent: _compile(terms) for intent, terms in INTENT_KEYWORDS.items()}
_ABUSE_PATTERN = _compile(ABUSIVE_TERMS)


def _find(pattern, text):
    return [re.sub(r"\s+", " ", m.lower()) for m in pattern.findall(text or "")]


def classify_intent(text):
    """Return {'intent': str, 'matched': [terms]} for the strongest keyword category."""
    best_intent, best_terms = DEFAULT_INTENT, []
    for intent, pattern in _INTENT_PATTERNS.items():
        terms = _find(pattern, text)
        if len(terms) > len(best_terms):
            best_intent, best_terms = intent, terms
    return {"intent": best_intent, "matched": best_terms}


def moderate(text):
    """Return {'flagged': bool, 'matched': [terms]} using whole-word matching."""
    terms = _find(_ABUSE_PATTERN, text)
    return {"flagged": bool(terms), "matched": terms}
