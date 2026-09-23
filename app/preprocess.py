"""Make user input look like the NaijaSenti training text.

The dataset authors released tweets lower-cased with handles, URLs, digits,
punctuation and emoji removed. Feeding raw input with those characters would
show the model a distribution it never trained on.
"""

import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_HANDLE = re.compile(r"@\w+")
_NON_LETTER = re.compile(r"[^\w\s]|[\d_]")
_SPACES = re.compile(r"\s+")


def normalize_for_model(text):
    text = (text or "").lower()
    text = _URL.sub(" ", text)
    text = _HANDLE.sub(" ", text)
    text = _NON_LETTER.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
