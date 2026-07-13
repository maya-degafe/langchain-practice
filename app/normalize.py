from __future__ import annotations

import re
import unicodedata


LEETSPEAK_MAP = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
})


def normalize_text(text: str) -> str:
    t = text.lower()
    t = unicodedata.normalize("NFKD", t)
    t = t.encode("ascii", "ignore").decode("ascii")
    t = t.translate(LEETSPEAK_MAP)

    # remove repeated punctuation / weird separators
    t = re.sub(r"[_\-\.\,\!\?\*]+", " ", t)

    # collapse repeated chars: "soooo" -> "soo"
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)

    # normalize whitespace
    t = " ".join(t.split())
    return t