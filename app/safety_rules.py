from __future__ import annotations

import re

SAFETY_PATTERNS = [
    re.compile(r"\bkill\s+myself\b"),
    re.compile(r"\bhurt\s+myself\b"),
    re.compile(r"\bsuicid(e|al)\b"),
    re.compile(r"\bself\s*harm\b"),
    re.compile(r"\bend\s+my\s+life\b"),
    re.compile(r"\bwant\s+to\s+die\b"),
    re.compile(r"\bdon'?t\s+want\s+to\s+live\b"),
    re.compile(r"\bwish\s+i\s+was\s+dead\b"),
]

def looks_like_safety_risk(text: str) -> bool:
    # Return True if the text looks like a safety risk (self-harm, suicide, etc.).
    return any(pattern.search(text) for pattern in SAFETY_PATTERNS)