from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyResult:
    """Outcome of a safety scan.

    category is one of: "none", "self_harm", "security_threat".
    """

    is_risk: bool
    category: str


# Self-harm / crisis language
SELF_HARM_PATTERNS = [
    re.compile(r"\bkill\s+myself\b"),
    re.compile(r"\bkill\s+my\s+self\b"),
    re.compile(r"\bhurt\s+myself\b"),
    re.compile(r"\bharm\s+myself\b"),
    re.compile(r"\bsuicid(e|al)\b"),
    re.compile(r"\bself\s*harm\b"),
    re.compile(r"\bend\s+my\s+life\b"),
    re.compile(r"\btake\s+my\s+(own\s+)?life\b"),
    re.compile(r"\bwant\s+to\s+die\b"),
    re.compile(r"\bdon'?t\s+want\s+to\s+live\b"),
    re.compile(r"\bwish\s+i\s+(was|were)\s+dead\b"),
    re.compile(r"\bwant\s+to\s+end\s+it\s+all\b"),
    re.compile(r"\boverdos(e|ing)\b"),
]

# Airline security threats 
# For an airline assistant, threats of violence toward aircraft/passengers must
# be caught and handled seriously (canned response + escalation), never fed to
# the LLM as a normal question.
SECURITY_THREAT_PATTERNS = [
    re.compile(r"\bbomb\b"),
    re.compile(r"\bexplosive(s)?\b"),
    re.compile(r"\bdetonat(e|or|ion)\b"),
    re.compile(r"\bhijack(ing|er)?\b"),
    re.compile(r"\bblow\s+up\b"),
    re.compile(r"\bshoot\b"),
    re.compile(r"\bweapon(s)?\b"),
    re.compile(r"\bgun\b"),
    re.compile(r"\bknife\b"),
    re.compile(r"\bkill\s+(everyone|passengers|people|the\s+crew|the\s+pilot)\b"),
    re.compile(r"\bterror(ist|ism)?\b"),
    re.compile(r"\bcrash\s+the\s+plane\b"),
]


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation so word-boundary patterns match reliably.

    Kept intentionally lightweight and dependency-free. If you already pass
    normalized text (e.g. from app.normalize.normalize_text), this is harmless.
    """
    t = text.lower()
    t = re.sub(r"[^a-z0-9'\s]+", " ", t)  
    t = " ".join(t.split())               
    return t


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def scan_safety(text: str) -> SafetyResult:
    """Classify a message for safety risks.

    Security threats are checked first because they require the most serious
    handling for an airline. Self-harm is checked next.
    """
    if not text or not text.strip():
        return SafetyResult(is_risk=False, category="none")

    normalized = _normalize(text)

    if _matches_any(normalized, SECURITY_THREAT_PATTERNS):
        return SafetyResult(is_risk=True, category="security_threat")

    if _matches_any(normalized, SELF_HARM_PATTERNS):
        return SafetyResult(is_risk=True, category="self_harm")

    return SafetyResult(is_risk=False, category="none")


def looks_like_safety_risk(text: str) -> bool:
    """Backwards-compatible boolean helper.

    Returns True for any self-harm or security-threat language. Existing callers
    that only need a yes/no answer can keep using this.
    """
    return scan_safety(text).is_risk

