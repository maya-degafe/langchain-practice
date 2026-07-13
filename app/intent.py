from __future__ import annotations

from dataclasses import dataclass
import re

from app.normalize import normalize_text


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float


SAFETY_PATTERNS = [
    re.compile(r"\bkill\s+myself\b"),
    re.compile(r"\bhurt\s+myself\b"),
    re.compile(r"\bsuicid(e|al)\b"),
    re.compile(r"\bself\s*harm\b"),
    re.compile(r"\bend\s+my\s+life\b"),
    re.compile(r"\bwant\s+to\s+die\b"),
    re.compile(r"\bdon'?t\s+want\s+to\s+live\b"),
    re.compile(r"\bwish\s+i\s+was\s+dead\b"),
    re.compile(r"\boverdos(e|ing)\b"),
]

REPAIR_PATTERNS = [
    re.compile(r"\byou('?| a)?re\s+not\s+listening\b"),
    re.compile(r"\bthat('?| i)?s\s+not\s+what\s+i\s+asked\b"),
    re.compile(r"\bwrong\s+answer\b"),
    re.compile(r"\bnot\s+helpful\b"),
    re.compile(r"\byou\s+misunderstood\b"),
    re.compile(r"\byou\s+missed\s+the\s+point\b"),
    re.compile(r"\bthat\s+doesn'?t\s+answer\s+my\s+question\b"),
]

OFF_DOMAIN_PATTERNS = [
    re.compile(r"\bflight(s)?\b"),
    re.compile(r"\bairline(s)?\b"),
    re.compile(r"\bticket(s)?\b"),
    re.compile(r"\bbaggage\b"),
    re.compile(r"\bboarding\b"),
    re.compile(r"\bpassport(s)?\b"),
    re.compile(r"\bhotel(s)?\b"),
    re.compile(r"\bcar\s+rental\b"),
]


def _matches_any_pattern(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def classify_intent(text: str) -> IntentResult:
    t = normalize_text(text)

    if not t:
        return IntentResult("other", 0.60)

    if _matches_any_pattern(t, SAFETY_PATTERNS):
        return IntentResult("safety", 0.98)

    if _matches_any_pattern(t, REPAIR_PATTERNS):
        return IntentResult("repair", 0.95)

    if _matches_any_pattern(t, OFF_DOMAIN_PATTERNS):
        return IntentResult("other", 0.80)

    return IntentResult("retrieve_or_redirect", 0.70)