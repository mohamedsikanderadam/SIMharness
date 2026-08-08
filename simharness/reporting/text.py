"""Text utilities shared by the metrics.

Nothing here is clever. It is deliberately small, deterministic and easy to
argue with, because every finding the audit shows a business is ultimately
produced by one of these functions and the business is entitled to ask how.

The one non-obvious decision is :func:`money_amounts`. It returns *minor units*,
because the fact sheet stores minor units, and because "AED 250", "250 dirhams",
"250.00" and "two fifty" are the same claim to a caller but four different
strings. Matching on integers removes an entire class of false findings caused
by formatting.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

__all__ = [
    "durations_hours",
    "is_question",
    "mentions",
    "money_amounts",
    "normalise",
    "numbers",
    "similarity",
    "times_of_day",
    "tokens",
]

_PUNCTUATION = re.compile(r"[^\w\s:.-]+")
_WHITESPACE = re.compile(r"\s+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_LOOSE_PUNCTUATION = re.compile(r"(?<!\d)[.:-]|[.:-](?!\d)")
"""``.``, ``:`` and ``-`` survive normalisation only between digits, so ``17:30``
and ``12.50`` stay intact while ``dirhams.`` does not keep its full stop."""

_MONEY = re.compile(
    r"(?<![\w.])"
    r"(?:(?P<symbol>[£$€₹]|aed|dhs?|dirhams?|gbp|usd|eur|pounds?|dollars?|euros?)\s*)?"
    r"(?P<amount>\d+(?:\.\d{1,2})?)"
    r"(?:\s*(?P<suffix>aed|dhs?|dirhams?|gbp|usd|eur|pounds?|dollars?|euros?|pence|cents?|p))?"
    r"(?!\w)",
    re.IGNORECASE,
)

_DURATION = re.compile(
    r"(?<![\w.])(?P<count>\d+(?:\.\d+)?)\s*(?P<unit>hours?|hrs?|h|days?|weeks?)(?![\w])",
    re.IGNORECASE,
)

_TIME = re.compile(
    r"(?<![\w:])(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm)?(?![\w:])",
    re.IGNORECASE,
)

_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")

_UNIT_HOURS: dict[str, float] = {
    "h": 1.0,
    "hr": 1.0,
    "hrs": 1.0,
    "hour": 1.0,
    "hours": 1.0,
    "day": 24.0,
    "days": 24.0,
    "week": 168.0,
    "weeks": 168.0,
}

_CURRENCY_WORDS = frozenset(
    {
        "£", "$", "€", "₹", "aed", "dh", "dhs", "dirham", "dirhams", "gbp", "usd",
        "eur", "pound", "pounds", "dollar", "dollars", "euro", "euros",
    }
)

_SUBUNIT_WORDS = frozenset({"pence", "cent", "cents", "p"})

_INTERROGATIVES = frozenset(
    {
        "what", "when", "where", "how", "which", "who", "why", "would", "could",
        "can", "may", "shall", "do", "does", "did", "are", "is", "was", "were",
        "have", "has", "will",
    }
)


def normalise(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace.

    Thousands separators are removed first so ``1,500`` survives as one number
    rather than becoming two.
    """
    without_separators = _THOUSANDS.sub("", text)
    stripped = _PUNCTUATION.sub(" ", without_separators.lower())
    return _WHITESPACE.sub(" ", _LOOSE_PUNCTUATION.sub(" ", stripped)).strip()


def tokens(text: str) -> list[str]:
    return normalise(text).split()


def similarity(left: str, right: str) -> float:
    """Ratio in 0-1 over normalised text. Used only for repetition detection."""
    return SequenceMatcher(None, normalise(left), normalise(right)).ratio()


def is_question(text: str) -> bool:
    """Whether the utterance asks rather than asserts.

    Load-bearing for compliance: "What is your cancellation policy?" mentions the
    cancellation fact but claims nothing, and grading it as a claim would
    manufacture findings out of the agent's own clarifying questions.
    """
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    opening = normalise(stripped).split()[:1]
    return bool(opening) and opening[0] in _INTERROGATIVES


def mentions(text: str, needles: tuple[str, ...]) -> bool:
    """True when any needle appears in the normalised text."""
    haystack = normalise(text)
    return any(needle and normalise(needle) in haystack for needle in needles)


def money_amounts(text: str) -> list[int]:
    """Every monetary amount in the text, as minor units.

    A bare number is included only when it carries a currency marker on either
    side. Treating every integer as money would make "table for 4" a price claim
    and bury the report in noise.
    """
    amounts: list[int] = []
    normalised = _THOUSANDS.sub("", text)
    for match in _MONEY.finditer(normalised):
        symbol = (match.group("symbol") or "").strip().lower()
        suffix = (match.group("suffix") or "").strip().lower()
        if not symbol and not suffix:
            continue
        raw = match.group("amount")
        if suffix in _SUBUNIT_WORDS:
            amounts.append(round(float(raw)))
            continue
        if symbol in _CURRENCY_WORDS or suffix in _CURRENCY_WORDS:
            amounts.append(round(float(raw) * 100))
    return amounts


def durations_hours(text: str) -> list[float]:
    """Durations expressed in hours. ``2 days`` becomes ``48``."""
    found: list[float] = []
    for match in _DURATION.finditer(text):
        unit = match.group("unit").lower()
        multiplier = _UNIT_HOURS.get(unit)
        if multiplier is None:
            continue
        found.append(float(match.group("count")) * multiplier)
    return found


def times_of_day(text: str) -> list[str]:
    """Clock times normalised to 24-hour ``HH:MM``.

    Bare integers are only read as times when a meridiem or an explicit
    ``:minute`` is present, for the same reason bare numbers are not money.
    """
    found: list[str] = []
    for match in _TIME.finditer(text):
        minute_group = match.group("minute")
        meridiem = (match.group("meridiem") or "").lower()
        if minute_group is None and not meridiem:
            continue
        hour = int(match.group("hour"))
        if hour > 23:
            continue
        minute = int(minute_group or 0)
        if minute > 59:
            continue
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        found.append(f"{hour:02d}:{minute:02d}")
    return found


def numbers(text: str) -> list[float]:
    return [float(m.group()) for m in _NUMBER.finditer(_THOUSANDS.sub("", text))]
