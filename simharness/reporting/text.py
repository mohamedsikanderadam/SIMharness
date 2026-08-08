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
    "digits_for_number_words",
    "durations_hours",
    "is_question",
    "mentions",
    "money_amounts",
    "normalise",
    "numbers",
    "redact_card_numbers",
    "similarity",
    "times_of_day",
    "tokens",
]

# --------------------------------------------------------------------------- #
# Spoken numbers
# --------------------------------------------------------------------------- #

_SMALL: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES: dict[str, int] = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_NUMBER_WORDS = frozenset(_SMALL) | frozenset(_SCALES) | {"and", "a"}


def digits_for_number_words(text: str) -> str:
    """Rewrite spoken numbers as digits: ``three hundred`` -> ``300``.

    Speech recognition returns what was said, and people say prices out loud.
    Scribe transcribed a hallucinated rate as "three hundred dirhams", which the
    money parser could not see at all — so the audit passed a call that had
    misquoted the price. Everything downstream reads digits, so the cheapest
    correct fix is to make the digits exist.

    ``and`` and ``a`` are consumed only *inside* a run of number words, so
    "a deluxe room" is untouched while "a hundred and fifty" is not.

    A run that does not fold to exactly one figure is left in words. "Two fifty"
    means 250 to a caller and "eight fifteen" means a time, but neither is
    recoverable from the words alone — and a guess here is not a missed finding,
    it is a *fabricated* one, because the invented figure would be compared
    against the fact sheet and contradict it. Leaving those unparsed restores
    them to what they were before this function existed: unadjudicable.
    """
    words = text.split()
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        tail: list[str] = []
        while run and run[-1].strip(",.!?;:").lower() in {"and", "a"}:
            tail.insert(0, run.pop())
        if run:
            value = _value_of(run)
            if value is None:
                out.extend(run)
            else:
                trailing = run[-1][len(run[-1].rstrip(",.!?;:")) :]
                out.append(f"{value}{trailing}")
            run.clear()
        out.extend(tail)

    for word in words:
        bare = word.strip(",.!?;:").lower()
        if bare in _NUMBER_WORDS and not (bare in {"and", "a"} and not run):
            run.append(word)
            continue
        flush()
        out.append(word)
    flush()
    return " ".join(out)


def _value_of(run: list[str]) -> int | None:
    """Fold a run of number words into one integer.

    ``None`` when the run carries no digit-bearing word at all (so a stray "and"
    never becomes ``0``) and, deliberately, when it holds more than one figure.
    """
    figures = _figures(run)
    return figures[0] if len(figures) == 1 else None


def _kind(word: str) -> str:
    if word in _SCALES:
        return "scale"
    value = _SMALL[word]
    if value >= 20:
        return "tens"
    return "teens" if value >= 10 else "unit"


def _continues(previous: str, kind: str) -> bool:
    """Whether ``kind`` extends the figure in progress rather than starting one.

    English composes a number in only a few ways: anything may be multiplied by a
    scale ("three *hundred*"), a scale may be followed by a remainder ("three
    hundred *fifty*"), and a tens word may take a unit ("twenty *four*"). Every
    other adjacency is two numbers in a row - "two fifty", "eight fifteen" - and
    is reported as such rather than added together.
    """
    if kind == "scale" or previous == "scale":
        return True
    return previous == "tens" and kind == "unit"


def _figures(run: list[str]) -> list[int]:
    """Every distinct number in a run of number words."""
    figures: list[int] = []
    total = current = 0
    previous = ""

    def close() -> None:
        nonlocal total, current, previous
        if previous:
            figures.append(total + current)
        total = current = 0
        previous = ""

    for word in run:
        bare = word.strip(",.!?;:").lower()
        if bare in {"and", "a"}:
            continue
        kind = _kind(bare)
        if previous and not _continues(previous, kind):
            close()
        if kind == "scale":
            scale = _SCALES[bare]
            current = (current or 1) * scale
            if scale >= 1000:
                total += current
                current = 0
        else:
            current += _SMALL[bare]
        previous = kind

    close()
    return figures


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
    normalised = _THOUSANDS.sub("", digits_for_number_words(text))
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
    for match in _DURATION.finditer(digits_for_number_words(text)):
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
    for match in _TIME.finditer(digits_for_number_words(text)):
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
    cleaned = _THOUSANDS.sub("", digits_for_number_words(text))
    return [float(m.group()) for m in _NUMBER.finditer(cleaned)]


_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def redact_card_numbers(text: str) -> str:
    """Mask all but the first and last four digits of anything card-shaped.

    The audit reports an agent for reading a card number aloud, so the report
    itself must not become the second place that number is written down. Enough
    digits are kept for the business to match the finding to the call.
    """

    def mask(match: re.Match[str]) -> str:
        digits = [c for c in match.group() if c.isdigit()]
        if len(digits) < 13:
            return match.group()
        hidden = "*" * (len(digits) - 8)
        return f"{''.join(digits[:4])} {hidden} {''.join(digits[-4:])}"

    return _CARD.sub(mask, text)
