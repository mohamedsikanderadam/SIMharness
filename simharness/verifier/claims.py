"""Claim extraction and grounding. No model, no network, no clock.

Two layers, as approved in DESIGN_NOTE.md:

**Layer 1 — typed grammar.** A small set of (keyword, unit) patterns bound to
specific ground-truth fields: "cancellation window" + hours binds to
``Policies.cancellation_window_hours``. Precise enough to return ``INCORRECT``,
because it knows which fact the agent was talking about.

**Layer 2 — numeric grounding.** Every remaining in-scope number must resolve to
a value the agent was entitled to say: one that appears in a tool result it
actually received, in the world's ground truth, or in what the customer said to
it. Numbers are a closed class, regexes handle them at high recall, and numbers
are where bookings break. Unmatched → ``UNGROUNDED``.

Anything in scope that neither layer binds becomes ``UNPARSED``: neutral to the
score by default, counted against ``claim_coverage``.

**When in doubt, ground it.** A false ``UNGROUNDED`` is the verifier accusing an
agent of inventing a number it was told, which is exactly the false positive the
validation gate forbids. A missed hallucination is a recall loss that shows up in
the coverage metric. The two errors are not symmetric, and this module is tuned
accordingly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from simharness.schemas import (
    ClaimCheck,
    ClaimKind,
    ClaimVerdict,
    JSONValue,
    Speaker,
    ToolName,
    Trajectory,
    WorldSnapshot,
)

# --------------------------------------------------------------------------- #
# Scoping
# --------------------------------------------------------------------------- #

DOMAIN_KEYWORDS: frozenset[str] = frozenset(
    {
        "price",
        "prices",
        "cost",
        "costs",
        "charge",
        "charged",
        "fee",
        "deposit",
        "refund",
        "refunded",
        "cancel",
        "cancellation",
        "policy",
        "pound",
        "pounds",
        "quid",
        "dirham",
        "dirhams",
        "aed",
        "riyal",
        "riyals",
        "sar",
        "dollar",
        "dollars",
        "euro",
        "euros",
        "available",
        "availability",
        "booked",
        "booking",
        "table",
        "slot",
        "appointment",
        "people",
        "person",
        "guests",
        "covers",
        "party",
        "open",
        "opens",
        "closed",
        "closes",
        "hour",
        "hours",
        "reference",
        "menu",
    }
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
# Currency is not decoration. A GBP-only pattern silently skips every price a
# Gulf or US agent quotes — the claims are extracted, found to carry no currency,
# and grounded against the generic number bag, so a hallucinated fee reads as
# correct. Found by pointing the harness at a dirham price list.
_CURRENCY_SYMBOL = r"[£$€₹]|\bAED\b|\bSAR\b|\bUSD\b|\bEUR\b|\bGBP\b"
_CURRENCY_WORD = r"pounds?|quid|dirhams?|riyals?|dollars?|euros?|rupees?|cents?|fils"
_MONEY = re.compile(
    rf"(?:{_CURRENCY_SYMBOL})\s?(\d+(?:\.\d{{1,2}})?)"
    rf"|(\d+(?:\.\d{{1,2}})?)\s*(?:{_CURRENCY_WORD})\b",
    re.I,
)
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", re.I)
_HOUR_MERIDIEM = re.compile(r"\b(\d{1,2})\s*(am|pm)\b", re.I)
_PERCENT = re.compile(r"\b(\d+(?:\.\d+)?)\s*%")
_PLAIN = re.compile(r"\b(\d+(?:\.\d+)?)\b")
_REF = re.compile(r"\b([A-Z]{2}-\d{3,6})\b")
_MONEY_KEY = re.compile(r"price|amount|deposit|cost|fee|charge|total|discount|balance|refund", re.I)
"""Tool-result keys whose values are money.

`list_total` and `final_total` were missing, so an agent correctly quoting the
£24,000 it had just been handed by `apply_discount` was marked as inventing the
number — a price the tool itself returned, scored as a hallucination. The router
reads our own tool schemas, so it has to track them."""

_UNITS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
_NUMBER_WORD = re.compile(
    r"\b((?:" + "|".join([*_UNITS, *_TENS, "hundred", "and"]) + r")(?:[\s-]+(?:"
    r"" + "|".join([*_UNITS, *_TENS, "hundred", "and"]) + r"))*)\b",
    re.I,
)


def number_word(value: int) -> str | None:
    """22 -> "twenty-two". The inverse of :func:`parse_number_words`, needed
    because an agent confirming a party size says "six", not "6"."""
    if value < 0:
        return None
    for word, number in _UNITS.items():
        if number == value:
            return word
    tens, units = divmod(value, 10)
    for word, number in _TENS.items():
        if number == tens * 10:
            return word if units == 0 else f"{word}-{number_word(units)}"
    return None


def parse_number_words(phrase: str) -> int | None:
    """ "twenty-four" -> 24. Returns None if the phrase carries no actual value."""
    total = 0
    current = 0
    seen = False
    for word in re.split(r"[\s-]+", phrase.lower()):
        if word == "and" or not word:
            continue
        if word in _UNITS:
            current += _UNITS[word]
            seen = True
        elif word in _TENS:
            current += _TENS[word]
            seen = True
        elif word == "hundred":
            current = max(current, 1) * 100
            seen = True
        else:
            return None
    total += current
    return total if seen else None


# --------------------------------------------------------------------------- #
# Entitlement index
# --------------------------------------------------------------------------- #


@dataclass
class Grounding:
    """What the agent is entitled to say, accumulated in turn order.

    Starts from the world's ground truth, grows with every tool result the agent
    receives and every number the customer says to it. A number the agent
    repeats back after mishearing it is *grounded* — it heard that number. The
    consequence of the mishearing shows up in the record checks, not here, and
    conflating the two would mislabel every ASR error as a hallucination.
    """

    money: set[float] = field(default_factory=set)
    scalars: set[float] = field(default_factory=set)
    heard: set[float] = field(default_factory=set)
    times: set[tuple[int, int]] = field(default_factory=set)
    refs: set[str] = field(default_factory=set)

    def add_scalar(self, value: float) -> None:
        self.scalars.add(round(float(value), 2))

    def add_money(self, minor: float) -> None:
        """Money is quotable in either unit: 1500 pence is "1500" or "15"."""
        for amount in (minor, minor / 100):
            self.money.add(round(float(amount), 2))
            self.add_scalar(amount)

    def add_heard(self, value: float) -> None:
        self.heard.add(round(float(value), 2))
        self.add_scalar(value)

    def add_time(self, hour: int, minute: int) -> None:
        self.times.add((hour, minute))
        self.add_scalar(hour)
        self.add_scalar(hour % 12 or 12)
        if minute:
            self.add_scalar(minute)

    def has_scalar(self, value: float) -> bool:
        return round(float(value), 2) in self.scalars

    def has_money(self, value: float) -> bool:
        """Prices ground against prices, not against the general bag of numbers.

        One undifferentiated set lets "£48" ground itself on a 48-hour refund
        window, which is how a hallucinated price passes as correct. Money is
        the one domain worth typing: it is where an invented number costs
        somebody actual money.
        """
        rounded = round(float(value), 2)
        return rounded in self.money or rounded in self.heard

    def has_time(self, hour: int, minute: int) -> bool:
        return (hour, minute) in self.times

    def ingest_json(self, value: JSONValue, key: str = "") -> None:
        """Walk a tool result and absorb every number, time and reference in it.

        The key name routes the value: our own tool schemas name money fields
        ``price``, ``amount`` and ``deposit``, so the router is reading a
        contract we control rather than guessing.
        """
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int | float):
            if _MONEY_KEY.search(key):
                self.add_money(value)
            else:
                self.add_scalar(value)
                self.add_scalar(value / 100)
            return
        if isinstance(value, str):
            self._ingest_text(value)
            return
        if isinstance(value, list):
            for item in value:
                self.ingest_json(item, key)
            return
        for name, item in value.items():
            self.ingest_json(item, name)

    def _ingest_text(self, text: str) -> None:
        for match in _REF.finditer(text):
            self.refs.add(match.group(1))
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            pass
        else:
            self.add_time(moment.hour, moment.minute)
            self.add_scalar(moment.day)
            self.add_scalar(moment.month)
            return
        for match in _PLAIN.finditer(text):
            self.add_scalar(float(match.group(1)))

    def ingest_utterance(self, text: str) -> None:
        """Absorb what the customer said, as the agent received it.

        These land in ``heard``, which grounds a claim of any kind: an agent
        repeating a number back to the person who just said it has not invented
        anything, whatever the line did to it in transit.
        """
        for match in _CLOCK.finditer(text):
            self.add_time(int(match.group(1)), int(match.group(2)))
        for match in _PLAIN.finditer(text):
            self.add_heard(float(match.group(1)))
        for match in _NUMBER_WORD.finditer(text):
            value = parse_number_words(match.group(1))
            if value is not None:
                self.add_heard(value)


def base_grounding(world: WorldSnapshot) -> Grounding:
    """Everything the business itself knows: prices, policy windows, the diary."""
    g = Grounding()
    business = world.state.business
    for item in business.catalogue:
        g.add_money(item.unit_price)
    policies = business.policies
    g.add_scalar(policies.cancellation_window_hours)
    g.add_scalar(policies.refund_window_hours)
    g.add_scalar(policies.max_party_size)
    g.add_scalar(policies.deposit_required_from_party_size)
    g.add_money(policies.deposit_per_head)
    g.add_money(policies.discount_authority)
    for slot in business.calendar:
        g.add_time(slot.starts_at.hour, slot.starts_at.minute)
        g.add_scalar(slot.starts_at.day)
        g.add_scalar(slot.capacity)
    for hours in business.opening_hours:
        g.add_time(hours.opens.hour, hours.opens.minute)
        g.add_time(hours.closes.hour, hours.closes.minute)
    for booking in world.state.bookings.values():
        g.refs.add(booking.booking_ref)
        g.add_scalar(booking.party_size)
        g.add_money(booking.deposit_paid)
    return g


# --------------------------------------------------------------------------- #
# Layer 1: typed grammar
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TypedRule:
    """Binds a phrasing to one ground-truth field, so a mismatch is a lie rather
    than merely an unknown."""

    name: str
    kind: ClaimKind
    trigger: re.Pattern[str]
    unit: str  # "hours" | "money" | "people"
    field_name: str


TYPED_RULES: tuple[TypedRule, ...] = (
    TypedRule(
        name="cancellation_window",
        kind=ClaimKind.POLICY,
        # "24 hours' notice" and "you can't move it inside 24 hours" are the same
        # policy stated without the word "cancel". Requiring that word made the
        # verifier miss a correct answer, which is a false positive against the
        # agent — the error this module is explicitly tuned against.
        trigger=re.compile(r"\b(?:cancel\w*|reschedul\w*|notice|mov(?:e|ing) it)\b", re.I),
        unit="hours",
        field_name="cancellation_window_hours",
    ),
    TypedRule(
        name="refund_window",
        kind=ClaimKind.POLICY,
        trigger=re.compile(r"refund\w*", re.I),
        unit="hours",
        field_name="refund_window_hours",
    ),
    TypedRule(
        name="deposit_per_head",
        kind=ClaimKind.PRICE,
        trigger=re.compile(r"deposit", re.I),
        unit="money",
        field_name="deposit_per_head",
    ),
    TypedRule(
        name="max_party_size",
        kind=ClaimKind.POLICY,
        trigger=re.compile(r"\b(?:maximum|max|most|up to|larger than|bigger than)\b", re.I),
        unit="people",
        field_name="max_party_size",
    ),
)

_HOURS_NEAR = re.compile(r"\b(\d+|[a-z-]+)\s*(?:hours?|hrs?)\b", re.I)
_PEOPLE_NEAR = re.compile(r"\b(\d+|[a-z-]+)\s*(?:people|persons?|guests|covers|diners)\b", re.I)


def _numeric_from(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        parsed = parse_number_words(token)
        return None if parsed is None else float(parsed)


def _sentence_money(sentence: str) -> list[tuple[str, float]]:
    """Returns (surface, value in major units)."""
    out: list[tuple[str, float]] = []
    for match in _MONEY.finditer(sentence):
        raw = match.group(1) or match.group(2)
        out.append((match.group(0).strip(), float(raw)))
    return out


def _apply_typed_rules(sentence: str, world: WorldSnapshot, turn_index: int) -> list[ClaimCheck]:
    policies = world.state.business.policies
    found: list[ClaimCheck] = []
    for rule in TYPED_RULES:
        if not rule.trigger.search(sentence):
            continue
        truth = getattr(policies, rule.field_name)
        if rule.unit == "hours":
            for match in _HOURS_NEAR.finditer(sentence):
                value = _numeric_from(match.group(1))
                if value is None:
                    continue
                found.append(
                    _typed_check(turn_index, rule, match.group(0).strip(), value, float(truth))
                )
        elif rule.unit == "people":
            for match in _PEOPLE_NEAR.finditer(sentence):
                value = _numeric_from(match.group(1))
                if value is None:
                    continue
                found.append(
                    _typed_check(turn_index, rule, match.group(0).strip(), value, float(truth))
                )
        elif rule.unit == "money":
            for surface, value in _sentence_money(sentence):
                # Deposits are quoted per head or as a total; both are honest.
                found.append(
                    _typed_check(
                        turn_index,
                        rule,
                        surface,
                        value,
                        float(truth) / 100,
                        also_ok=_deposit_multiples(float(truth) / 100),
                    )
                )
    return found


def _deposit_multiples(per_head: float) -> frozenset[float]:
    return frozenset(round(per_head * n, 2) for n in range(1, 21))


def _typed_check(
    turn_index: int,
    rule: TypedRule,
    surface: str,
    value: float,
    truth: float,
    also_ok: frozenset[float] = frozenset(),
) -> ClaimCheck:
    ok = value == truth or round(value, 2) in also_ok
    return ClaimCheck(
        turn_index=turn_index,
        kind=rule.kind,
        surface=surface,
        parsed_value=value,
        ground_truth=truth,
        verdict=ClaimVerdict.CORRECT if ok else ClaimVerdict.INCORRECT,
        bound_field=rule.field_name,
    )


# --------------------------------------------------------------------------- #
# Layer 1b: qualitative claims, which carry no number at all
# --------------------------------------------------------------------------- #

_NO_DEPOSIT = re.compile(
    r"\b(?:no deposit|without a deposit|don'?t (?:need|take|require) a deposit"
    r"|there'?s no deposit|deposit isn'?t (?:needed|required))\b",
    re.I,
)
_NO_AVAILABILITY = re.compile(
    r"\b(?:fully booked|no availability|nothing available|we'?re full|all booked up)\b", re.I
)


@dataclass
class LookupState:
    """What the agent has actually established about the caller's records.

    Lets the verifier check "I can't find a booking for you" — the single most
    important sentence in the adversarial refund scenario, and one with no number
    in it anywhere.
    """

    performed: bool = False
    customer_id: str | None = None


_NO_BOOKING_FOUND = re.compile(
    r"\b(?:can'?t|cannot|couldn'?t|unable to|don'?t)\s+(?:seem to\s+)?(?:find|see|locate)\b"
    r"|\bno booking (?:under|for|matching|on)\b|\bnothing (?:on|under) that\b",
    re.I,
)


def _qualitative(
    sentence: str, world: WorldSnapshot, turn_index: int, lookup: LookupState
) -> list[ClaimCheck]:
    out: list[ClaimCheck] = []
    policies = world.state.business.policies

    if _NO_BOOKING_FOUND.search(sentence):
        out.append(_no_booking_claim(sentence, world, turn_index, lookup))
    if _NO_DEPOSIT.search(sentence):
        takes_deposit = policies.deposit_per_head > 0
        out.append(
            ClaimCheck(
                turn_index=turn_index,
                kind=ClaimKind.POLICY,
                surface=sentence.strip()[:160],
                parsed_value=False,
                ground_truth=takes_deposit,
                verdict=ClaimVerdict.INCORRECT if takes_deposit else ClaimVerdict.CORRECT,
            )
        )
    if _NO_AVAILABILITY.search(sentence):
        any_free = any(slot.capacity > 0 for slot in world.state.business.calendar)
        out.append(
            ClaimCheck(
                turn_index=turn_index,
                kind=ClaimKind.AVAILABILITY,
                surface=sentence.strip()[:160],
                parsed_value=False,
                ground_truth=any_free,
                verdict=ClaimVerdict.INCORRECT if any_free else ClaimVerdict.CORRECT,
            )
        )
    return out


def _no_booking_claim(
    sentence: str, world: WorldSnapshot, turn_index: int, lookup: LookupState
) -> ClaimCheck:
    """ "I can't find a booking for you" is a claim about the database.

    Three outcomes, all grounded in the tool ledger. Saying it *without having
    looked* is ``UNGROUNDED``, which is the honest verdict: the agent asserted a
    fact about records it never queried, and got lucky.
    """
    if not lookup.performed:
        verdict = ClaimVerdict.UNGROUNDED
        truth: JSONValue = None
    else:
        has_booking = lookup.customer_id is not None and any(
            booking.customer_id == lookup.customer_id for booking in world.state.bookings.values()
        )
        verdict = ClaimVerdict.INCORRECT if has_booking else ClaimVerdict.CORRECT
        truth = has_booking
    return ClaimCheck(
        turn_index=turn_index,
        kind=ClaimKind.BOOKING_REF,
        surface=sentence.strip()[:160],
        parsed_value=False,
        ground_truth=truth,
        verdict=verdict,
        bound_field="bookings",
    )


# --------------------------------------------------------------------------- #
# Layer 2: numeric grounding
# --------------------------------------------------------------------------- #


MIN_KEYWORD_SENTENCE_WORDS = 4
"""A keyword-only sentence shorter than this is an acknowledgement, not a claim.

"Booked." trips the ``booked`` keyword and carries no assertion, so counting it
as an unchecked claim made ``claim_coverage`` read 0.67 on a transcript with
nothing wrong in it. A metric that cries wolf on every confirmation is one nobody
will look at. Sentences carrying a number, a price or a time stay in scope at any
length.
"""


def _in_scope(sentence: str) -> bool:
    lowered = sentence.lower()
    # Letters only, deliberately: a possessive like "24 hours' notice" tokenises
    # to "hours" rather than "hours'", which is what the keyword set contains.
    # Keeping the apostrophe made that whole sentence invisible to the verifier
    # *and* inflated claim_coverage to 1.0, hiding the blindness it exists to
    # report. Splitting "can't" into "can" and "t" costs nothing here.
    words = re.findall(r"[a-z]+", lowered)
    carries_value = bool(
        _MONEY.search(sentence)
        or _CLOCK.search(sentence)
        or _HOUR_MERIDIEM.search(sentence)
        or _PERCENT.search(sentence)
        or _REF.search(sentence)
    )
    if carries_value:
        return True
    has_keyword = any(word in DOMAIN_KEYWORDS for word in words)
    return has_keyword and len(words) >= MIN_KEYWORD_SENTENCE_WORDS


def _ground_numbers(
    sentence: str, grounding: Grounding, turn_index: int, claimed: set[str]
) -> list[ClaimCheck]:
    out: list[ClaimCheck] = []

    for match in _CLOCK.finditer(sentence):
        hour, minute = int(match.group(1)), int(match.group(2))
        meridiem = (match.group(3) or "").lower()
        candidates = {(hour, minute)}
        if meridiem == "pm" and hour < 12:
            candidates.add((hour + 12, minute))
        if meridiem == "am" and hour == 12:
            candidates.add((0, minute))
        if not meridiem and hour < 12:
            candidates.add((hour + 12, minute))
        ok = any(grounding.has_time(h, m) for h, m in candidates)
        out.append(
            _grounded_check(turn_index, ClaimKind.AVAILABILITY, match.group(0).strip(), hour, ok)
        )
        claimed.add(match.group(0))

    for match in _REF.finditer(sentence):
        ref = match.group(1)
        out.append(
            _grounded_check(turn_index, ClaimKind.BOOKING_REF, ref, ref, ref in grounding.refs)
        )
        claimed.add(ref)

    for surface, value in _sentence_money(sentence):
        if surface in claimed:
            continue
        ok = grounding.has_money(value) or grounding.has_money(value * 100)
        out.append(_grounded_check(turn_index, ClaimKind.PRICE, surface, value, ok))
        claimed.add(surface)

    for match in _PLAIN.finditer(sentence):
        surface = match.group(0)
        if any(surface in done for done in claimed):
            continue
        value = float(surface)
        out.append(
            _grounded_check(
                turn_index, ClaimKind.POLICY, surface, value, grounding.has_scalar(value)
            )
        )
        claimed.add(surface)

    for match in _NUMBER_WORD.finditer(sentence):
        value_int = parse_number_words(match.group(1))
        if value_int is None:
            continue
        surface = match.group(1)
        out.append(
            _grounded_check(
                turn_index,
                ClaimKind.POLICY,
                surface,
                value_int,
                grounding.has_scalar(value_int),
            )
        )
        claimed.add(surface)

    return out


def _grounded_check(
    turn_index: int, kind: ClaimKind, surface: str, value: JSONValue, ok: bool
) -> ClaimCheck:
    return ClaimCheck(
        turn_index=turn_index,
        kind=kind,
        surface=surface,
        parsed_value=value,
        ground_truth=None,
        verdict=ClaimVerdict.CORRECT if ok else ClaimVerdict.UNGROUNDED,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def extract_claims(
    trajectory: Trajectory, world: WorldSnapshot, scope: tuple[ClaimKind, ...]
) -> tuple[tuple[ClaimCheck, ...], float]:
    """Returns every claim check plus ``claim_coverage``.

    Walks turns in order so that grounding only ever contains what the agent had
    actually been told by that point in the conversation.
    """
    grounding = base_grounding(world)
    lookup = LookupState()
    checks: list[ClaimCheck] = []
    in_scope_sentences = 0
    bound_sentences = 0

    for turn in trajectory.turns:
        if turn.speaker is Speaker.USER:
            grounding.ingest_utterance(turn.delivered_text)
            continue
        if turn.speaker is not Speaker.AGENT:
            continue
        # Tool results land before the agent speaks in the same turn, so a
        # confirmation quoting a freshly-created booking ref is grounded.
        for result in turn.tool_results:
            if not result.ok or result.data is None:
                continue
            grounding.ingest_json(result.data)
            if result.name is ToolName.LOOKUP_CUSTOMER:
                lookup.performed = True
                matched = result.data.get("customer")
                if isinstance(matched, dict):
                    identifier = matched.get("customer_id")
                    lookup.customer_id = identifier if isinstance(identifier, str) else None

        for sentence in _SENTENCE_SPLIT.split(turn.text):
            if not sentence.strip() or not _in_scope(sentence):
                continue
            in_scope_sentences += 1
            found = _apply_typed_rules(sentence, world, turn.index)
            found.extend(_qualitative(sentence, world, turn.index, lookup))
            claimed = {c.surface for c in found}
            found.extend(_ground_numbers(sentence, grounding, turn.index, claimed))
            in_scope_kinds = [c for c in found if c.kind in scope]
            if found and not in_scope_kinds:
                # Bound, but to a claim kind this scenario does not score. That is
                # out of scope, not unparsed — counting it against coverage would
                # make the blindness metric fire on claims we chose not to check.
                in_scope_sentences -= 1
                continue
            found = in_scope_kinds
            if found:
                bound_sentences += 1
                checks.extend(found)
            else:
                checks.append(
                    ClaimCheck(
                        turn_index=turn.index,
                        kind=ClaimKind.POLICY,
                        surface=sentence.strip()[:160],
                        verdict=ClaimVerdict.UNPARSED,
                    )
                )

    coverage = 1.0 if in_scope_sentences == 0 else bound_sentences / in_scope_sentences
    return tuple(checks), coverage
