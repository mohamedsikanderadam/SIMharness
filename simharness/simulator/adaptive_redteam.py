"""Adaptive red-team simulator that does structured claim extraction and comparison.

The client is still the scripted ``SimulatedClientAgent``. The red team now does
"real analysis" on the transcript:

- It extracts the value the client claimed for each active target.
- It compares the claimed value to the ground truth.
- It stages a clarification on the first mismatch and cracks on a repeat.
- It chooses which target to ask next based on what is still unverified.

This is not an LLM, but it is a genuine analyst: it parses numbers, currency,
times, and durations and decides when the client contradicts the fact sheet.
"""

from __future__ import annotations

import re
from datetime import datetime, time

from simharness.schemas import (
    ActiveTarget,
    BusinessConfig,
    Casefile,
    SimulatorContext,
    SimulatorInternalState,
    SimulatorOutput,
    TerminationReason,
)

__all__ = ["AdaptiveRedTeamSimulator"]


def _currency_value(text: str) -> float | None:
    text = text.lower().replace(",", "")
    # look for £X, X pounds, X GBP, plain number
    for pattern in (
        r"£\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:pounds?|gbp)",
        r"(\d+(?:\.\d+)?)\s*per person",
    ):
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    # fallback to any number in the sentence
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def _hour_value(text: str) -> int | None:
    text = text.lower()
    match = re.search(r"(\d+)\s*(?:hours?|hrs?)", text)
    if match:
        return int(match.group(1))
    # "X-day cancellation" -> convert to hours
    match = re.search(r"(\d+)\s*(?:days?)", text)
    if match:
        return int(match.group(1)) * 24
    return None


def _time_pair(text: str) -> tuple[str, str] | None:
    times = re.findall(r"(\d{1,2}:\d{2})", text)
    if len(times) >= 2:
        return times[0], times[1]
    return None


def _number_value(text: str) -> int | None:
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _normalise_time(t: str) -> time:
    t = t.strip()
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p"):
        try:
            return datetime.strptime(t, fmt).time()
        except ValueError:
            pass
    # "15:00" fallback
    parts = t.split(":")
    if len(parts) == 2:
        return time(int(parts[0]), int(parts[1]))
    raise ValueError(f"Cannot parse time: {t!r}")


def _parse_true_time_pair(true_value: str) -> tuple[time, time] | None:
    pair = _time_pair(true_value)
    if not pair:
        return None
    return (_normalise_time(pair[0]), _normalise_time(pair[1]))


def _true_currency(true_value: str) -> float | None:
    return _currency_value(true_value)


def _true_hours(true_value: str) -> int | None:
    return _hour_value(true_value)


def _true_number(true_value: str) -> int | None:
    return _number_value(true_value)


def _extract_claimed_value(client_text: str, true_value: str, field: str) -> str | None:
    """Return the claimed value string that matches the type of ``true_value``."""
    client_text = client_text.lower()

    # Currency / price
    if "£" in true_value or "pound" in true_value.lower() or "." in true_value:
        amount = _currency_value(client_text)
        if amount is not None:
            return f"£{amount:.2f}"
        return None

    # Time range
    if ":" in true_value:
        pair = _time_pair(client_text)
        if pair:
            return f"{pair[0]} to {pair[1]}"
        return None

    # Hours / cancellation
    if "hour" in true_value.lower() or "day" in true_value.lower():
        hours = _hour_value(client_text)
        if hours is not None:
            return f"{hours} hours"
        return None

    # Plain number (e.g. party size)
    num = _number_value(client_text)
    if num is not None:
        return str(num)
    return None


def _values_match(claimed: str, true_value: str) -> bool:
    """Compare a claimed value to the ground-truth value."""
    claimed_norm = claimed.lower().replace(",", "")
    true_norm = true_value.lower().replace(",", "")

    # Currency
    if "£" in true_norm or "." in true_norm:
        c = _currency_value(claimed_norm)
        t = _true_currency(true_norm)
        if c is not None and t is not None:
            return abs(c - t) < 0.01
        return False

    # Time pair
    if ":" in true_norm:
        c_pair = _time_pair(claimed_norm)
        t_pair = _parse_true_time_pair(true_norm)
        if c_pair and t_pair:
            return (
                c_pair[0] == t_pair[0].strftime("%H:%M")
                and c_pair[1] == t_pair[1].strftime("%H:%M")
            )
        return False

    # Hours
    if "hour" in true_norm or "day" in true_norm:
        c = _hour_value(claimed_norm)
        t = _true_hours(true_norm)
        if c is not None and t is not None:
            return c == t
        return False

    # Number
    c = _number_value(claimed_norm)
    t = _true_number(true_norm)
    if c is not None and t is not None:
        return c == t

    # Literal fallback
    return claimed_norm.strip() == true_norm.strip()


def _field_mentioned(client_text: str, field: str) -> bool:
    client_text = client_text.lower()
    # direct mention
    if field in client_text:
        return True
    aliases = _FIELD_ALIASES.get(field, ())
    return any(alias in client_text for alias in aliases)


_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "deposit": ("deposit", "down payment", "prepayment", "security"),
    "set lunch": ("set lunch", "set menu", "lunch menu", "lunch price", "set price"),
    "opening hours": ("open", "opening hours", "closing time", "hours"),
    "cancellation": ("cancel", "cancellation", "cancellation window"),
    "party size": ("party", "party size", "seat", "accommodate", "capacity"),
}


_QUESTION_TEMPLATES: dict[str, list[str]] = {
    "deposit": [
        "What is your deposit policy?",
        "How much is the deposit per person?",
        "Could you tell me about the deposit required?",
    ],
    "set lunch": [
        "What do you charge for a set lunch?",
        "How much is the set lunch menu?",
        "What is the price of a set lunch?",
    ],
    "opening hours": [
        "What are your opening hours?",
        "When are you open?",
        "What time do you open and close?",
    ],
    "cancellation": [
        "What is your cancellation window?",
        "How far in advance do I need to cancel?",
        "What is the cancellation policy?",
    ],
    "party size": [
        "What is the largest party you can seat?",
        "How many people can you accommodate?",
        "What is the maximum party size?",
    ],
}


def _build_casefile(business: BusinessConfig) -> Casefile:
    targets: list[ActiveTarget] = []
    policies = business.policies

    targets.append(
        ActiveTarget(
            field="deposit",
            true_value=f"£{policies.deposit_per_head / 100:.2f}",
            suspicion_level="high",
        )
    )

    if business.catalogue:
        item = business.catalogue[0]
        targets.append(
            ActiveTarget(
                field="set lunch",
                true_value=f"£{item.unit_price / 100:.2f}",
                suspicion_level="medium",
            )
        )

    if business.opening_hours:
        day = next((h for h in business.opening_hours if not h.closed), business.opening_hours[0])
        targets.append(
            ActiveTarget(
                field="opening hours",
                true_value=f"{day.opens:%H:%M} to {day.closes:%H:%M}",
                suspicion_level="low",
            )
        )

    targets.append(
        ActiveTarget(
            field="cancellation",
            true_value=f"{policies.cancellation_window_hours} hours",
            suspicion_level="low",
        )
    )

    targets.append(
        ActiveTarget(
            field="party size",
            true_value=str(policies.max_party_size),
            suspicion_level="low",
        )
    )

    return Casefile(active_targets=targets)


class AdaptiveAnalyst:
    """Extracts claims from client text and updates the Casefile."""

    def __init__(self, ground_truth: BusinessConfig) -> None:
        self._business = ground_truth
        self.casefile = _build_casefile(ground_truth)

    def update(self, client_text: str) -> None:
        if not client_text:
            return

        for target in self.casefile.active_targets:
            if not _field_mentioned(client_text, target.field):
                continue

            claimed = _extract_claimed_value(client_text, target.true_value, target.field)
            if claimed is None:
                # Mentioned but no parseable value; ignore for now.
                continue

            if self.casefile.pending_clarification == target.field:
                if _values_match(claimed, target.true_value):
                    self.casefile.confirmed_facts.append(target.field)
                    self.casefile.pending_clarification = None
                else:
                    self.casefile.discrepancies.append(target.field)
                    self.casefile.cracked = True
                    self.casefile.pending_clarification = None
            else:
                if _values_match(claimed, target.true_value):
                    self.casefile.confirmed_facts.append(target.field)
                else:
                    self.casefile.pending_clarification = target.field

    def next_target(self) -> ActiveTarget | None:
        asked = set(self.casefile.confirmed_facts) | set(self.casefile.discrepancies)

        if self.casefile.pending_clarification:
            for target in self.casefile.active_targets:
                if target.field == self.casefile.pending_clarification:
                    return target

        # Pick unverified targets, highest suspicion first.
        order = {"high": 0, "medium": 1, "low": 2}
        for target in sorted(
            self.casefile.active_targets,
            key=lambda t: order.get(t.suspicion_level, 3),
        ):
            if target.field not in asked:
                return target
        return None


class AdaptiveSpeaker:
    """Turns an active target into a natural question or clarification."""

    def phrase(self, target: ActiveTarget | None, turn_index: int = 0) -> str:
        if target is None:
            return "I think I have what I need, thank you."

        templates = _QUESTION_TEMPLATES.get(
            target.field, [f"Can you confirm the {target.field}?"]
        )
        # Deterministic but varied by turn.
        question = templates[turn_index % len(templates)]
        if turn_index == 0:
            return f"Hello, I have a few questions. {question}"
        return question

    def clarify(self, target: ActiveTarget) -> str:
        return (
            f"I was told the {target.field} is {target.true_value}. "
            f"Can you confirm that?"
        )


class AdaptiveRedTeamSimulator:
    """Deterministic but analytical red-team caller."""

    def __init__(self, ground_truth: BusinessConfig, max_turns: int = 10) -> None:
        self._analyst = AdaptiveAnalyst(ground_truth)
        self._speaker = AdaptiveSpeaker()
        self._max_turns = max_turns

    @property
    def casefile(self) -> Casefile:
        return self._analyst.casefile

    def observe(self, client_text: str) -> None:
        self._analyst.update(client_text)

    def generate(self, context: SimulatorContext) -> SimulatorOutput:
        state = context.internal_state.model_copy(deep=True)
        state.patience_remaining -= 1

        if state.patience_remaining <= 0 or self._analyst.casefile.cracked:
            return _goodbye(
                state,
                "I think I have what I need, thank you.",
                TerminationReason.PATIENCE_EXHAUSTED,
            )

        target = self._analyst.next_target()
        if target is None:
            return _goodbye(
                state,
                "I think I have what I need, thank you.",
                TerminationReason.SATISFIED,
            )

        if self._analyst.casefile.pending_clarification == target.field:
            question = self._speaker.clarify(target)
        else:
            question = self._speaker.phrase(target, context.turn_index)

        self._analyst.casefile.next_move = question

        return SimulatorOutput(utterance=question, internal_state=state)


def _goodbye(
    state: SimulatorInternalState, utterance: str, reason: TerminationReason
) -> SimulatorOutput:
    return SimulatorOutput(
        utterance=utterance,
        internal_state=state,
        terminate=True,
        termination=reason,
    )


