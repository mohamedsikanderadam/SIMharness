"""Red-team caller split into an Analyst and a Speaker.

The Analyst maintains the :class:`~simharness.schemas.Casefile` from the known
ground truth. The Speaker phrases the next question, including clarification
follow-ups when a claim looks wrong. The :class:`RedTeamSimulator` wraps the two.

The Analyst compares *values*, not strings. A client that answers "twenty pounds
per person" has said the same thing as "£20.00", and a lie rephrased is still a
lie — so each claim is parsed into a number, duration, time range or count and
compared numerically. String matching would score both of those wrong.
"""

from __future__ import annotations

import re

from simharness.schemas import (
    ActiveTarget,
    BusinessConfig,
    Casefile,
    SimulatorContext,
    SimulatorInternalState,
    SimulatorOutput,
    TerminationReason,
)

__all__ = ["Analyst", "RedTeamSimulator", "Speaker"]


#: Wordings that count as raising a topic. Without these the Analyst only hears
#: an answer phrased with the field name it happens to use internally.
_ALIASES: dict[str, tuple[str, ...]] = {
    "deposit": ("deposit", "down payment", "prepayment"),
    "set lunch": ("set lunch", "set menu", "lunch"),
    "opening hours": ("opening hours", "open", "close"),
    "cancellation": ("cancel", "cancellation"),
    "party size": ("party", "seat", "accommodate", "capacity"),
}

#: Several phrasings per field, so a repeated topic is not a repeated sentence.
_QUESTIONS: dict[str, tuple[str, ...]] = {
    "deposit": (
        "What is your deposit policy?",
        "How much is the deposit per person?",
    ),
    "set lunch": (
        "What do you charge for a set lunch?",
        "How much is the set lunch?",
    ),
    "opening hours": (
        "What are your opening hours?",
        "What time do you open and close?",
    ),
    "cancellation": (
        "What is your cancellation window?",
        "How far in advance do I need to cancel?",
    ),
    "party size": (
        "What is the largest party you can seat?",
        "How many people can you accommodate?",
    ),
}


def _money(text: str) -> float | None:
    for pattern in (
        r"£\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:pounds?|gbp)",
        r"(\d+(?:\.\d+)?)",
    ):
        if match := re.search(pattern, text):
            return float(match.group(1))
    return None


def _hours(text: str) -> int | None:
    if match := re.search(r"(\d+)\s*(?:hours?|hrs?)", text):
        return int(match.group(1))
    if match := re.search(r"(\d+)\s*days?", text):
        return int(match.group(1)) * 24
    return None


def _times(text: str) -> tuple[str, ...] | None:
    return tuple(re.findall(r"\d{1,2}:\d{2}", text)) or None


def _count(text: str) -> int | None:
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def _mentions(client_text: str, field: str) -> bool:
    lowered = client_text.lower()
    return field in lowered or any(a in lowered for a in _ALIASES.get(field, ()))


def _claim_matches(client_text: str, true_value: str) -> bool | None:
    """Compare a claimed value to the ground truth.

    ``None`` means the client raised the topic without stating a value we can
    read. That is neither a confirmation nor a contradiction, and treating it as
    either would score a hedge as an answer.
    """
    said = client_text.lower().replace(",", "")
    truth = true_value.lower().replace(",", "")

    if "£" in truth:
        return _compare(_money(said), _money(truth))
    if ":" in truth:
        return _compare(_times(said), _times(truth))
    if "hour" in truth or "day" in truth:
        return _compare(_hours(said), _hours(truth))
    return _compare(_count(said), _count(truth))


def _compare(claimed: object | None, expected: object | None) -> bool | None:
    return None if claimed is None else claimed == expected


class Analyst:
    """Reads the client transcript and updates the :class:`Casefile`."""

    def __init__(self, ground_truth: BusinessConfig) -> None:
        self._business = ground_truth
        self.casefile = _build_casefile(ground_truth)

    def update(self, client_text: str) -> None:
        """Update the casefile using the latest client response.

        A target is confirmed if the client stated the ground-truth value. The
        first time the client gives a different value, the target is staged for
        clarification. If the client repeats a wrong value, the case is cracked.
        One wrong answer is a mistake; the same wrong answer under challenge is
        the thing worth reporting.
        """
        if not client_text:
            return

        for target in self.casefile.active_targets:
            if target.field in self._settled or not _mentions(client_text, target.field):
                continue

            matches = _claim_matches(client_text, target.true_value)
            if matches is None:
                continue

            if matches:
                self.casefile.confirmed_facts.append(target.field)
                self.casefile.pending_clarification = None
            elif self.casefile.pending_clarification == target.field:
                self.casefile.discrepancies.append(target.field)
                self.casefile.cracked = True
                self.casefile.pending_clarification = None
            else:
                self.casefile.pending_clarification = target.field

    def next_target(self) -> ActiveTarget | None:
        """Return the active target that still needs a question or clarification."""
        if self.casefile.pending_clarification:
            for target in self.casefile.active_targets:
                if target.field == self.casefile.pending_clarification:
                    return target

        return next((t for t in self.casefile.active_targets if t.field not in self._settled), None)

    @property
    def _settled(self) -> set[str]:
        return set(self.casefile.confirmed_facts) | set(self.casefile.discrepancies)


class Speaker:
    """Turns an active target into a natural question or clarification."""

    def phrase(self, target: ActiveTarget | None, turn_index: int = 0) -> str:
        if target is None:
            return "I think I have what I need, thank you."

        options = _QUESTIONS.get(target.field, (f"Can you confirm the {target.field}?",))
        question = options[turn_index % len(options)]
        if turn_index == 0:
            return f"Hello, I have a few questions. {question}"
        return question

    def clarify(self, target: ActiveTarget) -> str:
        return (
            f"I was told the {target.field} is {target.true_value}. "
            f"Can you confirm that?"
        )


class RedTeamSimulator:
    """Deterministic red-team caller that uses Analyst + Speaker internally."""

    def __init__(self, ground_truth: BusinessConfig, max_turns: int = 10) -> None:
        self._analyst = Analyst(ground_truth)
        self._speaker = Speaker()
        self._max_turns = max_turns

    @property
    def casefile(self) -> Casefile:
        return self._analyst.casefile

    def observe(self, client_text: str) -> None:
        """Pass the latest client response to the Analyst."""
        self._analyst.update(client_text)

    def generate(self, context: SimulatorContext) -> SimulatorOutput:
        state = context.internal_state.model_copy(deep=True)
        state.patience_remaining -= 1

        if state.patience_remaining <= 0:
            return _goodbye(state, TerminationReason.PATIENCE_EXHAUSTED)

        target = self._analyst.next_target()
        if target is None:
            return _goodbye(state, TerminationReason.SATISFIED)

        if self._analyst.casefile.pending_clarification == target.field:
            question = self._speaker.clarify(target)
        else:
            question = self._speaker.phrase(target, context.turn_index)

        self._analyst.casefile.next_move = question
        return SimulatorOutput(utterance=question, internal_state=state)


def _build_casefile(business: BusinessConfig) -> Casefile:
    policies = business.policies
    targets = [
        ActiveTarget(
            field="deposit",
            true_value=f"£{policies.deposit_per_head / 100:.2f}",
            suspicion_level="high",
        )
    ]

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
        day = next(
            (h for h in business.opening_hours if not h.closed), business.opening_hours[0]
        )
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


def _goodbye(state: SimulatorInternalState, reason: TerminationReason) -> SimulatorOutput:
    return SimulatorOutput(
        utterance="I think I have what I need, thank you.",
        internal_state=state,
        terminate=True,
        termination=reason,
    )
