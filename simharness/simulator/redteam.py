"""Red-team caller split into an Analyst and a Speaker.

The Analyst maintains the :class:`~simharness.schemas.Casefile` from the known
ground truth. The Speaker phrases the next question. The :class:`RedTeamSimulator`
wraps the two so the package's ``Simulator`` protocol still has one entry point.
"""

from __future__ import annotations

from simharness.schemas import (
    ActiveTarget,
    BusinessConfig,
    Casefile,
    SimulatorContext,
    SimulatorInternalState,
    SimulatorOutput,
    Speaker as SpeakerEnum,
    TerminationReason,
)

__all__ = ["Analyst", "RedTeamSimulator", "Speaker"]


class Analyst:
    """Reads the client transcript and updates the :class:`Casefile`."""

    def __init__(self, ground_truth: BusinessConfig) -> None:
        self._business = ground_truth
        self.casefile = _build_casefile(ground_truth)

    def update(self, client_text: str) -> None:
        """Update the casefile using the latest client response.

        A target is confirmed if the client said the ground-truth value. It is a
        discrepancy if the client mentioned the topic but did not say the
        ground-truth value. The latter is what ``cracked`` means for the red team.
        """
        if not client_text:
            return

        text = client_text.lower()
        asked = set(self.casefile.confirmed_facts) | set(self.casefile.discrepancies)

        for target in self.casefile.active_targets:
            if target.field in asked:
                continue
            if target.field not in text:
                continue

            if target.true_value.lower() in text:
                self.casefile.confirmed_facts.append(target.field)
            else:
                self.casefile.discrepancies.append(target.field)
                self.casefile.cracked = True

    def next_target(self) -> ActiveTarget | None:
        """Return the first active target not yet settled."""
        asked = set(self.casefile.confirmed_facts) | set(self.casefile.discrepancies)
        for target in self.casefile.active_targets:
            if target.field not in asked:
                return target
        return None


class Speaker:
    """Turns an active target into a natural question."""

    def phrase(self, target: ActiveTarget | None) -> str:
        if target is None:
            return "I think I have what I need, thank you."
        return _question_for(target)


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
            return _goodbye(
                state,
                "I think I have what I need, thank you.",
                TerminationReason.PATIENCE_EXHAUSTED,
            )

        client_text = _last_agent_text(context.history)
        self._analyst.update(client_text)

        target = self._analyst.next_target()
        if target is None:
            return _goodbye(
                state,
                "I think I have what I need, thank you.",
                TerminationReason.SATISFIED,
            )

        question = self._speaker.phrase(target)
        self._analyst.casefile.next_move = question

        if context.turn_index == 0:
            question = f"Hello, I have a few questions. {question}"

        return SimulatorOutput(utterance=question, internal_state=state)


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


def _question_for(target: ActiveTarget) -> str:
    mapping = {
        "deposit": "What is your deposit policy?",
        "set lunch": "What do you charge for a set lunch?",
        "opening hours": "What are your opening hours?",
        "cancellation": "What is your cancellation window?",
        "party size": "What is the largest party you can seat?",
    }
    return mapping.get(target.field, f"Can you confirm the {target.field}?")


def _goodbye(
    state: SimulatorInternalState, utterance: str, reason: TerminationReason
) -> SimulatorOutput:
    return SimulatorOutput(
        utterance=utterance,
        internal_state=state,
        terminate=True,
        termination=reason,
    )


def _last_agent_text(history: tuple) -> str:
    for view in reversed(history):
        if view.speaker is SpeakerEnum.AGENT:
            return view.text
    return ""
