"""A scripted red-team caller: a Speaker and an Analyst in one shell.

The Analyst maintains the :class:`~simharness.schemas.Casefile` from the known
ground truth; the Speaker phrases the next question. In a later phase the two
roles can be split into separate models; the interface stays the same.
"""

from __future__ import annotations

from simharness.schemas import (
    ActiveTarget,
    BusinessConfig,
    Casefile,
    Simulator,
    SimulatorContext,
    SimulatorInternalState,
    SimulatorOutput,
    TerminationReason,
)

__all__ = ["RedTeamSimulator"]


class RedTeamSimulator:
    """Deterministic red-team caller that probes a fixed list of ground-truth topics."""

    def __init__(self, ground_truth: BusinessConfig, max_turns: int = 10) -> None:
        self._business = ground_truth
        self._max_turns = max_turns
        self.casefile = _build_casefile(ground_truth)

    def generate(self, context: SimulatorContext) -> SimulatorOutput:
        state = context.internal_state.model_copy(deep=True)
        state.patience_remaining -= 1

        if state.patience_remaining <= 0:
            return SimulatorOutput(
                utterance="I think I have what I need, thank you.",
                internal_state=state,
                terminate=True,
                termination=TerminationReason.PATIENCE_EXHAUSTED,
            )

        if context.turn_index == 0:
            target = self._next_target()
            if target is None:
                return _hang_up(state)
            question = _question_for(target)
            self.casefile.next_move = question
            return SimulatorOutput(
                utterance=f"Hello, I have a few questions. {question}",
                internal_state=state,
            )

        target = self._next_target()
        if target is None:
            return _hang_up(state)

        question = _question_for(target)
        self.casefile.next_move = question
        return SimulatorOutput(utterance=question, internal_state=state)

    def _next_target(self) -> ActiveTarget | None:
        asked = set(self.casefile.confirmed_facts) | set(self.casefile.discrepancies)
        for target in self.casefile.active_targets:
            if target.field not in asked:
                return target
        return None


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


def _hang_up(state: SimulatorInternalState) -> SimulatorOutput:
    return SimulatorOutput(
        utterance="I think I have what I need, thank you.",
        internal_state=state,
        terminate=True,
        termination=TerminationReason.SATISFIED,
    )
