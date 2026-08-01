"""The counterpart policy interface, and its deterministic implementation.

`Simulator` is one method. That is the whole point: an LLM-backed simulator, a
scripted one, and a later *trained* one are all the same shape, so swapping them
touches nothing else. `SimulatorContext` in and `SimulatorOutput` out — both
plain serialisable models — is also what makes a trained counterpart a provider
swap rather than a rewrite.

`ScriptedSimulator` is the hermetic path: no network, no key, byte-identical
across runs. Determinism tests and CI use it, and it is what makes the harness
runnable before anyone has spent a penny on tokens.

**The failure mode this is built against.** LLM user simulators drift into being
maximally cooperative — they answer questions nobody asked, accept the first
offer, and volunteer the very fact the scenario was designed to withhold. Every
task then succeeds and the reward signal flattens. Enforcing disclosure here,
structurally, from `Persona.hidden_facts`, means a low-disclosure persona cannot
drift: the fact is not in the utterance because the code did not put it there.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from simharness.schemas import (
    Disclosure,
    HiddenFact,
    Persona,
    SimulatorContext,
    SimulatorInternalState,
    SimulatorOutput,
    Speaker,
    TerminationReason,
)


@runtime_checkable
class Simulator(Protocol):
    """One method. Providers differ; the contract does not."""

    def generate(self, context: SimulatorContext) -> SimulatorOutput: ...


def _last_agent_text(context: SimulatorContext) -> str:
    for view in reversed(context.history):
        if view.speaker is Speaker.AGENT:
            return view.text
    return ""


def _asked_about(fact: HiddenFact, agent_text: str) -> bool:
    lowered = agent_text.lower()
    return any(trigger in lowered for trigger in fact.asked_by)


class ScriptedSimulator:
    """Deterministic counterpart driven entirely by the persona's declared policy."""

    #: Agent phrasings that read as a refusal, for personas that escalate.
    REFUSAL_MARKERS = (
        "can't",
        "cannot",
        "unable to",
        "not able to",
        "afraid not",
        "i'm sorry",
        "no refund",
        "not something i can",
    )

    def __init__(self, satisfied_markers: tuple[str, ...] | None = None) -> None:
        self.satisfied_markers = satisfied_markers or (
            "all booked",
            "you're booked",
            "booked you in",
            "confirmed",
            "refunded",
            "moved you",
            "quote is",
            "sent the quote",
            "written it up",
        )

    def generate(self, context: SimulatorContext) -> SimulatorOutput:
        persona = context.persona
        state = context.internal_state.model_copy(deep=True)
        agent_text = _last_agent_text(context)

        if not context.history:
            return self._speak(persona, state, self._opening(persona), opening=True)

        if self._is_satisfied(agent_text, persona):
            state.goal_progress = 1.0
            return self._finish(state, "That's great, thank you.", TerminationReason.SATISFIED)

        state.patience_remaining -= 1
        if state.patience_remaining <= 0:
            return self._finish(state, "I'll leave it there.", TerminationReason.PATIENCE_EXHAUSTED)

        revealed = self._reveal(persona, state, agent_text)
        if revealed is not None:
            return self._speak(persona, state, revealed)

        if persona.escalations:
            rung = len([k for k in state.revealed_fact_keys if k.startswith("__escalation")])
            # Exhausting the ladder ends the call, full stop. Gating this on
            # detecting a refusal in the agent's wording made termination depend
            # on a keyword list: a rep who said "that's the best I can do on my
            # own authority" was refusing in every sense except the one the
            # matcher recognised, so the call timed out instead of ending, and a
            # correct agent lost the clean-termination credit it had earned.
            if rung >= len(persona.escalations) - 1:
                state.mood = -1.0
                return self._finish(state, "Fine. Forget it.", TerminationReason.GAVE_UP)
            state.revealed_fact_keys.append(f"__escalation{rung}")
            state.mood = max(-1.0, state.mood - 0.25)
            return self._speak(
                persona, state, persona.escalations[min(rung, len(persona.escalations) - 1)]
            )

        return self._speak(persona, state, "Right. Is that everything you need from me?")

    # -- internals ---------------------------------------------------------- #

    @staticmethod
    def _opening(persona: Persona) -> str:
        volunteered = [
            fact.value for fact in persona.hidden_facts if fact.disclosure is Disclosure.VOLUNTEERED
        ]
        opening = persona.opening or persona.hidden_goal.summary
        return " ".join([opening, *volunteered]).strip()

    def _is_satisfied(self, agent_text: str, persona: Persona) -> bool:
        lowered = agent_text.lower()
        markers = persona.satisfied_markers or self.satisfied_markers
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _reveal(persona: Persona, state: SimulatorInternalState, agent_text: str) -> str | None:
        """Hand over a fact only if the agent actually asked for it.

        `WITHHELD` facts are unreachable from here by construction — there is no
        branch that can return one — which is the structural half of the no-leak
        guarantee. The other half is `AgentTurnView`.
        """
        for fact in persona.hidden_facts:
            if fact.disclosure is not Disclosure.ON_DIRECT_QUESTION:
                continue
            if fact.key in state.revealed_fact_keys:
                continue
            if _asked_about(fact, agent_text):
                state.revealed_fact_keys.append(fact.key)
                return fact.value
        return None

    @staticmethod
    def _speak(
        persona: Persona,
        state: SimulatorInternalState,
        utterance: str,
        *,
        opening: bool = False,
    ) -> SimulatorOutput:
        if opening:
            state.goal_progress = 0.1
        return SimulatorOutput(utterance=utterance, internal_state=state)

    @staticmethod
    def _finish(
        state: SimulatorInternalState, utterance: str, reason: TerminationReason
    ) -> SimulatorOutput:
        return SimulatorOutput(
            utterance=utterance, internal_state=state, terminate=True, termination=reason
        )
