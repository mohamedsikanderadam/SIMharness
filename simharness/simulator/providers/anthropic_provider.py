"""Anthropic-backed counterpart policy.

The only module in the package that imports an LLM SDK on the simulator side —
`tests/test_boundaries.py` enforces that, so the verifier can never acquire a
provider dependency by accident.

**Structural no-leak, still.** The model is given the persona's goal and facts
because it has to play the part, but its output is constrained to a JSON schema
with exactly one text field: `utterance`. Withheld facts are never placed in the
prompt at all — the code filters them out before the request is built — so the
guarantee does not rest on the model's discretion.

API surface notes that are easy to get wrong on current models:
- `temperature` / `top_p` / `top_k` were removed; sending one is a 400.
- Thinking is on by default and `max_tokens` caps thinking *and* text together.
- A response can arrive with `stop_reason: "refusal"` and empty content — check
  it before indexing into `content`.
"""

from __future__ import annotations

import json
from typing import Any

from simharness.pricing import PRICE_TABLE_ID, estimate
from simharness.schemas import (
    Disclosure,
    Persona,
    SimulatorConfig,
    SimulatorContext,
    SimulatorOutput,
    Speaker,
    TerminationReason,
    TokenUsage,
)

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "utterance": {
            "type": "string",
            "description": "Exactly what you say out loud. One or two sentences.",
        },
        "terminate": {
            "type": "boolean",
            "description": "True if you are ending the call now.",
        },
        "termination": {
            "type": ["string", "null"],
            "enum": ["satisfied", "gave_up", "hung_up_angry", None],
            "description": "Why you are ending the call. Null unless terminate is true.",
        },
    },
    "required": ["utterance", "terminate", "termination"],
    "additionalProperties": False,
}

SYSTEM = """You are role-playing a customer on a phone call to a business. \
You are NOT an assistant and you are NOT trying to be helpful — you are the \
person who wants something.

Rules you must not break:
- Say only what this customer would say out loud. Never describe your own \
goal, your constraints, or your strategy.
- Facts listed as "only if asked" must NOT be volunteered. Wait until the agent \
actually asks for that specific thing, then give it.
- Stay in character even if the agent is slow, wrong, or unhelpful. Real \
customers get impatient; they do not break character to explain themselves.
- Keep it short. Real callers speak in about seven words per turn."""


class AnthropicSimulator:
    """LLM counterpart. Satisfies `simharness.simulator.base.Simulator`."""

    def __init__(self, config: SimulatorConfig | None = None, client: Any = None) -> None:
        self.config = config or SimulatorConfig()
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self.usd = 0.0
        self.usage = TokenUsage()
        self.price_table_id = PRICE_TABLE_ID
        self.refusals = 0

    # -- prompt ------------------------------------------------------------- #

    def _brief(self, persona: Persona) -> str:
        volunteered = [
            f.value for f in persona.hidden_facts if f.disclosure is Disclosure.VOLUNTEERED
        ]
        on_ask = [
            f"- {f.value}  (only if the agent asks about: {', '.join(f.asked_by) or f.key})"
            for f in persona.hidden_facts
            if f.disclosure is Disclosure.ON_DIRECT_QUESTION
        ]
        # WITHHELD facts are filtered out here and never reach the model. The
        # simulator cannot leak what it was never told.
        lines = [
            f"You are {persona.display_name}. Temperament: {persona.temperament}.",
            f"What you want: {persona.hidden_goal.summary}",
            "You are done when: "
            + (persona.hidden_goal.satisfied_when or "you get what you came for"),
        ]
        if volunteered:
            lines += ["", "You mention these freely:", *(f"- {v}" for v in volunteered)]
        if on_ask:
            lines += ["", "You give these ONLY when asked:", *on_ask]
        if persona.escalations:
            lines += [
                "",
                "If told no, push back in roughly this order, then give up:",
                *(f"- {e}" for e in persona.escalations),
            ]
        if persona.style_notes:
            lines += ["", f"Style: {persona.style_notes}"]
        return "\n".join(lines)

    def _messages(self, context: SimulatorContext) -> list[dict[str, Any]]:
        if not context.history:
            return [{"role": "user", "content": "[The call connects. You speak first.]"}]
        turns = []
        for view in context.history:
            role = "assistant" if view.speaker is Speaker.USER else "user"
            turns.append({"role": role, "content": view.text or "[silence]"})
        if turns[0]["role"] != "user":
            turns.insert(0, {"role": "user", "content": "[The call connects.]"})
        if turns[-1]["role"] != "user":
            turns.append({"role": "user", "content": "[The agent says nothing.]"})
        return turns

    # -- the one method ----------------------------------------------------- #

    def generate(self, context: SimulatorContext) -> SimulatorOutput:
        state = context.internal_state.model_copy(deep=True)
        state.patience_remaining -= 1
        if state.patience_remaining <= 0:
            return SimulatorOutput(
                utterance="I'll have to leave it there.",
                internal_state=state,
                terminate=True,
                termination=TerminationReason.PATIENCE_EXHAUSTED,
            )

        response = self._client.beta.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            # Opting into server-side fallbacks by default: a policy decline on
            # a benign booking call is unlikely but not impossible, and without
            # this the request simply stops.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=[
                {
                    "type": "text",
                    "text": SYSTEM,
                    # Identical on every call of every episode — worth a breakpoint.
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": self._brief(context.persona)},
            ],
            output_config={
                "effort": self.config.effort,
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
            messages=self._messages(context),
        )
        self._account(response)

        if response.stop_reason == "refusal":
            self.refusals += 1
            return SimulatorOutput(
                utterance="Sorry, I have to go.",
                internal_state=state,
                terminate=True,
                termination=TerminationReason.GAVE_UP,
            )

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # The schema makes this near-impossible, but a malformed body must
            # not take the episode down — it ends as a harness error, visibly.
            return SimulatorOutput(
                utterance="Sorry?",
                internal_state=state,
                terminate=True,
                termination=TerminationReason.HARNESS_ERROR,
            )

        terminate = bool(payload.get("terminate"))
        reason = payload.get("termination")
        return SimulatorOutput(
            utterance=str(payload.get("utterance", "")).strip() or "...",
            internal_state=state,
            terminate=terminate,
            termination=TerminationReason(reason) if terminate and reason else None,
        )

    # -- cost --------------------------------------------------------------- #

    def _account(self, response: Any) -> None:
        usage = response.usage
        self.usage = TokenUsage(
            prompt_tokens=self.usage.prompt_tokens + usage.input_tokens,
            completion_tokens=self.usage.completion_tokens + usage.output_tokens,
        )
        self.usd += estimate(
            self.config.model,
            usage.input_tokens,
            usage.output_tokens,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
