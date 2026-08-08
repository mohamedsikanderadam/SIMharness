"""A simulated client agent that answers from a ground-truth business or a set
of injected false beliefs.

This is the agent-under-test for the red-team harness. It sees only the
``AgentRequest`` the runner hands it, exactly like a real adapter would.
"""

from __future__ import annotations

from simharness.schemas import (
    Agent,
    AgentRequest,
    AgentResponse,
    BusinessConfig,
    ClientBeliefs,
    Speaker,
)

__all__ = ["SimulatedClientAgent"]


class SimulatedClientAgent:
    """A scripted client that responds to keyword-matched topics.

    If the caller asks about a topic in :class:`~simharness.schemas.ClientBeliefs`,
    the agent parrots that (potentially false) belief. Otherwise it falls back to
    a ground-truth answer derived from the :class:`~simharness.schemas.BusinessConfig`.
    """

    def __init__(self, business: BusinessConfig, beliefs: ClientBeliefs | None = None) -> None:
        self._business = business
        self._beliefs = beliefs or ClientBeliefs()

    def respond(self, request: AgentRequest) -> AgentResponse:
        last_user = self._last_user_text(request)
        if not last_user:
            return AgentResponse(text=f"Hello, you're through to {self._business.name}.")

        lower = last_user.lower()

        for field, claim in self._beliefs.facts.items():
            if field in lower:
                return AgentResponse(text=claim)

        answer = self._ground_truth(lower)
        if answer:
            return AgentResponse(text=answer)

        return AgentResponse(text="I'm not sure I understand — could you rephrase that?")

    @staticmethod
    def _last_user_text(request: AgentRequest) -> str:
        for view in reversed(request.history):
            if view.speaker is Speaker.USER:
                return view.text
        return ""

    def _ground_truth(self, text: str) -> str | None:
        policies = self._business.policies

        if "deposit" in text:
            return f"The deposit is £{policies.deposit_per_head / 100:.2f} per person."

        if "cancel" in text or "cancellation" in text:
            return f"You can cancel up to {policies.cancellation_window_hours} hours before."

        if "open" in text or "hour" in text:
            hours = self._business.opening_hours
            if hours:
                day = next((h for h in hours if not h.closed), hours[0])
                return (
                    f"We're open from {day.opens:%H:%M} to {day.closes:%H:%M} "
                    f"on {self._weekday_name(day.weekday)}s."
                )

        if "lunch" in text or "price" in text or "cost" in text:
            if self._business.catalogue:
                item = self._business.catalogue[0]
                return f"The {item.name.lower()} is £{item.unit_price / 100:.2f}."

        if "party" in text or "capacity" in text or "seat" in text:
            return f"The largest party we can seat is {policies.max_party_size}."

        return None

    @staticmethod
    def _weekday_name(n: int) -> str:
        return ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")[n]
