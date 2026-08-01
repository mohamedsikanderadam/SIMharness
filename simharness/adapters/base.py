"""The agent-under-test boundary.

One protocol, and the runner holds something satisfying it. The runner must not
be able to tell which implementation it has — if a feature ever needs
`isinstance(agent, LocalPolicyAgent)` outside this package, the feature is wrong.

`HTTPAgent` (the eval product) and a real `LocalPolicyAgent` (RL rollouts) are
Phase 4. What is here is the protocol plus `CallableAgent`, which is enough to
drive the whole pipeline from a plain function — an in-process policy, a rule
based reference, or a test stub — and is how the runner gets exercised without a
network.

Note what an adapter receives: `AgentRequest`, whose history is a tuple of
`AgentTurnView`. It never sees the world, the persona, the scenario, or the
customer's pre-noise words. That is enforced by the type, not by this docstring.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from simharness.schemas import AgentRequest, AgentResponse


@runtime_checkable
class Agent(Protocol):
    """The agent under test. One method; adding a third implementation touches
    nothing else in the package."""

    def respond(self, request: AgentRequest) -> AgentResponse: ...


class CallableAgent:
    """Wraps a plain function as an agent.

    The function receives the full `AgentRequest` and returns an `AgentResponse`,
    so a policy written for this is already written for the real adapters.
    """

    def __init__(self, fn: Callable[[AgentRequest], AgentResponse]) -> None:
        self._fn = fn

    def respond(self, request: AgentRequest) -> AgentResponse:
        return self._fn(request)


class EchoAgent:
    """A deliberately useless agent: says one thing, calls nothing.

    Exists as the floor for any evaluation. If a scenario cannot tell this apart
    from a competent agent, the scenario is not measuring anything — which is a
    check worth having, because "do nothing" silently passes any criterion that
    is only a prohibition.
    """

    def __init__(self, line: str = "Right, yes, of course.") -> None:
        self.line = line

    def respond(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(text=self.line)
