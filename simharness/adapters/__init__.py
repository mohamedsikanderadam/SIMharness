"""The agent-under-test boundary."""

from simharness.adapters.base import Agent, CallableAgent, EchoAgent
from simharness.adapters.http import HTTPAgent, build_generic, parse_generic

__all__ = [
    "Agent",
    "CallableAgent",
    "EchoAgent",
    "HTTPAgent",
    "build_generic",
    "parse_generic",
]


def anthropic_agent(*args: object, **kwargs: object) -> Agent:
    """Lazy factory: importing the package must not require the SDK."""
    from simharness.adapters.llm import AnthropicAgent

    return AnthropicAgent(*args, **kwargs)  # type: ignore[arg-type]
