"""The counterpart policy, behind one swappable interface."""

from simharness.simulator.base import ScriptedSimulator, Simulator

__all__ = ["ScriptedSimulator", "Simulator"]


def anthropic_simulator(*args: object, **kwargs: object) -> Simulator:
    """Lazy factory: importing the package must not require the SDK."""
    from simharness.simulator.providers.anthropic_provider import AnthropicSimulator

    return AnthropicSimulator(*args, **kwargs)  # type: ignore[arg-type]
