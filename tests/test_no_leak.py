"""The agent cannot see hidden state, and this is enforced by types.

These assertions are deliberately made against the projection rather than against
a formatted prompt string. A test that greps a prompt for a secret passes right
up until someone changes the prompt template; a test that shows the *type* has
nowhere to put a secret keeps holding.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from simharness.schemas import (
    AgentTurnView,
    SimulatorInternalState,
    SimulatorTurnView,
    Speaker,
    Turn,
)

SECRET = "budget-is-90-pounds-do-not-reveal"


def _user_turn() -> Turn:
    return Turn(
        index=0,
        speaker=Speaker.USER,
        text="A table for fifteen, please.",
        delivered_text="A table for fifty, please.",
        internal_state=SimulatorInternalState(patience_remaining=5, scratchpad=SECRET),
    )


def test_the_agent_view_has_nowhere_to_put_hidden_state() -> None:
    assert set(AgentTurnView.model_fields) == {"speaker", "text"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentTurnView(speaker=Speaker.USER, text="hi", scratchpad=SECRET)  # type: ignore[call-arg]


def test_projection_drops_the_simulator_scratchpad() -> None:
    view = AgentTurnView.from_turn(_user_turn())
    assert SECRET not in view.model_dump_json()
    assert "internal_state" not in view.model_dump()


def test_the_agent_hears_the_delivered_text_and_the_customer_remembers_the_spoken_one() -> None:
    """The one asymmetry the whole noise model rests on."""
    turn = _user_turn()
    assert AgentTurnView.from_turn(turn).text == "A table for fifty, please."
    assert SimulatorTurnView.from_turn(turn).text == "A table for fifteen, please."


def test_system_turns_never_reach_the_agent() -> None:
    system = Turn(index=0, speaker=Speaker.SYSTEM, text=SECRET, delivered_text=SECRET)
    with pytest.raises(ValueError, match="system turns"):
        AgentTurnView.from_turn(system)


def test_tool_turns_never_reach_the_simulator() -> None:
    tool = Turn(index=0, speaker=Speaker.TOOL, text="{}", delivered_text="{}")
    with pytest.raises(ValueError, match="only user and agent"):
        SimulatorTurnView.from_turn(tool)


def test_a_whole_trajectory_projects_without_leaking() -> None:
    from simharness.scenarios import BOOKING
    from tests.helpers import Episode

    episode = Episode(BOOKING)
    episode.user("Table for six.", heard="Table for sixty.")
    episode.agent("Certainly.")
    episode.turns[0].internal_state = SimulatorInternalState(
        patience_remaining=3, scratchpad=SECRET
    )
    trajectory = episode.finish()

    serialised = "\n".join(view.model_dump_json() for view in trajectory.agent_view())
    assert SECRET not in serialised
    assert "Table for sixty." in serialised, "the agent should see the corrupted text"
    assert "Table for six." not in serialised, "the agent must not see the clean text"
