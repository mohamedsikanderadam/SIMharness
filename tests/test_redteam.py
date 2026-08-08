"""Tests for the red-team Analyst -> Speaker and Speaker -> Analyst handoffs."""

from __future__ import annotations

from datetime import time

from simharness.adapters.client import SimulatedClientAgent
from simharness.runner import run_red_team_episode
from simharness.schemas import (
    AgentTurnView,
    BusinessConfig,
    CatalogueItem,
    ClientBeliefs,
    OpeningHours,
    Policies,
    Speaker as SpeakerEnum,
    SimulatorContext,
    SimulatorInternalState,
    SimulatorTurnView,
)
from simharness.simulator.redteam import Analyst, RedTeamSimulator, Speaker as SpeakerClass


def _sample_business() -> BusinessConfig:
    return BusinessConfig(
        business_id="bistro-nine",
        name="Bistro Nine",
        opening_hours=tuple(
            OpeningHours(weekday=d, opens=time(12, 0), closes=time(23, 0))
            for d in range(7)
        ),
        policies=Policies(
            cancellation_window_hours=24,
            deposit_required_from_party_size=6,
            deposit_per_head=1500,
            refund_window_hours=48,
            max_party_size=12,
            discount_authority=0,
        ),
        catalogue=(
            CatalogueItem(sku="SET-LUNCH", name="Set lunch", unit_price=2400),
            CatalogueItem(sku="SET-DINNER", name="Set dinner", unit_price=4500),
        ),
    )


def test_analyst_marks_confirmed_when_client_says_truth() -> None:
    analyst = Analyst(_sample_business())
    analyst.update("The deposit is £15.00 per person.")

    assert "deposit" in analyst.casefile.confirmed_facts
    assert not analyst.casefile.discrepancies
    assert not analyst.casefile.cracked


def test_analyst_marks_discrepancy_when_client_lies() -> None:
    analyst = Analyst(_sample_business())
    analyst.update("The deposit is £20.00 per person.")

    assert "deposit" in analyst.casefile.discrepancies
    assert analyst.casefile.cracked
    assert "deposit" not in analyst.casefile.confirmed_facts


def test_speaker_phrases_question_for_active_target() -> None:
    analyst = Analyst(_sample_business())
    speaker = SpeakerClass()
    target = analyst.next_target()

    assert target is not None
    question = speaker.phrase(target)
    assert "deposit" in question.lower()


def test_speaker_phrases_goodbye_when_no_target() -> None:
    speaker = SpeakerClass()
    text = speaker.phrase(None)
    assert "thank you" in text.lower()


def test_analyst_to_speaker_handoff_skips_confirmed_target() -> None:
    """The Speaker must receive the updated Casefile and ask the next open target."""
    analyst = Analyst(_sample_business())
    speaker = SpeakerClass()

    # Client truthfully answers the deposit question.
    analyst.update("The deposit is £15.00 per person.")

    next_target = analyst.next_target()
    assert next_target is not None
    assert next_target.field != "deposit"

    question = speaker.phrase(next_target)
    assert next_target.field in question.lower()


def test_speaker_to_analyst_handoff_via_red_team_simulator() -> None:
    """After the client answers, the next simulator turn uses the updated Casefile."""
    business = _sample_business()
    red_team = RedTeamSimulator(business, max_turns=4)
    client = SimulatedClientAgent(
        business,
        ClientBeliefs(facts={"deposit": "The deposit is £20.00 per person."}),
    )

    # First red-team turn: ask about deposit.
    context = _context_for_turn(red_team, turn_index=0)
    output = red_team.generate(context)
    assert "deposit" in output.utterance.lower()

    # Client responds with the false deposit value.
    from simharness.schemas import AgentRequest

    request = AgentRequest(
        episode_id="test-0",
        turn_index=1,
        history=(AgentTurnView(speaker=SpeakerEnum.USER, text="What is your deposit policy?"),),
        tools=(),
        brief="",
    )
    client_response = client.respond(request).text

    # The simulator sees the client response on the next turn and updates the Casefile.
    red_team.observe(client_response)
    assert "deposit" in red_team.casefile.discrepancies
    assert red_team.casefile.cracked

    # The next simulator question should now skip the cracked deposit.
    context = _context_for_turn(red_team, turn_index=2)
    output = red_team.generate(context)
    assert "deposit" not in output.utterance.lower()


def test_run_red_team_episode_detects_false_deposit() -> None:
    result = run_red_team_episode(
        business=_sample_business(),
        client_beliefs=ClientBeliefs(
            facts={"deposit": "The deposit is £20.00 per person."}
        ),
        max_turns=4,
    )
    assert result.cracked
    assert "deposit" in result.casefile.discrepancies


def _context_for_turn(red_team: RedTeamSimulator, turn_index: int) -> SimulatorContext:
    from simharness.schemas import HiddenGoal, Persona, Temperament

    history: tuple[SimulatorTurnView, ...] = ()
    return SimulatorContext(
        persona=Persona(
            persona_id="red_team",
            display_name="Red Team",
            temperament=Temperament.BRISK,
            hidden_goal=HiddenGoal(summary="Expose false claims"),
            patience_turns=4,
        ),
        internal_state=SimulatorInternalState(patience_remaining=4),
        history=history,
        turn_index=turn_index,
        seed=0,
    )
