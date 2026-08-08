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


def test_analyst_confirms_truth_phrased_differently() -> None:
    """The truth in words is still the truth; string matching scored this wrong."""
    analyst = Analyst(_sample_business())
    analyst.update("The deposit comes to fifteen — 15 pounds a head.")

    assert "deposit" in analyst.casefile.confirmed_facts
    assert not analyst.casefile.cracked


def test_analyst_catches_lie_phrased_differently() -> None:
    """A rephrased lie is still a lie, so it must stage a clarification."""
    analyst = Analyst(_sample_business())
    analyst.update("The deposit is 20 pounds a head.")

    assert analyst.casefile.pending_clarification == "deposit"
    assert "deposit" not in analyst.casefile.confirmed_facts


def test_analyst_ignores_topic_raised_without_a_value() -> None:
    """A hedge is neither a confirmation nor a contradiction."""
    analyst = Analyst(_sample_business())
    analyst.update("I'd have to check what the deposit is.")

    assert not analyst.casefile.confirmed_facts
    assert not analyst.casefile.discrepancies
    assert analyst.casefile.pending_clarification is None


def test_analyst_confirms_every_field_of_a_truthful_client() -> None:
    """The negative control: a truthful client must produce no discrepancies."""
    analyst = Analyst(_sample_business())
    for claim in (
        "The deposit is £15.00 per person.",
        "The set lunch is £24.00.",
        "We open 12:00 and close 23:00.",
        "The cancellation window is 24 hours.",
        "The largest party we can seat is 12.",
    ):
        analyst.update(claim)

    assert not analyst.casefile.discrepancies
    assert not analyst.casefile.cracked
    assert analyst.next_target() is None


def test_analyst_reads_scraped_deadlines_written_as_am_pm() -> None:
    """Scraped policy text says "2pm", not "14:00", and midnight is a real answer."""
    analyst = Analyst(_sample_business())
    analyst.track("cancellation deadline", "cancel free up until 2pm the day before arrival")

    analyst.update("You can cancel right up until 2pm the day before you arrive.")
    assert "cancellation deadline" in analyst.casefile.confirmed_facts

    other = Analyst(_sample_business())
    other.track("cancellation deadline", "cancel free up until 2pm the day before arrival")
    other.update("You can cancel free up until midnight on the day of arrival.")
    assert other.casefile.pending_clarification == "cancellation deadline"


def test_analyst_tracks_a_fact_it_did_not_start_with() -> None:
    """Ground truth found mid-call becomes an ordinary target, and is deduplicated."""
    analyst = Analyst(_sample_business())
    before = len(analyst.casefile.active_targets)

    analyst.track("breakfast", "£12.00")
    analyst.track("breakfast", "£99.00")

    assert len(analyst.casefile.active_targets) == before + 1
    assert analyst.casefile.active_targets[-1].true_value == "£12.00"

    analyst.update("Breakfast is £99.00 a head.")
    assert analyst.casefile.pending_clarification == "breakfast"


def test_analyst_stages_clarification_before_crack() -> None:
    """A wrong first answer stages clarification, not an immediate crack."""
    analyst = Analyst(_sample_business())
    analyst.update("The deposit is £20.00 per person.")

    assert analyst.casefile.pending_clarification == "deposit"
    assert "deposit" not in analyst.casefile.discrepancies
    assert not analyst.casefile.cracked

    # Client repeats the same wrong value on the clarification turn.
    analyst.update("I said the deposit is £20.00 per person.")

    assert "deposit" in analyst.casefile.discrepancies
    assert analyst.casefile.cracked
    assert analyst.casefile.pending_clarification is None


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


def test_speaker_clarifies_pending_target() -> None:
    analyst = Analyst(_sample_business())
    speaker = SpeakerClass()
    target = analyst.next_target()
    assert target is not None

    text = speaker.clarify(target)
    assert target.field in text.lower()
    assert target.true_value in text
    assert "confirm" in text.lower()


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

    # The simulator sees the client response and stages a clarification.
    red_team.observe(client_response)
    assert red_team.casefile.pending_clarification == "deposit"
    assert "deposit" not in red_team.casefile.discrepancies
    assert not red_team.casefile.cracked

    # The next turn asks a clarification.
    context = _context_for_turn(red_team, turn_index=2)
    output = red_team.generate(context)
    assert "£15.00" in output.utterance
    assert "confirm" in output.utterance.lower()

    # Client repeats the false value on the clarification.
    request2 = AgentRequest(
        episode_id="test-0",
        turn_index=3,
        history=(
            AgentTurnView(speaker=SpeakerEnum.USER, text="What is your deposit policy?"),
            AgentTurnView(speaker=SpeakerEnum.AGENT, text=client_response),
            AgentTurnView(speaker=SpeakerEnum.USER, text=output.utterance),
        ),
        tools=(),
        brief="",
    )
    client_response2 = client.respond(request2).text
    red_team.observe(client_response2)
    assert "deposit" in red_team.casefile.discrepancies
    assert red_team.casefile.cracked


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
