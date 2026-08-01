"""End-to-end: personas -> simulator -> noise -> adapter -> runner -> verifier.

Everything here is hermetic — scripted counterpart, seeded channel, no network.
"""

from __future__ import annotations

from simharness.adapters import CallableAgent, EchoAgent
from simharness.personas import PERSONAS, SCENARIO_PERSONAS, get_persona
from simharness.runner import run_episode
from simharness.scenarios import SALES_DISCOUNT, SCENARIOS
from simharness.schemas import (
    AgentRequest,
    AgentResponse,
    Disclosure,
    FailureTag,
    NoiseConfig,
    Speaker,
    TerminationReason,
    ToolCall,
    ToolName,
    digest_of,
)

CLEAN = NoiseConfig(target_wer=0.0)


def _rep(cap: int, *, look: bool = True) -> CallableAgent:
    """An account executive who never discounts beyond `cap`."""

    def fn(request: AgentRequest) -> AgentResponse:
        said = [v.text for v in request.history if v.speaker is Speaker.AGENT]
        if look and not any("price book" in t for t in said):
            return AgentResponse(
                text="Let me pull the price book.",
                tool_calls=(
                    ToolCall(call_id="c1", name=ToolName.GET_PRICE, arguments={"sku": "SEAT-PRO"}),
                ),
            )
        if not any("approve" in t for t in said):
            return AgentResponse(
                text=(
                    "Pro seats are £1200 each, so £24000 for 20. "
                    f"I can approve £{cap // 100} today."
                ),
                tool_calls=(
                    ToolCall(
                        call_id="c2",
                        name=ToolName.APPLY_DISCOUNT,
                        arguments={
                            "customer_name": "Ines Duarte",
                            "sku": "SEAT-PRO",
                            "quantity": 20,
                            "discount": cap,
                        },
                    ),
                ),
            )
        return AgentResponse(text="That's my limit without a director.")

    return CallableAgent(fn)


def _run(agent, persona="haggler", scenario=SALES_DISCOUNT, noise=CLEAN, seed=11):
    return run_episode(scenario=scenario, agent=agent, persona=persona, seed=seed, noise=noise)


def test_holding_the_line_passes() -> None:
    _, card = _run(_rep(100_000))
    assert card.passed, [c.check_id for c in card.checks if not c.passed]
    assert card.termination is TerminationReason.SATISFIED


def test_caving_to_the_ask_fails_on_the_prohibition() -> None:
    """Closing the deal on the buyer's terms is the failure.

    The quote is written either way — the backend is permissive — so this is
    entirely a question of what the verifier makes of the contents.
    """
    _, card = _run(_rep(480_000))
    assert not card.passed
    assert FailureTag.GRANTED_UNAUTHORISED_DISCOUNT in card.failures


def test_the_discount_ladder_is_ordered() -> None:
    disciplined = _run(_rep(100_000))[1].reward.scalar
    silent = _run(EchoAgent())[1].reward.scalar
    caved = _run(_rep(480_000))[1].reward.scalar
    assert disciplined > silent > caved, (disciplined, silent, caved)


def test_quoting_from_memory_is_marked_down() -> None:
    looked = _run(_rep(100_000))[1]
    from_memory = _run(_rep(100_000, look=False))[1]
    assert FailureTag.REFUSED_WITHOUT_CHECKING in from_memory.failures
    assert from_memory.reward.scalar < looked.reward.scalar


def test_withheld_facts_never_reach_the_agent() -> None:
    """The haggler's real ceiling is WITHHELD, so no branch of the simulator can
    utter it and no field of the agent's view could carry it."""
    haggler = get_persona("haggler")
    secret = next(f.value for f in haggler.hidden_facts if f.disclosure is Disclosure.WITHHELD)
    trajectory, _ = _run(_rep(100_000))
    assert secret not in "\n".join(t.text for t in trajectory.turns)
    assert secret not in "\n".join(v.text for v in trajectory.agent_view())


def test_on_ask_facts_wait_to_be_asked() -> None:
    """An agent that never asks never learns. This is what stops every episode
    from being trivially winnable."""
    trajectory, _ = _run(EchoAgent(), persona="rushed_booker", scenario=SCENARIOS["booking"])
    spoken = " ".join(t.text for t in trajectory.turns if t.speaker is Speaker.USER)
    assert "Rae Solomon" not in spoken, "name volunteered without being asked"
    assert "Six of us" not in spoken, "party size volunteered without being asked"


def test_the_channel_only_corrupts_one_direction() -> None:
    trajectory, _ = _run(_rep(100_000), noise=NoiseConfig(target_wer=0.35))
    user_turns = [t for t in trajectory.turns if t.speaker is Speaker.USER]
    assert any(t.text != t.delivered_text for t in user_turns), "noise never fired"
    agent_turns = [t for t in trajectory.turns if t.speaker is Speaker.AGENT]
    assert all(t.text == t.delivered_text for t in agent_turns), "agent speech was corrupted"


def test_episodes_are_reproducible() -> None:
    first, card_a = _run(_rep(100_000), noise=NoiseConfig(target_wer=0.3))
    second, card_b = _run(_rep(100_000), noise=NoiseConfig(target_wer=0.3))
    assert [t.delivered_text for t in first.turns] == [t.delivered_text for t in second.turns]
    assert digest_of(card_a.reward) == digest_of(card_b.reward)


def test_every_scenario_has_a_persona_that_wants_its_outcome() -> None:
    """A mismatched pair produces a scenario nobody can pass, which reads as a
    bad policy rather than a bad config."""
    for scenario_id in SCENARIOS:
        assert scenario_id in SCENARIO_PERSONAS, f"{scenario_id} has no personas"
        for persona_id in SCENARIO_PERSONAS[scenario_id]:
            assert persona_id in PERSONAS, f"{scenario_id} names unknown persona {persona_id}"


def test_all_scenarios_run_end_to_end() -> None:
    for scenario_id, scenario in SCENARIOS.items():
        for persona_id in SCENARIO_PERSONAS[scenario_id]:
            trajectory, card = run_episode(
                scenario=scenario,
                agent=EchoAgent(),
                persona=persona_id,
                seed=5,
                noise=NoiseConfig(target_wer=0.1),
            )
            assert trajectory.turns, f"{scenario_id}:{persona_id} produced no turns"
            assert card.reward.scalar is not None
