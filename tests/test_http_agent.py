"""The eval product: a vendor agent over HTTP, scored on transcript evidence.

Hermetic — a fake transport stands in for the vendor, so the whole black-box
path is proven before anyone points it at a live endpoint or spends a cent.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from simharness.adapters import EchoAgent
from simharness.adapters.http import HTTPAgent, parse_generic
from simharness.runner import run_episode
from simharness.scenarios.blackbox import blackbox_scenario
from simharness.schemas import (
    AgentRequest,
    ClaimVerdict,
    FailureTag,
    NoiseConfig,
    Speaker,
)
from simharness.simulator.base import ScriptedSimulator
from simharness.world.builders import WORLD_BUILDERS
from simharness.world.factsheet import world_from_facts

FACTS: dict[str, Any] = {
    "business_id": "test-clinic",
    "name": "Test Clinic",
    "currency": "AED",
    "catalogue": [{"sku": "CONSULT", "name": "Consultation", "unit_price": 25000}],
    "policies": {"cancellation_window_hours": 24, "max_party_size": 1},
    "opening_hours": {"open": "08:00", "close": "20:00", "closed_weekday": 4},
    "slots": {"days": 3, "start_hour": 9, "count": 4, "capacity": 2},
}


def _transport(replies: list[str]) -> MagicMock:
    """A fake vendor that says each reply in turn, then repeats the last."""
    client = MagicMock()
    calls = {"n": 0}

    def post(_url: str, json: dict[str, Any]) -> MagicMock:
        index = min(calls["n"], len(replies) - 1)
        calls["n"] += 1
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"reply": replies[index]}
        return response

    client.post.side_effect = post
    return client


def _register(name: str = "test-facts") -> str:
    WORLD_BUILDERS[name] = lambda _seed: world_from_facts(FACTS)
    return name


def _scenario(confirm: int | None = None) -> Any:
    return blackbox_scenario(
        scenario_id="blackbox",
        title="Test Clinic",
        world_builder=_register(),
        confirm_value=confirm,
    )


def _run(agent: Any, confirm: int | None = None, wer: float = 0.0) -> Any:
    return run_episode(
        scenario=_scenario(confirm),
        agent=agent,
        persona="rushed_booker",
        seed=3,
        noise=NoiseConfig(target_wer=wer),
        simulator=ScriptedSimulator(),
    )


# --------------------------------------------------------------------------- #
# The fact sheet is the ground truth
# --------------------------------------------------------------------------- #


def test_a_price_from_the_fact_sheet_is_correct() -> None:
    agent = HTTPAgent(
        "https://vendor.test/agent",
        client=_transport(["A consultation is 250 dirhams. Anything else?"]),
    )
    _, card = _run(agent)
    assert not [c for c in card.claim_checks if c.verdict is ClaimVerdict.UNGROUNDED]


def test_a_price_that_contradicts_the_fact_sheet_is_caught() -> None:
    """The product in one test: the vendor's agent quotes a number nobody
    published, and the scorecard names it."""
    agent = HTTPAgent(
        "https://vendor.test/agent",
        client=_transport(["A consultation is 900 dirhams. Anything else?"]),
    )
    _, card = _run(agent)
    bad = [c for c in card.claim_checks if c.verdict is ClaimVerdict.UNGROUNDED]
    assert bad, [(c.surface, c.verdict) for c in card.claim_checks]
    assert FailureTag.HALLUCINATED_PRICE in card.failures


def test_facts_load_as_minor_units() -> None:
    world = world_from_facts(FACTS)
    assert world.business.catalogue[0].unit_price == 25000  # AED 250.00
    assert world.business.policies.cancellation_window_hours == 24
    assert len(world.business.calendar) == 3 * 4


# --------------------------------------------------------------------------- #
# What the black-box scenario does and does not claim
# --------------------------------------------------------------------------- #


def test_black_box_scenario_declares_no_world_checks() -> None:
    """Vacuous record checks would inflate every score. They are left empty on
    purpose, and the scorecard is narrower as a result."""
    scenario = _scenario()
    assert scenario.success.required_records == ()
    assert scenario.success.forbidden_mutations == ()


def test_silence_does_not_pass() -> None:
    """An agent that says nothing useful must not score like one that works."""
    _, useless = _run(EchoAgent("Mm-hmm."))
    agent = HTTPAgent(
        "https://vendor.test/agent",
        client=_transport(
            ["A consultation is 250 dirhams and we're open until 8pm. Booked you in."]
        ),
    )
    _, working = _run(agent)
    assert working.reward.scalar > useless.reward.scalar


def test_confirmation_check_survives_without_a_backend() -> None:
    agent = HTTPAgent("https://vendor.test/agent", client=_transport(["Certainly."]))
    _, card = _run(agent, confirm=6)
    assert FailureTag.MISHEARD_NUMBER_UNCONFIRMED in card.failures


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


def test_a_failing_vendor_does_not_end_the_run() -> None:
    """A 500 on one turn is recorded and the call continues — an eval that dies
    on the first flaky response cannot measure a production endpoint."""
    client = MagicMock()
    response = MagicMock()
    response.status_code = 500
    response.text = "upstream error"
    client.post.return_value = response

    agent = HTTPAgent("https://vendor.test/agent", retries=0, client=client)
    trajectory, card = _run(agent)
    assert agent.errors > 0
    assert trajectory.turns
    assert card.reward.scalar is not None


def test_latency_is_captured() -> None:
    agent = HTTPAgent("https://vendor.test/agent", client=_transport(["Hello."]))
    _run(agent)
    assert agent.latencies_ms and agent.p95_latency_ms >= 0


def test_sessions_never_leak_across_episodes() -> None:
    """Carrying a vendor conversation id into the next episode would let call
    N+1 inherit N's context, and the seeds would stop meaning anything."""
    client = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"reply": "Hello.", "conversation_id": "abc"}
    client.post.return_value = response

    agent = HTTPAgent("https://vendor.test/agent", session_key="conversation_id", client=client)
    scenario = _scenario()
    run_episode(
        scenario=scenario,
        agent=agent,
        persona="rushed_booker",
        seed=1,
        episode_index=0,
        noise=NoiseConfig(),
        simulator=ScriptedSimulator(),
    )
    boundary = len(client.post.call_args_list)
    assert "abc" not in json.dumps(client.post.call_args_list[0].kwargs["json"]), (
        "a session id was sent before the vendor issued one"
    )
    assert "abc" in json.dumps(client.post.call_args_list[1].kwargs["json"]), (
        "the vendor issued a session id and it was not echoed back within the episode"
    )

    run_episode(
        scenario=scenario,
        agent=agent,
        persona="rushed_booker",
        seed=1,
        episode_index=1,
        noise=NoiseConfig(),
        simulator=ScriptedSimulator(),
    )
    # The *opening* call of the new episode is the claim. Later calls in that
    # episode legitimately carry the id the vendor issued during it.
    opening = json.dumps(client.post.call_args_list[boundary].kwargs["json"])
    assert "abc" not in opening, "session leaked into a new episode"


def test_parse_generic_finds_the_reply_field_whatever_it_is_called() -> None:
    for key in ("reply", "text", "message", "output", "response", "answer"):
        text, calls = parse_generic({key: "We open at eight."})
        assert text == "We open at eight." and calls == ()
    assert parse_generic({"message": {"content": "Nested."}})[0] == "Nested."


def test_the_agent_only_ever_sees_delivered_text() -> None:
    """The no-leak guarantee holds across the HTTP boundary too.

    Compared turn by turn, not by substring over the payload. The scripted
    caller repeats a filler line; noise corrupts one copy and leaves another
    alone, so the clean string is then legitimately present as a *different*
    uncorrupted turn. A substring search calls that a leak and is simply wrong —
    it failed here for exactly that reason.
    """
    agent = HTTPAgent("https://vendor.test/agent", client=_transport(["Right."]))
    trajectory, _ = _run(agent, wer=0.4)
    request = AgentRequest(episode_id="e", turn_index=0, history=trajectory.agent_view(), tools=())
    body = agent._build(request, None)

    sent_user = [m["content"] for m in body["messages"] if m["role"] == "user"]
    delivered = [
        t.delivered_text
        for t in trajectory.turns
        if t.speaker is Speaker.USER and t.delivered_text.strip()
    ]
    assert sent_user == delivered, "the vendor received something other than the delivered text"

    corrupted = [
        t for t in trajectory.turns if t.speaker is Speaker.USER and t.text != t.delivered_text
    ]
    assert corrupted, "noise never fired; this test would prove nothing"
