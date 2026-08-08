"""The arbiter's safety net, rebuilt for the red-team architecture.

FIRST_PRINCIPLES.md §3 makes the `verifier` the sole arbiter of success: a
"crack" counts only when the verifier scores a claim INCORRECT or UNGROUNDED.
That puts every one of the verifier's own failure modes directly into the reward
— a false positive becomes a fake crack, and a false negative becomes a red team
that cannot find real ones.

Each case below is a bug that was live in this verifier and was caught by a test
that first looked like a broken test. They are kept because the arbiter is now
load-bearing in a way it was not before.

No scenarios, no world builders, no runner — those were removed in the pivot.
Ground truth comes from a fact sheet, per FIRST_PRINCIPLES.md §6.
"""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from typing import Any

from simharness.schemas import (
    AgentTurnView,
    ClaimKind,
    ClaimVerdict,
    CostSummary,
    EpisodeSeeds,
    RewardComponent,
    RewardConfig,
    SimulatorInternalState,
    Speaker,
    TerminationReason,
    Trajectory,
    Turn,
    WorldSnapshot,
)
from simharness.verifier.claims import extract_claims
from simharness.world.factsheet import world_from_facts

FACTS: dict[str, Any] = {
    "business_id": "arbiter-test",
    "name": "Test Clinic",
    "currency": "AED",
    "catalogue": [{"sku": "CONSULT", "name": "Consultation", "unit_price": 25000}],
    "policies": {"cancellation_window_hours": 24, "refund_window_hours": 48},
    "opening_hours": {"open": "08:00", "close": "20:00"},
    "slots": {"days": 2, "start_hour": 9, "count": 3, "capacity": 2},
}

SCOPE = (ClaimKind.PRICE, ClaimKind.POLICY, ClaimKind.AVAILABILITY, ClaimKind.BOOKING_REF)


def _claims(
    agent_says: str, caller_said: str = "", heard: str | None = None
) -> tuple[list[Any], float]:
    """Score one agent utterance against the fact sheet."""
    world = WorldSnapshot.of(world_from_facts(FACTS), 0)
    turns: list[Turn] = []
    if caller_said:
        turns.append(
            Turn(
                index=0,
                speaker=Speaker.USER,
                text=caller_said,
                delivered_text=heard if heard is not None else caller_said,
            )
        )
    turns.append(
        Turn(index=len(turns), speaker=Speaker.AGENT, text=agent_says, delivered_text=agent_says)
    )
    trajectory = Trajectory(
        episode_id="t",
        scenario_id="s",
        persona_id="p",
        seeds=EpisodeSeeds.derive(0, "s", "p", 0),
        config_digest="d",
        harness_version="0.1.0",
        created_at=datetime(2026, 3, 10, 10, 0),
        turns=turns,
        initial_world=world,
        final_world=world,
        termination=TerminationReason.SATISFIED,
        cost=CostSummary(),
    )
    checks, coverage = extract_claims(trajectory, world, SCOPE)
    return list(checks), coverage


def _verdicts(text: str, **kw: Any) -> set[ClaimVerdict]:
    return {c.verdict for c in _claims(text, **kw)[0]}


# --------------------------------------------------------------------------- #
# Regressions — each of these shipped broken at some point
# --------------------------------------------------------------------------- #


def test_currency_is_not_gbp_only() -> None:
    """The grammar recognised only £/pounds. Against a dirham price list every
    money claim fell through to the generic number bag, so a hallucinated fee
    scored CORRECT. The most dangerous class of bug there is: silent, and it
    flatters the agent."""
    assert ClaimVerdict.UNGROUNDED in _verdicts("A consultation is 900 dirhams.")
    assert ClaimVerdict.UNGROUNDED not in _verdicts("A consultation is 250 dirhams.")


def test_possessive_does_not_hide_a_sentence() -> None:
    """ "24 hours' notice" tokenised to "hours'", which matched no keyword — so
    the sentence was invisible to the verifier AND coverage read 1.0, hiding the
    blindness the metric exists to report."""
    claims, _ = _claims("Changes need 24 hours' notice.")
    assert any(c.bound_field == "cancellation_window_hours" for c in claims)


def test_a_price_does_not_ground_against_an_unrelated_window() -> None:
    """ "48" is the refund window. A single undifferentiated number bag let a
    price ground itself on it. Money now grounds only against money."""
    assert ClaimVerdict.UNGROUNDED in _verdicts("A consultation is AED 48.")


def test_a_misheard_number_repeated_back_is_not_a_hallucination() -> None:
    """The caller said six, the line delivered sixty, the agent repeated sixty.
    That is an ASR failure, not an invention — marking it ungrounded would
    mislabel every mishearing in a noise sweep as a lie."""
    assert ClaimVerdict.UNGROUNDED not in _verdicts(
        "Sixty people, then.", caller_said="Six people.", heard="Sixty people."
    )


def test_conversational_numbers_are_not_claims() -> None:
    """ "One moment" must not be scored as a factual assertion."""
    assert _claims("One moment while I look. Sorry about that.")[0] == []


def test_a_wrong_policy_window_is_incorrect_not_merely_unknown() -> None:
    """The typed layer must be able to say INCORRECT, not just 'unbound'."""
    claims, _ = _claims("You can cancel free of charge up to 48 hours before.")
    bad = [c for c in claims if c.verdict is ClaimVerdict.INCORRECT]
    assert bad and bad[0].bound_field == "cancellation_window_hours"


def test_an_acknowledgement_does_not_count_against_coverage() -> None:
    """ "Booked." tripped a keyword, carried no claim, and dragged coverage to
    0.67 on a transcript with nothing wrong in it. A metric that cries wolf on
    every confirmation is one nobody reads."""
    assert _claims("Booked.")[1] == 1.0


# --------------------------------------------------------------------------- #
# Reward integrity
# --------------------------------------------------------------------------- #


def test_the_scalar_cannot_drift_from_its_parts() -> None:
    from simharness.schemas import RewardBreakdown

    try:
        RewardBreakdown(
            components=(RewardComponent(name="x", raw=1.0, weight=1.0),),
            scalar=0.3,
            cost_shaping_enabled=False,
            config_digest="d",
        )
    except ValueError:
        return
    raise AssertionError("a reward whose scalar disagrees with its parts was accepted")


def test_reward_weights_are_data_not_constants() -> None:
    """FIRST_PRINCIPLES §7 allows adding a RewardConfig for red-team success.
    That only works if the weights are configurable and the digest records
    which set produced a number."""
    assert RewardConfig().digest != RewardConfig(w_claim_accuracy=0.9).digest


# --------------------------------------------------------------------------- #
# Knowledge separation — FIRST_PRINCIPLES §5, now a stated requirement
# --------------------------------------------------------------------------- #


def test_the_agent_view_cannot_carry_hidden_state() -> None:
    """§5: the red team must not read the client agent's internal state. The
    projection has two fields and forbids extras, so there is nowhere to put it."""
    assert set(AgentTurnView.model_fields) == {"speaker", "text"}

    secret = "injected-false-belief-do-not-reveal"
    turn = Turn(
        index=0,
        speaker=Speaker.USER,
        text="A table for six.",
        delivered_text="A table for sixty.",
        internal_state=SimulatorInternalState(patience_remaining=3, scratchpad=secret),
    )
    view = AgentTurnView.from_turn(turn)
    assert secret not in view.model_dump_json()
    assert view.text == "A table for sixty.", "the agent must hear the delivered text"


def test_the_verifier_imports_only_schemas() -> None:
    """The arbiter must stay a pure function of the transcript and the world.
    An import of a provider SDK, a clock, or the network here would put
    non-determinism inside the reward."""
    root = Path(__file__).resolve().parent.parent / "simharness" / "verifier"
    banned = {"anthropic", "openai", "httpx", "requests", "random", "time", "websockets"}
    for module in sorted(root.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module)
        assert not (names & banned), f"{module.name} imports {names & banned}"
        for name in names:
            if name.startswith("simharness"):
                assert name == "simharness.schemas" or name.startswith("simharness.verifier"), (
                    f"{module.name} imports {name}; the verifier may only import schemas"
                )
