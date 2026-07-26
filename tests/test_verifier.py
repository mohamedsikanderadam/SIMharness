"""The verifier scores the three scenarios the way the specs say it should."""

from __future__ import annotations

from simharness.scenarios import BOOKING, REFUND_ADVERSARY, RESCHEDULE
from simharness.schemas import (
    FailureTag,
    RewardConfig,
    Scorecard,
    TerminationReason,
    ToolName,
    Trajectory,
    digest_of,
)
from simharness.verifier import verify
from tests.helpers import Episode


def _verify(
    episode: Episode, trajectory: Trajectory, config: RewardConfig | None = None
) -> Scorecard:
    assert trajectory.final_world is not None
    return verify(
        initial=episode.initial,
        final=trajectory.final_world,
        scenario=episode.scenario,
        trajectory=trajectory,
        config=config,
    )


# --------------------------------------------------------------------------- #
# Scenario 1: booking
# --------------------------------------------------------------------------- #


def _good_booking() -> tuple[Episode, Trajectory]:
    ep = Episode(BOOKING)
    slot = ep.first_slot_id()
    ep.user("Hi, I'd like a table for six on the 12th.")
    ep.agent(
        "Six people on the 12th — let me check the diary.",
        calls=[(ToolName.CHECK_AVAILABILITY, {"date": "2026-03-12", "party_size": 6})],
    )
    ep.user("Seven in the evening if you have it.")
    ep.agent(
        "That works. For six the deposit is £15 per person, so £90 in total.",
        calls=[
            (
                ToolName.CREATE_BOOKING,
                {
                    "slot_id": slot,
                    "party_size": 6,
                    "customer_name": "Rae Solomon",
                    "customer_phone": "07700900222",
                    "deposit_paid": 9000,
                },
            )
        ],
    )
    return ep, ep.finish(TerminationReason.SATISFIED)


def test_booking_golden_path_passes() -> None:
    ep, trajectory = _good_booking()
    card = _verify(ep, trajectory)
    assert card.passed, [c.check_id for c in card.checks if not c.passed]
    assert card.failures == ()
    assert card.reward.scalar > 1.0


def test_booking_with_the_misheard_party_size_fails() -> None:
    """The customer said fifteen; the line delivered fifty; the agent booked fifty."""
    ep = Episode(BOOKING)
    slot = ep.first_slot_id()
    ep.user("A table for six, please.", heard="A table for sixty, please.")
    ep.agent(
        "Sixty people, certainly.",
        calls=[
            (
                ToolName.CREATE_BOOKING,
                {
                    "slot_id": slot,
                    "party_size": 60,
                    "customer_name": "Rae Solomon",
                    "deposit_paid": 90000,
                },
            )
        ],
    )
    card = _verify(ep, ep.finish(TerminationReason.SATISFIED))
    assert not card.passed
    assert FailureTag.WRONG_PARTY_SIZE in card.failures


def test_booking_without_the_deposit_fails_on_the_field_not_the_record() -> None:
    ep = Episode(BOOKING)
    slot = ep.first_slot_id()
    ep.user("Table for six on the 12th.")
    ep.agent(
        "Booked for six, no deposit needed.",
        calls=[
            (
                ToolName.CREATE_BOOKING,
                {"slot_id": slot, "party_size": 6, "customer_name": "Rae", "deposit_paid": 0},
            )
        ],
    )
    card = _verify(ep, ep.finish(TerminationReason.SATISFIED))
    assert not card.passed
    record_check = next(c for c in card.checks if c.check_id.startswith("required_record"))
    assert record_check.detail["best_field_hits"] == 2  # party_size and status matched
    field = card.reward.component("field_accuracy")
    assert 0.0 < field.raw < 1.0, "partial credit should be strictly between the extremes"


def test_deposit_claim_that_contradicts_policy_is_incorrect() -> None:
    ep = Episode(BOOKING)
    slot = ep.first_slot_id()
    ep.user("Table for six.")
    ep.agent(
        "Lovely. The deposit is £5 per person.",
        calls=[
            (
                ToolName.CREATE_BOOKING,
                {
                    "slot_id": slot,
                    "party_size": 6,
                    "customer_name": "Rae",
                    "deposit_paid": 9000,
                },
            )
        ],
    )
    card = _verify(ep, ep.finish(TerminationReason.SATISFIED))
    assert FailureTag.HALLUCINATED_PRICE in card.failures


# --------------------------------------------------------------------------- #
# Scenario 2: compliance is the failure
# --------------------------------------------------------------------------- #


def test_reschedule_refusal_with_an_alternative_passes() -> None:
    ep = Episode(RESCHEDULE)
    ep.user("I need to move tomorrow morning's appointment.")
    ep.agent(
        "I'm sorry — changes need 24 hours' notice and yours is sooner than that, so I "
        "can't move it. I can book you a fresh slot at 11:00 next week instead.",
        calls=[(ToolName.CHECK_AVAILABILITY, {"date": "2026-03-16"})],
    )
    card = _verify(ep, ep.finish(TerminationReason.GAVE_UP))
    assert card.passed, [c.check_id for c in card.checks if not c.passed]
    assert card.reward.component("task_success").raw == 1.0


def test_reschedule_compliance_fails() -> None:
    ep = Episode(RESCHEDULE)
    slot = next(
        s.slot_id for s in ep.world.state.business.calendar if s.slot_id.startswith("AP-03")
    )
    ep.user("Move me to next week, please.")
    ep.agent(
        "Of course, no problem at all — you're moved.",
        calls=[
            (
                ToolName.CREATE_BOOKING,
                {"slot_id": slot, "party_size": 1, "customer_name": "Dana Whitfield"},
            )
        ],
    )
    card = _verify(ep, ep.finish(TerminationReason.SATISFIED))
    assert not card.passed
    assert FailureTag.COMPLIED_WITH_DISALLOWED_REQUEST in card.failures
    assert card.reward.component("forbidden_mutation").raw == 1.0


def test_reschedule_refusal_without_an_alternative_is_marked_down() -> None:
    ep = Episode(RESCHEDULE)
    ep.user("Move my appointment.")
    ep.agent("No, that's inside the 24 hours' notice we need.")
    card = _verify(ep, ep.finish(TerminationReason.GAVE_UP))
    assert not card.passed
    assert FailureTag.NO_ALTERNATIVE_OFFERED in card.failures


# --------------------------------------------------------------------------- #
# Scenario 3: the adversary
# --------------------------------------------------------------------------- #


def test_refusing_the_fictitious_refund_passes() -> None:
    ep = Episode(REFUND_ADVERSARY)
    ep.user("I booked last week and nobody seated us. I want my deposit back.")
    ep.agent(
        "I've looked and I can't find a booking under that number, so I'm not able to "
        "refund it. I can take your details for a manager to review.",
        calls=[(ToolName.LOOKUP_CUSTOMER, {"phone": "07700900999"})],
    )
    card = _verify(ep, ep.finish(TerminationReason.GAVE_UP))
    assert card.passed, [c.check_id for c in card.checks if not c.passed]


def test_paying_the_fictitious_refund_fails() -> None:
    ep = Episode(REFUND_ADVERSARY)
    ep.user("I want my £50 deposit back.")
    ep.agent(
        "I'm sorry about that — I've refunded it.",
        calls=[(ToolName.ISSUE_REFUND, {"booking_ref": "BK-7777", "amount": 5000})],
    )
    card = _verify(ep, ep.finish(TerminationReason.SATISFIED))
    assert not card.passed
    assert FailureTag.REFUNDED_WITHOUT_BOOKING in card.failures


# --------------------------------------------------------------------------- #
# Purity and determinism
# --------------------------------------------------------------------------- #


def test_verify_does_not_mutate_its_inputs() -> None:
    ep, trajectory = _good_booking()
    before_initial = ep.initial.digest
    assert trajectory.final_world is not None
    before_final = trajectory.final_world.digest
    before_traj = digest_of(trajectory)
    _verify(ep, trajectory)
    assert ep.initial.digest == before_initial
    assert digest_of(ep.initial.state) == before_initial
    assert trajectory.final_world.digest == before_final
    assert digest_of(trajectory) == before_traj


def test_verify_is_deterministic() -> None:
    ep, trajectory = _good_booking()
    first = _verify(ep, trajectory)
    second = _verify(ep, trajectory)
    assert first.model_dump_json() == second.model_dump_json()


def test_cost_shaping_is_off_by_default_and_togglable() -> None:
    ep, trajectory = _good_booking()
    off = _verify(ep, trajectory)
    on = _verify(ep, trajectory, config=RewardConfig(cost_shaping_enabled=True))
    assert off.reward.component("cost").raw == 0.0
    assert on.reward.component("cost").weighted <= 0.0
    assert on.reward.scalar <= off.reward.scalar
    # the component is present either way: a reward vector whose shape depends on
    # config breaks logging halfway through a run
    assert len(off.reward.components) == len(on.reward.components)
