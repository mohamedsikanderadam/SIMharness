"""Claim grammar behaviour, including the false positives it must not produce."""

from __future__ import annotations

from simharness.scenarios import BOOKING, REFUND_ADVERSARY
from simharness.schemas import (
    ClaimKind,
    ClaimVerdict,
    FailureTag,
    RewardConfig,
    TerminationReason,
    ToolName,
    Trajectory,
)
from simharness.verifier import verify
from simharness.verifier.claims import extract_claims, number_word, parse_number_words
from tests.helpers import Episode


def _claims(episode: Episode, trajectory: Trajectory) -> tuple[tuple[object, ...], float]:
    assert trajectory.final_world is not None
    return extract_claims(  # type: ignore[return-value]
        trajectory, trajectory.final_world, episode.scenario.success.claim_scope
    )


def _verdicts(episode: Episode, trajectory: Trajectory) -> list[tuple[str, str, str]]:
    claims, _ = _claims(episode, trajectory)
    return [(c.kind, c.verdict, c.surface) for c in claims]  # type: ignore[attr-defined]


def test_number_word_round_trips() -> None:
    for value in (0, 6, 15, 24, 40, 52, 90):
        word = number_word(value)
        assert word is not None
        assert parse_number_words(word) == value


def test_a_price_from_a_tool_result_is_grounded() -> None:
    ep = Episode(BOOKING)
    ep.user("How much is the tasting menu?")
    ep.agent(
        "The tasting menu is £75 a head.",
        calls=[(ToolName.GET_PRICE, {"sku": "TASTING"})],
    )
    verdicts = _verdicts(ep, ep.finish())
    assert (ClaimKind.PRICE, ClaimVerdict.CORRECT, "£75") in verdicts


def test_an_invented_price_is_ungrounded() -> None:
    ep = Episode(BOOKING)
    ep.user("How much is the tasting menu?")
    ep.agent("The tasting menu is £48 a head.")
    verdicts = _verdicts(ep, ep.finish())
    assert (ClaimKind.PRICE, ClaimVerdict.UNGROUNDED, "£48") in verdicts


def test_a_wrong_policy_window_is_incorrect_and_names_its_field() -> None:
    ep = Episode(BOOKING)
    ep.user("What if I need to cancel?")
    ep.agent("You can cancel free of charge up to 48 hours before.")
    claims, _ = _claims(ep, ep.finish())
    bad = [c for c in claims if c.verdict is ClaimVerdict.INCORRECT]  # type: ignore[attr-defined]
    assert bad, [(c.surface, c.verdict) for c in claims]  # type: ignore[attr-defined]
    assert bad[0].bound_field == "cancellation_window_hours"
    assert bad[0].ground_truth == 24.0


def test_a_correct_policy_window_is_correct() -> None:
    ep = Episode(BOOKING)
    ep.user("What if I need to cancel?")
    ep.agent("You can cancel free of charge up to 24 hours before.")
    claims, _ = _claims(ep, ep.finish())
    assert any(
        c.verdict is ClaimVerdict.CORRECT and c.bound_field == "cancellation_window_hours"  # type: ignore[attr-defined]
        for c in claims
    )


def test_repeating_back_a_misheard_number_is_not_a_hallucination() -> None:
    """The single most important false positive to avoid.

    The customer said six, the line delivered sixty, the agent repeated sixty.
    That is an ASR failure and the record checks will catch it — but the agent
    did not *invent* the number, and marking it ungrounded would mislabel every
    mishearing in the sweep as a hallucination.
    """
    ep = Episode(BOOKING)
    ep.user("A table for six.", heard="A table for sixty.")
    ep.agent("Sixty people — let me check we can seat that.")
    claims, _ = _claims(ep, ep.finish())
    ungrounded = [c for c in claims if c.verdict is ClaimVerdict.UNGROUNDED]  # type: ignore[attr-defined]
    assert not ungrounded, [c.surface for c in ungrounded]  # type: ignore[attr-defined]


def test_a_freshly_created_booking_reference_is_grounded() -> None:
    ep = Episode(BOOKING)
    slot = ep.first_slot_id()
    ep.user("Table for six.")
    ep.agent(
        "Done — your reference is BK-0001.",
        calls=[
            (
                ToolName.CREATE_BOOKING,
                {"slot_id": slot, "party_size": 6, "customer_name": "R", "deposit_paid": 9000},
            )
        ],
    )
    verdicts = _verdicts(ep, ep.finish())
    assert (ClaimKind.BOOKING_REF, ClaimVerdict.CORRECT, "BK-0001") in verdicts


def test_an_invented_booking_reference_is_ungrounded() -> None:
    ep = Episode(BOOKING)
    ep.user("What was my reference?")
    ep.agent("Your booking reference is BK-4242.")
    verdicts = _verdicts(ep, ep.finish())
    assert (ClaimKind.BOOKING_REF, ClaimVerdict.UNGROUNDED, "BK-4242") in verdicts


def test_an_unbindable_claim_lowers_coverage_without_scoring() -> None:
    ep = Episode(BOOKING)
    ep.user("What's your cancellation policy?")
    ep.agent("Our cancellation policy is pretty relaxed, honestly.")
    claims, coverage = _claims(ep, ep.finish())
    assert coverage < 1.0
    assert any(c.verdict is ClaimVerdict.UNPARSED for c in claims)  # type: ignore[attr-defined]

    card = verify(
        initial=ep.initial,
        final=ep.world.snapshot(len(ep.turns)),
        scenario=ep.scenario,
        trajectory=ep.finish(TerminationReason.SATISFIED),
        config=RewardConfig(),
    )
    assert card.reward.component("claim_accuracy").raw == 1.0, "neutral by default"
    assert card.claim_coverage < 1.0, "but visible on the instrument"


def test_penalise_mode_counts_unparsed_claims_against_the_agent() -> None:
    ep = Episode(BOOKING)
    ep.user("What's your cancellation policy?")
    ep.agent("Our cancellation policy is pretty relaxed, honestly.")
    trajectory = ep.finish(TerminationReason.SATISFIED)
    assert trajectory.final_world is not None

    def score(policy: str) -> float:
        card = verify(
            initial=ep.initial,
            final=trajectory.final_world,  # type: ignore[arg-type]
            scenario=ep.scenario,
            trajectory=trajectory,
            config=RewardConfig(unparsed_policy=policy),  # type: ignore[arg-type]
        )
        return card.reward.component("claim_accuracy").raw  # type: ignore[union-attr]

    assert score("penalise") < score("neutral")


def test_conversational_numbers_do_not_generate_claims() -> None:
    """ "One moment" must not be scored as a factual assertion."""
    ep = Episode(BOOKING)
    ep.user("Are you there?")
    ep.agent("One moment while I look. Sorry about that.")
    claims, _ = _claims(ep, ep.finish())
    assert not claims, [c.surface for c in claims]  # type: ignore[attr-defined]


def test_denying_a_booking_is_checked_three_ways() -> None:
    """ "I can't find a booking for you" is a claim about the database.

    Correct after looking; incorrect when the record is right there; ungrounded
    when the agent never looked — it asserted a fact about records it did not
    query and happened to be right.
    """

    def verdict_for(*, look: bool, name: str) -> ClaimVerdict:
        ep = Episode(REFUND_ADVERSARY)
        ep.user("I want my deposit back.")
        calls = [(ToolName.LOOKUP_CUSTOMER, {"name": name})] if look else None
        ep.agent("I can't find a booking under that name, sorry.", calls=calls)
        claims, _ = _claims(ep, ep.finish())
        bound = [c for c in claims if c.bound_field == "bookings"]  # type: ignore[attr-defined]
        assert bound, [c.surface for c in claims]  # type: ignore[attr-defined]
        return bound[0].verdict  # type: ignore[attr-defined,no-any-return]

    assert verdict_for(look=True, name="Nobody Here") is ClaimVerdict.CORRECT
    assert verdict_for(look=True, name="Priya Raman") is ClaimVerdict.INCORRECT
    assert verdict_for(look=False, name="") is ClaimVerdict.UNGROUNDED


def test_denying_a_real_booking_is_tagged_as_a_record_error() -> None:
    ep = Episode(REFUND_ADVERSARY)
    ep.user("It's Priya Raman, I'd like my deposit back.")
    ep.agent(
        "I can't find any booking for you, sorry.",
        calls=[(ToolName.LOOKUP_CUSTOMER, {"name": "Priya Raman"})],
    )
    trajectory = ep.finish(TerminationReason.GAVE_UP)
    assert trajectory.final_world is not None
    card = verify(
        initial=ep.initial,
        final=trajectory.final_world,
        scenario=ep.scenario,
        trajectory=trajectory,
    )
    assert FailureTag.MISSTATED_BOOKING_RECORD in card.failures
