"""Scenario 1 — restaurant booking with a party-size constraint and a deposit.

Success is a confirmed booking for exactly six covers with the full deposit
taken: six covers at £15 a head, £90. The party size is the number the noise wrapper is most
likely to corrupt, which is the point — "six" surviving as "six" all the way into
``Booking.party_size`` is the thing the WER sweep measures.
"""

from typing import Final

from simharness.schemas import (
    ClaimKind,
    Entity,
    EvidenceKind,
    EvidenceRequirement,
    FailureTag,
    FieldMatch,
    ForbiddenMutation,
    MutationOp,
    RequiredRecord,
    Scenario,
    SuccessCriteria,
    ToolName,
)

PARTY_SIZE: Final = 6
DEPOSIT_TOTAL: Final = 9000  # pence: six covers at £15 a head


def multiwoz_booking_for(seed: int) -> Scenario:
    """The booking task against a real Cambridge restaurant from MultiWOZ.

    The deposit threshold and the brief are *derived from the world this seed
    builds*, because a £5-a-head curry house and a £20-a-head seafood restaurant
    cannot share a hardcoded £90 success criterion. The scenario stays plain
    frozen data — it is just data computed per seed rather than written by hand.

    This is the pattern for any real-data world: the corpus varies the facts, so
    the success criteria have to be a function of the facts, not constants.
    """
    from simharness.world.multiwoz import PRICE_BANDS, brief_for, restaurant_for

    record = restaurant_for(seed)
    _, _, deposit_per_head = PRICE_BANDS[record["pricerange"]]
    return BOOKING.model_copy(
        update={
            "scenario_id": "booking_multiwoz",
            "title": f"Table for six at {str(record['name']).title()}",
            "world_builder": "multiwoz_bistro",
            "world_seed": seed,
            "agent_brief": brief_for(seed),
            "success": BOOKING.success.model_copy(
                update={
                    "required_records": (
                        RequiredRecord(
                            entity=Entity.BOOKING,
                            matches=(
                                FieldMatch(path="party_size", equals=PARTY_SIZE),
                                FieldMatch(
                                    path="deposit_paid",
                                    at_least=deposit_per_head * PARTY_SIZE,
                                ),
                                FieldMatch(path="status", equals="confirmed"),
                            ),
                        ),
                    )
                }
            ),
        }
    )


BOOKING: Final = Scenario(
    scenario_id="booking",
    title="Table for six with a deposit",
    description=(
        "A caller wants a table for six on the 12th. The house takes a £15 deposit "
        "per head from six people upward, so a booking without one is incomplete."
    ),
    world_builder="bistro",
    world_seed=1201,
    enabled_tools=(
        ToolName.CHECK_AVAILABILITY,
        ToolName.GET_PRICE,
        ToolName.LOOKUP_CUSTOMER,
        ToolName.CREATE_BOOKING,
    ),
    max_turns=14,
    agent_brief=(
        "You take bookings for Bistro Nine. House rules: parties of six or more pay a "
        "£15 per person deposit at the time of booking; the largest party we seat is "
        "12; free cancellation up to 24 hours before. Check the diary before promising "
        "a table, and confirm the party size back to the caller before you book."
    ),
    success=SuccessCriteria(
        required_records=(
            RequiredRecord(
                entity=Entity.BOOKING,
                matches=(
                    FieldMatch(path="party_size", equals=PARTY_SIZE),
                    FieldMatch(path="deposit_paid", at_least=DEPOSIT_TOTAL),
                    FieldMatch(path="status", equals="confirmed"),
                ),
            ),
        ),
        forbidden_mutations=(
            ForbiddenMutation(
                entity=Entity.REFUND,
                op=MutationOp.CREATE,
                tag=FailureTag.REFUNDED_WITHOUT_BOOKING,
            ),
        ),
        required_evidence=(
            EvidenceRequirement(
                kind=EvidenceKind.CONFIRMED_CRITICAL_NUMBER,
                detail={"value": PARTY_SIZE},
                tag=FailureTag.MISHEARD_NUMBER_UNCONFIRMED,
            ),
        ),
        claim_scope=(
            ClaimKind.PRICE,
            ClaimKind.POLICY,
            ClaimKind.AVAILABILITY,
            ClaimKind.BOOKING_REF,
        ),
    ),
)
