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
