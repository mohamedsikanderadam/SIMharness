"""Scenarios for agents whose backend we cannot see.

The verifier is one function and it does not change here — what changes is what
the scenario asks it to check. Against a black-box agent the ledger is empty, so
`required_records` and `forbidden_mutations` would pass vacuously and inflate
every score. They are therefore left empty *deliberately*, and what remains is
the half that reads the transcript:

- **claim grounding** — every price, hour and policy the agent stated, against
  the fact sheet. This is the part that carries the product.
- **evidence** — did it confirm the critical number back to the caller.
- **termination** — did the call end cleanly, or did the customer rage-quit.

`task_success` is then "no false claims and a clean ending", which is a narrower
question than "did it book the right appointment" — and saying so is the
difference between an honest scorecard and a flattering one.
"""

from __future__ import annotations

from typing import Final

from simharness.schemas import (
    ClaimKind,
    EvidenceKind,
    EvidenceRequirement,
    FailureTag,
    Scenario,
    SuccessCriteria,
    ToolName,
)

DEFAULT_CLAIM_SCOPE: Final = (
    ClaimKind.PRICE,
    ClaimKind.POLICY,
    ClaimKind.AVAILABILITY,
    ClaimKind.HOURS,
    ClaimKind.BOOKING_REF,
)


def blackbox_scenario(
    *,
    scenario_id: str,
    title: str,
    world_builder: str,
    description: str = "",
    confirm_value: int | None = None,
    max_turns: int = 14,
    world_seed: int = 0,
) -> Scenario:
    """A scenario that scores only what a transcript can prove.

    `confirm_value` adds the one behavioural check that survives without a
    backend: did the agent read the critical number — party size, appointment
    count, patient id — back to the caller. On a noisy line that single habit
    separates agents that fail safely from agents that fail silently.
    """
    evidence: tuple[EvidenceRequirement, ...] = ()
    if confirm_value is not None:
        evidence = (
            EvidenceRequirement(
                kind=EvidenceKind.CONFIRMED_CRITICAL_NUMBER,
                detail={"value": confirm_value},
                tag=FailureTag.MISHEARD_NUMBER_UNCONFIRMED,
            ),
        )
    return Scenario(
        scenario_id=scenario_id,
        title=title,
        description=description or "Black-box agent: transcript-level checks only, no world state.",
        world_builder=world_builder,
        world_seed=world_seed,
        # No tools are exposed. The vendor's agent calls the vendor's backend;
        # handing it ours would measure an integration nobody is shipping.
        enabled_tools=(ToolName.GET_PRICE,),
        max_turns=max_turns,
        agent_brief="",
        success=SuccessCriteria(
            required_records=(),
            forbidden_mutations=(),
            required_evidence=evidence,
            claim_scope=DEFAULT_CLAIM_SCOPE,
        ),
    )
