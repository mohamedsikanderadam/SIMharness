"""Structural checks over the world snapshots and the transcript.

Every check here reads data and returns a verdict. None of them mutate their
inputs, call a model, look at a clock, or import anything outside
:mod:`simharness.schemas`.
"""

from __future__ import annotations

import re

from simharness.schemas import (
    CLEAN_TERMINATIONS,
    CheckResult,
    ClaimCheck,
    ClaimKind,
    ClaimVerdict,
    Entity,
    EvidenceKind,
    EvidenceRequirement,
    FailureTag,
    FieldMatch,
    JSONObject,
    JSONValue,
    RequiredRecord,
    Scenario,
    Severity,
    Speaker,
    Trajectory,
    WorldSnapshot,
)
from simharness.verifier.claims import number_word

_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", re.I)
_HOUR_MERIDIEM = re.compile(r"\b(\d{1,2})\s*(am|pm)\b", re.I)


def _records(world: WorldSnapshot, entity: Entity) -> list[JSONObject]:
    state = world.state
    if entity is Entity.BOOKING:
        return [record.model_dump(mode="json") for record in state.bookings.values()]
    if entity is Entity.CUSTOMER:
        return [record.model_dump(mode="json") for record in state.customers.values()]
    if entity is Entity.QUOTE:
        return [record.model_dump(mode="json") for record in state.quotes.values()]
    return [record.model_dump(mode="json") for record in state.refunds.values()]


def _resolve(record: JSONObject, path: str) -> JSONValue:
    current: JSONValue = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _satisfies(record: JSONObject, match: FieldMatch) -> bool:
    value = _resolve(record, match.path)
    if match.equals is not None:
        return bool(value == match.equals)
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if match.at_least is not None:
        return value >= match.at_least
    return value <= (match.at_most if match.at_most is not None else 0)


def check_required_records(
    final: WorldSnapshot, scenario: Scenario
) -> tuple[list[CheckResult], float]:
    """All-or-nothing per record, plus the fraction of individual field matches
    satisfied by the best candidate — which is where the RL gradient comes from
    when the task itself is failed."""
    results: list[CheckResult] = []
    matched_fields = 0
    total_fields = 0

    for index, requirement in enumerate(scenario.success.required_records):
        candidates = _records(final, requirement.entity)
        best = 0
        full_matches = 0
        for record in candidates:
            hits = sum(_satisfies(record, m) for m in requirement.matches)
            best = max(best, hits)
            if hits == len(requirement.matches):
                full_matches += 1
        matched_fields += best
        total_fields += len(requirement.matches)
        passed = full_matches >= requirement.count
        results.append(
            CheckResult(
                check_id=f"required_record[{index}]",
                description=(
                    f"{requirement.count}x {requirement.entity} matching "
                    + ", ".join(_describe(m) for m in requirement.matches)
                ),
                passed=passed,
                severity=Severity.CRITICAL,
                detail={
                    "candidates": len(candidates),
                    "best_field_hits": best,
                    "required_fields": len(requirement.matches),
                    "full_matches": full_matches,
                },
                tag=None if passed else _tag_for(requirement),
            )
        )

    accuracy = 1.0 if total_fields == 0 else matched_fields / total_fields
    return results, accuracy


def _describe(match: FieldMatch) -> str:
    if match.equals is not None:
        return f"{match.path}=={match.equals!r}"
    if match.at_least is not None:
        return f"{match.path}>={match.at_least}"
    return f"{match.path}<={match.at_most}"


def _tag_for(requirement: RequiredRecord) -> FailureTag:
    if requirement.tag is not None:
        return requirement.tag
    paths = {m.path for m in requirement.matches}
    if "party_size" in paths:
        return FailureTag.WRONG_PARTY_SIZE
    if "deposit_paid" in paths:
        return FailureTag.SKIPPED_DEPOSIT
    return FailureTag.ABANDONED_CUSTOMER


def check_forbidden_mutations(final: WorldSnapshot, scenario: Scenario) -> list[CheckResult]:
    """Reads the ledger rather than diffing snapshots, so a create-then-revert
    cannot hide."""
    results: list[CheckResult] = []
    for index, forbidden in enumerate(scenario.success.forbidden_mutations):
        offenders = [
            record
            for record in final.state.ledger
            if record.entity is forbidden.entity
            and (forbidden.op is None or record.op is forbidden.op)
            and (forbidden.where_entity_id is None or record.entity_id == forbidden.where_entity_id)
            and _payload_matches(record.after, forbidden.where_matches)
        ]
        results.append(
            CheckResult(
                check_id=f"forbidden_mutation[{index}]",
                description=(
                    f"no {forbidden.op or 'mutation'} on {forbidden.entity}"
                    + (f" {forbidden.where_entity_id}" if forbidden.where_entity_id else "")
                ),
                passed=not offenders,
                severity=Severity.CRITICAL,
                detail={"offending_seqs": [r.seq for r in offenders]},
                tag=None if not offenders else forbidden.tag,
            )
        )
    return results


def _payload_matches(after: JSONObject | None, matches: tuple[FieldMatch, ...]) -> bool:
    """Do the written values trip every predicate on this prohibition?

    With no predicates the prohibition is about the write existing at all. With
    them it is about what was written — "a quote is fine, a quote discounted
    beyond your authority is not".
    """
    if not matches:
        return True
    if after is None:
        return False
    return all(_satisfies(after, match) for match in matches)


def _stated_times(trajectory: Trajectory) -> set[tuple[int, int]]:
    times: set[tuple[int, int]] = set()
    for turn in trajectory.turns:
        if turn.speaker is not Speaker.AGENT:
            continue
        for match in _CLOCK.finditer(turn.text):
            hour, minute = int(match.group(1)), int(match.group(2))
            meridiem = (match.group(3) or "").lower()
            times.add((hour, minute))
            if hour < 12 and meridiem != "am":
                times.add((hour + 12, minute))
        for match in _HOUR_MERIDIEM.finditer(turn.text):
            hour = int(match.group(1))
            if match.group(2).lower() == "pm" and hour < 12:
                hour += 12
            times.add((hour, 0))
    return times


def check_evidence(
    trajectory: Trajectory,
    initial: WorldSnapshot,
    scenario: Scenario,
    claims: tuple[ClaimCheck, ...],
) -> list[CheckResult]:
    """Transcript-level requirements, grounded in the diary and the tool ledger
    rather than in sentiment or keyword matching.

    "Did the agent offer an alternative" is answered by checking whether it named
    a time that exists in the calendar and is not the slot it was asked about —
    not by looking for the word "alternatively"."""
    results: list[CheckResult] = []
    stated = _stated_times(trajectory)

    for index, requirement in enumerate(scenario.success.required_evidence):
        passed, detail = _evaluate_evidence(requirement, initial, stated, claims, trajectory)
        results.append(
            CheckResult(
                check_id=f"evidence[{index}]:{requirement.kind}",
                description=str(requirement.kind),
                passed=passed,
                severity=Severity.MAJOR,
                detail=detail,
                tag=None if passed else requirement.tag,
            )
        )
    return results


def _evaluate_evidence(
    requirement: EvidenceRequirement,
    initial: WorldSnapshot,
    stated: set[tuple[int, int]],
    claims: tuple[ClaimCheck, ...],
    trajectory: Trajectory,
) -> tuple[bool, JSONObject]:
    if requirement.kind is EvidenceKind.OFFERED_ALTERNATIVE_SLOT:
        excluded = requirement.detail.get("exclude_slot_id")
        alternatives = {
            (slot.starts_at.hour, slot.starts_at.minute)
            for slot in initial.state.business.calendar
            if slot.slot_id != excluded
        }
        held = {
            (slot.starts_at.hour, slot.starts_at.minute)
            for slot in initial.state.business.calendar
            if slot.slot_id == excluded
        }
        offered = sorted(stated & (alternatives - held))
        return bool(offered), {"offered_times": [f"{h:02d}:{m:02d}" for h, m in offered]}

    if requirement.kind is EvidenceKind.CHECKED_RECORDS:
        wanted_tool = requirement.detail.get("tool")
        calls = [
            result
            for turn in trajectory.turns
            for result in turn.tool_results
            if result.ok and (wanted_tool is None or result.name.value == wanted_tool)
        ]
        return bool(calls), {"tool": wanted_tool, "successful_calls": len(calls)}

    if requirement.kind is EvidenceKind.STATED_POLICY_CORRECTLY:
        wanted_field = requirement.detail.get("field")
        correct = [
            c
            for c in claims
            if c.kind is ClaimKind.POLICY
            and c.verdict is ClaimVerdict.CORRECT
            and (wanted_field is None or c.bound_field == wanted_field)
        ]
        return bool(correct), {"correct_policy_claims": len(correct), "field": wanted_field}

    # CONFIRMED_CRITICAL_NUMBER. An agent reading a party size back says "six",
    # not "6"; matching only the digit would fail a correct confirmation, and a
    # verifier false positive is the one error this design refuses to make.
    wanted = requirement.detail.get("value")
    surfaces = [re.escape(str(wanted))]
    if isinstance(wanted, int) and not isinstance(wanted, bool):
        spelled = number_word(wanted)
        if spelled:
            surfaces.append(re.escape(spelled))
    needle = re.compile(r"\b(?:" + "|".join(surfaces) + r")\b", re.I)
    said = any(
        needle.search(turn.text) for turn in trajectory.turns if turn.speaker is Speaker.AGENT
    )
    return said, {"value": wanted, "accepted_surfaces": ", ".join(surfaces)}


def check_termination(trajectory: Trajectory) -> CheckResult:
    reason = trajectory.termination
    clean = reason is not None and reason in CLEAN_TERMINATIONS
    tag: FailureTag | None = None
    if not clean:
        tag = {
            "hung_up_angry": FailureTag.ENRAGED_CUSTOMER,
            "patience_exhausted": FailureTag.ABANDONED_CUSTOMER,
            "max_turns": FailureTag.EXCEEDED_TURN_BUDGET,
        }.get(str(reason), FailureTag.ABANDONED_CUSTOMER)
    return CheckResult(
        check_id="termination",
        description="the call ended cleanly",
        passed=clean,
        severity=Severity.MAJOR,
        detail={"reason": str(reason)},
        tag=tag,
    )


def claim_check_result(claims: tuple[ClaimCheck, ...]) -> tuple[CheckResult, float, int]:
    """Collapses the claim checks into one pass/fail plus the accuracy ratio."""
    correct = sum(c.verdict is ClaimVerdict.CORRECT for c in claims)
    incorrect = sum(c.verdict is ClaimVerdict.INCORRECT for c in claims)
    ungrounded = sum(c.verdict is ClaimVerdict.UNGROUNDED for c in claims)
    unparsed = sum(c.verdict is ClaimVerdict.UNPARSED for c in claims)
    denominator = correct + incorrect + ungrounded
    accuracy = 1.0 if denominator == 0 else correct / denominator
    passed = incorrect == 0 and ungrounded == 0
    return (
        CheckResult(
            check_id="claims",
            description="every checkable factual claim matches ground truth",
            passed=passed,
            severity=Severity.CRITICAL,
            detail={
                "correct": correct,
                "incorrect": incorrect,
                "ungrounded": ungrounded,
                "unparsed": unparsed,
            },
            tag=None if passed else _claim_tag(claims),
        ),
        accuracy,
        unparsed,
    )


def _claim_tag(claims: tuple[ClaimCheck, ...]) -> FailureTag:
    bad = [c for c in claims if c.verdict in (ClaimVerdict.INCORRECT, ClaimVerdict.UNGROUNDED)]
    kinds = {c.kind for c in bad}
    if ClaimKind.PRICE in kinds:
        return FailureTag.HALLUCINATED_PRICE
    if ClaimKind.AVAILABILITY in kinds:
        return FailureTag.HALLUCINATED_AVAILABILITY
    if ClaimKind.BOOKING_REF in kinds:
        return FailureTag.MISSTATED_BOOKING_RECORD
    return FailureTag.HALLUCINATED_POLICY
