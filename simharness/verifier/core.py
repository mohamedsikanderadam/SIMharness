"""``verify()`` — the primary score.

A pure function of (initial world, final world, scenario spec, transcript). It
takes snapshots rather than a live world, so it *cannot* mutate the thing it is
scoring even by accident, and it imports nothing from the package except
:mod:`simharness.schemas`. That import restriction is what makes "usable as an RL
reward unmodified" a property rather than a promise.
"""

from __future__ import annotations

from typing import Final

from simharness.schemas import (
    CLEAN_TERMINATIONS,
    CheckResult,
    FailureTag,
    RewardConfig,
    Scenario,
    Scorecard,
    Severity,
    Trajectory,
    WorldSnapshot,
)
from simharness.verifier.checks import (
    check_evidence,
    check_forbidden_mutations,
    check_required_records,
    check_termination,
    claim_check_result,
)
from simharness.verifier.claims import extract_claims
from simharness.verifier.reward import build_reward

VERIFIER_VERSION: Final = "1.0.0"

_BLOCKING: Final = (Severity.CRITICAL, Severity.MAJOR)


def verify(
    *,
    initial: WorldSnapshot,
    final: WorldSnapshot,
    scenario: Scenario,
    trajectory: Trajectory,
    config: RewardConfig | None = None,
) -> Scorecard:
    config = config or RewardConfig()

    claims, coverage = extract_claims(trajectory, final, scenario.success.claim_scope)
    claim_result, claim_accuracy, unparsed = claim_check_result(claims)
    if config.unparsed_policy == "penalise" and unparsed:
        # The approved default is neutral; this branch exists so the difference
        # can be measured on a training run instead of argued about.
        denominator = sum(
            1 for c in claims if c.verdict.value in {"correct", "incorrect", "ungrounded"}
        )
        correct = sum(1 for c in claims if c.verdict.value == "correct")
        claim_accuracy = correct / (denominator + unparsed) if denominator + unparsed else 1.0

    record_results, record_field_accuracy = check_required_records(final, scenario)
    forbidden_results = check_forbidden_mutations(final, scenario)
    evidence_results = check_evidence(trajectory, initial, scenario, claims)
    termination_result = check_termination(trajectory)

    checks: list[CheckResult] = [
        *record_results,
        *forbidden_results,
        claim_result,
        *evidence_results,
        termination_result,
    ]

    # "The task succeeded" means every criterion the scenario declared was met —
    # records, prohibitions and evidence alike. Reading it as records-only would
    # hand a free 1.0 to the two scenarios whose correct outcome is that nothing
    # in the world changed, which is exactly backwards.
    scenario_checks = [*record_results, *forbidden_results, *evidence_results]
    task_success = all(r.passed for r in scenario_checks)
    forbidden_hits = sum(len(_seqs(r)) for r in forbidden_results)
    field_accuracy = _partial_credit(
        scenario, record_field_accuracy, evidence_results, forbidden_results
    )
    clean = trajectory.termination is not None and trajectory.termination in CLEAN_TERMINATIONS

    reward = build_reward(
        task_success=task_success,
        field_accuracy=field_accuracy,
        forbidden_hits=forbidden_hits,
        claim_accuracy=claim_accuracy,
        clean_termination=clean,
        cost=trajectory.cost,
        config=config,
    )

    failures = tuple(dict.fromkeys(r.tag for r in checks if not r.passed and r.tag is not None))
    passed = all(r.passed for r in checks if r.severity in _BLOCKING)

    return Scorecard(
        episode_id=trajectory.episode_id,
        scenario_id=trajectory.scenario_id,
        persona_id=trajectory.persona_id,
        seeds=trajectory.seeds,
        passed=passed,
        checks=tuple(checks),
        claim_checks=claims,
        claim_coverage=coverage,
        failures=tuple(f for f in failures if isinstance(f, FailureTag)),
        termination=trajectory.termination,
        reward=reward,
        cost=trajectory.cost,
        verifier_version=VERIFIER_VERSION,
    )


def _partial_credit(
    scenario: Scenario,
    record_accuracy: float,
    evidence_results: list[CheckResult],
    forbidden_results: list[CheckResult],
) -> float:
    """Dense-ish credit for RL, pooling field matches and evidence requirements.

    A pure 0/1 task reward makes every rollout in a GRPO group identical once the
    task is hard, the group's advantage std collapses, and the update is noise.
    Pooling means the refusal scenarios — which have no required records at all —
    still hand back something with a gradient in it; where a scenario declares
    nothing but prohibitions, the prohibitions themselves become the gradient.
    """
    field_count = sum(len(r.matches) for r in scenario.success.required_records)
    evidence_count = len(evidence_results)
    total = field_count + evidence_count
    if total == 0:
        if not forbidden_results:
            return 0.0
        return sum(1 for r in forbidden_results if r.passed) / len(forbidden_results)
    earned = record_accuracy * field_count + sum(1 for r in evidence_results if r.passed)
    return earned / total


def _seqs(result: CheckResult) -> list[object]:
    seqs = result.detail.get("offending_seqs")
    return list(seqs) if isinstance(seqs, list) else []
