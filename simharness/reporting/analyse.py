"""The one entry point: logs in, :class:`AuditReport` out.

Deliberately thin. All the judgement lives in the metric modules and the rubric;
this file only fixes the order of operations, and that order is load-bearing in
two places:

* the judge runs *after* the deterministic checks and is given the same fact
  sheet, so it can never overturn a rule that already had an answer;
* business metrics run *after* everything else, because "compliant conversion"
  needs to know which calls already carry a finding.

The result is a pure function of (logs, fact sheet, rubric) whenever the judge is
off — the same inputs produce a byte-identical report, which is what makes two
audits of the same agent comparable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime

from simharness.reporting.grading import RUBRIC_V1, Rubric, grade_report, score_category
from simharness.reporting.judge import Judge, judge_calls
from simharness.reporting.metrics import (
    DEFAULT_POLICY_RULES,
    DEFAULT_REQUIRED_BEHAVIOURS,
    PolicyRule,
    RequiredBehaviour,
    adherence_metrics,
    business_metrics,
    compliance_metrics,
    quality_metrics,
    reliability_metrics,
)
from simharness.reporting.schemas import (
    AuditReport,
    CallLog,
    CallSummary,
    Category,
    CategoryScore,
    FactSheet,
    Finding,
    Gap,
    Metric,
    Severity,
)

__all__ = ["TOOL_VERSION", "analyse_calls"]

TOOL_VERSION = "1.0.0"

_SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.MAJOR: 1,
    Severity.MINOR: 2,
    Severity.INFO: 3,
}


def analyse_calls(
    logs: Sequence[CallLog],
    sheet: FactSheet,
    *,
    rubric: Rubric = RUBRIC_V1,
    judge: Judge | None = None,
    policy_rules: Sequence[PolicyRule] = DEFAULT_POLICY_RULES,
    behaviours: Sequence[RequiredBehaviour] = DEFAULT_REQUIRED_BEHAVIOURS,
    report_id: str = "",
    generated_at: datetime | None = None,
) -> AuditReport:
    """Grade a batch of production calls against a business's published facts."""
    compliance, compliance_findings = compliance_metrics(
        logs, sheet, rules=policy_rules, rubric=rubric
    )
    reliability, reliability_findings, reliability_gaps = reliability_metrics(logs, rubric=rubric)
    quality, quality_findings, quality_gaps = quality_metrics(logs, rubric=rubric)

    findings: list[Finding] = [
        *compliance_findings,
        *reliability_findings,
        *quality_findings,
    ]

    adherence, adherence_findings = adherence_metrics(
        logs, findings, behaviours=behaviours, rubric=rubric
    )
    findings.extend(adherence_findings)

    judged_metrics, judged_findings, verdicts = judge_calls(logs, sheet, judge, rubric=rubric)
    findings.extend(judged_findings)

    business, business_findings, business_gaps = business_metrics(logs, findings, rubric=rubric)
    findings.extend(business_findings)

    quality_all = quality + _for_category(judged_metrics, Category.QUALITY)
    adherence_all = adherence + _for_category(judged_metrics, Category.ADHERENCE)
    judged_naturalness = any(m.key == "naturalness" for m in judged_metrics)

    categories: tuple[CategoryScore, ...] = (
        score_category(Category.COMPLIANCE, compliance, rubric),
        score_category(Category.ADHERENCE, adherence_all, rubric),
        score_category(Category.RELIABILITY, reliability, rubric),
        score_category(Category.BUSINESS, business, rubric),
        score_category(Category.QUALITY, quality_all, rubric),
    )

    ordered = _order_findings(findings)
    gaps = _collect_gaps(
        reliability_gaps + quality_gaps + business_gaps,
        drop={"naturalness"} if judged_naturalness else set(),
    )
    gaps += _fact_sheet_gaps(sheet)

    return AuditReport(
        report_id=report_id or _report_id(logs, sheet, rubric),
        business_id=sheet.business_id,
        business_name=sheet.business_name,
        generated_at=generated_at or datetime.now(UTC),
        period_start=_earliest(logs),
        period_end=_latest(logs),
        calls_analysed=len(logs),
        turns_analysed=sum(len(log.turns) for log in logs),
        fact_sheet=sheet,
        categories=categories,
        grade=grade_report(categories, ordered, rubric),
        findings=ordered,
        call_summaries=_summaries(logs, ordered),
        gaps=gaps,
        judge_model=_judge_model(judge) if verdicts else "",
        rubric_id=rubric.rubric_id,
        rubric_digest=rubric.digest,
        tool_version=TOOL_VERSION,
    )


def _for_category(metrics: Sequence[Metric], category: Category) -> tuple[Metric, ...]:
    return tuple(m for m in metrics if m.category is category)


def _order_findings(findings: Sequence[Finding]) -> tuple[Finding, ...]:
    """Worst first, then by call and turn, so the report's first row is the one
    the owner should read."""
    return tuple(
        sorted(findings, key=lambda f: (_SEVERITY_ORDER[f.severity], f.call_id, f.turn_index))
    )


def _collect_gaps(gaps: Sequence[Gap], *, drop: set[str]) -> tuple[Gap, ...]:
    seen: dict[str, Gap] = {}
    for gap in gaps:
        if gap.metric_key in drop or gap.metric_key in seen:
            continue
        seen[gap.metric_key] = gap
    return tuple(seen.values())


def _fact_sheet_gaps(sheet: FactSheet) -> tuple[Gap, ...]:
    """A fact we hold no truth for is a fact the agent cannot be audited on.

    Surfaced as a gap rather than passed over, because the owner reading a clean
    compliance score deserves to know which of their policies went unchecked.
    """
    absent = [f for f in sheet.facts if not f.value]
    if not absent:
        return ()
    return (
        Gap(
            metric_key="fact_sheet_coverage",
            label="Unchecked business facts",
            reason=(
                f"{len(absent)} fact(s) had no value from either your configuration or "
                f"your website: {', '.join(f.label for f in absent[:6])}."
            ),
            needed="Add these to the business configuration so claims about them can be checked.",
        ),
    )


def _summaries(logs: Sequence[CallLog], findings: Sequence[Finding]) -> tuple[CallSummary, ...]:
    by_call: dict[str, list[Finding]] = {}
    for finding in findings:
        by_call.setdefault(finding.call_id, []).append(finding)

    summaries: list[CallSummary] = []
    for log in logs:
        call_findings = by_call.get(log.call_id, [])
        summaries.append(
            CallSummary(
                call_id=log.call_id,
                started_at=log.started_at,
                turns=len(log.turns),
                duration_s=_duration(log),
                outcome=log.outcome,
                score=None,
                findings=len(call_findings),
                critical_findings=sum(
                    1 for f in call_findings if f.severity is Severity.CRITICAL
                ),
            )
        )
    return tuple(sorted(summaries, key=lambda s: (-s.critical_findings, -s.findings, s.call_id)))


def _duration(log: CallLog) -> float | None:
    if log.started_at and log.ended_at:
        return (log.ended_at - log.started_at).total_seconds()
    return None


def _earliest(logs: Sequence[CallLog]) -> datetime | None:
    stamps = [log.started_at for log in logs if log.started_at]
    return min(stamps) if stamps else None


def _latest(logs: Sequence[CallLog]) -> datetime | None:
    stamps = [log.ended_at or log.started_at for log in logs if log.ended_at or log.started_at]
    return max(s for s in stamps if s is not None) if stamps else None


def _judge_model(judge: Judge | None) -> str:
    return str(getattr(judge, "model", "")) if judge is not None else ""


def _report_id(logs: Sequence[CallLog], sheet: FactSheet, rubric: Rubric) -> str:
    """Stable across identical inputs, so re-running the tool does not look like
    a new audit of the same calls."""
    payload = "|".join(
        [sheet.business_id, rubric.digest, *sorted(log.call_id for log in logs)]
    ).encode()
    return f"audit-{hashlib.sha256(payload).hexdigest()[:12]}"
