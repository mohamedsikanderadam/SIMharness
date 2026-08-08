"""Business outcomes: did the call earn anything, and did it finish.

The hard constraint here is that outcome is **read, never inferred**. A call
where the agent said "you're all booked" is not a booking; it is a claim, and
whether it became a booking lives in the business's reservation system, not in
the transcript. Inferring conversion from the agent's own closing line would
mean an agent that lies convincingly scores best on the metric the owner cares
about most — precisely the failure this product exists to expose.

So: outcomes come from :attr:`CallLog.outcome`, populated by ingest from the
source system. When the export carries none, this whole category reports
``UNAVAILABLE`` and the gaps section asks for it.

**Compliant conversion** is reported next to raw conversion. A booking taken by
promising a discount the agent had no authority to give is revenue the business
will hand back, and averaging it into a success rate hides that.
"""

from __future__ import annotations

from collections.abc import Sequence

from simharness.reporting.grading import RUBRIC_V1, Rubric, score_between
from simharness.reporting.schemas import (
    CallLog,
    CallOutcome,
    Category,
    Finding,
    FindingKind,
    Gap,
    Metric,
    MetricBasis,
    Severity,
)

__all__ = ["business_metrics"]

_CONVERTED = frozenset({CallOutcome.BOOKED, CallOutcome.QUOTED})
_COMPLETED = frozenset(
    {CallOutcome.BOOKED, CallOutcome.QUOTED, CallOutcome.ENQUIRY_ONLY}
)
"""``ENQUIRY_ONLY`` counts as completed: a caller who asked a question and got a
correct answer was served, even though nothing was sold. Grading it as a failure
would push an agent towards pressuring every caller into a booking."""

_ABANDONED = frozenset({CallOutcome.ABANDONED, CallOutcome.FAILED})


def _against_target(value: float, target: float) -> float:
    """Conversion is scored against a target rather than an ideal of 100%.

    No business converts every call, and an agent is not failing because most
    callers were asking about parking. Hitting the target scores full marks;
    exceeding it does not score more, because the point is to detect a drop.
    """
    if target <= 0:
        return 100.0
    return round(min(1.0, value / target) * 100, 2)


def business_metrics(
    logs: Sequence[CallLog],
    findings: Sequence[Finding],
    *,
    rubric: Rubric = RUBRIC_V1,
) -> tuple[tuple[Metric, ...], tuple[Finding, ...], tuple[Gap, ...]]:
    known = [log for log in logs if log.outcome is not CallOutcome.UNKNOWN]

    if not known:
        unavailable = tuple(
            Metric(
                key=key,
                label=label,
                category=Category.BUSINESS,
                basis=MetricBasis.UNAVAILABLE,
                unit="%",
                note="No call in these logs carries an outcome from the booking system.",
            )
            for key, label in (
                ("conversion_rate", "Conversion"),
                ("task_completion_rate", "Task completion"),
                ("abandonment_rate", "Calls abandoned"),
            )
        )
        gap = Gap(
            metric_key="conversion_rate",
            label="Conversion and task completion",
            reason="The logs carry no call outcome, and it must not be guessed from the words.",
            needed=(
                "Join each call to its result in the booking or CRM system and export "
                "an outcome field (booked / quoted / enquiry / abandoned / failed)."
            ),
        )
        return unavailable, (), (gap,)

    tainted = {f.call_id for f in findings if f.severity in (Severity.CRITICAL, Severity.MAJOR)}
    converted = [log for log in known if log.outcome in _CONVERTED]
    compliant = [log for log in converted if log.call_id not in tainted]

    conversion = len(converted) / len(known)
    completion = sum(1 for log in known if log.outcome in _COMPLETED) / len(known)
    abandonment = sum(1 for log in known if log.outcome in _ABANDONED) / len(known)

    abandon_findings = tuple(
        Finding(
            call_id=log.call_id,
            turn_index=log.turns[-1].index if log.turns else 0,
            kind=FindingKind.ABANDONED,
            severity=Severity.MAJOR,
            quote=log.turns[-1].text if log.turns else "(no turns)",
            explanation=(
                f"The call ended as {log.outcome.value}"
                + (f": {log.disconnect_reason}." if log.disconnect_reason else ".")
            ),
            at=log.started_at,
        )
        for log in known
        if log.outcome in _ABANDONED
    )

    metrics = (
        Metric(
            key="conversion_rate",
            label="Conversion",
            category=Category.BUSINESS,
            basis=MetricBasis.MEASURED,
            value=round(conversion * 100, 2),
            unit="%",
            score=_against_target(conversion, rubric.conversion_target),
            weight=2.0,
            sample_size=len(known),
            detail={
                "converted": len(converted),
                "target": rubric.conversion_target,
                "compliant_conversions": len(compliant),
            },
            note=(
                ""
                if len(compliant) == len(converted)
                else (
                    f"{len(converted) - len(compliant)} of {len(converted)} conversions came "
                    "from a call that also carries a compliance finding."
                )
            ),
        ),
        Metric(
            key="compliant_conversion_rate",
            label="Conversion without a compliance failure",
            category=Category.BUSINESS,
            basis=MetricBasis.MEASURED,
            value=round(len(compliant) / len(known) * 100, 2),
            unit="%",
            score=None,
            weight=0.0,
            sample_size=len(known),
            detail={"compliant_conversions": len(compliant)},
            note="Context for the conversion figure; graded through Compliance, not here.",
        ),
        Metric(
            key="task_completion_rate",
            label="Task completion",
            category=Category.BUSINESS,
            basis=MetricBasis.MEASURED,
            value=round(completion * 100, 2),
            unit="%",
            score=score_between(completion, rubric.completion_good, rubric.completion_bad),
            weight=3.0,
            sample_size=len(known),
            detail={"completed": sum(1 for log in known if log.outcome in _COMPLETED)},
        ),
        Metric(
            key="abandonment_rate",
            label="Calls abandoned",
            category=Category.BUSINESS,
            basis=MetricBasis.MEASURED,
            value=round(abandonment * 100, 2),
            unit="%",
            score=score_between(abandonment, rubric.abandonment_good, rubric.abandonment_bad),
            weight=1.0,
            sample_size=len(known),
            detail={"abandoned": len(abandon_findings)},
        ),
    )

    gaps = (
        ()
        if len(known) == len(logs)
        else (
            Gap(
                metric_key="conversion_rate",
                label="Conversion coverage",
                reason=f"Only {len(known)} of {len(logs)} calls carry an outcome.",
                needed="Export an outcome for every call, not only the ones that converted.",
            ),
        )
    )
    return metrics, abandon_findings, gaps
