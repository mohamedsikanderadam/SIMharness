"""Reliability: latency, silence, and things that outright broke.

Everything in this module depends on the export carrying a clock, and most do
not. The rule throughout is that a missing clock yields
:attr:`MetricBasis.UNAVAILABLE` and a :class:`Gap` explaining what the vendor
would have to export — never a zero, and never an estimate. An agent whose logs
are silent about timing must not be able to score full marks on timing.

The latency reported is *response* latency: the gap between the customer
finishing and the agent starting. Per-turn processing time inside the agent is a
different number, usually smaller, and is not what a caller experiences.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median

from simharness.reporting.grading import RUBRIC_V1, Rubric, score_between
from simharness.reporting.schemas import (
    CallLog,
    Category,
    Finding,
    FindingKind,
    Gap,
    LogSpeaker,
    Metric,
    MetricBasis,
    Severity,
)

__all__ = ["reliability_metrics"]


def reliability_metrics(
    logs: Sequence[CallLog], *, rubric: Rubric = RUBRIC_V1
) -> tuple[tuple[Metric, ...], tuple[Finding, ...], tuple[Gap, ...]]:
    latencies = _agent_latencies(logs)
    metrics: list[Metric] = []
    findings: list[Finding] = []
    gaps: list[Gap] = []

    latency_metric, latency_findings, latency_gap = _latency(logs, latencies, rubric)
    metrics.append(latency_metric)
    findings.extend(latency_findings)
    if latency_gap is not None:
        gaps.append(latency_gap)

    silence_metric, silence_findings, silence_gap = _dead_air(logs, rubric)
    metrics.append(silence_metric)
    findings.extend(silence_findings)
    if silence_gap is not None:
        gaps.append(silence_gap)

    error_metric, error_findings = _errors(logs, rubric)
    metrics.append(error_metric)
    findings.extend(error_findings)

    empty_metric, empty_findings = _empty_responses(logs, rubric)
    metrics.append(empty_metric)
    findings.extend(empty_findings)

    return tuple(metrics), tuple(findings), tuple(gaps)


def _agent_latencies(logs: Sequence[CallLog]) -> list[tuple[CallLog, int, float]]:
    return [
        (log, turn.index, turn.latency_ms)
        for log in logs
        for turn in log.turns
        if turn.speaker is LogSpeaker.AGENT and turn.latency_ms is not None
    ]


def _latency(
    logs: Sequence[CallLog], samples: list[tuple[CallLog, int, float]], rubric: Rubric
) -> tuple[Metric, list[Finding], Gap | None]:
    if not samples:
        return (
            Metric(
                key="latency_p95_ms",
                label="Response latency (p95)",
                category=Category.RELIABILITY,
                basis=MetricBasis.UNAVAILABLE,
                unit="ms",
                note="No turn in these logs carries a timestamp or a latency field.",
            ),
            [],
            Gap(
                metric_key="latency_p95_ms",
                label="Response latency",
                reason="The exported logs contain no per-turn timing.",
                needed=(
                    "Export absolute turn timestamps, or a per-turn latency field, "
                    "from the telephony or agent platform."
                ),
            ),
        )

    values = sorted(value for _, _, value in samples)
    p95 = _percentile(values, 0.95)
    findings = [
        Finding(
            call_id=log.call_id,
            turn_index=index,
            kind=FindingKind.SLOW_RESPONSE,
            severity=Severity.MINOR,
            quote=_quote(log, index),
            explanation=f"The agent took {value / 1000:.1f}s to reply.",
            expected=f"under {rubric.latency_p95_bad_ms / 1000:.1f}s",
        )
        for log, index, value in samples
        if value > rubric.latency_p95_bad_ms
    ]

    return (
        Metric(
            key="latency_p95_ms",
            label="Response latency (p95)",
            category=Category.RELIABILITY,
            basis=MetricBasis.MEASURED,
            value=round(p95, 1),
            unit="ms",
            score=score_between(p95, rubric.latency_p95_good_ms, rubric.latency_p95_bad_ms),
            weight=2.0,
            sample_size=len(values),
            detail={
                "median_ms": round(median(values), 1),
                "p99_ms": round(_percentile(values, 0.99), 1),
                "max_ms": round(values[-1], 1),
                "over_budget": len(findings),
            },
        ),
        findings,
        None,
    )


def _dead_air(
    logs: Sequence[CallLog], rubric: Rubric
) -> tuple[Metric, list[Finding], Gap | None]:
    """Silence between the end of one turn and the start of the next.

    Distinct from latency: latency is the agent thinking, dead air is the caller
    hearing nothing at all, including gaps the agent did not cause. Needs
    absolute timestamps *and* turn durations; latency alone cannot produce it.
    """
    timed = [log for log in logs if _has_absolute_timing(log)]
    if not timed:
        return (
            Metric(
                key="dead_air_per_call",
                label="Dead air",
                category=Category.RELIABILITY,
                basis=MetricBasis.UNAVAILABLE,
                unit="events/call",
                note="Needs turn start and end timestamps; these logs have neither.",
            ),
            [],
            Gap(
                metric_key="dead_air_per_call",
                label="Dead air",
                reason="Turn start and end timestamps are absent, so silence is not derivable.",
                needed="Export both the start and the end time of every turn.",
            ),
        )

    findings: list[Finding] = []
    for log in timed:
        previous_end = None
        for turn in log.turns:
            if previous_end is not None and turn.started_at is not None:
                gap = (turn.started_at - previous_end).total_seconds()
                if gap > rubric.dead_air_threshold_s:
                    findings.append(
                        Finding(
                            call_id=log.call_id,
                            turn_index=turn.index,
                            kind=FindingKind.DEAD_AIR,
                            severity=Severity.MINOR,
                            quote=turn.text or "(silence)",
                            explanation=f"{gap:.1f}s of silence before this turn.",
                            expected=f"under {rubric.dead_air_threshold_s:.0f}s",
                            at=turn.started_at,
                        )
                    )
            previous_end = turn.ended_at or turn.started_at

    per_call = len(findings) / len(timed)
    return (
        Metric(
            key="dead_air_per_call",
            label="Dead air",
            category=Category.RELIABILITY,
            basis=MetricBasis.MEASURED,
            value=round(per_call, 2),
            unit="events/call",
            score=score_between(per_call, rubric.dead_air_good, rubric.dead_air_bad),
            weight=1.0,
            sample_size=len(timed),
            detail={"events": len(findings), "threshold_s": rubric.dead_air_threshold_s},
            note=(
                ""
                if len(timed) == len(logs)
                else f"Measured over {len(timed)} of {len(logs)} calls; the rest lack timestamps."
            ),
        ),
        findings,
        None,
    )


def _errors(logs: Sequence[CallLog], rubric: Rubric) -> tuple[Metric, list[Finding]]:
    turns = sum(len(log.turns) for log in logs)
    findings: list[Finding] = []

    for log in logs:
        for turn in log.turns:
            if turn.error:
                findings.append(
                    Finding(
                        call_id=log.call_id,
                        turn_index=turn.index,
                        kind=FindingKind.ERROR,
                        severity=Severity.MAJOR,
                        quote=turn.text or "(no output)",
                        explanation=f"The platform reported an error: {turn.error}",
                        at=turn.started_at,
                    )
                )
            for tool in turn.tools:
                if not tool.ok:
                    findings.append(
                        Finding(
                            call_id=log.call_id,
                            turn_index=turn.index,
                            kind=FindingKind.ERROR,
                            severity=Severity.MAJOR,
                            quote=turn.text or f"(tool {tool.name})",
                            explanation=f"Tool {tool.name} failed: {tool.error or 'no detail'}.",
                            at=turn.started_at,
                        )
                    )

    rate = len(findings) / turns if turns else 0.0
    return (
        Metric(
            key="error_rate",
            label="Errors",
            category=Category.RELIABILITY,
            basis=MetricBasis.MEASURED,
            value=round(rate * 100, 2),
            unit="%",
            score=score_between(rate, rubric.error_rate_good, rubric.error_rate_bad),
            weight=2.0,
            sample_size=turns,
            detail={"errors": len(findings)},
        ),
        findings,
    )


def _empty_responses(logs: Sequence[CallLog], rubric: Rubric) -> tuple[Metric, list[Finding]]:
    """An agent turn with no words and no tool call — the caller heard nothing."""
    agent_turns = sum(len(log.agent_turns) for log in logs)
    findings = [
        Finding(
            call_id=log.call_id,
            turn_index=turn.index,
            kind=FindingKind.EMPTY_RESPONSE,
            severity=Severity.MINOR,
            quote="(no output)",
            explanation="The agent produced neither speech nor a tool call on its turn.",
            at=turn.started_at,
        )
        for log in logs
        for turn in log.agent_turns
        if not turn.text.strip() and not turn.tools
    ]

    rate = len(findings) / agent_turns if agent_turns else 0.0
    return (
        Metric(
            key="empty_response_rate",
            label="Silent turns",
            category=Category.RELIABILITY,
            basis=MetricBasis.MEASURED,
            value=round(rate * 100, 2),
            unit="%",
            score=score_between(rate, rubric.empty_response_good, rubric.empty_response_bad),
            weight=1.0,
            sample_size=agent_turns,
            detail={"silent_turns": len(findings)},
        ),
        findings,
    )


def _has_absolute_timing(log: CallLog) -> bool:
    return any(t.started_at is not None for t in log.turns) and any(
        t.ended_at is not None for t in log.turns
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Exact on small samples, where interpolation
    invents precision an audit over twelve calls does not have."""
    if not sorted_values:
        return 0.0
    rank = max(1, min(len(sorted_values), round(fraction * len(sorted_values) + 0.5)))
    return sorted_values[rank - 1]


def _quote(log: CallLog, turn_index: int) -> str:
    turn = next((t for t in log.turns if t.index == turn_index), None)
    return turn.text if turn and turn.text else "(no output)"
