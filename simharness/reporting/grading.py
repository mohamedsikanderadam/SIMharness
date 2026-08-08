"""The rubric: how a raw measurement becomes a 0-100 score and a letter.

Every threshold in :data:`RUBRIC_V1` is a judgement call, so all of them live
here, in one versioned object with a digest, rather than scattered through the
metric code. Two audits are comparable only if their ``rubric_digest`` matches,
and the report prints it for exactly that reason.

**Four decisions that shape the grade.**

*Scores are piecewise linear between a "good" and a "bad" anchor.* Anything
smoother implies a precision we do not have; anything coarser (banding) makes an
agent's improvement invisible until it crosses a boundary.

*Compliance caps the overall grade.* A weighted mean lets an agent that invents
prices average its way to a B on the back of fast responses and a high booking
rate. It should not. A single critical finding caps the whole audit at
:attr:`Rubric.critical_cap`, and the cap is named in the grade so the owner sees
why.

*Unavailable metrics are excluded, never zeroed.* A log without timestamps must
not score 0 for latency; it must not score at all. The alternative punishes the
business for their vendor's export format.

*Judged metrics are down-weighted.* An LLM's opinion of naturalness is worth
something, but not as much as a price that provably contradicts the price list.

The default weights are deliberately compliance-heavy. The product claim is
"audit your agent for things it got *wrong*", not "rate your agent's manners".
"""

from __future__ import annotations

import hashlib
import json

from pydantic import Field

from simharness.reporting.schemas import (
    AuditGrade,
    Category,
    CategoryScore,
    Finding,
    Metric,
    MetricBasis,
    Severity,
)
from simharness.schemas import Frozen

__all__ = [
    "GRADE_BANDS",
    "RUBRIC_V1",
    "Band",
    "Rubric",
    "grade_report",
    "letter_for",
    "score_between",
    "score_category",
]


class Band(Frozen):
    """A ``(minimum score, letter)`` pair. Bands are checked high to low."""

    minimum: float = Field(ge=0.0, le=100.0)
    letter: str
    verdict: str


GRADE_BANDS: tuple[Band, ...] = (
    Band(minimum=97.0, letter="A+", verdict="excellent"),
    Band(minimum=93.0, letter="A", verdict="excellent"),
    Band(minimum=90.0, letter="A-", verdict="excellent"),
    Band(minimum=87.0, letter="B+", verdict="good"),
    Band(minimum=83.0, letter="B", verdict="good"),
    Band(minimum=80.0, letter="B-", verdict="good"),
    Band(minimum=77.0, letter="C+", verdict="needs attention"),
    Band(minimum=73.0, letter="C", verdict="needs attention"),
    Band(minimum=70.0, letter="C-", verdict="needs attention"),
    Band(minimum=60.0, letter="D", verdict="at risk"),
    Band(minimum=0.0, letter="F", verdict="failing"),
)


class Rubric(Frozen):
    """Every threshold the grade depends on.

    ``good`` is the value at which a metric scores 100 and ``bad`` the value at
    which it scores 0; which end is better is implied by their order, so a
    metric where lower is better simply has ``good < bad``.
    """

    rubric_id: str = "simharness-audit-v1"

    # -- Quality ---------------------------------------------------------- #
    naturalness_good: float = 5.0
    naturalness_bad: float = 1.0
    """Judge score on a 1-5 rubric. 1 means the caller would know immediately."""
    repetition_good: float = 0.0
    repetition_bad: float = 0.15
    """Share of agent turns that near-duplicate an earlier one."""
    re_ask_good: float = 0.0
    re_ask_bad: float = 1.0
    """Re-asks per call. One per call is already a bad call."""
    interruption_good: float = 0.02
    interruption_bad: float = 0.20
    """Share of customer turns the agent talked over. Never zero in practice —
    barge-in handling is not free — so the "good" anchor is 2%, not 0%."""

    # -- Reliability ------------------------------------------------------ #
    latency_p95_good_ms: float = 800.0
    latency_p95_bad_ms: float = 2500.0
    """Perceived-turn-taking anchors for a voice agent: under ~0.8 s the pause
    reads as human, and by ~2.5 s callers start talking over it. These are
    *targets we chose*, not a measured psychoacoustic threshold."""
    dead_air_threshold_s: float = 3.0
    """A silence longer than this, mid-call, counts as dead air."""
    dead_air_good: float = 0.0
    dead_air_bad: float = 2.0
    """Dead-air events per call."""
    error_rate_good: float = 0.0
    error_rate_bad: float = 0.10
    empty_response_good: float = 0.0
    empty_response_bad: float = 0.05

    # -- Compliance ------------------------------------------------------- #
    hallucination_good: float = 0.0
    hallucination_bad: float = 0.05
    """Deliberately steep: one wrong price in twenty checkable claims scores 0.
    This is the metric the product exists for."""
    violation_good: float = 0.0
    violation_bad: float = 0.02
    claim_coverage_floor: float = 0.30
    """Below this share of checkable claims, compliance is reported as
    low-confidence rather than passed. A clean sheet on three claims out of a
    hundred is not a clean sheet."""

    # -- Business --------------------------------------------------------- #
    conversion_target: float = 0.30
    """Share of calls with an intent that ended converted. Scores 100 at target;
    a business can raise it once it knows its own baseline."""
    completion_good: float = 1.0
    completion_bad: float = 0.50
    abandonment_good: float = 0.0
    abandonment_bad: float = 0.20

    # -- Adherence -------------------------------------------------------- #
    instruction_good: float = 1.0
    instruction_bad: float = 0.70

    # -- Aggregation ------------------------------------------------------ #
    category_weights: dict[Category, float] = Field(
        default_factory=lambda: {
            Category.COMPLIANCE: 0.35,
            Category.ADHERENCE: 0.20,
            Category.RELIABILITY: 0.20,
            Category.BUSINESS: 0.15,
            Category.QUALITY: 0.10,
        }
    )
    judged_weight_multiplier: float = 0.5
    """Applied to any metric whose basis is ``JUDGED``."""
    critical_cap: float = 69.0
    """A single critical finding caps the overall score here — a C at best."""
    major_cap: float = 84.0
    """Three or more major findings cap the overall score here — a B at best."""
    major_cap_threshold: int = 3

    @property
    def digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


RUBRIC_V1 = Rubric()


def score_between(value: float, good: float, bad: float) -> float:
    """Map ``value`` onto 0-100, linearly, clamped at both anchors."""
    if good == bad:
        return 100.0 if value == good else 0.0
    fraction = (value - bad) / (good - bad)
    return round(max(0.0, min(1.0, fraction)) * 100.0, 2)


def score_category(
    category: Category, metrics: tuple[Metric, ...], rubric: Rubric = RUBRIC_V1
) -> CategoryScore:
    """Weight-normalised mean over the metrics that actually have a score."""
    weighted = tuple(_apply_judged_multiplier(m, rubric) for m in metrics)
    scored = [m for m in weighted if m.score is not None and m.weight > 0]
    unavailable = sum(1 for m in weighted if m.basis is MetricBasis.UNAVAILABLE)

    if not scored:
        return CategoryScore(
            category=category,
            metrics=weighted,
            score=None,
            weight=rubric.category_weights.get(category, 0.0),
            verdict="not_assessed",
            scored_metrics=0,
            unavailable_metrics=unavailable,
        )

    total_weight = sum(m.weight for m in scored)
    score = sum((m.score or 0.0) * m.weight for m in scored) / total_weight
    return CategoryScore(
        category=category,
        metrics=weighted,
        score=score,
        weight=rubric.category_weights.get(category, 0.0),
        verdict=_verdict_for(score),
        scored_metrics=len(scored),
        unavailable_metrics=unavailable,
    )


def _apply_judged_multiplier(metric: Metric, rubric: Rubric) -> Metric:
    if metric.basis is not MetricBasis.JUDGED:
        return metric
    return metric.model_copy(update={"weight": metric.weight * rubric.judged_weight_multiplier})


def letter_for(score: float) -> Band:
    for band in GRADE_BANDS:
        if score >= band.minimum:
            return band
    return GRADE_BANDS[-1]


def grade_report(
    categories: tuple[CategoryScore, ...],
    findings: tuple[Finding, ...],
    rubric: Rubric = RUBRIC_V1,
) -> AuditGrade:
    """Combine category scores into one grade, then apply the compliance caps.

    Category weights are renormalised over the categories that were assessed, so
    an audit that could not measure reliability is not silently marked down for
    it — the missing category appears in the gaps section instead.
    """
    assessed = [c for c in categories if c.score is not None and c.weight > 0]
    if not assessed:
        return AuditGrade(
            score=0.0,
            letter="N/A",
            verdict="not_assessed",
            rationale="No category could be scored from the supplied logs.",
        )

    total_weight = sum(c.weight for c in assessed)
    raw = sum((c.score or 0.0) * c.weight for c in assessed) / total_weight

    critical = sum(1 for f in findings if f.severity is Severity.CRITICAL)
    major = sum(1 for f in findings if f.severity is Severity.MAJOR)

    score = raw
    capped_by = ""
    if critical and score > rubric.critical_cap:
        score = rubric.critical_cap
        capped_by = f"{critical} critical finding{'s' if critical > 1 else ''}"
    elif major >= rubric.major_cap_threshold and score > rubric.major_cap:
        score = rubric.major_cap
        capped_by = f"{major} major findings"

    band = letter_for(score)
    return AuditGrade(
        score=round(score, 2),
        letter=band.letter,
        verdict=band.verdict,
        rationale=_rationale(raw, score, capped_by, assessed),
        capped_by=capped_by,
    )


def _rationale(raw: float, score: float, capped_by: str, assessed: list[CategoryScore]) -> str:
    weakest = min(assessed, key=lambda c: c.score or 0.0)
    parts = [
        f"Weighted across {len(assessed)} assessed categories; "
        f"weakest is {weakest.category.value} at {weakest.score:.0f}/100."
    ]
    if capped_by:
        parts.append(f"Capped from {raw:.0f} to {score:.0f} by {capped_by}.")
    return " ".join(parts)


def _verdict_for(score: float) -> str:
    if score >= 85.0:
        return "pass"
    if score >= 70.0:
        return "at_risk"
    return "fail"
