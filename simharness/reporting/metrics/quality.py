"""Quality: does the call sound like a competent person handled it.

Two of the three metrics here are deterministic and one is not, and the split
matters. Repetition and re-asking are structural failures visible in the
transcript — the agent said the same thing twice, or asked for a phone number
the caller already gave. Naturalness is a matter of taste, has no signal in the
log, and is therefore the judge's job (see :mod:`simharness.reporting.judge`);
this module does not attempt a proxy for it.

Interruptions are reported only when the export marks them. Inferring barge-in
from a transcript is guesswork: a short customer turn followed by a long agent
turn looks identical whether the agent talked over them or waited politely.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

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
from simharness.reporting.text import normalise, similarity

__all__ = ["quality_metrics"]

_REPETITION_THRESHOLD = 0.90
"""Two agent turns above this similarity are the same turn said twice. Chosen
high on purpose: "Certainly, one moment." twice in a call is not a defect, and a
lower threshold turns every polite filler into a finding."""

_MIN_REPEAT_TOKENS = 6
"""Short acknowledgements are excluded entirely. "Of course." repeated is fine."""

#: What the agent asked for -> how the customer's answer looks.
_ASK_PATTERNS: dict[str, re.Pattern[str]] = {
    "phone number": re.compile(r"\b(phone|mobile|contact) number\b", re.IGNORECASE),
    "name": re.compile(r"\b(your name|who am i speaking|may i take your name)\b", re.IGNORECASE),
    "date": re.compile(r"\b(what date|which date|what day)\b", re.IGNORECASE),
    "party size": re.compile(r"\b(how many (people|guests)|party size)\b", re.IGNORECASE),
    "email": re.compile(r"\b(email address|your email)\b", re.IGNORECASE),
}


def quality_metrics(
    logs: Sequence[CallLog], *, rubric: Rubric = RUBRIC_V1
) -> tuple[tuple[Metric, ...], tuple[Finding, ...], tuple[Gap, ...]]:
    repetition_metric, repetition_findings = _repetition(logs, rubric)
    re_ask_metric, re_ask_findings = _re_asks(logs, rubric)
    interruption_metric, interruption_gap = _interruptions(logs, rubric)

    gaps = [
        Gap(
            metric_key="naturalness",
            label="Naturalness",
            reason="No signal in a transcript scores phrasing.",
            needed="Enable the LLM judge (--judge) or supply human ratings.",
        )
    ]
    if interruption_gap is not None:
        gaps.append(interruption_gap)

    return (
        (repetition_metric, re_ask_metric, interruption_metric),
        repetition_findings + re_ask_findings,
        tuple(gaps),
    )


def _repetition(logs: Sequence[CallLog], rubric: Rubric) -> tuple[Metric, tuple[Finding, ...]]:
    agent_turns = sum(len(log.agent_turns) for log in logs)
    findings: list[Finding] = []

    for log in logs:
        seen: list[tuple[int, str]] = []
        for turn in log.agent_turns:
            text = turn.text.strip()
            if len(normalise(text).split()) < _MIN_REPEAT_TOKENS:
                continue
            match = next(
                (
                    (index, previous)
                    for index, previous in seen
                    if similarity(previous, text) >= _REPETITION_THRESHOLD
                ),
                None,
            )
            if match is not None:
                findings.append(
                    Finding(
                        call_id=log.call_id,
                        turn_index=turn.index,
                        kind=FindingKind.REPETITION,
                        severity=Severity.MINOR,
                        quote=text,
                        explanation=f"Near-identical to what it said at turn {match[0]}.",
                        at=turn.started_at,
                    )
                )
            seen.append((turn.index, text))

    rate = len(findings) / agent_turns if agent_turns else 0.0
    return (
        Metric(
            key="repetition_rate",
            label="Repeated itself",
            category=Category.QUALITY,
            basis=MetricBasis.MEASURED,
            value=round(rate * 100, 2),
            unit="%",
            score=score_between(rate, rubric.repetition_good, rubric.repetition_bad),
            weight=2.0,
            sample_size=agent_turns,
            detail={"repeats": len(findings), "similarity_threshold": _REPETITION_THRESHOLD},
        ),
        tuple(findings),
    )


def _re_asks(logs: Sequence[CallLog], rubric: Rubric) -> tuple[Metric, tuple[Finding, ...]]:
    """The agent asked for something the customer had already answered.

    Detected structurally: the agent asks for X, the customer replies with
    anything substantive, and the agent asks for X again. This is the single
    most complained-about failure in voice agents and it is fully visible in a
    plain transcript, which is why it is measured rather than judged.
    """
    findings: list[Finding] = []

    for log in logs:
        answered: set[str] = set()
        asked: set[str] = set()
        for turn in log.turns:
            if turn.speaker is LogSpeaker.AGENT:
                for field, pattern in _ASK_PATTERNS.items():
                    if not pattern.search(turn.text):
                        continue
                    if field in answered:
                        findings.append(
                            Finding(
                                call_id=log.call_id,
                                turn_index=turn.index,
                                kind=FindingKind.RE_ASK,
                                severity=Severity.MAJOR,
                                quote=turn.text,
                                explanation=(
                                    f"Asked for the caller's {field} again after they "
                                    "had already given it."
                                ),
                                at=turn.started_at,
                            )
                        )
                        answered.discard(field)
                    asked.add(field)
            elif turn.speaker is LogSpeaker.CUSTOMER and turn.text.strip():
                answered |= asked
                asked = set()

    per_call = len(findings) / len(logs) if logs else 0.0
    return (
        Metric(
            key="re_ask_per_call",
            label="Asked twice for the same thing",
            category=Category.QUALITY,
            basis=MetricBasis.MEASURED,
            value=round(per_call, 2),
            unit="per call",
            score=score_between(per_call, rubric.re_ask_good, rubric.re_ask_bad),
            weight=2.0,
            sample_size=len(logs),
            detail={"re_asks": len(findings), "fields_tracked": sorted(_ASK_PATTERNS)},
            note=(
                "Counts only details the agent had already asked for. A detail the "
                "caller volunteered unprompted is not tracked, so this is a floor."
            ),
        ),
        tuple(findings),
    )


def _interruptions(logs: Sequence[CallLog], rubric: Rubric) -> tuple[Metric, Gap | None]:
    marked = [t for log in logs for t in log.turns if "interrupted" in t.model_fields_set]
    customer_turns = sum(len(log.customer_turns) for log in logs)
    interrupted = sum(1 for log in logs for t in log.customer_turns if t.interrupted)

    if not marked or not customer_turns:
        return (
            Metric(
                key="interruption_rate",
                label="Talked over the caller",
                category=Category.QUALITY,
                basis=MetricBasis.UNAVAILABLE,
                unit="%",
                note="The export does not mark barge-in, and it cannot be inferred from text.",
            ),
            Gap(
                metric_key="interruption_rate",
                label="Interruptions",
                reason="No barge-in flag in the export; overlap is invisible in a transcript.",
                needed="Export the platform's interruption/barge-in event per turn.",
            ),
        )

    rate = interrupted / customer_turns
    return (
        Metric(
            key="interruption_rate",
            label="Talked over the caller",
            category=Category.QUALITY,
            basis=MetricBasis.MEASURED,
            value=round(rate * 100, 2),
            unit="%",
            score=score_between(rate, rubric.interruption_good, rubric.interruption_bad),
            weight=1.0,
            sample_size=customer_turns,
            detail={"interruptions": interrupted},
        ),
        None,
    )
