"""The LLM judge: the parts of an audit no rule can measure.

Naturalness has no signal in a transcript. Neither does "the agent answered a
different question from the one asked". Those need a reader, so this module
gives the audit one — behind an explicit opt-in, because a judge makes the
report non-deterministic and costs money per call.

**Three rules constrain it, and they are the reason the judge is safe to ship.**

1. *A judged number is labelled forever.* Everything here emits
   :attr:`MetricBasis.JUDGED`, which the rubric down-weights and the rendered
   page marks. An owner can discount it; they cannot mistake it for a measurement.
2. *A judged finding never caps the grade.* Severity is clamped to ``MAJOR``.
   Capping an audit at a C on a model's opinion, with no quotable contradiction
   behind it, is not defensible to a business that disputes it — and they will.
3. *The judge is never the arbiter of fact.* It is given the fact sheet and asked
   only about claims the deterministic checks could not adjudicate. Where a rule
   has an opinion, the rule wins.

With no API key, :func:`judge_calls` returns nothing and the report renders with
naturalness in the gaps section. That is the default path.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Any, Protocol

from simharness.reporting.grading import RUBRIC_V1, Rubric, score_between
from simharness.reporting.schemas import (
    CallLog,
    Category,
    FactSheet,
    Finding,
    FindingKind,
    Metric,
    MetricBasis,
    Severity,
)
from simharness.schemas import Frozen

__all__ = ["AnthropicJudge", "CallVerdict", "Judge", "judge_calls"]

_MAX_JUDGED_SEVERITY = Severity.MAJOR

_SYSTEM_PROMPT = """You audit recorded calls handled by a business's AI phone agent.
You are given the business's published facts and one call transcript.

Score only what a rule cannot: how the call reads to a customer, and whether the
agent answered what was actually asked.

Return STRICT JSON, no prose, no markdown fence:
{
  "naturalness": <integer 1-5>,
  "naturalness_reason": "<one sentence, max 20 words>",
  "answered_the_question": <true|false>,
  "issues": [
    {"turn_index": <int>, "quote": "<the agent's exact words>",
     "problem": "<one sentence>", "severity": "minor"|"major"}
  ]
}

Naturalness rubric — score the agent, not the caller:
5  Indistinguishable from a competent human receptionist.
4  Slightly stiff or over-formal, but no caller would object.
3  Noticeably robotic: canned phrasing, awkward transitions, over-apologising.
2  Hard to talk to: ignores conversational cues, restarts topics, lectures.
1  The caller would ask for a human.

Rules:
- Quote the agent verbatim in every issue. Never paraphrase.
- Do NOT report factual errors about prices, hours or policies. Those are
  checked separately and reporting them here double-counts.
- An empty issues list is a valid and common answer. Do not invent problems.
"""


class CallVerdict(Frozen):
    """What the judge concluded about one call."""

    call_id: str
    naturalness: float
    naturalness_reason: str = ""
    answered_the_question: bool = True
    findings: tuple[Finding, ...] = ()


class Judge(Protocol):
    """One method, so a real model, a cached transcript of one, and a test stub
    are interchangeable and nothing downstream imports an SDK."""

    def review(self, log: CallLog, sheet: FactSheet) -> CallVerdict | None: ...


class AnthropicJudge:
    """Judge backed by Anthropic. Imports the SDK lazily, on first use."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        max_turns: int = 60,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._max_turns = max_turns
        self._client: Any | None = None

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def review(self, log: CallLog, sheet: FactSheet) -> CallVerdict | None:
        if not self.available:
            return None
        response = self._complete(_user_prompt(log, sheet, self._max_turns))
        parsed = _parse_verdict(response)
        if parsed is None:
            return None
        return _to_verdict(log, parsed)

    def _complete(self, prompt: str) -> str:
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self._api_key)
        message = self._client.messages.create(
            model=self.model,
            max_tokens=1500,
            temperature=0.0,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            str(getattr(block, "text", ""))
            for block in message.content
            if getattr(block, "type", "") == "text"
        )


def judge_calls(
    logs: Sequence[CallLog],
    sheet: FactSheet,
    judge: Judge | None,
    *,
    rubric: Rubric = RUBRIC_V1,
) -> tuple[tuple[Metric, ...], tuple[Finding, ...], tuple[CallVerdict, ...]]:
    """Run the judge over every call and fold the verdicts into one metric.

    A call the judge fails to parse is skipped rather than scored zero: a
    malformed model response says nothing about the agent under audit.
    """
    if judge is None:
        return (), (), ()

    verdicts = [v for v in (judge.review(log, sheet) for log in logs) if v is not None]
    if not verdicts:
        return (), (), ()

    mean = sum(v.naturalness for v in verdicts) / len(verdicts)
    answered = sum(1 for v in verdicts if v.answered_the_question) / len(verdicts)

    metrics = (
        Metric(
            key="naturalness",
            label="Naturalness",
            category=Category.QUALITY,
            basis=MetricBasis.JUDGED,
            value=round(mean, 2),
            unit="/5",
            score=score_between(mean, rubric.naturalness_good, rubric.naturalness_bad),
            weight=2.0,
            sample_size=len(verdicts),
            detail={"distribution": _distribution(verdicts)},
            note="Model-scored against a written rubric, not measured. Treat as indicative.",
        ),
        Metric(
            key="answered_the_question",
            label="Answered what was asked",
            category=Category.ADHERENCE,
            basis=MetricBasis.JUDGED,
            value=round(answered * 100, 2),
            unit="%",
            score=round(answered * 100, 2),
            weight=1.0,
            sample_size=len(verdicts),
            note="Model-scored, not measured.",
        ),
    )

    findings = tuple(f for verdict in verdicts for f in verdict.findings)
    return metrics, findings, tuple(verdicts)


def _user_prompt(log: CallLog, sheet: FactSheet, max_turns: int) -> str:
    facts = "\n".join(f"- {f.label}: {f.value}" for f in sheet.facts if f.value)
    turns = log.turns[:max_turns]
    transcript = "\n".join(f"[{t.index}] {t.speaker.value}: {t.text}" for t in turns if t.text)
    truncated = "" if len(log.turns) <= max_turns else f"\n(transcript truncated at {max_turns})"
    return (
        f"Business: {sheet.business_name}\n"
        f"Published facts:\n{facts or '- (none supplied)'}\n\n"
        f"Call {log.call_id}:\n{transcript}{truncated}"
    )


def _parse_verdict(raw: str) -> dict[str, Any] | None:
    """Pull the JSON object out of a model response.

    Models fence JSON in markdown roughly half the time regardless of
    instructions, so the fence is stripped rather than treated as a failure.
    """
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _to_verdict(log: CallLog, parsed: dict[str, Any]) -> CallVerdict:
    findings: list[Finding] = []
    for issue in parsed.get("issues", []):
        if not isinstance(issue, dict):
            continue
        quote = str(issue.get("quote", "")).strip()
        if not quote:
            continue
        findings.append(
            Finding(
                call_id=log.call_id,
                turn_index=_coerce_int(issue.get("turn_index")),
                kind=FindingKind.UNNATURAL,
                severity=_clamp_severity(issue.get("severity")),
                quote=quote,
                explanation=str(issue.get("problem", "")).strip() or "Flagged by the judge.",
                basis=MetricBasis.JUDGED.value,
            )
        )

    return CallVerdict(
        call_id=log.call_id,
        naturalness=_clamp_naturalness(parsed.get("naturalness")),
        naturalness_reason=str(parsed.get("naturalness_reason", "")).strip(),
        answered_the_question=bool(parsed.get("answered_the_question", True)),
        findings=tuple(findings),
    )


def _clamp_severity(raw: Any) -> Severity:
    """Judged findings are capped at MAJOR so a model opinion cannot cap the grade."""
    try:
        severity = Severity(str(raw).lower())
    except ValueError:
        return Severity.MINOR
    return _MAX_JUDGED_SEVERITY if severity is Severity.CRITICAL else severity


def _clamp_naturalness(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 3.0
    return max(1.0, min(5.0, value))


def _coerce_int(raw: Any) -> int:
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _distribution(verdicts: Sequence[CallVerdict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for verdict in verdicts:
        key = str(round(verdict.naturalness))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
