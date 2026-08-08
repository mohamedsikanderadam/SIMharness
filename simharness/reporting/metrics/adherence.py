"""Adherence: did the agent do what the business told it to do.

Compliance asks whether the agent said anything false. Adherence asks the
narrower, more operational question the IT team actually acts on: of the
instructions in the system prompt, how many did it follow, turn by turn.

Two metrics, and the category score is the instruction-following score. There is
no third composite on top, because a composite of composites is untraceable —
when it drops, nobody can say which instruction slipped.

**Clean-turn rate** is the share of agent turns carrying no finding of a kind
that represents disobedience (a false fact, an ungrounded promise, a forbidden
behaviour, a re-ask). Deliberately per *turn*, not per call: a fifty-turn call
with one slip is not as bad as a four-turn call with one slip, and a per-call
rate would say they are identical.

**Required behaviours** is a checklist the business owns. Each entry is a thing
that must happen when a trigger occurs — state the cancellation policy when
taking a booking, offer a confirmation, identify as an automated assistant. The
default pack is small and generic on purpose; the value comes from a business
handing us their own.
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
    LogSpeaker,
    Metric,
    MetricBasis,
    Severity,
)
from simharness.schemas import Frozen

__all__ = ["DEFAULT_REQUIRED_BEHAVIOURS", "RequiredBehaviour", "adherence_metrics"]

_DISOBEDIENCE = frozenset(
    {
        FindingKind.WRONG_FACT,
        FindingKind.UNGROUNDED_CLAIM,
        FindingKind.PROMPT_VIOLATION,
        FindingKind.RE_ASK,
    }
)


class RequiredBehaviour(Frozen):
    """Something the agent must do, and the condition that makes it required.

    ``trigger`` is matched against what the caller asked for plus any turn where
    the agent committed to something; ``satisfied_by`` against the agent's turns.
    A behaviour whose trigger never fires is not counted either way — an agent is
    not marked down for failing to state a booking policy on a call where nobody
    tried to book.

    Deliberately *not* matched against the agent's whole transcript: "you can
    cancel up to 48 hours before arrival" contains the word ``cancel``, and
    triggering on it would demand the agent read back the details of a
    cancellation that was never requested.
    """

    rule_id: str
    label: str
    satisfied_by: str
    trigger: str = ""
    severity: Severity = Severity.MAJOR
    explanation: str = ""

    def triggered(self, transcript: str) -> bool:
        return not self.trigger or bool(re.search(self.trigger, transcript, re.IGNORECASE))

    def satisfied(self, agent_text: str) -> bool:
        return bool(re.search(self.satisfied_by, agent_text, re.IGNORECASE))


DEFAULT_REQUIRED_BEHAVIOURS: tuple[RequiredBehaviour, ...] = (
    RequiredBehaviour(
        rule_id="ai_disclosure",
        label="Identified itself as an automated assistant",
        satisfied_by=r"\b(virtual|automated|ai) (assistant|agent)\b|\bi(?:'m| am) an ai\b",
        severity=Severity.MAJOR,
        explanation="The caller was never told they were speaking to software.",
    ),
    RequiredBehaviour(
        rule_id="confirm_before_commit",
        label="Read the details back before committing",
        trigger=r"\b(book|reserve|cancel|refund)\b",
        satisfied_by=r"\b(just to confirm|let me confirm|so that(?:'s| is)|to recap|"
        r"can i just check)\b",
        severity=Severity.MAJOR,
        explanation="Committed to an action without reading the details back to the caller.",
    ),
    RequiredBehaviour(
        rule_id="state_cancellation_policy",
        label="Stated the cancellation policy when taking a booking",
        trigger=r"\b(book|reserve|reservation)\b",
        satisfied_by=r"\b(cancel|cancellation)\b",
        severity=Severity.MINOR,
        explanation="Took a booking without mentioning how it can be cancelled.",
    ),
    RequiredBehaviour(
        rule_id="offer_confirmation",
        label="Offered a written confirmation",
        trigger=r"\b(book|reserve|reservation)\b",
        satisfied_by=r"\b(confirmation (email|text|sms)"
        r"|send you (a|an) (email|text|confirmation))\b",
        severity=Severity.MINOR,
        explanation="Did not offer the caller anything in writing.",
    ),
)


def adherence_metrics(
    logs: Sequence[CallLog],
    findings: Sequence[Finding],
    *,
    behaviours: Sequence[RequiredBehaviour] = DEFAULT_REQUIRED_BEHAVIOURS,
    rubric: Rubric = RUBRIC_V1,
) -> tuple[tuple[Metric, ...], tuple[Finding, ...]]:
    clean_metric = _clean_turns(logs, findings, rubric)
    behaviour_metric, behaviour_findings = _required_behaviours(logs, behaviours, rubric)
    return (clean_metric, behaviour_metric), behaviour_findings


def _clean_turns(
    logs: Sequence[CallLog], findings: Sequence[Finding], rubric: Rubric
) -> Metric:
    agent_turns = sum(len(log.agent_turns) for log in logs)
    if not agent_turns:
        return Metric(
            key="clean_turn_rate",
            label="Turns that followed instructions",
            category=Category.ADHERENCE,
            basis=MetricBasis.UNAVAILABLE,
            unit="%",
            note="No agent turns in these logs.",
        )

    offending = {
        (f.call_id, f.turn_index) for f in findings if f.kind in _DISOBEDIENCE
    }
    clean = (agent_turns - len(offending)) / agent_turns
    return Metric(
        key="clean_turn_rate",
        label="Turns that followed instructions",
        category=Category.ADHERENCE,
        basis=MetricBasis.MEASURED,
        value=round(clean * 100, 2),
        unit="%",
        score=score_between(clean, rubric.instruction_good, rubric.instruction_bad),
        weight=3.0,
        sample_size=agent_turns,
        detail={
            "offending_turns": len(offending),
            "kinds_counted": sorted(k.value for k in _DISOBEDIENCE),
        },
    )


_COMMITMENT = re.compile(
    r"\b(i(?:'ve| have) (?:booked|reserved|confirmed|cancelled|refunded)"
    r"|(?:you(?:'re| are) )?(?:all )?(?:booked|confirmed)"
    r"|your (?:booking|reservation))\b",
    re.IGNORECASE,
)


def _trigger_text(log: CallLog) -> str:
    """What the caller asked for, plus any turn where the agent committed."""
    return "\n".join(
        turn.text
        for turn in log.turns
        if turn.speaker is not LogSpeaker.AGENT or _COMMITMENT.search(turn.text)
    )


def _required_behaviours(
    logs: Sequence[CallLog], behaviours: Sequence[RequiredBehaviour], rubric: Rubric
) -> tuple[Metric, tuple[Finding, ...]]:
    applicable = 0
    satisfied = 0
    findings: list[Finding] = []

    for log in logs:
        transcript = _trigger_text(log)
        agent_text = "\n".join(t.text for t in log.agent_turns)
        for behaviour in behaviours:
            if not behaviour.triggered(transcript):
                continue
            applicable += 1
            if behaviour.satisfied(agent_text):
                satisfied += 1
                continue
            opening = log.agent_turns[0] if log.agent_turns else None
            findings.append(
                Finding(
                    call_id=log.call_id,
                    turn_index=opening.index if opening else 0,
                    kind=FindingKind.PROMPT_VIOLATION,
                    severity=behaviour.severity,
                    quote=opening.text if opening else "(no agent turns)",
                    explanation=(behaviour.explanation or f"Missing: {behaviour.label}.")
                    + " This is an omission across the whole call, not a fault in the"
                    " quoted turn; the call opens as shown.",
                    expected=behaviour.label,
                    fact_key=behaviour.rule_id,
                    at=log.started_at,
                )
            )

    if not applicable:
        return (
            Metric(
                key="required_behaviour_rate",
                label="Required steps completed",
                category=Category.ADHERENCE,
                basis=MetricBasis.UNAVAILABLE,
                unit="%",
                note="None of the configured behaviours were triggered by these calls.",
            ),
            (),
        )

    rate = satisfied / applicable
    return (
        Metric(
            key="required_behaviour_rate",
            label="Required steps completed",
            category=Category.ADHERENCE,
            basis=MetricBasis.MEASURED,
            value=round(rate * 100, 2),
            unit="%",
            score=score_between(rate, rubric.instruction_good, rubric.instruction_bad),
            weight=2.0,
            sample_size=applicable,
            detail={
                "satisfied": satisfied,
                "applicable": applicable,
                "behaviours": [b.rule_id for b in behaviours],
            },
        ),
        tuple(findings),
    )

