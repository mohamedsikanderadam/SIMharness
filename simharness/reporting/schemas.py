"""The contract for the audit reporting module.

This module is deliberately separate from :mod:`simharness.schemas`. That one
describes an episode *we* ran, where the harness controlled the world and can
therefore see the tool ledger. This one describes a call the *business* ran, in
production, against their own backend — and the defining constraint is that we
usually cannot see inside it.

Three consequences are structural here rather than left to convention:

1. **Absence is a value, not a zero.** Every metric carries a
   :class:`MetricBasis`. A latency of ``None`` with basis ``UNAVAILABLE`` means
   the log had no timestamps; it must never be rendered as ``0 ms``, and it must
   never be scored. :class:`CategoryScore` recomputes its own score from only
   the metrics that were actually available.
2. **Measured and judged never merge.** A number an LLM produced carries basis
   ``JUDGED`` for the whole life of the report, so the rendered page can label
   it and the business can discount it accordingly.
3. **A finding must be quotable.** Telling an owner "your agent hallucinated
   4 times" is worthless without the sentence. :class:`Finding` therefore
   requires the utterance, and the ground truth it contradicted.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from simharness.schemas import Frozen, JSONObject, MinorUnits

# --------------------------------------------------------------------------- #
# Normalised call log
# --------------------------------------------------------------------------- #


class LogSpeaker(StrEnum):
    """Who produced a turn in a production call.

    ``CUSTOMER`` rather than ``USER``: in an audit the human is the business's
    customer, and the report is read by the business.
    """

    CUSTOMER = "customer"
    AGENT = "agent"
    TOOL = "tool"
    SYSTEM = "system"


class ToolInvocation(Frozen):
    """A tool call recovered from the log, when the vendor exports one.

    Optional throughout. Most transcript exports contain no tool records at all,
    and a report that silently assumed otherwise would grade every claim as
    ungrounded.
    """

    name: str
    arguments: JSONObject = Field(default_factory=dict)
    ok: bool = True
    result: JSONObject | None = None
    error: str | None = None


class CallTurn(Frozen):
    """One utterance in a production call.

    Every timing field is optional because a plain text transcript has none of
    them. ``interrupted`` means *this speaker was cut off by the other*, and is
    only ever set when the source log marks it explicitly — it is never inferred
    from punctuation.
    """

    index: int
    speaker: LogSpeaker
    text: str = ""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    latency_ms: float | None = None
    """Time from the end of the previous turn to the start of this one, when the
    log carries it directly. Otherwise derived in :mod:`.ingest` from timestamps,
    and left ``None`` when there are none."""
    audio_duration_s: float | None = None
    interrupted: bool = False
    tools: tuple[ToolInvocation, ...] = ()
    error: str | None = None
    metadata: JSONObject = Field(default_factory=dict)


class CallOutcome(StrEnum):
    """What the business got out of the call, as recorded by the source system.

    ``UNKNOWN`` is the honest default: most transcript exports do not say, and
    inferring conversion from the agent's own closing pleasantry is exactly the
    kind of flattery this report exists to catch.
    """

    UNKNOWN = "unknown"
    BOOKED = "booked"
    QUOTED = "quoted"
    ENQUIRY_ONLY = "enquiry_only"
    ABANDONED = "abandoned"
    TRANSFERRED_TO_HUMAN = "transferred_to_human"
    FAILED = "failed"


class CallLog(Frozen):
    """One production call, normalised. The unit of ingest and of drill-down."""

    call_id: str
    business_id: str = ""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    turns: tuple[CallTurn, ...] = ()
    outcome: CallOutcome = CallOutcome.UNKNOWN
    disconnect_reason: str = ""
    source: str = "unknown"
    """Which ingest parser produced this, e.g. ``elevenlabs`` or ``jsonl``."""
    metadata: JSONObject = Field(default_factory=dict)

    @property
    def agent_turns(self) -> tuple[CallTurn, ...]:
        return tuple(t for t in self.turns if t.speaker is LogSpeaker.AGENT)

    @property
    def customer_turns(self) -> tuple[CallTurn, ...]:
        return tuple(t for t in self.turns if t.speaker is LogSpeaker.CUSTOMER)

    @property
    def has_timing(self) -> bool:
        """True when at least one turn carries a usable latency or timestamp."""
        return any(t.latency_ms is not None or t.started_at is not None for t in self.turns)

    @property
    def has_tool_records(self) -> bool:
        """True when the source exported tool calls, which is what makes a claim
        groundable rather than merely checkable against the fact sheet."""
        return any(t.tools for t in self.turns)


# --------------------------------------------------------------------------- #
# The fact sheet the agent is audited against
# --------------------------------------------------------------------------- #


class FactSource(StrEnum):
    BUSINESS_CONFIG = "business_config"
    """Supplied by the business. Authoritative."""
    CONTEXT_DEV = "context_dev"
    """Scraped from the business's public web presence. Good enough to audit
    against, because a claim contradicting the business's own website is a
    finding either way — but flagged, because the scrape may be stale."""
    ABSENT = "absent"


class BusinessFact(Frozen):
    """One checkable public truth, plus where we got it.

    ``value`` is the display string the agent would have to match, and
    ``minor_units`` is set for money so a comparison is exact rather than a
    string match on a rounded figure.
    """

    key: str
    label: str
    value: str
    minor_units: MinorUnits | None = None
    currency: str = ""
    source: FactSource = FactSource.ABSENT
    source_detail: str = ""
    aliases: tuple[str, ...] = ()
    """Other phrasings that refer to this fact, used to decide whether the agent
    was talking about it at all."""


class FactSheet(Frozen):
    """Everything the agent's claims are checked against, with provenance.

    ``coverage`` is the share of facts that came from the business rather than a
    scrape. A report built on a thin fact sheet is a weak audit, and the page
    says so instead of implying completeness.
    """

    business_id: str
    business_name: str
    currency: str = ""
    timezone: str = ""
    facts: tuple[BusinessFact, ...] = ()
    scraped_at: datetime | None = None

    def get(self, key: str) -> BusinessFact | None:
        return next((f for f in self.facts if f.key == key), None)

    @property
    def coverage(self) -> float:
        if not self.facts:
            return 0.0
        authoritative = sum(1 for f in self.facts if f.source is FactSource.BUSINESS_CONFIG)
        return authoritative / len(self.facts)


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


class FindingKind(StrEnum):
    """What went wrong, in the vocabulary the report speaks to a business."""

    WRONG_FACT = "wrong_fact"
    """Said something that contradicts the fact sheet."""
    UNGROUNDED_CLAIM = "ungrounded_claim"
    """Asserted a specific figure or availability with nothing behind it."""
    PROMPT_VIOLATION = "prompt_violation"
    """Did something the business told it never to do."""
    REPETITION = "repetition"
    RE_ASK = "re_ask"
    """Asked again for something the customer had already given."""
    SLOW_RESPONSE = "slow_response"
    DEAD_AIR = "dead_air"
    EMPTY_RESPONSE = "empty_response"
    ERROR = "error"
    ABANDONED = "abandoned"
    UNNATURAL = "unnatural"


class Severity(StrEnum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class Finding(Frozen):
    """One thing worth showing the owner, with the evidence attached.

    ``quote`` is required and must be the agent's actual words. A finding that
    cannot be quoted cannot be argued with, and an audit the business cannot
    argue with is an audit it will not trust.
    """

    call_id: str
    turn_index: int
    kind: FindingKind
    severity: Severity
    quote: str
    explanation: str
    expected: str = ""
    fact_key: str = ""
    basis: str = "measured"
    at: datetime | None = None


# --------------------------------------------------------------------------- #
# Metrics and grading
# --------------------------------------------------------------------------- #


class MetricBasis(StrEnum):
    MEASURED = "measured"
    """Computed deterministically from the log. Reproducible."""
    JUDGED = "judged"
    """Produced by an LLM against a stated rubric. Not reproducible; labelled as
    such everywhere it is rendered."""
    UNAVAILABLE = "unavailable"
    """The log does not carry what this metric needs. Excluded from scoring
    rather than defaulted, so a silent log cannot earn marks."""


class Category(StrEnum):
    QUALITY = "quality"
    RELIABILITY = "reliability"
    COMPLIANCE = "compliance"
    BUSINESS = "business"
    ADHERENCE = "adherence"


class Metric(Frozen):
    """One measurement, its 0-100 score, and why it scored that.

    ``value`` is the raw quantity in ``unit``; ``score`` is that quantity mapped
    onto the rubric. Both are carried because the owner reads the score and the
    IT appendix reads the value.
    """

    key: str
    label: str
    category: Category
    basis: MetricBasis
    value: float | None = None
    unit: str = ""
    score: float | None = Field(default=None, ge=0.0, le=100.0)
    weight: float = Field(default=1.0, ge=0.0)
    sample_size: int = 0
    detail: JSONObject = Field(default_factory=dict)
    note: str = ""

    @model_validator(mode="after")
    def _unavailable_has_no_score(self) -> Self:
        if self.basis is MetricBasis.UNAVAILABLE and self.score is not None:
            raise ValueError(f"metric {self.key!r} is unavailable but carries a score")
        return self


class CategoryScore(Frozen):
    """A category's 0-100 score, recomputed from its own metrics.

    The score cannot drift from the parts: it is a weight-normalised mean over
    metrics that have a score, and ``None`` when none of them do. Verdict
    thresholds live in :mod:`.grading`.
    """

    category: Category
    metrics: tuple[Metric, ...]
    score: float | None = Field(default=None, ge=0.0, le=100.0)
    weight: float = Field(default=1.0, ge=0.0)
    verdict: str = "not_assessed"
    scored_metrics: int = 0
    unavailable_metrics: int = 0

    @model_validator(mode="after")
    def _score_matches_metrics(self) -> Self:
        scored = [m for m in self.metrics if m.score is not None and m.weight > 0]
        if not scored:
            if self.score is not None:
                raise ValueError(f"{self.category} has no scored metrics but carries a score")
            return self
        total = sum(m.weight for m in scored)
        expected = sum((m.score or 0.0) * m.weight for m in scored) / total
        if self.score is None or abs(self.score - expected) > 1e-6:
            raise ValueError(
                f"{self.category} score {self.score} does not match its metrics ({expected:.4f})"
            )
        return self


class AuditGrade(Frozen):
    """The single number and letter the owner will quote back at us."""

    score: float = Field(ge=0.0, le=100.0)
    letter: str
    verdict: str
    rationale: str = ""
    capped_by: str = ""
    """Set when a compliance failure capped the grade below what the weighted
    mean would have given. Without this the owner sees an A- on an agent that
    invented a price."""


class Gap(Frozen):
    """A metric the report could not produce, and what would fix it.

    Rendered as its own section. Reporting ``n/a`` honestly is the product;
    quietly dropping the row is how an audit becomes marketing.
    """

    metric_key: str
    label: str
    reason: str
    needed: str


class CallSummary(Frozen):
    """Per-call row for the IT appendix drill-down."""

    call_id: str
    started_at: datetime | None = None
    turns: int = 0
    duration_s: float | None = None
    outcome: CallOutcome = CallOutcome.UNKNOWN
    score: float | None = Field(default=None, ge=0.0, le=100.0)
    findings: int = 0
    critical_findings: int = 0


class AuditReport(Frozen):
    """The aggregated audit over a batch of calls. The renderer's only input."""

    report_id: str
    business_id: str
    business_name: str
    generated_at: datetime
    period_start: datetime | None = None
    period_end: datetime | None = None
    calls_analysed: int = 0
    turns_analysed: int = 0
    fact_sheet: FactSheet
    categories: tuple[CategoryScore, ...] = ()
    grade: AuditGrade
    findings: tuple[Finding, ...] = ()
    call_summaries: tuple[CallSummary, ...] = ()
    gaps: tuple[Gap, ...] = ()
    judge_model: str = ""
    rubric_id: str = ""
    rubric_digest: str = ""
    tool_version: str = ""

    def category(self, category: Category) -> CategoryScore | None:
        return next((c for c in self.categories if c.category is category), None)

    @property
    def critical_findings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.CRITICAL)
