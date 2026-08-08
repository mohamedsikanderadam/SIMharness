"""Tests for the audit reporting module.

The tests worth having here are the ones that pin the *refusals*: that a silent
log scores nothing rather than zero, that a judge cannot cap a grade, that a
question is not a claim. Those are the properties a business would sue over, and
they are exactly the ones a well-meaning refactor quietly breaks.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

import pytest

from simharness.reporting.analyse import analyse_calls
from simharness.reporting.factsheet import build_fact_sheet
from simharness.reporting.grading import RUBRIC_V1, grade_report, letter_for, score_category
from simharness.reporting.ingest import parse_any, parse_text_transcript
from simharness.reporting.judge import CallVerdict, judge_calls
from simharness.reporting.metrics.compliance import audit_claims, compliance_metrics
from simharness.reporting.render import render_html
from simharness.reporting.schemas import (
    CallLog,
    CallOutcome,
    CallTurn,
    Category,
    CategoryScore,
    FactSheet,
    FactSource,
    Finding,
    FindingKind,
    LogSpeaker,
    Metric,
    MetricBasis,
    Severity,
    ToolInvocation,
)
from simharness.schemas import (
    BusinessConfig,
    CatalogueItem,
    OpeningHours,
    Policies,
)

NOW = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)


def business() -> BusinessConfig:
    return BusinessConfig(
        business_id="marina-bay",
        name="Marina Bay Hotel",
        timezone="Asia/Dubai",
        catalogue=(
            CatalogueItem(sku="ROOM", name="Deluxe room", unit_price=45000, currency="AED"),
        ),
        opening_hours=tuple(
            OpeningHours(weekday=day, opens=time(8, 0), closes=time(20, 0))
            for day in range(7)
        ),
        policies=Policies(
            cancellation_window_hours=48,
            deposit_required_from_party_size=1,
            deposit_per_head=15000,
            refund_window_hours=72,
            max_party_size=4,
            discount_authority=5000,
        ),
    )


def sheet() -> FactSheet:
    return build_fact_sheet(business())


def call(*turns: tuple[LogSpeaker, str], call_id: str = "c1", **kwargs: object) -> CallLog:
    return CallLog(
        call_id=call_id,
        turns=tuple(
            CallTurn(index=i, speaker=speaker, text=text)
            for i, (speaker, text) in enumerate(turns)
        ),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #


def test_text_transcript_ignores_prose_colons() -> None:
    log = parse_text_transcript(
        "Agent: Good morning.\nNote: the caller sounded annoyed\nCustomer: Hello."
    )
    assert [t.speaker for t in log.turns] == [LogSpeaker.AGENT, LogSpeaker.CUSTOMER]
    assert "sounded annoyed" in log.turns[0].text


def test_transcript_without_stamps_has_no_latency() -> None:
    log = parse_text_transcript("Customer: Hi\nAgent: Hello there")
    assert all(turn.latency_ms is None for turn in log.turns)
    assert log.has_timing is False


def test_transcript_with_stamps_derives_agent_latency() -> None:
    log = parse_text_transcript("[00:10] Customer: Hi\n[00:13] Agent: Hello there")
    assert log.turns[1].latency_ms == pytest.approx(3000.0)


def test_vendor_export_maps_roles_and_offsets() -> None:
    logs = parse_any(
        """{"conversation_id": "abc",
            "metadata": {"start_time_unix_secs": 1777000000, "call_duration_secs": 30},
            "transcript": [
              {"role": "user", "message": "Hi", "time_in_call_secs": 0,
               "duration_secs": 2},
              {"role": "agent", "message": "Hello", "time_in_call_secs": 3}
            ]}"""
    )
    log = logs[0]
    assert log.call_id == "abc"
    assert log.turns[0].speaker is LogSpeaker.CUSTOMER
    assert log.turns[1].latency_ms == pytest.approx(1000.0)


def test_unknown_vendor_keys_survive_in_metadata() -> None:
    logs = parse_any('{"conversation_id": "x", "transcript": [], "sentiment": "warm"}')
    assert logs[0].metadata["sentiment"] == "warm"


# --------------------------------------------------------------------------- #
# Claim adjudication
# --------------------------------------------------------------------------- #


def test_wrong_price_is_a_critical_finding() -> None:
    log = call((LogSpeaker.AGENT, "The deluxe room is AED 300 per night."))
    result = audit_claims([log], sheet())
    assert result.wrong == 1
    assert result.findings[0].kind is FindingKind.WRONG_FACT
    assert result.findings[0].severity is Severity.CRITICAL


def test_correct_price_in_a_different_format_is_not_a_finding() -> None:
    log = call((LogSpeaker.AGENT, "A deluxe room costs 450 dirhams."))
    result = audit_claims([log], sheet())
    assert (result.correct, result.wrong) == (1, 0)


def test_a_question_is_never_a_claim() -> None:
    log = call((LogSpeaker.AGENT, "What deposit were you quoted for the deluxe room?"))
    assert audit_claims([log], sheet()).checked == 0


def test_mentioning_a_fact_without_a_value_is_not_adjudicated() -> None:
    log = call((LogSpeaker.AGENT, "Our cancellation policy is on the website."))
    result = audit_claims([log], sheet())
    assert result.checked == 0
    assert result.mentioned_unadjudicable == 1


def test_capacity_claim_is_only_read_from_a_limit_phrase() -> None:
    """A bare number near "party size" is not a claim about the limit."""
    overstated = call((LogSpeaker.AGENT, "We can seat up to 8 guests."))
    incidental = call((LogSpeaker.AGENT, "For a party that size I have 2 rooms free."))
    assert audit_claims([overstated], sheet()).wrong == 1
    assert audit_claims([incidental], sheet()).wrong == 0


def test_wrong_cancellation_window_is_caught() -> None:
    log = call((LogSpeaker.AGENT, "You can cancel up to 24 hours before arrival."))
    assert audit_claims([log], sheet()).wrong == 1


def test_ungrounded_confirmation_needs_the_export_to_have_tools() -> None:
    """Without any tool records we cannot tell an unlogged call from an invented
    one, and accusing every agent on a basic export tier would be slander."""
    log = call((LogSpeaker.AGENT, "I've booked that for you."))
    assert audit_claims([log], sheet()).ungrounded == 0


def test_ungrounded_confirmation_is_caught_when_tools_are_exported() -> None:
    log = CallLog(
        call_id="c1",
        turns=(
            CallTurn(
                index=0,
                speaker=LogSpeaker.AGENT,
                text="One moment.",
                tools=(ToolInvocation(name="check_availability"),),
            ),
            CallTurn(index=1, speaker=LogSpeaker.AGENT, text="I've booked that for you."),
        ),
    )
    assert audit_claims([log], sheet()).ungrounded == 1


def test_no_checkable_claims_reports_unavailable_not_a_perfect_score() -> None:
    metrics, _ = compliance_metrics([call((LogSpeaker.AGENT, "Hello."))], sheet())
    hallucination = next(m for m in metrics if m.key == "hallucination_rate")
    assert hallucination.basis is MetricBasis.UNAVAILABLE
    assert hallucination.score is None


def test_denying_being_an_ai_is_a_critical_violation() -> None:
    _, findings = compliance_metrics([call((LogSpeaker.AGENT, "No, I'm a real human."))], sheet())
    assert any(
        f.kind is FindingKind.PROMPT_VIOLATION and f.severity is Severity.CRITICAL
        for f in findings
    )


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def test_category_score_cannot_drift_from_its_metrics() -> None:
    metric = Metric(
        key="m", label="m", category=Category.QUALITY, basis=MetricBasis.MEASURED, score=80.0
    )
    with pytest.raises(ValueError, match="does not match its metrics"):
        CategoryScore(category=Category.QUALITY, metrics=(metric,), score=95.0)


def test_unavailable_metrics_are_excluded_rather_than_zeroed() -> None:
    metrics = (
        Metric(
            key="a", label="a", category=Category.RELIABILITY,
            basis=MetricBasis.MEASURED, score=90.0,
        ),
        Metric(
            key="b", label="b", category=Category.RELIABILITY, basis=MetricBasis.UNAVAILABLE
        ),
    )
    scored = score_category(Category.RELIABILITY, metrics)
    assert scored.score == pytest.approx(90.0)
    assert scored.unavailable_metrics == 1


def test_a_judged_metric_is_worth_half_a_measured_one() -> None:
    measured = Metric(
        key="a", label="a", category=Category.QUALITY, basis=MetricBasis.MEASURED,
        score=100.0, weight=1.0,
    )
    judged = Metric(
        key="b", label="b", category=Category.QUALITY, basis=MetricBasis.JUDGED,
        score=0.0, weight=1.0,
    )
    scored = score_category(Category.QUALITY, (measured, judged))
    assert scored.score == pytest.approx(100 / 1.5)


def test_one_critical_finding_caps_the_grade() -> None:
    perfect = score_category(
        Category.COMPLIANCE,
        (
            Metric(
                key="a", label="a", category=Category.COMPLIANCE,
                basis=MetricBasis.MEASURED, score=100.0,
            ),
        ),
    )
    finding = Finding(
        call_id="c1", turn_index=0, kind=FindingKind.WRONG_FACT,
        severity=Severity.CRITICAL, quote="q", explanation="e",
    )
    grade = grade_report((perfect,), (finding,))
    assert grade.score == RUBRIC_V1.critical_cap
    assert "critical" in grade.capped_by


def test_grade_is_not_assessed_when_nothing_could_be_scored() -> None:
    empty = score_category(
        Category.QUALITY,
        (Metric(key="a", label="a", category=Category.QUALITY, basis=MetricBasis.UNAVAILABLE),),
    )
    assert grade_report((empty,), ()).letter == "N/A"


def test_letter_bands_are_ordered() -> None:
    assert letter_for(100.0).letter == "A+"
    assert letter_for(83.0).letter == "B"
    assert letter_for(0.0).letter == "F"


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #


class StubJudge:
    model = "stub"

    def __init__(self, verdict: CallVerdict | None) -> None:
        self._verdict = verdict

    def review(self, log: CallLog, sheet: FactSheet) -> CallVerdict | None:
        return self._verdict


def test_judged_findings_never_reach_critical() -> None:
    from simharness.reporting.judge import _to_verdict

    verdict = _to_verdict(
        call((LogSpeaker.AGENT, "hi")),
        {
            "naturalness": 2,
            "issues": [{"turn_index": 0, "quote": "hi", "problem": "curt", "severity": "critical"}],
        },
    )
    assert verdict.findings[0].severity is Severity.MAJOR


def test_no_judge_means_no_judged_metrics() -> None:
    metrics, findings, verdicts = judge_calls([call((LogSpeaker.AGENT, "hi"))], sheet(), None)
    assert (metrics, findings, verdicts) == ((), (), ())


def test_judge_naturalness_becomes_a_labelled_metric() -> None:
    judge = StubJudge(CallVerdict(call_id="c1", naturalness=4.0))
    metrics, _, _ = judge_calls([call((LogSpeaker.AGENT, "hi"))], sheet(), judge)
    naturalness = next(m for m in metrics if m.key == "naturalness")
    assert naturalness.basis is MetricBasis.JUDGED


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def timed_call(call_id: str, outcome: CallOutcome) -> CallLog:
    start = NOW
    turns = []
    for index, (speaker, text, offset, length) in enumerate(
        [
            (LogSpeaker.CUSTOMER, "Hi, what's a deluxe room?", 0.0, 2.0),
            (LogSpeaker.AGENT, "A deluxe room is AED 450 a night.", 2.4, 3.0),
            (LogSpeaker.CUSTOMER, "And the deposit?", 6.0, 1.5),
            (LogSpeaker.AGENT, "The deposit is AED 200.", 8.0, 2.0),
        ]
    ):
        started = start + timedelta(seconds=offset)
        turns.append(
            CallTurn(
                index=index,
                speaker=speaker,
                text=text,
                started_at=started,
                ended_at=started + timedelta(seconds=length),
                latency_ms=400.0 if speaker is LogSpeaker.AGENT else None,
            )
        )
    return CallLog(
        call_id=call_id,
        started_at=start,
        ended_at=start + timedelta(seconds=12),
        turns=tuple(turns),
        outcome=outcome,
    )


def test_end_to_end_audit_grades_and_renders() -> None:
    logs = [timed_call("c1", CallOutcome.BOOKED), timed_call("c2", CallOutcome.ENQUIRY_ONLY)]
    report = analyse_calls(logs, sheet(), generated_at=NOW)

    assert report.calls_analysed == 2
    assert report.critical_findings, "the wrong deposit should be caught"
    assert report.grade.score <= RUBRIC_V1.critical_cap

    page = render_html(report)
    assert "For the business owner" in page
    assert "Technical appendix" in page
    assert "window.print()" in page
    assert report.rubric_digest in page


def test_report_is_deterministic_without_a_judge() -> None:
    logs = [timed_call("c1", CallOutcome.BOOKED)]
    first = analyse_calls(logs, sheet(), generated_at=NOW)
    second = analyse_calls(logs, sheet(), generated_at=NOW)
    assert first.model_dump_json() == second.model_dump_json()


def test_untimed_logs_report_latency_as_a_gap() -> None:
    report = analyse_calls([call((LogSpeaker.AGENT, "Hello."))], sheet(), generated_at=NOW)
    latency = next(
        m
        for category in report.categories
        for m in category.metrics
        if m.key == "latency_p95_ms"
    )
    assert latency.basis is MetricBasis.UNAVAILABLE
    assert any(gap.metric_key == "latency_p95_ms" for gap in report.gaps)


def test_outcomes_are_never_inferred_from_the_transcript() -> None:
    """The agent saying "you're all booked" must not become a conversion."""
    log = call((LogSpeaker.AGENT, "You're all booked, see you Friday."))
    report = analyse_calls([log], sheet(), generated_at=NOW)
    conversion = next(
        m
        for category in report.categories
        for m in category.metrics
        if m.key == "conversion_rate"
    )
    assert conversion.basis is MetricBasis.UNAVAILABLE


def test_a_card_number_is_masked_everywhere_a_finding_can_reach() -> None:
    """The report flags the agent for reciting a card number; writing it out
    again in the artefact would repeat the offence in a file the business
    emails around."""
    finding = Finding(
        call_id="c1",
        turn_index=1,
        kind=FindingKind.PROMPT_VIOLATION,
        severity=Severity.CRITICAL,
        quote="Let me read that back: 4111 1111 1111 1111.",
        explanation="Recited a card number.",
    )
    assert finding.quote == "Let me read that back: 4111 ******** 1111."
    assert "4111 1111" not in finding.model_dump_json()


def test_masking_leaves_ordinary_numbers_alone() -> None:
    finding = Finding(
        call_id="c1",
        turn_index=1,
        kind=FindingKind.WRONG_FACT,
        severity=Severity.CRITICAL,
        quote="A deluxe room is AED 450.00 and reference 12345678 applies.",
        explanation="Wrong price.",
    )
    assert "450.00" in finding.quote
    assert "12345678" in finding.quote


def test_a_file_that_looks_like_json_but_is_broken_is_refused() -> None:
    """Falling back to the transcript parser produced a zero-turn call that was
    then graded, handing the business a letter grade for a file we could not
    read."""
    with pytest.raises(ValueError, match="does not parse"):
        parse_any('{"transcript": [', default_call_id="broken")


def test_text_with_no_speaker_lines_is_refused_rather_than_graded() -> None:
    with pytest.raises(ValueError, match="no turns found"):
        parse_any("just some prose with no speaker labels", default_call_id="prose")


def test_the_cap_banner_does_not_claim_a_critical_when_the_cap_was_majors() -> None:
    majors = [
        Finding(
            call_id=f"c{i}",
            turn_index=1,
            kind=FindingKind.PROMPT_VIOLATION,
            severity=Severity.MAJOR,
            quote="Sure thing.",
            explanation="Missing disclosure.",
        )
        for i in range(3)
    ]
    log = call((LogSpeaker.AGENT, "Sure thing."))
    report = analyse_calls([log], sheet(), generated_at=NOW).model_copy(
        update={"findings": tuple(majors)}
    )
    report = report.model_copy(
        update={"grade": grade_report(report.categories, tuple(majors), RUBRIC_V1)}
    )
    html = render_html(report)
    assert "3 major findings" in html
    assert "A single critical finding limits" not in html


class _BrokenProvider:
    """A provider that fails the way a wrong key or a rate limit fails."""

    def public_facts(self, *, business_id: str) -> dict[str, str]:
        raise RuntimeError("Context.dev error 401: API key not found")


def test_a_failing_fact_provider_degrades_instead_of_killing_the_audit() -> None:
    """A bad key used to abort the run with a traceback and no report at all,
    which is a worse outcome than an audit that checked slightly less."""
    result = build_fact_sheet(
        business(), provider=_BrokenProvider(), required_keys=("check_in_time",)
    )
    assert result.provider_error.startswith("RuntimeError: Context.dev error 401")
    assert result.scraped_at is None
    assert result.get("check_in_time") is not None
    assert result.get("check_in_time").source is FactSource.ABSENT  # type: ignore[union-attr]
    assert result.get("cancellation_window").value == "48 hours"  # type: ignore[union-attr]


def test_the_owner_is_told_when_the_scrape_failed() -> None:
    """Silently checking fewer facts reads as a clean bill of health."""
    sheet_ = build_fact_sheet(
        business(), provider=_BrokenProvider(), required_keys=("check_in_time",)
    )
    log = call((LogSpeaker.AGENT, "Good morning."))
    html_out = render_html(analyse_calls([log], sheet_, generated_at=NOW))
    assert "could not reach your website" in html_out
