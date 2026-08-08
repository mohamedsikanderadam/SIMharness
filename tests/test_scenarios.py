"""End-to-end runs over the fixture call logs in ``examples/scenarios``.

The unit tests in ``test_reporting`` pin one behaviour each. These pin the
*whole pipeline* on ten deliberately different call sets, which is what catches
the failures a unit test cannot see: an agent that behaves perfectly still
losing marks, a vendor envelope silently parsing to zero turns, a metric that
should be unavailable quietly scoring zero.

Each scenario asserts a grade band rather than an exact score, so a rubric
tweak does not break the suite — but a scenario changing *category* (an A agent
becoming an F) does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simharness.reporting.analyse import analyse_calls
from simharness.reporting.factsheet import build_fact_sheet
from simharness.reporting.ingest import load_call_logs
from simharness.reporting.render import render_html
from simharness.reporting.schemas import AuditReport, Metric, MetricBasis, Severity
from simharness.schemas import BusinessConfig
from simharness.world.factsheet import load_facts, world_from_facts

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCENARIOS = EXAMPLES / "scenarios"


def business() -> BusinessConfig:
    return world_from_facts(load_facts(EXAMPLES / "marina_bay.json")).business


def audit(name: str) -> AuditReport:
    logs = load_call_logs(SCENARIOS / name)
    assert logs, f"{name} produced no call logs"
    return analyse_calls(logs, build_fact_sheet(business()))


def scores(report: AuditReport) -> dict[str, float | None]:
    return {c.category.value: c.score for c in report.categories}


def metric(report: AuditReport, key: str) -> Metric:
    for category in report.categories:
        for m in category.metrics:
            if m.key == key:
                return m
    raise AssertionError(f"no metric {key!r}")


def test_every_scenario_has_a_directory() -> None:
    """The runner walks the directory, so a missing fixture must fail loudly."""
    assert len(list(SCENARIOS.iterdir())) == 10


@pytest.mark.parametrize("name", sorted(p.name for p in SCENARIOS.iterdir()))
def test_scenario_renders_without_error(name: str) -> None:
    report = audit(name)
    html = render_html(report)
    assert "For the business owner" in html
    assert "Technical appendix" in html
    assert "http://" not in html and "https://" not in html
    json.dumps(report.model_dump(mode="json"))


def test_clean_agent_is_not_marked_down() -> None:
    """The one that matters most: a correct agent must score clean.

    A grader that finds fault everywhere is useless to a business, because the
    first thing they will do is check it against a call they know went well.
    """
    report = audit("01_clean_agent")
    assert report.grade.letter.startswith("A")
    assert report.findings == ()


def test_hallucinating_agent_fails_on_facts_alone() -> None:
    report = audit("02_hallucinating_agent")
    assert report.grade.letter in {"D", "F"}
    critical = [f for f in report.findings if f.severity is Severity.CRITICAL]
    assert len(critical) == 3
    assert {f.fact_key for f in critical} == {
        "price:DELUXE",
        "cancellation_window",
        "max_party_size",
    }
    assert scores(report)["reliability"] == 100.0


def test_slow_agent_loses_reliability_but_keeps_compliance() -> None:
    report = audit("03_slow_agent")
    assert scores(report)["compliance"] == 100.0
    assert (scores(report)["reliability"] or 100.0) < 70.0
    assert metric(report, "latency_p95_ms").value == pytest.approx(11200.0, rel=0.05)


def test_repetitive_agent_loses_quality() -> None:
    report = audit("04_repetitive_agent")
    assert (scores(report)["quality"] or 100.0) < 50.0
    # One, not two: the caller volunteered their name before being asked, and the
    # detector only counts details the agent had already asked for.
    assert metric(report, "re_ask_per_call").value == 1.0
    assert scores(report)["compliance"] == 100.0


def test_plain_transcript_leaves_timing_and_outcome_unmeasured() -> None:
    report = audit("05_plain_transcript")
    for key in ("latency_p95_ms", "dead_air_per_call", "conversion_rate"):
        assert metric(report, key).basis is MetricBasis.UNAVAILABLE
        assert metric(report, key).score is None
    assert scores(report)["business"] is None
    assert {g.metric_key for g in report.gaps} >= {"latency_p95_ms", "conversion_rate"}


def test_missing_outcomes_do_not_become_a_zero_conversion() -> None:
    report = audit("06_no_outcomes")
    assert metric(report, "conversion_rate").basis is MetricBasis.UNAVAILABLE
    assert scores(report)["business"] is None
    assert report.grade.letter.startswith("A")


def test_vendor_envelope_is_unwrapped() -> None:
    """``{"data": {...}}`` must not parse to a call with zero turns."""
    report = audit("07_elevenlabs_export")
    assert report.turns_analysed == 6
    assert report.call_summaries[0].call_id == "el-7788"
    assert metric(report, "latency_p95_ms").basis is MetricBasis.MEASURED


def test_nightly_batch_aggregates_every_call() -> None:
    report = audit("08_nightly_batch")
    assert report.calls_analysed == 8
    assert metric(report, "conversion_rate").value == pytest.approx(37.5)
    assert len(report.call_summaries) == 8


def test_policy_violations_are_critical_even_when_facts_are_right() -> None:
    report = audit("09_policy_violations")
    kinds = {f.fact_key for f in report.findings if f.severity is Severity.CRITICAL}
    assert kinds == {"claims_to_be_human", "personal_data_readback"}
    assert scores(report)["compliance"] == 0.0
    assert report.grade.letter == "F"


def test_ungrounded_confirmation_needs_tool_records_to_be_flagged() -> None:
    report = audit("10_ungrounded_confirmations")
    kinds = [f.kind.value for f in report.findings]
    assert "ungrounded_claim" in kinds
    assert "error" in kinds
