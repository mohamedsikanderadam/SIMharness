#!/usr/bin/env python3
"""Grade a business's live agent from its call logs and write an audit webpage.

    python scripts/generate_audit_report.py \
        --logs calls/ \
        --business examples/marina_bay.json \
        --out audit/

Writes ``audit/report.html`` (open it, click "Download PDF") and
``audit/report.json`` (the same data, for a dashboard). Deterministic unless
``--judge`` is passed.

Exit codes are meant for CI: ``0`` clean, ``2`` when the audit found a critical
issue or the grade fell below ``--fail-under``. A business can wire this into a
nightly job and be told when their agent starts lying.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simharness.reporting.analyse import analyse_calls
from simharness.reporting.factsheet import ContextDevFactProvider, build_fact_sheet
from simharness.reporting.grading import RUBRIC_V1, Rubric
from simharness.reporting.ingest import load_call_logs
from simharness.reporting.judge import AnthropicJudge, Judge
from simharness.reporting.render import render_html
from simharness.reporting.schemas import AuditReport
from simharness.schemas import BusinessConfig
from simharness.world.factsheet import load_facts, world_from_facts


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logs = load_call_logs(args.logs)
    if not logs:
        print(f"No call logs found at {args.logs}", file=sys.stderr)
        return 1

    config = _business_config(args.business)
    provider = (
        ContextDevFactProvider(args.context_url, fact_check=args.fact_check)
        if args.context_url
        else None
    )
    sheet = build_fact_sheet(
        config, provider=provider, required_keys=tuple(args.require_fact or ())
    )

    report = analyse_calls(
        logs,
        sheet,
        rubric=_rubric(args.rubric),
        judge=_judge(args),
    )

    _write(report, args.out)
    _print_summary(report, args.out)
    return _exit_code(report, args.fail_under)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", required=True, help="Call log file or directory.")
    parser.add_argument(
        "--business", required=True, help="BusinessConfig JSON, or a world fact sheet JSON."
    )
    parser.add_argument("--out", default="audit", help="Output directory.")
    parser.add_argument("--rubric", help="JSON file overriding rubric thresholds.")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable the LLM judge for naturalness. Needs ANTHROPIC_API_KEY and costs tokens.",
    )
    parser.add_argument("--judge-model", default="claude-sonnet-4-5")
    parser.add_argument(
        "--context-url",
        help="Business web page. Used only to fill facts the config does not supply.",
    )
    parser.add_argument("--fact-check", action="store_true", help="Ask Context.dev to verify.")
    parser.add_argument(
        "--require-fact",
        action="append",
        help="Fact key the audit needs even if the config lacks it. Repeatable.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="Exit non-zero when the overall score is below this.",
    )
    return parser.parse_args(argv)


def _business_config(path: str) -> BusinessConfig:
    """Accept either a ``BusinessConfig`` dump or a ``world_from_facts`` sheet.

    The two formats look similar to a user and completely different to pydantic,
    and making people care which one they have is friction with no upside.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "policies" in payload and isinstance(payload.get("opening_hours"), list):
        return BusinessConfig.model_validate(payload)
    return world_from_facts(load_facts(path)).business


def _rubric(path: str | None) -> Rubric:
    if not path:
        return RUBRIC_V1
    return Rubric.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def _judge(args: argparse.Namespace) -> Judge | None:
    if not args.judge:
        return None
    judge = AnthropicJudge(model=args.judge_model)
    if not judge.available:
        print(
            "--judge was requested but ANTHROPIC_API_KEY is not set; "
            "continuing without it. Naturalness will be reported as unmeasured.",
            file=sys.stderr,
        )
        return None
    return judge


def _write(report: AuditReport, out: str) -> None:
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.html").write_text(render_html(report), encoding="utf-8")
    (directory / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8"
    )


def _print_summary(report: AuditReport, out: str) -> None:
    print(f"{report.business_name}: {report.grade.letter} ({report.grade.score:.0f}/100)")
    for category in report.categories:
        score = "not assessed" if category.score is None else f"{category.score:5.1f}"
        print(f"  {category.category.value:<12} {score}")
    print(f"  findings: {len(report.findings)} ({len(report.critical_findings)} critical)")
    print(f"  written to {Path(out) / 'report.html'}")


def _exit_code(report: AuditReport, fail_under: float) -> int:
    if report.critical_findings:
        return 2
    if fail_under and report.grade.score < fail_under:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
