"""Post-run audit reporting: grade a live agent from its own call logs.

The rest of ``simharness`` red-teams an agent in a world we control. This
package does the opposite job — it takes calls the business's agent already
handled in production, checks what it said against what the business publishes,
and grades it 0-100 per category with an overall letter.

The pipeline is four steps and each is swappable:

    ingest.load_call_logs(path)          # vendor export -> CallLog
    factsheet.build_fact_sheet(config)   # BusinessConfig (+ context.dev) -> FactSheet
    analyse.analyse_calls(logs, sheet)   # -> AuditReport
    render.render_html(report)           # -> one self-contained page

Deterministic end to end unless a judge is passed to ``analyse_calls``.
"""

from simharness.reporting.analyse import TOOL_VERSION, analyse_calls
from simharness.reporting.factsheet import build_fact_sheet
from simharness.reporting.grading import RUBRIC_V1, Rubric
from simharness.reporting.ingest import load_call_logs
from simharness.reporting.render import render_html
from simharness.reporting.schemas import AuditReport, CallLog, FactSheet

__all__ = [
    "RUBRIC_V1",
    "TOOL_VERSION",
    "AuditReport",
    "CallLog",
    "FactSheet",
    "Rubric",
    "analyse_calls",
    "build_fact_sheet",
    "load_call_logs",
    "render_html",
]
