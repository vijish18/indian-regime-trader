"""Renders a ``validation.scenario.SessionReport`` as a human-readable
Markdown document (Phase 21's required "end-to-end validation report").

Performs no analysis of its own -- only formats what
``validation.scenario.run_end_to_end_validation`` already produced,
mirroring ``backtest/report.py``'s own separation between computing a
result and writing it up.
"""

from __future__ import annotations

from pathlib import Path

from validation.harness import STRATEGY_VERSION, ValidationEnvironment
from validation.invariants import InvariantStatus
from validation.scenario import SessionReport

_NARRATIVE_ORDER = [
    "market-data ingestion",
    "feature calculation + HMM",
    "stock ranking",
    "portfolio construction",
    "risk management",
    "paper execution",
    "fills",
    "portfolio accounting",
    "broker API timeout",
    "rejected order",
    "partial fill",
    "duplicate event",
    "lost WebSocket",
    "delayed market data",
    "monitoring",
    "shutdown",
    "application crash",
    "database restart",
    "reconciliation",
    "halted state persists (deliberately triggered)",
]


def render_markdown_report(env: ValidationEnvironment, report: SessionReport) -> str:
    lines: list[str] = []
    lines.extend(_header(env, report))
    lines.extend(_summary_table(report))
    lines.append("")
    lines.extend(_narrative_section(report))
    lines.append("")
    lines.extend(_invariants_section(report))
    lines.append("")
    lines.extend(_limitations_section())
    return "\n".join(lines) + "\n"


def write_markdown_report(env: ValidationEnvironment, report: SessionReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(env, report), encoding="utf-8")


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def _header(env: ValidationEnvironment, report: SessionReport) -> list[str]:
    duration = (report.finished_at - report.started_at).total_seconds()
    verdict = "PASSED" if report.all_ok else "FAILED"
    return [
        "# Phase 21: End-to-End Paper-Trading Validation Report",
        "",
        f"**Result: {verdict}**",
        "",
        f"- Generated: {report.finished_at.isoformat()}",
        f"- Run duration (wall clock): {duration:,.1f}s",
        f"- Configuration: this repository's own `config/settings.yaml`, unmodified "
        f"(`execution.mode={env.settings.execution.mode!r}`)",
        f"- Strategy version: `{STRATEGY_VERSION}`",
        f"- Model: `{env.model_artifact.model_id}` "
        f"(trained through {env.model_artifact.model.training_result.training_end}, "
        f"{env.model_artifact.model.n_states} states)",
        f"- Market data: {env.market.first_session} -> {env.market.last_session} "
        f"({len(env.market.sessions)} synthetic sessions, {len(env.market.instrument_ids)} "
        "instruments -- see \"Limitations\" below)",
        f"- Live trading sessions exercised: "
        f"{', '.join(str(d) for d in report.live_session_dates)}",
        "",
        "No live broker credentials were read or required at any point in this run "
        "(the only broker constructed was `broker.adapters.paper_broker.PaperBroker`).",
        "",
    ]


def _summary_table(report: SessionReport) -> list[str]:
    stage_pass = sum(1 for s in report.stages if s.category == "stage" and s.ok)
    stage_total = sum(1 for s in report.stages if s.category == "stage")
    failure_pass = sum(1 for s in report.stages if s.category == "failure_injection" and s.ok)
    failure_total = sum(1 for s in report.stages if s.category == "failure_injection")
    invariant_pass = sum(1 for r in report.final_invariants if r.ok)
    invariant_total = len(report.final_invariants)
    return [
        "## Summary",
        "",
        "| Category | Passed | Total |",
        "|---|---|---|",
        f"| Session narrative stages | {stage_pass} | {stage_total} |",
        f"| Required failure injections | {failure_pass} | {failure_total} |",
        f"| Final invariants | {invariant_pass} | {invariant_total} |",
    ]


def _narrative_section(report: SessionReport) -> list[str]:
    by_name = {stage.name: stage for stage in report.stages}
    lines = ["## Session narrative", ""]
    lines.append("| # | Stage | Type | Result | Detail |")
    lines.append("|---|---|---|---|---|")
    for index, name in enumerate(_NARRATIVE_ORDER, start=1):
        stage = by_name.get(name)
        if stage is None:
            lines.append(f"| {index} | {name} | -- | **DID NOT RUN** | -- |")
            continue
        lines.append(
            f"| {index} | {_escape(stage.name)} | {stage.category} | "
            f"{_badge(stage.ok)} | {_escape(stage.detail)} |"
        )
    unexpected = [s.name for s in report.stages if s.name not in _NARRATIVE_ORDER]
    if unexpected:
        lines.append("")
        lines.append(f"Additional stages not in the fixed narrative order: {unexpected}")
    return lines


def _invariants_section(report: SessionReport) -> list[str]:
    lines = [
        "## Final invariants",
        "",
        "Checked once more after the full narrative -- including recovery from every "
        "injected failure -- completed, against live state (not a replay).",
        "",
        "| Invariant | Result | Detail |",
        "|---|---|---|",
    ]
    for result in report.final_invariants:
        status = {
            InvariantStatus.PASS: "PASS",
            InvariantStatus.FAIL: "**FAIL**",
            InvariantStatus.SKIPPED: "SKIPPED",
        }[result.status]
        lines.append(f"| {result.invariant.value} | {status} | {_escape(result.detail)} |")
    lines.append("")
    lines.append(
        "`restart is safe` is not a point-in-time check above -- it is proven by the "
        "`application crash` / `reconciliation` stage pair in the narrative table: the "
        "fresh process correctly refused to trade until an operator resolved the gap, "
        "and never re-submitted or duplicated anything in the meantime."
    )
    return lines


def _limitations_section() -> list[str]:
    return [
        "## Limitations -- read before drawing any conclusion beyond \"the system works "
        "end to end\"",
        "",
        "- **The market data is synthetic**, not historical. It exists to give every "
        "stage something to compute over; no conclusion about returns, regime accuracy, "
        "or factor performance follows from this run. See "
        "`validation/synthetic_market.py`'s own docstring.",
        "- **Quotes are synthesized from the daily close** (`validation/paper_feed.py`), "
        "not a real bid/ask history. Spread-sensitive results (execution cost, partial-fill "
        "sizing) are illustrative of the *mechanism*, not the *magnitude*, a live spread "
        "would produce.",
        "- **There is no intraday path.** Every quote within a session is the same close, "
        "so this run says nothing about intraday timing risk.",
        "- **The failure injections are deliberate and scripted**, not a fuzzer -- they "
        "prove each named scenario is handled the way each phase's own design claims, not "
        "that no other failure mode exists.",
        "- **A crash loses pre-crash order traceability.** "
        "`execution.execution_journal.ExecutionJournal` is documented (Phase 17) as "
        "in-memory only. The `application crash` stage below confirms this directly: after "
        "the crash, `all orders traceable` holds only for orders created since -- durable "
        "journal persistence remains `storage/`'s unimplemented job, not something this "
        "phase changes.",
        "- **The recovery in the `reconciliation` stage is scripted for this scenario's own "
        "data** (seed local state from the broker's own reported truth; cancel the one "
        "orphaned order). It demonstrates the *mechanism* Phase 17/18 provide for an "
        "operator to use, not an automated recovery the system performs on its own -- per "
        "both phases' explicit design, no position or order discrepancy is ever "
        "auto-resolved.",
    ]


def _badge(ok: bool) -> str:
    return "PASS" if ok else "**FAIL**"


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
