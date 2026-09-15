"""The operational command-line entry point.

    python -m app.cli preflight

Runs the Phase 22 pre-live checklist (``live.preflight.run_preflight``),
prints a PASS/FAIL report with detailed reasons to the terminal, writes
``docs/preflight_report.md``, and exits non-zero if anything failed.
Exit code is a deliberate, checkable signal: this is meant to gate a
deploy step, not just to be read by a human.

This command never submits an order and never constructs a live-capable
broker. All it does is inspect configuration, the git-tracked source
tree, and run this repository's own test suite as subprocesses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from live.preflight import run_preflight
from live.report import render_text_report, write_reports

DEFAULT_MARKDOWN_REPORT = Path("docs/preflight_report.md")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Operational commands for the Indian Market Regime Trading System.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Run the pre-live checklist and report PASS/FAIL with reasons.",
    )
    preflight_parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_MARKDOWN_REPORT,
        help=f"Where to write the Markdown report (default: {DEFAULT_MARKDOWN_REPORT}).",
    )
    preflight_parser.add_argument(
        "--skip-test-suites",
        action="store_true",
        help=(
            "Skip the checks that run this repository's own test suite as a "
            "subprocess. For fast local iteration only -- a run with this flag "
            "set can never report an overall PASS (see live.preflight.run_preflight)."
        ),
    )
    return parser


def main(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv[1:])

    if args.command == "preflight":
        return _run_preflight_command(args.report, run_test_suites=not args.skip_test_suites)

    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover - argparse.error already exits


def _run_preflight_command(report_path: Path, *, run_test_suites: bool) -> int:
    print("Running the pre-live checklist. This runs this repository's own test suite as")
    print("subprocesses and can take several minutes.\n" if run_test_suites else "")
    report = run_preflight(run_test_suites=run_test_suites)
    print(render_text_report(report))

    write_reports(report, markdown_path=report_path)
    print(f"\nMarkdown report written to {report_path.resolve()}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
