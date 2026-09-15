"""Run the Phase 21 end-to-end paper-trading validation and write its
report.

Usage:
    python scripts/run_e2e_validation.py [output/path.md]

Wires the entire system in paper mode against a synthetic vendor drop
(no live credentials, no network access), runs a full session with all
required failure injections, and writes a Markdown report. Exits non-zero
if any stage or final invariant failed, so this can also run as a
pre-deploy gate independent of ``pytest``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from validation.harness import build_validation_environment
from validation.report import write_markdown_report
from validation.scenario import run_end_to_end_validation

DEFAULT_OUTPUT = Path("validation_report.md")


def main(argv: list[str]) -> int:
    output_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUTPUT

    with tempfile.TemporaryDirectory(prefix="e2e-validation-") as scratch:
        print("Building the validation environment (ingesting synthetic data, "
              "fitting and approving a model)...")
        env = build_validation_environment(Path(scratch), sessions=1100, train_end_index=900)

        print("Running the end-to-end session, including all required failure injections...")
        report = run_end_to_end_validation(env)

        write_markdown_report(env, report, output_path)

    for stage in report.stages:
        marker = "PASS" if stage.ok else "FAIL"
        print(f"[{marker}] {stage.category:17s} {stage.name}")
    print()
    print(f"Report written to {output_path.resolve()}")
    print(f"Overall result: {'PASSED' if report.all_ok else 'FAILED'}")

    return 0 if report.all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
