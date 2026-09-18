"""Combine per-strategy walk-forward JSON into the one comparison table.

    python scripts/merge_walk_forward.py --json-dir state/wf_json

The five-way comparison takes roughly 24 CPU hours in a single process, so
``run_walk_forward.py --strategy X --json-out ...`` runs each strategy in
its own process and this merges the results. The split is exact rather than
approximate: each strategy chains equity only through its own folds and owns
its circuit-breaker state file, which
``tests/unit/test_walk_forward.py::test_a_subset_run_is_identical_to_the_same_names_in_a_whole_run``
asserts directly.

This refuses to render a partial table. A comparison missing the control, or
missing the baseline the HMM is supposed to beat, is not a weaker answer --
it is one that invites the wrong conclusion from whichever rows happen to be
present.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from backtest.performance import PerformanceReport  # noqa: E402
from backtest.walk_forward import STRATEGY_NAMES  # noqa: E402
from scripts.run_walk_forward import REPORT_PATH, render  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="render even if some strategies are missing (they are listed as absent)",
    )
    args = parser.parse_args(argv[1:])

    files = sorted(args.json_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no JSON files in {args.json_dir}")

    reports: dict[str, PerformanceReport] = {}
    windows: set[tuple[str, str]] = set()
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        windows.add((payload["start"], payload["end"]))
        for name, fields in payload["reports"].items():
            if name in reports:
                raise SystemExit(f"{name} appears in more than one JSON file ({path})")
            reports[name] = PerformanceReport(**fields)

    if len(windows) != 1:
        # Merging runs over different periods would produce a table whose
        # rows are not comparable, which is the one thing this table exists
        # to be.
        raise SystemExit(f"JSON files cover different periods: {sorted(windows)}")

    missing = [name for name in STRATEGY_NAMES if name not in reports]
    if missing and not args.allow_partial:
        raise SystemExit(
            f"missing strategies: {missing}. The comparison is only meaningful "
            "whole -- the HMM is judged against the baselines and the shuffled "
            "control, so a table without them invites the wrong conclusion. "
            "Re-run the missing ones, or pass --allow-partial deliberately."
        )
    if missing:
        print(f"WARNING: rendering a partial table; missing {missing}")

    start_text, end_text = next(iter(windows))
    start, end = dt.date.fromisoformat(start_text), dt.date.fromisoformat(end_text)
    ordered = {name: reports[name] for name in STRATEGY_NAMES if name in reports}

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(ordered, start, end), encoding="utf-8")

    header = (
        f"{'strategy':<28} {'CAGR':>8} {'vol':>8} {'Sharpe':>8} "
        f"{'max DD':>9} {'inv':>7} {'trades':>7}"
    )
    print(header)
    for name, report in ordered.items():
        print(
            f"{name:<28} {report.cagr:>7.2%} {report.volatility:>7.2%} "
            f"{report.sharpe:>8.2f} {report.max_drawdown:>8.2%} "
            f"{report.pct_invested:>6.1%} {report.trade_count:>7}"
        )
    print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
