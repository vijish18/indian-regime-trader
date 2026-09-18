"""Gather everything the dashboard shows into one JSON file.

    python scripts/collect_dashboard_data.py --out state/dashboard_data.json

Collection is kept apart from rendering on purpose. Every number on the
dashboard comes from a file on disk that something else produced -- the
bhavcopy backfill, the walk-forward run, the cost schedule -- so the
rendering step has nothing to compute and no opportunity to invent. If a
figure looks wrong, it is wrong in the artifact named beside it here.

Panels whose inputs are absent are reported as absent rather than
defaulted. A dashboard that shows a plausible zero where it means "the
backtest has not finished" is worse than one that says so: the zero gets
read as a result.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from backtest.walk_forward import STRATEGY_NAMES  # noqa: E402
from data.calendar import NSETradingCalendar  # noqa: E402

DATA_CACHE = REPO_ROOT / "data_cache"
REFERENCE = DATA_CACHE / "reference"
HOLIDAY_FILE = REPO_ROOT / "config" / "nse_holidays.csv"
COST_SCHEDULE = REPO_ROOT / "config" / "cost_schedules.yaml"


def _count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def provenance() -> dict[str, Any]:
    """Where the numbers came from, counted rather than asserted."""
    bars = DATA_CACHE / "raw" / "equity_bars"
    bhavcopy = list(DATA_CACHE.glob("bhavcopy/*.zip")) or list(DATA_CACHE.glob("**/bhavcopy-*.zip"))
    bar_files = sorted(bars.glob("*.csv")) if bars.is_dir() else []
    total_bars = sum(_count_rows(p) for p in bar_files)
    calendar = NSETradingCalendar.from_file(HOLIDAY_FILE)
    return {
        "bhavcopy_archives": len(bhavcopy),
        "instrument_bar_files": len(bar_files),
        "total_bars": total_bars,
        "instruments": _count_rows(REFERENCE / "instruments.csv"),
        "corporate_actions": _count_rows(REFERENCE / "corporate_actions.csv"),
        "membership_spans": _count_rows(REFERENCE / "index_membership.csv"),
        "calendar_years": sorted(calendar.covered_years),
        "index_sessions": _count_rows(DATA_CACHE / "raw" / "index" / "NIFTY50.csv"),
    }


def universe_growth(sample_every_days: int = 90) -> list[dict[str, Any]]:
    """Point-in-time universe size over history.

    This is the survivorship-bias fix made visible: the universe is small
    early and large late because that is what actually traded, not because
    today's list was projected backwards.
    """
    path = REFERENCE / "index_membership.csv"
    if not path.is_file():
        return []
    spans: list[tuple[dt.date, dt.date]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                start = dt.date.fromisoformat(row["effective_from"])
            except (KeyError, ValueError):
                continue
            end_text = row.get("effective_to") or ""
            end = dt.date.fromisoformat(end_text) if end_text else dt.date(2100, 1, 1)
            spans.append((start, end))
    if not spans:
        return []

    first = min(s for s, _ in spans)
    last = min(max(e for _, e in spans), dt.date.today())
    out: list[dict[str, Any]] = []
    day = first
    while day <= last:
        out.append(
            {"date": day.isoformat(), "count": sum(1 for s, e in spans if s <= day <= e)}
        )
        day += dt.timedelta(days=sample_every_days)
    return out


def cost_eras() -> list[dict[str, Any]]:
    if not COST_SCHEDULE.is_file():
        return []
    payload = yaml.safe_load(COST_SCHEDULE.read_text(encoding="utf-8"))
    out = []
    for entry in payload.get("schedules", []):
        out.append(
            {
                "effective_from": str(entry["effective_from"]),
                "label": entry.get("label", ""),
                "stt_round_trip_bps": (entry["stt_buy_pct"] + entry["stt_sell_pct"]) * 10_000,
                "stamp_duty_buy_bps": entry["stamp_duty_buy_pct"] * 10_000,
                "exchange_txn_bps": entry["exchange_txn_pct"] * 10_000,
                "gst_pct": entry["gst_pct"] * 100,
            }
        )
    return out


def strategy_results(json_dir: Path | None) -> dict[str, Any]:
    """The five-way comparison, if the walk-forward run has produced it."""
    if json_dir is None or not json_dir.is_dir():
        return {"available": False, "reason": "no --json-dir given", "reports": {}}
    files = sorted(json_dir.glob("*.json"))
    if not files:
        return {"available": False, "reason": f"no JSON in {json_dir}", "reports": {}}

    reports: dict[str, Any] = {}
    window: dict[str, str] = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        window = {"start": payload["start"], "end": payload["end"]}
        reports.update(payload["reports"])

    missing = [name for name in STRATEGY_NAMES if name not in reports]
    return {
        # Deliberately not "available" until every strategy is in: the HMM is
        # judged against the baselines and the shuffled control, and a table
        # with some of them missing invites a conclusion the data does not
        # support.
        "available": not missing,
        "reason": f"still running: {missing}" if missing else "",
        "missing": missing,
        "window": window,
        "reports": {name: reports[name] for name in STRATEGY_NAMES if name in reports},
    }


def equity_curves(series_dir: Path | None, sample_every: int = 5) -> dict[str, Any]:
    """Equity curves, thinned for the browser.

    Sampling every 5th session keeps ~800 points per strategy instead of
    ~4,000, which is below the resolution of any chart this size and keeps
    the page from carrying four times the data it can draw.
    """
    if series_dir is None or not series_dir.is_dir():
        return {}
    out: dict[str, Any] = {}
    for path in sorted(series_dir.glob("*.equity.csv")):
        name = path.name.removesuffix(".equity.csv")
        points: list[dict[str, Any]] = []
        with path.open(encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if index % sample_every:
                    continue
                try:
                    points.append(
                        {"date": row["session_date"][:10], "equity": float(row["equity"])}
                    )
                except (KeyError, ValueError):
                    continue
        if points:
            out[name] = points
    return out


def regime_distribution(cache: Path) -> dict[str, Any]:
    """Per-fold out-of-sample regime mix, if it has been collected.

    Produced by scripts/precheck_folds.py, not here: it needs a fitted model
    per fold, and recomputing that every time the dashboard is rendered
    would put minutes of HMM fitting behind a page refresh.
    """
    if not cache.is_file():
        return {"available": False, "folds": []}
    return {"available": True, **json.loads(cache.read_text(encoding="utf-8"))}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, default=REPO_ROOT / "state" / "wf_json")
    parser.add_argument("--series-dir", type=Path, default=REPO_ROOT / "state" / "wf_series")
    parser.add_argument(
        "--regime-cache", type=Path, default=REPO_ROOT / "state" / "fold_regimes.json"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "state" / "dashboard_data.json")
    args = parser.parse_args(argv[1:])

    data: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "provenance": provenance(),
        "universe_growth": universe_growth(),
        "cost_eras": cost_eras(),
        "strategies": strategy_results(args.json_dir),
        "equity_curves": equity_curves(args.series_dir),
        "regimes": regime_distribution(args.regime_cache),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    prov = data["provenance"]
    print(f"wrote {args.out}")
    print(f"  {prov['instruments']:,} instruments, {prov['total_bars']:,} bars, "
          f"{prov['corporate_actions']:,} corporate actions")
    print(f"  universe samples : {len(data['universe_growth'])}")
    print(f"  cost eras        : {len(data['cost_eras'])}")
    print(f"  equity curves    : {len(data['equity_curves'])}")
    strategies = data["strategies"]
    if strategies["available"]:
        print(f"  strategies       : all {len(strategies['reports'])} present")
    else:
        print(f"  strategies       : NOT COMPLETE ({strategies['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
