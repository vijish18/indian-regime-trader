"""Bring the local data store up to a session's close, for the bot.

    python scripts/daily_data_update.py                    # through the last session
    python scripts/daily_data_update.py --to 2026-09-25
    python scripts/daily_data_update.py --corporate-actions  # weekly: refetch the feed

Runs, in order, and stops at the first failure:

1. ``backfill_bhavcopy.py --from 2015-01-01 --to D`` -- always the full
   range, because it rewrites ``index_membership.csv`` from what it covers.
   ``--skip-corporate-actions`` unless ``--corporate-actions``, which also
   re-applies ``repair_corporate_action_symbols.py`` to the fresh feed.
2. ``build_equity_bars.py --to D``.
3. NIFTY 50 and India VIX appended from NSE's public
   ``ind_close_all_DDMMYYYY.csv`` -- no Kite login needed, which matters
   because the bot only logs in on rebalance mornings.

Then it verifies the result reached D (NIFTY and VIX rows dated D) and
exits non-zero if not, so the scheduler and the evening job can refuse to
rank on stale data.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_walk_forward import DATA_CACHE, HOLIDAY_FILE  # noqa: E402

INDEX_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{d:%d%m%Y}.csv"
INDEX_NAMES = {"Nifty 50": "NIFTY50", "India VIX": "INDIAVIX"}
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0"


def parse_index_close(text: str, day: dt.date) -> dict[str, list[str]]:
    """``{symbol: [symbol, date, open, high, low, close, ""]}`` for the two
    indices, in the store's row layout. Raises if either is missing or dated
    differently."""
    rows: dict[str, list[str]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        symbol = INDEX_NAMES.get((row.get("Index Name") or "").strip())
        if symbol is None:
            continue
        stamp = dt.datetime.strptime(row["Index Date"].strip(), "%d-%m-%Y").date()
        if stamp != day:
            raise ValueError(f"{symbol}: file is dated {stamp}, expected {day}")
        values = [
            row[k].strip()
            for k in (
                "Open Index Value",
                "High Index Value",
                "Low Index Value",
                "Closing Index Value",
            )
        ]
        for v in values:
            float(v)  # a non-numeric field is a malformed file, not a zero
        rows[symbol] = [symbol, day.isoformat(), *values, ""]
    missing = sorted(set(INDEX_NAMES.values()) - set(rows))
    if missing:
        raise ValueError(f"index file for {day} lacks {missing}")
    return rows


def _last_date(path: Path) -> dt.date:
    last = path.read_text(encoding="utf-8").strip().splitlines()[-1].split(",")
    return dt.date.fromisoformat(last[1])


def append_index_closes(sessions: list[dt.date], index_dir: Path) -> int:
    """Fetch and append every session after each file's last row. Returns
    the number of sessions appended."""
    paths = {s: index_dir / f"{s}.csv" for s in INDEX_NAMES.values()}
    start = min(_last_date(p) for p in paths.values())
    todo = [d for d in sessions if d > start]
    for day in todo:
        request = urllib.request.Request(
            INDEX_URL.format(d=day), headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8-sig")
        rows = parse_index_close(text, day)
        for symbol, path in paths.items():
            if _last_date(path) < day:
                with path.open("a", encoding="utf-8", newline="") as handle:
                    handle.write(",".join(rows[symbol]) + "\n")
        time.sleep(1)
    return len(todo)


def _run(args: list[str]) -> None:
    print("$ " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=REPO_ROOT, check=True)


def main(argv: list[str]) -> int:
    from data.calendar import NSETradingCalendar

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", dest="end", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--corporate-actions", action="store_true")
    args = parser.parse_args(argv[1:])

    calendar = NSETradingCalendar.from_file(HOLIDAY_FILE)
    today = dt.date.today()
    end = args.end or (
        today if calendar.is_trading_day(today) else calendar.previous_trading_day(today)
    )

    backfill = ["scripts/backfill_bhavcopy.py", "--from", "2015-01-01", "--to", end.isoformat()]
    if not args.corporate_actions:
        backfill.append("--skip-corporate-actions")
    _run(backfill)
    if args.corporate_actions:
        _run(["scripts/repair_corporate_action_symbols.py"])
    _run(["scripts/build_equity_bars.py", "--to", end.isoformat()])

    index_dir = DATA_CACHE / "raw" / "index"
    sessions = calendar.trading_days_between(end - dt.timedelta(days=30), end)
    added = append_index_closes(sessions, index_dir)
    print(f"index sessions appended: {added}")

    stale = {s: _last_date(index_dir / f"{s}.csv") for s in INDEX_NAMES.values()}
    stale = {s: d for s, d in stale.items() if d != end}
    if stale:
        print(f"STALE after update: {stale} (expected {end})", file=sys.stderr)
        return 1
    print(f"data store is current through {end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
