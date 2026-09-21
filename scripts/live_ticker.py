"""Poll Zerodha for the picked stocks and keep a rolling intraday tick log.

    python scripts/live_ticker.py            # runs until stopped
    python scripts/live_ticker.py --once     # single sample, for testing

Why a local loop rather than the dashboard polling directly: the artifact
sandbox does not permit api.kite.trade, and an access token embedded in a
shared page would hand anyone who opened it the ability to trade. So ticks
are captured where the token already lives.

Why a loop rather than a scheduled job: the schedule that publishes to the
dashboard costs a model turn per firing, so it runs every few minutes. A
sparkline needs finer resolution than that. This process samples every 30
seconds into a local file, and the publish step ships whatever has
accumulated -- fine-grained history, coarse-grained publishing.

The log is bounded and per-day. Ticks are dropped when they age past
``--keep-minutes`` so the file cannot grow without limit over a session,
and a new trading day starts a fresh series rather than drawing a line
across an overnight gap.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from scripts._dashboard_live import live_quotes  # noqa: E402

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _now_ist() -> dt.datetime:
    return dt.datetime.now(IST)


def sample(picks: list[dict[str, Any]], session_path: Path) -> dict[str, Any] | None:
    quotes = live_quotes([p["instrument_id"] for p in picks], session_path)
    if not quotes.get("available"):
        return None
    now = _now_ist()
    return {
        "t": now.isoformat(timespec="seconds"),
        "hhmm": f"{now:%H:%M}",
        "px": {i: round(q["last"], 2) for i, q in quotes["quotes"].items()},
    }


def prune(ticks: list[dict[str, Any]], keep_minutes: int) -> list[dict[str, Any]]:
    """Drop ticks older than the window, and anything from a previous day.

    A series spanning an overnight gap would draw a sparkline across hours
    the market was shut, which reads as a price move that never happened.
    """
    if not ticks:
        return ticks
    now = _now_ist()
    cutoff = now - dt.timedelta(minutes=keep_minutes)
    kept = []
    for tick in ticks:
        try:
            when = dt.datetime.fromisoformat(tick["t"])
        except (KeyError, ValueError):
            continue
        if when.date() == now.date() and when >= cutoff:
            kept.append(tick)
    return kept


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "state" / "dashboard_data.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "state" / "live_ticks.json")
    parser.add_argument("--interval", type=int, default=30, help="seconds between samples")
    parser.add_argument("--keep-minutes", type=int, default=420, help="tick window to retain")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv[1:])

    picks = (json.loads(args.data.read_text(encoding="utf-8")).get("selection") or {}).get("picks")
    if not picks:
        raise SystemExit("no selection to track; run scripts/collect_dashboard_data.py first")
    session_path = REPO_ROOT / "state" / "kite_session.json"

    while True:
        ticks: list[dict[str, Any]] = []
        if args.out.is_file():
            try:
                ticks = json.loads(args.out.read_text(encoding="utf-8")).get("ticks", [])
            except (OSError, ValueError):
                ticks = []

        tick = sample(picks, session_path)
        if tick is not None:
            ticks.append(tick)
            ticks = prune(ticks, args.keep_minutes)
            args.out.write_text(
                json.dumps({"updated_at": tick["t"], "ticks": ticks}), encoding="utf-8"
            )
            print(f"{tick['hhmm']}  {len(tick['px'])} quotes  ({len(ticks)} ticks retained)")
        else:
            print(f"{_now_ist():%H:%M}  no quote (session expired or API unreachable)")

        if args.once:
            return 0
        time.sleep(max(args.interval, 5))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
