"""Compute every walk-forward ranking once, in parallel, into the selection cache.

    python scripts/precompute_selections.py --from 2015-01-01 --to 2026-09-01 \\
        --cache state/selection_cache --workers 10

Selection was 97% of the walk-forward's run time, and it was being done five
times over -- once per strategy, for rankings that do not depend on the
strategy. This computes each test-window date's ranking once and stores it;
``run_walk_forward.py --selection-cache`` then reads them back.

Dates are independent, so they parallelise cleanly. Each worker takes a
**contiguous** block of dates rather than every Nth one: consecutive rankings
read overlapping price windows, and a contiguous block keeps the frame cache
warm. Dates already in the cache are skipped, so an interrupted precompute
resumes by being run again.

Use the same ``--to`` as the backtest. It is the snapshot date the validator
is built with, and it is part of the cache fingerprint -- a different one is a
different cache.
"""

from __future__ import annotations

import argparse
import datetime as dt
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))


def _worker(job: tuple[int, list[str], str, str, str, int]) -> tuple[int, int, int, float]:
    index, dates, snapshot, cache, state_dir, frame_cache = job
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.run_walk_forward import build_validator

    started = time.time()
    validator = build_validator(
        snapshot_date=dt.date.fromisoformat(snapshot),
        circuit_breaker_dir=Path(state_dir) / f"w{index}",
        frame_cache=frame_cache,
        selection_cache=Path(cache),
    )
    selector = validator.engine.stock_selector
    done = skipped = 0
    for iso in dates:
        day = dt.date.fromisoformat(iso)
        if selector.has(day):  # type: ignore[attr-defined]
            skipped += 1
            continue
        selector.select(day)
        done += 1
        if done % 25 == 0:
            rate = (time.time() - started) / done
            print(f"  worker {index:>2}: {done + skipped}/{len(dates)}  "
                  f"({rate:.0f}s/ranking)  at {iso}", flush=True)
    return index, done, skipped, time.time() - started


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=dt.date.fromisoformat,
                        default=dt.date(2015, 1, 1))
    parser.add_argument("--to", dest="end", type=dt.date.fromisoformat,
                        default=dt.date(2026, 9, 1))
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "state" / "selection_cache")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-cache", type=int, default=600)
    parser.add_argument("--state-dir", type=Path,
                        default=REPO_ROOT / "state" / "precompute_state")
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N dates -- for timing a sample")
    args = parser.parse_args(argv[1:])

    from scripts.run_walk_forward import build_validator

    validator = build_validator(
        snapshot_date=args.end,
        circuit_breaker_dir=args.state_dir / "plan",
        frame_cache=50,
        selection_cache=args.cache,
    )
    folds = validator.generate_folds(args.start, args.end)
    dates: list[dt.date] = sorted({
        day
        for _, _, test_start, test_end in folds
        for day in validator.calendar.trading_days_between(test_start, test_end)
    })
    selector = validator.engine.stock_selector
    pending = [d for d in dates if not selector.has(d)]  # type: ignore[attr-defined]
    print(f"{len(folds)} folds, {len(dates):,} test sessions, "
          f"{len(dates) - len(pending):,} already cached, {len(pending):,} to compute")
    print(f"cache: {selector.directory}")  # type: ignore[attr-defined]
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        return 0

    workers = max(1, min(args.workers, len(pending)))
    size = -(-len(pending) // workers)
    blocks = [pending[i : i + size] for i in range(0, len(pending), size)]
    jobs = [
        (i, [d.isoformat() for d in block], args.end.isoformat(), str(args.cache),
         str(args.state_dir), args.frame_cache)
        for i, block in enumerate(blocks)
    ]
    print(f"{len(jobs)} workers x ~{size} dates each\n", flush=True)

    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        for index, done, skipped, took in pool.imap_unordered(_worker, jobs):
            print(f"worker {index:>2} finished: {done} computed, {skipped} skipped, "
                  f"{took/60:.1f} min", flush=True)
    print(f"\nall done in {(time.time() - started)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
