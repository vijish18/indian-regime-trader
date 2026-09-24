"""Fetch dated PR evidence for held events/anomalies; no automatic clearance."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.nse_pr_actions import load_archive  # noqa: E402
from storage.atomic import atomic_write  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--anomalies", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, help="Also audit all dates in this index CSV")
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    anomalies = json.loads(args.anomalies.read_text())
    dates = {r["ex_date"] for r in audit["events"]}
    dates.update(r["session_date"] for r in anomalies)
    if args.sessions:
        with args.sessions.open(encoding="utf-8-sig") as handle:
            dates.update(r["session_date"] for r in csv.DictReader(handle))
    results = {}
    if args.output.exists():
        results = json.loads(args.output.read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(load_archive, dt.date.fromisoformat(day), args.cache): day
            for day in sorted(dates)
            if "rows" not in results.get(day, {})
        }
        for future in as_completed(futures):
            day = futures[future]
            try:
                results[day] = future.result()
            except Exception as exc:  # evidence failures stay visible and resumable
                results[day] = {"error": str(exc)}
            # Archives are individually durable; periodically publish the growing index.
            if len(results) % 25 == 0:
                atomic_write(args.output, json.dumps(results, indent=2))
            print(
                f"{len(results)}/{len(dates)} {day}: {results[day].get('error', 'downloaded')}",
                flush=True,
            )
    atomic_write(args.output, json.dumps(results, indent=2))
