"""Re-attribute corporate actions to the ticker in force on their ex-date.

    python scripts/repair_corporate_action_symbols.py --dry-run
    python scripts/repair_corporate_action_symbols.py

NSE's corporate-action feed reports each event under the company's **current**
trading symbol, whatever symbol it was actually trading under when the event
happened. The bhavcopy does the opposite, and correctly: it records the symbol
in use that day. So for any company that has been renamed, its history splits
in two -- the bars under the old ticker, every corporate action under the new
one -- and nothing joins them.

The engine looks up actions by instrument id. For a renamed company it finds
none, and the event is silently lost. Both halves of the damage follow:

* the price is never adjusted, so an ex-date reads as a collapse
* the cash or the shares are never credited

The case that exposed it: **Majesco** paid a Rs 974 special dividend with
ex-date 2020-12-23 after selling its US business. The price went 985.65 ->
12.20 and the payout never arrived, because the action was filed under
``NSE:AURUM`` -- the name the company took ten months later -- while the bars
sat under ``NSE:MAJESCO``. A 1,194-share position recorded a Rs 10,89,917
loss on a trade that was roughly flat.

ISIN is the join. It does not change when a company renames, which is exactly
what it is for. This walks every action, finds the instrument master record
whose ``[effective_from, effective_to]`` span contains the ex-date and whose
ISIN matches, and rewrites the action's instrument id to that one.

Idempotent: an action already filed under the right ticker is left alone, so
re-running changes nothing. Safe to run after every backfill, and
``--dry-run`` reports without writing.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = REPO_ROOT / "data_cache" / "reference"


def _date(raw: str) -> dt.date | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return dt.date.fromisoformat(raw)
    except ValueError:
        return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruments", type=Path, default=REFERENCE / "instruments.csv")
    parser.add_argument("--actions", type=Path, default=REFERENCE / "corporate_actions.csv")
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would change, write nothing"
    )
    args = parser.parse_args(argv[1:])

    for path in (args.instruments, args.actions):
        if not path.is_file():
            raise SystemExit(f"missing {path}")

    # ISIN -> every ticker that ISIN has ever traded under, with its span
    spans: dict[str, list[tuple[str, dt.date, dt.date | None]]] = defaultdict(list)
    isin_of: dict[str, str] = {}
    with args.instruments.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            isin = (row.get("isin") or "").strip()
            iid = row["instrument_id"]
            start = _date(row.get("effective_from", ""))
            if not isin or start is None:
                continue
            spans[isin].append((iid, start, _date(row.get("effective_to", ""))))
            isin_of[iid] = isin

    with args.actions.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        actions = list(reader)

    moved: list[tuple[str, str, str, str]] = []
    unresolved: list[tuple[str, str, str]] = []
    renamed_isins = {k for k, v in spans.items() if len({i for i, _, _ in v}) > 1}

    for action in actions:
        iid = action["instrument_id"]
        isin = isin_of.get(iid)
        if isin is None or isin not in renamed_isins:
            continue
        ex_date = _date(action.get("ex_date", ""))
        if ex_date is None:
            continue

        candidates = [
            candidate
            for candidate, start, end in spans[isin]
            if start <= ex_date and (end is None or ex_date <= end)
        ]
        if len(candidates) != 1:
            # Zero: the ex-date predates every record we hold for this ISIN.
            # More than one: overlapping spans in the master. Either way the
            # answer is not knowable here, and a guess would be worse than
            # the status quo, which is at least consistent.
            if candidates != [iid]:
                unresolved.append((iid, action["ex_date"], action["action_type"]))
            continue

        correct = candidates[0]
        if correct != iid:
            moved.append((iid, correct, action["ex_date"], action["action_type"]))
            action["instrument_id"] = correct

    print(f"instruments      : {len(isin_of):,}")
    print(f"ISINs renamed    : {len(renamed_isins):,}")
    print(f"actions read     : {len(actions):,}")
    print(f"actions moved    : {len(moved):,}")
    print(f"unresolved       : {len(unresolved):,}")

    if moved:
        print("\nlargest moves by action type:")
        by_type: dict[str, int] = defaultdict(int)
        for _, _, _, kind in moved:
            by_type[kind] += 1
        for kind, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
            print(f"  {kind:<12} {count:>5}")
        print("\nfirst 15 moves:")
        for was, now, when, kind in moved[:15]:
            print(f"  {when}  {kind:<10} {was:<24} -> {now}")

    if unresolved:
        print("\nleft alone, no single ticker covers the ex-date:")
        for iid, when, kind in unresolved[:10]:
            print(f"  {when}  {kind:<10} {iid}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    if not moved:
        print("\nnothing to change")
        return 0

    backup = args.actions.with_suffix(".csv.pre-isin-repair")
    if not backup.exists():
        shutil.copy2(args.actions, backup)
        print(f"\nbackup -> {backup}")
    with args.actions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(actions)
    print(f"rewrote {args.actions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
