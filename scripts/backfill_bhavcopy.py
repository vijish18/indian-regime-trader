"""Backfill NSE bhavcopy and emit the point-in-time reference data.

    python scripts/backfill_bhavcopy.py --from 2015-01-01           # ~2,700 sessions
    python scripts/backfill_bhavcopy.py --from 2024-01-01 --to 2024-12-31
    python scripts/backfill_bhavcopy.py --from 2015-01-01 --skip-download

Produces, under ``data_cache/``:

    raw/bhavcopy/bhavcopy-YYYY-MM-DD.zip   one file per session, the cache
    reference/index_membership.csv         point-in-time universe spans
    reference/corporate_actions.csv        splits, bonuses, dividends
    reference/backfill_report.md           what was fetched, what was not

**Resumable, because it has to be.** Roughly 2,700 files at a polite
request rate is a long run, and a run that has to start again from 2015
whenever the network hiccups at file 2,000 is a run nobody completes.
Every file is cached on disk, and a cached day is never re-fetched -- so
re-running after a failure costs only the days that are actually missing.

**Which days to fetch comes from the trading calendar**, not from a date
range. Asking NSE for a Sunday wastes a request and produces a 404 that
looks exactly like a real failure. The calendar already knows
(docs/MARKET_CALENDAR.md), so the only 404s that survive are ones worth
reporting.

**Nothing is inferred from a gap.** A session the calendar says happened
but NSE does not publish is recorded in the report, not silently treated
as "nothing traded". That distinction is the difference between a hole in
the data and a day the market was quiet, and only one of them is safe to
build a universe from.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from config.loader import load_settings  # noqa: E402
from data.calendar import NSETradingCalendar  # noqa: E402
from data.models import CorporateAction, IndexMembership  # noqa: E402
from data.nse_bhavcopy import (  # noqa: E402
    BhavcopyError,
    BhavcopyNotPublished,
    BhavcopyRow,
    load_bhavcopy,
)
from data.nse_corporate_actions import (  # noqa: E402
    CorporateActionFeedError,
    fetch_raw,
    to_corporate_actions,
)
from universe.bhavcopy_universe import (  # noqa: E402
    DERIVED_INDEX_SYMBOL,
    EligibilityRules,
    UniverseSnapshot,
    snapshots_to_membership,
    stream_snapshots,
)

HOLIDAY_FILE = REPO_ROOT / "config" / "nse_holidays.csv"
DATA_CACHE = REPO_ROOT / "data_cache"
BHAVCOPY_CACHE = DATA_CACHE / "raw" / "bhavcopy"
REFERENCE = DATA_CACHE / "reference"

REQUEST_INTERVAL_SECONDS = 0.35
"""Spacing between downloads. NSE publishes no rate limit for the archive,
so this is politeness rather than compliance: roughly three requests a
second, which is what their documented API limit allows elsewhere. A
backfill is not urgent and being throttled mid-run costs far more than
pacing does."""


@dataclass
class BackfillOutcome:
    fetched: int = 0
    cached: int = 0
    missing: list[dt.date] = None  # type: ignore[assignment]
    failed: list[tuple[dt.date, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.missing = self.missing or []
        self.failed = self.failed or []


def download(days: list[dt.date], *, pause: float = REQUEST_INTERVAL_SECONDS) -> BackfillOutcome:
    outcome = BackfillOutcome()
    BHAVCOPY_CACHE.mkdir(parents=True, exist_ok=True)
    total = len(days)

    for index, day in enumerate(days, start=1):
        cached_file = BHAVCOPY_CACHE / f"bhavcopy-{day:%Y-%m-%d}.zip"
        already = cached_file.is_file()
        try:
            load_bhavcopy(day, cache_dir=BHAVCOPY_CACHE)
        except BhavcopyNotPublished:
            outcome.missing.append(day)
        except BhavcopyError as exc:
            outcome.failed.append((day, str(exc)[:120]))
        else:
            if already:
                outcome.cached += 1
            else:
                outcome.fetched += 1
                time.sleep(pause)

        if index % 100 == 0 or index == total:
            print(
                f"  {index:>5}/{total}  {day}  "
                f"fetched={outcome.fetched} cached={outcome.cached} "
                f"missing={len(outcome.missing)} failed={len(outcome.failed)}",
                flush=True,
            )
    return outcome


def stream_cached(
    days: list[dt.date], counter: list[int]
) -> Iterator[tuple[dt.date, tuple[BhavcopyRow, ...]]]:
    """Yield cached sessions one at a time, in date order.

    A generator rather than a dict because a full 2015-2026 backfill is
    roughly 5.4 million rows; materialising them all costs gigabytes to
    compute a result whose memory is otherwise bounded by the trailing
    window. Days absent from the cache are skipped, not re-fetched, so
    this stage is offline and repeatable.

    ``counter`` is a one-element list the caller reads afterwards: a
    generator cannot return a count alongside its items.
    """
    for day in sorted(days):
        if not (BHAVCOPY_CACHE / f"bhavcopy-{day:%Y-%m-%d}.zip").is_file():
            continue
        try:
            rows = load_bhavcopy(day, cache_dir=BHAVCOPY_CACHE)
        except BhavcopyError as exc:
            print(f"  skipping {day}: {exc}", file=sys.stderr)
            continue
        counter[0] += 1
        yield day, rows


def write_membership(memberships: list[IndexMembership], target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["index_symbol", "instrument_id", "effective_from", "effective_to"])
        for membership in memberships:
            writer.writerow(
                [
                    membership.index_symbol,
                    membership.instrument_id,
                    membership.effective_from.isoformat(),
                    membership.effective_to.isoformat() if membership.effective_to else "",
                ]
            )
    return len(memberships)


def write_corporate_actions(start: dt.date, end: dt.date, target: Path) -> tuple[int, int, int]:
    """Fetch the corporate-actions feed year by year and write one file.

    Year by year because the endpoint degrades on very long ranges, and a
    year is the natural unit anyway -- a failure costs one year's refetch.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    actions: list[CorporateAction] = []
    unparsed: list[tuple[str, str]] = []
    review: set[str] = set()

    for year in range(start.year, end.year + 1):
        window_start = max(start, dt.date(year, 1, 1))
        window_end = min(end, dt.date(year, 12, 31))
        try:
            records = fetch_raw(window_start, window_end)
        except CorporateActionFeedError as exc:
            print(f"  corporate actions {year}: FAILED ({exc})", file=sys.stderr)
            continue
        result = to_corporate_actions(records)
        actions.extend(result.actions)
        unparsed.extend(result.unparsed)
        review |= result.instruments_needing_review()
        print(
            f"  {year}: {len(result.actions):>5} actions, "
            f"{len(result.unparsed):>3} unparsed",
            flush=True,
        )
        time.sleep(REQUEST_INTERVAL_SECONDS)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "instrument_id",
                "action_type",
                "ex_date",
                "ratio_new",
                "ratio_old",
                "cash_amount",
                "explicit_price_factor",
                "record_date",
            ]
        )
        for action in sorted(actions, key=lambda a: (a.ex_date, a.instrument_id)):
            writer.writerow(
                [
                    action.instrument_id,
                    action.action_type.value,
                    action.ex_date.isoformat(),
                    action.ratio_new if action.ratio_new is not None else "",
                    action.ratio_old if action.ratio_old is not None else "",
                    action.cash_amount if action.cash_amount is not None else "",
                    "",  # explicit_price_factor: only a human supplies this
                    action.record_date.isoformat() if action.record_date else "",
                ]
            )
    return len(actions), len(unparsed), len(review)


def write_report(
    target: Path,
    *,
    start: dt.date,
    end: dt.date,
    sessions: int,
    outcome: BackfillOutcome,
    loaded: int,
    snapshots: list[object],
    memberships: int,
    instruments: int,
    action_counts: tuple[int, int, int],
) -> None:
    actions, unparsed, review = action_counts
    lines = [
        "# Backfill report",
        "",
        f"- Generated: {dt.datetime.now(dt.UTC).isoformat()}",
        f"- Range: {start} to {end} ({sessions} trading sessions per the calendar)",
        "",
        "## Bhavcopy",
        "",
        f"- Downloaded this run: {outcome.fetched}",
        f"- Already cached: {outcome.cached}",
        f"- Parsed into the universe: {loaded}",
        f"- Not published by NSE: {len(outcome.missing)}",
        f"- Failed: {len(outcome.failed)}",
        "",
    ]
    if outcome.missing:
        lines += [
            "### Sessions the calendar expected but NSE did not publish",
            "",
            "These are holes, not quiet days. A universe built across one is "
            "missing a session it believes it has.",
            "",
        ]
        lines += [f"- {day} ({day:%a})" for day in outcome.missing]
        lines.append("")
    if outcome.failed:
        lines += ["### Failures", ""]
        lines += [f"- {day}: {reason}" for day, reason in outcome.failed]
        lines.append("")

    eligible_counts = [len(s.eligible) for s in snapshots]  # type: ignore[attr-defined]
    lines += [
        "## Point-in-time universe",
        "",
        f"- Sessions with a snapshot: {len(snapshots)}",
        f"- Membership spans: {memberships}",
        f"- Distinct instruments ever eligible: {instruments}",
        f"- Eligible per session: min {min(eligible_counts, default=0)}, "
        f"max {max(eligible_counts, default=0)}",
        f"- Index symbol: `{DERIVED_INDEX_SYMBOL}` (NOT NIFTY 50 -- see "
        "universe/bhavcopy_universe.py)",
        "",
        "## Corporate actions",
        "",
        f"- Parsed: {actions}",
        f"- Unparsed subject lines: {unparsed}",
        f"- Instruments needing an operator-supplied factor: {review}",
        "",
        "Rights, mergers and demergers carry no computable terms in NSE's feed. "
        "They are recorded with no price factor and must be excluded from the "
        "universe for any window spanning their ex-date, per "
        "docs/SPECIFICATION.md section 2.1.",
        "",
        "## What this does not give you",
        "",
        "Adjusted prices. The actions are ingested; applying them to build an "
        "adjustment-factor series per instrument is the next step. Until then "
        "`PriceBasis.ADJUSTED` remains unavailable and factors computed on raw "
        "prices are wrong for any name with a split or bonus in the window.",
        "",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=dt.date.fromisoformat,
                        default=dt.date(2015, 1, 1))
    parser.add_argument("--to", dest="end", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--skip-download", action="store_true",
                        help="build outputs from the existing cache only")
    parser.add_argument("--skip-corporate-actions", action="store_true")
    args = parser.parse_args(argv[1:])

    end = args.end or dt.date.today()
    calendar = NSETradingCalendar.from_file(HOLIDAY_FILE)
    covered = calendar.covered_years
    missing_years = sorted(set(range(args.start.year, end.year + 1)) - covered)
    if missing_years:
        raise SystemExit(
            f"the trading calendar does not cover {missing_years} (covers {sorted(covered)}).\n"
            "Run: python scripts/build_nse_holidays.py --reconcile-with-kite"
        )

    days = calendar.trading_days_between(args.start, end)
    print(f"{len(days)} trading sessions from {args.start} to {end}\n")

    outcome = BackfillOutcome()
    if not args.skip_download:
        print("downloading bhavcopy (cached days are skipped)...")
        outcome = download(days)
        print()

    settings = load_settings()
    rules = EligibilityRules(
        min_avg_daily_value_inr=Decimal(str(settings.universe.min_avg_daily_value_inr))
    )

    print("streaming the cache into the universe builder...")
    parsed = [0]
    snapshots: list[UniverseSnapshot] = []
    for snapshot in stream_snapshots(stream_cached(days, parsed), rules):
        # Only the per-session eligible sets are kept, never the rows
        # that produced them: a decade of snapshots is small, a decade
        # of bars is not.
        snapshots.append(snapshot)
        if len(snapshots) % 250 == 0:
            print(f"  {len(snapshots):>5} sessions, last {snapshot.session_date}", flush=True)
    print(f"  {parsed[0]} sessions parsed")
    if not snapshots:
        raise SystemExit("no bhavcopy data available; run without --skip-download first")

    memberships = snapshots_to_membership(snapshots)
    instruments = len({m.instrument_id for m in memberships})
    membership_path = REFERENCE / "index_membership.csv"
    write_membership(memberships, membership_path)
    print(f"  {len(memberships)} spans, {instruments} instruments -> {membership_path}\n")

    action_counts = (0, 0, 0)
    if not args.skip_corporate_actions:
        print("fetching corporate actions...")
        action_counts = write_corporate_actions(
            args.start, end, REFERENCE / "corporate_actions.csv"
        )
        print(f"  {action_counts[0]} actions -> {REFERENCE / 'corporate_actions.csv'}\n")

    report_path = REFERENCE / "backfill_report.md"
    write_report(
        report_path,
        start=args.start,
        end=end,
        sessions=len(days),
        outcome=outcome,
        loaded=parsed[0],
        snapshots=list(snapshots),
        memberships=len(memberships),
        instruments=instruments,
        action_counts=action_counts,
    )
    print(f"report -> {report_path}")
    return 1 if outcome.failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
