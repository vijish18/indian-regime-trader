"""Turn the bhavcopy cache into per-instrument bar files the providers read.

    python scripts/build_equity_bars.py                       # universe names only
    python scripts/build_equity_bars.py --all-instruments

``scripts/backfill_bhavcopy.py`` produces one file per *session*, which is
the right shape for building a universe and the wrong shape for asking
"what did RELIANCE do over the last 200 days". ``LocalMarketDataProvider``
reads one file per *instrument*, so this transposes the cache.

**Why adjusted prices are not written here.** It would be easy to apply
the corporate-action factors once and store the result. That is exactly
what must not happen: an adjustment factor is only correct *as of* a
particular date, because a split announced next March retroactively
changes what last January's price means. Baking one in freezes a single
vantage point into the data, and every backtest that later reads it
inherits a view of the past that was not available at the time.

So bars are stored RAW, and ``LocalMarketDataProvider._adjust`` applies
``cumulative_adjustment_factor(instrument_id, bar_date, as_of)`` at read
time, per request. The machinery for that already exists and is
point-in-time correct; this script's job is only to put raw bars where it
can find them.

Verified against NESTLEIND's 10:1 split (ex-date 2024-01-05): raw closes
fall from Rs 27,116 to Rs 2,666, a -90.2% "return" that never happened.
Adjusted, the same two sessions are Rs 2,711.64 and Rs 2,666.40 -- a -1.7%
move, which is what actually occurred.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from data.calendar import NSETradingCalendar  # noqa: E402
from data.nse_bhavcopy import BhavcopyError, load_bhavcopy  # noqa: E402

HOLIDAY_FILE = REPO_ROOT / "config" / "nse_holidays.csv"
DATA_CACHE = REPO_ROOT / "data_cache"
BHAVCOPY_CACHE = DATA_CACHE / "raw" / "bhavcopy"
MEMBERSHIP = DATA_CACHE / "reference" / "index_membership.csv"
BARS_OUT = DATA_CACHE / "raw" / "equity_bars"
INSTRUMENTS_OUT = DATA_CACHE / "reference" / "instruments.csv"
KITE_INSTRUMENTS = DATA_CACHE / "raw" / "kite" / "instruments.csv"

DEFAULT_TICK_SIZE = "0.05"
"""Used when Kite's live dump has no entry for an instrument, which is
the case for every name delisted before today. 0.05 is NSE's common
tick for the price bands most equities trade in.

This is an approximation and it is worth knowing where it bites: tick
size only affects order-price rounding in fill simulation, so a wrong
tick shifts a simulated fill by at most one tick. It does not affect
which names are eligible, what they are ranked on, or what the bars
say. Real tick sizes are used for every instrument still listed."""

INSTRUMENT_HEADER = (
    "instrument_id",
    "symbol",
    "exchange",
    "segment",
    "tick_size",
    "price_precision",
    "effective_from",
    "effective_to",
    "isin",
)

BAR_HEADER = (
    "instrument_id",
    "session_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_basis",
    "data_source",
)

# One row is a tuple of plain strings rather than a DailyBar: a decade of
# NSE is several million rows, and dataclass instances with Decimal fields
# cost an order of magnitude more memory than the strings that will be
# written out anyway.
_Row = tuple[str, str, str, str, str, str]


def universe_instruments(path: Path) -> set[str]:
    """Every instrument that was ever eligible, from the membership file.

    Restricting to these is not an optimisation. The full EQ list is about
    ten thousand names, most of which this system can never trade -- and
    writing files for them would invite a later change to quietly widen
    the universe by reading whatever happens to be on disk, rather than by
    changing the eligibility rules on purpose.
    """
    if not path.is_file():
        raise SystemExit(
            f"no membership file at {path}. Run: python scripts/backfill_bhavcopy.py"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["instrument_id"] for row in csv.DictReader(handle)}


def load_kite_tick_sizes(path: Path) -> dict[str, str]:
    """Real tick sizes for instruments still listed today.

    Only the tick is taken from this file. Its effective_from is the
    snapshot date, which would make every instrument invalid for every
    historical date -- the reason the instrument master is rebuilt from
    bhavcopy rather than taken from here (docs/KITE_DATA.md).
    """
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["instrument_id"]: row["tick_size"]
            for row in csv.DictReader(handle)
            if row.get("tick_size")
        }


def _price_precision(tick_size: str) -> int:
    text = tick_size.rstrip("0").rstrip(".")
    return len(text.split(".", 1)[1]) if "." in text else 0


def _safe_name(instrument_id: str) -> str:
    return instrument_id.replace(":", "_").replace("/", "_").replace(" ", "_")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=dt.date.fromisoformat,
                        default=dt.date(2015, 1, 1))
    parser.add_argument("--to", dest="end", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--all-instruments", action="store_true",
                        help="write every EQ instrument, not only universe names")
    parser.add_argument("--out", type=Path, default=BARS_OUT)
    args = parser.parse_args(argv[1:])

    end = args.end or dt.date.today()
    calendar = NSETradingCalendar.from_file(HOLIDAY_FILE)
    days = calendar.trading_days_between(args.start, end)

    wanted: set[str] | None = None
    if not args.all_instruments:
        wanted = universe_instruments(MEMBERSHIP)
        print(f"restricting to {len(wanted):,} instruments that were ever eligible")

    bars: dict[str, list[_Row]] = defaultdict(list)
    # instrument_id -> (first seen, last seen, symbol, isin). Built from
    # the same pass: an instrument existed on the days it printed a bar,
    # which is the only point-in-time statement available, and the only
    # one that is survivorship-free.
    identity: dict[str, tuple[str, str, str, str]] = {}
    sessions = 0
    skipped = 0

    print(f"transposing {len(days):,} sessions...")
    for index, day in enumerate(sorted(days), start=1):
        if not (BHAVCOPY_CACHE / f"bhavcopy-{day:%Y-%m-%d}.zip").is_file():
            skipped += 1
            continue
        try:
            rows = load_bhavcopy(day, cache_dir=BHAVCOPY_CACHE)
        except BhavcopyError as exc:
            print(f"  skipping {day}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        sessions += 1
        for row in rows:
            instrument_id = row.instrument_id
            if wanted is not None and instrument_id not in wanted:
                continue
            day_iso = row.session_date.isoformat()
            seen = identity.get(instrument_id)
            identity[instrument_id] = (
                seen[0] if seen else day_iso,
                day_iso,
                row.symbol,
                row.isin,
            )
            bars[instrument_id].append(
                (
                    row.session_date.isoformat(),
                    str(row.open),
                    str(row.high),
                    str(row.low),
                    str(row.close),
                    str(row.volume),
                )
            )
        if index % 250 == 0:
            print(
                f"  {index:>5}/{len(days)}  {day}  "
                f"{len(bars):,} instruments, {sum(len(v) for v in bars.values()):,} bars",
                flush=True,
            )

    if not bars:
        raise SystemExit("no bars produced; is the bhavcopy cache populated?")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting {len(bars):,} instrument files to {args.out}...")
    total = 0
    for instrument_id, rows_out in sorted(bars.items()):
        # Sorted because the cache is read in date order, but an instrument
        # that relisted under the same id could otherwise interleave.
        rows_out.sort()
        target = args.out / f"{_safe_name(instrument_id)}.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(BAR_HEADER)
            for session_date, o, h, low, close, volume in rows_out:
                writer.writerow(
                    [instrument_id, session_date, o, h, low, close, volume, "raw", "nse_bhavcopy"]
                )
        total += len(rows_out)

    # Files this run did not write are from a previous, narrower run --
    # a Q1-only build leaves behind names the full-range universe never
    # contains. Nothing would read them (the universe decides what is
    # asked for), but a stale bar file on disk is indistinguishable from
    # a current one to anyone looking, which is how a later change comes
    # to read data nobody meant to keep.
    written = {_safe_name(i) for i in bars}
    orphans = sorted(
        path for path in args.out.glob("*.csv") if path.stem not in written
    )
    if orphans:
        print()
        print(
            f"{len(orphans)} stale file(s) from an earlier run, "
            "not written by this one:"
        )
        for path in orphans[:10]:
            print(f"  {path.name}")
        if len(orphans) > 10:
            print(f"  ... and {len(orphans) - 10} more")
        print("  delete them, or re-run over the same range that produced them")

    ticks = load_kite_tick_sizes(KITE_INSTRUMENTS)
    INSTRUMENTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    last_session = max(identity[i][1] for i in identity)
    with INSTRUMENTS_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(INSTRUMENT_HEADER)
        for instrument_id, (first, last, symbol, isin) in sorted(identity.items()):
            tick = ticks.get(instrument_id, DEFAULT_TICK_SIZE)
            writer.writerow(
                [
                    instrument_id,
                    symbol,
                    "NSE",
                    "equity",
                    tick,
                    _price_precision(tick),
                    first,
                    # Blank means "still listed". An instrument whose last bar
                    # is the last session in the data has not been shown to
                    # have delisted -- it has only been shown not to have
                    # traded since, which is a different claim.
                    "" if last == last_session else last,
                    isin,
                ]
            )
    with_real_tick = sum(1 for i in identity if i in ticks)

    print(f"  {total:,} bars across {len(bars):,} instruments")
    print(f"  instrument master -> {INSTRUMENTS_OUT}")
    print(
        f"    {with_real_tick:,} with a real tick size, "
        f"{len(identity) - with_real_tick:,} defaulted to {DEFAULT_TICK_SIZE}"
    )
    print(f"  {sessions:,} sessions used, {skipped:,} not in the cache")
    print("\nPrices are RAW. Adjusted prices are produced at read time by")
    print("LocalMarketDataProvider, using the corporate actions -- never baked in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
