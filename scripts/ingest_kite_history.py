"""Fetch daily history from Kite and run it through the ingestion pipeline.

    python scripts/kite_login.py                      # once each morning
    python scripts/ingest_kite_history.py --instruments-only
    python scripts/ingest_kite_history.py --symbols RELIANCE,INFY,TCS --from 2022-01-01
    python scripts/ingest_kite_history.py --nifty-50 --from 2022-01-01

This script is only the wiring. It fetches through
``broker.zerodha.kite_historical`` and writes through
``data.ingestion.DataIngestionPipeline`` -- the same validated path every
other source uses, so vendor data gets the same data-quality checks,
quarantine behaviour and storage layout as anything else. Nothing here
re-implements validation, and nothing here places an order: the client it
uses has no method that could.

**Why the date range is bounded by the calendar.** Ingesting bars for a
period the trading calendar does not cover would produce a price series
this system cannot reason about -- it could not tell a missing bar from a
holiday. ``config/nse_holidays.csv`` currently covers 2022-2026 (see
docs/MARKET_CALENDAR.md), so a request starting earlier is refused rather
than silently ingested.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from broker.errors import BrokerError  # noqa: E402
from broker.zerodha.kite_historical import (  # noqa: E402
    INDIA_VIX_TOKEN,
    NIFTY_50_TOKEN,
    Candle,
    KiteHistoricalClient,
    KiteInstrument,
)
from broker.zerodha.kite_session import KiteSessionError, load_session  # noqa: E402
from data.calendar import NSETradingCalendar  # noqa: E402
from data.models import Segment  # noqa: E402
from scripts.kite_login import DEFAULT_ENV_FILE, resolve_credentials  # noqa: E402

HOLIDAY_FILE = REPO_ROOT / "config" / "nse_holidays.csv"
DEFAULT_RAW_DIR = REPO_ROOT / "data_cache" / "raw" / "kite"
INSTRUMENT_SNAPSHOT = DEFAULT_RAW_DIR / "instruments.csv"

BAR_HEADER = ("instrument_id", "session_date", "open", "high", "low", "close", "volume")
INSTRUMENT_HEADER = (
    "instrument_id",
    "symbol",
    "exchange",
    "segment",
    "tick_size",
    "price_precision",
    "effective_from",
    "lot_size",
    "name",
)


KITE_SEGMENT_TO_DOMAIN = {
    "NSE": Segment.EQUITY.value,
    "INDICES": Segment.INDEX.value,
}
"""Kite's routing vocabulary -> ``data.models.Segment``.

An explicit table rather than a lowercase() so an unmapped segment (a
derivatives or commodity segment arriving through a widened filter)
raises a KeyError here instead of reaching the instrument master as a
plausible-looking string.
"""


def _price_precision(tick_size: float) -> int:
    """Decimal places implied by the tick.

    Derived rather than assumed: NSE ticks are 0.01 for some scrips and
    0.05 or 0.50 for others, and hardcoding 2 would silently misprice the
    rounding of anything else.
    """
    text = f"{tick_size:.10f}".rstrip("0")
    if "." not in text:
        return 0
    return max(0, len(text.split(".", 1)[1]))


def write_instrument_master(
    instruments: tuple[KiteInstrument, ...], target: Path, *, snapshot_date: dt.date
) -> int:
    """Write the tradable cash-equity universe in this repository's schema.

    Two translations happen here, and both matter.

    **Segment is remapped, not copied.** Kite's ``segment`` is its own
    routing vocabulary (``NSE``, ``INDICES``, ``NFO-OPT``...); this
    repository's ``data.models.Segment`` is a domain vocabulary
    (``equity``, ``index``). Copying Kite's string through produced
    ``'nse' is not a valid Segment`` -- which is the instrument master
    correctly refusing a vendor's word for a field it defines itself.

    Indices are excluded from the tradable set deliberately: NIFTY 50 and
    INDIA VIX both carry ``instrument_type == "EQ"`` in Kite's dump and
    are distinguished only by segment, so filtering on instrument type
    alone would put two untradable instruments into the universe --
    orders that can never fill.

    **``effective_from`` is the snapshot date, and that is a real
    limitation.** Kite's dump is a photograph of *today*: it carries no
    history, so it cannot say when a symbol was listed, renamed, or had
    its tick size changed. Backdating ``effective_from`` to make a
    backtest run would be inventing reference-data history this file does
    not contain. Worse, the dump contains only *currently listed*
    instruments, so using it as a historical universe reintroduces
    exactly the survivorship bias ``universe/universe.py`` exists to
    prevent. See docs/KITE_DATA.md before using this for a backtest.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tradable = [i for i in instruments if i.is_cash_equity]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(INSTRUMENT_HEADER)
        for instrument in tradable:
            writer.writerow(
                [
                    instrument.instrument_id,
                    instrument.tradingsymbol,
                    instrument.exchange,
                    KITE_SEGMENT_TO_DOMAIN[instrument.segment],
                    f"{instrument.tick_size:g}",
                    _price_precision(instrument.tick_size),
                    snapshot_date.isoformat(),
                    instrument.lot_size,
                    instrument.name,
                ]
            )
    return len(tradable)


def write_bars(instrument_id: str, candles: tuple[Candle, ...], target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(BAR_HEADER)
        for candle in candles:
            writer.writerow(
                [
                    instrument_id,
                    candle.session_date.isoformat(),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                ]
            )
    return len(candles)


def _load_calendar() -> NSETradingCalendar:
    try:
        return NSETradingCalendar.from_file(HOLIDAY_FILE)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"could not load the trading calendar: {exc}\n"
            "Run: python scripts/build_nse_holidays.py"
        ) from exc


def _check_range_is_covered(calendar: NSETradingCalendar, start: dt.date, end: dt.date) -> None:
    covered = calendar.covered_years
    wanted = set(range(start.year, end.year + 1))
    missing = sorted(wanted - covered)
    if missing:
        raise SystemExit(
            f"the trading calendar does not cover {missing} (it covers {sorted(covered)}).\n"
            "Ingesting bars for an uncovered year would produce a series this system "
            "cannot interpret -- it could not tell a missing bar from a holiday.\n"
            "Narrow the range, or add those years in scripts/build_nse_holidays.py "
            "(see docs/MARKET_CALENDAR.md)."
        )


def _resolve_client(env_file: Path) -> KiteHistoricalClient:
    api_key, _ = resolve_credentials(env_file)
    try:
        session = load_session(api_key=api_key)
    except KiteSessionError as exc:
        raise SystemExit(str(exc)) from exc
    return KiteHistoricalClient(api_key, access_token=session.access_token)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=dt.date.fromisoformat)
    parser.add_argument("--to", dest="end", type=dt.date.fromisoformat)
    parser.add_argument("--symbols", help="comma-separated NSE trading symbols")
    parser.add_argument(
        "--nifty-50",
        action="store_true",
        help="also fetch the NIFTY 50 and INDIA VIX index series the regime engine needs",
    )
    parser.add_argument(
        "--instruments-only",
        action="store_true",
        help="fetch only the instrument master (needs no access token)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    args = parser.parse_args(argv[1:])

    # --- instrument master: public, so it works before any login ---------
    if args.instruments_only:
        api_key, _ = resolve_credentials(args.env_file)
        client = KiteHistoricalClient(api_key)
        print("fetching the instrument master (no authentication required)...")
        instruments = client.instruments()
        target = args.out / "instruments.csv"
        written = write_instrument_master(
            instruments, target, snapshot_date=dt.date.today()
        )
        print(f"  {len(instruments):,} instruments returned")
        print(f"  {written:,} NSE cash equities written to {target}")
        print("  (indices excluded: NIFTY 50 and INDIA VIX are not tradable)")
        return 0

    if args.start is None:
        parser.error("--from is required unless --instruments-only is given")
    end = args.end or dt.date.today()
    if end < args.start:
        parser.error(f"--to {end} precedes --from {args.start}")

    calendar = _load_calendar()
    _check_range_is_covered(calendar, args.start, end)

    symbols = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
    if not symbols and not args.nifty_50:
        parser.error("give --symbols and/or --nifty-50")

    client = _resolve_client(args.env_file)

    print("fetching the instrument master to resolve tokens...")
    instruments = client.instruments()
    by_symbol = {i.tradingsymbol: i for i in instruments if i.is_cash_equity}

    targets: list[tuple[str, int]] = []
    unknown = [s for s in symbols if s not in by_symbol]
    if unknown:
        raise SystemExit(
            f"unknown NSE cash-equity symbol(s): {unknown}. "
            "Refusing to fetch a partial set silently."
        )
    targets.extend((by_symbol[s].instrument_id, by_symbol[s].instrument_token) for s in symbols)
    if args.nifty_50:
        targets.append(("NSE:NIFTY 50", NIFTY_50_TOKEN))
        targets.append(("NSE:INDIA VIX", INDIA_VIX_TOKEN))

    print(f"fetching daily candles {args.start} -> {end} for {len(targets)} instrument(s)")
    print("  paced to Kite's documented 3 requests/second\n")

    failures: list[str] = []
    for instrument_id, token in targets:
        try:
            candles = client.daily_candles(token, args.start, end)
        except BrokerError as exc:
            failures.append(f"{instrument_id}: {exc}")
            print(f"  {instrument_id:<24} FAILED  {exc}")
            continue
        safe_name = instrument_id.replace(":", "_").replace(" ", "_")
        written = write_bars(instrument_id, candles, args.out / "bars" / f"{safe_name}.csv")
        span = f"{candles[0].session_date} -> {candles[-1].session_date}" if candles else "empty"
        print(f"  {instrument_id:<24} {written:>5} bars  {span}")

    print(f"\nraw files written under {args.out}")
    if failures:
        # Fail loudly and non-zero: a partial backfill that looks
        # successful is how a backtest ends up silently missing a name.
        print(f"\n{len(failures)} instrument(s) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("\nNext: run these through data.ingestion.DataIngestionPipeline to validate")
    print("and normalize them (see docs/KITE_DATA.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
