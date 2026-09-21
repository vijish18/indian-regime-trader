"""Write NIFTY 50 and India VIX history into the store, for the feature pipeline.

    python scripts/kite_login.py            # once each morning
    python scripts/build_index_observations.py --from 2015-01-01

The regime features are computed from these two series alone
(``core/features/feature_engineering.py``), so this is what the HMM
actually trains on. Equity bars come from bhavcopy; index levels do not
appear there, so they come from Kite.

**Volume is written as empty, not zero.** Kite reports ``volume = 0`` for
an index, because an index has no traded quantity of its own.
``MarketFeatureInputs`` decides whether to include the volume-stress
feature by testing ``isna()``, so a column of zeros would count as real
volume and the feature -- a rolling z-score of ``log(volume)`` -- would be
``log(0) = -inf`` for every row. That does not raise. It quietly poisons
the feature matrix. Empty makes the pipeline correctly omit the feature
and fit on the remaining eight, which its own docstring calls an expected
and valid case.

**Sessions the calendar does not recognise are dropped**, with the dates
printed. Kite returns bars for Muhurat and for NSE's Saturday special
sessions (Union Budget days, disaster-recovery drills); this system
trades neither, and splicing a Saturday between Friday and Monday
distorts every rolling window that spans it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from broker.zerodha.kite_historical import (  # noqa: E402
    INDIA_VIX_TOKEN,
    NIFTY_50_TOKEN,
    KiteHistoricalClient,
)
from broker.zerodha.kite_session import KiteSessionError, load_session  # noqa: E402
from data.calendar import NSETradingCalendar  # noqa: E402

HOLIDAY_FILE = REPO_ROOT / "config" / "nse_holidays.csv"
INDEX_OUT = Path(os.environ.get("IRT_DATA_ROOT", REPO_ROOT / "data_cache")) / "raw" / "index"

NIFTY_SYMBOL = "NIFTY50"
VIX_SYMBOL = "INDIAVIX"
"""Store-side symbols. Deliberately without the space Kite uses ("NIFTY
50"), because these become filenames and a space in a path is a
long-running source of quoting bugs. The Kite token is the identifier
that matters; this is only a label."""

HEADER = ("index_symbol", "session_date", "open", "high", "low", "close", "volume")


def write_series(
    symbol: str,
    token: int,
    client: KiteHistoricalClient,
    calendar: NSETradingCalendar,
    start: dt.date,
    end: dt.date,
    out_dir: Path,
) -> tuple[int, list[dt.date]]:
    candles = client.daily_candles(token, start, end)
    kept = []
    dropped = []
    for candle in candles:
        if calendar.is_trading_day(candle.session_date):
            kept.append(candle)
        else:
            dropped.append(candle.session_date)

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{symbol}.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        for candle in kept:
            writer.writerow(
                [
                    symbol,
                    candle.session_date.isoformat(),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    "",  # volume: absent, not zero -- see the module docstring
                ]
            )
    return len(kept), dropped


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=dt.date.fromisoformat,
                        default=dt.date(2015, 1, 1))
    parser.add_argument("--to", dest="end", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--out", type=Path, default=INDEX_OUT)
    args = parser.parse_args(argv[1:])

    end = args.end or dt.date.today()
    calendar = NSETradingCalendar.from_file(HOLIDAY_FILE)
    missing = sorted(set(range(args.start.year, end.year + 1)) - calendar.covered_years)
    if missing:
        raise SystemExit(
            f"the trading calendar does not cover {missing}. "
            "Run: python scripts/build_nse_holidays.py --reconcile-with-kite"
        )

    try:
        session = load_session()
    except KiteSessionError as exc:
        raise SystemExit(str(exc)) from exc
    client = KiteHistoricalClient(session.api_key, access_token=session.access_token)

    for symbol, token in ((NIFTY_SYMBOL, NIFTY_50_TOKEN), (VIX_SYMBOL, INDIA_VIX_TOKEN)):
        print(f"fetching {symbol} ({args.start} -> {end})...")
        count, dropped = write_series(
            symbol, token, client, calendar, args.start, end, args.out
        )
        print(f"  {count:,} sessions -> {args.out / f'{symbol}.csv'}")
        if dropped:
            print(f"  dropped {len(dropped)} session(s) the calendar does not recognise:")
            for day in dropped:
                print(f"    {day} ({day:%a})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
