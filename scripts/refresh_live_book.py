"""Re-price the current picks from Zerodha and update the dashboard payload.

    python scripts/refresh_live_book.py

Deliberately separate from ``collect_dashboard_data.py``. That script runs
the real stock selector, which takes minutes; this one reads the selection
it already produced and only asks Zerodha for prices, so it finishes in
about a second and can run every few minutes while the market is open.

The published page cannot fetch these itself: the artifact sandbox only
permits a handful of script and font hosts, api.kite.trade is not among
them, and putting an access token in a shared page would hand anyone who
opened it the ability to trade. So quotes are fetched here, where the
token already lives, and pushed to the page's data channel.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from scripts._dashboard_live import live_book, live_quotes  # noqa: E402

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)


def market_is_open(now_ist: dt.datetime) -> bool:
    """NSE regular session, weekdays only.

    Holidays are not checked here on purpose -- a quote fetched on a holiday
    simply returns the previous close, which is harmless, and consulting the
    calendar would make a one-second script load configuration it otherwise
    does not need.
    """
    return now_ist.weekday() < 5 and MARKET_OPEN <= now_ist.time() <= MARKET_CLOSE


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "state" / "dashboard_data.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "state" / "dashboard_db.json")
    args = parser.parse_args(argv[1:])

    if not args.data.is_file():
        raise SystemExit(f"missing {args.data}; run scripts/collect_dashboard_data.py first")
    data = json.loads(args.data.read_text(encoding="utf-8"))

    picks = (data.get("selection") or {}).get("picks") or []
    if not picks:
        raise SystemExit("no selection in the payload to price")

    quotes = live_quotes(
        [p["instrument_id"] for p in picks], REPO_ROOT / "state" / "kite_session.json"
    )
    if not quotes.get("available"):
        raise SystemExit(f"no live quotes: {quotes.get('reason')}")

    book = live_book(data.get("paper_sizing") or {}, quotes, picks)
    data["live"], data["live_book"] = quotes, book
    data["generated_at"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    args.data.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Only the fast-moving fields. The full payload with trade history is
    # well past the data channel's document limit, and the page merges
    # rather than replaces, so omitting the rest leaves it intact.
    slim = {k: data[k] for k in ("generated_at", "live", "live_book", "paper_sizing") if k in data}
    args.out.write_text(
        json.dumps({"payload": json.dumps(slim, separators=(",", ":")),
                    "updated_at": data["generated_at"]}),
        encoding="utf-8",
    )

    ist = dt.datetime.now(dt.UTC) + dt.timedelta(hours=5, minutes=30)
    state = "OPEN" if market_is_open(ist) else "closed"
    if book.get("available"):
        print(f"{ist:%H:%M} IST  market {state}  "
              f"{book['winners']}W/{book['losers']}L  "
              f"value {book['market_value']:,.0f}  "
              f"P&L {book['pnl']:+,.0f} ({book['pnl_pct']*100:+.2f}%)")
    else:
        print(f"{ist:%H:%M} IST  market {state}  book unavailable: {book.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
