"""Re-price the paper book from Zerodha, act on its stops, update the dashboard.

    python scripts/refresh_live_book.py

Deliberately separate from ``collect_dashboard_data.py``. That script runs
the real stock selector, which takes minutes; this one reads the selection it
already produced and only asks Zerodha for prices, so it finishes in about a
second and can run every few minutes while the market is open.

What it does each time, in order:

1. Opens ``state/paper_book.json``, or creates it from the hypothetical
   sizing the collector produced if this is the first run.
2. Fetches quotes for everything held *and* everything on the bench -- the
   bench prices are needed before a replacement can be bought, not after.
3. Closes any position whose stop fired (risk/stop_loss.py), crediting the
   net proceeds to cash.
4. Spends the free cash on the best-ranked candidates not already held.
5. Writes the book back and pushes the marked payload to the dashboard.

Steps 3 and 4 make this the only process that changes the account. It is
still paper: no order reaches a broker, and ``execution.mode`` stays
``paper``. What is no longer pretend is the bookkeeping -- entry prices,
cost bases, realised P&L and the cash balance are a ledger with a history,
not a snapshot recomputed from the current ranking.

The published page cannot fetch quotes itself: the artifact sandbox permits
only a handful of script and font hosts, api.kite.trade is not among them,
and putting an access token in a shared page would hand anyone who opened it
the ability to trade. So quotes are fetched here, where the token already
lives, and pushed to the page's data channel.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from backtest.cost_schedule import CostScheduleRepository  # noqa: E402
from backtest.costs import CostModel  # noqa: E402
from config.loader import load_settings  # noqa: E402
from execution.paper_book import PaperBook  # noqa: E402
from risk.stop_loss import StopLossPolicy  # noqa: E402
from scripts._dashboard_live import book_view, live_quotes  # noqa: E402

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)
IST = dt.timedelta(hours=5, minutes=30)


def market_is_open(now_ist: dt.datetime) -> bool:
    """NSE regular session, weekdays only.

    Holidays are not checked here on purpose -- a quote fetched on a holiday
    simply returns the previous close, which is harmless, and consulting the
    calendar would make a one-second script load configuration it otherwise
    does not need.
    """
    return now_ist.weekday() < 5 and MARKET_OPEN <= now_ist.time() <= MARKET_CLOSE


def build_cost_model(settings: object) -> CostModel:
    return CostModel(
        CostScheduleRepository.from_file(REPO_ROOT / "config" / "cost_schedules.yaml"),
        min_slippage_bps=settings.backtest.slippage_min_bps,  # type: ignore[attr-defined]
        impact_coefficient=settings.backtest.slippage_impact_coefficient,  # type: ignore[attr-defined]
    )


def candidate_bench(
    selection: dict[str, Any], picks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The ranking a replacement is drawn from, best first.

    ``selection.bench`` carries ranks beyond the book's own ten when the
    collector was run with one. Without it the only candidates are the ten
    already held, so a stop frees cash that has nowhere to go -- which is a
    real state of the account and is reported rather than papered over.
    """
    bench = selection.get("bench") or []
    seen, ordered = set(), []
    for row in [*picks, *bench]:
        instrument_id = row.get("instrument_id")
        if not instrument_id or instrument_id in seen:
            continue
        seen.add(instrument_id)
        ordered.append(row)
    ordered.sort(key=lambda r: r.get("rank") or 10_000)
    return ordered


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "state" / "dashboard_data.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "state" / "dashboard_db.json")
    parser.add_argument("--book", type=Path, default=REPO_ROOT / "state" / "paper_book.json")
    parser.add_argument(
        "--no-trade",
        action="store_true",
        help="re-price only: do not close stopped positions or redeploy cash",
    )
    args = parser.parse_args(argv[1:])

    if not args.data.is_file():
        raise SystemExit(f"missing {args.data}; run scripts/collect_dashboard_data.py first")
    data = json.loads(args.data.read_text(encoding="utf-8"))

    selection = data.get("selection") or {}
    picks = selection.get("picks") or []
    if not picks:
        raise SystemExit("no selection in the payload to price")

    settings = load_settings()
    costs = build_cost_model(settings)
    policy = StopLossPolicy.from_mapping(settings.risk.stop_loss.model_dump())
    max_positions = settings.selection.max_holdings

    ist_now = dt.datetime.now(dt.UTC) + IST
    today = ist_now.date()

    sizing = data.get("paper_sizing") or {}
    book = PaperBook.load(args.book)
    if book is None:
        if not sizing.get("available"):
            raise SystemExit("no paper sizing to open a book from")
        seed_date = dt.date.fromisoformat(selection.get("as_of") or today.isoformat())
        book = PaperBook.seed(
            budget=float(sizing["capital"]),
            sizing_positions=sizing.get("positions") or [],
            picks=picks,
            costs=costs,
            as_of=seed_date,
        )
        print(
            f"opened paper book: {len(book.positions)} positions, "
            f"cash {book.cash:,.0f} of {book.budget:,.0f}"
        )
    book.roll_session(today)

    bench = candidate_bench(selection, picks)
    instrument_ids = sorted(
        {p.instrument_id for p in book.positions.values()}
        | {str(row["instrument_id"]) for row in bench}
    )
    quotes = live_quotes(instrument_ids, REPO_ROOT / "state" / "kite_session.json")
    if not quotes.get("available"):
        raise SystemExit(f"no live quotes: {quotes.get('reason')}")

    closed_now, opened_now = [], []
    if not args.no_trade:
        quote_rows = quotes["quotes"]
        closed_now = book.apply_stops(quote_rows, policy=policy, costs=costs, today=today)
        slot = book.budget / max_positions if max_positions else book.budget
        opened_now = book.reallocate(
            bench,
            quote_rows,
            costs=costs,
            today=today,
            slot=slot,
            max_positions=max_positions,
        )
        book.updated_at = ist_now.isoformat(timespec="seconds")
        book.save(args.book)

    for trade in closed_now:
        print(
            f"  CLOSED {trade.symbol:<12} {trade.exit_reason:<21} "
            f"{trade.shares} @ {trade.exit_price:,.2f}  "
            f"net {trade.net_pnl:+,.0f} ({trade.net_pnl_pct * 100:+.2f}%)"
        )
    for position in opened_now:
        print(
            f"  BOUGHT {position.symbol:<12} rank {position.rank_at_entry:<3} "
            f"{position.shares} @ {position.entry_price:,.2f}  "
            f"cost {position.entry_cost:,.0f}"
        )
    if closed_now and not opened_now and book.cash > 0:
        print(f"  cash {book.cash:,.0f} idle: no ranked candidate available to buy")

    book_payload = book_view(book, quotes, policy, costs)
    data["live"] = quotes
    data["live_book"] = book_payload
    data["realized"] = book.realized()

    # Intraday ticks from the background poller, if it is running. Thinned
    # to a fixed point count AND stripped to two decimals: the published
    # document has a 262 KB limit and every refresh adds a full session's
    # worth of ticks, so an unthinned payload grows past it before the close
    # and the push simply starts failing. A 74-pixel sparkline cannot
    # resolve more than this anyway.
    ticks_path = REPO_ROOT / "state" / "live_ticks.json"
    if ticks_path.is_file():
        try:
            ticks = json.loads(ticks_path.read_text(encoding="utf-8")).get("ticks", [])
        except (OSError, ValueError):
            ticks = []
        step = max(1, len(ticks) // 90)
        thinned = ticks[::step][-90:]
        data["ticks"] = {
            "updated_at": ticks[-1]["t"] if ticks else None,
            "points": [
                {"hhmm": p["hhmm"], "px": {k: round(v, 2) for k, v in p["px"].items()}}
                for p in thinned
            ],
        }
    data["generated_at"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    args.data.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")

    # Only the fast-moving fields. The full payload with backtest history is
    # well past the data channel's document limit, and the page merges
    # rather than replaces, so omitting the rest leaves it intact.
    slim = {
        k: data[k]
        for k in ("generated_at", "live", "live_book", "realized", "paper_sizing", "ticks")
        if k in data
    }
    args.out.write_text(
        json.dumps(
            {
                "payload": json.dumps(slim, separators=(",", ":"), allow_nan=False),
                "updated_at": data["generated_at"],
            }
        ),
        encoding="utf-8",
    )

    state = "OPEN" if market_is_open(ist_now) else "closed"
    realized = data["realized"]
    if book_payload.get("available"):
        print(
            f"{ist_now:%H:%M} IST  market {state}  "
            f"{book_payload['winners']}W/{book_payload['losers']}L  "
            f"equity {book_payload['equity']:,.0f}  "
            f"open {book_payload['pnl']:+,.0f} ({book_payload['pnl_pct'] * 100:+.2f}%)  "
            f"realised {realized['net_pnl']:+,.0f} over {realized['trades']} closed"
        )
    else:
        print(f"{ist_now:%H:%M} IST  market {state}  book unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
