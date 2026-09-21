"""Re-price the paper book from Zerodha, act on its stops, update the dashboard.

    python scripts/refresh_live_book.py

Deliberately separate from ``collect_dashboard_data.py``. That script runs
the real stock selector, which takes minutes; this one reads the selection it
already produced and only asks Zerodha for prices, so it finishes in about a
second and can run every few minutes while the market is open.

What it does each time, in order:

1. Opens ``state/paper_book.json``, or creates it from the hypothetical
   sizing the collector produced if this is the first run.
2. Fetches quotes for everything held *and* every name in the current
   ranking -- a candidate's price is needed before it can be bought, not
   after.
3. Closes any position whose stop fired (risk/stop_loss.py), crediting the
   net proceeds to cash.
4. If the stops have taken the book down to ``paper_book.rerank_at_positions``
   holdings or fewer, recomputes the ranking and the regime. This is the one
   slow step -- minutes, not a second -- and is rate-limited and locked.
5. Spends the free cash on the best-ranked candidates not already held,
   never reaching below ``paper_book.max_rank_to_buy``.
6. Writes the book back and pushes the marked payload to the dashboard.

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
from scripts._rerank import (  # noqa: E402
    RerankUnavailable,
    latest_session,
    rerank_lock,
    rerun_ranking,
)

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


def buy_candidates(picks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The names this account may open a position in, best rank first.

    The current ranking and nothing else. ``selection.bench`` -- the names
    just outside it -- is carried in the payload for display and is
    deliberately *not* here: rank 11 is a name the selector considered and
    did not choose, and an account that reaches for it when a stop frees
    cash ends up holding what the strategy rejected. When every ranked name
    is held or blocked the cash waits, and if the book runs far enough down
    the ranking is recomputed instead (see ``paper_book.rerank_at_positions``).
    """
    ordered = [row for row in picks if row.get("instrument_id")]
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
    parser.add_argument(
        "--force-rerank",
        action="store_true",
        help=(
            "recompute the ranking and the regime now, whatever the book's size "
            "and the cooldown (takes minutes). Entries still only happen while "
            "the market is open."
        ),
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
    book_config = settings.paper_book

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

    def fetch_quotes(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        instrument_ids = sorted(
            {p.instrument_id for p in book.positions.values()}
            | {str(row["instrument_id"]) for row in candidates}
        )
        return live_quotes(instrument_ids, REPO_ROOT / "state" / "kite_session.json")

    candidates = buy_candidates(picks)
    quotes = fetch_quotes(candidates)
    if not quotes.get("available"):
        raise SystemExit(f"no live quotes: {quotes.get('reason')}")

    is_open = market_is_open(ist_now)
    closed_now, opened_now = [], []
    if not args.no_trade:
        quote_rows = quotes["quotes"]
        # Exits are always allowed, entries only while the market is open.
        # The asymmetry is deliberate: a quote after the close is the closing
        # print, which is a real price a resting stop could have sold into,
        # so acting on a breach is honest. Opening a *new* position on it is
        # not -- the account would be buying at a price that is no longer
        # available, and the cash is better left idle until the next session
        # prices the bench properly.
        ranked_ids = [str(row["instrument_id"]) for row in candidates]
        # Grace granted by an earlier ranking expires on a later session and
        # nothing else would notice, so this runs every refresh, before the
        # stops: a position due to go today goes at today's price, and is not
        # first re-examined against a stop level it no longer needs.
        closed_now = book.apply_scheduled_exits(
            ranked_ids, quote_rows, costs=costs, today=today
        )
        closed_now += book.apply_stops(
            quote_rows, policy=policy, costs=costs, today=today
        )

        # A book the stops have run down is mostly cash held against a
        # ranking several stops old. Recompute it rather than reach further
        # down the stale one -- which is the whole reason max_rank_to_buy
        # exists and why there is nothing else to buy by this point.
        wants_rerank = args.force_rerank or (
            is_open
            and book.needs_rerank(
                at_or_below=book_config.rerank_at_positions,
                now=ist_now.replace(tzinfo=None),
                cooldown_minutes=book_config.rerank_cooldown_minutes,
            )
        )
        if wants_rerank:
            with rerank_lock(REPO_ROOT / "state" / "rerank.lock") as acquired:
                if not acquired:
                    print("  rerank already running elsewhere; skipping")
                else:
                    why = (
                        "forced"
                        if args.force_rerank
                        else f"book down to {len(book.positions)} positions "
                        f"(<= {book_config.rerank_at_positions})"
                    )
                    print(
                        f"  {why}; recomputing the ranking and the regime "
                        "-- this takes minutes"
                    )
                    try:
                        fresh = rerun_ranking(latest_session(today))
                    except RerankUnavailable as exc:
                        print(f"  rerank failed, keeping the old ranking: {exc}")
                    else:
                        selection = fresh["selection"]
                        picks = selection.get("picks") or picks
                        data["selection"] = selection
                        data["regime_now"] = fresh["regime_now"]
                        candidates = buy_candidates(picks)
                        ranked_ids = [str(row["instrument_id"]) for row in candidates]
                        book.last_rerank_at = ist_now.replace(tzinfo=None).isoformat(
                            timespec="seconds"
                        )
                        refreshed = fetch_quotes(candidates)
                        if refreshed.get("available"):
                            quotes, quote_rows = refreshed, refreshed["quotes"]
                        print(
                            "  new top "
                            f"{len(picks)}: "
                            + ", ".join(str(row["symbol"]) for row in picks)
                        )
                        # Reconcile what is held against what was just
                        # computed: keep the names it still backs, sell the
                        # ones it dropped that are losing, give the ones it
                        # dropped that are winning the rest of the day.
                        closed_now += book.apply_ranking(
                            ranked_ids, quote_rows, costs=costs, today=today
                        )
                        graced = [
                            p.symbol
                            for p in book.positions.values()
                            if p.ranking_exit_after == today.isoformat()
                        ]
                        if graced:
                            print(
                                "  dropped from the ranking but in profit, held to the "
                                f"close and sold next session: {', '.join(sorted(graced))}"
                            )

        if is_open:
            slot = book.budget / max_positions if max_positions else book.budget
            opened_now = book.reallocate(
                candidates,
                quote_rows,
                costs=costs,
                today=today,
                slot=slot,
                max_positions=max_positions,
                max_rank=book_config.max_rank_to_buy,
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
        why = (
            "market closed"
            if not is_open
            else f"every name in the top {book_config.max_rank_to_buy} is held or blocked"
        )
        print(f"  cash {book.cash:,.0f} idle: {why}")

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

    state = "OPEN" if is_open else "closed"
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
