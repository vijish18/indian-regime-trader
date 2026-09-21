"""Live quotes and trade analytics for the dashboard.

Kept apart from ``collect_dashboard_data`` because these two are the only
parts that reach outside the repository -- one to Zerodha for a price, one
to the trade logs for outcomes -- and both fail in ways the rest does not.

**On "live P&L" before anything is held.** Paper trading has no entry point
yet, so there are no positions. What can honestly be shown is the current
ranking priced live: if the book had been entered at the selection date's
close, this is what it would be worth now. That is a hypothetical and is
labelled as one; it becomes real holdings the moment a paper session runs.

**On trade statistics.** A fold boundary flattens the book -- a retrain is
a flatten-and-reassess point -- so a fold's final holdings are dropped
rather than sold and never generate a fill. Measured on the HMM run, 220,468
of 2,138,185 bought shares never appear as a sale. Closed round trips
therefore cover only the positions the strategy chose to exit, which biases
them toward winners: they show a profit factor of 1.68 on a strategy whose
equity curve lost money. Both numbers are reported, from their own source,
and never added together.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

QUOTE_URL = "https://api.kite.trade/quote"


def live_quotes(instrument_ids: list[str], session_path: Path) -> dict[str, Any]:
    """Last traded price per instrument, or a stated reason there is none."""
    if not instrument_ids:
        return {"available": False, "reason": "no instruments"}
    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
        api_key, token = session["api_key"], session["access_token"]
        expires = dt.datetime.fromisoformat(session["expires_at"])
    except (OSError, ValueError, KeyError) as exc:
        return {"available": False, "reason": f"no usable Kite session ({exc})"}
    if expires <= dt.datetime.now(dt.UTC):
        return {"available": False, "reason": f"Kite session expired {expires.isoformat()}"}

    query = "&".join(f"i={urllib.parse.quote(i)}" for i in instrument_ids)
    request = urllib.request.Request(
        f"{QUOTE_URL}?{query}",
        headers={"X-Kite-Version": "3", "Authorization": f"token {api_key}:{token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            payload = json.loads(response.read())["data"]
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    quotes = {}
    for instrument_id in instrument_ids:
        row = payload.get(instrument_id)
        if not row:
            continue
        last = float(row["last_price"])
        ohlc = row.get("ohlc") or {}
        prev = float(ohlc.get("close") or last)
        # Kite's `ohlc` is TODAY's open/high/low plus the PREVIOUS close --
        # the high and low are exactly what the trailing stop is measured
        # from, and they are running values, so they are correct at the
        # moment they are read rather than only after the close.
        quotes[instrument_id] = {
            "last": last,
            "prev_close": prev,
            "day_open": float(ohlc.get("open") or last),
            "day_high": float(ohlc.get("high") or last),
            "day_low": float(ohlc.get("low") or last),
            "day_pct": (last / prev - 1) if prev else 0.0,
        }
    return {
        "available": bool(quotes),
        "fetched_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "quotes": quotes,
    }


def _stop_policy_and_costs() -> tuple[Any, Any]:
    """The configured stop policy and a cost model, or ``(None, None)``.

    Imported lazily and failing soft on purpose: this is a monitoring script
    that runs every few minutes while the market is open, and a config
    problem should cost the stop column, not the whole price refresh. The
    backtest path has the opposite rule -- there a missing policy is a
    configuration error and fails loudly.
    """
    try:
        from backtest.cost_schedule import CostScheduleRepository
        from backtest.costs import CostModel
        from config.loader import load_settings
        from risk.stop_loss import StopLossPolicy

        settings = load_settings()
        policy = StopLossPolicy.from_mapping(settings.risk.stop_loss.model_dump())
        costs = CostModel(
            CostScheduleRepository.from_file(
                Path(__file__).resolve().parents[1] / "config" / "cost_schedules.yaml"
            ),
            min_slippage_bps=settings.backtest.slippage_min_bps,
            impact_coefficient=settings.backtest.slippage_impact_coefficient,
        )
        return policy, costs
    except Exception as exc:  # noqa: BLE001 - monitoring must not die on config
        print(f"stop status unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None, None


def stop_status(
    *,
    symbol: str,
    entry: float,
    shares: int,
    last: float,
    day_high: float,
    day_low: float,
    policy: Any,
    costs: Any,
) -> dict[str, Any]:
    """Where this holding stands against both stop rules, right now.

    The live counterpart to ``risk.stop_loss.evaluate``. It answers the same
    two questions with the same thresholds, but reports rather than sells:
    nothing here places an order, and the paper book is not flattened by a
    breach. What it gives the dashboard is the level, the distance to it,
    and whether it has already traded today.

    Unlike the backtest, the trailing rule can use the day's low as well as
    the last price, because a live running high is known at the moment it is
    read -- there is no question of which extreme came first when both are
    read from the same snapshot. The backtest cannot do this; see
    risk/stop_loss.py, "What a daily bar proves".
    """
    from backtest.costs import TradeSide
    from risk.stop_loss import hard_stop_level, trailing_stop_level

    if policy is None or costs is None or entry <= 0 or shares <= 0:
        return {"available": False}

    def net_sale(price: float) -> float:
        return float(
            costs.estimate_execution_cost(
                symbol,
                TradeSide.SELL,
                shares,
                price,
                dt.date.today(),
                spread_bps=10.0,
                avg_daily_value=0.0,
                volatility=0.0,
            ).net_value
        )

    basis = entry * shares
    hard = hard_stop_level(entry, policy.hard_stop_pct)
    trail = trailing_stop_level(day_high, policy.trail_drop_pct) if day_high > 0 else 0.0
    net_now = (net_sale(last) / basis) - 1.0 if basis > 0 else 0.0

    hard_hit = day_low > 0 and day_low <= hard
    trail_armed = trail > 0 and net_now > policy.trail_arm_net_profit_pct
    trail_hit = trail_armed and last <= trail

    if hard_hit:
        state = "hard_stop_hit"
    elif trail_hit:
        state = "trail_hit"
    elif trail_armed:
        state = "trail_armed"
    else:
        state = "ok"

    return {
        "available": True,
        "state": state,
        "hard_level": hard,
        "hard_distance_pct": (last / hard - 1) if hard > 0 else 0.0,
        "trail_level": trail if trail_armed else None,
        "trail_distance_pct": (last / trail - 1) if trail_armed and trail > 0 else None,
        "net_profit_pct": net_now,
        "arm_threshold_pct": policy.trail_arm_net_profit_pct,
    }


def live_book(
    sizing: dict[str, Any], quotes: dict[str, Any], picks: list[dict[str, Any]]
) -> dict[str, Any]:
    """The proposed book marked at live prices.

    Entry is the selection date's close, because that is the price the
    ranking was computed from and the one a session opening tomorrow would
    have acted on. Nothing is actually held, so this is what the book would
    be worth, not what it is worth.
    """
    if not sizing.get("available") or not quotes.get("available"):
        return {"available": False, "reason": "needs both sizing and live quotes"}

    by_symbol = {p["symbol"]: p for p in picks}
    policy, costs = _stop_policy_and_costs()
    rows, cost_basis, market_value = [], 0.0, 0.0
    for position in sizing["positions"]:
        pick = by_symbol.get(position["symbol"])
        if pick is None or position["shares"] == 0:
            continue
        quote = quotes["quotes"].get(pick["instrument_id"])
        if quote is None:
            continue
        entry, shares = position["price"], position["shares"]
        value = shares * quote["last"]
        basis = shares * entry
        cost_basis += basis
        market_value += value
        rows.append(
            {
                "symbol": position["symbol"],
                "rank": position["rank"],
                "shares": shares,
                "entry": entry,
                "last": quote["last"],
                "value": value,
                "pnl": value - basis,
                "pnl_pct": (quote["last"] / entry - 1) if entry else 0.0,
                "day_pct": quote["day_pct"],
                "day_high": quote.get("day_high"),
                "day_low": quote.get("day_low"),
                "stop": stop_status(
                    symbol=pick["instrument_id"],
                    entry=entry,
                    shares=shares,
                    last=quote["last"],
                    day_high=quote.get("day_high") or quote["last"],
                    day_low=quote.get("day_low") or quote["last"],
                    policy=policy,
                    costs=costs,
                ),
            }
        )
    rows.sort(key=lambda r: r["pnl_pct"], reverse=True)
    return {
        "available": bool(rows),
        "fetched_at": quotes.get("fetched_at"),
        "positions": rows,
        "cost_basis": cost_basis,
        "market_value": market_value,
        "pnl": market_value - cost_basis,
        "pnl_pct": (market_value / cost_basis - 1) if cost_basis else 0.0,
        "winners": sum(1 for r in rows if r["pnl"] > 0),
        "losers": sum(1 for r in rows if r["pnl"] < 0),
        "stops": {
            "available": policy is not None,
            "hard_stop_pct": policy.hard_stop_pct if policy else None,
            "trail_drop_pct": policy.trail_drop_pct if policy else None,
            "trail_arm_net_profit_pct": policy.trail_arm_net_profit_pct if policy else None,
            # Counted, not acted on. Nothing in this script sells; a breach
            # here says the level traded, and the decision to exit is still
            # the paper session's to make.
            "hard_hit": sum(1 for r in rows if (r["stop"] or {}).get("state") == "hard_stop_hit"),
            "trail_hit": sum(1 for r in rows if (r["stop"] or {}).get("state") == "trail_hit"),
            "trail_armed": sum(
                1 for r in rows if (r["stop"] or {}).get("state") == "trail_armed"
            ),
        },
        "hypothetical": True,
    }


def book_view(book: Any, quotes: dict[str, Any], policy: Any, costs: Any) -> dict[str, Any]:
    """The persistent paper book marked at live prices.

    Replaces the hypothetical ``live_book`` once an account exists. The
    difference is not cosmetic: entry is the price the position was really
    bought at, cash is a balance that moves, and a name that was stopped out
    is gone from here and present in the realised section instead.
    """
    rows, cost_basis, market_value = [], 0.0, 0.0
    quote_rows = quotes.get("quotes") or {}
    for position in book.positions.values():
        quote = quote_rows.get(position.instrument_id)
        if quote is None:
            continue
        value = position.shares * quote["last"]
        cost_basis += position.entry_cost
        market_value += value
        rows.append(
            {
                "symbol": position.symbol,
                "rank": position.rank_at_entry,
                "shares": position.shares,
                "entry": position.entry_price,
                "entry_date": position.entry_date,
                "last": quote["last"],
                "value": value,
                # Against cash paid, not shares x price: the position has to
                # earn back its own buy charges before it is ahead.
                "pnl": value - position.entry_cost,
                "pnl_pct": (value / position.entry_cost - 1.0)
                if position.entry_cost
                else 0.0,
                "day_pct": quote["day_pct"],
                "day_high": quote.get("day_high"),
                "day_low": quote.get("day_low"),
                # The extremes the stop actually used -- since entry, not
                # since the opening bell. For a position bought mid-session
                # these differ, and showing the session's own high/low here
                # would contradict the ledger's own decision.
                "held_high": position.session_high or None,
                "held_low": position.session_low or None,
                "stop": stop_status(
                    symbol=position.instrument_id,
                    entry=position.entry_price,
                    shares=position.shares,
                    last=quote["last"],
                    day_high=position.session_high or quote["last"],
                    day_low=position.session_low or quote["last"],
                    policy=policy,
                    costs=costs,
                ),
            }
        )
    rows.sort(key=lambda r: r["pnl_pct"], reverse=True)
    equity = market_value + book.cash
    return {
        "available": bool(rows) or bool(book.closed),
        "fetched_at": quotes.get("fetched_at"),
        "positions": rows,
        "cost_basis": cost_basis,
        "market_value": market_value,
        "cash": book.cash,
        "budget": book.budget,
        "equity": equity,
        "equity_pct": (equity / book.budget - 1.0) if book.budget else 0.0,
        "pnl": market_value - cost_basis,
        "pnl_pct": (market_value / cost_basis - 1.0) if cost_basis else 0.0,
        "winners": sum(1 for r in rows if r["pnl"] > 0),
        "losers": sum(1 for r in rows if r["pnl"] < 0),
        "opened_at": book.opened_at,
        "blocked_today": sorted(book.blocked_today),
        "stops": {
            "available": policy is not None,
            "hard_stop_pct": policy.hard_stop_pct if policy else None,
            "trail_drop_pct": policy.trail_drop_pct if policy else None,
            "trail_arm_net_profit_pct": policy.trail_arm_net_profit_pct if policy else None,
            "hard_hit": sum(1 for r in rows if (r["stop"] or {}).get("state") == "hard_stop_hit"),
            "trail_hit": sum(1 for r in rows if (r["stop"] or {}).get("state") == "trail_hit"),
            "trail_armed": sum(
                1 for r in rows if (r["stop"] or {}).get("state") == "trail_armed"
            ),
        },
        # The book is real now: these positions were bought and are held.
        # What remains simulated is the broker, not the bookkeeping.
        "hypothetical": False,
    }
