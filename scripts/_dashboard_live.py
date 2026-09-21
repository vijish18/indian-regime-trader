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
        prev = float((row.get("ohlc") or {}).get("close") or last)
        quotes[instrument_id] = {
            "last": last,
            "prev_close": prev,
            "day_pct": (last / prev - 1) if prev else 0.0,
        }
    return {
        "available": bool(quotes),
        "fetched_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "quotes": quotes,
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
        "hypothetical": True,
    }
