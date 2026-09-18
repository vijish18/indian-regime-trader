"""Turn a fill-by-fill trade log into closed round trips.

A trade log records fills, not outcomes: a buy row carries no idea whether
that position eventually made money. Answering "how many calls were right,
and what did they earn" means pairing each sell against the buys it closed,
which is what this module does.

**FIFO, and why it is not arbitrary.** Indian cash equity is delivery-based
and settled per scrip, so a sell removes the oldest shares held. Matching
last-in-first-out instead would report different holding periods and
different per-trip P&L from the same fills, while the portfolio total
stayed identical -- so the convention has to be stated rather than
defaulted into. The totals here reconcile with
``performance.PerformanceCalculator`` regardless of matching order; only
the attribution to individual trips depends on it.

**Costs are charged to the leg that incurred them.** A round trip carries
the entry fills' costs pro-rata for the quantity matched, plus the exit
fill's, so a trip's ``net_pnl`` is what the account actually kept. Summing
``net_pnl`` over all closed trips plus the marked value of what is still
open reproduces the strategy's realised P&L.

**Open positions are not round trips.** Shares still held at the end have
no exit price, and marking them at the last close would mix a realised
result with an unrealised one. They are returned separately.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict, deque
from dataclasses import dataclass

import pandas as pd

BUY = "buy"
SELL = "sell"


@dataclass(frozen=True, slots=True)
class RoundTrip:
    """One closed position: shares bought, later sold, net of both legs' costs."""

    instrument_id: str
    entry_date: dt.date
    exit_date: dt.date
    quantity: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    cost: float
    net_pnl: float

    @property
    def is_win(self) -> bool:
        """Net of costs, deliberately. A trip that gained on price and lost
        it to brokerage and STT was not a correct call in any sense that
        matters to the account."""
        return self.net_pnl > 0.0

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days

    @property
    def return_pct(self) -> float:
        basis = self.entry_price * self.quantity
        return self.net_pnl / basis if basis else 0.0


@dataclass(frozen=True, slots=True)
class OpenPosition:
    instrument_id: str
    entry_date: dt.date
    quantity: int
    entry_price: float


@dataclass(frozen=True, slots=True)
class TradeSummary:
    round_trips: int
    wins: int
    losses: int
    win_rate: float
    gross_profit: float
    """Sum of the winning trips' net P&L (net of their own costs)."""

    gross_loss: float
    """Absolute value of the losing trips' net P&L."""

    profit_factor: float
    net_pnl: float
    avg_win: float
    avg_loss: float
    avg_holding_days: float
    total_costs: float


@dataclass(slots=True)
class _Lot:
    """Shares bought and not yet sold. Mutable because a sell consumes part
    of one."""

    entry_date: dt.date
    remaining: float
    price: float
    cost_per_share: float


def _as_date(value: object) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(str(value)).date()


def round_trips(trade_log: pd.DataFrame) -> tuple[list[RoundTrip], list[OpenPosition]]:
    """Pair sells against earlier buys, per instrument, first in first out."""
    if trade_log.empty:
        return [], []

    required = {"execution_date", "instrument_id", "side", "quantity", "fill_price", "cost"}
    missing = required - set(trade_log.columns)
    if missing:
        raise ValueError(f"trade log is missing column(s): {sorted(missing)}")

    frame = trade_log.sort_values("execution_date", kind="stable")
    lots: dict[str, deque[_Lot]] = defaultdict(deque)
    closed: list[RoundTrip] = []

    for row in frame.to_dict("records"):
        instrument = str(row["instrument_id"])
        quantity = int(float(row["quantity"]))
        if quantity <= 0:
            continue
        price = float(row["fill_price"])
        cost_per_share = float(row["cost"]) / quantity if quantity else 0.0
        when = _as_date(row["execution_date"])

        if str(row["side"]).lower() == BUY:
            lots[instrument].append(_Lot(when, float(quantity), price, cost_per_share))
            continue

        # A sell consumes the oldest lots first.
        outstanding = float(quantity)
        while outstanding > 0 and lots[instrument]:
            lot = lots[instrument][0]
            matched = min(outstanding, lot.remaining)

            gross = (price - lot.price) * matched
            cost = (lot.cost_per_share + cost_per_share) * matched
            closed.append(
                RoundTrip(
                    instrument_id=instrument,
                    entry_date=lot.entry_date,
                    exit_date=when,
                    quantity=int(matched),
                    entry_price=lot.price,
                    exit_price=price,
                    gross_pnl=gross,
                    cost=cost,
                    net_pnl=gross - cost,
                )
            )
            lot.remaining -= matched
            outstanding -= matched
            if lot.remaining <= 0:
                lots[instrument].popleft()
        # A sell with nothing left to match cannot happen in a long-only
        # system; if it ever does, it is dropped rather than recorded as a
        # short, because this module must not invent a position the
        # portfolio never held.

    still_open = [
        OpenPosition(
            instrument_id=instrument,
            entry_date=lot.entry_date,
            quantity=int(lot.remaining),
            entry_price=lot.price,
        )
        for instrument, queue in lots.items()
        for lot in queue
        if lot.remaining > 0
    ]
    return closed, still_open


def summarize(trips: list[RoundTrip]) -> TradeSummary:
    """Headline accuracy and profitability over closed trips."""
    if not trips:
        return TradeSummary(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    wins = [t for t in trips if t.is_win]
    losses = [t for t in trips if not t.is_win]
    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    return TradeSummary(
        round_trips=len(trips),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(trips),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        # Infinite rather than a sentinel when nothing lost: a caller that
        # formats it will show the truth, and one that compares it will
        # rank a lossless strategy correctly.
        profit_factor=(gross_profit / gross_loss) if gross_loss else float("inf"),
        net_pnl=sum(t.net_pnl for t in trips),
        avg_win=(gross_profit / len(wins)) if wins else 0.0,
        avg_loss=(-gross_loss / len(losses)) if losses else 0.0,
        avg_holding_days=sum(t.holding_days for t in trips) / len(trips),
        total_costs=sum(t.cost for t in trips),
    )


def by_instrument(trips: list[RoundTrip]) -> pd.DataFrame:
    """Per-stock outcome: how often it was traded, how often that worked."""
    if not trips:
        return pd.DataFrame(
            columns=["instrument_id", "round_trips", "wins", "win_rate", "net_pnl", "cost"]
        )
    rows: dict[str, dict[str, float]] = defaultdict(
        lambda: {"round_trips": 0.0, "wins": 0.0, "net_pnl": 0.0, "cost": 0.0}
    )
    for trip in trips:
        row = rows[trip.instrument_id]
        row["round_trips"] += 1
        row["wins"] += 1 if trip.is_win else 0
        row["net_pnl"] += trip.net_pnl
        row["cost"] += trip.cost
    frame = pd.DataFrame(
        [
            {
                "instrument_id": instrument,
                "round_trips": int(values["round_trips"]),
                "wins": int(values["wins"]),
                "win_rate": values["wins"] / values["round_trips"],
                "net_pnl": values["net_pnl"],
                "cost": values["cost"],
            }
            for instrument, values in rows.items()
        ]
    )
    return frame.sort_values("net_pnl", ascending=False, ignore_index=True)
