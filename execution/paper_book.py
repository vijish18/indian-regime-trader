"""The paper account's durable ledger: what is held, what was closed, what cash is free.

Until now the dashboard's "live book" was recomputed from scratch on every
refresh: take the current ranking, price it at the selection date's close,
mark it live. That is a *hypothetical* -- it answers "what would this be
worth" and it cannot answer "what happened", because nothing persisted
between refreshes. A stop that fired at 11:40 left no trace by 11:45.

This is the other thing. It is a ledger: positions with an entry price and a
cost basis, closed trades with a realised P&L, and a cash balance that goes
up when something is sold and down when something is bought. It is written
to disk after every change, so the account has a history rather than a
snapshot.

## What closes a position

Only a stop, for now. ``apply_stops`` runs both rules from
:mod:`risk.stop_loss` against live quotes and closes whatever breached.
There is no ranking-driven exit here: a name dropping out of the top ten is
a *rebalance* decision that belongs to a trading session, not to a price
refresh running every few minutes.

## What opens one

Free cash, and only free cash, and only on a name the strategy currently
ranks. ``reallocate`` spends what the closures released on the best
candidate that is not already held, and ``max_rank`` caps how far down the
ranking it may reach -- set to the book's own size, it means the account
never holds a name the selector did not pick.

That cap has a cost, and it is the point: when every ranked name is already
held, a stop frees cash that has nowhere to go and it sits idle. The
alternative is buying rank 11, 12, 13 -- names the strategy considered and
did not choose. Idle cash earns nothing; an unranked position can lose.

When the book has shrunk far enough that most of it is cash, the answer is
not to reach further down a stale ranking but to compute a new one. See
``needs_rerank``.

**A name stopped out today cannot be re-bought today.** Without that rule a
position stopped at -3% is immediately re-entered a few paise lower, the
stop fires again, and the account grinds its capital into transaction costs
on a single falling stock. The block lifts at the next session, when the
ranking has been recomputed against a new day's prices.

## Prices

Entries and exits both transact at prices that were really available. A buy
fills at the live last traded price, because that is what a market order
would pay at the moment cash is redeployed. An exit fills at
``risk.stop_loss.stop_fill_price`` -- the worse of the stop level and the
last price -- because a stop that gapped through never offered its level.

Costs are the full :class:`~backtest.costs.CostModel` breakdown on both
legs, DP charge included. A position's ``entry_cost`` is cash out of the
door and its ``net_proceeds`` is cash back in, so realised P&L is the
difference between two cash amounts and never a price ratio dressed up as
one.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from backtest.costs import CostModel, TradeSide
from risk.stop_loss import StopBreach, StopLossPolicy, evaluate, stop_fill_price

SEED_REASON = "seed"
"""``exit_reason`` never takes this value; it marks a position created when
the book was first opened from an existing hypothetical sizing, rather than
bought by this module."""


class PaperBookError(RuntimeError):
    """The ledger is not in a state that can be acted on."""


@dataclass(frozen=True, slots=True)
class PaperPosition:
    """One open holding, with what it actually cost to acquire."""

    symbol: str
    instrument_id: str
    shares: int
    entry_price: float
    """Price per share. What the hard stop is measured from."""

    entry_cost: float
    """Total cash paid, buy-side charges included. What profit is measured
    against -- a position is not ahead until it has earned back its costs."""

    entry_date: str
    rank_at_entry: int
    source: str = SEED_REASON
    session_date: str = ""
    """Which session ``session_high``/``session_low`` describe. They reset
    when it changes."""

    session_high: float = 0.0
    """The highest price seen *while this position was held*, this session.
    Not the same as the session's high: a position bought at 15:05 was not
    held at 09:20, and a stop must not fire on a price that predates it."""

    session_low: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "instrument_id": self.instrument_id,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_cost": self.entry_cost,
            "entry_date": self.entry_date,
            "rank_at_entry": self.rank_at_entry,
            "source": self.source,
            "session_date": self.session_date,
            "session_high": self.session_high,
            "session_low": self.session_low,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PaperPosition:
        return cls(
            symbol=str(raw["symbol"]),
            instrument_id=str(raw["instrument_id"]),
            shares=int(raw["shares"]),
            entry_price=float(raw["entry_price"]),
            entry_cost=float(raw["entry_cost"]),
            entry_date=str(raw["entry_date"]),
            rank_at_entry=int(raw["rank_at_entry"]),
            source=str(raw.get("source", SEED_REASON)),
            session_date=str(raw.get("session_date", "")),
            session_high=float(raw.get("session_high") or 0.0),
            session_low=float(raw.get("session_low") or 0.0),
        )


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """One round trip, from cash out to cash back."""

    symbol: str
    instrument_id: str
    shares: int
    entry_price: float
    entry_cost: float
    entry_date: str
    exit_price: float
    exit_date: str
    exit_reason: str
    stop_level: float
    net_proceeds: float
    """Cash credited after every sell-side charge, DP charge included."""

    @property
    def net_pnl(self) -> float:
        return self.net_proceeds - self.entry_cost

    @property
    def net_pnl_pct(self) -> float:
        return (self.net_proceeds / self.entry_cost - 1.0) if self.entry_cost > 0 else 0.0

    @property
    def gross_pnl(self) -> float:
        """Price move alone, before either leg's charges -- shown next to
        ``net_pnl`` so the cost drag on a round trip is visible rather than
        buried."""
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def costs(self) -> float:
        return self.gross_pnl - self.net_pnl

    @property
    def holding_days(self) -> int:
        try:
            entered = dt.date.fromisoformat(self.entry_date)
            exited = dt.date.fromisoformat(self.exit_date)
        except ValueError:
            return 0
        return (exited - entered).days

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "instrument_id": self.instrument_id,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_cost": self.entry_cost,
            "entry_date": self.entry_date,
            "exit_price": self.exit_price,
            "exit_date": self.exit_date,
            "exit_reason": self.exit_reason,
            "stop_level": self.stop_level,
            "net_proceeds": self.net_proceeds,
            "gross_pnl": self.gross_pnl,
            "costs": self.costs,
            "net_pnl": self.net_pnl,
            "net_pnl_pct": self.net_pnl_pct,
            "holding_days": self.holding_days,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ClosedTrade:
        return cls(
            symbol=str(raw["symbol"]),
            instrument_id=str(raw["instrument_id"]),
            shares=int(raw["shares"]),
            entry_price=float(raw["entry_price"]),
            entry_cost=float(raw["entry_cost"]),
            entry_date=str(raw["entry_date"]),
            exit_price=float(raw["exit_price"]),
            exit_date=str(raw["exit_date"]),
            exit_reason=str(raw["exit_reason"]),
            stop_level=float(raw.get("stop_level", 0.0)),
            net_proceeds=float(raw["net_proceeds"]),
        )


@dataclass
class PaperBook:
    """The account. Mutable on purpose -- it is a ledger, not a value."""

    budget: float
    cash: float
    positions: dict[str, PaperPosition] = field(default_factory=dict)
    closed: list[ClosedTrade] = field(default_factory=list)
    opened_at: str = ""
    updated_at: str = ""
    blocked_today: dict[str, str] = field(default_factory=dict)
    """instrument -> the session it was stopped out on. Prevents the same
    name being bought back the moment it is sold; cleared when the date
    changes."""

    last_rerank_at: str = ""
    """When the ranking was last recomputed for this book, ISO timestamp.
    Rate-limits ``needs_rerank``: recomputing takes minutes, and a book that
    cannot be refilled (because every ranked name is blocked) would otherwise
    ask for a fresh ranking on every refresh."""

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def seed(
        cls,
        *,
        budget: float,
        sizing_positions: Iterable[Mapping[str, Any]],
        picks: Iterable[Mapping[str, Any]],
        costs: CostModel,
        as_of: dt.date,
    ) -> PaperBook:
        """Open a book from an existing hypothetical sizing.

        The positions it describes are treated as already bought at the
        price recorded there -- which is the selection date's close, the
        price the ranking was computed from. Their buy-side charges are
        computed now rather than assumed away, so the cost basis is the
        real one from the first refresh onwards.
        """
        if budget <= 0:
            raise PaperBookError(f"budget must be positive, got {budget}")
        instrument_by_symbol = {
            str(pick["symbol"]): str(pick["instrument_id"]) for pick in picks
        }
        book = cls(
            budget=budget,
            cash=budget,
            opened_at=as_of.isoformat(),
            updated_at=as_of.isoformat(),
        )
        for row in sizing_positions:
            shares = int(row.get("shares") or 0)
            symbol = str(row["symbol"])
            instrument_id = instrument_by_symbol.get(symbol)
            if shares <= 0 or instrument_id is None:
                continue
            price = float(row["price"])
            paid = costs.estimate_execution_cost(
                instrument_id,
                TradeSide.BUY,
                shares,
                price,
                as_of,
                spread_bps=10.0,
                avg_daily_value=0.0,
                volatility=0.0,
            ).net_value
            book.positions[instrument_id] = PaperPosition(
                symbol=symbol,
                instrument_id=instrument_id,
                shares=shares,
                entry_price=price,
                entry_cost=paid,
                entry_date=as_of.isoformat(),
                rank_at_entry=int(row.get("rank") or 0),
                source=SEED_REASON,
            )
            book.cash -= paid
        return book

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "cash": self.cash,
            "opened_at": self.opened_at,
            "updated_at": self.updated_at,
            "positions": [p.to_dict() for p in self.positions.values()],
            "closed": [t.to_dict() for t in self.closed],
            "blocked_today": self.blocked_today,
            "last_rerank_at": self.last_rerank_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PaperBook:
        positions = [PaperPosition.from_dict(p) for p in raw.get("positions") or []]
        return cls(
            budget=float(raw["budget"]),
            cash=float(raw["cash"]),
            positions={p.instrument_id: p for p in positions},
            closed=[ClosedTrade.from_dict(t) for t in raw.get("closed") or []],
            opened_at=str(raw.get("opened_at", "")),
            updated_at=str(raw.get("updated_at", "")),
            blocked_today=dict(raw.get("blocked_today") or {}),
            last_rerank_at=str(raw.get("last_rerank_at", "")),
        )

    @classmethod
    def load(cls, path: Path) -> PaperBook | None:
        """The book on disk, or ``None`` if there is not one yet.

        A corrupt file raises rather than being silently replaced with an
        empty book: losing an account's history to a bad parse and starting
        again at full cash would look exactly like a flat day.
        """
        if not path.is_file():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError) as exc:
            raise PaperBookError(f"{path} is not a readable paper book: {exc}") from exc

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
        )

    # -- the two operations ------------------------------------------------

    def roll_session(self, today: dt.date) -> None:
        """Clear the same-day re-entry blocks once the date has moved on."""
        self.blocked_today = {
            instrument_id: day
            for instrument_id, day in self.blocked_today.items()
            if day == today.isoformat()
        }

    @staticmethod
    def _mark_session(
        position: PaperPosition, quote: Mapping[str, Any], today: dt.date
    ) -> PaperPosition:
        """Update the high and low seen *while this position was held*.

        The distinction the first version of this module got wrong. A quote's
        ``ohlc`` is the whole session's high and low, and for a position
        carried in from a previous day that is exactly right -- it was held
        through all of it. For a position bought at 15:05 it is not: the
        day's low may have printed at 09:20, hours before the account owned
        a single share, and a hard stop measured against it fires instantly
        on a price the position never experienced. (It did: SKYGOLD was
        bought at 796.45 and "stopped out" six minutes later at 772.56, the
        session low from before the purchase.)

        So a position entered today starts its high and low at its own entry
        price and accumulates from there, one observed quote at a time. The
        cost is resolution -- between two refreshes the price is unobserved,
        so a spike that reverses inside the gap is missed. That under-detects
        and never invents, which is the right direction for a stop.
        """
        today_iso = today.isoformat()
        last = float(quote.get("last") or 0.0)
        entered_today = position.entry_date == today_iso

        if position.session_date != today_iso:
            if entered_today:
                base_high = base_low = position.entry_price
            else:
                base_high = float(quote.get("day_high") or last)
                base_low = float(quote.get("day_low") or last)
        else:
            base_high, base_low = position.session_high, position.session_low
            if not entered_today:
                base_high = max(base_high, float(quote.get("day_high") or last))
                base_low = min(base_low, float(quote.get("day_low") or last))

        return replace(
            position,
            session_date=today_iso,
            session_high=max(base_high, last),
            session_low=min(base_low, last) if base_low > 0 else last,
        )

    def apply_stops(
        self,
        quotes: Mapping[str, Mapping[str, Any]],
        *,
        policy: StopLossPolicy,
        costs: CostModel,
        today: dt.date,
    ) -> list[ClosedTrade]:
        """Close every holding whose stop fired, crediting the proceeds.

        Returns the trades closed by this call, newest business first. The
        book is mutated: the positions are gone, the cash is in, and the
        names are blocked from re-entry for the rest of the session.
        """
        closed_now: list[ClosedTrade] = []
        for instrument_id in sorted(self.positions):
            position = self.positions[instrument_id]
            quote = quotes.get(instrument_id)
            if not quote:
                continue
            last = float(quote.get("last") or 0.0)
            if last <= 0:
                continue
            position = self._mark_session(position, quote, today)
            self.positions[instrument_id] = position
            high, low = position.session_high, position.session_low

            def net_sale(price: float, _p: PaperPosition = position) -> float:
                return costs.estimate_execution_cost(
                    _p.instrument_id,
                    TradeSide.SELL,
                    _p.shares,
                    price,
                    today,
                    spread_bps=10.0,
                    avg_daily_value=0.0,
                    volatility=0.0,
                ).net_value

            breach: StopBreach | None = evaluate(
                instrument_id,
                entry_price=position.entry_price,
                cost_basis=position.entry_cost,
                session_high=high,
                session_low=low,
                reference_price=last,
                net_sale_value=net_sale,
                policy=policy,
            )
            if breach is None:
                continue

            # The live fill: the worse of the level and the price on the
            # screen right now. Same rule the backtest uses against a close.
            fill = stop_fill_price(breach.stop_level, last)
            proceeds = net_sale(fill)
            trade = ClosedTrade(
                symbol=position.symbol,
                instrument_id=instrument_id,
                shares=position.shares,
                entry_price=position.entry_price,
                entry_cost=position.entry_cost,
                entry_date=position.entry_date,
                exit_price=fill,
                exit_date=today.isoformat(),
                exit_reason=str(breach.reason),
                stop_level=breach.stop_level,
                net_proceeds=proceeds,
            )
            self.cash += proceeds
            del self.positions[instrument_id]
            self.closed.append(trade)
            self.blocked_today[instrument_id] = today.isoformat()
            closed_now.append(trade)
        return closed_now

    def reallocate(
        self,
        candidates: Iterable[Mapping[str, Any]],
        quotes: Mapping[str, Mapping[str, Any]],
        *,
        costs: CostModel,
        today: dt.date,
        slot: float,
        max_positions: int,
        max_rank: int | None = None,
    ) -> list[PaperPosition]:
        """Spend free cash on the best candidates not already held.

        Walks the candidates in rank order and buys the first names that fit.
        Each new position is capped at ``slot`` so one replacement cannot
        take a share of the account no ranking asked for, and at whatever
        cash is actually free -- an account cannot buy what it cannot pay
        for, and there is no margin here.

        ``max_rank`` is the floor of the ranking this account will reach
        down to. Set to the book's own size it means every position is a
        name the selector actually picked; a candidate ranked below it is
        skipped even when there is cash and nothing else to buy, because
        "the strategy did not choose this" is a reason not to own something.
        ``None`` disables the cap.

        Returns the positions opened. Nothing is sold to raise cash: this is
        a redeployment of what the stops released, not a rebalance.
        """
        opened: list[PaperPosition] = []
        for candidate in candidates:
            if len(self.positions) >= max_positions:
                break
            instrument_id = str(candidate.get("instrument_id") or "")
            if not instrument_id or instrument_id in self.positions:
                continue
            if instrument_id in self.blocked_today:
                continue
            rank = int(candidate.get("rank") or 0)
            if max_rank is not None and (rank <= 0 or rank > max_rank):
                continue
            quote = quotes.get(instrument_id)
            if not quote:
                continue
            price = float(quote.get("last") or 0.0)
            if price <= 0:
                continue

            budget_here = min(slot, self.cash)
            shares = int(budget_here // price)
            while shares > 0:
                payable = costs.estimate_execution_cost(
                    instrument_id,
                    TradeSide.BUY,
                    shares,
                    price,
                    today,
                    spread_bps=10.0,
                    avg_daily_value=0.0,
                    volatility=0.0,
                ).net_value
                if payable <= self.cash:
                    break
                # Charges pushed the bill over the free cash. Drop a share
                # rather than overdraw; at a small slot the DP-scale fees
                # are a real fraction of the order.
                shares -= 1
            if shares <= 0:
                continue

            position = PaperPosition(
                symbol=str(candidate.get("symbol") or instrument_id),
                instrument_id=instrument_id,
                shares=shares,
                entry_price=price,
                entry_cost=payable,
                entry_date=today.isoformat(),
                rank_at_entry=int(candidate.get("rank") or 0),
                source="reallocation",
            )
            self.cash -= payable
            self.positions[instrument_id] = position
            opened.append(position)
        return opened

    def needs_rerank(
        self, *, at_or_below: int, now: dt.datetime, cooldown_minutes: int
    ) -> bool:
        """Whether the book has shrunk enough to deserve a fresh ranking.

        A book down to a handful of names is mostly cash, and the ranking
        that chose those names is by then several stops old -- the market
        that stopped them out is not the one the selector last looked at.
        Reaching further down that stale list buys names it already passed
        over; recomputing it asks the current question instead.

        Rate-limited because recomputing means running the selector and the
        regime model over real history, which takes minutes. Without the
        cooldown a book that cannot be refilled -- every ranked name held or
        blocked -- would ask again on every refresh.
        """
        if len(self.positions) > at_or_below:
            return False
        if not self.last_rerank_at:
            return True
        try:
            last = dt.datetime.fromisoformat(self.last_rerank_at)
        except ValueError:
            return True
        if last.tzinfo is not None and now.tzinfo is None:
            last = last.replace(tzinfo=None)
        return (now - last) >= dt.timedelta(minutes=cooldown_minutes)

    # -- reporting ---------------------------------------------------------

    def realized(self) -> dict[str, Any]:
        """Closed-trade performance. Cash in minus cash out, nothing modelled.

        Deliberately separate from the unrealised marks: a realised number is
        a fact about trades that finished, and mixing it with open positions
        that may yet turn produces a figure that is neither.
        """
        trades = self.closed
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl < 0]
        gross_win = sum(t.net_pnl for t in wins)
        gross_loss = -sum(t.net_pnl for t in losses)
        by_reason: dict[str, dict[str, Any]] = {}
        for trade in trades:
            bucket = by_reason.setdefault(
                trade.exit_reason, {"count": 0, "net_pnl": 0.0}
            )
            bucket["count"] += 1
            bucket["net_pnl"] += trade.net_pnl
        return {
            "available": bool(trades),
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(trades)) if trades else 0.0,
            "net_pnl": sum(t.net_pnl for t in trades),
            "gross_pnl": sum(t.gross_pnl for t in trades),
            "costs": sum(t.costs for t in trades),
            "best": max((t.net_pnl for t in trades), default=0.0),
            "worst": min((t.net_pnl for t in trades), default=0.0),
            "avg_win": (gross_win / len(wins)) if wins else 0.0,
            "avg_loss": (-gross_loss / len(losses)) if losses else 0.0,
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
            "by_reason": by_reason,
            "history": [t.to_dict() for t in reversed(trades)],
        }
