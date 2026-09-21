"""Per-position stops: the two rules that take a single holding out.

Until these existed nothing limited one name's loss. The circuit breakers
act on the *book* -- daily loss, rolling loss, drawdown -- so a single
position could bleed 30% without tripping anything, and a holding only left
the portfolio when it stopped ranking in the top ten, which is usually well
after the damage.

Two independent rules, either of which exits the whole position:

``HARD_STOP``
    The price falls ``hard_stop_pct`` below **the price it was bought at**.
    A loss limiter: it does not move, it does not care about the day, and it
    is the only thing standing between a position and an unbounded loss.

``TRAILING_PROFIT_STOP``
    The price falls ``trail_drop_pct`` below **the session's high**, but only
    once selling there would realise more than ``trail_arm_net_profit_pct``
    *net of the costs of selling*, DP charge included. A profit-taker, not a
    loss limiter: it gives back at most 2% of a gain to avoid round-tripping
    a winner, and it is silent on any position that isn't already well ahead.

The arming test is deliberately computed at the price the exit would fill
at, not at the session high: a position 3.1% up at its high and 1.1% up
after a 2% pullback was never a 3% winner at any price this rule could have
sold at, and arming on the high would book a "profit-protecting" exit that
protects a profit nobody could have taken.

## What a daily bar proves, and what it doesn't

``low <= level`` proves the level *traded*: the hard stop's breach is a
fact, not an estimate. What the bar cannot give is the fill -- the stop may
have been hit on the way down and filled at the level, or the stock may have
gapped straight through it. :func:`stop_fill_price` takes the worse of the
level and the session's close, which is pessimistic by construction; a
backtest that assumes every stop filled exactly at its trigger reports
protection the market does not sell.

The trailing rule cannot use ``low`` at all, and this is the subtle one. A
daily bar records the high and the low but **not which came first**. If the
low preceded the high, then at the moment of the low the running high -- and
so the trail level -- was lower than the level computed from the full
session's high, and the rule may never have fired. Testing ``low`` against a
level derived from the whole day's high is look-ahead. The close is the one
price guaranteed to occur *at or after* the high, so ``close <= level`` is
ordering-safe, and that is what this module tests. The cost is real: an
intraday spike-and-recover that ends the day above the trail level is
missed. That is the conservative direction, and the live path does not have
the problem at all -- it passes its running intraday high and the last
traded price, both known at the moment the question is asked.

## Precedence

If both rules fire on one session the hard stop is reported. The two can
only coexist on a session that opened well up, made a high, then collapsed
through the buy price; chronologically the trail would have sold first, at a
profit, and the hard stop would never have been reached. Reporting the hard
stop therefore understates that exit -- again the conservative direction,
and it never credits a backtest with a sale it cannot prove.

Nothing here decides position size or ranking. It answers one question about
one holding on one session.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

NetSaleValue = Callable[[float], float]
"""``price -> net INR received`` for selling the entire position at ``price``,
after every sell-side charge (STT, exchange, SEBI, GST, stamp, **DP charge**).
Supplied by the caller so this module stays independent of the cost model and
of whichever broker schedule is in force."""


class StopLossError(ValueError):
    """A stop policy or evaluation input is not usable."""


class StopReason(StrEnum):
    HARD_STOP = "hard_stop"
    """Fell ``hard_stop_pct`` below the buy price."""

    TRAILING_PROFIT_STOP = "trailing_profit_stop"
    """Fell ``trail_drop_pct`` below the session high while more than
    ``trail_arm_net_profit_pct`` ahead, net of selling costs."""


@dataclass(frozen=True, slots=True)
class StopLossPolicy:
    """The three numbers the two rules need, and whether they are live.

    Config-driven rather than constants so a fold, a paper session and live
    trading can differ without a code change -- and so a backtest records
    which thresholds produced its result.
    """

    hard_stop_pct: float
    trail_drop_pct: float
    trail_arm_net_profit_pct: float
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("hard_stop_pct", "trail_drop_pct", "trail_arm_net_profit_pct"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise StopLossError(f"{name} must be a fraction in (0, 1), got {value}")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StopLossPolicy:
        """Build from the ``risk.stop_loss`` block of settings.yaml.

        Every key is required. A stop policy with a silently defaulted
        threshold is worse than no stop policy: it would report protection
        at a level nobody chose.
        """
        required = ("hard_stop_pct", "trail_drop_pct", "trail_arm_net_profit_pct", "enabled")
        missing = [key for key in required if key not in raw]
        if missing:
            raise StopLossError(f"stop_loss config is missing required keys: {missing}")
        return cls(
            hard_stop_pct=float(raw["hard_stop_pct"]),
            trail_drop_pct=float(raw["trail_drop_pct"]),
            trail_arm_net_profit_pct=float(raw["trail_arm_net_profit_pct"]),
            enabled=bool(raw["enabled"]),
        )


@dataclass(frozen=True, slots=True)
class StopBreach:
    """One holding that a stop takes out of the book on one session."""

    instrument_id: str
    reason: StopReason
    entry_price: float
    """The average price the position was bought at."""

    reference_price: float
    """What the level was measured from: the buy price for a hard stop, the
    session's high for a trailing stop."""

    stop_level: float
    """The price at or below which the rule fires."""

    fill_price: float
    """What the exit is assumed to fill at -- never better than the level."""

    net_profit_pct: float
    """Realised profit on the whole position at ``fill_price``, after every
    sell-side charge including the DP charge, as a fraction of what was paid
    for it. Negative for a hard stop."""


def hard_stop_level(entry_price: float, hard_stop_pct: float) -> float:
    """The price below which a loss is cut, measured from the buy price."""
    if entry_price <= 0:
        raise StopLossError(f"entry_price must be positive, got {entry_price}")
    return entry_price * (1.0 - hard_stop_pct)


def trailing_stop_level(session_high: float, trail_drop_pct: float) -> float:
    """The price below which a gain is banked, measured from the session high."""
    if session_high <= 0:
        raise StopLossError(f"session_high must be positive, got {session_high}")
    return session_high * (1.0 - trail_drop_pct)


def stop_fill_price(level: float, reference_price: float) -> float:
    """What the exit is assumed to fill at: the worse of the level and the
    price the session ended at (or, live, the last traded price).

    A session that gapped below the level never offered it, so filling there
    would credit the book with a price nobody could have got; a session that
    dipped and recovered did offer it. Taking the minimum covers both without
    needing intraday data to tell them apart.
    """
    return min(level, reference_price) if reference_price > 0 else level


def net_profit_pct(cost_basis: float, net_sale_value: NetSaleValue, price: float) -> float:
    """Profit on selling the whole position at ``price``, after sell costs,
    as a fraction of what the position cost to acquire.

    ``cost_basis`` is what was actually paid -- quantity times buy price plus
    the buy leg's own charges -- so this is realised P&L, not a gross price
    ratio. A 3% move on the screen is not a 3% profit: STT on both legs, the
    exchange and SEBI charges, GST, buy-side stamp duty and the flat DP
    charge all sit between them, and on a small position the DP charge alone
    can be 15-20bps.
    """
    if cost_basis <= 0:
        raise StopLossError(f"cost_basis must be positive, got {cost_basis}")
    return (net_sale_value(price) / cost_basis) - 1.0


def evaluate(
    instrument_id: str,
    *,
    entry_price: float,
    cost_basis: float,
    session_high: float,
    session_low: float,
    reference_price: float,
    net_sale_value: NetSaleValue,
    policy: StopLossPolicy,
) -> StopBreach | None:
    """Whether either stop takes this holding out on this session.

    ``reference_price`` is the session's close in a backtest and the last
    traded price when running live; ``session_high``/``session_low`` are the
    completed session's extremes in a backtest and the running intraday
    extremes when running live. The module docstring explains why the
    trailing rule tests ``reference_price`` while the hard stop tests
    ``session_low``.

    Returns ``None`` when nothing fires, when the policy is disabled, or when
    the inputs are not usable -- a stop is never inferred from a price this
    function cannot verify.
    """
    if not policy.enabled:
        return None
    if entry_price <= 0 or cost_basis <= 0:
        return None
    if session_low <= 0 or session_high <= 0 or reference_price <= 0:
        return None
    if session_high < session_low:
        raise StopLossError(
            f"{instrument_id}: session_high ({session_high}) is below "
            f"session_low ({session_low})"
        )

    hard_level = hard_stop_level(entry_price, policy.hard_stop_pct)
    if session_low <= hard_level:
        fill = stop_fill_price(hard_level, reference_price)
        return StopBreach(
            instrument_id=instrument_id,
            reason=StopReason.HARD_STOP,
            entry_price=entry_price,
            reference_price=entry_price,
            stop_level=hard_level,
            fill_price=fill,
            net_profit_pct=net_profit_pct(cost_basis, net_sale_value, fill),
        )

    trail_level = trailing_stop_level(session_high, policy.trail_drop_pct)
    if reference_price <= trail_level:
        fill = stop_fill_price(trail_level, reference_price)
        profit = net_profit_pct(cost_basis, net_sale_value, fill)
        if profit > policy.trail_arm_net_profit_pct:
            return StopBreach(
                instrument_id=instrument_id,
                reason=StopReason.TRAILING_PROFIT_STOP,
                entry_price=entry_price,
                reference_price=session_high,
                stop_level=trail_level,
                fill_price=fill,
                net_profit_pct=profit,
            )
    return None
