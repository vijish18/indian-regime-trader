"""Per-position stops: the two rules that take a single holding out.

The master rule, as configured (``risk.stop_loss`` in settings.yaml). A
position leaves the book when either holds:

1. its average buy price is above INR 100 and it trades 3% below that price, or
2. it is more than 5% in profit **after every sell-side charge, DP charge
   included**, and has fallen 2% below **today's running high**.

The asymmetry is the design. Losses are cut at 3%; winners are left alone
until they are 5% ahead and then trailed rather than capped, so a name that
keeps running keeps running. An earlier setting closed winners at the
threshold itself, which made every win the same size as a loss and put the
whole result on the win rate; ``close_on_arm`` still selects that behaviour
so a backtest can measure the difference instead of arguing about it.

Until these existed nothing limited one name's loss. The circuit breakers
act on the *book* -- daily loss, rolling loss, drawdown -- so a single
position could bleed 30% without tripping anything, and a holding only left
the portfolio when it stopped ranking in the top ten, which is usually well
after the damage.

Two independent rules, either of which exits the whole position:

``HARD_STOP``
    The price falls ``hard_stop_pct`` below **the price it was bought at**.
    Applies only when average entry exceeds ``hard_stop_min_entry_price``.
    Lower-priced positions remain eligible for rank-based replacement.

``TRAILING_PROFIT_STOP`` (while ``close_on_arm`` is off -- the shipped setting)
    The price falls ``trail_drop_pct`` below **the session's high**, but only
    once selling there would realise more than ``trail_arm_net_profit_pct``
    net. A profit-taker that gives back at most 2% of a gain rather than
    capping the gain, so a name that keeps running is still held.

``TAKE_PROFIT`` (while ``close_on_arm`` is on)
    Selling right now would realise more than ``trail_arm_net_profit_pct``
    net. The position is closed there and then, giving up everything above
    the threshold.

The two profit rules are different bets. Closing on arm banks every winner
at the threshold and keeps none of the upside beyond it -- a name that goes
on to +20% is sold at +3%, and the strategy's returns become a stream of
small wins against whatever the hard stop lets through. Waiting for the
pullback keeps the position while it is still rising and pays for that with
the 2% given back at the top. Which is better depends on the return
distribution of the names this strategy picks, which is an empirical
question, so it is configuration rather than a constant.

Both profit tests are computed at the price the exit would fill at, never at
a high the sale could not reach: a position 3.1% up at its high and 1.1% up
after a pullback was never a 3% winner at any price the rule could have sold
at, and testing the high would book a "profit-protecting" exit that protects
a profit nobody could have taken.

## What a daily bar proves, and what it doesn't

This section is about the two rules that read a session's extremes: the hard
stop, always, and the trailing stop when ``close_on_arm`` is off. The
take-profit reads neither -- it tests the price on the screen and sells
there, so there is nothing to infer.

``low <= level`` establishes a breach, but does not prove the trigger price
itself traded or was executable. With a daily bar, a hard stop is modelled at
the opening price when it opens below the trigger, otherwise at the trigger.
The caller charges execution costs separately. This cannot resolve intraday
gaps, queue position or circuit-limit liquidity. The closing price is not a
substitute for the price at which the earlier hard stop would have executed.

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

If both a loss rule and a profit rule fire on one session the hard stop is
reported. They can only coexist on a session that ran well up, then
collapsed through the buy price; chronologically the profit rule would have
sold first, at a gain, and the hard stop would never have been reached.
Reporting the hard stop therefore understates that exit -- the conservative
direction, and it never credits a backtest with a sale it cannot prove.

Nothing here decides position size or ranking. It answers one question about
one holding on one session.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
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
    ``trail_arm_net_profit_pct`` ahead, net of selling costs. Only reachable
    when ``close_on_arm`` is off."""

    TAKE_PROFIT = "take_profit"
    """Reached ``trail_arm_net_profit_pct`` net of selling costs and was
    closed there, without waiting for a pullback. What ``close_on_arm``
    does."""


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
    close_on_arm: bool = True
    """Whether reaching ``trail_arm_net_profit_pct`` closes the position
    outright (``TAKE_PROFIT``) instead of arming a trailing stop that waits
    for a ``trail_drop_pct`` pullback (``TRAILING_PROFIT_STOP``).

    The two are different bets, not different spellings of one. Closing on
    arm banks every winner at the threshold and keeps none of the upside
    beyond it: a name that goes on to +20% is sold at +3%. Waiting for the
    pullback gives back up to ``trail_drop_pct`` of whatever the high
    reached, and in exchange keeps the position while it is still rising.
    Which is better is a question about the return distribution of the names
    this strategy picks, so it is configuration and not a constant.

    ``trail_drop_pct`` is unused while this is on. It stays configured
    rather than being deleted so the setting can be turned off again without
    having to rediscover the number."""

    enabled: bool = True
    hard_stop_min_entry_price: float = 0.0
    profit_exit_enabled: bool = True
    """False leaves the hard stop as the only exit rule: no trailing stop and
    no take-profit. A switch rather than an unreachable threshold, so the
    config says what the rule is instead of hiding it in a number."""

    """Apply the hard stop only above this average purchase price, strictly.

    Uses the same corporate-action-adjusted entry price as the stop level.
    Zero preserves historical policies; settings.yaml selects INR 100.
    Profit exits and rank-based rebalancing remain independent.
    """

    def __post_init__(self) -> None:
        if not isfinite(self.hard_stop_min_entry_price) or self.hard_stop_min_entry_price < 0:
            raise StopLossError("hard_stop_min_entry_price must be finite and non-negative")
        for name in ("hard_stop_pct", "trail_drop_pct", "trail_arm_net_profit_pct"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise StopLossError(f"{name} must be a fraction in (0, 1), got {value}")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StopLossPolicy:
        """Build from the ``risk.stop_loss`` block of settings.yaml.

        The original keys are required. The optional minimum entry price
        defaults to zero for compatibility with historical configurations.
        A stop policy with a silently defaulted
        threshold is worse than no stop policy: it would report protection
        at a level nobody chose.
        """
        required = (
            "hard_stop_pct",
            "trail_drop_pct",
            "trail_arm_net_profit_pct",
            "close_on_arm",
            "enabled",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise StopLossError(f"stop_loss config is missing required keys: {missing}")
        return cls(
            hard_stop_pct=float(raw["hard_stop_pct"]),
            trail_drop_pct=float(raw["trail_drop_pct"]),
            trail_arm_net_profit_pct=float(raw["trail_arm_net_profit_pct"]),
            close_on_arm=bool(raw["close_on_arm"]),
            enabled=bool(raw["enabled"]),
            hard_stop_min_entry_price=float(raw.get("hard_stop_min_entry_price", 0.0)),
            profit_exit_enabled=bool(raw.get("profit_exit_enabled", True)),
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
    """Worse of trigger and supplied executable-price proxy.

    Hard-stop backtests supply the open; paper/live supply last observed
    price. Trailing-profit evaluation uses the close as described above.
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
    session_open: float | None = None,
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

    Backtests supply ``session_open``: a hard stop fills at the open if it
    gaps below the trigger, otherwise at the trigger after a low breach.
    Execution costs are applied by the caller. Without an open, retain the
    last-observed-price convention for live/paper callers. Daily bars cannot
    establish liquidity or intraday gaps; these are modelled fills.
    """
    if not policy.enabled:
        return None
    if entry_price <= 0 or cost_basis <= 0:
        return None
    if session_low <= 0 or session_high <= 0 or reference_price <= 0:
        return None
    if session_high < session_low:
        raise StopLossError(
            f"{instrument_id}: session_high ({session_high}) is below session_low ({session_low})"
        )
    if session_open is not None and (
        not isfinite(session_open) or not session_low <= session_open <= session_high
    ):
        raise StopLossError("session_open must be finite and within the session range")

    hard_level = hard_stop_level(entry_price, policy.hard_stop_pct)
    if entry_price > policy.hard_stop_min_entry_price and session_low <= hard_level:
        fill = stop_fill_price(
            hard_level, session_open if session_open is not None else reference_price
        )
        return StopBreach(
            instrument_id=instrument_id,
            reason=StopReason.HARD_STOP,
            entry_price=entry_price,
            reference_price=entry_price,
            stop_level=hard_level,
            fill_price=fill,
            net_profit_pct=net_profit_pct(cost_basis, net_sale_value, fill),
        )

    if not policy.profit_exit_enabled:
        return None

    if policy.close_on_arm:
        # The take-profit sells at the price on the screen, so the threshold
        # is tested there too -- no separate level to fill at, and nothing
        # about the session's extremes is involved. In a backtest
        # ``reference_price`` is the close, so a session that crossed the
        # threshold intraday and closed below it is not counted: the profit
        # has to be there at a price the rule could actually have sold at.
        profit = net_profit_pct(cost_basis, net_sale_value, reference_price)
        if profit > policy.trail_arm_net_profit_pct:
            return StopBreach(
                instrument_id=instrument_id,
                reason=StopReason.TAKE_PROFIT,
                entry_price=entry_price,
                reference_price=reference_price,
                stop_level=reference_price,
                fill_price=reference_price,
                net_profit_pct=profit,
            )
        return None

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
