"""Indian transaction-cost and execution-cost model.

Do this before trusting any backtest (docs/SPECIFICATION.md section 9): a
zero-commission or single-flat-fee assumption silently flatters every
Indian equity strategy, since delivery (CNC) trading carries several
independent statutory charges on top of brokerage. This module models each
one separately rather than folding them into one generic number, and keeps
them classified by *where the uncertainty actually is* (see
:class:`CostCategory`) so a reader can tell at a glance which numbers are
exact charges and which are a model's estimate.

## Deterministic vs. estimated

:class:`TradeCost` is the **deterministic** side: brokerage plus every
statutory levy, computed from a dated :class:`~backtest.cost_schedule.CostSchedule`
and the trade's own quantity/price/side. Given the same schedule and the
same trade, it is bit-for-bit reproducible -- no randomness, no market
data beyond the trade itself.

:class:`ExecutionCostEstimate` adds the **estimated** side: slippage,
built from a spread-cost component and a price-impact component
(docs/SPECIFICATION.md section 9.1's research slippage model). These are
never charged amounts on a contract note; they are this system's best
guess at how much the market will move against an order before it fills,
and are labeled as such throughout (``CostCategory.ESTIMATED``).

## Rounding

Every individual charge is rounded to the nearest paisa (2 decimal places,
half-up, matching how a real contract note itemizes charges) *before* it
is summed, and GST is computed on the already-rounded brokerage/exchange/SEBI
figures -- exactly as a broker's contract note would show it. This keeps
the displayed total always exactly equal to the sum of the displayed line
items; computing GST or the total from unrounded intermediates can silently
produce a total that doesn't match its own breakdown by a paisa.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import ClassVar

from backtest.cost_schedule import CostScheduleRepository

_PAISA = Decimal("0.01")


def _round_inr(value: float) -> float:
    """Round to the nearest paisa (2dp), half-up -- the convention Indian
    brokerage contract notes use, not Python's banker's-rounding default."""
    return float(Decimal(str(value)).quantize(_PAISA, rounding=ROUND_HALF_UP))


class TradeSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class CostCategory(StrEnum):
    """Where the uncertainty in a cost component actually lives -- the
    classification docs/SPECIFICATION.md section 9 asks for, made
    machine-readable rather than left to a docstring.
    """

    DETERMINISTIC = "deterministic"
    """A uniform, formula-computed statutory charge: same for every broker
    and every exchange once the schedule is known (STT, SEBI turnover fee,
    GST, stamp duty)."""

    BROKER_DEPENDENT = "broker_dependent"
    """Set by whichever broker executes the trade (brokerage, DP charges --
    DP charges are levied by the Depository Participant, which is
    typically the broker's own DP arm, and vary materially between
    brokers)."""

    EXCHANGE_DEPENDENT = "exchange_dependent"
    """Set by the exchange the trade routes to (exchange transaction
    charges differ between NSE and BSE)."""

    ESTIMATED = "estimated"
    """Not a charged amount at all -- a model's estimate of market impact
    the trade will experience (slippage, spread cost, price impact)."""


class CostModelError(RuntimeError):
    """A trade cost could not be computed -- invalid trade inputs, not a
    missing schedule (that raises
    :class:`~backtest.cost_schedule.MissingCostScheduleError` instead)."""


@dataclass(frozen=True, slots=True)
class TradeCost:
    """The complete, deterministic statutory + brokerage cost of one trade
    leg. Excludes slippage -- see :class:`ExecutionCostEstimate`, and this
    module's docstring, "Deterministic vs. estimated".
    """

    instrument_id: str
    side: TradeSide
    quantity: int
    price: float
    turnover: float
    """``quantity * price`` -- the gross trade value before any cost."""

    schedule_effective_from: dt.date
    """Which dated :class:`~backtest.cost_schedule.CostSchedule` produced
    this breakdown, for audit -- never the trade date itself, since a
    schedule can predate the trade by any amount."""

    brokerage: float
    stt: float
    exchange_txn_charge: float
    sebi_turnover_fee: float
    gst: float
    stamp_duty: float
    dp_charge: float
    other_charges: float
    total: float

    CATEGORY: ClassVar[dict[str, CostCategory]] = {
        "brokerage": CostCategory.BROKER_DEPENDENT,
        "stt": CostCategory.DETERMINISTIC,
        "exchange_txn_charge": CostCategory.EXCHANGE_DEPENDENT,
        "sebi_turnover_fee": CostCategory.DETERMINISTIC,
        "gst": CostCategory.DETERMINISTIC,
        "stamp_duty": CostCategory.DETERMINISTIC,
        "dp_charge": CostCategory.BROKER_DEPENDENT,
        "other_charges": CostCategory.DETERMINISTIC,
    }
    """Which :class:`CostCategory` each line item belongs to. A classmethod
    rather than per-field metadata so it stays trivially inspectable
    (``TradeCost.CATEGORY["brokerage"]``) without reaching into dataclass
    internals."""

    _COMPONENT_FIELDS: ClassVar[tuple[str, ...]] = (
        "brokerage",
        "stt",
        "exchange_txn_charge",
        "sebi_turnover_fee",
        "gst",
        "stamp_duty",
        "dp_charge",
        "other_charges",
    )

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"price must be > 0, got {self.price}")
        expected_turnover = self.quantity * self.price
        if abs(self.turnover - expected_turnover) > 1e-6:
            raise ValueError(
                f"turnover ({self.turnover}) must equal quantity * price ({expected_turnover})"
            )
        for field_name in self._COMPONENT_FIELDS:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0, got {value}")
        expected_total = sum(getattr(self, field_name) for field_name in self._COMPONENT_FIELDS)
        if abs(self.total - expected_total) > 1e-6:
            raise ValueError(
                f"total ({self.total}) must equal the sum of its components ({expected_total})"
            )

    def components_by_category(self) -> dict[CostCategory, float]:
        """Total cost grouped by :class:`CostCategory` -- e.g. "how much of
        this trade's cost is broker-dependent vs. fixed by law"."""
        totals: dict[CostCategory, float] = dict.fromkeys(CostCategory, 0.0)
        for field_name in self._COMPONENT_FIELDS:
            totals[self.CATEGORY[field_name]] += getattr(self, field_name)
        return totals


@dataclass(frozen=True, slots=True)
class ExecutionCostEstimate:
    """A trade's full expected cost: :class:`TradeCost` (deterministic)
    plus a slippage estimate (:class:`CostCategory.ESTIMATED`), combined
    into the numbers a backtest report actually needs -- gross value, total
    cost, net value, and cost as a percentage of turnover.
    """

    trade_cost: TradeCost
    spread_cost_bps: float
    """``0.5 * spread_bps`` -- half the quoted spread, in basis points of
    order value. A sub-component of ``slippage_bps``, not additive to it."""

    price_impact_bps: float
    """The market-impact component, in basis points of order value. Also a
    sub-component of ``slippage_bps``, not additive to it."""

    slippage_bps: float
    """``max(min_bps, spread_cost_bps + price_impact_bps)`` -- the combined
    estimated execution cost (docs/SPECIFICATION.md section 9.1)."""

    slippage_amount: float
    """``slippage_bps`` converted to INR on this trade's turnover."""

    gross_value: float
    """The trade's turnover before any cost -- ``trade_cost.turnover``."""

    total_cost: float
    """``trade_cost.total + slippage_amount`` -- deterministic plus
    estimated, the number a backtest sums across trades to report total
    cost drag."""

    net_value: float
    """The effective trade value after costs, as shown on a contract note:
    for a BUY, ``gross_value + total_cost`` (you pay more than the sticker
    value); for a SELL, ``gross_value - total_cost`` (you receive less)."""

    cost_pct_of_turnover: float
    """``total_cost / gross_value`` -- the number
    docs/SPECIFICATION.md's backtest reporting requirement asks for
    directly; a backtest's portfolio-level version is the turnover-weighted
    average of this across every trade, not a simple average."""

    def __post_init__(self) -> None:
        for field_name in (
            "spread_cost_bps",
            "price_impact_bps",
            "slippage_bps",
            "slippage_amount",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0, got {value}")
        if self.gross_value <= 0:
            raise ValueError(f"gross_value must be > 0, got {self.gross_value}")
        # Both checks below compare against the nearest-paisa-rounded expected
        # value, not the raw float sum: `total_cost` and `net_value` are
        # themselves rounded to the paisa (this module's docstring,
        # "Rounding"), and `gross_value` (quantity * a float price) is not
        # guaranteed to itself be a clean 2-decimal number, so an exact
        # (1e-6) comparison against the unrounded sum would reject a
        # correctly-rounded value over ordinary paisa-level rounding.
        expected_total = _round_inr(self.trade_cost.total + self.slippage_amount)
        if abs(self.total_cost - expected_total) > 0.01:
            raise ValueError(
                f"total_cost ({self.total_cost}) must equal trade_cost.total + "
                f"slippage_amount, rounded ({expected_total})"
            )
        expected_net = _round_inr(
            self.gross_value + self.total_cost
            if self.trade_cost.side is TradeSide.BUY
            else self.gross_value - self.total_cost
        )
        if abs(self.net_value - expected_net) > 0.01:
            raise ValueError(f"net_value ({self.net_value}) must equal {expected_net}")
        expected_pct = self.total_cost / self.gross_value
        if abs(self.cost_pct_of_turnover - expected_pct) > 1e-9:
            raise ValueError(
                f"cost_pct_of_turnover ({self.cost_pct_of_turnover}) must equal {expected_pct}"
            )


class CostModel:
    """Computes deterministic :class:`TradeCost` and full
    :class:`ExecutionCostEstimate` for one Indian delivery-equity trade
    leg, from a versioned :class:`~backtest.cost_schedule.CostScheduleRepository`.

    ``min_slippage_bps`` and ``impact_coefficient`` are the research
    slippage model's own tunable parameters (docs/SPECIFICATION.md section
    9.1) -- unlike the statutory rates in ``CostSchedule``, they are not a
    regulatory fact with an effective date, so they are plain
    configuration (``BacktestConfig``), not a versioned schedule entry.
    """

    def __init__(
        self,
        schedules: CostScheduleRepository,
        min_slippage_bps: float,
        impact_coefficient: float,
    ) -> None:
        if min_slippage_bps < 0:
            raise ValueError(f"min_slippage_bps must be >= 0, got {min_slippage_bps}")
        if impact_coefficient < 0:
            raise ValueError(f"impact_coefficient must be >= 0, got {impact_coefficient}")
        self.schedules = schedules
        self.min_slippage_bps = min_slippage_bps
        self.impact_coefficient = impact_coefficient

    def compute_trade_cost(
        self,
        instrument_id: str,
        side: TradeSide,
        quantity: int,
        price: float,
        trade_date: dt.date,
    ) -> TradeCost:
        """The deterministic brokerage + statutory cost breakdown for one
        trade leg, using the schedule effective on ``trade_date``.

        Raises :class:`CostModelError` for non-positive quantity/price, and
        :class:`~backtest.cost_schedule.MissingCostScheduleError` (from the
        repository) if no schedule covers ``trade_date``.
        """
        if quantity <= 0:
            raise CostModelError(f"quantity must be > 0, got {quantity}")
        if price <= 0:
            raise CostModelError(f"price must be > 0, got {price}")

        schedule = self.schedules.schedule_as_of(trade_date)
        turnover = quantity * price

        brokerage = _round_inr(schedule.brokerage_flat_inr + schedule.brokerage_pct * turnover)
        stt_pct = schedule.stt_buy_pct if side is TradeSide.BUY else schedule.stt_sell_pct
        stt = _round_inr(stt_pct * turnover)
        exchange_txn_charge = _round_inr(schedule.exchange_txn_pct * turnover)
        sebi_turnover_fee = _round_inr(schedule.sebi_turnover_pct * turnover)
        gst = _round_inr(schedule.gst_pct * (brokerage + exchange_txn_charge + sebi_turnover_fee))
        stamp_duty_pct = (
            schedule.stamp_duty_buy_pct if side is TradeSide.BUY else schedule.stamp_duty_sell_pct
        )
        stamp_duty = _round_inr(stamp_duty_pct * turnover)
        dp_charge = _round_inr(schedule.dp_charges_inr) if side is TradeSide.SELL else 0.0
        other_charges = _round_inr(
            schedule.other_charges_flat_inr + schedule.other_charges_pct * turnover
        )

        total = _round_inr(
            brokerage
            + stt
            + exchange_txn_charge
            + sebi_turnover_fee
            + gst
            + stamp_duty
            + dp_charge
            + other_charges
        )

        return TradeCost(
            instrument_id=instrument_id,
            side=side,
            quantity=quantity,
            price=price,
            turnover=turnover,
            schedule_effective_from=schedule.effective_from,
            brokerage=brokerage,
            stt=stt,
            exchange_txn_charge=exchange_txn_charge,
            sebi_turnover_fee=sebi_turnover_fee,
            gst=gst,
            stamp_duty=stamp_duty,
            dp_charge=dp_charge,
            other_charges=other_charges,
            total=total,
        )

    def estimate_slippage_bps(
        self,
        order_value: float,
        spread_bps: float,
        avg_daily_value: float,
        volatility: float,
    ) -> tuple[float, float, float]:
        """``(spread_cost_bps, price_impact_bps, slippage_bps)`` --
        ``slippage_bps = max(min_bps, spread_cost_bps + price_impact_bps)``
        (docs/SPECIFICATION.md section 9.1).

        The impact term uses a square-root participation model
        (``impact_coefficient * volatility * sqrt(order_value / avg_daily_value)``,
        in bps): impact grows with the square root of how large the order
        is relative to the instrument's liquidity, and with the
        instrument's own volatility, which is the standard qualitative
        shape research market-impact models use. ``avg_daily_value <= 0``
        is treated as "no reliable liquidity estimate" and produces the
        maximum-participation case (``sqrt(1.0)``) rather than a fabricated
        near-zero impact.
        """
        if order_value < 0:
            raise CostModelError(f"order_value must be >= 0, got {order_value}")
        if spread_bps < 0:
            raise CostModelError(f"spread_bps must be >= 0, got {spread_bps}")
        if volatility < 0:
            raise CostModelError(f"volatility must be >= 0, got {volatility}")

        spread_cost_bps = 0.5 * spread_bps
        participation = order_value / avg_daily_value if avg_daily_value > 0 else 1.0
        participation = min(max(participation, 0.0), 1.0) if avg_daily_value > 0 else 1.0
        price_impact_bps = self.impact_coefficient * volatility * math.sqrt(participation)
        slippage_bps = max(self.min_slippage_bps, spread_cost_bps + price_impact_bps)
        return spread_cost_bps, price_impact_bps, slippage_bps

    def estimate_execution_cost(
        self,
        instrument_id: str,
        side: TradeSide,
        quantity: int,
        price: float,
        trade_date: dt.date,
        spread_bps: float,
        avg_daily_value: float,
        volatility: float,
    ) -> ExecutionCostEstimate:
        """The full expected cost of one trade leg: deterministic
        :class:`TradeCost` plus the estimated slippage on top.
        """
        trade_cost = self.compute_trade_cost(instrument_id, side, quantity, price, trade_date)
        spread_cost_bps, price_impact_bps, slippage_bps = self.estimate_slippage_bps(
            order_value=trade_cost.turnover,
            spread_bps=spread_bps,
            avg_daily_value=avg_daily_value,
            volatility=volatility,
        )
        slippage_amount = _round_inr(slippage_bps / 10_000 * trade_cost.turnover)
        gross_value = trade_cost.turnover
        total_cost = _round_inr(trade_cost.total + slippage_amount)
        net_value = _round_inr(
            gross_value + total_cost if side is TradeSide.BUY else gross_value - total_cost
        )
        cost_pct_of_turnover = total_cost / gross_value

        return ExecutionCostEstimate(
            trade_cost=trade_cost,
            spread_cost_bps=spread_cost_bps,
            price_impact_bps=price_impact_bps,
            slippage_bps=slippage_bps,
            slippage_amount=slippage_amount,
            gross_value=gross_value,
            total_cost=total_cost,
            net_value=net_value,
            cost_pct_of_turnover=cost_pct_of_turnover,
        )


def net_pnl(gross_pnl: float, total_cost: float) -> float:
    """``gross_pnl - total_cost``. ``backtest/performance.py``'s
    ``PerformanceCalculator`` derives gross P&L from the equity curve in
    the *other* direction (``gross = net + costs``, since the curve is
    already net of every cost) rather than calling this function directly,
    but the two are the same equation read in opposite directions -- this
    is the canonical statement of it, so "net P&L" isn't left as an
    implicit convention scattered across whoever reports it.
    """
    return gross_pnl - total_cost


def cost_pct_of_turnover(total_cost: float, total_turnover: float) -> float:
    """Portfolio-level cost ratio: total cost across every trade divided by
    total turnover across every trade -- the turnover-weighted figure a
    backtest report needs, not an average of each trade's own
    ``cost_pct_of_turnover`` (which would over-weight small trades).
    """
    if total_turnover <= 0:
        raise CostModelError(f"total_turnover must be > 0, got {total_turnover}")
    return total_cost / total_turnover
