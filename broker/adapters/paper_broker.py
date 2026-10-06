"""Paper-trading broker adapter: simulates fills against real market data
without sending any order to a real exchange. Required before live capital
is enabled (docs/SPECIFICATION.md section 2 and section 20).

Implements ``broker.base.Broker`` exactly -- no method here is visible to
strategy code as anything other than a generic broker call, which is the
whole point: swapping this for a future ``LiveBroker`` changes nothing
above this module. Applies the identical cost/slippage model as
``backtest.costs.CostModel`` (and the identical liquidity/volatility
estimate, ``backtest.engine.market_liquidity_stats``) so paper results are
directly comparable to backtest results, with one genuine improvement over
the backtest: this adapter prices spread from a real live ``Quote``
(``data.interfaces.MarketDataProvider.get_quote``), not the assumed
constant the backtest falls back on for lack of real bid/ask history.

## What is simulated, and how

- **Order validation** (broker-side, defense in depth on top of whatever
  ``risk.risk_manager.RiskManager`` already approved upstream): order
  type must match ``execution.order_type`` (NSE algo orders may not use
  market orders -- see ``broker/base.py``'s module docstring), the quote
  must not be stale (``execution.stale_quote_seconds``) or crossed, the
  limit price must sit within ``execution.order_price_guard_bps`` of the
  quote's mid, a sell must not exceed the held quantity (no shorting,
  V1 long-only), and a buy must not exceed available cash at the worst
  case (ask) price.
- **Idempotency**: ``place_order`` is keyed by ``client_order_id``. A
  resubmission of the identical payload returns the already-recorded
  order unchanged, never creating a second one; a resubmission carrying a
  *different* payload under a reused ID is a caller bug and raises rather
  than silently accepting whichever payload arrived first or last.
- **Latency**: ``paper_trading.submission_latency_ms`` is reflected in the
  order's recorded acknowledgement timestamp, not an actual ``sleep`` --
  this adapter is synchronous and in-process, so "latency" here means
  "what the timestamps would show", not simulated wall-clock delay.
- **Fills and partial fills**: a marketable limit order fills against the
  quote's own displayed depth (``bid_quantity``/``ask_quantity``), capped
  per match by ``paper_trading.max_fill_participation_pct`` of that depth
  -- an order larger than one match's share of the book rests
  ``PARTIALLY_FILLED`` rather than filling instantly in full. A quote with
  no depth information at all (``None``) is filled in full as a documented
  simplification (this system's most conservative default is depth
  information should shrink a fill, never expand one, and there is
  nothing here to shrink it by).
- **Slippage**: every fill is priced through
  ``backtest.costs.CostModel.estimate_execution_cost`` exactly as a
  backtest fill is, using the real quote's own spread plus a trailing
  liquidity/volatility estimate (``paper_trading.default_avg_daily_value_inr``/
  ``default_volatility`` when too little history exists to measure one).
- **Expiry**: a resting order older than ``paper_trading.order_expiry_seconds``
  transitions to ``EXPIRED`` the next time any broker method runs -- a
  duration-based simplification, not session-aware (see
  ``config.models.PaperTradingConfig``).
- **Not simulated**: partial-fill *timing* beyond one match per call (a
  resting remainder only fills on a later ``place_order``/``modify_order``
  call against a fresh quote, never on a background timer), and any form
  of real network failure -- ``UNKNOWN``/ambiguous-response handling is
  exercised by wrapping this adapter, not by anything inside it, since
  this adapter's own responses are never actually lost.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from backtest.costs import CostModel, ExecutionCostEstimate, TradeSide
from backtest.engine import market_liquidity_stats
from broker.base import (
    Broker,
    BrokerAccount,
    BrokerCapabilities,
    BrokerFill,
    BrokerOrder,
    BrokerPosition,
    BrokerQuote,
    HealthStatus,
)
from broker.errors import BrokerCapabilityError
from config.models import ExecutionConfig, PaperTradingConfig
from data.errors import DataNotAvailableError
from data.interfaces import MarketDataProvider
from data.models import Quote
from execution.order_manager import OrderState
from execution.position_tracker import PositionTracker

_BROKER_OPEN_STATES: frozenset[OrderState] = frozenset(
    {OrderState.OPEN, OrderState.PARTIALLY_FILLED}
)
"""States this adapter itself reports about a live order. Distinct from
``execution.order_manager.OPEN_STATES``, which also covers client-only
states (``SUBMITTED``, ``CANCEL_REQUESTED``) this adapter never emits
about its own book -- by the time it can report on an order at all, that
order has already been received and processed."""

_RESTING_STATES: frozenset[OrderState] = frozenset({OrderState.OPEN, OrderState.PARTIALLY_FILLED})


class PaperBrokerError(RuntimeError):
    """A broker-level operation could not proceed -- an unknown order ID,
    an invalid modification, or a client_order_id reused for a different
    order payload. Fail closed rather than guessing."""


@dataclass(frozen=True, slots=True)
class PaperFill:
    """One simulated fill, for audit -- not part of ``Broker``'s interface,
    a paper-only introspection convenience."""

    client_order_id: str
    instrument_id: str
    side: TradeSide
    quantity: int
    fill_price: float
    execution_cost: ExecutionCostEstimate
    as_of: dt.datetime


class PaperBroker(Broker):
    """Simulated broker used for paper trading and backtesting-adjacent
    dry runs. Must apply the same cost/slippage model as backtest/costs.py
    so paper results are comparable to backtest results.
    """

    def __init__(
        self,
        market_data: MarketDataProvider,
        cost_model: CostModel,
        execution_config: ExecutionConfig,
        paper_config: PaperTradingConfig,
        position_tracker: PositionTracker,
        initial_cash: float,
        clock: Callable[[], dt.datetime] | None = None,
    ) -> None:
        if initial_cash <= 0:
            raise PaperBrokerError(f"initial_cash must be > 0, got {initial_cash}")
        if execution_config.mode != "paper":
            raise PaperBrokerError(
                f"PaperBroker requires execution.mode == 'paper', got {execution_config.mode!r}"
            )
        self.market_data = market_data
        self.cost_model = cost_model
        self.execution_config = execution_config
        self.paper_config = paper_config
        self.position_tracker = position_tracker
        self._cash = initial_cash
        self._clock: Callable[[], dt.datetime] = clock or (lambda: dt.datetime.now(dt.UTC))
        self._orders: dict[str, BrokerOrder] = {}
        self._created_at: dict[str, dt.datetime] = {}
        self._fills: list[PaperFill] = []
        self._login_time = self._clock()

    @property
    def fills(self) -> tuple[PaperFill, ...]:
        return tuple(self._fills)

    @property
    def cash(self) -> float:
        return self._cash

    def acknowledged_at(self, client_order_id: str) -> dt.datetime:
        """When this order's submission was simulated as acknowledged --
        ``paper_trading.submission_latency_ms`` after it was received. Not
        part of ``Broker``; a paper-only introspection convenience for
        verifying latency simulation directly."""
        try:
            return self._created_at[client_order_id]
        except KeyError:
            raise PaperBrokerError(f"unknown order_id: {client_order_id}") from None

    def process_resting_orders(self) -> list[BrokerOrder]:
        """Re-attempt a match for every resting order against its
        instrument's current quote. Not part of ``Broker`` -- a live/paper
        trading loop is expected to call this periodically (the same
        "tick" a real exchange connection would deliver as a stream of
        book updates) so a partially-filled order's remainder can fill as
        fresh quotes arrive, not only once at submission time.
        """
        self._expire_stale_orders()
        updated = []
        for client_order_id, order in list(self._orders.items()):
            if OrderState(order.status) not in _RESTING_STATES:
                continue
            try:
                quote = self._get_quote_or_raise(order.instrument_id)
            except PaperBrokerError:
                # No quote for this name this tick (e.g. locked at a price
                # band): the order keeps resting, as it would on the exchange,
                # and the other resting orders are still matched.
                continue
            updated.append(self._attempt_match(client_order_id, quote, self._clock()))
        return updated

    # -- Broker interface ---------------------------------------------

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            broker_name="paper",
            supports_order_modification=True,
            supports_market_data_streaming=False,
            supported_exchanges=frozenset({"NSE", "BSE"}),
            supported_products=frozenset({"CNC"}),
            supported_order_types=frozenset({self.execution_config.order_type}),
            supported_varieties=frozenset({"regular"}),
        )

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        """A no-op: this adapter has no real session to establish, and is
        always ready to accept calls once constructed. Present so callers
        can treat every ``Broker`` uniformly rather than special-casing
        the paper adapter."""
        self._login_time = self._clock()

    def get_account(self) -> BrokerAccount:
        self._mark_positions_to_market()
        equity = self._cash
        for position in self.position_tracker.current_positions():
            equity += position.quantity * position.current_price
        return BrokerAccount(
            account_id="paper",
            equity=equity,
            cash=self._cash,
            buying_power=self._cash,
            as_of=self._clock(),
        )

    def get_positions(self) -> list[BrokerPosition]:
        self._mark_positions_to_market()
        return [
            BrokerPosition(
                instrument_id=position.instrument_id,
                quantity=position.quantity,
                avg_price=position.avg_price,
            )
            for position in self.position_tracker.current_positions()
        ]

    def get_open_orders(self) -> list[BrokerOrder]:
        self._expire_stale_orders()
        return [
            order
            for order in self._orders.values()
            if OrderState(order.status) in _BROKER_OPEN_STATES
        ]

    def get_order(self, order_id: str) -> BrokerOrder:
        self._expire_stale_orders()
        return self._require_order(order_id)

    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]:
        quotes = []
        for instrument_id in instrument_ids:
            quote = self._get_quote_or_raise(instrument_id)
            quotes.append(
                BrokerQuote(
                    instrument_id=instrument_id,
                    bid=float(quote.bid),
                    ask=float(quote.ask),
                    last_price=float(quote.last_price),
                    as_of=quote.as_of,
                )
            )
        return quotes

    def subscribe_market_data(
        self, instrument_ids: list[str], on_tick: Callable[[BrokerQuote], None]
    ) -> Callable[[], None]:
        raise BrokerCapabilityError(
            "PaperBroker has no live streaming feed -- poll get_quotes() instead"
        )

    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        return [
            BrokerFill(
                trade_id=str(index),
                order_id=fill.client_order_id,
                instrument_id=fill.instrument_id,
                side=fill.side.value,
                quantity=fill.quantity,
                price=fill.fill_price,
                product="CNC",
                as_of=fill.as_of,
            )
            for index, fill in enumerate(self._fills, start=1)
            if order_id is None or fill.client_order_id == order_id
        ]

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        self._expire_stale_orders()
        existing = self._orders.get(order.client_order_id)
        if existing is not None:
            if not self._same_payload(existing, order):
                raise PaperBrokerError(
                    f"client_order_id {order.client_order_id} was already used for a "
                    "different order payload -- idempotent resubmission requires an "
                    "identical request"
                )
            return existing

        now = self._clock()
        rejection = self._validate(order)
        if rejection is not None:
            return self._reject(order, rejection)

        acknowledged_at = now + dt.timedelta(milliseconds=self.paper_config.submission_latency_ms)
        resting = replace(
            order,
            broker_order_id=str(uuid.uuid4()),
            status=OrderState.OPEN.value,
            filled_quantity=0,
            avg_fill_price=None,
            reject_reason=None,
        )
        self._orders[order.client_order_id] = resting
        self._created_at[order.client_order_id] = acknowledged_at

        quote = self._get_quote_or_raise(order.instrument_id)
        return self._attempt_match(order.client_order_id, quote, acknowledged_at)

    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder:
        self._expire_stale_orders()
        order = self._require_order(order_id)
        state = OrderState(order.status)
        if state not in _RESTING_STATES:
            raise PaperBrokerError(f"cannot modify order {order_id} in state {state.value}")

        new_limit_price = changes.get("limit_price", order.limit_price)
        new_quantity = changes.get("quantity", order.quantity)
        if not isinstance(new_limit_price, int | float) or new_limit_price <= 0:
            raise PaperBrokerError(
                f"limit_price must be a positive number, got {new_limit_price!r}"
            )
        if not isinstance(new_quantity, int) or new_quantity <= order.filled_quantity:
            raise PaperBrokerError(
                f"quantity must be an int greater than filled_quantity "
                f"({order.filled_quantity}), got {new_quantity!r}"
            )

        updated = replace(order, limit_price=float(new_limit_price), quantity=new_quantity)
        self._orders[order_id] = updated
        quote = self._get_quote_or_raise(order.instrument_id)
        return self._attempt_match(order_id, quote, self._clock())

    def cancel_order(self, order_id: str) -> BrokerOrder:
        self._expire_stale_orders()
        order = self._require_order(order_id)
        state = OrderState(order.status)
        if state not in _RESTING_STATES:
            raise PaperBrokerError(
                f"cannot cancel order {order_id} in terminal state {state.value}"
            )
        cancelled = replace(order, status=OrderState.CANCELLED.value)
        self._orders[order_id] = cancelled
        return cancelled

    def close_position(self, instrument_id: str) -> BrokerOrder:
        held = self.position_tracker.held_quantity(instrument_id)
        if held <= 0:
            raise PaperBrokerError(f"no open position in {instrument_id} to close")
        quote = self._get_quote_or_raise(instrument_id)
        order = BrokerOrder(
            client_order_id=str(uuid.uuid4()),
            broker_order_id=None,
            instrument_id=instrument_id,
            side=TradeSide.SELL.value,
            quantity=held,
            order_type=self.execution_config.order_type,
            limit_price=float(quote.bid),
            status=OrderState.CREATED.value,
        )
        return self.place_order(order)

    def close_all_positions(self) -> list[BrokerOrder]:
        return [
            self.close_position(position.instrument_id)
            for position in self.position_tracker.current_positions()
        ]

    def health_check(self) -> HealthStatus:
        return HealthStatus(
            healthy=True,
            detail="paper broker: in-process simulation, always reachable",
            checked_at=self._clock(),
            session_active=True,
            login_time=self._login_time,
        )

    # -- internals -------------------------------------------------------

    @staticmethod
    def _same_payload(a: BrokerOrder, b: BrokerOrder) -> bool:
        return (
            a.instrument_id,
            a.side,
            a.quantity,
            a.order_type,
            a.limit_price,
            a.product,
            a.variety,
        ) == (
            b.instrument_id,
            b.side,
            b.quantity,
            b.order_type,
            b.limit_price,
            b.product,
            b.variety,
        )

    def _validate(self, order: BrokerOrder) -> str | None:
        """Returns a rejection reason, or ``None`` if the order passes
        every broker-side check."""
        if order.quantity <= 0:
            return f"quantity must be > 0, got {order.quantity}"
        if order.order_type != self.execution_config.order_type:
            return (
                f"order_type {order.order_type!r} not supported "
                f"(only {self.execution_config.order_type!r})"
            )
        if order.limit_price is None or order.limit_price <= 0:
            return "a limit order requires a positive limit_price"
        try:
            side = TradeSide(order.side)
        except ValueError:
            return f"unrecognized side: {order.side!r}"

        try:
            quote = self.market_data.get_quote(order.instrument_id)
        except DataNotAvailableError as exc:
            return f"no live quote for {order.instrument_id}: {exc}"

        if quote.is_crossed():
            return "quote is crossed (bid > ask); refusing to trade a broken book"
        if quote.is_stale(self._clock(), self.execution_config.stale_quote_seconds):
            return (
                f"quote is stale (older than "
                f"stale_quote_seconds={self.execution_config.stale_quote_seconds})"
            )

        mid = float(quote.mid)
        distance_bps = abs(order.limit_price - mid) / mid * 10_000
        guard_bps = self.execution_config.order_price_guard_bps
        if distance_bps > guard_bps:
            return (
                f"limit_price {order.limit_price} is {distance_bps:.1f}bps from mid "
                f"{mid:.2f}, exceeds order_price_guard_bps={guard_bps}"
            )

        if side is TradeSide.SELL:
            held = self.position_tracker.held_quantity(order.instrument_id)
            if order.quantity > held:
                return (
                    f"sell quantity {order.quantity} exceeds held quantity {held}; "
                    "this system is long-only and never shorts"
                )
        else:
            worst_case_cost = order.quantity * float(quote.ask)
            if worst_case_cost > self._cash:
                return (
                    f"insufficient cash: order could cost up to {worst_case_cost:.2f}, "
                    f"cash available {self._cash:.2f}"
                )
        return None

    def _reject(self, order: BrokerOrder, reason: str) -> BrokerOrder:
        rejected = replace(
            order,
            broker_order_id=None,
            status=OrderState.REJECTED.value,
            filled_quantity=0,
            avg_fill_price=None,
            reject_reason=reason,
        )
        self._orders[order.client_order_id] = rejected
        return rejected

    def _attempt_match(self, client_order_id: str, quote: Quote, now: dt.datetime) -> BrokerOrder:
        order = self._orders[client_order_id]
        if OrderState(order.status) not in _RESTING_STATES:
            return order
        side = TradeSide(order.side)
        assert order.limit_price is not None  # enforced at validation time

        touch_price = float(quote.ask) if side is TradeSide.BUY else float(quote.bid)
        marketable = (
            touch_price <= order.limit_price
            if side is TradeSide.BUY
            else touch_price >= order.limit_price
        )
        if not marketable:
            return order

        depth = quote.ask_quantity if side is TradeSide.BUY else quote.bid_quantity
        remaining = order.quantity - order.filled_quantity
        if depth is None:
            fillable = remaining
        else:
            participation_cap = max(
                1, math.floor(depth * self.paper_config.max_fill_participation_pct)
            )
            fillable = min(remaining, participation_cap)
        if fillable <= 0:
            return order

        execution_cost = self._price_fill(
            order.instrument_id, side, fillable, touch_price, now, quote
        )
        if side is TradeSide.BUY:
            self._cash -= execution_cost.net_value
        else:
            self._cash += execution_cost.net_value
        self.position_tracker.apply_fill(order.instrument_id, fillable, touch_price, side, now)
        self._fills.append(
            PaperFill(
                client_order_id=client_order_id,
                instrument_id=order.instrument_id,
                side=side,
                quantity=fillable,
                fill_price=touch_price,
                execution_cost=execution_cost,
                as_of=now,
            )
        )

        new_filled = order.filled_quantity + fillable
        prior_notional = (order.avg_fill_price or 0.0) * order.filled_quantity
        new_avg_fill_price = (prior_notional + touch_price * fillable) / new_filled
        new_status = (
            OrderState.FILLED if new_filled >= order.quantity else OrderState.PARTIALLY_FILLED
        )
        updated = replace(
            order,
            filled_quantity=new_filled,
            avg_fill_price=new_avg_fill_price,
            status=new_status.value,
        )
        self._orders[client_order_id] = updated
        return updated

    def _price_fill(
        self,
        instrument_id: str,
        side: TradeSide,
        quantity: int,
        fill_price: float,
        now: dt.datetime,
        quote: Quote,
    ) -> ExecutionCostEstimate:
        avg_daily_value, volatility = market_liquidity_stats(
            self.market_data, instrument_id, now.date()
        )
        if avg_daily_value <= 0:
            avg_daily_value = self.paper_config.default_avg_daily_value_inr
        if volatility <= 0:
            volatility = self.paper_config.default_volatility
        return self.cost_model.estimate_execution_cost(
            instrument_id,
            side,
            quantity,
            fill_price,
            now.date(),
            spread_bps=float(quote.spread_bps),
            avg_daily_value=avg_daily_value,
            volatility=volatility,
        )

    def _mark_positions_to_market(self) -> None:
        """Refresh every held instrument's last-known price from a current
        quote before reporting -- without this, ``unrealized_pnl`` would
        stay frozen at whatever price the instrument last *traded* at,
        rather than reflecting the market it is currently held in. A
        quote that cannot be fetched for one instrument is skipped, not
        fatal to the whole report -- reporting must not fail just because
        one name's live feed is momentarily unavailable."""
        now = self._clock()
        prices: dict[str, float] = {}
        for position in self.position_tracker.current_positions():
            try:
                quote = self.market_data.get_quote(position.instrument_id)
            except DataNotAvailableError:
                continue
            prices[position.instrument_id] = float(quote.last_price)
        if prices:
            self.position_tracker.mark_to_market(prices, now)

    def _expire_stale_orders(self) -> None:
        now = self._clock()
        for client_order_id, order in list(self._orders.items()):
            if OrderState(order.status) not in _RESTING_STATES:
                continue
            created = self._created_at.get(client_order_id)
            if created is None:
                continue
            age = (now - created).total_seconds()
            if age >= self.paper_config.order_expiry_seconds:
                self._orders[client_order_id] = replace(order, status=OrderState.EXPIRED.value)

    def _require_order(self, order_id: str) -> BrokerOrder:
        try:
            return self._orders[order_id]
        except KeyError:
            raise PaperBrokerError(f"unknown order_id: {order_id}") from None

    def _get_quote_or_raise(self, instrument_id: str) -> Quote:
        try:
            return self.market_data.get_quote(instrument_id)
        except DataNotAvailableError as exc:
            raise PaperBrokerError(f"no live quote for {instrument_id}: {exc}") from exc
