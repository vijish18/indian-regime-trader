"""Integration tests for the paper-trading engine (Phase 14):
``broker.adapters.paper_broker.PaperBroker`` wired to a real
``backtest.costs.CostModel``, a real ``execution.position_tracker.PositionTracker``,
and (for the end-to-end scenarios) a real ``execution.order_manager.OrderManager`` --
everything except the market-data feed, which is an in-process fake, and
the exchange itself, which this system never touches (paper trading, by
definition, connects to nothing real).

These tests exercise the module wiring the same way a live trading loop
would use it: submit an order, observe fills/partial fills/rejections,
inspect positions and P&L, cancel, and recover from a lost response --
never by reaching into ``PaperBroker``'s private state.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from decimal import Decimal

import pytest

from backtest.cost_schedule import CostScheduleRepository
from backtest.costs import CostModel, TradeSide
from broker.adapters.paper_broker import PaperBroker, PaperBrokerError
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
from config.models import ExecutionConfig, PaperTradingConfig
from data.errors import DataNotAvailableError
from data.interfaces import MarketDataProvider
from data.models import DailyBar, IndexObservation, PriceBasis, Quote
from execution.order_manager import OrderManager, OrderState
from execution.position_tracker import PositionTracker
from tests.unit._wf_support import cost_schedule, execution_config

_T0 = dt.datetime(2024, 6, 3, 9, 30, tzinfo=dt.UTC)


class _ClockBox:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + dt.timedelta(**kwargs)


class _FakeLiveMarketData(MarketDataProvider):
    """Unlike ``tests/unit/_wf_support.py``'s ``FakeMarketDataProvider``
    (deliberately historical-only, matching the real
    ``LocalMarketDataProvider``), this fake supports live quotes --
    paper trading needs a feed that can, since it is not a backtest.
    """

    def __init__(self) -> None:
        self._quotes: dict[str, Quote] = {}
        self._bars: dict[str, list[DailyBar]] = {}

    def set_quote(self, quote: Quote) -> None:
        self._quotes[quote.instrument_id] = quote

    def get_equity_bars(
        self,
        instrument_id: str,
        start: dt.date,
        end: dt.date,
        price_basis: PriceBasis = PriceBasis.RAW,
    ) -> list[DailyBar]:
        bars = self._bars.get(instrument_id, [])
        return [bar for bar in bars if start <= bar.session_date <= end]

    def get_index_observations(
        self, index_symbol: str, start: dt.date, end: dt.date
    ) -> list[IndexObservation]:
        return []

    def available_range(self, instrument_id: str) -> tuple[dt.date, dt.date] | None:
        bars = self._bars.get(instrument_id)
        if not bars:
            return None
        return bars[0].session_date, bars[-1].session_date

    def get_quote(self, instrument_id: str) -> Quote:
        try:
            return self._quotes[instrument_id]
        except KeyError:
            raise DataNotAvailableError(f"no quote for {instrument_id}") from None


def make_quote(
    instrument_id: str = "NSE:INFY",
    *,
    bid: float = 99.70,
    ask: float = 100.30,
    last_price: float = 100.00,
    as_of: dt.datetime = _T0,
    bid_quantity: int | None = None,
    ask_quantity: int | None = None,
) -> Quote:
    return Quote(
        instrument_id=instrument_id,
        bid=Decimal(str(bid)),
        ask=Decimal(str(ask)),
        last_price=Decimal(str(last_price)),
        as_of=as_of,
        bid_quantity=bid_quantity,
        ask_quantity=ask_quantity,
    )


def paper_trading_config(**overrides: object) -> PaperTradingConfig:
    defaults: dict[str, object] = {
        "submission_latency_ms": 100,
        "max_fill_participation_pct": 0.25,
        "order_expiry_seconds": 3600,
        "default_avg_daily_value_inr": 0.0,
        "default_volatility": 0.30,
    }
    defaults.update(overrides)
    return PaperTradingConfig.model_validate(defaults)


def make_execution_config(**overrides: object) -> ExecutionConfig:
    return execution_config(order_price_guard_bps=50, stale_quote_seconds=15, **overrides)


def make_cost_model() -> CostModel:
    return CostModel(
        CostScheduleRepository([cost_schedule()]), min_slippage_bps=2.0, impact_coefficient=10.0
    )


def new_order(
    *,
    client_order_id: str = "co-1",
    instrument_id: str = "NSE:INFY",
    side: str = "buy",
    quantity: int = 10,
    order_type: str = "limit",
    limit_price: float | None = 100.30,
) -> BrokerOrder:
    return BrokerOrder(
        client_order_id=client_order_id,
        broker_order_id=None,
        instrument_id=instrument_id,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        status=OrderState.CREATED.value,
    )


@pytest.fixture
def clock() -> _ClockBox:
    return _ClockBox(_T0)


@pytest.fixture
def market_data() -> _FakeLiveMarketData:
    return _FakeLiveMarketData()


@pytest.fixture
def cost_model() -> CostModel:
    return make_cost_model()


@pytest.fixture
def broker(
    market_data: _FakeLiveMarketData, cost_model: CostModel, clock: _ClockBox
) -> PaperBroker:
    return PaperBroker(
        market_data=market_data,
        cost_model=cost_model,
        execution_config=make_execution_config(),
        paper_config=paper_trading_config(),
        position_tracker=PositionTracker(),
        initial_cash=1_000_000.0,
        clock=clock,
    )


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_construction_rejects_a_non_paper_execution_mode(
    market_data: _FakeLiveMarketData, cost_model: CostModel
) -> None:
    with pytest.raises(PaperBrokerError, match="paper"):
        PaperBroker(
            market_data=market_data,
            cost_model=cost_model,
            execution_config=make_execution_config(mode="live"),
            paper_config=paper_trading_config(),
            position_tracker=PositionTracker(),
            initial_cash=1_000_000.0,
        )


def test_construction_rejects_non_positive_initial_cash(
    market_data: _FakeLiveMarketData, cost_model: CostModel
) -> None:
    with pytest.raises(PaperBrokerError, match="initial_cash"):
        PaperBroker(
            market_data=market_data,
            cost_model=cost_model,
            execution_config=make_execution_config(),
            paper_config=paper_trading_config(),
            position_tracker=PositionTracker(),
            initial_cash=0.0,
        )


# --------------------------------------------------------------------------
# Submission and fills
# --------------------------------------------------------------------------


def test_a_marketable_order_fills_immediately_in_full(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    result = broker.place_order(new_order(quantity=10, limit_price=100.30))
    assert result.status == OrderState.FILLED.value
    assert result.filled_quantity == 10
    assert result.avg_fill_price == pytest.approx(100.30)
    assert broker.position_tracker.held_quantity("NSE:INFY") == 10


def test_an_unmarketable_limit_order_rests_open_unfilled(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    result = broker.place_order(new_order(quantity=10, limit_price=99.80))
    assert result.status == OrderState.OPEN.value
    assert result.filled_quantity == 0
    assert broker.position_tracker.held_quantity("NSE:INFY") == 0


def test_get_open_orders_reports_a_resting_order(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    broker.place_order(new_order(quantity=10, limit_price=99.80))
    open_orders = broker.get_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].status == OrderState.OPEN.value


def test_a_filled_order_no_longer_appears_in_open_orders(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    broker.place_order(new_order(quantity=10, limit_price=100.30))
    assert broker.get_open_orders() == []


# --------------------------------------------------------------------------
# Partial fills
# --------------------------------------------------------------------------


def test_a_large_order_partially_fills_against_limited_depth(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=100))
    result = broker.place_order(new_order(quantity=100, limit_price=100.30))
    # max_fill_participation_pct=0.25 of a 100-share book -> 25 per match
    assert result.status == OrderState.PARTIALLY_FILLED.value
    assert result.filled_quantity == 25
    assert broker.position_tracker.held_quantity("NSE:INFY") == 25


def test_processing_resting_orders_fills_further_partials_over_time(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=100))
    order = broker.place_order(new_order(quantity=100, limit_price=100.30))
    assert order.filled_quantity == 25

    updated = broker.process_resting_orders()
    assert len(updated) == 1
    assert updated[0].filled_quantity == 50
    assert updated[0].status == OrderState.PARTIALLY_FILLED.value

    broker.process_resting_orders()
    broker.process_resting_orders()
    final = broker.get_order("co-1")
    assert final.status == OrderState.FILLED.value
    assert final.filled_quantity == 100
    assert broker.position_tracker.held_quantity("NSE:INFY") == 100


def test_a_quote_with_no_depth_information_fills_in_full(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=None))
    result = broker.place_order(new_order(quantity=500, limit_price=100.30))
    assert result.status == OrderState.FILLED.value
    assert result.filled_quantity == 500


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------


def test_resubmitting_the_identical_order_does_not_duplicate_the_fill(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    order = new_order(quantity=10, limit_price=100.30)
    first = broker.place_order(order)
    cash_after_first = broker.cash
    second = broker.place_order(order)
    assert second == first
    assert broker.cash == cash_after_first
    assert broker.position_tracker.held_quantity("NSE:INFY") == 10
    assert len(broker.fills) == 1


def test_resubmitting_a_different_payload_under_the_same_id_raises(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    broker.place_order(new_order(quantity=10, limit_price=100.30))
    with pytest.raises(PaperBrokerError, match="different order payload"):
        broker.place_order(new_order(quantity=20, limit_price=100.30))


# --------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------


def test_an_unsupported_order_type_is_rejected(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    result = broker.place_order(new_order(order_type="market"))
    assert result.status == OrderState.REJECTED.value
    assert result.reject_reason is not None and "order_type" in result.reject_reason


def test_a_stale_quote_is_rejected(
    broker: PaperBroker, market_data: _FakeLiveMarketData, clock: _ClockBox
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000, as_of=_T0))
    clock.advance(seconds=30)  # stale_quote_seconds=15
    result = broker.place_order(new_order())
    assert result.status == OrderState.REJECTED.value
    assert result.reject_reason is not None and "stale" in result.reject_reason


def test_a_price_far_outside_the_guard_is_rejected(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))  # mid = 100.0
    result = broker.place_order(new_order(limit_price=110.0))  # 1000bps away
    assert result.status == OrderState.REJECTED.value
    assert result.reject_reason is not None and "order_price_guard_bps" in result.reject_reason


def test_a_buy_beyond_available_cash_is_rejected(
    market_data: _FakeLiveMarketData, cost_model: CostModel, clock: _ClockBox
) -> None:
    broker = PaperBroker(
        market_data=market_data,
        cost_model=cost_model,
        execution_config=make_execution_config(),
        paper_config=paper_trading_config(),
        position_tracker=PositionTracker(),
        initial_cash=500.0,
        clock=clock,
    )
    market_data.set_quote(make_quote(ask_quantity=1000))
    result = broker.place_order(new_order(quantity=10, limit_price=100.30))
    assert result.status == OrderState.REJECTED.value
    assert result.reject_reason is not None and "insufficient cash" in result.reject_reason


def test_a_sell_beyond_the_held_quantity_is_rejected_as_no_shorting(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    result = broker.place_order(new_order(side="sell", quantity=5, limit_price=99.70))
    assert result.status == OrderState.REJECTED.value
    assert result.reject_reason is not None and "shorts" in result.reject_reason


def test_an_order_for_an_instrument_with_no_quote_is_rejected(broker: PaperBroker) -> None:
    result = broker.place_order(new_order(instrument_id="NSE:UNKNOWN"))
    assert result.status == OrderState.REJECTED.value
    assert result.reject_reason is not None and "no live quote" in result.reject_reason


def test_a_crossed_quote_is_rejected(broker: PaperBroker, market_data: _FakeLiveMarketData) -> None:
    market_data.set_quote(make_quote(bid=101.0, ask=100.0))  # crossed
    result = broker.place_order(new_order())
    assert result.status == OrderState.REJECTED.value
    assert result.reject_reason is not None and "crossed" in result.reject_reason


def test_an_invalid_limit_price_is_rejected(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    result = broker.place_order(new_order(limit_price=None))
    assert result.status == OrderState.REJECTED.value


def test_a_non_positive_quantity_is_rejected(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    result = broker.place_order(new_order(quantity=0))
    assert result.status == OrderState.REJECTED.value


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------


def test_cancel_order_cancels_a_resting_order(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    broker.place_order(new_order(quantity=10, limit_price=99.80))
    cancelled = broker.cancel_order("co-1")
    assert cancelled.status == OrderState.CANCELLED.value
    assert broker.get_open_orders() == []


def test_cannot_cancel_an_order_already_filled(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    broker.place_order(new_order(quantity=10, limit_price=100.30))
    with pytest.raises(PaperBrokerError, match="terminal state"):
        broker.cancel_order("co-1")


def test_cancel_of_an_unknown_order_raises(broker: PaperBroker) -> None:
    with pytest.raises(PaperBrokerError, match="unknown order_id"):
        broker.cancel_order("does-not-exist")


# --------------------------------------------------------------------------
# Modification
# --------------------------------------------------------------------------


def test_modify_order_can_move_a_price_to_become_marketable(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    broker.place_order(new_order(quantity=10, limit_price=99.80))
    updated = broker.modify_order("co-1", {"limit_price": 100.30})
    assert updated.status == OrderState.FILLED.value


def test_modify_order_on_a_terminal_order_raises(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    broker.place_order(new_order(quantity=10, limit_price=100.30))
    with pytest.raises(PaperBrokerError):
        broker.modify_order("co-1", {"limit_price": 100.20})


# --------------------------------------------------------------------------
# Latency simulation
# --------------------------------------------------------------------------


def test_submission_latency_is_reflected_in_the_acknowledgement_timestamp(
    broker: PaperBroker, market_data: _FakeLiveMarketData, clock: _ClockBox
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    submitted_at = clock.now
    broker.place_order(new_order(quantity=10, limit_price=100.30))
    ack = broker.acknowledged_at("co-1")
    assert ack - submitted_at == dt.timedelta(milliseconds=100)


# --------------------------------------------------------------------------
# Slippage / cost-model parity with the backtester
# --------------------------------------------------------------------------


def test_fill_cost_matches_the_shared_cost_model_directly(
    broker: PaperBroker, market_data: _FakeLiveMarketData, cost_model: CostModel, clock: _ClockBox
) -> None:
    quote = make_quote(ask_quantity=1000)
    market_data.set_quote(quote)
    cash_before = broker.cash

    result = broker.place_order(new_order(quantity=10, limit_price=100.30))
    assert result.status == OrderState.FILLED.value

    expected = cost_model.estimate_execution_cost(
        "NSE:INFY",
        TradeSide.BUY,
        10,
        100.30,
        clock.now.date(),
        spread_bps=float(quote.spread_bps),
        avg_daily_value=0.0,  # no bars -> config default_avg_daily_value_inr (0.0)
        volatility=0.30,  # no bars -> config default_volatility
    )
    assert broker.cash == pytest.approx(cash_before - expected.net_value)
    assert len(broker.fills) == 1
    assert broker.fills[0].execution_cost.total_cost == pytest.approx(expected.total_cost)


# --------------------------------------------------------------------------
# Position and P&L updates
# --------------------------------------------------------------------------


def test_position_and_pnl_update_across_a_buy_then_a_partial_sell(
    broker: PaperBroker, market_data: _FakeLiveMarketData, clock: _ClockBox
) -> None:
    market_data.set_quote(
        make_quote(bid=99.70, ask=100.30, last_price=100.00, as_of=clock.now, ask_quantity=1000)
    )
    broker.place_order(new_order(client_order_id="co-buy", quantity=10, limit_price=100.30))
    position = broker.position_tracker.current_positions()[0]
    assert position.quantity == 10
    assert position.avg_price == pytest.approx(100.30)
    assert position.realized_pnl == 0.0

    # The market moves up; mark-to-market should show unrealized P&L
    # against the *current* quote, not the stale fill price.
    clock.advance(seconds=5)
    market_data.set_quote(
        make_quote(bid=104.70, ask=105.30, last_price=105.00, as_of=clock.now, bid_quantity=1000)
    )
    account = broker.get_account()
    marked = broker.position_tracker.current_positions()[0]
    assert marked.current_price == pytest.approx(105.00)
    assert marked.unrealized_pnl == pytest.approx((105.00 - 100.30) * 10)
    assert account.equity == pytest.approx(broker.cash + 10 * 105.00)

    sell_result = broker.place_order(
        new_order(client_order_id="co-sell", side="sell", quantity=4, limit_price=104.70)
    )
    assert sell_result.status == OrderState.FILLED.value
    after_sell = broker.position_tracker.current_positions()[0]
    assert after_sell.quantity == 6
    assert after_sell.realized_pnl == pytest.approx((104.70 - 100.30) * 4)


def test_get_positions_reports_broker_neutral_position_shape(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    broker.place_order(new_order(quantity=10, limit_price=100.30))
    positions = broker.get_positions()
    assert positions == [BrokerPosition(instrument_id="NSE:INFY", quantity=10, avg_price=100.30)]


# --------------------------------------------------------------------------
# Order expiry
# --------------------------------------------------------------------------


def test_a_resting_order_expires_after_its_configured_lifetime(
    market_data: _FakeLiveMarketData, cost_model: CostModel, clock: _ClockBox
) -> None:
    broker = PaperBroker(
        market_data=market_data,
        cost_model=cost_model,
        execution_config=make_execution_config(),
        paper_config=paper_trading_config(order_expiry_seconds=60, submission_latency_ms=0),
        position_tracker=PositionTracker(),
        initial_cash=1_000_000.0,
        clock=clock,
    )
    market_data.set_quote(make_quote(ask_quantity=1000))
    broker.place_order(new_order(quantity=10, limit_price=99.80))  # not marketable, rests open

    clock.advance(seconds=61)
    expired = broker.get_order("co-1")
    assert expired.status == OrderState.EXPIRED.value
    assert broker.get_open_orders() == []


# --------------------------------------------------------------------------
# close_position / close_all_positions / health_check
# --------------------------------------------------------------------------


def test_close_position_sells_the_full_held_quantity(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000, bid_quantity=1000))
    broker.place_order(new_order(quantity=10, limit_price=100.30))
    closing_order = broker.close_position("NSE:INFY")
    assert closing_order.side == TradeSide.SELL.value
    assert closing_order.status == OrderState.FILLED.value
    assert broker.position_tracker.held_quantity("NSE:INFY") == 0


def test_close_position_with_nothing_held_raises(broker: PaperBroker) -> None:
    with pytest.raises(PaperBrokerError, match="no open position"):
        broker.close_position("NSE:INFY")


def test_close_all_positions_closes_every_held_instrument(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote("NSE:INFY", ask_quantity=1000, bid_quantity=1000))
    market_data.set_quote(
        make_quote(
            "NSE:TCS",
            bid=3499.70,
            ask=3500.30,
            last_price=3500.0,
            ask_quantity=1000,
            bid_quantity=1000,
        )
    )
    broker.place_order(
        new_order(
            client_order_id="co-1", instrument_id="NSE:INFY", quantity=10, limit_price=100.30
        )
    )
    broker.place_order(
        new_order(client_order_id="co-2", instrument_id="NSE:TCS", quantity=5, limit_price=3500.30)
    )
    closing_orders = broker.close_all_positions()
    assert len(closing_orders) == 2
    assert broker.position_tracker.current_positions() == []


def test_health_check_reports_healthy(broker: PaperBroker) -> None:
    status = broker.health_check()
    assert status.healthy is True


def test_get_quotes_maps_the_live_feed(
    broker: PaperBroker, market_data: _FakeLiveMarketData
) -> None:
    market_data.set_quote(make_quote(bid=99.70, ask=100.30, last_price=100.0))
    quotes = broker.get_quotes(["NSE:INFY"])
    assert quotes == [
        BrokerQuote(instrument_id="NSE:INFY", bid=99.70, ask=100.30, last_price=100.0, as_of=_T0)
    ]


def test_get_order_for_an_unknown_id_raises(broker: PaperBroker) -> None:
    with pytest.raises(PaperBrokerError, match="unknown order_id"):
        broker.get_order("does-not-exist")


# --------------------------------------------------------------------------
# End-to-end via OrderManager, including a lost/ambiguous response
# --------------------------------------------------------------------------


class _FlakyBroker(Broker):
    """Wraps a real ``Broker`` but drops the response to exactly one
    ``place_order`` call -- the order is actually processed underneath,
    the caller just never learns the outcome from that call. Models a
    network timeout/connection drop after the broker already acted.
    """

    def __init__(self, inner: Broker, fail_client_order_id: str) -> None:
        self._inner = inner
        self._fail_for = fail_client_order_id
        self._already_failed = False

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        if order.client_order_id == self._fail_for and not self._already_failed:
            self._already_failed = True
            self._inner.place_order(order)
            raise TimeoutError("simulated network timeout waiting for broker ack")
        return self._inner.place_order(order)

    def capabilities(self) -> BrokerCapabilities:
        return self._inner.capabilities()

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        self._inner.authenticate(credentials)

    def get_account(self) -> BrokerAccount:
        return self._inner.get_account()

    def get_positions(self) -> list[BrokerPosition]:
        return self._inner.get_positions()

    def get_open_orders(self) -> list[BrokerOrder]:
        return self._inner.get_open_orders()

    def get_order(self, order_id: str) -> BrokerOrder:
        return self._inner.get_order(order_id)

    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        return self._inner.get_trades(order_id)

    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]:
        return self._inner.get_quotes(instrument_ids)

    def subscribe_market_data(
        self, instrument_ids: list[str], on_tick: Callable[[BrokerQuote], None]
    ) -> Callable[[], None]:
        return self._inner.subscribe_market_data(instrument_ids, on_tick)

    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder:
        return self._inner.modify_order(order_id, changes)

    def cancel_order(self, order_id: str) -> BrokerOrder:
        return self._inner.cancel_order(order_id)

    def close_position(self, instrument_id: str) -> BrokerOrder:
        return self._inner.close_position(instrument_id)

    def close_all_positions(self) -> list[BrokerOrder]:
        return self._inner.close_all_positions()

    def health_check(self) -> HealthStatus:
        return self._inner.health_check()


def test_order_manager_end_to_end_submission_through_a_real_paper_broker(
    broker: PaperBroker, market_data: _FakeLiveMarketData, clock: _ClockBox
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    order_manager = OrderManager(clock=clock)
    created = order_manager.create(
        "NSE:INFY",
        "buy",
        10,
        "limit",
        100.30,
        idempotency_key="strategy-signal-1",
        signal_id="sig-1",
        risk_decision_id="rd-1",
    )
    record = order_manager.submit(created.order.client_order_id, broker)

    assert record.state is OrderState.FILLED
    assert record.filled_quantity == 10
    # The strategy-facing OrderManager view and the broker's own ledger agree
    # -- this is "the same portfolio state model" the phase requires.
    assert broker.position_tracker.held_quantity("NSE:INFY") == 10


def test_order_manager_resubmission_with_the_same_idempotency_key_never_duplicates(
    broker: PaperBroker, market_data: _FakeLiveMarketData, clock: _ClockBox
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    order_manager = OrderManager(clock=clock)
    first = order_manager.create(
        "NSE:INFY",
        "buy",
        10,
        "limit",
        100.30,
        idempotency_key="strategy-signal-1",
        signal_id="sig-1",
        risk_decision_id="rd-1",
    )
    order_manager.submit(first.order.client_order_id, broker)

    replay = order_manager.create(
        "NSE:INFY",
        "buy",
        10,
        "limit",
        100.30,
        idempotency_key="strategy-signal-1",
        signal_id="sig-1",
        risk_decision_id="rd-1",
    )
    assert replay.was_duplicate is True
    assert replay.order.client_order_id == first.order.client_order_id
    # Never resubmitted to the broker at all -- create() catches the
    # duplicate before submit() would ever be called again.
    assert broker.position_tracker.held_quantity("NSE:INFY") == 10
    assert len(broker.fills) == 1


def test_a_lost_submission_response_resolves_via_the_brokers_own_truth(
    broker: PaperBroker, market_data: _FakeLiveMarketData, clock: _ClockBox
) -> None:
    market_data.set_quote(make_quote(ask_quantity=1000))
    order_manager = OrderManager(clock=clock)
    created = order_manager.create(
        "NSE:INFY",
        "buy",
        10,
        "limit",
        100.30,
        idempotency_key="strategy-signal-1",
        signal_id="sig-1",
        risk_decision_id="rd-1",
    )
    client_order_id = created.order.client_order_id
    flaky = _FlakyBroker(broker, fail_client_order_id=client_order_id)

    record = order_manager.submit(client_order_id, flaky)
    assert record.state is OrderState.UNKNOWN
    # The broker, unseen by the client, actually filled the order already.
    assert broker.position_tracker.held_quantity("NSE:INFY") == 10

    resolved = order_manager.handle_ambiguous_response(client_order_id, flaky)
    assert resolved.state is OrderState.FILLED
    assert resolved.filled_quantity == 10
