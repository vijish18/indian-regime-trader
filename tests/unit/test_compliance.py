"""Unit and integration tests for ``broker/compliance.py`` (Phase 16):
the India API/algo operational-controls gate. Every test here proves one
of the phase's explicit requirements -- a hard failure, never a
best-effort order, and no silent substitution of a prohibited order type.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping

import pytest

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
from broker.compliance import ComplianceError, ComplianceGate, ComplianceGuardedBroker
from broker.errors import BrokerRequestError
from config.models import ComplianceConfig
from execution.order_manager import OrderState

_T0 = dt.datetime(2024, 6, 3, 9, 0, 0, tzinfo=dt.UTC)


class _ClockBox:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + dt.timedelta(**kwargs)


def compliance_config(**overrides: object) -> ComplianceConfig:
    defaults: dict[str, object] = {
        "compliance_version": "test-fixture-v1",
        "broker_authorization_confirmed": True,
        "static_ip_primary": "203.0.113.10",
        "static_ip_secondary": None,
        "algo_identifier": "ALGO-TEST-0001",
        "allowed_order_types": ("LIMIT", "SL", "SL-M"),
        "allowed_validities": ("DAY", "TTL"),
        "max_orders_per_second": 3,
        "session_max_age_hours": 12.0,
        "audit_log_retention_years": 5,
    }
    defaults.update(overrides)
    return ComplianceConfig.model_validate(defaults)


def new_order(
    client_order_id: str = "c-1",
    *,
    order_type: str = "LIMIT",
    validity: str = "DAY",
    side: str = "buy",
    quantity: int = 10,
    limit_price: float = 1500.0,
) -> BrokerOrder:
    return BrokerOrder(
        client_order_id=client_order_id,
        broker_order_id=None,
        instrument_id="NSE:INFY",
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        status=OrderState.CREATED.value,
        validity=validity,
    )


# --------------------------------------------------------------------------
# ComplianceConfig -- schema-level validation
# --------------------------------------------------------------------------


def test_compliance_config_rejects_market_in_allowed_order_types() -> None:
    with pytest.raises(ValueError, match="prohibited type"):
        compliance_config(allowed_order_types=("LIMIT", "MARKET"))


def test_compliance_config_rejects_ioc_in_allowed_validities() -> None:
    with pytest.raises(ValueError, match="prohibited value"):
        compliance_config(allowed_validities=("DAY", "IOC"))


def test_compliance_config_rejects_a_malformed_static_ip() -> None:
    with pytest.raises(ValueError, match="valid IP address"):
        compliance_config(static_ip_primary="not-an-ip")


def test_compliance_config_accepts_the_placeholder_ip_as_syntactically_valid() -> None:
    # 0.0.0.0 is a real, parseable IP -- ComplianceGate is what rejects it
    # as a placeholder, not schema validation, so settings.yaml can load
    # in paper mode without a real IP on hand.
    config = compliance_config(static_ip_primary="0.0.0.0")
    assert config.static_ip_primary == "0.0.0.0"


def test_compliance_config_caps_max_orders_per_second_at_ten() -> None:
    with pytest.raises(ValueError):
        compliance_config(max_orders_per_second=11)


# --------------------------------------------------------------------------
# ComplianceGate -- construction-time "refuse to operate" gate
# --------------------------------------------------------------------------


def test_gate_constructs_successfully_with_a_fully_valid_config() -> None:
    ComplianceGate(compliance_config())  # must not raise


def test_gate_refuses_when_broker_authorization_not_confirmed() -> None:
    with pytest.raises(ComplianceError, match="broker authorization"):
        ComplianceGate(compliance_config(broker_authorization_confirmed=False))


def test_gate_refuses_a_placeholder_static_ip() -> None:
    with pytest.raises(ComplianceError, match="static IP"):
        ComplianceGate(compliance_config(static_ip_primary="0.0.0.0"))


def test_gate_refuses_a_placeholder_algo_identifier() -> None:
    with pytest.raises(ComplianceError, match="algo identifier"):
        ComplianceGate(compliance_config(algo_identifier="UNSET"))


# --------------------------------------------------------------------------
# ComplianceGate -- per-order checks, never substituting a prohibited value
# --------------------------------------------------------------------------


def test_check_order_type_accepts_an_allowed_type() -> None:
    gate = ComplianceGate(compliance_config())
    gate.check_order_type("LIMIT")  # must not raise


def test_check_order_type_rejects_market() -> None:
    gate = ComplianceGate(compliance_config())
    with pytest.raises(ComplianceError, match="unsupported order type"):
        gate.check_order_type("MARKET")


def test_check_validity_rejects_ioc() -> None:
    gate = ComplianceGate(compliance_config())
    with pytest.raises(ComplianceError, match="unsupported order validity"):
        gate.check_validity("IOC")


def test_check_order_injects_the_configured_algo_identifier_as_the_tag() -> None:
    gate = ComplianceGate(compliance_config(algo_identifier="ALGO-XYZ-42"))
    checked = gate.check_order(new_order())
    assert checked.tag == "ALGO-XYZ-42"


def test_check_order_never_substitutes_a_prohibited_type_it_just_refuses() -> None:
    gate = ComplianceGate(compliance_config())
    order = new_order(order_type="MARKET")
    with pytest.raises(ComplianceError):
        gate.check_order(order)
    # The original order is untouched -- there is nothing to "recover" a
    # substituted order from, because none was ever created.
    assert order.order_type == "MARKET"


# --------------------------------------------------------------------------
# ComplianceGate -- session / expired-authentication checks
# --------------------------------------------------------------------------


def test_check_session_accepts_a_fresh_active_session() -> None:
    gate = ComplianceGate(compliance_config(session_max_age_hours=12.0), clock=lambda: _T0)
    status = HealthStatus(
        healthy=True, detail="ok", checked_at=_T0, session_active=True, login_time=_T0
    )
    gate.check_session(status)  # must not raise


def test_check_session_rejects_an_inactive_session() -> None:
    gate = ComplianceGate(compliance_config(), clock=lambda: _T0)
    status = HealthStatus(
        healthy=False, detail="expired", checked_at=_T0, session_active=False, login_time=_T0
    )
    with pytest.raises(ComplianceError, match="expired authentication"):
        gate.check_session(status)


def test_check_session_rejects_a_missing_login_time() -> None:
    gate = ComplianceGate(compliance_config(), clock=lambda: _T0)
    status = HealthStatus(
        healthy=True, detail="ok", checked_at=_T0, session_active=True, login_time=None
    )
    with pytest.raises(ComplianceError, match="expired authentication"):
        gate.check_session(status)


def test_check_session_rejects_a_session_older_than_the_configured_max_age() -> None:
    clock = _ClockBox(_T0)
    gate = ComplianceGate(compliance_config(session_max_age_hours=1.0), clock=clock)
    status = HealthStatus(
        healthy=True, detail="ok", checked_at=_T0, session_active=True, login_time=_T0
    )
    clock.advance(hours=2)
    with pytest.raises(ComplianceError, match="expired authentication"):
        gate.check_session(status)


# --------------------------------------------------------------------------
# ComplianceGate -- order-per-second rate limiting
# --------------------------------------------------------------------------


def test_rate_limit_allows_orders_up_to_the_configured_maximum() -> None:
    clock = _ClockBox(_T0)
    gate = ComplianceGate(compliance_config(max_orders_per_second=3), clock=clock)
    for _ in range(3):
        gate.check_rate_limit()
        gate.record_order_submission()


def test_rate_limit_refuses_the_order_exceeding_the_maximum() -> None:
    clock = _ClockBox(_T0)
    gate = ComplianceGate(compliance_config(max_orders_per_second=3), clock=clock)
    for _ in range(3):
        gate.check_rate_limit()
        gate.record_order_submission()
    with pytest.raises(ComplianceError, match="order-per-second limit exceeded"):
        gate.check_rate_limit()


def test_rate_limit_window_slides_forward_over_time() -> None:
    clock = _ClockBox(_T0)
    gate = ComplianceGate(compliance_config(max_orders_per_second=1), clock=clock)
    gate.check_rate_limit()
    gate.record_order_submission()
    with pytest.raises(ComplianceError):
        gate.check_rate_limit()

    clock.advance(seconds=1.1)
    gate.check_rate_limit()  # must not raise -- the earlier order aged out


# --------------------------------------------------------------------------
# ComplianceGuardedBroker
# --------------------------------------------------------------------------


def _gate(**overrides: object) -> ComplianceGate:
    """A ComplianceGate pinned to _T0 -- matches _StubBroker's default
    health.login_time, so these tests are exercising the specific check
    they name, not incidentally tripping the real-clock session-age check
    against a fixed 2024 login_time.
    """
    return ComplianceGate(compliance_config(**overrides), clock=lambda: _T0)


class _StubBroker(Broker):
    """A fully controllable ``Broker`` double -- tracks every call so
    tests can assert the wrapper never reached the inner broker when it
    should have refused first.
    """

    def __init__(self) -> None:
        self.place_order_calls: list[BrokerOrder] = []
        self.modify_order_calls: list[tuple[str, dict[str, object]]] = []
        self.cancel_order_calls: list[str] = []
        self.positions: list[BrokerPosition] = []
        self.quotes: dict[str, BrokerQuote] = {}
        self.health = HealthStatus(
            healthy=True, detail="ok", checked_at=_T0, session_active=True, login_time=_T0
        )

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            broker_name="stub",
            supports_order_modification=True,
            supports_market_data_streaming=False,
            supported_exchanges=frozenset({"NSE"}),
            supported_products=frozenset({"CNC"}),
            supported_order_types=frozenset({"LIMIT"}),
            supported_varieties=frozenset({"regular"}),
        )

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        pass

    def get_account(self) -> BrokerAccount:
        return BrokerAccount(account_id="stub", equity=0.0, cash=0.0, buying_power=0.0, as_of=_T0)

    def get_positions(self) -> list[BrokerPosition]:
        return list(self.positions)

    def get_open_orders(self) -> list[BrokerOrder]:
        return []

    def get_order(self, order_id: str) -> BrokerOrder:
        raise NotImplementedError

    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        return []

    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]:
        return [self.quotes[i] for i in instrument_ids if i in self.quotes]

    def subscribe_market_data(
        self, instrument_ids: list[str], on_tick: Callable[[BrokerQuote], None]
    ) -> Callable[[], None]:
        raise NotImplementedError

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        self.place_order_calls.append(order)
        return order

    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder:
        self.modify_order_calls.append((order_id, changes))
        return new_order(order_id)

    def cancel_order(self, order_id: str) -> BrokerOrder:
        self.cancel_order_calls.append(order_id)
        return new_order(order_id)

    def close_position(self, instrument_id: str) -> BrokerOrder:
        raise NotImplementedError("ComplianceGuardedBroker must not delegate to this")

    def close_all_positions(self) -> list[BrokerOrder]:
        raise NotImplementedError("ComplianceGuardedBroker must not delegate to this")

    def health_check(self) -> HealthStatus:
        return self.health


def test_read_only_calls_pass_straight_through() -> None:
    inner = _StubBroker()
    inner.positions = [BrokerPosition(instrument_id="NSE:INFY", quantity=5, avg_price=1500.0)]
    guarded = ComplianceGuardedBroker(inner, _gate())

    assert guarded.get_positions() == inner.positions
    assert guarded.get_account().account_id == "stub"
    assert guarded.health_check() is inner.health
    assert guarded.capabilities().broker_name == "stub"


def test_place_order_injects_the_algo_identifier_tag_before_delegating() -> None:
    inner = _StubBroker()
    guarded = ComplianceGuardedBroker(
        inner, _gate(algo_identifier="ALGO-9")
    )
    guarded.place_order(new_order())
    assert len(inner.place_order_calls) == 1
    assert inner.place_order_calls[0].tag == "ALGO-9"


def test_place_order_rejects_a_prohibited_type_without_reaching_the_inner_broker() -> None:
    inner = _StubBroker()
    guarded = ComplianceGuardedBroker(inner, _gate())
    with pytest.raises(ComplianceError):
        guarded.place_order(new_order(order_type="MARKET"))
    assert inner.place_order_calls == []


def test_place_order_rejects_when_the_session_is_expired() -> None:
    inner = _StubBroker()
    inner.health = HealthStatus(
        healthy=False, detail="expired", checked_at=_T0, session_active=False, login_time=_T0
    )
    guarded = ComplianceGuardedBroker(inner, _gate())
    with pytest.raises(ComplianceError, match="expired authentication"):
        guarded.place_order(new_order())
    assert inner.place_order_calls == []


def test_place_order_rejects_beyond_the_rate_limit() -> None:
    inner = _StubBroker()
    clock = _ClockBox(_T0)
    inner.health = HealthStatus(
        healthy=True, detail="ok", checked_at=_T0, session_active=True, login_time=_T0
    )
    guarded = ComplianceGuardedBroker(
        inner, ComplianceGate(compliance_config(max_orders_per_second=2), clock=clock)
    )
    guarded.place_order(new_order("c-1"))
    guarded.place_order(new_order("c-2"))
    with pytest.raises(ComplianceError, match="order-per-second"):
        guarded.place_order(new_order("c-3"))
    assert len(inner.place_order_calls) == 2


def test_modify_order_rejects_a_change_to_a_prohibited_order_type() -> None:
    inner = _StubBroker()
    guarded = ComplianceGuardedBroker(inner, _gate())
    with pytest.raises(ComplianceError):
        guarded.modify_order("c-1", {"order_type": "MARKET"})
    assert inner.modify_order_calls == []


def test_modify_order_rejects_a_change_to_a_prohibited_validity() -> None:
    inner = _StubBroker()
    guarded = ComplianceGuardedBroker(inner, _gate())
    with pytest.raises(ComplianceError):
        guarded.modify_order("c-1", {"validity": "IOC"})
    assert inner.modify_order_calls == []


def test_modify_order_allows_a_change_to_an_allowed_field() -> None:
    inner = _StubBroker()
    guarded = ComplianceGuardedBroker(inner, _gate())
    guarded.modify_order("c-1", {"limit_price": 1510.0})
    assert inner.modify_order_calls == [("c-1", {"limit_price": 1510.0})]


def test_cancel_order_is_session_checked_before_delegating() -> None:
    inner = _StubBroker()
    inner.health = HealthStatus(
        healthy=False, detail="expired", checked_at=_T0, session_active=False, login_time=_T0
    )
    guarded = ComplianceGuardedBroker(inner, _gate())
    with pytest.raises(ComplianceError):
        guarded.cancel_order("c-1")
    assert inner.cancel_order_calls == []


def test_cancel_order_delegates_when_the_session_is_valid() -> None:
    inner = _StubBroker()
    guarded = ComplianceGuardedBroker(inner, _gate())
    guarded.cancel_order("c-1")
    assert inner.cancel_order_calls == ["c-1"]


def test_close_position_builds_and_tags_a_sell_order_via_place_order() -> None:
    inner = _StubBroker()
    inner.positions = [BrokerPosition(instrument_id="NSE:INFY", quantity=10, avg_price=1500.0)]
    inner.quotes = {
        "NSE:INFY": BrokerQuote(
            instrument_id="NSE:INFY", bid=1490.0, ask=1495.0, last_price=1492.0, as_of=_T0
        )
    }
    guarded = ComplianceGuardedBroker(
        inner, _gate(algo_identifier="ALGO-CLOSE")
    )

    order = guarded.close_position("NSE:INFY")
    assert order.side == "sell"
    assert order.quantity == 10
    assert order.limit_price == pytest.approx(1490.0)
    assert len(inner.place_order_calls) == 1
    assert inner.place_order_calls[0].tag == "ALGO-CLOSE"


def test_close_position_with_no_open_position_raises() -> None:
    inner = _StubBroker()
    guarded = ComplianceGuardedBroker(inner, _gate())
    with pytest.raises(BrokerRequestError, match="no open long position"):
        guarded.close_position("NSE:INFY")


def test_close_position_with_no_quote_raises() -> None:
    inner = _StubBroker()
    inner.positions = [BrokerPosition(instrument_id="NSE:INFY", quantity=10, avg_price=1500.0)]
    guarded = ComplianceGuardedBroker(inner, _gate())
    with pytest.raises(BrokerRequestError, match="no live quote"):
        guarded.close_position("NSE:INFY")


def test_close_all_positions_closes_every_position_via_place_order() -> None:
    inner = _StubBroker()
    inner.positions = [
        BrokerPosition(instrument_id="NSE:INFY", quantity=10, avg_price=1500.0),
        BrokerPosition(instrument_id="NSE:TCS", quantity=5, avg_price=3500.0),
    ]
    inner.quotes = {
        "NSE:INFY": BrokerQuote(
            instrument_id="NSE:INFY", bid=1490.0, ask=1495.0, last_price=1492.0, as_of=_T0
        ),
        "NSE:TCS": BrokerQuote(
            instrument_id="NSE:TCS", bid=3490.0, ask=3495.0, last_price=3492.0, as_of=_T0
        ),
    }
    guarded = ComplianceGuardedBroker(inner, _gate())

    orders = guarded.close_all_positions()
    assert len(orders) == 2
    assert {order.instrument_id for order in orders} == {"NSE:INFY", "NSE:TCS"}
    assert len(inner.place_order_calls) == 2
