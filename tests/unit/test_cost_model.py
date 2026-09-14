"""CostModel: deterministic per-leg trade costs, the slippage estimate on
top, and the invariants a cost breakdown must always satisfy (its total
equals the sum of its rounded components, GST applies only where it
should, buy/sell asymmetry is correct, rounding is exact to the paisa).
"""

from __future__ import annotations

import datetime as dt

import pytest

from backtest.cost_schedule import CostSchedule, CostScheduleRepository
from backtest.costs import (
    CostCategory,
    CostModel,
    CostModelError,
    TradeCost,
    TradeSide,
    cost_pct_of_turnover,
    net_pnl,
)

TRADE_DATE = dt.date(2024, 1, 15)


def schedule(effective_from: dt.date = dt.date(2019, 1, 1), **overrides: object) -> CostSchedule:
    defaults: dict[str, object] = dict(
        effective_from=effective_from,
        label="test schedule",
        source="unit test fixture",
        brokerage_flat_inr=0.0,
        brokerage_pct=0.0,
        stt_buy_pct=0.001,
        stt_sell_pct=0.001,
        exchange_txn_pct=0.0000345,
        sebi_turnover_pct=0.0000010,
        gst_pct=0.18,
        stamp_duty_buy_pct=0.00015,
        stamp_duty_sell_pct=0.0,
        dp_charges_inr=15.93,
        other_charges_flat_inr=0.0,
        other_charges_pct=0.0,
    )
    defaults.update(overrides)
    return CostSchedule(**defaults)  # type: ignore[arg-type]


def model(
    *schedules: CostSchedule,
    min_slippage_bps: float = 5.0,
    impact_coefficient: float = 50.0,
) -> CostModel:
    repo = CostScheduleRepository(schedules or (schedule(),))
    return CostModel(repo, min_slippage_bps=min_slippage_bps, impact_coefficient=impact_coefficient)


# --------------------------------------------------------------------------
# Buy transaction
# --------------------------------------------------------------------------


def test_buy_transaction_charges_stt_and_stamp_duty() -> None:
    cost = model().compute_trade_cost("NSE:TCS", TradeSide.BUY, 100, 3500.0, TRADE_DATE)

    assert cost.turnover == pytest.approx(350_000.0)
    assert cost.stt == pytest.approx(350.0)  # 0.001 * 350000
    assert cost.stamp_duty == pytest.approx(52.5)  # 0.00015 * 350000
    assert cost.dp_charge == 0.0  # DP charge never applies on a buy


def test_buy_transaction_total_equals_sum_of_components() -> None:
    cost = model().compute_trade_cost("NSE:TCS", TradeSide.BUY, 100, 3500.0, TRADE_DATE)
    component_sum = (
        cost.brokerage
        + cost.stt
        + cost.exchange_txn_charge
        + cost.sebi_turnover_fee
        + cost.gst
        + cost.stamp_duty
        + cost.dp_charge
        + cost.other_charges
    )
    assert cost.total == pytest.approx(component_sum)


# --------------------------------------------------------------------------
# Sell transaction
# --------------------------------------------------------------------------


def test_sell_transaction_charges_dp_charge_not_stamp_duty() -> None:
    cost = model().compute_trade_cost("NSE:TCS", TradeSide.SELL, 100, 3500.0, TRADE_DATE)

    assert cost.stamp_duty == 0.0  # stamp duty never applies on a sell (V1 default schedule)
    assert cost.dp_charge == pytest.approx(15.93)
    assert cost.stt == pytest.approx(350.0)  # STT applies on both legs for delivery equity


def test_buy_and_sell_costs_differ_only_in_stamp_duty_and_dp_charge() -> None:
    buy = model().compute_trade_cost("NSE:TCS", TradeSide.BUY, 100, 3500.0, TRADE_DATE)
    sell = model().compute_trade_cost("NSE:TCS", TradeSide.SELL, 100, 3500.0, TRADE_DATE)

    assert buy.brokerage == sell.brokerage
    assert buy.stt == sell.stt
    assert buy.exchange_txn_charge == sell.exchange_txn_charge
    assert buy.sebi_turnover_fee == sell.sebi_turnover_fee
    assert buy.stamp_duty != sell.stamp_duty
    assert buy.dp_charge != sell.dp_charge


def test_asymmetric_stt_schedule_is_applied_per_side() -> None:
    """STT is 'instrument and transaction-side specific' -- a schedule with
    genuinely different buy/sell STT must be honored, not silently
    averaged or applied uniformly."""
    asymmetric = schedule(stt_buy_pct=0.002, stt_sell_pct=0.0005)
    m = model(asymmetric)

    buy = m.compute_trade_cost("NSE:X", TradeSide.BUY, 10, 100.0, TRADE_DATE)
    sell = m.compute_trade_cost("NSE:X", TradeSide.SELL, 10, 100.0, TRADE_DATE)

    assert buy.stt == pytest.approx(2.0)  # 0.002 * 1000
    assert sell.stt == pytest.approx(0.5)  # 0.0005 * 1000


# --------------------------------------------------------------------------
# Different turnover values
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quantity", "price"),
    [(1, 10.0), (10, 100.0), (500, 250.0), (10_000, 15.5), (1, 1_000_000.0)],
)
def test_percentage_components_scale_linearly_with_turnover(
    quantity: int, price: float
) -> None:
    cost = model().compute_trade_cost("NSE:X", TradeSide.BUY, quantity, price, TRADE_DATE)
    turnover = quantity * price
    assert cost.stt == pytest.approx(0.001 * turnover, abs=0.01)
    assert cost.exchange_txn_charge == pytest.approx(0.0000345 * turnover, abs=0.01)


def test_flat_components_do_not_scale_with_turnover() -> None:
    flat_schedule = schedule(brokerage_flat_inr=20.0, dp_charges_inr=15.93)
    m = model(flat_schedule)

    small = m.compute_trade_cost("NSE:X", TradeSide.SELL, 1, 10.0, TRADE_DATE)
    large = m.compute_trade_cost("NSE:X", TradeSide.SELL, 10_000, 1000.0, TRADE_DATE)

    assert small.brokerage == large.brokerage == pytest.approx(20.0)
    assert small.dp_charge == large.dp_charge == pytest.approx(15.93)


def test_larger_turnover_produces_larger_total_cost() -> None:
    m = model()
    small = m.compute_trade_cost("NSE:X", TradeSide.BUY, 10, 100.0, TRADE_DATE)
    large = m.compute_trade_cost("NSE:X", TradeSide.BUY, 10_000, 100.0, TRADE_DATE)
    assert large.total > small.total


def test_zero_or_negative_quantity_is_rejected() -> None:
    with pytest.raises(CostModelError, match="quantity"):
        model().compute_trade_cost("NSE:X", TradeSide.BUY, 0, 100.0, TRADE_DATE)
    with pytest.raises(CostModelError, match="quantity"):
        model().compute_trade_cost("NSE:X", TradeSide.BUY, -5, 100.0, TRADE_DATE)


def test_zero_or_negative_price_is_rejected() -> None:
    with pytest.raises(CostModelError, match="price"):
        model().compute_trade_cost("NSE:X", TradeSide.BUY, 10, 0.0, TRADE_DATE)
    with pytest.raises(CostModelError, match="price"):
        model().compute_trade_cost("NSE:X", TradeSide.BUY, 10, -50.0, TRADE_DATE)


# --------------------------------------------------------------------------
# Effective-date changes
# --------------------------------------------------------------------------


def test_cost_model_uses_the_schedule_effective_on_the_trade_date() -> None:
    old = schedule(dt.date(2019, 1, 1), stt_buy_pct=0.00125, label="old")
    new = schedule(dt.date(2023, 1, 1), stt_buy_pct=0.00100, label="new")
    m = model(old, new)

    before = m.compute_trade_cost("NSE:X", TradeSide.BUY, 100, 100.0, dt.date(2022, 12, 31))
    after = m.compute_trade_cost("NSE:X", TradeSide.BUY, 100, 100.0, dt.date(2023, 1, 1))

    assert before.schedule_effective_from == dt.date(2019, 1, 1)
    assert after.schedule_effective_from == dt.date(2023, 1, 1)
    assert before.stt == pytest.approx(12.5)
    assert after.stt == pytest.approx(10.0)


def test_same_trade_different_dates_can_produce_different_totals() -> None:
    old = schedule(dt.date(2019, 1, 1), gst_pct=0.15, label="old")
    new = schedule(dt.date(2022, 1, 1), gst_pct=0.18, label="new")
    m = model(old, new)

    before = m.compute_trade_cost("NSE:X", TradeSide.BUY, 100, 500.0, dt.date(2020, 1, 1))
    after = m.compute_trade_cost("NSE:X", TradeSide.BUY, 100, 500.0, dt.date(2023, 1, 1))

    assert before.total != after.total


# --------------------------------------------------------------------------
# Missing rates
# --------------------------------------------------------------------------


def test_trade_date_before_earliest_schedule_is_rejected() -> None:
    from backtest.cost_schedule import MissingCostScheduleError

    m = model(schedule(dt.date(2022, 1, 1)))
    with pytest.raises(MissingCostScheduleError):
        m.compute_trade_cost("NSE:X", TradeSide.BUY, 10, 100.0, dt.date(2021, 1, 1))


# --------------------------------------------------------------------------
# Rounding
# --------------------------------------------------------------------------


def test_charges_are_rounded_to_the_nearest_paisa() -> None:
    # 33 shares * 101.10 = 3336.30; stt = 0.001 * 3336.30 = 3.3363 -> 3.34
    cost = model().compute_trade_cost("NSE:X", TradeSide.BUY, 33, 101.10, TRADE_DATE)
    assert cost.stt == pytest.approx(3.34)
    for value in (
        cost.brokerage,
        cost.stt,
        cost.exchange_txn_charge,
        cost.sebi_turnover_fee,
        cost.gst,
        cost.stamp_duty,
        cost.dp_charge,
        cost.other_charges,
        cost.total,
    ):
        assert round(value, 2) == value


def test_rounding_is_half_up_not_bankers_rounding() -> None:
    """A charge landing exactly on a half-paisa must round up, not to even
    -- Python's built-in round() uses banker's rounding (round(0.125, 2)
    == 0.12), but a contract note rounds 0.125 up to 0.13."""
    half_paisa_schedule = schedule(brokerage_flat_inr=0.0, brokerage_pct=0.0000125)
    m = model(half_paisa_schedule)
    # turnover = 100 * 100 = 10000; brokerage = 0.0000125 * 10000 = 0.125 exactly
    cost = m.compute_trade_cost("NSE:X", TradeSide.BUY, 100, 100.0, TRADE_DATE)
    assert cost.brokerage == 0.13
    assert round(0.125, 2) == 0.12  # documents why plain round() would be wrong here


def test_total_is_exactly_the_sum_of_the_rounded_line_items() -> None:
    """GST and total must be computed from already-rounded components, so
    the displayed total never drifts a paisa from the sum of displayed
    rows -- a classic rounding-order bug."""
    odd_schedule = schedule(
        brokerage_flat_inr=1.0,
        brokerage_pct=0.0001234567,
        exchange_txn_pct=0.0000345678,
    )
    m = model(odd_schedule)
    cost = m.compute_trade_cost("NSE:X", TradeSide.BUY, 137, 733.33, TRADE_DATE)
    component_sum = (
        cost.brokerage
        + cost.stt
        + cost.exchange_txn_charge
        + cost.sebi_turnover_fee
        + cost.gst
        + cost.stamp_duty
        + cost.dp_charge
        + cost.other_charges
    )
    assert cost.total == round(component_sum, 2)


def test_gst_is_computed_on_rounded_brokerage_exchange_and_sebi_charges() -> None:
    gst_schedule = schedule(brokerage_flat_inr=0.0, brokerage_pct=0.0, gst_pct=0.18)
    m = model(gst_schedule)
    cost = m.compute_trade_cost("NSE:X", TradeSide.BUY, 137, 733.33, TRADE_DATE)
    expected_base = cost.brokerage + cost.exchange_txn_charge + cost.sebi_turnover_fee
    assert cost.gst == round(0.18 * expected_base, 2)


def test_gst_excludes_stt_and_stamp_duty() -> None:
    """GST applies to brokerage + exchange charges + SEBI fee only -- STT
    and stamp duty are themselves statutory levies, not a taxable
    service."""
    high_stt_schedule = schedule(stt_buy_pct=0.01, stamp_duty_buy_pct=0.01, gst_pct=0.18)
    m = model(high_stt_schedule)
    cost = m.compute_trade_cost("NSE:X", TradeSide.BUY, 100, 100.0, TRADE_DATE)
    # With brokerage/exchange/sebi all tiny relative to the inflated STT and
    # stamp duty, GST must stay small -- it is not 18% of the whole trade.
    assert cost.gst < cost.stt
    assert cost.gst < cost.stamp_duty


# --------------------------------------------------------------------------
# TradeCost dataclass invariants
# --------------------------------------------------------------------------


def test_trade_cost_rejects_total_not_matching_components() -> None:
    with pytest.raises(ValueError, match="total"):
        TradeCost(
            instrument_id="NSE:X",
            side=TradeSide.BUY,
            quantity=10,
            price=100.0,
            turnover=1000.0,
            schedule_effective_from=TRADE_DATE,
            brokerage=1.0,
            stt=1.0,
            exchange_txn_charge=1.0,
            sebi_turnover_fee=1.0,
            gst=1.0,
            stamp_duty=1.0,
            dp_charge=1.0,
            other_charges=1.0,
            total=999.0,
        )


def test_trade_cost_rejects_turnover_not_matching_quantity_times_price() -> None:
    with pytest.raises(ValueError, match="turnover"):
        TradeCost(
            instrument_id="NSE:X",
            side=TradeSide.BUY,
            quantity=10,
            price=100.0,
            turnover=500.0,
            schedule_effective_from=TRADE_DATE,
            brokerage=0.0,
            stt=0.0,
            exchange_txn_charge=0.0,
            sebi_turnover_fee=0.0,
            gst=0.0,
            stamp_duty=0.0,
            dp_charge=0.0,
            other_charges=0.0,
            total=0.0,
        )


def test_components_by_category_groups_correctly() -> None:
    cost = model().compute_trade_cost("NSE:X", TradeSide.SELL, 100, 3500.0, TRADE_DATE)
    grouped = cost.components_by_category()

    assert grouped[CostCategory.BROKER_DEPENDENT] == pytest.approx(
        cost.brokerage + cost.dp_charge
    )
    assert grouped[CostCategory.EXCHANGE_DEPENDENT] == pytest.approx(cost.exchange_txn_charge)
    assert grouped[CostCategory.DETERMINISTIC] == pytest.approx(
        cost.stt + cost.sebi_turnover_fee + cost.gst + cost.stamp_duty + cost.other_charges
    )
    assert grouped[CostCategory.ESTIMATED] == 0.0  # TradeCost never includes slippage


# --------------------------------------------------------------------------
# ExecutionCostEstimate: slippage on top of the deterministic cost
# --------------------------------------------------------------------------


def test_execution_cost_estimate_adds_slippage_to_trade_cost() -> None:
    m = model()
    est = m.estimate_execution_cost(
        "NSE:X",
        TradeSide.BUY,
        100,
        3500.0,
        TRADE_DATE,
        spread_bps=10.0,
        avg_daily_value=50_000_000.0,
        volatility=0.25,
    )
    assert est.total_cost == pytest.approx(est.trade_cost.total + est.slippage_amount)
    assert est.slippage_bps >= m.min_slippage_bps


def test_slippage_floor_applies_when_spread_and_impact_are_tiny() -> None:
    m = model(min_slippage_bps=25.0)
    est = m.estimate_execution_cost(
        "NSE:X",
        TradeSide.BUY,
        1,
        100.0,
        TRADE_DATE,
        spread_bps=0.0,
        avg_daily_value=1_000_000_000_000.0,
        volatility=0.0,
    )
    assert est.slippage_bps == pytest.approx(25.0)


def test_larger_participation_produces_larger_price_impact() -> None:
    m = model()
    thin = m.estimate_execution_cost(
        "NSE:X",
        TradeSide.BUY,
        1000,
        100.0,
        TRADE_DATE,
        spread_bps=5.0,
        avg_daily_value=10_000_000.0,
        volatility=0.3,
    )
    deep = m.estimate_execution_cost(
        "NSE:X",
        TradeSide.BUY,
        1000,
        100.0,
        TRADE_DATE,
        spread_bps=5.0,
        avg_daily_value=10_000_000_000.0,
        volatility=0.3,
    )
    assert thin.price_impact_bps > deep.price_impact_bps


def test_missing_liquidity_estimate_assumes_maximum_participation() -> None:
    """avg_daily_value <= 0 means "no reliable liquidity estimate" and must
    fail toward the worst case, not a fabricated near-zero impact."""
    m = model()
    unknown = m.estimate_execution_cost(
        "NSE:X",
        TradeSide.BUY,
        1000,
        100.0,
        TRADE_DATE,
        spread_bps=5.0,
        avg_daily_value=0.0,
        volatility=0.3,
    )
    known_full_participation = m.estimate_execution_cost(
        "NSE:X",
        TradeSide.BUY,
        1000,
        100.0,
        TRADE_DATE,
        spread_bps=5.0,
        avg_daily_value=100_000.0,  # order value == ADV -> participation 1.0
        volatility=0.3,
    )
    assert unknown.price_impact_bps == pytest.approx(known_full_participation.price_impact_bps)


def test_net_value_is_higher_than_gross_for_a_buy() -> None:
    est = model().estimate_execution_cost(
        "NSE:X", TradeSide.BUY, 100, 100.0, TRADE_DATE, 5.0, 10_000_000.0, 0.2
    )
    assert est.net_value > est.gross_value


def test_net_value_is_lower_than_gross_for_a_sell() -> None:
    est = model().estimate_execution_cost(
        "NSE:X", TradeSide.SELL, 100, 100.0, TRADE_DATE, 5.0, 10_000_000.0, 0.2
    )
    assert est.net_value < est.gross_value


def test_cost_pct_of_turnover_matches_total_cost_over_gross_value() -> None:
    est = model().estimate_execution_cost(
        "NSE:X", TradeSide.BUY, 100, 100.0, TRADE_DATE, 5.0, 10_000_000.0, 0.2
    )
    assert est.cost_pct_of_turnover == pytest.approx(est.total_cost / est.gross_value)


def test_negative_spread_bps_is_rejected() -> None:
    with pytest.raises(CostModelError, match="spread_bps"):
        model().estimate_slippage_bps(
            order_value=1000.0, spread_bps=-1.0, avg_daily_value=1_000_000.0, volatility=0.2
        )


def test_negative_volatility_is_rejected() -> None:
    with pytest.raises(CostModelError, match="volatility"):
        model().estimate_slippage_bps(
            order_value=1000.0, spread_bps=5.0, avg_daily_value=1_000_000.0, volatility=-0.1
        )


def test_cost_model_rejects_negative_min_slippage_bps() -> None:
    repo = CostScheduleRepository([schedule()])
    with pytest.raises(ValueError, match="min_slippage_bps"):
        CostModel(repo, min_slippage_bps=-1.0, impact_coefficient=10.0)


def test_cost_model_rejects_negative_impact_coefficient() -> None:
    repo = CostScheduleRepository([schedule()])
    with pytest.raises(ValueError, match="impact_coefficient"):
        CostModel(repo, min_slippage_bps=1.0, impact_coefficient=-10.0)


# --------------------------------------------------------------------------
# Backtest-report building blocks: gross P&L, costs, net P&L, cost % turnover
# --------------------------------------------------------------------------


def test_net_pnl_subtracts_total_cost_from_gross_pnl() -> None:
    assert net_pnl(gross_pnl=10_000.0, total_cost=250.0) == pytest.approx(9_750.0)


def test_net_pnl_can_go_negative_when_costs_exceed_gross_gains() -> None:
    assert net_pnl(gross_pnl=50.0, total_cost=250.0) == pytest.approx(-200.0)


def test_cost_pct_of_turnover_is_turnover_weighted() -> None:
    # Not the average of two trades' own cost ratios -- the ratio of totals.
    assert cost_pct_of_turnover(total_cost=300.0, total_turnover=1_000_000.0) == pytest.approx(
        0.0003
    )


def test_cost_pct_of_turnover_rejects_zero_turnover() -> None:
    with pytest.raises(CostModelError, match="total_turnover"):
        cost_pct_of_turnover(total_cost=100.0, total_turnover=0.0)
