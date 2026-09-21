"""BacktestEngine: the single-strategy simulation loop -- next-session
execution timing, cost deduction, risk-veto folding into the executed
portfolio, dividend crediting, and input validation. Look-ahead/leakage
properties specifically are covered in
``tests/unit/test_no_lookahead_walkforward.py``.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import pytest

from backtest.cost_schedule import CostScheduleRepository
from backtest.costs import CostModel, TradeSide
from backtest.engine import (
    BacktestEngine,
    BacktestEngineError,
    FillRecord,
    OrderRecord,
    _apply_rebalance_threshold,
    _apply_risk_decisions,
)
from core.regime.allocation import AllocationRegime, AllocationTarget
from core.regime.regime_policy import RegimePolicy
from data.models import DailyBar
from portfolio.portfolio_constructor import TargetPortfolio, TargetPosition, TradeAction
from risk.circuit_breaker import CircuitState
from risk.risk_manager import RiskCheck, RiskDecision, RiskViolation
from risk.stop_loss import StopLossPolicy, StopReason
from tests.unit._wf_support import Environment, cost_schedule, dividend


def full_exposure_target(as_of: dt.date, regime_policy: RegimePolicy) -> AllocationTarget:
    band = regime_policy.band_for(AllocationRegime.LOW_RISK)
    return AllocationTarget(
        as_of=as_of,
        regime=AllocationRegime.LOW_RISK,
        target_gross_exposure=band.max_gross_exposure,
        min_gross_exposure=band.max_gross_exposure,
        max_gross_exposure=band.max_gross_exposure,
        allow_new_positions=True,
        confidence=1.0,
        expected_volatility=0.1,
        reason="test",
    )


# --------------------------------------------------------------------------
# Buy / sell transaction mechanics
# --------------------------------------------------------------------------


def test_buy_transaction_fills_at_next_session_open_not_signal_close(tmp_path: Path) -> None:
    env = Environment(n_days=120, n_stocks=3)
    engine = env.engine(tmp_path)
    signal_date = env.dates[80]
    execution_date = env.calendar.next_trading_day(signal_date)
    target = full_exposure_target(signal_date, env.regime_policy)

    result = engine.run("buy_test", {signal_date: target}, [signal_date], 1_000_000.0)

    assert len(result.fills) == 3
    execution_bars = {
        instrument_id: env.market_data.get_equity_bars(
            instrument_id, execution_date, execution_date
        )[0]
        for instrument_id in env.instrument_ids
    }
    for fill in result.fills:
        assert fill.order.execution_date == execution_date
        assert fill.order.signal_date == signal_date
        expected_open = float(execution_bars[fill.order.instrument_id].open)
        assert fill.fill_price == pytest.approx(expected_open)
        assert fill.side is TradeSide.BUY


def test_buy_transaction_deducts_gross_value_plus_cost_from_cash(tmp_path: Path) -> None:
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path)
    signal_date = env.dates[80]
    target = full_exposure_target(signal_date, env.regime_policy)

    result = engine.run("buy_cash_test", {signal_date: target}, [signal_date], 1_000_000.0)

    (fill,) = result.fills
    assert fill.side is TradeSide.BUY
    assert fill.execution_cost.total_cost > 0
    # net_value for a buy is gross_value + total_cost -- what actually left
    # the cash account, not the sticker gross trade value.
    assert fill.execution_cost.net_value > fill.execution_cost.gross_value
    remaining_cash = 1_000_000.0 - fill.execution_cost.net_value

    execution_close = float(
        env.market_data.get_equity_bars(
            fill.order.instrument_id, fill.order.execution_date, fill.order.execution_date
        )[0].close
    )
    expected_equity = remaining_cash + fill.quantity * execution_close
    assert result.equity_curve.iloc[0] == pytest.approx(expected_equity)


def test_sell_transaction_never_exceeds_currently_held_quantity(tmp_path: Path) -> None:
    """Holdings persist across signal dates *within one run() call* -- a
    later day's sell must be sized against what was actually accumulated
    by the fills before it, not against the proposed weight alone."""
    env = Environment(n_days=140, n_stocks=1)
    engine = env.engine(tmp_path)
    buy_date = env.dates[80]
    sell_date = env.dates[82]
    full = full_exposure_target(buy_date, env.regime_policy)
    reduced_band_target = AllocationTarget(
        as_of=sell_date,
        regime=AllocationRegime.HIGH_RISK,
        target_gross_exposure=0.05,
        min_gross_exposure=0.0,
        max_gross_exposure=0.10,
        allow_new_positions=True,
        confidence=1.0,
        expected_volatility=0.3,
        reason="test",
    )

    result = engine.run(
        "buy_then_sell",
        {buy_date: full, sell_date: reduced_band_target},
        [buy_date, sell_date],
        1_000_000.0,
    )

    buy_fill = next(fill for fill in result.fills if fill.order.signal_date == buy_date)
    sell_fill = next(fill for fill in result.fills if fill.order.signal_date == sell_date)
    assert buy_fill.side is TradeSide.BUY
    assert sell_fill.side is TradeSide.SELL
    assert sell_fill.quantity <= buy_fill.quantity


def test_exit_sells_the_entire_position_not_a_weight_derived_approximation(
    tmp_path: Path,
) -> None:
    env = Environment(n_days=140, n_stocks=2)
    engine = env.engine(tmp_path)
    buy_date = env.dates[80]
    exit_date = env.dates[82]
    full = full_exposure_target(buy_date, env.regime_policy)
    all_cash_target = AllocationTarget(
        as_of=exit_date,
        regime=AllocationRegime.UNCERTAIN,
        target_gross_exposure=0.0,
        min_gross_exposure=0.0,
        max_gross_exposure=0.0,
        allow_new_positions=False,
        confidence=1.0,
        expected_volatility=0.3,
        reason="test",
    )

    result = engine.run(
        "buy_then_exit",
        {buy_date: full, exit_date: all_cash_target},
        [buy_date, exit_date],
        1_000_000.0,
    )

    held_quantities = {
        fill.order.instrument_id: fill.quantity
        for fill in result.fills
        if fill.order.signal_date == buy_date
    }
    exit_fills = {
        fill.order.instrument_id: fill
        for fill in result.fills
        if fill.order.signal_date == exit_date
    }
    assert held_quantities  # sanity: the opening buy actually happened
    for instrument_id, quantity in held_quantities.items():
        assert exit_fills[instrument_id].order.action is TradeAction.EXIT
        assert exit_fills[instrument_id].quantity == quantity
    assert result.positions_history[
        engine.calendar.next_trading_day(exit_date)
    ] == {}


# --------------------------------------------------------------------------
# Dividends
# --------------------------------------------------------------------------


def test_dividend_ex_date_credits_cash_proportional_to_held_quantity(tmp_path: Path) -> None:
    env = Environment(n_days=140, n_stocks=1)
    buy_date = env.dates[80]
    ex_date = env.calendar.next_trading_day(buy_date)
    env.corporate_actions = env.corporate_actions.__class__(
        [dividend(env.instrument_ids[0], ex_date, cash_amount=2.5)]
    )
    engine = env.engine(tmp_path)
    full = full_exposure_target(buy_date, env.regime_policy)

    result = engine.run("dividend_test", {buy_date: full}, [buy_date], 1_000_000.0)
    (fill,) = result.fills

    credit = engine.dividend_cash_credit({fill.order.instrument_id: fill.quantity}, ex_date)
    assert credit == pytest.approx(2.5 * fill.quantity)


def test_dividend_cash_credit_is_zero_without_a_corporate_action_provider(
    tmp_path: Path,
) -> None:
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path)
    engine.corporate_actions = None
    assert engine.dividend_cash_credit({"NSE:S00": 100}, env.dates[50]) == 0.0


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_run_rejects_empty_signal_dates(tmp_path: Path) -> None:
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path)
    with pytest.raises(BacktestEngineError, match="must not be empty"):
        engine.run("x", {}, [], 1_000_000.0)


def test_run_rejects_unsorted_signal_dates(tmp_path: Path) -> None:
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path)
    day_a, day_b = env.dates[80], env.dates[81]
    target = full_exposure_target(day_a, env.regime_policy)
    with pytest.raises(BacktestEngineError, match="ascending"):
        engine.run("x", {day_a: target, day_b: target}, [day_b, day_a], 1_000_000.0)


def test_run_rejects_missing_exposure_target_entries(tmp_path: Path) -> None:
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path)
    day_a, day_b = env.dates[80], env.dates[81]
    target = full_exposure_target(day_a, env.regime_policy)
    with pytest.raises(BacktestEngineError, match="missing"):
        engine.run("x", {day_a: target}, [day_a, day_b], 1_000_000.0)


def test_run_rejects_non_positive_initial_equity(tmp_path: Path) -> None:
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path)
    day = env.dates[80]
    target = full_exposure_target(day, env.regime_policy)
    with pytest.raises(BacktestEngineError, match="initial_equity"):
        engine.run("x", {day: target}, [day], 0.0)


# --------------------------------------------------------------------------
# _apply_risk_decisions
# --------------------------------------------------------------------------


def _position(instrument_id: str, weight: float, rank: int = 1) -> TargetPosition:
    return TargetPosition(
        instrument_id=instrument_id,
        symbol=instrument_id.split(":")[-1],
        target_weight=weight,
        sector=instrument_id,
        rank=rank,
        score=1.0,
        binding_constraint="unconstrained",
    )


def _portfolio(positions: list[TargetPosition]) -> TargetPortfolio:
    gross = sum(p.target_weight for p in positions)
    return TargetPortfolio(
        as_of=dt.date(2024, 1, 1),
        positions=tuple(positions),
        cash_weight=round(1.0 - gross, 12),
        regime=AllocationRegime.NORMAL_RISK,
        gross_exposure=gross,
    )


def _decision(instrument_id: str, approved: bool, weight: float) -> RiskDecision:
    violations = (
        ()
        if approved
        else (RiskViolation(RiskCheck.SINGLE_NAME_EXPOSURE, "test", None, None),)
    )
    return RiskDecision(
        instrument_id=instrument_id,
        approved=approved,
        target_weight=weight,
        circuit_state=CircuitState.NORMAL,
        violations=violations,
    )


def test_apply_risk_decisions_keeps_approved_positions_at_proposed_weight() -> None:
    proposed = _portfolio([_position("NSE:A", 0.3)])
    decisions = [_decision("NSE:A", True, 0.3)]
    result = _apply_risk_decisions(proposed, decisions, current=None)
    assert result.weight_for("NSE:A") == pytest.approx(0.3)


def test_apply_risk_decisions_keeps_rejected_held_position_at_current_weight() -> None:
    proposed = _portfolio([_position("NSE:A", 0.5)])
    current = _portfolio([_position("NSE:A", 0.2)])
    decisions = [_decision("NSE:A", False, 0.5)]
    result = _apply_risk_decisions(proposed, decisions, current)
    assert result.weight_for("NSE:A") == pytest.approx(0.2)


def test_apply_risk_decisions_excludes_rejected_position_not_currently_held() -> None:
    proposed = _portfolio([_position("NSE:A", 0.5)])
    decisions = [_decision("NSE:A", False, 0.5)]
    result = _apply_risk_decisions(proposed, decisions, current=None)
    assert "NSE:A" not in result
    assert result.cash_weight == pytest.approx(1.0)


def test_apply_risk_decisions_scales_down_if_kept_weights_exceed_one() -> None:
    proposed = _portfolio([_position("NSE:A", 0.9, rank=1), _position("NSE:B", 0.05, rank=2)])
    current = _portfolio([_position("NSE:A", 0.9)])
    decisions = [_decision("NSE:A", False, 0.9), _decision("NSE:B", True, 0.05)]
    result = _apply_risk_decisions(proposed, decisions, current)
    assert result.gross_exposure <= 1.0 + 1e-9
    assert result.cash_weight >= -1e-9


def test_apply_risk_decisions_never_produces_negative_cash() -> None:
    import numpy as np

    rng = np.random.default_rng(21)
    for _ in range(30):
        n = int(rng.integers(1, 5))
        proposed_positions = [
            _position(f"NSE:S{i}", float(rng.uniform(0.05, 0.3)), rank=i + 1) for i in range(n)
        ]
        current_positions = [
            _position(f"NSE:S{i}", float(rng.uniform(0.05, 0.3)), rank=i + 1)
            for i in range(n)
            if rng.random() > 0.5
        ]
        proposed = _portfolio(proposed_positions)
        current = _portfolio(current_positions) if current_positions else None
        decisions = [
            _decision(p.instrument_id, bool(rng.random() > 0.4), p.target_weight)
            for p in proposed_positions
        ]
        result = _apply_risk_decisions(proposed, decisions, current)
        assert result.cash_weight >= -1e-9
        assert result.gross_exposure <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# cash_history
# --------------------------------------------------------------------------


def test_cash_history_shares_the_equity_curve_index(tmp_path: Path) -> None:
    env = Environment(n_days=120, n_stocks=3)
    engine = env.engine(tmp_path)
    signal_date = env.dates[80]
    target = full_exposure_target(signal_date, env.regime_policy)

    result = engine.run("cash_test", {signal_date: target}, [signal_date], 1_000_000.0)

    assert list(result.cash_history.index) == list(result.equity_curve.index)


def test_cash_history_plus_position_value_equals_equity(tmp_path: Path) -> None:
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path)
    signal_date = env.dates[80]
    target = full_exposure_target(signal_date, env.regime_policy)

    result = engine.run("cash_test", {signal_date: target}, [signal_date], 1_000_000.0)

    (fill,) = result.fills
    execution_date = fill.order.execution_date
    close = float(
        env.market_data.get_equity_bars(fill.order.instrument_id, execution_date, execution_date)[
            0
        ].close
    )
    position_value = fill.quantity * close
    assert result.cash_history.loc[execution_date] + position_value == pytest.approx(
        result.equity_curve.loc[execution_date]
    )


# --------------------------------------------------------------------------
# _apply_rebalance_threshold
# --------------------------------------------------------------------------


def test_rebalance_threshold_reverts_a_small_drift_to_the_current_weight() -> None:
    current = _portfolio([_position("NSE:A", 0.30)])
    executed = _portfolio([_position("NSE:A", 0.305)])  # a 0.5pp drift
    result = _apply_rebalance_threshold(executed, current, threshold=0.02)
    assert result.weight_for("NSE:A") == pytest.approx(0.30)


def test_rebalance_threshold_allows_a_drift_above_the_threshold() -> None:
    current = _portfolio([_position("NSE:A", 0.30)])
    executed = _portfolio([_position("NSE:A", 0.40)])  # a 10pp drift
    result = _apply_rebalance_threshold(executed, current, threshold=0.02)
    assert result.weight_for("NSE:A") == pytest.approx(0.40)


def test_rebalance_threshold_never_opens_a_tiny_new_position() -> None:
    executed = _portfolio([_position("NSE:NEW", 0.01)])
    result = _apply_rebalance_threshold(executed, current=None, threshold=0.02)
    assert "NSE:NEW" not in result
    assert result.cash_weight == pytest.approx(1.0)


def test_rebalance_threshold_skips_a_small_exit() -> None:
    current = _portfolio([_position("NSE:A", 0.01)])
    executed = _portfolio([])  # proposal exits NSE:A entirely
    result = _apply_rebalance_threshold(executed, current, threshold=0.02)
    assert result.weight_for("NSE:A") == pytest.approx(0.01)


def test_rebalance_threshold_allows_a_large_exit() -> None:
    current = _portfolio([_position("NSE:A", 0.30)])
    executed = _portfolio([])
    result = _apply_rebalance_threshold(executed, current, threshold=0.02)
    assert "NSE:A" not in result


def test_rebalance_threshold_zero_is_a_no_op() -> None:
    current = _portfolio([_position("NSE:A", 0.30)])
    executed = _portfolio([_position("NSE:A", 0.305)])
    result = _apply_rebalance_threshold(executed, current, threshold=0.0)
    assert result.weight_for("NSE:A") == pytest.approx(0.305)


def test_rebalance_threshold_scales_down_if_reverted_weights_exceed_one() -> None:
    """Reverting a below-threshold exit back to its current weight while
    another position keeps its already-near-the-cap executed weight can
    push the total slightly over 1.0 -- must be scaled back down, never
    let through as implied leverage."""
    current = _portfolio([_position("NSE:A", 0.99, rank=1), _position("NSE:B", 0.005, rank=2)])
    executed = _portfolio([_position("NSE:A", 0.99, rank=1)])  # proposal drops tiny NSE:B
    result = _apply_rebalance_threshold(executed, current, threshold=0.01)
    assert result.gross_exposure <= 1.0 + 1e-9
    assert result.cash_weight >= -1e-9


def test_engine_with_a_rebalance_threshold_produces_fewer_or_equal_fills(
    tmp_path: Path,
) -> None:
    env = Environment(n_days=140, n_stocks=3)
    band = env.regime_policy.band_for(AllocationRegime.LOW_RISK)
    dates = env.dates[70:100]
    targets = {
        day: AllocationTarget(
            as_of=day,
            regime=AllocationRegime.LOW_RISK,
            target_gross_exposure=band.max_gross_exposure,
            min_gross_exposure=band.max_gross_exposure,
            max_gross_exposure=band.max_gross_exposure,
            allow_new_positions=True,
            confidence=1.0,
            expected_volatility=0.1,
            reason="test",
        )
        for day in dates
    }

    unthresholded = env.engine(tmp_path / "none")
    result_none = unthresholded.run("none", targets, dates, 1_000_000.0)

    thresholded = BacktestEngine(
        calendar=env.calendar,
        market_data=env.market_data,
        stock_selector=env.stock_selector,
        portfolio_constructor=env.portfolio_constructor,
        risk_config=env.risk_cfg,
        cost_model=env.cost_model,
        circuit_breaker_state_dir=tmp_path / "thresholded",
        corporate_actions=env.corporate_actions,
        min_rebalance_weight_delta=0.05,
    )
    result_thresholded = thresholded.run("thresholded", targets, dates, 1_000_000.0)

    assert len(result_thresholded.fills) <= len(result_none.fills)


# ---------------------------------------------------------------------------
# Marking a suspended holding
# ---------------------------------------------------------------------------


class _SuspendedMarketData:
    """Bars that stop dead, the way a suspended Indian equity's do."""

    def __init__(self, last_traded: dt.date, close: float) -> None:
        self.last_traded = last_traded
        self.close = close
        self.windows: list[int] = []

    def get_equity_bars(
        self, instrument_id: str, start: dt.date, end: dt.date, **_: object
    ) -> list[object]:
        self.windows.append((end - start).days)
        if start > self.last_traded:
            return []

        class _Bar:
            session_date = self.last_traded
            close = self.close

        return [_Bar()]


def _engine_with(market_data: object, tmp_path: Path) -> BacktestEngine:
    env = Environment(n_days=40)
    engine = env.engine(tmp_path)
    engine.market_data = market_data  # type: ignore[assignment]
    return engine


def test_a_holding_suspended_past_the_narrow_window_is_still_marked(tmp_path: Path) -> None:
    """Regression: this killed four 32-fold runs eleven folds in.

    ADANITRANS last traded 2021-06-08 and next traded 2021-09-13. Marking
    the portfolio on 2021-06-24 missed the 15-day window by one day, and the
    engine raised rather than valuing a position it was still holding.

    Refusing to value a held position is the wrong answer: the position
    exists, the portfolio is worth something, and last-traded-price is the
    convention for a suspended holding. Execution prices are unaffected --
    they come from _next_open, so a stale mark can never be mistaken for a
    price something traded at.
    """
    data = _SuspendedMarketData(dt.date(2021, 6, 8), 1592.6)
    engine = _engine_with(data, tmp_path)

    price = engine._last_close("NSE:ADANITRANS", dt.date(2021, 6, 24))

    assert price == pytest.approx(1592.6)
    assert max(data.windows) >= 400, "the wide fallback window must actually be tried"


def test_a_months_old_mark_is_recorded_rather_than_passing_unnoticed(tmp_path: Path) -> None:
    """A run that leaned on a months-old price is not equivalent to one that
    did not, so it has to be auditable afterwards.

    ADANITRANS stayed suspended until 2021-09-13, so the marks got steadily
    older through that window; the worst age is what gets kept.
    """
    data = _SuspendedMarketData(dt.date(2021, 6, 8), 1592.6)
    engine = _engine_with(data, tmp_path)

    engine._last_close("NSE:ADANITRANS", dt.date(2021, 9, 6))

    assert engine.stale_marks == {"NSE:ADANITRANS": 90}


def test_a_mark_inside_the_warn_threshold_is_not_flagged(tmp_path: Path) -> None:
    """The common case must stay clean or the record becomes noise nobody
    reads. Sixteen days without a trade is a thin stock, not an incident --
    and it is exactly the gap that broke the run, so this pins that the fix
    is about valuing the position, not about raising an alarm."""
    data = _SuspendedMarketData(dt.date(2021, 6, 8), 1592.6)
    engine = _engine_with(data, tmp_path)

    engine._last_close("NSE:ADANITRANS", dt.date(2021, 6, 24))

    assert engine.stale_marks == {}


def test_an_instrument_with_no_price_at_all_still_raises(tmp_path: Path) -> None:
    """Widening the search must not turn into never failing. An instrument
    that never traded is a real error, not a stale mark."""
    data = _SuspendedMarketData(dt.date(1990, 1, 1), 1.0)
    engine = _engine_with(data, tmp_path)

    with pytest.raises(BacktestEngineError, match="no price data"):
        engine._last_close("NSE:NEVERTRADED", dt.date(2021, 6, 24))


class _UnadjustableMarketData:
    """A provider whose ADJUSTED path raises the way an underivable
    corporate action makes it raise, and whose RAW path still works."""

    def __init__(self, last_traded: dt.date, close: float) -> None:
        self.last_traded = last_traded
        self.close = close
        self.raw_calls = 0

    def get_equity_bars(
        self, instrument_id: str, start: dt.date, end: dt.date, price_basis: object = None
    ) -> list[object]:
        from data.models import PriceBasis

        if price_basis is PriceBasis.ADJUSTED:
            raise ValueError(
                f"{instrument_id} demerger requires an explicit_price_factor; "
                "it cannot be derived from the action terms alone"
            )
        self.raw_calls += 1
        if start > self.last_traded:
            return []

        class _Bar:
            session_date = self.last_traded
            close = self.close

        return [_Bar()]


def test_a_holding_through_an_underivable_action_is_marked_not_refused(
    tmp_path: Path,
) -> None:
    """Regression: NSE:VEDL demerged 2026-04-30 and killed four 33-fold runs
    on their final fold, after ten hours each.

    The adjustment factor genuinely cannot be computed from the action
    terms. But the position is held and the portfolio has to be worth
    something, so it is marked at the last price the market actually
    printed, unadjusted -- which is the most defensible number available
    when no adjustment exists.
    """
    data = _UnadjustableMarketData(dt.date(2026, 4, 29), 412.5)
    engine = _engine_with(data, tmp_path)

    price = engine._last_close("NSE:VEDL", dt.date(2026, 5, 4))

    assert price == pytest.approx(412.5)
    assert data.raw_calls >= 1, "must fall back to the raw traded price"


def test_an_unadjustable_mark_is_counted_rather_than_silent(tmp_path: Path) -> None:
    """A run that valued a holding on unadjusted prices is not equivalent to
    one that did not, so it has to be visible afterwards."""
    data = _UnadjustableMarketData(dt.date(2026, 4, 29), 412.5)
    engine = _engine_with(data, tmp_path)

    engine._last_close("NSE:VEDL", dt.date(2026, 5, 4))

    assert engine.unadjustable_marks == {"NSE:VEDL": 1}


def test_a_normal_holding_records_no_unadjustable_mark(tmp_path: Path) -> None:
    data = _SuspendedMarketData(dt.date(2026, 5, 3), 412.5)
    engine = _engine_with(data, tmp_path)

    engine._last_close("NSE:OK", dt.date(2026, 5, 4))

    assert engine.unadjustable_marks == {}


def _stop_test_cost_model() -> CostModel:
    return CostModel(
        CostScheduleRepository([cost_schedule()]),
        min_slippage_bps=1.0,
        impact_coefficient=0.1,
    )


def _ledger_fill(instrument_id: str, quantity: int, price: float, side: TradeSide) -> FillRecord:
    order = OrderRecord(
        signal_date=dt.date(2024, 1, 1),
        execution_date=dt.date(2024, 1, 2),
        instrument_id=instrument_id,
        action=TradeAction.BUY if side is TradeSide.BUY else TradeAction.SELL,
        current_weight=0.0,
        target_weight=0.1,
        delta_weight=0.1,
    )
    estimate = _stop_test_cost_model().estimate_execution_cost(
        instrument_id,
        side,
        quantity,
        price,
        dt.date(2024, 1, 2),
        spread_bps=10.0,
        avg_daily_value=50_000_000.0,
        volatility=0.2,
    )
    return FillRecord(
        order=order, side=side, quantity=quantity, fill_price=price, execution_cost=estimate
    )


def _buy_fill(instrument_id: str, quantity: int, price: float) -> FillRecord:
    return _ledger_fill(instrument_id, quantity, price, TradeSide.BUY)


def _sell_fill(instrument_id: str, quantity: int, price: float) -> FillRecord:
    return _ledger_fill(instrument_id, quantity, price, TradeSide.SELL)



# --------------------------------------------------------------------------
# Per-position stops
# --------------------------------------------------------------------------


STOP_POLICY = StopLossPolicy(
    hard_stop_pct=0.03, trail_drop_pct=0.02, trail_arm_net_profit_pct=0.03
)
"""The shipped setting: +3% net closes the position outright."""

TRAIL_STOP_POLICY = StopLossPolicy(
    hard_stop_pct=0.03,
    trail_drop_pct=0.02,
    trail_arm_net_profit_pct=0.03,
    close_on_arm=False,
)


def _reshape_session(
    env: Environment,
    instrument_id: str,
    session_date: dt.date,
    *,
    high_mult: float,
    low_mult: float,
    close_mult: float,
) -> DailyBar:
    """Rewrite one session's bar around its own open, and return it.

    The multipliers are applied to the *open*, which is the price the buy
    filled at, so a test can say "this name fell 5% from where we bought it"
    without knowing anything about the synthetic price series underneath.
    """
    bars = env.market_data._bars[instrument_id]
    index = next(i for i, bar in enumerate(bars) if bar.session_date == session_date)
    open_price = float(bars[index].open)
    reshaped = DailyBar(
        instrument_id=instrument_id,
        session_date=session_date,
        open=Decimal(str(round(open_price, 4))),
        high=Decimal(str(round(open_price * high_mult, 4))),
        low=Decimal(str(round(open_price * low_mult, 4))),
        close=Decimal(str(round(open_price * close_mult, 4))),
        volume=bars[index].volume,
    )
    bars[index] = reshaped
    return reshaped


def test_a_hard_stop_exits_the_position_inside_the_execution_session(
    tmp_path: Path,
) -> None:
    """Bought at the open, fell 5% below it the same day. The stop is a
    resting order placed before the session, so it resolves within it rather
    than waiting for the next open -- an exit that waits a day is not a stop.
    """
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path, stop_loss_policy=STOP_POLICY)
    signal_date = env.dates[80]
    execution_date = env.calendar.next_trading_day(signal_date)
    instrument_id = env.instrument_ids[0]
    bar = _reshape_session(
        env, instrument_id, execution_date, high_mult=1.002, low_mult=0.95, close_mult=0.96
    )
    target = full_exposure_target(signal_date, env.regime_policy)

    result = engine.run("hard_stop", {signal_date: target}, [signal_date], 1_000_000.0)

    assert len(result.stop_exits) == 1
    exit_record = result.stop_exits[0]
    assert exit_record.breach.reason is StopReason.HARD_STOP
    assert exit_record.execution_date == execution_date
    assert exit_record.breach.stop_level == pytest.approx(float(bar.open) * 0.97)
    # Closed below the level, so the close is the fill -- the level was not
    # on offer at the end of the session.
    assert exit_record.breach.fill_price == pytest.approx(float(bar.close))
    assert exit_record.realized_pnl < 0

    sells = [fill for fill in result.fills if fill.side is TradeSide.SELL]
    assert len(sells) == 1
    assert sells[0].quantity == [f for f in result.fills if f.side is TradeSide.BUY][0].quantity
    assert result.positions_history[execution_date] == {}


def test_a_trailing_stop_banks_a_profitable_pullback(tmp_path: Path) -> None:
    """Up 10% at the high, closed 7% up -- a fall of more than 2% from the
    high, and the exit still clears 3% after costs."""
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path, stop_loss_policy=TRAIL_STOP_POLICY)
    signal_date = env.dates[80]
    execution_date = env.calendar.next_trading_day(signal_date)
    instrument_id = env.instrument_ids[0]
    bar = _reshape_session(
        env, instrument_id, execution_date, high_mult=1.10, low_mult=1.06, close_mult=1.07
    )
    target = full_exposure_target(signal_date, env.regime_policy)

    result = engine.run("trail_stop", {signal_date: target}, [signal_date], 1_000_000.0)

    assert len(result.stop_exits) == 1
    exit_record = result.stop_exits[0]
    assert exit_record.breach.reason is StopReason.TRAILING_PROFIT_STOP
    assert exit_record.breach.reference_price == pytest.approx(float(bar.high))
    assert exit_record.breach.stop_level == pytest.approx(float(bar.high) * 0.98)
    assert exit_record.realized_pnl > 0
    assert exit_record.breach.net_profit_pct > TRAIL_STOP_POLICY.trail_arm_net_profit_pct
    assert result.positions_history[execution_date] == {}


def test_a_trailing_stop_does_not_fire_on_a_pullback_that_is_not_yet_profitable(
    tmp_path: Path,
) -> None:
    """Same 2% fall from the high, but the high was only 2.5% up: selling
    here books a loss after costs, which is not what a profit-protecting
    rule is for. Only the hard stop may sell at a loss."""
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path, stop_loss_policy=STOP_POLICY)
    signal_date = env.dates[80]
    execution_date = env.calendar.next_trading_day(signal_date)
    _reshape_session(
        env,
        env.instrument_ids[0],
        execution_date,
        high_mult=1.025,
        low_mult=0.995,
        close_mult=1.0,
    )
    target = full_exposure_target(signal_date, env.regime_policy)

    result = engine.run("no_trail", {signal_date: target}, [signal_date], 1_000_000.0)

    assert result.stop_exits == ()
    assert result.positions_history[execution_date] != {}


def test_no_policy_means_no_stops_and_an_empty_record(tmp_path: Path) -> None:
    """The same crashing session, run without a policy, is left alone. An
    empty ``stop_exits`` on a run with no policy is not the same claim as an
    empty one on a run whose stops never fired, and the engine keeps both
    readable by never inventing a policy of its own."""
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path)
    signal_date = env.dates[80]
    execution_date = env.calendar.next_trading_day(signal_date)
    _reshape_session(
        env,
        env.instrument_ids[0],
        execution_date,
        high_mult=1.002,
        low_mult=0.95,
        close_mult=0.96,
    )
    target = full_exposure_target(signal_date, env.regime_policy)

    result = engine.run("no_policy", {signal_date: target}, [signal_date], 1_000_000.0)

    assert engine.stop_loss_policy is None
    assert result.stop_exits == ()
    assert result.positions_history[execution_date] != {}


def test_a_stopped_position_leaves_the_target_so_it_is_not_sold_twice(
    tmp_path: Path,
) -> None:
    """After a stop the name is gone from the book, so the next session must
    diff against a portfolio that no longer holds it. Left in, the engine
    would carry a weight for a position it does not own."""
    env = Environment(n_days=120, n_stocks=2)
    engine = env.engine(tmp_path, stop_loss_policy=STOP_POLICY)
    first, second = env.dates[80], env.dates[81]
    execution_date = env.calendar.next_trading_day(first)
    stopped = env.instrument_ids[0]
    _reshape_session(
        env, stopped, execution_date, high_mult=1.002, low_mult=0.95, close_mult=0.96
    )
    targets = {
        first: full_exposure_target(first, env.regime_policy),
        second: full_exposure_target(second, env.regime_policy),
    }

    result = engine.run("target_cleanup", targets, [first, second], 1_000_000.0)

    assert [exit_record.breach.instrument_id for exit_record in result.stop_exits] == [stopped]
    stop_sells = [
        fill
        for fill in result.fills
        if fill.side is TradeSide.SELL
        and fill.order.execution_date == execution_date
        and fill.order.instrument_id == stopped
    ]
    assert len(stop_sells) == 1
    # No second sale of a position that is already gone.
    later_sells = [
        fill
        for fill in result.fills
        if fill.side is TradeSide.SELL and fill.order.instrument_id == stopped
    ]
    assert len(later_sells) == 1


def test_a_holding_with_no_bar_is_recorded_as_unchecked_not_as_safe(
    tmp_path: Path,
) -> None:
    """A session the stop could not see is an unprotected session, and the
    run says so rather than passing over it silently. Inventing a breach from
    a price that does not exist would be worse."""
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path, stop_loss_policy=STOP_POLICY)
    first, second = env.dates[80], env.dates[81]
    instrument_id = env.instrument_ids[0]
    second_execution = env.calendar.next_trading_day(second)
    bars = env.market_data._bars[instrument_id]
    env.market_data._bars[instrument_id] = [
        bar for bar in bars if bar.session_date != second_execution
    ]
    targets = {
        first: full_exposure_target(first, env.regime_policy),
        second: full_exposure_target(second, env.regime_policy),
    }

    result = engine.run("unchecked", targets, [first, second], 1_000_000.0)

    assert result.stop_checks_skipped.get(instrument_id, 0) >= 1
    assert result.stop_exits == ()


# -- the ledger the stops read ---------------------------------------------


def test_average_price_is_weighted_across_buys_and_basis_includes_costs() -> None:
    """The hard stop is measured from ``avg_price`` and profitability from
    ``cost_basis``, and the two are different numbers: one is a price per
    share, the other is cash out of the door including charges. Conflating
    them would put the stop level in the wrong place on any position built
    in more than one go."""
    holdings: dict[str, int] = {}
    avg_price: dict[str, float] = {}
    cost_basis: dict[str, float] = {}
    cash = 100_000.0

    for quantity, price in ((100, 100.0), (100, 120.0)):
        fill = _buy_fill("NSE:ACME", quantity, price)
        cash = BacktestEngine._apply_fill(fill, cash, holdings, avg_price, cost_basis)

    assert holdings["NSE:ACME"] == 200
    assert avg_price["NSE:ACME"] == pytest.approx(110.0)
    # 22,000 of stock plus the buy leg's charges.
    assert cost_basis["NSE:ACME"] > 22_000.0
    assert cost_basis["NSE:ACME"] == pytest.approx(100_000.0 - cash)


def test_a_partial_sale_retires_its_share_of_the_basis_and_keeps_the_price() -> None:
    holdings: dict[str, int] = {}
    avg_price: dict[str, float] = {}
    cost_basis: dict[str, float] = {}
    cash = BacktestEngine._apply_fill(
        _buy_fill("NSE:ACME", 200, 110.0), 100_000.0, holdings, avg_price, cost_basis
    )
    full_basis = cost_basis["NSE:ACME"]

    BacktestEngine._apply_fill(
        _sell_fill("NSE:ACME", 50, 115.0), cash, holdings, avg_price, cost_basis
    )

    assert holdings["NSE:ACME"] == 150
    assert avg_price["NSE:ACME"] == pytest.approx(110.0)
    assert cost_basis["NSE:ACME"] == pytest.approx(full_basis * 0.75)


def test_a_take_profit_banks_the_gain_without_waiting_for_a_pullback(
    tmp_path: Path,
) -> None:
    """The shipped rule. Closed 7% up and still near its high: the trailing
    stop would hold on, the take profit sells."""
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path, stop_loss_policy=STOP_POLICY)
    signal_date = env.dates[80]
    execution_date = env.calendar.next_trading_day(signal_date)
    bar = _reshape_session(
        env,
        env.instrument_ids[0],
        execution_date,
        high_mult=1.075,
        low_mult=1.0,
        close_mult=1.07,
    )
    target = full_exposure_target(signal_date, env.regime_policy)

    result = engine.run("take_profit", {signal_date: target}, [signal_date], 1_000_000.0)

    assert len(result.stop_exits) == 1
    exit_record = result.stop_exits[0]
    assert exit_record.breach.reason is StopReason.TAKE_PROFIT
    # Sold at the close -- the price the rule was tested at, not the high.
    assert exit_record.breach.fill_price == pytest.approx(float(bar.close))
    assert exit_record.realized_pnl > 0
    assert result.positions_history[execution_date] == {}
