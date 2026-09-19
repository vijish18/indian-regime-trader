"""BacktestEngine: the single-strategy simulation loop -- next-session
execution timing, cost deduction, risk-veto folding into the executed
portfolio, dividend crediting, and input validation. Look-ahead/leakage
properties specifically are covered in
``tests/unit/test_no_lookahead_walkforward.py``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from backtest.costs import TradeSide
from backtest.engine import (
    BacktestEngine,
    BacktestEngineError,
    _apply_rebalance_threshold,
    _apply_risk_decisions,
)
from core.regime.allocation import AllocationRegime, AllocationTarget
from core.regime.regime_policy import RegimePolicy
from portfolio.portfolio_constructor import TargetPortfolio, TargetPosition, TradeAction
from risk.circuit_breaker import CircuitState
from risk.risk_manager import RiskCheck, RiskDecision, RiskViolation
from tests.unit._wf_support import Environment, dividend


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
