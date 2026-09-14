"""Factor math: correctness of each formula, point-in-time causality (no
future observation may change a value already computed), and defensive
handling of missing/insufficient data.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import pytest

from config.models import SelectionConfig
from universe.factor_calculator import (
    FactorCalculator,
    InsufficientFactorHistoryError,
    drawdown,
    liquidity,
    log_return,
    momentum,
    realized_volatility,
    relative_strength,
    trend_persistence,
)


def series(values: list[float], start: dt.date = dt.date(2023, 1, 2)) -> pd.Series:
    dates = [start + dt.timedelta(days=i) for i in range(len(values))]
    return pd.Series(values, index=dates)


def constant_series(value: float, n: int, start: dt.date = dt.date(2023, 1, 2)) -> pd.Series:
    return series([value] * n, start)


def selection_config(**overrides: object) -> SelectionConfig:
    defaults: dict[str, object] = {
        "min_holdings": 3,
        "max_holdings": 5,
        "momentum_lookback_months": [1, 2],
        "momentum_skip_days": 5,
        "trend_ma_window_days": 5,
        "trend_window_days": 10,
        "relative_strength_window_days": 20,
        "volatility_window_days": 10,
        "drawdown_window_days": 20,
        "liquidity_window_days": 5,
        "min_history_days": 47,
        "factor_weights": {
            "momentum": 0.3,
            "trend_persistence": 0.15,
            "relative_strength": 0.2,
            "volatility": 0.15,
            "drawdown": 0.1,
            "liquidity": 0.1,
        },
    }
    defaults.update(overrides)
    return SelectionConfig.model_validate(defaults)


# --------------------------------------------------------------------------
# log_return
# --------------------------------------------------------------------------


def test_log_return_matches_hand_computation() -> None:
    prices = series([100.0, 101.0, 99.0, 105.0, 110.0])
    # start_index=4 (oldest, 100.0), end_index=0 (newest, 110.0)
    assert log_return(prices, start_index=4, end_index=0) == pytest.approx(math.log(110.0 / 100.0))


def test_log_return_zero_offset_is_zero() -> None:
    prices = series([100.0, 105.0])
    assert log_return(prices, start_index=0, end_index=0) == pytest.approx(0.0)


def test_log_return_rejects_non_positive_price() -> None:
    prices = series([100.0, -5.0])
    with pytest.raises(InsufficientFactorHistoryError, match="non-positive"):
        log_return(prices, start_index=1, end_index=0)


def test_log_return_rejects_out_of_range_index() -> None:
    prices = series([100.0, 101.0])
    with pytest.raises(InsufficientFactorHistoryError, match="needs at least"):
        log_return(prices, start_index=5, end_index=0)


# --------------------------------------------------------------------------
# realized_volatility
# --------------------------------------------------------------------------


def test_realized_volatility_of_a_constant_series_is_zero() -> None:
    prices = constant_series(100.0, 20)
    assert realized_volatility(prices, window=10) == pytest.approx(0.0, abs=1e-10)


def test_realized_volatility_matches_hand_computation() -> None:
    # Alternating +1%/-1% daily returns, a simple hand-checkable case.
    values = [100.0]
    for i in range(10):
        values.append(values[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
    prices = series(values)
    result = realized_volatility(prices, window=10)

    returns = [math.log(values[i] / values[i - 1]) for i in range(1, len(values))]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    expected = math.sqrt(variance) * math.sqrt(252)
    assert result == pytest.approx(expected)


def test_realized_volatility_needs_window_plus_one_observations() -> None:
    prices = constant_series(100.0, 10)
    with pytest.raises(InsufficientFactorHistoryError, match="needs 11"):
        realized_volatility(prices, window=10)


# --------------------------------------------------------------------------
# momentum
# --------------------------------------------------------------------------


def test_momentum_skips_the_most_recent_window() -> None:
    """A crash in the last skip_days must not affect the momentum score --
    only the return up to the skip boundary counts.
    """
    n = 60
    values = [100.0] * (n - 5) + [100.0, 50.0, 40.0, 30.0, 20.0]  # crash in the last 5 days
    prices = series(values)

    result = momentum(prices, horizons_days=[20], skip_days=5, volatility_window=10)
    # Flat for the first n-5 days, so momentum over [start=19, end=5] is ~0.
    assert result == pytest.approx(0.0, abs=1e-6)


def test_momentum_is_positive_for_a_rising_series() -> None:
    values = [100.0 * (1.01**i) for i in range(60)]
    prices = series(values)
    result = momentum(prices, horizons_days=[20, 40], skip_days=2, volatility_window=10)
    assert result > 0


def test_momentum_divides_by_realized_volatility() -> None:
    """The 'risk-adjusted' part: the same raw price path, scaled to be
    noisier, must produce a smaller momentum score.
    """
    n = 60
    base = [100.0 * (1.005**i) for i in range(n)]
    noisy = list(base)
    for i in range(1, n, 2):
        noisy[i] *= 1.05  # inject alternating noise, same trend underneath

    calm_prices = series(base)
    noisy_prices = series(noisy)

    calm_score = momentum(calm_prices, horizons_days=[20], skip_days=2, volatility_window=10)
    noisy_score = momentum(noisy_prices, horizons_days=[20], skip_days=2, volatility_window=10)
    assert calm_score > noisy_score > 0


def test_momentum_is_unaffected_by_future_prices() -> None:
    values = [100.0 * (1.004**i) for i in range(60)]
    prices = series(values)
    before = momentum(prices, horizons_days=[20, 40], skip_days=2, volatility_window=10)

    extended = pd.concat([prices, series([9999.0], start=prices.index[-1] + dt.timedelta(days=1))])
    after = momentum(extended.iloc[:60], horizons_days=[20, 40], skip_days=2, volatility_window=10)
    assert after == pytest.approx(before)


# --------------------------------------------------------------------------
# trend_persistence
# --------------------------------------------------------------------------


def test_trend_persistence_is_one_when_always_above_moving_average() -> None:
    values = [100.0 + i for i in range(30)]  # strictly increasing -> always above trailing MA
    prices = series(values)
    result = trend_persistence(prices, trend_window=10, ma_window=5)
    assert result == pytest.approx(1.0)


def test_trend_persistence_is_zero_when_always_below_moving_average() -> None:
    values = [200.0 - i for i in range(30)]  # strictly decreasing -> always below trailing MA
    prices = series(values)
    result = trend_persistence(prices, trend_window=10, ma_window=5)
    assert result == pytest.approx(0.0)


def test_trend_persistence_is_a_fraction_between_zero_and_one() -> None:
    values = [100.0, 102, 98, 103, 97, 104, 96, 105, 95, 106, 94, 107, 93, 108, 92]
    prices = series(values)
    result = trend_persistence(prices, trend_window=8, ma_window=4)
    assert 0.0 <= result <= 1.0


def test_trend_persistence_needs_window_plus_ma_minus_one_observations() -> None:
    prices = constant_series(100.0, 10)
    with pytest.raises(InsufficientFactorHistoryError, match="needs 14"):
        trend_persistence(prices, trend_window=10, ma_window=5)


# --------------------------------------------------------------------------
# relative_strength
# --------------------------------------------------------------------------


def test_relative_strength_is_positive_when_outperforming_the_index() -> None:
    stock = series([100.0 * (1.02**i) for i in range(20)])
    index = series([100.0 * (1.005**i) for i in range(20)])
    assert relative_strength(stock, index, window=15) > 0


def test_relative_strength_is_negative_when_underperforming_the_index() -> None:
    stock = series([100.0 * (1.001**i) for i in range(20)])
    index = series([100.0 * (1.02**i) for i in range(20)])
    assert relative_strength(stock, index, window=15) < 0


def test_relative_strength_is_zero_for_identical_paths() -> None:
    path = [100.0 * (1.01**i) for i in range(20)]
    assert relative_strength(series(path), series(path), window=15) == pytest.approx(0.0, abs=1e-10)


def test_relative_strength_matches_the_difference_of_log_returns() -> None:
    stock = series([100.0, 101.0, 103.0, 99.0, 108.0])
    index = series([100.0, 100.5, 101.0, 100.0, 102.0])
    result = relative_strength(stock, index, window=4)
    # relative_strength uses log_return(..., start_index=window-1, end_index=0).
    expected = log_return(stock, 3, 0) - log_return(index, 3, 0)
    assert result == pytest.approx(expected)


# --------------------------------------------------------------------------
# drawdown
# --------------------------------------------------------------------------


def test_drawdown_is_zero_at_a_new_high() -> None:
    values = [100.0 + i for i in range(20)]  # strictly increasing: today is the high
    assert drawdown(series(values), window=10) == pytest.approx(0.0)


def test_drawdown_is_negative_below_the_trailing_high() -> None:
    values = [100.0, 110.0, 105.0, 95.0, 90.0]
    result = drawdown(series(values), window=5)
    assert result == pytest.approx(90.0 / 110.0 - 1.0)


def test_drawdown_only_considers_the_trailing_window() -> None:
    """A high from before the window must not suppress today's drawdown
    reading -- only the trailing `window` sessions count.
    """
    old_high = [500.0] + [100.0] * 9  # spike far in the past, outside the window
    recent = [100.0, 105.0, 95.0]
    values = old_high + recent
    result = drawdown(series(values), window=3)
    assert result == pytest.approx(95.0 / 105.0 - 1.0)


# --------------------------------------------------------------------------
# liquidity
# --------------------------------------------------------------------------


def test_liquidity_is_the_mean_traded_value_over_the_window() -> None:
    closes = series([100.0, 100.0, 100.0, 100.0])
    volumes = series([1000.0, 2000.0, 3000.0, 4000.0])
    result = liquidity(closes, volumes, window=4)
    assert result == pytest.approx(100.0 * (1000 + 2000 + 3000 + 4000) / 4)


def test_liquidity_only_considers_the_trailing_window() -> None:
    closes = series([100.0] * 10)
    volumes = series([1_000_000.0] * 5 + [100.0] * 5)  # heavy volume outside the window
    result = liquidity(closes, volumes, window=5)
    assert result == pytest.approx(100.0 * 100.0)


def test_liquidity_rejects_mismatched_lengths() -> None:
    closes = series([100.0, 101.0, 102.0])
    volumes = series([1000.0, 2000.0])
    with pytest.raises(ValueError, match="closes has 3"):
        liquidity(closes, volumes, window=2)


# --------------------------------------------------------------------------
# Missing values / insufficient data
# --------------------------------------------------------------------------


def test_empty_series_raises_for_every_factor() -> None:
    empty = pd.Series(dtype=float)
    with pytest.raises(InsufficientFactorHistoryError, match="empty"):
        realized_volatility(empty, window=5)
    with pytest.raises(InsufficientFactorHistoryError, match="empty"):
        trend_persistence(empty, trend_window=5, ma_window=3)
    with pytest.raises(InsufficientFactorHistoryError, match="empty"):
        drawdown(empty, window=5)
    with pytest.raises(InsufficientFactorHistoryError, match="empty"):
        liquidity(empty, empty, window=5)


def test_nan_in_the_series_raises_rather_than_silently_propagating() -> None:
    values = [100.0, 101.0, float("nan"), 103.0, 104.0]
    with pytest.raises(InsufficientFactorHistoryError, match="NaN"):
        realized_volatility(series(values), window=3)


def test_insufficient_history_error_message_reports_shortfall() -> None:
    prices = constant_series(100.0, 5)
    with pytest.raises(InsufficientFactorHistoryError, match="needs 10 observations, series has 5"):
        drawdown(prices, window=10)


# --------------------------------------------------------------------------
# FactorCalculator.compute -- the full factor set together
# --------------------------------------------------------------------------


def _bull_market_setup(n: int = 60) -> tuple[pd.Series, pd.Series, pd.Series]:
    closes = series([100.0 * (1.006**i) for i in range(n)])
    volumes = series([1_000_000.0] * n)
    index_closes = series([100.0 * (1.002**i) for i in range(n)])
    return closes, volumes, index_closes


def test_compute_returns_every_factor() -> None:
    closes, volumes, index_closes = _bull_market_setup()
    calculator = FactorCalculator(selection_config())
    result = calculator.compute(closes, volumes, index_closes)

    assert result.momentum > 0
    assert 0.0 <= result.trend_persistence <= 1.0
    assert result.relative_strength > 0  # outperforming a slower-rising index
    assert result.volatility >= 0
    assert result.drawdown <= 0
    assert result.liquidity_inr > 0


def test_compute_is_deterministic() -> None:
    closes, volumes, index_closes = _bull_market_setup()
    calculator = FactorCalculator(selection_config())
    first = calculator.compute(closes, volumes, index_closes)
    second = calculator.compute(closes, volumes, index_closes)
    assert first == second


def test_compute_is_unaffected_by_data_appended_after_the_decision_date() -> None:
    """The end-to-end causality guarantee for the combined factor set, not
    just individual formulas.
    """
    closes, volumes, index_closes = _bull_market_setup(n=80)
    calculator = FactorCalculator(selection_config())

    cutoff = 60
    before = calculator.compute(
        closes.iloc[:cutoff], volumes.iloc[:cutoff], index_closes.iloc[:cutoff]
    )

    # Shock every series after the cutoff with wild, unrelated values.
    shocked_closes = closes.copy()
    shocked_closes.iloc[cutoff:] = 1_000_000.0
    shocked_volumes = volumes.copy()
    shocked_volumes.iloc[cutoff:] = 1.0
    shocked_index = index_closes.copy()
    shocked_index.iloc[cutoff:] = 0.01

    after = calculator.compute(
        shocked_closes.iloc[:cutoff], shocked_volumes.iloc[:cutoff], shocked_index.iloc[:cutoff]
    )
    assert after == before
