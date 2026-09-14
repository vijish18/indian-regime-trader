"""Pure, point-in-time factor computations for stock selection.

Every function here takes a price series that already ends at the decision
date and returns one number computed only from that series -- there is no
argument through which a later observation could reach these calculations.
The caller (``universe.stock_selector.StockSelector``) is responsible for
never fetching data past ``as_of`` in the first place; this module has no way
to check that from inside a bare ``pd.Series``, which is exactly why
``tests/unit/test_factor_calculator.py`` proves it externally, the same way
``tests/unit/test_feature_engineering.py`` proves causality for the market
features: append a future observation and check the earlier value is
unchanged.

## Why every factor uses adjusted prices, including liquidity

``data.models.DailyBar.adjusted()`` scales volume inversely to the price
factor specifically so that ``close * volume`` (traded value) is invariant
under adjustment: a 5-for-1 split scales price by 1/5 and volume by 5, and
the product is unchanged. That means every factor here -- including the
liquidity/traded-value factor, which needs the *real* INR value that changed
hands -- can safely use one uniformly adjusted bar series. Using raw prices
for liquidity and adjusted prices for the rest would be its own subtle bug;
using raw prices for momentum/trend/volatility/drawdown would be a much
louder one (a split would register as a fake 80% single-day crash).

## Sign convention

Every factor here is defined so that a *higher* value is more attractive,
with one deliberate exception: ``volatility`` is reported as the actual
annualized realized volatility (a positive number), not negated, because
that is the economically meaningful quantity to audit. The sign flip needed
to treat "calmer is better" as a ranking input happens once, explicitly, in
``universe.stock_selector.StockSelector`` where factors are combined into a
composite score -- not hidden inside this module's arithmetic.

## What's deliberately not here

There is no quality/fundamentals factor. docs/SPECIFICATION.md section 7.1
allows one "only when clean, point-in-time fundamental data is available" --
this codebase has no fundamentals data source (``data/`` covers price,
instrument, and corporate-action data only), so adding one now would mean
either fabricating a factor from nothing or building an entire new data
pipeline this phase doesn't need. The gap is documented, not silently
skipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from config.models import SelectionConfig

_TRADING_DAYS_PER_YEAR = 252


class InsufficientFactorHistoryError(ValueError):
    """A factor was asked to compute over more history than the series has.

    Raised rather than silently truncating the window, which would quietly
    compute a "6-month momentum" over three weeks of data for a
    recently-listed stock.
    """


@dataclass(frozen=True, slots=True)
class FactorSet:
    """One instrument's raw factor values as of one date, before
    cross-sectional standardization. Every field is directly interpretable
    on its own (a fraction, a log return, an INR value) -- the composite
    scoring in ``StockSelector`` is a separate, later step.
    """

    momentum: float
    """Risk-adjusted momentum: blended multi-horizon return (skipping the
    most recent short window) divided by realized volatility."""

    trend_persistence: float
    """Fraction of the trailing trend window spent with the close above its
    own trailing moving average, in ``[0, 1]``."""

    relative_strength: float
    """Log return of the stock minus the log return of NIFTY 50 over the
    same window -- positive means the stock outperformed the index."""

    volatility: float
    """Annualized realized volatility of daily log returns (positive; not
    sign-flipped -- see the module docstring)."""

    drawdown: float
    """Current drawdown from the trailing window's high, in ``(-inf, 0]``;
    0 means a new high today."""

    liquidity_inr: float
    """Average daily traded value (close * volume) over the liquidity
    window, in INR."""


def _closes(bars: pd.Series) -> pd.Series:
    if bars.empty:
        raise InsufficientFactorHistoryError("cannot compute a factor from an empty price series")
    if bars.isna().any():
        raise InsufficientFactorHistoryError(
            "price series contains NaN; StockSelector must filter for sufficient, "
            "complete history before handing it to a factor calculation"
        )
    return bars


def _tail(series: pd.Series, window: int, factor_name: str) -> pd.Series:
    if len(series) < window:
        raise InsufficientFactorHistoryError(
            f"{factor_name} needs {window} observations, series has {len(series)}"
        )
    return series.tail(window)


def log_return(closes: pd.Series, start_index: int, end_index: int) -> float:
    """``ln(closes[end_index] / closes[start_index])`` using positions
    counted from the end of the series (0 = the most recent close), so
    callers express "N sessions ago" directly without recomputing offsets.
    """
    closes = _closes(closes)
    if start_index >= len(closes) or end_index >= len(closes):
        raise InsufficientFactorHistoryError(
            f"log_return needs at least {max(start_index, end_index) + 1} observations, "
            f"series has {len(closes)}"
        )
    start_price = float(closes.iloc[-(start_index + 1)])
    end_price = float(closes.iloc[-(end_index + 1)])
    if start_price <= 0 or end_price <= 0:
        raise InsufficientFactorHistoryError("cannot take a log return of a non-positive price")
    return math.log(end_price / start_price)


def realized_volatility(closes: pd.Series, window: int) -> float:
    """Annualized population std of daily log returns over the trailing
    ``window`` sessions. Same convention (population std, ``sqrt(252)``) as
    ``core.features.feature_engineering`` and
    ``core.regime.baseline_policy``, so a factor and a regime feature
    measuring "volatility" always mean the same thing.
    """
    closes = _closes(closes)
    windowed = _tail(closes, window + 1, "realized_volatility")
    returns = cast(pd.Series, np.log(windowed / windowed.shift(1))).dropna()
    return float(returns.std(ddof=0) * math.sqrt(_TRADING_DAYS_PER_YEAR))


def momentum(
    closes: pd.Series, horizons_days: list[int], skip_days: int, volatility_window: int
) -> float:
    """Blended multi-horizon momentum, skipping the most recent
    ``skip_days`` sessions, divided by realized volatility.

    Skipping the most recent window follows docs/SPECIFICATION.md section
    7.1 ("excluding the most recent short window") -- a stock's most recent
    few weeks are dominated by short-term reversal effects that a medium-term
    momentum factor is not trying to capture.
    """
    closes = _closes(closes)
    horizon_returns = [
        log_return(closes, start_index=horizon - 1, end_index=skip_days)
        for horizon in horizons_days
    ]
    blended = sum(horizon_returns) / len(horizon_returns)
    volatility = realized_volatility(closes, volatility_window)
    denominator = max(volatility, 1e-4)  # floor, not a silent division by ~0
    return blended / denominator


def trend_persistence(closes: pd.Series, trend_window: int, ma_window: int) -> float:
    """Fraction of the trailing ``trend_window`` sessions where the close was
    above its own trailing ``ma_window``-session moving average.

    Distinct from momentum: a stock can have a large positive return (high
    momentum) while spending much of the window below its moving average
    (choppy, not persistent) -- this factor measures the consistency, not the
    magnitude, of the trend.
    """
    closes = _closes(closes)
    needed = trend_window + ma_window - 1
    windowed = _tail(closes, needed, "trend_persistence")
    moving_average = windowed.rolling(window=ma_window, min_periods=ma_window).mean()
    comparison = (windowed > moving_average).tail(trend_window)
    return float(comparison.mean())


def relative_strength(closes: pd.Series, index_closes: pd.Series, window: int) -> float:
    """Log return of ``closes`` minus the log return of ``index_closes``
    over the trailing ``window`` sessions -- both series must already be
    aligned to end on the same decision date.
    """
    stock_return = log_return(closes, start_index=window - 1, end_index=0)
    index_return = log_return(index_closes, start_index=window - 1, end_index=0)
    return stock_return - index_return


def drawdown(closes: pd.Series, window: int) -> float:
    """``close_t / rolling_max(closes, window)_t - 1``, always <= 0.

    Same construction as
    ``core.features.feature_engineering``'s ``nifty_drawdown_from_high_252d``,
    applied per-stock instead of to the index.
    """
    closes = _closes(closes)
    windowed = _tail(closes, window, "drawdown")
    return float(windowed.iloc[-1] / windowed.max() - 1.0)


def liquidity(closes: pd.Series, volumes: pd.Series, window: int) -> float:
    """Average daily traded value (close * volume) over the trailing
    ``window`` sessions, in the same currency as ``closes``.
    """
    closes = _closes(closes)
    if len(volumes) != len(closes):
        raise ValueError(
            f"closes has {len(closes)} observations but volumes has {len(volumes)}"
        )
    traded_value = (closes * volumes).tail(window)
    if len(traded_value) < window:
        raise InsufficientFactorHistoryError(
            f"liquidity needs {window} observations, series has {len(traded_value)}"
        )
    return float(traded_value.mean())


class FactorCalculator:
    """Computes the full ``FactorSet`` for one instrument from its adjusted
    price/volume history and the index's price history, both already ending
    at the decision date.
    """

    def __init__(self, config: SelectionConfig) -> None:
        self.config = config

    def compute(
        self, closes: pd.Series, volumes: pd.Series, index_closes: pd.Series
    ) -> FactorSet:
        """Raises ``InsufficientFactorHistoryError`` if any input series is
        shorter than the window a factor needs -- callers should have
        already filtered for sufficient history
        (``StockSelector.build_candidate_universe``) and should treat this
        as a bug, not an expected outcome, if it happens here.
        """
        config = self.config
        horizons_days = [months * 21 for months in config.momentum_lookback_months]
        return FactorSet(
            momentum=momentum(
                closes, horizons_days, config.momentum_skip_days, config.volatility_window_days
            ),
            trend_persistence=trend_persistence(
                closes, config.trend_window_days, config.trend_ma_window_days
            ),
            relative_strength=relative_strength(
                closes, index_closes, config.relative_strength_window_days
            ),
            volatility=realized_volatility(closes, config.volatility_window_days),
            drawdown=drawdown(closes, config.drawdown_window_days),
            liquidity_inr=liquidity(closes, volumes, config.liquidity_window_days),
        )
