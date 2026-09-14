"""Market-level feature engineering for the HMM regime engine.

The HMM (``core/regime/hmm_engine.py``, not yet implemented) is a MARKET
REGIME CLASSIFIER, not a directional stock predictor. Every feature here is
chosen to describe *how risky the market currently is*, not *which way it is
about to move* -- see docs/SPECIFICATION.md section 1.2, "Too many
directional features", and docs/ARCHITECTURE.md's note (B5) on this exact
risk. Where a feature uses signed returns at all (``nifty_return_1d``), it is
one single short-horizon return used as shock/momentum *context* alongside
eight volatility/trend/participation features -- not a bank of signed
lookback returns that could let the model rediscover market direction.

This is a deliberately small feature set (9 features). Nothing here is added
"because the data is available" -- every feature maps to one bullet in this
phase's brief, and no bullet has more than one feature.

## The feature table

Full documentation for each feature -- economic interpretation, calculation,
required lookback, and whether it is known at the decision timestamp -- lives
as structured data on its ``FeatureDefinition``, built by
``build_default_feature_definitions()`` below. That is the single source of
truth (inspect it directly, or call ``FeaturePipeline.audit()`` for a
per-value provenance trail); the summary below is just an index into it:

- ``nifty_return_1d`` -- 1-day log return: shock/momentum context.
- ``nifty_realized_vol_20d`` -- annualized trailing realized volatility.
- ``nifty_vol_ratio_5_20`` -- short vs. long realized vol: is vol accelerating?
- ``india_vix_level_z`` -- India VIX level, rescaled to a rolling z-score.
- ``india_vix_change_5d`` -- 5-day log change in India VIX: stress acceleration.
- ``nifty_trend_200d`` -- close vs. its 200-day moving average.
- ``nifty_drawdown_from_high_252d`` -- distance below the trailing 1-year high.
- ``nifty_atr_normalized_14d`` -- normalized average true range: gap/range environment.
- ``nifty_volume_stress_20d`` -- rolling z-score of log volume (optional, see below).

Every feature's last raw input is session t's own close (or VIX close) --
which only exists once session t has finished trading. That is exactly the
timestamp at which docs/SPECIFICATION.md section 10 says the daily decision
is made ("decision after session t; execution on session t+1"), so every
feature in this module is causal by construction, not by convention.

## Causality rules enforced here

- **Trailing windows only.** Every rolling computation uses
  ``pandas.Series.rolling(window=W)`` with the default ``center=False``.
  Centered windows are never used anywhere in this module -- a centered
  window at time t needs observations *after* t, which does not exist yet
  when the decision at t is made.
- **No global/expanding statistics.** ``rolling_standardize`` computes a
  z-score from a fixed trailing window, never from all data seen so far and
  never from the full series. An expanding-window or full-history normalizer
  would let an early-history z-score be defined using a mean/std that only
  becomes final once the *whole* dataset (including the future) is known.
- **Warm-up is NaN, not a guess.** Every rolling computation sets
  ``min_periods`` equal to its declared lookback, so a feature is NaN until
  it has a full window of real history -- never zero, never
  forward-filled, never estimated from a partial window.
- **Missing observations fail safely.** ``MarketFeatureInputs`` aligns NIFTY
  50 and India VIX by inner join on session date: a date present in only one
  series is dropped, not interpolated. Within an aligned date, pandas'
  documented ``rolling(..., min_periods=W)`` behavior skips individual NaN
  values inside the window and only returns NaN itself if fewer than ``W``
  non-NaN observations remain -- so an isolated bad/missing print degrades
  gracefully instead of poisoning every subsequent value or crashing the
  pipeline. See ``tests/unit/test_feature_engineering.py`` for both
  properties tested directly.

## Volume is optional ("where reliable")

Index-level volume for NIFTY 50 is not a uniformly reliable data series
across vendors (unlike volume for an individual equity). ``nifty_volume_stress_20d``
is therefore the only feature marked ``requires_volume=True``: if the supplied
volume series is entirely absent, ``FeaturePipeline`` silently omits this one
column from the output rather than failing the whole feature matrix or
fabricating a value. This is a documented data limitation, not a bug: a
feature matrix built without volume data is expected and valid, and callers
should check ``"nifty_volume_stress_20d" in matrix.columns`` rather than
assume it is always present.

## Deliberately deferred (not in this phase)

docs/SPECIFICATION.md section 5 also lists overnight gap, generic range
expansion, and breadth stress (advance/decline). They are left out of this
deliberately small set: overnight gap and range expansion are close enough in
economic content to ``nifty_atr_normalized_14d`` that adding both would
violate "do not add dozens of indicators just because they are available";
breadth stress needs point-in-time index membership joined against every
constituent's price history (``universe/universe.py`` plus per-stock bars),
which is a heavier lift only worth taking once Phase 6 stock-level data flows
exist. Both are natural follow-ups, not oversights.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from config.models import FeaturesConfig
from data.models import IndexObservation

_TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarketFeatureInputs:
    """Raw causal inputs to the feature pipeline: one row per session date
    present in *both* the NIFTY 50 and India VIX series, ascending, with no
    duplicate dates.

    This class performs its own defensive alignment (inner join, sort,
    duplicate rejection) but is not a substitute for calendar validation --
    build it from series that have already passed
    ``data.data_quality.IndexObservationValidator``, so a date missing here
    reflects a genuine vendor gap rather than an unvalidated series.
    """

    frame: pd.DataFrame
    """Index: ``pd.DatetimeIndex`` (session date). Columns: ``nifty_close``,
    ``nifty_high``, ``nifty_low``, ``vix_close``, ``volume`` (``volume`` is
    all-NaN when no participation data was supplied -- see "Volume is
    optional" in the module docstring)."""

    _REQUIRED_COLUMNS = ("nifty_close", "nifty_high", "nifty_low", "vix_close", "volume")

    def __post_init__(self) -> None:
        missing = set(self._REQUIRED_COLUMNS) - set(self.frame.columns)
        if missing:
            raise ValueError(
                f"MarketFeatureInputs.frame is missing column(s): {sorted(missing)}"
            )
        if not isinstance(self.frame.index, pd.DatetimeIndex):
            raise ValueError("MarketFeatureInputs.frame must be indexed by a DatetimeIndex")
        if self.frame.index.has_duplicates:
            raise ValueError("MarketFeatureInputs.frame has duplicate session dates")
        if not self.frame.index.is_monotonic_increasing:
            raise ValueError("MarketFeatureInputs.frame index must be sorted ascending")

    @property
    def dates(self) -> tuple[dt.date, ...]:
        return tuple(timestamp.date() for timestamp in self.frame.index)

    @property
    def has_volume(self) -> bool:
        return not self.frame["volume"].isna().all()

    @classmethod
    def from_index_observations(
        cls,
        nifty: Sequence[IndexObservation],
        india_vix: Sequence[IndexObservation],
    ) -> MarketFeatureInputs:
        """Build inputs from the Phase 2-3 domain objects.

        Raises:
            ValueError: if either series is empty, contains a duplicate
                session date, or the two series share no common date.
        """
        if not nifty:
            raise ValueError("cannot build feature inputs from an empty NIFTY 50 series")
        if not india_vix:
            raise ValueError("cannot build feature inputs from an empty India VIX series")

        nifty_frame = _observations_to_frame(nifty, prefix="nifty")
        vix_frame = _observations_to_frame(india_vix, prefix="vix")
        joined = nifty_frame.join(vix_frame[["vix_close"]], how="inner")
        if joined.empty:
            raise ValueError(
                "NIFTY 50 and India VIX series share no common session date"
            )
        return cls(joined[list(cls._REQUIRED_COLUMNS)])


def _observations_to_frame(observations: Sequence[IndexObservation], prefix: str) -> pd.DataFrame:
    """Convert one index's observations to a sorted, Decimal-to-float frame.

    Float conversion happens exactly here, at the pandas boundary -- see
    data/models.py's module docstring on why domain objects stay Decimal.
    """
    dates = [pd.Timestamp(observation.session_date) for observation in observations]
    if len(set(dates)) != len(dates):
        raise ValueError(f"{prefix} observations contain a duplicate session date")
    frame = pd.DataFrame(
        {
            f"{prefix}_close": [float(observation.close) for observation in observations],
            f"{prefix}_high": [
                float(observation.high) if observation.high is not None else float("nan")
                for observation in observations
            ],
            f"{prefix}_low": [
                float(observation.low) if observation.low is not None else float("nan")
                for observation in observations
            ],
            "volume": [
                float(observation.volume) if observation.volume is not None else float("nan")
                for observation in observations
            ],
        },
        index=pd.DatetimeIndex(dates, name="session_date"),
    )
    return frame.sort_index()


# --------------------------------------------------------------------------
# Pure, causal transforms
# --------------------------------------------------------------------------


def rolling_standardize(
    series: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    """Causal rolling z-score: ``(x_t - trailing_mean_t) / trailing_std_t``,
    computed from a trailing window ending at (and including) ``t``.

    Never centered (``center=False`` always -- a centered window at ``t``
    would use observations after ``t``) and never fit on more than the
    trailing ``window`` observations (no expanding or full-series
    statistics), so appending future rows can never change an
    already-computed value. Both properties are tested directly in
    ``tests/unit/test_feature_engineering.py``.

    Uses the population standard deviation (``ddof=0``): this is a
    descriptive statistic of the observed window itself, not an estimate
    extrapolated to a larger population.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2 to compute a standard deviation, got {window}")
    resolved_min_periods = window if min_periods is None else min_periods
    rolling = series.rolling(window=window, min_periods=resolved_min_periods, center=False)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _log_returns(close: pd.Series) -> pd.Series:
    return cast(pd.Series, np.log(close / close.shift(1)))


def _realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Annualized population std of trailing daily log returns."""
    returns = _log_returns(close)
    return returns.rolling(window=window, min_periods=window, center=False).std(
        ddof=0
    ) * math.sqrt(_TRADING_DAYS_PER_YEAR)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range, strictly requiring the prior close: the first session of
    any series has an undefined true range by this definition (not just a
    high-low approximation), so a 14-session ATR needs exactly 15 raw bars.
    """
    prior_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prior_close).abs(), (low - prior_close).abs()], axis=1
    )
    return ranges.max(axis=1, skipna=False)


# --------------------------------------------------------------------------
# Feature definitions and snapshots
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Documents and computes exactly one feature.

    ``compute`` must be a pure function of ``MarketFeatureInputs``: the same
    inputs always produce the same output series, and appending rows after
    time t must never change the value already computed for t (enforced by
    the no-look-ahead tests).
    """

    name: str
    economic_interpretation: str
    calculation: str
    required_lookback: int
    known_at_decision_timestamp: bool
    compute: Callable[[MarketFeatureInputs], pd.Series]
    requires_volume: bool = False


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """One feature's value at one timestamp, with enough provenance to audit
    it: which raw session dates fed into this value, and how large a window
    the feature declares it needs.
    """

    name: str
    timestamp: dt.date
    value: float
    source_observations: tuple[dt.date, ...]
    lookback: int

    @property
    def is_warmup(self) -> bool:
        """True if this value is NaN because fewer than ``lookback``
        observations were available yet.
        """
        return math.isnan(self.value)


def build_default_feature_definitions(config: FeaturesConfig) -> tuple[FeatureDefinition, ...]:
    """The default, deliberately small feature set, parameterized by
    ``config.features`` (see config/settings.yaml).
    """

    def nifty_return_1d(inputs: MarketFeatureInputs) -> pd.Series:
        return _log_returns(inputs.frame["nifty_close"])

    def nifty_realized_vol(inputs: MarketFeatureInputs) -> pd.Series:
        return _realized_volatility(inputs.frame["nifty_close"], config.realized_vol_window)

    def nifty_vol_ratio(inputs: MarketFeatureInputs) -> pd.Series:
        short = _realized_volatility(inputs.frame["nifty_close"], config.vol_ratio_short_window)
        long = _realized_volatility(inputs.frame["nifty_close"], config.vol_ratio_long_window)
        return short / long.replace(0.0, np.nan)

    def india_vix_level_z(inputs: MarketFeatureInputs) -> pd.Series:
        return rolling_standardize(inputs.frame["vix_close"], config.vix_zscore_window)

    def india_vix_change(inputs: MarketFeatureInputs) -> pd.Series:
        vix = inputs.frame["vix_close"]
        return cast(pd.Series, np.log(vix / vix.shift(config.vix_change_window)))

    def nifty_trend(inputs: MarketFeatureInputs) -> pd.Series:
        close = inputs.frame["nifty_close"]
        sma = close.rolling(window=config.trend_window, min_periods=config.trend_window).mean()
        return close / sma - 1

    def nifty_drawdown(inputs: MarketFeatureInputs) -> pd.Series:
        close = inputs.frame["nifty_close"]
        trailing_high = close.rolling(
            window=config.drawdown_window, min_periods=config.drawdown_window
        ).max()
        return close / trailing_high - 1

    def nifty_atr_normalized(inputs: MarketFeatureInputs) -> pd.Series:
        frame = inputs.frame
        true_range = _true_range(frame["nifty_high"], frame["nifty_low"], frame["nifty_close"])
        atr = true_range.rolling(window=config.atr_window, min_periods=config.atr_window).mean()
        return atr / frame["nifty_close"]

    def nifty_volume_stress(inputs: MarketFeatureInputs) -> pd.Series:
        log_volume = np.log(inputs.frame["volume"])
        return rolling_standardize(log_volume, config.volume_stress_window)

    return (
        FeatureDefinition(
            name="nifty_return_1d",
            economic_interpretation=(
                "Immediate shock/momentum context for the current session. "
                "The only signed-return feature in this set, kept singular and "
                "short-horizon per docs/ARCHITECTURE.md's B5 note so the HMM is "
                "not handed a bank of directional signals."
            ),
            calculation="ln(close_t / close_{t-1})",
            required_lookback=2,
            known_at_decision_timestamp=True,
            compute=nifty_return_1d,
        ),
        FeatureDefinition(
            name="nifty_realized_vol_20d",
            economic_interpretation=(
                "Core volatility/risk-state signal: dispersion of recent daily "
                "returns, annualized."
            ),
            calculation=(
                f"population_std(daily log returns, trailing {config.realized_vol_window} "
                "sessions) * sqrt(252)"
            ),
            required_lookback=config.realized_vol_window + 1,
            known_at_decision_timestamp=True,
            compute=nifty_realized_vol,
        ),
        FeatureDefinition(
            name="nifty_vol_ratio_5_20",
            economic_interpretation=(
                "Volatility acceleration: short-horizon realized vol relative to "
                "the longer baseline -- a common signature of a regime transition."
            ),
            calculation=(
                f"realized_vol({config.vol_ratio_short_window}d) / "
                f"realized_vol({config.vol_ratio_long_window}d)"
            ),
            required_lookback=config.vol_ratio_long_window + 1,
            known_at_decision_timestamp=True,
            compute=nifty_vol_ratio,
        ),
        FeatureDefinition(
            name="india_vix_level_z",
            economic_interpretation=(
                "Forward-looking implied volatility, rescaled so its level is "
                "comparable across years rather than compared to an absolute "
                "threshold that drifts over time."
            ),
            calculation=(
                f"rolling z-score of India VIX close over the trailing "
                f"{config.vix_zscore_window} sessions"
            ),
            required_lookback=config.vix_zscore_window,
            known_at_decision_timestamp=True,
            compute=india_vix_level_z,
        ),
        FeatureDefinition(
            name="india_vix_change_5d",
            economic_interpretation="Stress acceleration: has implied volatility risen sharply?",
            calculation=f"ln(vix_t / vix_{{t-{config.vix_change_window}}})",
            required_lookback=config.vix_change_window + 1,
            known_at_decision_timestamp=True,
            compute=india_vix_change,
        ),
        FeatureDefinition(
            name="nifty_trend_200d",
            economic_interpretation=(
                "Medium/long-term trend context: is the market above or below "
                "its long-run trend? Descriptive regime context, not a trading rule."
            ),
            calculation=f"close_t / SMA_{config.trend_window}(close)_t - 1",
            required_lookback=config.trend_window,
            known_at_decision_timestamp=True,
            compute=nifty_trend,
        ),
        FeatureDefinition(
            name="nifty_drawdown_from_high_252d",
            economic_interpretation=(
                "Classic stress proxy: distance below the trailing 1-year high."
            ),
            calculation=f"close_t / rolling_max_{config.drawdown_window}(close)_t - 1",
            required_lookback=config.drawdown_window,
            known_at_decision_timestamp=True,
            compute=nifty_drawdown,
        ),
        FeatureDefinition(
            name="nifty_atr_normalized_14d",
            economic_interpretation="Range/gap environment, independent of price level.",
            calculation=(
                f"mean(true_range, trailing {config.atr_window} sessions) / close_t "
                "(true_range requires the prior close, so the first session of "
                "any series is undefined by construction)"
            ),
            required_lookback=config.atr_window + 1,
            known_at_decision_timestamp=True,
            compute=nifty_atr_normalized,
        ),
        FeatureDefinition(
            name="nifty_volume_stress_20d",
            economic_interpretation=(
                "Participation/stress: is trading volume abnormal relative to its "
                "recent baseline? Included only when volume data is supplied and "
                "reliable -- see 'Volume is optional' in this module's docstring."
            ),
            calculation=(
                f"rolling z-score of ln(volume) over the trailing "
                f"{config.volume_stress_window} sessions"
            ),
            required_lookback=config.volume_stress_window,
            known_at_decision_timestamp=True,
            compute=nifty_volume_stress,
            requires_volume=True,
        ),
    )


DEFAULT_FEATURE_CONFIG = FeaturesConfig(
    realized_vol_window=20,
    vol_ratio_short_window=5,
    vol_ratio_long_window=20,
    vix_zscore_window=252,
    vix_change_window=5,
    trend_window=200,
    drawdown_window=252,
    atr_window=14,
    volume_stress_window=20,
)
"""Mirrors the ``features:`` section of config/settings.yaml. Used to build
``DEFAULT_FEATURES`` below; a real deployment should load
``config.load_settings().features`` instead of relying on this mirror
drifting in sync by hand.
"""

DEFAULT_FEATURES: tuple[FeatureDefinition, ...] = build_default_feature_definitions(
    DEFAULT_FEATURE_CONFIG
)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


class FeaturePipeline:
    """Computes the market-level feature matrix for the HMM from raw NIFTY
    50 / India VIX data.

    Every column is a pure function of ``MarketFeatureInputs``: identical
    inputs always produce an identical matrix, and appending rows after time
    t never changes the value already computed for t. This property is
    enforced by the no-look-ahead tests in
    ``tests/unit/test_feature_engineering.py``, not merely asserted here.
    """

    def __init__(self, definitions: Sequence[FeatureDefinition] = DEFAULT_FEATURES) -> None:
        self.definitions = tuple(definitions)

    def compute(self, inputs: MarketFeatureInputs) -> pd.DataFrame:
        """The full feature matrix: one column per applicable definition, one
        row per aligned session date, ascending.

        A ``requires_volume`` definition is silently omitted (not filled with
        NaN, not an error) when ``inputs.has_volume`` is False -- see "Volume
        is optional" in this module's docstring.
        """
        columns: dict[str, pd.Series] = {}
        for definition in self.definitions:
            if definition.requires_volume and not inputs.has_volume:
                continue
            columns[definition.name] = definition.compute(inputs)
        return pd.DataFrame(columns, index=inputs.frame.index)

    def audit(self, inputs: MarketFeatureInputs) -> list[FeatureSnapshot]:
        """One ``FeatureSnapshot`` per (feature, timestamp) pair actually
        present in :meth:`compute`'s output -- a full provenance record of
        every value the pipeline produced, including NaN warm-up values.

        ``source_observations`` is derived structurally from each
        definition's declared ``required_lookback`` and the row's position in
        the aligned date index -- every feature here is a simple trailing
        window over that one shared index, so "the last N aligned dates
        ending at t" is exactly what each ``compute`` function consumed.
        """
        matrix = self.compute(inputs)
        dates = inputs.dates
        by_name = {definition.name: definition for definition in self.definitions}

        snapshots: list[FeatureSnapshot] = []
        for name in matrix.columns:
            definition = by_name[name]
            series = matrix[name]
            for position, timestamp in enumerate(dates):
                lower = max(0, position - definition.required_lookback + 1)
                raw_value = series.iloc[position]
                snapshots.append(
                    FeatureSnapshot(
                        name=name,
                        timestamp=timestamp,
                        value=float(raw_value) if pd.notna(raw_value) else float("nan"),
                        source_observations=dates[lower : position + 1],
                        lookback=definition.required_lookback,
                    )
                )
        return snapshots


def snapshots_to_frame(snapshots: Sequence[FeatureSnapshot]) -> pd.DataFrame:
    """Flatten an audit into a tabular form for inspection: one row per
    (feature, timestamp), columns ``name, timestamp, value, lookback,
    source_observations, is_warmup``.
    """
    return pd.DataFrame(
        {
            "name": [snapshot.name for snapshot in snapshots],
            "timestamp": [snapshot.timestamp for snapshot in snapshots],
            "value": [snapshot.value for snapshot in snapshots],
            "lookback": [snapshot.lookback for snapshot in snapshots],
            "source_observations": [snapshot.source_observations for snapshot in snapshots],
            "is_warmup": [snapshot.is_warmup for snapshot in snapshots],
        }
    )
