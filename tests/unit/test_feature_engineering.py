"""Feature engineering: causality (no future observations, no centered
windows, no future-fit normalizers), NaN warm-up handling, missing-data
safety, and the audit trail.
"""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from config.models import FeaturesConfig
from core.features.feature_engineering import (
    DEFAULT_FEATURE_CONFIG,
    FeaturePipeline,
    FeatureSnapshot,
    MarketFeatureInputs,
    build_default_feature_definitions,
    rolling_standardize,
    snapshots_to_frame,
)
from data.models import IndexObservation

# Small windows so warm-up boundaries are cheap to compute exactly and easy
# to hand-verify, while exercising the exact same code paths as production.
TINY_CONFIG = FeaturesConfig(
    realized_vol_window=3,
    vol_ratio_short_window=2,
    vol_ratio_long_window=3,
    vix_zscore_window=4,
    vix_change_window=2,
    trend_window=3,
    drawdown_window=3,
    atr_window=2,
    volume_stress_window=3,
)


def _obs(
    index_symbol: str,
    day: dt.date,
    close: Decimal | float | str,
    *,
    high: Decimal | float | str | None = None,
    low: Decimal | float | str | None = None,
    volume: int | None = None,
) -> IndexObservation:
    return IndexObservation(
        index_symbol=index_symbol,
        session_date=day,
        close=Decimal(str(close)),
        high=Decimal(str(high)) if high is not None else None,
        low=Decimal(str(low)) if low is not None else None,
        volume=volume,
    )


def _dates(n: int, start: dt.date = dt.date(2023, 1, 2)) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range(n)]


def _random_walk(n: int, seed: int, base: float, vol: float) -> list[float]:
    rng = np.random.default_rng(seed)
    values = [base]
    for _ in range(n - 1):
        values.append(max(0.01, values[-1] * (1 + rng.normal(0, vol))))
    return values


def _nifty_series(
    n: int, seed: int = 1, start: dt.date = dt.date(2023, 1, 2)
) -> list[IndexObservation]:
    closes = _random_walk(n, seed, base=20_000.0, vol=0.01)
    rng = np.random.default_rng(seed + 1000)
    return [
        _obs(
            "NIFTY50",
            day,
            round(close, 2),
            high=round(close * 1.004, 2),
            low=round(close * 0.996, 2),
            volume=int(abs(rng.normal(1_000_000, 50_000))),
        )
        for day, close in zip(_dates(n, start), closes, strict=True)
    ]


def _vix_series(
    n: int, seed: int = 2, start: dt.date = dt.date(2023, 1, 2)
) -> list[IndexObservation]:
    closes = _random_walk(n, seed, base=14.0, vol=0.03)
    return [
        _obs("INDIAVIX", day, round(close, 2))
        for day, close in zip(_dates(n, start), closes, strict=True)
    ]


def _pipeline(config: FeaturesConfig = TINY_CONFIG) -> FeaturePipeline:
    return FeaturePipeline(build_default_feature_definitions(config))


# --------------------------------------------------------------------------
# 1. No future observations affect a feature
# --------------------------------------------------------------------------


def test_appending_future_rows_does_not_change_past_feature_values() -> None:
    n = 60
    nifty_full = _nifty_series(n)
    vix_full = _vix_series(n)
    cutoff = 40

    pipeline = _pipeline()
    matrix_short = pipeline.compute(
        MarketFeatureInputs.from_index_observations(nifty_full[:cutoff], vix_full[:cutoff])
    )
    matrix_full = pipeline.compute(
        MarketFeatureInputs.from_index_observations(nifty_full, vix_full)
    )

    pd.testing.assert_frame_equal(matrix_full.loc[matrix_short.index], matrix_short)


def test_appending_a_wild_future_outlier_does_not_change_past_values() -> None:
    """A single extreme future print is the sharpest version of this test: a
    non-causal implementation (expanding stats, centered windows, a
    look-ahead bug) would visibly move earlier values; a causal one cannot.
    """
    n = 50
    nifty = _nifty_series(n)
    vix = _vix_series(n)
    pipeline = _pipeline()

    before = pipeline.compute(MarketFeatureInputs.from_index_observations(nifty, vix))

    outlier_day = nifty[-1].session_date + dt.timedelta(days=1)
    shocked_nifty = [
        *nifty,
        _obs("NIFTY50", outlier_day, 100_000, high=100_500, low=99_500, volume=999_999_999),
    ]
    shocked_vix = [*vix, _obs("INDIAVIX", outlier_day, 90.0)]
    after = pipeline.compute(
        MarketFeatureInputs.from_index_observations(shocked_nifty, shocked_vix)
    )

    pd.testing.assert_frame_equal(after.loc[before.index], before)


def test_audit_source_observations_never_include_a_later_date() -> None:
    n = 25
    inputs = MarketFeatureInputs.from_index_observations(_nifty_series(n), _vix_series(n))
    snapshots = _pipeline().audit(inputs)
    for snapshot in snapshots:
        assert all(source <= snapshot.timestamp for source in snapshot.source_observations)


# --------------------------------------------------------------------------
# 2. Rolling normalization uses only past/current data
# --------------------------------------------------------------------------


def test_rolling_standardize_matches_hand_computed_trailing_window() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = rolling_standardize(series, window=3, min_periods=3)

    window_values = [3.0, 4.0, 5.0]  # the 3 values trailing (and including) index 4
    mean = sum(window_values) / 3
    std = math.sqrt(sum((v - mean) ** 2 for v in window_values) / 3)
    assert result.iloc[4] == pytest.approx((5.0 - mean) / std)


def test_rolling_standardize_window_is_trailing_not_centered() -> None:
    """At position 3 (0-indexed) of a 10-point series with window=4, a
    trailing window uses indices [0,1,2,3]; a centered window would use
    something straddling index 3 instead (e.g. [1,2,3,4]). Asserting the
    exact trailing set pins down which one this function actually computes.
    """
    series = pd.Series(range(1, 11), dtype=float)
    z = rolling_standardize(series, window=4, min_periods=4)

    trailing_values = series.iloc[0:4].tolist()
    mean = sum(trailing_values) / 4
    std = math.sqrt(sum((v - mean) ** 2 for v in trailing_values) / 4)
    assert z.iloc[3] == pytest.approx((series.iloc[3] - mean) / std)


def test_rolling_standardize_value_is_unaffected_by_future_appends() -> None:
    series = pd.Series([10.0, 11.0, 9.0, 12.0, 8.0, 20.0, 21.0, 19.0, 22.0, 18.0])
    before = rolling_standardize(series, window=4)

    extended = pd.concat([series, pd.Series([1_000_000.0])], ignore_index=True)
    after = rolling_standardize(extended, window=4)

    pd.testing.assert_series_equal(after.iloc[: len(series)], before)


def test_rolling_standardize_rejects_window_below_two() -> None:
    with pytest.raises(ValueError, match="window must be >= 2"):
        rolling_standardize(pd.Series([1.0, 2.0, 3.0]), window=1)


def test_rolling_standardize_handles_a_zero_variance_window_without_inf() -> None:
    """A constant window has zero std; dividing by it must produce NaN, not
    +/-inf, which would silently poison every downstream computation that
    touches it.
    """
    series = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
    result = rolling_standardize(series, window=3, min_periods=3)
    assert result.iloc[2:].apply(lambda v: math.isnan(v)).all()
    assert not np.isinf(result.fillna(0.0)).any()


# --------------------------------------------------------------------------
# 3. Missing observations fail safely
# --------------------------------------------------------------------------


def test_dates_present_in_only_one_series_are_dropped_not_interpolated() -> None:
    nifty = _nifty_series(10)
    vix = _vix_series(10)
    dropped_date = vix[5].session_date
    del vix[5]  # a genuine vendor gap on the VIX side

    inputs = MarketFeatureInputs.from_index_observations(nifty, vix)
    assert len(inputs.dates) == 9
    assert dropped_date not in inputs.dates


def test_empty_nifty_series_raises() -> None:
    with pytest.raises(ValueError, match="empty NIFTY 50"):
        MarketFeatureInputs.from_index_observations([], _vix_series(5))


def test_empty_vix_series_raises() -> None:
    with pytest.raises(ValueError, match="empty India VIX"):
        MarketFeatureInputs.from_index_observations(_nifty_series(5), [])


def test_no_overlapping_dates_raises() -> None:
    nifty = _nifty_series(5, start=dt.date(2023, 1, 2))
    vix = _vix_series(5, start=dt.date(2024, 6, 1))
    with pytest.raises(ValueError, match="no common session date"):
        MarketFeatureInputs.from_index_observations(nifty, vix)


def test_duplicate_session_date_in_source_observations_raises() -> None:
    nifty = _nifty_series(5)
    nifty.append(nifty[-1])
    with pytest.raises(ValueError, match="duplicate session date"):
        MarketFeatureInputs.from_index_observations(nifty, _vix_series(6))


def test_market_feature_inputs_rejects_unsorted_frame_index() -> None:
    frame = pd.DataFrame(
        {
            "nifty_close": [1.0, 2.0],
            "nifty_high": [1.0, 2.0],
            "nifty_low": [1.0, 2.0],
            "vix_close": [10.0, 11.0],
            "volume": [100.0, 100.0],
        },
        index=pd.DatetimeIndex([dt.date(2023, 1, 3), dt.date(2023, 1, 2)]),
    )
    with pytest.raises(ValueError, match="sorted ascending"):
        MarketFeatureInputs(frame)


def test_market_feature_inputs_rejects_duplicate_frame_index() -> None:
    frame = pd.DataFrame(
        {
            "nifty_close": [1.0, 2.0],
            "nifty_high": [1.0, 2.0],
            "nifty_low": [1.0, 2.0],
            "vix_close": [10.0, 11.0],
            "volume": [100.0, 100.0],
        },
        index=pd.DatetimeIndex([dt.date(2023, 1, 2), dt.date(2023, 1, 2)]),
    )
    with pytest.raises(ValueError, match="duplicate session dates"):
        MarketFeatureInputs(frame)


def test_market_feature_inputs_requires_all_columns() -> None:
    frame = pd.DataFrame(
        {"nifty_close": [1.0]}, index=pd.DatetimeIndex([dt.date(2023, 1, 2)])
    )
    with pytest.raises(ValueError, match="missing column"):
        MarketFeatureInputs(frame)


def test_isolated_missing_price_field_degrades_only_the_affected_feature() -> None:
    """A single missing high/low mid-series (a partial vendor gap) must NaN
    only the ATR windows that overlap it -- not the whole series, and not
    features that never touch high/low at all.
    """
    n = 20
    nifty = _nifty_series(n)
    broken_day = nifty[10]
    nifty[10] = _obs(
        "NIFTY50", broken_day.session_date, broken_day.close, high=None, low=None, volume=1_000_000
    )
    vix = _vix_series(n)

    inputs = MarketFeatureInputs.from_index_observations(nifty, vix)
    matrix = _pipeline().compute(inputs)

    atr = matrix["nifty_atr_normalized_14d"]
    window = TINY_CONFIG.atr_window
    warmup = TINY_CONFIG.atr_window  # required_lookback - 1 leading NaNs, unrelated to the fault
    assert atr.iloc[10 : 10 + window].isna().all()
    assert atr.iloc[warmup:10].notna().all()  # past warm-up, unaffected by a future fault
    assert atr.iloc[10 + window :].notna().all()  # recovers once the bad day rolls out

    # A feature that never reads high/low must be completely untouched.
    assert matrix["nifty_return_1d"].notna().sum() == n - 1


def test_pipeline_does_not_crash_on_minimal_two_row_input() -> None:
    inputs = MarketFeatureInputs.from_index_observations(_nifty_series(2), _vix_series(2))
    matrix = _pipeline().compute(inputs)
    assert len(matrix) == 2
    assert matrix["nifty_return_1d"].notna().sum() == 1  # 1 warmup row, 1 real value


# --------------------------------------------------------------------------
# 4. NaN warm-up periods are handled correctly
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("feature_name", "expected_lookback"),
    [
        ("nifty_return_1d", 2),
        ("nifty_realized_vol_20d", TINY_CONFIG.realized_vol_window + 1),
        ("nifty_vol_ratio_5_20", TINY_CONFIG.vol_ratio_long_window + 1),
        ("india_vix_level_z", TINY_CONFIG.vix_zscore_window),
        ("india_vix_change_5d", TINY_CONFIG.vix_change_window + 1),
        ("nifty_trend_200d", TINY_CONFIG.trend_window),
        ("nifty_drawdown_from_high_252d", TINY_CONFIG.drawdown_window),
        ("nifty_atr_normalized_14d", TINY_CONFIG.atr_window + 1),
        ("nifty_volume_stress_20d", TINY_CONFIG.volume_stress_window),
    ],
)
def test_warmup_boundary_is_exact(feature_name: str, expected_lookback: int) -> None:
    n = 30
    inputs = MarketFeatureInputs.from_index_observations(_nifty_series(n), _vix_series(n))
    matrix = _pipeline().compute(inputs)
    series = matrix[feature_name]

    assert series.iloc[: expected_lookback - 1].isna().all(), (
        f"{feature_name}: expected NaN warm-up for the first {expected_lookback - 1} rows"
    )
    assert series.iloc[expected_lookback - 1 :].notna().all(), (
        f"{feature_name}: expected non-NaN values from row {expected_lookback - 1} onward"
    )


def test_declared_lookback_matches_the_warmup_boundary_for_every_feature() -> None:
    """A structural version of the parametrized test above: for every
    definition, the declared required_lookback is exactly where NaN ends.
    """
    n = 30
    inputs = MarketFeatureInputs.from_index_observations(_nifty_series(n), _vix_series(n))
    pipeline = _pipeline()
    matrix = pipeline.compute(inputs)

    for definition in pipeline.definitions:
        series = matrix[definition.name]
        first_valid = series.first_valid_index()
        assert first_valid is not None
        position = matrix.index.get_loc(first_valid)
        assert position == definition.required_lookback - 1, definition.name


def test_snapshot_is_warmup_matches_the_declared_lookback() -> None:
    n = 20
    inputs = MarketFeatureInputs.from_index_observations(_nifty_series(n), _vix_series(n))
    snapshots = _pipeline().audit(inputs)

    by_feature: dict[str, list[FeatureSnapshot]] = {}
    for snapshot in snapshots:
        by_feature.setdefault(snapshot.name, []).append(snapshot)

    for name, snaps in by_feature.items():
        ordered = sorted(snaps, key=lambda s: s.timestamp)
        lookback = ordered[0].lookback
        for position, snapshot in enumerate(ordered):
            assert snapshot.is_warmup == (position < lookback - 1), (name, position)


# --------------------------------------------------------------------------
# 5. Feature audit output
# --------------------------------------------------------------------------


def test_audit_produces_one_snapshot_per_feature_per_date() -> None:
    n = 10
    inputs = MarketFeatureInputs.from_index_observations(_nifty_series(n), _vix_series(n))
    pipeline = _pipeline()
    snapshots = pipeline.audit(inputs)
    assert len(snapshots) == len(pipeline.definitions) * n


def test_audit_snapshot_fields_are_correct() -> None:
    n = 10
    inputs = MarketFeatureInputs.from_index_observations(_nifty_series(n), _vix_series(n))
    snapshots = _pipeline().audit(inputs)

    return_snapshots = sorted(
        (s for s in snapshots if s.name == "nifty_return_1d"), key=lambda s: s.timestamp
    )

    warmup = return_snapshots[0]
    assert warmup.timestamp == inputs.dates[0]
    assert warmup.lookback == 2
    assert warmup.source_observations == inputs.dates[0:1]
    assert math.isnan(warmup.value)
    assert warmup.is_warmup

    third = return_snapshots[2]
    assert third.timestamp == inputs.dates[2]
    assert third.source_observations == inputs.dates[1:3]
    assert not third.is_warmup
    assert not math.isnan(third.value)


def test_snapshots_to_frame_has_the_documented_columns() -> None:
    inputs = MarketFeatureInputs.from_index_observations(_nifty_series(10), _vix_series(10))
    snapshots = _pipeline().audit(inputs)
    frame = snapshots_to_frame(snapshots)

    assert set(frame.columns) == {
        "name",
        "timestamp",
        "value",
        "lookback",
        "source_observations",
        "is_warmup",
    }
    assert len(frame) == len(snapshots)


# --------------------------------------------------------------------------
# Volume optionality ("where reliable")
# --------------------------------------------------------------------------


def test_volume_stress_is_omitted_when_no_volume_data_is_supplied() -> None:
    nifty_no_volume = [
        _obs("NIFTY50", o.session_date, o.close, high=o.high, low=o.low, volume=None)
        for o in _nifty_series(15)
    ]
    inputs = MarketFeatureInputs.from_index_observations(nifty_no_volume, _vix_series(15))
    assert not inputs.has_volume

    pipeline = _pipeline()
    matrix = pipeline.compute(inputs)
    assert "nifty_volume_stress_20d" not in matrix.columns
    assert len(matrix.columns) == len(pipeline.definitions) - 1


def test_volume_stress_is_included_when_volume_data_is_supplied() -> None:
    inputs = MarketFeatureInputs.from_index_observations(_nifty_series(15), _vix_series(15))
    assert inputs.has_volume
    matrix = _pipeline().compute(inputs)
    assert "nifty_volume_stress_20d" in matrix.columns


# --------------------------------------------------------------------------
# Determinism, documentation, and config wiring
# --------------------------------------------------------------------------


def test_pipeline_output_is_deterministic() -> None:
    inputs = MarketFeatureInputs.from_index_observations(_nifty_series(15), _vix_series(15))
    pipeline = _pipeline()
    pd.testing.assert_frame_equal(pipeline.compute(inputs), pipeline.compute(inputs))


def test_lookback_reflects_the_configured_window_not_a_hardcoded_constant() -> None:
    small = {d.name: d for d in build_default_feature_definitions(TINY_CONFIG)}
    large = {d.name: d for d in build_default_feature_definitions(DEFAULT_FEATURE_CONFIG)}
    assert small["nifty_trend_200d"].required_lookback == TINY_CONFIG.trend_window
    assert large["nifty_trend_200d"].required_lookback == DEFAULT_FEATURE_CONFIG.trend_window
    assert (
        small["nifty_trend_200d"].required_lookback
        != large["nifty_trend_200d"].required_lookback
    )


def test_every_default_feature_documents_itself() -> None:
    for definition in build_default_feature_definitions(TINY_CONFIG):
        assert definition.economic_interpretation.strip(), definition.name
        assert definition.calculation.strip(), definition.name
        assert definition.required_lookback >= 1, definition.name
        assert definition.known_at_decision_timestamp is True, definition.name


def test_feature_set_is_deliberately_small() -> None:
    """A budget, not a target: this asserts the set stays small, so adding a
    10th feature requires deliberately raising this number, not silently
    drifting past it.
    """
    assert len(build_default_feature_definitions(TINY_CONFIG)) == 9
