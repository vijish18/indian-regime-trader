"""Regime-aware portfolio allocation: volatility-tier mapping, confidence
handling, uncertainty mode, regime confirmation/transitions, allocation
bounds, no-leverage, and the non-HMM rolling-volatility baseline.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import pandas as pd
import pytest

from config.models import AllocationConfig, ExposureBand, HMMConfig, RegimePolicyConfig
from core.regime.allocation import (
    AllocationRegime,
    AllocationTarget,
    RegimeAllocationEngine,
    confirmed_tier_sequence,
    count_transitions,
    volatility_tier,
)
from core.regime.baseline_policy import MovingAverageTrendBaseline, RollingVolatilityBaseline
from core.regime.hmm_engine import RegimeLabel, RegimeState
from core.regime.regime_policy import RegimePolicy


def allocation_config(**overrides: object) -> AllocationConfig:
    defaults: dict[str, object] = {
        "low_risk_volatility_threshold": 0.12,
        "high_risk_volatility_threshold": 0.25,
        "extreme_confidence_threshold": 0.90,
        "max_flicker_transitions": 3,
        "baseline_volatility_window": 20,
        "trend_ma_window_days": 200,
    }
    defaults.update(overrides)
    return AllocationConfig.model_validate(defaults)


def band(low: float, high: float) -> ExposureBand:
    return ExposureBand(min_gross_exposure=low, max_gross_exposure=high)


def regime_policy_config(**overrides: ExposureBand) -> RegimePolicyConfig:
    defaults = {
        "low_risk": band(0.75, 1.00),
        "normal_risk": band(0.45, 0.75),
        "high_risk": band(0.15, 0.45),
        "uncertain": band(0.00, 0.15),
    }
    defaults.update(overrides)
    return RegimePolicyConfig.model_validate(defaults)


def hmm_config(**overrides: object) -> HMMConfig:
    defaults: dict[str, object] = {
        "candidate_states": [3],
        "covariance_type": "diag",
        "training_window_days": 252,
        "retrain_interval_sessions": 63,
        "min_confidence": 0.60,
        "confirmation_bars": 2,
        "flicker_window_sessions": 10,
        "max_covariance_condition_number": 1_000_000.0,
        "random_seeds": [17],
        "max_iterations": 200,
        "convergence_tolerance": 0.0001,
        "covariance_regularization": 0.000001,
        "min_state_occupancy": 0.02,
    }
    defaults.update(overrides)
    return HMMConfig.model_validate(defaults)


def state(
    day: dt.date,
    volatility: float,
    confidence: float,
    label: RegimeLabel = RegimeLabel.NORMAL,
    *,
    expected_return: float = 0.0,
    persistence: float = 0.9,
) -> RegimeState:
    return RegimeState(
        as_of=day,
        state_id=0,
        label=label,
        probabilities=(confidence, 1.0 - confidence),
        confidence=confidence,
        expected_volatility=volatility,
        expected_return=expected_return,
        persistence=persistence,
    )


def history(
    volatility: float, confidence: float, n: int, *, start: dt.date = dt.date(2024, 1, 2)
) -> list[RegimeState]:
    """n consecutive states with the same volatility and confidence -- enough
    to be fully confirmed under any reasonable confirmation_bars.
    """
    return [state(start + dt.timedelta(days=i), volatility, confidence) for i in range(n)]


def engine(
    hmm: HMMConfig | None = None,
    allocation: AllocationConfig | None = None,
    policy_config: RegimePolicyConfig | None = None,
) -> RegimeAllocationEngine:
    return RegimeAllocationEngine(
        hmm or hmm_config(),
        allocation or allocation_config(),
        RegimePolicy(policy_config or regime_policy_config()),
    )


# --------------------------------------------------------------------------
# Regime mapping (volatility -> tier)
# --------------------------------------------------------------------------


def test_low_volatility_maps_to_low_risk() -> None:
    assert volatility_tier(0.05, allocation_config()) is AllocationRegime.LOW_RISK


def test_mid_volatility_maps_to_normal_risk() -> None:
    assert volatility_tier(0.18, allocation_config()) is AllocationRegime.NORMAL_RISK


def test_high_volatility_maps_to_high_risk() -> None:
    assert volatility_tier(0.40, allocation_config()) is AllocationRegime.HIGH_RISK


def test_low_risk_threshold_is_exclusive_upper_bound() -> None:
    config = allocation_config()
    assert volatility_tier(0.11999, config) is AllocationRegime.LOW_RISK
    assert volatility_tier(0.12, config) is AllocationRegime.NORMAL_RISK  # boundary itself


def test_high_risk_threshold_is_inclusive_lower_bound() -> None:
    config = allocation_config()
    assert volatility_tier(0.24999, config) is AllocationRegime.NORMAL_RISK
    assert volatility_tier(0.25, config) is AllocationRegime.HIGH_RISK  # boundary itself


def test_volatility_tier_never_returns_uncertain() -> None:
    """UNCERTAIN is about trust in the classification, not a volatility
    level -- a bare number can never produce it.
    """
    config = allocation_config()
    for volatility in (0.0, 0.01, 0.12, 0.18, 0.25, 0.5, 2.0):
        assert volatility_tier(volatility, config) is not AllocationRegime.UNCERTAIN


def test_volatility_tier_rejects_negative_volatility() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        volatility_tier(-0.01, allocation_config())


def test_end_to_end_mapping_through_the_engine() -> None:
    """The example conceptual structure from the brief, checked directly:
    LOW_RISK -> highest exposure, ..., UNCERTAIN -> the most conservative.
    """
    hmm = hmm_config(confirmation_bars=1)
    e = engine(hmm=hmm)

    low = e.evaluate(history(0.05, 0.95, 5))
    normal = e.evaluate(history(0.18, 0.95, 5))
    high = e.evaluate(history(0.40, 0.95, 5))

    assert low.regime is AllocationRegime.LOW_RISK
    assert normal.regime is AllocationRegime.NORMAL_RISK
    assert high.regime is AllocationRegime.HIGH_RISK
    assert low.target_gross_exposure > normal.target_gross_exposure > high.target_gross_exposure


# --------------------------------------------------------------------------
# Confidence handling
# --------------------------------------------------------------------------


def test_confidence_below_minimum_forces_uncertain() -> None:
    e = engine()
    target = e.evaluate(history(0.05, 0.40, 5))  # calm but not trusted
    assert target.regime is AllocationRegime.UNCERTAIN
    assert "confidence" in target.reason


def test_confidence_at_exactly_the_minimum_is_accepted() -> None:
    hmm = hmm_config(min_confidence=0.60, confirmation_bars=1)
    e = engine(hmm=hmm)
    target = e.evaluate(history(0.05, 0.60, 5))
    assert target.regime is AllocationRegime.LOW_RISK


def test_higher_confidence_yields_higher_exposure_within_the_same_tier() -> None:
    """Confidence must scale the target continuously within the band, not
    just gate it on or off.
    """
    hmm = hmm_config(confirmation_bars=1, min_confidence=0.60)
    e = engine(hmm=hmm)

    low_conf = e.evaluate(history(0.05, 0.62, 5))
    mid_conf = e.evaluate(history(0.05, 0.80, 5))
    high_conf = e.evaluate(history(0.05, 1.00, 5))

    assert low_conf.regime is mid_conf.regime is high_conf.regime is AllocationRegime.LOW_RISK
    assert (
        low_conf.target_gross_exposure
        < mid_conf.target_gross_exposure
        < high_conf.target_gross_exposure
    )


def test_confidence_scaling_spans_the_full_band() -> None:
    hmm = hmm_config(confirmation_bars=1, min_confidence=0.50)
    e = engine(hmm=hmm)
    policy_bands = regime_policy_config()

    at_minimum = e.evaluate(history(0.05, 0.50, 5))
    at_maximum = e.evaluate(history(0.05, 1.00, 5))

    assert at_minimum.target_gross_exposure == pytest.approx(
        policy_bands.low_risk.min_gross_exposure
    )
    assert at_maximum.target_gross_exposure == pytest.approx(
        policy_bands.low_risk.max_gross_exposure
    )


def test_extreme_confidence_confirms_a_transition_in_a_single_bar() -> None:
    """docs/SPECIFICATION.md section 6: '2 consecutive observations unless
    confidence is extreme'.
    """
    hmm = hmm_config(confirmation_bars=5)  # would otherwise take 5 bars
    config = allocation_config(extreme_confidence_threshold=0.95)
    e = engine(hmm=hmm, allocation=config)

    calm = history(0.05, 0.99, 10)
    one_extreme_spike = calm + [state(calm[-1].as_of + dt.timedelta(days=1), 0.40, 0.97)]

    target = e.evaluate(one_extreme_spike)
    assert target.regime is AllocationRegime.HIGH_RISK
    assert target.reason == "confirmed"


def test_regime_label_never_affects_the_allocation() -> None:
    """The requirement stated plainly: regime names must not determine
    behavior. Two histories differing only in RegimeLabel must produce
    identical AllocationTargets.
    """
    e = engine(hmm=hmm_config(confirmation_bars=1))
    base = history(0.18, 0.85, 5)
    relabelled = [
        state(s.as_of, s.expected_volatility, s.confidence, label=RegimeLabel.CRISIS)
        for s in base
    ]

    assert e.evaluate(base) == e.evaluate(relabelled)


# --------------------------------------------------------------------------
# Uncertainty mode
# --------------------------------------------------------------------------


def test_uncertain_disallows_new_positions() -> None:
    e = engine()
    target = e.evaluate(history(0.05, 0.10, 5))  # very low confidence
    assert target.regime is AllocationRegime.UNCERTAIN
    assert target.allow_new_positions is False


def test_non_uncertain_tiers_allow_new_positions() -> None:
    e = engine(hmm=hmm_config(confirmation_bars=1))
    for volatility in (0.05, 0.18, 0.40):
        target = e.evaluate(history(volatility, 0.95, 5))
        assert target.regime is not AllocationRegime.UNCERTAIN
        assert target.allow_new_positions is True


def test_uncertain_is_the_most_conservative_configured_band() -> None:
    config = regime_policy_config()
    assert config.uncertain.max_gross_exposure <= config.high_risk.min_gross_exposure


def test_insufficient_history_to_confirm_anything_is_uncertain() -> None:
    """A single observation cannot satisfy confirmation_bars=2 -- and with
    nothing ever confirmed, the engine must not guess. Confidence is kept
    below the extreme-confidence threshold so the single-bar override does
    not mask the case being tested.
    """
    e = engine(hmm=hmm_config(confirmation_bars=2, min_confidence=0.0))
    target = e.evaluate([state(dt.date(2024, 1, 2), 0.05, 0.70)])
    assert target.regime is AllocationRegime.UNCERTAIN
    assert "not enough history" in target.reason


def test_empty_history_raises() -> None:
    with pytest.raises(ValueError, match="empty regime history"):
        engine().evaluate([])


# --------------------------------------------------------------------------
# Regime transitions and confirmation
# --------------------------------------------------------------------------


def test_a_single_dissenting_observation_does_not_flip_the_regime() -> None:
    """confirmation_bars=2: one high-vol day among a calm run must not yet
    move exposure -- the previous confirmed tier is held.
    """
    hmm = hmm_config(confirmation_bars=2, min_confidence=0.0)
    e = engine(hmm=hmm)

    calm = history(0.05, 0.80, 10)
    one_spike = calm + [state(calm[-1].as_of + dt.timedelta(days=1), 0.40, 0.80)]

    target = e.evaluate(one_spike)
    assert target.regime is AllocationRegime.LOW_RISK
    assert target.reason == "confirmed"


def test_two_consecutive_observations_confirm_the_new_regime() -> None:
    hmm = hmm_config(confirmation_bars=2, min_confidence=0.0)
    e = engine(hmm=hmm)

    calm = history(0.05, 0.80, 10)
    two_spikes = calm + [
        state(calm[-1].as_of + dt.timedelta(days=1), 0.40, 0.80),
        state(calm[-1].as_of + dt.timedelta(days=2), 0.40, 0.80),
    ]

    target = e.evaluate(two_spikes)
    assert target.regime is AllocationRegime.HIGH_RISK


def test_confirmed_tier_sequence_holds_the_previous_tier_during_a_pending_transition() -> None:
    tiers = [
        AllocationRegime.LOW_RISK,
        AllocationRegime.LOW_RISK,
        AllocationRegime.HIGH_RISK,  # 1st dissent
        AllocationRegime.LOW_RISK,  # reverts before confirming
        AllocationRegime.HIGH_RISK,  # 1st again
        AllocationRegime.HIGH_RISK,  # 2nd consecutive -> confirms
    ]
    confidences = [0.8] * len(tiers)

    confirmed = confirmed_tier_sequence(
        tiers, confidences, confirmation_bars=2, extreme_confidence_threshold=0.99
    )

    assert confirmed == [
        None,  # nothing confirmed yet (needs 2 consecutive)
        AllocationRegime.LOW_RISK,  # 2 consecutive LOW_RISK confirms
        AllocationRegime.LOW_RISK,  # single HIGH_RISK dissent: still LOW_RISK
        AllocationRegime.LOW_RISK,  # reverted, still LOW_RISK
        AllocationRegime.LOW_RISK,  # 1st HIGH_RISK again: not yet confirmed
        AllocationRegime.HIGH_RISK,  # 2nd consecutive HIGH_RISK: now confirmed
    ]


def test_confirmed_tier_sequence_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="raw_tiers has"):
        confirmed_tier_sequence([AllocationRegime.LOW_RISK], [0.5, 0.6], 2, 0.9)


def test_confirmed_tier_sequence_rejects_confirmation_bars_below_one() -> None:
    with pytest.raises(ValueError, match="confirmation_bars must be >= 1"):
        confirmed_tier_sequence([AllocationRegime.LOW_RISK], [0.9], 0, 0.9)


def test_flickering_regime_is_forced_uncertain() -> None:
    """A regime that keeps flipping is not one anyone should size a
    portfolio against, even if the latest single reading looks confirmed.
    """
    hmm = hmm_config(confirmation_bars=1, flicker_window_sessions=8, min_confidence=0.0)
    config = allocation_config(max_flicker_transitions=2)
    e = engine(hmm=hmm, allocation=config)

    start = dt.date(2024, 1, 2)
    volatilities = [0.05, 0.40, 0.05, 0.40, 0.05, 0.40, 0.05, 0.40]  # flips every day
    flickering = [
        state(start + dt.timedelta(days=i), v, 0.85) for i, v in enumerate(volatilities)
    ]

    target = e.evaluate(flickering)
    assert target.regime is AllocationRegime.UNCERTAIN
    assert "flicker" in target.reason.lower()


def test_a_stable_regime_does_not_trigger_the_flicker_gate() -> None:
    hmm = hmm_config(confirmation_bars=1, flicker_window_sessions=8, min_confidence=0.0)
    config = allocation_config(max_flicker_transitions=2)
    e = engine(hmm=hmm, allocation=config)

    target = e.evaluate(history(0.05, 0.85, 15))
    assert target.regime is AllocationRegime.LOW_RISK
    assert target.reason == "confirmed"


def test_flicker_window_only_looks_at_the_trailing_window() -> None:
    """Old instability outside the flicker window must not haunt a regime
    that has since settled down.
    """
    hmm = hmm_config(confirmation_bars=1, flicker_window_sessions=5, min_confidence=0.0)
    config = allocation_config(max_flicker_transitions=1)
    e = engine(hmm=hmm, allocation=config)

    start = dt.date(2024, 1, 2)
    noisy_past = [
        state(start + dt.timedelta(days=i), v, 0.85)
        for i, v in enumerate([0.05, 0.40, 0.05, 0.40, 0.05, 0.40, 0.05, 0.40])
    ]
    settled = noisy_past + [
        state(start + dt.timedelta(days=8 + i), 0.05, 0.85) for i in range(6)
    ]

    target = e.evaluate(settled)
    assert target.regime is AllocationRegime.LOW_RISK


def test_count_transitions_ignores_unconfirmed_none_entries() -> None:
    tiers = [
        None,
        None,
        AllocationRegime.LOW_RISK,
        AllocationRegime.LOW_RISK,
        AllocationRegime.HIGH_RISK,
    ]
    assert count_transitions(tiers) == 1


def test_count_transitions_on_empty_and_singleton_lists() -> None:
    assert count_transitions([]) == 0
    assert count_transitions([AllocationRegime.LOW_RISK]) == 0
    assert count_transitions([None]) == 0


# --------------------------------------------------------------------------
# Allocation bounds and no leverage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("volatility", [0.0, 0.05, 0.11, 0.12, 0.18, 0.25, 0.30, 1.0, 5.0])
@pytest.mark.parametrize("confidence", [0.0, 0.3, 0.6, 0.75, 0.9, 1.0])
def test_target_is_always_within_its_own_band_and_never_exceeds_one(
    volatility: float, confidence: float
) -> None:
    e = engine(hmm=hmm_config(confirmation_bars=1, min_confidence=0.0))
    target = e.evaluate(history(volatility, confidence, 3))

    assert target.min_gross_exposure <= target.target_gross_exposure <= target.max_gross_exposure
    assert 0.0 <= target.min_gross_exposure
    assert target.max_gross_exposure <= 1.0  # no leverage, ever


def test_no_configured_band_permits_leverage() -> None:
    """Defense in depth at the config layer: every band's max is a Percent
    field bounded to [0, 1] (config/models.py::ExposureBand).
    """
    with pytest.raises(Exception, match="max_gross_exposure"):
        ExposureBand(min_gross_exposure=0.0, max_gross_exposure=1.5)


def test_allocation_target_rejects_a_band_that_implies_leverage() -> None:
    with pytest.raises(ValueError, match="no leverage"):
        AllocationTarget(
            as_of=dt.date(2024, 1, 2),
            regime=AllocationRegime.LOW_RISK,
            target_gross_exposure=1.0,
            min_gross_exposure=0.5,
            max_gross_exposure=1.2,
            allow_new_positions=True,
            confidence=0.9,
            expected_volatility=0.1,
            reason="test",
        )


def test_allocation_target_rejects_a_target_outside_its_own_band() -> None:
    with pytest.raises(ValueError, match="outside the band"):
        AllocationTarget(
            as_of=dt.date(2024, 1, 2),
            regime=AllocationRegime.LOW_RISK,
            target_gross_exposure=0.99,
            min_gross_exposure=0.5,
            max_gross_exposure=0.9,
            allow_new_positions=True,
            confidence=0.9,
            expected_volatility=0.1,
            reason="test",
        )


def test_allocation_target_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        AllocationTarget(
            as_of=dt.date(2024, 1, 2),
            regime=AllocationRegime.LOW_RISK,
            target_gross_exposure=0.5,
            min_gross_exposure=0.0,
            max_gross_exposure=1.0,
            allow_new_positions=True,
            confidence=1.5,
            expected_volatility=0.1,
            reason="test",
        )


def test_allocation_target_rejects_negative_volatility() -> None:
    with pytest.raises(ValueError, match="expected_volatility"):
        AllocationTarget(
            as_of=dt.date(2024, 1, 2),
            regime=AllocationRegime.LOW_RISK,
            target_gross_exposure=0.5,
            min_gross_exposure=0.0,
            max_gross_exposure=1.0,
            allow_new_positions=True,
            confidence=0.9,
            expected_volatility=-0.1,
            reason="test",
        )


def test_high_risk_band_is_strictly_below_low_risk_band() -> None:
    """The example conceptual structure, checked at the config level:
    reduced exposure for HIGH_RISK relative to LOW_RISK."""
    config = regime_policy_config()
    assert config.high_risk.max_gross_exposure <= config.low_risk.min_gross_exposure


# --------------------------------------------------------------------------
# RegimePolicy
# --------------------------------------------------------------------------


def test_regime_policy_looks_up_every_tier() -> None:
    config = regime_policy_config()
    policy = RegimePolicy(config)

    assert policy.band_for(AllocationRegime.LOW_RISK) == config.low_risk
    assert policy.band_for(AllocationRegime.NORMAL_RISK) == config.normal_risk
    assert policy.band_for(AllocationRegime.HIGH_RISK) == config.high_risk
    assert policy.band_for(AllocationRegime.UNCERTAIN) == config.uncertain


# --------------------------------------------------------------------------
# Baseline: simple rolling realized volatility, no HMM
# --------------------------------------------------------------------------


def _returns(values: list[float], start: dt.date = dt.date(2024, 1, 2)) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_baseline_classifies_a_calm_series_as_low_risk() -> None:
    policy = RegimePolicy(regime_policy_config())
    baseline = RollingVolatilityBaseline(allocation_config(), policy)

    quiet = _returns([0.0005] * 25)
    target = baseline.evaluate(quiet)

    assert target.regime is AllocationRegime.LOW_RISK
    assert target.allow_new_positions is True


def test_baseline_classifies_a_turbulent_series_as_high_risk() -> None:
    import numpy as np

    policy = RegimePolicy(regime_policy_config())
    baseline = RollingVolatilityBaseline(allocation_config(), policy)

    rng = np.random.default_rng(3)
    turbulent = _returns(list(rng.normal(0.0, 0.04, 25)))
    target = baseline.evaluate(turbulent)

    assert target.regime is AllocationRegime.HIGH_RISK


def test_baseline_target_is_the_band_midpoint() -> None:
    """No model, no confidence to scale by -- the only honest point estimate
    is the middle of the tier's configured band.
    """
    policy = RegimePolicy(regime_policy_config())
    baseline = RollingVolatilityBaseline(allocation_config(), policy)

    target = baseline.evaluate(_returns([0.0] * 25))
    band_config = regime_policy_config().low_risk
    expected_midpoint = (band_config.min_gross_exposure + band_config.max_gross_exposure) / 2
    assert target.target_gross_exposure == pytest.approx(expected_midpoint)


def test_baseline_reports_full_confidence_and_always_allows_new_positions() -> None:
    """A documented limitation, not an oversight: the baseline has no
    concept of "I don't know" the way the HMM's UNCERTAIN tier does.
    """
    policy = RegimePolicy(regime_policy_config())
    baseline = RollingVolatilityBaseline(allocation_config(), policy)

    for values in ([0.0001] * 25, [0.05] * 25, [-0.05] * 25):
        target = baseline.evaluate(_returns(values))
        assert target.confidence == 1.0
        assert target.allow_new_positions is True
        assert target.regime is not AllocationRegime.UNCERTAIN


def test_baseline_uses_the_same_annualization_as_the_feature_engine() -> None:
    """The HMM and the baseline must measure volatility the same way, or a
    difference in outcome could just be a different volatility estimator
    rather than the classification logic actually being better.
    """
    import math

    policy = RegimePolicy(regime_policy_config())
    baseline = RollingVolatilityBaseline(allocation_config(baseline_volatility_window=20), policy)

    values = [0.01, -0.02, 0.015, -0.005, 0.02] * 5
    computed = baseline.realized_volatility(_returns(values))

    series = pd.Series(values)
    expected = float(series.std(ddof=0) * math.sqrt(252))
    assert computed == pytest.approx(expected)


def test_baseline_rejects_a_window_shorter_than_configured() -> None:
    policy = RegimePolicy(regime_policy_config())
    baseline = RollingVolatilityBaseline(allocation_config(baseline_volatility_window=20), policy)

    with pytest.raises(ValueError, match="need at least 20"):
        baseline.realized_volatility(_returns([0.001] * 10))


def test_baseline_rejects_an_empty_return_series() -> None:
    policy = RegimePolicy(regime_policy_config())
    baseline = RollingVolatilityBaseline(allocation_config(), policy)
    with pytest.raises(ValueError, match="empty return series"):
        baseline.evaluate(pd.Series(dtype=float))


def test_baseline_rejects_nan_in_the_window() -> None:
    policy = RegimePolicy(regime_policy_config())
    baseline = RollingVolatilityBaseline(allocation_config(), policy)

    values = [0.001] * 25
    series = _returns(values)
    series.iloc[10] = float("nan")

    with pytest.raises(ValueError, match="NaN"):
        baseline.realized_volatility(series)


def test_baseline_is_structurally_interchangeable_with_the_hmm_engine() -> None:
    """The entire point: a later walk-forward comparison must be able to
    swap one for the other with no other code change. Confirmed here by
    checking both produce the AllocationTarget shape with the same field set.
    """
    policy = RegimePolicy(regime_policy_config())
    baseline_target = RollingVolatilityBaseline(allocation_config(), policy).evaluate(
        _returns([0.001] * 25)
    )
    hmm_target = engine(hmm=hmm_config(confirmation_bars=1)).evaluate(history(0.05, 0.9, 5))

    assert type(baseline_target) is type(hmm_target) is AllocationTarget
    field_names = {f.name for f in dataclasses.fields(AllocationTarget)}
    assert field_names  # sanity: the dataclass actually has fields
    for name in field_names:
        getattr(baseline_target, name)  # every field is present on both
        getattr(hmm_target, name)


# --------------------------------------------------------------------------
# Baseline: simple moving-average trend filter, no HMM
# --------------------------------------------------------------------------


def _prices(values: list[float], start: dt.date = dt.date(2024, 1, 2)) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_trend_baseline_classifies_price_above_average_as_low_risk() -> None:
    policy = RegimePolicy(regime_policy_config())
    baseline = MovingAverageTrendBaseline(allocation_config(trend_ma_window_days=10), policy)

    rising = _prices([100.0 + i for i in range(15)])
    target = baseline.evaluate(rising)

    assert target.regime is AllocationRegime.LOW_RISK
    assert target.allow_new_positions is True


def test_trend_baseline_classifies_price_at_or_below_average_as_high_risk() -> None:
    policy = RegimePolicy(regime_policy_config())
    baseline = MovingAverageTrendBaseline(allocation_config(trend_ma_window_days=10), policy)

    falling = _prices([100.0 - i for i in range(15)])
    target = baseline.evaluate(falling)

    assert target.regime is AllocationRegime.HIGH_RISK


def test_trend_baseline_target_is_the_band_midpoint() -> None:
    policy = RegimePolicy(regime_policy_config())
    baseline = MovingAverageTrendBaseline(allocation_config(trend_ma_window_days=5), policy)

    target = baseline.evaluate(_prices([100.0] * 10))
    band_config = regime_policy_config().high_risk  # flat price == moving average -> not > MA
    expected_midpoint = (band_config.min_gross_exposure + band_config.max_gross_exposure) / 2
    assert target.target_gross_exposure == pytest.approx(expected_midpoint)


def test_trend_baseline_reports_full_confidence_and_always_allows_new_positions() -> None:
    policy = RegimePolicy(regime_policy_config())
    baseline = MovingAverageTrendBaseline(allocation_config(trend_ma_window_days=5), policy)

    for values in ([100.0 + i for i in range(10)], [100.0 - i for i in range(10)]):
        target = baseline.evaluate(_prices(values))
        assert target.confidence == 1.0
        assert target.allow_new_positions is True
        assert target.regime is not AllocationRegime.UNCERTAIN


def test_trend_baseline_rejects_a_window_shorter_than_configured() -> None:
    policy = RegimePolicy(regime_policy_config())
    baseline = MovingAverageTrendBaseline(allocation_config(trend_ma_window_days=20), policy)

    with pytest.raises(ValueError, match="need at least 20"):
        baseline.moving_average(_prices([100.0] * 10))


def test_trend_baseline_rejects_an_empty_price_series() -> None:
    policy = RegimePolicy(regime_policy_config())
    baseline = MovingAverageTrendBaseline(allocation_config(), policy)
    with pytest.raises(ValueError, match="empty price series"):
        baseline.evaluate(pd.Series(dtype=float))


def test_trend_baseline_rejects_nan_in_the_window() -> None:
    policy = RegimePolicy(regime_policy_config())
    baseline = MovingAverageTrendBaseline(allocation_config(trend_ma_window_days=10), policy)

    series = _prices([100.0] * 15)
    series.iloc[5] = float("nan")

    with pytest.raises(ValueError, match="NaN"):
        baseline.moving_average(series)


def test_trend_baseline_moving_average_is_a_plain_trailing_mean() -> None:
    policy = RegimePolicy(regime_policy_config())
    baseline = MovingAverageTrendBaseline(allocation_config(trend_ma_window_days=5), policy)

    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert baseline.moving_average(_prices(values)) == pytest.approx(30.0)


def test_trend_baseline_is_structurally_interchangeable_with_the_hmm_engine() -> None:
    policy = RegimePolicy(regime_policy_config())
    baseline_target = MovingAverageTrendBaseline(
        allocation_config(trend_ma_window_days=5), policy
    ).evaluate(_prices([100.0 + i for i in range(10)]))
    hmm_target = engine(hmm=hmm_config(confirmation_bars=1)).evaluate(history(0.05, 0.9, 5))

    assert type(baseline_target) is type(hmm_target) is AllocationTarget
    for name in {f.name for f in dataclasses.fields(AllocationTarget)}:
        getattr(baseline_target, name)
        getattr(hmm_target, name)
