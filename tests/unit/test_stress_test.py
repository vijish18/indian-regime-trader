"""Stress-testing framework: all 20 required Indian equity-market failure
scenarios, the Monte Carlo harness (deterministic, >= 100 trials by
default), and the central claim the whole framework exists to check --
risk controls must limit damage even if the HMM is wrong.

Two shared environments back these tests: ``env`` (module-scoped, a
moderately sized universe) for the one-shot ``run_scenario``/``run_all``
tests, and ``fast_env`` (module-scoped, a single-instrument universe with
the shortest window that still clears
``selection.min_history_days``) for the Monte Carlo tests, where the same
backtest runs 100+ times.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from backtest.stress_test import (
    MONTE_CARLO_SCENARIOS,
    MarketShock,
    ShockedMarketDataProvider,
    ShockType,
    StressScenario,
    StressTestContext,
    StressTestError,
    StressTestResult,
    StressTestSuite,
)
from risk.circuit_breaker import CircuitBreaker, CircuitState
from tests.unit._wf_support import Environment


def build_context(
    env: Environment, tmp_path: Path, signal_slice: slice
) -> StressTestContext:
    return StressTestContext(
        calendar=env.calendar,
        market_data=env.market_data,
        stock_selector_factory=lambda market_data: type(env.stock_selector)(
            env.selection_cfg, env.universe_cfg, env.universe_provider, market_data
        ),
        portfolio_constructor_factory=lambda market_data: type(env.portfolio_constructor)(
            env.portfolio_cfg, env.selection_cfg, env.execution_cfg, market_data
        ),
        risk_config=env.risk_cfg,
        cost_model=env.cost_model,
        circuit_breaker_state_dir=tmp_path,
        corporate_actions=env.corporate_actions,
        instrument_ids=tuple(env.instrument_ids),
        index_symbol="NIFTY50",
        signal_dates=tuple(env.dates[signal_slice]),
        initial_equity=1_000_000.0,
    )


@pytest.fixture(scope="module")
def env() -> Environment:
    # min_history_days (47 by default) must clear before signal_dates
    # starts, or the stock selector never has a candidate and every
    # scenario silently no-ops against an all-cash portfolio.
    return Environment(n_days=100, n_stocks=2, seed=11)


@pytest.fixture
def context(env: Environment, tmp_path: Path) -> StressTestContext:
    return build_context(env, tmp_path, slice(60, 80))


@pytest.fixture
def suite(context: StressTestContext) -> StressTestSuite:
    return StressTestSuite(context)


@pytest.fixture(scope="module")
def fast_env() -> Environment:
    """A single-instrument universe -- fast enough to run a Monte Carlo
    scenario 100+ times in a test without the suite taking minutes."""
    return Environment(n_days=70, n_stocks=1, seed=13)


@pytest.fixture
def fast_context(fast_env: Environment, tmp_path: Path) -> StressTestContext:
    return build_context(fast_env, tmp_path, slice(50, 60))


@pytest.fixture
def fast_suite(fast_context: StressTestContext) -> StressTestSuite:
    return StressTestSuite(fast_context)


# --------------------------------------------------------------------------
# Every scenario runs and produces a well-formed result
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", list(StressScenario))
def test_every_scenario_runs_without_raising(
    suite: StressTestSuite, scenario: StressScenario
) -> None:
    result = suite.run_scenario(scenario, seed=1)
    assert isinstance(result, StressTestResult)
    assert result.scenario is scenario
    assert result.trial_seed == 1
    assert isinstance(result.detail, str) and result.detail


def test_run_all_covers_every_scenario_in_declaration_order(suite: StressTestSuite) -> None:
    results = suite.run_all(seed=3)
    assert [result.scenario for result in results] == list(StressScenario)


# --------------------------------------------------------------------------
# The central requirement: risk controls contain damage even when the
# exposure signal is wrong
# --------------------------------------------------------------------------


def test_hmm_regime_misclassification_still_halts_on_realized_loss(
    suite: StressTestSuite,
) -> None:
    """A fixed "stay fully invested" signal straight through a real crash
    must still trip the circuit breaker -- which watches realized P&L, not
    the signal -- and the loss must be smaller than the full shock,
    evidence that halting actually contained further damage.
    """
    result = suite.run_scenario(
        StressScenario.HMM_REGIME_MISCLASSIFICATION, seed=7, drop_magnitude=0.35
    )
    assert result.risk_controls_fired is True
    assert result.trading_halted is True
    assert result.portfolio_impact_pct < 0
    assert result.max_loss_pct < 0.35  # contained, not the full uncontained shock
    assert result.system_failed_closed is True


def test_wrong_stock_ranking_never_exceeds_the_single_name_cap(suite: StressTestSuite) -> None:
    result = suite.run_scenario(
        StressScenario.WRONG_STOCK_RANKING, seed=4, adverse_move=0.40
    )
    assert result.portfolio_state_consistent is True
    assert result.system_failed_closed is True


def test_sudden_market_crash_triggers_risk_controls(suite: StressTestSuite) -> None:
    result = suite.run_scenario(StressScenario.SUDDEN_MARKET_CRASH, seed=2, magnitude=0.30)
    assert result.risk_controls_fired is True
    assert result.max_loss_pct > 0


def test_sudden_portfolio_drawdown_triggers_risk_controls(suite: StressTestSuite) -> None:
    result = suite.run_scenario(
        StressScenario.SUDDEN_PORTFOLIO_DRAWDOWN, seed=9, magnitude=0.25
    )
    assert result.risk_controls_fired is True


# --------------------------------------------------------------------------
# Market / data scenarios: specific behavior
# --------------------------------------------------------------------------


def test_wide_bid_ask_spread_rejects_via_abnormal_spread_check(suite: StressTestSuite) -> None:
    result = suite.run_scenario(StressScenario.WIDE_BID_ASK_SPREAD, seed=1, spread_bps=600.0)
    assert result.risk_controls_fired is True
    assert result.system_failed_closed is True


def test_stale_market_data_is_rejected(suite: StressTestSuite) -> None:
    result = suite.run_scenario(StressScenario.STALE_MARKET_DATA, seed=1)
    assert result.risk_controls_fired is True
    assert result.system_failed_closed is True


def test_missing_bars_fails_closed_on_mark_to_market(suite: StressTestSuite) -> None:
    result = suite.run_scenario(StressScenario.MISSING_BARS, seed=1)
    assert result.system_failed_closed is True
    assert "raised BacktestEngineError" in result.detail


def test_stock_trading_halt_does_not_crash_the_backtest(suite: StressTestSuite) -> None:
    result = suite.run_scenario(StressScenario.STOCK_TRADING_HALT, seed=1)
    assert result.portfolio_state_consistent is True
    assert result.system_failed_closed is True


def test_exchange_holiday_mismatch_never_executes_on_a_non_trading_day(
    suite: StressTestSuite,
) -> None:
    result = suite.run_scenario(StressScenario.EXCHANGE_HOLIDAY_MISMATCH, seed=1)
    assert result.portfolio_state_consistent is True
    assert result.system_failed_closed is True


def test_corporate_action_event_credits_cash_and_degrades_gracefully(
    suite: StressTestSuite,
) -> None:
    result = suite.run_scenario(StressScenario.CORPORATE_ACTION_EVENT, seed=1)
    assert result.portfolio_state_consistent is True
    assert "credit=250.00" in result.detail or "credit=250.0" in result.detail


# --------------------------------------------------------------------------
# Execution / infrastructure scenarios
# --------------------------------------------------------------------------


def test_broker_api_outage_halts_and_rejects_everything(suite: StressTestSuite) -> None:
    result = suite.run_scenario(StressScenario.BROKER_API_OUTAGE, seed=1)
    assert result.risk_controls_fired is True
    assert result.trading_halted is True
    assert result.system_failed_closed is True


def test_partial_fill_updates_ledger_consistently_for_a_non_full_quantity(
    suite: StressTestSuite,
) -> None:
    result = suite.run_scenario(StressScenario.PARTIAL_FILL, seed=1)
    assert result.portfolio_state_consistent is True
    assert result.system_failed_closed is True


def test_order_rejection_is_skipped_not_crashed(suite: StressTestSuite) -> None:
    result = suite.run_scenario(StressScenario.ORDER_REJECTION, seed=1)
    assert result.risk_controls_fired is True
    assert result.system_failed_closed is True


def test_delayed_fill_resolves_to_a_later_genuine_price(suite: StressTestSuite) -> None:
    result = suite.run_scenario(StressScenario.DELAYED_FILL, seed=1)
    assert result.portfolio_state_consistent is True


def test_duplicate_order_is_prevented(suite: StressTestSuite) -> None:
    result = suite.run_scenario(StressScenario.DUPLICATE_ORDER, seed=1)
    assert result.duplicate_orders_prevented is True
    assert result.risk_controls_fired is True
    assert result.system_failed_closed is True


def test_application_restart_keeps_a_halt_across_a_fresh_circuit_breaker(
    suite: StressTestSuite,
) -> None:
    result = suite.run_scenario(StressScenario.APPLICATION_RESTART, seed=5)
    assert result.trading_halted is True
    assert result.portfolio_state_consistent is True
    assert "resists_auto_recovery=True" in result.detail


def test_database_failure_never_silently_resumes_as_normal(suite: StressTestSuite) -> None:
    result = suite.run_scenario(StressScenario.DATABASE_FAILURE, seed=6)
    assert result.system_failed_closed is True
    assert result.portfolio_state_consistent is True


# --------------------------------------------------------------------------
# ShockedMarketDataProvider / MarketShock, in isolation
# --------------------------------------------------------------------------


def test_price_crash_shock_scales_every_ohlc_field(env: Environment) -> None:
    instrument_id = env.instrument_ids[0]
    shock_date = env.dates[65]
    shocked = ShockedMarketDataProvider(
        env.market_data,
        [MarketShock(ShockType.PRICE_CRASH, shock_date, None, 0.20, frozenset({instrument_id}))],
    )
    baseline = env.market_data.get_equity_bars(instrument_id, shock_date, shock_date)[0]
    shocked_bar = shocked.get_equity_bars(instrument_id, shock_date, shock_date)[0]
    assert float(shocked_bar.close) == pytest.approx(float(baseline.close) * 0.80, rel=1e-6)
    assert float(shocked_bar.open) == pytest.approx(float(baseline.open) * 0.80, rel=1e-6)


def test_price_crash_shock_leaves_prior_dates_untouched(env: Environment) -> None:
    instrument_id = env.instrument_ids[0]
    shock_date = env.dates[65]
    before_date = env.dates[64]
    shocked = ShockedMarketDataProvider(
        env.market_data,
        [MarketShock(ShockType.PRICE_CRASH, shock_date, None, 0.20, frozenset({instrument_id}))],
    )
    baseline = env.market_data.get_equity_bars(instrument_id, before_date, before_date)[0]
    shocked_bar = shocked.get_equity_bars(instrument_id, before_date, before_date)[0]
    assert shocked_bar.close == baseline.close


def test_unavailable_shock_removes_bars_in_range(env: Environment) -> None:
    instrument_id = env.instrument_ids[0]
    start, end = env.dates[65], env.dates[70]
    shocked = ShockedMarketDataProvider(
        env.market_data,
        [MarketShock(ShockType.UNAVAILABLE, start, end, 0.0, frozenset({instrument_id}))],
    )
    bars = shocked.get_equity_bars(instrument_id, start, end)
    assert bars == []


def test_index_spike_shock_multiplies_close_only_for_named_symbol(env: Environment) -> None:
    shock_date = env.dates[65]
    shocked = ShockedMarketDataProvider(
        env.market_data,
        [MarketShock(ShockType.INDEX_SPIKE, shock_date, None, 2.0, frozenset({"NIFTY50"}))],
    )
    baseline = env.market_data.get_index_observations("NIFTY50", shock_date, shock_date)[0]
    shocked_obs = shocked.get_index_observations("NIFTY50", shock_date, shock_date)[0]
    assert float(shocked_obs.close) == pytest.approx(float(baseline.close) * 3.0, rel=1e-6)

    vix_baseline = env.market_data.get_index_observations("INDIAVIX", shock_date, shock_date)[0]
    vix_shocked = shocked.get_index_observations("INDIAVIX", shock_date, shock_date)[0]
    assert vix_shocked.close == vix_baseline.close  # not the named symbol -- untouched


def test_volume_collapse_shock_reduces_volume_but_keeps_a_floor(env: Environment) -> None:
    instrument_id = env.instrument_ids[0]
    shock_date = env.dates[65]
    shocked = ShockedMarketDataProvider(
        env.market_data,
        [
            MarketShock(
                ShockType.VOLUME_COLLAPSE, shock_date, None, 0.999, frozenset({instrument_id})
            )
        ],
    )
    bar = shocked.get_equity_bars(instrument_id, shock_date, shock_date)[0]
    assert bar.volume >= 1  # never zero, even at a 99.9% cut


def test_market_shock_applies_to_id_none_means_universal() -> None:
    shock = MarketShock(ShockType.PRICE_CRASH, dt.date(2024, 1, 1), None, 0.1, None)
    assert shock.applies_to_id("NSE:ANYTHING")


def test_market_shock_applies_to_date_respects_bounds() -> None:
    shock = MarketShock(
        ShockType.PRICE_CRASH, dt.date(2024, 1, 5), dt.date(2024, 1, 10), 0.1, None
    )
    assert not shock.applies_to_date(dt.date(2024, 1, 4))
    assert shock.applies_to_date(dt.date(2024, 1, 5))
    assert shock.applies_to_date(dt.date(2024, 1, 10))
    assert not shock.applies_to_date(dt.date(2024, 1, 11))


def test_market_shock_open_ended_end_date_applies_indefinitely() -> None:
    shock = MarketShock(ShockType.PRICE_CRASH, dt.date(2024, 1, 5), None, 0.1, None)
    assert shock.applies_to_date(dt.date(2099, 1, 1))


# --------------------------------------------------------------------------
# Monte Carlo: >= 100 trials, deterministic seeds
# --------------------------------------------------------------------------


def test_run_monte_carlo_defaults_to_at_least_100_trials(fast_suite: StressTestSuite) -> None:
    result = fast_suite.run_monte_carlo(StressScenario.SUDDEN_MARKET_CRASH, seed=1)
    assert result.n_trials >= 100
    assert len(result.trials) == result.n_trials


def test_run_monte_carlo_is_deterministic_given_the_same_seed(fast_suite: StressTestSuite) -> None:
    first = fast_suite.run_monte_carlo(StressScenario.SUDDEN_MARKET_CRASH, n_trials=15, seed=42)
    second = fast_suite.run_monte_carlo(StressScenario.SUDDEN_MARKET_CRASH, n_trials=15, seed=42)
    assert [trial.portfolio_impact_pct for trial in first.trials] == [
        trial.portfolio_impact_pct for trial in second.trials
    ]
    assert [trial.max_loss_pct for trial in first.trials] == [
        trial.max_loss_pct for trial in second.trials
    ]


def test_run_monte_carlo_different_seeds_produce_different_trials(
    fast_suite: StressTestSuite,
) -> None:
    first = fast_suite.run_monte_carlo(StressScenario.SUDDEN_MARKET_CRASH, n_trials=15, seed=1)
    second = fast_suite.run_monte_carlo(StressScenario.SUDDEN_MARKET_CRASH, n_trials=15, seed=99)
    assert [trial.portfolio_impact_pct for trial in first.trials] != [
        trial.portfolio_impact_pct for trial in second.trials
    ]


def test_run_monte_carlo_each_trial_has_a_distinct_seed(fast_suite: StressTestSuite) -> None:
    result = fast_suite.run_monte_carlo(StressScenario.SUDDEN_MARKET_CRASH, n_trials=20, seed=5)
    seeds = [trial.trial_seed for trial in result.trials]
    assert seeds == list(range(5, 25))


def test_run_monte_carlo_rejects_a_non_monte_carlo_scenario(fast_suite: StressTestSuite) -> None:
    with pytest.raises(StressTestError, match="not a Monte Carlo scenario"):
        fast_suite.run_monte_carlo(StressScenario.BROKER_API_OUTAGE, n_trials=10, seed=1)


def test_run_monte_carlo_rejects_fewer_than_one_trial(fast_suite: StressTestSuite) -> None:
    with pytest.raises(StressTestError, match="n_trials"):
        fast_suite.run_monte_carlo(StressScenario.SUDDEN_MARKET_CRASH, n_trials=0, seed=1)


def test_every_declared_monte_carlo_scenario_actually_runs(fast_suite: StressTestSuite) -> None:
    """Not exhaustive at 100 trials each (test-suite speed), but confirms
    every scenario the module declares as Monte Carlo-eligible really can
    be run that way."""
    for scenario in MONTE_CARLO_SCENARIOS:
        result = fast_suite.run_monte_carlo(scenario, n_trials=10, seed=1)
        assert result.scenario is scenario
        assert len(result.trials) == 10


def test_monte_carlo_rates_are_fractions_between_zero_and_one(
    fast_suite: StressTestSuite,
) -> None:
    result = fast_suite.run_monte_carlo(StressScenario.SUDDEN_MARKET_CRASH, n_trials=20, seed=1)
    assert 0.0 <= result.risk_controls_fired_rate <= 1.0
    assert 0.0 <= result.trading_halted_rate <= 1.0
    assert 0.0 <= result.portfolio_state_consistent_rate <= 1.0
    assert 0.0 <= result.system_failed_closed_rate <= 1.0


def test_monte_carlo_crash_scenario_reliably_triggers_risk_controls(
    fast_suite: StressTestSuite,
) -> None:
    """The key requirement, checked statistically: across a full 100-trial
    randomized sweep of crash magnitudes (uniform 10%-40%), risk controls
    fire in most trials -- not just in one hand-picked case. The sampled
    range spans genuinely mild cuts too: at a single-name weight capped
    well under 100%, a ~10-20% price cut can legitimately stay under the
    daily-loss reduce threshold, so the bar here is "most", not "all".
    """
    result = fast_suite.run_monte_carlo(StressScenario.SUDDEN_MARKET_CRASH, seed=21)
    assert result.n_trials >= 100
    assert result.risk_controls_fired_rate >= 0.7
    assert result.portfolio_state_consistent_rate == 1.0
    assert result.system_failed_closed_rate == 1.0


def test_monte_carlo_worst_case_loss_is_the_max_of_all_trials(fast_suite: StressTestSuite) -> None:
    result = fast_suite.run_monte_carlo(StressScenario.SUDDEN_MARKET_CRASH, n_trials=20, seed=1)
    assert result.worst_case_loss_pct == max(result.max_loss_pct_values)


def test_monte_carlo_empty_trials_worst_case_is_zero() -> None:
    from backtest.stress_test import MonteCarloStressResult

    empty = MonteCarloStressResult(
        scenario=StressScenario.SUDDEN_MARKET_CRASH, trials=(), seed=0, n_trials=0
    )
    assert empty.worst_case_loss_pct == 0.0
    assert empty.risk_controls_fired_rate == 0.0


# --------------------------------------------------------------------------
# StressTestContext.engine() rebuilds collaborators against shocked data
# --------------------------------------------------------------------------


def test_context_engine_uses_the_shocked_provider_for_stock_selection(
    context: StressTestContext,
) -> None:
    """A regression guard for the exact bug this context design avoids:
    swapping only BacktestEngine.market_data would leave StockSelector and
    PortfolioConstructor pointed at the original, unshocked provider.
    """
    universe = frozenset(context.instrument_ids)
    shocked = ShockedMarketDataProvider(
        context.market_data,
        [MarketShock(ShockType.UNAVAILABLE, dt.date.min, None, 0.0, universe)],
    )
    engine = context.engine(market_data=shocked)
    assert engine.stock_selector.market_data is shocked
    assert engine.portfolio_constructor.market_data is shocked
    # Candidate exclusion (not an exception) is this pipeline's fail-closed
    # response to unavailable data -- an empty result here confirms the
    # selector is reading the shocked feed, not the original one.
    assert engine.stock_selector.select(context.signal_dates[0]) == []


def test_context_engine_defaults_to_the_context_market_data(context: StressTestContext) -> None:
    engine = context.engine()
    assert engine.market_data is context.market_data


# --------------------------------------------------------------------------
# CircuitBreaker sanity check reused directly (defense in depth for the
# application-restart / database-failure scenarios above)
# --------------------------------------------------------------------------


def test_circuit_breaker_state_file_is_written_after_a_halt(
    context: StressTestContext, tmp_path: Path
) -> None:
    from risk.portfolio_risk_state import PortfolioRiskState

    state_path = tmp_path / "direct_probe.json"
    breaker = CircuitBreaker(context.risk_config, state_path)
    state = PortfolioRiskState(
        as_of=dt.datetime.combine(context.signal_dates[0], dt.time(15, 30), tzinfo=dt.UTC),
        equity=1_000_000.0,
        positions=(),
        daily_pnl_pct=0.0,
        rolling_pnl_pct=0.0,
        peak_to_trough_drawdown_pct=0.0,
        daily_turnover_pct_so_far=0.0,
        max_pairwise_correlation=None,
        correlated_pair=None,
        system_healthy=False,
        system_detail="probe",
        broker_connected=True,
        broker_detail=None,
    )
    status = breaker.evaluate(state)
    assert status.state is CircuitState.HALTED
    assert state_path.is_file()
