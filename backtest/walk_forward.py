"""Walk-forward validation: the fit -> freeze -> out-of-sample -> advance ->
retrain loop, and the strategy comparison that answers docs/SPECIFICATION.md
section 10's central question -- does the HMM provide incremental value.

    historical data
        v
    training window            <- fit HMM + scaler on this window ONLY
        v
    fit HMM / factors / normalizers
        v
    freeze model                <- ScalerParams and FittedRegimeModel never change again
        v
    OOS simulation               <- backtest.engine.BacktestEngine, test window only
        v
    advance window                <- roll forward by roll_step_sessions
        v
    retrain
        v
    next OOS period

Every fold fits a *fresh* model on its own training window and evaluates it
strictly on the following test window -- the training window for fold N+1
overlaps fold N's training window (it slides forward, it does not expand),
but a fold's test window is never touched by that fold's own fit. The one
exception, and it is not an exception to that rule: :meth:`_hmm_regime_states`
extends *inference* (never fitting) a little before ``test_start`` so the
already-frozen model's confirmation/flicker logic has real context on the
test window's first sessions, instead of manufacturing an artificial
"not enough history" gap at every single fold boundary. Forward filtering
is causal regardless of where it starts (docs/SPECIFICATION.md section
6.3) -- extending how far back an already-frozen model is *run* leaks
nothing, because no parameter is ever re-estimated from it.

## Dates, not timestamps

Every date here is ``datetime.date``, matching every other module in this
codebase (``TradingCalendar``, ``AllocationTarget.as_of``,
``core.regime.hmm_engine.RegimeState.as_of``, ...). The original Phase 1
stub this module replaces used ``pandas.Timestamp`` throughout; that was a
placeholder guess made before any of those other modules existed, and it
is corrected here rather than preserved for its own sake.

## Five strategies, one identical pipeline

:meth:`WalkForwardValidator.run_all_strategies` runs buy-and-hold, the
rolling-volatility baseline, the moving-average trend baseline, the HMM,
and a shuffled-regime control over the *same* fold sequence, the *same*
stock selector, portfolio constructor, risk manager, and cost model --
the only thing that ever differs between them is which
``AllocationTarget`` time series each one produces
(docs/SPECIFICATION.md section 10.1). This is what makes "the HMM beats
the simple baseline after costs" a checkable claim rather than an assumed
one (docs/SPECIFICATION.md section 10.3).

Folds chain into one continuous ledger: each fold's ``BacktestEngine.run()``
starts from the *previous* fold's ending equity, not a fresh
``initial_equity`` every time. Holdings themselves do not carry across a
fold boundary -- a retrain is treated as a flatten-and-reassess point, since
the new fold's stock selector may not even rank the same instruments. This
is a documented simplification, not an attempt to model a real
zero-cost liquidation.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import CostModel
from backtest.engine import BacktestEngine, BacktestResult
from backtest.performance import PerformanceCalculator, PerformanceReport
from config.models import AllocationConfig, BacktestConfig, HMMConfig, RiskConfig
from core.features.feature_engineering import (
    FeaturePipeline,
    MarketFeatureInputs,
    drop_warmup_rows,
    feature_set_version,
)
from core.features.feature_scaler import CausalFeatureScaler, ScalerParams
from core.regime.allocation import AllocationRegime, AllocationTarget, RegimeAllocationEngine
from core.regime.baseline_policy import MovingAverageTrendBaseline, RollingVolatilityBaseline
from core.regime.hmm_engine import FittedRegimeModel, HMMRegimeEngine, RegimeState
from core.regime.model_registry import build_model_id
from core.regime.regime_policy import RegimePolicy
from data.interfaces import CorporateActionProvider, MarketDataProvider, TradingCalendar
from portfolio.portfolio_constructor import PortfolioConstructor
from universe.stock_selector import StockSelector

BUY_AND_HOLD = "buy_and_hold"
ROLLING_VOLATILITY = "rolling_volatility"
MOVING_AVERAGE_TREND = "moving_average_trend"
HMM = "hmm"
SHUFFLED_REGIME_CONTROL = "shuffled_regime_control"

STRATEGY_NAMES = (
    BUY_AND_HOLD,
    ROLLING_VOLATILITY,
    MOVING_AVERAGE_TREND,
    HMM,
    SHUFFLED_REGIME_CONTROL,
)


class WalkForwardError(RuntimeError):
    """A walk-forward run could not proceed."""


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date
    model_id: str
    performance: PerformanceReport


class WalkForwardValidator:
    def __init__(
        self,
        config: BacktestConfig,
        calendar: TradingCalendar,
        market_data: MarketDataProvider,
        stock_selector: StockSelector,
        portfolio_constructor: PortfolioConstructor,
        risk_config: RiskConfig,
        cost_model: CostModel,
        circuit_breaker_state_dir: Path,
        hmm_config: HMMConfig,
        allocation_config: AllocationConfig,
        regime_policy: RegimePolicy,
        feature_pipeline: FeaturePipeline,
        index_symbol: str,
        vix_symbol: str,
        corporate_actions: CorporateActionProvider | None = None,
        initial_equity: float = 10_000_000.0,
        random_seed: int = 7,
        feature_warmup_buffer_days: int = 500,
    ) -> None:
        self.config = config
        self.calendar = calendar
        self.market_data = market_data
        self.hmm_config = hmm_config
        self.allocation_config = allocation_config
        self.regime_policy = regime_policy
        self.feature_pipeline = feature_pipeline
        self.index_symbol = index_symbol
        self.vix_symbol = vix_symbol
        self.initial_equity = initial_equity
        self.random_seed = random_seed
        self.feature_warmup_buffer_days = feature_warmup_buffer_days
        self.performance_calculator = PerformanceCalculator()

        self.engine = BacktestEngine(
            calendar=calendar,
            market_data=market_data,
            stock_selector=stock_selector,
            portfolio_constructor=portfolio_constructor,
            risk_config=risk_config,
            cost_model=cost_model,
            circuit_breaker_state_dir=circuit_breaker_state_dir,
            corporate_actions=corporate_actions,
        )

    # -- fold generation ----------------------------------------------------

    def generate_folds(
        self, start: dt.date, end: dt.date
    ) -> list[tuple[dt.date, dt.date, dt.date, dt.date]]:
        """Train/test window boundaries per ``backtest.training_window_sessions``,
        ``backtest.test_window_sessions``, and ``backtest.roll_step_sessions``.

        A rolling (not expanding) training window: each fold's training
        window is the same length as every other's, sliding forward by
        ``roll_step_sessions`` each time -- so a later fold's model is
        never fit on strictly more history than an earlier one's, isolating
        "did the regime change" from "did the model just see more data".
        """
        trading_days = self.calendar.trading_days_between(start, end)
        train_n = self.config.training_window_sessions
        test_n = self.config.test_window_sessions
        step_n = self.config.roll_step_sessions

        folds: list[tuple[dt.date, dt.date, dt.date, dt.date]] = []
        train_start_idx = 0
        while True:
            train_end_idx = train_start_idx + train_n - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = test_start_idx + test_n - 1
            if test_end_idx >= len(trading_days):
                break
            folds.append(
                (
                    trading_days[train_start_idx],
                    trading_days[train_end_idx],
                    trading_days[test_start_idx],
                    trading_days[test_end_idx],
                )
            )
            train_start_idx += step_n
        return folds

    # -- HMM: fit once per fold, freeze, infer -------------------------------

    def _fit_fold(
        self, train_start: dt.date, train_end: dt.date
    ) -> tuple[FittedRegimeModel, ScalerParams, HMMRegimeEngine, str]:
        """Fit a fresh HMM + scaler on ``[train_start, train_end]`` only.

        Nothing computed here ever sees a date after ``train_end`` --
        the training feature matrix, the scaler's mean/std, and the fitted
        HMM's parameters are all a pure function of this window alone.
        """
        raw_features = self._raw_features(train_start, train_end)
        matrix = drop_warmup_rows(self.feature_pipeline.compute(
            MarketFeatureInputs(raw_features)
        ))
        if matrix.empty:
            raise WalkForwardError(
                f"no complete (post-warmup) feature rows in training window "
                f"[{train_start}, {train_end}]"
            )
        scaler = CausalFeatureScaler()
        scaled, scaler_params = scaler.fit_transform(matrix)
        returns = raw_features["nifty_close"].pct_change().reindex(scaled.index)

        engine = HMMRegimeEngine(self.hmm_config)
        model = engine.fit(scaled, returns)
        model_id = build_model_id(
            train_end,
            model.n_states,
            model.training_result.seed,
            feature_set_version(self.feature_pipeline.definitions),
        )
        return model, scaler_params, engine, model_id

    def _raw_features(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        nifty = self.market_data.get_index_observations(self.index_symbol, start, end)
        vix = self.market_data.get_index_observations(self.vix_symbol, start, end)
        inputs = MarketFeatureInputs.from_index_observations(nifty, vix)
        return inputs.frame

    def _hmm_exposure_targets(
        self,
        model: FittedRegimeModel,
        params: ScalerParams,
        engine: HMMRegimeEngine,
        test_start: dt.date,
        test_end: dt.date,
    ) -> tuple[dict[dt.date, AllocationTarget], list[RegimeState]]:
        buffer_start = test_start - dt.timedelta(days=self.feature_warmup_buffer_days)
        nifty = self.market_data.get_index_observations(self.index_symbol, buffer_start, test_end)
        vix = self.market_data.get_index_observations(self.vix_symbol, buffer_start, test_end)
        inputs = MarketFeatureInputs.from_index_observations(nifty, vix)
        matrix = drop_warmup_rows(self.feature_pipeline.compute(inputs))
        scaler = CausalFeatureScaler()
        scaled = scaler.transform(matrix, params)
        all_states = engine.filter(model, scaled)

        allocation_engine = RegimeAllocationEngine(
            self.hmm_config, self.allocation_config, self.regime_policy
        )
        history: list[RegimeState] = []
        targets: dict[dt.date, AllocationTarget] = {}
        test_states: list[RegimeState] = []
        for state in all_states:
            history.append(state)
            if state.as_of >= test_start:
                targets[state.as_of] = allocation_engine.evaluate(history)
                test_states.append(state)
        return targets, test_states

    def _shuffled_exposure_targets(
        self, test_states: list[RegimeState], seed: int
    ) -> dict[dt.date, AllocationTarget]:
        """Re-derive exposure targets from the *same* states the HMM
        actually produced for this fold, with their date assignment
        randomly permuted (docs/SPECIFICATION.md section 10.1's
        shuffled-regime control) -- the same total time spent in each
        regime, just reordered, to test whether *when* the HMM called a
        regime mattered, not merely that it called some regime some of the
        time.
        """
        dates = [state.as_of for state in test_states]
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(test_states))
        shuffled_states = [
            replace(test_states[index], as_of=date)
            for index, date in zip(permutation, dates, strict=True)
        ]
        shuffled_states.sort(key=lambda state: state.as_of)

        allocation_engine = RegimeAllocationEngine(
            self.hmm_config, self.allocation_config, self.regime_policy
        )
        history: list[RegimeState] = []
        targets: dict[dt.date, AllocationTarget] = {}
        for state in shuffled_states:
            history.append(state)
            targets[state.as_of] = allocation_engine.evaluate(history)
        return targets

    # -- non-HMM baselines ----------------------------------------------------

    def _buy_and_hold_targets(
        self, dates: list[dt.date]
    ) -> dict[dt.date, AllocationTarget]:
        band = self.regime_policy.band_for(AllocationRegime.LOW_RISK)
        return {
            day: AllocationTarget(
                as_of=day,
                regime=AllocationRegime.LOW_RISK,
                target_gross_exposure=band.max_gross_exposure,
                min_gross_exposure=band.max_gross_exposure,
                max_gross_exposure=band.max_gross_exposure,
                allow_new_positions=True,
                confidence=1.0,
                expected_volatility=0.0,
                reason="buy_and_hold: always fully invested, no market timing",
            )
            for day in dates
        }

    def _rolling_volatility_targets(
        self, dates: list[dt.date]
    ) -> dict[dt.date, AllocationTarget]:
        buffer_start = dates[0] - dt.timedelta(days=self.feature_warmup_buffer_days)
        raw = self._raw_features(buffer_start, dates[-1])
        returns = raw["nifty_close"].pct_change().dropna()
        baseline = RollingVolatilityBaseline(self.allocation_config, self.regime_policy)
        targets: dict[dt.date, AllocationTarget] = {}
        for day in dates:
            window = returns[returns.index <= pd.Timestamp(day)]
            targets[day] = baseline.evaluate(window, as_of=day)
        return targets

    def _moving_average_trend_targets(
        self, dates: list[dt.date]
    ) -> dict[dt.date, AllocationTarget]:
        buffer_start = dates[0] - dt.timedelta(days=self.feature_warmup_buffer_days)
        raw = self._raw_features(buffer_start, dates[-1])
        prices = raw["nifty_close"]
        baseline = MovingAverageTrendBaseline(self.allocation_config, self.regime_policy)
        targets: dict[dt.date, AllocationTarget] = {}
        for day in dates:
            window = prices[prices.index <= pd.Timestamp(day)]
            targets[day] = baseline.evaluate(window, as_of=day)
        return targets

    # -- running one fold, one strategy ----------------------------------------

    def _run_strategy_on_fold(
        self,
        strategy_name: str,
        exposure_targets: dict[dt.date, AllocationTarget],
        dates: list[dt.date],
        initial_equity: float,
    ) -> BacktestResult:
        return self.engine.run(strategy_name, exposure_targets, dates, initial_equity)

    # -- public entry points --------------------------------------------------

    def run(self, start: dt.date, end: dt.date) -> list[WalkForwardFold]:
        """The HMM strategy only, one entry per fold -- walk-forward
        validation proper (docs/SPECIFICATION.md section 10).
        """
        folds = self.generate_folds(start, end)
        if not folds:
            raise WalkForwardError(
                f"no fold fits in [{start}, {end}] given the configured window sizes"
            )
        results: list[WalkForwardFold] = []
        equity = self.initial_equity
        for train_start, train_end, test_start, test_end in folds:
            model, params, engine, model_id = self._fit_fold(train_start, train_end)
            targets, _states = self._hmm_exposure_targets(
                model, params, engine, test_start, test_end
            )
            dates = sorted(targets)
            result = self._run_strategy_on_fold(HMM, targets, dates, equity)
            equity = float(result.equity_curve.iloc[-1])
            performance = self.performance_calculator.compute(
                result.equity_curve, result.trade_log
            )
            results.append(
                WalkForwardFold(train_start, train_end, test_start, test_end, model_id, performance)
            )
        return results

    def run_shuffled_regime_control(self, start: dt.date, end: dt.date) -> PerformanceReport:
        """Re-run with each fold's regime states randomly permuted, chained
        across folds exactly like :meth:`run`, and reported as one overall
        performance figure -- docs/SPECIFICATION.md section 10.1's
        shuffled-regime control.
        """
        folds = self.generate_folds(start, end)
        if not folds:
            raise WalkForwardError(
                f"no fold fits in [{start}, {end}] given the configured window sizes"
            )
        equity_curves: list[pd.Series] = []
        trade_logs: list[pd.DataFrame] = []
        equity = self.initial_equity
        for fold_index, (train_start, train_end, test_start, test_end) in enumerate(folds):
            model, params, engine, _model_id = self._fit_fold(train_start, train_end)
            _targets, test_states = self._hmm_exposure_targets(
                model, params, engine, test_start, test_end
            )
            shuffled = self._shuffled_exposure_targets(
                test_states, seed=self.random_seed + fold_index
            )
            dates = sorted(shuffled)
            result = self._run_strategy_on_fold(
                SHUFFLED_REGIME_CONTROL, shuffled, dates, equity
            )
            equity = float(result.equity_curve.iloc[-1])
            equity_curves.append(result.equity_curve)
            trade_logs.append(result.trade_log)
        return self.performance_calculator.compute(
            pd.concat(equity_curves).sort_index(), pd.concat(trade_logs, ignore_index=True)
        )

    def run_all_strategies(self, start: dt.date, end: dt.date) -> dict[str, PerformanceReport]:
        """Buy-and-hold, the rolling-volatility baseline, the moving-average
        trend baseline, the HMM, and the shuffled-regime control, all over
        the identical fold sequence and the identical downstream
        selection/construction/risk/cost pipeline -- the comparison
        docs/SPECIFICATION.md section 10.1 asks for, to determine whether
        the HMM provides incremental value.
        """
        folds = self.generate_folds(start, end)
        if not folds:
            raise WalkForwardError(
                f"no fold fits in [{start}, {end}] given the configured window sizes"
            )

        equity_curves: dict[str, list[pd.Series]] = {name: [] for name in STRATEGY_NAMES}
        trade_logs: dict[str, list[pd.DataFrame]] = {name: [] for name in STRATEGY_NAMES}
        running_equity: dict[str, float] = dict.fromkeys(STRATEGY_NAMES, self.initial_equity)

        for fold_index, (train_start, train_end, test_start, test_end) in enumerate(folds):
            model, params, engine, _model_id = self._fit_fold(train_start, train_end)
            hmm_targets, test_states = self._hmm_exposure_targets(
                model, params, engine, test_start, test_end
            )
            dates = sorted(hmm_targets)
            shuffled_targets = self._shuffled_exposure_targets(
                test_states, seed=self.random_seed + fold_index
            )

            per_strategy_targets = {
                BUY_AND_HOLD: self._buy_and_hold_targets(dates),
                ROLLING_VOLATILITY: self._rolling_volatility_targets(dates),
                MOVING_AVERAGE_TREND: self._moving_average_trend_targets(dates),
                HMM: hmm_targets,
                SHUFFLED_REGIME_CONTROL: shuffled_targets,
            }

            for name in STRATEGY_NAMES:
                result = self._run_strategy_on_fold(
                    name, per_strategy_targets[name], dates, running_equity[name]
                )
                running_equity[name] = float(result.equity_curve.iloc[-1])
                equity_curves[name].append(result.equity_curve)
                trade_logs[name].append(result.trade_log)

        return {
            name: self.performance_calculator.compute(
                pd.concat(equity_curves[name]).sort_index(),
                pd.concat(trade_logs[name], ignore_index=True),
            )
            for name in STRATEGY_NAMES
        }
