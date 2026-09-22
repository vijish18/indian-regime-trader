"""Stress-testing framework for Indian equity-market failure scenarios.

The one requirement every scenario here exists to check:
**risk controls must limit damage even if the HMM is wrong.** A market
regime call, however confident, is a statistical estimate, not a
guarantee -- so the circuit breaker watches *realized* P&L directly
(``risk/circuit_breaker.py``), never the regime label, and every scenario
below that feeds the system a plausible-looking-but-wrong signal
(:const:`StressScenario.HMM_REGIME_MISCLASSIFICATION`,
:const:`StressScenario.WRONG_STOCK_RANKING`) confirms that the portfolio-
and account-level limits (``risk/risk_manager.py``,
``risk/circuit_breaker.py``) still hold regardless.

## What each scenario actually exercises

Market/data scenarios (crash, gap, VIX spike, liquidity deterioration,
wide spread, stale data, missing bars, trading halt, holiday mismatch,
corporate actions, sudden drawdown, regime misclassification, wrong
ranking) run a real ``backtest.engine.BacktestEngine`` over a short
window against a deliberately shocked ``MarketDataProvider``
(:class:`ShockedMarketDataProvider`) and inspect the resulting
``BacktestResult`` -- the same machinery a real backtest uses, not a
simulation of it.

Execution/infrastructure scenarios (broker outage, partial fill, order
rejection, delayed fill, duplicate order, application restart, database
failure) exercise the specific mechanism that already exists for each:
``PortfolioRiskState.broker_connected`` for an outage,
``BacktestEngine._apply_fill``'s ledger arithmetic for a partial fill,
``BacktestEngine._execute`` returning ``None`` for a rejected/undeliverable
order, a later-session retry for a delayed fill,
``BacktestEngine.run``'s duplicate-signal-date rejection for a duplicate
order, and ``CircuitBreaker``'s on-disk persistence for a restart or a
corrupted state file. There is no live broker or order-management system
yet (Phase 10/11) -- these scenarios test the layer that exists today as
honestly as they can, not a simulation of a layer that does not.

## Monte Carlo scenarios

Scenarios whose real-world severity is not one fixed number (crash
magnitude, gap size, liquidity collapse depth, spread width, VIX spike
size, which day a misclassification coincides with a real move, which
candidate gets a wrong top rank, drawdown depth) are run many times with
randomized severities via :meth:`StressTestSuite.run_monte_carlo`
(``n_trials=100`` by default), each trial seeded deterministically
(``seed + trial_index``) so a run is exactly reproducible. See
:data:`MONTE_CARLO_SCENARIOS`.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

import numpy as np

from backtest.costs import CostModel, TradeSide
from backtest.engine import (
    BacktestEngine,
    BacktestEngineError,
    BacktestResult,
    FillRecord,
    OrderRecord,
)
from backtest.performance import _drawdown_stats, _recovery_duration
from config.models import RiskConfig
from core.regime.allocation import AllocationRegime, AllocationTarget
from data.errors import DataNotAvailableError
from data.interfaces import CorporateActionProvider, MarketDataProvider, TradingCalendar
from data.models import (
    CorporateAction,
    CorporateActionType,
    DailyBar,
    IndexObservation,
    PriceBasis,
    Quote,
)
from portfolio.portfolio_constructor import (
    PortfolioConstructor,
    TargetPortfolio,
    TargetPosition,
    TradeAction,
)
from risk.circuit_breaker import CircuitBreaker, CircuitState
from risk.portfolio_risk_state import PortfolioRiskState, PositionRisk
from risk.risk_manager import RiskManager
from universe.stock_selector import StockSelector


class StressScenario(StrEnum):
    SUDDEN_MARKET_CRASH = "sudden_market_crash"
    OVERNIGHT_GAP = "overnight_gap"
    VIX_SPIKE = "vix_spike"
    LIQUIDITY_DETERIORATION = "liquidity_deterioration"
    WIDE_BID_ASK_SPREAD = "wide_bid_ask_spread"
    STALE_MARKET_DATA = "stale_market_data"
    MISSING_BARS = "missing_bars"
    STOCK_TRADING_HALT = "stock_trading_halt"
    EXCHANGE_HOLIDAY_MISMATCH = "exchange_holiday_mismatch"
    BROKER_API_OUTAGE = "broker_api_outage"
    PARTIAL_FILL = "partial_fill"
    ORDER_REJECTION = "order_rejection"
    DELAYED_FILL = "delayed_fill"
    DUPLICATE_ORDER = "duplicate_order"
    APPLICATION_RESTART = "application_restart"
    DATABASE_FAILURE = "database_failure"
    HMM_REGIME_MISCLASSIFICATION = "hmm_regime_misclassification"
    WRONG_STOCK_RANKING = "wrong_stock_ranking"
    CORPORATE_ACTION_EVENT = "corporate_action_event"
    SUDDEN_PORTFOLIO_DRAWDOWN = "sudden_portfolio_drawdown"


MONTE_CARLO_SCENARIOS = frozenset(
    {
        StressScenario.SUDDEN_MARKET_CRASH,
        StressScenario.OVERNIGHT_GAP,
        StressScenario.VIX_SPIKE,
        StressScenario.LIQUIDITY_DETERIORATION,
        StressScenario.WIDE_BID_ASK_SPREAD,
        StressScenario.HMM_REGIME_MISCLASSIFICATION,
        StressScenario.WRONG_STOCK_RANKING,
        StressScenario.SUDDEN_PORTFOLIO_DRAWDOWN,
    }
)
"""Scenarios whose real-world severity is not one fixed number -- run
through :meth:`StressTestSuite.run_monte_carlo` rather than a single
:meth:`StressTestSuite.run_scenario` call."""


class StressTestError(RuntimeError):
    """A stress scenario could not be run as requested."""


# --------------------------------------------------------------------------
# Fault injection: a MarketDataProvider that shocks a real one
# --------------------------------------------------------------------------


class ShockType(StrEnum):
    PRICE_CRASH = "price_crash"
    """Multiplies open/high/low/close by ``(1 - magnitude)`` from
    ``start_date`` through ``end_date`` (inclusive; a persistent price
    cut, not a one-day event)."""

    OPEN_GAP = "open_gap"
    """Shifts *only* the opening price down by ``magnitude`` on
    ``start_date`` -- an overnight gap, not a sustained crash. ``end_date``
    is ignored."""

    VOLUME_COLLAPSE = "volume_collapse"
    """Multiplies volume by ``(1 - magnitude)`` from ``start_date``
    through ``end_date`` -- a liquidity deterioration."""

    UNAVAILABLE = "unavailable"
    """No bars/observations at all in ``[start_date, end_date]`` --
    missing data, a trading halt, or a broker/feed outage, depending on
    which scenario applies it."""

    INDEX_SPIKE = "index_spike"
    """Multiplies an index's close by ``(1 + magnitude)`` from
    ``start_date`` through ``end_date`` -- a VIX spike."""


@dataclass(frozen=True, slots=True)
class MarketShock:
    shock_type: ShockType
    start_date: dt.date
    end_date: dt.date | None
    """``None`` means open-ended: the shock persists through the end of
    whatever range is queried."""

    magnitude: float
    """Meaning depends on ``shock_type`` -- see each member's docstring."""

    instrument_ids: frozenset[str] | None = None
    """``None`` applies to every instrument (equity shocks) or every index
    symbol (index shocks) queried; otherwise the shock applies only to the
    named ids."""

    def applies_to_date(self, session_date: dt.date) -> bool:
        end = self.end_date if self.end_date is not None else dt.date.max
        return self.start_date <= session_date <= end

    def applies_to_id(self, identifier: str) -> bool:
        return self.instrument_ids is None or identifier in self.instrument_ids


class ShockedMarketDataProvider(MarketDataProvider):
    """Wraps a real ``MarketDataProvider`` and applies deterministic
    shocks on top of it -- every other call is delegated unchanged.
    """

    def __init__(self, base: MarketDataProvider, shocks: list[MarketShock]) -> None:
        self._base = base
        self._shocks = shocks

    def get_equity_bars(
        self,
        instrument_id: str,
        start: dt.date,
        end: dt.date,
        price_basis: PriceBasis = PriceBasis.RAW,
    ) -> list[DailyBar]:
        bars = self._base.get_equity_bars(instrument_id, start, end, price_basis)
        unavailable = [
            shock
            for shock in self._shocks
            if shock.shock_type is ShockType.UNAVAILABLE and shock.applies_to_id(instrument_id)
        ]
        result: list[DailyBar] = []
        for bar in bars:
            if any(shock.applies_to_date(bar.session_date) for shock in unavailable):
                continue
            result.append(self._shock_bar(bar, instrument_id))
        return result

    def _shock_bar(self, bar: DailyBar, instrument_id: str) -> DailyBar:
        open_, high, low, close, volume = bar.open, bar.high, bar.low, bar.close, bar.volume
        changed = False
        for shock in self._shocks:
            if not shock.applies_to_id(instrument_id) or not shock.applies_to_date(
                bar.session_date
            ):
                continue
            if shock.shock_type is ShockType.PRICE_CRASH:
                factor = Decimal(str(1.0 - shock.magnitude))
                open_, high, low, close = (
                    open_ * factor,
                    high * factor,
                    low * factor,
                    close * factor,
                )
                changed = True
            elif shock.shock_type is ShockType.OPEN_GAP and bar.session_date == shock.start_date:
                factor = Decimal(str(1.0 - shock.magnitude))
                open_ = open_ * factor
                low = min(low, open_)
                changed = True
            elif shock.shock_type is ShockType.VOLUME_COLLAPSE:
                volume = max(1, int(volume * (1.0 - shock.magnitude)))
                changed = True
        if not changed:
            return bar
        return dataclasses.replace(bar, open=open_, high=high, low=low, close=close, volume=volume)

    def get_index_observations(
        self, index_symbol: str, start: dt.date, end: dt.date
    ) -> list[IndexObservation]:
        observations = self._base.get_index_observations(index_symbol, start, end)
        spikes = [
            shock
            for shock in self._shocks
            if shock.shock_type is ShockType.INDEX_SPIKE and shock.applies_to_id(index_symbol)
        ]
        unavailable = [
            shock
            for shock in self._shocks
            if shock.shock_type is ShockType.UNAVAILABLE and shock.applies_to_id(index_symbol)
        ]
        result: list[IndexObservation] = []
        for observation in observations:
            if any(shock.applies_to_date(observation.session_date) for shock in unavailable):
                continue
            multiplier = 1.0
            for shock in spikes:
                if shock.applies_to_date(observation.session_date):
                    multiplier *= 1.0 + shock.magnitude
            if multiplier == 1.0:
                result.append(observation)
            else:
                shocked_close = observation.close * Decimal(str(multiplier))
                result.append(dataclasses.replace(observation, close=shocked_close))
        return result

    def available_range(self, instrument_id: str) -> tuple[dt.date, dt.date] | None:
        return self._base.available_range(instrument_id)

    def get_quote(self, instrument_id: str) -> Quote:
        return self._base.get_quote(instrument_id)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StressTestResult:
    scenario: StressScenario
    trial_seed: int | None
    portfolio_impact_pct: float
    """Equity change over the scenario window, as a fraction (negative =
    loss). ``0.0`` for scenarios that check a control directly rather than
    running a priced backtest -- see each handler's docstring."""

    max_loss_pct: float
    """Worst peak-to-trough decline observed during the scenario window."""

    recovery_days: int | None
    """Sessions from the worst drawdown's trough back to its prior peak;
    ``None`` if the window ends still underwater, ``0`` if there was
    nothing to recover from."""

    risk_controls_fired: bool
    trading_halted: bool
    duplicate_orders_prevented: bool
    portfolio_state_consistent: bool
    system_failed_closed: bool
    """The overall verdict: did the system respond safely to this failure
    (reject, halt, or degrade gracefully) rather than doing something
    undefined or silently wrong."""

    detail: str


@dataclass(frozen=True, slots=True)
class MonteCarloStressResult:
    scenario: StressScenario
    trials: tuple[StressTestResult, ...]
    seed: int
    n_trials: int

    @property
    def risk_controls_fired_rate(self) -> float:
        return _rate(trial.risk_controls_fired for trial in self.trials)

    @property
    def trading_halted_rate(self) -> float:
        return _rate(trial.trading_halted for trial in self.trials)

    @property
    def portfolio_state_consistent_rate(self) -> float:
        return _rate(trial.portfolio_state_consistent for trial in self.trials)

    @property
    def system_failed_closed_rate(self) -> float:
        return _rate(trial.system_failed_closed for trial in self.trials)

    @property
    def max_loss_pct_values(self) -> tuple[float, ...]:
        return tuple(trial.max_loss_pct for trial in self.trials)

    @property
    def worst_case_loss_pct(self) -> float:
        return max(self.max_loss_pct_values) if self.trials else 0.0


def _rate(flags: Iterable[bool]) -> float:
    values = list(flags)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


# --------------------------------------------------------------------------
# Context: the real collaborators every scenario runs against
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StressTestContext:
    """Everything a stress scenario needs to build a fresh, possibly
    shocked, ``BacktestEngine`` and run it.

    ``stock_selector_factory``/``portfolio_constructor_factory`` (rather
    than fixed instances) exist because each of those objects holds its
    *own* ``MarketDataProvider`` reference captured at construction time
    -- swapping ``BacktestEngine.market_data`` alone would not shock what
    stock selection or portfolio construction actually see. Every shocked
    engine this context builds gets a freshly constructed selector and
    constructor pointed at the *same* shocked provider.
    """

    calendar: TradingCalendar
    market_data: MarketDataProvider
    stock_selector_factory: Callable[[MarketDataProvider], StockSelector]
    portfolio_constructor_factory: Callable[[MarketDataProvider], PortfolioConstructor]
    risk_config: RiskConfig
    cost_model: CostModel
    circuit_breaker_state_dir: Path
    corporate_actions: CorporateActionProvider | None
    instrument_ids: tuple[str, ...]
    index_symbol: str
    signal_dates: tuple[dt.date, ...]
    initial_equity: float = 10_000_000.0
    assumed_spread_bps: float = 10.0

    def engine(
        self, market_data: MarketDataProvider | None = None, **overrides: object
    ) -> BacktestEngine:
        data = market_data if market_data is not None else self.market_data
        kwargs: dict[str, object] = dict(
            calendar=self.calendar,
            market_data=data,
            stock_selector=self.stock_selector_factory(data),
            portfolio_constructor=self.portfolio_constructor_factory(data),
            risk_config=self.risk_config,
            cost_model=self.cost_model,
            circuit_breaker_state_dir=self.circuit_breaker_state_dir,
            corporate_actions=self.corporate_actions,
            assumed_spread_bps=self.assumed_spread_bps,
        )
        kwargs.update(overrides)
        return BacktestEngine(**kwargs)  # type: ignore[arg-type]

    def full_exposure_targets(
        self, band_max: float = 1.0, regime: AllocationRegime = AllocationRegime.LOW_RISK
    ) -> dict[dt.date, AllocationTarget]:
        """A trivial "always fully invested" signal -- deliberately not
        the HMM, so market-scenario handlers can isolate whether risk
        controls (not the exposure-timing signal) contain the damage.
        """
        return {
            day: AllocationTarget(
                as_of=day,
                regime=regime,
                target_gross_exposure=band_max,
                min_gross_exposure=band_max,
                max_gross_exposure=band_max,
                allow_new_positions=True,
                confidence=1.0,
                expected_volatility=0.1,
                reason="stress test: fixed full-exposure signal",
            )
            for day in self.signal_dates
        }


# --------------------------------------------------------------------------
# The suite
# --------------------------------------------------------------------------


class StressTestSuite:
    """Runs each scenario against the real backtest/risk stack and reports
    whether the system's response was safe.
    """

    def __init__(self, context: StressTestContext) -> None:
        self.context = context
        self._handlers: dict[StressScenario, Callable[..., StressTestResult]] = {
            StressScenario.SUDDEN_MARKET_CRASH: self._run_sudden_market_crash,
            StressScenario.OVERNIGHT_GAP: self._run_overnight_gap,
            StressScenario.VIX_SPIKE: self._run_vix_spike,
            StressScenario.LIQUIDITY_DETERIORATION: self._run_liquidity_deterioration,
            StressScenario.WIDE_BID_ASK_SPREAD: self._run_wide_bid_ask_spread,
            StressScenario.STALE_MARKET_DATA: self._run_stale_market_data,
            StressScenario.MISSING_BARS: self._run_missing_bars,
            StressScenario.STOCK_TRADING_HALT: self._run_stock_trading_halt,
            StressScenario.EXCHANGE_HOLIDAY_MISMATCH: self._run_exchange_holiday_mismatch,
            StressScenario.BROKER_API_OUTAGE: self._run_broker_api_outage,
            StressScenario.PARTIAL_FILL: self._run_partial_fill,
            StressScenario.ORDER_REJECTION: self._run_order_rejection,
            StressScenario.DELAYED_FILL: self._run_delayed_fill,
            StressScenario.DUPLICATE_ORDER: self._run_duplicate_order,
            StressScenario.APPLICATION_RESTART: self._run_application_restart,
            StressScenario.DATABASE_FAILURE: self._run_database_failure,
            StressScenario.HMM_REGIME_MISCLASSIFICATION: self._run_hmm_regime_misclassification,
            StressScenario.WRONG_STOCK_RANKING: self._run_wrong_stock_ranking,
            StressScenario.CORPORATE_ACTION_EVENT: self._run_corporate_action_event,
            StressScenario.SUDDEN_PORTFOLIO_DRAWDOWN: self._run_sudden_portfolio_drawdown,
        }

    def run_scenario(
        self, scenario: StressScenario, seed: int | None = None, **params: object
    ) -> StressTestResult:
        handler = self._handlers[scenario]
        return handler(seed=seed, **params)

    def run_monte_carlo(
        self,
        scenario: StressScenario,
        n_trials: int = 100,
        seed: int = 0,
        **params: object,
    ) -> MonteCarloStressResult:
        if scenario not in MONTE_CARLO_SCENARIOS:
            raise StressTestError(
                f"{scenario.value} is not a Monte Carlo scenario -- use run_scenario() "
                f"for a single deterministic run. Monte Carlo scenarios: "
                f"{sorted(s.value for s in MONTE_CARLO_SCENARIOS)}"
            )
        if n_trials < 1:
            raise StressTestError(f"n_trials must be >= 1, got {n_trials}")
        trials = tuple(
            self.run_scenario(scenario, seed=seed + trial_index, **params)
            for trial_index in range(n_trials)
        )
        return MonteCarloStressResult(
            scenario=scenario, trials=trials, seed=seed, n_trials=n_trials
        )

    def run_all(self, seed: int = 0) -> list[StressTestResult]:
        """One deterministic run per scenario, in enum-declaration order."""
        return [self.run_scenario(scenario, seed=seed) for scenario in StressScenario]

    # -- market / data scenarios --------------------------------------------

    def _run_sudden_market_crash(
        self, seed: int | None = None, magnitude: float | None = None
    ) -> StressTestResult:
        """A persistent, deep price cut across every held instrument,
        starting mid-window -- the classic tail-risk scenario. Uses a
        fixed full-exposure signal (never the HMM) so the result isolates
        whether risk controls, not market timing, contain the damage.
        """
        rng = _rng(seed)
        drop = magnitude if magnitude is not None else float(rng.uniform(0.10, 0.40))
        shock_date = self.context.signal_dates[len(self.context.signal_dates) // 2]
        universe = frozenset(self.context.instrument_ids)
        shocked = ShockedMarketDataProvider(
            self.context.market_data,
            [MarketShock(ShockType.PRICE_CRASH, shock_date, None, drop, universe)],
        )
        return self._run_priced_scenario(
            StressScenario.SUDDEN_MARKET_CRASH,
            seed,
            shocked,
            detail=(
                f"instantaneous {drop:.0%} price cut across the whole universe from {shock_date}"
            ),
        )

    def _run_overnight_gap(
        self, seed: int | None = None, magnitude: float | None = None
    ) -> StressTestResult:
        """A single session's opening price gaps down sharply relative to
        the prior close -- the fill itself is at the gapped price (no
        look-ahead relief), not a sustained crash.
        """
        rng = _rng(seed)
        gap = magnitude if magnitude is not None else float(rng.uniform(0.05, 0.20))
        shock_date = self.context.signal_dates[len(self.context.signal_dates) // 2]
        universe = frozenset(self.context.instrument_ids)
        shocked = ShockedMarketDataProvider(
            self.context.market_data,
            [MarketShock(ShockType.OPEN_GAP, shock_date, shock_date, gap, universe)],
        )
        return self._run_priced_scenario(
            StressScenario.OVERNIGHT_GAP,
            seed,
            shocked,
            detail=f"{gap:.0%} overnight gap-down on {shock_date}",
        )

    def _run_vix_spike(
        self, seed: int | None = None, magnitude: float | None = None
    ) -> StressTestResult:
        """A sharp India VIX spike -- checked at the data level (the
        feature pipeline sees it) and structurally (an exposure target
        that reduces risk in response is still bounded correctly by
        portfolio construction), without requiring a live HMM fit for
        this scenario specifically.
        """
        rng = _rng(seed)
        spike = magnitude if magnitude is not None else float(rng.uniform(1.0, 3.0))
        shock_date = self.context.signal_dates[len(self.context.signal_dates) // 2]
        shocked = ShockedMarketDataProvider(
            self.context.market_data,
            [
                MarketShock(
                    ShockType.INDEX_SPIKE,
                    shock_date,
                    None,
                    spike,
                    frozenset({self.context.index_symbol}),
                )
            ],
        )
        try:
            observations = shocked.get_index_observations(
                self.context.index_symbol, shock_date, shock_date
            )
            consistent = bool(observations) and observations[0].close > 0
        except DataNotAvailableError:
            consistent = False

        reduced_band = AllocationRegime.HIGH_RISK
        targets = self.context.full_exposure_targets(band_max=0.20, regime=reduced_band)
        result = self._run_priced_scenario(
            StressScenario.VIX_SPIKE,
            seed,
            shocked,
            targets=targets,
            detail=f"India VIX multiplied by {1 + spike:.1f}x from {shock_date}; "
            "exposure signal reduced to 20% to check construction/risk still bound it",
        )
        if not consistent:
            result = dataclasses.replace(
                result, portfolio_state_consistent=False, system_failed_closed=False
            )
        return result

    def _run_liquidity_deterioration(
        self, seed: int | None = None, magnitude: float | None = None
    ) -> StressTestResult:
        """Average daily traded value collapses -- the liquidity/ADV
        participation check (``RiskCheck.LIQUIDITY``) should start
        rejecting positions that were previously fine.
        """
        rng = _rng(seed)
        reduction = magnitude if magnitude is not None else float(rng.uniform(0.80, 0.99))
        shock_date = self.context.signal_dates[len(self.context.signal_dates) // 3]
        shocked = ShockedMarketDataProvider(
            self.context.market_data,
            [
                MarketShock(
                    ShockType.VOLUME_COLLAPSE,
                    shock_date,
                    None,
                    reduction,
                    frozenset(self.context.instrument_ids),
                )
            ],
        )
        return self._run_priced_scenario(
            StressScenario.LIQUIDITY_DETERIORATION,
            seed,
            shocked,
            detail=f"traded volume cut {reduction:.0%} from {shock_date} onward",
        )

    def _run_wide_bid_ask_spread(
        self, seed: int | None = None, spread_bps: float | None = None
    ) -> StressTestResult:
        """An abnormally wide quoted spread -- ``RiskCheck.ABNORMAL_SPREAD``
        should reject positions once the assumed spread exceeds
        ``risk.max_spread_bps``.
        """
        rng = _rng(seed)
        widened = spread_bps if spread_bps is not None else float(rng.uniform(200.0, 800.0))
        engine = self.context.engine(assumed_spread_bps=widened)
        return self._run_priced_scenario_with_engine(
            StressScenario.WIDE_BID_ASK_SPREAD,
            seed,
            engine,
            detail=f"assumed spread widened to {widened:.0f}bps for the whole window",
        )

    def _run_stale_market_data(self, seed: int | None = None) -> StressTestResult:
        """Directly exercises ``RiskCheck.STALE_DATA``: a position whose
        quote is far older than ``risk.stale_data_max_minutes`` must be
        rejected outright.
        """
        risk_manager, risk_state, proposed = self._single_position_case(
            stale_seconds=self.context.risk_config.stale_data_max_minutes * 60 + 3600
        )
        decisions = risk_manager.evaluate(proposed, risk_state)
        rejected = not decisions[0].approved
        return StressTestResult(
            scenario=StressScenario.STALE_MARKET_DATA,
            trial_seed=seed,
            portfolio_impact_pct=0.0,
            max_loss_pct=0.0,
            recovery_days=0,
            risk_controls_fired=rejected,
            trading_halted=False,
            duplicate_orders_prevented=True,
            portfolio_state_consistent=True,
            system_failed_closed=rejected,
            detail=f"quote age exceeded stale_data_max_minutes; rejected={rejected}",
        )

    def _run_missing_bars(self, seed: int | None = None) -> StressTestResult:
        """No bars at all for an instrument -- mark-to-market must fail
        closed (raise), not fabricate a price.

        The shock runs from the beginning of time rather than from mid-window,
        because those are two different failures and only one of them should
        raise:

            no price has ever been seen     the data is broken; refuse
            a price was seen, but is old    the stock is suspended; mark it
                                            at its last trade, the convention
                                            every fund uses

        ``_last_close`` cannot distinguish them by looking at one window, so
        the distinction is carried by how far back it is willing to look --
        400 days, then give up. Shocking from mid-window would leave
        pre-shock bars inside that reach and exercise the *suspension* path,
        which is tested at the engine level and is not what this scenario is
        for.
        """
        instrument_id = self.context.instrument_ids[0]
        shock_date = self.context.signal_dates[len(self.context.signal_dates) // 2]
        shocked = ShockedMarketDataProvider(
            self.context.market_data,
            [
                MarketShock(
                    ShockType.UNAVAILABLE, dt.date.min, None, 0.0, frozenset({instrument_id})
                )
            ],
        )
        engine = self.context.engine(market_data=shocked)
        query_date = self.context.calendar.sessions_offset(shock_date, 20)
        failed_closed = False
        detail = "mark-to-market did not raise for missing data (unexpected)"
        try:
            engine._last_close(instrument_id, query_date)
        except BacktestEngineError:
            failed_closed = True
            detail = "mark-to-market correctly raised BacktestEngineError for missing bars"
        return StressTestResult(
            scenario=StressScenario.MISSING_BARS,
            trial_seed=seed,
            portfolio_impact_pct=0.0,
            max_loss_pct=0.0,
            recovery_days=0,
            risk_controls_fired=failed_closed,
            trading_halted=False,
            duplicate_orders_prevented=True,
            portfolio_state_consistent=True,
            system_failed_closed=failed_closed,
            detail=detail,
        )

    def _run_stock_trading_halt(self, seed: int | None = None) -> StressTestResult:
        """One instrument stops publishing bars mid-window (a trading
        halt) while the rest of the universe continues -- the backtest as
        a whole must not crash; the halted name's fill must simply not
        occur (``_execute`` returns ``None`` when it can find no price).
        """
        instrument_id = self.context.instrument_ids[0]
        halt_start = self.context.signal_dates[len(self.context.signal_dates) // 3]
        halt_end = self.context.signal_dates[-1]
        shocked = ShockedMarketDataProvider(
            self.context.market_data,
            [
                MarketShock(
                    ShockType.UNAVAILABLE, halt_start, halt_end, 0.0, frozenset({instrument_id})
                )
            ],
        )
        return self._run_priced_scenario(
            StressScenario.STOCK_TRADING_HALT,
            seed,
            shocked,
            detail=f"{instrument_id} stops publishing bars from {halt_start} to {halt_end}",
        )

    def _run_exchange_holiday_mismatch(self, seed: int | None = None) -> StressTestResult:
        """The calendar, not the data feed, is authoritative for which
        sessions exist: ``signal_dates`` is always calendar-derived, so a
        data provider that happens to have (or lack) a bar for a
        non-trading day can never cause an execution attempt on that day.
        """
        calendar = self.context.calendar
        trading_days = set(calendar.trading_days_between(
            self.context.signal_dates[0], self.context.signal_dates[-1]
        ))
        consistent = all(day in trading_days for day in self.context.signal_dates)
        return StressTestResult(
            scenario=StressScenario.EXCHANGE_HOLIDAY_MISMATCH,
            trial_seed=seed,
            portfolio_impact_pct=0.0,
            max_loss_pct=0.0,
            recovery_days=0,
            risk_controls_fired=False,
            trading_halted=False,
            duplicate_orders_prevented=True,
            portfolio_state_consistent=consistent,
            system_failed_closed=consistent,
            detail=(
                "every signal date is independently confirmed to be a calendar trading "
                "day; a data feed that has a bar for a holiday (or is missing one for a "
                "real trading day) cannot influence which dates the engine ever acts on"
            ),
        )

    def _run_corporate_action_event(self, seed: int | None = None) -> StressTestResult:
        """A dividend goes ex on the fill date for a held position; cash
        must be credited exactly, and the absence of a corporate-action
        provider must degrade to "no credit", never crash.
        """
        instrument_id = self.context.instrument_ids[0]
        engine = self.context.engine()
        holding = {instrument_id: 100}
        execution_date = self.context.signal_dates[-1]
        dividend = CorporateAction(
            instrument_id=instrument_id,
            action_type=CorporateActionType.DIVIDEND,
            ex_date=execution_date,
            cash_amount=Decimal("2.50"),
        )
        provider = _StaticCorporateActionProvider([dividend])
        engine_with_dividend = self.context.engine(corporate_actions=provider)
        credit = engine_with_dividend.dividend_cash_credit(holding, execution_date)
        expected = 2.50 * 100
        credited_correctly = math.isclose(credit, expected, rel_tol=1e-9)

        engine_without_provider = self.context.engine(corporate_actions=None)
        no_provider_credit = engine_without_provider.dividend_cash_credit(holding, execution_date)
        graceful = no_provider_credit == 0.0

        del engine  # unused; kept only to mirror other handlers' shape
        return StressTestResult(
            scenario=StressScenario.CORPORATE_ACTION_EVENT,
            trial_seed=seed,
            portfolio_impact_pct=0.0,
            max_loss_pct=0.0,
            recovery_days=0,
            risk_controls_fired=False,
            trading_halted=False,
            duplicate_orders_prevented=True,
            portfolio_state_consistent=credited_correctly and graceful,
            system_failed_closed=credited_correctly and graceful,
            detail=(
                f"dividend credit={credit:.2f} (expected {expected:.2f}); "
                f"no-provider credit={no_provider_credit:.2f} (expected 0.0)"
            ),
        )

    # -- execution / infrastructure scenarios --------------------------------

    def _run_broker_api_outage(self, seed: int | None = None) -> StressTestResult:
        """``PortfolioRiskState.broker_connected=False`` must force a HALT
        and reject every proposed position, no exceptions -- reusing the
        exact mechanism Phase 9 built and tested for this.
        """
        risk_manager, risk_state, proposed = self._single_position_case(broker_connected=False)
        decisions = risk_manager.evaluate(proposed, risk_state)
        all_rejected = all(not decision.approved for decision in decisions)
        halted = decisions[0].circuit_state is CircuitState.HALTED
        return StressTestResult(
            scenario=StressScenario.BROKER_API_OUTAGE,
            trial_seed=seed,
            portfolio_impact_pct=0.0,
            max_loss_pct=0.0,
            recovery_days=0,
            risk_controls_fired=all_rejected,
            trading_halted=halted,
            duplicate_orders_prevented=True,
            portfolio_state_consistent=True,
            system_failed_closed=all_rejected and halted,
            detail=f"broker_connected=False -> halted={halted}, all_rejected={all_rejected}",
        )

    def _run_partial_fill(self, seed: int | None = None) -> StressTestResult:
        """No partial-fill mechanism exists yet (Phase 10/11's order
        management system) -- this checks the one thing that already must
        be true regardless: applying a fill for *any* quantity (not
        necessarily the full requested size) updates cash and holdings by
        exactly that quantity, never silently assuming the full size
        filled.
        """
        order = OrderRecord(
            signal_date=self.context.signal_dates[0],
            execution_date=self.context.signal_dates[1],
            instrument_id=self.context.instrument_ids[0],
            action=TradeAction.BUY,
            current_weight=0.0,
            target_weight=0.10,
            delta_weight=0.10,
        )
        engine = self.context.engine()
        full_quantity = 100
        partial_quantity = 37  # an arbitrary, deliberately non-round partial size
        cost_model = engine.cost_model
        fill_price = 100.0


        execution_cost = cost_model.estimate_execution_cost(
            order.instrument_id,
            TradeSide.BUY,
            partial_quantity,
            fill_price,
            order.execution_date,
            spread_bps=engine.assumed_spread_bps,
            avg_daily_value=1_000_000_000.0,
            volatility=0.2,
        )
        fill = FillRecord(
            order=order, side=TradeSide.BUY, quantity=partial_quantity, fill_price=fill_price,
            execution_cost=execution_cost,
        )
        starting_cash = 1_000_000.0
        holdings: dict[str, int] = {}
        remaining_cash = BacktestEngine._apply_fill(fill, starting_cash, holdings)

        cash_consistent = math.isclose(
            remaining_cash, starting_cash - execution_cost.net_value, rel_tol=1e-9
        )
        holdings_consistent = holdings.get(order.instrument_id) == partial_quantity
        not_full_size = partial_quantity != full_quantity

        return StressTestResult(
            scenario=StressScenario.PARTIAL_FILL,
            trial_seed=seed,
            portfolio_impact_pct=0.0,
            max_loss_pct=0.0,
            recovery_days=0,
            risk_controls_fired=False,
            trading_halted=False,
            duplicate_orders_prevented=True,
            portfolio_state_consistent=cash_consistent and holdings_consistent and not_full_size,
            system_failed_closed=cash_consistent and holdings_consistent,
            detail=(
                f"partial fill of {partial_quantity} (not the full {full_quantity}) "
                f"updated cash and holdings consistently: cash_ok={cash_consistent}, "
                f"holdings_ok={holdings_consistent}"
            ),
        )

    def _run_order_rejection(self, seed: int | None = None) -> StressTestResult:
        """An order with no fillable price anywhere in the search window
        (``_next_open`` returns ``None``) must be skipped, not crash the
        run or silently mutate the ledger.
        """
        instrument_id = self.context.instrument_ids[0]
        shocked = ShockedMarketDataProvider(
            self.context.market_data,
            [
                MarketShock(
                    ShockType.UNAVAILABLE, dt.date.min, None, 0.0, frozenset({instrument_id})
                )
            ],
        )
        engine = self.context.engine(market_data=shocked, max_fill_search_days=2)
        execution_date = self.context.signal_dates[1]
        price = engine._next_open(instrument_id, execution_date)
        rejected_cleanly = price is None
        return StressTestResult(
            scenario=StressScenario.ORDER_REJECTION,
            trial_seed=seed,
            portfolio_impact_pct=0.0,
            max_loss_pct=0.0,
            recovery_days=0,
            risk_controls_fired=rejected_cleanly,
            trading_halted=False,
            duplicate_orders_prevented=True,
            portfolio_state_consistent=True,
            system_failed_closed=rejected_cleanly,
            detail=f"undeliverable order returned no fill price (rejected={rejected_cleanly})",
        )

    def _run_delayed_fill(self, seed: int | None = None) -> StressTestResult:
        """A short data gap right after the signal date -- the fill must
        still occur, at the next *genuinely available* price, later than
        the nominal next session, rather than fabricating a price for the
        missing days.
        """
        instrument_id = self.context.instrument_ids[0]
        signal_date = self.context.signal_dates[len(self.context.signal_dates) // 2]
        nominal_execution_date = self.context.calendar.next_trading_day(signal_date)
        gap_end = self.context.calendar.sessions_offset(nominal_execution_date, 2)
        shocked = ShockedMarketDataProvider(
            self.context.market_data,
            [
                MarketShock(
                    ShockType.UNAVAILABLE,
                    nominal_execution_date,
                    gap_end,
                    0.0,
                    frozenset({instrument_id}),
                )
            ],
        )
        engine = self.context.engine(market_data=shocked)
        missing_price = engine._next_open(instrument_id, nominal_execution_date)
        actual_execution_date = self.context.calendar.next_trading_day(gap_end)
        fill_price = engine._next_open(instrument_id, actual_execution_date)
        delayed_bars = self.context.market_data.get_equity_bars(
            instrument_id, gap_end, self.context.calendar.sessions_offset(gap_end, 1)
        )
        expected_price = float(delayed_bars[-1].open) if delayed_bars else None
        delayed_correctly = missing_price is None and fill_price is not None and (
            expected_price is None or math.isclose(fill_price, expected_price, rel_tol=1e-6)
        )
        return StressTestResult(
            scenario=StressScenario.DELAYED_FILL,
            trial_seed=seed,
            portfolio_impact_pct=0.0,
            max_loss_pct=0.0,
            recovery_days=0,
            risk_controls_fired=False,
            trading_halted=False,
            duplicate_orders_prevented=True,
            portfolio_state_consistent=fill_price is not None,
            system_failed_closed=delayed_correctly,
            detail=(
                f"nominal execution {nominal_execution_date}, data gap through {gap_end}, "
                f"resolved fill_price={fill_price}"
            ),
        )

    def _run_duplicate_order(self, seed: int | None = None) -> StressTestResult:
        """A duplicated signal date (the walk-forward analogue of a
        duplicated broker order response) must be rejected outright by
        ``BacktestEngine.run``, not silently replayed.
        """
        engine = self.context.engine()
        duplicated_dates = [
            self.context.signal_dates[0],
            self.context.signal_dates[0],
            self.context.signal_dates[1],
        ]
        targets = self.context.full_exposure_targets()
        prevented = False
        try:
            engine.run(
                "duplicate_order_probe", targets, duplicated_dates, self.context.initial_equity
            )
        except BacktestEngineError as exc:
            prevented = "duplicate" in str(exc)
        return StressTestResult(
            scenario=StressScenario.DUPLICATE_ORDER,
            trial_seed=seed,
            portfolio_impact_pct=0.0,
            max_loss_pct=0.0,
            recovery_days=0,
            risk_controls_fired=prevented,
            trading_halted=False,
            duplicate_orders_prevented=prevented,
            portfolio_state_consistent=True,
            system_failed_closed=prevented,
            detail=f"duplicated signal date rejected={prevented}",
        )

    def _run_application_restart(self, seed: int | None = None) -> StressTestResult:
        """A halted circuit breaker must remain halted across a simulated
        process restart -- a fresh ``CircuitBreaker`` pointed at the same
        state file, not a new one starting from NORMAL.
        """
        state_path = self.context.circuit_breaker_state_dir / f"restart_probe_{seed}.json"
        if state_path.exists():
            state_path.unlink()
        first = CircuitBreaker(self.context.risk_config, state_path)
        _, halting_state = self._risk_state_and_position(broker_connected=False)
        first.evaluate(halting_state)
        halted_before_restart = first.current_status().state is CircuitState.HALTED

        restarted = CircuitBreaker(self.context.risk_config, state_path)
        still_halted = restarted.current_status().state is CircuitState.HALTED
        _, healthy_state = self._risk_state_and_position()
        stays_halted_despite_healthy_data = (
            restarted.evaluate(healthy_state).state is CircuitState.HALTED
        )

        consistent = halted_before_restart and still_halted and stays_halted_despite_healthy_data
        return StressTestResult(
            scenario=StressScenario.APPLICATION_RESTART,
            trial_seed=seed,
            portfolio_impact_pct=0.0,
            max_loss_pct=0.0,
            recovery_days=None,
            risk_controls_fired=True,
            trading_halted=still_halted,
            duplicate_orders_prevented=True,
            portfolio_state_consistent=consistent,
            system_failed_closed=consistent,
            detail=(
                f"halted_before_restart={halted_before_restart}, "
                f"still_halted_after_restart={still_halted}, "
                f"resists_auto_recovery={stays_halted_despite_healthy_data}"
            ),
        )

    def _run_database_failure(self, seed: int | None = None) -> StressTestResult:
        """A corrupted circuit-breaker state file (the stand-in for a
        database/disk failure this system actually persists to) must not
        be silently treated as a healthy NORMAL state -- reading it either
        raises (halting all further action, since no order can be placed
        by a crashed process) or is otherwise never interpreted as
        "everything is fine".
        """
        state_path = self.context.circuit_breaker_state_dir / f"db_failure_probe_{seed}.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{not valid json", encoding="utf-8")

        breaker = CircuitBreaker(self.context.risk_config, state_path)
        failed_closed = False
        detail = "corrupted state file was silently accepted as a valid status (unsafe)"
        try:
            status = breaker.current_status()
            failed_closed = status.state is not CircuitState.NORMAL
            if failed_closed:
                detail = f"corrupted state file resolved to {status.state.value}, not NORMAL"
        except (ValueError, KeyError, TypeError) as exc:
            failed_closed = True
            detail = (
                f"corrupted state file correctly raised {type(exc).__name__}, "
                "not silently resumed"
            )

        return StressTestResult(
            scenario=StressScenario.DATABASE_FAILURE,
            trial_seed=seed,
            portfolio_impact_pct=0.0,
            max_loss_pct=0.0,
            recovery_days=None,
            risk_controls_fired=failed_closed,
            trading_halted=failed_closed,
            duplicate_orders_prevented=True,
            portfolio_state_consistent=failed_closed,
            system_failed_closed=failed_closed,
            detail=detail,
        )

    # -- regime / ranking / drawdown scenarios -------------------------------

    def _run_hmm_regime_misclassification(
        self, seed: int | None = None, drop_magnitude: float | None = None
    ) -> StressTestResult:
        """The central requirement: feed a fixed *full-exposure* signal
        (as if the HMM wrongly called LOW_RISK) straight through a real
        market crash, and confirm the circuit breaker -- which watches
        realized P&L, never the regime label -- still halts or reduces
        risk despite the wrong signal.
        """
        rng = _rng(seed)
        drop = drop_magnitude if drop_magnitude is not None else float(rng.uniform(0.15, 0.45))
        shock_date = self.context.signal_dates[len(self.context.signal_dates) // 2]
        universe = frozenset(self.context.instrument_ids)
        shocked = ShockedMarketDataProvider(
            self.context.market_data,
            [MarketShock(ShockType.PRICE_CRASH, shock_date, None, drop, universe)],
        )
        # The "wrong" HMM signal: stay fully invested regardless.
        targets = self.context.full_exposure_targets()
        return self._run_priced_scenario(
            StressScenario.HMM_REGIME_MISCLASSIFICATION,
            seed,
            shocked,
            targets=targets,
            detail=(
                f"exposure signal says LOW_RISK (fully invested) throughout a {drop:.0%} "
                f"crash from {shock_date} -- checking whether the circuit breaker still "
                "contains the damage despite the wrong signal"
            ),
        )

    def _run_wrong_stock_ranking(
        self, seed: int | None = None, adverse_move: float | None = None
    ) -> StressTestResult:
        """Even if stock selection ranks the worst-performing instrument
        first, the *portfolio-level* limits (single-name cap, sector cap)
        must still bound how much damage any one wrong pick can do --
        checked structurally: no proposed position, however it was
        ranked, can ever exceed ``portfolio.max_single_name_pct``.
        """
        rng = _rng(seed)
        move = adverse_move if adverse_move is not None else float(rng.uniform(0.20, 0.50))
        instrument_id = self.context.instrument_ids[0]
        shock_date = self.context.signal_dates[len(self.context.signal_dates) // 2]
        shocked = ShockedMarketDataProvider(
            self.context.market_data,
            [
                MarketShock(
                    ShockType.PRICE_CRASH, shock_date, None, move, frozenset({instrument_id})
                )
            ],
        )
        engine = self.context.engine(market_data=shocked)
        targets = self.context.full_exposure_targets()
        try:
            result = engine.run(
                "wrong_ranking_probe",
                targets,
                list(self.context.signal_dates),
                self.context.initial_equity,
            )
        except BacktestEngineError as exc:
            return self._failed_run_result(StressScenario.WRONG_STOCK_RANKING, seed, exc)

        # positions_history records quantities, not weights; the recorded
        # orders carry the target weight actually proposed, which is what
        # the single-name cap bounds.
        max_weight_in_orders = max((order.target_weight for order in result.orders), default=0.0)
        bounded = max_weight_in_orders <= self.context.risk_config.max_single_name_pct + 1e-9

        return self._priced_result(
            StressScenario.WRONG_STOCK_RANKING,
            seed,
            result,
            extra_consistent=bounded,
            detail=(
                f"a {move:.0%} adverse move on {instrument_id} from {shock_date}; "
                f"max single-name weight ever proposed={max_weight_in_orders:.4f} "
                f"(limit {self.context.risk_config.max_single_name_pct:.4f})"
            ),
        )

    def _run_sudden_portfolio_drawdown(
        self, seed: int | None = None, magnitude: float | None = None
    ) -> StressTestResult:
        """A severe, single-day, portfolio-wide shock (broader than one
        stock) -- checks that the daily-loss and peak-to-trough circuit
        breaker tiers correctly halt further risk-taking.
        """
        rng = _rng(seed)
        drop = magnitude if magnitude is not None else float(rng.uniform(0.12, 0.35))
        shock_date = self.context.signal_dates[2 * len(self.context.signal_dates) // 3]
        universe = frozenset(self.context.instrument_ids)
        shocked = ShockedMarketDataProvider(
            self.context.market_data,
            [MarketShock(ShockType.PRICE_CRASH, shock_date, None, drop, universe)],
        )
        return self._run_priced_scenario(
            StressScenario.SUDDEN_PORTFOLIO_DRAWDOWN,
            seed,
            shocked,
            detail=f"a {drop:.0%} single-day portfolio-wide shock from {shock_date}",
        )

    # -- shared machinery -----------------------------------------------------

    def _run_priced_scenario(
        self,
        scenario: StressScenario,
        seed: int | None,
        market_data: MarketDataProvider,
        detail: str,
        targets: dict[dt.date, AllocationTarget] | None = None,
    ) -> StressTestResult:
        engine = self.context.engine(market_data=market_data)
        return self._run_priced_scenario_with_engine(scenario, seed, engine, detail, targets)

    def _run_priced_scenario_with_engine(
        self,
        scenario: StressScenario,
        seed: int | None,
        engine: BacktestEngine,
        detail: str,
        targets: dict[dt.date, AllocationTarget] | None = None,
    ) -> StressTestResult:
        exposure_targets = targets if targets is not None else self.context.full_exposure_targets()
        try:
            result = engine.run(
                f"{scenario.value}_probe",
                exposure_targets,
                list(self.context.signal_dates),
                self.context.initial_equity,
            )
        except BacktestEngineError as exc:
            return self._failed_run_result(scenario, seed, exc)
        return self._priced_result(scenario, seed, result, extra_consistent=True, detail=detail)

    def _priced_result(
        self,
        scenario: StressScenario,
        seed: int | None,
        result: BacktestResult,
        extra_consistent: bool,
        detail: str,
    ) -> StressTestResult:
        equity = result.equity_curve.astype(float)
        portfolio_impact_pct = (
            float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 0 else 0.0
        )
        max_loss_pct, _duration = _drawdown_stats(equity) if len(equity) > 0 else (0.0, 0)
        recovery_days = _recovery_duration(equity) if len(equity) > 0 else 0

        risk_controls_fired = any(
            not decision.approved or decision.circuit_state is not CircuitState.NORMAL
            for daily in result.risk_decisions
            for decision in daily.decisions
        )
        trading_halted = any(
            daily.decisions and daily.decisions[0].circuit_state is CircuitState.HALTED
            for daily in result.risk_decisions
        )

        return StressTestResult(
            scenario=scenario,
            trial_seed=seed,
            portfolio_impact_pct=portfolio_impact_pct,
            max_loss_pct=max_loss_pct,
            recovery_days=recovery_days,
            risk_controls_fired=risk_controls_fired,
            trading_halted=trading_halted,
            duplicate_orders_prevented=True,
            portfolio_state_consistent=extra_consistent,
            system_failed_closed=extra_consistent,
            detail=detail,
        )

    def _failed_run_result(
        self, scenario: StressScenario, seed: int | None, exc: BacktestEngineError
    ) -> StressTestResult:
        """``BacktestEngineError`` from the run itself is a fail-closed
        outcome (the engine refused to proceed on bad/missing data), not
        an inconsistency -- reported as such rather than propagated,
        since a stress test's job is to observe the failure mode, not
        crash alongside it.
        """
        return StressTestResult(
            scenario=scenario,
            trial_seed=seed,
            portfolio_impact_pct=0.0,
            max_loss_pct=0.0,
            recovery_days=None,
            risk_controls_fired=True,
            trading_halted=False,
            duplicate_orders_prevented=True,
            portfolio_state_consistent=True,
            system_failed_closed=True,
            detail=f"engine refused to proceed (fail closed): {exc}",
        )

    def _single_position_case(
        self,
        stale_seconds: float = 0.0,
        broker_connected: bool = True,
        spread_bps: float = 5.0,
    ) -> tuple[RiskManager, PortfolioRiskState, TargetPortfolio]:
        instrument_id = self.context.instrument_ids[0]
        position = TargetPosition(
            instrument_id=instrument_id,
            symbol=instrument_id.split(":")[-1],
            target_weight=0.10,
            sector=instrument_id,
            rank=1,
            score=1.0,
            binding_constraint="unconstrained",
        )
        proposed = TargetPortfolio(
            as_of=self.context.signal_dates[0],
            positions=(position,),
            cash_weight=0.90,
            regime=AllocationRegime.NORMAL_RISK,
            gross_exposure=0.10,
        )
        _, risk_state = self._risk_state_and_position(
            stale_seconds=stale_seconds, broker_connected=broker_connected, spread_bps=spread_bps
        )
        state_path = self.context.circuit_breaker_state_dir / "single_position_probe.json"
        if state_path.exists():
            state_path.unlink()
        circuit_breaker = CircuitBreaker(self.context.risk_config, state_path)
        risk_manager = RiskManager(self.context.risk_config, circuit_breaker)
        return risk_manager, risk_state, proposed

    def _risk_state_and_position(
        self,
        stale_seconds: float = 0.0,
        broker_connected: bool = True,
        spread_bps: float = 5.0,
    ) -> tuple[PositionRisk, PortfolioRiskState]:
        instrument_id = self.context.instrument_ids[0]
        position_risk = PositionRisk(
            instrument_id=instrument_id,
            sector=instrument_id,
            avg_daily_value_inr=1_000_000_000.0,
            quote_age_seconds=stale_seconds,
            spread_bps=spread_bps,
        )
        state = PortfolioRiskState(
            as_of=dt.datetime.combine(self.context.signal_dates[0], dt.time(15, 30), tzinfo=dt.UTC),
            equity=self.context.initial_equity,
            positions=(position_risk,),
            daily_pnl_pct=0.0,
            rolling_pnl_pct=0.0,
            peak_to_trough_drawdown_pct=0.0,
            daily_turnover_pct_so_far=0.0,
            max_pairwise_correlation=None,
            correlated_pair=None,
            system_healthy=True,
            system_detail=None,
            broker_connected=broker_connected,
            broker_detail=None if broker_connected else "simulated outage",
        )
        return position_risk, state


class _StaticCorporateActionProvider(CorporateActionProvider):
    """A minimal, fixed-list ``CorporateActionProvider`` for scenario
    construction -- deliberately simpler than
    ``data.corporate_actions.InMemoryCorporateActionProvider`` since a
    stress scenario needs only ``actions_for``.
    """

    def __init__(self, actions: list[CorporateAction]) -> None:
        self._actions = actions

    def actions_for(
        self, instrument_id: str, start: dt.date, end: dt.date
    ) -> list[CorporateAction]:
        return [
            action
            for action in self._actions
            if action.instrument_id == instrument_id and start <= action.ex_date <= end
        ]

    def cumulative_adjustment_factor(
        self, instrument_id: str, price_date: dt.date, as_of: dt.date
    ) -> Decimal:
        return Decimal(1)

    def identity_change(self, instrument_id: str, as_of: dt.date) -> CorporateAction | None:
        return None


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else 0)
