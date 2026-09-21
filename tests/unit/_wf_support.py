"""Shared, non-collected test support for the Phase 11 walk-forward suite:
a synthetic multi-year market, an in-memory data provider, config
factories with small/fast values, and an ``Environment`` bundling
everything into ready-to-use ``BacktestEngine``/``WalkForwardValidator``
instances.

Not a test module itself (no ``test_`` prefix) -- imported by
``test_backtest_engine.py``, ``test_walk_forward.py``, and
``test_no_lookahead_walkforward.py``.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import numpy as np

from backtest.cost_schedule import CostSchedule, CostScheduleRepository
from backtest.costs import CostModel
from backtest.engine import BacktestEngine
from backtest.walk_forward import WalkForwardValidator
from config.models import (
    AllocationConfig,
    BacktestConfig,
    ExecutionConfig,
    FeaturesConfig,
    HMMConfig,
    PortfolioConfig,
    RegimePolicyConfig,
    RiskConfig,
    SelectionConfig,
    UniverseConfig,
)
from core.features.feature_engineering import FeaturePipeline, build_default_feature_definitions
from core.regime.regime_policy import RegimePolicy
from data.calendar import NSETradingCalendar
from data.corporate_actions import InMemoryCorporateActionProvider
from data.errors import DataNotAvailableError
from data.instrument_master import InMemoryInstrumentRepository
from data.interfaces import MarketDataProvider
from data.membership import InMemoryIndexMembershipProvider
from data.models import (
    CorporateAction,
    CorporateActionType,
    DailyBar,
    Exchange,
    IndexMembership,
    IndexObservation,
    Instrument,
    InstrumentStatus,
    PriceBasis,
    Quote,
    Segment,
)
from portfolio.portfolio_constructor import PortfolioConstructor
from universe.stock_selector import StockSelector
from universe.universe import UniverseProvider

INDEX_SYMBOL = "NIFTY50"
VIX_SYMBOL = "INDIAVIX"


# --------------------------------------------------------------------------
# In-memory market data
# --------------------------------------------------------------------------


class FakeMarketDataProvider(MarketDataProvider):
    """Mirrors ``tests/unit/test_stock_selector.py``'s test double: an
    instrument never added behaves like a missing data file
    (``DataNotAvailableError``); adjustment is not modeled (Phase 2-3's
    tested responsibility, not this suite's).
    """

    def __init__(self) -> None:
        self._bars: dict[str, list[DailyBar]] = {}
        self._index: dict[str, list[IndexObservation]] = {}

    def add_equity(
        self,
        instrument_id: str,
        dates: list[dt.date],
        closes: list[float],
        *,
        volume: int = 5_000_000,
    ) -> None:
        self._bars[instrument_id] = [
            DailyBar(
                instrument_id=instrument_id,
                session_date=day,
                open=Decimal(str(round(close, 4))),
                high=Decimal(str(round(close * 1.01, 4))),
                low=Decimal(str(round(close * 0.99, 4))),
                close=Decimal(str(round(close, 4))),
                volume=volume,
            )
            for day, close in zip(dates, closes, strict=True)
        ]

    def add_index(
        self,
        index_symbol: str,
        dates: list[dt.date],
        closes: list[float],
        *,
        with_ohlcv: bool = False,
        seed: int = 11,
    ) -> None:
        """``with_ohlcv=True`` also synthesizes high/low/volume around each
        close -- needed for features (ATR, volume stress) that read them;
        ``IndexObservation`` otherwise leaves them ``None`` (a real vendor
        gap this dataclass models on purpose -- see its docstring). Volume
        varies (not a constant) so the volume-stress feature's rolling
        z-score has nonzero variance to divide by."""
        rng = np.random.default_rng(seed)
        volumes = rng.integers(1_500_000_000, 2_500_000_000, size=len(dates))
        self._index[index_symbol] = [
            IndexObservation(
                index_symbol,
                day,
                close=Decimal(str(round(close, 4))),
                high=Decimal(str(round(close * 1.006, 4))) if with_ohlcv else None,
                low=Decimal(str(round(close * 0.994, 4))) if with_ohlcv else None,
                volume=int(volumes[i]) if with_ohlcv else None,
            )
            for i, (day, close) in enumerate(zip(dates, closes, strict=True))
        ]

    def get_equity_bars(
        self,
        instrument_id: str,
        start: dt.date,
        end: dt.date,
        price_basis: PriceBasis = PriceBasis.RAW,
    ) -> list[DailyBar]:
        if instrument_id not in self._bars:
            raise DataNotAvailableError(f"no data for {instrument_id}")
        return [bar for bar in self._bars[instrument_id] if start <= bar.session_date <= end]

    def get_index_observations(
        self, index_symbol: str, start: dt.date, end: dt.date
    ) -> list[IndexObservation]:
        return [
            observation
            for observation in self._index.get(index_symbol, [])
            if start <= observation.session_date <= end
        ]

    def available_range(self, instrument_id: str) -> tuple[dt.date, dt.date] | None:
        bars = self._bars.get(instrument_id)
        if not bars:
            return None
        return bars[0].session_date, bars[-1].session_date

    def get_quote(self, instrument_id: str) -> Quote:
        raise DataNotAvailableError("fake provider has no live quotes")


# --------------------------------------------------------------------------
# Synthetic market generation
# --------------------------------------------------------------------------


def business_days(start: dt.date, n: int) -> list[dt.date]:
    dates: list[dt.date] = []
    current = start
    while len(dates) < n:
        if current.weekday() < 5:
            dates.append(current)
        current += dt.timedelta(days=1)
    return dates


def synthetic_index_market(
    start: dt.date, n_days: int, seed: int = 5, regime_block: int = 20
) -> tuple[list[dt.date], list[float], list[float]]:
    """A NIFTY/VIX pair with an alternating calm/volatile block structure --
    the same style ``tests/unit/test_hmm_engine.py``'s ``market()`` helper
    uses, extended to price *levels* (not just returns), since the trend
    baseline and the feature pipeline both need real levels.
    """
    # Two independent RNGs, not one shared sequentially: drawing `returns`
    # (size n_days) before `vix`'s noise term from one shared generator
    # would make vix's stream position depend on n_days, breaking prefix
    # determinism between differently-truncated environments built from
    # the same seed (tests/unit/test_no_lookahead_walkforward.py's
    # premise) even though nothing about *this* series is actually
    # different for the overlapping dates.
    returns_rng = np.random.default_rng(seed)
    vix_rng = np.random.default_rng(seed + 1)
    dates = business_days(start, n_days)
    calm = (np.arange(n_days) // regime_block) % 2 == 0
    daily_vol = np.where(calm, 0.004, 0.020)
    returns = returns_rng.normal(0.0002, daily_vol)
    nifty = 10_000.0 * np.cumprod(1.0 + returns)
    vix = np.where(calm, 12.0, 30.0) + vix_rng.normal(0.0, 1.0, n_days)
    vix = np.clip(vix, 8.0, 65.0)
    return dates, nifty.tolist(), vix.tolist()


def synthetic_stock_prices(
    dates: list[dt.date], start_price: float, daily_return: float, daily_vol: float, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(daily_return, daily_vol, len(dates))
    prices: list[float] = (start_price * np.cumprod(1.0 + returns)).tolist()
    return prices


# --------------------------------------------------------------------------
# Config factories -- small, fast values; every test overrides what it cares about
# --------------------------------------------------------------------------


def features_config(**overrides: object) -> FeaturesConfig:
    defaults: dict[str, object] = {
        "realized_vol_window": 10,
        "vol_ratio_short_window": 5,
        "vol_ratio_long_window": 10,
        "vix_zscore_window": 10,
        "vix_change_window": 1,
        "trend_window": 10,
        "drawdown_window": 10,
        "atr_window": 10,
        "volume_stress_window": 10,
    }
    defaults.update(overrides)
    return FeaturesConfig.model_validate(defaults)


def hmm_config(**overrides: object) -> HMMConfig:
    defaults: dict[str, object] = {
        "candidate_states": [2, 3],
        "covariance_type": "diag",
        "training_window_days": 252,
        "retrain_interval_sessions": 20,
        "min_confidence": 0.5,
        "confirmation_bars": 2,
        "flicker_window_sessions": 10,
        "max_covariance_condition_number": 1_000_000.0,
        "random_seeds": [17, 29],
        "max_iterations": 100,
        "convergence_tolerance": 0.0001,
        "covariance_regularization": 0.000001,
        "min_state_occupancy": 0.02,
    }
    defaults.update(overrides)
    return HMMConfig.model_validate(defaults)


def allocation_config(**overrides: object) -> AllocationConfig:
    defaults: dict[str, object] = {
        "low_risk_volatility_threshold": 0.15,
        "high_risk_volatility_threshold": 0.30,
        "extreme_confidence_threshold": 0.90,
        "max_flicker_transitions": 3,
        "baseline_volatility_window": 10,
        "trend_ma_window_days": 20,
    }
    defaults.update(overrides)
    return AllocationConfig.model_validate(defaults)


def regime_policy_config(**overrides: object) -> RegimePolicyConfig:
    defaults: dict[str, object] = {
        "low_risk": {"min_gross_exposure": 0.60, "max_gross_exposure": 1.00},
        "normal_risk": {"min_gross_exposure": 0.30, "max_gross_exposure": 0.60},
        "high_risk": {"min_gross_exposure": 0.00, "max_gross_exposure": 0.30},
        "uncertain": {"min_gross_exposure": 0.00, "max_gross_exposure": 0.20},
    }
    defaults.update(overrides)
    return RegimePolicyConfig.model_validate(defaults)


def universe_config(**overrides: object) -> UniverseConfig:
    defaults: dict[str, object] = {
        "index_reference": INDEX_SYMBOL,
        "min_avg_daily_value_inr": 1_000.0,
        "point_in_time": True,
        "exclude_illiquid": True,
    }
    defaults.update(overrides)
    return UniverseConfig.model_validate(defaults)


def selection_config(**overrides: object) -> SelectionConfig:
    defaults: dict[str, object] = {
        "min_holdings": 2,
        "max_holdings": 4,
        "momentum_lookback_months": [1, 2],
        "momentum_skip_days": 2,
        "trend_ma_window_days": 5,
        "trend_window_days": 10,
        "relative_strength_window_days": 15,
        "volatility_window_days": 10,
        "drawdown_window_days": 15,
        "liquidity_window_days": 5,
        "min_history_days": 47,
        "factor_weights": {
            "momentum": 0.25,
            "trend_persistence": 0.15,
            "relative_strength": 0.2,
            "volatility": 0.15,
            "drawdown": 0.1,
            "liquidity": 0.15,
        },
    }
    defaults.update(overrides)
    return SelectionConfig.model_validate(defaults)


def portfolio_config(**overrides: object) -> PortfolioConfig:
    defaults: dict[str, object] = {
        "max_single_name_pct": 0.40,
        "max_sector_pct": 1.0,
        "min_position_weight_pct": 0.02,
        "max_pairwise_correlation": 0.98,
        "correlation_lookback_days": 20,
        "correlation_penalty_pct": 0.5,
        "min_correlation_observations": 5,
    }
    defaults.update(overrides)
    return PortfolioConfig.model_validate(defaults)


def execution_config(**overrides: object) -> ExecutionConfig:
    defaults: dict[str, object] = {
        "mode": "paper",
        "order_type": "limit",
        "order_price_guard_bps": 50,
        "stale_quote_seconds": 15,
        "max_participation_adv_pct": 0.5,
        "no_market_order_fallback": True,
    }
    defaults.update(overrides)
    return ExecutionConfig.model_validate(defaults)


def risk_config(**overrides: object) -> RiskConfig:
    defaults: dict[str, object] = {
        "max_gross_exposure": 1.0,
        "max_leverage": 1.0,
        "max_single_name_pct": 0.5,
        "max_sector_pct": 1.0,
        "max_concurrent_positions": 10,
        "max_risk_per_position_pct": 0.05,
        "daily_loss_warning_pct": 0.05,
        "daily_loss_reduce_pct": 0.08,
        "daily_loss_halt_pct": 0.15,
        "rolling_loss_reduce_pct": 0.10,
        "rolling_loss_halt_pct": 0.20,
        "peak_to_trough_drawdown_halt_pct": 0.30,
        "stale_data_max_minutes": 60,
        "max_pairwise_correlation": 0.99,
        "max_adv_participation_pct": 0.5,
        "max_spread_bps": 300.0,
        "max_daily_turnover_pct": 2.0,
        "reduced_risk_exposure_multiplier": 0.5,
        # Per-position stops are OFF in this fixture so every expectation
        # below reads as it did before risk/stop_loss.py existed -- a stop
        # firing mid-run would change trade counts and returns for reasons
        # that have nothing to do with what these tests assert. Tests that
        # want the stops enable them explicitly.
        "stop_loss": {
            "enabled": False,
            "hard_stop_pct": 0.03,
            "trail_drop_pct": 0.02,
            "trail_arm_net_profit_pct": 0.03,
        },
    }
    defaults.update(overrides)
    return RiskConfig.model_validate(defaults)


def backtest_config(**overrides: object) -> BacktestConfig:
    defaults: dict[str, object] = {
        "training_window_sessions": 80,
        "test_window_sessions": 15,
        "roll_step_sessions": 15,
        "include_indian_costs": True,
        "include_slippage": True,
        "point_in_time_universe": True,
        "slippage_min_bps": 2.0,
        "slippage_impact_coefficient": 10.0,
    }
    defaults.update(overrides)
    return BacktestConfig.model_validate(defaults)


def cost_schedule(**overrides: object) -> CostSchedule:
    defaults: dict[str, object] = dict(
        effective_from=dt.date(2010, 1, 1),
        label="test schedule",
        source="unit test fixture",
        brokerage_flat_inr=0.0,
        brokerage_pct=0.0,
        stt_buy_pct=0.001,
        stt_sell_pct=0.001,
        exchange_txn_pct=0.0000345,
        sebi_turnover_pct=0.0000010,
        gst_pct=0.18,
        stamp_duty_buy_pct=0.00015,
        stamp_duty_sell_pct=0.0,
        dp_charges_inr=15.93,
        other_charges_flat_inr=0.0,
        other_charges_pct=0.0,
    )
    defaults.update(overrides)
    return CostSchedule(**defaults)  # type: ignore[arg-type]


def instrument(
    instrument_id: str, *, effective_from: dt.date, symbol: str | None = None
) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        symbol=symbol or instrument_id.split(":")[-1],
        exchange=Exchange.NSE,
        segment=Segment.EQUITY,
        tick_size=Decimal("0.05"),
        price_precision=2,
        effective_from=effective_from,
        status=InstrumentStatus.ACTIVE,
    )


def membership(instrument_id: str, *, effective_from: dt.date) -> IndexMembership:
    return IndexMembership(INDEX_SYMBOL, instrument_id, effective_from, None)


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


class Environment:
    """Bundles a synthetic multi-year market, a point-in-time universe, and
    every config needed to build a ``BacktestEngine``/``WalkForwardValidator``.
    """

    def __init__(
        self,
        n_days: int = 260,
        start: dt.date = dt.date(2015, 1, 1),
        n_stocks: int = 6,
        seed: int = 5,
        regime_block: int = 20,
        dividends: list[CorporateAction] | None = None,
    ) -> None:
        self.start = start
        self.dates, nifty_closes, vix_closes = synthetic_index_market(
            start, n_days, seed=seed, regime_block=regime_block
        )
        self.calendar = NSETradingCalendar(
            holidays={}, covered_years=frozenset(range(start.year - 1, start.year + 10))
        )

        self.market_data = FakeMarketDataProvider()
        self.market_data.add_index(INDEX_SYMBOL, self.dates, nifty_closes, with_ohlcv=True)
        self.market_data.add_index(VIX_SYMBOL, self.dates, vix_closes)

        self.instrument_ids = [f"NSE:S{i:02d}" for i in range(n_stocks)]
        instrument_effective_from = start - dt.timedelta(days=730)
        instruments = [
            instrument(instrument_id, effective_from=instrument_effective_from)
            for instrument_id in self.instrument_ids
        ]
        memberships = [
            membership(instrument_id, effective_from=instrument_effective_from)
            for instrument_id in self.instrument_ids
        ]
        for index, instrument_id in enumerate(self.instrument_ids):
            prices = synthetic_stock_prices(
                self.dates,
                start_price=100.0 + index * 15.0,
                daily_return=0.0002 + index * 0.00015,
                daily_vol=0.012 + index * 0.002,
                seed=100 + index,
            )
            self.market_data.add_equity(instrument_id, self.dates, prices)

        self.universe_provider = UniverseProvider(
            INDEX_SYMBOL,
            InMemoryIndexMembershipProvider(memberships),
            InMemoryInstrumentRepository(instruments),
        )

        self.features_cfg = features_config()
        self.hmm_cfg = hmm_config()
        self.allocation_cfg = allocation_config()
        self.regime_policy_cfg = regime_policy_config()
        self.universe_cfg = universe_config()
        self.selection_cfg = selection_config()
        self.portfolio_cfg = portfolio_config()
        self.execution_cfg = execution_config()
        self.risk_cfg = risk_config()
        self.backtest_cfg = backtest_config()

        self.regime_policy = RegimePolicy(self.regime_policy_cfg)
        self.feature_pipeline = FeaturePipeline(
            build_default_feature_definitions(self.features_cfg)
        )
        self.stock_selector = StockSelector(
            self.selection_cfg, self.universe_cfg, self.universe_provider, self.market_data
        )
        self.portfolio_constructor = PortfolioConstructor(
            self.portfolio_cfg, self.selection_cfg, self.execution_cfg, self.market_data
        )
        self.cost_model = CostModel(
            CostScheduleRepository([cost_schedule()]),
            min_slippage_bps=self.backtest_cfg.slippage_min_bps,
            impact_coefficient=self.backtest_cfg.slippage_impact_coefficient,
        )
        self.corporate_actions = InMemoryCorporateActionProvider(dividends or [])

    def engine(self, tmp_path: Path, **overrides: object) -> BacktestEngine:
        kwargs: dict[str, object] = dict(
            calendar=self.calendar,
            market_data=self.market_data,
            stock_selector=self.stock_selector,
            portfolio_constructor=self.portfolio_constructor,
            risk_config=self.risk_cfg,
            cost_model=self.cost_model,
            circuit_breaker_state_dir=tmp_path,
            corporate_actions=self.corporate_actions,
            assumed_spread_bps=10.0,
        )
        kwargs.update(overrides)
        return BacktestEngine(**kwargs)  # type: ignore[arg-type]

    def validator(self, tmp_path: Path, **overrides: object) -> WalkForwardValidator:
        kwargs: dict[str, object] = dict(
            config=self.backtest_cfg,
            calendar=self.calendar,
            market_data=self.market_data,
            stock_selector=self.stock_selector,
            portfolio_constructor=self.portfolio_constructor,
            risk_config=self.risk_cfg,
            cost_model=self.cost_model,
            circuit_breaker_state_dir=tmp_path,
            hmm_config=self.hmm_cfg,
            allocation_config=self.allocation_cfg,
            regime_policy=self.regime_policy,
            feature_pipeline=self.feature_pipeline,
            index_symbol=INDEX_SYMBOL,
            vix_symbol=VIX_SYMBOL,
            corporate_actions=self.corporate_actions,
            initial_equity=10_000_000.0,
            random_seed=7,
            feature_warmup_buffer_days=40,
        )
        kwargs.update(overrides)
        return WalkForwardValidator(**kwargs)  # type: ignore[arg-type]


def dividend(
    instrument_id: str, ex_date: dt.date, cash_amount: float
) -> CorporateAction:
    return CorporateAction(
        instrument_id=instrument_id,
        action_type=CorporateActionType.DIVIDEND,
        ex_date=ex_date,
        cash_amount=Decimal(str(cash_amount)),
    )
