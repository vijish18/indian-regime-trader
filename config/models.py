"""Typed configuration models for the Indian Market Regime Trading System.

These mirror the structure validated by ``config/settings.schema.yaml``. The
schema catches malformed YAML at load time; the cross-field validators here
catch combinations that are individually valid but jointly inconsistent with
the system's own architectural constraints (see docs/ARCHITECTURE.md,
"Resolved specification ambiguities").
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field, model_validator

Percent = float  # semantic alias for a 0..1 fraction; enforced per-field below


def _parse_time(field_name: str, value: str) -> dt.time:
    try:
        return dt.time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} {value!r} is not an HH:MM time") from exc


class SessionTimesConfig(BaseModel):
    """Exchange session boundaries as ``HH:MM`` in the market timezone."""

    model_config = {"frozen": True}

    pre_open_start: str
    pre_open_end: str
    regular_open: str
    regular_close: str

    @model_validator(mode="after")
    def _times_are_ordered(self) -> SessionTimesConfig:
        times = [
            _parse_time(name, getattr(self, name))
            for name in ("pre_open_start", "pre_open_end", "regular_open", "regular_close")
        ]
        if not times[0] < times[1] <= times[2] < times[3]:
            raise ValueError(
                "session times must satisfy pre_open_start < pre_open_end <= "
                "regular_open < regular_close"
            )
        return self

    def as_times(self) -> dict[str, dt.time]:
        """Parsed ``datetime.time`` values, keyed by field name."""
        return {
            name: _parse_time(name, getattr(self, name))
            for name in ("pre_open_start", "pre_open_end", "regular_open", "regular_close")
        }


class MarketConfig(BaseModel):
    model_config = {"frozen": True}

    exchange: str
    timezone: str
    bar_timeframe: str
    execution_delay_sessions: int = Field(ge=1)
    sessions: SessionTimesConfig


class DataConfig(BaseModel):
    model_config = {"frozen": True}

    raw_data_path: str
    normalized_data_path: str
    reference_data_path: str
    quarantine_data_path: str
    calendar_file: str
    storage_format: str
    instrument_master_max_age_days: int = Field(ge=1)
    market_data_max_staleness_minutes: int = Field(ge=1)
    stale_price_run_sessions: int = Field(ge=2)
    volume_spike_multiple: float = Field(gt=1)


class FeaturesConfig(BaseModel):
    """Rolling-window sizes for the market-regime feature set
    (core/features/feature_engineering.py). Deliberately few knobs: this is a
    small, documented feature set by design, not a parameter grid.
    """

    model_config = {"frozen": True}

    realized_vol_window: int = Field(ge=2)
    vol_ratio_short_window: int = Field(ge=2)
    vol_ratio_long_window: int = Field(ge=2)
    vix_zscore_window: int = Field(ge=2)
    vix_change_window: int = Field(ge=1)
    trend_window: int = Field(ge=2)
    drawdown_window: int = Field(ge=2)
    atr_window: int = Field(ge=2)
    volume_stress_window: int = Field(ge=2)

    @model_validator(mode="after")
    def _vol_ratio_windows_are_ordered(self) -> FeaturesConfig:
        if self.vol_ratio_short_window >= self.vol_ratio_long_window:
            raise ValueError(
                "vol_ratio_short_window must be < vol_ratio_long_window "
                "(a volatility 'ratio' needs two different horizons)"
            )
        return self


class HMMConfig(BaseModel):
    model_config = {"frozen": True}

    candidate_states: list[int]
    covariance_type: str
    training_window_days: int = Field(ge=252)
    retrain_interval_sessions: int = Field(ge=1)
    min_confidence: Percent = Field(ge=0, le=1)
    confirmation_bars: int = Field(ge=1)
    flicker_window_sessions: int = Field(ge=1)
    max_covariance_condition_number: float = Field(gt=0)
    random_seeds: list[int]
    max_iterations: int = Field(ge=1)
    convergence_tolerance: float = Field(gt=0)
    covariance_regularization: float = Field(gt=0)
    min_state_occupancy: Percent = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _selection_inputs_are_usable(self) -> HMMConfig:
        if not self.candidate_states:
            raise ValueError("candidate_states must list at least one state count")
        if min(self.candidate_states) < 2:
            raise ValueError("candidate_states must all be >= 2; a 1-state HMM has no regimes")
        if not self.random_seeds:
            raise ValueError(
                "random_seeds must list at least one seed; multiple seeds are how "
                "docs/SPECIFICATION.md section 6.3's stability check is performed"
            )
        if len(set(self.random_seeds)) != len(self.random_seeds):
            raise ValueError(
                "random_seeds must be distinct; a repeated seed refits the same model"
            )
        if self.covariance_type not in ("diag", "full"):
            raise ValueError(
                f"covariance_type must be 'diag' or 'full', got {self.covariance_type!r}"
            )
        return self


class ExposureBand(BaseModel):
    model_config = {"frozen": True}

    min_gross_exposure: Percent = Field(ge=0, le=1)
    max_gross_exposure: Percent = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _min_not_above_max(self) -> ExposureBand:
        if self.min_gross_exposure > self.max_gross_exposure:
            raise ValueError("min_gross_exposure must be <= max_gross_exposure")
        return self


class RegimePolicyConfig(BaseModel):
    """Gross-exposure bands per allocation tier
    (core/regime/allocation.py::AllocationRegime). Named by risk tier
    (low_risk/normal_risk/high_risk/uncertain), not by the HMM's own
    calm/normal/elevated/crisis regime labels -- see docs/ARCHITECTURE.md,
    "Resolved specification ambiguities" (B6), for why these are two
    deliberately separate vocabularies.
    """

    model_config = {"frozen": True}

    low_risk: ExposureBand
    normal_risk: ExposureBand
    high_risk: ExposureBand
    uncertain: ExposureBand


class AllocationConfig(BaseModel):
    """Parameters for core/regime/allocation.py's volatility-tier
    classification, regime-confirmation, and flicker handling.
    """

    model_config = {"frozen": True}

    low_risk_volatility_threshold: float = Field(gt=0)
    """Annualized expected/realized volatility strictly below this is
    LOW_RISK."""

    high_risk_volatility_threshold: float = Field(gt=0)
    """Annualized expected/realized volatility at or above this is
    HIGH_RISK; between the two thresholds is NORMAL_RISK."""

    extreme_confidence_threshold: Percent = Field(gt=0, le=1)
    """A regime observed at or above this confidence confirms immediately
    (1 bar) rather than waiting for ``hmm.confirmation_bars`` consecutive
    observations (docs/SPECIFICATION.md section 6, "unless confidence is
    extreme")."""

    max_flicker_transitions: int = Field(ge=0)
    """Confirmed-regime changes tolerated within ``hmm.flicker_window_sessions``
    before the allocation is forced to UNCERTAIN regardless of the latest
    call."""

    baseline_volatility_window: int = Field(ge=2)
    """Trailing sessions used by the non-HMM rolling-volatility baseline
    (core/regime/baseline_policy.py)."""

    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> AllocationConfig:
        if self.low_risk_volatility_threshold >= self.high_risk_volatility_threshold:
            raise ValueError(
                "low_risk_volatility_threshold must be < high_risk_volatility_threshold"
            )
        return self


class UniverseConfig(BaseModel):
    model_config = {"frozen": True}

    index_reference: str
    min_avg_daily_value_inr: float = Field(ge=0)
    point_in_time: bool
    exclude_illiquid: bool


class FactorWeights(BaseModel):
    """Non-negative weights combining
    ``universe.factor_calculator.FactorSet`` into one composite score
    (``universe/stock_selector.py``). Every factor is defined so "higher is
    better"; a weight only needs a sign flip for ``volatility``, and that
    flip happens once, explicitly, in ``StockSelector`` -- not here.
    """

    model_config = {"frozen": True}

    momentum: float = Field(ge=0)
    trend_persistence: float = Field(ge=0)
    relative_strength: float = Field(ge=0)
    volatility: float = Field(ge=0)
    drawdown: float = Field(ge=0)
    liquidity: float = Field(ge=0)

    @model_validator(mode="after")
    def _at_least_one_weight_is_positive(self) -> FactorWeights:
        if not any(
            weight > 0
            for weight in (
                self.momentum,
                self.trend_persistence,
                self.relative_strength,
                self.volatility,
                self.drawdown,
                self.liquidity,
            )
        ):
            raise ValueError(
                "at least one factor weight must be > 0; an all-zero weight set "
                "makes every candidate's composite score identical"
            )
        return self


class SelectionConfig(BaseModel):
    model_config = {"frozen": True}

    min_holdings: int = Field(ge=1)
    max_holdings: int = Field(ge=1)
    momentum_lookback_months: list[int]
    """Exactly two horizons in months, ascending: ``[short, long]``."""

    momentum_skip_days: int = Field(ge=0)
    """Most recent sessions excluded from momentum, per
    docs/SPECIFICATION.md section 7.1 ("excluding the most recent short
    window")."""

    trend_ma_window_days: int = Field(ge=2)
    trend_window_days: int = Field(ge=2)
    relative_strength_window_days: int = Field(ge=2)
    volatility_window_days: int = Field(ge=2)
    drawdown_window_days: int = Field(ge=2)
    liquidity_window_days: int = Field(ge=2)
    min_history_days: int = Field(ge=2)
    """Minimum trading-day bar count required before an instrument is even
    considered a candidate (docs/ARCHITECTURE.md's "remove securities with
    insufficient data" step). Validated below to always cover the longest
    individual factor window, so "enough data" here can never mean "not
    enough" to a specific factor."""

    factor_weights: FactorWeights

    @model_validator(mode="after")
    def _min_not_above_max(self) -> SelectionConfig:
        if self.min_holdings > self.max_holdings:
            raise ValueError("min_holdings must be <= max_holdings")
        return self

    @model_validator(mode="after")
    def _momentum_lookback_is_two_ascending_horizons(self) -> SelectionConfig:
        if len(self.momentum_lookback_months) != 2:
            raise ValueError(
                "momentum_lookback_months must have exactly 2 entries: [short, long] months"
            )
        if self.momentum_lookback_months[0] >= self.momentum_lookback_months[1]:
            raise ValueError("momentum_lookback_months short horizon must be < long horizon")
        return self

    @model_validator(mode="after")
    def _min_history_covers_every_factor_window(self) -> SelectionConfig:
        trading_days_per_month = 21  # approximation, used only to size the history buffer
        longest_horizon_days = max(self.momentum_lookback_months) * trading_days_per_month
        longest_needed = max(
            longest_horizon_days + self.momentum_skip_days,
            self.trend_window_days + self.trend_ma_window_days,
            self.relative_strength_window_days,
            self.volatility_window_days + 1,
            self.drawdown_window_days,
            self.liquidity_window_days,
        )
        if self.min_history_days < longest_needed:
            raise ValueError(
                f"min_history_days ({self.min_history_days}) must be >= the longest "
                f"individual factor window ({longest_needed}); otherwise a candidate "
                "judged to have 'enough' history could still fail to compute one factor"
            )
        return self


class PortfolioConfig(BaseModel):
    model_config = {"frozen": True}

    max_single_name_pct: Percent = Field(gt=0, le=1)
    max_sector_pct: Percent = Field(gt=0, le=1)
    min_position_weight_pct: Percent = Field(gt=0, le=1)
    """A position below this fraction of total equity is dropped rather than
    held -- not worth the operational overhead of tracking it. Its freed
    weight becomes cash, not redistributed further (portfolio/portfolio_constructor.py)."""

    max_pairwise_correlation: Percent = Field(gt=0, le=1)
    correlation_lookback_days: int = Field(ge=2)
    correlation_penalty_pct: Percent = Field(ge=0, lt=1)
    """Weight multiplier applied to the lower-ranked member of an
    over-correlated pair, e.g. 0.5 halves it. Never fully excludes -- this is
    a penalty, not a veto; risk/risk_manager.py (Phase 7/8) has veto authority."""

    min_correlation_observations: int = Field(ge=2)
    """Minimum overlapping return observations required to trust a pairwise
    correlation estimate; below this, the pair's correlation is skipped
    rather than acted on."""

    @model_validator(mode="after")
    def _min_weight_below_max_weight(self) -> PortfolioConfig:
        if self.min_position_weight_pct >= self.max_single_name_pct:
            raise ValueError(
                "min_position_weight_pct must be < max_single_name_pct, or every "
                "position would be simultaneously too small to hold and too large to allow"
            )
        return self


class RiskConfig(BaseModel):
    model_config = {"frozen": True}

    max_gross_exposure: Percent = Field(ge=0, le=1)
    max_leverage: float = Field(ge=1, le=1)  # V1: no leverage, hard-pinned to 1.0
    max_single_name_pct: Percent = Field(gt=0, le=1)
    max_sector_pct: Percent = Field(gt=0, le=1)
    max_concurrent_positions: int = Field(ge=1)
    max_risk_per_position_pct: Percent = Field(gt=0, le=1)
    daily_loss_warning_pct: Percent = Field(gt=0, le=1)
    daily_loss_reduce_pct: Percent = Field(gt=0, le=1)
    daily_loss_halt_pct: Percent = Field(gt=0, le=1)
    rolling_loss_reduce_pct: Percent = Field(gt=0, le=1)
    """The "rolling drawdown" check (docs/SPECIFICATION.md section 8): P&L
    over a trailing multi-day window the caller defines when building
    ``PortfolioRiskState.rolling_pnl_pct`` -- distinct from same-day P&L
    (``daily_loss_*``) and from since-inception peak-to-trough
    (``peak_to_trough_drawdown_halt_pct``)."""
    rolling_loss_halt_pct: Percent = Field(gt=0, le=1)
    peak_to_trough_drawdown_halt_pct: Percent = Field(gt=0, le=1)
    stale_data_max_minutes: int = Field(ge=1)
    max_pairwise_correlation: Percent = Field(gt=0, le=1)
    """Risk-layer hard backstop on correlation concentration. Deliberately a
    separate, independently configured limit from
    ``portfolio.max_pairwise_correlation`` (which only halves a weight) --
    this one is a veto, evaluated against whatever correlation actually
    survived construction, not the construction-time estimate."""
    max_adv_participation_pct: Percent = Field(gt=0, le=1)
    max_spread_bps: float = Field(gt=0)
    max_daily_turnover_pct: float = Field(gt=0)
    """Not a ``Percent``/``le=1`` field on purpose: turnover (bought +
    sold, as a fraction of equity) can legitimately exceed 100% in one day
    if a position is both exited and re-entered."""
    reduced_risk_exposure_multiplier: Percent = Field(gt=0, lt=1)
    """Applied to ``max_gross_exposure``, ``max_single_name_pct``, and
    ``max_sector_pct`` while the circuit breaker is in REDUCED_RISK --
    strictly less than 1 so REDUCED_RISK is always actually tighter than
    NORMAL, never a no-op."""

    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> RiskConfig:
        if not (
            self.daily_loss_warning_pct
            < self.daily_loss_reduce_pct
            < self.daily_loss_halt_pct
        ):
            raise ValueError(
                "daily_loss_warning_pct < daily_loss_reduce_pct < daily_loss_halt_pct must hold"
            )
        if not (self.rolling_loss_reduce_pct < self.rolling_loss_halt_pct):
            raise ValueError("rolling_loss_reduce_pct < rolling_loss_halt_pct must hold")
        return self


class ExecutionConfig(BaseModel):
    model_config = {"frozen": True}

    mode: str
    order_type: str
    order_price_guard_bps: float = Field(ge=0)
    stale_quote_seconds: int = Field(ge=1)
    max_participation_adv_pct: Percent = Field(gt=0, le=1)
    no_market_order_fallback: bool


class BrokerConfig(BaseModel):
    model_config = {"frozen": True}

    provider: str
    static_ip_required: bool


class BacktestConfig(BaseModel):
    model_config = {"frozen": True}

    training_window_sessions: int = Field(ge=1)
    test_window_sessions: int = Field(ge=1)
    roll_step_sessions: int = Field(ge=1)
    include_indian_costs: bool
    include_slippage: bool
    point_in_time_universe: bool
    slippage_min_bps: float = Field(ge=0)
    """The research slippage model's floor (docs/SPECIFICATION.md section
    9.1's ``min_bps``) -- a tunable estimation-model parameter, not a
    regulatory rate, so unlike ``CostsConfig`` it needs no effective date."""
    slippage_impact_coefficient: float = Field(ge=0)
    """Scales the square-root price-impact term in
    ``backtest.costs.CostModel.estimate_slippage_bps`` -- also a model
    parameter, not a statutory rate."""


class CostsConfig(BaseModel):
    model_config = {"frozen": True}

    schedule_file: str
    """Path (relative to the project root) to the versioned cost-schedule
    data file (see ``backtest/cost_schedule.py``). Rates are never
    hardcoded in Python -- this file is the single source of truth for
    brokerage/STT/exchange/SEBI/GST/stamp-duty/DP rates, and it grows a new
    dated entry whenever NSE/SEBI/CDSL publish revised levies; an existing
    entry is never edited in place."""


class DatabaseConfig(BaseModel):
    model_config = {"frozen": True}

    url_env_var: str
    pool_size: int = Field(ge=1)


class LoggingConfig(BaseModel):
    model_config = {"frozen": True}

    level: str
    format: str


class MonitoringConfig(BaseModel):
    model_config = {"frozen": True}

    heartbeat_interval_seconds: int = Field(ge=1)
    alert_channels: list[str]


class Settings(BaseModel):
    """Root, validated configuration object for the whole system."""

    model_config = {"frozen": True}

    market: MarketConfig
    data: DataConfig
    features: FeaturesConfig
    hmm: HMMConfig
    regime_policy: RegimePolicyConfig
    allocation: AllocationConfig
    universe: UniverseConfig
    selection: SelectionConfig
    portfolio: PortfolioConfig
    risk: RiskConfig
    execution: ExecutionConfig
    broker: BrokerConfig
    backtest: BacktestConfig
    costs: CostsConfig
    database: DatabaseConfig
    logging: LoggingConfig
    monitoring: MonitoringConfig

    @model_validator(mode="after")
    def _diversification_can_reach_low_risk_exposure(self) -> Settings:
        """The system must be able to reach the low-risk-tier exposure
        ceiling without breaching the single-name cap. See
        docs/ARCHITECTURE.md, "Resolved specification ambiguities" (B4 in the
        architecture review).
        """
        max_reachable = self.selection.min_holdings * self.portfolio.max_single_name_pct
        ceiling = self.regime_policy.low_risk.max_gross_exposure
        if max_reachable < ceiling:
            raise ValueError(
                "selection.min_holdings * portfolio.max_single_name_pct "
                f"({max_reachable:.2f}) cannot reach regime_policy.low_risk.max_gross_exposure "
                f"({ceiling:.2f}); raise min_holdings, raise max_single_name_pct, or lower "
                "the low_risk exposure ceiling."
            )
        return self
