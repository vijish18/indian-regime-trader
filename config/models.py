"""Typed configuration models for the Indian Market Regime Trading System.

These mirror the structure validated by ``config/settings.schema.yaml``. The
schema catches malformed YAML at load time; the cross-field validators here
catch combinations that are individually valid but jointly inconsistent with
the system's own architectural constraints (see docs/ARCHITECTURE.md,
"Resolved specification ambiguities").
"""

from __future__ import annotations

import datetime as dt
import ipaddress

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

    trend_ma_window_days: int = Field(ge=2)
    """Trailing sessions used by the non-HMM moving-average trend baseline
    (core/regime/baseline_policy.py's ``MovingAverageTrendBaseline`` --
    docs/SPECIFICATION.md section 10.1's "simple 200-day moving-average
    risk filter" walk-forward benchmark)."""

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


class StopLossConfig(BaseModel):
    """Per-position stop thresholds (risk/stop_loss.py).

    Nested under ``risk`` because these are risk limits, but structurally
    separate from the book-level thresholds around them: those describe when
    the *portfolio* stops trading, these describe when *one holding* is sold.
    """

    model_config = {"frozen": True}

    enabled: bool
    hard_stop_pct: Percent = Field(gt=0, lt=1)
    """Exit when the price falls this far below the price the position was
    bought at. A fixed level, not a trailing one."""

    trail_drop_pct: Percent = Field(gt=0, lt=1)
    """Exit when the price falls this far below the session's high -- but
    only if ``trail_arm_net_profit_pct`` is also satisfied."""

    trail_arm_net_profit_pct: Percent = Field(gt=0, lt=1)
    """The trailing stop stays disarmed until selling would realise more than
    this, **net of every sell-side charge including the DP charge**. Without
    it the trailing rule would fire on positions that are flat or losing,
    which is the hard stop's job at a much wider level."""


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
    stop_loss: StopLossConfig

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


class PaperTradingConfig(BaseModel):
    """Simulation-only parameters for ``broker.adapters.paper_broker.PaperBroker``
    -- never consulted by a live adapter, which experiences real latency and
    real exchange-matching behavior instead of a configured stand-in for it.
    """

    model_config = {"frozen": True}

    submission_latency_ms: int = Field(ge=0)
    """Simulated round-trip time from ``place_order`` to broker
    acknowledgement, reflected in the order's recorded timestamps -- the
    paper broker never actually sleeps for this long."""

    max_fill_participation_pct: Percent = Field(gt=0, le=1)
    """The fraction of a quote's displayed depth (``bid_quantity``/
    ``ask_quantity``) one simulated match is willing to assume is takeable,
    so a large order against a real order book fills partially over
    multiple matches rather than assuming unlimited depth at the touch."""

    order_expiry_seconds: int = Field(ge=1)
    """A resting (unfilled or partially filled) order past this age
    transitions to ``EXPIRED`` the next time it is checked -- this is a
    duration-based simplification, not session-aware (a real
    good-for-day order expires at session close); documented, not hidden."""

    default_avg_daily_value_inr: float = Field(ge=0)
    """Fallback liquidity estimate for an instrument with fewer than two
    trailing bars of history -- feeds the same square-root impact model as
    ``backtest.costs.CostModel``, never a fabricated near-zero impact."""

    default_volatility: float = Field(ge=0)
    """Fallback annualized volatility estimate, same purpose as
    ``default_avg_daily_value_inr`` above."""


class ComplianceConfig(BaseModel):
    """India API/algo operational controls (Phase 16, ``docs/COMPLIANCE.md``).

    Governs whether this system may operate a live-capable broker at all,
    not just how it behaves once running: ``broker.compliance.ComplianceGate``
    refuses to construct itself if any of this is missing or invalid, and
    refuses every order that would violate it -- "must produce a HARD
    FAILURE, not a best-effort order" is this config's entire reason to
    exist. See ``docs/COMPLIANCE.md`` for the research behind every field
    below and, just as importantly, what remains unverified.
    """

    model_config = {"frozen": True}

    compliance_version: str = Field(min_length=1)
    """A human-readable tag for the source rules this configuration was
    last checked against (e.g. ``"2025-11-03-nse-retail-algo-faq"``) --
    docs/SPECIFICATION.md section 13's "compliance configuration version"
    requirement, made concrete: a later regulatory or broker change is a
    deliberate version bump here, not silent drift."""

    broker_authorization_confirmed: bool
    """Must be explicitly ``True``. Attests a human has confirmed with the
    broker -- not assumed, not defaulted -- that this account's API/algo
    setup is actually eligible for live trading (docs/COMPLIANCE.md,
    "Broker-specific algo onboarding requirements"). ``False`` is this
    system's honest default state before that confirmation has actually
    happened, not a placeholder to be flipped without it."""

    static_ip_primary: str = Field(min_length=1)
    """This system's own attestation of the static IP registered with the
    broker for order-placing endpoints (docs/COMPLIANCE.md section 2) --
    this code has no documented API to verify the broker's own dashboard
    state, so this field exists to make "was this actually registered" a
    recorded human decision rather than an assumption. ``"0.0.0.0"`` is
    the not-yet-configured placeholder used in ``settings.yaml``'s default
    (paper-mode) configuration -- syntactically a valid IP so the file
    loads, but ``ComplianceGate`` treats it as equivalent to missing."""

    static_ip_secondary: str | None = None

    algo_identifier: str = Field(min_length=1)
    """The identifier this account/strategy must tag every order with, per
    NSE's algo-tagging requirement -- sourced from the broker's own
    algo-onboarding confirmation (docs/COMPLIANCE.md section 4), never
    invented by this codebase. ``"UNSET"`` is the not-yet-configured
    placeholder, treated as equivalent to missing by ``ComplianceGate``."""

    allowed_order_types: tuple[str, ...] = Field(min_length=1)
    """Order types (Kite's ``order_type`` vocabulary) this system may
    submit under the retail-algo framework -- ``MARKET`` is excluded by
    NSE's own rule (docs/COMPLIANCE.md section 5); rejected below if
    present rather than left to a caller to remember to omit it."""

    allowed_validities: tuple[str, ...] = Field(min_length=1)
    """Order validities (Kite's ``validity`` vocabulary) this system may
    submit -- ``IOC`` is excluded by the same rule."""

    max_orders_per_second: int = Field(gt=0, le=10)
    """The unregistered tech-savvy route's regulatory ceiling is 10 OPS
    (docs/COMPLIANCE.md section 8); capped here by field validation so
    this configuration cannot claim a rate the route this system actually
    uses does not cover -- exceeding 10 OPS requires formal exchange algo
    registration, a materially different, unimplemented compliance
    posture, not a config value to raise."""

    session_max_age_hours: float = Field(gt=0, le=24)
    """A Kite access token is valid for one trading day only
    (docs/COMPLIANCE.md section 9); a session older than this is treated
    as expired authentication and refuses further order placement."""

    audit_log_retention_years: int = Field(ge=1)
    """See docs/COMPLIANCE.md section 12 -- this figure is not verified
    against a retail-algo-specific primary source during this phase's
    research; confirm with the broker/compliance officer before relying
    on it as an actual retention policy."""

    @model_validator(mode="after")
    def _order_types_and_validities_exclude_what_nse_prohibits(self) -> ComplianceConfig:
        """MARKET orders and IOC validity are not permitted for algo flow
        (docs/COMPLIANCE.md section 5) -- Kite's own API is technically
        permissive enough to accept both, so this system must be the one
        to refuse them, not assume the broker will."""
        prohibited_order_types = {"MARKET"}
        prohibited_validities = {"IOC"}
        found_types = prohibited_order_types & {v.upper() for v in self.allowed_order_types}
        if found_types:
            raise ValueError(
                f"allowed_order_types must not include a prohibited type: {sorted(found_types)} "
                "-- NSE's retail-algo framework does not permit market orders for algo flow "
                "(see docs/COMPLIANCE.md section 5)"
            )
        found_validities = prohibited_validities & {v.upper() for v in self.allowed_validities}
        if found_validities:
            raise ValueError(
                f"allowed_validities must not include a prohibited value: "
                f"{sorted(found_validities)} -- NSE's retail-algo framework does not permit "
                "IOC orders for algo flow (see docs/COMPLIANCE.md section 5)"
            )
        return self

    @model_validator(mode="after")
    def _static_ips_are_well_formed(self) -> ComplianceConfig:
        """A malformed value is exactly as unsafe as a missing one -- this
        is a syntactic check only (does this parse as an IP address), not
        a claim that it is actually registered with the broker; see this
        field's own docstring and docs/COMPLIANCE.md section 2."""
        for label, value in (
            ("static_ip_primary", self.static_ip_primary),
            ("static_ip_secondary", self.static_ip_secondary),
        ):
            if value is None:
                continue
            try:
                ipaddress.ip_address(value)
            except ValueError as exc:
                raise ValueError(f"{label} must be a valid IP address, got {value!r}") from exc
        return self


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

    audit_log_path: str | None = None
    """Where to additionally write every log record as a durable,
    rotating file (Phase 23). ``None`` disables it and leaves stdout as
    the only destination. Set in production to a path inside a mounted
    volume, so the audit trail survives the container that wrote it --
    stdout alone lives in the container runtime's own storage, which a
    ``docker system prune`` discards. Never a secret, so it belongs here
    rather than in ``.env``."""

    audit_log_max_bytes: int = Field(default=50_000_000, ge=1024)
    """Rotation threshold for ``audit_log_path``. Rotation is this
    process's own job as well as the container runtime's: the stdout
    stream is rotated by Docker's ``json-file`` driver, but a file this
    process opens itself is not."""

    audit_log_backup_count: int = Field(default=10, ge=1)
    """How many rotated audit files to keep. With the defaults above, an
    upper bound of roughly 550 MB of audit history on disk."""


class MonitoringConfig(BaseModel):
    model_config = {"frozen": True}

    heartbeat_interval_seconds: int = Field(ge=1)
    alert_channels: list[str]

    alert_webhook_url_env: str = "ALERT_WEBHOOK_URL"
    """Name of the environment variable holding the webhook URL -- never the
    URL. A Slack or Telegram webhook URL is a credential: anyone holding it
    can post as you, and Telegram's embeds the bot token outright. Keeping
    only the variable *name* here is what lets this file stay in Git."""

    alert_webhook_message_field: str = "text"
    """Which JSON key carries the message. ``text`` suits Slack, Telegram,
    Google Chat and ntfy; Discord wants ``content``."""

    alert_webhook_static_fields: dict[str, str] = Field(default_factory=dict)
    """Extra JSON fields sent with every alert, for services that need
    routing information -- Telegram's ``chat_id``, for instance. Do not put
    a secret here; this file is committed."""

    alert_webhook_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    """Bounds how long a hanging pager can delay a trading cycle. Delivery is
    synchronous on purpose -- for the one process whose job is to stop
    safely, knowing the page arrived is worth the seconds -- so this is the
    control that keeps that from becoming unbounded."""
    alert_cooldown_seconds: int = Field(ge=1)
    """Minimum gap between two alerts of the same kind about the same
    subject (``monitoring.alerts.AlertManager``'s rate limit). A condition
    that stays true -- a broker that stays disconnected -- must not
    generate one alert per monitoring-loop iteration, or the alerts that
    matter drown in the ones that are already known."""


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
    paper_trading: PaperTradingConfig
    compliance: ComplianceConfig
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
