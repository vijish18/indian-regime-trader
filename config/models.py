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
    model_config = {"frozen": True}

    calm: ExposureBand
    normal: ExposureBand
    elevated: ExposureBand
    crisis: ExposureBand


class UniverseConfig(BaseModel):
    model_config = {"frozen": True}

    index_reference: str
    min_avg_daily_value_inr: float = Field(ge=0)
    point_in_time: bool
    exclude_illiquid: bool


class SelectionConfig(BaseModel):
    model_config = {"frozen": True}

    min_holdings: int = Field(ge=1)
    max_holdings: int = Field(ge=1)
    momentum_lookback_months: list[int]

    @model_validator(mode="after")
    def _min_not_above_max(self) -> SelectionConfig:
        if self.min_holdings > self.max_holdings:
            raise ValueError("min_holdings must be <= max_holdings")
        return self


class PortfolioConfig(BaseModel):
    model_config = {"frozen": True}

    max_single_name_pct: Percent = Field(gt=0, le=1)
    max_sector_pct: Percent = Field(gt=0, le=1)


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
    weekly_loss_reduce_pct: Percent = Field(gt=0, le=1)
    weekly_loss_halt_pct: Percent = Field(gt=0, le=1)
    peak_drawdown_halt_pct: Percent = Field(gt=0, le=1)
    stale_data_max_minutes: int = Field(ge=1)

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
        if not (self.weekly_loss_reduce_pct < self.weekly_loss_halt_pct):
            raise ValueError("weekly_loss_reduce_pct < weekly_loss_halt_pct must hold")
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


class CostsConfig(BaseModel):
    model_config = {"frozen": True}

    brokerage_flat_inr: float = Field(ge=0)
    brokerage_pct: float = Field(ge=0)
    stt_delivery_pct: float = Field(ge=0)
    exchange_txn_pct: float = Field(ge=0)
    sebi_turnover_pct: float = Field(ge=0)
    gst_pct: float = Field(ge=0)
    stamp_duty_pct: float = Field(ge=0)
    dp_charges_inr: float = Field(ge=0)


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
    hmm: HMMConfig
    regime_policy: RegimePolicyConfig
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
    def _diversification_can_reach_calm_exposure(self) -> Settings:
        """The system must be able to reach the calm-regime exposure ceiling
        without breaching the single-name cap. See docs/ARCHITECTURE.md,
        "Resolved specification ambiguities" (B4 in the architecture review).
        """
        max_reachable = self.selection.min_holdings * self.portfolio.max_single_name_pct
        if max_reachable < self.regime_policy.calm.max_gross_exposure:
            raise ValueError(
                "selection.min_holdings * portfolio.max_single_name_pct "
                f"({max_reachable:.2f}) cannot reach regime_policy.calm.max_gross_exposure "
                f"({self.regime_policy.calm.max_gross_exposure:.2f}); raise min_holdings, "
                "raise max_single_name_pct, or lower the calm exposure ceiling."
            )
        return self
