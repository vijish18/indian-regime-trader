"""Persisted table schemas (docs/SPECIFICATION.md section 17).

These are plain, typed row schemas describing what gets persisted -- they
are intentionally separate from the domain dataclasses used elsewhere
(e.g. ``data.instrument_master.Instrument``, ``execution.position_tracker.Position``).
The mapping between domain objects and these rows, and the choice of ORM/
migration tooling, is a Phase 2/3 decision (docs/DEVELOPMENT.md).

Not implemented as a database mapping yet (Phase 2/3). These are schema
definitions only.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentRow:
    instrument_id: str
    symbol: str
    isin: str
    exchange: str
    segment: str
    tick_size: float
    lot_size: int
    price_band_pct: float | None
    effective_from: dt.date
    effective_to: dt.date | None


@dataclass(frozen=True)
class UniverseSnapshotRow:
    as_of: dt.date
    symbol: str
    inclusion_reason: str | None
    exclusion_reason: str | None


@dataclass(frozen=True)
class BarRow:
    instrument_id: str
    timestamp: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    data_source: str
    adjustment_version: int


@dataclass(frozen=True)
class FeatureRow:
    timestamp: dt.datetime
    feature_set_version: str
    feature_values: dict[str, float]


@dataclass(frozen=True)
class HMMModelRow:
    model_id: str
    train_start: dt.date
    train_end: dt.date
    feature_set_version: str
    state_count: int
    bic: float
    seed: int
    artifact_path: str
    approved: bool


@dataclass(frozen=True)
class RegimeHistoryRow:
    timestamp: dt.datetime
    model_id: str
    state_id: int
    state_label: str
    probabilities: tuple[float, ...]
    confidence: float


@dataclass(frozen=True)
class SignalRow:
    signal_id: str
    timestamp: dt.datetime
    instrument_id: str
    target_weight: float
    reason: str
    regime_id: str


@dataclass(frozen=True)
class RiskDecisionRow:
    signal_id: str
    approved: bool
    modified_weight: float | None
    rejection_reason: str | None


@dataclass(frozen=True)
class OrderRow:
    trade_id: str
    broker_order_id: str | None
    instrument_id: str
    side: str
    quantity: int
    order_type: str
    limit_price: float | None
    status: str


@dataclass(frozen=True)
class FillRow:
    order_id: str
    fill_quantity: int
    fill_price: float
    timestamp: dt.datetime
    fees: float


@dataclass(frozen=True)
class PositionRow:
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    target_weight: float


@dataclass(frozen=True)
class PortfolioSnapshotRow:
    timestamp: dt.datetime
    equity: float
    cash: float
    gross_exposure: float
    drawdown: float


@dataclass(frozen=True)
class AuditEventRow:
    timestamp: dt.datetime
    component: str
    event_type: str
    payload_hash: str
    severity: str


@dataclass(frozen=True)
class SystemStateRow:
    snapshot_version: int
    last_processed_market_timestamp: dt.datetime
    health_state: str
