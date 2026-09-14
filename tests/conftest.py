from __future__ import annotations

import copy
import datetime as dt
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.calendar import NSETradingCalendar
from data.models import (
    DailyBar,
    Exchange,
    Instrument,
    InstrumentStatus,
    Segment,
)
from data.storage import LocalDataStore, StorageFormat

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def valid_settings_dict() -> dict[str, Any]:
    """A full, schema-valid settings dict loaded from the real
    config/settings.yaml and deep-copied so tests can mutate it freely
    without affecting other tests or the file on disk.
    """
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return copy.deepcopy(data)


# --------------------------------------------------------------------------
# Data-layer fixtures
#
# Tests build their own calendar rather than reading config/nse_holidays.csv,
# so the suite neither depends on that file being populated nor breaks when
# the real holiday list is updated.
# --------------------------------------------------------------------------


@pytest.fixture
def calendar() -> NSETradingCalendar:
    """A calendar covering 2024 only, with two closures and one special session.

    2024-01-26 (Fri) is a holiday, 2024-03-08 (Fri) is a holiday, and
    2024-11-03 (a Sunday) is a Muhurat-style special session -- which is what
    makes "weekday and not a holiday" an insufficient rule.
    """
    return NSETradingCalendar(
        holidays={
            dt.date(2024, 1, 26): "Republic Day",
            dt.date(2024, 3, 8): "Mahashivratri",
        },
        special_sessions={dt.date(2024, 11, 3): "Muhurat Trading"},
    )


@pytest.fixture
def store(tmp_path: Path) -> LocalDataStore:
    """A CSV-backed store in a temp directory.

    CSV keeps failures readable in test output; Parquet is exercised
    separately in the storage round-trip tests.
    """
    return LocalDataStore(
        raw_root=tmp_path / "raw",
        normalized_root=tmp_path / "normalized",
        reference_root=tmp_path / "reference",
        storage_format=StorageFormat.CSV,
    )


@pytest.fixture
def infosys() -> Instrument:
    return Instrument(
        instrument_id="NSE:INFY",
        symbol="INFY",
        exchange=Exchange.NSE,
        segment=Segment.EQUITY,
        tick_size=Decimal("0.05"),
        price_precision=2,
        effective_from=dt.date(2020, 1, 1),
        isin="INE009A01021",
        lot_size=1,
        status=InstrumentStatus.ACTIVE,
        name="Infosys Limited",
    )


def make_bar(
    session_date: dt.date,
    close: str = "100.00",
    *,
    instrument_id: str = "NSE:INFY",
    open_: str | None = None,
    high: str | None = None,
    low: str | None = None,
    volume: int = 10_000,
) -> DailyBar:
    """Build a well-formed bar, overriding only what a test cares about."""
    close_price = Decimal(close)
    return DailyBar(
        instrument_id=instrument_id,
        session_date=session_date,
        open=Decimal(open_) if open_ is not None else close_price,
        high=Decimal(high) if high is not None else close_price,
        low=Decimal(low) if low is not None else close_price,
        close=close_price,
        volume=volume,
    )


BarFactory = Callable[..., DailyBar]


@pytest.fixture
def bar_factory() -> BarFactory:
    """Expose :func:`make_bar` as a fixture for readability in tests."""
    return make_bar
