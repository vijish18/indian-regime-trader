"""Data-quality validation rules applied before raw data is trusted anywhere
downstream. See docs/SPECIFICATION.md section 4.1.

Not implemented yet (Phase 3).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class DataQualityIssue(StrEnum):
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    IMPOSSIBLE_OHLC = "impossible_ohlc"
    NEGATIVE_OR_ZERO_PRICE = "negative_or_zero_price"
    MISSING_SESSION = "missing_session"
    STALE_PRICE = "stale_price"
    SUSPICIOUS_VOLUME_SPIKE = "suspicious_volume_spike"


@dataclass(frozen=True)
class DataQualityReport:
    instrument_id: str
    issues: tuple[DataQualityIssue, ...]
    checked_rows: int


class DataQualityValidator:
    """Validates a raw OHLCV series against the rules in
    docs/SPECIFICATION.md section 4.1 before it is normalized or used.
    """

    def check_duplicate_timestamps(self, bars: pd.DataFrame) -> list[int]:
        raise NotImplementedError("Phase 3: data quality checks are not implemented yet.")

    def check_ohlc_relationships(self, bars: pd.DataFrame) -> list[int]:
        """Rows where high < low, close outside [low, high], etc."""
        raise NotImplementedError("Phase 3: data quality checks are not implemented yet.")

    def check_negative_or_zero_prices(self, bars: pd.DataFrame) -> list[int]:
        raise NotImplementedError("Phase 3: data quality checks are not implemented yet.")

    def check_missing_sessions(
        self, bars: pd.DataFrame, expected_trading_days: list[dt.date]
    ) -> list[dt.date]:
        raise NotImplementedError("Phase 3: data quality checks are not implemented yet.")

    def check_zero_volume_is_genuine(self, bars: pd.DataFrame) -> list[int]:
        """Distinguishes a genuine zero-volume session from missing data."""
        raise NotImplementedError("Phase 3: data quality checks are not implemented yet.")

    def validate(self, instrument_id: str, bars: pd.DataFrame) -> DataQualityReport:
        """Run all checks and return a consolidated report."""
        raise NotImplementedError("Phase 3: data quality checks are not implemented yet.")
