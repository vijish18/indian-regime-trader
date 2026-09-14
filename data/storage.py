"""Local data storage: file layout, CSV/Parquet I/O, and parsing helpers.

Layout (rooted at ``data.raw_data_path`` / ``normalized_data_path`` /
``reference_data_path`` from config)::

    data_cache/
      raw/                      vendor data exactly as received, never rewritten
        equity_bars/<instrument_id>.<ext>
        index/<index_symbol>.<ext>
      normalized/               derived, versioned, safe to regenerate
        equity_bars/<instrument_id>.<ext>
      reference/
        instruments.<ext>
        corporate_actions.<ext>
        index_membership.<ext>

The raw/normalized split is a hard rule from docs/SPECIFICATION.md section
4.1: raw vendor data is immutable, so that a corporate-action adjustment or a
parsing fix can always be recomputed from source rather than being a
destructive, unrepeatable edit.

CSV columns are read as text so that decimal prices survive the round trip
exactly; Parquet keeps its native types. Both paths reconstruct ``Decimal``
through :func:`parse_decimal`, so storage format never changes a value.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from data.errors import DataNotAvailableError
from data.models import PRICE_PRECISION


class StorageFormat(StrEnum):
    CSV = "csv"
    PARQUET = "parquet"

    @property
    def suffix(self) -> str:
        return f".{self.value}"


class DataLayer(StrEnum):
    """Which copy of a dataset a path refers to."""

    RAW = "raw"
    NORMALIZED = "normalized"
    REFERENCE = "reference"


# --------------------------------------------------------------------------
# Parsing helpers
#
# Every value that reaches a domain model goes through one of these, so that
# CSV text and Parquet native types converge on the same Python object.
# --------------------------------------------------------------------------


def parse_decimal(value: Any, quantum: Decimal = PRICE_PRECISION) -> Decimal:
    """Convert a stored value to an exactly-quantized ``Decimal``.

    Floats are routed through ``str`` first: ``Decimal(0.1)`` captures binary
    noise, while ``Decimal(str(0.1))`` gives the intended ``0.1``.
    """
    if isinstance(value, Decimal):
        return value.quantize(quantum)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("cannot parse a missing value as Decimal")
    try:
        return Decimal(str(value).strip()).quantize(quantum)
    except InvalidOperation as exc:
        raise ValueError(f"{value!r} is not a valid decimal") from exc


def parse_optional_decimal(
    value: Any, quantum: Decimal = PRICE_PRECISION
) -> Decimal | None:
    if is_missing(value):
        return None
    return parse_decimal(value, quantum)


def parse_date(value: Any) -> dt.date:
    """Convert a stored value to a ``date``.

    Accepts ISO-8601 text, ``date``, ``datetime`` and pandas timestamps, and
    rejects anything else loudly -- a silently mis-parsed date shifts an entire
    price series against the calendar.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().date()
    if is_missing(value):
        raise ValueError("cannot parse a missing value as a date")
    text = str(value).strip()
    try:
        return dt.date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{text!r} is not an ISO-8601 date (YYYY-MM-DD)") from exc


def parse_optional_date(value: Any) -> dt.date | None:
    if is_missing(value):
        return None
    return parse_date(value)


def parse_int(value: Any) -> int:
    if is_missing(value):
        raise ValueError("cannot parse a missing value as int")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        return int(Decimal(text))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{value!r} is not a valid integer") from exc


def parse_optional_int(value: Any) -> int | None:
    if is_missing(value):
        return None
    return parse_int(value)


def parse_optional_str(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def is_missing(value: Any) -> bool:
    """True for ``None``, NaN/NaT, and empty-or-whitespace text."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if value is pd.NaT:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], source: str) -> None:
    """Raise if ``frame`` is missing any required column."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required column(s): {', '.join(missing)}; "
            f"found {list(frame.columns)}"
        )


# --------------------------------------------------------------------------
# Table I/O
# --------------------------------------------------------------------------


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame.

    Raises:
        DataNotAvailableError: if the file does not exist.
        ValueError: if the extension is not a supported format.
    """
    if not path.is_file():
        raise DataNotAvailableError(f"no data file at {path}")
    match path.suffix.lower():
        case ".csv":
            return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        case ".parquet":
            return pd.read_parquet(path)
        case other:
            raise ValueError(f"unsupported storage format {other!r} for {path}")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as CSV or Parquet, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    match path.suffix.lower():
        case ".csv":
            frame.to_csv(path, index=False)
        case ".parquet":
            frame.to_parquet(path, index=False)
        case other:
            raise ValueError(f"unsupported storage format {other!r} for {path}")


class LocalDataStore:
    """Owns the on-disk layout. Every other module asks this class where data
    lives instead of building paths itself.
    """

    def __init__(
        self,
        raw_root: Path,
        normalized_root: Path,
        reference_root: Path,
        storage_format: StorageFormat = StorageFormat.PARQUET,
    ) -> None:
        self.raw_root = raw_root
        self.normalized_root = normalized_root
        self.reference_root = reference_root
        self.storage_format = storage_format

    def root_for(self, layer: DataLayer) -> Path:
        match layer:
            case DataLayer.RAW:
                return self.raw_root
            case DataLayer.NORMALIZED:
                return self.normalized_root
            case DataLayer.REFERENCE:
                return self.reference_root

    def equity_bars_path(
        self, instrument_id: str, layer: DataLayer = DataLayer.RAW
    ) -> Path:
        return (
            self.root_for(layer)
            / "equity_bars"
            / f"{_safe_name(instrument_id)}{self.storage_format.suffix}"
        )

    def index_path(self, index_symbol: str, layer: DataLayer = DataLayer.RAW) -> Path:
        return (
            self.root_for(layer)
            / "index"
            / f"{_safe_name(index_symbol)}{self.storage_format.suffix}"
        )

    def instruments_path(self) -> Path:
        return self.reference_root / f"instruments{self.storage_format.suffix}"

    def corporate_actions_path(self) -> Path:
        return self.reference_root / f"corporate_actions{self.storage_format.suffix}"

    def index_membership_path(self) -> Path:
        return self.reference_root / f"index_membership{self.storage_format.suffix}"

    def exists(self, path: Path) -> bool:
        return path.is_file()

    def read(self, path: Path) -> pd.DataFrame:
        return read_table(path)

    def write(self, frame: pd.DataFrame, path: Path) -> None:
        write_table(frame, path)


def _safe_name(identifier: str) -> str:
    """Make an instrument id safe to use as a filename.

    Symbols contain characters that are legal on one platform and not another
    (``&`` in ``M&M``, ``/`` in some series codes); normalizing here keeps the
    same data readable on Windows and Linux.
    """
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in identifier
    )
    if not cleaned.strip("_"):
        raise ValueError(f"identifier {identifier!r} has no usable filename characters")
    return cleaned
