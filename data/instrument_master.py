"""Point-in-time instrument reference data.

Instrument records are versioned by effective date rather than overwritten,
because reference data changes and a backtest must see the values that were in
force at the time: a tick size that changed in 2023 must not be applied to a
2019 order simulation, and a symbol reassigned after a delisting must not
resolve to the wrong company.

Overlapping effective ranges for the same instrument are rejected at load
time: if two records claim the same day, every downstream lookup becomes
order-dependent, which is the kind of bug that only shows up as an
unexplainable backtest discrepancy months later.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path

import pandas as pd

from data.errors import InstrumentNotFoundError
from data.interfaces import InstrumentRepository
from data.models import Exchange, Instrument, InstrumentStatus, Segment
from data.storage import (
    LocalDataStore,
    parse_date,
    parse_decimal,
    parse_int,
    parse_optional_date,
    parse_optional_int,
    parse_optional_str,
    read_table,
    require_columns,
)

INSTRUMENT_COLUMNS = (
    "instrument_id",
    "symbol",
    "exchange",
    "segment",
    "tick_size",
    "price_precision",
    "effective_from",
)
"""Required columns. ``effective_to``, ``isin``, ``lot_size``, ``status`` and
``name`` are optional."""

_TICK_QUANTUM = Decimal("0.0001")


class InMemoryInstrumentRepository(InstrumentRepository):
    """Instrument repository backed by an in-memory set of versioned records.

    Construct it from a file with :meth:`from_file`, or directly from
    ``Instrument`` objects in tests.
    """

    def __init__(
        self,
        instruments: Iterable[Instrument],
        snapshot_date: dt.date | None = None,
    ) -> None:
        self._by_id: dict[str, list[Instrument]] = defaultdict(list)
        for instrument in instruments:
            self._by_id[instrument.instrument_id].append(instrument)
        for versions in self._by_id.values():
            versions.sort(key=lambda record: record.effective_from)
        _reject_overlapping_versions(self._by_id)
        self._snapshot_date = snapshot_date

    @classmethod
    def from_file(
        cls, path: Path, snapshot_date: dt.date | None = None
    ) -> InMemoryInstrumentRepository:
        """Load instruments from a CSV/Parquet file.

        Raises:
            ValueError: if required columns are missing, a row is unparseable,
                or two records for one instrument cover the same date.
        """
        frame = read_table(path)
        require_columns(frame, INSTRUMENT_COLUMNS, str(path))
        return cls(parse_instrument_rows(frame, source=str(path)), snapshot_date)

    @classmethod
    def from_store(cls, store: LocalDataStore) -> InMemoryInstrumentRepository:
        path = store.instruments_path()
        snapshot = dt.datetime.fromtimestamp(
            path.stat().st_mtime, tz=dt.UTC
        ).date() if path.is_file() else None
        return cls.from_file(path, snapshot_date=snapshot)

    def get(self, instrument_id: str, as_of: dt.date) -> Instrument:
        for instrument in self._by_id.get(instrument_id, ()):
            if instrument.is_effective_on(as_of):
                return instrument
        raise InstrumentNotFoundError(
            f"no instrument record for {instrument_id!r} effective on {as_of}"
        )

    def get_by_symbol(self, symbol: str, exchange: str, as_of: dt.date) -> Instrument:
        matches = [
            instrument
            for versions in self._by_id.values()
            for instrument in versions
            if instrument.symbol == symbol
            and instrument.exchange == exchange
            and instrument.is_effective_on(as_of)
        ]
        if not matches:
            raise InstrumentNotFoundError(
                f"no instrument with symbol {symbol!r} on {exchange} effective on {as_of}"
            )
        if len(matches) > 1:
            ids = ", ".join(sorted(match.instrument_id for match in matches))
            raise InstrumentNotFoundError(
                f"symbol {symbol!r} on {exchange} is ambiguous on {as_of}: {ids}"
            )
        return matches[0]

    def list_effective(
        self, as_of: dt.date, segment: Segment | None = None
    ) -> list[Instrument]:
        effective = [
            instrument
            for versions in self._by_id.values()
            for instrument in versions
            if instrument.is_effective_on(as_of)
            and (segment is None or instrument.segment == segment)
        ]
        return sorted(effective, key=lambda record: record.instrument_id)

    def history(self, instrument_id: str) -> list[Instrument]:
        return list(self._by_id.get(instrument_id, ()))

    def instrument_ids(self) -> list[str]:
        """Every instrument id known to this repository, ascending."""
        return sorted(self._by_id)

    def all_versions(self) -> list[Instrument]:
        """Every version of every instrument, for whole-master validation."""
        return [
            instrument
            for instrument_id in self.instrument_ids()
            for instrument in self._by_id[instrument_id]
        ]

    def snapshot_date(self) -> dt.date | None:
        return self._snapshot_date

    def is_fresh(self, max_age_days: int, reference_date: dt.date) -> bool:
        """True if the snapshot is no older than ``max_age_days``.

        Used by the startup sequence, which refuses to trade on a stale
        instrument master (docs/SPECIFICATION.md section 15.2).
        """
        if self._snapshot_date is None:
            return False
        return (reference_date - self._snapshot_date).days <= max_age_days


def parse_instrument_rows(frame: pd.DataFrame, source: str) -> list[Instrument]:
    """Convert a reference-data table into ``Instrument`` records."""
    instruments: list[Instrument] = []
    for position, row in enumerate(frame.to_dict("records"), start=2):
        try:
            instruments.append(
                Instrument(
                    instrument_id=str(row["instrument_id"]).strip(),
                    symbol=str(row["symbol"]).strip(),
                    exchange=Exchange(str(row["exchange"]).strip().upper()),
                    segment=Segment(str(row["segment"]).strip().lower()),
                    tick_size=parse_decimal(row["tick_size"], _TICK_QUANTUM),
                    price_precision=parse_int(row["price_precision"]),
                    effective_from=parse_date(row["effective_from"]),
                    effective_to=parse_optional_date(row.get("effective_to")),
                    isin=parse_optional_str(row.get("isin")),
                    lot_size=parse_optional_int(row.get("lot_size")),
                    status=InstrumentStatus(
                        (parse_optional_str(row.get("status")) or "active").lower()
                    ),
                    name=parse_optional_str(row.get("name")),
                )
            )
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{source} line {position}: {exc}") from exc
    return instruments


def _reject_overlapping_versions(by_id: dict[str, list[Instrument]]) -> None:
    """Raise if any instrument has two records effective on the same date."""
    for instrument_id, versions in by_id.items():
        for earlier, later in zip(versions, versions[1:], strict=False):
            if earlier.effective_to is None:
                raise ValueError(
                    f"{instrument_id}: record effective from {earlier.effective_from} "
                    "is open-ended but is followed by another record "
                    f"from {later.effective_from}; close it with effective_to"
                )
            if later.effective_from <= earlier.effective_to:
                raise ValueError(
                    f"{instrument_id}: overlapping effective ranges "
                    f"[{earlier.effective_from}..{earlier.effective_to}] and "
                    f"[{later.effective_from}..{later.effective_to}]"
                )
