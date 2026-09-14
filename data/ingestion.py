"""Ingestion: read a vendor file, validate it, store it.

The pipeline is deliberately boring and strict:

1. Parse the source file into domain objects (representable even if wrong).
2. Validate against ``data.data_quality``.
3. Write to the **raw** layer only if validation found no errors.

Step 3 is what keeps the raw layer trustworthy. Rejecting on error rather than
storing-and-flagging means a later phase can treat anything under ``raw/`` as
having passed structural validation, instead of every consumer re-deriving
whether a file is usable. ``quarantine=True`` overrides this for the case
where operations explicitly wants the bad file persisted for inspection -- it
writes to a separate quarantine path, never over good data.

Ingestion never touches the normalized layer: normalization (corporate-action
adjustment) is a derived step that must stay reproducible from raw
(docs/SPECIFICATION.md section 4.1).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from data.corporate_actions import (
    CORPORATE_ACTION_COLUMNS,
    parse_corporate_action_rows,
)
from data.data_quality import (
    BarValidator,
    DataQualityReport,
    IndexObservationValidator,
    InstrumentLookup,
    validate_corporate_actions,
    validate_instruments,
)
from data.instrument_master import (
    INSTRUMENT_COLUMNS,
    InMemoryInstrumentRepository,
    parse_instrument_rows,
)
from data.interfaces import TradingCalendar
from data.market_data import (
    BAR_COLUMNS,
    INDEX_COLUMNS,
    bars_to_frame,
    index_observations_to_frame,
    parse_bar_rows,
    parse_index_rows,
)
from data.membership import (
    MEMBERSHIP_COLUMNS,
    InMemoryIndexMembershipProvider,
    parse_membership_rows,
)
from data.models import DailyBar, IndexObservation
from data.storage import (
    DataLayer,
    LocalDataStore,
    read_table,
    require_columns,
    write_table,
)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """What one ingest did, and why it did or did not store anything."""

    source: Path
    destination: Path | None
    records_read: int
    records_stored: int
    report: DataQualityReport
    quarantined: bool = False

    @property
    def accepted(self) -> bool:
        return self.records_stored > 0 and not self.quarantined

    def summary(self) -> str:
        verdict = (
            "quarantined"
            if self.quarantined
            else ("stored" if self.accepted else "rejected")
        )
        return (
            f"{self.source.name}: read {self.records_read}, {verdict}"
            f" ({len(self.report.errors)} error(s), {len(self.report.warnings)} warning(s))"
        )


class DataIngestionPipeline:
    """Validates and stores vendor files into the local raw layer."""

    def __init__(
        self,
        store: LocalDataStore,
        calendar: TradingCalendar | None = None,
        *,
        quarantine_root: Path | None = None,
    ) -> None:
        self._store = store
        self._calendar = calendar
        self._bar_validator = BarValidator(calendar)
        self._index_validator = IndexObservationValidator(calendar)
        self._quarantine_root = quarantine_root

    def ingest_equity_bars(
        self,
        source: Path,
        *,
        instrument_id: str | None = None,
        expected_start: dt.date | None = None,
        expected_end: dt.date | None = None,
        today: dt.date | None = None,
        quarantine: bool = False,
    ) -> IngestionResult:
        """Ingest one instrument's daily bars from a CSV/Parquet file.

        ``instrument_id`` defaults to the id carried in the file; passing it
        explicitly also asserts the file contains the instrument the caller
        expected, which catches mislabeled vendor drops.
        """
        frame = read_table(source)
        require_columns(frame, BAR_COLUMNS, str(source))
        bars = parse_bar_rows(frame, source=str(source))
        resolved_id = instrument_id or (bars[0].instrument_id if bars else None)
        if resolved_id is None:
            raise ValueError(f"{source} contains no rows and no instrument_id was given")

        report = self._bar_validator.validate(
            bars,
            instrument_id=instrument_id,
            expected_start=expected_start,
            expected_end=expected_end,
            today=today,
        )
        return self._store_bars(source, resolved_id, bars, report, quarantine)

    def ingest_index_observations(
        self,
        source: Path,
        *,
        index_symbol: str | None = None,
        expected_start: dt.date | None = None,
        expected_end: dt.date | None = None,
        quarantine: bool = False,
    ) -> IngestionResult:
        """Ingest a NIFTY 50 or India VIX series from a CSV/Parquet file."""
        frame = read_table(source)
        require_columns(frame, INDEX_COLUMNS, str(source))
        observations = parse_index_rows(frame, source=str(source))
        resolved_symbol = index_symbol or (
            observations[0].index_symbol if observations else None
        )
        if resolved_symbol is None:
            raise ValueError(f"{source} contains no rows and no index_symbol was given")

        report = self._index_validator.validate(
            observations,
            expected_start=expected_start,
            expected_end=expected_end,
        )
        return self._store_index(
            source, resolved_symbol, observations, report, quarantine
        )

    def ingest_instruments(self, source: Path) -> IngestionResult:
        """Ingest the instrument master into the reference layer."""
        frame = read_table(source)
        require_columns(frame, INSTRUMENT_COLUMNS, str(source))
        instruments = parse_instrument_rows(frame, source=str(source))
        report = validate_instruments(instruments)

        destination = self._store.instruments_path()
        if not report.is_usable:
            return IngestionResult(source, None, len(instruments), 0, report)
        write_table(frame, destination)
        return IngestionResult(
            source, destination, len(instruments), len(instruments), report
        )

    def ingest_corporate_actions(self, source: Path) -> IngestionResult:
        """Ingest corporate actions into the reference layer.

        Validates against the instrument master when one is already stored, so
        an action for an unlisted date is caught at ingest rather than
        surfacing as a wrong adjustment factor mid-backtest.
        """
        frame = read_table(source)
        require_columns(frame, CORPORATE_ACTION_COLUMNS, str(source))
        actions = parse_corporate_action_rows(frame, source=str(source))
        report = validate_corporate_actions(
            actions, calendar=self._calendar, instruments=self._instrument_lookup()
        )

        destination = self._store.corporate_actions_path()
        if not report.is_usable:
            return IngestionResult(source, None, len(actions), 0, report)
        write_table(frame, destination)
        return IngestionResult(source, destination, len(actions), len(actions), report)

    def ingest_index_membership(self, source: Path) -> IngestionResult:
        """Ingest point-in-time index membership into the reference layer.

        Overlap validation lives in the provider's constructor, so building one
        here is the validation: it raises on overlapping or open-ended-then-
        superseded spells.
        """
        frame = read_table(source)
        require_columns(frame, MEMBERSHIP_COLUMNS, str(source))
        memberships = parse_membership_rows(frame, source=str(source))
        InMemoryIndexMembershipProvider(memberships)

        destination = self._store.index_membership_path()
        write_table(frame, destination)
        return IngestionResult(
            source,
            destination,
            len(memberships),
            len(memberships),
            DataQualityReport(checked_rows=len(memberships)),
        )

    def _store_bars(
        self,
        source: Path,
        instrument_id: str,
        bars: list[DailyBar],
        report: DataQualityReport,
        quarantine: bool,
    ) -> IngestionResult:
        if not report.is_usable:
            if not quarantine:
                return IngestionResult(source, None, len(bars), 0, report)
            destination = self._quarantine_path(f"{instrument_id}_bars{source.suffix}")
            write_table(bars_to_frame(bars), destination)
            return IngestionResult(
                source, destination, len(bars), len(bars), report, quarantined=True
            )
        destination = self._store.equity_bars_path(instrument_id, DataLayer.RAW)
        write_table(bars_to_frame(bars), destination)
        return IngestionResult(source, destination, len(bars), len(bars), report)

    def _store_index(
        self,
        source: Path,
        index_symbol: str,
        observations: list[IndexObservation],
        report: DataQualityReport,
        quarantine: bool,
    ) -> IngestionResult:
        if not report.is_usable:
            if not quarantine:
                return IngestionResult(source, None, len(observations), 0, report)
            destination = self._quarantine_path(f"{index_symbol}_index{source.suffix}")
            write_table(index_observations_to_frame(observations), destination)
            return IngestionResult(
                source,
                destination,
                len(observations),
                len(observations),
                report,
                quarantined=True,
            )
        destination = self._store.index_path(index_symbol, DataLayer.RAW)
        write_table(index_observations_to_frame(observations), destination)
        return IngestionResult(
            source, destination, len(observations), len(observations), report
        )

    def _quarantine_path(self, filename: str) -> Path:
        if self._quarantine_root is None:
            raise ValueError(
                "quarantine was requested but the pipeline has no quarantine_root"
            )
        return self._quarantine_root / filename

    def _instrument_lookup(self) -> InstrumentLookup | None:
        """The stored instrument master, if one has been ingested yet."""
        path = self._store.instruments_path()
        if not path.is_file():
            return None
        repository = InMemoryInstrumentRepository.from_file(path)
        return InstrumentLookup(repository.all_versions())
