"""Point-in-time instrument lookups, duplicate detection, and invalid dates."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import pytest

from data.errors import InstrumentNotFoundError
from data.instrument_master import InMemoryInstrumentRepository
from data.models import Exchange, Instrument, InstrumentStatus, Segment

HEADER = (
    "instrument_id,symbol,exchange,segment,tick_size,price_precision,"
    "effective_from,effective_to,isin,lot_size,status,name\n"
)


def _instrument(
    instrument_id: str = "NSE:INFY",
    symbol: str = "INFY",
    *,
    effective_from: dt.date = dt.date(2020, 1, 1),
    effective_to: dt.date | None = None,
    tick_size: str = "0.05",
    isin: str | None = "INE009A01021",
    status: InstrumentStatus = InstrumentStatus.ACTIVE,
) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        symbol=symbol,
        exchange=Exchange.NSE,
        segment=Segment.EQUITY,
        tick_size=Decimal(tick_size),
        price_precision=2,
        effective_from=effective_from,
        effective_to=effective_to,
        isin=isin,
        lot_size=1,
        status=status,
    )


def test_get_returns_the_version_in_force_on_the_date() -> None:
    """Reference data is versioned, so a 2021 backtest must see the 2021 tick
    size rather than today's.
    """
    repository = InMemoryInstrumentRepository(
        [
            _instrument(
                effective_from=dt.date(2020, 1, 1),
                effective_to=dt.date(2022, 12, 31),
                tick_size="0.05",
            ),
            _instrument(effective_from=dt.date(2023, 1, 1), tick_size="0.01"),
        ]
    )
    assert repository.get("NSE:INFY", dt.date(2021, 6, 1)).tick_size == Decimal("0.05")
    assert repository.get("NSE:INFY", dt.date(2024, 6, 1)).tick_size == Decimal("0.01")


def test_get_raises_before_the_instrument_existed() -> None:
    repository = InMemoryInstrumentRepository([_instrument()])
    with pytest.raises(InstrumentNotFoundError, match="effective on"):
        repository.get("NSE:INFY", dt.date(2019, 1, 1))


def test_get_raises_for_unknown_instrument() -> None:
    repository = InMemoryInstrumentRepository([_instrument()])
    with pytest.raises(InstrumentNotFoundError):
        repository.get("NSE:NOPE", dt.date(2021, 1, 1))


def test_symbol_lookup_is_point_in_time() -> None:
    """A symbol can be reassigned to a different company after a delisting, so
    resolving it without a date would silently return the wrong instrument.
    """
    repository = InMemoryInstrumentRepository(
        [
            _instrument(
                instrument_id="NSE:OLDCO",
                symbol="ABC",
                effective_from=dt.date(2015, 1, 1),
                effective_to=dt.date(2018, 12, 31),
                isin="INE111A01011",
            ),
            _instrument(
                instrument_id="NSE:NEWCO",
                symbol="ABC",
                effective_from=dt.date(2021, 1, 1),
                isin="INE222A01012",
            ),
        ]
    )
    assert repository.get_by_symbol("ABC", "NSE", dt.date(2016, 6, 1)).instrument_id == "NSE:OLDCO"
    assert repository.get_by_symbol("ABC", "NSE", dt.date(2022, 6, 1)).instrument_id == "NSE:NEWCO"
    with pytest.raises(InstrumentNotFoundError):
        repository.get_by_symbol("ABC", "NSE", dt.date(2019, 6, 1))


def test_duplicate_instrument_versions_are_rejected_at_construction() -> None:
    """Two records covering the same day make every lookup order-dependent."""
    with pytest.raises(ValueError, match="overlapping effective ranges"):
        InMemoryInstrumentRepository(
            [
                _instrument(
                    effective_from=dt.date(2020, 1, 1), effective_to=dt.date(2022, 12, 31)
                ),
                _instrument(
                    effective_from=dt.date(2022, 1, 1), effective_to=dt.date(2023, 12, 31)
                ),
            ]
        )


def test_open_ended_version_followed_by_another_is_rejected() -> None:
    with pytest.raises(ValueError, match="open-ended"):
        InMemoryInstrumentRepository(
            [
                _instrument(effective_from=dt.date(2020, 1, 1), effective_to=None),
                _instrument(effective_from=dt.date(2023, 1, 1)),
            ]
        )


def test_list_effective_filters_by_date_and_segment() -> None:
    repository = InMemoryInstrumentRepository(
        [
            _instrument(instrument_id="NSE:A", symbol="A", isin="INE111A01011"),
            _instrument(
                instrument_id="NSE:B",
                symbol="B",
                effective_from=dt.date(2023, 1, 1),
                isin="INE222A01012",
            ),
        ]
    )
    assert [i.instrument_id for i in repository.list_effective(dt.date(2021, 1, 1))] == [
        "NSE:A"
    ]
    assert [i.instrument_id for i in repository.list_effective(dt.date(2023, 6, 1))] == [
        "NSE:A",
        "NSE:B",
    ]
    assert repository.list_effective(dt.date(2023, 6, 1), segment=Segment.INDEX) == []


def test_history_returns_every_version_ascending() -> None:
    repository = InMemoryInstrumentRepository(
        [
            _instrument(effective_from=dt.date(2023, 1, 1)),
            _instrument(
                effective_from=dt.date(2020, 1, 1), effective_to=dt.date(2022, 12, 31)
            ),
        ]
    )
    history = repository.history("NSE:INFY")
    assert [record.effective_from for record in history] == [
        dt.date(2020, 1, 1),
        dt.date(2023, 1, 1),
    ]


def test_from_file_parses_full_metadata(tmp_path: Path) -> None:
    path = tmp_path / "instruments.csv"
    path.write_text(
        HEADER
        + "NSE:INFY,INFY,NSE,equity,0.05,2,2020-01-01,,INE009A01021,1,active,Infosys\n"
        + "NSE:NIFTY50,NIFTY50,NSE,index,0.05,2,2020-01-01,,,,active,Nifty 50\n",
        encoding="utf-8",
    )
    repository = InMemoryInstrumentRepository.from_file(path)
    infy = repository.get("NSE:INFY", dt.date(2024, 1, 2))
    assert infy.isin == "INE009A01021"
    assert infy.lot_size == 1
    assert infy.segment is Segment.EQUITY

    index = repository.get("NSE:NIFTY50", dt.date(2024, 1, 2))
    assert index.segment is Segment.INDEX
    assert index.isin is None
    assert index.lot_size is None


def test_from_file_reports_the_offending_line(tmp_path: Path) -> None:
    path = tmp_path / "instruments.csv"
    path.write_text(
        HEADER
        + "NSE:INFY,INFY,NSE,equity,0.05,2,2020-01-01,,INE009A01021,1,active,Infosys\n"
        + "NSE:BAD,BAD,NSE,equity,-1,2,2020-01-01,,,1,active,Bad Tick\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 3"):
        InMemoryInstrumentRepository.from_file(path)


def test_from_file_rejects_invalid_effective_dates(tmp_path: Path) -> None:
    path = tmp_path / "instruments.csv"
    path.write_text(
        HEADER
        + "NSE:INFY,INFY,NSE,equity,0.05,2,2024-06-01,2024-01-01,,1,active,Inverted\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="precedes"):
        InMemoryInstrumentRepository.from_file(path)


def test_from_file_requires_mandatory_columns(tmp_path: Path) -> None:
    path = tmp_path / "instruments.csv"
    path.write_text("instrument_id,symbol\nNSE:INFY,INFY\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        InMemoryInstrumentRepository.from_file(path)


def test_freshness_check_fails_closed_without_a_snapshot_date() -> None:
    repository = InMemoryInstrumentRepository([_instrument()])
    assert not repository.is_fresh(max_age_days=1, reference_date=dt.date(2024, 1, 2))

    dated = InMemoryInstrumentRepository([_instrument()], snapshot_date=dt.date(2024, 1, 2))
    assert dated.is_fresh(1, dt.date(2024, 1, 3))
    assert not dated.is_fresh(1, dt.date(2024, 1, 5))
