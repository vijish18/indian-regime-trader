"""Point-in-time index membership, and the ingest pipeline's accept/reject
behavior.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from data.calendar import NSETradingCalendar
from data.data_quality import IssueCode
from data.ingestion import DataIngestionPipeline
from data.instrument_master import InMemoryInstrumentRepository
from data.market_data import bars_to_frame
from data.membership import InMemoryIndexMembershipProvider
from data.models import IndexMembership
from data.storage import LocalDataStore, read_table, write_table
from tests.conftest import make_bar

BAR_HEADER = "instrument_id,session_date,open,high,low,close,volume\n"


# --------------------------------------------------------------------------
# Point-in-time membership
# --------------------------------------------------------------------------


def test_membership_is_resolved_as_of_a_date() -> None:
    """The survivorship-bias control: a company deleted from the index in 2022
    must still appear as a member for 2021 dates.
    """
    provider = InMemoryIndexMembershipProvider(
        [
            IndexMembership("NIFTY50", "NSE:SURVIVOR", dt.date(2015, 1, 1)),
            IndexMembership(
                "NIFTY50",
                "NSE:DELETED",
                dt.date(2015, 1, 1),
                effective_to=dt.date(2022, 3, 31),
                exclusion_reason="index review",
            ),
            IndexMembership("NIFTY50", "NSE:ADDED", dt.date(2022, 4, 1)),
        ]
    )
    assert provider.members_on("NIFTY50", dt.date(2021, 6, 1)) == {
        "NSE:SURVIVOR",
        "NSE:DELETED",
    }
    assert provider.members_on("NIFTY50", dt.date(2023, 6, 1)) == {
        "NSE:SURVIVOR",
        "NSE:ADDED",
    }


def test_membership_changes_between_dates() -> None:
    provider = InMemoryIndexMembershipProvider(
        [
            IndexMembership(
                "NIFTY50", "NSE:DELETED", dt.date(2015, 1, 1), effective_to=dt.date(2022, 3, 31)
            ),
            IndexMembership("NIFTY50", "NSE:ADDED", dt.date(2022, 4, 1)),
        ]
    )
    added, removed = provider.changes_between(
        "NIFTY50", dt.date(2022, 1, 1), dt.date(2022, 6, 1)
    )
    assert added == {"NSE:ADDED"}
    assert removed == {"NSE:DELETED"}


def test_readmitted_instrument_keeps_both_spells() -> None:
    """Indices do re-admit companies they deleted, so membership is a list of
    intervals rather than one range per instrument.
    """
    provider = InMemoryIndexMembershipProvider(
        [
            IndexMembership(
                "NIFTY50", "NSE:X", dt.date(2015, 1, 1), effective_to=dt.date(2018, 12, 31)
            ),
            IndexMembership("NIFTY50", "NSE:X", dt.date(2021, 1, 1)),
        ]
    )
    assert provider.members_on("NIFTY50", dt.date(2016, 1, 1)) == {"NSE:X"}
    assert provider.members_on("NIFTY50", dt.date(2019, 6, 1)) == frozenset()
    assert provider.members_on("NIFTY50", dt.date(2022, 1, 1)) == {"NSE:X"}


def test_overlapping_membership_spells_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlapping membership"):
        InMemoryIndexMembershipProvider(
            [
                IndexMembership(
                    "NIFTY50", "NSE:X", dt.date(2015, 1, 1), effective_to=dt.date(2019, 12, 31)
                ),
                IndexMembership("NIFTY50", "NSE:X", dt.date(2018, 1, 1)),
            ]
        )


def test_unknown_index_returns_empty_membership() -> None:
    provider = InMemoryIndexMembershipProvider([])
    assert provider.members_on("NIFTY50", dt.date(2024, 1, 2)) == frozenset()


def test_membership_loads_from_file(tmp_path: Path) -> None:
    path = tmp_path / "membership.csv"
    path.write_text(
        "index_symbol,instrument_id,effective_from,effective_to,exclusion_reason\n"
        "NIFTY50,NSE:INFY,2015-01-01,,\n"
        "NIFTY50,NSE:GONE,2015-01-01,2022-03-31,index review\n",
        encoding="utf-8",
    )
    provider = InMemoryIndexMembershipProvider.from_file(path)
    assert provider.members_on("NIFTY50", dt.date(2021, 1, 1)) == {"NSE:INFY", "NSE:GONE"}
    assert provider.members_on("NIFTY50", dt.date(2023, 1, 1)) == {"NSE:INFY"}


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


@pytest.fixture
def pipeline(store: LocalDataStore, calendar: NSETradingCalendar) -> DataIngestionPipeline:
    return DataIngestionPipeline(store, calendar)


def test_clean_bars_are_ingested_and_stored(
    pipeline: DataIngestionPipeline, store: LocalDataStore, tmp_path: Path
) -> None:
    source = tmp_path / "infy.csv"
    write_table(
        bars_to_frame(
            [make_bar(dt.date(2024, 1, 2), "100"), make_bar(dt.date(2024, 1, 3), "101")]
        ),
        source,
    )
    result = pipeline.ingest_equity_bars(source, instrument_id="NSE:INFY")

    assert result.accepted
    assert result.records_stored == 2
    assert result.destination == store.equity_bars_path("NSE:INFY")
    assert result.destination.is_file()


def test_bars_with_errors_are_rejected_and_not_stored(
    pipeline: DataIngestionPipeline, store: LocalDataStore, tmp_path: Path
) -> None:
    """Rejecting rather than storing-and-flagging is what lets later phases
    treat everything under raw/ as structurally valid.
    """
    source = tmp_path / "infy.csv"
    source.write_text(
        BAR_HEADER
        + "NSE:INFY,2024-01-02,100,101,99,100,1000\n"
        + "NSE:INFY,2024-01-02,100,101,99,100,1000\n",  # duplicate session
        encoding="utf-8",
    )
    result = pipeline.ingest_equity_bars(source, instrument_id="NSE:INFY")

    assert not result.accepted
    assert result.records_stored == 0
    assert result.destination is None
    assert IssueCode.DUPLICATE_BAR in result.report.codes()
    assert not store.equity_bars_path("NSE:INFY").exists()


def test_quarantine_writes_bad_data_away_from_good_data(
    store: LocalDataStore, calendar: NSETradingCalendar, tmp_path: Path
) -> None:
    quarantine = tmp_path / "quarantine"
    pipeline = DataIngestionPipeline(store, calendar, quarantine_root=quarantine)
    source = tmp_path / "infy.csv"
    source.write_text(
        BAR_HEADER + "NSE:INFY,2024-01-02,100,95,99,98,1000\n",  # high < low
        encoding="utf-8",
    )
    result = pipeline.ingest_equity_bars(source, instrument_id="NSE:INFY", quarantine=True)

    assert result.quarantined
    assert not result.accepted
    assert result.destination is not None
    assert quarantine in result.destination.parents
    assert not store.equity_bars_path("NSE:INFY").exists()


def test_missing_sessions_block_ingest(
    pipeline: DataIngestionPipeline, tmp_path: Path
) -> None:
    source = tmp_path / "infy.csv"
    source.write_text(
        BAR_HEADER
        + "NSE:INFY,2024-01-02,100,101,99,100,1000\n"
        + "NSE:INFY,2024-01-04,100,101,99,100,1000\n",  # 3rd is missing
        encoding="utf-8",
    )
    result = pipeline.ingest_equity_bars(source, instrument_id="NSE:INFY")
    assert not result.accepted
    assert IssueCode.MISSING_SESSION in result.report.codes()


def test_mislabeled_file_is_rejected(
    pipeline: DataIngestionPipeline, tmp_path: Path
) -> None:
    source = tmp_path / "tcs.csv"
    source.write_text(
        BAR_HEADER + "NSE:TCS,2024-01-02,100,101,99,100,1000\n", encoding="utf-8"
    )
    result = pipeline.ingest_equity_bars(source, instrument_id="NSE:INFY")
    assert not result.accepted
    assert IssueCode.MIXED_INSTRUMENTS in result.report.codes()


def test_missing_required_columns_raise(
    pipeline: DataIngestionPipeline, tmp_path: Path
) -> None:
    source = tmp_path / "infy.csv"
    source.write_text("instrument_id,session_date\nNSE:INFY,2024-01-02\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        pipeline.ingest_equity_bars(source)


def test_index_series_is_ingested(
    pipeline: DataIngestionPipeline, store: LocalDataStore, tmp_path: Path
) -> None:
    source = tmp_path / "vix.csv"
    source.write_text(
        "index_symbol,session_date,close\n"
        "INDIAVIX,2024-01-02,13.45\n"
        "INDIAVIX,2024-01-03,13.90\n",
        encoding="utf-8",
    )
    result = pipeline.ingest_index_observations(source)
    assert result.accepted
    assert result.destination == store.index_path("INDIAVIX")


def test_instruments_are_ingested_into_reference_layer(
    pipeline: DataIngestionPipeline, store: LocalDataStore, tmp_path: Path
) -> None:
    source = tmp_path / "instruments.csv"
    source.write_text(
        "instrument_id,symbol,exchange,segment,tick_size,price_precision,effective_from,isin\n"
        "NSE:INFY,INFY,NSE,equity,0.05,2,2020-01-01,INE009A01021\n",
        encoding="utf-8",
    )
    result = pipeline.ingest_instruments(source)
    assert result.accepted
    assert store.instruments_path().is_file()

    repository = InMemoryInstrumentRepository.from_file(store.instruments_path())
    assert repository.get("NSE:INFY", dt.date(2024, 1, 2)).symbol == "INFY"


def test_duplicate_instruments_block_ingest(
    pipeline: DataIngestionPipeline, store: LocalDataStore, tmp_path: Path
) -> None:
    source = tmp_path / "instruments.csv"
    source.write_text(
        "instrument_id,symbol,exchange,segment,tick_size,price_precision,effective_from,isin\n"
        "NSE:INFY,INFY,NSE,equity,0.05,2,2020-01-01,INE009A01021\n"
        "NSE:INFY,INFY,NSE,equity,0.05,2,2020-01-01,INE009A01021\n",
        encoding="utf-8",
    )
    result = pipeline.ingest_instruments(source)
    assert not result.accepted
    assert IssueCode.DUPLICATE_INSTRUMENT in result.report.codes()
    assert not store.instruments_path().exists()


def test_corporate_actions_are_validated_against_the_instrument_master(
    pipeline: DataIngestionPipeline, store: LocalDataStore, tmp_path: Path
) -> None:
    instruments = tmp_path / "instruments.csv"
    instruments.write_text(
        "instrument_id,symbol,exchange,segment,tick_size,price_precision,effective_from\n"
        "NSE:INFY,INFY,NSE,equity,0.05,2,2020-01-01\n",
        encoding="utf-8",
    )
    pipeline.ingest_instruments(instruments)

    actions = tmp_path / "actions.csv"
    actions.write_text(
        "instrument_id,action_type,ex_date,ratio_new,ratio_old\n"
        "NSE:INFY,split,2019-05-01,5,1\n",  # before listing
        encoding="utf-8",
    )
    result = pipeline.ingest_corporate_actions(actions)
    assert not result.accepted
    assert IssueCode.ACTION_BEFORE_LISTING in result.report.codes()


def test_valid_corporate_actions_are_stored(
    pipeline: DataIngestionPipeline, store: LocalDataStore, tmp_path: Path
) -> None:
    actions = tmp_path / "actions.csv"
    actions.write_text(
        "instrument_id,action_type,ex_date,ratio_new,ratio_old\n"
        "NSE:INFY,split,2024-01-03,5,1\n",
        encoding="utf-8",
    )
    result = pipeline.ingest_corporate_actions(actions)
    assert result.accepted
    assert store.corporate_actions_path().is_file()


def test_membership_ingest_rejects_overlaps(
    pipeline: DataIngestionPipeline, tmp_path: Path
) -> None:
    source = tmp_path / "membership.csv"
    source.write_text(
        "index_symbol,instrument_id,effective_from,effective_to\n"
        "NIFTY50,NSE:X,2015-01-01,2019-12-31\n"
        "NIFTY50,NSE:X,2018-01-01,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlapping membership"):
        pipeline.ingest_index_membership(source)


def test_membership_ingest_stores_valid_records(
    pipeline: DataIngestionPipeline, store: LocalDataStore, tmp_path: Path
) -> None:
    source = tmp_path / "membership.csv"
    source.write_text(
        "index_symbol,instrument_id,effective_from,effective_to\n"
        "NIFTY50,NSE:INFY,2015-01-01,\n",
        encoding="utf-8",
    )
    result = pipeline.ingest_index_membership(source)
    assert result.accepted
    assert read_table(store.index_membership_path()).shape[0] == 1


def test_result_summary_is_readable(
    pipeline: DataIngestionPipeline, tmp_path: Path
) -> None:
    source = tmp_path / "infy.csv"
    source.write_text(
        BAR_HEADER + "NSE:INFY,2024-01-02,100,101,99,100,1000\n", encoding="utf-8"
    )
    result = pipeline.ingest_equity_bars(source, instrument_id="NSE:INFY")
    assert "infy.csv" in result.summary()
    assert "stored" in result.summary()
