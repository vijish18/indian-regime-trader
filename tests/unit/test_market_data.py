"""Storage round-trips, the local market data provider, and corporate-action
adjustment as-of a decision date.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import pytest

from data.corporate_actions import InMemoryCorporateActionProvider
from data.errors import DataNotAvailableError
from data.market_data import (
    LocalMarketDataProvider,
    bars_to_frame,
    index_observations_to_frame,
    parse_bar_rows,
)
from data.models import (
    CorporateAction,
    CorporateActionType,
    IndexObservation,
    PriceBasis,
)
from data.storage import (
    DataLayer,
    LocalDataStore,
    StorageFormat,
    parse_date,
    parse_decimal,
    read_table,
    write_table,
)
from tests.conftest import make_bar

SERIES = [
    make_bar(dt.date(2024, 1, 2), "500.25", open_="499.10", high="501.00", low="498.05"),
    make_bar(dt.date(2024, 1, 3), "505.55", open_="500.30", high="506.00", low="500.00"),
    make_bar(dt.date(2024, 1, 4), "498.90", open_="505.00", high="505.50", low="498.00"),
]


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("storage_format", [StorageFormat.CSV, StorageFormat.PARQUET])
def test_bars_round_trip_without_losing_precision(
    tmp_path: Path, storage_format: StorageFormat
) -> None:
    """Decimal prices must survive storage exactly in both formats -- a price
    that changes by a float epsilon breaks tick-size and cost arithmetic.
    """
    path = tmp_path / f"bars{storage_format.suffix}"
    write_table(bars_to_frame(SERIES), path)
    restored = parse_bar_rows(read_table(path), source=str(path))
    assert restored == SERIES


@pytest.mark.parametrize("storage_format", [StorageFormat.CSV, StorageFormat.PARQUET])
def test_index_observations_round_trip(
    tmp_path: Path, storage_format: StorageFormat
) -> None:
    from data.market_data import parse_index_rows

    observations = [
        IndexObservation("INDIAVIX", dt.date(2024, 1, 2), Decimal("13.4500")),
        IndexObservation(
            "INDIAVIX",
            dt.date(2024, 1, 3),
            Decimal("14.0250"),
            open=Decimal("13.5000"),
            high=Decimal("14.1000"),
            low=Decimal("13.4000"),
        ),
    ]
    path = tmp_path / f"vix{storage_format.suffix}"
    write_table(index_observations_to_frame(observations), path)
    assert parse_index_rows(read_table(path), source=str(path)) == observations


def test_reading_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DataNotAvailableError, match="no data file"):
        read_table(tmp_path / "absent.csv")


def test_unsupported_format_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bars.xlsx"
    path.write_text("not really a spreadsheet", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported storage format"):
        read_table(path)


def test_store_layout_separates_raw_from_normalized(store: LocalDataStore) -> None:
    raw = store.equity_bars_path("NSE:INFY", DataLayer.RAW)
    normalized = store.equity_bars_path("NSE:INFY", DataLayer.NORMALIZED)
    assert raw != normalized
    assert raw.parent.parent.name == "raw"
    assert normalized.parent.parent.name == "normalized"


def test_store_sanitizes_symbols_that_are_illegal_filenames(
    store: LocalDataStore,
) -> None:
    """``M&M`` and slash-bearing series codes are real NSE symbols that would
    otherwise produce unusable paths on Windows.
    """
    path = store.equity_bars_path("NSE:M&M")
    assert "&" not in path.name
    assert ":" not in path.name


def test_parse_decimal_avoids_binary_float_noise() -> None:
    assert parse_decimal(0.1) == Decimal("0.1000")
    assert parse_decimal("100.05") == Decimal("100.0500")


def test_parse_date_rejects_impossible_dates() -> None:
    with pytest.raises(ValueError, match="ISO-8601"):
        parse_date("2024-02-31")
    with pytest.raises(ValueError, match="ISO-8601"):
        parse_date("02/01/2024")


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


def _seed(store: LocalDataStore) -> LocalMarketDataProvider:
    write_table(bars_to_frame(SERIES), store.equity_bars_path("NSE:INFY"))
    return LocalMarketDataProvider(store)


def test_provider_returns_bars_in_range_ascending(store: LocalDataStore) -> None:
    provider = _seed(store)
    bars = provider.get_equity_bars("NSE:INFY", dt.date(2024, 1, 2), dt.date(2024, 1, 3))
    assert [bar.session_date for bar in bars] == [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]


def test_provider_rejects_inverted_range(store: LocalDataStore) -> None:
    provider = _seed(store)
    with pytest.raises(ValueError, match="precedes"):
        provider.get_equity_bars("NSE:INFY", dt.date(2024, 1, 5), dt.date(2024, 1, 2))


def test_provider_reports_available_range(store: LocalDataStore) -> None:
    provider = _seed(store)
    assert provider.available_range("NSE:INFY") == (dt.date(2024, 1, 2), dt.date(2024, 1, 4))
    assert provider.available_range("NSE:UNKNOWN") is None


def test_provider_raises_for_unknown_instrument(store: LocalDataStore) -> None:
    provider = _seed(store)
    with pytest.raises(DataNotAvailableError):
        provider.get_equity_bars("NSE:UNKNOWN", dt.date(2024, 1, 2), dt.date(2024, 1, 4))


def test_historical_provider_has_no_live_quotes(store: LocalDataStore) -> None:
    """A backtest must never be able to consult a live book by accident."""
    provider = _seed(store)
    with pytest.raises(DataNotAvailableError, match="historical bars only"):
        provider.get_quote("NSE:INFY")


def test_adjusted_prices_require_a_corporate_action_provider(
    store: LocalDataStore,
) -> None:
    provider = _seed(store)
    with pytest.raises(DataNotAvailableError, match="CorporateActionProvider"):
        provider.get_equity_bars(
            "NSE:INFY",
            dt.date(2024, 1, 2),
            dt.date(2024, 1, 4),
            price_basis=PriceBasis.ADJUSTED,
        )


# --------------------------------------------------------------------------
# Corporate-action adjustment
# --------------------------------------------------------------------------


def _split_provider() -> InMemoryCorporateActionProvider:
    return InMemoryCorporateActionProvider(
        [
            CorporateAction(
                instrument_id="NSE:INFY",
                action_type=CorporateActionType.SPLIT,
                ex_date=dt.date(2024, 1, 4),
                ratio_new=Decimal(5),
                ratio_old=Decimal(1),
            )
        ]
    )


def test_cumulative_factor_applies_only_to_prices_before_the_ex_date() -> None:
    actions = _split_provider()
    as_of = dt.date(2024, 1, 10)
    assert actions.cumulative_adjustment_factor("NSE:INFY", dt.date(2024, 1, 3), as_of) == Decimal(
        "0.2"
    )
    # A price observed on the ex-date already reflects the split.
    on_ex_date = actions.cumulative_adjustment_factor("NSE:INFY", dt.date(2024, 1, 4), as_of)
    assert on_ex_date == Decimal(1)


def test_cumulative_factor_ignores_actions_after_the_as_of_date() -> None:
    """This is the look-ahead guard: on 2024-01-03 the split had not happened,
    so a decision made that day must see unadjusted prices.
    """
    actions = _split_provider()
    assert actions.cumulative_adjustment_factor(
        "NSE:INFY", dt.date(2024, 1, 2), dt.date(2024, 1, 3)
    ) == Decimal(1)


def test_cumulative_factor_compounds_multiple_actions() -> None:
    actions = InMemoryCorporateActionProvider(
        [
            CorporateAction(
                instrument_id="NSE:X",
                action_type=CorporateActionType.SPLIT,
                ex_date=dt.date(2024, 2, 1),
                ratio_new=Decimal(2),
                ratio_old=Decimal(1),
            ),
            CorporateAction(
                instrument_id="NSE:X",
                action_type=CorporateActionType.BONUS,
                ex_date=dt.date(2024, 3, 1),
                ratio_new=Decimal(1),
                ratio_old=Decimal(1),
            ),
        ]
    )
    factor = actions.cumulative_adjustment_factor(
        "NSE:X", dt.date(2024, 1, 15), dt.date(2024, 4, 1)
    )
    assert factor == Decimal("0.25")  # 1/2 for the split, then 1/2 for the bonus


def test_cumulative_factor_rejects_as_of_before_price_date() -> None:
    with pytest.raises(ValueError, match="backwards in time"):
        _split_provider().cumulative_adjustment_factor(
            "NSE:INFY", dt.date(2024, 1, 10), dt.date(2024, 1, 2)
        )


def test_provider_serves_adjusted_series_as_of_window_end(store: LocalDataStore) -> None:
    write_table(bars_to_frame(SERIES), store.equity_bars_path("NSE:INFY"))
    provider = LocalMarketDataProvider(store, _split_provider())

    adjusted = provider.get_equity_bars(
        "NSE:INFY",
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 4),
        price_basis=PriceBasis.ADJUSTED,
    )
    # Pre-split sessions are scaled by 1/5; the ex-date session is untouched.
    assert adjusted[0].close == Decimal("100.0500")
    assert adjusted[1].close == Decimal("101.1100")
    assert adjusted[2].close == Decimal("498.9000")
    assert adjusted[0].price_basis is PriceBasis.ADJUSTED


def test_window_ending_before_the_split_sees_unadjusted_prices(
    store: LocalDataStore,
) -> None:
    """The same historical bar is served unadjusted to a fold that ended before
    the split and adjusted to one that ended after it. That asymmetry is the
    whole point.
    """
    write_table(bars_to_frame(SERIES), store.equity_bars_path("NSE:INFY"))
    provider = LocalMarketDataProvider(store, _split_provider())

    before = provider.get_equity_bars(
        "NSE:INFY",
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
        price_basis=PriceBasis.ADJUSTED,
    )
    assert before[0].close == Decimal("500.25")
    assert before[0].price_basis is PriceBasis.RAW


def test_identity_change_is_surfaced() -> None:
    actions = InMemoryCorporateActionProvider(
        [
            CorporateAction(
                instrument_id="NSE:OLD",
                action_type=CorporateActionType.MERGER,
                ex_date=dt.date(2024, 6, 1),
                explicit_price_factor=Decimal(1),
                successor_instrument_id="NSE:NEW",
            )
        ]
    )
    change = actions.identity_change("NSE:OLD", dt.date(2024, 1, 1))
    assert change is not None
    assert change.successor_instrument_id == "NSE:NEW"
    assert actions.identity_change("NSE:OLD", dt.date(2024, 12, 1)) is None


def test_dividends_are_listed_separately_from_price_adjustments() -> None:
    actions = InMemoryCorporateActionProvider(
        [
            CorporateAction(
                instrument_id="NSE:INFY",
                action_type=CorporateActionType.DIVIDEND,
                ex_date=dt.date(2024, 5, 2),
                cash_amount=Decimal("18.00"),
            )
        ]
    )
    dividends = actions.cash_dividends("NSE:INFY", dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    assert [action.cash_amount for action in dividends] == [Decimal("18.0000")]
    assert actions.cumulative_adjustment_factor(
        "NSE:INFY", dt.date(2024, 1, 1), dt.date(2024, 12, 31)
    ) == Decimal(1)
