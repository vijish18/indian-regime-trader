"""Versioned cost-schedule storage: point-in-time selection by effective
date, fail-closed behavior for dates with no covering schedule, and the
shipped data file.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import yaml

from backtest.cost_schedule import (
    CostSchedule,
    CostScheduleFileError,
    CostScheduleRepository,
    MissingCostScheduleError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def schedule(effective_from: dt.date, **overrides: object) -> CostSchedule:
    defaults: dict[str, object] = dict(
        effective_from=effective_from,
        label="test schedule",
        source="unit test fixture",
        brokerage_flat_inr=0.0,
        brokerage_pct=0.0,
        stt_buy_pct=0.001,
        stt_sell_pct=0.001,
        exchange_txn_pct=0.0000345,
        sebi_turnover_pct=0.0000010,
        gst_pct=0.18,
        stamp_duty_buy_pct=0.00015,
        stamp_duty_sell_pct=0.0,
        dp_charges_inr=15.93,
        other_charges_flat_inr=0.0,
        other_charges_pct=0.0,
    )
    defaults.update(overrides)
    return CostSchedule(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# CostSchedule validation
# --------------------------------------------------------------------------


def test_schedule_rejects_negative_rate() -> None:
    with pytest.raises(ValueError, match="brokerage_pct"):
        schedule(dt.date(2020, 1, 1), brokerage_pct=-0.001)


def test_schedule_rejects_empty_label() -> None:
    with pytest.raises(ValueError, match="label"):
        schedule(dt.date(2020, 1, 1), label="")


def test_schedule_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="source"):
        schedule(dt.date(2020, 1, 1), source="")


# --------------------------------------------------------------------------
# Repository construction
# --------------------------------------------------------------------------


def test_repository_requires_at_least_one_schedule() -> None:
    with pytest.raises(CostScheduleFileError, match="at least one"):
        CostScheduleRepository([])


def test_repository_rejects_duplicate_effective_from() -> None:
    with pytest.raises(CostScheduleFileError, match="duplicate"):
        CostScheduleRepository(
            [schedule(dt.date(2020, 1, 1)), schedule(dt.date(2020, 1, 1), brokerage_pct=0.01)]
        )


def test_repository_orders_schedules_regardless_of_input_order() -> None:
    later = schedule(dt.date(2022, 1, 1))
    earlier = schedule(dt.date(2020, 1, 1))
    repo = CostScheduleRepository([later, earlier])
    assert [s.effective_from for s in repo.all_schedules()] == [
        dt.date(2020, 1, 1),
        dt.date(2022, 1, 1),
    ]


# --------------------------------------------------------------------------
# Effective-date selection ("effective-date changes")
# --------------------------------------------------------------------------


def test_schedule_as_of_selects_most_recent_covering_entry() -> None:
    old = schedule(dt.date(2019, 1, 1), stt_buy_pct=0.00125, label="pre-2020")
    new = schedule(dt.date(2020, 7, 1), stt_buy_pct=0.00100, label="post-2020")
    repo = CostScheduleRepository([old, new])

    assert repo.schedule_as_of(dt.date(2019, 6, 1)) is old
    assert repo.schedule_as_of(dt.date(2020, 6, 30)) is old
    assert repo.schedule_as_of(dt.date(2020, 7, 1)) is new
    assert repo.schedule_as_of(dt.date(2025, 1, 1)) is new


def test_schedule_as_of_is_exclusive_of_dates_before_effective_from() -> None:
    early = schedule(dt.date(2019, 1, 1))
    late = schedule(dt.date(2023, 1, 1))
    repo = CostScheduleRepository([early, late])
    assert repo.schedule_as_of(dt.date(2022, 12, 31)) is early


def test_three_schedules_select_correctly_across_all_boundaries() -> None:
    s1 = schedule(dt.date(2018, 1, 1), label="s1")
    s2 = schedule(dt.date(2020, 1, 1), label="s2")
    s3 = schedule(dt.date(2023, 1, 1), label="s3")
    repo = CostScheduleRepository([s1, s2, s3])

    assert repo.schedule_as_of(dt.date(2019, 1, 1)).label == "s1"
    assert repo.schedule_as_of(dt.date(2020, 1, 1)).label == "s2"
    assert repo.schedule_as_of(dt.date(2022, 12, 31)).label == "s2"
    assert repo.schedule_as_of(dt.date(2023, 6, 1)).label == "s3"


# --------------------------------------------------------------------------
# Missing rates
# --------------------------------------------------------------------------


def test_schedule_as_of_fails_closed_before_earliest_schedule() -> None:
    repo = CostScheduleRepository([schedule(dt.date(2022, 1, 1))])
    with pytest.raises(MissingCostScheduleError, match="2022-01-01"):
        repo.schedule_as_of(dt.date(2021, 12, 31))


def test_missing_rates_error_is_distinct_from_file_error() -> None:
    """A date with no covering schedule and a malformed data file are
    different failure modes with different exception types."""
    repo = CostScheduleRepository([schedule(dt.date(2022, 1, 1))])
    with pytest.raises(MissingCostScheduleError):
        repo.schedule_as_of(dt.date(2000, 1, 1))
    assert not issubclass(MissingCostScheduleError, CostScheduleFileError)


# --------------------------------------------------------------------------
# Loading from file
# --------------------------------------------------------------------------


def _write_schedule_file(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(yaml.safe_dump({"schedules": entries}), encoding="utf-8")


def _entry(effective_from: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "effective_from": effective_from,
        "label": "test",
        "source": "test",
        "brokerage_flat_inr": 0.0,
        "brokerage_pct": 0.0,
        "stt_buy_pct": 0.001,
        "stt_sell_pct": 0.001,
        "exchange_txn_pct": 0.0000345,
        "sebi_turnover_pct": 0.0000010,
        "gst_pct": 0.18,
        "stamp_duty_buy_pct": 0.00015,
        "stamp_duty_sell_pct": 0.0,
        "dp_charges_inr": 15.93,
        "other_charges_flat_inr": 0.0,
        "other_charges_pct": 0.0,
    }
    base.update(overrides)
    return base


def test_from_file_loads_a_valid_schedule_file(tmp_path: Path) -> None:
    path = tmp_path / "cost_schedules.yaml"
    _write_schedule_file(path, [_entry("2020-01-01"), _entry("2023-01-01", stt_buy_pct=0.0009)])

    repo = CostScheduleRepository.from_file(path)

    assert len(repo.all_schedules()) == 2
    assert repo.schedule_as_of(dt.date(2024, 1, 1)).stt_buy_pct == 0.0009


def test_from_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CostScheduleFileError, match="no cost schedule file"):
        CostScheduleRepository.from_file(tmp_path / "does_not_exist.yaml")


def test_from_file_rejects_empty_schedules_list(tmp_path: Path) -> None:
    path = tmp_path / "cost_schedules.yaml"
    path.write_text(yaml.safe_dump({"schedules": []}), encoding="utf-8")
    with pytest.raises(CostScheduleFileError, match="non-empty"):
        CostScheduleRepository.from_file(path)


def test_from_file_rejects_missing_top_level_key(tmp_path: Path) -> None:
    path = tmp_path / "cost_schedules.yaml"
    path.write_text(yaml.safe_dump({"not_schedules": []}), encoding="utf-8")
    with pytest.raises(CostScheduleFileError, match="schedules"):
        CostScheduleRepository.from_file(path)


def test_from_file_rejects_entry_missing_a_required_field(tmp_path: Path) -> None:
    entry = _entry("2020-01-01")
    del entry["stt_buy_pct"]
    path = tmp_path / "cost_schedules.yaml"
    _write_schedule_file(path, [entry])
    with pytest.raises(CostScheduleFileError, match="stt_buy_pct"):
        CostScheduleRepository.from_file(path)


def test_from_file_rejects_malformed_date(tmp_path: Path) -> None:
    path = tmp_path / "cost_schedules.yaml"
    _write_schedule_file(path, [_entry("not-a-date")])
    with pytest.raises(CostScheduleFileError):
        CostScheduleRepository.from_file(path)


def test_shipped_cost_schedule_file_loads_successfully() -> None:
    path = PROJECT_ROOT / "config" / "cost_schedules.yaml"
    repo = CostScheduleRepository.from_file(path)
    assert len(repo.all_schedules()) >= 1
    schedule_at = repo.schedule_as_of(dt.date.today())
    assert schedule_at.gst_pct == pytest.approx(0.18)


def test_the_shipped_schedule_covers_the_whole_backfill_period() -> None:
    """Regression: the walk-forward run over 2018-2026 died on its first
    fold because the only shipped entry began 2019-01-01.

    Failing closed was correct -- pricing a 2018 trade with 2019 rates is
    exactly what this module exists to prevent -- but the gap should be
    caught here, in a second, rather than by a multi-hour backtest.

    The bhavcopy backfill starts 2015-01-01, so that is the earliest date
    any backtest can reach.
    """
    repo = CostScheduleRepository.from_file(PROJECT_ROOT / "config" / "cost_schedules.yaml")

    repo.schedule_as_of(dt.date(2015, 1, 1))  # raises if uncovered


def test_the_shipped_schedule_prices_each_era_with_its_own_rates() -> None:
    """The point of a dated file is that history is not repriced with
    today's card. These three are the changes large enough to matter.

    Stamp duty was state-wise until the exchanges began collecting it
    uniformly on 2020-07-01; service tax became GST on 2017-07-01; and the
    NSE cash charge moved four times between 2021 and 2024.
    """
    repo = CostScheduleRepository.from_file(PROJECT_ROOT / "config" / "cost_schedules.yaml")

    assert repo.schedule_as_of(dt.date(2016, 1, 4)).gst_pct == pytest.approx(0.145)
    assert repo.schedule_as_of(dt.date(2018, 1, 23)).gst_pct == pytest.approx(0.18)

    assert repo.schedule_as_of(dt.date(2020, 6, 30)).stamp_duty_buy_pct == pytest.approx(0.0001)
    assert repo.schedule_as_of(dt.date(2020, 7, 1)).stamp_duty_buy_pct == pytest.approx(0.00015)

    assert repo.schedule_as_of(dt.date(2020, 12, 31)).exchange_txn_pct == pytest.approx(0.0000325)
    assert repo.schedule_as_of(dt.date(2021, 1, 4)).exchange_txn_pct == pytest.approx(0.0000345)
    assert repo.schedule_as_of(dt.date(2024, 10, 1)).exchange_txn_pct == pytest.approx(0.0000297)


def test_delivery_stt_is_unchanged_across_every_shipped_era() -> None:
    """STT is ~20 bps of a round trip -- an order of magnitude more than
    every other statutory component combined -- and delivery equity has
    been charged 0.1% on both legs throughout this period. If a future
    edit makes it drift by era, that is either a real change worth a
    citation or a typo worth catching here.
    """
    repo = CostScheduleRepository.from_file(PROJECT_ROOT / "config" / "cost_schedules.yaml")

    for schedule in repo.all_schedules():
        assert schedule.stt_buy_pct == pytest.approx(0.001), schedule.effective_from
        assert schedule.stt_sell_pct == pytest.approx(0.001), schedule.effective_from
