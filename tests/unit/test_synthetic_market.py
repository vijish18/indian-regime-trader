"""Unit tests for ``validation/synthetic_market.py`` (Phase 21): the
generated vendor drop is internally consistent, weekday-only, and
ingests cleanly through the real pipeline (the ingestion round-trip
itself is exercised end to end in ``validation.harness``'s own tests;
this file checks what's actually written to disk).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from data.calendar import NSETradingCalendar
from data.ingestion import DataIngestionPipeline
from data.storage import LocalDataStore, StorageFormat
from validation.synthetic_market import (
    INDEX_SYMBOL,
    VIX_SYMBOL,
    write_synthetic_market,
)


def test_writes_one_bar_file_per_instrument(tmp_path: Path) -> None:
    market = write_synthetic_market(tmp_path, sessions=40, instruments=4)
    assert len(market.bar_files) == 4
    assert len(market.instrument_ids) == 4
    for path in market.bar_files:
        assert path.is_file()


def test_sessions_are_weekdays_only(tmp_path: Path) -> None:
    market = write_synthetic_market(tmp_path, sessions=60, start=dt.date(2024, 1, 1))
    assert all(day.weekday() < 5 for day in market.sessions)
    assert len(market.sessions) == 60


def test_sessions_are_contiguous_weekdays_in_order(tmp_path: Path) -> None:
    market = write_synthetic_market(tmp_path, sessions=30, start=dt.date(2024, 1, 1))
    assert list(market.sessions) == sorted(market.sessions)
    assert len(set(market.sessions)) == len(market.sessions)


def test_first_and_last_session_properties(tmp_path: Path) -> None:
    market = write_synthetic_market(tmp_path, sessions=20)
    assert market.first_session == market.sessions[0]
    assert market.last_session == market.sessions[-1]


def test_bar_file_has_the_required_columns_and_row_count(tmp_path: Path) -> None:
    market = write_synthetic_market(tmp_path, sessions=25, instruments=1)
    frame = pd.read_csv(market.bar_files[0])
    for column in ("instrument_id", "session_date", "open", "high", "low", "close", "volume"):
        assert column in frame.columns
    assert len(frame) == 25


def test_high_is_always_at_or_above_close_and_low_at_or_below(tmp_path: Path) -> None:
    market = write_synthetic_market(tmp_path, sessions=50, instruments=1)
    frame = pd.read_csv(market.bar_files[0])
    assert (frame["high"] >= frame["close"]).all()
    assert (frame["low"] <= frame["close"]).all()
    assert (frame["close"] > 0).all()


def test_index_and_vix_files_use_distinct_symbols(tmp_path: Path) -> None:
    market = write_synthetic_market(tmp_path, sessions=30)
    index_frame = pd.read_csv(market.index_file)
    vix_frame = pd.read_csv(market.vix_file)
    assert set(index_frame["index_symbol"]) == {INDEX_SYMBOL}
    assert set(vix_frame["index_symbol"]) == {VIX_SYMBOL}


def test_instruments_file_lists_every_generated_instrument(tmp_path: Path) -> None:
    market = write_synthetic_market(tmp_path, sessions=20, instruments=5)
    frame = pd.read_csv(market.instruments_file)
    assert set(frame["instrument_id"]) == set(market.instrument_ids)
    assert set(frame["exchange"]) == {"NSE"}
    assert set(frame["segment"]) == {"equity"}


def test_membership_file_references_every_instrument_under_the_index(tmp_path: Path) -> None:
    market = write_synthetic_market(tmp_path, sessions=20, instruments=3)
    frame = pd.read_csv(market.membership_file)
    assert set(frame["instrument_id"]) == set(market.instrument_ids)
    assert set(frame["index_symbol"]) == {INDEX_SYMBOL}


def test_same_seed_is_reproducible(tmp_path: Path) -> None:
    market_a = write_synthetic_market(tmp_path / "a", sessions=30, seed=7)
    market_b = write_synthetic_market(tmp_path / "b", sessions=30, seed=7)
    frame_a = pd.read_csv(market_a.bar_files[0])
    frame_b = pd.read_csv(market_b.bar_files[0])
    pd.testing.assert_frame_equal(frame_a, frame_b)


def test_volume_is_reproducible_across_python_processes(tmp_path: Path) -> None:
    """Regression: volume was originally seeded from ``hash(instrument_id)``,
    which Python randomizes per-process by default (``PYTHONHASHSEED``) --
    so the exact same call could write different volumes on every run.
    Proven across two literal subprocesses, since a single process's own
    hash randomization is already fixed for its own lifetime and would
    not have caught this.
    """
    import os
    import subprocess
    import sys

    script = (
        "from pathlib import Path; "
        "from validation.synthetic_market import write_synthetic_market; "
        "m = write_synthetic_market(Path(r'{root}'), sessions=15, instruments=1, seed=3); "
        "print(Path(m.bar_files[0]).read_text(encoding='utf-8'), end='')"
    )
    env = {**os.environ, "PYTHONHASHSEED": "random"}
    outputs = []
    for label in ("a", "b"):
        root = tmp_path / label
        result = subprocess.run(
            [sys.executable, "-c", script.format(root=root)],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


def test_different_seeds_produce_different_prices(tmp_path: Path) -> None:
    market_a = write_synthetic_market(tmp_path / "a", sessions=30, seed=1)
    market_b = write_synthetic_market(tmp_path / "b", sessions=30, seed=2)
    frame_a = pd.read_csv(market_a.bar_files[0])
    frame_b = pd.read_csv(market_b.bar_files[0])
    assert not frame_a["close"].equals(frame_b["close"])


def test_the_generated_drop_ingests_cleanly_through_the_real_pipeline(tmp_path: Path) -> None:
    market = write_synthetic_market(tmp_path / "vendor", sessions=45, instruments=3)
    calendar = NSETradingCalendar(holidays={}, covered_years=frozenset(range(2014, 2016)))
    store = LocalDataStore(
        tmp_path / "raw", tmp_path / "normalized", tmp_path / "reference", StorageFormat.PARQUET
    )
    pipeline = DataIngestionPipeline(store, calendar)

    results = [
        *(pipeline.ingest_equity_bars(path) for path in market.bar_files),
        pipeline.ingest_index_observations(market.index_file, index_symbol=INDEX_SYMBOL),
        pipeline.ingest_index_observations(market.vix_file, index_symbol=VIX_SYMBOL),
        pipeline.ingest_instruments(market.instruments_file),
        pipeline.ingest_index_membership(market.membership_file),
    ]
    rejected = [r.summary() for r in results if not r.accepted]
    assert rejected == []
