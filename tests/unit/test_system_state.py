"""Unit tests for ``execution/system_state.py`` (Phase 18): the
JSON-file persistence layer for system state across restarts.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from execution.system_state import (
    STATE_SCHEMA_VERSION,
    PersistedState,
    PortfolioPositionSnapshot,
    PortfolioSnapshot,
    SystemState,
    SystemStateStore,
    SystemStateStoreError,
)

_T0 = dt.datetime(2024, 6, 3, 9, 0, 0, tzinfo=dt.UTC)


def _make_state(**overrides: object) -> PersistedState:
    defaults: dict[str, object] = {
        "schema_version": STATE_SCHEMA_VERSION,
        "app_version": "0.1.0",
        "system_state": SystemState.READY,
        "model_version": "model-2024-06-01",
        "strategy_version": "strategy-v1",
        "portfolio_snapshot": PortfolioSnapshot(
            as_of=_T0,
            cash=1_000_000.0,
            positions=(PortfolioPositionSnapshot("NSE:INFY", 10, 1500.0),),
        ),
        "last_market_data_timestamp": _T0,
        "last_broker_event_id": "T-1",
        "last_broker_event_timestamp": _T0,
        "updated_at": _T0,
    }
    defaults.update(overrides)
    return PersistedState(**defaults)  # type: ignore[arg-type]


def test_load_returns_none_when_no_file_exists(tmp_path: Path) -> None:
    store = SystemStateStore(tmp_path / "state.json")
    assert store.load() is None


def test_save_then_load_round_trips_exactly(tmp_path: Path) -> None:
    store = SystemStateStore(tmp_path / "state.json")
    state = _make_state()
    store.save(state)
    loaded = store.load()
    assert loaded == state


def test_save_then_load_round_trips_with_no_portfolio_snapshot(tmp_path: Path) -> None:
    store = SystemStateStore(tmp_path / "state.json")
    state = _make_state(
        portfolio_snapshot=None,
        last_market_data_timestamp=None,
        last_broker_event_id=None,
        last_broker_event_timestamp=None,
    )
    store.save(state)
    assert store.load() == state


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    store = SystemStateStore(tmp_path / "nested" / "dir" / "state.json")
    store.save(_make_state())
    assert (tmp_path / "nested" / "dir" / "state.json").is_file()


def test_load_raises_on_corrupted_json(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = SystemStateStore(path)
    with pytest.raises(SystemStateStoreError, match="corrupted"):
        store.load()


def test_load_raises_on_wrong_shape(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"unexpected": "shape"}', encoding="utf-8")
    store = SystemStateStore(path)
    with pytest.raises(SystemStateStoreError, match="expected shape"):
        store.load()


def test_verify_accessible_passes_for_a_fresh_writable_directory(tmp_path: Path) -> None:
    store = SystemStateStore(tmp_path / "state.json")
    store.verify_accessible()  # must not raise


def test_verify_accessible_does_not_leave_a_probe_file_behind(tmp_path: Path) -> None:
    store = SystemStateStore(tmp_path / "state.json")
    store.verify_accessible()
    leftovers = list(tmp_path.glob(".probe-*"))
    assert leftovers == []


def test_verify_accessible_raises_on_a_corrupted_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = SystemStateStore(path)
    with pytest.raises(SystemStateStoreError):
        store.verify_accessible()


def test_save_overwrites_the_previous_state(tmp_path: Path) -> None:
    store = SystemStateStore(tmp_path / "state.json")
    store.save(_make_state(system_state=SystemState.READY))
    store.save(_make_state(system_state=SystemState.RECONCILIATION_REQUIRED))
    loaded = store.load()
    assert loaded is not None
    assert loaded.system_state is SystemState.RECONCILIATION_REQUIRED
