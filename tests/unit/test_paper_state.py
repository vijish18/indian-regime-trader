from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from app.paper_state import completed_days, paper_identity
from backtest.checkpoint import save


def test_completed_days_reads_the_ledger_metadata(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    save(ledger, "id", {"metadata": {"completed_days": ["2026-10-05"], "extremes": {}}})
    assert completed_days(ledger, "id") == [dt.date(2026, 10, 5)]


def test_no_ledger_means_no_rebalance_yet(tmp_path: Path) -> None:
    assert completed_days(tmp_path / "ledger.json", "id") == []


def test_a_ledger_from_other_settings_is_refused(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    save(ledger, "id", {"metadata": {"completed_days": [], "extremes": {}}})
    with pytest.raises(ValueError):
        completed_days(ledger, "other")


def test_identity_tracks_settings_and_budget() -> None:
    class _Settings:
        def __init__(self, text: str) -> None:
            self.text = text

        def model_dump_json(self) -> str:
            return self.text

    assert paper_identity(_Settings("a"), 100_000) == paper_identity(_Settings("a"), 100_000)
    assert paper_identity(_Settings("a"), 100_000) != paper_identity(_Settings("b"), 100_000)
    assert paper_identity(_Settings("a"), 100_000) != paper_identity(_Settings("a"), 50_000)
