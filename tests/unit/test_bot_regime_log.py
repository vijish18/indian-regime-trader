from __future__ import annotations

import csv
import json
from pathlib import Path

from app.bot import record_regime


def _publish(path: Path, as_of: str, label: str = "normal", available: bool = True) -> None:
    regime = {
        "available": available,
        "as_of": as_of,
        "label": label,
        "state_id": 3,
        "confidence": 0.61234,
        "sizes_book": False,
        "model_id": "hmm_x",
        "probabilities": [0.1, 0.0, 0.29, 0.61, 0.0],
    }
    path.write_text(json.dumps({"regime_now": regime}), encoding="utf-8")


def test_each_session_is_logged_once(tmp_path: Path) -> None:
    pub, log = tmp_path / "dashboard.json", tmp_path / "regime_history.csv"
    _publish(pub, "2026-10-01")
    assert record_regime(pub, log) is True
    assert record_regime(pub, log) is False  # the 21:30 retry or a rerun adds nothing
    _publish(pub, "2026-10-05", "elevated")
    assert record_regime(pub, log) is True
    with log.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["as_of"], r["label"]) for r in rows] == [
        ("2026-10-01", "normal"),
        ("2026-10-05", "elevated"),
    ]
    assert rows[0]["confidence"] == "0.6123"
    assert json.loads(rows[0]["probabilities"])[3] == 0.61


def test_an_unavailable_regime_is_not_logged(tmp_path: Path) -> None:
    pub, log = tmp_path / "dashboard.json", tmp_path / "regime_history.csv"
    _publish(pub, "2026-10-01", available=False)
    assert record_regime(pub, log) is False
    assert not log.exists()


def test_trade_runs_when_due_or_to_finish_todays_rebalance() -> None:
    import datetime as dt

    from app.bot import trades_today
    from data.calendar import NSETradingCalendar

    calendar = NSETradingCalendar(holidays={}, covered_years=frozenset({2026}))
    tue, wed = dt.date(2026, 10, 6), dt.date(2026, 10, 7)
    assert trades_today("weekly", calendar, tue, [])
    assert trades_today("weekly", calendar, tue, [tue])
    assert not trades_today("weekly", calendar, wed, [tue])
