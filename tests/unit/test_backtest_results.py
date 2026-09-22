import json
from pathlib import Path

import pytest

from monitoring.backtest_results import export_run


def write_strategy(root: Path, name: str, complete: bool, identity: str = "same") -> None:
    folder = root / name
    (folder / "series").mkdir(parents=True)
    manifest = {
        "fingerprint": identity,
        "start": "2015-01-01",
        "end": "2026-09-21",
        "initial_equity": 100000,
        "folds_total": 34,
        "git_commit": "test",
        "completed_folds": {name: 34 if complete else 19},
        "updated_at": "test",
    }
    (folder / "series/run.manifest.json").write_text(json.dumps(manifest))
    # A partial strategy can have an old report file; completion is mandatory.
    (folder / "reports.json").write_text(
        json.dumps(
            {
                "start": manifest["start"],
                "end": manifest["end"],
                "reports": {name: {"cagr": -0.3, "sortino": float("nan")}},
            }
        )
    )
    if complete:
        (folder / "completed.txt").write_text("done")
        (folder / f"series/{name}.equity.csv").write_text(
            "session_date,equity\n2018-01-23,99000\n2026-09-21,4334.62\n"
        )


def test_pending_reports_excluded_and_balance_uses_actual_starting_capital(tmp_path: Path) -> None:
    write_strategy(tmp_path, "hmm", True)
    write_strategy(tmp_path, "shuffled_regime_control", False)
    result = export_run(tmp_path)
    assert set(result["strategies"]["reports"]) == {"hmm"}
    report = result["strategies"]["reports"]["hmm"]
    assert report["ending_equity"] == 4334.62
    assert report["return_from_capital"] == pytest.approx(4334.62 / 100000 - 1)
    assert report["sortino"] is None
    assert result["backtest_run"]["progress"]["shuffled_regime_control"]["completed"] == 19
    assert result["backtest_run"]["complete"] is False
    json.dumps(result, allow_nan=False)


def test_mixed_fingerprints_are_refused(tmp_path: Path) -> None:
    write_strategy(tmp_path, "hmm", True)
    write_strategy(tmp_path, "buy_and_hold", False, identity="old-run")
    with pytest.raises(ValueError, match="different backtest runs"):
        export_run(tmp_path)
