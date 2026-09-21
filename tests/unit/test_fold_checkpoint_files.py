from pathlib import Path

import pandas as pd

from scripts.run_walk_forward import _save_fold_csv, _truncate_checkpoints


def test_replayed_empty_fold_removes_old_trades_without_losing_prior_fold(tmp_path: Path) -> None:
    path = tmp_path / "hmm.trades.partial.csv"
    _save_fold_csv(path, pd.DataFrame({"fold": [1], "quantity": [10]}), 1)
    _save_fold_csv(path, pd.DataFrame({"fold": [2], "quantity": [20]}), 2)
    _save_fold_csv(path, pd.DataFrame(columns=["fold", "quantity"]), 2)
    assert pd.read_csv(path).to_dict("records") == [{"fold": 1, "quantity": 10}]


def test_resume_truncation_keeps_only_committed_folds(tmp_path: Path) -> None:
    path = tmp_path / "hmm.cash.partial.csv"
    pd.DataFrame({"fold": [1, 2], "cash": [100, 200]}).to_csv(path, index=False)
    _truncate_checkpoints(tmp_path, "hmm", 1)
    assert pd.read_csv(path).to_dict("records") == [{"fold": 1, "cash": 100}]
