"""Walk-forward validation harness. See docs/SPECIFICATION.md section 10.

Each fold: fit HMM + scaler on the training window only (BIC model
selection on training only), freeze both, evaluate strictly out-of-sample on
the following test window, roll forward. Also runs the required benchmarks
and the shuffled-regime control (section 10.1).

Not implemented yet (Phase 9).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.performance import PerformanceReport
from config.models import BacktestConfig


@dataclass(frozen=True)
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    model_id: str
    performance: PerformanceReport


class WalkForwardValidator:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def generate_folds(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Train/test window boundaries per ``backtest.training_window_sessions``,
        ``backtest.test_window_sessions``, and ``backtest.roll_step_sessions``.
        """
        raise NotImplementedError("Phase 9: walk-forward folds are not implemented yet.")

    def run(self, start: pd.Timestamp, end: pd.Timestamp) -> list[WalkForwardFold]:
        raise NotImplementedError("Phase 9: walk-forward validation is not implemented yet.")

    def run_shuffled_regime_control(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> PerformanceReport:
        """Re-run with regime labels randomly permuted, to test whether the
        HMM's specific state ordering (not just its existence) matters.
        """
        raise NotImplementedError("Phase 9: shuffled-regime control is not implemented yet.")
