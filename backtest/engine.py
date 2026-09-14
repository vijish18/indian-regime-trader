"""Single-pass backtest engine: replays signal -> risk -> execution decisions
session by session, applying next-session execution timing and full Indian
costs/slippage. See docs/SPECIFICATION.md section 10.

Not implemented yet (Phase 8).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    regime_history: pd.Series


class BacktestEngine:
    """Runs one deterministic backtest over a fixed date range with a fixed,
    already-fitted model -- walk_forward.py is responsible for the
    fit/freeze/roll loop across folds.
    """

    def run(self, start: pd.Timestamp, end: pd.Timestamp, model_id: str) -> BacktestResult:
        raise NotImplementedError("Phase 8: backtest engine is not implemented yet.")
