"""Controlled real-bar regression through the previously failing fold boundary.

This isolates execution: force a small HEXAWARE target each day, with stops
disabled, to prove delisting handling rather than obtain a strategy return.
The no-policy control must reproduce the missing-bar failure. The fixed case
must exit at an observed pre-suspension open, retain costs/provenance and
reach 2020-11-13 with no stranded shares.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.engine import BacktestEngineError  # noqa: E402
from portfolio.portfolio_constructor import TargetPortfolio, TargetPosition  # noqa: E402
from scripts.run_walk_forward import build_validator  # noqa: E402
from storage.atomic import atomic_write  # noqa: E402


def replay(output: Path) -> dict:
    results = {}
    for enabled in (False, True):
        label = "fixed" if enabled else "control"
        validator = build_validator(dt.date(2026, 9, 21), output / label, frame_cache=100)
        engine = validator.engine
        engine.stop_loss_policy = None
        engine.liquidate_at_end = True
        if not enabled:
            engine.delisting_notices = ()
        engine.stock_selector = SimpleNamespace(select=lambda day: [])

        def construct(candidates, target, day, equity, **kwargs):
            position = TargetPosition(
                "NSE:HEXAWARE", "HEXAWARE", 0.05, "IT", 1, 1.0, "controlled replay"
            )
            return TargetPortfolio(day, (position,), 0.95, target.regime, 0.05)

        engine.portfolio_constructor = SimpleNamespace(construct=construct)
        dates = validator.calendar.trading_days_between(dt.date(2020, 10, 1), dt.date(2020, 11, 12))
        targets = validator._buy_and_hold_targets(dates)
        try:
            result = engine.run(label, targets, dates, 100_000)
        except BacktestEngineError as exc:
            if enabled or "Cannot liquidate NSE:HEXAWARE at fold end 2020-11-13" not in str(exc):
                raise
            results[label] = {"expected_failure": str(exc)}
            continue
        if not enabled:
            raise AssertionError("Control did not reproduce the failure")
        exits = result.trade_log[result.trade_log.exit_reason == "announced_delisting_exit"]
        assert len(exits) == 1
        exit_row = exits.iloc[0]
        assert exit_row.execution_date == dt.date(2020, 10, 22)
        assert abs(exit_row.fill_price - 470.45) < 1e-8
        assert exit_row.cost > 0
        assert result.positions_history[dt.date(2020, 11, 13)] == {}
        atomic_write(output / "fixed.trades.csv", result.trade_log.to_csv(index=False))
        results[label] = {
            "exit_date": str(exit_row.execution_date),
            "fill_price": float(exit_row.fill_price),
            "quantity": int(exit_row.quantity),
            "cost": float(exit_row.cost),
            "last_execution": str(max(result.positions_history)),
            "terminal_holdings": {},
            "passed": True,
        }
    atomic_write(output / "replay.json", json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(replay(parser.parse_args().output), indent=2))
