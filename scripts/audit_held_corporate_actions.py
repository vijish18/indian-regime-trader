"""Extract corporate-action exposure from saved daily holdings, not guesses.

Read-only; can audit completed and partial diagnostic runs. Each fold begins
flat, as the engine does. Include ex-date buys separately from entitled prior
holders so dividend attribution errors are visible. JSON is an audit queue,
never a claim that source terms or accounting have been verified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.backtest_trades import plain  # noqa: E402
from storage.atomic import atomic_write  # noqa: E402


def audit(run_root: Path, data_root: Path) -> dict:
    actions = defaultdict(list)
    with (data_root / "reference/corporate_actions.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            actions[(row["instrument_id"], row["ex_date"])].append(row)
    exposure = defaultdict(list)
    held_days = defaultdict(set)
    strategies = {}
    adjustments = []
    dividend_checks = []
    for folder in sorted(run_root.iterdir()):
        paths = sorted((folder / "series/sessions").glob("*.session.json"))
        if not paths:
            continue
        strategies[folder.name] = {
            "saved_folds": len(paths),
            "complete": (folder / "completed.txt").exists(),
        }
        for path in paths:
            state = plain(json.loads(path.read_text())["state"])
            buys = defaultdict(set)
            cash_flows = defaultdict(float)
            for fill in state["trade_log_rows"]:
                sign = -1 if fill["side"] == "buy" else 1
                cash_flows[fill["execution_date"]] += sign * float(fill["net_value"])
                if fill["side"] == "buy":
                    buys[fill["execution_date"]].add(fill["instrument_id"])
            prior = {}
            prior_cash = float(state["equity_history"][0])
            for day, holdings in sorted(state["positions_history"].items()):
                entitled = sum(
                    quantity * float(action["cash_amount"])
                    for iid, quantity in prior.items()
                    for action in actions.get((iid, day), [])
                    if action["action_type"] == "dividend" and action.get("cash_amount")
                )
                actual = float(state["cash_points"][day]) - prior_cash - cash_flows[day]
                if abs(entitled) > 0.01 or abs(actual) > 0.01:
                    dividend_checks.append(
                        {
                            "strategy": folder.name,
                            "fold": path.name,
                            "ex_date": day,
                            "entitled_amount_from_reference": round(entitled, 6),
                            "credited_amount": round(actual, 6),
                            "difference": round(actual - entitled, 6),
                            "quantity_check_passed": abs(actual - entitled) < 0.01,
                            "payment_timing_verified": False,
                        }
                    )
                instruments = set(prior) | set(holdings) | buys[day]
                for iid in instruments:
                    if prior.get(iid, 0) or holdings.get(iid, 0) or iid in buys[day]:
                        held_days[iid].add(day)
                    if (iid, day) in actions and (prior.get(iid, 0) or iid in buys[day]):
                        exposure[(iid, day)].append(
                            {
                                "strategy": folder.name,
                                "fold": path.name,
                                "prior_close_quantity": prior.get(iid, 0),
                                "end_of_session_quantity": holdings.get(iid, 0),
                                "bought_ex_date": iid in buys[day],
                            }
                        )
                prior = holdings
                prior_cash = float(state["cash_points"][day])
            adjustments.extend(
                {"strategy": folder.name, **a} for a in state.get("share_adjustments", [])
            )
    return {
        "validation_status": "unverified",
        "run_root": str(run_root),
        "data_root": str(data_root),
        "strategies": strategies,
        "events": [
            {"instrument_id": iid, "ex_date": day, "actions": actions[(iid, day)], "exposure": rows}
            for (iid, day), rows in sorted(exposure.items())
        ],
        "share_adjustments": adjustments,
        "dividend_checks": dividend_checks,
        "held_days": {iid: sorted(days) for iid, days in sorted(held_days.items())},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.run_root, args.data_root)
    atomic_write(args.output, json.dumps(payload, indent=2))
    print(
        f"{len(payload['events'])} exposed corporate-action dates; "
        f"{len(payload['held_days'])} held instruments"
    )
