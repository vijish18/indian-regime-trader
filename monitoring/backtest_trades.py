"""Reconstruct HMM sell history from saved fills, using the engine's average cost.

Standalone standard-library exporter, runnable over SSH stdin without altering
the research checkout. Checkpoints supply stop reasons omitted by older CSVs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shlex
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


def plain(value: Any) -> Any:
    """Read checkpoint data without importing or instantiating domain classes."""
    if not isinstance(value, dict):
        return value
    kind, payload = value["kind"], value["value"]
    if kind in {"date", "datetime", "enum"}:
        return payload
    if kind == "decimal":
        return float(payload)
    if kind == "mapping":
        return {plain(k): plain(v) for k, v in payload}
    if kind in {"list", "tuple", "set"}:
        return [plain(v) for v in payload]
    if kind == "record":
        return {k: plain(v) for k, v in payload.items()}
    raise ValueError(f"Unsupported checkpoint tag: {kind}")


def fill_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["execution_date"],
        row["instrument_id"],
        row["side"],
        int(row["quantity"]),
        round(float(row["fill_price"]), 8),
    )


def history_from_states(states: list[dict[str, Any]]) -> dict[str, Any]:
    positions: dict[str, dict[str, Any]] = {}
    exits: list[dict[str, Any]] = []
    all_fills: list[dict[str, Any]] = []
    buy_cash = sell_cash = fees = 0.0
    for fold, state in enumerate(states, start=1):
        stops = {}
        for stop in state["stop_exits"]:
            breach = stop["breach"]
            key = (
                stop["execution_date"],
                breach["instrument_id"],
                "sell",
                stop["quantity"],
                round(breach["fill_price"], 8),
            )
            if key in stops:
                raise ValueError("Ambiguous stop attribution")
            stops[key] = breach
        fills, rows = state["fills"], state["trade_log_rows"]
        if len(fills) != len(rows):
            raise ValueError("Checkpoint fill count does not match ledger")
        for row, fill in zip(rows, fills, strict=True):
            order = fill["order"]
            instrument, quantity = row["instrument_id"], int(row["quantity"])
            if quantity <= 0 or not all(
                math.isfinite(float(row[k]))
                for k in ("fill_price", "gross_value", "cost", "net_value")
            ):
                raise ValueError("Invalid fill amounts")
            all_fills.append(row)
            fees += row["cost"]
            if row["side"] == "buy":
                buy_cash += row["net_value"]
                p = positions.setdefault(
                    instrument,
                    {
                        "quantity": 0,
                        "price_basis": 0.0,
                        "cash_basis": 0.0,
                        "fees": 0.0,
                        "first_entry": row["execution_date"],
                        "entry_count": 0,
                    },
                )
                p["quantity"] += quantity
                p["price_basis"] += quantity * row["fill_price"]
                p["cash_basis"] += row["net_value"]
                p["fees"] += row["cost"]
                p["entry_count"] += 1
                p["last_entry"] = row["execution_date"]
                continue
            if row["side"] != "sell":
                raise ValueError("Unknown fill side")
            if instrument not in positions or quantity > positions[instrument]["quantity"]:
                raise ValueError(f"Sell exceeds recorded holdings: {instrument}")
            p = positions[instrument]
            fraction = quantity / p["quantity"]
            entry_price = p["price_basis"] / p["quantity"]
            basis, buy_fees = p["cash_basis"] * fraction, p["fees"] * fraction
            pnl = row["net_value"] - basis
            sell_cash += row["net_value"]
            stop = stops.pop(fill_key(row), None)
            reason = row.get("exit_reason") or (stop["reason"] if stop else "portfolio_rebalance")
            date = row["execution_date"]
            exits.append(
                {
                    "id": len(exits) + 1,
                    "fold": fold,
                    "instrument_id": instrument,
                    "symbol": instrument.removeprefix("NSE:"),
                    "entry_date": p["first_entry"],
                    "last_entry_date": p["last_entry"],
                    "entry_fill_count": p["entry_count"],
                    "signal_date": row["signal_date"],
                    "exit_date": date,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "exit_price": row["fill_price"],
                    "entry_cost": basis,
                    "net_proceeds": row["net_value"],
                    "net_pnl": pnl,
                    "return_pct": pnl / basis if basis else None,
                    "gross_pnl": row["gross_value"] - entry_price * quantity,
                    "entry_fees": buy_fees,
                    "exit_fees": row["cost"],
                    "total_fees": buy_fees + row["cost"],
                    "reason": reason,
                    "reason_source": "checkpoint stop event"
                    if stop
                    else "recorded fold liquidation"
                    if reason == "fold_end_liquidation"
                    else "rebalance fill; specific ranking/risk cause not recorded",
                    "stop_level": stop["stop_level"] if stop else None,
                    "regime": state["regime_points"].get(row["signal_date"]),
                    "confidence": state["confidence_points"].get(row["signal_date"]),
                    "target_weight_before": order["current_weight"],
                    "target_weight_after": order["target_weight"],
                    "shares_after_exit": p["quantity"] - quantity,
                    "holding_days": (
                        dt.date.fromisoformat(date) - dt.date.fromisoformat(p["first_entry"])
                    ).days,
                }
            )
            p["quantity"] -= quantity
            for field in ("price_basis", "cash_basis", "fees"):
                p[field] *= 1 - fraction
            if p["quantity"] == 0:
                del positions[instrument]
        if stops:
            raise ValueError("Some recorded stops could not be matched to a sell")
        recorded = {k: v for k, v in state["holdings"].items() if v}
        if recorded != {k: v["quantity"] for k, v in positions.items()}:
            raise ValueError("Reconstructed holdings do not match checkpoint")
    for row in exits:
        row["shares_at_end"] = positions.get(row["instrument_id"], {}).get("quantity", 0)
    return {
        "rows": exits,
        "raw_fills": all_fills,
        "current_positions": positions,
        "summary": {
            "exit_fills": len(exits),
            "total_fills": len(all_fills),
            "realized_net_pnl": sum(r["net_pnl"] for r in exits),
            "total_costs": fees,
            "buy_cash": buy_cash,
            "sell_cash": sell_cash,
            "reason_counts": dict(Counter(r["reason"] for r in exits)),
        },
    }


def export_history(root: Path, data_root: Path) -> dict[str, Any]:
    folder = root / "hmm"
    manifest = json.loads((folder / "series/run.manifest.json").read_text())
    if not (folder / "completed.txt").exists():
        raise ValueError("HMM is not complete")
    paths = sorted((folder / "series/sessions").glob("hmm.*.session.json"))
    if len(paths) != manifest["folds_total"]:
        raise ValueError("Missing final fold checkpoints")
    states = [plain(json.loads(p.read_text())["state"]) for p in paths]
    result = history_from_states(states)
    with (folder / "series/hmm.trades.csv").open(newline="") as handle:
        published = list(csv.DictReader(handle))
    if [fill_key(r) for r in published] != [fill_key(r) for r in result["raw_fills"]]:
        raise ValueError("Checkpoint history does not match published trade CSV")
    result.pop("raw_fills")
    prices = {}
    for instrument in sorted({r["instrument_id"] for r in result["rows"]}):
        name = instrument.replace(":", "_").replace("/", "_").replace(" ", "_")
        path = data_root / "raw/equity_bars" / f"{name}.csv"
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["session_date"] <= manifest["end"]]
        if rows:
            last = max(rows, key=lambda r: r["session_date"])
            prices[instrument] = {"price": float(last["close"]), "as_of": last["session_date"]}
    with (folder / "series/hmm.equity.csv").open(newline="") as handle:
        equity = list(csv.DictReader(handle))
    ending = float(equity[-1]["equity"])
    result["summary"]["other_cash_movements"] = (
        ending - manifest["initial_equity"] - result["summary"]["realized_net_pnl"]
    )
    result.update(
        available=True,
        run_id=root.name,
        fingerprint=manifest["fingerprint"],
        as_of=manifest["end"],
        historical_prices=prices,
        generated_at=dt.datetime.now(dt.UTC).isoformat(),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-key", type=Path)
    parser.add_argument("--output", type=Path, default=Path("state/hmm_backtest_trades.json"))
    parser.add_argument("--watch", type=int, default=0)
    args = parser.parse_args()
    if not args.ssh_host:
        print(
            json.dumps(export_history(Path(args.run_root), Path(args.data_root)), allow_nan=False)
        )
        return
    if not args.ssh_key:
        parser.error("--ssh-key is required")
    command = [
        "ssh",
        "-i",
        str(args.ssh_key.resolve()),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        args.ssh_host,
        "python3 - --run-root "
        + shlex.quote(args.run_root)
        + " --data-root "
        + shlex.quote(args.data_root),
    ]
    response = subprocess.run(
        command,
        input=Path(__file__).read_text(encoding="utf-8"),
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    history = json.loads(response.stdout)
    # Quotes are fetched locally; broker credentials never go to Azure.
    from scripts._dashboard_live import live_quotes

    ids = sorted({r["instrument_id"] for r in history["rows"]})
    while True:
        quotes: dict[str, Any] = {}
        reason = None
        for offset in range(0, len(ids), 100):
            batch = live_quotes(ids[offset : offset + 100], Path("state/kite_session.json"))
            if not batch.get("available"):
                reason = batch.get("reason", "No current quotes")
                break
            for instrument, q in batch["quotes"].items():
                quotes[instrument] = {
                    "price": q["last"],
                    "as_of": q.get("last_trade_time") or q.get("exchange_timestamp"),
                }
        history["current_quotes"] = {
            "prices": quotes,
            "reason": reason,
            "checked_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=args.output.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(history, handle, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, args.output)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(
            f"Published {len(history['rows'])} HMM exits; {len(quotes)} current quotes", flush=True
        )
        if not args.watch:
            return
        time.sleep(max(args.watch, 30))


if __name__ == "__main__":
    main()
