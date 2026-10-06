"""One atomic paper transaction: broker, orders, fills and deduplication together."""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Any

from backtest.checkpoint import load, save

TRACKER_FIELDS = (
    "_quantity",
    "_avg_price",
    "_realized_pnl",
    "_target_weight",
    "_last_price",
    "_last_update",
)


def components(orchestrator: Any) -> dict[str, tuple[Any, tuple[str, ...]]]:
    return {
        "broker": (orchestrator.broker, ("_cash", "_orders", "_created_at", "_fills")),
        "broker_positions": (orchestrator.broker.position_tracker, TRACKER_FIELDS),
        "positions": (orchestrator.position_tracker, TRACKER_FIELDS),
        "orders": (orchestrator.order_manager, ("_orders", "_idempotency_index", "_signal_index")),
        "journal": (orchestrator.order_manager.journal, ("_entries", "_sequence")),
        "fills": (orchestrator.fill_tracker, ("_applied_fill_ids", "_cumulative_cash_flow")),
        "equity": (orchestrator.equity_history, ("values",)),
    }


def paper_identity(settings: Any, budget: float) -> str:
    """The ledger's identity: a settings or budget change starts a new ledger
    rather than continuing one built under different rules."""
    return hashlib.sha256((settings.model_dump_json() + str(budget)).encode()).hexdigest()


def completed_days(path: Path, identity: str) -> list[dt.date]:
    """Sessions whose rebalance cycle completed, read without building a
    runtime -- the scheduler asks this before deciding to prompt a login."""
    state = load(path, identity)
    if state is None:
        return []
    return [dt.date.fromisoformat(day) for day in state["metadata"]["completed_days"]]


def save_paper(path: Path, identity: str, orchestrator: Any, metadata: dict[str, Any]) -> None:
    state = {
        name: {field: getattr(obj, field) for field in names}
        for name, (obj, names) in components(orchestrator).items()
    }
    state["metadata"] = metadata
    save(path, identity, state)


def restore_paper(path: Path, identity: str, orchestrator: Any) -> dict[str, Any]:
    state = load(path, identity)
    if state is None:
        return {"completed_days": [], "extremes": {}}
    for name, (obj, names) in components(orchestrator).items():
        for field in names:
            setattr(obj, field, state[name][field])
    return dict(state["metadata"])
