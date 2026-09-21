"""JSON session checkpoints; only explicitly allowed domain data types can be loaded."""

from __future__ import annotations

import datetime as dt
import importlib
import json
from dataclasses import fields, is_dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from storage.atomic import atomic_write

MODULES = frozenset(
    {
        "backtest.engine",
        "backtest.costs",
        "core.regime.allocation",
        "portfolio.portfolio_constructor",
        "risk.risk_manager",
        "risk.circuit_breaker",
        "risk.stop_loss",
        "broker.base",
        "broker.adapters.paper_broker",
        "execution.order_manager",
        "execution.execution_journal",
        "execution.position_tracker",
    }
)


def encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return {
            "kind": "enum",
            "module": type(value).__module__,
            "name": type(value).__name__,
            "value": value.value,
        }
    if isinstance(value, dt.datetime):
        return {"kind": "datetime", "value": value.isoformat()}
    if isinstance(value, dt.date):
        return {"kind": "date", "value": value.isoformat()}
    if isinstance(value, Decimal):
        return {"kind": "decimal", "value": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "record",
            "module": type(value).__module__,
            "name": type(value).__name__,
            "value": {f.name: encode(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, dict):
        return {"kind": "mapping", "value": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (set, frozenset)):
        return {"kind": "set", "value": [encode(v) for v in sorted(value)]}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "value": [encode(v) for v in value],
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value)}")


def decode(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    kind, payload = value["kind"], value["value"]
    if kind == "date":
        return dt.date.fromisoformat(payload)
    if kind == "datetime":
        return dt.datetime.fromisoformat(payload)
    if kind == "decimal":
        return Decimal(payload)
    if kind == "mapping":
        return {decode(k): decode(v) for k, v in payload}
    if kind == "set":
        return {decode(v) for v in payload}
    if kind in {"list", "tuple"}:
        decoded = [decode(v) for v in payload]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind in {"record", "enum"} and value["module"] in MODULES:
        cls = getattr(importlib.import_module(value["module"]), value["name"])
        if kind == "enum" and isinstance(cls, type) and issubclass(cls, Enum):
            return cls(payload)
        if kind == "record" and isinstance(cls, type) and is_dataclass(cls):
            return cls(**{k: decode(v) for k, v in payload.items()})
    raise ValueError("Unknown checkpoint type")


def save(path: Path, identity: str, state: dict[str, Any]) -> None:
    atomic_write(
        path,
        json.dumps(
            {"version": 1, "identity": identity, "state": encode(state)},
            allow_nan=False,
            separators=(",", ":"),
        ),
    )


def load(path: Path, identity: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("identity") != identity:
        raise ValueError("Checkpoint identity changed; use a fresh run directory")
    state = decode(payload["state"])
    if not isinstance(state, dict):
        raise ValueError("Invalid checkpoint state")
    return state
