"""Durable, per-session record of what the system decided and why.

``ExecutionJournal`` is explicit that it is "an in-memory list for the
current run", and ``storage/database.py`` is still a stub, so until now a
daily cycle's reasoning existed only while the process did. That is fine
for a backtest, which reports at the end, and useless for paper trading,
where the whole point is to read back what happened on a day that has
already gone.

One file per session, written once and never rewritten. A trading day's
record is an audit artifact: if a second run happens for the same date,
that is a fact worth seeing, not a reason to overwrite the first. Re-runs
land beside the original with a suffix.

What is kept is the chain, not just the outcome:

    candidates      every ranked name with its factor breakdown
    target          the weights the constructor proposed
    risk decisions  which were approved, which vetoed, and on what check
    sized trades    the quantities those became
    submitted       the orders that actually went to the broker
    skipped         what was dropped, so absence is explained

That chain is what makes "why did it buy this" answerable afterwards. A
list of fills alone cannot answer it -- it shows what happened and not
what the alternatives were.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _factor_dict(score: Any) -> dict[str, Any]:
    raw, std = score.raw_factors, score.standardized_factors
    return {
        "rank": score.rank,
        "symbol": score.symbol,
        "instrument_id": score.instrument_id,
        "score": float(score.score),
        "z": {
            "momentum": float(std.momentum),
            "trend": float(std.trend_persistence),
            "relative_strength": float(std.relative_strength),
            "volatility": float(std.volatility),
        },
        "raw": {
            "momentum": float(raw.momentum),
            "volatility": float(raw.volatility),
        },
    }


def to_record(report: Any) -> dict[str, Any]:
    """The report as plain JSON-safe data. Pure, so it is testable without
    touching a filesystem."""
    target = report.target_portfolio
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": report.as_of.isoformat(),
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "state": str(report.state),
        "is_trading_day": bool(report.is_trading_day),
        "permit_trading": bool(report.permit_trading),
        "fail_closed_reason": (
            str(report.fail_closed_reason) if report.fail_closed_reason else None
        ),
        "candidates": [_factor_dict(s) for s in report.candidates],
        "target_positions": [
            {
                "instrument_id": p.instrument_id,
                "target_weight": float(p.target_weight),
                "rank": p.rank,
            }
            for p in (target.positions if target is not None else ())
        ],
        "risk_decisions": [
            {
                "instrument_id": d.instrument_id,
                "approved": bool(d.approved),
                "target_weight": float(d.target_weight),
                "circuit_state": str(d.circuit_state),
                # The check that stopped it, not just that something did --
                # "rejected" without a reason cannot be acted on.
                "violations": [str(v.check) for v in d.violations],
            }
            for d in report.risk_decisions
        ],
        "sized_trades": [
            {
                "instrument_id": getattr(t, "instrument_id", None),
                "quantity": getattr(t, "quantity", None),
                "side": str(getattr(t, "side", "")),
            }
            for t in report.sized_trades
        ],
        "submitted_order_ids": list(report.submitted_order_ids),
        "skipped_trades": list(report.skipped_trades),
        "messages": list(report.messages),
    }


def record_cycle(report: Any, root: Path) -> Path:
    """Write one session's record, without ever replacing another.

    A second run for the same date is a real event -- a restart, a manual
    re-run, an operator retrying after a halt -- and overwriting the first
    would erase the evidence of whichever one mattered.
    """
    root.mkdir(parents=True, exist_ok=True)
    stem = report.as_of.isoformat()
    path = root / f"{stem}.json"
    attempt = 1
    while path.exists():
        attempt += 1
        path = root / f"{stem}.run{attempt}.json"
    path.write_text(json.dumps(to_record(report), indent=2), encoding="utf-8")
    return path


def load_sessions(root: Path) -> list[dict[str, Any]]:
    """Every recorded session, oldest first, for after-the-fact analysis."""
    if not root.is_dir():
        return []
    records = []
    for path in sorted(root.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(records, key=lambda r: (r.get("as_of", ""), r.get("recorded_at", "")))
