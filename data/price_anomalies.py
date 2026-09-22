"""Raw-price loss screen. Flags require evidence, never automatic price repair."""

from __future__ import annotations

import csv
import datetime as dt
from decimal import Decimal
from pathlib import Path


def scan_price_losses(root: Path, start: dt.date, end: dt.date) -> list[dict]:
    """Flag strictly >10% declines without assuming a corporate action.

    Compare close and low with the previous available close, and low with
    today's open. High-to-low is excluded: daily OHLC cannot establish order.
    Keep previous dates explicit so missing sessions are not called one day.
    Action matches are research leads, not proof that accounting is correct.
    """
    actions: dict[tuple[str, str], list[dict]] = {}
    with (root / "reference/corporate_actions.csv").open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            actions.setdefault((row["instrument_id"], row["ex_date"]), []).append(row)
    paths = sorted((root / "raw/equity_bars").glob("*.csv"))
    if not paths:
        raise ValueError("No equity bars available for price-loss audit")
    findings = []
    for path in paths:
        previous = None
        with path.open(encoding="utf-8-sig") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda row: row["session_date"])
        for row in rows:
            day = dt.date.fromisoformat(row["session_date"])
            if day > end:
                break
            prices = {key: Decimal(row[key]) for key in ("open", "low", "close")}
            if any(not p.is_finite() or p <= 0 for p in prices.values()):
                raise ValueError(f"Invalid price in {path.name} on {day}")
            comparisons = {"open_to_low": (prices["open"], prices["low"])}
            if previous is not None:
                if previous["session_date"] == row["session_date"]:
                    raise ValueError(f"Duplicate session in {path.name}: {day}")
                prior = Decimal(previous["close"])
                comparisons.update(
                    {
                        "previous_close_to_close": (prior, prices["close"]),
                        "previous_close_to_low": (prior, prices["low"]),
                    }
                )
            breaches = {
                key: float((value / base - 1) * 100)
                for key, (base, value) in comparisons.items()
                if value < base * Decimal("0.90")
            }
            if day >= start and breaches:
                symbol = row["instrument_id"]
                findings.append(
                    {
                        "instrument_id": symbol,
                        "session_date": str(day),
                        "previous_session_date": previous["session_date"] if previous else None,
                        "previous_close": previous["close"] if previous else None,
                        "prices": {key: str(value) for key, value in prices.items()},
                        "declines_pct": breaches,
                        "recorded_actions": actions.get((symbol, str(day)), []),
                        "status": "requires_source_and_accounting_review",
                        "research_query": (
                            f"{symbol.removeprefix('NSE:')} {day} "
                            "corporate action split bonus dividend merger NSE"
                        ),
                    }
                )
            previous = row
    return findings
