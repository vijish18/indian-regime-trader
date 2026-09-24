"""Apply reviewed reference-data corrections to a NEW file, preserving input.

Corrections name expected old amounts (null for missing rows) and source
provenance. They cannot silently overwrite different data or create duplicate
entitlements. These corrections do not certify engine accounting.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage.atomic import atomic_write  # noqa: E402


def apply(rows: list[dict], corrections: list[dict]) -> list[dict]:
    result = [dict(r) for r in rows]
    seen = set()
    for correction in corrections:
        key = tuple(correction[k] for k in ("instrument_id", "ex_date", "action_type"))
        if key in seen:
            raise ValueError(f"Duplicate correction: {key}")
        seen.add(key)
        if not correction.get("source_url") or not correction.get("source_sha256"):
            raise ValueError(f"Missing provenance: {key}")
        if correction["action_type"] != "dividend":
            raise ValueError("This patcher only applies reviewed dividend corrections")
        amount = Decimal(correction["cash_amount"])
        if not amount.is_finite() or amount < 0:
            raise ValueError("Invalid dividend amount")
        matched = [
            r
            for r in result
            if tuple(r[k] for k in ("instrument_id", "ex_date", "action_type")) == key
        ]
        expected = correction["expected_cash_amount"]
        if expected is None:
            if matched:
                raise ValueError(f"Expected missing action already exists: {key}")
            row = {k: correction[k] for k in ("instrument_id", "ex_date", "action_type")}
            row["cash_amount"] = str(amount)
            row["record_date"] = correction.get("record_date") or ""
            result.append(row)
        else:
            if len(matched) != 1 or Decimal(matched[0]["cash_amount"]) != Decimal(expected):
                raise ValueError(f"Old amount does not match reviewed evidence: {key}")
            matched[0]["cash_amount"] = str(amount)
    return sorted(result, key=lambda r: (r["ex_date"], r["instrument_id"], r["action_type"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "corrections", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("output must be a new file, distinct from the input")
    with args.input.open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    corrected = apply(rows, json.loads(args.corrections.read_text())["corrections"])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(corrected)
    atomic_write(args.output, buffer.getvalue())
    print(f"Wrote {len(corrected)} reference actions to {args.output}; raw bars unchanged")
