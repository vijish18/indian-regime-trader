"""Reconcile held corporate events against dated PR evidence; fail closed.

Source terms being matched is distinct from engine accounting being verified.
This report cannot certify a run with outstanding entitlement-model gaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.nse_pr_actions import all_terms, parse_date  # noqa: E402
from storage.atomic import atomic_write  # noqa: E402


def reconcile(
    audit: dict, evidence: dict, references: list[dict], supplements: Sequence[dict] = ()
) -> dict:
    indexed = defaultdict(dict)
    for day, archive in sorted(evidence.items()):
        for row in archive.get("rows", []):
            if row.get("SERIES") not in {"EQ", "BE", "BZ"}:
                continue
            ex_date = parse_date(row["EX_DT"])
            if not ex_date:
                continue
            key = ("NSE:" + row["SYMBOL"], ex_date)
            # Repeated EQ/BE/BL lines and repeated archives are evidence of
            # the same event, not multiple entitlements. Preserve variants.
            indexed[key].setdefault(
                row["PURPOSE"],
                {
                    "purpose": row["PURPOSE"],
                    "terms": all_terms(row["PURPOSE"]),
                    "source_url": archive["source_url"],
                    "sha256": archive["sha256"],
                    "archive_date": day,
                    "record_date": parse_date(row.get("RECORD_DT", "")),
                },
            )
    for source in supplements:
        key = (source["instrument_id"], source["ex_date"])
        indexed[key][source["source_url"]] = {
            "purpose": source["notes"],
            "terms": [
                {
                    k: source[k]
                    for k in ("action_type", "ratio_new", "ratio_old", "cash_amount")
                    if k in source
                }
            ],
            "source_url": source["source_url"],
            "archive_date": source.get("announcement_date"),
            "supplemental_terms": source,
        }
    refs = defaultdict(list)
    for row in references:
        refs[(row["instrument_id"], row["ex_date"])].append(row)
    events = []
    for event in audit["events"]:
        key = (event["instrument_id"], event["ex_date"])
        sources = list(indexed[key].values())
        matches = []
        for action in event["actions"]:

            def same(source: dict, action: dict = action) -> bool:
                fields = (
                    ("cash_amount",)
                    if action["action_type"] == "dividend"
                    else (
                        ("ratio_new", "ratio_old")
                        if action["action_type"] in {"split", "bonus"}
                        else ()
                    )
                )
                return any(
                    parsed["action_type"] == action["action_type"]
                    and all(
                        action.get(k) and parsed.get(k) and Decimal(action[k]) == Decimal(parsed[k])
                        for k in fields
                    )
                    for parsed in source["terms"]
                )

            matches.append({"action": action, "source_matches": [s for s in sources if same(s)]})
        events.append(
            {
                **event,
                "source_checks": matches,
                "source_status": "matched"
                if all(m["source_matches"] for m in matches)
                else "unresolved",
                "all_source_variants": sources,
            }
        )
    missing = []
    unparsed = []
    for iid, days in audit["held_days"].items():
        for day in days:
            key = (iid, day)
            existing_types = {r["action_type"] for r in refs[key]}
            for source in indexed[key].values():
                purpose = source["purpose"].upper()
                if any(p["action_type"] == "unresolved" for p in source["terms"]) and any(
                    word in purpose for word in ("DIV", "BONUS", "SPLT", "SPLIT", "RIGHT", "MERGER")
                ):
                    unparsed.append({"instrument_id": iid, "ex_date": day, "source": source})
            missing_sources = [
                s
                for s in indexed[key].values()
                if any(
                    p["action_type"] not in existing_types and p["action_type"] != "unresolved"
                    for p in s["terms"]
                )
            ]
            if missing_sources:
                missing.append(
                    {"instrument_id": iid, "ex_date": day, "source_events": missing_sources}
                )
    return {
        "validation_status": "blocked",
        "scope": "source reconciliation for holdings in the supplied diagnostic run",
        "source_archive_count": len(evidence),
        "source_errors": {k: v["error"] for k, v in evidence.items() if "error" in v},
        "events": events,
        "missing_reference_events": missing,
        "unparsed_held_purposes": unparsed,
        "accounting_blockers": [
            "Dividend payment dates and receivables are not modelled; cash credited on ex-date",
            "Bonus share-credit/tradability dates and fractional entitlements not modelled",
            "Demerger successor shares, basis allocation and availability not modelled",
            "Delisting tender submission/acceptance/payment not modelled",
            "Adjusted-price fallback can bypass unresolved event terms",
            "Price-drop anomalies without a matching event still require classification",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "evidence", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--supplement", type=Path)
    args = parser.parse_args()
    with args.reference.open(encoding="utf-8-sig") as f:
        refs = list(csv.DictReader(f))
    result = reconcile(
        json.loads(args.audit.read_text()),
        json.loads(args.evidence.read_text()),
        refs,
        json.loads(args.supplement.read_text())["events"] if args.supplement else (),
    )
    atomic_write(args.output, json.dumps(result, indent=2))
    matched = sum(e["source_status"] == "matched" for e in result["events"])
    print(
        f"{matched}/{len(result['events'])} source-matched; "
        f"{len(result['missing_reference_events'])} missing-reference event dates; "
        "accounting validation BLOCKED"
    )
