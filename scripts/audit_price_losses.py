"""Create a research queue for raw-price falls >10%; never rewrite bhavcopy.

Exit 1 means unresolved cases exist, 0 means this screen found none.
Neither outcome certifies completeness of corporate-action accounting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.price_anomalies import scan_price_losses  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--from", dest="start", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--to", dest="end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("start must not follow end")
    findings = scan_price_losses(args.data_root, args.start, args.end)
    payload = {
        "threshold_pct": 10,
        "start": str(args.start),
        "end": str(args.end),
        "unresolved_count": len(findings),
        "findings": findings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"{len(findings)} unresolved price-loss cases: {args.output}")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
