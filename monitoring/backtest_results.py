"""Export one Azure research run and optionally sync it to the local dashboard.

Uses only the standard library so the exporter can run over SSH stdin without
changing a frozen backtest checkout. Only completed strategy reports get curves.
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
from pathlib import Path
from typing import Any

NAMES = (
    "hmm",
    "buy_and_hold",
    "rolling_volatility",
    "moving_average_trend",
    "shuffled_regime_control",
)


def clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def export_run(root: Path) -> dict[str, Any]:
    reports, curves, progress = {}, {}, {}
    identity = None
    metadata: dict[str, Any] = {}
    for name in NAMES:
        folder = root / name
        manifest_path = folder / "series/run.manifest.json"
        if not manifest_path.exists():
            progress[name] = {"status": "pending", "completed": 0, "total": None}
            continue
        manifest = json.loads(manifest_path.read_text())
        current = (
            manifest["fingerprint"],
            manifest["start"],
            manifest["end"],
            manifest["initial_equity"],
            manifest["folds_total"],
        )
        if identity is not None and identity != current:
            raise ValueError("Refusing to combine different backtest runs")
        identity = current
        metadata = {
            k: manifest[k] for k in ("start", "end", "initial_equity", "folds_total", "git_commit")
        }
        done = manifest["completed_folds"].get(name, 0)
        complete = (folder / "completed.txt").exists() and done == manifest["folds_total"]
        progress[name] = {
            "status": "complete" if complete else "incomplete",
            "completed": done,
            "total": manifest["folds_total"],
            "updated_at": manifest["updated_at"],
        }
        if not complete:
            continue
        payload = json.loads((folder / "reports.json").read_text())
        if (payload["start"], payload["end"]) != (manifest["start"], manifest["end"]):
            raise ValueError("Report dates do not match manifest")
        report = payload["reports"][name]
        with (folder / f"series/{name}.equity.csv").open(newline="") as handle:
            points: list[dict[str, Any]] = [
                {"date": row["session_date"], "equity": float(row["equity"])}
                for row in csv.DictReader(handle)
            ]
        if not points or any(not math.isfinite(p["equity"]) for p in points):
            raise ValueError("Invalid completed equity curve")
        dates = [p["date"] for p in points]
        if dates != sorted(set(dates)) or dates[-1] > manifest["end"]:
            raise ValueError("Invalid equity-curve dates")
        starting = float(manifest["initial_equity"])
        report.update(
            starting_equity=starting,
            ending_equity=points[-1]["equity"],
            return_from_capital=points[-1]["equity"] / starting - 1,
            first_execution=dates[0],
            last_execution=dates[-1],
        )
        reports[name], curves[name] = report, points
    if identity is None:
        raise ValueError("No run manifests found")
    for item in progress.values():
        if item["total"] is None:
            item["total"] = metadata["folds_total"]
    result: dict[str, Any] = clean(
        {
            "strategies": {"available": bool(reports), "reports": reports},
            "equity_curves": curves,
            "backtest_run": {
                **metadata,
                "run_id": root.name,
                "progress": progress,
                "synced_at": dt.datetime.now(dt.UTC).isoformat(),
                "complete": len(reports) == len(NAMES),
            },
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-key", type=Path)
    parser.add_argument("--output", type=Path, default=Path("state/backtest_dashboard.json"))
    parser.add_argument("--watch", type=int, default=0)
    args = parser.parse_args()
    if not args.ssh_host:
        print(json.dumps(export_run(Path(args.run_root)), allow_nan=False))
        return
    if args.ssh_key is None:
        parser.error("--ssh-key is required with --ssh-host")
    command = [
        "ssh",
        "-i",
        str(args.ssh_key.resolve()),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        args.ssh_host,
        "python3 - --run-root " + shlex.quote(args.run_root),
    ]
    source = Path(__file__).read_text(encoding="utf-8")
    while True:
        try:
            result = subprocess.run(
                command, input=source, capture_output=True, text=True, timeout=60, check=True
            )
            payload = json.loads(result.stdout)
            body = json.dumps(payload, allow_nan=False)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=args.output.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(body)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, args.output)
            finally:
                Path(temporary).unlink(missing_ok=True)
            count = len(payload["strategies"]["reports"])
            print(f"{payload['backtest_run']['synced_at']}: {count}/5 reports synced", flush=True)
            if payload["backtest_run"]["complete"] or not args.watch:
                return
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f"Sync failed; retaining last snapshot: {exc}", flush=True)
            if not args.watch:
                raise
        time.sleep(max(args.watch, 10))


if __name__ == "__main__":
    main()
