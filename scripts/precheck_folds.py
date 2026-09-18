"""Fit and filter every walk-forward fold before committing to a long run.

    python scripts/precheck_folds.py --from 2015-01-01 --to 2026-09-15

``_fit_fold`` reads only the NIFTY 50 and India VIX series -- no equity bars
-- so all 32 folds cost about two minutes, against the hours the full
backtest takes. A fold that cannot fit or cannot be filtered would
otherwise kill the run partway through, and nothing about that run is
checkpointed.

This is not hypothetical. It found fold 11 dying in inference:

    HMMNumericalError: belief collapsed to zero probability mass; the
    observation is impossible under every state of this model

which would have ended a 24-hour run about a third of the way in. The cause
was the filter starting from the model's fitted ``start_probabilities`` --
one-hot, because Baum-Welch on a single sequence drives it there -- on a
window beginning in April 2020, which that state explained 2,019 nats
worse than the state that fit. See ``HMMRegimeEngine.filter``.

It also writes the per-fold out-of-sample regime mix, which is what the
dashboard's regime panel draws.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_walk_forward import build_validator  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=dt.date.fromisoformat,
                        default=dt.date(2015, 1, 1))
    parser.add_argument("--to", dest="end", type=dt.date.fromisoformat,
                        default=dt.date(2026, 9, 15))
    parser.add_argument("--state-dir", type=Path,
                        default=REPO_ROOT / "state" / "precheck")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "state" / "fold_regimes.json")
    args = parser.parse_args(argv[1:])

    args.state_dir.mkdir(parents=True, exist_ok=True)
    validator = build_validator(
        snapshot_date=args.end,
        circuit_breaker_dir=args.state_dir,
        frame_cache=0,  # equity bars are not read on this path at all
    )
    folds = validator.generate_folds(args.start, args.end)
    if not folds:
        raise SystemExit("no folds in range")

    print(f"{len(folds)} folds\n")
    print(f"{'#':>3} {'test window':<26} {'fit':>5} {'inf':>5}  regimes")

    failures: list[tuple[int, str, str]] = []
    collected: list[dict[str, object]] = []

    for index, (train_start, train_end, test_start, test_end) in enumerate(folds, 1):
        started = time.perf_counter()
        try:
            model, params, engine, model_id = validator._fit_fold(train_start, train_end)
        except Exception as exc:  # noqa: BLE001 - this is the thing being probed
            failures.append((index, "FIT", f"{type(exc).__name__}: {exc}"))
            print(f"{index:>3} {test_start}..{test_end}  "
                  f"{time.perf_counter() - started:>5.1f}     -  *** FIT FAILED: {exc}")
            continue
        fitted = time.perf_counter()

        try:
            targets, states = validator._hmm_exposure_targets(
                model, params, engine, test_start, test_end
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((index, "INFERENCE", f"{type(exc).__name__}: {exc}"))
            print(f"{index:>3} {test_start}..{test_end}  {fitted - started:>5.1f} "
                  f"{time.perf_counter() - fitted:>5.1f}  *** INFERENCE FAILED: {exc}")
            continue

        mix = Counter(state.label.value for state in states)
        allowed = sum(1 for target in targets.values() if target.allow_new_positions)
        collected.append(
            {
                "fold": index,
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "model_id": model_id,
                "regimes": dict(mix),
                "sessions": len(targets),
                "allow_new_positions": allowed,
            }
        )
        print(f"{index:>3} {test_start}..{test_end}  {fitted - started:>5.1f} "
              f"{time.perf_counter() - fitted:>5.1f}  {dict(mix)} "
              f"allow_new {allowed}/{len(targets)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"folds": collected, "failures": failures}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.out}")

    if failures:
        print(f"\n{len(failures)} of {len(folds)} fold(s) would kill the full run:")
        for index, stage, message in failures:
            print(f"  fold {index:>2} [{stage}]: {message}")
        return 1
    print(f"all {len(folds)} folds fit and infer -- the run will not die on the model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
