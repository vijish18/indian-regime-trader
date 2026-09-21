"""Score the whole liquid universe and write every row to CSV.

    python scripts/export_factor_ranking.py
    python scripts/export_factor_ranking.py --as-of 2026-09-21 --out ranking.csv

``select()`` returns the top ``selection.max_holdings`` and throws the rest
away. This keeps all of it: every stock that survived the point-in-time
universe, the history filter and the liquidity floor, with each of the six
factors in both its raw units and the cross-sectional z-score the composite
actually used.

**Why both.** A z-score says where a stock sits relative to the others that
day and is the only thing the ranking reads; a raw value says what it
actually is. "Momentum z = 3.10" is meaningless without knowing that the
underlying number is a 21-day-lagged blended log return divided by realised
volatility, and the raw column is the only place that shows up.

Two columns need reading carefully:

``stability_z``
    The z-score of **negated** volatility, which is what the composite uses:
    every factor in this model is "higher is better", so a calm stock has to
    score high. ``volatility_raw`` beside it is the ordinary annualised
    figure, where higher means wilder. They move in opposite directions on
    purpose.

``drawdown_raw``
    ``close / 252-day high - 1``, so always <= 0. Closer to zero means
    closer to its own high, and scores higher.

The composite is reproduced in ``score_check`` from the columns in this file
alone -- if it does not match ``score`` the export and the selector have
drifted apart, and the file says so rather than being quietly wrong.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from config.loader import load_settings  # noqa: E402
from scripts._rerank import latest_session  # noqa: E402

COLUMNS = [
    "rank",
    "symbol",
    "instrument_id",
    "score",
    "score_check",
    "momentum_z",
    "momentum_raw",
    "relative_strength_z",
    "relative_strength_raw",
    "trend_persistence_z",
    "trend_persistence_raw",
    "stability_z",
    "volatility_raw",
    "drawdown_z",
    "drawdown_raw",
    "liquidity_z",
    "liquidity_raw_inr",
]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of",
        default=None,
        help="decision date (YYYY-MM-DD). Default: the latest session the local data covers.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--frame-cache", type=int, default=1200)
    args = parser.parse_args(argv[1:])

    as_of = (
        dt.date.fromisoformat(args.as_of)
        if args.as_of
        else latest_session(dt.date.today())
    )
    out = args.out or REPO_ROOT / "state" / f"factor_ranking_{as_of.isoformat()}.csv"

    settings = load_settings()
    weights = settings.selection.factor_weights

    from scripts.run_walk_forward import build_validator

    print(f"scoring the liquid universe as of {as_of.isoformat()} ...")
    validator = build_validator(
        snapshot_date=as_of,
        circuit_breaker_dir=REPO_ROOT / "state" / "precheck",
        frame_cache=args.frame_cache,
    )
    selector = validator.engine.stock_selector

    candidates = selector.build_candidate_universe(as_of)
    print(
        f"  {len(candidates.universe.constituents):,} in the point-in-time universe"
        f" -> {len(candidates.candidate_ids):,} liquid with enough history"
    )
    ranked = selector.score_candidates(candidates, as_of)
    print(f"  scored {len(ranked):,}")

    mismatches = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for stock in ranked:
            raw, z = stock.raw_factors, stock.standardized_factors
            check = (
                weights.momentum * z.momentum
                + weights.trend_persistence * z.trend_persistence
                + weights.relative_strength * z.relative_strength
                + weights.volatility * z.volatility
                + weights.drawdown * z.drawdown
                + weights.liquidity * z.liquidity_inr
            )
            if abs(check - stock.score) > 1e-9:
                mismatches += 1
            writer.writerow(
                {
                    "rank": stock.rank,
                    "symbol": stock.symbol,
                    "instrument_id": stock.instrument_id,
                    "score": round(stock.score, 6),
                    "score_check": round(check, 6),
                    "momentum_z": round(z.momentum, 4),
                    "momentum_raw": round(raw.momentum, 6),
                    "relative_strength_z": round(z.relative_strength, 4),
                    "relative_strength_raw": round(raw.relative_strength, 6),
                    "trend_persistence_z": round(z.trend_persistence, 4),
                    "trend_persistence_raw": round(raw.trend_persistence, 6),
                    "stability_z": round(z.volatility, 4),
                    "volatility_raw": round(raw.volatility, 6),
                    "drawdown_z": round(z.drawdown, 4),
                    "drawdown_raw": round(raw.drawdown, 6),
                    "liquidity_z": round(z.liquidity_inr, 4),
                    "liquidity_raw_inr": round(raw.liquidity_inr, 2),
                }
            )

    print(f"wrote {out}  ({len(ranked):,} rows)")
    print(
        "  weights: "
        + ", ".join(
            f"{name} {value:.2f}"
            for name, value in (
                ("momentum", weights.momentum),
                ("relative_strength", weights.relative_strength),
                ("trend_persistence", weights.trend_persistence),
                ("volatility", weights.volatility),
                ("drawdown", weights.drawdown),
                ("liquidity", weights.liquidity),
            )
        )
    )
    if mismatches:
        print(
            f"  WARNING: {mismatches} rows where score_check != score -- the export "
            "and the selector disagree about the composite"
        )
    else:
        print("  score_check reproduces score on every row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
