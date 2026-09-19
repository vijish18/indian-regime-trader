"""Gather everything the dashboard shows into one JSON file.

    python scripts/collect_dashboard_data.py --out state/dashboard_data.json

Collection is kept apart from rendering on purpose. Every number on the
dashboard comes from a file on disk that something else produced -- the
bhavcopy backfill, the walk-forward run, the cost schedule -- so the
rendering step has nothing to compute and no opportunity to invent. If a
figure looks wrong, it is wrong in the artifact named beside it here.

Panels whose inputs are absent are reported as absent rather than
defaulted. A dashboard that shows a plausible zero where it means "the
backtest has not finished" is worse than one that says so: the zero gets
read as a result.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from backtest.walk_forward import STRATEGY_NAMES  # noqa: E402
from data.calendar import NSETradingCalendar  # noqa: E402

DATA_CACHE = REPO_ROOT / "data_cache"
REFERENCE = DATA_CACHE / "reference"
HOLIDAY_FILE = REPO_ROOT / "config" / "nse_holidays.csv"
COST_SCHEDULE = REPO_ROOT / "config" / "cost_schedules.yaml"


def _count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def provenance() -> dict[str, Any]:
    """Where the numbers came from, counted rather than asserted."""
    bars = DATA_CACHE / "raw" / "equity_bars"
    bhavcopy = list(DATA_CACHE.glob("bhavcopy/*.zip")) or list(DATA_CACHE.glob("**/bhavcopy-*.zip"))
    bar_files = sorted(bars.glob("*.csv")) if bars.is_dir() else []
    total_bars = sum(_count_rows(p) for p in bar_files)
    calendar = NSETradingCalendar.from_file(HOLIDAY_FILE)
    return {
        "bhavcopy_archives": len(bhavcopy),
        "instrument_bar_files": len(bar_files),
        "total_bars": total_bars,
        "instruments": _count_rows(REFERENCE / "instruments.csv"),
        "corporate_actions": _count_rows(REFERENCE / "corporate_actions.csv"),
        "membership_spans": _count_rows(REFERENCE / "index_membership.csv"),
        "calendar_years": sorted(calendar.covered_years),
        "index_sessions": _count_rows(DATA_CACHE / "raw" / "index" / "NIFTY50.csv"),
    }


def universe_growth(sample_every_days: int = 90) -> list[dict[str, Any]]:
    """Point-in-time universe size over history.

    This is the survivorship-bias fix made visible: the universe is small
    early and large late because that is what actually traded, not because
    today's list was projected backwards.
    """
    path = REFERENCE / "index_membership.csv"
    if not path.is_file():
        return []
    spans: list[tuple[dt.date, dt.date]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                start = dt.date.fromisoformat(row["effective_from"])
            except (KeyError, ValueError):
                continue
            end_text = row.get("effective_to") or ""
            end = dt.date.fromisoformat(end_text) if end_text else dt.date(2100, 1, 1)
            spans.append((start, end))
    if not spans:
        return []

    first = min(s for s, _ in spans)
    last = min(max(e for _, e in spans), dt.date.today())
    out: list[dict[str, Any]] = []
    day = first
    while day <= last:
        out.append(
            {"date": day.isoformat(), "count": sum(1 for s, e in spans if s <= day <= e)}
        )
        day += dt.timedelta(days=sample_every_days)
    return out


def cost_eras() -> list[dict[str, Any]]:
    if not COST_SCHEDULE.is_file():
        return []
    payload = yaml.safe_load(COST_SCHEDULE.read_text(encoding="utf-8"))
    out = []
    for entry in payload.get("schedules", []):
        out.append(
            {
                "effective_from": str(entry["effective_from"]),
                "label": entry.get("label", ""),
                "stt_round_trip_bps": (entry["stt_buy_pct"] + entry["stt_sell_pct"]) * 10_000,
                "stamp_duty_buy_bps": entry["stamp_duty_buy_pct"] * 10_000,
                "exchange_txn_bps": entry["exchange_txn_pct"] * 10_000,
                "gst_pct": entry["gst_pct"] * 100,
            }
        )
    return out


def strategy_results(json_dir: Path | None) -> dict[str, Any]:
    """The five-way comparison, if the walk-forward run has produced it."""
    if json_dir is None or not json_dir.is_dir():
        return {"available": False, "reason": "no --json-dir given", "reports": {}}
    files = sorted(json_dir.glob("*.json"))
    if not files:
        return {"available": False, "reason": f"no JSON in {json_dir}", "reports": {}}

    reports: dict[str, Any] = {}
    window: dict[str, str] = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        window = {"start": payload["start"], "end": payload["end"]}
        reports.update(payload["reports"])

    missing = [name for name in STRATEGY_NAMES if name not in reports]
    return {
        # Deliberately not "available" until every strategy is in: the HMM is
        # judged against the baselines and the shuffled control, and a table
        # with some of them missing invites a conclusion the data does not
        # support.
        "available": not missing,
        "reason": f"still running: {missing}" if missing else "",
        "missing": missing,
        "window": window,
        "reports": {name: reports[name] for name in STRATEGY_NAMES if name in reports},
    }


def equity_curves(series_dir: Path | None, sample_every: int = 5) -> dict[str, Any]:
    """Equity curves, thinned for the browser.

    Sampling every 5th session keeps ~800 points per strategy instead of
    ~4,000, which is below the resolution of any chart this size and keeps
    the page from carrying four times the data it can draw.
    """
    if series_dir is None or not series_dir.is_dir():
        return {}
    out: dict[str, Any] = {}
    for path in sorted(series_dir.glob("*.equity.csv")):
        name = path.name.removesuffix(".equity.csv")
        points: list[dict[str, Any]] = []
        with path.open(encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if index % sample_every:
                    continue
                try:
                    points.append(
                        {"date": row["session_date"][:10], "equity": float(row["equity"])}
                    )
                except (KeyError, ValueError):
                    continue
        if points:
            out[name] = points
    return out


def hmm_model() -> dict[str, Any]:
    """The approved model's internals, for the regime panel.

    The transition matrix and the per-state statistics are what the HMM
    actually *is* -- a regime label is a name someone chose, but the measured
    volatility, return and persistence behind it are the things the risk
    policy consumes. Showing the label without them would put the one part a
    human picked on screen and hide the parts the machine measured.
    """
    try:
        from core.regime.model_registry import ModelRegistry

        artifact = ModelRegistry(REPO_ROOT / "model_registry").load_current_approved()
    except Exception as exc:  # noqa: BLE001 - absent registry is a normal state here
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    model = artifact.model
    states = []
    for state_id in range(model.n_states):
        stat = model.statistics_for(state_id)
        states.append(
            {
                "state_id": state_id,
                "label": stat.label.value,
                "expected_volatility": stat.expected_volatility,
                "expected_return": stat.expected_return,
                "persistence": stat.self_transition_probability,
            }
        )
    # Volatility order, which is the order the labels were assigned in:
    # assign_labels ranks states by measured volatility and spreads them
    # across calm/normal/elevated/crisis. Presenting them by state_id hides
    # that and puts crisis in the middle of the row.
    #
    # It also makes the collision legible. With five states and four labels
    # two states share a name -- here s2 (+49.8% expected return) and s4
    # (-0.06%) are both "elevated" -- and sorted by volatility they sit next
    # to each other where the difference is obvious, rather than looking like
    # one thing mentioned twice.
    states.sort(key=lambda s: s["expected_volatility"])
    training = model.training_result
    return {
        "available": True,
        "model_id": artifact.model_id,
        "created_at": artifact.created_at.isoformat(timespec="seconds"),
        "feature_version": artifact.feature_version,
        "n_states": model.n_states,
        "features": list(training.feature_columns),
        "states": states,
        "transition_matrix": [
            [float(v) for v in row] for row in model.parameters.transition_matrix
        ],
        "bic": float(training.bic),
        "aic": float(training.aic),
        "converged": bool(training.converged),
        "iterations": int(training.iterations),
    }


def stock_selection(as_of_text: str | None, frame_cache: int = 400) -> dict[str, Any]:
    """What the model would hold, and why each name earned its place.

    This is the honest answer to "which stocks does the model predict". It
    predicts none: there is no price forecast anywhere in this system. What
    it produces is a *ranking* -- a composite of momentum, trend persistence,
    relative strength and (negated) volatility, each standardised
    cross-sectionally against the rest of that day's candidates -- and a
    target weight derived from it. So the factor z-scores travel with every
    row: a rank without them is an opinion, and with them it is auditable.

    Running the real selector rather than reading a cache, because the
    selector is the thing under inspection. It takes a couple of minutes.
    """
    if as_of_text is None:
        return {"available": False, "reason": "not requested"}
    try:
        as_of = dt.date.fromisoformat(as_of_text)
        from scripts.run_walk_forward import build_validator

        validator = build_validator(
            snapshot_date=as_of,
            circuit_breaker_dir=REPO_ROOT / "state" / "precheck",
            frame_cache=frame_cache,
        )
        scores = validator.engine.stock_selector.select(as_of)
    except Exception as exc:  # noqa: BLE001 - reported, never faked
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    picks = []
    for score in scores:
        raw, std = score.raw_factors, score.standardized_factors
        picks.append(
            {
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
        )
    return {
        "available": True,
        "as_of": as_of.isoformat(),
        "count": len(picks),
        "picks": picks,
        "note": (
            "A ranking, not a price forecast. This system produces target weights; "
            "nothing in it predicts where a stock will trade."
        ),
    }


def broker_account() -> dict[str, Any]:
    """Zerodha session state, reported rather than assumed.

    Two facts have to travel together here or the panel misleads. The access
    token is valid for one trading day, so a stale file is the normal state
    rather than an error. And this system has never placed an order: live
    trading has been disabled throughout, so whatever sits in that account is
    the operator's own investing and must never be rendered as though the
    strategy produced it.
    """
    path = REPO_ROOT / "state" / "kite_session.json"
    if not path.is_file():
        return {"session": "absent", "traded_live": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"session": "unreadable", "reason": str(exc), "traded_live": False}

    expires = str(payload.get("expires_at", ""))
    expired = True
    try:
        expired = dt.datetime.fromisoformat(expires) <= dt.datetime.now(dt.UTC)
    except ValueError:
        pass
    return {
        "session": "expired" if expired else "valid",
        "obtained_at": str(payload.get("obtained_at", "")),
        "expires_at": expires,
        # Deliberately not the token, and not the api_key.
        "traded_live": False,
        "note": (
            "This system has never placed an order. Holdings in this account, if any, "
            "are manual investments and are not strategy performance."
        ),
    }


def regime_distribution(cache: Path) -> dict[str, Any]:
    """Per-fold out-of-sample regime mix, if it has been collected.

    Produced by scripts/precheck_folds.py, not here: it needs a fitted model
    per fold, and recomputing that every time the dashboard is rendered
    would put minutes of HMM fitting behind a page refresh.
    """
    if not cache.is_file():
        return {"available": False, "folds": []}
    return {"available": True, **json.loads(cache.read_text(encoding="utf-8"))}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, default=REPO_ROOT / "state" / "wf_json")
    parser.add_argument("--series-dir", type=Path, default=REPO_ROOT / "state" / "wf_series")
    parser.add_argument(
        "--regime-cache", type=Path, default=REPO_ROOT / "state" / "fold_regimes.json"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "state" / "dashboard_data.json")
    parser.add_argument(
        "--selection-as-of",
        default=None,
        help="run the real stock selector for this date (YYYY-MM-DD); takes a few minutes",
    )
    args = parser.parse_args(argv[1:])

    data: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "provenance": provenance(),
        "universe_growth": universe_growth(),
        "cost_eras": cost_eras(),
        "strategies": strategy_results(args.json_dir),
        "equity_curves": equity_curves(args.series_dir),
        "regimes": regime_distribution(args.regime_cache),
        "hmm": hmm_model(),
        "selection": stock_selection(args.selection_as_of),
        "broker": broker_account(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    prov = data["provenance"]
    print(f"wrote {args.out}")
    print(f"  {prov['instruments']:,} instruments, {prov['total_bars']:,} bars, "
          f"{prov['corporate_actions']:,} corporate actions")
    print(f"  universe samples : {len(data['universe_growth'])}")
    print(f"  cost eras        : {len(data['cost_eras'])}")
    sel = data["selection"]
    print(f"  stock selection  : {sel['count'] if sel.get('available') else sel.get('reason')}")
    print(f"  equity curves    : {len(data['equity_curves'])}")
    strategies = data["strategies"]
    if strategies["available"]:
        print(f"  strategies       : all {len(strategies['reports'])} present")
    else:
        print(f"  strategies       : NOT COMPLETE ({strategies['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
