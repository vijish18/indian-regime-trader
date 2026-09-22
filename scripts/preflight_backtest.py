"""Check everything a walk-forward run needs, before it costs hours.

    python scripts/preflight_backtest.py --from 2015-01-01 --to 2026-09-01

A full run is measured in hours. Every check here exists because something
once failed *after* those hours had been spent, on a fold in the twenties or
thirties, and discarded the lot. They are cheap -- the whole preflight is
seconds -- and they are the difference between finding out now and finding
out tomorrow morning.

What each one is for:

``config``
    The schema and the model both have to accept settings.yaml. A run that
    starts and then fails to load a config block has wasted the fitting.

``stop policy``
    Printed, not just validated. The run is *about* the stop rule, and a
    result whose thresholds nobody recorded cannot be compared to anything.

``folds``
    Non-empty, and tiling rather than overlapping. Overlapping folds double
    count sessions and quietly inflate every statistic; the validator now
    refuses them at construction, and this says so before the run.

``cost schedule``
    Must cover every fold's dates. A run over 2015-2026 once failed closed on
    its first fold because the schedule started in 2019 -- correct behaviour,
    discovered expensively.

``index history``
    NIFTY50 and INDIAVIX over the whole range. The features are built from
    them, so a gap is a fold that cannot be fitted.

``universe``
    Membership spans covering the range, or the selector has nothing to rank.

``unadjustable actions``
    The historic killer. A demerger has no derivable price factor, so
    adjusting a bar across one raises -- and a held position still has to be
    marked and sold through the event even after the universe stops selecting
    it. Four separate runs died this way at four different call sites
    (ADANITRANS, TATACHEM, VEDL twice). The fix was to catch it at the
    source, in ``data.market_data._adjust``; this check proves the fix is
    still in place by adjusting a real bar across a real demerger and
    requiring that it returns rather than raises.

``output``
    The directories are writable and say whether a resumable checkpoint is
    already there.

Exit code is 0 when everything passed, 1 otherwise, so this can gate a run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

DATA_CACHE = Path(os.environ.get("IRT_DATA_ROOT", REPO_ROOT / "data_cache"))
REFERENCE = DATA_CACHE / "reference"

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Check:
    """One preflight result. ``WARN`` does not fail the run."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = PASS
        self.notes: list[str] = []

    def note(self, text: str) -> None:
        self.notes.append(text)

    def warn(self, text: str) -> None:
        self.status = WARN if self.status == PASS else self.status
        self.notes.append(text)

    def fail(self, text: str) -> None:
        self.status = FAIL
        self.notes.append(text)


def check_config() -> Check:
    check = Check("config")
    try:
        from config.loader import load_settings

        settings = load_settings()
    except Exception as exc:  # noqa: BLE001 - the whole point is to report it
        check.fail(f"{type(exc).__name__}: {exc}")
        return check
    check.note("settings.yaml validates against the schema and the model")
    check.note(
        f"selection: top {settings.selection.max_holdings}, "
        f"liquidity floor Rs {settings.universe.min_avg_daily_value_inr:,.0f}/day"
    )
    return check


def check_stop_policy() -> Check:
    check = Check("stop policy")
    try:
        from config.loader import load_settings
        from risk.stop_loss import StopLossPolicy

        policy = StopLossPolicy.from_mapping(load_settings().risk.stop_loss.model_dump())
    except Exception as exc:  # noqa: BLE001
        check.fail(f"{type(exc).__name__}: {exc}")
        return check
    if not policy.enabled:
        check.warn("stops are DISABLED for this run -- that is a baseline, not the rule")
        return check
    profit = (
        f"take profit at +{policy.trail_arm_net_profit_pct:.1%} net"
        if policy.close_on_arm
        else (
            f"trail {policy.trail_drop_pct:.1%} off the session high once "
            f"+{policy.trail_arm_net_profit_pct:.1%} net ahead"
        )
    )
    check.note(f"hard stop at -{policy.hard_stop_pct:.1%} from the buy price")
    check.note(profit)
    return check


def check_folds(start: dt.date, end: dt.date) -> tuple[Check, int]:
    check = Check("folds")
    try:
        from config.loader import load_settings
        from data.calendar import NSETradingCalendar

        config = load_settings().backtest
        calendar = NSETradingCalendar.from_file(REPO_ROOT / "config" / "nse_holidays.csv")
    except Exception as exc:  # noqa: BLE001
        check.fail(f"{type(exc).__name__}: {exc}")
        return check, 0
    if config.roll_step_sessions < config.test_window_sessions:
        check.fail(
            f"roll_step_sessions={config.roll_step_sessions} < "
            f"test_window_sessions={config.test_window_sessions}: folds would overlap "
            "and every statistic would double count the overlap"
        )
        return check, 0
    sessions = len(calendar.trading_days_between(start, end))
    needed = config.training_window_sessions + 2
    estimate = max(0, (sessions - needed) // config.roll_step_sessions + 1)
    if estimate <= 0:
        check.fail(
            f"{sessions} sessions in range, but a fold needs at least {needed} "
            f"({config.training_window_sessions} train + one signal + one execution)"
        )
        return check, 0
    check.note(
        f"{estimate} folds: {config.training_window_sessions} train, "
        f"up to {config.test_window_sessions} test signals, step {config.roll_step_sessions}; "
        "partial final window included, execution bounded by cutoff"
    )
    return check, estimate


def check_cost_schedule(start: dt.date, end: dt.date) -> Check:
    check = Check("cost schedule")
    try:
        from backtest.cost_schedule import CostScheduleRepository

        repo = CostScheduleRepository.from_file(REPO_ROOT / "config" / "cost_schedules.yaml")
    except Exception as exc:  # noqa: BLE001
        check.fail(f"{type(exc).__name__}: {exc}")
        return check
    probe = start
    missing: list[dt.date] = []
    while probe <= end:
        try:
            repo.schedule_as_of(probe)
        except Exception:  # noqa: BLE001 - MissingCostScheduleError and anything else
            missing.append(probe)
        probe += dt.timedelta(days=30)
    if missing:
        check.fail(
            f"no schedule covers {len(missing)} probed dates, earliest {missing[0]}. "
            "The run would fail closed on the first fold that reaches one."
        )
        return check
    check.note(f"covers every probed date from {start} to {end}")
    return check


def check_index_history(start: dt.date, end: dt.date) -> Check:
    check = Check("index history")
    try:
        from data.market_data import LocalMarketDataProvider
        from data.storage import LocalDataStore, StorageFormat

        store = LocalDataStore(
            raw_root=DATA_CACHE / "raw",
            normalized_root=DATA_CACHE / "normalized",
            reference_root=REFERENCE,
            storage_format=StorageFormat.CSV,
        )
        market = LocalMarketDataProvider(store)
    except Exception as exc:  # noqa: BLE001
        check.fail(f"{type(exc).__name__}: {exc}")
        return check

    for symbol in ("NIFTY50", "INDIAVIX"):
        try:
            observations = market.get_index_observations(symbol, start, end)
        except Exception as exc:  # noqa: BLE001
            check.fail(f"{symbol}: {type(exc).__name__}: {exc}")
            continue
        if not observations:
            check.fail(f"{symbol}: no observations in range")
            continue
        first, last = observations[0].session_date, observations[-1].session_date
        check.note(f"{symbol}: {len(observations):,} sessions, {first} to {last}")
        if (first - start).days > 10:
            check.warn(f"{symbol} starts {(first - start).days} days after --from")
        if (end - last).days > 10:
            check.warn(f"{symbol} ends {(end - last).days} days before --to")
    return check


def check_universe(start: dt.date, end: dt.date) -> Check:
    check = Check("universe")
    path = REFERENCE / "index_membership.csv"
    if not path.exists():
        check.fail(f"{path} is missing; run scripts/backfill_bhavcopy.py")
        return check
    import csv

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        check.fail(f"{path} is empty")
        return check
    starts = [dt.date.fromisoformat(r["effective_from"]) for r in rows]
    check.note(f"{len(rows):,} membership spans, earliest {min(starts)}")

    # Probe from the first date the run actually TRADES, not from --from.
    # Nothing is selected during the training window, and the universe is
    # empty for the first 20 sessions of the data by construction -- the
    # liquidity floor is a 20-session median, so there is nothing to be
    # eligible until there are 20 sessions to measure. Probing --from
    # reported "0 eligible" on a range that is perfectly fine.
    from config.loader import load_settings

    training_days = int(load_settings().backtest.training_window_sessions * 365 / 252)
    first_trade = min(start + dt.timedelta(days=training_days), end)
    if min(starts) > first_trade:
        check.fail(
            f"membership starts {min(starts)}, after the first test date {first_trade}"
        )
        return check
    check.note(f"first test date is about {first_trade} (after the training window)")
    for probe_date in (first_trade, first_trade + (end - first_trade) / 2, end):
        active = sum(
            1
            for r in rows
            if dt.date.fromisoformat(r["effective_from"]) <= probe_date
            and (not r["effective_to"] or dt.date.fromisoformat(r["effective_to"]) >= probe_date)
        )
        check.note(f"{probe_date}: {active:,} eligible")
        if active < 50:
            check.fail(f"only {active} eligible on {probe_date}; the selector needs candidates")
    return check


def check_unadjustable_actions() -> Check:
    """The crash that killed four runs, proven handled rather than assumed."""
    check = Check("unadjustable actions")
    try:
        import csv

        from data.corporate_actions import InMemoryCorporateActionProvider
        from data.errors import DataNotAvailableError
        from data.market_data import LocalMarketDataProvider
        from data.models import PriceBasis
        from data.storage import LocalDataStore, StorageFormat

        actions = InMemoryCorporateActionProvider.from_file(REFERENCE / "corporate_actions.csv")
        store = LocalDataStore(
            raw_root=DATA_CACHE / "raw",
            normalized_root=DATA_CACHE / "normalized",
            reference_root=REFERENCE,
            storage_format=StorageFormat.CSV,
        )
        market = LocalMarketDataProvider(store, corporate_actions=actions)
    except Exception as exc:  # noqa: BLE001
        check.fail(f"{type(exc).__name__}: {exc}")
        return check

    rows = list((REFERENCE / "corporate_actions.csv").open(encoding="utf-8"))
    reader = csv.DictReader(rows)
    risky = [
        r
        for r in reader
        if r["action_type"] in {"demerger", "merger"}
        and not r.get("explicit_price_factor")
    ]
    if not risky:
        check.note("no demergers or mergers without an explicit factor in the reference data")
        return check
    check.note(f"{len(risky)} demerger/merger events with no derivable price factor")

    tested = crashed = survived = no_data = 0
    for row in risky[:40]:
        instrument_id = row["instrument_id"]
        ex_date = dt.date.fromisoformat(row["ex_date"])
        try:
            market.get_equity_bars(
                instrument_id,
                ex_date - dt.timedelta(days=20),
                ex_date + dt.timedelta(days=5),
                price_basis=PriceBasis.ADJUSTED,
            )
        except DataNotAvailableError:
            # A different failure class, and not the one this checks. These
            # are names with no local bar file at all -- long delisted, or
            # never liquid enough to be stored. Every caller in the engine
            # already treats "no data" as "no price" and moves on; it is the
            # *adjustment* raising on data that exists that killed the runs.
            no_data += 1
            continue
        except Exception as exc:  # noqa: BLE001 - this is the regression being checked
            tested += 1
            crashed += 1
            if crashed <= 3:
                check.fail(
                    f"{instrument_id} ex {ex_date}: adjusting across the event raised "
                    f"{type(exc).__name__}: {exc}"
                )
            continue
        tested += 1
        survived += 1
    if no_data:
        check.note(f"{no_data} sampled events are on names with no local bars (not applicable)")
    if not tested:
        check.warn("no sampled event had local bars to adjust; the fallback was not exercised")
        return check
    if crashed:
        check.fail(
            f"{crashed} of {tested} sampled events still raise. This is the failure that "
            "killed folds 9, 20 and 32 of earlier runs; data/market_data._adjust is "
            "supposed to count them and fall back to unadjusted prices."
        )
    else:
        check.note(
            f"adjusted real bars across {survived} of them without raising "
            "(the fallback in data/market_data._adjust is doing its job)"
        )
    return check


def check_output(series_dir: Path | None, json_out: Path | None) -> Check:
    check = Check("output")
    for label, path in (("series-dir", series_dir), ("json-out", json_out)):
        if path is None:
            check.warn(f"--{label} not set: this run keeps no checkpoint and cannot resume")
            continue
        target = path if label == "series-dir" else path.parent
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".preflight"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            check.fail(f"{target} is not writable: {exc}")
            continue
        check.note(f"{target} writable")

    if series_dir is not None:
        manifest = series_dir / "run.manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                done = data.get("completed_folds") or {}
                total = data.get("folds_total")
                summary = ", ".join(f"{k} {v}/{total}" for k, v in sorted(done.items()))
                check.note(f"existing checkpoint: {summary or 'no folds yet'}")
                check.note("  pass --resume to continue it, or use a new --series-dir")
            except (OSError, ValueError) as exc:
                check.warn(f"checkpoint manifest is unreadable: {exc}")
        else:
            check.note("no existing checkpoint; this will be a fresh run")
    return check


def check_series_history() -> Check:
    """Reject old EQ-only exports and ambiguous per-session price records."""
    from collections import Counter

    import pandas as pd

    from data.nse_bhavcopy import VALUATION_SERIES

    check = Check("equity series history")
    paths = sorted((DATA_CACHE / "raw" / "equity_bars").glob("*.csv"))
    counts: Counter[str] = Counter()
    if not paths:
        check.fail("No equity bar files")
        return check
    for path in paths:
        try:
            frame = pd.read_csv(path, usecols=["instrument_id", "session_date", "trading_series"])
            if frame.duplicated(["instrument_id", "session_date"]).any():
                raise ValueError("duplicate instrument/session prices")
            if not frame.trading_series.isin(VALUATION_SERIES).all():
                raise ValueError("missing or unsupported trading series")
            counts.update(frame.trading_series.value_counts().to_dict())
        except (OSError, ValueError) as exc:
            check.fail(f"{path.name}: {exc}; rebuild with scripts/build_equity_bars.py")
            return check
    check.note(f"{len(paths):,} instruments; observed series counts {dict(sorted(counts.items()))}")
    check.note("BE/BZ retained for held-position pricing/exits; new entries remain EQ-only")
    return check


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=dt.date.fromisoformat,
                        default=dt.date(2015, 1, 1))
    parser.add_argument("--to", dest="end", type=dt.date.fromisoformat,
                        default=dt.date(2026, 9, 1))
    parser.add_argument("--series-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv[1:])

    print(f"preflight for a walk-forward run over {args.start} .. {args.end}\n")

    checks: list[Check] = [check_config(), check_stop_policy()]
    fold_check, _ = check_folds(args.start, args.end)
    checks.append(fold_check)
    checks.append(check_cost_schedule(args.start, args.end))
    checks.append(check_index_history(args.start, args.end))
    checks.append(check_universe(args.start, args.end))
    checks.append(check_series_history())
    checks.append(check_unadjustable_actions())
    checks.append(check_output(args.series_dir, args.json_out))

    for check in checks:
        print(f"[{check.status}] {check.name}")
        for note in check.notes:
            print(f"       {note}")
    failed = [c for c in checks if c.status == FAIL]
    warned = [c for c in checks if c.status == WARN]
    print()
    if failed:
        print(f"{len(failed)} check(s) FAILED: " + ", ".join(c.name for c in failed))
        print("Do not start the run. Fixing these now costs minutes; finding them at "
              "fold 30 costs the whole night.")
        return 1
    print("all checks passed" + (f" ({len(warned)} warning(s))" if warned else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
