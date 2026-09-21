"""Walk-forward backtest on real data: the composition root.

    python scripts/run_walk_forward.py --from 2016-01-01 --to 2024-12-31

Everything this assembles already existed. ``WalkForwardValidator`` was
built in Phase 9 and has been exercised against synthetic data ever
since; what it never had was real prices, a real universe and real
corporate actions. This script is the wiring, and deliberately contains
no strategy logic of its own -- if a number here looks wrong, it is wrong
in a module that has its own tests, not in this file.

**What it answers.** ``run_all_strategies`` runs the HMM-gated strategy
alongside buy-and-hold, a rolling-volatility filter, a moving-average
trend filter, and a shuffled-regime control. docs/SPECIFICATION.md
section 10.3 requires the HMM to beat the simple baseline *after costs*;
the shuffled control exists to catch the case where it beats the baseline
only because it is sometimes out of the market, which a coin flip would
also achieve.

So there are three possible honest outcomes, and two of them are "no":

    HMM > baseline and HMM > shuffled     the regime layer adds something
    HMM > baseline but ~= shuffled        the layer is just reducing exposure
    HMM <= baseline                       it does not work

**Prerequisites**, in order:

    python scripts/backfill_bhavcopy.py --from 2015-01-01
    python scripts/build_equity_bars.py
    python scripts/kite_login.py
    python scripts/build_index_observations.py --from 2015-01-01

Training uses only sessions inside each fold's training window, and the
feature pipeline is causal, so a fold's test window is genuinely out of
sample. That is a property of the modules being wired, not of this
script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from backtest.cost_schedule import CostScheduleRepository  # noqa: E402
from backtest.costs import CostModel  # noqa: E402
from backtest.engine import BacktestResult  # noqa: E402
from backtest.performance import PerformanceReport  # noqa: E402
from backtest.walk_forward import (  # noqa: E402
    STRATEGY_NAMES,
    CompletedFolds,
    WalkForwardValidator,
)
from config.loader import load_settings  # noqa: E402
from core.features.feature_engineering import (  # noqa: E402
    FeaturePipeline,
    build_default_feature_definitions,
)
from core.regime.regime_policy import RegimePolicy  # noqa: E402
from data.calendar import NSETradingCalendar  # noqa: E402
from data.corporate_actions import InMemoryCorporateActionProvider  # noqa: E402
from data.instrument_master import InMemoryInstrumentRepository  # noqa: E402
from data.market_data import LocalMarketDataProvider  # noqa: E402
from data.membership import InMemoryIndexMembershipProvider  # noqa: E402
from data.storage import LocalDataStore, StorageFormat  # noqa: E402
from portfolio.portfolio_constructor import PortfolioConstructor  # noqa: E402
from scripts.build_index_observations import NIFTY_SYMBOL, VIX_SYMBOL  # noqa: E402
from storage.atomic import atomic_write  # noqa: E402
from universe.bhavcopy_universe import DERIVED_INDEX_SYMBOL  # noqa: E402
from universe.stock_selector import StockSelector  # noqa: E402
from universe.universe import UniverseProvider  # noqa: E402

DATA_CACHE = Path(os.environ.get("IRT_DATA_ROOT", REPO_ROOT / "data_cache"))
REFERENCE = DATA_CACHE / "reference"
HOLIDAY_FILE = REPO_ROOT / "config" / "nse_holidays.csv"
COST_SCHEDULE = REPO_ROOT / "config" / "cost_schedules.yaml"
REPORT_PATH = REPO_ROOT / "docs" / "walk_forward_report.md"


def _require(path: Path, how: str) -> Path:
    if not path.exists():
        raise SystemExit(f"missing {path}\n  produce it with: {how}")
    return path


def build_validator(
    snapshot_date: dt.date, circuit_breaker_dir: Path, frame_cache: int = 1000
) -> WalkForwardValidator:
    settings = load_settings()

    _require(REFERENCE / "corporate_actions.csv", "python scripts/backfill_bhavcopy.py")
    _require(REFERENCE / "index_membership.csv", "python scripts/backfill_bhavcopy.py")
    _require(REFERENCE / "instruments.csv", "python scripts/build_equity_bars.py")
    _require(
        DATA_CACHE / "raw" / "index" / f"{NIFTY_SYMBOL}.csv",
        "python scripts/build_index_observations.py",
    )

    calendar = NSETradingCalendar.from_file(HOLIDAY_FILE)

    # CSV, not the default Parquet: the backfill writes CSV so the files
    # stay greppable, which matters more for data being brought into a
    # system for the first time than the read speed does.
    store = LocalDataStore(
        raw_root=DATA_CACHE / "raw",
        normalized_root=DATA_CACHE / "normalized",
        reference_root=REFERENCE,
        storage_format=StorageFormat.CSV,
    )

    corporate_actions = InMemoryCorporateActionProvider.from_file(
        REFERENCE / "corporate_actions.csv"
    )
    # A backtest reads an immutable snapshot, so memoising parsed files is
    # free correctness-wise and removes most of the cost -- a fold asks for
    # the same instrument's file on every session, for every strategy. It
    # is purely a speed/memory trade: the cached object is a parsed copy of
    # a file that cannot change during a run, so no value of --frame-cache
    # can change the result, only how long it takes to get it.
    #
    # Tunable because the full 32-fold run is unattended and long. Measured
    # on the 2015-2026 store (2,205 instruments, 2,886 sessions): the
    # largest single frame is 0.79 MB in memory, so 1,000 frames is 0.79 GB
    # and the entire set would be 1.74 GB. A frame holds an instrument's
    # whole history regardless of the fold window, so this does not grow
    # with the period -- only with how much of the universe is touched.
    # Lower it if the machine starts swapping: a slower run beats a run
    # that dies at hour twelve.
    #
    # Live trading must never set this: files change daily there, and a
    # cache would serve yesterday's bars as today's with nothing to show
    # anything was stale.
    market_data = LocalMarketDataProvider(
        store, corporate_actions=corporate_actions, frame_cache_size=frame_cache
    )
    instruments = InMemoryInstrumentRepository.from_file(
        REFERENCE / "instruments.csv", snapshot_date=snapshot_date
    )
    membership = InMemoryIndexMembershipProvider.from_file(REFERENCE / "index_membership.csv")

    universe_provider = UniverseProvider(
        DERIVED_INDEX_SYMBOL, membership, instruments, corporate_actions
    )
    stock_selector = StockSelector(
        settings.selection, settings.universe, universe_provider, market_data
    )
    portfolio_constructor = PortfolioConstructor(
        settings.portfolio, settings.selection, settings.execution, market_data
    )
    cost_model = CostModel(
        CostScheduleRepository.from_file(COST_SCHEDULE),
        min_slippage_bps=settings.backtest.slippage_min_bps,
        impact_coefficient=settings.backtest.slippage_impact_coefficient,
    )

    return WalkForwardValidator(
        config=settings.backtest,
        calendar=calendar,
        market_data=market_data,
        stock_selector=stock_selector,
        portfolio_constructor=portfolio_constructor,
        risk_config=settings.risk,
        cost_model=cost_model,
        circuit_breaker_state_dir=circuit_breaker_dir,
        hmm_config=settings.hmm,
        allocation_config=settings.allocation,
        regime_policy=RegimePolicy(settings.regime_policy),
        feature_pipeline=FeaturePipeline(build_default_feature_definitions(settings.features)),
        index_symbol=NIFTY_SYMBOL,
        vix_symbol=VIX_SYMBOL,
        corporate_actions=corporate_actions,
    )


def render(reports: dict[str, PerformanceReport], start: dt.date, end: dt.date) -> str:
    lines = [
        "# Walk-forward backtest",
        "",
        f"- Generated: {dt.datetime.now(dt.UTC).isoformat()}",
        f"- Period: {start} to {end}",
        "- Data: NSE bhavcopy (point-in-time universe), Kite index history,",
        "  NSE corporate actions applied at read time.",
        "",
        "| Strategy | CAGR | Vol | Sharpe | Max DD | Invested | Trades | Costs (Rs) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, report in reports.items():
        lines.append(
            f"| {name} | {report.cagr:.2%} | {report.volatility:.2%} | "
            f"{report.sharpe:.2f} | {report.max_drawdown:.2%} | "
            f"{report.pct_invested:.1%} | {report.trade_count} | "
            f"{report.total_costs:,.0f} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "docs/SPECIFICATION.md section 10.3 requires the HMM strategy to beat the",
        "simple baseline **after costs**. The shuffled-regime control is the second",
        "test: it keeps the same exposure *distribution* but destroys the timing, so",
        "if the HMM does not beat it, the regime layer is only reducing average",
        "exposure and a coin flip would do as well.",
        "",
        "All figures are net of the Indian statutory cost model",
        "(`backtest/costs.py`), not gross.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Checkpoint, fingerprint, resume
# --------------------------------------------------------------------------

TRADE_LOG_COLUMNS = (
    "signal_date",
    "execution_date",
    "instrument_id",
    "side",
    "quantity",
    "fill_price",
    "gross_value",
    "cost",
    "net_value",
)
"""Matches ``BacktestEngine.run``'s empty trade log, so a resumed run with
no recorded trades concatenates against the same columns the engine would
have produced rather than an empty frame with none."""

MANIFEST_NAME = "run.manifest.json"
"""Written into ``--series-dir`` after every completed fold. Names what the
run is, which folds are done, and a fingerprint of the configuration that
produced them."""


def _fingerprint(
    args: argparse.Namespace,
    folds: Sequence[tuple[dt.date, ...]],
    initial_equity: float,
) -> str:
    """A hash of everything that would change the numbers.

    Resuming across a configuration change would splice two different models'
    output into one equity curve and report it as one run. The result would
    look entirely plausible: a continuous curve, sensible drawdowns, and no
    indication that folds 1-20 used a 3% take-profit and folds 21-33 a 5%
    trailing stop. So the fingerprint covers the fold boundaries, the money,
    and every config block that reaches the engine -- and a resume against a
    different one is refused rather than warned about.

    ``git_commit`` is recorded but deliberately NOT hashed: a commit that
    only touches a docstring or a test must not invalidate ten hours of
    completed folds. What matters is whether the inputs changed, and the
    config blocks below are the inputs.
    """
    settings = load_settings()
    payload = {
        "schema": 2,
        "inputs": _input_digest(),
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "folds": [[d.isoformat() for d in fold] for fold in folds],
        "initial_equity": initial_equity,
        "selection": settings.selection.model_dump(mode="json"),
        "universe": settings.universe.model_dump(mode="json"),
        "portfolio": settings.portfolio.model_dump(mode="json"),
        "risk": settings.risk.model_dump(mode="json"),
        "backtest": settings.backtest.model_dump(mode="json"),
        "hmm": settings.hmm.model_dump(mode="json"),
        "allocation": settings.allocation.model_dump(mode="json"),
        "regime_policy": settings.regime_policy.model_dump(mode="json"),
        "features": settings.features.model_dump(mode="json"),
        "execution": settings.execution.model_dump(mode="json"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _input_digest() -> str:
    """Content identity, not mtimes: resume cannot mix code or data revisions."""
    digest = hashlib.sha256()
    paths = list(DATA_CACHE.rglob("*.csv"))
    paths += [HOLIDAY_FILE, COST_SCHEDULE]
    for folder in ("backtest", "core", "data", "universe", "portfolio", "risk", "config"):
        paths += list((REPO_ROOT / folder).rglob("*.py"))
    paths += [Path(__file__)]
    for path in sorted(set(paths)):
        relative = (
            path.relative_to(DATA_CACHE)
            if path.is_relative_to(DATA_CACHE)
            else path.relative_to(REPO_ROOT)
        )
        digest.update(str(relative).replace("\\", "/").encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _save_fold_csv(path: Path, frame: pd.DataFrame, fold: int) -> None:
    """Replace a fold as one atomic operation, including its empty trade log."""
    if path.exists():
        previous = pd.read_csv(path)
        previous = previous[previous["fold"] != fold]
        if not previous.empty:
            frame = pd.concat([previous, frame], ignore_index=True)
    atomic_write(path, frame.to_csv(index=False))


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write_manifest(
    args: argparse.Namespace,
    folds_total: int,
    completed: dict[str, int],
    fingerprint: str,
    initial_equity: float,
) -> None:
    if args.series_dir is None:
        return
    args.series_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        args.series_dir / MANIFEST_NAME,
        json.dumps(
            {
                "fingerprint": fingerprint,
                "start": args.start.isoformat(),
                "end": args.end.isoformat(),
                "folds_total": folds_total,
                "completed_folds": completed,
                "initial_equity": initial_equity,
                "git_commit": _git_commit(),
                "updated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
    )


def _load_resume(
    series_dir: Path, selected: tuple[str, ...], fingerprint: str, folds_total: int
) -> dict[str, CompletedFolds]:
    """Read back what an earlier run of this exact configuration finished.

    Raises rather than returning a partial state. Every failure here means
    "the checkpoint on disk is not what this run is computing", and the only
    safe answers are to start over or to point --series-dir somewhere else;
    quietly continuing would produce a curve spliced from two different runs.
    """
    manifest_path = series_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(
            f"--resume: no {MANIFEST_NAME} in {series_dir}. Nothing to resume from; "
            "drop --resume to start the run."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--resume: {manifest_path} is unreadable: {exc}") from exc

    if manifest.get("fingerprint") != fingerprint:
        raise SystemExit(
            "--resume: the configuration has changed since those folds were computed.\n"
            f"  checkpoint: {manifest.get('fingerprint', '?')[:16]}  "
            f"({manifest.get('git_commit', '?')[:8]}, {manifest.get('updated_at', '?')})\n"
            f"  this run:   {fingerprint[:16]}\n"
            "Resuming would splice folds computed under two different configurations "
            "into one equity curve and report it as one run. Start over, or point "
            "--series-dir at a new directory."
        )
    if manifest.get("folds_total") != folds_total:
        raise SystemExit(
            f"--resume: the checkpoint covers {manifest.get('folds_total')} folds, "
            f"this run generates {folds_total}."
        )

    completed = manifest.get("completed_folds") or {}
    resume: dict[str, CompletedFolds] = {}
    for name in selected:
        folds_done = int(completed.get(name, 0))
        if folds_done <= 0:
            continue
        equity = _read_checkpoint_series(series_dir / f"{name}.folds.csv", "equity", folds_done)
        cash = _read_checkpoint_series(series_dir / f"{name}.cash.partial.csv", "cash", folds_done)
        trades = _read_checkpoint_trades(series_dir / f"{name}.trades.partial.csv", folds_done)
        resume[name] = CompletedFolds(
            folds_done=folds_done, equity=equity, trades=trades, cash=cash
        )
    return resume


def _read_checkpoint_trades(path: Path, folds_done: int | None = None) -> pd.DataFrame:
    """The checkpointed trade log, with its dates back as dates.

    ``pd.read_csv`` hands back strings, and the engine produces
    ``datetime.date``. Concatenating the two and asking
    ``PerformanceCalculator`` for an average holding period then fails on
    ``str - str`` -- which is exactly how the first resumed run died, after
    correctly recomputing every fold it was missing.
    """
    if not path.exists():
        return pd.DataFrame(columns=list(TRADE_LOG_COLUMNS))
    frame = pd.read_csv(path)
    if "fold" in frame.columns and folds_done is not None:
        frame = _keep_last_run_per_fold(frame, folds_done).drop(columns=["fold"])
    for column in ("signal_date", "execution_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column]).dt.date
    return frame


def _keep_last_run_per_fold(frame: pd.DataFrame, folds_done: int) -> pd.DataFrame:
    """Rows for folds 1..folds_done, keeping only the LAST copy of each fold.

    Two distinct kinds of stale row live in a checkpoint after a crash:

    *Beyond the manifest* -- a fold whose rows were appended before the
    manifest was rewritten, so the resume does not count it as done. Dropped
    by the fold number.

    *Duplicated* -- that same fold, recomputed by the resumed run and
    appended a second time, leaving 126 rows for a 63-session fold. Dropped
    by the natural key, keeping the later copy.

    The key is ``(fold, session_date)`` where there is one value per session,
    and the whole row for the trade log, where a fold can hold many rows per
    session. A recomputed fold is byte-identical -- the backtest is
    deterministic -- and the engine emits at most one order per instrument
    per signal date, so two identical trade rows inside one fold cannot
    occur legitimately.

    The obvious approach, marking a new block wherever the fold number
    changes, does not work: the two copies of a fold are *contiguous*, so the
    fold number never changes between them and they read as one block.
    """
    if "fold" not in frame.columns or frame.empty:
        return frame
    frame = frame[frame["fold"] <= folds_done]
    if frame.empty:
        return frame
    if "session_date" in frame.columns:
        return frame.drop_duplicates(subset=["fold", "session_date"], keep="last")
    return frame.drop_duplicates(keep="last")


def _truncate_checkpoints(series_dir: Path, name: str, folds_done: int) -> None:
    """Rewrite this strategy's checkpoints to exactly folds 1..folds_done.

    Run before a resumed pass appends anything. Reading is already guarded by
    fold number, but the file itself accumulates duplicates otherwise -- and
    the file is what a person opens.
    """
    for filename in (
        f"{name}.folds.csv",
        f"{name}.cash.partial.csv",
        f"{name}.trades.partial.csv",
    ):
        path = series_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "fold" not in frame.columns:
            # Written by a version that did not tag trades with a fold. It
            # cannot be truncated safely, so it is not trusted: a resumed run
            # rebuilds the trade log from the folds it recomputes plus
            # whatever this file holds, and a silent partial is worse than a
            # loud restart.
            raise SystemExit(
                f"--resume: {path} has no 'fold' column, so the rows belonging to "
                "completed folds cannot be identified. It was written by an older "
                "build. Start the run over, or point --series-dir at a new directory."
            )
        kept = _keep_last_run_per_fold(frame, folds_done)
        if len(kept) != len(frame):
            print(
                f"  checkpoint: {filename} {len(frame)} -> {len(kept)} rows "
                f"(stale or duplicated folds removed)"
            )
            atomic_write(path, kept.to_csv(index=False))


def _read_checkpoint_series(path: Path, column: str, folds_done: int) -> pd.Series:
    """One checkpoint file as a session-indexed series, truncated to the
    folds the manifest says are complete.

    The truncation matters. A crash mid-fold can leave rows for a fold the
    manifest never recorded, because the CSV is appended before the manifest
    is rewritten. Trusting the file over the manifest would count a fold that
    was interrupted partway through as finished.
    """
    if not path.exists():
        raise SystemExit(
            f"--resume: {path} is missing but the manifest says {folds_done} folds "
            "are complete. The checkpoint is incomplete; start over."
        )
    frame = _keep_last_run_per_fold(pd.read_csv(path, parse_dates=["session_date"]), folds_done)
    if frame.empty:
        raise SystemExit(f"--resume: {path} has no rows for folds 1..{folds_done}")
    series = pd.Series(
        frame[column].to_numpy(dtype=float),
        index=pd.Index([d.date() for d in frame["session_date"]]),
    )
    return series.sort_index()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from", dest="start", type=dt.date.fromisoformat, default=dt.date(2016, 1, 1)
    )
    parser.add_argument(
        "--to", dest="end", type=dt.date.fromisoformat, default=dt.date(2024, 12, 31)
    )
    parser.add_argument("--state-dir", type=Path, default=REPO_ROOT / "state" / "walk_forward")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--initial-equity", type=float, default=10_000_000.0)
    parser.add_argument(
        "--frame-cache",
        type=int,
        default=1000,
        help="parsed bar frames to memoise; speed/memory only, never the result",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        default=None,
        help=(
            "run only this strategy (repeatable). The five are independent -- "
            "each chains its own equity and owns its circuit-breaker state -- so "
            "splitting them across processes gives identical numbers in a "
            f"fifth of the wall clock. One of: {', '.join(STRATEGY_NAMES)}"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="write the reports as JSON here, for merge_walk_forward.py to combine",
    )
    parser.add_argument(
        "--series-dir",
        type=Path,
        default=None,
        help="write each strategy's equity curve and trade log as CSV here",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "continue from the folds already checkpointed in --series-dir instead of "
            "starting over. Refused if the configuration has changed since those folds "
            "were computed."
        ),
    )
    args = parser.parse_args(argv[1:])

    if args.series_dir is None:
        args.series_dir = args.state_dir / "series"
    if not args.resume and (args.series_dir / MANIFEST_NAME).exists():
        raise SystemExit("Run directory already exists; use --resume or a fresh --series-dir")

    if args.resume and args.series_dir is None:
        raise SystemExit("--resume needs --series-dir: that is where the checkpoint lives")

    args.state_dir.mkdir(parents=True, exist_ok=True)
    validator = build_validator(
        snapshot_date=args.end,
        circuit_breaker_dir=args.state_dir,
        frame_cache=args.frame_cache,
    )

    if not math.isfinite(args.initial_equity) or args.initial_equity <= 0:
        raise SystemExit("--initial-equity must be finite and positive")
    validator.initial_equity = args.initial_equity
    folds = validator.generate_folds(args.start, args.end)
    print(f"{len(folds)} walk-forward folds from {args.start} to {args.end}")
    if not folds:
        raise SystemExit(
            "no folds generated: the period is shorter than "
            "backtest.training_window_sessions + backtest.test_window_sessions"
        )
    for train_start, train_end, test_start, test_end in folds[:3]:
        print(f"  train {train_start}..{train_end}   test {test_start}..{test_end}")
    if len(folds) > 3:
        print(f"  ... and {len(folds) - 3} more")

    folds_total = len(folds)
    selected_names = tuple(args.strategy) if args.strategy else tuple(STRATEGY_NAMES)
    fingerprint = _fingerprint(args, folds, validator.initial_equity)
    validator.engine.checkpoint_dir = args.series_dir / "sessions"
    validator.engine.run_identity = fingerprint
    validator.engine.liquidate_at_end = True
    resume_state: dict[str, CompletedFolds] = {}
    if args.resume:
        resume_state = _load_resume(args.series_dir, selected_names, fingerprint, folds_total)
        for name, state in resume_state.items():
            _truncate_checkpoints(args.series_dir, name, state.folds_done)
        if resume_state:
            for name, state in sorted(resume_state.items()):
                print(
                    f"  resuming {name}: {state.folds_done}/{folds_total} folds already "
                    f"done, equity {state.running_equity:,.0f}"
                )
        else:
            print("  resume: the checkpoint has no completed folds; starting from fold 1")
    resumed_counts = {name: state.folds_done for name, state in resume_state.items()}
    _write_manifest(args, folds_total, dict(resumed_counts), fingerprint, validator.initial_equity)

    print("\nrunning all strategies (this refits the HMM per fold)...")
    started = dt.datetime.now(dt.UTC)
    completed = 0

    def progress(line: str) -> None:
        """Per-fold heartbeat with an ETA extrapolated from folds so far.

        The full run takes most of a day. Without this, a hang is
        indistinguishable from slow progress until the whole thing is over.
        """
        nonlocal completed
        completed += 1
        elapsed = (dt.datetime.now(dt.UTC) - started).total_seconds()
        remaining = elapsed / completed * (len(folds) - completed)
        eta = (dt.datetime.now(dt.UTC) + dt.timedelta(seconds=remaining)).astimezone()
        print(
            f"[{elapsed / 60:6.1f}m] {line}"
            + (f"  eta {eta:%H:%M %Z}" if completed < len(folds) else "  done")
        )

    def on_series(name: str, curve: pd.Series[float], trades: pd.DataFrame) -> None:
        """Persist the raw series a dashboard needs, next to the JSON.

        Summary statistics cannot be un-summarised: without these, drawing
        an equity curve later means re-running the whole backtest.
        """
        if args.series_dir is None:
            return
        args.series_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(
            args.series_dir / f"{name}.equity.csv",
            curve.rename("equity").to_csv(index_label="session_date"),
        )
        atomic_write(args.series_dir / f"{name}.trades.csv", trades.to_csv(index=False))
        print(f"  series -> {args.series_dir / f'{name}.equity.csv'} ({len(curve):,} rows)")

    completed_folds: dict[str, int] = dict(resumed_counts)

    def on_fold(name: str, fold_index: int, result: BacktestResult) -> None:
        """Keep the two things only the fold result knows.

        A veto is a position the strategy wanted and risk refused -- the
        only honest basis for asking what refusing it cost. And the
        holdings after the final session are what the strategy would be
        holding now, which no summary statistic can reconstruct.
        """
        if args.series_dir is None:
            return

        # Checkpoint this fold before moving on. Everything else here is
        # written only after all 33 folds finish, which meant a crash on the
        # last fold discarded the whole run: four strategies lost ten hours
        # each to a corporate action in fold 33, having computed folds 1-32
        # correctly and kept them nowhere.
        #
        # Appending per fold makes a failure cost one fold instead of all of
        # them. The completed run still overwrites these with the clean
        # concatenated series, so this is a safety net, not the product.
        checkpoint = args.series_dir / f"{name}.folds.csv"
        frame = result.equity_curve.rename("equity").to_frame()
        frame.insert(0, "fold", fold_index + 1)
        _save_fold_csv(checkpoint, frame.rename_axis("session_date").reset_index(), fold_index + 1)

        trades_checkpoint = args.series_dir / f"{name}.trades.partial.csv"
        # Tagged with the fold, like the other two checkpoints. Without it
        # a resumed run cannot tell which trades belong to a fold it is
        # about to recompute, and the only alternative -- truncating by
        # date -- cannot distinguish a re-run fold from a duplicated one.
        tagged = result.trade_log.copy()
        tagged.insert(0, "fold", fold_index + 1)
        _save_fold_csv(trades_checkpoint, tagged, fold_index + 1)

        # Cash too, and for the same reason the completed run keeps it: a
        # resumed run that cannot read back the cash balance of the folds it
        # skipped reports pct_invested as nan for the whole period. Honest,
        # and useless -- it is the one figure showing how much of the run was
        # actually spent in the market.
        cash_checkpoint = args.series_dir / f"{name}.cash.partial.csv"
        cash_frame = result.cash_history.rename("cash").to_frame()
        cash_frame.insert(0, "fold", fold_index + 1)
        _save_fold_csv(
            cash_checkpoint, cash_frame.rename_axis("session_date").reset_index(), fold_index + 1
        )

        vetoes: list[dict[str, object]] = []
        holdings: list[dict[str, object]] = []
        for day in result.risk_decisions:
            for decision in day.decisions:
                if decision.approved:
                    continue
                vetoes.append(
                    {
                        "strategy": name,
                        "fold": fold_index + 1,
                        "as_of": day.as_of.isoformat(),
                        "instrument_id": decision.instrument_id,
                        "wanted_weight": decision.target_weight,
                        "circuit_state": str(decision.circuit_state),
                        "violations": "|".join(str(v.check) for v in decision.violations),
                    }
                )
        if result.positions_history:
            final_day = max(result.positions_history)
            for instrument_id, quantity in result.positions_history[final_day].items():
                holdings.append(
                    {
                        "strategy": name,
                        "fold": fold_index + 1,
                        "as_of": final_day.isoformat(),
                        "instrument_id": instrument_id,
                        "quantity": quantity,
                    }
                )

        audit_dir = args.series_dir / "audit"
        for label, rows in (("vetoes", vetoes), ("holdings", holdings)):
            atomic_write(audit_dir / f"{name}.{fold_index + 1}.{label}.json", json.dumps(rows))
        completed_folds[name] = fold_index + 1
        _write_manifest(args, folds_total, completed_folds, fingerprint, validator.initial_equity)

    reports = validator.run_all_strategies(
        args.start,
        args.end,
        progress=progress,
        strategies=args.strategy,
        on_series=on_series,
        on_fold=on_fold,
        resume=resume_state or None,
    )

    if args.series_dir is not None:
        args.series_dir.mkdir(parents=True, exist_ok=True)
        tag = args.strategy[0] if args.strategy else "all"
        for label in ("vetoes", "holdings"):
            rows = []
            for name in selected_names:
                for fold_number in range(1, completed_folds[name] + 1):
                    audit = args.series_dir / "audit" / f"{name}.{fold_number}.{label}.json"
                    rows.extend(json.loads(audit.read_text(encoding="utf-8")))
            atomic_write(
                args.series_dir / f"{tag}.{label}.csv", pd.DataFrame(rows).to_csv(index=False)
            )
            print(f"  {label} -> {len(rows):,} rows")

    print()
    header = (
        f"{'strategy':<28} {'CAGR':>8} {'vol':>8} {'Sharpe':>8} "
        f"{'max DD':>9} {'inv':>7} {'trades':>7}"
    )
    print(header)
    for name, report in reports.items():
        print(
            f"{name:<28} {report.cagr:>7.2%} {report.volatility:>7.2%} "
            f"{report.sharpe:>8.2f} {report.max_drawdown:>8.2%} "
            f"{report.pct_invested:>6.1%} {report.trade_count:>7}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            args.json_out,
            json.dumps(
                {
                    "start": args.start.isoformat(),
                    "end": args.end.isoformat(),
                    "reports": {name: asdict(report) for name, report in reports.items()},
                },
                indent=2,
            ),
        )
        print(f"\njson -> {args.json_out}")

    # A subset run is one process of a split; the combined markdown is
    # merge_walk_forward.py's job, and writing a partial table to the
    # report path would leave a four-row comparison looking like the answer.
    if args.strategy is None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(args.report, render(reports, args.start, args.end))
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
