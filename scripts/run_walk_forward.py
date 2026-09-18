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
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from backtest.cost_schedule import CostScheduleRepository  # noqa: E402
from backtest.costs import CostModel  # noqa: E402
from backtest.performance import PerformanceReport  # noqa: E402
from backtest.walk_forward import STRATEGY_NAMES, WalkForwardValidator  # noqa: E402
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
from universe.bhavcopy_universe import DERIVED_INDEX_SYMBOL  # noqa: E402
from universe.stock_selector import StockSelector  # noqa: E402
from universe.universe import UniverseProvider  # noqa: E402

DATA_CACHE = REPO_ROOT / "data_cache"
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
    _require(DATA_CACHE / "raw" / "index" / f"{NIFTY_SYMBOL}.csv",
             "python scripts/build_index_observations.py")

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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=dt.date.fromisoformat,
                        default=dt.date(2016, 1, 1))
    parser.add_argument("--to", dest="end", type=dt.date.fromisoformat,
                        default=dt.date(2024, 12, 31))
    parser.add_argument("--state-dir", type=Path, default=REPO_ROOT / "state" / "walk_forward")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
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
    args = parser.parse_args(argv[1:])

    args.state_dir.mkdir(parents=True, exist_ok=True)
    validator = build_validator(
        snapshot_date=args.end,
        circuit_breaker_dir=args.state_dir,
        frame_cache=args.frame_cache,
    )

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

    reports = validator.run_all_strategies(
        args.start, args.end, progress=progress, strategies=args.strategy
    )

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
        args.json_out.write_text(
            json.dumps(
                {
                    "start": args.start.isoformat(),
                    "end": args.end.isoformat(),
                    "reports": {name: asdict(report) for name, report in reports.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\njson -> {args.json_out}")

    # A subset run is one process of a split; the combined markdown is
    # merge_walk_forward.py's job, and writing a partial table to the
    # report path would leave a four-row comparison looking like the answer.
    if args.strategy is None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render(reports, args.start, args.end), encoding="utf-8")
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
