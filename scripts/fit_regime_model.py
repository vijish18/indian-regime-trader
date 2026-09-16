"""Fit the HMM regime model on real NIFTY 50 / India VIX history.

    python scripts/kite_login.py            # once each morning
    python scripts/fit_regime_model.py --train-end 2024-12-31
    python scripts/fit_regime_model.py --train-end 2024-12-31 --approve

**Why this can be done honestly today, when a stock-level backtest cannot.**
The regime engine's features come from NIFTY 50 and India VIX alone --
never from individual equities. So neither of the two defects documented
in docs/KITE_DATA.md applies here:

* **Survivorship bias** is a property of a *universe* of stocks. An index
  has no universe to survive; NIFTY 50's published level already accounts
  for its own constituent changes.
* **Corporate-action adjustment** is a property of a single company's
  share price. An index level is not adjusted for splits because it is
  not a share price.

That is not a loophole, it is why the architecture separates the regime
layer from selection: the thing that decides *how much* market exposure
to take can be fitted long before the thing that decides *which stocks*.

**What still cannot be done** is measuring whether this model makes
money. That needs stock-level returns, and those need the two missing
feeds. This script fits and characterises a model; it does not claim the
model is profitable, and ``--approve`` marks it usable, not validated.

Training uses only data at or before ``--train-end``, which is what makes
a later walk-forward evaluation meaningful rather than circular.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from decimal import Decimal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from broker.zerodha.kite_historical import (  # noqa: E402
    INDIA_VIX_TOKEN,
    NIFTY_50_TOKEN,
    Candle,
    KiteHistoricalClient,
)
from broker.zerodha.kite_session import load_session  # noqa: E402
from config.loader import load_settings  # noqa: E402
from core.features.feature_engineering import (  # noqa: E402
    FeaturePipeline,
    MarketFeatureInputs,
    build_default_feature_definitions,
    drop_warmup_rows,
)
from core.features.feature_scaler import CausalFeatureScaler  # noqa: E402
from core.regime.hmm_engine import HMMRegimeEngine  # noqa: E402
from core.regime.model_registry import ModelArtifact, ModelRegistry, build_model_id  # noqa: E402
from data.calendar import NSETradingCalendar  # noqa: E402
from data.models import IndexObservation  # noqa: E402

HOLIDAY_FILE = REPO_ROOT / "config" / "nse_holidays.csv"
DEFAULT_REGISTRY = REPO_ROOT / "model_registry"
FEATURE_VERSION = "kite-nifty-vix-v1"


def to_index_observations(symbol: str, candles: tuple[Candle, ...]) -> list[IndexObservation]:
    """Kite candles -> domain observations, with one critical translation.

    **Index volume is discarded, not passed through as zero.** Kite
    reports ``volume = 0`` for NIFTY 50 and INDIA VIX, because an index
    has no traded quantity of its own. Zero is not the same as absent,
    and the difference is load-bearing here: ``MarketFeatureInputs``
    decides whether to include ``nifty_volume_stress_20d`` by testing
    ``isna()``, so a column of zeros would count as real volume, and the
    feature -- a rolling z-score of ``log(volume)`` -- would be computed
    as ``log(0) = -inf`` for every row.

    That would not raise. It would produce a feature matrix full of
    infinities that silently poisons the fit. Passing ``None`` makes the
    pipeline correctly omit the feature and fit on the remaining eight,
    which its own docstring describes as an expected and valid case.
    """
    return [
        IndexObservation(
            index_symbol=symbol,
            session_date=candle.session_date,
            close=Decimal(str(candle.close)),
            open=Decimal(str(candle.open)),
            high=Decimal(str(candle.high)),
            low=Decimal(str(candle.low)),
            volume=None,
        )
        for candle in candles
    ]


def fetch_index_history(start: dt.date, end: dt.date) -> tuple[list[IndexObservation], ...]:
    session = load_session()
    client = KiteHistoricalClient(session.api_key, access_token=session.access_token)
    print(f"fetching NIFTY 50 and INDIA VIX, {start} -> {end}...")
    nifty = client.daily_candles(NIFTY_50_TOKEN, start, end)
    vix = client.daily_candles(INDIA_VIX_TOKEN, start, end)
    print(f"  NIFTY 50 : {len(nifty):,} bars")
    print(f"  INDIA VIX: {len(vix):,} bars")
    return (
        to_index_observations("NIFTY 50", nifty),
        to_index_observations("INDIA VIX", vix),
    )


def restrict_to_trading_days(
    observations: list[IndexObservation],
) -> tuple[list[IndexObservation], list[dt.date]]:
    """Keep only sessions the calendar calls trading days.

    Kite returns bars for two kinds of session this system never trades,
    and both would otherwise land in the training data:

    * **Muhurat** -- the ~1 hour ceremonial Diwali session.
    * **Saturday special sessions** -- Union Budget days (2015-02-28,
      2020-02-01) and NSE's special live / disaster-recovery sessions
      (2024-01-20, 2024-03-02, 2024-05-18).

    Dropping the Budget Saturdays is a real cost, stated rather than
    hidden: they are among the most economically significant sessions of
    their year. It is still the right call, because **the model should be
    fitted on exactly the sessions the system can act on.** This strategy
    trades daily, on weekdays, in the regular session. Training on a
    session it will never participate in models a market it does not
    trade, and splicing a Saturday between Friday and Monday distorts
    every rolling window that spans it.

    Returns the kept observations and the dropped dates, so the caller
    reports what was excluded instead of it vanishing silently.
    """
    calendar = NSETradingCalendar.from_file(HOLIDAY_FILE)
    covered = calendar.covered_years

    uncovered = sorted({o.session_date.year for o in observations} - covered)
    if uncovered:
        raise SystemExit(
            f"bars fall in year(s) the calendar does not cover: {uncovered}. "
            "Run: python scripts/build_nse_holidays.py --reconcile-with-kite"
        )

    kept = [o for o in observations if calendar.is_trading_day(o.session_date)]
    dropped = [
        o.session_date for o in observations if not calendar.is_trading_day(o.session_date)
    ]
    return kept, dropped


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=dt.date.fromisoformat,
                        default=dt.date(2015, 1, 1))
    parser.add_argument("--train-end", type=dt.date.fromisoformat, required=True,
                        help="last session included in training (nothing after it is seen)")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--approve", action="store_true",
                        help="mark the fitted model approved for use")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args(argv[1:])

    settings = load_settings()

    nifty, vix = fetch_index_history(args.start, args.train_end)
    if not nifty or not vix:
        raise SystemExit("no index history returned")

    nifty, dropped = restrict_to_trading_days(nifty)
    vix, _ = restrict_to_trading_days(vix)
    if dropped:
        print(f"  dropped {len(dropped)} non-trading session(s) the system cannot act on:")
        for day in dropped:
            print(f"    {day} ({day:%a})")

    # --- features -------------------------------------------------------
    inputs = MarketFeatureInputs.from_index_observations(nifty, vix)
    pipeline = FeaturePipeline(build_default_feature_definitions(settings.features))
    matrix = drop_warmup_rows(pipeline.compute(inputs))

    print(f"\nfeature matrix: {matrix.shape[0]:,} sessions x {matrix.shape[1]} features")
    print(f"  {matrix.index[0].date()} -> {matrix.index[-1].date()}")
    for column in matrix.columns:
        print(f"    {column}")
    if "nifty_volume_stress_20d" not in matrix.columns:
        print("  (volume-stress omitted: indices carry no traded volume -- expected)")

    # --- scale, then fit -------------------------------------------------
    scaled, scaler_params = CausalFeatureScaler().fit_transform(matrix)
    returns = inputs.frame["nifty_close"].pct_change().reindex(matrix.index).fillna(0.0)

    print("\nfitting (every candidate state count x seed, selected by BIC)...")
    engine = HMMRegimeEngine(settings.hmm)
    fitted = engine.fit(scaled, returns)
    training = fitted.training_result

    print(f"\nselected: {training.n_states} states, seed {training.seed}")
    print(f"  BIC {training.bic:,.1f}   log-likelihood {training.log_likelihood:,.1f}")
    print("\nstate characteristics (measured from real returns, not assumed):")
    print(
        f"  {'state':>5} {'ann.vol':>8} {'ann.ret':>8} "
        f"{'downside':>9} {'occupancy':>9} {'days':>6}  label"
    )
    for stats in sorted(fitted.statistics, key=lambda s: s.expected_volatility):
        print(
            f"  {stats.state_id:>5} {stats.expected_volatility:>7.1%} "
            f"{stats.expected_return:>7.1%} {stats.downside_volatility:>8.1%} "
            f"{stats.occupancy:>8.1%} {stats.expected_duration:>6.1f}  {stats.label}"
        )
    print("  (labels are reporting-only and never a decision input)")

    # --- persist ---------------------------------------------------------
    model_id = build_model_id(args.train_end, training.n_states, training.seed, FEATURE_VERSION)
    artifact = ModelArtifact(
        model_id=model_id,
        created_at=dt.datetime.now(dt.UTC),
        model=fitted,
        scaler=scaler_params,
        feature_version=FEATURE_VERSION,
        notes=args.notes
        or (
            f"Fitted on real NIFTY 50 / India VIX from Kite, {args.start} to "
            f"{args.train_end}. Index-only inputs, so unaffected by the "
            f"survivorship and corporate-action limits in docs/KITE_DATA.md. "
            f"Not evaluated for profitability -- that needs stock-level data."
        ),
    )
    registry = ModelRegistry(args.registry)
    path = registry.save(artifact)
    print(f"\nsaved {model_id}")
    print(f"  {path}")

    if args.approve:
        registry.approve(model_id)
        print(f"\napproved {model_id}")
        print("  NOTE: approval marks this model usable, not validated. It has not")
        print("  been shown to make money -- that needs a stock-level backtest.")
    else:
        print(f"\nnot approved. To approve: --approve, or registry.approve({model_id!r})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
