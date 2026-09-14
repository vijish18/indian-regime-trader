"""Non-HMM benchmark: classifies risk by trailing realized volatility alone.

This exists to make a specific claim checkable rather than assumed:
docs/SPECIFICATION.md section 10.1 requires a "rolling-volatility threshold
strategy without HMM" among the walk-forward benchmarks, and section 10.3
requires the HMM strategy to beat it, after costs, on multiple out-of-sample
windows, before the HMM is allowed to matter to a live decision. Nothing else
in this codebase can produce that comparison -- Phase 5's HMM engine measures
regimes, but "is it *better than the simple thing*" needs the simple thing to
actually exist as code, not as an assumption.

``RollingVolatilityBaseline.evaluate`` returns the exact same
``AllocationTarget`` shape as
``core.regime.allocation.RegimeAllocationEngine.evaluate``, on purpose: a
walk-forward comparison (Phase 8/9) should be able to run the same downstream
portfolio/risk/cost pipeline against either one's output, with the *only*
difference being which of these two produced the exposure target. If the HMM
version cannot be shown to win under that setup, docs/SPECIFICATION.md's own
verdict is that it should not be trusted with capital.

Deliberately simple, and deliberately not extended to imitate the HMM's
confirmation/flicker/confidence machinery: it is supposed to be the "dumb"
alternative. Its only real free parameter is the trailing window
(``config.allocation.baseline_volatility_window``), which is shared with
the HMM's own ``config.features.realized_vol_window`` by default so the two
are measuring volatility the same way and a difference in outcome reflects
the *regime classification*, not a different volatility estimator.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from config.models import AllocationConfig
from core.regime.allocation import AllocationTarget, volatility_tier
from core.regime.regime_policy import RegimePolicy

_TRADING_DAYS_PER_YEAR = 252


class RollingVolatilityBaseline:
    """Classifies the current risk tier from trailing realized volatility of
    raw market returns. No HMM, no feature engineering, no model fitting, no
    confirmation or flicker handling.
    """

    def __init__(self, config: AllocationConfig, policy: RegimePolicy) -> None:
        self.config = config
        self.policy = policy

    def realized_volatility(self, returns: pd.Series) -> float:
        """Annualized population std of the trailing
        ``baseline_volatility_window`` daily returns.

        Same annualization convention (population std, ``sqrt(252)``) as
        ``core.features.feature_engineering``'s realized-volatility feature,
        so the HMM and the baseline are measuring volatility on equal terms
        and any difference in outcome is attributable to the classification
        logic, not the volatility estimator.
        """
        window = returns.tail(self.config.baseline_volatility_window)
        if len(window) < self.config.baseline_volatility_window:
            raise ValueError(
                f"need at least {self.config.baseline_volatility_window} trailing "
                f"returns, got {len(window)}"
            )
        if window.isna().any():
            raise ValueError("trailing return window contains NaN")
        return float(window.std(ddof=0) * math.sqrt(_TRADING_DAYS_PER_YEAR))

    def evaluate(self, returns: pd.Series, as_of: dt.date | None = None) -> AllocationTarget:
        """The allocation implied by trailing realized volatility alone.

        ``returns`` must be ascending and end at the decision date;
        ``as_of`` defaults to that last date. The target is the midpoint of
        the tier's configured band: with no model and no confidence score to
        scale by, a fixed point within the band is the only honest choice --
        picking anything closer to one edge would imply a certainty this
        method has no basis for.

        ``confidence`` is reported as 1.0 and ``allow_new_positions`` as
        always True: this method has no concept of "I don't know" the way
        the HMM's UNCERTAIN tier does, which is itself a real, documented
        limitation of the baseline, not an oversight.
        """
        if returns.empty:
            raise ValueError("cannot evaluate the baseline on an empty return series")

        volatility = self.realized_volatility(returns)
        regime = volatility_tier(volatility, self.config)
        band = self.policy.band_for(regime)
        target = (band.min_gross_exposure + band.max_gross_exposure) / 2.0
        decision_date = as_of if as_of is not None else _as_date(returns.index[-1])

        return AllocationTarget(
            as_of=decision_date,
            regime=regime,
            target_gross_exposure=target,
            min_gross_exposure=band.min_gross_exposure,
            max_gross_exposure=band.max_gross_exposure,
            allow_new_positions=True,
            confidence=1.0,
            expected_volatility=volatility,
            reason="rolling_realized_volatility",
        )


def _as_date(value: pd.Timestamp | dt.datetime | dt.date | str) -> dt.date:
    return pd.Timestamp(value).date()
