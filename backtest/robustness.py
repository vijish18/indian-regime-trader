"""Robustness diagnostics: does a strategy's result survive being run under
systematically varied conditions, or does it collapse the moment one
configuration choice changes?

This module runs no backtests itself -- it deliberately knows nothing
about ``WalkForwardValidator``, ``BacktestEngine``, config objects, or
market data. Each variant is supplied as a zero-argument closure the
caller builds (typically "construct a validator with this one setting
changed, run it, return the aggregate performance") because assembling
that closure requires wiring together config, data, and every other layer
this module has no business depending on; ``RobustnessSuite`` only knows
how to run a batch of them and measure how much the results disagree with
each other.

Six of the seven dimensions docs/SPECIFICATION.md-adjacent diligence asks
for are direct config changes the caller already has the knobs for
(training window, transaction costs, slippage, universe size, market
period, and the rebalance-threshold control ``backtest/engine.py``'s
``min_rebalance_weight_delta`` adds); "parameter perturbation" is a
catch-all for small, deliberate jitters to any other numeric config (HMM
seeds/candidate state counts, allocation thresholds) that a caller wires
up the same way.

Nothing here declares a result "robust" or "fragile" as a final verdict --
:meth:`RobustnessReport.is_stable` is an explicit, named threshold check a
caller opts into, not an automatic judgment, matching
``backtest/comparison.py``'s refusal to emit a bare success/failure flag.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from backtest.performance import PerformanceReport

_KEY_METRICS = ("cagr", "total_return", "sharpe", "sortino", "calmar", "max_drawdown")


class RobustnessDimension(StrEnum):
    PARAMETER_PERTURBATION = "parameter_perturbation"
    TRAINING_WINDOW = "training_window"
    REBALANCE_THRESHOLD = "rebalance_threshold"
    TRANSACTION_COST = "transaction_cost"
    SLIPPAGE = "slippage"
    UNIVERSE_SIZE = "universe_size"
    MARKET_PERIOD = "market_period"


class RobustnessError(RuntimeError):
    """A robustness run could not proceed -- e.g. no variants supplied."""


@dataclass(frozen=True, slots=True)
class RobustnessVariantResult:
    dimension: RobustnessDimension
    variant_label: str
    performance: PerformanceReport


@dataclass(frozen=True, slots=True)
class MetricDispersion:
    """How much one metric varied across every variant in a robustness
    run."""

    metric: str
    values: tuple[float, ...]
    minimum: float
    maximum: float
    mean: float
    stdev: float

    @property
    def relative_range(self) -> float:
        """``(max - min) / |mean|`` -- ``0.0`` for a perfectly stable
        metric, growing as the spread across variants grows relative to
        the metric's own typical size. ``math.inf`` when the mean is ~0
        but the values still differ (a relative comparison is meaningless
        there); ``0.0`` when the mean is ~0 and the values do not differ
        either.
        """
        if abs(self.mean) < 1e-12:
            return math.inf if (self.maximum - self.minimum) > 1e-12 else 0.0
        return (self.maximum - self.minimum) / abs(self.mean)


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    dimension: RobustnessDimension
    variants: tuple[RobustnessVariantResult, ...]
    dispersion: dict[str, MetricDispersion]

    def is_stable(self, metric: str, max_relative_range: float = 0.5) -> bool:
        """Whether ``metric``'s spread across every variant stays within
        ``max_relative_range`` of its own mean.

        An explicit, named threshold check a caller opts into -- not an
        automatic verdict. A "stable" metric by this check can still
        describe a bad strategy; an "unstable" one is not automatically
        disqualifying, just something that needs explaining before anyone
        trusts the comparison it came from.
        """
        if metric not in self.dispersion:
            raise KeyError(f"no dispersion computed for metric {metric!r}")
        return self.dispersion[metric].relative_range <= max_relative_range


class RobustnessSuite:
    """Runs a batch of already-built variant closures and measures how
    much their key performance metrics disagree with each other.
    """

    def run(
        self,
        dimension: RobustnessDimension,
        variants: dict[str, Callable[[], PerformanceReport]],
    ) -> RobustnessReport:
        if not variants:
            raise RobustnessError(
                f"at least one variant is required to run a {dimension.value} sweep"
            )

        results = tuple(
            RobustnessVariantResult(dimension, label, factory())
            for label, factory in variants.items()
        )
        dispersion = {metric: _dispersion(metric, results) for metric in _KEY_METRICS}
        return RobustnessReport(dimension=dimension, variants=results, dispersion=dispersion)


def _dispersion(metric: str, results: tuple[RobustnessVariantResult, ...]) -> MetricDispersion:
    values = tuple(float(getattr(result.performance, metric)) for result in results)
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return MetricDispersion(metric, values, math.nan, math.nan, math.nan, math.nan)
    mean = statistics.fmean(finite_values)
    stdev = statistics.pstdev(finite_values) if len(finite_values) > 1 else 0.0
    return MetricDispersion(
        metric=metric,
        values=values,
        minimum=min(finite_values),
        maximum=max(finite_values),
        mean=mean,
        stdev=stdev,
    )
