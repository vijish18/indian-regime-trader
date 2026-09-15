"""Strategy comparison: the HMM against each non-HMM baseline, with
explicit, programmatically-generated caveats attached to every comparison.

This module makes one structural commitment: **nothing here ever emits a
verdict**. There is no ``success: bool`` field anywhere in this file, on
purpose. A `Sharpe ratio, on its own, has repeatedly been enough to
convince people a strategy "works" when it is really an artifact of a
short sample, one lucky regime, or unaccounted-for costs -- so this module
never lets a single number stand in for that judgment. What it produces
instead is a per-metric delta table plus a set of caveats a reader has to
actually read past before concluding anything. The robustness diagnostics
in ``backtest/robustness.py`` are the other half of this discipline: a
comparison that looks good here and falls apart under parameter
perturbation was never a real edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from backtest.performance import PerformanceReport

_COMPARISON_METRICS = (
    "cagr",
    "total_return",
    "volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "net_pnl",
    "net_return",
    "gross_return",
    "total_costs",
    "cost_pct_of_turnover",
    "turnover",
    "trade_count",
)
"""Every metric compared between the HMM and a baseline. Not every one of
these is "higher is better" -- ``max_drawdown``, ``total_costs``, and
``cost_pct_of_turnover`` are lower-is-better; ``MetricDelta`` reports the
raw signed difference (``hmm - baseline``) for all of them uniformly and
leaves the direction of "better" to the reader, rather than silently
flipping signs, which would make the same field mean different things for
different metrics."""

_LOWER_IS_BETTER = frozenset({"max_drawdown", "total_costs", "cost_pct_of_turnover"})


@dataclass(frozen=True, slots=True)
class MetricDelta:
    metric: str
    hmm_value: float
    baseline_value: float
    delta: float
    """``hmm_value - baseline_value``, always -- see this module's
    docstring on ``_COMPARISON_METRICS`` for why this is never sign-flipped
    per metric."""

    lower_is_better: bool


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    baseline_name: str
    hmm: PerformanceReport
    baseline: PerformanceReport
    deltas: tuple[MetricDelta, ...]
    caveats: tuple[str, ...]
    """Programmatically generated warnings about over-interpreting this
    comparison. Always non-empty -- see :func:`_generate_caveats`."""

    def delta_for(self, metric: str) -> MetricDelta:
        for delta in self.deltas:
            if delta.metric == metric:
                return delta
        raise KeyError(f"no delta computed for metric {metric!r}")


def compare_to_baseline(
    hmm: PerformanceReport, baseline: PerformanceReport, baseline_name: str
) -> BaselineComparison:
    deltas = tuple(
        MetricDelta(
            metric=metric,
            hmm_value=float(getattr(hmm, metric)),
            baseline_value=float(getattr(baseline, metric)),
            delta=float(getattr(hmm, metric)) - float(getattr(baseline, metric)),
            lower_is_better=metric in _LOWER_IS_BETTER,
        )
        for metric in _COMPARISON_METRICS
    )
    caveats = _generate_caveats(hmm, baseline_name, deltas)
    return BaselineComparison(
        baseline_name=baseline_name, hmm=hmm, baseline=baseline, deltas=deltas, caveats=caveats
    )


def compare_all(
    reports: dict[str, PerformanceReport], hmm_key: str = "hmm"
) -> dict[str, BaselineComparison]:
    """One :class:`BaselineComparison` per non-HMM entry in ``reports`` --
    the four required comparisons (simple volatility classifier,
    buy-and-hold, trend-only baseline, randomized control) when ``reports``
    is ``WalkForwardValidator.run_all_strategies``'s own output.
    """
    if hmm_key not in reports:
        raise ValueError(f"reports has no {hmm_key!r} entry to compare the others against")
    hmm = reports[hmm_key]
    return {
        name: compare_to_baseline(hmm, report, name)
        for name, report in reports.items()
        if name != hmm_key
    }


def _generate_caveats(
    hmm: PerformanceReport, baseline_name: str, deltas: tuple[MetricDelta, ...]
) -> tuple[str, ...]:
    caveats: list[str] = [
        "A per-metric comparison alone does not establish that an outperformance is "
        "durable -- see the accompanying robustness diagnostics (parameter "
        "perturbation, alternate training windows, rebalance thresholds, transaction "
        "costs, slippage, universe sizes, and market periods) before treating any "
        "result here as a genuine edge rather than a configuration artifact."
    ]

    if hmm.trade_count < 30:
        caveats.append(
            f"HMM trade_count is only {hmm.trade_count}; a Sharpe/Sortino/Calmar ratio "
            "computed from this few trades is not statistically reliable on its own."
        )

    if hmm.sharpe > 1.0 and hmm.max_drawdown > 0.20:
        caveats.append(
            f"Sharpe ({hmm.sharpe:.2f}) is high alongside a {hmm.max_drawdown:.1%} "
            "maximum drawdown -- a single ratio hides the magnitude of loss an investor "
            "would actually have experienced along the way."
        )

    return_delta = _find(deltas, "total_return")
    drawdown_delta = _find(deltas, "max_drawdown")
    if return_delta.delta > 0 and drawdown_delta.delta > 0:
        caveats.append(
            f"HMM's {return_delta.delta:+.2%} return advantage over {baseline_name} comes "
            f"with a {drawdown_delta.delta:+.2%} *larger* maximum drawdown, not for free -- "
            "the extra return may simply be compensation for extra risk taken, not skill."
        )

    cost_delta = _find(deltas, "total_costs")
    if cost_delta.hmm_value > 0 and return_delta.hmm_value != 0:
        cost_share = abs(cost_delta.hmm_value / (hmm.gross_pnl if hmm.gross_pnl != 0 else 1.0))
        if cost_share > 0.30:
            caveats.append(
                f"transaction costs consume {cost_share:.0%} of HMM's gross P&L -- a "
                "meaningful share of the strategy's own turnover-driven cost, worth "
                "checking against the transaction-cost and slippage robustness variants."
            )

    if not math.isfinite(hmm.profit_factor):
        caveats.append(
            "HMM's profit_factor is infinite (no losing days in this window) -- likely "
            "reflects a short or unusually one-sided sample, not a robust edge."
        )

    return tuple(caveats)


def _find(deltas: tuple[MetricDelta, ...], metric: str) -> MetricDelta:
    for delta in deltas:
        if delta.metric == metric:
            return delta
    raise KeyError(f"no delta for metric {metric!r}")
