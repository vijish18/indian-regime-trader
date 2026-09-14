"""V1 stock selector: a transparent, systematic factor-ranking model.

This is the layer that decides *which* stocks receive the risk budget the
regime layer (``core/regime/allocation.py``) has already set. It has no
access to the current regime, the exposure target, or any risk state -- see
``universe/__init__.py`` for why that boundary matters (it is what lets a
walk-forward run measure the regime layer's contribution independently of
selection).

For V1 this is deliberately *not* a machine-learning model. It is six
transparent, individually-interpretable factors
(``universe.factor_calculator.FactorSet``) combined by configured,
non-negative weights into one composite score. Every number that goes into a
ranking decision can be read off a ``StockScore`` and traced back to the raw
price history that produced it -- there is no opaque intermediate
representation to audit.

## The six-step selection algorithm

``select(as_of)`` runs, in order:

1. **Obtain the correct historical universe** --
   ``universe.universe.UniverseProvider.get_universe(as_of)``, the
   point-in-time eligible constituent list (survivorship-bias-free by
   construction; see that module).
2. **Remove securities with insufficient data** -- fewer than
   ``selection.min_history_days`` bars of adjusted price history ending at
   ``as_of``, or missing from local market data entirely.
3. **Remove securities failing liquidity rules** -- average daily traded
   value over ``selection.liquidity_window_days`` below
   ``universe.min_avg_daily_value_inr``.
4. **Calculate factors** -- ``universe.factor_calculator.FactorCalculator``,
   from adjusted bars covering only ``[as_of - lookback, as_of]``.
5. **Rank securities** -- each factor is cross-sectionally standardized
   *within this date's surviving candidate set* (never over time, and never
   including instruments already excluded in steps 2-3), then combined by
   ``selection.factor_weights`` into one composite score.
6. **Return a configurable number of candidates** -- the top
   ``selection.max_holdings`` by composite score.

Steps 1-3 are captured as one auditable ``CandidateUniverse``; steps 4-6
produce a ranked list of ``StockScore``. This class places no orders and
returns no target weights -- that is ``portfolio/portfolio_constructor.py``'s
job, one layer up.

## Point-in-time discipline

Every price/volume fetch in this module requests
``[as_of - lookback, as_of]`` with ``price_basis=ADJUSTED`` -- adjusted *as
of* ``as_of``, per ``data.interfaces.MarketDataProvider``'s own contract, so
a split announced after the decision date cannot retroactively change a
factor computed for it. No fundamentals are used (none are available -- see
``universe/factor_calculator.py``), and future index membership cannot leak
in because the universe itself is point-in-time
(``universe.universe.UniverseProvider``).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from config.models import SelectionConfig, UniverseConfig
from data.errors import DataNotAvailableError
from data.interfaces import MarketDataProvider
from data.models import DailyBar, PriceBasis
from universe.factor_calculator import (
    FactorCalculator,
    FactorSet,
    InsufficientFactorHistoryError,
)
from universe.factor_calculator import (
    liquidity as _liquidity,
)
from universe.universe import ConstituentSnapshot, UniverseProvider, UniverseSnapshot


class SelectionExclusionReason(StrEnum):
    """Why an otherwise-eligible constituent did not become a candidate."""

    INSUFFICIENT_HISTORY = "insufficient_history"
    ILLIQUID = "illiquid"
    DATA_UNAVAILABLE = "data_unavailable"


@dataclass(frozen=True, slots=True)
class SelectionExclusion:
    """One eligible constituent excluded before ranking, and why -- kept
    alongside the surviving candidates so an excluded name is an auditable
    decision, matching ``universe.universe.UniverseExclusion``'s pattern one
    layer up.
    """

    instrument_id: str
    reason: SelectionExclusionReason
    detail: str


@dataclass(frozen=True, slots=True)
class CandidateUniverse:
    """The point-in-time eligible universe (steps 1), further filtered for
    sufficient price history and liquidity (steps 2-3) -- the pool
    ``StockSelector`` actually ranks. Every excluded eligible constituent is
    recorded with a reason, never silently dropped.
    """

    as_of: dt.date
    universe: UniverseSnapshot
    candidate_ids: tuple[str, ...]
    excluded: tuple[SelectionExclusion, ...] = ()

    def __contains__(self, instrument_id: str) -> bool:
        return instrument_id in self.candidate_ids

    def __len__(self) -> int:
        return len(self.candidate_ids)


@dataclass(frozen=True, slots=True)
class StockScore:
    """One candidate's composite ranking score, its raw and
    cross-sectionally-standardized factor breakdown, and its rank -- enough
    to audit exactly why it ranked where it did relative to its peers on
    this date.
    """

    instrument_id: str
    symbol: str
    as_of: dt.date
    rank: int
    """1 = best. Assigned only among candidates that were actually scored;
    excluded constituents have no rank."""

    score: float
    raw_factors: FactorSet
    standardized_factors: FactorSet
    """Same shape as ``raw_factors``, but each value is a cross-sectional
    z-score within this date's candidate set -- comparable across factors,
    unlike the raw values. ``standardized_factors.volatility`` is the
    z-score of *negated* volatility (so higher is still "better" here),
    matching ``universe.factor_calculator``'s documented sign convention.
    """


def _cross_sectional_zscore(values: dict[str, float]) -> dict[str, float]:
    """Standardize ``values`` against each other -- a snapshot across
    instruments on one date, never a rolling window over time (that is
    ``core.features.feature_engineering.rolling_standardize``'s job, and a
    different problem). A single candidate or a zero-variance set maps every
    value to 0.0: with nothing to compare against, no candidate can be said
    to rank above another on that factor.
    """
    if not values:
        return {}
    sample = list(values.values())
    mean = sum(sample) / len(sample)
    variance = sum((value - mean) ** 2 for value in sample) / len(sample)
    std = math.sqrt(variance)
    if std <= 1e-12:
        return dict.fromkeys(values, 0.0)
    return {key: (value - mean) / std for key, value in values.items()}


class StockSelector:
    """Builds the point-in-time candidate universe and ranks it by a
    transparent, configurable composite factor score.
    """

    def __init__(
        self,
        config: SelectionConfig,
        universe_config: UniverseConfig,
        universe_provider: UniverseProvider,
        market_data: MarketDataProvider,
        factor_calculator: FactorCalculator | None = None,
    ) -> None:
        self.config = config
        self.universe_config = universe_config
        self.universe_provider = universe_provider
        self.market_data = market_data
        self.factor_calculator = factor_calculator or FactorCalculator(config)

    def build_candidate_universe(self, as_of: dt.date) -> CandidateUniverse:
        """Steps 1-3: point-in-time eligible universe, minus insufficient-
        data and illiquid names.
        """
        universe = self.universe_provider.get_universe(as_of)
        candidate_ids: list[str] = []
        excluded: list[SelectionExclusion] = []

        for constituent in universe.constituents:
            try:
                bars = self._fetch_bars(constituent.instrument_id, as_of)
            except DataNotAvailableError as exc:
                excluded.append(
                    SelectionExclusion(
                        constituent.instrument_id,
                        SelectionExclusionReason.DATA_UNAVAILABLE,
                        str(exc),
                    )
                )
                continue

            if len(bars) < self.config.min_history_days:
                excluded.append(
                    SelectionExclusion(
                        constituent.instrument_id,
                        SelectionExclusionReason.INSUFFICIENT_HISTORY,
                        f"{len(bars)} bars available, "
                        f"{self.config.min_history_days} required",
                    )
                )
                continue

            closes = pd.Series(
                [float(bar.close) for bar in bars],
                index=[bar.session_date for bar in bars],
            )
            volumes = pd.Series(
                [float(bar.volume) for bar in bars],
                index=[bar.session_date for bar in bars],
            )
            average_traded_value = _liquidity(
                closes, volumes, self.config.liquidity_window_days
            )
            if average_traded_value < self.universe_config.min_avg_daily_value_inr:
                excluded.append(
                    SelectionExclusion(
                        constituent.instrument_id,
                        SelectionExclusionReason.ILLIQUID,
                        f"average traded value {average_traded_value:,.0f} INR over "
                        f"{self.config.liquidity_window_days} sessions is below the "
                        f"{self.universe_config.min_avg_daily_value_inr:,.0f} INR minimum",
                    )
                )
                continue

            candidate_ids.append(constituent.instrument_id)

        candidate_ids.sort()
        excluded.sort(key=lambda item: item.instrument_id)
        return CandidateUniverse(
            as_of=as_of,
            universe=universe,
            candidate_ids=tuple(candidate_ids),
            excluded=tuple(excluded),
        )

    def score_candidates(
        self, candidates: CandidateUniverse, as_of: dt.date
    ) -> list[StockScore]:
        """Steps 4-5: compute factors for every candidate, cross-sectionally
        standardize each factor across exactly this candidate set, combine
        by ``selection.factor_weights``, and rank descending by composite
        score (ties broken by ``instrument_id`` so the ordering is always
        deterministic, never dependent on dict/set iteration order).
        """
        if not candidates.candidate_ids:
            return []

        index_closes = self._fetch_index_closes(as_of)
        by_id: dict[str, ConstituentSnapshot] = {
            constituent.instrument_id: constituent
            for constituent in candidates.universe.constituents
        }
        raw_factors: dict[str, FactorSet] = {}
        for instrument_id in candidates.candidate_ids:
            bars = self._fetch_bars(instrument_id, as_of)
            closes = pd.Series(
                [float(bar.close) for bar in bars], index=[bar.session_date for bar in bars]
            )
            volumes = pd.Series(
                [float(bar.volume) for bar in bars], index=[bar.session_date for bar in bars]
            )
            try:
                raw_factors[instrument_id] = self.factor_calculator.compute(
                    closes, volumes, index_closes
                )
            except InsufficientFactorHistoryError as exc:  # pragma: no cover - defensive
                raise InsufficientFactorHistoryError(
                    f"{instrument_id} passed the min_history_days filter but a factor "
                    f"still needed more data than it had: {exc}"
                ) from exc

        standardized = self._standardize(raw_factors)
        weights = self.config.factor_weights
        scored: list[tuple[str, float]] = [
            (
                instrument_id,
                (
                    weights.momentum * standardized[instrument_id].momentum
                    + weights.trend_persistence * standardized[instrument_id].trend_persistence
                    + weights.relative_strength * standardized[instrument_id].relative_strength
                    + weights.volatility * standardized[instrument_id].volatility
                    + weights.drawdown * standardized[instrument_id].drawdown
                    + weights.liquidity * standardized[instrument_id].liquidity_inr
                ),
            )
            for instrument_id in candidates.candidate_ids
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))

        return [
            StockScore(
                instrument_id=instrument_id,
                symbol=by_id[instrument_id].symbol,
                as_of=as_of,
                rank=rank,
                score=score,
                raw_factors=raw_factors[instrument_id],
                standardized_factors=standardized[instrument_id],
            )
            for rank, (instrument_id, score) in enumerate(scored, start=1)
        ]

    def select(self, as_of: dt.date) -> list[StockScore]:
        """The full pipeline: build the candidate universe, score and rank
        it, and return the top ``selection.max_holdings`` (step 6). Places
        no orders and returns no target weights.
        """
        candidates = self.build_candidate_universe(as_of)
        ranked = self.score_candidates(candidates, as_of)
        return ranked[: self.config.max_holdings]

    def _standardize(self, raw_factors: dict[str, FactorSet]) -> dict[str, FactorSet]:
        momentum_z = _cross_sectional_zscore(
            {key: value.momentum for key, value in raw_factors.items()}
        )
        trend_z = _cross_sectional_zscore(
            {key: value.trend_persistence for key, value in raw_factors.items()}
        )
        relative_strength_z = _cross_sectional_zscore(
            {key: value.relative_strength for key, value in raw_factors.items()}
        )
        # Negated before standardizing: lower realized volatility should
        # produce a higher standardized value, matching every other factor's
        # "higher is better" convention (see this module's docstring).
        stability_z = _cross_sectional_zscore(
            {key: -value.volatility for key, value in raw_factors.items()}
        )
        drawdown_z = _cross_sectional_zscore(
            {key: value.drawdown for key, value in raw_factors.items()}
        )
        liquidity_z = _cross_sectional_zscore(
            {key: value.liquidity_inr for key, value in raw_factors.items()}
        )
        return {
            instrument_id: FactorSet(
                momentum=momentum_z[instrument_id],
                trend_persistence=trend_z[instrument_id],
                relative_strength=relative_strength_z[instrument_id],
                volatility=stability_z[instrument_id],
                drawdown=drawdown_z[instrument_id],
                liquidity_inr=liquidity_z[instrument_id],
            )
            for instrument_id in raw_factors
        }

    def _fetch_bars(self, instrument_id: str, as_of: dt.date) -> list[DailyBar]:
        start = as_of - dt.timedelta(days=self.config.min_history_days * 2)
        return self.market_data.get_equity_bars(
            instrument_id, start, as_of, price_basis=PriceBasis.ADJUSTED
        )

    def _fetch_index_closes(self, as_of: dt.date) -> pd.Series:
        longest_window = max(
            self.config.relative_strength_window_days,
            self.config.min_history_days,
        )
        start = as_of - dt.timedelta(days=longest_window * 2)
        observations = self.market_data.get_index_observations(
            self.universe_provider.index_symbol, start, as_of
        )
        return pd.Series(
            [float(observation.close) for observation in observations],
            index=[observation.session_date for observation in observations],
        )
