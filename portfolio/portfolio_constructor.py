"""Combines the regime's risk budget, the ranked candidates, and every
position-level limit into one target portfolio.

    MARKET REGIME + RISK BUDGET   (core.regime.allocation.AllocationTarget)
        +
    STOCK RANKINGS                (universe.stock_selector.StockScore, ranked)
        +
    POSITION LIMITS               (config.models.PortfolioConfig, plus
                                    execution.max_participation_adv_pct for
                                    liquidity and selection.max_holdings for
                                    the position count)
        =
    TARGET PORTFOLIO              (TargetPortfolio)

"Market regime" and "risk budget" are the same input here, not two: the
regime *is* what determines the risk budget in this system
(``core/regime/allocation.py``'s docstring states the arrow explicitly --
observations -> regime -> risk budget), so there is exactly one
``exposure_target: AllocationTarget`` parameter, not a separate one for each.

This module produces weights only. It does not compute order quantities --
that is ``risk/position_sizer.py``'s job, one layer down, and it never
touches a broker. **No orders are sent from here.**

## The weighting waterfall

``construct()`` runs, in order, each step a pure reduction of the previous
one (never an increase), which is what makes the whole pipeline converge in
one pass with no iteration:

1. **Candidate selection.** Top ``selection.max_holdings`` ranked candidates,
   restricted to names already held when ``exposure_target.allow_new_positions``
   is False (UNCERTAIN regime) -- no new names are ever opened in that state,
   matching the semantics ``AllocationTarget.allow_new_positions`` already
   documents.
2. **Raw weights**: ``(score - floor) / volatility`` per candidate --
   risk-adjusted, shifted to be strictly positive first (composite scores are
   cross-sectional z-score sums and can be negative or zero, which a raw
   proportional weight cannot use directly).
3. **Correlation penalty**: for any pair of selected candidates whose
   trailing return correlation exceeds ``portfolio.max_pairwise_correlation``,
   the lower-ranked member's raw weight is scaled down by
   ``portfolio.correlation_penalty_pct`` -- a penalty, not an exclusion; nothing
   in this module has veto authority (that is ``risk/risk_manager.py``'s job).
4. **Normalize** to sum to 1.0 -- relative proportions within the selected set.
5. **Scale to the risk budget**: multiply every weight by
   ``exposure_target.target_gross_exposure``, so the pre-cap sum equals the
   regime's specific point target, not just something inside its band.
6. **Position limits**, each a pure clip/scale-down, applied in this fixed
   order: single-name cap, liquidity cap (position value vs.
   ``execution.max_participation_adv_pct`` of the candidate's own average
   daily traded value), sector cap (aggregate, scaled down proportionally
   within the sector). None of these ever redistributes freed weight to
   other names -- see "Why capped weight is never redistributed" below.
7. **Minimum weight floor**: a position below ``portfolio.min_position_weight_pct``
   is dropped; its freed weight becomes cash.

Cash is always ``1.0 - sum(final position weights)``, which is the entire
reason step 6 never over-allocates: nothing here can produce leverage or a
negative cash balance, and ``TargetPortfolio.__post_init__`` re-validates
both as defense in depth.

## Why capped weight is never redistributed

A clip-and-redistribute scheme (give a capped name's excess to the
next-best-ranked uncapped name) sounds like it uses the risk budget more
fully, but it can cascade: redistributing into another name can push *that*
name over its own cap, requiring another round, which can push a third name
over a sector cap, and so on, with no guarantee of a clean stopping point.
Every step here is a pure reduction instead: capped or excluded weight simply
becomes cash. The result is deterministic, always converges in one pass, and
fails toward *less* risk rather than more when a limit binds -- the same bias
every risk control in this system is built with.

## Target vs. current vs. required trades

``construct()`` returns a ``TargetPortfolio`` -- what the portfolio *should*
hold. It is not told what is currently held except through the optional
``current_portfolio`` parameter, used only to (a) restrict candidates when
new positions are disallowed, and (b) let a caller compute the diff. That diff
is ``required_trades()``: given a target and the current
``TargetPortfolio`` (structurally identical -- "the portfolio right now" is
just another point-in-time set of weights), it returns one ``RequiredTrade``
per instrument that appears in either, classified BUY/SELL/EXIT/HOLD. No
order is created from this either -- a ``RequiredTrade`` is a weight delta,
not a share count or a broker instruction.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import numpy as np
import pandas as pd

from config.models import ExecutionConfig, PortfolioConfig, SelectionConfig
from core.regime.allocation import AllocationRegime, AllocationTarget
from data.errors import DataNotAvailableError
from data.interfaces import MarketDataProvider
from data.models import PriceBasis
from universe.stock_selector import StockScore

_EPSILON = 1e-9
_TRADE_TOLERANCE = 1e-9


class PortfolioConstructionError(RuntimeError):
    """A constructed portfolio failed to satisfy its own configured limits.

    Raised rather than returned: a portfolio that silently violates the
    config it was built under is worse than no portfolio, the same fail-closed
    principle ``core/regime/hmm_engine.py``'s ``ModelSelectionError`` and
    ``core/regime/model_registry.py``'s approval gate already apply elsewhere.
    """


@dataclass(frozen=True, slots=True)
class TargetPosition:
    """One instrument's target weight, with enough provenance to audit why it
    ended up at that weight.
    """

    instrument_id: str
    symbol: str
    target_weight: float
    """Fraction of total portfolio equity (including cash), in (0, 1]."""

    sector: str
    rank: int
    """The candidate's rank from ``StockScore``, ascending (1 = best)."""

    score: float
    """The candidate's composite score from ``StockScore``, for audit."""

    binding_constraint: str
    """Which step last reduced this weight below what the risk budget alone
    would have given it: ``"unconstrained"``, ``"correlation"``,
    ``"max_single_name"``, ``"liquidity"``, or ``"max_sector"``."""

    def __post_init__(self) -> None:
        if not (0.0 < self.target_weight <= 1.0):
            raise ValueError(
                f"{self.instrument_id}: target_weight must be in (0, 1], "
                f"got {self.target_weight}"
            )


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    """A complete, self-consistent target allocation: every position weight
    plus cash, always summing to exactly 1.0 -- structurally identical
    whether it represents what the portfolio *should* hold (the output of
    :meth:`PortfolioConstructor.construct`) or what it *currently* holds
    (a caller-supplied snapshot passed back in as ``current_portfolio``).
    """

    as_of: dt.date
    positions: tuple[TargetPosition, ...]
    cash_weight: float
    regime: AllocationRegime
    gross_exposure: float
    """Sum of position weights; ``1.0 - cash_weight`` by construction, kept
    as an explicit field so a caller never has to recompute it."""

    def __post_init__(self) -> None:
        instrument_ids = [position.instrument_id for position in self.positions]
        if len(instrument_ids) != len(set(instrument_ids)):
            raise ValueError("TargetPortfolio has duplicate instrument_id entries")
        if self.cash_weight < -_EPSILON:
            raise ValueError(f"cash_weight must be >= 0, got {self.cash_weight}")
        total_positions = sum(position.target_weight for position in self.positions)
        if abs(total_positions - self.gross_exposure) > 1e-6:
            raise ValueError(
                f"gross_exposure ({self.gross_exposure}) must equal the sum of position "
                f"weights ({total_positions})"
            )
        if abs(total_positions + self.cash_weight - 1.0) > 1e-6:
            raise ValueError(
                f"positions ({total_positions}) + cash ({self.cash_weight}) must sum to "
                f"1.0, got {total_positions + self.cash_weight}"
            )

    @property
    def instrument_ids(self) -> frozenset[str]:
        return frozenset(position.instrument_id for position in self.positions)

    def weight_for(self, instrument_id: str) -> float:
        for position in self.positions:
            if position.instrument_id == instrument_id:
                return position.target_weight
        return 0.0

    def __contains__(self, instrument_id: str) -> bool:
        return instrument_id in self.instrument_ids

    def __len__(self) -> int:
        return len(self.positions)


class TradeAction(StrEnum):
    BUY = "buy"
    SELL = "sell"
    EXIT = "exit"
    """Target weight is zero; the instrument was held before and now should
    not be."""

    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class RequiredTrade:
    """The weight delta for one instrument between a current and a target
    portfolio -- not a share count, not a broker order.
    """

    instrument_id: str
    current_weight: float
    target_weight: float
    delta_weight: float
    action: TradeAction


def empty_portfolio(as_of: dt.date, regime: AllocationRegime) -> TargetPortfolio:
    """An all-cash target portfolio -- the correct output when there are no
    eligible candidates, or when UNCERTAIN disallows new positions and
    nothing eligible is currently held.
    """
    return TargetPortfolio(
        as_of=as_of, positions=(), cash_weight=1.0, regime=regime, gross_exposure=0.0
    )


class PortfolioConstructor:
    """Distributes a regime-determined gross-exposure budget across ranked
    candidates, subject to single-name, liquidity, sector, and correlation
    limits.
    """

    def __init__(
        self,
        config: PortfolioConfig,
        selection_config: SelectionConfig,
        execution_config: ExecutionConfig,
        market_data: MarketDataProvider,
    ) -> None:
        self.config = config
        self.selection_config = selection_config
        self.execution_config = execution_config
        self.market_data = market_data

    # -- construction -------------------------------------------------

    def construct(
        self,
        candidates: list[StockScore],
        exposure_target: AllocationTarget,
        as_of: dt.date,
        equity: float,
        sector_map: dict[str, str] | None = None,
        current_portfolio: TargetPortfolio | None = None,
    ) -> TargetPortfolio:
        """The full pipeline: select candidates, weight, penalize, scale,
        cap, floor. Never sends an order.

        Args:
            candidates: ranked, best first (``StockSelector.select``'s output).
            exposure_target: the regime's risk budget.
            as_of: the decision date; also used as the upper bound for every
                price fetch this method makes (correlation), so nothing here
                can see data past it.
            equity: total portfolio equity in INR, > 0. Used only to convert
                the liquidity cap from an INR amount into a weight fraction.
            sector_map: instrument_id -> sector label. An instrument absent
                from the map is treated as its own single-member sector
                (``instrument_id`` itself), so sector capping degrades
                gracefully rather than crashing when sector data is
                unavailable -- there is no sector data source in this
                codebase yet (the same documented gap as the missing
                fundamentals factor in ``universe/factor_calculator.py``).
            current_portfolio: what is held right now, if anything. Only
                consulted to restrict candidates when
                ``exposure_target.allow_new_positions`` is False.
        """
        if equity <= 0:
            raise ValueError(f"equity must be positive, got {equity}")

        sectors = sector_map or {}
        selected = self._select_candidates(candidates, exposure_target, current_portfolio)
        if not selected:
            return empty_portfolio(as_of, exposure_target.regime)

        raw = self.raw_weights(selected)
        penalized, binding = self.apply_correlation_penalty(selected, raw, as_of)
        normalized = self._normalize(penalized)
        scaled = {
            instrument_id: weight * exposure_target.target_gross_exposure
            for instrument_id, weight in normalized.items()
        }
        capped, binding = self._reconcile_caps(scaled, selected, sectors, equity, binding)
        floored = self._apply_min_weight_floor(capped)

        by_id = {candidate.instrument_id: candidate for candidate in selected}
        positions = tuple(
            TargetPosition(
                instrument_id=instrument_id,
                symbol=by_id[instrument_id].symbol,
                target_weight=weight,
                sector=sectors.get(instrument_id, instrument_id),
                rank=by_id[instrument_id].rank,
                score=by_id[instrument_id].score,
                binding_constraint=binding.get(instrument_id, "unconstrained"),
            )
            for instrument_id, weight in sorted(floored.items())
            if weight > 0.0
        )
        gross_exposure = sum(position.target_weight for position in positions)
        portfolio = TargetPortfolio(
            as_of=as_of,
            positions=positions,
            cash_weight=1.0 - gross_exposure,
            regime=exposure_target.regime,
            gross_exposure=gross_exposure,
        )
        self._validate_against_config(portfolio, exposure_target)
        return portfolio

    def _select_candidates(
        self,
        candidates: list[StockScore],
        exposure_target: AllocationTarget,
        current_portfolio: TargetPortfolio | None,
    ) -> list[StockScore]:
        if not exposure_target.allow_new_positions:
            held = current_portfolio.instrument_ids if current_portfolio else frozenset()
            return [candidate for candidate in candidates if candidate.instrument_id in held]
        return candidates[: self.selection_config.max_holdings]

    # -- weighting steps, each independently callable and testable ----

    def raw_weights(self, candidates: list[StockScore]) -> dict[str, float]:
        """Risk-adjusted raw weight per candidate: a positive-shifted
        composite score divided by realized volatility.

        Composite scores are cross-sectional z-score sums
        (``universe.stock_selector.StockScore.score``) and can be zero or
        negative, so they are shifted to be strictly positive before being
        used as a weight numerator -- the shift preserves relative ordering
        and relative spacing; it does not change which candidate is more or
        less attractive relative to another.
        """
        if not candidates:
            return {}
        if self.config.weighting == "equal":
            return {candidate.instrument_id: 1.0 for candidate in candidates}
        scores = [candidate.score for candidate in candidates]
        score_range = max(scores) - min(scores)
        shift = -min(scores) + max(1.0, 0.1 * score_range)
        return {
            candidate.instrument_id: (candidate.score + shift)
            / max(candidate.raw_factors.volatility, 1e-4)
            for candidate in candidates
        }

    def apply_correlation_penalty(
        self, candidates: list[StockScore], weights: dict[str, float], as_of: dt.date
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Halve (by ``portfolio.correlation_penalty_pct``) the lower-ranked
        member of any pair whose trailing return correlation exceeds
        ``portfolio.max_pairwise_correlation``. A penalty, never an
        exclusion -- this module has no veto authority.

        Gracefully skipped (returns ``weights`` unchanged) when fewer than
        two candidates are given or too little overlapping price history
        exists to trust a correlation estimate.
        """
        binding = dict.fromkeys(weights, "unconstrained")
        if len(candidates) < 2:
            return dict(weights), binding

        matrix = self._correlation_matrix(
            sorted(candidate.instrument_id for candidate in candidates), as_of
        )
        if matrix is None:
            return dict(weights), binding

        rank_by_id = {candidate.instrument_id: candidate.rank for candidate in candidates}
        penalized = dict(weights)
        ids = sorted(weights)
        for i, first in enumerate(ids):
            if first not in matrix.index:
                continue
            for second in ids[i + 1 :]:
                if second not in matrix.columns:
                    continue
                correlation = cast(float, matrix.loc[first, second])
                if pd.isna(correlation) or correlation <= self.config.max_pairwise_correlation:
                    continue
                worse = first if rank_by_id[first] > rank_by_id[second] else second
                penalized[worse] *= 1.0 - self.config.correlation_penalty_pct
                binding[worse] = "correlation"
        return penalized, binding

    def _reconcile_caps(
        self,
        weights: dict[str, float],
        candidates: list[StockScore],
        sector_map: dict[str, str],
        equity: float,
        binding: dict[str, str],
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Single-name cap, then liquidity cap, then sector cap -- each a
        pure reduction, so this fixed order always converges in one pass
        (see the module docstring, "Why capped weight is never
        redistributed").
        """
        by_id = {candidate.instrument_id: candidate for candidate in candidates}
        binding = dict(binding)

        stage = {}
        for instrument_id, weight in weights.items():
            capped = min(weight, self.config.max_single_name_pct)
            if capped < weight - _EPSILON:
                binding[instrument_id] = "max_single_name"
            stage[instrument_id] = capped

        next_stage = {}
        for instrument_id, weight in stage.items():
            liquidity_cap = self._liquidity_cap(by_id[instrument_id], equity)
            capped = min(weight, liquidity_cap)
            if capped < weight - _EPSILON:
                binding[instrument_id] = "liquidity"
            next_stage[instrument_id] = capped
        stage = next_stage

        sector_totals: dict[str, float] = {}
        for instrument_id, weight in stage.items():
            sector = sector_map.get(instrument_id, instrument_id)
            sector_totals[sector] = sector_totals.get(sector, 0.0) + weight

        final_stage = {}
        for instrument_id, weight in stage.items():
            sector = sector_map.get(instrument_id, instrument_id)
            total = sector_totals[sector]
            if total > self.config.max_sector_pct + _EPSILON:
                scaled = weight * (self.config.max_sector_pct / total)
                if scaled < weight - _EPSILON:
                    binding[instrument_id] = "max_sector"
                final_stage[instrument_id] = scaled
            else:
                final_stage[instrument_id] = weight

        return final_stage, binding

    def _apply_min_weight_floor(self, weights: dict[str, float]) -> dict[str, float]:
        """Drop any position below ``portfolio.min_position_weight_pct``; its
        freed weight becomes cash, not redistributed further -- the same
        reasoning as every other step in this pipeline.
        """
        return {
            instrument_id: weight
            for instrument_id, weight in weights.items()
            if weight >= self.config.min_position_weight_pct
        }

    def _liquidity_cap(self, candidate: StockScore, equity: float) -> float:
        """The maximum weight such that the resulting INR position size does
        not exceed ``execution.max_participation_adv_pct`` of the
        candidate's own average daily traded value
        (``StockScore.raw_factors.liquidity_inr``, already computed
        point-in-time by ``universe/factor_calculator.py``).
        """
        max_position_value = (
            self.execution_config.max_participation_adv_pct * candidate.raw_factors.liquidity_inr
        )
        return max_position_value / equity

    def _correlation_matrix(
        self, instrument_ids: list[str], as_of: dt.date
    ) -> pd.DataFrame | None:
        """Pairwise correlation of adjusted daily log returns over
        ``portfolio.correlation_lookback_days`` ending at ``as_of`` --
        causal by construction, since every price fetch is bounded by
        ``end=as_of``, the same discipline
        ``universe/stock_selector.py`` uses for factor computation.

        Returns ``None`` (skip the penalty entirely) rather than a
        partially-reliable matrix when too few overlapping observations
        exist, per ``portfolio.min_correlation_observations``.

        An instrument the market data provider has no data for at all
        (``DataNotAvailableError``) is skipped exactly like one with too few
        bars -- a data gap here must degrade the correlation estimate, never
        crash the whole construction. In practice every candidate has
        already passed ``StockSelector``'s own history check, but this
        module's correlation window is a separate, independently configured
        lookback, so the two can legitimately disagree.
        """
        window = self.config.correlation_lookback_days
        start = as_of - dt.timedelta(days=window * 3)
        returns: dict[str, pd.Series] = {}
        for instrument_id in instrument_ids:
            try:
                bars = self.market_data.get_equity_bars(
                    instrument_id, start, as_of, price_basis=PriceBasis.ADJUSTED
                )
            except DataNotAvailableError:
                continue
            if len(bars) < 2:
                continue
            closes = pd.Series(
                [float(bar.close) for bar in bars], index=[bar.session_date for bar in bars]
            ).tail(window + 1)
            log_returns = cast(pd.Series, np.log(closes / closes.shift(1))).dropna()
            if not log_returns.empty:
                returns[instrument_id] = log_returns

        if len(returns) < 2:
            return None
        frame = pd.DataFrame(returns).dropna(how="any")
        if len(frame) < self.config.min_correlation_observations:
            return None
        return frame.corr()

    @staticmethod
    def _normalize(weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if total <= 0 or not math.isfinite(total):
            raise PortfolioConstructionError(
                f"candidate weights sum to {total}; cannot normalize a non-positive total"
            )
        return {instrument_id: weight / total for instrument_id, weight in weights.items()}

    def _validate_against_config(
        self, portfolio: TargetPortfolio, exposure_target: AllocationTarget
    ) -> None:
        """Defense in depth: re-check the portfolio this method is about to
        return against the exact limits it was supposed to respect, and fail
        closed rather than silently returning a violation.
        """
        if portfolio.gross_exposure > exposure_target.max_gross_exposure + _EPSILON:
            raise PortfolioConstructionError(
                f"gross_exposure {portfolio.gross_exposure} exceeds the regime's "
                f"max_gross_exposure {exposure_target.max_gross_exposure}"
            )
        for position in portfolio.positions:
            if position.target_weight > self.config.max_single_name_pct + _EPSILON:
                raise PortfolioConstructionError(
                    f"{position.instrument_id}: weight {position.target_weight} exceeds "
                    f"max_single_name_pct {self.config.max_single_name_pct}"
                )
        if portfolio.cash_weight < -_EPSILON:
            raise PortfolioConstructionError(
                f"cash_weight {portfolio.cash_weight} is negative"
            )


# -- target vs. current vs. required trades ----------------------------
#
# A module-level function, deliberately not a PortfolioConstructor method:
# it needs no config, no market data, nothing but the two portfolios being
# compared -- constructing a PortfolioConstructor just to diff two
# TargetPortfolios would be a dependency with no purpose.


def required_trades(
    target: TargetPortfolio, current: TargetPortfolio | None
) -> list[RequiredTrade]:
    """The weight delta between ``current`` (what is held now) and
    ``target`` (what should be held), one ``RequiredTrade`` per instrument
    appearing in either. Not an order -- no quantity, no price, no broker
    call.
    """
    current_weights = {
        position.instrument_id: position.target_weight
        for position in (current.positions if current else ())
    }
    target_weights = {
        position.instrument_id: position.target_weight for position in target.positions
    }
    all_ids = sorted(set(current_weights) | set(target_weights))

    trades = []
    for instrument_id in all_ids:
        current_weight = current_weights.get(instrument_id, 0.0)
        target_weight = target_weights.get(instrument_id, 0.0)
        delta = target_weight - current_weight
        if abs(delta) <= _TRADE_TOLERANCE:
            action = TradeAction.HOLD
        elif target_weight <= _TRADE_TOLERANCE:
            action = TradeAction.EXIT
        elif delta > 0:
            action = TradeAction.BUY
        else:
            action = TradeAction.SELL
        trades.append(
            RequiredTrade(
                instrument_id=instrument_id,
                current_weight=current_weight,
                target_weight=target_weight,
                delta_weight=delta,
                action=action,
            )
        )
    return trades
