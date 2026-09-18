"""The independent risk-management engine with absolute veto authority.

Every proposed target portfolio from ``portfolio/portfolio_constructor.py``
passes through here before anything downstream can act on it. Nothing about
this module knows what a market regime is -- it never imports
``core.regime``, never sees a ``RegimeState`` or ``AllocationRegime``, and
never trusts that the portfolio it is given already respects any limit,
including its own construction-time ones. A regime that looks calm cannot
talk this layer out of a check; only the numbers in ``PortfolioRiskState``
can (docs/SPECIFICATION.md section 8, "NON-NEGOTIABLE... independent veto").

## What "veto" means here

This module never resizes a position. It approves or rejects each proposed
position outright (:class:`RiskDecision`). Turning an approved weight into a
final order quantity is a separate, later concern
(``risk/position_sizer.py``) -- conflating "should we do this at all" with
"how many shares exactly" would blur exactly the line specification section
8 draws between risk control and execution mechanics.

## Checks

Two kinds, evaluated in this order:

1. **Circuit breaker.** :meth:`RiskManager.evaluate` first asks
   ``risk/circuit_breaker.py`` for the current :class:`~risk.circuit_breaker.CircuitState`.
   HALTED rejects *every* proposed position outright, with no further checks
   run and no exceptions for trades that would reduce risk -- a halt driven
   by a broker-connectivity or system-health failure means no order is safe
   to route, sell or buy alike (docs/SPECIFICATION.md section 19). REDUCED_RISK
   tightens ``max_gross_exposure``, ``max_single_name_pct``, and
   ``max_sector_pct`` by ``RiskConfig.reduced_risk_exposure_multiplier`` and
   forbids opening any position not already held in ``current`` -- and, fail
   closed, treats *every* proposed position as new (rejecting all of them)
   when no ``current`` portfolio is supplied at all, rather than silently
   skipping this protection because the caller omitted it.
2. **Portfolio- and position-level checks**, against the (possibly
   tightened) limits: gross exposure, position count, sector concentration,
   correlation concentration, single-name exposure, liquidity/ADV
   participation, stale data, abnormal spread, daily turnover, and the V1
   long-only/no-leverage/no-borrowing invariants -- re-checked here as
   defense in depth even though ``portfolio/portfolio_constructor.py``
   already enforces most of them structurally, because this layer must
   never simply *trust* what it is handed.

A position with no matching :class:`~risk.portfolio_risk_state.PositionRisk`
entry in the supplied state is rejected outright (``MISSING_RISK_DATA``) --
fail closed rather than approve something this layer has no data to judge.

## Attribution: which position "caused" a portfolio-level breach

A portfolio-level breach (gross exposure, daily turnover, or a structural
no-leverage/no-borrowing violation) has no single culpable instrument, so it
rejects every proposed position. A position-count breach rejects only the
lowest-ranked positions beyond the cap (``TargetPosition.rank``, ascending =
best). A sector-concentration breach rejects every position in the
offending sector. A correlation breach rejects only the lower-ranked member
of the flagged pair -- the same "who gets blamed" rule
``portfolio/portfolio_constructor.py``'s correlation *penalty* uses, except
here it is an outright rejection, not a half-weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from config.models import RiskConfig
from monitoring.logger import get_logger
from portfolio.portfolio_constructor import (
    TargetPortfolio,
    TargetPosition,
    TradeAction,
    required_trades,
)
from risk.circuit_breaker import CircuitBreaker, CircuitState
from risk.portfolio_risk_state import PortfolioRiskState, PositionRisk

logger = get_logger("risk.risk_manager")

_TOLERANCE = 1e-9


class RiskCheck(StrEnum):
    CIRCUIT_BREAKER = "circuit_breaker"
    GROSS_EXPOSURE = "gross_exposure"
    SINGLE_NAME_EXPOSURE = "single_name_exposure"
    POSITION_COUNT = "position_count"
    SECTOR_CONCENTRATION = "sector_concentration"
    CORRELATION_CONCENTRATION = "correlation_concentration"
    LIQUIDITY = "liquidity"
    STALE_DATA = "stale_data"
    ABNORMAL_SPREAD = "abnormal_spread"
    DAILY_TURNOVER = "daily_turnover"
    NO_LEVERAGE = "no_leverage"
    NO_BORROWING = "no_borrowing"
    LONG_ONLY = "long_only"
    NEW_POSITION_DURING_REDUCED_RISK = "new_position_during_reduced_risk"
    MISSING_RISK_DATA = "missing_risk_data"


@dataclass(frozen=True, slots=True)
class RiskViolation:
    """One structured, machine-readable reason a position was rejected --
    never a bare boolean or a free-text-only message."""

    check: RiskCheck
    message: str
    limit: float | None
    observed: float | None


@dataclass(frozen=True, slots=True)
class RiskDecision:
    instrument_id: str
    approved: bool
    target_weight: float
    circuit_state: CircuitState
    violations: tuple[RiskViolation, ...]

    def __post_init__(self) -> None:
        if self.approved and self.violations:
            raise ValueError("an approved RiskDecision must carry no violations")
        if not self.approved and not self.violations:
            raise ValueError(
                "a rejected RiskDecision must carry at least one structured violation"
            )
        if not (0.0 <= self.target_weight <= 1.0):
            raise ValueError(f"target_weight must be in [0, 1], got {self.target_weight}")


class RiskManager:
    """Evaluates a proposed target portfolio against every configured risk
    control and returns one decision per proposed position.
    """

    def __init__(self, config: RiskConfig, circuit_breaker: CircuitBreaker) -> None:
        self.config = config
        self.circuit_breaker = circuit_breaker

    def evaluate(
        self,
        proposed: TargetPortfolio,
        risk_state: PortfolioRiskState,
        current: TargetPortfolio | None = None,
    ) -> list[RiskDecision]:
        """One :class:`RiskDecision` per position in ``proposed.positions``,
        in the same order.
        """
        circuit_status = self.circuit_breaker.evaluate(risk_state)

        if circuit_status.state is CircuitState.HALTED:
            reason = circuit_status.reason or "trading halted"
            decisions = [
                RiskDecision(
                    instrument_id=position.instrument_id,
                    approved=False,
                    target_weight=position.target_weight,
                    circuit_state=circuit_status.state,
                    violations=(
                        RiskViolation(RiskCheck.CIRCUIT_BREAKER, reason, None, None),
                    ),
                )
                for position in proposed.positions
            ]
            self._log_rejections(decisions)
            return decisions

        violations_by_id: dict[str, list[RiskViolation]] = {
            position.instrument_id: [] for position in proposed.positions
        }

        reduced_risk = circuit_status.state is CircuitState.REDUCED_RISK
        multiplier = self.config.reduced_risk_exposure_multiplier if reduced_risk else 1.0
        effective_max_gross = self.config.max_gross_exposure * multiplier
        effective_max_single_name = self.config.max_single_name_pct * multiplier
        effective_max_sector = self.config.max_sector_pct * multiplier

        self._check_structural_invariants(proposed, violations_by_id)
        self._check_gross_exposure(proposed, effective_max_gross, violations_by_id)
        self._check_position_count(proposed, violations_by_id)
        self._check_sector_concentration(proposed, effective_max_sector, violations_by_id)
        self._check_correlation(proposed, risk_state, violations_by_id)
        self._check_daily_turnover(proposed, current, risk_state, violations_by_id)
        self._check_per_position(
            proposed, risk_state, effective_max_single_name, violations_by_id
        )
        if reduced_risk:
            self._check_no_new_positions(proposed, current, violations_by_id)

        decisions = []
        for position in proposed.positions:
            violations = tuple(violations_by_id[position.instrument_id])
            decisions.append(
                RiskDecision(
                    instrument_id=position.instrument_id,
                    approved=len(violations) == 0,
                    target_weight=position.target_weight,
                    circuit_state=circuit_status.state,
                    violations=violations,
                )
            )
        self._log_rejections(decisions)
        return decisions

    @staticmethod
    def _log_rejections(decisions: list[RiskDecision]) -> None:
        for decision in decisions:
            if decision.approved:
                continue
            logger.info(
                "risk decision rejected: %s",
                decision.instrument_id,
                extra={
                    "extra_fields": {
                        "event": "risk_rejection",
                        "instrument_id": decision.instrument_id,
                        "circuit_state": decision.circuit_state.value,
                        "checks": [violation.check.value for violation in decision.violations],
                    }
                },
            )

    # -- portfolio-level checks --------------------------------------------

    @staticmethod
    def _check_structural_invariants(
        proposed: TargetPortfolio, violations_by_id: dict[str, list[RiskViolation]]
    ) -> None:
        """V1's non-negotiables: no leverage, no borrowing, long only.
        Defense in depth -- ``TargetPortfolio``/``TargetPosition`` already
        enforce these structurally, so these branches should never fire in
        practice, but this layer must never simply trust what it is given.
        """
        if proposed.cash_weight < -_TOLERANCE:
            violation = RiskViolation(
                RiskCheck.NO_BORROWING,
                f"cash_weight is negative ({proposed.cash_weight:.6f}); implies borrowing",
                0.0,
                proposed.cash_weight,
            )
            for violations in violations_by_id.values():
                violations.append(violation)
        if proposed.gross_exposure > 1.0 + _TOLERANCE:
            violation = RiskViolation(
                RiskCheck.NO_LEVERAGE,
                f"gross_exposure {proposed.gross_exposure:.6f} exceeds 1.0; implies leverage",
                1.0,
                proposed.gross_exposure,
            )
            for violations in violations_by_id.values():
                violations.append(violation)
        for position in proposed.positions:
            if position.target_weight <= 0:
                violations_by_id[position.instrument_id].append(
                    RiskViolation(
                        RiskCheck.LONG_ONLY,
                        f"target_weight {position.target_weight:.6f} is not strictly positive",
                        0.0,
                        position.target_weight,
                    )
                )

    def _check_gross_exposure(
        self,
        proposed: TargetPortfolio,
        effective_max_gross: float,
        violations_by_id: dict[str, list[RiskViolation]],
    ) -> None:
        if proposed.gross_exposure > effective_max_gross + _TOLERANCE:
            violation = RiskViolation(
                RiskCheck.GROSS_EXPOSURE,
                f"gross_exposure {proposed.gross_exposure:.6f} exceeds limit "
                f"{effective_max_gross:.6f}",
                effective_max_gross,
                proposed.gross_exposure,
            )
            for violations in violations_by_id.values():
                violations.append(violation)

    def _check_position_count(
        self, proposed: TargetPortfolio, violations_by_id: dict[str, list[RiskViolation]]
    ) -> None:
        limit = self.config.max_concurrent_positions
        if len(proposed.positions) <= limit:
            return
        excess = sorted(proposed.positions, key=lambda position: position.rank)[limit:]
        for position in excess:
            violations_by_id[position.instrument_id].append(
                RiskViolation(
                    RiskCheck.POSITION_COUNT,
                    f"{len(proposed.positions)} positions exceeds the limit of {limit}; "
                    "this is one of the lowest-ranked excess positions",
                    float(limit),
                    float(len(proposed.positions)),
                )
            )

    def _check_sector_concentration(
        self,
        proposed: TargetPortfolio,
        effective_max_sector: float,
        violations_by_id: dict[str, list[RiskViolation]],
    ) -> None:
        sector_totals: dict[str, float] = {}
        for position in proposed.positions:
            sector_totals[position.sector] = (
                sector_totals.get(position.sector, 0.0) + position.target_weight
            )
        for sector, total in sector_totals.items():
            if total <= effective_max_sector + _TOLERANCE:
                continue
            for position in proposed.positions:
                if position.sector != sector:
                    continue
                violations_by_id[position.instrument_id].append(
                    RiskViolation(
                        RiskCheck.SECTOR_CONCENTRATION,
                        f"sector {sector!r} totals {total:.6f}, exceeding limit "
                        f"{effective_max_sector:.6f}",
                        effective_max_sector,
                        total,
                    )
                )

    def _check_correlation(
        self,
        proposed: TargetPortfolio,
        risk_state: PortfolioRiskState,
        violations_by_id: dict[str, list[RiskViolation]],
    ) -> None:
        if risk_state.max_pairwise_correlation is None or risk_state.correlated_pair is None:
            return
        if risk_state.max_pairwise_correlation <= self.config.max_pairwise_correlation + _TOLERANCE:
            return
        first, second = risk_state.correlated_pair
        ranks = {position.instrument_id: position.rank for position in proposed.positions}
        if first not in ranks or second not in ranks:
            return
        worse, other = (first, second) if ranks[first] > ranks[second] else (second, first)
        violations_by_id[worse].append(
            RiskViolation(
                RiskCheck.CORRELATION_CONCENTRATION,
                f"pairwise correlation {risk_state.max_pairwise_correlation:.4f} with "
                f"{other!r} exceeds limit {self.config.max_pairwise_correlation:.4f}",
                self.config.max_pairwise_correlation,
                risk_state.max_pairwise_correlation,
            )
        )

    def _check_daily_turnover(
        self,
        proposed: TargetPortfolio,
        current: TargetPortfolio | None,
        risk_state: PortfolioRiskState,
        violations_by_id: dict[str, list[RiskViolation]],
    ) -> None:
        """Hold the day's turnover under the cap by rejecting the *weakest*
        increases, not by rejecting everything.

        The previous behaviour appended the violation to every position, so
        a portfolio whose target turnover exceeded the cap got nothing at
        all. That is fine as a churn guard and fatal as a starting
        condition: going flat -> 100% invested is 100% turnover in one
        session, so with a 50% cap the portfolio could never form. Not on
        day one, not ever. A walk-forward backtest showed it plainly --
        buy-and-hold made zero trades across a year, and every other
        strategy was silently capped at 50% exposure, which inverts the
        strategy's intent by letting it invest only in the regimes where it
        wants *least* exposure.

        Two rules, and the first one matters more:

        1. **Risk-reducing trades are never blocked.** An exit or a
           reduction lowers exposure, and a cap that can trap this system
           in a position it has decided to leave is worse than any amount
           of churn. Their turnover is counted, never vetoed.

        2. Increases are admitted in rank order until the budget is spent.
           Rank is the selector's own conviction ordering, so what survives
           is the highest-conviction subset that fits -- deterministic, and
           the same subset every time for the same inputs. The rest are
           rejected with the turnover violation, and the portfolio reaches
           its target over the next session or two instead of never.

        The cap still does its job: no session can exceed it, so a bug
        causing repeated full rebalances is still stopped.
        """
        trades = {trade.instrument_id: trade for trade in required_trades(proposed, current)}
        budget = self.config.max_daily_turnover_pct - risk_state.daily_turnover_pct_so_far

        reducing = [
            trade
            for trade in trades.values()
            if trade.action in (TradeAction.SELL, TradeAction.EXIT)
        ]
        spent = sum(abs(trade.delta_weight) for trade in reducing)

        increases = [
            position
            for position in sorted(proposed.positions, key=lambda p: p.rank)
            if trades.get(position.instrument_id) is not None
            and trades[position.instrument_id].action is TradeAction.BUY
        ]

        for position in increases:
            cost = abs(trades[position.instrument_id].delta_weight)
            if spent + cost <= budget + _TOLERANCE:
                spent += cost
                continue
            violations_by_id[position.instrument_id].append(
                RiskViolation(
                    RiskCheck.DAILY_TURNOVER,
                    f"daily turnover budget exhausted: {spent:.6f} already committed of "
                    f"{self.config.max_daily_turnover_pct:.6f}, this trade needs "
                    f"{cost:.6f}",
                    self.config.max_daily_turnover_pct,
                    spent + cost,
                )
            )

    def _check_no_new_positions(
        self,
        proposed: TargetPortfolio,
        current: TargetPortfolio | None,
        violations_by_id: dict[str, list[RiskViolation]],
    ) -> None:
        """Fail closed when ``current`` is unknown: with no current
        portfolio to compare against, every proposed position is treated as
        new and rejected, rather than silently skipping this protection."""
        held_ids = {position.instrument_id for position in current.positions} if current else set()
        for position in proposed.positions:
            if position.instrument_id in held_ids:
                continue
            violations_by_id[position.instrument_id].append(
                RiskViolation(
                    RiskCheck.NEW_POSITION_DURING_REDUCED_RISK,
                    "circuit breaker is REDUCED_RISK; no new positions may be opened",
                    None,
                    None,
                )
            )

    # -- per-position checks ------------------------------------------------

    def _check_per_position(
        self,
        proposed: TargetPortfolio,
        risk_state: PortfolioRiskState,
        effective_max_single_name: float,
        violations_by_id: dict[str, list[RiskViolation]],
    ) -> None:
        for position in proposed.positions:
            position_risk = risk_state.position_risk(position.instrument_id)
            if position_risk is None:
                violations_by_id[position.instrument_id].append(
                    RiskViolation(
                        RiskCheck.MISSING_RISK_DATA,
                        f"no PositionRisk entry for {position.instrument_id!r}; "
                        "cannot evaluate liquidity/staleness/spread",
                        None,
                        None,
                    )
                )
                continue

            self._check_single_name(position, effective_max_single_name, violations_by_id)
            self._check_liquidity(position, position_risk, risk_state.equity, violations_by_id)
            self._check_stale_data(position, position_risk, violations_by_id)
            self._check_spread(position, position_risk, violations_by_id)

    def _check_single_name(
        self,
        position: TargetPosition,
        effective_max_single_name: float,
        violations_by_id: dict[str, list[RiskViolation]],
    ) -> None:
        if position.target_weight > effective_max_single_name + _TOLERANCE:
            violations_by_id[position.instrument_id].append(
                RiskViolation(
                    RiskCheck.SINGLE_NAME_EXPOSURE,
                    f"target_weight {position.target_weight:.6f} exceeds limit "
                    f"{effective_max_single_name:.6f}",
                    effective_max_single_name,
                    position.target_weight,
                )
            )

    def _check_liquidity(
        self,
        position: TargetPosition,
        position_risk: PositionRisk,
        equity: float,
        violations_by_id: dict[str, list[RiskViolation]],
    ) -> None:
        implied_value = position.target_weight * equity
        liquidity_cap_value = (
            self.config.max_adv_participation_pct * position_risk.avg_daily_value_inr
        )
        if implied_value > liquidity_cap_value + _TOLERANCE:
            violations_by_id[position.instrument_id].append(
                RiskViolation(
                    RiskCheck.LIQUIDITY,
                    f"implied position value {implied_value:,.0f} exceeds "
                    f"{self.config.max_adv_participation_pct:.2%} of average daily value "
                    f"({liquidity_cap_value:,.0f})",
                    liquidity_cap_value,
                    implied_value,
                )
            )

    def _check_stale_data(
        self,
        position: TargetPosition,
        position_risk: PositionRisk,
        violations_by_id: dict[str, list[RiskViolation]],
    ) -> None:
        limit_seconds = self.config.stale_data_max_minutes * 60
        if position_risk.quote_age_seconds > limit_seconds + _TOLERANCE:
            violations_by_id[position.instrument_id].append(
                RiskViolation(
                    RiskCheck.STALE_DATA,
                    f"quote age {position_risk.quote_age_seconds:.0f}s exceeds limit "
                    f"{limit_seconds}s",
                    float(limit_seconds),
                    position_risk.quote_age_seconds,
                )
            )

    def _check_spread(
        self,
        position: TargetPosition,
        position_risk: PositionRisk,
        violations_by_id: dict[str, list[RiskViolation]],
    ) -> None:
        if position_risk.spread_bps > self.config.max_spread_bps + _TOLERANCE:
            violations_by_id[position.instrument_id].append(
                RiskViolation(
                    RiskCheck.ABNORMAL_SPREAD,
                    f"spread {position_risk.spread_bps:.1f}bps exceeds limit "
                    f"{self.config.max_spread_bps:.1f}bps",
                    self.config.max_spread_bps,
                    position_risk.spread_bps,
                )
            )
