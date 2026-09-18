"""RiskManager: the independent veto layer over a proposed target
portfolio. Covers the circuit-breaker interaction (HALTED rejects
everything, REDUCED_RISK tightens caps and forbids new positions),
every individual risk check, attribution of portfolio-level breaches to
specific positions, and the hard invariants a risk decision must never
violate.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from config.models import RiskConfig
from core.regime.allocation import AllocationRegime
from portfolio.portfolio_constructor import TargetPortfolio, TargetPosition
from risk.circuit_breaker import CircuitBreaker, CircuitState
from risk.portfolio_risk_state import PortfolioRiskState, PositionRisk
from risk.risk_manager import RiskCheck, RiskDecision, RiskManager, RiskViolation

AS_OF_DATE = dt.date(2023, 6, 1)
AS_OF = dt.datetime(2023, 6, 1, 10, 0, tzinfo=dt.UTC)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def risk_config(**overrides: object) -> RiskConfig:
    defaults: dict[str, object] = {
        "max_gross_exposure": 0.75,
        "max_leverage": 1.0,
        "max_single_name_pct": 0.15,
        "max_sector_pct": 0.30,
        "max_concurrent_positions": 5,
        "max_risk_per_position_pct": 0.01,
        "daily_loss_warning_pct": 0.01,
        "daily_loss_reduce_pct": 0.02,
        "daily_loss_halt_pct": 0.03,
        "rolling_loss_reduce_pct": 0.04,
        "rolling_loss_halt_pct": 0.06,
        "peak_to_trough_drawdown_halt_pct": 0.10,
        "stale_data_max_minutes": 30,
        "max_pairwise_correlation": 0.85,
        "max_adv_participation_pct": 0.08,
        "max_spread_bps": 100.0,
        "max_daily_turnover_pct": 0.50,
        "reduced_risk_exposure_multiplier": 0.50,
    }
    defaults.update(overrides)
    return RiskConfig.model_validate(defaults)


def target_position(
    instrument_id: str,
    weight: float,
    *,
    rank: int = 1,
    sector: str = "SECTOR",
    symbol: str | None = None,
) -> TargetPosition:
    return TargetPosition(
        instrument_id=instrument_id,
        symbol=symbol or instrument_id.split(":")[-1],
        target_weight=weight,
        sector=sector,
        rank=rank,
        score=1.0,
        binding_constraint="unconstrained",
    )


def target_portfolio(
    positions: list[TargetPosition], regime: AllocationRegime = AllocationRegime.NORMAL_RISK
) -> TargetPortfolio:
    gross = sum(position.target_weight for position in positions)
    return TargetPortfolio(
        as_of=AS_OF_DATE,
        positions=tuple(positions),
        cash_weight=round(1.0 - gross, 12),
        regime=regime,
        gross_exposure=gross,
    )


def position_risk(
    instrument_id: str,
    *,
    sector: str = "SECTOR",
    avg_daily_value_inr: float = 1_000_000_000.0,
    quote_age_seconds: float = 10.0,
    spread_bps: float = 5.0,
) -> PositionRisk:
    return PositionRisk(
        instrument_id=instrument_id,
        sector=sector,
        avg_daily_value_inr=avg_daily_value_inr,
        quote_age_seconds=quote_age_seconds,
        spread_bps=spread_bps,
    )


def risk_state(
    positions: tuple[PositionRisk, ...] = (), equity: float = 10_000_000.0, **overrides: object
) -> PortfolioRiskState:
    defaults: dict[str, object] = dict(
        as_of=AS_OF,
        equity=equity,
        positions=positions,
        daily_pnl_pct=0.0,
        rolling_pnl_pct=0.0,
        peak_to_trough_drawdown_pct=0.0,
        daily_turnover_pct_so_far=0.0,
        max_pairwise_correlation=None,
        correlated_pair=None,
        system_healthy=True,
        system_detail=None,
        broker_connected=True,
        broker_detail=None,
    )
    defaults.update(overrides)
    return PortfolioRiskState(**defaults)  # type: ignore[arg-type]


def risk_manager(tmp_path: Path, config: RiskConfig | None = None) -> RiskManager:
    cfg = config or risk_config()
    return RiskManager(cfg, CircuitBreaker(cfg, tmp_path / "circuit_breaker_state.json"))


def default_position_risks(instrument_ids: list[str], **kwargs: object) -> tuple[PositionRisk, ...]:
    return tuple(position_risk(instrument_id, **kwargs) for instrument_id in instrument_ids)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Circuit breaker: HALTED rejects everything, no exceptions
# --------------------------------------------------------------------------


def test_halted_rejects_every_proposed_position(tmp_path: Path) -> None:
    manager = risk_manager(tmp_path)
    positions = [target_position("NSE:A", 0.10), target_position("NSE:B", 0.10)]
    proposed = target_portfolio(positions)
    state = risk_state(
        default_position_risks(["NSE:A", "NSE:B"]), broker_connected=False
    )

    decisions = manager.evaluate(proposed, state)

    assert len(decisions) == 2
    assert all(not decision.approved for decision in decisions)
    assert all(decision.circuit_state is CircuitState.HALTED for decision in decisions)
    assert all(
        decision.violations[0].check is RiskCheck.CIRCUIT_BREAKER for decision in decisions
    )


def test_halted_rejects_even_a_risk_reducing_trade(tmp_path: Path) -> None:
    """A halt driven by broker/system failure must block everything, even a
    trade that only shrinks exposure -- no order is safe to route."""
    manager = risk_manager(tmp_path)
    current = target_portfolio([target_position("NSE:A", 0.30)])
    proposed = target_portfolio([target_position("NSE:A", 0.05)])
    state = risk_state(default_position_risks(["NSE:A"]), system_healthy=False)

    decisions = manager.evaluate(proposed, state, current=current)

    assert all(not decision.approved for decision in decisions)


def test_halted_decision_has_a_structured_reason(tmp_path: Path) -> None:
    manager = risk_manager(tmp_path)
    proposed = target_portfolio([target_position("NSE:A", 0.10)])
    state = risk_state(default_position_risks(["NSE:A"]), broker_connected=False)

    (decision,) = manager.evaluate(proposed, state)

    assert len(decision.violations) == 1
    violation = decision.violations[0]
    assert violation.check is RiskCheck.CIRCUIT_BREAKER
    assert violation.message


# --------------------------------------------------------------------------
# Circuit breaker: REDUCED_RISK tightens caps, forbids new positions
# --------------------------------------------------------------------------


def test_reduced_risk_tightens_gross_exposure_cap(tmp_path: Path) -> None:
    cfg = risk_config(max_gross_exposure=0.75, reduced_risk_exposure_multiplier=0.5)
    manager = risk_manager(tmp_path, cfg)
    # 0.60 is under the normal 0.75 cap but over the reduced 0.375 cap.
    positions = [target_position("NSE:A", 0.30, rank=1), target_position("NSE:B", 0.30, rank=2)]
    proposed = target_portfolio(positions)
    state = risk_state(
        default_position_risks(["NSE:A", "NSE:B"]), daily_pnl_pct=cfg.daily_loss_reduce_pct
    )

    decisions = manager.evaluate(proposed, state)

    assert all(decision.circuit_state is CircuitState.REDUCED_RISK for decision in decisions)
    assert all(not decision.approved for decision in decisions)
    assert all(
        any(v.check is RiskCheck.GROSS_EXPOSURE for v in decision.violations)
        for decision in decisions
    )


def test_reduced_risk_forbids_opening_a_new_position(tmp_path: Path) -> None:
    cfg = risk_config()
    manager = risk_manager(tmp_path, cfg)
    current = target_portfolio([target_position("NSE:HELD", 0.05)])
    proposed = target_portfolio(
        [target_position("NSE:HELD", 0.05), target_position("NSE:NEW", 0.03, rank=2)]
    )
    state = risk_state(
        default_position_risks(["NSE:HELD", "NSE:NEW"]), daily_pnl_pct=cfg.daily_loss_reduce_pct
    )

    decisions = manager.evaluate(proposed, state, current=current)
    by_id = {decision.instrument_id: decision for decision in decisions}

    assert by_id["NSE:HELD"].approved is True
    assert by_id["NSE:NEW"].approved is False
    assert any(
        v.check is RiskCheck.NEW_POSITION_DURING_REDUCED_RISK
        for v in by_id["NSE:NEW"].violations
    )


def test_reduced_risk_without_current_portfolio_fails_closed(tmp_path: Path) -> None:
    """Fail-closed design: without a current portfolio, RiskManager cannot
    tell what's "new", so it treats every proposed position as new (and
    rejects it) rather than silently skipping the protection."""
    cfg = risk_config()
    manager = risk_manager(tmp_path, cfg)
    proposed = target_portfolio([target_position("NSE:A", 0.05)])
    state = risk_state(default_position_risks(["NSE:A"]), daily_pnl_pct=cfg.daily_loss_reduce_pct)

    (decision,) = manager.evaluate(proposed, state, current=None)

    assert decision.approved is False
    assert any(
        v.check is RiskCheck.NEW_POSITION_DURING_REDUCED_RISK for v in decision.violations
    )


# --------------------------------------------------------------------------
# Portfolio-level checks
# --------------------------------------------------------------------------


def test_gross_exposure_breach_rejects_every_position(tmp_path: Path) -> None:
    cfg = risk_config(max_gross_exposure=0.50)
    manager = risk_manager(tmp_path, cfg)
    positions = [target_position("NSE:A", 0.30, rank=1), target_position("NSE:B", 0.30, rank=2)]
    proposed = target_portfolio(positions)
    state = risk_state(default_position_risks(["NSE:A", "NSE:B"]))

    decisions = manager.evaluate(proposed, state)

    assert all(not decision.approved for decision in decisions)
    assert all(
        any(v.check is RiskCheck.GROSS_EXPOSURE for v in decision.violations)
        for decision in decisions
    )


def test_position_count_breach_rejects_lowest_ranked_excess(tmp_path: Path) -> None:
    cfg = risk_config(max_concurrent_positions=2, max_gross_exposure=1.0, max_sector_pct=1.0)
    manager = risk_manager(tmp_path, cfg)
    positions = [
        target_position("NSE:A", 0.10, rank=1),
        target_position("NSE:B", 0.10, rank=2),
        target_position("NSE:C", 0.10, rank=3),
    ]
    proposed = target_portfolio(positions)
    state = risk_state(default_position_risks(["NSE:A", "NSE:B", "NSE:C"]))

    decisions = manager.evaluate(proposed, state)
    by_id = {decision.instrument_id: decision for decision in decisions}

    assert by_id["NSE:A"].approved is True
    assert by_id["NSE:B"].approved is True
    assert by_id["NSE:C"].approved is False
    assert any(v.check is RiskCheck.POSITION_COUNT for v in by_id["NSE:C"].violations)


def test_sector_concentration_breach_rejects_whole_sector(tmp_path: Path) -> None:
    cfg = risk_config(max_sector_pct=0.20, max_gross_exposure=1.0)
    manager = risk_manager(tmp_path, cfg)
    positions = [
        target_position("NSE:A", 0.15, rank=1, sector="TECH"),
        target_position("NSE:B", 0.15, rank=2, sector="TECH"),
        target_position("NSE:C", 0.15, rank=3, sector="OTHER"),
    ]
    proposed = target_portfolio(positions)
    state = risk_state(default_position_risks(["NSE:A", "NSE:B", "NSE:C"]))

    decisions = manager.evaluate(proposed, state)
    by_id = {decision.instrument_id: decision for decision in decisions}

    assert by_id["NSE:A"].approved is False
    assert by_id["NSE:B"].approved is False
    assert by_id["NSE:C"].approved is True
    assert all(
        any(v.check is RiskCheck.SECTOR_CONCENTRATION for v in by_id[iid].violations)
        for iid in ("NSE:A", "NSE:B")
    )


def test_correlation_breach_rejects_only_lower_ranked_member(tmp_path: Path) -> None:
    cfg = risk_config(max_pairwise_correlation=0.80, max_gross_exposure=1.0)
    manager = risk_manager(tmp_path, cfg)
    positions = [
        target_position("NSE:A", 0.10, rank=1),
        target_position("NSE:B", 0.10, rank=2),
    ]
    proposed = target_portfolio(positions)
    state = risk_state(
        default_position_risks(["NSE:A", "NSE:B"]),
        max_pairwise_correlation=0.95,
        correlated_pair=("NSE:A", "NSE:B"),
    )

    decisions = manager.evaluate(proposed, state)
    by_id = {decision.instrument_id: decision for decision in decisions}

    assert by_id["NSE:A"].approved is True
    assert by_id["NSE:B"].approved is False
    assert any(v.check is RiskCheck.CORRELATION_CONCENTRATION for v in by_id["NSE:B"].violations)


def test_correlation_within_limit_is_not_flagged(tmp_path: Path) -> None:
    cfg = risk_config(max_pairwise_correlation=0.95, max_gross_exposure=1.0)
    manager = risk_manager(tmp_path, cfg)
    positions = [target_position("NSE:A", 0.10, rank=1), target_position("NSE:B", 0.10, rank=2)]
    proposed = target_portfolio(positions)
    state = risk_state(
        default_position_risks(["NSE:A", "NSE:B"]),
        max_pairwise_correlation=0.85,
        correlated_pair=("NSE:A", "NSE:B"),
    )

    decisions = manager.evaluate(proposed, state)

    assert all(decision.approved for decision in decisions)


def test_a_single_position_over_the_turnover_budget_is_rejected(tmp_path: Path) -> None:
    """One position that alone exceeds the budget leaves nothing to
    admit, so the outcome is the same as rejecting everything. With
    several positions it is not -- see the subset tests below."""
    cfg = risk_config(max_daily_turnover_pct=0.10, max_gross_exposure=1.0)
    manager = risk_manager(tmp_path, cfg)
    positions = [target_position("NSE:A", 0.30, rank=1)]
    proposed = target_portfolio(positions)
    state = risk_state(default_position_risks(["NSE:A"]))

    decisions = manager.evaluate(proposed, state, current=None)

    assert all(not decision.approved for decision in decisions)
    assert all(
        any(v.check is RiskCheck.DAILY_TURNOVER for v in decision.violations)
        for decision in decisions
    )


def test_daily_turnover_uses_the_delta_not_the_full_target_weight(tmp_path: Path) -> None:
    """A large but nearly-unchanged position should not trip a turnover
    limit sized for the actual trade, only for the full weight."""
    cfg = risk_config(
        max_daily_turnover_pct=0.02,
        max_gross_exposure=1.0,
        max_single_name_pct=1.0,
        max_sector_pct=1.0,
    )
    manager = risk_manager(tmp_path, cfg)
    current = target_portfolio([target_position("NSE:A", 0.30)])
    proposed = target_portfolio([target_position("NSE:A", 0.305)])
    state = risk_state(default_position_risks(["NSE:A"]))

    decisions = manager.evaluate(proposed, state, current=current)

    assert all(decision.approved for decision in decisions)


def test_daily_turnover_accounts_for_turnover_already_used_today(tmp_path: Path) -> None:
    cfg = risk_config(max_daily_turnover_pct=0.10, max_gross_exposure=1.0)
    manager = risk_manager(tmp_path, cfg)
    proposed = target_portfolio([target_position("NSE:A", 0.05)])
    state = risk_state(default_position_risks(["NSE:A"]), daily_turnover_pct_so_far=0.08)

    decisions = manager.evaluate(proposed, state, current=None)

    assert all(not decision.approved for decision in decisions)


# --------------------------------------------------------------------------
# Per-position checks
# --------------------------------------------------------------------------


def test_single_name_breach_rejects_only_that_position(tmp_path: Path) -> None:
    cfg = risk_config(max_single_name_pct=0.12, max_gross_exposure=1.0, max_sector_pct=1.0)
    manager = risk_manager(tmp_path, cfg)
    positions = [
        target_position("NSE:BIG", 0.20, rank=1),
        target_position("NSE:SMALL", 0.05, rank=2),
    ]
    proposed = target_portfolio(positions)
    state = risk_state(default_position_risks(["NSE:BIG", "NSE:SMALL"]))

    decisions = manager.evaluate(proposed, state)
    by_id = {decision.instrument_id: decision for decision in decisions}

    assert by_id["NSE:BIG"].approved is False
    assert by_id["NSE:SMALL"].approved is True
    assert any(v.check is RiskCheck.SINGLE_NAME_EXPOSURE for v in by_id["NSE:BIG"].violations)


def test_liquidity_breach_rejects_illiquid_position(tmp_path: Path) -> None:
    cfg = risk_config(max_adv_participation_pct=0.05)
    manager = risk_manager(tmp_path, cfg)
    proposed = target_portfolio([target_position("NSE:THIN", 0.10)])
    # implied value = 0.10 * 10,000,000 = 1,000,000; cap = 0.05 * ADV
    state = risk_state(
        (position_risk("NSE:THIN", avg_daily_value_inr=1_000_000.0),), equity=10_000_000.0
    )

    (decision,) = manager.evaluate(proposed, state)

    assert decision.approved is False
    assert any(v.check is RiskCheck.LIQUIDITY for v in decision.violations)


def test_stale_data_rejects_stale_position(tmp_path: Path) -> None:
    cfg = risk_config(stale_data_max_minutes=5)
    manager = risk_manager(tmp_path, cfg)
    proposed = target_portfolio([target_position("NSE:A", 0.10)])
    state = risk_state((position_risk("NSE:A", quote_age_seconds=600.0),))

    (decision,) = manager.evaluate(proposed, state)

    assert decision.approved is False
    assert any(v.check is RiskCheck.STALE_DATA for v in decision.violations)


def test_abnormal_spread_rejects_wide_spread_position(tmp_path: Path) -> None:
    cfg = risk_config(max_spread_bps=50.0)
    manager = risk_manager(tmp_path, cfg)
    proposed = target_portfolio([target_position("NSE:A", 0.10)])
    state = risk_state((position_risk("NSE:A", spread_bps=200.0),))

    (decision,) = manager.evaluate(proposed, state)

    assert decision.approved is False
    assert any(v.check is RiskCheck.ABNORMAL_SPREAD for v in decision.violations)


def test_missing_risk_data_fails_closed(tmp_path: Path) -> None:
    manager = risk_manager(tmp_path)
    proposed = target_portfolio([target_position("NSE:UNKNOWN", 0.10)])
    state = risk_state(())

    (decision,) = manager.evaluate(proposed, state)

    assert decision.approved is False
    assert any(v.check is RiskCheck.MISSING_RISK_DATA for v in decision.violations)


def test_clean_proposal_is_fully_approved(tmp_path: Path) -> None:
    manager = risk_manager(tmp_path)
    positions = [target_position("NSE:A", 0.10, rank=1), target_position("NSE:B", 0.10, rank=2)]
    proposed = target_portfolio(positions)
    state = risk_state(default_position_risks(["NSE:A", "NSE:B"]))

    decisions = manager.evaluate(proposed, state)

    assert all(decision.approved for decision in decisions)
    assert all(decision.violations == () for decision in decisions)
    assert all(decision.circuit_state is CircuitState.NORMAL for decision in decisions)


# --------------------------------------------------------------------------
# RiskDecision / RiskViolation dataclass invariants
# --------------------------------------------------------------------------


def test_risk_decision_rejects_approved_with_violations() -> None:
    with pytest.raises(ValueError, match="no violations"):
        RiskDecision(
            instrument_id="NSE:A",
            approved=True,
            target_weight=0.1,
            circuit_state=CircuitState.NORMAL,
            violations=(RiskViolation(RiskCheck.STALE_DATA, "x", None, None),),
        )


def test_risk_decision_rejects_unapproved_without_violations() -> None:
    with pytest.raises(ValueError, match="structured violation"):
        RiskDecision(
            instrument_id="NSE:A",
            approved=False,
            target_weight=0.1,
            circuit_state=CircuitState.NORMAL,
            violations=(),
        )


def test_risk_decision_rejects_weight_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="target_weight"):
        RiskDecision(
            instrument_id="NSE:A",
            approved=True,
            target_weight=1.5,
            circuit_state=CircuitState.NORMAL,
            violations=(),
        )


# --------------------------------------------------------------------------
# Structural defense-in-depth: no leverage / no borrowing / long only.
#
# TargetPortfolio's own __post_init__ makes these unconstructible through
# normal use (cash_weight >= 0 and gross_exposure == sum(positions) are
# already enforced there), so these checks are exercised directly against
# duck-typed fakes shaped like TargetPortfolio/TargetPosition -- the only
# way to prove the *defense-in-depth* branch itself is correct, since no
# real TargetPortfolio can ever reach it.
# --------------------------------------------------------------------------


class _FakePosition:
    def __init__(self, instrument_id: str, target_weight: float, sector: str = "SECTOR") -> None:
        self.instrument_id = instrument_id
        self.target_weight = target_weight
        self.sector = sector
        self.rank = 1


class _FakePortfolio:
    def __init__(
        self, positions: list[_FakePosition], cash_weight: float, gross_exposure: float
    ) -> None:
        self.positions = positions
        self.cash_weight = cash_weight
        self.gross_exposure = gross_exposure


def test_structural_check_flags_negative_cash_as_borrowing() -> None:
    fake = _FakePortfolio([_FakePosition("NSE:A", 0.5)], cash_weight=-0.1, gross_exposure=1.1)
    violations: dict[str, list[RiskViolation]] = {"NSE:A": []}
    RiskManager._check_structural_invariants(fake, violations)  # type: ignore[arg-type]
    assert any(v.check is RiskCheck.NO_BORROWING for v in violations["NSE:A"])


def test_structural_check_flags_gross_exposure_over_one_as_leverage() -> None:
    fake = _FakePortfolio([_FakePosition("NSE:A", 1.1)], cash_weight=0.0, gross_exposure=1.1)
    violations: dict[str, list[RiskViolation]] = {"NSE:A": []}
    RiskManager._check_structural_invariants(fake, violations)  # type: ignore[arg-type]
    assert any(v.check is RiskCheck.NO_LEVERAGE for v in violations["NSE:A"])


def test_structural_check_flags_non_positive_weight_as_not_long_only() -> None:
    fake = _FakePortfolio(
        [_FakePosition("NSE:A", 0.0), _FakePosition("NSE:B", 0.2)],
        cash_weight=0.8,
        gross_exposure=0.2,
    )
    violations: dict[str, list[RiskViolation]] = {"NSE:A": [], "NSE:B": []}
    RiskManager._check_structural_invariants(fake, violations)  # type: ignore[arg-type]
    assert any(v.check is RiskCheck.LONG_ONLY for v in violations["NSE:A"])
    assert violations["NSE:B"] == []


# --------------------------------------------------------------------------
# Invariant / property tests over randomized proposals
# --------------------------------------------------------------------------


def _random_portfolio(rng: np.random.Generator, n: int, max_weight: float) -> TargetPortfolio:
    weights = rng.uniform(0.01, max_weight, size=n)
    total = weights.sum()
    if total > 0.9:
        weights = weights * (0.9 / total)
    sectors = [f"SEC{i % 3}" for i in range(n)]
    positions = [
        target_position(f"NSE:S{i:02d}", float(weights[i]), rank=i + 1, sector=sectors[i])
        for i in range(n)
    ]
    return target_portfolio(positions)


def test_invariant_halted_state_always_rejects_every_position(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    cfg = risk_config()
    for i in range(15):
        n = int(rng.integers(1, 6))
        proposed = _random_portfolio(rng, n, max_weight=0.10)
        state = risk_state(
            default_position_risks([p.instrument_id for p in proposed.positions]),
            broker_connected=False,
        )
        manager = RiskManager(cfg, CircuitBreaker(cfg, tmp_path / f"halt_{i}.json"))
        decisions = manager.evaluate(proposed, state)
        assert all(not decision.approved for decision in decisions)


def test_invariant_rejection_always_has_a_structured_reason(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    cfg = risk_config(max_single_name_pct=0.08, max_gross_exposure=0.5)
    manager = risk_manager(tmp_path, cfg)
    for _ in range(25):
        n = int(rng.integers(1, 8))
        proposed = _random_portfolio(rng, n, max_weight=0.20)
        state = risk_state(
            default_position_risks([p.instrument_id for p in proposed.positions])
        )
        for decision in manager.evaluate(proposed, state):
            if decision.approved:
                assert decision.violations == ()
            else:
                assert len(decision.violations) >= 1
                for violation in decision.violations:
                    assert isinstance(violation.check, RiskCheck)
                    assert violation.message


def test_invariant_no_approved_position_exceeds_the_effective_single_name_cap(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(9)
    cfg = risk_config(max_single_name_pct=0.15, max_gross_exposure=1.0, max_sector_pct=1.0)
    manager = risk_manager(tmp_path, cfg)
    for _ in range(25):
        n = int(rng.integers(1, 6))
        proposed = _random_portfolio(rng, n, max_weight=0.30)
        state = risk_state(
            default_position_risks([p.instrument_id for p in proposed.positions])
        )
        decisions = manager.evaluate(proposed, state)
        by_id = {position.instrument_id: position for position in proposed.positions}
        for decision in decisions:
            if decision.approved:
                assert by_id[decision.instrument_id].target_weight <= cfg.max_single_name_pct + 1e-9


def test_invariant_reduced_risk_caps_are_never_looser_than_normal(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    for _ in range(20):
        multiplier = float(rng.uniform(0.01, 0.99))
        cfg = risk_config(reduced_risk_exposure_multiplier=multiplier)
        assert cfg.max_gross_exposure * multiplier <= cfg.max_gross_exposure
        assert cfg.max_single_name_pct * multiplier <= cfg.max_single_name_pct
        assert cfg.max_sector_pct * multiplier <= cfg.max_sector_pct


def test_invariant_gross_exposure_breach_is_never_partially_approved(tmp_path: Path) -> None:
    """If the whole proposed portfolio's gross exposure exceeds the
    effective cap, no individual position may sneak through approved --
    the breach has no single culpable instrument."""
    rng = np.random.default_rng(17)
    cfg = risk_config(max_gross_exposure=0.30, max_single_name_pct=1.0, max_sector_pct=1.0)
    manager = risk_manager(tmp_path, cfg)
    for _ in range(20):
        n = int(rng.integers(2, 6))
        proposed = _random_portfolio(rng, n, max_weight=0.20)
        if proposed.gross_exposure <= cfg.max_gross_exposure:
            continue
        state = risk_state(
            default_position_risks([p.instrument_id for p in proposed.positions])
        )
        decisions = manager.evaluate(proposed, state)
        assert all(not decision.approved for decision in decisions)


# --------------------------------------------------------------------------
# Daily turnover admits a fitting subset rather than rejecting everything
# --------------------------------------------------------------------------


def test_turnover_admits_the_highest_ranked_positions_that_fit(tmp_path: Path) -> None:
    """The fix for a defect a walk-forward backtest exposed.

    Rejecting every position when the total exceeded the cap meant a
    portfolio could never form: flat -> fully invested is 100% turnover in
    one session, so with a 50% cap nothing was ever approved. Not on day
    one, not ever. Buy-and-hold made zero trades across a year, and every
    other strategy was silently capped at 50% exposure.

    Rank order decides who gets in, because rank is the selector's own
    conviction ordering -- so the surviving subset is the best available
    one, and it is the same subset every time for the same inputs.
    """
    cfg = risk_config(
        max_daily_turnover_pct=0.50,
        max_gross_exposure=1.0,
        max_single_name_pct=1.0,
        max_sector_pct=1.0,
    )
    manager = risk_manager(tmp_path, cfg)
    positions = [target_position(f"NSE:{c}", 0.20, rank=i + 1) for i, c in enumerate("ABCDE")]
    proposed = target_portfolio(positions)
    state = risk_state(default_position_risks([p.instrument_id for p in positions]))

    decisions = {d.instrument_id: d for d in manager.evaluate(proposed, state, current=None)}

    # 0.20 each, budget 0.50 -> the top two fit, the rest do not.
    assert decisions["NSE:A"].approved is True
    assert decisions["NSE:B"].approved is True
    assert decisions["NSE:C"].approved is False
    assert decisions["NSE:D"].approved is False
    assert decisions["NSE:E"].approved is False


def test_the_turnover_cap_is_still_enforced(tmp_path: Path) -> None:
    """Admitting a subset must not become admitting everything. The guard
    exists to stop a bug causing repeated full rebalances, and it still
    does."""
    cfg = risk_config(
        max_daily_turnover_pct=0.50,
        max_gross_exposure=1.0,
        max_single_name_pct=1.0,
        max_sector_pct=1.0,
    )
    manager = risk_manager(tmp_path, cfg)
    positions = [target_position(f"NSE:{c}", 0.20, rank=i + 1) for i, c in enumerate("ABCDE")]
    proposed = target_portfolio(positions)
    state = risk_state(default_position_risks([p.instrument_id for p in positions]))

    decisions = manager.evaluate(proposed, state, current=None)
    admitted = sum(d.target_weight for d in decisions if d.approved)
    assert admitted <= cfg.max_daily_turnover_pct + 1e-9


def test_a_risk_reducing_trade_is_never_blocked_by_turnover(tmp_path: Path) -> None:
    """The safety property, and the more important half of the rule.

    A cap that can trap this system in a position it has decided to leave
    is worse than any amount of churn. Exits and reductions consume the
    budget but are never vetoed by it.
    """
    cfg = risk_config(
        max_daily_turnover_pct=0.05,
        max_gross_exposure=1.0,
        max_single_name_pct=1.0,
        max_sector_pct=1.0,
    )
    manager = risk_manager(tmp_path, cfg)
    current = target_portfolio(
        [target_position("NSE:A", 0.40, rank=1), target_position("NSE:B", 0.40, rank=2)]
    )
    # Exit A entirely and halve B: 0.40 + 0.20 = 0.60 turnover, far over
    # the 0.05 cap, and every bit of it reduces exposure.
    proposed = target_portfolio([target_position("NSE:B", 0.20, rank=2)])
    state = risk_state(default_position_risks(["NSE:A", "NSE:B"]))

    decisions = manager.evaluate(proposed, state, current=current)

    assert all(decision.approved for decision in decisions), (
        "a reduction was blocked by the turnover cap"
    )


def test_turnover_already_spent_today_reduces_the_remaining_budget(tmp_path: Path) -> None:
    cfg = risk_config(
        max_daily_turnover_pct=0.50,
        max_gross_exposure=1.0,
        max_single_name_pct=1.0,
        max_sector_pct=1.0,
    )
    manager = risk_manager(tmp_path, cfg)
    positions = [target_position(f"NSE:{c}", 0.20, rank=i + 1) for i, c in enumerate("ABC")]
    proposed = target_portfolio(positions)
    state = risk_state(
        default_position_risks([p.instrument_id for p in positions]),
        daily_turnover_pct_so_far=0.30,
    )

    decisions = {d.instrument_id: d for d in manager.evaluate(proposed, state, current=None)}

    # Only 0.20 of budget remains, so exactly one position fits.
    assert decisions["NSE:A"].approved is True
    assert decisions["NSE:B"].approved is False
    assert decisions["NSE:C"].approved is False
