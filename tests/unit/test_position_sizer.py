"""Position sizing: the one place a final order quantity is computed.

The specification gives two sizing formulas and never says how they
combine (docs/ARCHITECTURE.md, "Resolved specification ambiguities"), so
the rule this module implements -- take the minimum, then narrow -- is a
decision rather than a transcription. These tests pin that decision and
the direction of every cap.

The recurring property is **no constraint is ever widened**. Each cap is
an upper bound, so the final quantity must satisfy all of them at once;
if any single cap could increase the answer, the module would be
producing positions that violate a limit someone believes is in force.
"""

from __future__ import annotations

import math

import pytest

from config.loader import load_settings
from config.models import RiskConfig
from portfolio.portfolio_constructor import TargetPosition
from risk.position_sizer import (
    OVERNIGHT_GAP_STRESS_MULTIPLE,
    PositionSizer,
    PositionSizingError,
)


@pytest.fixture(scope="module")
def risk_config() -> RiskConfig:
    """This repository's own shipped limits, not invented ones -- so these
    tests also fail if a configured limit becomes incoherent."""
    return load_settings().risk


@pytest.fixture
def sizer(risk_config: RiskConfig) -> PositionSizer:
    return PositionSizer(risk_config)


def _position(weight: float = 0.10, instrument_id: str = "NSE:RELIANCE") -> TargetPosition:
    return TargetPosition(
        instrument_id=instrument_id,
        symbol=instrument_id.split(":")[-1],
        target_weight=weight,
        sector="Energy",
        rank=1,
        score=1.0,
        binding_constraint="unconstrained",
    )


UNBOUNDED = 10**9
"""A cap high enough not to bind, so a test can isolate one constraint."""


# ---------------------------------------------------------------------------
# The two formulas
# ---------------------------------------------------------------------------


def test_weight_based_quantity_is_equity_times_weight_over_price(sizer: PositionSizer) -> None:
    assert sizer.weight_based_quantity(_position(0.10), equity=1_000_000, price=250.0) == 400


def test_weight_based_quantity_floors_rather_than_rounds(sizer: PositionSizer) -> None:
    """Rounding up would push the position past the very weight cap that
    produced the target, and a fractional share cannot be bought."""
    # 1,000,000 * 0.10 / 300 = 333.33...
    assert sizer.weight_based_quantity(_position(0.10), equity=1_000_000, price=300.0) == 333


def test_risk_based_quantity_bounds_loss_per_position(
    sizer: PositionSizer, risk_config: RiskConfig
) -> None:
    """Section 8.1's formula: the risk budget divided by the distance price
    must travel against the position before it is exited."""
    equity, stop = 1_000_000.0, 10.0
    quantity = sizer.risk_based_quantity(equity, entry_price=250.0, stop_distance=stop)

    budget = equity * risk_config.max_risk_per_position_pct
    expected = math.floor(budget / (stop * OVERNIGHT_GAP_STRESS_MULTIPLE))
    assert quantity == expected


def test_the_gap_stress_makes_positions_smaller_not_larger(sizer: PositionSizer) -> None:
    """The direction is the whole point. This system trades daily bars, so
    every position is held overnight and can reopen through its stop. A
    stress that increased size would be worse than no stress at all."""
    stressed = sizer.risk_based_quantity(1_000_000, entry_price=250.0, stop_distance=10.0)
    unstressed = math.floor(
        1_000_000 * sizer.config.max_risk_per_position_pct / 10.0
    )
    assert stressed < unstressed
    assert OVERNIGHT_GAP_STRESS_MULTIPLE > 1.0


def test_a_wider_stop_distance_gives_a_smaller_position(sizer: PositionSizer) -> None:
    """The core risk relationship: a more volatile name, needing a wider
    risk distance, gets less capital for the same loss budget."""
    tight = sizer.risk_based_quantity(1_000_000, entry_price=250.0, stop_distance=5.0)
    wide = sizer.risk_based_quantity(1_000_000, entry_price=250.0, stop_distance=50.0)
    assert wide < tight


# ---------------------------------------------------------------------------
# Reconciliation: the minimum, and which constraint bound
# ---------------------------------------------------------------------------


def test_reconcile_takes_the_minimum_of_both_formulas(sizer: PositionSizer) -> None:
    """The reconciliation the specification leaves open.

    The two formulas answer different questions, and a quantity that
    satisfies one can violate the other. Only the minimum satisfies both.
    """
    order = sizer.reconcile(
        _position(0.10),
        equity=1_000_000,
        price=250.0,
        stop_distance=10.0,
        available_cash=UNBOUNDED,
        liquidity_participation_cap=UNBOUNDED,
    )
    by_weight = sizer.weight_based_quantity(_position(0.10), 1_000_000, 250.0)
    by_risk = sizer.risk_based_quantity(1_000_000, 250.0, 10.0)
    assert order.quantity == min(by_weight, by_risk)


@pytest.mark.parametrize(
    ("kwargs", "expected_binding"),
    [
        # A tiny target weight is what limits it.
        (
            {"weight": 0.001, "stop_distance": 1.0, "cash": UNBOUNDED, "liquidity": UNBOUNDED},
            "target_weight",
        ),
        # A very wide risk distance starves the risk budget first.
        (
            {"weight": 0.99, "stop_distance": 5000.0, "cash": UNBOUNDED, "liquidity": UNBOUNDED},
            "risk_per_position",
        ),
        # Cash runs out before anything else.
        (
            {"weight": 0.99, "stop_distance": 0.01, "cash": 1000.0, "liquidity": UNBOUNDED},
            "available_cash",
        ),
        # The book is too thin to participate further.
        (
            {"weight": 0.99, "stop_distance": 0.01, "cash": UNBOUNDED, "liquidity": 7},
            "liquidity_participation",
        ),
    ],
)
def test_the_binding_constraint_is_recorded(
    sizer: PositionSizer, kwargs: dict[str, float], expected_binding: str
) -> None:
    """Auditability. "Why is this position so small?" is asked after the
    fact, usually during an incident, and the answer has to be in the
    record rather than re-derived from inputs nobody kept."""
    order = sizer.reconcile(
        _position(float(kwargs["weight"])),
        equity=1_000_000,
        price=250.0,
        stop_distance=float(kwargs["stop_distance"]),
        available_cash=float(kwargs["cash"]),
        liquidity_participation_cap=int(kwargs["liquidity"]),
    )
    assert order.binding_constraint == expected_binding


def test_the_single_name_cap_binds_even_when_the_target_weight_exceeds_it(
    sizer: PositionSizer, risk_config: RiskConfig
) -> None:
    """A proposal asking for more than the configured single-name limit
    must not get it. This is the last line of defence if portfolio
    construction ever proposes an over-weight position."""
    over = risk_config.max_single_name_pct + 0.30
    order = sizer.reconcile(
        _position(over),
        equity=1_000_000,
        price=250.0,
        stop_distance=0.01,
        available_cash=UNBOUNDED,
        liquidity_participation_cap=UNBOUNDED,
    )
    cap_quantity = math.floor(1_000_000 * risk_config.max_single_name_pct / 250.0)
    assert order.quantity <= cap_quantity
    assert order.binding_constraint == "max_single_name_pct"


def test_no_cap_can_ever_increase_the_quantity(sizer: PositionSizer) -> None:
    """The invariant behind every case above: each constraint is an upper
    bound, so the result satisfies all of them simultaneously."""
    order = sizer.reconcile(
        _position(0.12),
        equity=1_000_000,
        price=250.0,
        stop_distance=8.0,
        available_cash=40_000.0,
        liquidity_participation_cap=90,
    )
    assert order.quantity <= sizer.weight_based_quantity(_position(0.12), 1_000_000, 250.0)
    assert order.quantity <= sizer.risk_based_quantity(1_000_000, 250.0, 8.0)
    assert order.quantity <= math.floor(
        1_000_000 * sizer.config.max_single_name_pct / 250.0
    )
    assert order.quantity <= math.floor(40_000.0 / 250.0)
    assert order.quantity <= 90


def test_no_cash_means_no_order_rather_than_an_error(sizer: PositionSizer) -> None:
    """Zero is a legitimate answer -- the constraints leave no room -- and
    must be distinguishable from a rejection or a crash."""
    order = sizer.reconcile(
        _position(0.10),
        equity=1_000_000,
        price=250.0,
        stop_distance=10.0,
        available_cash=0.0,
        liquidity_participation_cap=UNBOUNDED,
    )
    assert order.quantity == 0
    assert order.binding_constraint == "available_cash"


def test_quantity_is_never_negative(sizer: PositionSizer) -> None:
    """Long-only, so a negative quantity is not a short -- it is a bug that
    would reach the broker as a sell of stock this system does not own."""
    order = sizer.reconcile(
        _position(0.01),
        equity=1_000_000,
        price=250.0,
        stop_distance=10.0,
        available_cash=0.0,
        liquidity_participation_cap=0,
    )
    assert order.quantity >= 0


def test_the_order_carries_its_own_provenance(sizer: PositionSizer) -> None:
    order = sizer.reconcile(
        _position(0.10, "NSE:INFY"),
        equity=1_000_000,
        price=250.0,
        stop_distance=10.0,
        available_cash=UNBOUNDED,
        liquidity_participation_cap=UNBOUNDED,
    )
    assert order.instrument_id == "NSE:INFY"
    assert order.target_weight == 0.10
    assert order.stop_distance == 10.0


# ---------------------------------------------------------------------------
# Broken inputs raise; they do not quietly size to zero
# ---------------------------------------------------------------------------


def test_a_zero_stop_distance_raises_rather_than_dividing(sizer: PositionSizer) -> None:
    """A missing risk-distance estimate is a broken caller, not a risk
    decision. Sizing off it would be sizing off nothing."""
    with pytest.raises(PositionSizingError, match="stop_distance"):
        sizer.risk_based_quantity(1_000_000, entry_price=250.0, stop_distance=0.0)


@pytest.mark.parametrize(
    ("equity", "price"),
    [(0.0, 250.0), (-1.0, 250.0), (1_000_000.0, 0.0), (1_000_000.0, -5.0)],
)
def test_nonsensical_equity_or_price_raises(
    sizer: PositionSizer, equity: float, price: float
) -> None:
    """Returning zero here would make a programming error look like a
    legitimately binding constraint, and it would look identical on the
    report to "no cash"."""
    with pytest.raises(PositionSizingError):
        sizer.reconcile(
            _position(0.10),
            equity=equity,
            price=price,
            stop_distance=10.0,
            available_cash=UNBOUNDED,
            liquidity_participation_cap=UNBOUNDED,
        )


def test_a_weight_above_one_cannot_even_be_constructed() -> None:
    """A weight above 1.0 is leverage, which V1 does not do at any layer.

    The enforcement that matters is in ``TargetPosition.__post_init__``:
    the illegal value is unrepresentable, so it cannot reach the sizer.
    ``PositionSizer.weight_based_quantity`` re-checks anyway as defence in
    depth, but this asserts the guarantee at the point it is actually
    made.
    """
    with pytest.raises(ValueError, match="target_weight"):
        _position(1.5)


def test_negative_cash_raises(sizer: PositionSizer) -> None:
    with pytest.raises(PositionSizingError, match="available_cash"):
        sizer.reconcile(
            _position(0.10),
            equity=1_000_000,
            price=250.0,
            stop_distance=10.0,
            available_cash=-1.0,
            liquidity_participation_cap=UNBOUNDED,
        )


# ---------------------------------------------------------------------------
# What this module deliberately does not do
# ---------------------------------------------------------------------------


def test_sector_exposure_is_not_claimed_to_be_enforced_here() -> None:
    """Sector limits are a property of the portfolio, not of one order, and
    cannot be evaluated from a single TargetPosition. Pretending otherwise
    would give a false assurance -- so ``reconcile`` never reports a sector
    constraint as binding, and the docstring says where it is enforced.
    """
    import inspect

    source = inspect.getsource(PositionSizer.reconcile)
    assert "max_sector_pct" not in source
    assert "portfolio_constructor" in source or "risk_manager" in source
