"""Tests for risk/stop_loss.py -- the two per-position exit rules.

The cases that matter here are not "does 3% arithmetic work". They are the
three places the rules could quietly be wrong: the trailing stop arming on a
profit nobody could have taken, the trailing stop reading a daily bar in an
order the bar does not record, and either rule reporting a fill better than
the market offered.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from risk.stop_loss import (
    StopBreach,
    StopLossError,
    StopLossPolicy,
    StopReason,
    evaluate,
    hard_stop_level,
    net_profit_pct,
    stop_fill_price,
    trailing_stop_level,
)

POLICY = StopLossPolicy(hard_stop_pct=0.03, trail_drop_pct=0.02, trail_arm_net_profit_pct=0.03)
"""The shipped setting: reaching +3% net closes the position outright."""

TRAIL_POLICY = StopLossPolicy(
    hard_stop_pct=0.03,
    trail_drop_pct=0.02,
    trail_arm_net_profit_pct=0.03,
    close_on_arm=False,
)
"""The alternative: +3% net arms a trailing stop that waits for a 2% pullback
off the session high, so a name still running is still held."""

QUANTITY = 10
SELL_COST_PCT = 0.0012
"""STT + exchange + SEBI + GST on a delivery sell, near enough for a test."""

DP_CHARGE = 15.93
"""Flat per-scrip sell charge. On a small position this alone is ~15bps, which
is the whole reason the arming test is computed net rather than gross."""


@pytest.mark.parametrize(
    "entry,expected", [(50.0, False), (99.99, False), (100.0, False), (100.01, True), (200.0, True)]
)
def test_hard_stop_purchase_price_exemption(entry: float, expected: bool) -> None:
    policy = replace(TRAIL_POLICY, hard_stop_min_entry_price=100.0)
    breach = evaluate(
        "NSE:TEST",
        entry_price=entry,
        cost_basis=entry * 10,
        session_high=entry,
        session_low=entry * 0.90,
        reference_price=entry * 0.95,
        net_sale_value=lambda p: p * 10,
        policy=policy,
    )
    assert (breach is not None) is expected
    if breach:
        assert breach.reason is StopReason.HARD_STOP


def test_low_purchase_price_remains_eligible_for_profit_exit() -> None:
    breach = evaluate(
        "NSE:TEST",
        entry_price=50,
        cost_basis=500,
        session_high=60,
        session_low=45,
        reference_price=58,
        net_sale_value=lambda p: p * 10,
        policy=replace(TRAIL_POLICY, hard_stop_min_entry_price=100),
    )
    assert breach is not None
    assert breach.reason is StopReason.TRAILING_PROFIT_STOP


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf")])
def test_invalid_entry_price_threshold(threshold: float) -> None:
    with pytest.raises(StopLossError):
        replace(POLICY, hard_stop_min_entry_price=threshold)


def test_shipped_policy_stops_every_position_whatever_its_price() -> None:
    """The one rule applies to a Rs 22 stock exactly as to a Rs 2,200 one."""
    from config.loader import load_settings

    policy = StopLossPolicy.from_mapping(load_settings().risk.stop_loss.model_dump())
    assert policy.hard_stop_min_entry_price == 0


def net_sale(price: float, quantity: int = QUANTITY) -> float:
    return price * quantity * (1.0 - SELL_COST_PCT) - DP_CHARGE


def basis_for(entry_price: float, quantity: int = QUANTITY) -> float:
    """What the buy actually cost, buy-side charges included."""
    return entry_price * quantity * 1.0013


# -- levels -----------------------------------------------------------------


def test_hard_stop_level_is_measured_from_the_buy_price() -> None:
    assert hard_stop_level(100.0, 0.03) == pytest.approx(97.0)


def test_trailing_stop_level_is_measured_from_the_session_high() -> None:
    assert trailing_stop_level(120.0, 0.02) == pytest.approx(117.6)


def test_levels_reject_a_non_positive_reference_price() -> None:
    with pytest.raises(StopLossError):
        hard_stop_level(0.0, 0.03)
    with pytest.raises(StopLossError):
        trailing_stop_level(-1.0, 0.02)


# -- fills ------------------------------------------------------------------


def test_fill_is_the_level_when_the_session_closed_above_it() -> None:
    # The level traded and the stock recovered: the stop got its price.
    assert stop_fill_price(97.0, 99.0) == pytest.approx(97.0)


def test_fill_is_the_close_when_the_session_gapped_through_the_level() -> None:
    # 97 was never offered. Filling there would credit a price nobody could get.
    assert stop_fill_price(97.0, 92.0) == pytest.approx(92.0)


# -- the hard stop ----------------------------------------------------------


@pytest.mark.parametrize(
    "opening,closing,expected",
    [(200.0, 182.0, 194.0), (190.0, 198.0, 190.0), (194.0, 200.0, 194.0)],
)
def test_daily_hard_stop_uses_open_not_later_close(
    opening: float, closing: float, expected: float
) -> None:
    breach = evaluate(
        "NSE:TEST",
        entry_price=200,
        cost_basis=2000,
        session_open=opening,
        session_high=205,
        session_low=180,
        reference_price=closing,
        net_sale_value=lambda p: p * 10 - 20,
        policy=replace(TRAIL_POLICY, hard_stop_min_entry_price=100),
    )
    assert breach is not None
    assert breach.reason is StopReason.HARD_STOP
    assert breach.fill_price == expected
    assert breach.net_profit_pct == pytest.approx((expected * 10 - 20) / 2000 - 1)


def test_hard_stop_fires_on_an_intraday_low_even_if_the_close_recovers() -> None:
    """A stop is not a close-only rule: 96 traded, so the stop was hit."""
    breach = evaluate(
        "ACME",
        entry_price=100.0,
        cost_basis=basis_for(100.0),
        session_high=101.0,
        session_low=96.0,
        reference_price=99.5,
        net_sale_value=net_sale,
        policy=POLICY,
    )
    assert breach is not None
    assert breach.reason is StopReason.HARD_STOP
    assert breach.stop_level == pytest.approx(97.0)
    assert breach.fill_price == pytest.approx(97.0)
    assert breach.net_profit_pct < 0


def test_hard_stop_is_silent_one_tick_above_the_level() -> None:
    assert (
        evaluate(
            "ACME",
            entry_price=100.0,
            cost_basis=basis_for(100.0),
            session_high=101.0,
            session_low=97.01,
            reference_price=98.0,
            net_sale_value=net_sale,
            policy=POLICY,
        )
        is None
    )


def test_hard_stop_does_not_move_with_the_position() -> None:
    """The level is fixed at the buy price, so a name far ahead is not
    protected 3% below *today's* price -- it is protected 3% below what it
    cost. A position bought at 100 and trading at 140 does not stop out at
    135.8.

    Shown against the trailing policy because under the shipped one a
    position never gets to be 40% ahead: it is sold at 3%."""
    assert (
        evaluate(
            "ACME",
            entry_price=100.0,
            cost_basis=basis_for(100.0),
            session_high=141.0,
            session_low=135.0,
            reference_price=140.0,
            net_sale_value=net_sale,
            policy=TRAIL_POLICY,
        )
        is None
    )


# -- the trailing profit stop ----------------------------------------------


def test_trailing_stop_fires_on_a_pullback_from_a_profitable_high() -> None:
    breach = evaluate(
        "ACME",
        entry_price=100.0,
        cost_basis=basis_for(100.0),
        session_high=110.0,
        session_low=106.5,
        reference_price=107.0,
        net_sale_value=net_sale,
        policy=TRAIL_POLICY,
    )
    assert breach is not None
    assert breach.reason is StopReason.TRAILING_PROFIT_STOP
    assert breach.reference_price == pytest.approx(110.0)
    assert breach.stop_level == pytest.approx(107.8)
    # Closed below the level, so that -- not the level -- is the fill.
    assert breach.fill_price == pytest.approx(107.0)
    assert breach.net_profit_pct > POLICY.trail_arm_net_profit_pct


def test_trailing_stop_stays_disarmed_when_the_exit_would_not_clear_3_pct() -> None:
    """The high was 4% up; after a 2% pullback the sale nets under 2%. Arming
    on the high would book a 'profit-protecting' exit at a price that was
    never a 3% profit."""
    breach = evaluate(
        "ACME",
        entry_price=100.0,
        cost_basis=basis_for(100.0),
        session_high=104.0,
        session_low=101.0,
        reference_price=101.8,
        net_sale_value=net_sale,
        policy=TRAIL_POLICY,
    )
    assert breach is None


def test_trailing_stop_accounts_for_the_dp_charge() -> None:
    """The same price move arms the stop on a large position and not on a
    small one, because the flat DP charge is a much bigger share of a small
    sale. If the rule ignored it, the small position would be exited on a
    'profit' that the contract note turns into less than 3%."""
    entry, high, close = 100.0, 107.0, 104.8

    def small_sale(price: float) -> float:
        return net_sale(price, quantity=5)

    def large_sale(price: float) -> float:
        return net_sale(price, quantity=5_000)

    small = evaluate(
        "ACME",
        entry_price=entry,
        cost_basis=basis_for(entry, 5),
        session_high=high,
        session_low=104.0,
        reference_price=close,
        net_sale_value=small_sale,
        policy=TRAIL_POLICY,
    )
    large = evaluate(
        "ACME",
        entry_price=entry,
        cost_basis=basis_for(entry, 5_000),
        session_high=high,
        session_low=104.0,
        reference_price=close,
        net_sale_value=large_sale,
        policy=TRAIL_POLICY,
    )
    assert small is None
    assert large is not None
    assert large.reason is StopReason.TRAILING_PROFIT_STOP


def test_trailing_stop_ignores_the_session_low() -> None:
    """The look-ahead guard. The low is 3% below the high and deeply through
    the trail level, but the bar does not say the low came *after* the high --
    if it came first, the trail level at that moment was lower and the rule
    never fired. Only the close is guaranteed to follow the high, and this
    session closed above the level."""
    breach = evaluate(
        "ACME",
        entry_price=100.0,
        cost_basis=basis_for(100.0),
        session_high=110.0,
        session_low=106.0,
        reference_price=109.5,
        net_sale_value=net_sale,
        policy=TRAIL_POLICY,
    )
    assert breach is None


# -- interaction and guards -------------------------------------------------


def test_hard_stop_wins_when_both_rules_fire() -> None:
    """Opened up, made 115, collapsed to 95. Chronologically the profit rule
    sold first at a gain; reporting the hard stop understates the exit, which
    is the safe direction for a backtest."""
    breach = evaluate(
        "ACME",
        entry_price=100.0,
        cost_basis=basis_for(100.0),
        session_high=115.0,
        session_low=95.0,
        reference_price=95.5,
        net_sale_value=net_sale,
        policy=TRAIL_POLICY,
    )
    assert breach is not None
    assert breach.reason is StopReason.HARD_STOP


def test_a_disabled_policy_never_fires() -> None:
    disabled = StopLossPolicy(
        hard_stop_pct=0.03,
        trail_drop_pct=0.02,
        trail_arm_net_profit_pct=0.03,
        enabled=False,
    )
    assert (
        evaluate(
            "ACME",
            entry_price=100.0,
            cost_basis=basis_for(100.0),
            session_high=101.0,
            session_low=80.0,
            reference_price=81.0,
            net_sale_value=net_sale,
            policy=disabled,
        )
        is None
    )


def test_unusable_inputs_produce_no_breach_rather_than_a_guess() -> None:
    for kwargs in (
        {"entry_price": 0.0},
        {"cost_basis": 0.0},
        {"session_low": 0.0},
        {"session_high": 0.0},
        {"reference_price": 0.0},
    ):
        base = {
            "entry_price": 100.0,
            "cost_basis": basis_for(100.0),
            "session_high": 101.0,
            "session_low": 90.0,
            "reference_price": 91.0,
        }
        base.update(kwargs)
        assert evaluate("ACME", net_sale_value=net_sale, policy=POLICY, **base) is None


def test_an_impossible_bar_fails_loudly() -> None:
    """high < low is corrupt data, not a quiet no-breach: every other guard
    here is about a price that cannot be verified, but this one is a price
    that cannot exist."""
    with pytest.raises(StopLossError):
        evaluate(
            "ACME",
            entry_price=100.0,
            cost_basis=basis_for(100.0),
            session_high=90.0,
            session_low=95.0,
            reference_price=92.0,
            net_sale_value=net_sale,
            policy=POLICY,
        )


# -- profitability and config ----------------------------------------------


def test_net_profit_is_measured_against_what_the_position_cost() -> None:
    """A 3% price move is not a 3% profit: both legs' charges sit between."""
    entry = 100.0
    gross_move = 0.03
    profit = net_profit_pct(basis_for(entry), net_sale, entry * (1 + gross_move))
    assert profit < gross_move
    assert profit == pytest.approx(0.0114, abs=5e-4)


def test_policy_rejects_a_threshold_outside_zero_to_one() -> None:
    with pytest.raises(StopLossError):
        StopLossPolicy(hard_stop_pct=3.0, trail_drop_pct=0.02, trail_arm_net_profit_pct=0.03)
    with pytest.raises(StopLossError):
        StopLossPolicy(hard_stop_pct=0.03, trail_drop_pct=0.0, trail_arm_net_profit_pct=0.03)


def test_policy_from_mapping_requires_every_key() -> None:
    """No silent defaults: a stop at a threshold nobody chose reports
    protection that was never configured."""
    with pytest.raises(StopLossError, match="missing required keys"):
        StopLossPolicy.from_mapping({"enabled": True, "hard_stop_pct": 0.03})


def test_the_shipped_settings_are_the_one_rule() -> None:
    """A position leaves the book on one rule only: it trades more than 3%
    below the price it was BOUGHT at. Every position, whatever its price. No
    trailing stop and no take-profit -- a winner leaves only when the ranking
    drops it.

    Pinned here because changing it is a decision about the strategy, not a
    tweak. If this test fails, the rule was edited -- which is allowed, but it
    should be on purpose.
    """
    from config.loader import load_settings

    policy = StopLossPolicy.from_mapping(load_settings().risk.stop_loss.model_dump())

    assert policy.enabled is True
    assert policy.hard_stop_pct == pytest.approx(0.03)
    assert policy.hard_stop_min_entry_price == 0
    assert policy.profit_exit_enabled is False

def test_take_profit_closes_as_soon_as_the_sale_clears_the_threshold() -> None:
    """No pullback required and no session extreme involved: the price on the
    screen nets more than 3%, so the position is sold there."""
    breach = evaluate(
        "ACME",
        entry_price=100.0,
        cost_basis=basis_for(100.0),
        session_high=105.5,
        session_low=99.0,
        reference_price=105.5,
        net_sale_value=net_sale,
        policy=POLICY,
    )
    assert breach is not None
    assert breach.reason is StopReason.TAKE_PROFIT
    assert breach.fill_price == pytest.approx(105.5)
    assert breach.stop_level == pytest.approx(105.5)
    assert breach.net_profit_pct > POLICY.trail_arm_net_profit_pct


def test_take_profit_is_measured_net_so_a_3_pct_move_is_not_enough() -> None:
    """The whole point of the net test. Up 3.2% on the screen, and on a
    position large enough that the flat DP charge is negligible, STT on both
    legs plus the exchange and SEBI charges, GST and stamp duty still leave
    2.94% in the account. Under the threshold, so the position is held."""
    quantity = 5_000

    def large_sale(price: float) -> float:
        return net_sale(price, quantity=quantity)

    breach = evaluate(
        "ACME",
        entry_price=100.0,
        cost_basis=basis_for(100.0, quantity),
        session_high=103.2,
        session_low=99.5,
        reference_price=103.2,
        net_sale_value=large_sale,
        policy=POLICY,
    )
    assert breach is None
    assert net_profit_pct(basis_for(100.0, quantity), large_sale, 103.2) == pytest.approx(
        0.0294, abs=5e-4
    )


def test_take_profit_does_not_fire_on_a_high_the_sale_missed() -> None:
    """Touched 110 intraday and came back to 101. A backtest passes the close
    as the reference price, and the rule sells at the price it is given --
    it never books a profit at a high the exit could not reach."""
    breach = evaluate(
        "ACME",
        entry_price=100.0,
        cost_basis=basis_for(100.0),
        session_high=110.0,
        session_low=99.0,
        reference_price=101.0,
        net_sale_value=net_sale,
        policy=POLICY,
    )
    assert breach is None


def test_the_hard_stop_still_wins_over_a_take_profit_on_the_same_session() -> None:
    """Ran to 115, collapsed to 95. The take profit would have sold first;
    reporting the hard stop understates the exit, which is the safe
    direction."""
    breach = evaluate(
        "ACME",
        entry_price=100.0,
        cost_basis=basis_for(100.0),
        session_high=115.0,
        session_low=95.0,
        reference_price=95.5,
        net_sale_value=net_sale,
        policy=POLICY,
    )
    assert breach is not None
    assert breach.reason is StopReason.HARD_STOP


def test_the_two_profit_policies_disagree_on_the_same_session() -> None:
    """The reason this is configuration. Up 4% and still at its high: the
    take profit banks it, the trailing stop holds on for more."""

    def at(policy: StopLossPolicy) -> StopBreach | None:
        return evaluate(
            "ACME",
            entry_price=100.0,
            cost_basis=basis_for(100.0),
            session_high=105.5,
            session_low=99.8,
            reference_price=105.5,
            net_sale_value=net_sale,
            policy=policy,
        )

    banked = at(POLICY)
    held = at(TRAIL_POLICY)

    assert banked is not None and banked.reason is StopReason.TAKE_PROFIT
    assert held is None


def test_close_on_arm_is_required_in_config() -> None:
    with pytest.raises(StopLossError, match="missing required keys"):
        StopLossPolicy.from_mapping(
            {
                "enabled": True,
                "hard_stop_pct": 0.03,
                "trail_drop_pct": 0.02,
                "trail_arm_net_profit_pct": 0.03,
            }
        )


def test_the_master_rule_holds_a_winner_that_has_not_pulled_back() -> None:
    """The point of arming rather than closing. Up 7% net and still at its
    high: nothing fires, and the position keeps running."""
    master = StopLossPolicy(
        hard_stop_pct=0.03,
        trail_drop_pct=0.02,
        trail_arm_net_profit_pct=0.05,
        close_on_arm=False,
    )
    assert (
        evaluate(
            "ACME",
            entry_price=100.0,
            cost_basis=basis_for(100.0),
            session_high=109.0,
            session_low=101.0,
            reference_price=109.0,
            net_sale_value=net_sale,
            policy=master,
        )
        is None
    )


def test_the_master_rule_needs_five_percent_not_three() -> None:
    """A 2% pullback from a high that only ever showed a 4% net gain does
    nothing: the trail is disarmed below the threshold, and the hard stop is
    the only thing live until the position is properly ahead."""
    master = StopLossPolicy(
        hard_stop_pct=0.03,
        trail_drop_pct=0.02,
        trail_arm_net_profit_pct=0.05,
        close_on_arm=False,
    )
    # High 106.5, pulled back to 104.3 -- through the trail level, netting
    # about 2.5%, which is under 5%.
    assert (
        evaluate(
            "ACME",
            entry_price=100.0,
            cost_basis=basis_for(100.0),
            session_high=106.5,
            session_low=104.0,
            reference_price=104.3,
            net_sale_value=net_sale,
            policy=master,
        )
        is None
    )


def test_the_master_rule_banks_a_five_percent_winner_on_a_two_percent_pullback() -> None:
    quantity = 5_000

    def large_sale(price: float) -> float:
        return net_sale(price, quantity=quantity)

    master = StopLossPolicy(
        hard_stop_pct=0.03,
        trail_drop_pct=0.02,
        trail_arm_net_profit_pct=0.05,
        close_on_arm=False,
    )
    # Ran to 112, fell to 109.5 -- below 112 * 0.98, and netting about 9%.
    breach = evaluate(
        "ACME",
        entry_price=100.0,
        cost_basis=basis_for(100.0, quantity),
        session_high=112.0,
        session_low=109.0,
        reference_price=109.5,
        net_sale_value=large_sale,
        policy=master,
    )

    assert breach is not None
    assert breach.reason is StopReason.TRAILING_PROFIT_STOP
    assert breach.net_profit_pct > 0.05


def test_with_profit_exits_off_a_pullback_from_a_big_gain_does_nothing() -> None:
    """Up 12% and pulled back 2.3% off the high -- the trailing rule would
    sell. With profit exits off, nothing fires: the hard stop is the only
    rule, and this position is nowhere near it."""
    policy = replace(TRAIL_POLICY, profit_exit_enabled=False)
    assert (
        evaluate(
            "ACME",
            entry_price=100.0,
            cost_basis=basis_for(100.0),
            session_high=112.0,
            session_low=109.0,
            reference_price=109.4,
            net_sale_value=net_sale,
            policy=policy,
        )
        is None
    )


def test_with_profit_exits_off_the_hard_stop_still_fires() -> None:
    """Switching off the profit exit must not switch off the loss exit."""
    policy = replace(TRAIL_POLICY, profit_exit_enabled=False)
    breach = evaluate(
        "ACME",
        entry_price=100.0,
        cost_basis=basis_for(100.0),
        session_high=100.5,
        session_low=96.0,
        reference_price=96.5,
        net_sale_value=net_sale,
        policy=policy,
    )
    assert breach is not None
    assert breach.reason is StopReason.HARD_STOP


def test_the_hard_stop_protects_a_cheap_stock_when_the_floor_is_zero() -> None:
    """SBC traded near Rs 22. Under the old Rs 100 floor it had no stop."""
    policy = replace(TRAIL_POLICY, profit_exit_enabled=False, hard_stop_min_entry_price=0.0)
    breach = evaluate(
        "SBC",
        entry_price=22.0,
        cost_basis=basis_for(22.0),
        session_high=22.1,
        session_low=21.3,
        reference_price=21.5,
        net_sale_value=net_sale,
        policy=policy,
    )
    assert breach is not None
    assert breach.reason is StopReason.HARD_STOP


def test_profit_exit_enabled_defaults_on_for_older_configs() -> None:
    """A config written before the switch existed keeps its old behaviour."""
    policy = StopLossPolicy.from_mapping(
        {
            "enabled": True,
            "hard_stop_pct": 0.03,
            "trail_drop_pct": 0.02,
            "trail_arm_net_profit_pct": 0.05,
            "close_on_arm": False,
        }
    )
    assert policy.profit_exit_enabled is True
