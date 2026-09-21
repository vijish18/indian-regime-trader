"""Tests for execution/paper_book.py -- the paper account's ledger.

The things that matter here are the ones a snapshot-based "live book" could
never get wrong because it never tried: that cash released by a stop is
really credited, that it is spent only on names the account may hold, that a
stopped-out name is not immediately bought back, and that realised P&L is the
difference between two cash amounts rather than a price ratio.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from backtest.cost_schedule import CostScheduleRepository
from backtest.costs import CostModel
from execution.paper_book import ClosedTrade, PaperBook, PaperBookError
from risk.stop_loss import StopLossPolicy

POLICY = StopLossPolicy(
    hard_stop_pct=0.03, trail_drop_pct=0.02, trail_arm_net_profit_pct=0.03
)
"""The shipped setting: +3% net closes the position outright."""

TRAIL_POLICY = StopLossPolicy(
    hard_stop_pct=0.03,
    trail_drop_pct=0.02,
    trail_arm_net_profit_pct=0.03,
    close_on_arm=False,
)
TODAY = dt.date(2026, 9, 21)
SEED_DATE = TODAY - dt.timedelta(days=4)
"""The fixture book is opened from a previous session's selection, which is
how a real one starts: the ranking is computed at a close and the account is
sized against it. Positions are therefore held *through* the session under
test, so the session's own high and low are prices they really traded at."""


@pytest.fixture
def costs() -> CostModel:
    return CostModel(
        CostScheduleRepository.from_file(
            Path(__file__).resolve().parents[2] / "config" / "cost_schedules.yaml"
        ),
        min_slippage_bps=5.0,
        impact_coefficient=50.0,
    )


def quote(last: float, *, high: float | None = None, low: float | None = None) -> dict:
    return {
        "last": last,
        "day_high": high if high is not None else last,
        "day_low": low if low is not None else last,
    }


def seeded(costs: CostModel, **overrides) -> PaperBook:
    """A two-name book: ALPHA at 100, BETA at 200, out of a 100k budget,
    opened four sessions ago."""
    sizing = [
        {"symbol": "ALPHA", "rank": 1, "price": 100.0, "shares": 100},
        {"symbol": "BETA", "rank": 2, "price": 200.0, "shares": 50},
    ]
    picks = [
        {"symbol": "ALPHA", "instrument_id": "NSE:ALPHA"},
        {"symbol": "BETA", "instrument_id": "NSE:BETA"},
    ]
    kwargs = dict(
        budget=100_000.0,
        sizing_positions=sizing,
        picks=picks,
        costs=costs,
        as_of=SEED_DATE,
    )
    kwargs.update(overrides)
    return PaperBook.seed(**kwargs)  # type: ignore[arg-type]


# -- opening the book -------------------------------------------------------


def test_seeding_pays_for_the_positions_out_of_cash(costs: CostModel) -> None:
    book = seeded(costs)

    assert set(book.positions) == {"NSE:ALPHA", "NSE:BETA"}
    spent = sum(p.entry_cost for p in book.positions.values())
    assert book.cash == pytest.approx(100_000.0 - spent)
    # 10,000 + 10,000 of stock, plus charges on both -- never exactly 20,000.
    assert spent > 20_000.0


def test_entry_cost_is_cash_paid_not_shares_times_price(costs: CostModel) -> None:
    """A position is not ahead until it has earned back what it cost to
    open, so the basis has to include the buy leg's charges."""
    book = seeded(costs)
    alpha = book.positions["NSE:ALPHA"]

    assert alpha.entry_price == pytest.approx(100.0)
    assert alpha.entry_cost > alpha.shares * alpha.entry_price


def test_a_zero_share_sizing_row_is_not_a_position(costs: CostModel) -> None:
    """Unbuyable names (one share costs more than the slot) size to zero
    shares. They are not holdings and must not consume cash."""
    book = PaperBook.seed(
        budget=100_000.0,
        sizing_positions=[{"symbol": "PRICEY", "rank": 1, "price": 47_710.0, "shares": 0}],
        picks=[{"symbol": "PRICEY", "instrument_id": "NSE:PRICEY"}],
        costs=costs,
        as_of=TODAY,
    )

    assert book.positions == {}
    assert book.cash == pytest.approx(100_000.0)


# -- stops closing positions ------------------------------------------------


def test_a_hard_stop_closes_the_position_and_credits_the_cash(costs: CostModel) -> None:
    book = seeded(costs)
    cash_before = book.cash
    quotes = {
        "NSE:ALPHA": quote(96.5, high=100.5, low=95.0),  # through 97
        "NSE:BETA": quote(201.0, high=202.0, low=200.0),
    }

    closed = book.apply_stops(quotes, policy=POLICY, costs=costs, today=TODAY)

    assert [t.symbol for t in closed] == ["ALPHA"]
    assert "NSE:ALPHA" not in book.positions
    assert "NSE:BETA" in book.positions
    assert book.cash == pytest.approx(cash_before + closed[0].net_proceeds)
    assert closed[0].exit_reason == "hard_stop"
    assert closed[0].net_pnl < 0


def test_the_exit_fills_at_the_worse_of_the_level_and_the_screen(
    costs: CostModel,
) -> None:
    """Gapped well below 97: the level was never on offer, so filling there
    would credit the account with a price it could not have got."""
    book = seeded(costs)
    quotes = {"NSE:ALPHA": quote(92.0, high=99.0, low=91.0)}

    closed = book.apply_stops(quotes, policy=POLICY, costs=costs, today=TODAY)

    assert closed[0].exit_price == pytest.approx(92.0)


def test_a_take_profit_closes_a_winner_and_books_the_profit(
    costs: CostModel,
) -> None:
    """The shipped rule: reaching the threshold sells, no pullback needed."""
    book = seeded(costs)
    quotes = {"NSE:ALPHA": quote(107.0, high=110.0, low=106.0)}

    closed = book.apply_stops(quotes, policy=POLICY, costs=costs, today=TODAY)

    assert [t.exit_reason for t in closed] == ["take_profit"]
    assert closed[0].exit_price == pytest.approx(107.0)
    assert closed[0].net_pnl > 0
    assert closed[0].net_pnl_pct > POLICY.trail_arm_net_profit_pct


def test_a_trailing_stop_closes_a_winner_and_books_the_profit(
    costs: CostModel,
) -> None:
    """The alternative rule, still supported: wait for the 2% pullback."""
    book = seeded(costs)
    quotes = {"NSE:ALPHA": quote(107.0, high=110.0, low=106.0)}

    closed = book.apply_stops(quotes, policy=TRAIL_POLICY, costs=costs, today=TODAY)

    assert [t.exit_reason for t in closed] == ["trailing_profit_stop"]
    assert closed[0].net_pnl > 0
    assert closed[0].net_pnl_pct > POLICY.trail_arm_net_profit_pct


def test_a_holding_with_no_quote_is_left_alone(costs: CostModel) -> None:
    book = seeded(costs)

    closed = book.apply_stops({}, policy=POLICY, costs=costs, today=TODAY)

    assert closed == []
    assert len(book.positions) == 2


def test_realized_pnl_is_cash_in_minus_cash_out(costs: CostModel) -> None:
    """Not a price ratio. Both legs' charges sit between the two, and the
    gap between gross and net is the cost of the round trip."""
    book = seeded(costs)
    quotes = {"NSE:ALPHA": quote(107.0, high=110.0, low=106.0)}

    trade = book.apply_stops(quotes, policy=POLICY, costs=costs, today=TODAY)[0]

    assert trade.net_pnl == pytest.approx(trade.net_proceeds - trade.entry_cost)
    assert trade.gross_pnl == pytest.approx(
        (trade.exit_price - trade.entry_price) * trade.shares
    )
    assert trade.costs == pytest.approx(trade.gross_pnl - trade.net_pnl)
    assert trade.costs > 0


# -- redeploying the cash ---------------------------------------------------


BENCH = [
    {"rank": 3, "symbol": "GAMMA", "instrument_id": "NSE:GAMMA"},
    {"rank": 4, "symbol": "DELTA", "instrument_id": "NSE:DELTA"},
]


def test_freed_cash_buys_the_best_candidate_not_already_held(
    costs: CostModel,
) -> None:
    book = seeded(costs)
    quotes = {
        "NSE:ALPHA": quote(96.0, high=100.0, low=95.0),
        "NSE:GAMMA": quote(50.0),
        "NSE:DELTA": quote(25.0),
    }
    book.apply_stops(quotes, policy=POLICY, costs=costs, today=TODAY)
    cash_before = book.cash

    opened = book.reallocate(
        BENCH, quotes, costs=costs, today=TODAY, slot=10_000.0, max_positions=10
    )

    assert [p.symbol for p in opened] == ["GAMMA", "DELTA"]
    assert book.cash < cash_before
    assert book.cash >= 0
    assert opened[0].source == "reallocation"
    assert opened[0].entry_price == pytest.approx(50.0)


def test_a_name_stopped_out_today_is_not_bought_back_today(
    costs: CostModel,
) -> None:
    """Otherwise a falling stock is re-entered a few paise below its own
    stop, fires again, and grinds the account into transaction costs."""
    book = seeded(costs)
    quotes = {
        "NSE:ALPHA": quote(96.0, high=100.0, low=95.0),
        "NSE:GAMMA": quote(50.0),
    }
    book.apply_stops(quotes, policy=POLICY, costs=costs, today=TODAY)
    bench_with_alpha = [
        {"rank": 1, "symbol": "ALPHA", "instrument_id": "NSE:ALPHA"},
        *BENCH,
    ]

    opened = book.reallocate(
        bench_with_alpha, quotes, costs=costs, today=TODAY, slot=10_000.0, max_positions=10
    )

    assert "NSE:ALPHA" not in {p.instrument_id for p in opened}
    assert "NSE:ALPHA" not in book.positions


def test_the_block_lifts_at_the_next_session(costs: CostModel) -> None:
    book = seeded(costs)
    book.apply_stops(
        {"NSE:ALPHA": quote(96.0, high=100.0, low=95.0)},
        policy=POLICY,
        costs=costs,
        today=TODAY,
    )
    assert "NSE:ALPHA" in book.blocked_today

    book.roll_session(TODAY + dt.timedelta(days=1))

    assert book.blocked_today == {}


def test_a_name_already_held_is_never_bought_twice(costs: CostModel) -> None:
    book = seeded(costs)
    quotes = {"NSE:ALPHA": quote(100.0), "NSE:BETA": quote(200.0)}
    bench = [{"rank": 1, "symbol": "ALPHA", "instrument_id": "NSE:ALPHA"}]

    opened = book.reallocate(
        bench, quotes, costs=costs, today=TODAY, slot=10_000.0, max_positions=10
    )

    assert opened == []
    assert book.positions["NSE:ALPHA"].shares == 100


def test_reallocation_never_overdraws_the_account(costs: CostModel) -> None:
    """There is no margin here. Charges can push a bill that fits on price
    alone past the free cash, and the order is trimmed rather than the
    balance going negative."""
    book = seeded(costs)
    book.cash = 1_000.0
    quotes = {"NSE:GAMMA": quote(999.0)}

    book.reallocate(
        [BENCH[0]], quotes, costs=costs, today=TODAY, slot=10_000.0, max_positions=10
    )

    assert book.cash >= 0


def test_reallocation_respects_the_position_cap(costs: CostModel) -> None:
    book = seeded(costs)
    quotes = {"NSE:GAMMA": quote(50.0), "NSE:DELTA": quote(25.0)}

    opened = book.reallocate(
        BENCH, quotes, costs=costs, today=TODAY, slot=10_000.0, max_positions=2
    )

    assert opened == []
    assert len(book.positions) == 2


def test_a_new_position_is_capped_at_the_slot_not_the_whole_balance(
    costs: CostModel,
) -> None:
    """A replacement must not take a share of the account no ranking asked
    for, even when a stop has just released a lot of cash."""
    book = seeded(costs)
    book.cash = 60_000.0
    quotes = {"NSE:GAMMA": quote(100.0)}

    opened = book.reallocate(
        [BENCH[0]], quotes, costs=costs, today=TODAY, slot=10_000.0, max_positions=10
    )

    assert opened[0].shares <= 100
    assert opened[0].entry_cost < 10_100.0


# -- reporting --------------------------------------------------------------


def test_realized_summary_counts_wins_losses_and_cost_drag(costs: CostModel) -> None:
    book = seeded(costs)
    book.apply_stops(
        {
            "NSE:ALPHA": quote(107.0, high=110.0, low=106.0),
            "NSE:BETA": quote(193.0, high=201.0, low=192.0),
        },
        policy=POLICY,
        costs=costs,
        today=TODAY,
    )

    summary = book.realized()

    assert summary["available"] is True
    assert summary["trades"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["costs"] > 0
    assert set(summary["by_reason"]) == {"hard_stop", "take_profit"}
    assert len(summary["history"]) == 2
    # Newest first.
    assert summary["history"][0]["symbol"] == "BETA"


def test_an_empty_book_reports_nothing_rather_than_zeroes_that_look_real(
    costs: CostModel,
) -> None:
    summary = seeded(costs).realized()

    assert summary["available"] is False
    assert summary["trades"] == 0
    assert summary["profit_factor"] is None


# -- persistence ------------------------------------------------------------


def test_the_book_round_trips_through_disk(costs: CostModel, tmp_path: Path) -> None:
    book = seeded(costs)
    book.apply_stops(
        {"NSE:ALPHA": quote(96.0, high=100.0, low=95.0)},
        policy=POLICY,
        costs=costs,
        today=TODAY,
    )
    path = tmp_path / "paper_book.json"
    book.save(path)

    restored = PaperBook.load(path)

    assert restored is not None
    assert restored.cash == pytest.approx(book.cash)
    assert set(restored.positions) == set(book.positions)
    assert [t.symbol for t in restored.closed] == [t.symbol for t in book.closed]
    assert restored.blocked_today == book.blocked_today


def test_a_missing_book_is_absence_not_an_empty_account(tmp_path: Path) -> None:
    assert PaperBook.load(tmp_path / "nothing.json") is None


def test_a_corrupt_book_raises_rather_than_starting_again_at_full_cash(
    tmp_path: Path,
) -> None:
    """Silently replacing an unreadable ledger with a fresh one would erase
    the account's history and look exactly like a flat day."""
    path = tmp_path / "paper_book.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(PaperBookError):
        PaperBook.load(path)


def test_the_saved_book_is_valid_json_for_a_browser(
    costs: CostModel, tmp_path: Path
) -> None:
    """Python writes NaN as a bare token that json.loads accepts and every
    browser's JSON.parse rejects. The page's parse is wrapped in a try/catch
    that falls back to an empty payload, so one NaN blanks the dashboard."""
    book = seeded(costs)
    path = tmp_path / "paper_book.json"
    book.save(path)

    def reject(constant: str) -> float:
        raise AssertionError(f"non-finite {constant} in the saved book")

    json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def test_holding_days_survives_a_trade_written_by_an_older_version() -> None:
    trade = ClosedTrade.from_dict(
        {
            "symbol": "ALPHA",
            "instrument_id": "NSE:ALPHA",
            "shares": 10,
            "entry_price": 100.0,
            "entry_cost": 1_001.0,
            "entry_date": "2026-09-15",
            "exit_price": 97.0,
            "exit_date": "2026-09-21",
            "exit_reason": "hard_stop",
            "net_proceeds": 968.0,
        }
    )

    assert trade.holding_days == 6
    assert trade.stop_level == 0.0


# -- prices from before the position existed --------------------------------


def test_a_position_bought_today_ignores_the_session_low_before_entry(
    costs: CostModel,
) -> None:
    """The bug this suite did not have. SKYGOLD was bought at 796.45 at
    15:05 and 'stopped out' six minutes later at 772.56 -- the session's low,
    printed hours before the account owned a share. A stop must only see
    prices from while the position was held."""
    book = seeded(costs)
    book.cash = 20_000.0
    bench = [{"rank": 3, "symbol": "GAMMA", "instrument_id": "NSE:GAMMA"}]
    # Bought at 800 while the day's low, set in the morning, was 770.
    buy_quotes = {"NSE:GAMMA": quote(800.0, high=810.0, low=770.0)}
    opened = book.reallocate(
        bench, buy_quotes, costs=costs, today=TODAY, slot=10_000.0, max_positions=10
    )
    assert opened and opened[0].symbol == "GAMMA"

    closed = book.apply_stops(buy_quotes, policy=POLICY, costs=costs, today=TODAY)

    assert closed == []
    assert "NSE:GAMMA" in book.positions


def test_a_position_bought_today_still_stops_on_a_fall_after_entry(
    costs: CostModel,
) -> None:
    """The guard must not disable the stop -- only anchor it. Bought at 800,
    the price then falls to 775, which is below 776 and is a price the
    position really experienced."""
    book = seeded(costs)
    book.cash = 20_000.0
    bench = [{"rank": 3, "symbol": "GAMMA", "instrument_id": "NSE:GAMMA"}]
    book.reallocate(
        bench,
        {"NSE:GAMMA": quote(800.0, high=810.0, low=770.0)},
        costs=costs,
        today=TODAY,
        slot=10_000.0,
        max_positions=10,
    )

    closed = book.apply_stops(
        {"NSE:GAMMA": quote(775.0, high=810.0, low=770.0)},
        policy=POLICY,
        costs=costs,
        today=TODAY,
    )

    assert [t.symbol for t in closed] == ["GAMMA"]
    assert closed[0].exit_reason == "hard_stop"


def test_a_position_held_from_a_previous_day_does_use_the_session_low(
    costs: CostModel,
) -> None:
    """The other half. A name carried in overnight was held through the whole
    session, so the session's low is a price it really traded at while owned
    -- even if it has since recovered."""
    book = seeded(costs)  # opened four sessions ago, so held all day

    closed = book.apply_stops(
        {"NSE:BETA": quote(199.0, high=201.0, low=190.0)},
        policy=POLICY,
        costs=costs,
        today=TODAY,
    )

    assert [t.symbol for t in closed] == ["BETA"]
    assert closed[0].exit_reason == "hard_stop"


def test_the_trailing_high_for_a_same_day_entry_starts_at_the_entry_price(
    costs: CostModel,
) -> None:
    """A morning spike the position missed must not arm a trailing stop
    against it. Bought at 800 after the day high of 900; a fall to 790 is
    2% below neither 800 nor anything the position saw."""
    book = seeded(costs)
    book.cash = 20_000.0
    book.reallocate(
        [{"rank": 3, "symbol": "GAMMA", "instrument_id": "NSE:GAMMA"}],
        {"NSE:GAMMA": quote(800.0, high=900.0, low=790.0)},
        costs=costs,
        today=TODAY,
        slot=10_000.0,
        max_positions=10,
    )

    closed = book.apply_stops(
        {"NSE:GAMMA": quote(795.0, high=900.0, low=790.0)},
        policy=POLICY,
        costs=costs,
        today=TODAY,
    )

    assert closed == []


def test_the_held_extremes_survive_a_save_and_reload(
    costs: CostModel, tmp_path: Path
) -> None:
    """They are the stop's memory between refreshes. Lost on reload, a
    mid-session entry would reset its anchor to the entry price every few
    minutes and never trail."""
    book = seeded(costs)
    book.apply_stops(
        {"NSE:BETA": quote(205.0, high=206.0, low=204.0)},
        policy=POLICY,
        costs=costs,
        today=TODAY,
    )
    path = tmp_path / "book.json"
    book.save(path)

    restored = PaperBook.load(path)

    assert restored is not None
    beta = restored.positions["NSE:BETA"]
    assert beta.session_date == TODAY.isoformat()
    # Held since before today, so the session high applies in full.
    assert beta.session_high == pytest.approx(206.0)
