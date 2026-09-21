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


def quote(
    last: float, *, high: float | None = None, low: float | None = None
) -> dict[str, float]:
    return {
        "last": last,
        "day_high": high if high is not None else last,
        "day_low": low if low is not None else last,
    }


def seeded(costs: CostModel, **overrides: object) -> PaperBook:
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


# -- only names the strategy actually ranks ---------------------------------


RANKED = [
    {"rank": 3, "symbol": "GAMMA", "instrument_id": "NSE:GAMMA"},
    {"rank": 11, "symbol": "ELEVEN", "instrument_id": "NSE:ELEVEN"},
    {"rank": 12, "symbol": "TWELVE", "instrument_id": "NSE:TWELVE"},
]


def test_a_candidate_below_the_rank_cap_is_never_bought(costs: CostModel) -> None:
    """Rank 11 is a name the selector considered and did not choose. The
    account holds what the strategy picked, or it holds cash."""
    book = seeded(costs)
    book.cash = 40_000.0
    quotes = {
        "NSE:ELEVEN": quote(100.0),
        "NSE:TWELVE": quote(100.0),
    }

    opened = book.reallocate(
        RANKED[1:],
        quotes,
        costs=costs,
        today=TODAY,
        slot=10_000.0,
        max_positions=10,
        max_rank=10,
    )

    assert opened == []
    assert book.cash == pytest.approx(40_000.0)


def test_cash_waits_rather_than_reaching_past_the_cap(costs: CostModel) -> None:
    """The ranked name is taken and the unranked ones are left, even though
    there is cash for all three."""
    book = seeded(costs)
    book.cash = 40_000.0
    quotes = {
        "NSE:GAMMA": quote(100.0),
        "NSE:ELEVEN": quote(100.0),
        "NSE:TWELVE": quote(100.0),
    }

    opened = book.reallocate(
        RANKED, quotes, costs=costs, today=TODAY, slot=10_000.0, max_positions=10, max_rank=10
    )

    assert [p.symbol for p in opened] == ["GAMMA"]
    assert book.cash > 29_000.0


def test_a_candidate_with_no_rank_is_refused_under_a_cap(costs: CostModel) -> None:
    """An unranked row cannot be shown to satisfy the cap, so it does not."""
    book = seeded(costs)
    book.cash = 20_000.0

    opened = book.reallocate(
        [{"symbol": "NORANK", "instrument_id": "NSE:NORANK"}],
        {"NSE:NORANK": quote(100.0)},
        costs=costs,
        today=TODAY,
        slot=10_000.0,
        max_positions=10,
        max_rank=10,
    )

    assert opened == []


def test_without_a_cap_any_candidate_is_eligible(costs: CostModel) -> None:
    """The cap is a policy, not a hard-wired rule -- a backtest comparing
    with and against it needs the other side to exist."""
    book = seeded(costs)
    book.cash = 20_000.0

    opened = book.reallocate(
        RANKED[1:],
        {"NSE:ELEVEN": quote(100.0), "NSE:TWELVE": quote(100.0)},
        costs=costs,
        today=TODAY,
        slot=10_000.0,
        max_positions=10,
        max_rank=None,
    )

    assert [p.symbol for p in opened] == ["ELEVEN", "TWELVE"]


# -- asking for a new ranking ----------------------------------------------


NOW = dt.datetime(2026, 9, 21, 11, 0)


def test_a_full_book_does_not_ask_for_a_new_ranking(costs: CostModel) -> None:
    book = seeded(costs)  # two positions

    assert not book.needs_rerank(at_or_below=1, now=NOW, cooldown_minutes=45)


def test_a_book_run_down_to_the_threshold_asks_for_one(costs: CostModel) -> None:
    book = seeded(costs)

    assert book.needs_rerank(at_or_below=2, now=NOW, cooldown_minutes=45)


def test_the_cooldown_stops_it_asking_again_immediately(costs: CostModel) -> None:
    """Recomputing runs the selector over real history. A book that cannot be
    refilled -- every ranked name held or blocked -- would otherwise ask on
    every refresh, which is every few minutes."""
    book = seeded(costs)
    book.last_rerank_at = (NOW - dt.timedelta(minutes=10)).isoformat()

    assert not book.needs_rerank(at_or_below=2, now=NOW, cooldown_minutes=45)


def test_the_cooldown_expires(costs: CostModel) -> None:
    book = seeded(costs)
    book.last_rerank_at = (NOW - dt.timedelta(minutes=60)).isoformat()

    assert book.needs_rerank(at_or_below=2, now=NOW, cooldown_minutes=45)


def test_an_unreadable_rerank_timestamp_allows_one_rather_than_blocking(
    costs: CostModel,
) -> None:
    """Failing closed here would mean never recomputing again."""
    book = seeded(costs)
    book.last_rerank_at = "not a timestamp"

    assert book.needs_rerank(at_or_below=2, now=NOW, cooldown_minutes=45)


def test_the_rerank_timestamp_survives_a_save(costs: CostModel, tmp_path: Path) -> None:
    book = seeded(costs)
    book.last_rerank_at = NOW.isoformat()
    path = tmp_path / "book.json"
    book.save(path)

    restored = PaperBook.load(path)

    assert restored is not None
    assert restored.last_rerank_at == NOW.isoformat()


# -- reconciling the book against a new ranking -----------------------------


def held_ids(book: PaperBook) -> set[str]:
    return set(book.positions)


def test_a_name_the_new_ranking_still_holds_is_retained(costs: CostModel) -> None:
    book = seeded(costs)
    quotes = {"NSE:ALPHA": quote(100.0), "NSE:BETA": quote(200.0)}

    closed = book.apply_ranking(
        ["NSE:ALPHA", "NSE:BETA"], quotes, costs=costs, today=TODAY
    )

    assert closed == []
    assert held_ids(book) == {"NSE:ALPHA", "NSE:BETA"}
    assert all(p.ranking_exit_after == "" for p in book.positions.values())


def test_a_dropped_name_at_a_loss_is_sold_at_once_however_small_the_loss(
    costs: CostModel,
) -> None:
    """No threshold. The 3% stop asks whether the name broke down; this asks
    whether the strategy still backs it, and a name it has dropped that is
    also losing has nothing left arguing to hold it."""
    book = seeded(costs)
    # Down about 0.5% -- nowhere near the hard stop.
    quotes = {"NSE:ALPHA": quote(99.5), "NSE:BETA": quote(200.0)}

    closed = book.apply_ranking(["NSE:BETA"], quotes, costs=costs, today=TODAY)

    assert [t.symbol for t in closed] == ["ALPHA"]
    assert closed[0].exit_reason == "dropped_from_ranking"
    assert closed[0].exit_price == pytest.approx(99.5)
    assert closed[0].net_pnl < 0
    assert held_ids(book) == {"NSE:BETA"}


def test_a_name_up_on_the_screen_but_down_after_charges_counts_as_losing(
    costs: CostModel,
) -> None:
    """Profit is measured net, as everywhere else: a position up 0.1% on the
    screen has not covered its own charges and is not in profit."""
    book = seeded(costs)
    quotes = {"NSE:ALPHA": quote(100.1), "NSE:BETA": quote(200.0)}

    closed = book.apply_ranking(["NSE:BETA"], quotes, costs=costs, today=TODAY)

    assert [t.symbol for t in closed] == ["ALPHA"]
    assert closed[0].exit_reason == "dropped_from_ranking"


def test_a_dropped_name_in_profit_keeps_the_day_and_is_marked_to_go(
    costs: CostModel,
) -> None:
    book = seeded(costs)
    quotes = {"NSE:ALPHA": quote(102.0), "NSE:BETA": quote(200.0)}

    closed = book.apply_ranking(["NSE:BETA"], quotes, costs=costs, today=TODAY)

    assert closed == []
    assert held_ids(book) == {"NSE:ALPHA", "NSE:BETA"}
    assert book.positions["NSE:ALPHA"].ranking_exit_after == TODAY.isoformat()


def test_a_graced_position_still_has_its_stops(costs: CostModel) -> None:
    """The grace is from the ranking, not from risk management. A name held
    to the close because it was winning can still break down before it."""
    book = seeded(costs)
    book.apply_ranking(
        {"NSE:BETA"}, {"NSE:ALPHA": quote(102.0)}, costs=costs, today=TODAY
    )
    assert book.positions["NSE:ALPHA"].ranking_exit_after == TODAY.isoformat()

    closed = book.apply_stops(
        {"NSE:ALPHA": quote(96.0, high=102.0, low=95.0)},
        policy=POLICY,
        costs=costs,
        today=TODAY,
    )

    assert [t.exit_reason for t in closed] == ["hard_stop"]


def test_the_grace_lasts_exactly_one_session(costs: CostModel) -> None:
    book = seeded(costs)
    book.apply_ranking(
        {"NSE:BETA"}, {"NSE:ALPHA": quote(102.0)}, costs=costs, today=TODAY
    )
    quotes = {"NSE:ALPHA": quote(103.0), "NSE:BETA": quote(200.0)}

    same_day = book.apply_scheduled_exits(
        {"NSE:BETA"}, quotes, costs=costs, today=TODAY
    )
    assert same_day == []
    assert "NSE:ALPHA" in book.positions

    tomorrow = book.apply_scheduled_exits(
        {"NSE:BETA"}, quotes, costs=costs, today=TODAY + dt.timedelta(days=1)
    )

    assert [t.symbol for t in tomorrow] == ["ALPHA"]
    assert tomorrow[0].exit_reason == "ranking_exit"
    assert tomorrow[0].net_pnl > 0


def test_the_grace_is_not_renewed_by_a_second_drop(costs: CostModel) -> None:
    """Still unranked the next day and still winning: it goes anyway. One
    session, not a rolling reprieve."""
    book = seeded(costs)
    book.apply_ranking(
        {"NSE:BETA"}, {"NSE:ALPHA": quote(102.0)}, costs=costs, today=TODAY
    )
    tomorrow = TODAY + dt.timedelta(days=1)
    quotes = {"NSE:ALPHA": quote(104.0), "NSE:BETA": quote(200.0)}

    book.apply_ranking({"NSE:BETA"}, quotes, costs=costs, today=tomorrow)
    closed = book.apply_scheduled_exits(
        {"NSE:BETA"}, quotes, costs=costs, today=tomorrow
    )

    assert [t.symbol for t in closed] == ["ALPHA"]


def test_a_ranking_that_takes_the_name_back_cancels_the_grace(
    costs: CostModel,
) -> None:
    """A full reprieve, not a pause. Selling a name the strategy has just
    re-picked only to buy it back on the next refresh pays two sets of
    charges to end up where it started."""
    book = seeded(costs)
    book.apply_ranking(
        {"NSE:BETA"}, {"NSE:ALPHA": quote(102.0)}, costs=costs, today=TODAY
    )

    book.apply_ranking(
        {"NSE:ALPHA", "NSE:BETA"},
        {"NSE:ALPHA": quote(102.0), "NSE:BETA": quote(200.0)},
        costs=costs,
        today=TODAY,
    )

    assert book.positions["NSE:ALPHA"].ranking_exit_after == ""


def test_a_scheduled_exit_is_cancelled_if_the_name_is_ranked_again(
    costs: CostModel,
) -> None:
    book = seeded(costs)
    book.apply_ranking(
        {"NSE:BETA"}, {"NSE:ALPHA": quote(102.0)}, costs=costs, today=TODAY
    )

    closed = book.apply_scheduled_exits(
        {"NSE:ALPHA", "NSE:BETA"},
        {"NSE:ALPHA": quote(102.0)},
        costs=costs,
        today=TODAY + dt.timedelta(days=1),
    )

    assert closed == []
    assert book.positions["NSE:ALPHA"].ranking_exit_after == ""


def test_a_holding_with_no_price_is_neither_sold_nor_graced(
    costs: CostModel,
) -> None:
    """A position that cannot be valued cannot be judged. Guessing either way
    would be a decision made on no information."""
    book = seeded(costs)

    closed = book.apply_ranking({"NSE:BETA"}, {}, costs=costs, today=TODAY)

    assert closed == []
    assert book.positions["NSE:ALPHA"].ranking_exit_after == ""


def test_the_ranking_exit_flag_survives_a_save(
    costs: CostModel, tmp_path: Path
) -> None:
    """It is the only record that a position is living on borrowed time."""
    book = seeded(costs)
    book.apply_ranking(
        {"NSE:BETA"}, {"NSE:ALPHA": quote(102.0)}, costs=costs, today=TODAY
    )
    path = tmp_path / "book.json"
    book.save(path)

    restored = PaperBook.load(path)

    assert restored is not None
    assert restored.positions["NSE:ALPHA"].ranking_exit_after == TODAY.isoformat()


def test_ranking_exits_credit_cash_like_any_other_sale(costs: CostModel) -> None:
    book = seeded(costs)
    cash_before = book.cash

    closed = book.apply_ranking(
        {"NSE:BETA"}, {"NSE:ALPHA": quote(99.5)}, costs=costs, today=TODAY
    )

    assert book.cash == pytest.approx(cash_before + closed[0].net_proceeds)
    assert book.realized()["trades"] == 1
