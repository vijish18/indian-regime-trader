"""StockSelector end to end: point-in-time universe, factor look-ahead,
ranking stability, missing values, and liquidity filtering.

Uses a lightweight in-memory ``MarketDataProvider`` test double (not the
filesystem-backed one) so these tests stay fast and so "an instrument has no
data at all" versus "an instrument has some data, just not enough" can be
constructed precisely -- see ``_FakeMarketDataProvider`` below. One
integration-style test at the bottom uses the real, file-backed
``LocalMarketDataProvider`` to confirm the wiring actually requests adjusted
prices, since the fake provider does not model corporate-action adjustment at
all.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import pytest

from config.models import SelectionConfig, UniverseConfig
from data.corporate_actions import InMemoryCorporateActionProvider
from data.errors import DataNotAvailableError
from data.instrument_master import InMemoryInstrumentRepository
from data.interfaces import MarketDataProvider
from data.market_data import LocalMarketDataProvider, bars_to_frame, index_observations_to_frame
from data.membership import InMemoryIndexMembershipProvider
from data.models import (
    CorporateAction,
    CorporateActionType,
    DailyBar,
    Exchange,
    IndexMembership,
    IndexObservation,
    Instrument,
    PriceBasis,
    Quote,
    Segment,
)
from data.storage import LocalDataStore, StorageFormat, write_table
from universe.stock_selector import SelectionExclusionReason, StockSelector
from universe.universe import UniverseProvider

INDEX_SYMBOL = "NIFTY50"


# --------------------------------------------------------------------------
# In-memory market data test double
# --------------------------------------------------------------------------


class _FakeMarketDataProvider(MarketDataProvider):
    """An instrument never passed to ``add_equity`` behaves like a missing
    data file (raises ``DataNotAvailableError``, matching
    ``LocalMarketDataProvider``); one passed with a short series behaves like
    a file that exists but does not cover enough history. ``price_basis`` is
    accepted but not modeled -- adjustment itself is Phase 2-3's tested
    responsibility, not this module's.
    """

    def __init__(self) -> None:
        self._bars: dict[str, list[DailyBar]] = {}
        self._index: dict[str, list[IndexObservation]] = {}

    def add_equity(
        self,
        instrument_id: str,
        dates: list[dt.date],
        closes: list[float],
        *,
        volume: int | list[int] = 2_000_000,
    ) -> None:
        volumes = volume if isinstance(volume, list) else [volume] * len(dates)
        self._bars[instrument_id] = [
            DailyBar(
                instrument_id=instrument_id,
                session_date=day,
                open=Decimal(str(close)),
                high=Decimal(str(close * 1.01)),
                low=Decimal(str(close * 0.99)),
                close=Decimal(str(close)),
                volume=vol,
            )
            for day, close, vol in zip(dates, closes, volumes, strict=True)
        ]

    def add_index(self, index_symbol: str, dates: list[dt.date], closes: list[float]) -> None:
        self._index[index_symbol] = [
            IndexObservation(index_symbol, day, Decimal(str(close)))
            for day, close in zip(dates, closes, strict=True)
        ]

    def get_equity_bars(
        self,
        instrument_id: str,
        start: dt.date,
        end: dt.date,
        price_basis: PriceBasis = PriceBasis.RAW,
    ) -> list[DailyBar]:
        if instrument_id not in self._bars:
            raise DataNotAvailableError(f"no data for {instrument_id}")
        return [bar for bar in self._bars[instrument_id] if start <= bar.session_date <= end]

    def get_index_observations(
        self, index_symbol: str, start: dt.date, end: dt.date
    ) -> list[IndexObservation]:
        return [
            observation
            for observation in self._index.get(index_symbol, [])
            if start <= observation.session_date <= end
        ]

    def available_range(self, instrument_id: str) -> tuple[dt.date, dt.date] | None:
        raise NotImplementedError

    def get_quote(self, instrument_id: str) -> Quote:
        raise DataNotAvailableError("fake provider has no live quotes")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _dates(n: int, start: dt.date = dt.date(2022, 1, 3)) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range(n)]


def _flat(value: float, n: int) -> list[float]:
    return [value] * n


def _rising(start_price: float, daily_return: float, n: int) -> list[float]:
    return [start_price * (1 + daily_return) ** i for i in range(n)]


def selection_config(**overrides: object) -> SelectionConfig:
    defaults: dict[str, object] = {
        "min_holdings": 2,
        "max_holdings": 3,
        "momentum_lookback_months": [1, 2],
        "momentum_skip_days": 3,
        "trend_ma_window_days": 5,
        "trend_window_days": 10,
        "relative_strength_window_days": 20,
        "volatility_window_days": 10,
        "drawdown_window_days": 20,
        "liquidity_window_days": 5,
        "min_history_days": 47,
        "factor_weights": {
            "momentum": 0.3,
            "trend_persistence": 0.15,
            "relative_strength": 0.2,
            "volatility": 0.15,
            "drawdown": 0.1,
            "liquidity": 0.1,
        },
    }
    defaults.update(overrides)
    return SelectionConfig.model_validate(defaults)


def universe_config(**overrides: object) -> UniverseConfig:
    defaults: dict[str, object] = {
        "index_reference": INDEX_SYMBOL,
        "min_avg_daily_value_inr": 10_000_000.0,
        "point_in_time": True,
        "exclude_illiquid": True,
    }
    defaults.update(overrides)
    return UniverseConfig.model_validate(defaults)


def instrument(
    instrument_id: str,
    symbol: str | None = None,
    *,
    effective_from: dt.date = dt.date(2015, 1, 1),
) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        symbol=symbol or instrument_id.split(":")[-1],
        exchange=Exchange.NSE,
        segment=Segment.EQUITY,
        tick_size=Decimal("0.05"),
        price_precision=2,
        effective_from=effective_from,
    )


def membership(
    instrument_id: str,
    *,
    effective_from: dt.date = dt.date(2015, 1, 1),
    effective_to: dt.date | None = None,
) -> IndexMembership:
    return IndexMembership(INDEX_SYMBOL, instrument_id, effective_from, effective_to)


class Environment:
    """Bundles a StockSelector with the fake market data provider backing
    it, so tests can add price series and immediately call select/
    build_candidate_universe.
    """

    def __init__(
        self,
        instruments: list[Instrument],
        memberships: list[IndexMembership],
        *,
        config: SelectionConfig | None = None,
        universe: UniverseConfig | None = None,
    ) -> None:
        self.market_data = _FakeMarketDataProvider()
        self.universe_provider = UniverseProvider(
            INDEX_SYMBOL,
            InMemoryIndexMembershipProvider(memberships),
            InMemoryInstrumentRepository(instruments),
        )
        self.config = config or selection_config()
        self.universe_config = universe or universe_config()
        self.selector = StockSelector(
            self.config, self.universe_config, self.universe_provider, self.market_data
        )
        # A neutral, flat index series spanning every as_of used anywhere in
        # this test file, with wide margin on both sides for the longest
        # configured lookback (including the shipped production config's).
        index_start = dt.date(2015, 1, 1)
        index_end = dt.date(2025, 12, 31)
        span_days = (index_end - index_start).days + 1
        self.market_data.add_index(
            INDEX_SYMBOL, _dates(span_days, index_start), _flat(100.0, span_days)
        )

    def add_stock(
        self,
        instrument_id: str,
        as_of: dt.date,
        closes: list[float],
        *,
        volume: int | list[int] = 2_000_000,
        days_of_history: int | None = None,
    ) -> None:
        n = days_of_history if days_of_history is not None else len(closes)
        start = as_of - dt.timedelta(days=n - 1)
        dates = _dates(n, start)
        self.market_data.add_equity(instrument_id, dates, closes[-n:], volume=volume)


def _default_environment(n_stocks: int = 3) -> tuple[Environment, dt.date, list[str]]:
    ids = [f"NSE:S{i}" for i in range(n_stocks)]
    env = Environment(
        [instrument(i) for i in ids],
        [membership(i) for i in ids],
    )
    as_of = dt.date(2022, 6, 1)
    for i, instrument_id in enumerate(ids):
        closes = _rising(100.0, 0.001 * (i + 1), 100)
        env.add_stock(instrument_id, as_of, closes, days_of_history=100)
    return env, as_of, ids


# --------------------------------------------------------------------------
# Point-in-time universe
# --------------------------------------------------------------------------


def test_a_constituent_added_later_is_absent_from_an_earlier_selection() -> None:
    """StockSelector must not bypass UniverseProvider's point-in-time
    guarantee: a name that joined the index in 2022 cannot appear in a
    candidate universe built for a 2021 date.
    """
    early_joiner = "NSE:EARLY"
    late_joiner = "NSE:LATE"
    env = Environment(
        [instrument(early_joiner), instrument(late_joiner)],
        [
            membership(early_joiner, effective_from=dt.date(2015, 1, 1)),
            membership(late_joiner, effective_from=dt.date(2022, 4, 1)),
        ],
    )
    as_of = dt.date(2021, 12, 1)
    env.add_stock(early_joiner, as_of, _rising(100.0, 0.001, 100), days_of_history=100)
    env.add_stock(late_joiner, as_of, _rising(100.0, 0.001, 100), days_of_history=100)

    candidates = env.selector.build_candidate_universe(as_of)

    assert early_joiner in candidates
    assert late_joiner not in candidates


def test_a_constituent_removed_earlier_is_absent_from_a_later_selection() -> None:
    removed = "NSE:REMOVED"
    stays = "NSE:STAYS"
    env = Environment(
        [instrument(removed), instrument(stays)],
        [
            membership(
                removed, effective_from=dt.date(2015, 1, 1), effective_to=dt.date(2020, 12, 31)
            ),
            membership(stays, effective_from=dt.date(2015, 1, 1)),
        ],
    )
    as_of = dt.date(2022, 6, 1)
    env.add_stock(removed, as_of, _rising(100.0, 0.001, 100), days_of_history=100)
    env.add_stock(stays, as_of, _rising(100.0, 0.001, 100), days_of_history=100)

    candidates = env.selector.build_candidate_universe(as_of)

    assert removed not in candidates
    assert stays in candidates


def test_suspended_instrument_is_excluded_from_the_candidate_universe() -> None:
    """The eligibility layer (universe.universe.UniverseProvider) already
    excludes suspended/delisted names -- confirm StockSelector honors that
    rather than re-including everything a membership file mentions.
    """
    from data.models import InstrumentStatus

    suspended_id = "NSE:HALTED"
    healthy_id = "NSE:HEALTHY"
    env = Environment(
        [
            Instrument(
                instrument_id=suspended_id,
                symbol="HALTED",
                exchange=Exchange.NSE,
                segment=Segment.EQUITY,
                tick_size=Decimal("0.05"),
                price_precision=2,
                effective_from=dt.date(2015, 1, 1),
                status=InstrumentStatus.SUSPENDED,
            ),
            instrument(healthy_id),
        ],
        [membership(suspended_id), membership(healthy_id)],
    )
    as_of = dt.date(2022, 6, 1)
    env.add_stock(suspended_id, as_of, _rising(100.0, 0.001, 100), days_of_history=100)
    env.add_stock(healthy_id, as_of, _rising(100.0, 0.001, 100), days_of_history=100)

    candidates = env.selector.build_candidate_universe(as_of)

    assert suspended_id not in candidates
    assert healthy_id in candidates


# --------------------------------------------------------------------------
# Factor look-ahead
# --------------------------------------------------------------------------


def test_selection_is_unaffected_by_prices_added_after_the_decision_date() -> None:
    """The sharpest version of the causality guarantee, at the level a
    caller actually consumes: select() for a fixed as_of must return
    identical scores whether or not the underlying provider also happens to
    hold data for sessions after as_of.
    """
    env, as_of, ids = _default_environment()
    before = env.selector.select(as_of)

    # Extend every stock and the index with wild future prices.
    future_dates = _dates(30, as_of + dt.timedelta(days=1))
    for instrument_id in ids:
        env.market_data._bars[instrument_id].extend(
            DailyBar(
                instrument_id,
                day,
                Decimal("9999"),
                Decimal("9999"),
                Decimal("9999"),
                Decimal("9999"),
                1,
            )
            for day in future_dates
        )
    env.market_data._index[INDEX_SYMBOL].extend(
        IndexObservation(INDEX_SYMBOL, day, Decimal("1")) for day in future_dates
    )

    after = env.selector.select(as_of)
    assert after == before


def test_candidate_universe_membership_is_unaffected_by_future_prices() -> None:
    env, as_of, ids = _default_environment()
    before = env.selector.build_candidate_universe(as_of)

    future_dates = _dates(30, as_of + dt.timedelta(days=1))
    for instrument_id in ids:
        env.market_data._bars[instrument_id].extend(
            DailyBar(instrument_id, day, Decimal("1"), Decimal("1"), Decimal("1"), Decimal("1"), 1)
            for day in future_dates
        )

    after = env.selector.build_candidate_universe(as_of)
    assert after.candidate_ids == before.candidate_ids


def test_selector_requests_adjusted_prices_from_the_real_provider(tmp_path: Path) -> None:
    """Integration check with the real, adjustment-aware provider: a split
    that goes ex before as_of must be reflected in the factors computed for
    it, proving StockSelector actually asked for ADJUSTED prices rather than
    RAW ones.
    """
    store = LocalDataStore(
        tmp_path / "raw", tmp_path / "norm", tmp_path / "ref", StorageFormat.CSV
    )
    instrument_id = "NSE:SPLIT"
    as_of = dt.date(2022, 6, 1)
    dates = _dates(100, as_of - dt.timedelta(days=99))
    split_date = dates[60]

    # A flat 500-price series, with a 5-for-1 split going ex mid-window.
    # Post-split RAW prices are ~100 (500 / 5); pre-split RAW prices are ~500.
    raw_closes = [500.0] * 60 + [100.0] * 40
    bars = [
        DailyBar(
            instrument_id,
            day,
            Decimal(str(c)),
            Decimal(str(c)),
            Decimal(str(c)),
            Decimal(str(c)),
            500_000,
        )
        for day, c in zip(dates, raw_closes, strict=True)
    ]
    write_table(bars_to_frame(bars), store.equity_bars_path(instrument_id))
    # Wide margin: _fetch_index_closes requests back to
    # as_of - 2 * max(relative_strength_window_days, min_history_days).
    index_dates = _dates(400, as_of - dt.timedelta(days=399))
    write_table(
        index_observations_to_frame(
            [IndexObservation(INDEX_SYMBOL, day, Decimal("100")) for day in index_dates]
        ),
        store.index_path(INDEX_SYMBOL),
    )

    actions = InMemoryCorporateActionProvider(
        [
            CorporateAction(
                instrument_id=instrument_id,
                action_type=CorporateActionType.SPLIT,
                ex_date=split_date,
                ratio_new=Decimal(5),
                ratio_old=Decimal(1),
            )
        ]
    )
    instruments_repo = InMemoryInstrumentRepository([instrument(instrument_id)])
    membership_repo = InMemoryIndexMembershipProvider([membership(instrument_id)])
    universe_provider = UniverseProvider(INDEX_SYMBOL, membership_repo, instruments_repo, actions)
    market_data = LocalMarketDataProvider(store, actions)

    config = selection_config(min_history_days=90, drawdown_window_days=90)
    selector = StockSelector(config, universe_config(), universe_provider, market_data)

    candidates = selector.build_candidate_universe(as_of)
    assert instrument_id in candidates, [e.detail for e in candidates.excluded]

    scores = selector.score_candidates(candidates, as_of)
    # On an ADJUSTED basis, the whole series is a flat ~100, so drawdown must
    # be ~0 -- on a RAW basis it would show a fake -80% single-day crash.
    assert scores[0].raw_factors.drawdown == pytest.approx(0.0, abs=0.02)


# --------------------------------------------------------------------------
# Ranking stability
# --------------------------------------------------------------------------


def test_selection_is_deterministic() -> None:
    env, as_of, _ids = _default_environment()
    first = env.selector.select(as_of)
    second = env.selector.select(as_of)
    assert first == second


def test_ranks_are_assigned_consecutively_from_one() -> None:
    env, as_of, _ids = _default_environment()
    ranked = env.selector.score_candidates(env.selector.build_candidate_universe(as_of), as_of)
    assert [candidate.rank for candidate in ranked] == list(range(1, len(ranked) + 1))


def test_tied_scores_break_ties_by_instrument_id() -> None:
    """Two candidates with literally identical price histories must not have
    their relative order depend on dict/set iteration order.
    """
    ids = ["NSE:AAA", "NSE:ZZZ"]
    env = Environment([instrument(i) for i in ids], [membership(i) for i in ids])
    as_of = dt.date(2022, 6, 1)
    identical_closes = _rising(100.0, 0.002, 100)
    for instrument_id in ids:
        env.add_stock(instrument_id, as_of, list(identical_closes), days_of_history=100)

    ranked = env.selector.select(as_of)
    assert ranked[0].score == pytest.approx(ranked[1].score)
    assert [candidate.instrument_id for candidate in ranked] == sorted(ids)


def test_excluding_an_unrelated_failing_candidate_does_not_change_survivors_scores() -> None:
    """Exclusions happen before cross-sectional standardization, so adding
    one more instrument that fails filters must not perturb the scores of
    the candidates that do pass.
    """
    env, as_of, ids = _default_environment()
    baseline = {score.instrument_id: score.score for score in env.selector.select(as_of)}

    # Add an illiquid instrument to the same universe/environment.
    illiquid_id = "NSE:ILLIQUID"
    env.universe_provider = UniverseProvider(
        INDEX_SYMBOL,
        InMemoryIndexMembershipProvider([membership(i) for i in [*ids, illiquid_id]]),
        InMemoryInstrumentRepository([instrument(i) for i in [*ids, illiquid_id]]),
    )
    env.selector = StockSelector(
        env.config, env.universe_config, env.universe_provider, env.market_data
    )
    env.add_stock(illiquid_id, as_of, _rising(100.0, 0.001, 100), volume=10, days_of_history=100)

    with_extra = {score.instrument_id: score.score for score in env.selector.select(as_of)}

    assert illiquid_id not in with_extra
    for instrument_id, score in baseline.items():
        assert with_extra[instrument_id] == pytest.approx(score)


def test_selection_respects_max_holdings() -> None:
    env, as_of, ids = _default_environment(n_stocks=5)
    config = selection_config(max_holdings=2, min_holdings=1)
    selector = StockSelector(config, env.universe_config, env.universe_provider, env.market_data)
    ranked = selector.select(as_of)
    assert len(ranked) == 2
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2


# --------------------------------------------------------------------------
# Missing values
# --------------------------------------------------------------------------


def test_instrument_with_no_data_at_all_is_excluded_not_a_crash() -> None:
    present_id = "NSE:PRESENT"
    missing_id = "NSE:MISSING"
    env = Environment(
        [instrument(present_id), instrument(missing_id)],
        [membership(present_id), membership(missing_id)],
    )
    as_of = dt.date(2022, 6, 1)
    env.add_stock(present_id, as_of, _rising(100.0, 0.001, 100), days_of_history=100)
    # missing_id is never added to the market data provider at all.

    candidates = env.selector.build_candidate_universe(as_of)

    assert present_id in candidates
    assert missing_id not in candidates
    exclusion = next(e for e in candidates.excluded if e.instrument_id == missing_id)
    assert exclusion.reason is SelectionExclusionReason.DATA_UNAVAILABLE


def test_instrument_with_too_little_history_is_excluded() -> None:
    short_id = "NSE:SHORT"
    long_id = "NSE:LONG"
    env = Environment(
        [instrument(short_id), instrument(long_id)],
        [membership(short_id), membership(long_id)],
    )
    as_of = dt.date(2022, 6, 1)
    env.add_stock(short_id, as_of, _rising(100.0, 0.001, 20), days_of_history=20)  # too short
    env.add_stock(long_id, as_of, _rising(100.0, 0.001, 100), days_of_history=100)

    candidates = env.selector.build_candidate_universe(as_of)

    assert long_id in candidates
    assert short_id not in candidates
    exclusion = next(e for e in candidates.excluded if e.instrument_id == short_id)
    assert exclusion.reason is SelectionExclusionReason.INSUFFICIENT_HISTORY
    assert "20" in exclusion.detail


def test_recently_listed_instrument_becomes_eligible_once_it_has_enough_history() -> None:
    """The same instrument must transition from excluded to included purely
    as a function of how much history exists by the decision date -- no
    special-casing of "new listings" beyond the generic history check.
    """
    instrument_id = "NSE:NEWCO"
    env = Environment([instrument(instrument_id)], [membership(instrument_id)])
    listing_date = dt.date(2022, 1, 3)
    all_dates = _dates(80, listing_date)
    all_closes = _rising(100.0, 0.001, 80)
    env.market_data.add_equity(instrument_id, all_dates, all_closes)

    too_early = env.selector.build_candidate_universe(listing_date + dt.timedelta(days=10))
    assert instrument_id not in too_early

    late_enough = env.selector.build_candidate_universe(all_dates[-1])
    assert instrument_id in late_enough


def test_selection_succeeds_when_every_candidate_survives(tmp_path: Path) -> None:
    """A basic sanity check that missing-value handling for *other*
    instruments doesn't accidentally break a clean run."""
    env, as_of, ids = _default_environment()
    ranked = env.selector.select(as_of)
    assert {score.instrument_id for score in ranked} <= set(ids)
    assert len(ranked) > 0


# --------------------------------------------------------------------------
# Liquidity filtering
# --------------------------------------------------------------------------


def test_instrument_below_the_liquidity_threshold_is_excluded() -> None:
    liquid_id = "NSE:LIQUID"
    illiquid_id = "NSE:ILLIQUID"
    env = Environment(
        [instrument(liquid_id), instrument(illiquid_id)],
        [membership(liquid_id), membership(illiquid_id)],
        universe=universe_config(min_avg_daily_value_inr=10_000_000.0),
    )
    as_of = dt.date(2022, 6, 1)
    env.add_stock(
        liquid_id, as_of, _flat(100.0, 100), volume=500_000, days_of_history=100
    )  # 50M/day
    env.add_stock(
        illiquid_id, as_of, _flat(100.0, 100), volume=100, days_of_history=100
    )  # 10K/day

    candidates = env.selector.build_candidate_universe(as_of)

    assert liquid_id in candidates
    assert illiquid_id not in candidates
    exclusion = next(e for e in candidates.excluded if e.instrument_id == illiquid_id)
    assert exclusion.reason is SelectionExclusionReason.ILLIQUID


def test_instrument_exactly_at_the_liquidity_threshold_passes() -> None:
    at_threshold_id = "NSE:ATLINE"
    env = Environment(
        [instrument(at_threshold_id)],
        [membership(at_threshold_id)],
        universe=universe_config(min_avg_daily_value_inr=1_000_000.0),
    )
    as_of = dt.date(2022, 6, 1)
    # close=100, volume=10_000 -> traded value exactly 1_000_000 INR/day.
    env.add_stock(at_threshold_id, as_of, _flat(100.0, 100), volume=10_000, days_of_history=100)

    candidates = env.selector.build_candidate_universe(as_of)
    assert at_threshold_id in candidates


def test_liquidity_filter_uses_only_the_configured_trailing_window() -> None:
    """A stock that used to be illiquid long ago but has since become liquid
    must pass -- and vice versa -- because only the trailing
    liquidity_window_days is averaged, not the whole history.
    """
    instrument_id = "NSE:RECOVERED"
    env = Environment(
        [instrument(instrument_id)],
        [membership(instrument_id)],
        config=selection_config(liquidity_window_days=5),
        universe=universe_config(min_avg_daily_value_inr=10_000_000.0),
    )
    as_of = dt.date(2022, 6, 1)
    # 95 days of illiquid trading, then 5 days of heavy volume just before as_of.
    volumes = [10] * 95 + [1_000_000] * 5
    env.add_stock(instrument_id, as_of, _flat(100.0, 100), volume=volumes, days_of_history=100)

    candidates = env.selector.build_candidate_universe(as_of)
    assert instrument_id in candidates


def test_liquidity_exclusion_detail_reports_the_computed_value() -> None:
    instrument_id = "NSE:QUIET"
    env = Environment(
        [instrument(instrument_id)],
        [membership(instrument_id)],
        universe=universe_config(min_avg_daily_value_inr=5_000_000.0),
    )
    as_of = dt.date(2022, 6, 1)
    env.add_stock(instrument_id, as_of, _flat(100.0, 100), volume=1_000, days_of_history=100)

    candidates = env.selector.build_candidate_universe(as_of)
    exclusion = next(e for e in candidates.excluded if e.instrument_id == instrument_id)
    assert "100,000" in exclusion.detail  # 100 * 1000 traded value
    assert "5,000,000" in exclusion.detail  # the configured threshold


# --------------------------------------------------------------------------
# The shipped production config actually works end to end
# --------------------------------------------------------------------------


def test_default_settings_selection_config_runs_end_to_end() -> None:
    from config.loader import load_settings

    settings = load_settings()
    ids = [f"NSE:P{i}" for i in range(4)]
    env = Environment(
        [instrument(i) for i in ids],
        [membership(i) for i in ids],
        config=settings.selection,
        universe=settings.universe,
    )
    as_of = dt.date(2023, 1, 2)
    n = settings.selection.min_history_days + 5
    for i, instrument_id in enumerate(ids):
        closes = _rising(500.0, 0.0004 * (i + 1), n)
        env.add_stock(instrument_id, as_of, closes, volume=1_000_000, days_of_history=n)

    ranked = env.selector.select(as_of)
    assert len(ranked) <= settings.selection.max_holdings
    assert len(ranked) > 0


def test_a_single_momentum_horizon_is_accepted() -> None:
    config = selection_config(momentum_lookback_months=[2])
    assert config.momentum_lookback_months == [2]


def test_momentum_horizons_must_still_ascend_or_number_one_or_two() -> None:
    with pytest.raises(ValueError):
        selection_config(momentum_lookback_months=[2, 1])
    with pytest.raises(ValueError):
        selection_config(momentum_lookback_months=[1, 2, 3])
    with pytest.raises(ValueError):
        selection_config(momentum_lookback_months=[])
