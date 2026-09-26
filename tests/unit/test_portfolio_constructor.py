"""Portfolio construction: the weighting waterfall, target vs. current vs.
required trades, and the hard invariants a target portfolio must never
violate (no leverage, no negative positions, cash never negative, no
security outside the candidate set).
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import numpy as np
import pytest

from config.models import ExecutionConfig, PortfolioConfig, SelectionConfig
from core.regime.allocation import AllocationRegime, AllocationTarget
from data.errors import DataNotAvailableError
from data.interfaces import MarketDataProvider
from data.models import DailyBar, IndexObservation, PriceBasis, Quote
from portfolio.portfolio_constructor import (
    PortfolioConstructionError,
    PortfolioConstructor,
    TargetPortfolio,
    TargetPosition,
    TradeAction,
    empty_portfolio,
    required_trades,
)
from universe.factor_calculator import FactorSet
from universe.stock_selector import StockScore

AS_OF = dt.date(2023, 6, 1)


# --------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------


def portfolio_config(**overrides: object) -> PortfolioConfig:
    defaults: dict[str, object] = {
        "max_single_name_pct": 0.15,
        "max_sector_pct": 0.30,
        "min_position_weight_pct": 0.02,
        "max_pairwise_correlation": 0.80,
        "correlation_lookback_days": 30,
        "correlation_penalty_pct": 0.50,
        "min_correlation_observations": 10,
    }
    defaults.update(overrides)
    return PortfolioConfig.model_validate(defaults)


def selection_config(**overrides: object) -> SelectionConfig:
    defaults: dict[str, object] = {
        "min_holdings": 2,
        "max_holdings": 6,
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


def execution_config(**overrides: object) -> ExecutionConfig:
    defaults: dict[str, object] = {
        "mode": "paper",
        "order_type": "limit",
        "order_price_guard_bps": 50,
        "stale_quote_seconds": 15,
        "max_participation_adv_pct": 0.05,
        "no_market_order_fallback": True,
    }
    defaults.update(overrides)
    return ExecutionConfig.model_validate(defaults)


def exposure_target(
    regime: AllocationRegime = AllocationRegime.NORMAL_RISK,
    target: float = 0.70,
    *,
    min_gross_exposure: float = 0.45,
    max_gross_exposure: float = 0.75,
    allow_new_positions: bool = True,
    confidence: float = 0.85,
) -> AllocationTarget:
    return AllocationTarget(
        as_of=AS_OF,
        regime=regime,
        target_gross_exposure=target,
        min_gross_exposure=min_gross_exposure,
        max_gross_exposure=max_gross_exposure,
        allow_new_positions=allow_new_positions,
        confidence=confidence,
        expected_volatility=0.15,
        reason="test",
    )


# --------------------------------------------------------------------------
# Candidate helpers
# --------------------------------------------------------------------------


def factor_set(volatility: float = 0.15, liquidity_inr: float = 100_000_000.0) -> FactorSet:
    return FactorSet(
        momentum=0.1,
        trend_persistence=0.6,
        relative_strength=0.05,
        volatility=volatility,
        drawdown=-0.05,
        liquidity_inr=liquidity_inr,
    )


def candidate(
    instrument_id: str,
    rank: int,
    score: float,
    *,
    volatility: float = 0.15,
    liquidity_inr: float = 100_000_000.0,
    symbol: str | None = None,
) -> StockScore:
    return StockScore(
        instrument_id=instrument_id,
        symbol=symbol or instrument_id.split(":")[-1],
        as_of=AS_OF,
        rank=rank,
        score=score,
        raw_factors=factor_set(volatility, liquidity_inr),
        standardized_factors=factor_set(0.0, 0.0),
    )


def make_candidates(n: int, **kwargs: object) -> list[StockScore]:
    """``n`` distinctly-ranked candidates. A single ``score=`` override
    applies the same score to every candidate (used where the test only
    cares about weighting downstream of scoring, e.g. cap enforcement); the
    default gives each a distinct, descending score so rank and score agree.
    """
    if "score" in kwargs:
        fixed_score = kwargs.pop("score")
        return [
            candidate(f"NSE:S{i:02d}", rank=i + 1, score=fixed_score, **kwargs)  # type: ignore[arg-type]
            for i in range(n)
        ]
    return [
        candidate(f"NSE:S{i:02d}", rank=i + 1, score=2.0 - i * 0.3, **kwargs)  # type: ignore[arg-type]
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# Fake market data (for correlation)
# --------------------------------------------------------------------------


class FakeMarketDataProvider(MarketDataProvider):
    def __init__(self) -> None:
        self._bars: dict[str, list[DailyBar]] = {}

    def add_equity(
        self, instrument_id: str, dates: list[dt.date], closes: list[float]
    ) -> None:
        self._bars[instrument_id] = [
            DailyBar(
                instrument_id,
                day,
                Decimal(str(c)),
                Decimal(str(c)),
                Decimal(str(c)),
                Decimal(str(c)),
                1_000_000,
            )
            for day, c in zip(dates, closes, strict=True)
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
        raise NotImplementedError

    def available_range(self, instrument_id: str) -> tuple[dt.date, dt.date] | None:
        raise NotImplementedError

    def get_quote(self, instrument_id: str) -> Quote:
        raise DataNotAvailableError("fake provider has no live quotes")


def _dates(n: int, end: dt.date = AS_OF) -> list[dt.date]:
    start = end - dt.timedelta(days=n - 1)
    return [start + dt.timedelta(days=i) for i in range(n)]


def market_with_no_correlation_data() -> FakeMarketDataProvider:
    """A provider with no price history at all -- the correlation penalty
    must degrade to a no-op rather than crash.
    """
    return FakeMarketDataProvider()


def market_with_two_identical_series(n: int = 150) -> FakeMarketDataProvider:
    """Two perfectly correlated series (S00 and S01), plus a third
    uncorrelated one, so the correlation penalty has something real to act
    on.
    """
    provider = FakeMarketDataProvider()
    dates = _dates(n)
    rng = np.random.default_rng(7)
    shared = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    independent = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    provider.add_equity("NSE:S00", dates, shared.tolist())
    provider.add_equity("NSE:S01", dates, shared.tolist())  # identical -> correlation 1.0
    provider.add_equity("NSE:S02", dates, independent.tolist())
    return provider


def constructor(
    provider: MarketDataProvider | None = None,
    *,
    config: PortfolioConfig | None = None,
    selection: SelectionConfig | None = None,
    execution: ExecutionConfig | None = None,
) -> PortfolioConstructor:
    return PortfolioConstructor(
        config or portfolio_config(),
        selection or selection_config(),
        execution or execution_config(),
        provider or market_with_no_correlation_data(),
    )


# --------------------------------------------------------------------------
# raw_weights
# --------------------------------------------------------------------------


def test_raw_weights_are_all_positive_even_with_negative_scores() -> None:
    candidates = [
        candidate("NSE:A", 1, score=5.0),
        candidate("NSE:B", 2, score=-2.0),
        candidate("NSE:C", 3, score=-8.0),
    ]
    weights = constructor().raw_weights(candidates)
    assert all(weight > 0 for weight in weights.values())


def test_raw_weights_preserve_score_ordering() -> None:
    candidates = [
        candidate("NSE:A", 1, score=5.0, volatility=0.15),
        candidate("NSE:B", 2, score=1.0, volatility=0.15),
        candidate("NSE:C", 3, score=-3.0, volatility=0.15),
    ]
    weights = constructor().raw_weights(candidates)
    assert weights["NSE:A"] > weights["NSE:B"] > weights["NSE:C"]


def test_raw_weights_are_risk_adjusted() -> None:
    """Same score, higher volatility -> smaller raw weight."""
    calm = candidate("NSE:CALM", 1, score=1.0, volatility=0.10)
    volatile = candidate("NSE:VOL", 2, score=1.0, volatility=0.40)
    weights = constructor().raw_weights([calm, volatile])
    assert weights["NSE:CALM"] > weights["NSE:VOL"]


def test_raw_weights_of_empty_candidates_is_empty() -> None:
    assert constructor().raw_weights([]) == {}


def test_equal_weighting_ignores_score_and_volatility() -> None:
    candidates = [
        candidate("NSE:A", 1, score=5.0, volatility=0.10),
        candidate("NSE:B", 2, score=-3.0, volatility=0.60),
    ]
    weights = constructor(config=portfolio_config(weighting="equal")).raw_weights(candidates)
    assert weights["NSE:A"] == weights["NSE:B"]


def test_weighting_defaults_to_score_over_volatility() -> None:
    assert portfolio_config().weighting == "score_over_volatility"


# --------------------------------------------------------------------------
# Weighting waterfall: normalization and caps
# --------------------------------------------------------------------------


def test_weights_are_normalized_correctly_before_scaling() -> None:
    """The core numerical property: proportional weights, summing to 1.0,
    before the regime's exposure scaling is applied.
    """
    candidates = make_candidates(4)
    raw = constructor().raw_weights(candidates)
    normalized = PortfolioConstructor._normalize(raw)
    assert sum(normalized.values()) == pytest.approx(1.0)
    for instrument_id in raw:
        assert normalized[instrument_id] == pytest.approx(raw[instrument_id] / sum(raw.values()))


def test_normalize_rejects_a_non_positive_total() -> None:
    with pytest.raises(PortfolioConstructionError, match="cannot normalize"):
        PortfolioConstructor._normalize({"NSE:A": 0.0, "NSE:B": 0.0})


def test_single_name_cap_is_respected() -> None:
    """One dominant candidate must be clipped to max_single_name_pct, not
    allowed to swamp the portfolio.
    """
    candidates = [
        candidate("NSE:DOMINANT", 1, score=100.0),
        *make_candidates(5, score=1.0),
    ]
    portfolio = constructor(config=portfolio_config(max_single_name_pct=0.15)).construct(
        candidates, exposure_target(target=1.0, max_gross_exposure=1.0), AS_OF, equity=10_000_000.0
    )
    dominant = next(p for p in portfolio.positions if p.instrument_id == "NSE:DOMINANT")
    assert dominant.target_weight == pytest.approx(0.15)
    assert dominant.binding_constraint == "max_single_name"


def test_capped_weight_becomes_cash_not_redistributed() -> None:
    """The documented design choice: excess from a capped name is never
    handed to another candidate.
    """
    candidates = [candidate("NSE:DOMINANT", 1, score=100.0), candidate("NSE:OTHER", 2, score=0.01)]
    portfolio = constructor().construct(
        candidates, exposure_target(target=1.0, max_gross_exposure=1.0), AS_OF, equity=10_000_000.0
    )
    # DOMINANT capped at 15%; OTHER's weight is whatever its own tiny share
    # of the normalized basket was, scaled -- it must NOT have grown to
    # absorb DOMINANT's excess.
    assert portfolio.cash_weight > 0.5  # most of the budget went unused, not redistributed


def test_sector_cap_scales_down_the_whole_sector_proportionally() -> None:
    candidates = make_candidates(4, score=1.0)
    sector_map = {c.instrument_id: "TECH" for c in candidates[:3]}
    sector_map[candidates[3].instrument_id] = "OTHER"

    portfolio = constructor(config=portfolio_config(max_sector_pct=0.30)).construct(
        candidates,
        exposure_target(target=1.0, max_gross_exposure=1.0),
        AS_OF,
        equity=10_000_000.0,
        sector_map=sector_map,
    )
    tech_total = sum(p.target_weight for p in portfolio.positions if p.sector == "TECH")
    assert tech_total <= 0.30 + 1e-6


def test_missing_sector_map_entries_default_to_their_own_instrument_id() -> None:
    """Sector data is optional; an instrument absent from the map must not
    crash or be silently grouped with unrelated names.
    """
    candidates = make_candidates(3)
    portfolio = constructor().construct(
        candidates, exposure_target(), AS_OF, equity=10_000_000.0, sector_map=None
    )
    for position in portfolio.positions:
        assert position.sector == position.instrument_id


def test_liquidity_cap_limits_weight_relative_to_average_daily_value() -> None:
    thin = candidate("NSE:THIN", 1, score=5.0, liquidity_inr=1_000_000.0)  # tiny ADV
    deep = candidate("NSE:DEEP", 2, score=5.0, liquidity_inr=1_000_000_000.0)

    equity = 10_000_000.0
    execution = execution_config(max_participation_adv_pct=0.05)
    # A near-zero floor isolates the liquidity cap: THIN's capped weight
    # (0.005) would otherwise also be dropped by the default min-weight floor.
    config = portfolio_config(min_position_weight_pct=1e-6)
    portfolio = constructor(execution=execution, config=config).construct(
        [thin, deep], exposure_target(target=1.0, max_gross_exposure=1.0), AS_OF, equity=equity
    )
    thin_position = next(p for p in portfolio.positions if p.instrument_id == "NSE:THIN")
    # max weight = 0.05 * 1_000_000 / 10_000_000 = 0.005
    assert thin_position.target_weight == pytest.approx(0.005, abs=1e-6)
    assert thin_position.binding_constraint == "liquidity"


def test_min_weight_floor_drops_tiny_positions_to_cash() -> None:
    candidates = make_candidates(8, score=0.5)
    portfolio = constructor(config=portfolio_config(min_position_weight_pct=0.05)).construct(
        candidates,
        exposure_target(target=0.10, min_gross_exposure=0.0, max_gross_exposure=0.75),
        AS_OF,
        equity=10_000_000.0,
    )
    # 0.10 spread over 8 near-equal candidates is ~0.0125 each -- below the 5% floor.
    assert len(portfolio.positions) == 0
    assert portfolio.cash_weight == pytest.approx(1.0)


def test_scaling_hits_the_regime_specific_target_when_nothing_binds() -> None:
    """With no cap binding, the portfolio's gross exposure must equal the
    regime's exact target_gross_exposure, not just fall inside its band.
    """
    candidates = make_candidates(3, score=1.0)
    target = exposure_target(target=0.55, min_gross_exposure=0.45, max_gross_exposure=0.90)
    portfolio = constructor(config=portfolio_config(max_single_name_pct=0.90)).construct(
        candidates, target, AS_OF, equity=10_000_000.0
    )
    assert portfolio.gross_exposure == pytest.approx(0.55)


# --------------------------------------------------------------------------
# Correlation constraint
# --------------------------------------------------------------------------


def test_highly_correlated_pair_penalizes_the_lower_ranked_member() -> None:
    candidates = [
        candidate("NSE:S00", 1, score=2.0),
        candidate("NSE:S01", 2, score=1.5),
        candidate("NSE:S02", 3, score=1.0),
    ]
    provider = market_with_two_identical_series()
    pc = constructor(provider, config=portfolio_config(max_pairwise_correlation=0.80))
    weights = pc.raw_weights(candidates)
    penalized, binding = constructor(
        provider, config=portfolio_config(max_pairwise_correlation=0.80)
    ).apply_correlation_penalty(candidates, weights, AS_OF)

    assert penalized["NSE:S01"] < weights["NSE:S01"]  # the lower-ranked of the identical pair
    assert penalized["NSE:S00"] == pytest.approx(weights["NSE:S00"])  # better-ranked untouched
    assert binding["NSE:S01"] == "correlation"


def test_correlation_penalty_is_a_penalty_not_an_exclusion() -> None:
    candidates = [
        candidate("NSE:S00", 1, score=2.0),
        candidate("NSE:S01", 2, score=1.5),
        candidate("NSE:S02", 3, score=1.0),
    ]
    provider = market_with_two_identical_series()
    portfolio = constructor(provider).construct(
        candidates, exposure_target(target=0.6, max_gross_exposure=0.9), AS_OF, equity=10_000_000.0
    )
    assert "NSE:S01" in portfolio  # penalized, still present


def test_correlation_penalty_skips_gracefully_with_no_price_data() -> None:
    candidates = make_candidates(3)
    weights = constructor().raw_weights(candidates)
    penalized, binding = constructor().apply_correlation_penalty(candidates, weights, AS_OF)
    assert penalized == weights
    assert all(reason == "unconstrained" for reason in binding.values())


def test_correlation_penalty_is_skipped_with_a_single_candidate() -> None:
    candidates = [candidate("NSE:ONLY", 1, score=1.0)]
    weights = constructor().raw_weights(candidates)
    penalized, _binding = constructor().apply_correlation_penalty(candidates, weights, AS_OF)
    assert penalized == weights


def test_correlation_penalty_requires_minimum_overlapping_observations() -> None:
    provider = market_with_two_identical_series(n=5)  # far fewer than min_correlation_observations
    candidates = [candidate("NSE:S00", 1, score=2.0), candidate("NSE:S01", 2, score=1.5)]
    pc = constructor(provider, config=portfolio_config(min_correlation_observations=40))
    weights = pc.raw_weights(candidates)
    penalized, binding = constructor(
        provider, config=portfolio_config(min_correlation_observations=40)
    ).apply_correlation_penalty(candidates, weights, AS_OF)
    assert penalized == weights
    assert binding["NSE:S01"] == "unconstrained"


# --------------------------------------------------------------------------
# UNCERTAIN regime: no new positions
# --------------------------------------------------------------------------


def test_uncertain_regime_adds_no_new_names() -> None:
    candidates = make_candidates(5)
    target = exposure_target(
        regime=AllocationRegime.UNCERTAIN,
        target=0.10,
        min_gross_exposure=0.0,
        max_gross_exposure=0.15,
        allow_new_positions=False,
    )
    portfolio = constructor().construct(candidates, target, AS_OF, equity=10_000_000.0)
    assert len(portfolio.positions) == 0
    assert portfolio.cash_weight == pytest.approx(1.0)


def test_uncertain_regime_may_retain_currently_held_names() -> None:
    candidates = make_candidates(5)
    held_ids = {candidates[0].instrument_id, candidates[2].instrument_id}
    current = TargetPortfolio(
        as_of=AS_OF,
        positions=tuple(
            TargetPosition(
                c.instrument_id, c.symbol, 0.10, c.instrument_id, c.rank, c.score, "unconstrained"
            )
            for c in candidates
            if c.instrument_id in held_ids
        ),
        cash_weight=0.80,
        regime=AllocationRegime.NORMAL_RISK,
        gross_exposure=0.20,
    )
    target = exposure_target(
        regime=AllocationRegime.UNCERTAIN,
        target=0.10,
        min_gross_exposure=0.0,
        max_gross_exposure=0.15,
        allow_new_positions=False,
    )
    portfolio = constructor(config=portfolio_config(min_position_weight_pct=0.01)).construct(
        candidates, target, AS_OF, equity=10_000_000.0, current_portfolio=current
    )
    assert portfolio.instrument_ids <= held_ids
    assert portfolio.instrument_ids  # at least something survived the floor


def test_uncertain_regime_with_nothing_held_is_all_cash() -> None:
    candidates = make_candidates(5)
    target = exposure_target(regime=AllocationRegime.UNCERTAIN, allow_new_positions=False)
    portfolio = constructor().construct(candidates, target, AS_OF, equity=10_000_000.0)
    assert portfolio == empty_portfolio(AS_OF, AllocationRegime.UNCERTAIN)


def test_empty_candidates_is_all_cash() -> None:
    portfolio = constructor().construct([], exposure_target(), AS_OF, equity=10_000_000.0)
    assert len(portfolio.positions) == 0
    assert portfolio.cash_weight == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Target vs. current vs. required trades
# --------------------------------------------------------------------------


def _portfolio(
    weights: dict[str, float], regime: AllocationRegime = AllocationRegime.NORMAL_RISK
) -> TargetPortfolio:
    positions = tuple(
        TargetPosition(
            instrument_id,
            instrument_id.split(":")[-1],
            weight,
            instrument_id,
            1,
            1.0,
            "unconstrained",
        )
        for instrument_id, weight in weights.items()
    )
    gross = sum(weights.values())
    return TargetPortfolio(AS_OF, positions, 1.0 - gross, regime, gross)


def test_required_trades_classifies_buy_sell_exit_hold() -> None:
    current = _portfolio({"NSE:A": 0.10, "NSE:B": 0.20, "NSE:C": 0.05})
    target = _portfolio({"NSE:A": 0.10, "NSE:B": 0.05, "NSE:D": 0.15})

    trades = {trade.instrument_id: trade for trade in required_trades(target, current)}

    assert trades["NSE:A"].action is TradeAction.HOLD
    assert trades["NSE:B"].action is TradeAction.SELL
    assert trades["NSE:C"].action is TradeAction.EXIT
    assert trades["NSE:D"].action is TradeAction.BUY
    assert trades["NSE:D"].delta_weight == pytest.approx(0.15)
    assert trades["NSE:C"].delta_weight == pytest.approx(-0.05)


def test_required_trades_with_no_current_portfolio_are_all_buys() -> None:
    target = _portfolio({"NSE:A": 0.10, "NSE:B": 0.20})
    trades = required_trades(target, None)
    assert all(trade.action is TradeAction.BUY for trade in trades)
    assert {trade.instrument_id for trade in trades} == {"NSE:A", "NSE:B"}


def test_required_trades_produce_no_orders_only_weight_deltas() -> None:
    """Confirms the type-level guarantee: a RequiredTrade carries weights,
    never a quantity or a broker reference.
    """
    target = _portfolio({"NSE:A": 0.10})
    trade = required_trades(target, None)[0]
    for forbidden in ("quantity", "price", "order_id", "broker_order_id"):
        assert not hasattr(trade, forbidden)


def test_required_trades_between_identical_portfolios_are_all_holds() -> None:
    portfolio = _portfolio({"NSE:A": 0.10, "NSE:B": 0.20})
    trades = required_trades(portfolio, portfolio)
    assert all(trade.action is TradeAction.HOLD for trade in trades)


def test_target_and_current_portfolios_are_structurally_interchangeable() -> None:
    """The design claim stated in the module docstring: TargetPortfolio
    represents both what should be held and what currently is."""
    target = constructor().construct(
        make_candidates(3), exposure_target(), AS_OF, equity=1_000_000.0
    )
    # Using the constructed target AS a "current" portfolio for a second
    # construct() call must work without any type conversion.
    again = constructor().construct(
        make_candidates(3), exposure_target(), AS_OF, equity=1_000_000.0, current_portfolio=target
    )
    assert isinstance(again, TargetPortfolio)


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------


def test_invariant_sum_of_weights_never_exceeds_permitted_exposure() -> None:
    candidates = make_candidates(10, score=1.0)
    for max_gross in (0.30, 0.50, 0.75, 1.00):
        portfolio = constructor().construct(
            candidates,
            exposure_target(target=max_gross, min_gross_exposure=0.0, max_gross_exposure=max_gross),
            AS_OF,
            equity=10_000_000.0,
        )
        assert portfolio.gross_exposure <= max_gross + 1e-9


def test_invariant_each_position_never_exceeds_max_single_name() -> None:
    candidates = [candidate("NSE:HUGE", 1, score=1000.0), *make_candidates(6, score=0.1)]
    cap = 0.12
    portfolio = constructor(config=portfolio_config(max_single_name_pct=cap)).construct(
        candidates, exposure_target(target=1.0, max_gross_exposure=1.0), AS_OF, equity=10_000_000.0
    )
    for position in portfolio.positions:
        assert position.target_weight <= cap + 1e-9


def test_invariant_no_negative_positions() -> None:
    candidates = make_candidates(6, score=1.0)
    portfolio = constructor().construct(candidates, exposure_target(), AS_OF, equity=10_000_000.0)
    for position in portfolio.positions:
        assert position.target_weight > 0.0


def test_invariant_no_leverage() -> None:
    """No configuration this constructor can be given should ever be able to
    produce gross_exposure + cash_weight != 1.0 with cash negative."""
    candidates = make_candidates(10, score=5.0)
    portfolio = constructor().construct(
        candidates, exposure_target(target=1.0, max_gross_exposure=1.0), AS_OF, equity=10_000_000.0
    )
    assert portfolio.gross_exposure <= 1.0 + 1e-9
    assert portfolio.cash_weight >= -1e-9
    assert portfolio.gross_exposure + portfolio.cash_weight == pytest.approx(1.0)


def test_invariant_no_invalid_securities() -> None:
    """Every position in the target portfolio must trace back to an
    instrument the caller actually offered as a candidate (or already held)
    -- the constructor must never invent an instrument_id.
    """
    candidates = make_candidates(5)
    offered_ids = {c.instrument_id for c in candidates}
    portfolio = constructor().construct(candidates, exposure_target(), AS_OF, equity=10_000_000.0)
    assert portfolio.instrument_ids <= offered_ids


def test_invariant_cash_remains_non_negative_across_a_sweep_of_scenarios() -> None:
    rng = np.random.default_rng(11)
    for _ in range(20):
        n = rng.integers(1, 10)
        candidates = [
            candidate(
                f"NSE:R{i}",
                rank=i + 1,
                score=float(rng.normal(0, 3)),
                volatility=float(rng.uniform(0.05, 0.5)),
                liquidity_inr=float(rng.uniform(1e6, 1e9)),
            )
            for i in range(n)
        ]
        target = float(rng.uniform(0.0, 1.0))
        portfolio = constructor().construct(
            candidates,
            exposure_target(target=target, min_gross_exposure=0.0, max_gross_exposure=1.0),
            AS_OF,
            equity=float(rng.uniform(1e6, 1e8)),
        )
        assert portfolio.cash_weight >= -1e-9


def test_target_portfolio_rejects_duplicate_instrument_ids() -> None:
    position = TargetPosition("NSE:A", "A", 0.1, "SEC", 1, 1.0, "unconstrained")
    with pytest.raises(ValueError, match="duplicate"):
        TargetPortfolio(AS_OF, (position, position), 0.8, AllocationRegime.NORMAL_RISK, 0.2)


def test_target_portfolio_rejects_negative_cash() -> None:
    position = TargetPosition("NSE:A", "A", 0.5, "SEC", 1, 1.0, "unconstrained")
    with pytest.raises(ValueError, match="cash_weight"):
        TargetPortfolio(AS_OF, (position,), -0.1, AllocationRegime.NORMAL_RISK, 0.5)


def test_target_portfolio_rejects_inconsistent_gross_exposure() -> None:
    position = TargetPosition("NSE:A", "A", 0.5, "SEC", 1, 1.0, "unconstrained")
    with pytest.raises(ValueError, match="gross_exposure"):
        TargetPortfolio(AS_OF, (position,), 0.5, AllocationRegime.NORMAL_RISK, 0.9)


def test_target_portfolio_rejects_weights_not_summing_to_one() -> None:
    position = TargetPosition("NSE:A", "A", 0.5, "SEC", 1, 1.0, "unconstrained")
    with pytest.raises(ValueError, match="sum to"):
        TargetPortfolio(AS_OF, (position,), 0.2, AllocationRegime.NORMAL_RISK, 0.5)


def test_target_position_rejects_non_positive_weight() -> None:
    with pytest.raises(ValueError, match="target_weight"):
        TargetPosition("NSE:A", "A", 0.0, "SEC", 1, 1.0, "unconstrained")


def test_target_position_rejects_weight_above_one() -> None:
    with pytest.raises(ValueError, match="target_weight"):
        TargetPosition("NSE:A", "A", 1.5, "SEC", 1, 1.0, "unconstrained")


def test_construct_rejects_non_positive_equity() -> None:
    with pytest.raises(ValueError, match="equity"):
        constructor().construct(make_candidates(2), exposure_target(), AS_OF, equity=0.0)


# --------------------------------------------------------------------------
# The shipped production config actually works end to end
# --------------------------------------------------------------------------


def test_default_settings_portfolio_config_runs_end_to_end() -> None:
    from config.loader import load_settings

    settings = load_settings()
    candidates = make_candidates(8)
    pc = PortfolioConstructor(
        settings.portfolio,
        settings.selection,
        settings.execution,
        market_with_no_correlation_data(),
    )
    portfolio = pc.construct(candidates, exposure_target(), AS_OF, equity=50_000_000.0)
    assert portfolio.gross_exposure <= 0.75 + 1e-9
    assert portfolio.cash_weight >= 0.0
