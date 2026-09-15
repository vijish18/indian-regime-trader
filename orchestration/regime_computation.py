"""Computes today's market regime and exposure target from an already
fitted, already approved model (Phase 19, step 9: "compute market
regime"). Contains no HMM/allocation mathematics of its own -- it only
pulls the trailing window of index data a live inference call needs and
hands it to ``core.regime.hmm_engine.HMMRegimeEngine`` and
``core.regime.allocation.RegimeAllocationEngine``, exactly the same two
calls ``backtest.walk_forward.WalkForwardValidator._hmm_exposure_targets``
already makes for its own out-of-sample inference -- this module exists so
live orchestration does not need to import a backtesting-only class to get
that behavior.
"""

from __future__ import annotations

import datetime as dt

from config.models import AllocationConfig, HMMConfig
from core.features.feature_engineering import (
    FeaturePipeline,
    MarketFeatureInputs,
    drop_warmup_rows,
)
from core.features.feature_scaler import CausalFeatureScaler, ScalerParams
from core.regime.allocation import AllocationTarget, RegimeAllocationEngine
from core.regime.hmm_engine import FittedRegimeModel, HMMRegimeEngine, RegimeState
from core.regime.model_registry import ModelArtifact
from core.regime.regime_policy import RegimePolicy
from data.interfaces import MarketDataProvider


class RegimeComputationError(RuntimeError):
    """Raised when there is not enough trailing history to compute today's
    regime -- fail-closed, the caller must not proceed to sizing/orders
    without a regime decision."""


class RegimeComputer:
    def __init__(
        self,
        market_data: MarketDataProvider,
        hmm_config: HMMConfig,
        allocation_config: AllocationConfig,
        regime_policy: RegimePolicy,
        feature_pipeline: FeaturePipeline,
        index_symbol: str,
        vix_symbol: str,
        feature_warmup_buffer_days: int = 500,
    ) -> None:
        self.market_data = market_data
        self.hmm_config = hmm_config
        self.allocation_config = allocation_config
        self.regime_policy = regime_policy
        self.feature_pipeline = feature_pipeline
        self.index_symbol = index_symbol
        self.vix_symbol = vix_symbol
        self.feature_warmup_buffer_days = feature_warmup_buffer_days

    def compute_today(
        self, artifact: ModelArtifact, as_of: dt.date
    ) -> tuple[AllocationTarget, RegimeState]:
        """Filters (never smooths, never re-fits) up to and including
        ``as_of`` using ``artifact``'s already-frozen model and scaler, and
        turns the resulting regime history into today's exposure target.
        """
        states = self._filter_history(artifact.model, artifact.scaler, as_of)
        if not states:
            raise RegimeComputationError(
                f"no regime states could be computed through {as_of} -- insufficient "
                "trailing index history"
            )
        if states[-1].as_of != as_of:
            raise RegimeComputationError(
                f"latest computable regime state is dated {states[-1].as_of}, not {as_of} "
                "-- index data for today is not yet available"
            )
        engine = RegimeAllocationEngine(self.hmm_config, self.allocation_config, self.regime_policy)
        target = engine.evaluate(states)
        return target, states[-1]

    def _filter_history(
        self, model: FittedRegimeModel, scaler_params: ScalerParams, as_of: dt.date
    ) -> list[RegimeState]:
        buffer_start = as_of - dt.timedelta(days=self.feature_warmup_buffer_days)
        nifty = self.market_data.get_index_observations(self.index_symbol, buffer_start, as_of)
        vix = self.market_data.get_index_observations(self.vix_symbol, buffer_start, as_of)
        inputs = MarketFeatureInputs.from_index_observations(nifty, vix)
        matrix = drop_warmup_rows(self.feature_pipeline.compute(inputs))
        if matrix.empty:
            return []
        scaler = CausalFeatureScaler()
        scaled = scaler.transform(matrix, scaler_params)
        engine = HMMRegimeEngine(self.hmm_config)
        return engine.filter(model, scaled)
