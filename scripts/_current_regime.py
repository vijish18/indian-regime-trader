"""What regime the market is in right now, per the approved model.

The dashboard showed the model's *states* -- their volatilities, the
transition matrix -- but never which one is live today, which is the single
thing a person looks at a regime system to find out.

Inference runs the same way live trading would: the approved artifact,
filtered forward over recent sessions, reading the last row. Nothing is
refitted, so what appears here is what the deployed model believes, not a
fresh fit that happens to agree.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def current_regime(as_of: dt.date, history_days: int = 600) -> dict[str, Any]:
    try:
        from config.loader import load_settings
        from core.features.feature_engineering import (
            FeaturePipeline,
            MarketFeatureInputs,
            build_default_feature_definitions,
            drop_warmup_rows,
        )
        from core.features.feature_scaler import CausalFeatureScaler
        from core.regime.hmm_engine import HMMRegimeEngine, assign_labels
        from core.regime.model_registry import ModelRegistry
        from data.corporate_actions import InMemoryCorporateActionProvider
        from data.market_data import LocalMarketDataProvider
        from data.storage import LocalDataStore, StorageFormat

        settings = load_settings()
        artifact = ModelRegistry(REPO_ROOT / "model_registry").load_current_approved()
        store = LocalDataStore(
            raw_root=REPO_ROOT / "data_cache" / "raw",
            normalized_root=REPO_ROOT / "data_cache" / "normalized",
            reference_root=REPO_ROOT / "data_cache" / "reference",
            storage_format=StorageFormat.CSV,
        )
        actions = InMemoryCorporateActionProvider.from_file(
            REPO_ROOT / "data_cache" / "reference" / "corporate_actions.csv"
        )
        market = LocalMarketDataProvider(store, corporate_actions=actions)

        start = as_of - dt.timedelta(days=history_days)
        pipeline = FeaturePipeline(build_default_feature_definitions(settings.features))
        matrix = drop_warmup_rows(
            pipeline.compute(
                MarketFeatureInputs.from_index_observations(
                    market.get_index_observations("NIFTY50", start, as_of),
                    market.get_index_observations("INDIAVIX", start, as_of),
                )
            )
        )
        scaled = CausalFeatureScaler().transform(matrix, artifact.scaler)
        states = HMMRegimeEngine(settings.hmm).filter(artifact.model, scaled)
    except Exception as exc:  # noqa: BLE001 - reported, never faked
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    if not states:
        return {"available": False, "reason": "no filtered states"}

    model = artifact.model
    labels = assign_labels(
        [model.statistics_for(i).expected_volatility for i in range(model.n_states)]
    )
    # Same risk-order display index the model panel uses, so "s3" means the
    # same thing in both places.
    order = sorted(range(model.n_states), key=lambda i: model.statistics_for(i).expected_volatility)
    display = {state_id: index for index, state_id in enumerate(order)}
    latest = states[-1]
    min_confidence = settings.hmm.min_confidence
    return {
        "available": True,
        "as_of": latest.as_of.isoformat(),
        "model_id": artifact.model_id,
        "state_id": display[latest.state_id],
        "model_state_id": latest.state_id,
        "label": labels[latest.state_id].value,
        "confidence": float(latest.confidence),
        "min_confidence": float(min_confidence),
        # Below the configured bar the call is not acted on as certain --
        # the allocation engine falls back to the more conservative regime.
        # Showing the flag beside the number stops a 53% reading being read
        # as a decision.
        "confident": bool(latest.confidence >= min_confidence),
        "expected_volatility": float(latest.expected_volatility),
        "expected_return": float(latest.expected_return),
        "persistence": float(latest.persistence),
        "probabilities": [float(latest.probabilities[i]) for i in order],
        "recent": [
            {
                "as_of": s.as_of.isoformat(),
                "state_id": display[s.state_id],
                "label": labels[s.state_id].value,
                "confidence": float(s.confidence),
            }
            for s in states[-30:]
        ],
    }
