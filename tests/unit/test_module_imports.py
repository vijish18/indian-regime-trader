"""Every module in the project must at least import cleanly. This is a cheap
smoke test that catches broken imports, typos in type hints, and circular
imports across the nine-layer package boundary before any real logic exists.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "config",
    "config.loader",
    "config.models",
    "core",
    "core.regime",
    "core.regime.hmm_engine",
    "core.regime.regime_policy",
    "core.regime.allocation",
    "core.regime.baseline_policy",
    "core.regime.model_registry",
    "core.regime.gaussian_hmm",
    "core.features",
    "core.features.feature_engineering",
    "core.features.feature_scaler",
    "data",
    "data.calendar",
    "data.corporate_actions",
    "data.data_quality",
    "data.errors",
    "data.ingestion",
    "data.instrument_master",
    "data.interfaces",
    "data.market_data",
    "data.membership",
    "data.models",
    "data.storage",
    "universe",
    "universe.universe",
    "universe.stock_selector",
    "universe.factor_calculator",
    "portfolio",
    "portfolio.portfolio_constructor",
    "risk",
    "risk.risk_manager",
    "risk.position_sizer",
    "risk.exposure",
    "risk.circuit_breaker",
    "risk.order_validator",
    "risk.portfolio_risk_state",
    "execution",
    "execution.execution_journal",
    "execution.order_manager",
    "execution.order_reconciler",
    "execution.position_tracker",
    "execution.reconciliation",
    "execution.startup",
    "execution.system_state",
    "broker",
    "broker.base",
    "broker.compliance",
    "broker.errors",
    "broker.factory",
    "broker.adapters",
    "broker.adapters.paper_broker",
    "broker.zerodha",
    "broker.zerodha.kite_broker",
    "broker.zerodha.kite_mappings",
    "broker.zerodha.kite_ticker",
    "broker.zerodha.kite_transport",
    "backtest",
    "backtest.engine",
    "backtest.costs",
    "backtest.cost_schedule",
    "backtest.performance",
    "backtest.comparison",
    "backtest.robustness",
    "backtest.report",
    "backtest.walk_forward",
    "backtest.stress_test",
    "monitoring",
    "monitoring.logger",
    "monitoring.alerts",
    "monitoring.health",
    "monitoring.dashboard",
    "storage",
    "storage.models",
    "storage.database",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_core_does_not_import_downstream_layers() -> None:
    """Structural guard for docs/ARCHITECTURE.md's dependency rule: core/
    (regime detection + feature engineering) must not import from universe,
    portfolio, risk, execution, or broker.
    """
    import core.features.feature_engineering as fe
    import core.features.feature_scaler as fs
    import core.regime.allocation as allocation
    import core.regime.baseline_policy as baseline
    import core.regime.hmm_engine as hmm
    import core.regime.model_registry as registry
    import core.regime.regime_policy as policy

    forbidden_prefixes = ("universe", "portfolio", "risk", "execution", "broker")
    for module in (fe, fs, hmm, registry, policy, allocation, baseline):
        for name, value in vars(module).items():
            module_name = getattr(value, "__module__", "")
            assert not module_name.startswith(forbidden_prefixes), (
                f"{module.__name__}.{name} pulls in {module_name}, "
                "violating the core/ dependency rule in docs/ARCHITECTURE.md"
            )
