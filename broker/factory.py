"""Builds the ``Broker`` this run should use. The one place that decides
paper vs. live, so nothing else in the codebase branches on
``settings.execution.mode`` or constructs a concrete adapter directly.

**Default mode stays PAPER**, and constructing a live-capable broker needs
four independent, explicit confirmations, none of which alone is enough:

1. ``settings.execution.mode == "live"`` in configuration,
2. ``enable_live_trading=True`` passed explicitly to :func:`build_broker`
   by the caller,
3. ``preflight_confirmed=True`` passed explicitly, attesting the caller
   just ran the full pre-live checklist (Phase 22,
   ``live.preflight.run_preflight``, ``docs/PRE_LIVE_CHECKLIST.md``) and
   it reported ``passed`` -- a remembered or hardcoded ``True`` from an
   earlier day defeats the entire point of this gate, and
4. ``settings.compliance`` (``config.models.ComplianceConfig``) passing
   every check in ``broker.compliance.ComplianceGate`` -- broker
   authorization confirmed, a real static IP and algo identifier on
   record, not the ``settings.yaml`` placeholders (Phase 16,
   ``docs/COMPLIANCE.md``).

A config-file value alone (easy to typo, easy to leave set from a prior
session) must never be sufficient to place a real order -- the second,
third and fourth, independently-checked confirmations are deliberate
friction, not an oversight. Credentials (``BROKER_API_KEY``/
``BROKER_API_SECRET``) are read from the environment, never from
``settings.yaml`` (see ``.env.example`` and ``config/loader.py``'s own
documented secrets convention).

Deliberately *not* wired the other way: this module never imports
``live.preflight`` and never re-runs the checklist itself (that would
mean this module executing this repository's own test suite as a side
effect of constructing a broker, and would tie ``broker/`` -- an earlier
layer -- to ``live/``, a later one; see docs/ARCHITECTURE.md's dependency
rules). ``preflight_confirmed`` is a plain boolean, exactly like
``enable_live_trading`` already is: the caller is trusted to have
obtained it honestly, the same threat model this codebase already applies
to that flag.
"""

from __future__ import annotations

import os

from backtest.costs import CostModel
from broker.adapters.paper_broker import PaperBroker
from broker.base import Broker
from broker.compliance import ComplianceGate, ComplianceGuardedBroker
from config.models import Settings
from data.interfaces import MarketDataProvider
from execution.position_tracker import PositionTracker


class BrokerFactoryError(RuntimeError):
    """The configured broker could not be constructed -- missing
    credentials, an unsupported provider, or a live request that was not
    explicitly confirmed. A compliance failure raises
    ``broker.compliance.ComplianceError`` instead (a more specific,
    still-hard failure), not this exception."""


def build_broker(
    settings: Settings,
    market_data: MarketDataProvider,
    cost_model: CostModel,
    initial_cash: float,
    *,
    position_tracker: PositionTracker | None = None,
    enable_live_trading: bool = False,
    preflight_confirmed: bool = False,
) -> Broker:
    if settings.execution.mode != "live":
        return PaperBroker(
            market_data=market_data,
            cost_model=cost_model,
            execution_config=settings.execution,
            paper_config=settings.paper_trading,
            position_tracker=position_tracker or PositionTracker(),
            initial_cash=initial_cash,
        )

    if settings.broker.provider != "zerodha":
        raise BrokerFactoryError(
            f"execution.mode is 'live' but broker.provider={settings.broker.provider!r} "
            "has no live adapter implemented (only 'zerodha' exists)"
        )
    if not enable_live_trading:
        raise BrokerFactoryError(
            "execution.mode is 'live' but enable_live_trading was not passed as True to "
            "build_broker() -- this is a deliberate second confirmation on top of "
            "settings.yaml, since a config-file value alone must never be enough to place "
            "a real order"
        )
    if not preflight_confirmed:
        raise BrokerFactoryError(
            "execution.mode is 'live' but preflight_confirmed was not passed as True -- "
            "this is a third, independent confirmation (Phase 22): run "
            "`python -m app.cli preflight` (or live.preflight.run_preflight() directly), "
            "confirm the result's .passed is True, and pass that current result here. "
            "A remembered or hardcoded True defeats the entire point of this gate -- see "
            "docs/PRE_LIVE_CHECKLIST.md."
        )

    # Fourth gate: refuses to construct anything further if compliance
    # configuration is missing or still a placeholder (Phase 16). Raises
    # broker.compliance.ComplianceError, not caught here -- a compliance
    # failure must propagate as exactly what it is, not be laundered into
    # a generic BrokerFactoryError.
    gate = ComplianceGate(settings.compliance)

    api_key = os.environ.get("BROKER_API_KEY")
    api_secret = os.environ.get("BROKER_API_SECRET")
    if not api_key or not api_secret:
        raise BrokerFactoryError(
            "BROKER_API_KEY and BROKER_API_SECRET must both be set in the environment "
            "(see .env.example) to construct a live KiteBroker"
        )

    from broker.zerodha.kite_broker import KiteBroker

    kite_broker = KiteBroker(api_key=api_key, api_secret=api_secret, enable_live_trading=True)
    return ComplianceGuardedBroker(inner=kite_broker, gate=gate)
