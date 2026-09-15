"""Builds the ``Broker`` this run should use. The one place that decides
paper vs. live, so nothing else in the codebase branches on
``settings.execution.mode`` or constructs a concrete adapter directly.

**Default mode stays PAPER**, and constructing a live-capable broker
needs two independent, explicit confirmations, neither of which alone is
enough:

1. ``settings.execution.mode == "live"`` in configuration, and
2. ``enable_live_trading=True`` passed explicitly to :func:`build_broker`
   by the caller.

A config-file value alone (easy to typo, easy to leave set from a prior
session) must never be sufficient to place a real order -- the second,
code-level confirmation is deliberate friction, not an oversight.
Credentials (``BROKER_API_KEY``/``BROKER_API_SECRET``) are read from the
environment, never from ``settings.yaml`` (see ``.env.example`` and
``config/loader.py``'s own documented secrets convention).
"""

from __future__ import annotations

import os

from backtest.costs import CostModel
from broker.adapters.paper_broker import PaperBroker
from broker.base import Broker
from config.models import Settings
from data.interfaces import MarketDataProvider
from execution.position_tracker import PositionTracker


class BrokerFactoryError(RuntimeError):
    """The configured broker could not be constructed -- missing
    credentials, an unsupported provider, or a live request that was not
    explicitly confirmed."""


def build_broker(
    settings: Settings,
    market_data: MarketDataProvider,
    cost_model: CostModel,
    initial_cash: float,
    *,
    position_tracker: PositionTracker | None = None,
    enable_live_trading: bool = False,
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

    api_key = os.environ.get("BROKER_API_KEY")
    api_secret = os.environ.get("BROKER_API_SECRET")
    if not api_key or not api_secret:
        raise BrokerFactoryError(
            "BROKER_API_KEY and BROKER_API_SECRET must both be set in the environment "
            "(see .env.example) to construct a live KiteBroker"
        )

    from broker.zerodha.kite_broker import KiteBroker

    return KiteBroker(api_key=api_key, api_secret=api_secret, enable_live_trading=True)
