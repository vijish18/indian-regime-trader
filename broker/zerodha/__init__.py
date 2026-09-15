"""Zerodha Kite Connect v3 adapter -- the one concrete implementation of
``broker.base.Broker`` built against a real, live broker's published API
(https://kite.trade/docs/connect/v3/). Everything Kite-specific (request
shapes, response field names, status vocabulary, WebSocket framing) is
isolated inside this package; no other module in this codebase imports
from here except ``broker/factory.py``.
"""

from __future__ import annotations
