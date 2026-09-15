"""Deduplicates broker fills and applies each one to a canonical
``PositionTracker`` exactly once (Phase 19, step 15/16: "track fills" /
"update portfolio").

Why this exists as its own module rather than being inline in the
orchestrator: ``broker.adapters.paper_broker.PaperBroker`` needs *some*
``PositionTracker`` instance of its own to validate its own fills (reject
a sell beyond held quantity, etc.) regardless of what the orchestrator
does with the result -- so ``Orchestrator`` never shares its own canonical
``PositionTracker`` with the broker it constructs. Instead, fills reach
the canonical tracker uniformly for every broker type (paper or live)
through this one path: pull ``Broker.get_trades()``, keep only fills not
already applied (deduplicated by ``trade_id``, mirroring
``execution.startup.StartupSequence``'s own fill-dedup precedent), and
apply each exactly once via ``PositionTracker.apply_fill``.
"""

from __future__ import annotations

from backtest.costs import TradeSide
from broker.base import Broker, BrokerFill
from execution.position_tracker import PositionTracker


class FillTracker:
    """Applies each broker fill to ``position_tracker`` exactly once."""

    def __init__(self, position_tracker: PositionTracker) -> None:
        self.position_tracker = position_tracker
        self._applied_fill_ids: set[str] = set()

    def poll(self, broker: Broker) -> list[BrokerFill]:
        """Fetches every fill the broker currently reports, applies the
        ones not yet seen to ``position_tracker``, and returns just the
        newly-applied fills (empty if nothing new)."""
        fills = broker.get_trades()
        new_fills = [fill for fill in fills if fill.trade_id not in self._applied_fill_ids]
        for fill in new_fills:
            side = TradeSide.BUY if fill.side.lower() == "buy" else TradeSide.SELL
            self.position_tracker.apply_fill(
                fill.instrument_id, fill.quantity, fill.price, side, fill.as_of
            )
            self._applied_fill_ids.add(fill.trade_id)
        return new_fills

    def seen_fill_ids(self) -> set[str]:
        return set(self._applied_fill_ids)

    def mark_seen(self, trade_ids: set[str]) -> None:
        """Seeds already-applied fills without re-applying them -- used at
        startup when the canonical ``PositionTracker`` was rebuilt from a
        persisted snapshot that already reflects fills up to some point,
        so those same fills must not be double-counted the next time
        ``poll`` runs."""
        self._applied_fill_ids |= trade_ids
