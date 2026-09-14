"""Paper-trading broker adapter: simulates fills against real market data
without sending any order to a real exchange. Required before live capital
is enabled (docs/SPECIFICATION.md section 2 and section 20).

Not implemented yet (Phase 10/11).
"""

from __future__ import annotations

from broker.base import (
    Account,
    Broker,
    BrokerOrder,
    BrokerPosition,
    BrokerQuote,
    HealthStatus,
)


class PaperBroker(Broker):
    """Simulated broker used for paper trading and backtesting-adjacent
    dry runs. Must apply the same cost/slippage model as backtest/costs.py
    so paper results are comparable to backtest results.
    """

    def get_account(self) -> Account:
        raise NotImplementedError("Phase 10/11: paper broker is not implemented yet.")

    def get_positions(self) -> list[BrokerPosition]:
        raise NotImplementedError("Phase 10/11: paper broker is not implemented yet.")

    def get_open_orders(self) -> list[BrokerOrder]:
        raise NotImplementedError("Phase 10/11: paper broker is not implemented yet.")

    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]:
        raise NotImplementedError("Phase 10/11: paper broker is not implemented yet.")

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        raise NotImplementedError("Phase 10/11: paper broker is not implemented yet.")

    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder:
        raise NotImplementedError("Phase 10/11: paper broker is not implemented yet.")

    def cancel_order(self, order_id: str) -> BrokerOrder:
        raise NotImplementedError("Phase 10/11: paper broker is not implemented yet.")

    def close_position(self, instrument_id: str) -> BrokerOrder:
        raise NotImplementedError("Phase 10/11: paper broker is not implemented yet.")

    def close_all_positions(self) -> list[BrokerOrder]:
        raise NotImplementedError("Phase 10/11: paper broker is not implemented yet.")

    def health_check(self) -> HealthStatus:
        raise NotImplementedError("Phase 10/11: paper broker is not implemented yet.")
