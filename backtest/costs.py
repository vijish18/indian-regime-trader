"""Indian transaction cost and slippage model. See
docs/SPECIFICATION.md section 9.

This module is the single source of cost assumptions shared by both the
backtester and any live/paper pre-trade cost estimate (broker/execution),
so the two can never silently drift apart. Rates are configuration-driven
via ``config.models.CostsConfig`` because they change over time.

Delivery (CNC) equity STT is charged on both the buy and sell leg, unlike
intraday trades -- this V1 system is delivery-only long-only, so both legs
apply (docs/ARCHITECTURE.md).

Not implemented yet (Phase 8/9).
"""

from __future__ import annotations

from dataclasses import dataclass

from config.models import CostsConfig


@dataclass(frozen=True)
class TradeCostBreakdown:
    brokerage: float
    stt: float
    exchange_txn_charge: float
    sebi_turnover_fee: float
    gst: float
    stamp_duty: float
    dp_charge: float
    slippage: float
    total: float


class IndianCostModel:
    def __init__(self, config: CostsConfig) -> None:
        self.config = config

    def compute_costs(self, side: str, quantity: int, price: float) -> TradeCostBreakdown:
        """Full statutory + brokerage cost breakdown for one delivery-equity
        trade leg (excludes slippage; see ``estimate_slippage``).
        """
        raise NotImplementedError("Phase 8/9: cost modeling is not implemented yet.")

    def estimate_slippage(
        self,
        order_value: float,
        spread_bps: float,
        avg_daily_value: float,
        volatility: float,
    ) -> float:
        """``max(min_bps, 0.5 * spread_bps + impact_bps(...))`` research
        slippage model (docs/SPECIFICATION.md section 9.1).
        """
        raise NotImplementedError("Phase 8/9: slippage modeling is not implemented yet.")
