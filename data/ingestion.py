"""Orchestrates fetching market data from a vendor, validating it via
data_quality.py, and persisting raw + normalized series via storage/.
See docs/SPECIFICATION.md section 4.

Not implemented yet (Phase 3).
"""

from __future__ import annotations

import datetime as dt


class DataIngestionPipeline:
    """Runs one ingestion cycle: fetch -> validate -> store raw -> normalize
    -> store normalized. Raw vendor data is immutable once stored.
    """

    def ingest_nifty50(self, as_of: dt.date) -> None:
        raise NotImplementedError("Phase 3: data ingestion is not implemented yet.")

    def ingest_india_vix(self, as_of: dt.date) -> None:
        raise NotImplementedError("Phase 3: data ingestion is not implemented yet.")

    def ingest_equity_universe(self, as_of: dt.date) -> None:
        raise NotImplementedError("Phase 3: data ingestion is not implemented yet.")

    def ingest_instrument_master(self, as_of: dt.date) -> None:
        raise NotImplementedError("Phase 3: data ingestion is not implemented yet.")

    def ingest_corporate_actions(self, as_of: dt.date) -> None:
        raise NotImplementedError("Phase 3: data ingestion is not implemented yet.")
