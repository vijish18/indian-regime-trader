"""Recompute the ranking and the regime for a paper book that has run down.

Called from ``scripts/refresh_live_book.py`` when stops have taken the book
below ``paper_book.rerank_at_positions``. At that point the account is mostly
cash held against a ranking that is several stops old: the market that
stopped those names out is not the one the selector last looked at, and
reaching further down that list only buys names it already passed over.

This is the expensive path -- it runs the real stock selector over real
history and re-filters the approved regime model, which takes minutes, not
the second a price refresh takes. It is deliberately rare, rate-limited by
``paper_book.rerank_cooldown_minutes``, and guarded by a lock so two
overlapping refreshes cannot both start one.

**Nothing is refitted.** The regime model is the approved artifact, filtered
forward; the ranking is the same selector the backtest ran. A "rerank" is
asking today's question of the deployed system, not training a new one.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

LOCK_STALE_MINUTES = 30
"""A lock older than this is assumed to belong to a run that died. Without
this a crashed rerank would block every future one for good."""


class RerankUnavailable(RuntimeError):
    """A ranking could not be recomputed. The caller keeps the old one."""


@contextmanager
def rerank_lock(path: Path) -> Iterator[bool]:
    """Yield True if this process took the lock, False if another holds it.

    A refresh runs every few minutes and a rerank takes longer than that, so
    without this two of them would run the selector at once and race to write
    the same payload.
    """
    try:
        if path.is_file():
            age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
            if age < dt.timedelta(minutes=LOCK_STALE_MINUTES):
                yield False
                return
            path.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{os.getpid()} {dt.datetime.now().isoformat()}\n", encoding="utf-8")
    except OSError as exc:
        raise RerankUnavailable(f"could not take the rerank lock: {exc}") from exc
    try:
        yield True
    finally:
        path.unlink(missing_ok=True)


def latest_session(as_of: dt.date, lookback_days: int = 15) -> dt.date:
    """The most recent session the local data actually covers.

    Not ``date.today()``: the bhavcopy for a session lands after it closes,
    and on a holiday or before the day's download there is none at all.
    Asking the selector for a date it has no data for fails the whole rerank,
    so the date comes from the data rather than the clock.
    """
    from data.market_data import LocalMarketDataProvider
    from data.storage import LocalDataStore, StorageFormat

    store = LocalDataStore(
        raw_root=REPO_ROOT / "data_cache" / "raw",
        normalized_root=REPO_ROOT / "data_cache" / "normalized",
        reference_root=REPO_ROOT / "data_cache" / "reference",
        storage_format=StorageFormat.CSV,
    )
    market = LocalMarketDataProvider(store)
    try:
        observations = market.get_index_observations(
            "NIFTY50", as_of - dt.timedelta(days=lookback_days), as_of
        )
    except Exception as exc:  # noqa: BLE001 - reported, never guessed
        raise RerankUnavailable(f"no index history to date the rerank: {exc}") from exc
    if not observations:
        raise RerankUnavailable(
            f"no NIFTY50 session in the {lookback_days} days to {as_of.isoformat()}"
        )
    return observations[-1].session_date


def rerun_ranking(as_of: dt.date) -> dict[str, Any]:
    """``{"selection": ..., "regime_now": ...}`` for ``as_of``.

    Raises :class:`RerankUnavailable` rather than returning a partial result:
    a book that refills against half a ranking is worse off than one that
    keeps the ranking it had and leaves the cash alone.
    """
    from scripts._current_regime import current_regime
    from scripts.collect_dashboard_data import stock_selection

    selection = stock_selection(as_of.isoformat())
    if not selection.get("available"):
        raise RerankUnavailable(str(selection.get("reason") or "selector returned nothing"))
    picks = selection.get("picks") or []
    if not picks:
        raise RerankUnavailable("selector returned an empty ranking")
    return {"selection": selection, "regime_now": current_regime(as_of)}
