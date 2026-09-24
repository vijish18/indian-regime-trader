"""Disk cache for ``StockSelector.select``, shared across strategies and processes.

Profiling one fold of the walk-forward put 302 of 310 seconds inside
``select()``. Two facts make almost all of that avoidable without changing a
single number:

1. **Selection does not depend on the strategy.** The selector has no access
   to the regime, the exposure target or any risk state -- that separation is
   deliberate (see ``universe/__init__.py``) and it is what lets a
   walk-forward measure the regime layer at all. So the five strategies ask
   for identical rankings on identical dates, and the run computed each one
   five times.

2. **Selection on one date does not depend on selection on another.** The
   backtest's equity chains across sessions; the ranking does not. That makes
   it embarrassingly parallel, where the backtest itself is not.

So rankings are computed once per date, in parallel, and read back by every
strategy. ``scripts/precompute_selections.py`` fills the cache; a miss during
a run computes and stores, so a partial cache is still correct.

## When a cached ranking is stale

A ranking is a function of the selection and universe config, the price and
reference data, and the code that computes it. All of those feed a
fingerprint, and the fingerprint names the directory, so a change to any of
them simply looks in a different, empty directory -- stale entries are never
read, and nothing has to remember to clear them:

* ``selection`` and ``universe`` config blocks
* ``corporate_actions.csv``, ``instruments.csv``, ``index_membership.csv``
  by content hash -- the ISIN repair rewrote the first, and that must miss
* the equity-bar directory by file count and newest modification time
* the source of every module on the ranking path
* the validator's snapshot date

Writes are atomic (temp file, then rename), so parallel workers can share one
directory without a reader ever seeing half a file.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from universe.stock_selector import StockScore, StockSelector

REPO_ROOT = Path(__file__).resolve().parents[1]

RANKING_SOURCES = (
    "universe/stock_selector.py",
    "universe/factor_calculator.py",
    "universe/universe.py",
    "universe/bhavcopy_universe.py",
    "data/market_data.py",
    "data/corporate_actions.py",
    "data/models.py",
    "data/storage.py",
)
REFERENCE_FILES = ("corporate_actions.csv", "instruments.csv", "index_membership.csv")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_fingerprint(
    settings: Any,
    snapshot_date: dt.date,
    reference_dir: Path,
    bars_dir: Path,
    sources: Iterable[str] = RANKING_SOURCES,
) -> str:
    """Everything that could change a ranking, hashed."""
    bar_files = list(bars_dir.glob("*.csv")) if bars_dir.is_dir() else []
    payload = {
        "selection": settings.selection.model_dump(mode="json"),
        "universe": settings.universe.model_dump(mode="json"),
        "snapshot_date": snapshot_date.isoformat(),
        "reference": {
            name: _sha(reference_dir / name)
            for name in REFERENCE_FILES
            if (reference_dir / name).is_file()
        },
        "bars": {
            "files": len(bar_files),
            "newest_mtime_ns": max((f.stat().st_mtime_ns for f in bar_files), default=0),
        },
        "sources": {
            name: _sha(REPO_ROOT / name) for name in sources if (REPO_ROOT / name).is_file()
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CachingStockSelector:
    """A ``StockSelector`` whose ``select`` reads from and writes to disk.

    Everything else is delegated untouched, so it can stand in wherever a
    selector is expected. Only ``select`` is cached because only ``select``
    is on the hot path; ``build_candidate_universe`` and ``score_candidates``
    are for inspection and stay live.
    """

    def __init__(self, inner: StockSelector, cache_dir: Path, fingerprint: str) -> None:
        self._inner = inner
        self.directory = cache_dir / fingerprint[:20]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def path_for(self, as_of: dt.date) -> Path:
        return self.directory / f"{as_of.isoformat()}.pkl"

    def has(self, as_of: dt.date) -> bool:
        return self.path_for(as_of).is_file()

    def select(self, as_of: dt.date) -> list[StockScore]:
        path = self.path_for(as_of)
        if path.is_file():
            try:
                with path.open("rb") as handle:
                    cached = pickle.load(handle)
            except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
                # A damaged entry is recomputed, never trusted and never fatal.
                cached = None
            if isinstance(cached, list):
                self.hits += 1
                return cached

        self.misses += 1
        ranked = self._inner.select(as_of)
        self._store(path, ranked)
        return ranked

    def _store(self, path: Path, ranked: list[StockScore]) -> None:
        handle, temp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as out:
                pickle.dump(ranked, out, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp, path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise
