"""Tests for universe/selection_cache.py.

The cache exists only for speed, so the one property that matters is that it
cannot change a result: a hit returns exactly what a live call returned, a
damaged entry is recomputed rather than trusted, and anything that could
change a ranking changes the directory it is looked up in.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from universe.selection_cache import CachingStockSelector, selection_fingerprint

DAY = dt.date(2021, 12, 2)


class _Inner:
    def __init__(self) -> None:
        self.calls: list[dt.date] = []
        self.config = "passthrough"

    def select(self, as_of: dt.date) -> list[Any]:
        self.calls.append(as_of)
        return [("IEX", 1, 1.5), ("CUPID", 2, 1.2), as_of.isoformat()]


def _cache(tmp_path: Path, fingerprint: str = "f" * 64) -> tuple[CachingStockSelector, _Inner]:
    inner = _Inner()
    return CachingStockSelector(inner, tmp_path, fingerprint), inner  # type: ignore[arg-type]


def test_a_hit_returns_exactly_what_the_live_call_returned(tmp_path: Path) -> None:
    cache, inner = _cache(tmp_path)
    first = cache.select(DAY)
    second = cache.select(DAY)

    assert first == second
    assert inner.calls == [DAY]
    assert (cache.hits, cache.misses) == (1, 1)


def test_a_second_selector_on_the_same_directory_shares_the_entry(tmp_path: Path) -> None:
    """The point of the cache: five strategies, one computation."""
    a, inner_a = _cache(tmp_path)
    b, inner_b = _cache(tmp_path)
    a.select(DAY)
    b.select(DAY)

    assert inner_a.calls == [DAY]
    assert inner_b.calls == []


def test_a_different_fingerprint_never_reads_the_old_entry(tmp_path: Path) -> None:
    """A config, data or code change must miss, not return a stale ranking."""
    old, _ = _cache(tmp_path, "a" * 64)
    old.select(DAY)
    new, inner = _cache(tmp_path, "b" * 64)
    new.select(DAY)

    assert inner.calls == [DAY]


def test_a_damaged_entry_is_recomputed_not_trusted(tmp_path: Path) -> None:
    cache, inner = _cache(tmp_path)
    cache.path_for(DAY).write_bytes(b"not a pickle")

    result: list[Any] = cache.select(DAY)

    assert inner.calls == [DAY]
    assert result[-1] == DAY.isoformat()


def test_other_attributes_pass_through(tmp_path: Path) -> None:
    cache, _ = _cache(tmp_path)
    assert cache.config == "passthrough"


def test_the_fingerprint_moves_with_reference_data(tmp_path: Path) -> None:
    """The ISIN repair rewrote corporate_actions.csv. Rankings computed before
    it must not be read after it."""
    ref = tmp_path / "ref"
    ref.mkdir()
    bars = tmp_path / "bars"
    bars.mkdir()
    (ref / "corporate_actions.csv").write_text("a\n", encoding="utf-8")
    settings = SimpleNamespace(
        selection=SimpleNamespace(model_dump=lambda mode: {"k": 1}),
        universe=SimpleNamespace(model_dump=lambda mode: {"u": 1}),
    )

    before = selection_fingerprint(settings, DAY, ref, bars, sources=())
    (ref / "corporate_actions.csv").write_text("b\n", encoding="utf-8")
    after = selection_fingerprint(settings, DAY, ref, bars, sources=())

    assert before != after


def test_the_fingerprint_moves_with_selection_config(tmp_path: Path) -> None:
    ref = tmp_path / "ref"
    ref.mkdir()

    def settings(weight: float) -> Any:
        return SimpleNamespace(
            selection=SimpleNamespace(model_dump=lambda mode: {"liquidity": weight}),
            universe=SimpleNamespace(model_dump=lambda mode: {}),
        )

    assert selection_fingerprint(settings(0.10), DAY, ref, ref, sources=()) != (
        selection_fingerprint(settings(0.05), DAY, ref, ref, sources=())
    )
