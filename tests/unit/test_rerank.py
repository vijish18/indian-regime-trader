"""Tests for scripts/_rerank.py -- the lock and the dating of a recomputation.

``rerun_ranking`` itself runs the real selector over real history and is not
exercised here; what is testable without minutes of compute is the machinery
that decides *whether* and *for when* it runs, which is where a mistake would
either block every future rerank or point one at a date with no data.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pytest

from scripts._rerank import LOCK_STALE_MINUTES, rerank_lock


def test_the_lock_is_taken_and_released(tmp_path: Path) -> None:
    path = tmp_path / "rerank.lock"

    with rerank_lock(path) as acquired:
        assert acquired is True
        assert path.is_file()

    assert not path.exists()


def test_a_second_holder_is_refused_rather_than_made_to_wait(tmp_path: Path) -> None:
    """A refresh runs every few minutes and a rerank takes longer than that.
    Blocking would pile refreshes up behind it; skipping lets the price
    update go out on time and the rerank happen on the next one."""
    path = tmp_path / "rerank.lock"

    with rerank_lock(path) as outer:
        assert outer is True
        with rerank_lock(path) as inner:
            assert inner is False
        # The refused attempt must not have released the holder's lock.
        assert path.is_file()


def test_the_lock_is_released_even_when_the_rerank_raises(tmp_path: Path) -> None:
    path = tmp_path / "rerank.lock"

    with pytest.raises(RuntimeError):
        with rerank_lock(path):
            raise RuntimeError("selector blew up")

    assert not path.exists()


def test_a_stale_lock_is_broken(tmp_path: Path) -> None:
    """Otherwise one crashed rerank blocks every future one for good."""
    path = tmp_path / "rerank.lock"
    path.write_text("999999 old\n", encoding="utf-8")
    stale = dt.datetime.now() - dt.timedelta(minutes=LOCK_STALE_MINUTES + 5)
    os.utime(path, (stale.timestamp(), stale.timestamp()))

    with rerank_lock(path) as acquired:
        assert acquired is True


def test_a_fresh_lock_is_respected(tmp_path: Path) -> None:
    path = tmp_path / "rerank.lock"
    path.write_text("999999 recent\n", encoding="utf-8")
    recent = dt.datetime.now() - dt.timedelta(minutes=1)
    os.utime(path, (recent.timestamp(), recent.timestamp()))

    with rerank_lock(path) as acquired:
        assert acquired is False

    # A refused attempt leaves the existing lock alone.
    assert path.is_file()


def test_latest_session_comes_from_the_data_not_the_clock() -> None:
    """The bhavcopy for a session lands after it closes, and on a holiday
    there is none at all. Asking the selector for a date it has no data for
    fails the whole rerank."""
    from scripts._rerank import latest_session

    asked = dt.date(2026, 9, 21)
    found = latest_session(asked)

    assert found <= asked
    assert found.weekday() < 5
