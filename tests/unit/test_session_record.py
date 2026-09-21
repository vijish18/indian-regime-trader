"""Persisting one trading session's reasoning.

Paper trading exists to be read back afterwards, so the test that matters
is not "a file appeared" but "the chain from candidate to order survived".
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path

from orchestration.session_record import load_sessions, record_cycle, to_record


@dataclasses.dataclass
class _Factors:
    momentum: float = 1.0
    trend_persistence: float = 0.5
    relative_strength: float = 0.2
    volatility: float = -0.3


@dataclasses.dataclass
class _Score:
    rank: int
    symbol: str
    instrument_id: str
    score: float
    raw_factors: _Factors = dataclasses.field(default_factory=_Factors)
    standardized_factors: _Factors = dataclasses.field(default_factory=_Factors)


@dataclasses.dataclass
class _Position:
    instrument_id: str
    target_weight: float
    rank: int


@dataclasses.dataclass
class _Portfolio:
    positions: tuple[_Position, ...]


@dataclasses.dataclass
class _Violation:
    check: str


@dataclasses.dataclass
class _Decision:
    instrument_id: str
    approved: bool
    target_weight: float
    circuit_state: str = "normal"
    violations: tuple[_Violation, ...] = ()


@dataclasses.dataclass
class _Report:
    as_of: dt.date = dt.date(2026, 9, 21)
    state: str = "running"
    is_trading_day: bool = True
    permit_trading: bool = True
    candidates: tuple[_Score, ...] = ()
    target_portfolio: _Portfolio | None = None
    risk_decisions: tuple[_Decision, ...] = ()
    sized_trades: tuple[object, ...] = ()
    submitted_order_ids: tuple[str, ...] = ()
    skipped_trades: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()
    fail_closed_reason: str | None = None


def _report() -> _Report:
    return _Report(
        candidates=(_Score(1, "CUPID", "NSE:CUPID", 2.2),),
        target_portfolio=_Portfolio((_Position("NSE:CUPID", 0.10, 1),)),
        risk_decisions=(
            _Decision("NSE:CUPID", True, 0.10),
            _Decision("NSE:RISKY", False, 0.08, violations=(_Violation("max_single_name"),)),
        ),
        submitted_order_ids=("ord-1",),
        skipped_trades=("NSE:RISKY",),
    )


def test_the_whole_chain_from_candidate_to_order_is_kept() -> None:
    """A list of fills cannot answer "why did it buy this" -- it shows what
    happened, never what the alternatives were."""
    record = to_record(_report())

    assert record["candidates"][0]["symbol"] == "CUPID"
    assert record["candidates"][0]["z"]["momentum"] == 1.0
    assert record["target_positions"][0]["target_weight"] == 0.10
    assert record["submitted_order_ids"] == ["ord-1"]


def test_a_veto_records_which_check_stopped_it() -> None:
    """"Rejected" without a reason cannot be acted on."""
    record = to_record(_report())

    vetoed = [d for d in record["risk_decisions"] if not d["approved"]]
    assert vetoed[0]["instrument_id"] == "NSE:RISKY"
    assert vetoed[0]["violations"] == ["max_single_name"]


def test_a_halted_session_is_recorded_too(tmp_path: Path) -> None:
    """A day the system refused to trade is evidence, not an absence. If
    only trading days were written, a fortnight of paper trading could not
    distinguish "nothing to do" from "it never ran"."""
    halted = _Report(permit_trading=False, fail_closed_reason="stale_market_data")

    path = record_cycle(halted, tmp_path)

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["permit_trading"] is False
    assert written["fail_closed_reason"] == "stale_market_data"


def test_a_second_run_never_overwrites_the_first(tmp_path: Path) -> None:
    """A re-run for the same date is a real event -- a restart, an operator
    retrying after a halt -- and overwriting would erase whichever one
    mattered."""
    first = record_cycle(_report(), tmp_path)
    second = record_cycle(_report(), tmp_path)

    assert first != second
    assert first.exists() and second.exists()
    assert "run2" in second.name


def test_sessions_load_back_in_order(tmp_path: Path) -> None:
    record_cycle(_Report(as_of=dt.date(2026, 9, 22)), tmp_path)
    record_cycle(_Report(as_of=dt.date(2026, 9, 21)), tmp_path)

    loaded = load_sessions(tmp_path)

    assert [r["as_of"] for r in loaded] == ["2026-09-21", "2026-09-22"]


def test_loading_an_empty_directory_is_not_an_error(tmp_path: Path) -> None:
    assert load_sessions(tmp_path / "nothing") == []
