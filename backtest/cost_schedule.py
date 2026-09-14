"""Versioned Indian cash-equity cost schedules.

NSE, SEBI, and CDSL/NSDL publish transaction-charge, STT, turnover-fee, and
DP-charge levies *separately* from any single circular, and revise them
periodically. Treating a rate as a permanent constant baked into Python
would silently misprice every trade the day a rate changes and would
retroactively misprice historical backtests the day it changes *again*
(a later run would apply today's rate to yesterday's trade). This module's
job is to make that impossible: rates live only in the data file this
module reads (``config/cost_schedules.yaml`` by default, one entry per
change), never in code, and every cost computed elsewhere in ``backtest/``
records which dated schedule it used.

Add a new dated entry when rates change; never edit an existing one in
place -- a backtest that already ran against a schedule must remain
reproducible, the same discipline
``core/regime/model_registry.py`` applies to fitted models.

The rates shipped in ``config/cost_schedules.yaml`` are illustrative
approximations of typical NSE cash-equity delivery (CNC) costs, assembled
from publicly documented STT/GST/SEBI-fee rules and a representative
discount-broker rate card. They are **not** a substitute for the current
NSE/SEBI/CDSL circulars or your broker's actual rate card -- verify before
using this for anything beyond research backtests.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml


class MissingCostScheduleError(RuntimeError):
    """No :class:`CostSchedule` covers the requested trade date.

    Raised rather than silently falling back to the earliest or latest
    known rates, either of which would misprice a trade with rates that
    were never actually in force on that date -- fail closed
    (docs/SPECIFICATION.md's standing "no silent fallback" principle).
    """


class CostScheduleFileError(RuntimeError):
    """The cost-schedule data file is malformed, empty, or contains
    duplicate/out-of-order effective dates."""


@dataclass(frozen=True, slots=True)
class CostSchedule:
    """One dated rate card: every statutory and brokerage rate needed to
    price one trade leg, all in force as of ``effective_from`` until the
    next later-dated schedule (if any) supersedes it.

    Percent fields are fractions (``0.001`` = 0.1%), matching the rest of
    this codebase's ``Percent`` convention.
    """

    effective_from: dt.date
    label: str
    """Short human-readable description, e.g. what circular/broker plan
    this schedule represents -- for audit trails, not used in arithmetic."""

    source: str
    """Where these rates came from, for audit -- a circular reference, a
    broker rate-card URL, or an explicit "illustrative, verify before use"
    note."""

    brokerage_flat_inr: float
    brokerage_pct: float
    stt_buy_pct: float
    stt_sell_pct: float
    exchange_txn_pct: float
    sebi_turnover_pct: float
    gst_pct: float
    stamp_duty_buy_pct: float
    stamp_duty_sell_pct: float
    dp_charges_inr: float
    other_charges_flat_inr: float
    other_charges_pct: float

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("CostSchedule.label must not be empty")
        if not self.source:
            raise ValueError("CostSchedule.source must not be empty")
        for field_name in (
            "brokerage_flat_inr",
            "brokerage_pct",
            "stt_buy_pct",
            "stt_sell_pct",
            "exchange_txn_pct",
            "sebi_turnover_pct",
            "gst_pct",
            "stamp_duty_buy_pct",
            "stamp_duty_sell_pct",
            "dp_charges_inr",
            "other_charges_flat_inr",
            "other_charges_pct",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"CostSchedule.{field_name} must be >= 0, got {value}")


class CostScheduleRepository:
    """Point-in-time lookup over a set of dated :class:`CostSchedule`
    entries -- the same "most recent entry on or before the date wins"
    pattern ``data/corporate_actions.py`` and
    ``core/regime/model_registry.py`` use for their own versioned records.
    """

    def __init__(self, schedules: Iterable[CostSchedule]) -> None:
        ordered = sorted(schedules, key=lambda schedule: schedule.effective_from)
        if not ordered:
            raise CostScheduleFileError("at least one CostSchedule is required")
        seen_dates: set[dt.date] = set()
        for schedule in ordered:
            if schedule.effective_from in seen_dates:
                raise CostScheduleFileError(
                    f"duplicate effective_from date {schedule.effective_from} in cost schedules"
                )
            seen_dates.add(schedule.effective_from)
        self._schedules: tuple[CostSchedule, ...] = tuple(ordered)

    @classmethod
    def from_file(cls, path: Path) -> CostScheduleRepository:
        if not path.is_file():
            raise CostScheduleFileError(f"no cost schedule file at {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "schedules" not in raw:
            raise CostScheduleFileError(f"{path}: expected a top-level 'schedules' list")
        entries = raw["schedules"]
        if not isinstance(entries, list) or not entries:
            raise CostScheduleFileError(f"{path}: 'schedules' must be a non-empty list")
        try:
            schedules = [_schedule_from_dict(entry) for entry in entries]
        except (KeyError, TypeError, ValueError) as exc:
            raise CostScheduleFileError(f"{path}: {exc}") from exc
        return cls(schedules)

    def schedule_as_of(self, trade_date: dt.date) -> CostSchedule:
        """The most recently effective schedule on or before ``trade_date``.

        Raises :class:`MissingCostScheduleError` if every known schedule's
        ``effective_from`` is later than ``trade_date`` -- there is no rate
        card this repository can honestly attribute to that date.
        """
        applicable = [
            schedule for schedule in self._schedules if schedule.effective_from <= trade_date
        ]
        if not applicable:
            raise MissingCostScheduleError(
                f"no cost schedule is effective on or before {trade_date}; the earliest known "
                f"schedule starts {self._schedules[0].effective_from} -- add a dated entry to "
                "the cost-schedule file if this date is genuinely in scope"
            )
        return applicable[-1]

    def all_schedules(self) -> tuple[CostSchedule, ...]:
        """Every known schedule, ascending by ``effective_from``."""
        return self._schedules


def _schedule_from_dict(entry: dict[str, object]) -> CostSchedule:
    return CostSchedule(
        effective_from=_parse_date(entry["effective_from"]),
        label=str(entry["label"]),
        source=str(entry["source"]),
        brokerage_flat_inr=float(entry["brokerage_flat_inr"]),  # type: ignore[arg-type]
        brokerage_pct=float(entry["brokerage_pct"]),  # type: ignore[arg-type]
        stt_buy_pct=float(entry["stt_buy_pct"]),  # type: ignore[arg-type]
        stt_sell_pct=float(entry["stt_sell_pct"]),  # type: ignore[arg-type]
        exchange_txn_pct=float(entry["exchange_txn_pct"]),  # type: ignore[arg-type]
        sebi_turnover_pct=float(entry["sebi_turnover_pct"]),  # type: ignore[arg-type]
        gst_pct=float(entry["gst_pct"]),  # type: ignore[arg-type]
        stamp_duty_buy_pct=float(entry["stamp_duty_buy_pct"]),  # type: ignore[arg-type]
        stamp_duty_sell_pct=float(entry["stamp_duty_sell_pct"]),  # type: ignore[arg-type]
        dp_charges_inr=float(entry["dp_charges_inr"]),  # type: ignore[arg-type]
        other_charges_flat_inr=float(entry["other_charges_flat_inr"]),  # type: ignore[arg-type]
        other_charges_pct=float(entry["other_charges_pct"]),  # type: ignore[arg-type]
    )


def _parse_date(value: object) -> dt.date:
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    raise ValueError(f"effective_from must be a date or ISO-8601 string, got {value!r}")
