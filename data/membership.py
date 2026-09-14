"""Point-in-time index membership.

No constituent list is hardcoded anywhere in this package. Membership is
loaded from a maintained dataset of ``(index, instrument, effective_from,
effective_to)`` records, so asking "who was in the NIFTY 50 on 2019-03-14"
returns the index as it actually was on that date.

Hardcoding today's constituents and running them through history is the
textbook survivorship bias: the backtest would only ever hold companies that
survived to the present, inflating returns by silently excluding the ones that
were deleted after they fell apart (docs/SPECIFICATION.md section 2.1).
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from data.interfaces import IndexMembershipProvider
from data.models import IndexMembership
from data.storage import (
    LocalDataStore,
    parse_date,
    parse_optional_date,
    parse_optional_str,
    read_table,
    require_columns,
)

MEMBERSHIP_COLUMNS = ("index_symbol", "instrument_id", "effective_from")
"""Required columns. ``effective_to`` empty means "still a member";
``inclusion_reason``/``exclusion_reason`` are optional free text."""


class InMemoryIndexMembershipProvider(IndexMembershipProvider):
    """Membership provider backed by an in-memory record set.

    An instrument may appear several times for the same index -- indices do
    re-admit companies they previously deleted -- so records are kept as a
    list of intervals rather than collapsed into one range per instrument.
    """

    def __init__(self, memberships: Iterable[IndexMembership]) -> None:
        self._by_index: dict[str, list[IndexMembership]] = defaultdict(list)
        for membership in memberships:
            self._by_index[membership.index_symbol].append(membership)
        for records in self._by_index.values():
            records.sort(key=lambda record: (record.effective_from, record.instrument_id))
        _reject_overlapping_intervals(self._by_index)

    @classmethod
    def from_file(cls, path: Path) -> InMemoryIndexMembershipProvider:
        frame = read_table(path)
        require_columns(frame, MEMBERSHIP_COLUMNS, str(path))
        return cls(parse_membership_rows(frame, source=str(path)))

    @classmethod
    def from_store(cls, store: LocalDataStore) -> InMemoryIndexMembershipProvider:
        return cls.from_file(store.index_membership_path())

    def members_on(self, index_symbol: str, as_of: dt.date) -> frozenset[str]:
        return frozenset(
            membership.instrument_id
            for membership in self._by_index.get(index_symbol, ())
            if membership.is_member_on(as_of)
        )

    def membership_history(self, index_symbol: str) -> list[IndexMembership]:
        return list(self._by_index.get(index_symbol, ()))

    def known_indices(self) -> frozenset[str]:
        return frozenset(self._by_index)

    def changes_between(
        self, index_symbol: str, start: dt.date, end: dt.date
    ) -> tuple[frozenset[str], frozenset[str]]:
        """Instruments added and removed between ``start`` and ``end``.

        Used to keep universe rebalancing honest: an instrument leaving the
        index is a position the portfolio must exit, not one that silently
        disappears from the books.
        """
        before = self.members_on(index_symbol, start)
        after = self.members_on(index_symbol, end)
        return after - before, before - after


def parse_membership_rows(frame: pd.DataFrame, source: str) -> list[IndexMembership]:
    """Convert a membership table into ``IndexMembership`` records."""
    memberships: list[IndexMembership] = []
    for position, row in enumerate(frame.to_dict("records"), start=2):
        try:
            memberships.append(
                IndexMembership(
                    index_symbol=str(row["index_symbol"]).strip(),
                    instrument_id=str(row["instrument_id"]).strip(),
                    effective_from=parse_date(row["effective_from"]),
                    effective_to=parse_optional_date(row.get("effective_to")),
                    inclusion_reason=parse_optional_str(row.get("inclusion_reason")),
                    exclusion_reason=parse_optional_str(row.get("exclusion_reason")),
                )
            )
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{source} line {position}: {exc}") from exc
    return memberships


def _reject_overlapping_intervals(
    by_index: dict[str, list[IndexMembership]]
) -> None:
    """Raise if one instrument has two overlapping membership spells in the
    same index, which would make ``members_on`` ambiguous about when it
    actually joined.
    """
    for index_symbol, records in by_index.items():
        by_instrument: dict[str, list[IndexMembership]] = defaultdict(list)
        for record in records:
            by_instrument[record.instrument_id].append(record)
        for instrument_id, spells in by_instrument.items():
            ordered = sorted(spells, key=lambda spell: spell.effective_from)
            for earlier, later in zip(ordered, ordered[1:], strict=False):
                if earlier.effective_to is None:
                    raise ValueError(
                        f"{index_symbol}/{instrument_id}: open-ended membership from "
                        f"{earlier.effective_from} is followed by another spell from "
                        f"{later.effective_from}; close it with effective_to"
                    )
                if later.effective_from <= earlier.effective_to:
                    raise ValueError(
                        f"{index_symbol}/{instrument_id}: overlapping membership "
                        f"[{earlier.effective_from}..{earlier.effective_to}] and "
                        f"[{later.effective_from}..{later.effective_to}]"
                    )
