"""Point-in-time tradable universe construction.

Never backtest today's index membership into the past. ``UniverseProvider``
joins three point-in-time data sources so a query for any historical date
returns exactly what was investable *on that date*, using only information
that existed as of it:

1. **Index membership** (``data.interfaces.IndexMembershipProvider``) --
   who was a constituent, and since when.
2. **Instrument reference data** (``data.interfaces.InstrumentRepository``)
   -- was the instrument active, suspended, or delisted on that date, and
   under what symbol.
3. **Corporate actions** (``data.interfaces.CorporateActionProvider``,
   optional) -- a cross-check for identity-changing events (merger,
   demerger, delisting) that had already occurred by the query date.

This is the primary survivorship-bias control described in
docs/SPECIFICATION.md section 2.1: a constituent added to the index next year
must never appear in this year's universe, and a constituent removed last
year must still appear in a query for a date while it was still a member.

Layer boundary: this module answers "was X eligible to be considered", not
"was X liquid enough to trade" or "does X pass the selection filters" -- those
are ``universe.stock_selector``'s job, applied *within* the eligible universe
this module returns. Keeping the two separate is what lets a walk-forward run
attribute performance to selection vs. eligibility independently.

## Data limitations (read before trusting a historical universe)

- **This module cannot recover history the source data does not have.** If
  the membership dataset only records today's constituents without historical
  add/remove dates, no logic here can reconstruct who was a member in 2019 --
  it will simply be wrong. Point-in-time correctness here is only as good as
  the point-in-time correctness of the ``IndexMembershipProvider`` and
  ``InstrumentRepository`` it is given.
- **Delisting coverage depends on what was fed in.** An instrument is treated
  as delisted only if its instrument-master record's ``status`` says so, or
  (when a ``CorporateActionProvider`` is supplied) if a merger/demerger/
  delisting action with ``ex_date <= as_of`` is on file. If neither source
  recorded the event, the instrument will incorrectly still appear eligible.
- **A stable ``instrument_id`` is assumed to survive symbol changes.** Symbol
  renames are handled by instrument versioning (multiple ``Instrument``
  records sharing one ``instrument_id``); if the upstream data instead keys
  by symbol and reissues a new id across a rename, this will misread a rename
  as a delisting plus a new listing.
- **A missing instrument record is excluded, not guessed.** If a membership
  record references an ``instrument_id`` absent from the instrument
  repository as of that date, the constituent is excluded with
  ``ExclusionReason.MISSING_INSTRUMENT_DATA`` rather than assumed eligible.
- **No snapshot persistence yet.** ``get_universe`` recomputes from current
  source data on every call; it is deterministic only if the underlying
  membership/instrument/corporate-action data has not been mutated since a
  prior call. Persisting the exact snapshot used at each historical rebalance
  (so a later data correction cannot retroactively change a past decision) is
  deferred to the storage layer (``storage/database.py``, not yet
  implemented).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum

from data.corporate_actions import InMemoryCorporateActionProvider
from data.errors import InstrumentNotFoundError
from data.instrument_master import InMemoryInstrumentRepository
from data.interfaces import (
    CorporateActionProvider,
    IndexMembershipProvider,
    InstrumentRepository,
)
from data.membership import InMemoryIndexMembershipProvider
from data.models import CorporateAction
from data.storage import LocalDataStore


class ExclusionReason(StrEnum):
    """Why an index constituent did not make it into the eligible universe."""

    MISSING_INSTRUMENT_DATA = "missing_instrument_data"
    NOT_TRADABLE = "not_tradable"
    CORPORATE_ACTION = "corporate_action"


@dataclass(frozen=True, slots=True)
class UniverseExclusion:
    """One index constituent excluded from an otherwise-eligible universe,
    and why -- kept alongside the accepted constituents so an excluded name
    is a visible, auditable decision rather than a silent omission.
    """

    instrument_id: str
    reason: ExclusionReason
    detail: str


@dataclass(frozen=True, slots=True)
class ConstituentSnapshot:
    """One eligible constituent, resolved against the instrument master as of
    the snapshot date.

    ``symbol`` and ``isin`` are the values in force *on that date* -- for an
    instrument that has been renamed, an early snapshot carries the old
    symbol and a later one carries the new symbol, while ``instrument_id``
    (the stable join key) is unchanged across the rename.
    """

    instrument_id: str
    symbol: str
    isin: str | None
    index_membership_effective_from: dt.date
    inclusion_reason: str | None


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    """The eligible universe for one index on one date, plus the constituents
    that were considered and excluded, with reasons.
    """

    index_symbol: str
    as_of: dt.date
    constituents: tuple[ConstituentSnapshot, ...]
    excluded: tuple[UniverseExclusion, ...] = ()

    @property
    def instrument_ids(self) -> frozenset[str]:
        return frozenset(c.instrument_id for c in self.constituents)

    def __contains__(self, instrument_id: str) -> bool:
        return instrument_id in self.instrument_ids

    def __len__(self) -> int:
        return len(self.constituents)


class UniverseProvider:
    """Builds the point-in-time eligible universe for one index by joining
    index membership, instrument reference data, and (optionally) corporate
    actions.

    All three collaborators are point-in-time abstractions
    (``data.interfaces``); this class adds no data of its own and hardcodes
    no constituent list.
    """

    def __init__(
        self,
        index_symbol: str,
        membership: IndexMembershipProvider,
        instruments: InstrumentRepository,
        corporate_actions: CorporateActionProvider | None = None,
    ) -> None:
        self._index_symbol = index_symbol
        self._membership = membership
        self._instruments = instruments
        self._corporate_actions = corporate_actions

    @classmethod
    def from_store(
        cls,
        index_symbol: str,
        store: LocalDataStore,
        *,
        use_corporate_actions: bool = True,
    ) -> UniverseProvider:
        """Build a provider from the local file-backed store used elsewhere
        in the data layer. Corporate-action cross-checking is skipped (not
        an error) if no corporate-actions file has been ingested yet.
        """
        membership = InMemoryIndexMembershipProvider.from_store(store)
        instruments = InMemoryInstrumentRepository.from_store(store)
        corporate_actions = (
            InMemoryCorporateActionProvider.from_store(store)
            if use_corporate_actions and store.corporate_actions_path().is_file()
            else None
        )
        return cls(index_symbol, membership, instruments, corporate_actions)

    @property
    def index_symbol(self) -> str:
        return self._index_symbol

    def get_universe(self, as_of: dt.date) -> UniverseSnapshot:
        """The eligible universe on ``as_of``, using only membership,
        instrument, and corporate-action facts dated on or before it.

        Returns only instrument ids that were (a) index members on this
        exact date, (b) tradable (active status, within their effective
        range) on this exact date, and (c) not already subject to a
        completed merger/demerger/delisting as of this date.
        """
        constituents: list[ConstituentSnapshot] = []
        excluded: list[UniverseExclusion] = []

        for record in self._membership.membership_history(self._index_symbol):
            if not record.is_member_on(as_of):
                continue
            instrument_id = record.instrument_id

            try:
                instrument = self._instruments.get(instrument_id, as_of)
            except InstrumentNotFoundError as exc:
                excluded.append(
                    UniverseExclusion(
                        instrument_id, ExclusionReason.MISSING_INSTRUMENT_DATA, str(exc)
                    )
                )
                continue

            if not instrument.is_tradable_on(as_of):
                excluded.append(
                    UniverseExclusion(
                        instrument_id,
                        ExclusionReason.NOT_TRADABLE,
                        f"status={instrument.status.value}, effective="
                        f"[{instrument.effective_from}..{instrument.effective_to or 'open'}]",
                    )
                )
                continue

            identity_action = self._completed_identity_change(
                instrument_id, instrument.effective_from, as_of
            )
            if identity_action is not None:
                excluded.append(
                    UniverseExclusion(
                        instrument_id,
                        ExclusionReason.CORPORATE_ACTION,
                        f"{identity_action.action_type.value} ex-date "
                        f"{identity_action.ex_date}",
                    )
                )
                continue

            constituents.append(
                ConstituentSnapshot(
                    instrument_id=instrument_id,
                    symbol=instrument.symbol,
                    isin=instrument.isin,
                    index_membership_effective_from=record.effective_from,
                    inclusion_reason=record.inclusion_reason,
                )
            )

        constituents.sort(key=lambda c: c.instrument_id)
        excluded.sort(key=lambda e: e.instrument_id)
        return UniverseSnapshot(
            index_symbol=self._index_symbol,
            as_of=as_of,
            constituents=tuple(constituents),
            excluded=tuple(excluded),
        )

    def _completed_identity_change(
        self, instrument_id: str, start: dt.date, as_of: dt.date
    ) -> CorporateAction | None:
        """The most recent merger/demerger/delisting with ``ex_date <= as_of``,
        if any.

        Bounded strictly to ``end=as_of`` so this can never see a
        not-yet-happened action -- using a future corporate action to
        exclude an instrument from a past universe would be a look-ahead
        leak, the same failure mode price adjustment guards against in
        ``data.corporate_actions``.
        """
        if self._corporate_actions is None:
            return None
        candidates = [
            action
            for action in self._corporate_actions.actions_for(instrument_id, start, as_of)
            if action.is_identity_change
        ]
        return max(candidates, key=lambda action: action.ex_date) if candidates else None
