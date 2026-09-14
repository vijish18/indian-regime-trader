"""Point-in-time universe construction: entry, removal, symbol changes,
missing constituent data, corporate actions, and the core survivorship-bias
guarantee that future constituents cannot appear in earlier periods.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from data.corporate_actions import InMemoryCorporateActionProvider
from data.instrument_master import InMemoryInstrumentRepository
from data.membership import InMemoryIndexMembershipProvider
from data.models import (
    CorporateAction,
    CorporateActionType,
    Exchange,
    IndexMembership,
    Instrument,
    InstrumentStatus,
    Segment,
)
from universe.universe import (
    ExclusionReason,
    UniverseProvider,
)

INDEX = "NIFTY50"


def _instrument(
    instrument_id: str,
    symbol: str,
    *,
    effective_from: dt.date,
    effective_to: dt.date | None = None,
    status: InstrumentStatus = InstrumentStatus.ACTIVE,
    isin: str | None = None,
) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        symbol=symbol,
        exchange=Exchange.NSE,
        segment=Segment.EQUITY,
        tick_size=Decimal("0.05"),
        price_precision=2,
        effective_from=effective_from,
        effective_to=effective_to,
        status=status,
        isin=isin,
    )


def _member(
    instrument_id: str,
    *,
    effective_from: dt.date,
    effective_to: dt.date | None = None,
    inclusion_reason: str | None = None,
    exclusion_reason: str | None = None,
) -> IndexMembership:
    return IndexMembership(
        index_symbol=INDEX,
        instrument_id=instrument_id,
        effective_from=effective_from,
        effective_to=effective_to,
        inclusion_reason=inclusion_reason,
        exclusion_reason=exclusion_reason,
    )


def _provider(
    memberships: list[IndexMembership],
    instruments: list[Instrument],
    actions: list[CorporateAction] | None = None,
) -> UniverseProvider:
    return UniverseProvider(
        INDEX,
        InMemoryIndexMembershipProvider(memberships),
        InMemoryInstrumentRepository(instruments),
        InMemoryCorporateActionProvider(actions or []),
    )


# --------------------------------------------------------------------------
# Entry into the index
# --------------------------------------------------------------------------


def test_entry_is_absent_before_and_present_from_effective_date() -> None:
    """A constituent added mid-history must be invisible before its entry
    date and visible from it -- this is the exact boundary docs/
    SPECIFICATION.md section 2.1 calls out.
    """
    provider = _provider(
        memberships=[_member("NSE:NEWCO", effective_from=dt.date(2022, 4, 1))],
        instruments=[_instrument("NSE:NEWCO", "NEWCO", effective_from=dt.date(2015, 1, 1))],
    )
    before = provider.get_universe(dt.date(2022, 3, 31))
    on_entry = provider.get_universe(dt.date(2022, 4, 1))

    assert "NSE:NEWCO" not in before
    assert "NSE:NEWCO" in on_entry


def test_inclusion_reason_is_carried_through() -> None:
    provider = _provider(
        memberships=[
            _member(
                "NSE:NEWCO",
                effective_from=dt.date(2022, 4, 1),
                inclusion_reason="quarterly index review",
            )
        ],
        instruments=[_instrument("NSE:NEWCO", "NEWCO", effective_from=dt.date(2015, 1, 1))],
    )
    snapshot = provider.get_universe(dt.date(2022, 6, 1))
    constituent = next(c for c in snapshot.constituents if c.instrument_id == "NSE:NEWCO")
    assert constituent.inclusion_reason == "quarterly index review"
    assert constituent.index_membership_effective_from == dt.date(2022, 4, 1)


# --------------------------------------------------------------------------
# Removal from the index
# --------------------------------------------------------------------------


def test_removal_is_present_through_last_day_and_absent_after() -> None:
    """A deleted constituent must still appear for every date it was actually
    a member -- excluding it early would be the opposite bias (understating
    historical exposure), and is just as wrong as including it too late.
    """
    provider = _provider(
        memberships=[
            _member(
                "NSE:OLDCO",
                effective_from=dt.date(2015, 1, 1),
                effective_to=dt.date(2022, 3, 31),
                exclusion_reason="index review",
            )
        ],
        instruments=[_instrument("NSE:OLDCO", "OLDCO", effective_from=dt.date(2010, 1, 1))],
    )
    last_day = provider.get_universe(dt.date(2022, 3, 31))
    after_removal = provider.get_universe(dt.date(2022, 4, 1))

    assert "NSE:OLDCO" in last_day
    assert "NSE:OLDCO" not in after_removal


def test_removed_constituent_does_not_appear_as_an_exclusion_after_it_left() -> None:
    """Once fully out of the membership history for a date, the instrument is
    simply absent -- it is not a "constituent considered and excluded" for a
    date it was never a member on, which would misleadingly suggest it was
    still under consideration.
    """
    provider = _provider(
        memberships=[
            _member(
                "NSE:OLDCO", effective_from=dt.date(2015, 1, 1), effective_to=dt.date(2022, 3, 31)
            )
        ],
        instruments=[_instrument("NSE:OLDCO", "OLDCO", effective_from=dt.date(2010, 1, 1))],
    )
    snapshot = provider.get_universe(dt.date(2023, 1, 1))
    excluded_ids = {exclusion.instrument_id for exclusion in snapshot.excluded}
    assert "NSE:OLDCO" not in excluded_ids
    assert "NSE:OLDCO" not in snapshot


def test_readmitted_constituent_has_two_visible_spells() -> None:
    """Indices do re-admit names they previously deleted; both spells of
    membership must be independently queryable.
    """
    provider = _provider(
        memberships=[
            _member(
                "NSE:X", effective_from=dt.date(2015, 1, 1), effective_to=dt.date(2018, 12, 31)
            ),
            _member("NSE:X", effective_from=dt.date(2021, 1, 1)),
        ],
        instruments=[_instrument("NSE:X", "X", effective_from=dt.date(2010, 1, 1))],
    )
    assert "NSE:X" in provider.get_universe(dt.date(2016, 6, 1))
    assert "NSE:X" not in provider.get_universe(dt.date(2019, 6, 1))
    assert "NSE:X" in provider.get_universe(dt.date(2022, 6, 1))


# --------------------------------------------------------------------------
# Symbol changes
# --------------------------------------------------------------------------


def test_symbol_change_does_not_interrupt_membership() -> None:
    """A rename must not look like a delisting: the stable instrument_id keeps
    the constituent in the universe continuously, while the resolved symbol
    on each snapshot reflects what was actually in force that day.
    """
    provider = _provider(
        memberships=[_member("NSE:INFY", effective_from=dt.date(2015, 1, 1))],
        instruments=[
            _instrument(
                "NSE:INFY",
                "INFOSYSTCH",
                effective_from=dt.date(2010, 1, 1),
                effective_to=dt.date(2019, 6, 3),
            ),
            _instrument("NSE:INFY", "INFY", effective_from=dt.date(2019, 6, 4)),
        ],
    )
    before_rename = provider.get_universe(dt.date(2019, 6, 3))
    after_rename = provider.get_universe(dt.date(2019, 6, 4))

    assert "NSE:INFY" in before_rename
    assert "NSE:INFY" in after_rename
    before_symbol = next(c for c in before_rename.constituents if c.instrument_id == "NSE:INFY")
    after_symbol = next(c for c in after_rename.constituents if c.instrument_id == "NSE:INFY")
    assert before_symbol.symbol == "INFOSYSTCH"
    assert after_symbol.symbol == "INFY"


# --------------------------------------------------------------------------
# Missing constituent data
# --------------------------------------------------------------------------


def test_membership_record_with_no_instrument_data_is_excluded_not_guessed() -> None:
    """A membership row referencing an instrument the instrument master has
    never heard of must be excluded with a clear reason, not silently
    treated as eligible or silently dropped.
    """
    provider = UniverseProvider(
        INDEX,
        InMemoryIndexMembershipProvider(
            [_member("NSE:GHOST", effective_from=dt.date(2015, 1, 1))]
        ),
        InMemoryInstrumentRepository([]),  # no instrument data at all
    )
    snapshot = provider.get_universe(dt.date(2020, 1, 1))

    assert "NSE:GHOST" not in snapshot
    assert len(snapshot.constituents) == 0
    assert len(snapshot.excluded) == 1
    exclusion = snapshot.excluded[0]
    assert exclusion.instrument_id == "NSE:GHOST"
    assert exclusion.reason is ExclusionReason.MISSING_INSTRUMENT_DATA


def test_missing_data_for_only_one_of_several_constituents_excludes_just_that_one() -> None:
    provider = UniverseProvider(
        INDEX,
        InMemoryIndexMembershipProvider(
            [
                _member("NSE:KNOWN", effective_from=dt.date(2015, 1, 1)),
                _member("NSE:GHOST", effective_from=dt.date(2015, 1, 1)),
            ]
        ),
        InMemoryInstrumentRepository(
            [_instrument("NSE:KNOWN", "KNOWN", effective_from=dt.date(2010, 1, 1))]
        ),
    )
    snapshot = provider.get_universe(dt.date(2020, 1, 1))
    assert "NSE:KNOWN" in snapshot
    assert "NSE:GHOST" not in snapshot
    assert snapshot.excluded[0].reason is ExclusionReason.MISSING_INSTRUMENT_DATA


def test_instrument_record_that_has_not_started_yet_is_missing_data() -> None:
    """A membership record can predate the instrument master's own coverage --
    e.g. the instrument master was only backfilled from a later date. That is
    indistinguishable from "no data" for this date and must be excluded the
    same way, not crash.
    """
    provider = UniverseProvider(
        INDEX,
        InMemoryIndexMembershipProvider(
            [_member("NSE:X", effective_from=dt.date(2015, 1, 1))]
        ),
        InMemoryInstrumentRepository(
            [_instrument("NSE:X", "X", effective_from=dt.date(2018, 1, 1))]  # starts later
        ),
    )
    snapshot = provider.get_universe(dt.date(2016, 1, 1))
    assert "NSE:X" not in snapshot
    assert snapshot.excluded[0].reason is ExclusionReason.MISSING_INSTRUMENT_DATA


# --------------------------------------------------------------------------
# Suspended / delisted instruments
# --------------------------------------------------------------------------


def test_suspended_instrument_is_excluded_while_suspended() -> None:
    provider = _provider(
        memberships=[_member("NSE:X", effective_from=dt.date(2015, 1, 1))],
        instruments=[
            _instrument(
                "NSE:X",
                "X",
                effective_from=dt.date(2015, 1, 1),
                effective_to=dt.date(2020, 12, 31),
                status=InstrumentStatus.ACTIVE,
            ),
            _instrument(
                "NSE:X",
                "X",
                effective_from=dt.date(2021, 1, 1),
                effective_to=dt.date(2021, 6, 30),
                status=InstrumentStatus.SUSPENDED,
            ),
            _instrument(
                "NSE:X", "X", effective_from=dt.date(2021, 7, 1), status=InstrumentStatus.ACTIVE
            ),
        ],
    )
    assert "NSE:X" in provider.get_universe(dt.date(2020, 6, 1))  # active
    suspended_snapshot = provider.get_universe(dt.date(2021, 3, 1))
    assert "NSE:X" not in suspended_snapshot
    assert suspended_snapshot.excluded[0].reason is ExclusionReason.NOT_TRADABLE
    assert "suspended" in suspended_snapshot.excluded[0].detail
    assert "NSE:X" in provider.get_universe(dt.date(2021, 8, 1))  # reinstated


def test_delisted_instrument_stays_a_member_until_delisting_takes_effect() -> None:
    provider = _provider(
        memberships=[_member("NSE:X", effective_from=dt.date(2015, 1, 1))],
        instruments=[
            _instrument(
                "NSE:X",
                "X",
                effective_from=dt.date(2015, 1, 1),
                effective_to=dt.date(2022, 8, 31),
                status=InstrumentStatus.ACTIVE,
            ),
            _instrument(
                "NSE:X",
                "X",
                effective_from=dt.date(2022, 9, 1),
                status=InstrumentStatus.DELISTED,
            ),
        ],
    )
    assert "NSE:X" in provider.get_universe(dt.date(2022, 8, 31))
    after = provider.get_universe(dt.date(2022, 9, 2))
    assert "NSE:X" not in after
    assert after.excluded[0].reason is ExclusionReason.NOT_TRADABLE


# --------------------------------------------------------------------------
# Corporate actions
# --------------------------------------------------------------------------


def test_completed_merger_excludes_the_absorbed_instrument() -> None:
    """A merger that has already happened by the query date removes the
    absorbed instrument even if the instrument master's status field was
    never updated -- corporate-action data is treated as an independent,
    authoritative cross-check.
    """
    provider = _provider(
        memberships=[_member("NSE:ABSORBED", effective_from=dt.date(2015, 1, 1))],
        instruments=[
            _instrument("NSE:ABSORBED", "ABSORBED", effective_from=dt.date(2010, 1, 1))
        ],  # status left ACTIVE on purpose
        actions=[
            CorporateAction(
                instrument_id="NSE:ABSORBED",
                action_type=CorporateActionType.MERGER,
                ex_date=dt.date(2022, 5, 1),
                explicit_price_factor=Decimal(1),
                successor_instrument_id="NSE:ACQUIRER",
            )
        ],
    )
    before_merger = provider.get_universe(dt.date(2022, 4, 30))
    after_merger = provider.get_universe(dt.date(2022, 5, 2))

    assert "NSE:ABSORBED" in before_merger
    assert "NSE:ABSORBED" not in after_merger
    exclusion = after_merger.excluded[0]
    assert exclusion.reason is ExclusionReason.CORPORATE_ACTION
    assert "merger" in exclusion.detail


def test_future_corporate_action_does_not_exclude_early_because_that_would_be_look_ahead() -> None:
    """The mirror image of the merger test: a merger scheduled for next month
    must not remove the instrument from today's universe. Excluding early
    would mean the universe construction "knows" about an event before it
    happened -- exactly the look-ahead leak price adjustment also guards
    against.
    """
    provider = _provider(
        memberships=[_member("NSE:X", effective_from=dt.date(2015, 1, 1))],
        instruments=[_instrument("NSE:X", "X", effective_from=dt.date(2010, 1, 1))],
        actions=[
            CorporateAction(
                instrument_id="NSE:X",
                action_type=CorporateActionType.MERGER,
                ex_date=dt.date(2022, 5, 1),
                explicit_price_factor=Decimal(1),
            )
        ],
    )
    still_eligible = provider.get_universe(dt.date(2022, 4, 1))
    assert "NSE:X" in still_eligible
    assert still_eligible.excluded == ()


def test_split_and_bonus_do_not_exclude_the_instrument() -> None:
    """Only identity-changing actions (merger, demerger, delisting) affect
    eligibility. A split or bonus changes the share count, not whether the
    company is investable.
    """
    provider = _provider(
        memberships=[_member("NSE:X", effective_from=dt.date(2015, 1, 1))],
        instruments=[_instrument("NSE:X", "X", effective_from=dt.date(2010, 1, 1))],
        actions=[
            CorporateAction(
                instrument_id="NSE:X",
                action_type=CorporateActionType.SPLIT,
                ex_date=dt.date(2022, 5, 1),
                ratio_new=Decimal(5),
                ratio_old=Decimal(1),
            ),
            CorporateAction(
                instrument_id="NSE:X",
                action_type=CorporateActionType.DIVIDEND,
                ex_date=dt.date(2022, 6, 1),
                cash_amount=Decimal("10.00"),
            ),
        ],
    )
    snapshot = provider.get_universe(dt.date(2022, 7, 1))
    assert "NSE:X" in snapshot
    assert snapshot.excluded == ()


def test_corporate_action_provider_is_optional() -> None:
    """Without a CorporateActionProvider, eligibility rests entirely on
    instrument status -- an intentional degraded mode, not a crash.
    """
    provider = UniverseProvider(
        INDEX,
        InMemoryIndexMembershipProvider(
            [_member("NSE:X", effective_from=dt.date(2015, 1, 1))]
        ),
        InMemoryInstrumentRepository(
            [_instrument("NSE:X", "X", effective_from=dt.date(2010, 1, 1))]
        ),
        corporate_actions=None,
    )
    snapshot = provider.get_universe(dt.date(2022, 1, 1))
    assert "NSE:X" in snapshot


# --------------------------------------------------------------------------
# The core guarantee: future constituents cannot enter earlier periods
# --------------------------------------------------------------------------


def test_future_constituents_cannot_enter_earlier_periods() -> None:
    """The central assertion of this module: given a membership dataset that
    spans years, querying an early date must never surface a name that only
    joined later -- regardless of how much *later* history the data source
    happens to contain.
    """
    provider = _provider(
        memberships=[
            _member("NSE:FOUNDER", effective_from=dt.date(2010, 1, 1)),
            _member("NSE:JOINED_2015", effective_from=dt.date(2015, 3, 1)),
            _member("NSE:JOINED_2020", effective_from=dt.date(2020, 9, 15)),
            _member("NSE:JOINED_2024", effective_from=dt.date(2024, 1, 2)),
        ],
        instruments=[
            _instrument("NSE:FOUNDER", "FOUNDER", effective_from=dt.date(2005, 1, 1)),
            _instrument("NSE:JOINED_2015", "J2015", effective_from=dt.date(2005, 1, 1)),
            _instrument("NSE:JOINED_2020", "J2020", effective_from=dt.date(2005, 1, 1)),
            _instrument("NSE:JOINED_2024", "J2024", effective_from=dt.date(2005, 1, 1)),
        ],
    )

    snapshot_2012 = provider.get_universe(dt.date(2012, 1, 1))
    assert snapshot_2012.instrument_ids == {"NSE:FOUNDER"}

    snapshot_2017 = provider.get_universe(dt.date(2017, 1, 1))
    assert snapshot_2017.instrument_ids == {"NSE:FOUNDER", "NSE:JOINED_2015"}

    snapshot_2022 = provider.get_universe(dt.date(2022, 1, 1))
    assert snapshot_2022.instrument_ids == {
        "NSE:FOUNDER",
        "NSE:JOINED_2015",
        "NSE:JOINED_2020",
    }

    # The full membership dataset -- including 2024 entries -- is loaded into
    # every one of the queries above. Their absence from the earlier
    # snapshots is not because the provider hasn't seen them yet; it is
    # because they were not members yet.
    snapshot_2024 = provider.get_universe(dt.date(2024, 1, 2))
    assert snapshot_2024.instrument_ids == {
        "NSE:FOUNDER",
        "NSE:JOINED_2015",
        "NSE:JOINED_2020",
        "NSE:JOINED_2024",
    }
    assert "NSE:JOINED_2024" not in snapshot_2012
    assert "NSE:JOINED_2024" not in snapshot_2017
    assert "NSE:JOINED_2024" not in snapshot_2022


def test_the_day_before_entry_is_the_precise_boundary() -> None:
    """One day matters: entry is effective_from-inclusive, so the day before
    must exclude and the day itself must include, with no ambiguity.
    """
    provider = _provider(
        memberships=[_member("NSE:X", effective_from=dt.date(2020, 6, 15))],
        instruments=[_instrument("NSE:X", "X", effective_from=dt.date(2010, 1, 1))],
    )
    assert "NSE:X" not in provider.get_universe(dt.date(2020, 6, 14))
    assert "NSE:X" in provider.get_universe(dt.date(2020, 6, 15))


def test_querying_a_date_decades_before_any_membership_returns_an_empty_universe() -> None:
    provider = _provider(
        memberships=[_member("NSE:X", effective_from=dt.date(2015, 1, 1))],
        instruments=[_instrument("NSE:X", "X", effective_from=dt.date(2010, 1, 1))],
    )
    snapshot = provider.get_universe(dt.date(1999, 1, 1))
    assert snapshot.instrument_ids == frozenset()
    assert snapshot.excluded == ()


# --------------------------------------------------------------------------
# Snapshot ergonomics
# --------------------------------------------------------------------------


def test_snapshot_reports_its_index_and_date() -> None:
    provider = _provider(
        memberships=[_member("NSE:X", effective_from=dt.date(2015, 1, 1))],
        instruments=[_instrument("NSE:X", "X", effective_from=dt.date(2010, 1, 1))],
    )
    snapshot = provider.get_universe(dt.date(2020, 1, 1))
    assert snapshot.index_symbol == INDEX
    assert snapshot.as_of == dt.date(2020, 1, 1)
    assert len(snapshot) == 1


def test_unknown_index_yields_an_empty_universe_not_an_error() -> None:
    provider = UniverseProvider(
        "UNKNOWN_INDEX", InMemoryIndexMembershipProvider([]), InMemoryInstrumentRepository([])
    )
    snapshot = provider.get_universe(dt.date(2020, 1, 1))
    assert snapshot.instrument_ids == frozenset()
