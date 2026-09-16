"""Build a point-in-time liquid universe from NSE's daily bhavcopy.

docs/SPECIFICATION.md section 2.1 asks for a "point-in-time liquid NSE
equity universe" and says plainly: *do not backtest today's NIFTY 50
members all the way into the past*. This module is how that is satisfied
without reconstructing index membership from announcements.

**Why derive membership instead of tracking an index.** The alternative
is to compile NIFTY 50 constituent-change press releases since 2015 and
replay them. Every announcement missed in that compilation is invisible:
the universe simply contains a name it should not, or lacks one it
should, and nothing downstream can tell. Deriving eligibility from what
actually traded has no such failure mode, because each day's file is a
record rather than a reconstruction.

**Eligibility on date T uses only data up to T.** The liquidity screen is
a trailing average over the preceding window, so a name qualifies on the
strength of turnover it had already printed. Using a centred or forward
window would let a stock enter the universe because of volume it was
about to have -- the most direct form of look-ahead there is, and one
that flatters a backtest enormously, because the days a stock is about to
become heavily traded are rarely uneventful.

**Identity is ISIN, not symbol.** NSE symbols get renamed. Keyed on
symbol, a rename looks like a delisting plus a fresh listing: a position
that vanished and a new candidate that appeared, on the same day, for the
same company. Membership spans here are keyed on ISIN and carry whatever
symbol was current, so a rename is a continuation.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from data.models import IndexMembership
from data.nse_bhavcopy import BhavcopyRow

DERIVED_INDEX_SYMBOL = "NSE_LIQUID"
"""The name this derived universe is registered under.

Deliberately *not* "NIFTY50". This is not NIFTY 50 membership and calling
it that would be a lie that survives into every audit record -- someone
would eventually compare a holding against the real index and find it
absent. ``universe.index_reference: NIFTY50`` in settings stays what it
is: a benchmark for relative strength, not the universe.
"""


@dataclass(frozen=True, slots=True)
class EligibilityRules:
    """What makes an instrument tradable by this system on a given day."""

    min_avg_daily_value_inr: Decimal
    """Trailing turnover floor, applied to the **median** session, not the
    mean.

    Turnover rather than share volume, because a million shares of a Rs 3
    stock is not liquidity this system can use and share counts are not
    comparable across prices.

    Median rather than mean, because the mean is trivially gamed by a
    single day. A name that traded Rs 0.1 crore on nineteen sessions and
    Rs 100 crore on one averages Rs 5.1 crore and clears a Rs 5 crore
    bar -- while being impossible to exit on nineteen days out of twenty.
    That is not a hypothetical: it is what one block trade in an illiquid
    name looks like, and the mean cannot tell it apart from steady
    turnover. The median is unmoved by it, and for a genuinely liquid
    name the two are nearly identical.

    The config key is still ``min_avg_daily_value_inr``; the name predates
    this and describes the intent ("typical daily traded value") rather
    than the estimator."""

    lookback_sessions: int = 20
    """Sessions in the trailing liquidity window. One trading month:
    long enough that a single block trade cannot qualify a name, short
    enough to drop one that has genuinely dried up."""

    min_sessions_traded: int = 15
    """Of the last ``lookback_sessions``, how many the instrument must
    actually have traded on. A name that trades on three days out of
    twenty can show a flattering average while being impossible to exit,
    which is precisely the illiquidity the average hides."""

    min_price_inr: Decimal = Decimal("5")
    """Penny stocks move in tick-sized jumps that swamp any modelled
    edge, and their quoted spreads are a large fraction of price."""


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    """The eligible set on one session, with enough detail to audit it."""

    session_date: dt.date
    eligible: frozenset[str]
    """Instrument ids (``NSE:SYMBOL``) eligible on this date."""

    traded: int
    """How many EQ instruments traded at all, eligible or not."""

    def __len__(self) -> int:
        return len(self.eligible)


def _median(values: Sequence[Decimal]) -> Decimal:
    """Middle value, averaging the two middles for an even count.

    Written out rather than using ``statistics.median`` so the arithmetic
    stays in ``Decimal``: turnover is money, and money should not make a
    round trip through binary floating point on its way to a threshold
    comparison.
    """
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def build_snapshots(
    rows_by_date: Mapping[dt.date, Sequence[BhavcopyRow]],
    rules: EligibilityRules,
) -> list[UniverseSnapshot]:
    """Eligible sets for every session, in date order.

    Convenience wrapper over :func:`stream_snapshots` for callers that
    already hold every session in memory. A full 2015-2026 backfill does
    not -- roughly 5.4 million rows -- and should stream instead.
    """
    return list(
        stream_snapshots(((day, rows_by_date[day]) for day in sorted(rows_by_date)), rules)
    )


def stream_snapshots(
    sessions: Iterable[tuple[dt.date, Sequence[BhavcopyRow]]],
    rules: EligibilityRules,
) -> Iterator[UniverseSnapshot]:
    """Eligible sets, yielded one session at a time.

    ``sessions`` must arrive in ascending date order; that ordering is
    what makes the result correct, so it is checked rather than assumed.

    Streaming is not only about memory. The trailing windows are advanced
    session by session as each arrives, so a snapshot *cannot* see a
    session that has not been consumed yet -- there is no structure
    holding the future for it to reach into. No-look-ahead is arranged
    here, not asserted afterwards.

    Memory is bounded by the number of instruments times
    ``lookback_sessions``, not by the length of the history, so a decade
    costs the same as a month.
    """
    if rules.lookback_sessions < 1:
        raise ValueError("lookback_sessions must be at least 1")
    if rules.min_sessions_traded > rules.lookback_sessions:
        raise ValueError(
            f"min_sessions_traded ({rules.min_sessions_traded}) cannot exceed "
            f"lookback_sessions ({rules.lookback_sessions})"
        )

    # isin -> trailing turnover observations, one per session traded.
    history: dict[str, deque[Decimal]] = defaultdict(
        lambda: deque(maxlen=rules.lookback_sessions)
    )
    # isin -> which of the last N sessions it traded on, as a rolling flag.
    presence: dict[str, deque[int]] = defaultdict(
        lambda: deque(maxlen=rules.lookback_sessions)
    )

    previous_date: dt.date | None = None
    for session_date, rows in sessions:
        if previous_date is not None and session_date <= previous_date:
            raise ValueError(
                f"sessions must ascend; {session_date} followed {previous_date}. "
                "Out-of-order input would let a later session's turnover into an "
                "earlier session's window, which is look-ahead."
            )
        previous_date = session_date
        traded_today = {row.isin: row for row in rows if row.isin}

        # Advance every instrument seen so far, not only today's, so that
        # a name which stops trading decays out of the window instead of
        # keeping its last average for ever.
        for isin in list(presence) + [i for i in traded_today if i not in presence]:
            row = traded_today.get(isin)
            presence[isin].append(1 if row is not None else 0)
            if row is not None:
                history[isin].append(row.traded_value)

        eligible = set()
        for isin, row in traded_today.items():
            sessions_traded = sum(presence[isin])
            if sessions_traded < rules.min_sessions_traded:
                continue
            turnovers = history[isin]
            if not turnovers:
                continue
            if _median(turnovers) < rules.min_avg_daily_value_inr:
                continue
            if row.close < rules.min_price_inr:
                continue
            eligible.add(row.instrument_id)

        yield UniverseSnapshot(
            session_date=session_date,
            eligible=frozenset(eligible),
            traded=len(traded_today),
        )


def snapshots_to_membership(
    snapshots: Iterable[UniverseSnapshot], *, index_symbol: str = DERIVED_INDEX_SYMBOL
) -> list[IndexMembership]:
    """Collapse per-session eligibility into ``[effective_from, effective_to]``
    spans, which is what ``IndexMembershipProvider`` consumes.

    An instrument that leaves and later returns gets two spans rather than
    one long one. Merging them would assert membership across a gap the
    data says did not exist, and the gap is usually the interesting part:
    it is where the name was too illiquid to trade.

    ``effective_to`` is the last session the instrument was eligible, and
    ``IndexMembership.is_member_on`` treats the range as inclusive, so a
    span never claims eligibility on a day the screen had already failed.
    """
    open_since: dict[str, dt.date] = {}
    last_eligible: dict[str, dt.date] = {}
    memberships: list[IndexMembership] = []

    for snapshot in snapshots:
        for instrument_id in snapshot.eligible:
            if instrument_id not in open_since:
                open_since[instrument_id] = snapshot.session_date
            last_eligible[instrument_id] = snapshot.session_date

        for instrument_id in list(open_since):
            if instrument_id not in snapshot.eligible:
                memberships.append(
                    IndexMembership(
                        index_symbol=index_symbol,
                        instrument_id=instrument_id,
                        effective_from=open_since.pop(instrument_id),
                        effective_to=last_eligible[instrument_id],
                    )
                )

    # Spans still open on the final session stay open: effective_to=None
    # means "still a member", which is true as far as this data goes.
    for instrument_id, start in open_since.items():
        memberships.append(
            IndexMembership(
                index_symbol=index_symbol,
                instrument_id=instrument_id,
                effective_from=start,
                effective_to=None,
            )
        )

    memberships.sort(key=lambda m: (m.effective_from, m.instrument_id))
    return memberships
