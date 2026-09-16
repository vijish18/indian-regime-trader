"""NSE's corporate-actions feed, parsed into domain ``CorporateAction`` records.

Source: ``https://www.nseindia.com/api/corporates-corporateActions``, free,
official, and available historically (2,208 records for 2020 alone).

**Why this matters more than its size suggests.** The strategy's factors are
computed on ``PriceBasis.ADJUSTED``. An unadjusted 1:2 split looks exactly
like a 50% overnight collapse -- which reads as momentum and drawdown that
never happened, on a stock that in fact did nothing. Every factor for that
name, for the whole lookback window, is then wrong in a way no test on the
price series alone can see.

The feed gives terms as free text in a ``subject`` field, and the real
variety over 2019/2022/2024 (7,324 records) is:

    DIVIDEND       4067   cash, no price adjustment
    OTHER          2856   buybacks, EGM/AGM notices -- not price events
    BONUS           144   "Bonus 1:4"
    SPLIT           126   "Face Value Split ... From Rs 10/- ... To Rs 2/-"
    RIGHTS           89   "Rights 588:1000@ Premium Rs 2/-"
    DEMERGER         38   "Demerger"            <- no terms at all
    MERGER            4   "Scheme Of Amalgamation"  <- no terms at all

That last group is the important one. **Rights, mergers and demergers carry
no computable factor in this feed** -- a demerger is literally the word
"Demerger" -- so they are recorded with ``explicit_price_factor=None`` and
reported as needing an operator-supplied factor. They are never silently
treated as "no adjustment", because a demerger with no adjustment is a
price cliff the strategy will read as a crash.

docs/SPECIFICATION.md section 2.1 already asks for this: "keep an explicit
exclusion list for instruments with corporate-action or data-quality
anomalies". :func:`instruments_needing_review` produces that list.

**Nothing here guesses.** A subject line whose terms cannot be parsed is
returned as unparsed, not skipped and not approximated.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import urllib.request
from dataclasses import dataclass
from decimal import Decimal

from data.models import CorporateAction, CorporateActionType

API_URL = "https://www.nseindia.com/api/corporates-corporateActions"
REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-actions"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# -- subject-line grammar ---------------------------------------------------
#
# Each pattern is anchored on the vocabulary NSE actually uses, sampled from
# three separate years rather than assumed. Ordering matters: "Face Value
# Split" must be tested before any generic ratio pattern, and demerger
# before merger, since "Composite Scheme Of Arrangement" for a demerger
# also contains arrangement wording.

_RUPEES = r"(?:rs|re)\.?"
"""Both spellings, because Indian usage inflects the currency: "Re 1" is
one rupee, "Rs 2" is two. NSE's feed uses both, and matching only "Rs"
left 788 records unparsed -- almost all of them "Interim Dividend - Re 1
Per Share"."""

_SPLIT = re.compile(
    rf"from\s*{_RUPEES}\s*(?P<old>[\d.]+)\s*/?-?\s*per\s*share\s*to\s*{_RUPEES}\s*(?P<new>[\d.]+)",
    re.IGNORECASE,
)
_BONUS = re.compile(r"bonus[\s\-]*(?P<new>\d+)\s*:\s*(?P<old>\d+)", re.IGNORECASE)
"""The optional dash matters: NSE publishes both "Bonus 1:4" and
"Bonus- 1:2". A bonus is price-affecting, so an unparsed one is a
silent 50% price cliff -- unlike an amount-less dividend, which
affects cash accounting but never the adjusted price series."""
_RIGHTS_RATIO = re.compile(r"rights\s*(?P<new>\d+)\s*:\s*(?P<old>\d+)", re.IGNORECASE)
_DIVIDEND_AMOUNT = re.compile(rf"{_RUPEES}\s*-?\s*(?P<amount>[\d.]+)", re.IGNORECASE)
"""The optional dash is not cosmetic: the feed writes both
"Dividend - Rs 11 Per Share" and "Dividend Rs - 11 Per Share"."""

_DIVIDEND_WORDS = re.compile(
    r"\bdividends?\b|\bdivdend\b|\bdiv\b",
    re.IGNORECASE,
)
r"""Matches the feed's real spellings of "dividend".

The abbreviated form appears as "Int Div - Rs 0.71 Per Sh", and
"divdend" is a typo NSE has published. Word boundaries matter: without
them ``div`` would also match "Sub-Division", which is a split.
"""

_SPLIT_WORDS = ("split", "sub-division", "sub division")
_CONSOLIDATION_WORDS = ("consolidation of shares", "reverse split")
_DEMERGER_WORDS = ("demerger", "de-merger")
_NO_TERMS_PRICE_EVENTS = ("capital reduction", "reduction of capital")
"""Price-affecting, terms not in this feed. Treated like a demerger:
recorded, flagged for an operator, never assumed to be a non-event."""
_MERGER_WORDS = ("amalgamation", "scheme of arrangement", "merger")
_NON_PRICE_WORDS = (
    "buy back",
    "buyback",
    "general meeting",
    "annual general",
    "postal ballot",
    "board meeting",
    "voluntary delisting",
    "open offer",
    "name change",
    "symbol change",
)


class CorporateActionFeedError(RuntimeError):
    """The feed could not be fetched or did not have the expected shape."""


@dataclass(frozen=True, slots=True)
class ParsedSubject:
    """What one ``subject`` line was understood to mean."""

    action_type: CorporateActionType | None
    """``None`` for a record that is not a price or cash event at all
    (a buyback notice, an AGM). Not an error -- just not an action."""

    ratio_new: Decimal | None = None
    ratio_old: Decimal | None = None
    cash_amount: Decimal | None = None

    needs_explicit_factor: bool = False
    """True when the action *does* move the price but this feed does not
    carry terms sufficient to compute a factor -- rights, mergers,
    demergers. An operator must supply one before the instrument can be
    priced through the event."""


def classify_subject(subject: str) -> ParsedSubject | None:
    """Parse one NSE subject line, or return ``None`` if it is unrecognised.

    ``None`` means "this needs a human", never "ignore it". The caller
    collects unrecognised subjects and reports them; a silently dropped
    split is indistinguishable from a 50% crash.
    """
    text = " ".join(subject.split())
    if not text:
        return None
    lowered = text.lower()

    # Order matters; see the note above the patterns.
    if any(word in lowered for word in _SPLIT_WORDS):
        match = _SPLIT.search(text)
        if not match:
            return None
        old_face, new_face = Decimal(match.group("old")), Decimal(match.group("new"))
        if old_face <= 0 or new_face <= 0 or new_face >= old_face:
            # A "split" that does not reduce the face value is either a
            # consolidation or a malformed line. Either way, not this.
            return None
        # One old share becomes old_face/new_face new shares. The face
        # values are kept as the ratio rather than reduced, so the record
        # still shows the numbers the circular used.
        return ParsedSubject(CorporateActionType.SPLIT, ratio_new=old_face, ratio_old=new_face)

    if any(word in lowered for word in _CONSOLIDATION_WORDS):
        # A reverse split: representable, but rare enough in this feed that
        # no sample was available to verify the wording against. Flagged
        # rather than parsed from a guess.
        return ParsedSubject(CorporateActionType.SPLIT, needs_explicit_factor=True)

    if "bonus" in lowered:
        match = _BONUS.search(text)
        if not match:
            return None
        return ParsedSubject(
            CorporateActionType.BONUS,
            ratio_new=Decimal(match.group("new")),
            ratio_old=Decimal(match.group("old")),
        )

    if "right" in lowered:
        # The ratio is present but a rights factor also needs the issue
        # price relative to the cum-rights price. The premium is quoted in
        # free text ("@ Premium Rs 2/-") and is not the issue price, so the
        # factor is not computable from this line alone.
        match = _RIGHTS_RATIO.search(text)
        return ParsedSubject(
            CorporateActionType.RIGHTS,
            ratio_new=Decimal(match.group("new")) if match else None,
            ratio_old=Decimal(match.group("old")) if match else None,
            needs_explicit_factor=True,
        )

    if any(word in lowered for word in _DEMERGER_WORDS) or any(
        word in lowered for word in _NO_TERMS_PRICE_EVENTS
    ):
        return ParsedSubject(CorporateActionType.DEMERGER, needs_explicit_factor=True)

    if any(word in lowered for word in _MERGER_WORDS):
        return ParsedSubject(CorporateActionType.MERGER, needs_explicit_factor=True)

    if _DIVIDEND_WORDS.search(lowered):
        match = _DIVIDEND_AMOUNT.search(text)
        if not match:
            return None
        return ParsedSubject(
            CorporateActionType.DIVIDEND, cash_amount=Decimal(match.group("amount"))
        )

    if any(word in lowered for word in _NON_PRICE_WORDS):
        return ParsedSubject(action_type=None)

    return None


# -- fetching ---------------------------------------------------------------


def fetch_raw(
    start: dt.date, end: dt.date, *, timeout: int = 60
) -> list[dict[str, str]]:
    """Fetch the feed for a date range. NSE wants a browser-shaped request."""
    params = (
        f"?index=equities&from_date={start:%d-%m-%Y}&to_date={end:%d-%m-%Y}"
    )
    request = urllib.request.Request(
        API_URL + params,
        headers={"User-Agent": USER_AGENT, "Referer": REFERER, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - network, JSON, or HTTP alike
        raise CorporateActionFeedError(
            f"could not fetch corporate actions {start}..{end}: {exc}"
        ) from exc

    if not isinstance(payload, list):
        raise CorporateActionFeedError(
            f"expected a JSON list from the corporate-actions feed, "
            f"got {type(payload).__name__}"
        )
    return [record for record in payload if isinstance(record, dict)]


# -- conversion -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConversionResult:
    actions: tuple[CorporateAction, ...]
    unparsed: tuple[tuple[str, str], ...]
    """``(symbol, subject)`` for lines that could not be understood."""

    skipped_non_price: int
    """Buybacks, meeting notices -- recognised and correctly ignored."""

    def instruments_needing_review(self) -> frozenset[str]:
        """Instruments that cannot be priced through their own corporate
        action without a human supplying terms.

        This is docs/SPECIFICATION.md section 2.1's "explicit exclusion list
        for instruments with corporate-action anomalies". An instrument in
        this set must be excluded from the universe for any window spanning
        its ex-date, not traded on prices nobody can adjust.
        """
        needing = {
            action.instrument_id
            for action in self.actions
            if action.action_type
            in (
                CorporateActionType.RIGHTS,
                CorporateActionType.MERGER,
                CorporateActionType.DEMERGER,
            )
            and action.explicit_price_factor is None
        }
        return frozenset(needing | {symbol for symbol, _ in self.unparsed})


def to_corporate_actions(
    records: list[dict[str, str]], *, exchange: str = "NSE"
) -> ConversionResult:
    """Convert raw feed records into domain actions.

    Only ``EQ`` series records are converted: the feed also carries
    debentures and other series whose "corporate actions" do not apply to
    the cash-equity instrument this system trades.
    """
    actions: list[CorporateAction] = []
    unparsed: list[tuple[str, str]] = []
    skipped = 0

    for record in records:
        if str(record.get("series", "")).strip().upper() != "EQ":
            continue
        symbol = str(record.get("symbol", "")).strip()
        subject = str(record.get("subject", "")).strip()
        if not symbol or not subject:
            continue

        ex_date = _parse_feed_date(record.get("exDate"))
        if ex_date is None:
            # Without an ex-date the action cannot be placed on a timeline,
            # so it cannot adjust anything. Report rather than drop.
            unparsed.append((symbol, f"{subject} (unusable exDate={record.get('exDate')!r})"))
            continue

        parsed = classify_subject(subject)
        if parsed is None:
            unparsed.append((symbol, subject))
            continue
        if parsed.action_type is None:
            skipped += 1
            continue

        actions.append(
            CorporateAction(
                instrument_id=f"{exchange}:{symbol}",
                action_type=parsed.action_type,
                ex_date=ex_date,
                ratio_new=parsed.ratio_new,
                ratio_old=parsed.ratio_old,
                cash_amount=parsed.cash_amount,
                explicit_price_factor=None,
                record_date=_parse_feed_date(record.get("recDate")),
            )
        )

    return ConversionResult(
        actions=tuple(actions),
        unparsed=tuple(unparsed),
        skipped_non_price=skipped,
    )


def _parse_feed_date(value: object) -> dt.date | None:
    """NSE quotes dates as ``16-Sep-2026``, and uses ``"-"`` for absent."""
    text = str(value or "").strip()
    if not text or text == "-":
        return None
    try:
        return dt.datetime.strptime(text, "%d-%b-%Y").date()
    except ValueError:
        return None
