# The market calendar

`config/nse_holidays.csv` decides, for every date, whether NSE was open. Almost
everything else depends on it: which bars exist, what "yesterday" means, when a
session opens, whether the system should be trading at all right now.

It is data, not code, so no type checker guards it — and a wrong entry fails
exactly like a right one until money is involved. This document records where
it came from, the two judgement calls encoded in it, and what is still missing.

---

## 1. Where the data comes from

Regenerate with:

```bash
pip install pypdf          # maintenance-tool dependency, not a project one
python scripts/build_nse_holidays.py
python scripts/build_nse_holidays.py --check    # verify, don't rewrite
```

Two source kinds, and **both are needed**:

| Source | What it is | Why it alone is not enough |
|---|---|---|
| **Annual circular** (`NSE/CMTR/…`) | NSE's authoritative notification for a calendar year, published each December | It is superseded during the year — see §2 |
| **Live holiday master** (`/api/holiday-master?type=trading`, `CM` segment) | NSE's current statement | Covers the current year only |

Every row in the CSV carries a `source` column naming the circular it was
transcribed from, so any entry can be traced back and re-verified. That column
is ignored by `NSETradingCalendar.from_file`, which reads only `date`,
`description` and `session_type`.

The builder cross-checks each parsed date against the **day of week NSE printed
next to it** and refuses to guess if they disagree. Two independent statements
of the same fact, from the same document, is a cheap and surprisingly effective
parse check.

Current coverage: **2015–2026**.

---

## 2. The annual circular is not the last word

NSE adds closures mid-year by partial modification. Two are already in this
file, and neither appears in its year's December circular:

| Date | Reason | How it was found |
|---|---|---|
| 2024-05-20 (Mon) | Parliamentary Elections in Mumbai | circular `NSE/CMTR/61518`, issued the following April |
| 2026-01-15 (Thu) | Municipal Corporation Election, Maharashtra | live holiday-master API; absent from `NSE/CMTR/71775` |

Both are ordinary weekdays. A build that read only the annual circulars would
have the system placing orders into a closed exchange on both.

This is why `scripts/build_nse_holidays.py` queries the live API on every run,
and why it merges with `setdefault` — the API can **add** a closure the
circulars missed, but can never remove one or overwrite a historical year.

**Operationally:** re-run the builder periodically during the year, not just
each December. A closure announced in April for May will not be in the file
otherwise.

---

## 2b. Reconciliation against what the market actually did

```bash
python scripts/build_nse_holidays.py --reconcile-with-kite
```

A circular is published in December for the year *ahead*, so it is a forecast.
NIFTY 50's bar history is the record: on any weekday the index printed no bar,
the cash market was shut. For dates that have already happened, the record wins.

This does two things. It **extends coverage back to 2015**, for years whose
circulars were not locatable in NSE's archive — those rows are sourced
`derived/nifty50-no-bar`, a fact from price data rather than a document. And it
**corrects the circular-derived years**. Run against 2022–2025 it found three
weekdays the exchange was shut and this calendar said were open:

| Date | What | Why the circular missed it |
|---|---|---|
| 2023-06-29 | Bakri Id | moved from the 28th after publication |
| 2024-01-22 | Ram Mandir consecration | announced ad hoc |
| 2024-11-20 | Maharashtra assembly elections | announced ad hoc |

All three are the dangerous direction — the system would have waited all day for
fills from a closed exchange. It also removed 2023-06-28, which the circular
listed as closed and the index demonstrably traded through.

Muhurat days are the deliberate exception: the index prints a bar because the
ceremonial session trades, but there is no regular session, so they stay closed
(§3).

---

## 2c. The bug this found, and why the earlier checks missed it

The 2022 circular renders row 13 as:

```
13 November 08,2022
08,2022
Tuesday Gurunanak Jayanti
```

— the date fragment repeated on its own line. The original line-anchored regex
required date and weekday on one line, so it **silently skipped the row**, and
the shipped calendar said NSE was open on Guru Nanak Jayanti 2022.

Nothing caught it. The day-of-week cross-check only validates rows that *did*
parse. The plausibility check passed because 12 closures is a perfectly normal
count. The file loaded, the tests were green, and the data was wrong.

It was found by reconciling against price data — an entirely independent source.

Two fixes, both now permanent:

1. The parser anchors on serial-number/date/weekday over whitespace-collapsed
   text and takes the description as what lies between anchors, so a row can
   wrap lines.
2. **NSE numbers its own rows.** The parser asserts each table's serials are a
   contiguous `1..N`, so the document states its own row count and a dropped row
   is detectable with no external source at all.

The general lesson is worth keeping: validating the records you parsed says
nothing about the records you didn't.

---

## 3. Muhurat sessions are recorded as closed

NSE marks the Diwali Laxmi Pujan row with an asterisk: the exchange is shut for
regular trading, but holds a ceremonial Muhurat session of roughly an hour,
whose timings are notified separately — for 2026, not yet at all.

`data/models.py` has a `SessionType.SPECIAL` for exactly this. **It is
deliberately unused, and nothing in the shipped file is marked `special`.**

The reason is a real limitation in `NSETradingCalendar.session()`: it returns
the *regular* 09:15–15:30 window for any day, including one marked `special`.
It has no field for per-session times. So marking a Muhurat day `special` would
make `is_trading_day` return `True` and hand out a full normal trading day —
telling this system that Sunday 8 November 2026 is open 09:15 to 15:30.

Recording them `closed` means the strategy does not trade Muhurat. That is the
fail-closed answer, and for a daily long-only cash-equity system it is the
right answer independently: Muhurat is a low-liquidity ceremonial session on a
separate settlement schedule, and nothing in this strategy's daily-bar logic
would handle it correctly.

The weekday cases are the ones that matter — 2022-10-24 (Mon), 2024-11-01 (Fri)
and 2025-10-21 (Tue) were all working days on which only the Muhurat session
ran. `tests/unit/test_nse_holidays.py` asserts each is not a trading day.

**If Muhurat trading is ever wanted**, the fix is in the calendar, not the data:
`TradingSession` needs explicit per-day session times, and `session()` needs to
read them. Until then a test rejects any `special` row, so the dangerous state
cannot be reached by editing the CSV.

---

## 4. The failure mode these tests are built around

While assembling this file, a web-search summary of "NSE trading holidays 2026"
returned five dates. Four were weekends. It read like an answer.

The actual circular lists **fifteen** weekday closures. Had the summary been
used, the system would have treated Republic Day, Holi, Shri Ram Navami,
Mahavir Jayanti, Good Friday, Ambedkar Jayanti, Maharashtra Day, Bakri Id,
Muharram, Ganesh Chaturthi, Gandhi Jayanti, Dussehra, Diwali-Balipratipada,
Guru Nanak Jayanti and Christmas as ordinary trading days.

Nothing would have raised. A year with *any* entries counts as "covered", and
every missing closure looks exactly like a normal open day. The system would
have quietly waited for fills from a shut exchange, fifteen times a year.

So `tests/unit/test_nse_holidays.py` checks **shape**, not just rows:

- at least 10 and at most 25 weekday closures per covered year — the only check
  that catches a truncated list in which every individual row is correct
- contiguous coverage (a gap mid-range means an extraction failure)
- no duplicates; rows sorted by date, so a mid-year insertion is a one-line diff
- both known mid-year closures present
- spot checks in both directions: known closures are closed, and known ordinary
  days are open (a file marking every day a holiday would pass a
  closures-only test and trade nothing)
- **the current year is covered** — this test is designed to start failing when
  the calendar goes stale, which is a thing to find out in December rather than
  at 09:15 on 2 January

---

## 5. What is still missing

**2015–2021 rest on derived data, not documents.** Those years' circulars were
not locatable in NSE's archive, so their closures come from NIFTY 50's bar
history (`derived/nifty50-no-bar`). That is a sound inference — the index is
computed from trades, so no bar means no cash session — but it inherits any gap
in the vendor's own history. If Kite were missing a day, this would record it as
a holiday. Cross-checked where both sources exist (2022–2026), the inference
agreed with the circulars on every date except the three genuine corrections in
§2b, which is the evidence for trusting it on the earlier years.

**Before 2015 is not covered**, because Kite's daily history does not reach
further back. `NSETradingCalendar` refuses to answer for an uncovered year, so a
backtest spanning 2014 fails loudly rather than assuming no holidays.

**Ad-hoc closures in years with no price coverage would still be invisible.**
Reconciliation closes this for every year the index covers, which is now all of
them — but the mechanism is worth understanding rather than assuming the problem
is permanently solved.

That is a known, bounded inaccuracy in historical results. It is recorded here
rather than discovered later.

**Session timings are assumed constant.** 09:15–15:30 with a 09:00–09:08
pre-open, hardcoded as defaults in `NSETradingCalendar`. NSE has changed
timings before, and does so for special circumstances. Nothing in this file
tracks that.
