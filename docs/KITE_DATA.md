# Kite market data

How real NSE price history gets into this system, what has actually been
verified against the live API, and the two limitations that decide whether the
result can be trusted for a backtest.

Companion: [MARKET_CALENDAR.md](MARKET_CALENDAR.md) for the trading calendar this
data is interpreted against.

---

## 1. Daily routine

A Kite access token expires at **6 AM IST** every day. Zerodha describes this as
a regulatory requirement, not a configurable session length, and obtaining a new
one needs a human to complete a browser login with password and 2FA on Zerodha's
own domain.

**This system does not try to automate around that**, and shouldn't: that login
is what stands between an API key and your account.

```bash
python scripts/kite_login.py          # once each morning
```

It prints a login URL, waits on `http://127.0.0.1:5000/kite/callback` for the
redirect, exchanges the `request_token`, and caches the session under `state/`
(gitignored) until 6 AM tomorrow. Over SSH or with a port conflict, use
`--paste`: the redirect target never has to serve anything, because the token is
in the browser's address bar even when the page fails to load.

The cached token is refused from 06:00 IST onward rather than at some point
after — giving up the last minutes before expiry is free, whereas discovering
expiry mid-write is not.

---

## 2. Fetching

```bash
# Instrument master — needs NO login, the endpoint is public
python scripts/ingest_kite_history.py --instruments-only

# Daily bars
python scripts/ingest_kite_history.py --symbols RELIANCE,INFY,TCS --from 2022-01-01
python scripts/ingest_kite_history.py --nifty-50 --from 2022-01-01
```

Requests are paced to Kite's documented **3 requests/second** for historical
data, and long ranges are split into windows that tile the range exactly (a
one-day gap at each chunk boundary would be invisible in the result).

The date range is refused if the trading calendar doesn't cover it. Bars for an
uncovered year produce a series this system cannot interpret — it could not tell
a missing bar from a holiday. Coverage is currently **2022–2026**.

### `broker/zerodha/kite_historical.py` cannot place an order

It touches three endpoints — `GET /instruments`, `GET /instruments/historical/…`,
`POST /session/token` — and has no method that submits, modifies or cancels
anything. A test asserts this.

That separation is deliberate. Fetching price history must never require
constructing a live-capable `KiteBroker`, because that would entangle data
ingestion with the four live-trading gates in `broker/factory.py` and create
pressure to weaken them. Ingesting history and placing orders are different
privileges.

---

## 3. What was verified live, and what wasn't

Being precise about this, because "the tests pass" and "it works against Kite"
are different claims.

**Verified against the live API:**

- `GET /instruments` returns 200 with no authentication. 112,731 rows.
- The CSV header in `kite_historical.INSTRUMENT_COLUMNS` is the literal live
  header, not a guess.
- The parser handles the real 9 MB response: 10,100 NSE cash equities,
  136 NSE indices.
- `NIFTY 50` = token `256265`, `INDIA VIX` = token `264969`.
- The generated instrument master loads through
  `data.instrument_master.InMemoryInstrumentRepository`.

**Not verified live — asserted only against a fake transport:**

- Everything involving `access_token`: the session exchange, historical candle
  fetching, chunking, pacing, and every error path (429, 403/`TokenException`,
  `InputException`).

Those need a completed login, which needs a human. The shapes come from
Zerodha's published documentation. **The first real backfill is the test**, and
should be treated as one: start with two or three symbols over a short range
before pulling years.

---

## 4. Two limitations that decide whether a backtest is honest

### 4.1 The instrument dump is survivorship-biased

Kite's `/instruments` is a photograph of **today**. It contains only currently
listed instruments — every company delisted, merged or suspended since 2022 is
simply absent.

Building a historical universe from it would mean back-testing on the set of
companies that *survived to today*, which is the single most common way a
strategy backtests beautifully and trades badly. It is exactly the bias
`universe/universe.py` and its point-in-time membership data exist to prevent.

The dump also carries no history at all: it cannot say when a symbol was listed,
renamed, or had its tick size changed. `effective_from` is therefore written as
the **snapshot date**, not backdated. Backdating it would fabricate
reference-data history the file does not contain — and would make the
survivorship problem invisible by letting the backtest run.

**Consequence:** this instrument master is fit for *live* universe construction
and for resolving tokens. It is **not** fit for historical universe
construction. That needs point-in-time index membership (NSE publishes NIFTY 50
constituent changes), which this system has an interface for
(`data.interfaces.IndexMembershipProvider`) and no data for yet.

### 4.2 Kite's daily candles are not corporate-action adjusted

The strategy's factors are computed on `PriceBasis.ADJUSTED` — a split or bonus
that isn't adjusted for shows up as a price collapse, which reads as momentum
and drawdown that never happened.

Kite's historical endpoint returns raw traded prices. So ingested bars are `RAW`,
and adjustment requires a corporate-actions feed through
`data.corporate_actions`. Until that exists, any factor computed on this data is
wrong for every instrument that had a corporate action in the window.

Phase 21 already found the failure mode: a `LocalMarketDataProvider` without a
`CorporateActionProvider` made every `ADJUSTED` request raise, which silently
excluded every candidate. That raise is the system being careful — do not work
around it by switching to `RAW`.

---

## 5. Where things stand

| Piece | State |
|---|---|
| Trading calendar 2022–2026 | Done |
| Instrument master (live universe, token resolution) | Done, verified live |
| Daily bars, paced and chunked | Built; needs a login to verify |
| Point-in-time index membership | **Missing** — blocks honest backtesting |
| Corporate actions / adjusted prices | **Missing** — blocks factor correctness |
| Fitted model on real data | Blocked on the two above |

Live trading remains gated independently of all of this: preflight fails,
`app/service.py` refuses live mode, and `broker/factory.py`'s four gates hold.
Getting data in changes none of that.
