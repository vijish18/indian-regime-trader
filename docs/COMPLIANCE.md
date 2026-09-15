# India API/Algo Operational Compliance (Phase 16)

This document is the research record behind `config.models.ComplianceConfig`
and `broker/compliance.py`. It exists because docs/SPECIFICATION.md section 13
already said the right thing before this phase started — "The production
build must be aligned with the chosen broker's supported route and current
exchange onboarding requirements" — and this phase is where that stopped
being an aspiration and became executable configuration with a hard-failure
gate in front of it.

**Every claim below is either quoted/paraphrased from a source with a URL
next to it, or explicitly marked as unverified and requiring confirmation
with the broker before going live.** Nothing here is a recollection or a
plausible guess. Where two sources disagreed (this happened — see the
effective-date note below), both are reported rather than one being quietly
picked.

## Scope and chosen execution model

**Broker:** Zerodha (chosen explicitly by the user in Phase 15 — see that
phase's summary; not guessed).

**Execution model:** "Tech-savvy client" / self-hosted API trading via Kite
Connect, automating strategies at **10 orders per second (OPS) or below**.
This is the unregistered retail-algo route — it does **not** require
exchange algo registration, unlike the higher-throughput "registered algo"
route this system does not use and this document does not cover. If this
system is ever changed to exceed 10 OPS or to route through an
exchange-registered/broker-hosted algo strategy, this entire document and
`ComplianceConfig` need to be re-verified against the registered-algo rules,
not extended by assumption.

## 1. API authentication requirements

Kite Connect's documented login flow (verified directly against
`kite.trade/docs/connect/v3/`, Phase 15's research):

1. A human completes login at `https://kite.zerodha.com/connect/login?v=3&api_key=...`
   in a browser (username/password + 2FA). There is no documented API for
   automating this step — see `broker/zerodha/kite_broker.py`'s module
   docstring, which explains why this codebase does not attempt to script it.
2. Kite redirects back to the app's registered redirect URL with a one-time
   `request_token`.
3. The app exchanges it via `POST /session/token` (`api_key`, `request_token`,
   `checksum = sha256(api_key + request_token + api_secret)`) for an
   `access_token`.
4. Every subsequent request carries `Authorization: token
   {api_key}:{access_token}` and `X-Kite-Version: 3`.

Source: `kite.trade/docs/connect/v3/user/` (fetched and verified in Phase 15).

## 2. Static-IP requirement

**Regulatory requirement (NSE, applies to the tech-savvy/self-hosted API
route specifically):** a client placing algo orders via a direct API must
register a static IP, and order requests from an unregistered IP are
rejected. Broker-hosted (non-self-hosted) strategies are compliant via the
*broker's* static IP instead — this does not apply to this system, which is
self-hosted.
Source: [Zerodha — comprehensive overview of NSE's retail algo circular](https://zerodha.com/z-connect/general/a-comprehensive-overview-of-nses-circular-on-the-new-retail-algo-trading-framework);
[NSE retail-algo rules explainer (fintrens)](https://blogs.fintrens.com/nse-retail-algo-trading-rules-nov-2025-static-ip-order-tagging-compliance-guide/).

**Effective date — sources disagree, not resolved here:** one summary of
Zerodha's own FAQ states the static-IP requirement for order placement took
effect **1 April 2025**; a summary of the NSE circular states **1 August
2025**. Both are already in the past relative to this system's current
environment date, so the requirement is live either way, but the exact
effective date should be confirmed against NSE's current circular before
any go-live checklist cites a specific date.
Sources: [support.zerodha.com Kite Connect API FAQ](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/kite-connect-api-faqs)
(April 2025); [fintrens NSE explainer](https://blogs.fintrens.com/nse-retail-algo-trading-rules-nov-2025-static-ip-order-tagging-compliance-guide/) (August 2025).

**Zerodha-specific implementation** (this is Zerodha's own product choice on
top of the baseline NSE requirement, not itself a SEBI/NSE mandate — the NSE
FAQ digest found no exchange-mandated count or change frequency):

- Up to **two** static IPs per developer app: one **primary (mandatory)**,
  one **secondary (optional)**.
- Configured in the Kite Connect developer console, app profile, "IP
  Whitelist" section.
- Both IPv4 and IPv6 are accepted; the IP need not be India-based.
- Changeable **once per calendar week**, per one source (not corroborated
  by a second source — treat as unverified until confirmed in Zerodha's
  live console UI).
- **Scope: order-placing endpoints only.** `place_order`/`modify_order`/
  `cancel_order` are validated against the registered IP; all other
  endpoints (quotes, WebSocket, positions, order book, funds) remain
  reachable from any IP.
- IP sharing between SEBI-defined family members (spouse, dependent
  children, dependent parents) is permitted with a written request + 2FA
  validation; one IP maps to exactly one account otherwise.

Sources: [Kite Connect forum — preparing for SEBI's retail algo rules](https://kite.trade/forum/discussion/15912/preparing-to-comply-with-sebis-retail-algo-rules-static-ip-ratelimits-order-types);
[support.zerodha.com Kite Connect API FAQ](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/kite-connect-api-faqs).

**What this system does with it:** `ComplianceConfig.static_ip_primary` is a
required, non-empty field. This code has no way to verify Zerodha's own
dashboard actually has this IP registered (there is no documented read API
for it) — the field is this system's own attestation record: if it is
empty, the compliance gate refuses to construct a live-capable broker at
all, on the principle that a human must have deliberately recorded what was
registered rather than the system silently assuming the operator remembered
to do it out of band.

## 3. API-key mapping

Static IPs are configured **at the developer-app level**, not per trading
account: multiple Zerodha accounts can be added under one developer profile
and share the same registered IP(s). Deleting an app immediately invalidates
its `api_key`, `api_secret`, and every access token issued under it.
Source: [Kite Connect API FAQ](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/kite-connect-api-faqs);
[How to sign up for Kite Connect](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/how-do-i-sign-up-for-kite-connect).

## 4. Order tagging / algo identification requirements

**Regulatory requirement:** NSE's implementation standard requires every
algo order (the FAQ digest states *all* orders placed via API count as algo
orders under this framework, regardless of OPS rate) to carry a tag
identifying it as such. One source states the exact format for
Internet/Mobile-API-originated orders is a 13-character tag whose **first
12 digits are `444444444444`** and whose **13th digit is `0`, `2`, or `4`**
depending on order category.
Source: [fintrens NSE explainer](https://blogs.fintrens.com/nse-retail-algo-trading-rules-nov-2025-static-ip-order-tagging-compliance-guide/);
corroborated by a secondary aggregation in web search results citing NSE
circular NSE/INVG/67858 (5 May 2025).

**What is explicitly NOT verified:** Kite Connect's own published API
documentation (`kite.trade/docs/connect/v3/orders/`, re-checked directly
during this phase) makes **no mention** of this tag format, an algo-ID
field, or SEBI/NSE retail-algo compliance at all — it documents only a
generic, optional `tag` parameter ("alphanumeric, max 20 chars") with no
connection to this requirement. It is therefore **not confirmed** whether:

- Zerodha's backend applies the exchange-mandated tag automatically based on
  the order's originating channel (most likely, since a client should not
  be trusted to self-report a regulatory classification), or
- the API client is expected to submit it explicitly via the existing `tag`
  field, or some other field this system has not seen documented.

**This codebase does not guess which.** `ComplianceConfig.algo_identifier`
is a required, non-empty field representing whatever identifier the broker's
own algo-onboarding process actually assigns to this account/strategy —
sourced from that confirmation, never invented. `broker/compliance.py`
enforces that it is present and passes it through the existing, documented
`tag` field (the only mechanism Kite's own docs actually describe for
attaching an identifier to an order) as this system's stated, honest choice
pending explicit written confirmation from Zerodha of the correct mechanism
— documented here, not silently assumed correct. **Do not go live until
Zerodha has confirmed in writing which of the above is actually true.**

## 5. Allowed order types

**Regulatory requirement:** "Market and IOC orders are not allowed" for
algo orders, across both equity and commodity segments.
Source: [fintrens NSE explainer](https://blogs.fintrens.com/nse-retail-algo-trading-rules-nov-2025-static-ip-order-tagging-compliance-guide/).
This corroborates, with a specific regulatory citation, what
docs/SPECIFICATION.md section 12.2/13 already required from the broker's
own retail-algo FAQ as understood at the time this project started.

**Kite-specific detail, verified in Phase 15:** Kite's own `order_type` enum
is `MARKET, LIMIT, SL, SL-M`; its `validity` enum is `DAY, IOC, TTL`. The
above regulatory rule excludes `order_type=MARKET` and `validity=IOC` from
algo flow — Kite's API is technically permissive enough to accept both, so
the exclusion must be enforced by this system, not assumed away by the
broker.

**What this system does with it:** `ComplianceConfig.allowed_order_types`
and `ComplianceConfig.allowed_validities` are the authoritative,
config-driven allow-lists `broker/compliance.py` checks every order
against, independent of and in addition to `broker.zerodha.kite_mappings`'s
own (currently coincident) `SUPPORTED_ORDER_TYPES` — two independent checks
for the same real-world constraint is deliberate defense in depth, not
redundancy to be trimmed, matching this codebase's established pattern
elsewhere (e.g. the V1 long-only invariant is checked in multiple
independent layers).

## 6. Broker RMS behavior

Even for the self-hosted "tech-savvy" route, every order still passes
through Zerodha's own Risk Management System before reaching the exchange —
margin sufficiency, holdings sufficiency for a sell, per-order/circuit-limit
checks. This system's own `risk.risk_manager.RiskManager` (Phase 7b) is
independent of and does not replace the broker's RMS: an order this
system's own risk layer approved can still be rejected by Zerodha's RMS
(already modeled in Phase 15's error taxonomy — `MarginException`,
`HoldingException` map to `broker.errors.BrokerRequestError`). The retail
algo framework additionally requires that for broker-hosted (not
self-hosted) strategies, order messages must originate from the broker's
own servers specifically so the broker's RMS can apply in real time; this
system's self-hosted route means *this system's* own outbound request still
must clear the same RMS on Zerodha's side before the exchange ever sees it.
Source: [Kite Connect forum — market protection requirement](https://kite.trade/forum/discussion/15912/preparing-to-comply-with-sebis-retail-algo-rules-static-ip-ratelimits-order-types)
(market orders/SL-M require `market_protection != 0`, rejected otherwise —
moot for this system since market orders are already prohibited under
section 5 above); fintrens NSE explainer (RMS/server-origination
requirement for broker-hosted strategies).

## 7. API rate limits

Verified directly against `kite.trade/docs/connect/v3/exceptions/` in
Phase 15:

| Endpoint class | Limit |
|---|---|
| `/quote` | 1 request/second |
| Historical candle data | 3 requests/second |
| Orders and most other endpoints | 10 requests/second |
| Order placement (daily) | 3,000 orders/day per user (Kite's own general platform limit) |

A request beyond the limit receives HTTP 429, mapped in this codebase to
`broker.errors.BrokerRateLimitError` (Phase 15).

## 8. Order-per-second limits

**Regulatory ceiling for the unregistered tech-savvy route this system
uses:** **10 orders per second**, per client (trading) account, per segment
per exchange. This coincides with Kite's own general-platform order-endpoint
rate limit (item 7 above) but is a *separate* regulatory ceiling specific to
retail algo orders, not merely a side effect of Kite's API rate limiting.
Source: [Zerodha comprehensive overview](https://zerodha.com/z-connect/general/a-comprehensive-overview-of-nses-circular-on-the-new-retail-algo-trading-framework);
[Kite Connect forum](https://kite.trade/forum/discussion/15912/preparing-to-comply-with-sebis-retail-algo-rules-static-ip-ratelimits-order-types)
("A strict 10 orders-per-second rate limit will apply. Requests exceeding
this will receive a 429 response"; "10 orders will be placed, and 5 orders
will be blocked with a 429 response" for an 15-order burst).
Exceeding this rate is exactly the trigger that converts this system from
the unregistered route into one requiring formal exchange algo
registration — a materially different, unimplemented compliance posture.
`ComplianceConfig.max_orders_per_second` is capped at 10 by schema/field
validation for this reason: this system structurally cannot be configured
to claim compliance with a rate the unregistered route does not cover.

## 9. Session/token expiry

A Kite Connect `access_token` is valid for a single trading day only.
Multiple sources converge: tokens are invalidated somewhere between roughly
5:00 AM and 7:30 AM IST daily, and a token generated **after 7:30 AM IST**
is reliably valid for the remainder of that day. There is no single
official documented exact expiry instant — this is drawn from Kite Connect
developer-forum discussions, not `kite.trade`'s own formal docs, which do
not state a precise expiry time.
Sources: [Kite Connect forum — access token expiry time](https://kite.trade/forum/discussion/3468/access-token-expiry-time-everyday);
[Kite Connect forum — earliest time to generate token](https://kite.trade/forum/discussion/13884/what-is-the-earliest-time-in-the-day-i-can-generate-the-access-token-for-the-day).

**What this system does with it:** rather than hard-coding a specific
clock time (which is not consistently documented and could legitimately
differ by a few minutes run to run), `ComplianceConfig.session_max_age_hours`
is a configured maximum session age; `broker/compliance.py` computes the
session's age from `Broker.health_check().login_time` (Phase 15) and
refuses to place an order once that age is exceeded — "expired
authentication" as one of this phase's required hard-failure conditions. A
conservative default (documented in `config/settings.yaml`) is used since
a single sub-day session is this broker's own documented reality, not this
system's invention.

## 10. IP whitelisting

Covered under item 2 (static IP) above — Kite Connect's "IP whitelisting" is
the same mechanism as its static-IP requirement (the developer-console "IP
Whitelist" section is literally where the registered static IP(s) are
entered), not a separate control.

## 11. Audit logging requirements

**Regulatory requirement:** every algo order must carry a traceable,
exchange-assigned identifier specifically to "establish an audit trail" —
i.e. audit logging is not merely good practice here, it is the explicit
purpose of the order-tagging requirement (item 4). This system already
satisfies the underlying need structurally: every `BrokerOrder`/`BrokerFill`
carries a `client_order_id`/`trade_id`, `execution.order_manager.OrderRecord`
timestamps every state transition (`created_at`/`updated_at`), and
`PaperBroker`/`KiteBroker` both log every rejection with a structured reason
(`risk.risk_manager.RiskManager._log_rejections`, `broker.errors.BrokerRequestError`).
What Phase 16 adds on top: `broker/compliance.py` logs every compliance
gate decision (a construction-time refusal, a per-order rejection, a
rate-limit refusal) through the standard `logging` module with a structured
`extra_fields` payload, in the same style already established by
`risk/circuit_breaker.py` and `risk/risk_manager.py` — so a compliance
refusal is auditable the same way a risk rejection already is, not a
new, inconsistent logging convention.
Source: [fintrens NSE explainer](https://blogs.fintrens.com/nse-retail-algo-trading-rules-nov-2025-static-ip-order-tagging-compliance-guide/).

## 12. Retention requirements

**Not found in NSE's retail-algo-specific materials fetched during this
phase's research** (no explicit retention period appeared in any of the
Zerodha/NSE-specific sources above). The figures below are general SEBI
stock-broker record-keeping requirements, reported by third-party legal/
compliance summaries, not read directly from the primary regulation text
during this phase (the primary NSE FAQ PDF could not be fetched — every
attempt timed out; see "What was not verified" below):

- **A commonly cited 5-year minimum** for API/algo activity logs, attributed
  to SEBI (Stock Brokers) Regulations 1992, Regulation 18.
- **A commonly cited 3-year minimum** for general client order records
  (extended indefinitely while a dispute is open), attributed to SEBI
  circular CIR/HO/MIRSD/MIRSD2/CIR/P/2017/124 (30 November 2017), on
  prevention of unauthorized trading.

**This is the one item in this phase's required list this document cannot
mark as verified against a primary source.**
`ComplianceConfig.audit_log_retention_years` defaults to **5** (the more
conservative/longer of the two commonly cited figures) but is explicitly
flagged in its own field docstring as requiring confirmation with the
broker's compliance desk or a SEBI-registered compliance officer before
this system is used to justify an actual retention/deletion policy.

## 13. Broker-specific algo onboarding requirements

1. Sign up at `developers.kite.trade`, subscribe to Kite Connect (₹500/month
   for the full API including WebSocket streaming and historical data, per
   Zerodha's published pricing at the time of this research), and create an
   "app" to obtain `api_key`/`api_secret`.
2. Register a static IP (primary mandatory, secondary optional) in that
   app's profile "IP Whitelist" section (items 2/3/10 above).
3. Confirm with Zerodha (support ticket or account-manager channel; no
   self-serve API for this was found) that the account's order-tagging
   mechanism (item 4) and the account's eligibility for the unregistered
   tech-savvy route (rather than requiring formal exchange algo
   registration) are both actually in place for this specific account.
4. Only after that confirmation should `ComplianceConfig.broker_authorization_confirmed`
   be set to `true` — this field exists specifically to make step 3 an
   explicit, recorded, human decision rather than an assumption a config
   file quietly encodes on someone's behalf.

Sources: [Kite Connect API FAQ](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/kite-connect-api-faqs);
[How to sign up for Kite Connect](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/how-do-i-sign-up-for-kite-connect);
[Kite Connect API pricing](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/what-are-the-charges-for-kite-apis).

## What was not verified (read this before going live)

- The primary source for this entire framework — NSE's official FAQ PDF,
  `FAQ_Retail Algo_03112025_NSE.pdf` — could not be fetched during this
  phase (every attempt timed out); everything above sourced from it was
  read through third-party summaries instead, and is flagged as such at
  each point above.
- The exact algo-tag format (item 4) and whether this system or Zerodha's
  backend is responsible for constructing it.
- The precise static-IP change-frequency limit (item 2) and the exact
  regulatory effective date (one source said April 2025, another August
  2025).
- The exact retention period for algo/API logs specifically (item 12) —
  only general stock-broker record-keeping norms were found.

**None of these gaps are treated as resolved by this document.** They are
exactly why `ComplianceConfig.broker_authorization_confirmed` exists as a
separate, explicit gate from every other field: even a fully and correctly
filled-in `ComplianceConfig` does not mean this system may go live — a
human must have separately confirmed the open items above with Zerodha
first.
