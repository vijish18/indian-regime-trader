# Takeover notes - 21 September 2026

Scope: design PDF, repository architecture and key decision/execution paths,
current saved dashboard payload, and the new local monitoring interface.
This is an initial engineering assessment, not a claim that every module has
been audited or that the multi-hour backtest has been independently reproduced.
The existing `scripts/run_walk_forward.py` change was preserved. Another process
committed it during this session (HEAD advanced from `0c953a7` to `0fa925d`), also
including some dashboard files already written here. No commits were created
by this takeover session. Runtime snapshots also changed during the review.

## Mental model

1. Reconstruct eligible NSE cash equities at each date from bhavcopy, instrument
   history and corporate actions. Use adjusted prices for comparisons and raw
   prices for fills and valuation.
2. Fit features/scaling and a Gaussian HMM on the training window. Forward-filter
   observations to obtain causal state probabilities. The HMM is a market-risk
   classifier, not a stock-price forecaster.
3. Translate measured volatility and confidence into exposure, with confirmation
   and flicker checks. State labels are descriptions, not allocation rules.
4. Independently rank stocks by six configured factors. Combine rankings with
   the exposure budget, then apply name, liquidity, correlation and sector caps.
5. Risk has veto power. Execution manages orders, fills, reconciliation and
   recovery. Daily decisions and next-session fills differ from quote updates.
6. Evaluate the HMM against exposure-policy baselines on an otherwise common
   downstream pipeline. This isolates the exposure policy, not selection alpha.

The paper-book implementation is a separate path: stop/ranking-driven exits
and equal-slot replenishment using current quotes. It must not be described as
the same strategy merely because it uses the same stock ranks and stop rules.

## First priorities

### 1. Unify the paper and production decision path

`scripts/refresh_live_book.py` computes `slot = book.budget / max_positions`
and calls `book.reallocate` without the allocation engine or independent risk
manager. The initially inspected snapshot had about 94.97% exposure while confidence
is 53.67%, below its 60% threshold. Low confidence would force UNCERTAIN in
`core/regime/allocation.py`; this observation demonstrates that the paper book
is not enforcing that policy. It is not a request to liquidate the account.

Build one composition root around the orchestrator and use the paper broker
through it. Publish decision IDs, approved exposure, risk reasons, order/fill
IDs and timestamps to the monitor. Test a low-confidence session and prove no
new positions are opened, then test restart recovery on the same path.

`app/service.py` explicitly only heartbeats today. A healthy container does not
mean the strategy is running. Display process health, strategy readiness and
data freshness as separate facts.

### 2. Repair experiment accounting before choosing a better factor

`backtest/walk_forward.py` carries equity but not holdings across folds.
Terminal positions therefore do not generate the same liquidation trade/cost
trail as real sales. The PDF also acknowledges biased closed-trade statistics.
Either carry holdings and cost bases across retraining or book explicit terminal
liquidations and all costs. Reconcile equity P&L to realized + unrealized P&L
and costs across every boundary. The impact on comparative results must be
measured, not dismissed because the behavior is documented.

The pipeline's `buy_and_hold` is an exposure-policy control over the same
selector, not an investable NIFTY index buy-and-hold benchmark. Add a separately
labelled total-return benchmark and a genuinely fixed-holdings comparison.

### 3. Treat the negative result and its explanation separately

The supplied report states HMM CAGR -5.35%, versus -6.08% for a shuffled control,
with all five strategies losing after costs. These are reported results, not
newly reproduced numbers. The document alternates between annual and whole-
period language for gross return; keep units and compounding explicit.

One shuffled sequence does not establish statistically reliable timing skill.
Use many independently seeded controls, paired fold comparisons, uncertainty
intervals that respect time dependence, and an untouched final evaluation
period. Repeatedly choosing factors against the same 33 folds turns those folds
into development data. See Bailey et al.,
[The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

Flat gross performance implicates the combined selection, sizing, timing and
execution assumptions; it does not by itself uniquely prove selection is the
only cause. Run factor ablations under fixed exposure before making that claim.

### 4. Reduce unnecessary turnover through controlled experiments

Compare the existing policy with scheduled rebalances, rank buffers and minimum
weight-change thresholds. Separately test liquidity log/rank transforms and
factor clipping. Do not change several weights and the stop rule together.
Record each hypothesis/configuration before running it; assess gross return,
net return, turnover, drawdown, sector exposure and fold stability together.
These are research proposals, not demonstrated profitable changes.

Compare the stop-enabled run with the preserved no-stop run on identical folds
only after confirming both share data, costs, terminal accounting and execution
semantics. Daily OHLC cannot establish all intraday high/low ordering.

### 5. Complete persistence and operational truth

`storage/database.py` remains a stub. `execution/execution_journal.py` is an
in-memory index, although events also go to logs. Add a durable order/fill
ledger with idempotency, transactional updates, and tested backup/restore.
Use the real trading calendar for paper mutations: the refresh script's
weekday/time-only check intentionally ignores holidays, so old quotes can
look actionable. Point-in-time sector data also needs evidence before sector
limits can be claimed effective on real historical universes.

## Dashboard delivered

The new version-controlled frontend and loopback server replace the need for
Claude's artifact DB channel when viewing locally. See [DASHBOARD.md](DASHBOARD.md).
It visualizes actual saved data, labels stale observations, and provides motion
without manufacturing market activity. It does not modify strategy settings,
start paper refresh scripts, enable live trading, or publish externally.
