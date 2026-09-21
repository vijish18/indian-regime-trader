# Paper runtime implementation status

The new `app.paper_runtime` entry point is work in progress. It has not been
validated end to end or deployed for unattended paper trading. Committing this
code does not start a backtest or a paper account.

Implemented: paper-only broker composition, timestamp validation for quotes,
atomic ledger snapshots, and closed-session signal selection. Backtests now
support session checkpoints, input fingerprints, atomic fold files, and explicit
fold-end liquidation with execution costs. `IRT_DATA_ROOT` selects an isolated
data rebuild; generated data and account credentials remain untracked.

Before enabling the paper runtime, verify ledger recovery and order
deduplication, enforce a single writer, handle partial-cycle failures, check
halt behavior before matching resting orders, and validate stop-loss cost
basis. Dashboard integration, fresh-data validation, retirement of legacy
state writers, and Azure deployment are still pending.
