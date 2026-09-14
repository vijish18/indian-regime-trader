# Indian Market Regime Trading System

An India-focused, long-only systematic equity trading system for NSE-listed cash
equities. Market exposure is gated by a Hidden Markov Model (HMM) that classifies
broad-market volatility regime from NIFTY 50 and India VIX; stock selection,
portfolio construction, risk management, and execution are kept as separate,
independently testable layers. See [docs/SPECIFICATION.md](docs/SPECIFICATION.md)
for the full engineering specification and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for how it maps onto this repository.

**V1 scope:** cash equities only, long only, no leverage, no shorting, no
derivatives, no intraday strategy, daily data/decisions, paper trading required
before any live capital.

## Project status

**Phases 1-4 complete: configuration, the broker-independent data layer,
point-in-time universe construction, and causal feature engineering.**

- **Phase 1** — repository structure, type-safe/validated configuration,
  structured logging, environment handling, unit-test framework.
- **Phases 2-3** — market calendar, point-in-time instrument master, corporate
  actions, point-in-time index membership, local CSV/Parquet storage,
  ingestion, and data-quality validation. Historical data only: there is no
  broker or vendor connection, and `LocalMarketDataProvider.get_quote` raises
  by design so a backtest cannot consult a live book.
- **Universe engine** (`universe/universe.py`) — joins index membership,
  instrument status, and (optionally) corporate actions into a point-in-time
  eligible universe. `UniverseProvider.get_universe(as_of)` returns only
  constituents that were actually members, tradable, and not already subject
  to a completed merger/demerger/delisting on that exact date — the primary
  survivorship-bias control described in `docs/SPECIFICATION.md` section 2.1.
  Every exclusion is recorded with a reason rather than silently dropped.
  Liquidity/trend/momentum filtering is a separate, later concern
  (`universe/stock_selector.py`, still stubbed) — see
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the scope boundary and
  documented data limitations.
- **Feature engineering** (`core/features/feature_engineering.py`) — a
  deliberately small, fully-documented set of 9 causal market-regime features
  from NIFTY 50 and India VIX (returns, realized vol, vol ratio, VIX level/
  change, trend, drawdown, normalized ATR, volume stress). Every rolling
  computation is a trailing, never-centered window with `min_periods` equal
  to its declared lookback, so warm-up is NaN rather than a guess, and
  appending future data never changes an already-computed value (tested
  directly, not just asserted). `FeaturePipeline.audit()` produces a full
  provenance trail (`FeatureSnapshot`: name, timestamp, value, source
  observations, lookback) for every value. This is the HMM's *input*
  pipeline only — the HMM itself (`core/regime/hmm_engine.py`) and the
  train/freeze/apply feature scaler used for walk-forward fitting
  (`core/features/feature_scaler.py`) remain Phase 5.

**No trading, regime, selection, portfolio, risk, or execution logic is
implemented yet** — those modules remain typed stubs that define the interfaces
for later phases (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).

Before running ingestion, populate `config/nse_holidays.csv` from NSE's
published holiday list — it ships empty and the calendar fails closed rather
than guessing whether the exchange was open (see
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)).

## Repository layout

```
config/       Typed, schema-validated configuration (settings.yaml + loader)
core/
  regime/     Market-regime detection (HMM engine, regime policy, model registry)
  features/   Feature engineering + causal feature scaling
data/         Broker-independent market data: interfaces, models, calendar,
              instrument master, corporate actions, membership, storage,
              ingestion, validation
universe/     Point-in-time universe construction + stock selection
portfolio/    Portfolio construction (target weights) + position sizing
risk/         Independent risk management with veto authority
execution/    Order management, position tracking, reconciliation
broker/       Broker-neutral interface + adapters (paper adapter first)
backtest/     Walk-forward backtesting, cost/slippage model, stress testing
monitoring/   Structured logging, alerts, health checks, dashboard
storage/      Persistence layer (table schemas, DB session management)
scripts/      Operational / one-off scripts
tests/        Unit and integration tests
docs/         Specification, architecture, development guide
```

## Getting started

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate.bat on cmd.exe
pip install -e ".[dev]"

cp .env.example .env            # fill in local/paper values; never commit .env

pytest                          # run the unit test suite (no network, no broker)
mypy .                          # type-check
ruff check .                    # lint
```

## Configuration

All parameters that affect trading behavior — regime thresholds, exposure bands,
risk limits, cost assumptions, execution guards — live in
[config/settings.yaml](config/settings.yaml) and are validated against
[config/settings.schema.yaml](config/settings.schema.yaml) and the typed models in
`config/models.py` at load time. Nothing that belongs in configuration should be
hardcoded in module logic. Secrets and per-deployment values (broker keys, database
URL) live in `.env`, never in `settings.yaml`.

## Documentation

- [docs/SPECIFICATION.md](docs/SPECIFICATION.md) — the source engineering specification.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the specification's layers map to this repo, module boundaries, and the phase build plan.
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — local setup, coding standards, test conventions.
