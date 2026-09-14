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

**Phase 1 — Repository & configuration.** This phase creates the repository
structure, a type-safe/validated configuration system, environment handling, and
the unit-test framework. **No trading, regime, selection, portfolio, risk, or
execution logic is implemented yet** — those modules exist as typed stubs that
define the intended interfaces for later phases (see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the phase plan).

## Repository layout

```
config/       Typed, schema-validated configuration (settings.yaml + loader)
core/
  regime/     Market-regime detection (HMM engine, regime policy, model registry)
  features/   Feature engineering + causal feature scaling
data/         Market data ingestion, instrument master, corporate actions, calendar
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
