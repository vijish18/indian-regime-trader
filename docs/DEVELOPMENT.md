# Development Guide

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows; .venv\Scripts\activate.bat for cmd.exe
pip install -e ".[dev]"
cp .env.example .env
```

## Running things

```bash
pytest                    # unit tests only (integration-marked tests are excluded by default)
pytest -m integration      # integration tests only, once any exist
pytest --cov               # with coverage
mypy .                      # strict type checking
ruff check .                 # lint
ruff format .                 # format
python scripts/validate_config.py   # load + print the current config, or fail with a readable error
python main.py                       # Phase 1: loads config, configures logging, exits
```

## Per-phase requirements (specification section 23)

Every phase, before moving to the next one, must satisfy all of:

1. Type hints on every public interface.
2. Unit tests for normal *and* failure paths — a stub that only tests the happy path is not
   done.
3. Deterministic fixtures (fixed seeds, fixed clocks) — no test may depend on wall-clock
   time or unseeded randomness.
4. Structured logging via `monitoring.logger`, not `print`.
5. No hidden globals — dependencies are passed in (constructor or function arguments), not
   imported as module-level singletons.
6. Config-driven thresholds — if a number affects trading behavior, it lives in
   `config/settings.yaml`, not inline in code. See "Adding a configuration parameter"
   below.
7. Explicit timestamps and timezones — `datetime` objects are always timezone-aware;
   internal strategy timestamps are `Asia/Kolkata`, persisted timestamps are UTC (see
   `data/calendar.py`).
8. Backward-compatible database migrations for every schema change (once `storage/` has a
   real migration tool wired up, from Phase 2/3 onward).
9. No network calls in unit tests unless the test is explicitly marked
   `@pytest.mark.integration`.
10. A short README/docstring note on how to run the phase's new code locally.

## Adding a configuration parameter

Configuration changes touch three files together, not one:

1. `config/settings.yaml` — add the value itself.
2. `config/settings.schema.yaml` — add it to the matching section's `required`/`properties`
   (JSON Schema draft 2020-12), including type and range where meaningful.
3. `config/models.py` — add the field to the matching pydantic model, with a `Field(...)`
   constraint mirroring the schema. Add a `model_validator` if the new parameter has a
   cross-field invariant (see `Settings._diversification_can_reach_calm_exposure` for an
   example).

`config/loader.py::load_settings` validates against the schema first (a cheap structural
check with a readable error) and then constructs the typed `Settings` object (which catches
cross-field invariants the schema can't express). Both must pass, so `pytest` and
`python scripts/validate_config.py` both exercise the full path.

## Writing a stub for a new phase

Match the existing style (see any file under `core/`, `risk/`, etc.):

```python
class Something:
    """One-line description of what this will do, referencing the relevant
    docs/SPECIFICATION.md section."""

    def do_thing(self, arg: SomeType) -> ResultType:
        """One-line description of the method's contract."""
        raise NotImplementedError("Phase N: <short description> is not implemented yet.")
```

Type hints and docstrings are not optional for a stub — they are the interface contract
that the next phase implements against, and what the test suite in that phase will be
written to.

## Before running anything that touches the calendar

`config/nse_holidays.csv` ships with only a header row. Populate it from NSE's
published holiday list before running ingestion or a backtest:

```csv
date,description,session_type
2026-01-26,Republic Day,closed
2026-11-08,Muhurat Trading,special
```

`session_type` defaults to `closed`; use `special` for Muhurat sessions, which
can fall on a weekend. The calendar treats a year with no entries as "not
maintained" and refuses to answer for it, rather than assuming the exchange was
open every weekday — so an unpopulated file fails loudly instead of silently
backtesting trades on closed days.

## Data file formats

Ingestion accepts CSV or Parquet, chosen by file extension. Required columns:

| Dataset | Required columns |
|---|---|
| Equity bars | `instrument_id, session_date, open, high, low, close, volume` |
| Index observations | `index_symbol, session_date, close` |
| Instruments | `instrument_id, symbol, exchange, segment, tick_size, price_precision, effective_from` |
| Corporate actions | `instrument_id, action_type, ex_date` |
| Index membership | `index_symbol, instrument_id, effective_from` |

Optional columns are documented on each module's `*_COLUMNS` constant. Dates are
ISO-8601 (`YYYY-MM-DD`); an empty `effective_to` means "still in force".

Split and bonus ratios are **not** interchangeable: a 5-for-1 split is
`ratio_new=5, ratio_old=1` (price × 1/5), while a 1:1 bonus is `ratio_new=1,
ratio_old=1` (price × 1/2). Rights, mergers and demergers require an
`explicit_price_factor` — no factor can be derived from their terms alone.

## Testing conventions

- `tests/unit/` mirrors the top-level package layout where it's useful (e.g.
  `tests/unit/test_config.py` for `config/`).
- Anything touching the filesystem uses `tmp_path`/`tmp_path_factory`, never a path inside
  the repo.
- Anything touching the network, a broker, or a real database is marked
  `@pytest.mark.integration` and excluded by the default `pytest` invocation (see
  `[tool.pytest.ini_options]` in `pyproject.toml`).
- No-look-ahead tests (from Phase 4 onward) follow the pattern: compute a feature/decision
  on data through day T, then again on the same data with days T+1..T+k appended, and assert
  the value at day T is unchanged.
