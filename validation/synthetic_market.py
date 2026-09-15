"""A reproducible, obviously-synthetic vendor drop (Phase 21).

The validation harness has to ingest *something*, and this repository
ships no market data: the calendar file is empty by design and there is no
vendor feed or broker connection. This module writes a self-consistent set
of vendor-shaped CSVs -- daily bars, NIFTY 50 and India VIX series, an
instrument master and index membership -- so an end-to-end paper run can
start from real ingestion rather than from a provider pre-loaded in
memory.

**These are not market data and must never be mistaken for it.** The
series are a seeded random walk with an alternating calm/volatile block
structure, chosen so the regime engine has something to distinguish and
the selector has something to rank. Nothing concluded about *returns* from
a run over this data means anything. What a run over it does establish is
that every stage accepts the previous stage's output, that the invariants
in ``validation.invariants`` hold throughout, and that failures are
handled the way each phase claimed -- none of which depends on the numbers
being real.

Point the harness at a real historical drop instead and the same checks
run unchanged; that is the intended use before go-live.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_SYMBOL = "NIFTY50"
VIX_SYMBOL = "INDIAVIX"


@dataclass(frozen=True)
class SyntheticMarket:
    """Where the generated vendor files landed, and what is in them."""

    root: Path
    instrument_ids: tuple[str, ...]
    sessions: tuple[dt.date, ...]
    bar_files: tuple[Path, ...]
    index_file: Path
    vix_file: Path
    instruments_file: Path
    membership_file: Path

    @property
    def first_session(self) -> dt.date:
        return self.sessions[0]

    @property
    def last_session(self) -> dt.date:
        return self.sessions[-1]


def write_synthetic_market(
    root: Path,
    *,
    start: dt.date = dt.date(2015, 1, 1),
    sessions: int = 260,
    instruments: int = 6,
    seed: int = 5,
    regime_block: int = 20,
) -> SyntheticMarket:
    """Write a complete vendor drop under ``root`` and describe it."""
    root.mkdir(parents=True, exist_ok=True)
    session_dates = _weekdays(start, sessions)
    nifty, vix = _index_series(sessions, seed=seed, regime_block=regime_block)

    index_file = _write_index(root, INDEX_SYMBOL, session_dates, nifty, seed=seed)
    vix_file = _write_index(root, VIX_SYMBOL, session_dates, vix, seed=seed + 1)

    instrument_ids = tuple(f"NSE:S{index:02d}" for index in range(instruments))
    bar_files = tuple(
        _write_bars(
            root,
            instrument_id,
            session_dates,
            _stock_series(
                sessions,
                start_price=100.0 + position * 15.0,
                drift=0.0002 + position * 0.00015,
                volatility=0.012 + position * 0.002,
                seed=seed * 1000 + 100 + position,
            ),
            seed=seed * 1000 + 200 + position,
        )
        for position, instrument_id in enumerate(instrument_ids)
    )

    listed_from = start - dt.timedelta(days=730)
    instruments_file = _write_instruments(root, instrument_ids, listed_from)
    membership_file = _write_membership(root, instrument_ids, listed_from)

    return SyntheticMarket(
        root=root,
        instrument_ids=instrument_ids,
        sessions=tuple(session_dates),
        bar_files=bar_files,
        index_file=index_file,
        vix_file=vix_file,
        instruments_file=instruments_file,
        membership_file=membership_file,
    )


# -- series ---------------------------------------------------------------


def _weekdays(start: dt.date, count: int) -> list[dt.date]:
    dates: list[dt.date] = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += dt.timedelta(days=1)
    return dates


def _index_series(
    sessions: int, *, seed: int, regime_block: int
) -> tuple[np.ndarray, np.ndarray]:
    """A NIFTY/VIX pair with alternating calm and volatile blocks, so the
    HMM has two genuinely different states to separate rather than one
    noisy one."""
    returns_rng = np.random.default_rng(seed)
    vix_rng = np.random.default_rng(seed + 1)
    calm = (np.arange(sessions) // regime_block) % 2 == 0
    daily_volatility = np.where(calm, 0.004, 0.020)
    returns = returns_rng.normal(0.0002, daily_volatility)
    nifty = 10_000.0 * np.cumprod(1.0 + returns)
    vix = np.where(calm, 12.0, 30.0) + vix_rng.normal(0.0, 1.0, sessions)
    return nifty, np.clip(vix, 8.0, 65.0)


def _stock_series(
    sessions: int, *, start_price: float, drift: float, volatility: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, volatility, sessions)
    return start_price * np.cumprod(1.0 + returns)


# -- files ----------------------------------------------------------------


def _write_bars(
    root: Path, instrument_id: str, dates: list[dt.date], closes: np.ndarray, *, seed: int
) -> Path:
    rng = np.random.default_rng(seed)
    volumes = rng.integers(2_000_000, 8_000_000, size=len(dates))
    frame = pd.DataFrame(
        {
            "instrument_id": instrument_id,
            "session_date": [day.isoformat() for day in dates],
            "open": [f"{close:.2f}" for close in closes],
            "high": [f"{close * 1.01:.2f}" for close in closes],
            "low": [f"{close * 0.99:.2f}" for close in closes],
            "close": [f"{close:.2f}" for close in closes],
            "volume": volumes,
        }
    )
    path = root / f"bars_{instrument_id.replace(':', '_')}.csv"
    frame.to_csv(path, index=False)
    return path


def _write_index(
    root: Path, symbol: str, dates: list[dt.date], closes: np.ndarray, *, seed: int
) -> Path:
    """High/low/volume are written even though only ``close`` is required:
    the feature pipeline's ATR and volume-stress features read them, and a
    drop without them would fail warm-up rather than the ingest."""
    rng = np.random.default_rng(seed)
    volumes = rng.integers(1_500_000_000, 2_500_000_000, size=len(dates))
    frame = pd.DataFrame(
        {
            "index_symbol": symbol,
            "session_date": [day.isoformat() for day in dates],
            "open": [f"{close:.2f}" for close in closes],
            "high": [f"{close * 1.006:.2f}" for close in closes],
            "low": [f"{close * 0.994:.2f}" for close in closes],
            "close": [f"{close:.2f}" for close in closes],
            "volume": volumes,
        }
    )
    path = root / f"index_{symbol}.csv"
    frame.to_csv(path, index=False)
    return path


def _write_instruments(
    root: Path, instrument_ids: tuple[str, ...], effective_from: dt.date
) -> Path:
    frame = pd.DataFrame(
        {
            "instrument_id": list(instrument_ids),
            "symbol": [instrument_id.split(":")[-1] for instrument_id in instrument_ids],
            "exchange": "NSE",
            "segment": "equity",
            "tick_size": "0.05",
            "price_precision": 2,
            "lot_size": 1,
            "status": "active",
            "effective_from": effective_from.isoformat(),
        }
    )
    path = root / "instruments.csv"
    frame.to_csv(path, index=False)
    return path


def _write_membership(
    root: Path, instrument_ids: tuple[str, ...], effective_from: dt.date
) -> Path:
    frame = pd.DataFrame(
        {
            "index_symbol": INDEX_SYMBOL,
            "instrument_id": list(instrument_ids),
            "effective_from": effective_from.isoformat(),
        }
    )
    path = root / "index_membership.csv"
    frame.to_csv(path, index=False)
    return path
