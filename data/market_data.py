"""Historical market data served from local storage.

This is the historical implementation of ``MarketDataProvider``. A live
implementation added in a later phase satisfies the same interface, so no
caller above the data layer changes when the source does -- which is the point
of keeping the strategy broker-independent (docs/SPECIFICATION.md section 12).

Adjusted prices are computed *as of the end of the requested window*, never as
of today. A walk-forward fold that ends on date T therefore sees the series
exactly as it looked on T, including splits that had gone ex by then and
excluding ones that had not.
"""

from __future__ import annotations

import datetime as dt
from collections import OrderedDict
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path

import pandas as pd

from data.errors import DataNotAvailableError
from data.interfaces import CorporateActionProvider, MarketDataProvider
from data.models import DailyBar, IndexObservation, PriceBasis, Quote
from data.storage import (
    DataLayer,
    LocalDataStore,
    parse_date,
    parse_decimal,
    parse_int,
    parse_optional_decimal,
    parse_optional_int,
    parse_optional_str,
    read_table,
    require_columns,
)

BAR_COLUMNS = (
    "instrument_id",
    "session_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
INDEX_COLUMNS = ("index_symbol", "session_date", "close")


class LocalMarketDataProvider(MarketDataProvider):
    """Serves daily bars and index observations from a :class:`LocalDataStore`."""

    def __init__(
        self,
        store: LocalDataStore,
        corporate_actions: CorporateActionProvider | None = None,
        layer: DataLayer = DataLayer.RAW,
        frame_cache_size: int = 0,
    ) -> None:
        """``frame_cache_size`` memoises parsed files, and defaults to OFF.

        The default matters more than the feature. A backtest reads an
        immutable snapshot, so caching a file is free correctness-wise and
        removes most of the cost: a walk-forward fold asks for the same
        instrument's file on every session, for every strategy, hundreds
        of thousands of times.

        A *live* system reads files that change every day. A cache there
        would serve yesterday's bars as today's, and the system would act
        on them with no indication anything was stale -- the exact class of
        silent wrongness the freshness checks exist to prevent. So this is
        opt-in, and only the backtest opts in.

        Bounded rather than unlimited: the full 2,205-instrument set is
        about 3 GB of frames, and a machine that starts swapping is slower
        than no cache at all. One fold's universe is roughly 900
        instruments, so a limit near that holds the working set.
        """
        self._store = store
        self._corporate_actions = corporate_actions
        self._layer = layer
        self._frame_cache_size = frame_cache_size
        self.unadjustable: dict[str, int] = {}
        """instrument -> bars served unadjusted because a corporate action
        between the bar and the as-of date has no derivable price factor.
        Empty for a run that never priced through such an event."""

        self._frames: OrderedDict[Path, pd.DataFrame] = OrderedDict()

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if self._frame_cache_size <= 0:
            return self._store.read(path)
        cached = self._frames.get(path)
        if cached is not None:
            self._frames.move_to_end(path)
            return cached
        frame = self._store.read(path)
        self._frames[path] = frame
        if len(self._frames) > self._frame_cache_size:
            self._frames.popitem(last=False)
        return frame

    def get_equity_bars(
        self,
        instrument_id: str,
        start: dt.date,
        end: dt.date,
        price_basis: PriceBasis = PriceBasis.RAW,
    ) -> list[DailyBar]:
        if end < start:
            raise ValueError(f"end {end} precedes start {start}")
        path = self._store.equity_bars_path(instrument_id, self._layer)
        frame = self._read_frame(path)
        require_columns(frame, BAR_COLUMNS, str(path))

        # Narrow the frame to the requested window *before* building
        # DailyBar objects. Parsing is where the cost is -- 35ms of a 46ms
        # call on an eleven-year file -- because each row becomes six
        # Decimals, and a backtest asks for a two-year window out of
        # eleven years, thousands of times. Parsing rows only to discard
        # them made a single walk-forward fold take seven hours.
        #
        # The comparison is on the raw column so no row is parsed to find
        # out whether it was wanted.
        session_dates = pd.to_datetime(frame["session_date"], errors="coerce").dt.date
        in_window = (session_dates >= start) & (session_dates <= end)
        bars = parse_bar_rows(frame[in_window], source=str(path))
        bars.sort(key=lambda bar: bar.session_date)
        if price_basis is PriceBasis.RAW:
            return bars
        return self._adjust(instrument_id, bars, as_of=end)

    def get_index_observations(
        self, index_symbol: str, start: dt.date, end: dt.date
    ) -> list[IndexObservation]:
        if end < start:
            raise ValueError(f"end {end} precedes start {start}")
        path = self._store.index_path(index_symbol, self._layer)
        frame = self._store.read(path)
        require_columns(frame, INDEX_COLUMNS, str(path))
        observations = [
            observation
            for observation in parse_index_rows(frame, source=str(path))
            if start <= observation.session_date <= end
        ]
        observations.sort(key=lambda observation: observation.session_date)
        return observations

    def available_range(self, instrument_id: str) -> tuple[dt.date, dt.date] | None:
        path = self._store.equity_bars_path(instrument_id, self._layer)
        if not self._store.exists(path):
            return None
        frame = self._store.read(path)
        require_columns(frame, ("session_date",), str(path))
        dates = [parse_date(value) for value in frame["session_date"]]
        if not dates:
            return None
        return min(dates), max(dates)

    def get_quote(self, instrument_id: str) -> Quote:
        raise DataNotAvailableError(
            "LocalMarketDataProvider serves historical bars only; live quotes "
            "require a broker-backed provider (not implemented until the broker "
            "adapter phase)"
        )

    def _adjust(
        self, instrument_id: str, bars: list[DailyBar], as_of: dt.date
    ) -> list[DailyBar]:
        if self._corporate_actions is None:
            raise DataNotAvailableError(
                f"cannot serve adjusted prices for {instrument_id}: this provider "
                "was constructed without a CorporateActionProvider"
            )
        adjusted: list[DailyBar] = []
        for bar in bars:
            try:
                factor = self._corporate_actions.cumulative_adjustment_factor(
                    instrument_id, bar.session_date, as_of
                )
            except ValueError:
                # Some corporate actions have no derivable price factor -- a
                # demerger's terms do not determine one, and nobody supplied
                # an explicit override. The bar is served unadjusted rather
                # than raising.
                #
                # Handled here rather than at each caller because it is not a
                # property of any one caller. NSE:VEDL's 2026-04-30 demerger
                # reached this through three separate paths on three separate
                # runs -- _next_open, _last_close, then market_liquidity_stats
                # -- each costing hours, and each "fix" only moved the crash
                # to the next call site. Refusing to serve a price for an
                # instrument the portfolio is holding is the wrong answer at
                # every one of them.
                #
                # Counted, never silent: `unadjustable` names every instrument
                # served this way, so a run that leaned on unadjusted prices
                # is distinguishable afterwards from one that did not. The
                # universe already excludes these instruments from selection
                # (universe/bhavcopy_universe.py), so this is the residue --
                # positions bought before the event and still held through it.
                self.unadjustable[instrument_id] = (
                    self.unadjustable.get(instrument_id, 0) + 1
                )
                adjusted.append(bar)
                continue
            adjusted.append(bar if factor == Decimal(1) else bar.adjusted(factor))
        return adjusted


def parse_bar_rows(frame: pd.DataFrame, source: str) -> list[DailyBar]:
    """Convert a bar table into ``DailyBar`` records.

    Deliberately does not reject economically impossible bars -- that is
    ``data.data_quality``'s job, and raw vendor data must stay representable
    so it can be reported rather than silently dropped.
    """
    bars: list[DailyBar] = []
    for position, row in enumerate(frame.to_dict("records"), start=2):
        try:
            bars.append(
                DailyBar(
                    instrument_id=str(row["instrument_id"]).strip(),
                    session_date=parse_date(row["session_date"]),
                    open=parse_decimal(row["open"]),
                    high=parse_decimal(row["high"]),
                    low=parse_decimal(row["low"]),
                    close=parse_decimal(row["close"]),
                    volume=parse_int(row["volume"]),
                    price_basis=PriceBasis(
                        (parse_optional_str(row.get("price_basis")) or "raw").lower()
                    ),
                    adjustment_factor=parse_optional_decimal(
                        row.get("adjustment_factor"), Decimal("0.00000001")
                    )
                    or Decimal(1),
                    data_source=parse_optional_str(row.get("data_source")) or "unknown",
                    trading_series=parse_optional_str(row.get("trading_series")),
                )
            )
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{source} line {position}: {exc}") from exc
    return bars


def parse_index_rows(frame: pd.DataFrame, source: str) -> list[IndexObservation]:
    """Convert an index table into ``IndexObservation`` records."""
    observations: list[IndexObservation] = []
    for position, row in enumerate(frame.to_dict("records"), start=2):
        try:
            observations.append(
                IndexObservation(
                    index_symbol=str(row["index_symbol"]).strip(),
                    session_date=parse_date(row["session_date"]),
                    close=parse_decimal(row["close"]),
                    open=parse_optional_decimal(row.get("open")),
                    high=parse_optional_decimal(row.get("high")),
                    low=parse_optional_decimal(row.get("low")),
                    volume=parse_optional_int(row.get("volume")),
                    data_source=parse_optional_str(row.get("data_source")) or "unknown",
                )
            )
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{source} line {position}: {exc}") from exc
    return observations


def bars_to_frame(bars: Iterable[DailyBar]) -> pd.DataFrame:
    """Serialize bars for storage. Prices become text so CSV round-trips
    exactly; Parquet keeps them as strings for the same reason.
    """
    return pd.DataFrame(
        [
            {
                "instrument_id": bar.instrument_id,
                "session_date": bar.session_date.isoformat(),
                "open": str(bar.open),
                "high": str(bar.high),
                "low": str(bar.low),
                "close": str(bar.close),
                "volume": bar.volume,
                "price_basis": bar.price_basis.value,
                "adjustment_factor": str(bar.adjustment_factor),
                "data_source": bar.data_source,
            }
            for bar in bars
        ],
        columns=[
            "instrument_id",
            "session_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "price_basis",
            "adjustment_factor",
            "data_source",
        ],
    )


def index_observations_to_frame(observations: Iterable[IndexObservation]) -> pd.DataFrame:
    """Serialize index observations for storage."""
    return pd.DataFrame(
        [
            {
                "index_symbol": observation.index_symbol,
                "session_date": observation.session_date.isoformat(),
                "open": None if observation.open is None else str(observation.open),
                "high": None if observation.high is None else str(observation.high),
                "low": None if observation.low is None else str(observation.low),
                "close": str(observation.close),
                "volume": observation.volume,
                "data_source": observation.data_source,
            }
            for observation in observations
        ],
        columns=[
            "index_symbol",
            "session_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "data_source",
        ],
    )


def load_bars_from_path(path: Path) -> list[DailyBar]:
    """Read bars from an arbitrary CSV/Parquet file (a vendor drop, say),
    without assuming the managed storage layout.
    """
    frame = read_table(path)
    require_columns(frame, BAR_COLUMNS, str(path))
    return parse_bar_rows(frame, source=str(path))
