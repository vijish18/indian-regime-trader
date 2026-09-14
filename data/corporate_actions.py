"""Corporate actions and price adjustment.

The adjustment rule that matters for backtest correctness:

    A price observed on ``price_date`` is made comparable with prices as of
    ``as_of`` by multiplying it by the product of the factors of all actions
    with ``price_date < ex_date <= as_of``.

Both bounds are load-bearing. The lower bound excludes actions that had
already gone ex when the price was observed (that price already reflects
them). The upper bound excludes actions that had not yet gone ex as of the
decision date -- including them would rewrite history with information that
did not exist yet, which is a look-ahead leak that quietly flatters every
backtest that touches a split (docs/SPECIFICATION.md section 4.1, section 10).
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path

import pandas as pd

from data.interfaces import CorporateActionProvider
from data.models import CorporateAction, CorporateActionType
from data.storage import (
    LocalDataStore,
    parse_date,
    parse_int,
    parse_optional_date,
    parse_optional_decimal,
    parse_optional_str,
    read_table,
    require_columns,
)

CORPORATE_ACTION_COLUMNS = ("instrument_id", "action_type", "ex_date")
"""Required columns. Ratio/amount/factor columns are type-dependent and
validated by ``CorporateAction`` itself."""

_RATIO_QUANTUM = Decimal("0.00000001")
_FACTOR_QUANTUM = Decimal("0.00000001")


class InMemoryCorporateActionProvider(CorporateActionProvider):
    """Corporate-action provider backed by an in-memory record set."""

    def __init__(self, actions: Iterable[CorporateAction]) -> None:
        self._by_instrument: dict[str, list[CorporateAction]] = defaultdict(list)
        for action in actions:
            self._by_instrument[action.instrument_id].append(action)
        for records in self._by_instrument.values():
            records.sort(key=lambda record: (record.ex_date, record.action_type))

    @classmethod
    def from_file(cls, path: Path) -> InMemoryCorporateActionProvider:
        frame = read_table(path)
        require_columns(frame, CORPORATE_ACTION_COLUMNS, str(path))
        return cls(parse_corporate_action_rows(frame, source=str(path)))

    @classmethod
    def from_store(cls, store: LocalDataStore) -> InMemoryCorporateActionProvider:
        return cls.from_file(store.corporate_actions_path())

    def actions_for(
        self, instrument_id: str, start: dt.date, end: dt.date
    ) -> list[CorporateAction]:
        if end < start:
            raise ValueError(f"end {end} precedes start {start}")
        return [
            action
            for action in self._by_instrument.get(instrument_id, ())
            if start <= action.ex_date <= end
        ]

    def cumulative_adjustment_factor(
        self, instrument_id: str, price_date: dt.date, as_of: dt.date
    ) -> Decimal:
        if as_of < price_date:
            raise ValueError(
                f"as_of {as_of} precedes price_date {price_date}; an adjustment "
                "cannot be computed backwards in time"
            )
        factor = Decimal(1)
        for action in self._by_instrument.get(instrument_id, ()):
            if price_date < action.ex_date <= as_of:
                factor *= action.price_adjustment_factor()
        return factor

    def identity_change(
        self, instrument_id: str, as_of: dt.date
    ) -> CorporateAction | None:
        for action in self._by_instrument.get(instrument_id, ()):
            if action.is_identity_change and action.ex_date >= as_of:
                return action
        return None

    def cash_dividends(
        self, instrument_id: str, start: dt.date, end: dt.date
    ) -> list[CorporateAction]:
        """Dividends with ``ex_date`` in ``[start, end]``.

        Dividends are cash events rather than price adjustments: the
        backtester credits them to cash so the equity curve is total-return
        and therefore comparable with a total-return benchmark
        (docs/ARCHITECTURE.md).
        """
        return [
            action
            for action in self.actions_for(instrument_id, start, end)
            if action.action_type is CorporateActionType.DIVIDEND
        ]

    def instruments_with_actions(self) -> frozenset[str]:
        return frozenset(self._by_instrument)


def parse_corporate_action_rows(
    frame: pd.DataFrame, source: str
) -> list[CorporateAction]:
    """Convert a corporate-action table into ``CorporateAction`` records."""
    actions: list[CorporateAction] = []
    for position, row in enumerate(frame.to_dict("records"), start=2):
        try:
            actions.append(
                CorporateAction(
                    instrument_id=str(row["instrument_id"]).strip(),
                    action_type=CorporateActionType(
                        str(row["action_type"]).strip().lower()
                    ),
                    ex_date=parse_date(row["ex_date"]),
                    ratio_new=parse_optional_decimal(row.get("ratio_new"), _RATIO_QUANTUM),
                    ratio_old=parse_optional_decimal(row.get("ratio_old"), _RATIO_QUANTUM),
                    cash_amount=parse_optional_decimal(row.get("cash_amount")),
                    explicit_price_factor=parse_optional_decimal(
                        row.get("explicit_price_factor"), _FACTOR_QUANTUM
                    ),
                    record_date=parse_optional_date(row.get("record_date")),
                    announcement_date=parse_optional_date(row.get("announcement_date")),
                    successor_instrument_id=parse_optional_str(
                        row.get("successor_instrument_id")
                    ),
                    adjustment_version=(
                        parse_int(row["adjustment_version"])
                        if "adjustment_version" in row
                        and not _is_blank(row["adjustment_version"])
                        else 1
                    ),
                )
            )
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{source} line {position}: {exc}") from exc
    return actions


def _is_blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())
