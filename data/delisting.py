"""Sourced, point-in-time delisting notices and the pre-suspension exit policy."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DelistingNotice:
    instrument_id: str
    known_on: dt.date
    suspension_date: dt.date
    source_url: str
    restriction_end: dt.date | None = None

    def __post_init__(self) -> None:
        if not self.instrument_id or not self.source_url.startswith("https://"):
            raise ValueError("Delisting notice requires an instrument and source URL")
        if self.known_on >= self.suspension_date:
            raise ValueError("Pre-suspension exit requires notice before suspension")
        if self.restriction_end and self.restriction_end < self.suspension_date:
            raise ValueError("Restriction cannot end before suspension")

    def active(self, signal_date: dt.date) -> bool:
        return self.known_on <= signal_date and (
            self.restriction_end is None or signal_date <= self.restriction_end
        )


def load_notices(path: Path) -> tuple[DelistingNotice, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    notices = tuple(
        DelistingNotice(
            instrument_id=r["instrument_id"],
            known_on=dt.date.fromisoformat(r["known_on"]),
            suspension_date=dt.date.fromisoformat(r["suspension_date"]),
            source_url=r["source_url"],
            restriction_end=dt.date.fromisoformat(r["restriction_end"])
            if r.get("restriction_end")
            else None,
        )
        for r in payload["notices"]
    )
    if len({(n.instrument_id, n.known_on) for n in notices}) != len(notices):
        raise ValueError("Duplicate delisting notices")
    return notices
