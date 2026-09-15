"""A tiny shared "last heartbeat" marker so ``Orchestrator`` (which beats
it, once per monitoring-loop iteration) and ``monitoring.health.HealthChecker``
(which reads it, for ``check_heartbeat``) agree on the same value without
either needing a reference to the other -- both are constructed against
this one object at wiring time instead.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable


class Heartbeat:
    def __init__(self, clock: Callable[[], dt.datetime] | None = None) -> None:
        self._clock: Callable[[], dt.datetime] = clock or (lambda: dt.datetime.now(dt.UTC))
        self._last: dt.datetime | None = None

    def beat(self) -> None:
        self._last = self._clock()

    def last(self) -> dt.datetime | None:
        return self._last
