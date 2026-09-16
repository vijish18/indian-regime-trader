"""Phase 1 must not silently implement trading logic. This is a spot-check,
not exhaustive coverage of every stub, that a representative method from
each of the nine separated layers still raises NotImplementedError with a
message identifying which phase will implement it.
"""

from __future__ import annotations

import pytest

from config.loader import load_settings


def test_position_sizer_is_implemented() -> None:
    """This asserted the Phase 7c stub still raised. It is now implemented,
    so the inverse is asserted instead: the one module allowed to compute a
    final order quantity actually computes one.

    The substantive tests live in ``tests/unit/test_position_sizer.py``;
    this only guards against a regression to the stub, which would leave
    the system unable to size any order at all.
    """
    from risk.position_sizer import PositionSizer, PositionSizingError

    settings = load_settings()
    sizer = PositionSizer(settings.risk)

    assert sizer.risk_based_quantity(1_000_000.0, 250.0, 10.0) > 0

    # A broken input still refuses rather than guessing -- and refuses with
    # this module's own error, not NotImplementedError.
    with pytest.raises(PositionSizingError):
        sizer.risk_based_quantity(1_000_000.0, 250.0, 0.0)
