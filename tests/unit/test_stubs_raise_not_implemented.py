"""Phase 1 must not silently implement trading logic. This is a spot-check,
not exhaustive coverage of every stub, that a representative method from
each of the nine separated layers still raises NotImplementedError with a
message identifying which phase will implement it.
"""

from __future__ import annotations

import pytest

from config.loader import load_settings


def test_position_sizer_is_unimplemented() -> None:
    """Portfolio construction, risk management (approve/veto), and
    walk-forward backtesting are implemented; converting an approved
    target weight into a final order quantity is still later work
    (Phase 7c).
    """
    from risk.position_sizer import PositionSizer

    settings = load_settings()
    sizer = PositionSizer(settings.risk)
    with pytest.raises(NotImplementedError, match="Phase 7c"):
        sizer.weight_based_quantity(proposed=None, equity=100.0, price=10.0)  # type: ignore[arg-type]
