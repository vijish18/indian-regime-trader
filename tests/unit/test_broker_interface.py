from __future__ import annotations

import pytest

from broker.base import Broker


def test_broker_base_cannot_be_instantiated_directly() -> None:
    """Broker is an abstract contract (docs/SPECIFICATION.md section 12) --
    only a concrete adapter should ever be instantiated.
    """
    with pytest.raises(TypeError):
        Broker()  # type: ignore[abstract]


def test_paper_broker_implements_full_interface() -> None:
    from broker.adapters.paper_broker import PaperBroker

    assert issubclass(PaperBroker, Broker)
    for method in (
        "get_account",
        "get_positions",
        "get_open_orders",
        "get_quotes",
        "place_order",
        "modify_order",
        "cancel_order",
        "close_position",
        "close_all_positions",
        "health_check",
    ):
        assert hasattr(PaperBroker, method)
