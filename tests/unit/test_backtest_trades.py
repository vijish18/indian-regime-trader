import pytest

from monitoring.backtest_trades import history_from_states


def state(rows, stops=(), holdings=None):  # type: ignore[no-untyped-def]
    return {
        "trade_log_rows": rows,
        "stop_exits": list(stops),
        "holdings": holdings or {},
        "regime_points": {},
        "confidence_points": {},
        "fills": [{"order": {"current_weight": 0.1, "target_weight": 0}} for _ in rows],
    }


def row(side, quantity, price, fee, day="2026-09-01"):  # type: ignore[no-untyped-def]
    return {
        "instrument_id": "NSE:TEST",
        "signal_date": day,
        "execution_date": day,
        "side": side,
        "quantity": quantity,
        "fill_price": price,
        "gross_value": quantity * price,
        "cost": fee,
        "net_value": quantity * price + (fee if side == "buy" else -fee),
    }


def test_average_entry_and_fees_are_allocated_across_partial_exits() -> None:
    rows = [
        row("buy", 10, 100, 10),
        row("buy", 10, 200, 20),
        row("sell", 5, 180, 5),
        row("sell", 15, 160, 15),
    ]
    result = history_from_states([state(rows)])
    first, last = result["rows"]
    assert first["entry_price"] == 150
    assert first["entry_fees"] == 7.5
    assert first["net_pnl"] == 137.5
    assert first["shares_after_exit"] == 15
    assert first["shares_at_end"] == last["shares_at_end"] == 0
    assert last["entry_fees"] == 22.5
    assert result["summary"]["realized_net_pnl"] == 250
    assert result["summary"]["total_costs"] == 50


def test_stop_reason_is_recovered_from_checkpoint_not_guessed() -> None:
    rows = [row("buy", 10, 100, 1), row("sell", 10, 96, 2)]
    stop = {
        "execution_date": "2026-09-01",
        "quantity": 10,
        "breach": {
            "instrument_id": "NSE:TEST",
            "fill_price": 96,
            "stop_level": 97,
            "reason": "hard_stop",
        },
    }
    result = history_from_states([state(rows, [stop])])
    assert result["rows"][0]["reason"] == "hard_stop"
    assert result["rows"][0]["stop_level"] == 97
    assert result["rows"][0]["net_pnl"] == -43
    assert history_from_states([state(rows)])["rows"][0]["reason"] == "portfolio_rebalance"


def test_fold_liquidation_and_reentry_keep_separate_cost_basis() -> None:
    sell = row("sell", 10, 105, 1)
    sell["exit_reason"] = "fold_end_liquidation"
    result = history_from_states(
        [
            state([row("buy", 10, 100, 1), sell]),
            state([row("buy", 1, 200, 2), row("sell", 1, 190, 2)]),
        ]
    )
    assert result["rows"][0]["reason"] == "fold_end_liquidation"
    assert result["rows"][1]["entry_price"] == 200
    assert result["rows"][1]["net_pnl"] == -14
    assert result["rows"][1]["fold"] == 2


def test_unmatched_sell_is_rejected() -> None:
    with pytest.raises(ValueError, match="Sell exceeds"):
        history_from_states([state([row("sell", 1, 100, 1)])])
