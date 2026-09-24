import datetime as dt
from decimal import Decimal

import pytest

from data.models import CorporateActionType
from data.nse_corporate_actions import to_corporate_actions
from scripts.apply_action_corrections import apply
from scripts.reconcile_action_evidence import reconcile


def test_api_preserves_special_dividend_and_combined_bonus() -> None:
    result = to_corporate_actions(
        [
            {
                "symbol": "TEST",
                "series": "EQ",
                "exDate": "31-May-2018",
                "subject": (
                    "Bonus 1:1/Dividend - Rs 20.50 Per Share/"
                    "Special Dividend - Rs 10 Per Share"
                ),
            }
        ]
    )
    assert not result.unparsed
    assert len(result.actions) == 2
    assert result.actions[0].action_type is CorporateActionType.BONUS
    assert result.actions[1].cash_amount == Decimal("30.50")
    assert result.actions[1].ex_date == dt.date(2018, 5, 31)


def test_reference_patch_requires_reviewed_old_value_and_preserves_input() -> None:
    row = {
        "instrument_id": "NSE:TEST",
        "ex_date": "2020-01-01",
        "action_type": "dividend",
        "cash_amount": "20.50",
    }
    correction = {
        **row,
        "cash_amount": "30.50",
        "expected_cash_amount": "20.50",
        "source_url": "https://nsearchives.nseindia.com/example.zip",
        "source_sha256": "a" * 64,
    }
    repaired = apply([row], [correction])
    assert row["cash_amount"] == "20.50"
    assert repaired[0]["cash_amount"] == "30.50"
    with pytest.raises(ValueError, match="Old amount"):
        apply(repaired, [correction])
    with pytest.raises(ValueError, match="Duplicate correction"):
        apply([row], [correction, correction])


def test_source_match_does_not_certify_accounting_or_duplicate_series() -> None:
    action = {
        "instrument_id": "NSE:TEST",
        "ex_date": "2020-01-01",
        "action_type": "dividend",
        "cash_amount": "30.50",
    }
    row = {
        "SYMBOL": "TEST",
        "EX_DT": "01/01/2020",
        "RECORD_DT": "02/01/2020",
        "PURPOSE": "DIV-RS20.50/SPLDIV-RS10",
        "SERIES": "EQ",
    }
    evidence = {
        "2019-12-30": {
            "source_url": "https://nsearchives.nseindia.com/example.zip",
            "sha256": "a" * 64,
            "rows": [row, {**row, "SERIES": "BE"}],
        }
    }
    audit = {
        "events": [{"instrument_id": "NSE:TEST", "ex_date": "2020-01-01", "actions": [action]}],
        "held_days": {"NSE:TEST": ["2020-01-01"]},
    }
    result = reconcile(audit, evidence, [action])
    assert result["events"][0]["source_status"] == "matched"
    assert len(result["events"][0]["source_checks"][0]["source_matches"]) == 1
    assert result["validation_status"] == "blocked"
    assert result["accounting_blockers"]


def test_complex_action_with_factor_still_fails_preflight(tmp_path, monkeypatch) -> None:
    from scripts import preflight_backtest

    monkeypatch.setattr(preflight_backtest, "REFERENCE", tmp_path)
    (tmp_path / "corporate_actions.csv").write_text(
        "instrument_id,action_type,ex_date,explicit_price_factor\n"
        "NSE:TEST,demerger,2020-01-01,0.5\n"
    )
    check = preflight_backtest.check_unadjustable_actions()
    assert check.status == preflight_backtest.FAIL
