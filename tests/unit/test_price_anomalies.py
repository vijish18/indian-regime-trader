import datetime as dt
from pathlib import Path

from data.price_anomalies import scan_price_losses


def test_loss_screen_boundaries_and_action_is_not_auto_accepted(tmp_path: Path) -> None:
    bars = tmp_path / "raw/equity_bars"
    bars.mkdir(parents=True)
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "corporate_actions.csv").write_text(
        "instrument_id,ex_date,action_type\nNSE:TEST,2020-01-03,bonus\n"
    )
    (bars / "test.csv").write_text(
        "instrument_id,session_date,open,low,close\n"
        "NSE:TEST,2020-01-01,100,100,100\n"
        "NSE:TEST,2020-01-02,100,90,90\n"
        "NSE:TEST,2020-01-03,45,44,45\n"
        "NSE:TEST,2020-01-06,50,44,49\n"
    )
    findings = scan_price_losses(tmp_path, dt.date(2020, 1, 2), dt.date(2020, 1, 6))
    assert len(findings) == 2  # exactly 10% excluded; >10% intraday included
    assert findings[0]["previous_session_date"] == "2020-01-02"
    assert findings[0]["recorded_actions"][0]["action_type"] == "bonus"
    assert findings[0]["status"] == "requires_source_and_accounting_review"
    assert findings[1]["declines_pct"] == {"open_to_low": -12.0}
