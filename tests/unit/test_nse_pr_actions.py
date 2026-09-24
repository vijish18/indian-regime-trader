import io
import zipfile

import pytest

from data.nse_pr_actions import all_terms, archive_actions, parse_date


@pytest.mark.parametrize("raw", ["23/12/2020", "2020-12-23", "23-12-2020"])
def test_pr_date_formats(raw: str) -> None:
    assert parse_date(raw) == "2020-12-23"


def test_special_dividend_and_bonus_in_compound_purposes() -> None:
    assert all_terms("DIV- 6.75 SPLDV- 2.75") == [
        {"action_type": "dividend", "cash_amount": "9.50"}
    ]
    assert all_terms("AGM/DIV-RS38/SPLDIV-RS27") == [
        {"action_type": "dividend", "cash_amount": "65"}
    ]
    assert all_terms("BONUS 1:1/DIV-RS 29") == [
        {"action_type": "bonus", "ratio_new": "1", "ratio_old": "1"},
        {"action_type": "dividend", "cash_amount": "29"},
    ]
    assert all_terms("INTDIV-RS 974 PER SH")[0]["cash_amount"] == "974"
    assert all_terms("INT DIV RS 30/- PER SHARE")[0]["cash_amount"] == "30"


def test_split_terms_and_incomplete_terms_are_not_invented() -> None:
    assert all_terms("FVSPLT FRM RS 10 TO RS 2") == [
        {"action_type": "split", "ratio_new": "10", "ratio_old": "2"}
    ]
    assert all_terms("INTERIM DIVIDEND") == [{"action_type": "unresolved"}]
    assert all_terms("DEMERGER") == [{"action_type": "demerger"}]


def test_pr_archive_member_and_raw_purpose_preserved() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr(
            "bc23122020.csv",
            "SYMBOL,EX_DT,PURPOSE\r\r\nMAJESCO,23/12/2020,INTDIV-RS 974 PER SH\r\r\n",
        )
    row = archive_actions(buffer.getvalue())[0]
    assert row["SYMBOL"] == "MAJESCO"
    assert row["PURPOSE"] == "INTDIV-RS 974 PER SH"
    assert row["archive_member"] == "bc23122020.csv"
