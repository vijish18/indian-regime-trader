from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from config.loader import ConfigError, load_schema, load_settings, validate_against_schema
from config.models import Settings


def test_default_settings_load_and_validate() -> None:
    settings = load_settings()
    assert settings.market.exchange == "NSE"
    assert settings.market.timezone == "Asia/Kolkata"
    assert settings.risk.max_leverage == 1.0
    assert settings.execution.no_market_order_fallback is True
    assert settings.selection.min_holdings == 7


def test_default_settings_pass_schema(valid_settings_dict: dict[str, Any]) -> None:
    schema = load_schema()
    validate_against_schema(valid_settings_dict, schema)  # must not raise


def test_settings_model_matches_yaml(valid_settings_dict: dict[str, Any]) -> None:
    settings = Settings.model_validate(valid_settings_dict)
    assert settings.risk.max_concurrent_positions == 10
    assert settings.risk.daily_loss_warning_pct < settings.risk.daily_loss_reduce_pct


def test_missing_required_section_fails_schema(valid_settings_dict: dict[str, Any]) -> None:
    del valid_settings_dict["risk"]
    schema = load_schema()
    with pytest.raises(ConfigError):
        validate_against_schema(valid_settings_dict, schema)


def test_wrong_type_fails_schema(valid_settings_dict: dict[str, Any]) -> None:
    valid_settings_dict["hmm"]["min_confidence"] = "not-a-number"
    schema = load_schema()
    with pytest.raises(ConfigError):
        validate_against_schema(valid_settings_dict, schema)


def test_load_settings_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigError):
        load_settings(settings_path=missing)


def test_load_settings_rejects_invalid_yaml(tmp_path: Path) -> None:
    bad_file = tmp_path / "settings.yaml"
    bad_file.write_text("not: [a, valid, settings, document", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(settings_path=bad_file)


def test_diversification_validator_rejects_infeasible_holdings(
    valid_settings_dict: dict[str, Any],
) -> None:
    """Regression test for the min_holdings x max_single_name_pct vs.
    calm-regime exposure invariant (see docs/ARCHITECTURE.md, "Resolved
    specification ambiguities").
    """
    valid_settings_dict["selection"]["min_holdings"] = 1
    with pytest.raises(ValidationError):
        Settings.model_validate(valid_settings_dict)


def test_risk_threshold_ordering_enforced(valid_settings_dict: dict[str, Any]) -> None:
    valid_settings_dict["risk"]["daily_loss_warning_pct"] = 0.05  # above reduce and halt
    with pytest.raises(ValidationError):
        Settings.model_validate(valid_settings_dict)


def test_exposure_band_rejects_min_above_max(valid_settings_dict: dict[str, Any]) -> None:
    valid_settings_dict["regime_policy"]["calm"]["min_gross_exposure"] = 0.99
    valid_settings_dict["regime_policy"]["calm"]["max_gross_exposure"] = 0.10
    with pytest.raises(ValidationError):
        Settings.model_validate(valid_settings_dict)


def test_settings_are_frozen(valid_settings_dict: dict[str, Any]) -> None:
    settings = Settings.model_validate(valid_settings_dict)
    with pytest.raises(ValidationError):
        settings.market.exchange = "BSE"
