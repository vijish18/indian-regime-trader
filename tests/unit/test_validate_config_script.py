from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate_config import main


def test_validate_config_script_succeeds_on_default_settings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["validate_config.py"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Configuration is valid." in captured.out


def test_validate_config_script_fails_on_bad_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    exit_code = main(["validate_config.py", str(missing)])
    assert exit_code == 1
