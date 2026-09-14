from __future__ import annotations

import json

import pytest

from config.models import LoggingConfig
from monitoring.logger import configure_logging, get_logger


def test_json_logging_emits_structured_records(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(LoggingConfig(level="INFO", format="json"))
    logger = get_logger("test.component")
    logger.info("hello world")

    captured = capsys.readouterr()
    record = json.loads(captured.out.strip().splitlines()[-1])

    assert record["component"] == "test.component"
    assert record["message"] == "hello world"
    assert record["level"] == "INFO"
    assert "timestamp" in record


def test_text_logging_does_not_emit_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(LoggingConfig(level="INFO", format="text"))
    logger = get_logger("test.component")
    logger.info("hello world")

    captured = capsys.readouterr()
    with pytest.raises(json.JSONDecodeError):
        json.loads(captured.out.strip().splitlines()[-1])
