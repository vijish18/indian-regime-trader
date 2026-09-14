from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def valid_settings_dict() -> dict[str, Any]:
    """A full, schema-valid settings dict loaded from the real
    config/settings.yaml and deep-copied so tests can mutate it freely
    without affecting other tests or the file on disk.
    """
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return copy.deepcopy(data)
