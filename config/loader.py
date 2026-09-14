"""Loads, schema-validates, and type-validates system configuration.

Configuration flows through two layers deliberately:

1. ``settings.schema.yaml`` (JSON Schema) catches structurally malformed YAML
   early, with an error message that doesn't require reading Python.
2. ``config.models.Settings`` (pydantic) is the typed object the rest of the
   codebase imports, and additionally enforces cross-field invariants that a
   structural schema can't express (e.g. risk threshold ordering).

Secrets and per-deployment values live in the environment (``.env``), never in
``settings.yaml``; use ``load_environment`` to populate ``os.environ`` before
reading anything that needs a secret.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema
import yaml
from dotenv import load_dotenv

from config.models import Settings

CONFIG_DIR = Path(__file__).parent
DEFAULT_SETTINGS_PATH = CONFIG_DIR / "settings.yaml"
DEFAULT_SCHEMA_PATH = CONFIG_DIR / "settings.schema.yaml"
DEFAULT_ENV_PATH = CONFIG_DIR.parent / ".env"


class ConfigError(RuntimeError):
    """Raised when configuration fails to load, parse, or validate."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            loaded = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"expected a mapping at the top level of {path}, got {type(loaded)!r}")
    return loaded


def load_schema(schema_path: Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    """Load the JSON Schema document used to validate settings.yaml."""
    return _read_yaml(schema_path)


def validate_against_schema(data: dict[str, Any], schema: dict[str, Any]) -> None:
    """Raise ``ConfigError`` if ``data`` does not conform to ``schema``."""
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        path = " -> ".join(str(p) for p in exc.absolute_path) or "<root>"
        raise ConfigError(
            f"settings.yaml failed schema validation at {path}: {exc.message}"
        ) from exc


def load_settings(
    settings_path: Path = DEFAULT_SETTINGS_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> Settings:
    """Load, schema-validate, and type-validate the system configuration.

    Raises:
        ConfigError: if the file is missing, malformed, fails schema
            validation, or fails a cross-field invariant in ``Settings``.
    """
    raw = _read_yaml(settings_path)
    schema = load_schema(schema_path)
    validate_against_schema(raw, schema)
    try:
        return Settings.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError, primarily
        raise ConfigError(f"settings.yaml failed type/invariant validation: {exc}") from exc


def load_environment(env_path: Path | None = DEFAULT_ENV_PATH) -> None:
    """Populate ``os.environ`` from a ``.env`` file, if present.

    Silently no-ops if the file is absent (e.g. in CI, where secrets are
    injected directly into the environment) rather than raising, since the
    presence of ``.env`` is optional by design.
    """
    if env_path is not None and env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)
