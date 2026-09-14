"""Load, validate, and print the current configuration.

Usage:
    python scripts/validate_config.py [path/to/settings.yaml]

Exits non-zero with a readable error if settings.yaml fails schema or
cross-field validation -- useful as a pre-deploy / CI check independent of
running the full test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

from config.loader import ConfigError, load_settings


def main(argv: list[str]) -> int:
    settings_path = Path(argv[1]) if len(argv) > 1 else None
    try:
        settings = load_settings(settings_path) if settings_path else load_settings()
    except ConfigError as exc:
        print(f"CONFIG INVALID: {exc}", file=sys.stderr)
        return 1

    print("Configuration is valid.")
    print(settings.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
