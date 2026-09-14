"""Entry point.

Phase 1 only wires configuration and logging. Regime detection, selection,
portfolio construction, risk, and execution are not implemented yet -- see
docs/ARCHITECTURE.md for the phase plan.
"""

from __future__ import annotations

from config.loader import load_environment, load_settings
from monitoring.logger import configure_logging, get_logger


def main() -> int:
    load_environment()
    settings = load_settings()
    configure_logging(settings.logging)

    logger = get_logger("main")
    logger.info(
        "Configuration loaded. Trading logic is not implemented yet (Phase 1 scaffold)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
