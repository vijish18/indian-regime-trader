"""Entry point: loads configuration and logging only.

The trading day itself is ``orchestration.orchestrator.Orchestrator``
(Phase 11d), which needs a composition root -- a market-data provider, a
calendar with a populated holiday file, an approved model artifact, and a
constructed broker -- none of which exist until this deployment is
actually provisioned with data and credentials. That wiring is
deliberately not faked here: see docs/ARCHITECTURE.md's "Application
lifecycle and daily workflow" section for what an `Orchestrator` needs,
and ``tests/unit/test_orchestrator.py``'s ``Harness`` for a complete,
working example of assembling one.
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
        "Configuration loaded. This entry point does not start a trading day: "
        "construct an orchestration.orchestrator.Orchestrator with this "
        "deployment's own data provider, calendar, model registry and broker."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
