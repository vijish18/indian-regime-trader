#!/bin/sh
# Container entrypoint (Phase 23).
#
# Deliberately small. Anything this script decides is a decision made in
# shell, outside the type checker and outside the test suite, so it does
# as little as possible: it refuses the one thing that must be refused
# before Python even starts, then hands over to Python with `exec`.
#
# Usage:  entrypoint.sh serve | health | preflight | <any app.cli command>

set -eu

# --- Refuse live mode ------------------------------------------------------
#
# app/service.py refuses live mode too, and that is the refusal that
# matters -- it is the one with tests. This is a second, cruder check in
# front of it, for one specific reason: a container is the thing most
# likely to be handed a stray `EXECUTION_MODE=live` by a copy-pasted
# deploy command or a promoted environment file, and the cheapest place
# to stop that is before any trading code is imported at all.
#
# Going live is authorized by the gates in docs/PRE_LIVE_CHECKLIST.md,
# never by an environment variable and never by a deployment.
#
# Note what these two variables are and are not. The application's actual
# execution mode comes from `execution.mode` in config/settings.yaml, not
# from here; these carry a human's *deploy-time intent*, and disagreeing
# with the file is itself worth stopping for.
if [ "${ENVIRONMENT:-paper}" = "live" ] || [ "${EXECUTION_MODE:-paper}" = "live" ]; then
    echo "FAIL CLOSED [configuration failure]: this deployment refuses to start in live mode." >&2
    echo "ENVIRONMENT/EXECUTION_MODE requested 'live'. Deployment is not one of the gates" >&2
    echo "that authorizes live trading. See docs/PRE_LIVE_CHECKLIST.md." >&2
    exit 3
fi

# --- Refuse to run as root -------------------------------------------------
#
# The image sets USER app, so reaching here as root means someone
# overrode it (`docker run --user 0`, or a compose file with `user: root`).
# A process that can rewrite its own source is a process whose source you
# can no longer trust to describe what it did.
if [ "$(id -u)" = "0" ]; then
    echo "FAIL CLOSED [configuration failure]: refusing to run as root." >&2
    echo "The image declares USER app (uid 10001); something overrode it." >&2
    exit 1
fi

# --- Hand over -------------------------------------------------------------
#
# `exec` so Python replaces this shell as PID 1 and receives SIGTERM
# directly. Without it, `docker stop` signals the shell, the shell does
# not forward it, and the trading process is SIGKILLed on timeout without
# ever running its shutdown path.
exec python -m app.cli "$@"
