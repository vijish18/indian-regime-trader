# Production image for the Indian Market Regime Trading System (Phase 23).
#
# Build:  docker build -t indian-regime-trader:local .
# Run:    see deploy/docker-compose.yml -- this image is not meant to be
#         run bare, because the volumes it expects for state and logs are
#         defined there.
#
# Two stages. The builder has a compiler toolchain (numpy/pyarrow wheels
# occasionally need one on a platform without a prebuilt wheel); the
# runtime does not, because a trading host should not carry a compiler it
# only needed for ten minutes at build time.

# ---------------------------------------------------------------------------
# Stage 1: build the virtualenv
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# A virtualenv rather than the system site-packages, so stage 2 can take
# the whole dependency set as one self-contained directory.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY app ./app
COPY backtest ./backtest
COPY broker ./broker
COPY config ./config
COPY core ./core
COPY data ./data
COPY execution ./execution
COPY live ./live
COPY monitoring ./monitoring
COPY orchestration ./orchestration
COPY portfolio ./portfolio
COPY risk ./risk
COPY storage ./storage
COPY universe ./universe
COPY validation ./validation
COPY main.py ./

# Install the project's *dependencies* into the venv. The project's own
# packages are not installed: the runtime stage puts the source at /app
# and runs from there, so the code that runs is the code you can read
# with `docker exec ... cat`, with no second installed copy to diverge
# from it -- and, importantly, no installed copy of `config` whose
# settings.yaml could differ from the one at /app/config/settings.yaml.
# Installing then uninstalling the project is how the dependency set gets
# resolved from pyproject.toml alone, with no duplicated requirements
# file to drift out of sync with it.
RUN pip install . && pip uninstall --yes indian-regime-trader

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

# UID/GID are fixed and match deploy/docker-compose.yml's volume
# ownership notes, so a bind-mounted host directory can be chowned to a
# known id ahead of time instead of being discovered after a failure.
ARG APP_UID=10001
ARG APP_GID=10001

# Must match `version` in pyproject.toml; a unit test asserts it does.
# The image does not install this project (see stage 1), so there is no
# package metadata to read the version from -- and without this, every
# state file and audit record the container writes says "unknown", which
# is precisely the field an incident review needs to answer "which build
# wrote this?".
ARG APP_VERSION=0.1.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_VERSION=${APP_VERSION} \
    STATE_FILE=/app/state/system_state.json

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Source is owned by root and readable-but-not-writable by the app user:
# the process has no business rewriting its own code, and an attacker who
# reaches it should not be able to either.
COPY --chown=root:root app ./app
COPY --chown=root:root backtest ./backtest
COPY --chown=root:root broker ./broker
COPY --chown=root:root config ./config
COPY --chown=root:root core ./core
COPY --chown=root:root data ./data
COPY --chown=root:root execution ./execution
COPY --chown=root:root live ./live
COPY --chown=root:root monitoring ./monitoring
COPY --chown=root:root orchestration ./orchestration
COPY --chown=root:root portfolio ./portfolio
COPY --chown=root:root risk ./risk
COPY --chown=root:root storage ./storage
COPY --chown=root:root universe ./universe
COPY --chown=root:root validation ./validation
COPY --chown=root:root main.py pyproject.toml README.md ./
COPY --chown=root:root deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

# The three writable paths, created here so they exist with the right
# owner even when no volume is mounted over them (a bare `docker run` for
# a smoke test). A volume mounted over a directory that already exists
# with this ownership inherits it.
RUN mkdir -p /app/state /app/logs /app/data_cache \
    && chown -R "${APP_UID}:${APP_GID}" /app/state /app/logs /app/data_cache

USER app

# The entrypoint ends in `exec python ...`, so Python -- not the shell --
# is PID 1 and SIGTERM from `docker stop` reaches the signal handlers in
# app/service.py instead of being swallowed by a shell that does not
# forward it. A trading process killed by SIGKILL after the 10-second
# grace period is a trading process that never ran its shutdown path.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]

# start-period covers first-boot: the service writes no heartbeat until
# configuration has loaded and the state store has been verified, and a
# container marked unhealthy before it has had the chance is a container
# restarted for no reason.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-m", "app.cli", "health"]

LABEL org.opencontainers.image.title="indian-regime-trader" \
      org.opencontainers.image.description="India-focused long-only systematic equity trading system" \
      org.opencontainers.image.licenses="Proprietary"
