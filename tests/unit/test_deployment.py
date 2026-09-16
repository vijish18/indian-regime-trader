"""Production smoke tests (Phase 23).

These test the deployment the way an operator experiences it, without
needing Docker: the service starts up or refuses to, the health check
answers the question a supervisor is actually asking, and the artifacts
that carry security decisions say what they are supposed to say.

The Docker-level smoke test -- build the image, bring the stack up, watch
it reach ``(healthy)`` -- lives in ``tests/integration/test_docker_smoke.py``
and is marked ``integration`` so it does not run by default. Both exist
because they prove different things: this file proves the *logic* is
right, that one proves the *packaging* is.

The recurring theme below is that the interesting assertions are about
refusals. A deployment's job in this system is mostly to not do things.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest

from app.health import DEFAULT_MAX_HEARTBEAT_AGE_SECONDS, check_health, render_health
from app.service import (
    DEFAULT_STATE_FILE,
    EXIT_LIVE_MODE_REFUSED,
    EXIT_OK,
    STATE_FILE_ENV_VAR,
    ServiceStartupError,
    _refuse_live_mode,
    run_service,
    state_file_path,
)
from config.loader import load_settings
from execution.system_state import (
    APP_VERSION,
    STATE_SCHEMA_VERSION,
    PersistedState,
    PortfolioPositionSnapshot,
    PortfolioSnapshot,
    SystemState,
    SystemStateStore,
)
from orchestration.fail_closed import FailClosedReason

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "deploy"


# ---------------------------------------------------------------------------
# The service: startup, refusal, heartbeat, shutdown
# ---------------------------------------------------------------------------


@pytest.fixture
def state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "state" / "system_state.json"
    monkeypatch.setenv(STATE_FILE_ENV_VAR, str(path))
    return path


def _run_once(**kwargs: object) -> int:
    return run_service(
        max_iterations=1,
        sleep_fn=lambda _seconds: None,
        install_signal_handlers=False,  # pytest is not the main thread's owner here
        **kwargs,  # type: ignore[arg-type]
    )


def test_the_service_starts_and_writes_a_heartbeat(state_file: Path) -> None:
    assert _run_once() == EXIT_OK
    persisted = SystemStateStore(state_file).load()
    assert persisted is not None
    assert persisted.system_state is SystemState.READY
    assert persisted.app_version == APP_VERSION


def test_the_service_refuses_to_start_in_live_mode() -> None:
    """Deployment is not one of the gates that authorizes live trading.

    A container image is copied between hosts and restarted by a
    supervisor; none of those events is a human deciding to risk real
    money, so none of them may be what causes it.
    """
    settings = load_settings()
    live_settings = settings.model_copy(
        update={"execution": settings.execution.model_copy(update={"mode": "live"})}
    )
    with pytest.raises(ServiceStartupError) as excinfo:
        _refuse_live_mode(live_settings)
    assert excinfo.value.reason is FailClosedReason.CONFIGURATION_FAILURE
    assert "live mode" in excinfo.value.detail
    assert "PRE_LIVE_CHECKLIST" in excinfo.value.detail


def test_live_mode_refusal_has_its_own_exit_code(
    state_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """So a deploy pipeline can tell "this tried to go live and was
    stopped" from an ordinary misconfiguration, and page accordingly."""
    def _live(settings: object) -> None:
        raise ServiceStartupError(
            FailClosedReason.CONFIGURATION_FAILURE,
            "execution.mode is 'live', and this service refuses to start in live mode.",
        )

    monkeypatch.setattr("app.service._refuse_live_mode", _live)
    assert _run_once() == EXIT_LIVE_MODE_REFUSED


def test_an_unwritable_state_directory_stops_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Database failure -> do not trade. Here the "database" is the
    persisted state store, which is what a restart reconciles from."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("this is a file, so it cannot also be a directory", encoding="utf-8")
    monkeypatch.setenv(STATE_FILE_ENV_VAR, str(blocker / "system_state.json"))
    assert _run_once() != EXIT_OK


def test_the_heartbeat_never_flattens_what_a_real_run_recorded(state_file: Path) -> None:
    """The portfolio snapshot and the last broker event id in this file
    are what Phase 18's reconciliation reads after a restart. A heartbeat
    that overwrote them would destroy the record of what this system had
    already done -- silently, and in the exact file whose whole purpose is
    to survive."""
    store = SystemStateStore(state_file)
    store.save(
        PersistedState(
            schema_version=STATE_SCHEMA_VERSION,
            app_version=APP_VERSION,
            system_state=SystemState.READY,
            model_version="hmm-2026-01",
            strategy_version="strategy-v1",
            portfolio_snapshot=PortfolioSnapshot(
                as_of=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
                cash=100_000.0,
                positions=(PortfolioPositionSnapshot("NSE:INFY", 10, 1500.0),),
            ),
            last_market_data_timestamp=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
            last_broker_event_id="evt-42",
            last_broker_event_timestamp=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
            updated_at=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
        )
    )

    assert _run_once() == EXIT_OK

    after = SystemStateStore(state_file).load()
    assert after is not None
    assert after.last_broker_event_id == "evt-42"
    assert after.model_version == "hmm-2026-01"
    assert after.strategy_version == "strategy-v1"
    assert after.portfolio_snapshot is not None
    assert after.portfolio_snapshot.positions[0].instrument_id == "NSE:INFY"
    assert after.updated_at > dt.datetime(2026, 1, 2, tzinfo=dt.UTC)


def test_a_shutdown_signal_interrupts_the_heartbeat_wait() -> None:
    """Regression test for a bug the Docker smoke test found.

    Since PEP 475, ``time.sleep`` resumes after a signal handler returns
    rather than raising, so a SIGTERM one second into a 60-second
    heartbeat interval used to leave the process sleeping for the
    remaining 59 -- past Docker's 30-second grace period, so the
    container was SIGKILLed every time and the shutdown path never ran.

    No unit test could have caught the original bug (they all pass their
    own ``sleep_fn`` and never block). This one guards the *fix*: the
    wait must return early, having slept a small fraction of the
    interval.
    """
    from app.service import _ShutdownRequest, _wait_for_shutdown

    shutdown = _ShutdownRequest()
    slept: list[float] = []

    def _sleep(seconds: float) -> None:
        slept.append(seconds)
        if len(slept) == 2:  # the signal arrives two slices in
            shutdown.requested = True

    _wait_for_shutdown(shutdown, 60.0, _sleep)

    assert sum(slept) < 5.0, f"waited {sum(slept)}s of a 60s interval after a shutdown request"
    assert all(s <= 1.0 for s in slept)


def test_the_state_file_path_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(STATE_FILE_ENV_VAR, raising=False)
    assert state_file_path() == DEFAULT_STATE_FILE
    monkeypatch.setenv(STATE_FILE_ENV_VAR, "/app/state/system_state.json")
    assert state_file_path() == Path("/app/state/system_state.json")


# ---------------------------------------------------------------------------
# The health check: what must and must not be reported unhealthy
# ---------------------------------------------------------------------------


def _persist(path: Path, state: SystemState, updated_at: dt.datetime) -> None:
    SystemStateStore(path).save(
        PersistedState(
            schema_version=STATE_SCHEMA_VERSION,
            app_version=APP_VERSION,
            system_state=state,
            model_version=None,
            strategy_version="strategy-v1",
            portfolio_snapshot=None,
            last_market_data_timestamp=None,
            last_broker_event_id=None,
            last_broker_event_timestamp=None,
            updated_at=updated_at,
        )
    )


def test_a_halted_system_is_reported_healthy(tmp_path: Path) -> None:
    """The single most important assertion in this file.

    HALTED means the system met a fail-closed condition and correctly
    refused to trade. Reporting that unhealthy would have the supervisor
    restart the process that knows why, re-run startup into the same
    condition, and halt again -- a crash loop whose only symptom is a
    restart counter. That is precisely the outcome
    ``orchestration.fail_closed`` exists to prevent, and a health check
    is a very easy place to reintroduce it by accident.
    """
    now = dt.datetime(2026, 3, 2, 10, 0, tzinfo=dt.UTC)
    path = tmp_path / "state.json"
    _persist(path, SystemState.HALTED, now - dt.timedelta(seconds=10))

    result = check_health(state_path=path, clock=lambda: now)

    assert result.healthy is True
    assert result.system_state is SystemState.HALTED
    assert "HALTED" in result.detail
    assert "operator" in result.detail  # it says what the remedy actually is
    assert "UNHEALTHY" not in render_health(result)


def test_a_stale_heartbeat_is_unhealthy(tmp_path: Path) -> None:
    """A wedged process is the case where replacing it is the correct and
    only remedy."""
    now = dt.datetime(2026, 3, 2, 10, 0, tzinfo=dt.UTC)
    path = tmp_path / "state.json"
    _persist(
        path,
        SystemState.READY,
        now - dt.timedelta(seconds=DEFAULT_MAX_HEARTBEAT_AGE_SECONDS + 60),
    )

    result = check_health(state_path=path, clock=lambda: now)

    assert result.healthy is False
    assert "not making progress" in result.detail


def test_a_missing_state_file_is_unhealthy(tmp_path: Path) -> None:
    result = check_health(state_path=tmp_path / "nothing.json")
    assert result.healthy is False
    assert "not mounted" in result.detail  # names the likely cause, not just the symptom


def test_a_corrupt_state_file_is_unhealthy(tmp_path: Path) -> None:
    """Never "assume empty and carry on": an unreadable store is a
    database failure, and a database failure means do not trade."""
    path = tmp_path / "state.json"
    path.write_text("{not valid json", encoding="utf-8")
    result = check_health(state_path=path)
    assert result.healthy is False
    assert "database failure" in result.detail


def test_the_health_bound_is_looser_than_the_heartbeat_interval() -> None:
    """Otherwise every deployment would flap: the supervisor would call a
    process dead in the ordinary gap between two of its own heartbeats."""
    interval = load_settings().monitoring.heartbeat_interval_seconds
    assert DEFAULT_MAX_HEARTBEAT_AGE_SECONDS > interval * 2


def test_health_survives_a_naive_timestamp(tmp_path: Path) -> None:
    """Defence in depth against a state file written by an older build:
    comparing a naive datetime to an aware one raises, and a health check
    that raises is a container that gets restarted for no reason."""
    now = dt.datetime(2026, 3, 2, 10, 0, tzinfo=dt.UTC)
    path = tmp_path / "state.json"
    _persist(path, SystemState.READY, now - dt.timedelta(seconds=5))
    contents = path.read_text(encoding="utf-8").replace("+00:00", "")
    path.write_text(contents, encoding="utf-8")

    result = check_health(state_path=path, clock=lambda: now)
    assert result.healthy is True


# ---------------------------------------------------------------------------
# Logging: persistent, rotating, auditable
# ---------------------------------------------------------------------------


def test_the_audit_log_is_written_and_configured_to_rotate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import logging

    from config.models import LoggingConfig
    from monitoring.logger import configure_logging, get_logger

    audit_path = tmp_path / "logs" / "audit.log"
    configure_logging(
        LoggingConfig(
            level="INFO",
            format="json",
            audit_log_path=str(audit_path),
            audit_log_max_bytes=2048,
            audit_log_backup_count=3,
        )
    )
    try:
        get_logger("test.audit").info("an order was submitted")
        assert audit_path.is_file()
        assert "an order was submitted" in audit_path.read_text(encoding="utf-8")

        handlers = [
            h
            for h in logging.getLogger().handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(handlers) == 1
        assert handlers[0].maxBytes == 2048
        assert handlers[0].backupCount == 3
    finally:
        configure_logging(LoggingConfig(level="INFO", format="json"))


def test_an_unwritable_audit_log_refuses_rather_than_logging_to_nowhere(
    tmp_path: Path,
) -> None:
    """A system that runs without the audit trail its operator believes
    it has is worse than one that refuses to start: the absence is only
    discovered during the incident where the trail was needed."""
    from config.models import LoggingConfig
    from monitoring.logger import AuditLogError, configure_logging

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")

    try:
        with pytest.raises(AuditLogError):
            configure_logging(
                LoggingConfig(
                    level="INFO", format="json", audit_log_path=str(blocker / "audit.log")
                )
            )
    finally:
        configure_logging(LoggingConfig(level="INFO", format="json"))


def test_the_shipped_settings_write_the_audit_log_into_the_logs_volume() -> None:
    """``logs/audit.log`` is relative, and the container's WORKDIR is
    /app with the logs volume mounted at /app/logs. If this path ever
    becomes absolute or moves outside ``logs/``, the audit trail silently
    starts living inside the container's writable layer and dies with it.
    """
    configured = load_settings().logging.audit_log_path
    assert configured is not None
    assert not Path(configured).is_absolute()
    assert configured.startswith("logs/")


# ---------------------------------------------------------------------------
# Deployment artifacts: the security decisions they encode
# ---------------------------------------------------------------------------


def _read(relative: str) -> str:
    path = REPO_ROOT / relative
    assert path.is_file(), f"{relative} is missing"
    return path.read_text(encoding="utf-8")


def _compose_without_comments() -> str:
    """The compose file with comment lines removed.

    Several of the comments in it quote the very anti-patterns they warn
    against (``${POSTGRES_PASSWORD}``, a published ``ports:`` entry), so a
    test that greps the raw text finds the warning and calls it the
    offence.
    """
    return "\n".join(
        line
        for line in _read("deploy/docker-compose.yml").splitlines()
        if not line.lstrip().startswith("#")
    )


def _compose_services() -> tuple[str, str]:
    """The ``app`` and ``db`` service blocks, split at the top-level key.

    Splitting on the bare string ``"  db:"`` would cut at ``depends_on``'s
    own nested ``db:`` instead, silently comparing the wrong halves.
    """
    compose = _compose_without_comments()
    app_start = compose.index("\n  app:")
    db_start = compose.index("\n  db:")
    assert app_start < db_start
    return compose[app_start:db_start], compose[db_start:]


def test_every_deployment_artifact_exists() -> None:
    for relative in (
        "Dockerfile",
        ".dockerignore",
        "deploy/docker-compose.yml",
        "deploy/entrypoint.sh",
        "deploy/backup.sh",
        "deploy/app.env.example",
        "deploy/db.env.example",
        "deploy/postgres-init/01-least-privilege.sql",
        "deploy/host/ufw.sh",
        "deploy/host/sshd_config.hardened",
        "deploy/host/Caddyfile",
        "docs/DEPLOYMENT.md",
        "docs/OPERATIONS.md",
        "docs/INCIDENT_RESPONSE.md",
    ):
        assert (REPO_ROOT / relative).is_file(), f"{relative} is missing"


def test_the_images_app_version_matches_pyproject() -> None:
    """Two copies of the version, so a test has to hold them together.

    The image cannot read the version from package metadata (it does not
    install the project), so the Dockerfile states it. A drifted value is
    not a cosmetic problem: ``app_version`` is written into every state
    file and audit record, and is how an incident review answers "which
    build wrote this?".
    """
    dockerfile = _read("Dockerfile")
    pyproject = _read("pyproject.toml")

    in_image = re.search(r"^ARG APP_VERSION=(.+)$", dockerfile, re.MULTILINE)
    in_project = re.search(r'^version = "(.+)"$', pyproject, re.MULTILINE)
    assert in_image is not None and in_project is not None
    assert in_image.group(1).strip() == in_project.group(1).strip()


def test_an_explicit_app_version_wins_over_absent_package_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from execution.system_state import APP_VERSION_ENV_VAR, _application_version

    monkeypatch.setenv(APP_VERSION_ENV_VAR, "9.9.9-test")
    assert _application_version() == "9.9.9-test"
    monkeypatch.setenv(APP_VERSION_ENV_VAR, "   ")
    assert _application_version() != "   "  # blank is not a version


def test_the_container_does_not_run_as_root() -> None:
    dockerfile = _read("Dockerfile")
    assert re.search(r"^USER app$", dockerfile, re.MULTILINE)
    # ...and the entrypoint refuses if something overrides that.
    assert 'id -u' in _read("deploy/entrypoint.sh")


def test_the_entrypoint_refuses_live_mode_before_python_starts() -> None:
    entrypoint = _read("deploy/entrypoint.sh")
    assert 'ENVIRONMENT:-paper' in entrypoint
    assert 'EXECUTION_MODE:-paper' in entrypoint
    assert "exit 3" in entrypoint


def test_the_compose_stack_publishes_no_ports() -> None:
    """Restricted network ports, enforced structurally.

    Docker's own iptables rules are traversed before ufw's, so a
    published port is reachable from the internet even behind
    `ufw default deny incoming`. The only reliable defence is to publish
    nothing -- which means this assertion, not the firewall script, is
    what actually keeps the database off the internet.
    """
    compose = _read("deploy/docker-compose.yml")
    published = [
        line
        for line in compose.splitlines()
        if re.match(r"^\s{4}ports:", line) or re.search(r"^\s*-\s*\"?\d+:\d+", line)
    ]
    assert published == [], f"deployment publishes ports: {published}"


def test_the_compose_stack_restarts_automatically_but_not_unconditionally() -> None:
    """`unless-stopped`, never `always`: an operator who deliberately
    stops a trading process during an incident must not have it come back
    on the next daemon restart."""
    compose = _read("deploy/docker-compose.yml")
    assert "restart: unless-stopped" in compose
    assert "restart: always" not in compose


def test_the_compose_stack_sets_resource_limits_and_log_rotation() -> None:
    compose = _read("deploy/docker-compose.yml")
    assert compose.count("limits:") >= 2  # both services
    assert "max-size:" in compose and "max-file:" in compose


def test_the_compose_stack_persists_state_logs_and_the_database() -> None:
    compose = _read("deploy/docker-compose.yml")
    for volume in (
        "app-state:/app/state",
        "app-logs:/app/logs",
        "db-data:/var/lib/postgresql/data",
    ):
        assert volume in compose, f"{volume} is not mounted"


def test_the_app_container_has_a_health_check() -> None:
    assert "HEALTHCHECK" in _read("Dockerfile")
    assert "healthcheck:" in _read("deploy/docker-compose.yml")


def test_no_secrets_are_committed_in_the_deployment_artifacts() -> None:
    """The example env files must stay examples. A filled-in value here
    would be committed, and an image layer or a git object holding a
    credential cannot be un-published by deleting it later."""
    for relative, secret_keys in (
        (
            "deploy/app.env.example",
            ("BROKER_API_KEY", "BROKER_API_SECRET", "MARKET_DATA_API_KEY"),
        ),
        ("deploy/db.env.example", ("POSTGRES_PASSWORD", "APP_DB_PASSWORD")),
    ):
        example = _read(relative)
        for secret_key in secret_keys:
            match = re.search(rf"^{secret_key}=(.*)$", example, re.MULTILINE)
            assert match is not None, f"{secret_key} is missing from {relative}"
            assert match.group(1).strip() == "", f"{secret_key} has a value in {relative}"


def test_the_application_container_never_sees_the_database_superuser_password() -> None:
    """Least privilege applied to the deployment itself, not just to SQL.

    Two env files rather than one: a compromised application container
    must not be able to read the credential that owns the schema, which
    would make the careful GRANTs in the init script irrelevant.
    """
    # The variables the file *sets*, not every mention of one: the
    # comments deliberately cross-reference the other file's names.
    app_env_keys = {
        line.split("=", 1)[0].strip()
        for line in _read("deploy/app.env.example").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    assert "POSTGRES_PASSWORD" not in app_env_keys
    assert "APP_DB_PASSWORD" not in app_env_keys

    app_section, db_section = _compose_services()
    assert "app.env" in app_section and "db.env" not in app_section
    assert "db.env" in db_section


def test_compose_does_not_try_to_interpolate_secrets_it_cannot_see() -> None:
    """A bug this deployment actually had.

    Compose interpolates ``${VAR}`` from the shell and from a ``.env`` in
    the project directory -- *not* from ``env_file:``. A compose file that
    writes ``POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}`` while the value
    lives in an env_file either yields an empty password silently or
    fails at ``docker compose config``. Secrets now reach the container
    through ``env_file`` alone and are never named in an interpolation.
    """
    interpolated = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)", _compose_without_comments()))
    secret_names = {
        "POSTGRES_PASSWORD",
        "POSTGRES_SUPERUSER_PASSWORD",
        "APP_DB_PASSWORD",
        "POSTGRES_APP_PASSWORD",
        "BROKER_API_KEY",
        "BROKER_API_SECRET",
        "MARKET_DATA_API_KEY",
        "DATABASE_URL",
    }
    assert not (interpolated & secret_names), f"interpolates secrets: {interpolated & secret_names}"


def test_the_filled_environment_files_cannot_be_committed_or_baked_in() -> None:
    gitignore = _read(".gitignore")
    dockerignore = _read(".dockerignore")
    for pattern in ("deploy/*.env",):
        assert pattern in gitignore, f"{pattern} is not gitignored"
        assert pattern in dockerignore, f"{pattern} is not dockerignored"
    # And the templates themselves stay available to both.
    assert "!deploy/*.env.example" in gitignore
    assert "!deploy/*.env.example" in dockerignore


def test_the_dockerignore_keeps_state_and_secrets_out_of_image_layers() -> None:
    """An image layer is not erasable. A secret COPYed in one layer and
    deleted in the next is still in the image and still readable by
    anyone who can pull it."""
    dockerignore = _read(".dockerignore")
    for pattern in (".env", "state/", "logs/", "*.pem", "*.key", ".git/"):
        assert pattern in dockerignore, f"{pattern} is not excluded from the build context"


def test_the_database_role_the_application_uses_cannot_change_the_schema() -> None:
    """Least privilege, stated as the thing it prevents: a SQL injection
    or a mistaken migration must not be able to drop the order history
    this system reconciles against after a restart."""
    sql = _read("deploy/postgres-init/01-least-privilege.sql")
    assert "NOSUPERUSER" in sql
    assert "NOCREATEDB" in sql
    assert "NOCREATEROLE" in sql
    assert "REVOKE CREATE ON SCHEMA public FROM PUBLIC" in sql
    assert "GRANT SELECT, INSERT, UPDATE, DELETE" in sql
    # TRUNCATE is the one row-shaped privilege deliberately withheld: it
    # is the only one that can erase a whole table's history in a single
    # statement. Checked against the GRANT statements alone, since the
    # surrounding comments name it precisely to explain its absence.
    grants = [line for line in sql.splitlines() if "GRANT" in line and "--" not in line]
    assert grants, "no GRANT statements found"
    assert not any("TRUNCATE" in line for line in grants)


def test_ssh_is_key_only() -> None:
    sshd = _read("deploy/host/sshd_config.hardened")
    assert "PasswordAuthentication no" in sshd
    assert "PubkeyAuthentication yes" in sshd
    assert "PermitRootLogin no" in sshd
    assert "AuthenticationMethods publickey" in sshd


def test_the_firewall_script_denies_by_default_and_never_opens_the_database() -> None:
    ufw = _read("deploy/host/ufw.sh")
    assert "ufw default deny incoming" in ufw
    assert "allow 5432" not in ufw
    # The DOCKER-USER caveat must stay: without it the script is
    # reassuring and wrong.
    assert "DOCKER-USER" in ufw
