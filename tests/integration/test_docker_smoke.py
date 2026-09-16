"""Docker production smoke test (Phase 23).

Builds the real image and runs the real container. Marked ``integration``
and therefore excluded from the default ``pytest`` run:

    pytest -m integration tests/integration/test_docker_smoke.py

``tests/unit/test_deployment.py`` proves the deployment *logic* is right.
This file proves the *packaging* is -- that the image builds, that the
dependency set resolves, that the process starts as a non-root user
inside it, that it writes state to the path the volumes are mounted at,
that the health check answers correctly from inside the container, and
that the live-mode refusal holds when the refusal is the only thing
standing between an environment variable and a trading process.

A packaging failure is invisible to every other test in this repository:
the code can be perfect and the container still fail to start because a
wheel does not exist for the base image's platform, or because a
directory the process needs is owned by root. That is exactly the class
of failure that shows up for the first time on the production host at
9:10am, so it is worth a test that actually builds.

No network trading, no credentials, no compose stack: each test below
runs one short-lived container with ``docker run``, so nothing is left
behind and nothing needs a database. The compose stack as a whole is
brought up by hand following ``docs/DEPLOYMENT.md``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_TAG = "indian-regime-trader:smoke-test"
BUILD_TIMEOUT_SECONDS = 1800
RUN_TIMEOUT_SECONDS = 180


def _capture(
    command: list[str], *, timeout: int = RUN_TIMEOUT_SECONDS, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a docker command and capture its output as text.

    The explicit ``encoding``/``errors`` are not decoration. Docker writes
    its build progress in UTF-8 (the spinner glyphs among others), while
    ``text=True`` on Windows decodes with the ANSI code page -- which
    raises inside subprocess's reader *thread*, where the exception
    surfaces as an unrelated-looking warning rather than as a failure at
    the call site. Decoding permissively here keeps a test about Docker
    from failing because of the host's locale.
    """
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = _capture(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=60)
    return probe.returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_available(), reason="a running Docker daemon is required"
)


@pytest.fixture(scope="module")
def image() -> str:
    """Build the image once for the module.

    Session-scoped would be tempting, but module scope keeps the failure
    attributable: if the build breaks, it breaks here rather than in
    whichever unrelated test happened to request it first.
    """
    build = _capture(
        ["docker", "build", "--tag", IMAGE_TAG, "--file", "Dockerfile", "."],
        timeout=BUILD_TIMEOUT_SECONDS,
        cwd=REPO_ROOT,
    )
    if build.returncode != 0:
        pytest.fail(f"docker build failed:\n{build.stdout[-4000:]}\n{build.stderr[-4000:]}")
    return IMAGE_TAG


def _run(
    image_tag: str, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    command = ["docker", "run", "--rm", "--name", f"irt-smoke-{uuid.uuid4().hex[:8]}"]
    for key, value in (env or {}).items():
        command += ["--env", f"{key}={value}"]
    command.append(image_tag)
    command += args
    return _capture(command)


@requires_docker
def test_the_image_builds(image: str) -> None:
    inspect = _capture(["docker", "image", "inspect", image])
    assert inspect.returncode == 0


@requires_docker
def test_the_process_does_not_run_as_root(image: str) -> None:
    """A process that can rewrite its own source is a process whose
    source you can no longer trust to describe what it did."""
    result = _run(image, "--help")  # any command; we inspect the image's user
    assert result.returncode in (0, 2), result.stderr

    inspect = _capture(["docker", "image", "inspect", "--format", "{{.Config.User}}", image])
    assert inspect.stdout.strip() == "app"


@requires_docker
def test_the_container_refuses_to_start_in_live_mode(image: str) -> None:
    """The assertion this whole phase exists to protect.

    An environment variable reaching a container must never be what turns
    real money on. Exit code 3 is the distinct "tried to go live and was
    refused" signal a deploy pipeline can page on.
    """
    result = _run(image, "serve", env={"ENVIRONMENT": "live"})
    assert result.returncode == 3, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "refuses to start in live mode" in (result.stdout + result.stderr)

    result = _run(image, "serve", env={"EXECUTION_MODE": "live"})
    assert result.returncode == 3


@requires_docker
def test_the_service_starts_and_the_health_check_agrees(image: str, tmp_path: Path) -> None:
    """Start the service in the background, wait for Docker's own health
    check to report healthy, then stop it. This is the end-to-end claim
    the deployment rests on: something is running, and something outside
    it can tell that it is."""
    state_volume = f"irt-smoke-state-{uuid.uuid4().hex[:8]}"
    logs_volume = f"irt-smoke-logs-{uuid.uuid4().hex[:8]}"
    container = f"irt-smoke-{uuid.uuid4().hex[:8]}"

    started = _capture(
        [
            "docker", "run", "--detach",
            "--name", container,
            "--volume", f"{state_volume}:/app/state",
            "--volume", f"{logs_volume}:/app/logs",
            # A shorter interval than the image's own, so this test does
            # not wait a minute to learn something it could learn in five
            # seconds.
            "--health-interval", "3s",
            "--health-start-period", "5s",
            "--health-retries", "3",
            image, "serve",
        ],
    )
    assert started.returncode == 0, started.stderr

    try:
        status = _await_health(container, timeout_seconds=120)
        logs = _capture(["docker", "logs", container])
        combined = logs.stdout + logs.stderr
        assert status == "healthy", f"health={status}\nlogs:\n{combined[-4000:]}"

        # The startup log is structured JSON on stdout, which is what the
        # json-file driver collects and what an operator greps.
        events = [
            json.loads(line)
            for line in combined.splitlines()
            if line.strip().startswith("{")
        ]
        assert any(e.get("event") == "service_started" for e in events)
        # It says out loud that it is not trading, rather than presenting
        # an idle process as a trading one.
        assert any(e.get("event") == "service_not_trading" for e in events)

        # The state file landed on the mounted volume, not in the
        # container's own writable layer -- the difference between state
        # that survives a redeploy and state that does not.
        listing = _capture(["docker", "exec", container, "ls", "/app/state"])
        assert "system_state.json" in listing.stdout

        # And the audit log landed on the logs volume.
        audit = _capture(["docker", "exec", container, "ls", "/app/logs"])
        assert "audit.log" in audit.stdout

        # The health command answers correctly from inside, too.
        health = _capture(["docker", "exec", container, "python", "-m", "app.cli", "health"])
        assert health.returncode == 0, health.stdout + health.stderr
        assert "HEALTHY" in health.stdout

        # SIGTERM reaches Python because the entrypoint `exec`s it. A
        # trading process SIGKILLed after the grace period is one that
        # never ran its shutdown path.
        stopped = _capture(["docker", "stop", "--timeout", "30", container])
        assert stopped.returncode == 0
        final_logs = _capture(["docker", "logs", container])
        assert "service_shutdown" in final_logs.stdout + final_logs.stderr
    finally:
        subprocess.run(["docker", "rm", "--force", container], capture_output=True, check=False)
        for volume in (state_volume, logs_volume):
            subprocess.run(["docker", "volume", "rm", "--force", volume],
                           capture_output=True, check=False)


def _await_health(container: str, *, timeout_seconds: int) -> str:
    import time

    deadline = time.monotonic() + timeout_seconds
    status = "unknown"
    while time.monotonic() < deadline:
        probe = _capture(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", container]
        )
        status = probe.stdout.strip()
        if status in ("healthy", "unhealthy"):
            return status
        time.sleep(2)
    return status
