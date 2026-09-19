"""The formal pre-live checklist (Phase 22), machine-checked wherever a
machine can honestly check it.

Eighteen conditions, from ``docs/PRE_LIVE_CHECKLIST.md``, every one
independently reported PASS/FAIL with a concrete reason -- never a bare
boolean, matching this codebase's own established convention for a
rejection reason (``risk.risk_manager.RiskViolation``,
``execution.order_manager.OrderRecord.reject_reason``). Fail-closed
throughout: a condition this module cannot verify (a missing file, an
unreachable check, a test run it could not execute) is reported FAIL, not
skipped and not assumed -- an unverifiable prerequisite for live trading
is not a satisfied one.

This module submits no order, contacts no broker, and reads no live
credentials. Checks 1-5, 7, 13-15 run this repository's own test suite
(as subprocesses, so a hung or crashing test cannot bring this process
down); checks 6, 8-12, 16-18 inspect configuration, generated reports,
and the git-tracked source tree directly.

**What "PASS" here actually proves, and what it does not.** Several of
the eighteen conditions (compliance sign-off, kill-switch testing,
monitoring, database backups) are things code can verify were *exercised
and passed their own tests*, not that a human has made the underlying
business judgment they represent -- ``ComplianceConfig.broker_authorization_confirmed``
is itself an attested boolean, not something this module can independently
verify against a broker's dashboard. A PASS here is necessary, not
sufficient, for a human decision to actually go live -- which is exactly
why ``broker.factory.build_broker`` still requires ``enable_live_trading=True``
as its own, separate, explicit confirmation even after this passes.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from broker.compliance import ComplianceError, ComplianceGate
from config.loader import ConfigError, load_settings
from core.regime.model_registry import ModelRegistry, NoApprovedModelError

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTEST_TIMEOUT_SECONDS = 900

_SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"-----BEGIN(?: RSA)? PRIVATE KEY-----", "an embedded private key"),
    (r"AKIA[0-9A-Z]{16}", "an AWS access key id"),
    (r"(?i)kite[_-]?api[_-]?secret\s*[:=]\s*['\"][^'\"\s]{8,}['\"]", "a literal Kite API secret"),
    (
        r"(?i)\bapi[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
        "a literal API key assigned as a string constant",
    ),
)
"""Deliberately narrow, high-precision patterns: a placeholder like
``BROKER_API_KEY=`` (empty, .env.example's own convention) or a field
*name* mentioning a secret must never match -- only what looks like an
actual embedded credential."""

_ALLOWED_SECRET_SCAN_EXCLUSIONS = frozenset({".env.example"})
"""``.env.example`` documents the *shape* of a secret with a comment
above an empty value -- not a match in practice, but excluded explicitly
so this stays true even if the example file's own comments get more
specific in the future."""


class PreflightCheck(StrEnum):
    UNIT_TESTS = "unit tests pass"
    INTEGRATION_TESTS = "integration tests pass"
    LOOKAHEAD_TESTS = "look-ahead tests pass"
    WALK_FORWARD_BACKTESTS = "walk-forward backtests complete"
    STRESS_TESTS = "stress tests complete"
    PAPER_TRADING = "paper trading completed successfully"
    BROKER_RECONCILIATION = "broker reconciliation verified"
    API_CONFIGURATION = "API configuration verified"
    COMPLIANCE_CONFIGURATION = "compliance configuration verified"
    STATIC_IP_CONFIGURATION = "static-IP configuration verified where applicable"
    ORDER_TYPE_CONFIGURATION = "order-type configuration verified"
    APPROVED_MODEL = "an approved model is present and loadable"
    RISK_LIMITS = "risk limits configured"
    KILL_SWITCH = "kill switch tested"
    RESTART_RECOVERY = "restart recovery tested"
    MONITORING = "monitoring tested"
    DATABASE_BACKUPS = "database backups tested"
    NO_SECRETS_IN_SOURCE = "secrets are not stored in source code"
    NO_LIVE_CREDENTIALS_IN_DEV = "live credentials are not present in development configuration"


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    check: PreflightCheck
    status: CheckStatus
    detail: str

    @property
    def ok(self) -> bool:
        return self.status is CheckStatus.PASS


@dataclass(frozen=True)
class PreflightReport:
    generated_at: dt.datetime
    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return len(self.results) == len(PreflightCheck) and all(r.ok for r in self.results)

    @property
    def failed_checks(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if not r.ok)


def run_preflight(
    repo_root: Path = _REPO_ROOT,
    *,
    run_test_suites: bool = True,
) -> PreflightReport:
    """Runs every check, in the fixed order of ``docs/PRE_LIVE_CHECKLIST.md``.

    ``run_test_suites=False`` skips the eight checks that invoke this
    repository's own test suite as a subprocess (checks 1-5, 7, 13-15) --
    used only by this module's own test suite, to avoid recursively
    re-running the whole thing on every test collection. The real CLI
    (``python -m app.cli preflight``) always leaves it ``True``: a
    preflight run that skipped the test suite would not be a preflight
    check at all.
    """
    results = [
        _check_unit_tests(repo_root, run_test_suites),
        _check_integration_tests(repo_root, run_test_suites),
        _check_lookahead_tests(repo_root, run_test_suites),
        _check_walk_forward_backtests(repo_root, run_test_suites),
        _check_stress_tests(repo_root, run_test_suites),
        _check_paper_trading(repo_root),
        _check_broker_reconciliation(repo_root, run_test_suites),
        _check_api_configuration(repo_root, run_test_suites),
        _check_compliance_configuration(),
        _check_static_ip_configuration(),
        _check_order_type_configuration(),
        _check_approved_model(repo_root),
        _check_risk_limits(),
        _check_kill_switch(repo_root, run_test_suites),
        _check_restart_recovery(repo_root, run_test_suites),
        _check_monitoring(repo_root, run_test_suites),
        _check_database_backups(repo_root),
        _check_no_secrets_in_source(repo_root),
        _check_no_live_credentials_in_dev(repo_root),
    ]
    return PreflightReport(generated_at=dt.datetime.now(dt.UTC), results=tuple(results))


# --------------------------------------------------------------------------
# 1-5, 7, 13-15: this repository's own test suite, as evidence
# --------------------------------------------------------------------------


def _run_pytest(
    repo_root: Path, paths: Sequence[str], check: PreflightCheck, *, skip: bool
) -> CheckResult:
    if skip:
        return CheckResult(
            check, CheckStatus.FAIL, "test suite run skipped (run_test_suites=False)"
        )
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *paths],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=_PYTEST_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            check, CheckStatus.FAIL, f"pytest {' '.join(paths)} did not finish within "
            f"{_PYTEST_TIMEOUT_SECONDS}s"
        )
    summary = _last_nonblank_line(result.stdout) or _last_nonblank_line(result.stderr)
    status = CheckStatus.PASS if result.returncode == 0 else CheckStatus.FAIL
    return CheckResult(check, status, f"pytest {' '.join(paths)}: {summary}")


def _last_nonblank_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return "(no output)"


def _check_unit_tests(repo_root: Path, run_test_suites: bool) -> CheckResult:
    return _run_pytest(
        repo_root, ["tests/unit"], PreflightCheck.UNIT_TESTS, skip=not run_test_suites
    )


def _check_integration_tests(repo_root: Path, run_test_suites: bool) -> CheckResult:
    return _run_pytest(
        repo_root,
        ["tests/unit/test_orchestrator.py", "tests/unit/test_e2e_validation.py"],
        PreflightCheck.INTEGRATION_TESTS,
        skip=not run_test_suites,
    )


def _check_lookahead_tests(repo_root: Path, run_test_suites: bool) -> CheckResult:
    return _run_pytest(
        repo_root,
        ["tests/unit/test_no_lookahead_walkforward.py"],
        PreflightCheck.LOOKAHEAD_TESTS,
        skip=not run_test_suites,
    )


def _check_walk_forward_backtests(repo_root: Path, run_test_suites: bool) -> CheckResult:
    return _run_pytest(
        repo_root,
        ["tests/unit/test_walk_forward.py"],
        PreflightCheck.WALK_FORWARD_BACKTESTS,
        skip=not run_test_suites,
    )


def _check_stress_tests(repo_root: Path, run_test_suites: bool) -> CheckResult:
    return _run_pytest(
        repo_root,
        ["tests/unit/test_stress_test.py"],
        PreflightCheck.STRESS_TESTS,
        skip=not run_test_suites,
    )


def _check_broker_reconciliation(repo_root: Path, run_test_suites: bool) -> CheckResult:
    return _run_pytest(
        repo_root,
        ["tests/unit/test_reconciliation.py", "tests/unit/test_startup.py"],
        PreflightCheck.BROKER_RECONCILIATION,
        skip=not run_test_suites,
    )


def _check_kill_switch(repo_root: Path, run_test_suites: bool) -> CheckResult:
    return _run_pytest(
        repo_root, ["tests/unit/test_kill_switch.py"], PreflightCheck.KILL_SWITCH,
        skip=not run_test_suites,
    )


def _check_restart_recovery(repo_root: Path, run_test_suites: bool) -> CheckResult:
    return _run_pytest(
        repo_root, ["tests/unit/test_startup.py"], PreflightCheck.RESTART_RECOVERY,
        skip=not run_test_suites,
    )


def _check_monitoring(repo_root: Path, run_test_suites: bool) -> CheckResult:
    return _run_pytest(
        repo_root,
        [
            "tests/unit/test_alerts.py",
            "tests/unit/test_terminal_dashboard.py",
            "tests/unit/test_monitoring_snapshot.py",
        ],
        PreflightCheck.MONITORING,
        skip=not run_test_suites,
    )


def _check_api_configuration(repo_root: Path, run_test_suites: bool) -> CheckResult:
    """The adapter code (Kite request/response mapping, WebSocket framing)
    is validated by its own test suite; the *configured* provider must be
    one that adapter actually implements."""
    try:
        settings = load_settings()
    except ConfigError as exc:
        return CheckResult(
            PreflightCheck.API_CONFIGURATION, CheckStatus.FAIL, f"configuration invalid: {exc}"
        )
    if settings.broker.provider not in {"paper", "zerodha"}:
        return CheckResult(
            PreflightCheck.API_CONFIGURATION,
            CheckStatus.FAIL,
            f"broker.provider={settings.broker.provider!r} has no adapter in this codebase",
        )
    adapter_tests = _run_pytest(
        repo_root,
        [
            "tests/unit/test_kite_broker.py",
            "tests/unit/test_kite_mappings.py",
            "tests/unit/test_kite_ticker.py",
        ],
        PreflightCheck.API_CONFIGURATION,
        skip=not run_test_suites,
    )
    if not adapter_tests.ok:
        return adapter_tests
    return CheckResult(
        PreflightCheck.API_CONFIGURATION,
        CheckStatus.PASS,
        f"broker.provider={settings.broker.provider!r} is a supported adapter; "
        f"{adapter_tests.detail}",
    )


# --------------------------------------------------------------------------
# 6: paper trading (Phase 21's own generated evidence)
# --------------------------------------------------------------------------


def _check_paper_trading(repo_root: Path) -> CheckResult:
    report_path = repo_root / "docs" / "validation_report.md"
    if not report_path.is_file():
        return CheckResult(
            PreflightCheck.PAPER_TRADING,
            CheckStatus.FAIL,
            f"{report_path} does not exist -- run scripts/run_e2e_validation.py first",
        )
    text = report_path.read_text(encoding="utf-8")
    generated_line = next(
        (line for line in text.splitlines() if line.startswith("- Generated:")), None
    )
    if "**Result: PASSED**" not in text:
        return CheckResult(
            PreflightCheck.PAPER_TRADING,
            CheckStatus.FAIL,
            f"{report_path} does not record a PASSED result ({generated_line or 'no date found'})",
        )
    return CheckResult(
        PreflightCheck.PAPER_TRADING,
        CheckStatus.PASS,
        f"{report_path} records PASSED ({generated_line or 'generation date not found'}). "
        "Confirm this is recent enough before relying on it -- this check does not enforce "
        "a staleness window on its own.",
    )


# --------------------------------------------------------------------------
# 9-11: compliance / static IP / order type -- ComplianceGate is authoritative
# --------------------------------------------------------------------------


def _check_compliance_configuration() -> CheckResult:
    try:
        settings = load_settings()
    except ConfigError as exc:
        return CheckResult(
            PreflightCheck.COMPLIANCE_CONFIGURATION,
            CheckStatus.FAIL,
            f"configuration invalid: {exc}",
        )
    try:
        ComplianceGate(settings.compliance)
    except ComplianceError as exc:
        return CheckResult(PreflightCheck.COMPLIANCE_CONFIGURATION, CheckStatus.FAIL, str(exc))
    return CheckResult(
        PreflightCheck.COMPLIANCE_CONFIGURATION,
        CheckStatus.PASS,
        f"ComplianceGate accepted compliance_version={settings.compliance.compliance_version!r}",
    )


def _check_static_ip_configuration() -> CheckResult:
    try:
        settings = load_settings()
    except ConfigError as exc:
        return CheckResult(
            PreflightCheck.STATIC_IP_CONFIGURATION,
            CheckStatus.FAIL,
            f"configuration invalid: {exc}",
        )
    if not settings.broker.static_ip_required:
        return CheckResult(
            PreflightCheck.STATIC_IP_CONFIGURATION,
            CheckStatus.PASS,
            "broker.static_ip_required is False -- not applicable",
        )
    if settings.compliance.static_ip_primary == "0.0.0.0":
        return CheckResult(
            PreflightCheck.STATIC_IP_CONFIGURATION,
            CheckStatus.FAIL,
            "broker.static_ip_required is True but compliance.static_ip_primary is still "
            "the unconfigured placeholder '0.0.0.0'",
        )
    return CheckResult(
        PreflightCheck.STATIC_IP_CONFIGURATION,
        CheckStatus.PASS,
        f"static_ip_primary={settings.compliance.static_ip_primary!r} is configured",
    )


def _check_order_type_configuration() -> CheckResult:
    try:
        settings = load_settings()
    except ConfigError as exc:
        return CheckResult(
            PreflightCheck.ORDER_TYPE_CONFIGURATION,
            CheckStatus.FAIL,
            f"configuration invalid: {exc}",
        )
    problems = []
    allowed = {value.upper() for value in settings.compliance.allowed_order_types}
    if settings.execution.order_type.upper() not in allowed:
        problems.append(
            f"execution.order_type={settings.execution.order_type!r} is not in "
            f"compliance.allowed_order_types={sorted(allowed)}"
        )
    if not settings.execution.no_market_order_fallback:
        problems.append(
            "execution.no_market_order_fallback is False -- NSE algo orders may not fall "
            "back to a market order"
        )
    if "MARKET" in allowed:
        problems.append("compliance.allowed_order_types still includes MARKET")
    if problems:
        return CheckResult(
            PreflightCheck.ORDER_TYPE_CONFIGURATION, CheckStatus.FAIL, "; ".join(problems)
        )
    return CheckResult(
        PreflightCheck.ORDER_TYPE_CONFIGURATION,
        CheckStatus.PASS,
        f"order_type={settings.execution.order_type!r}, allowed_order_types={sorted(allowed)}, "
        "no_market_order_fallback=True",
    )


# --------------------------------------------------------------------------
# 12: an approved model exists and loads
# --------------------------------------------------------------------------


def _check_approved_model(repo_root: Path) -> CheckResult:
    """The regime layer cannot run without a model, and nothing else in
    this checklist looks for one.

    That gap is reachable in a way that matters. ``model_registry/`` is in
    both ``.gitignore`` and ``.dockerignore``, and ``deploy/docker-compose.yml``
    mounts ``state``, ``logs`` and ``data_cache`` but not the registry -- so a
    container built and started from a clean checkout has no artifact at all.
    Every other check would pass, and the failure would surface at the first
    regime call of the first live session.

    Loading the artifact, rather than stat-ing the file, is the point: a
    truncated or hand-edited JSON is exactly the kind of thing that survives
    an existence check and fails at 09:15. ``load_current_approved`` also
    refuses to fall back to the newest fit when nothing is approved, so this
    confirms a *reviewed* model, not merely a present one.
    """
    registry_root = repo_root / "model_registry"
    if not registry_root.is_dir():
        return CheckResult(
            PreflightCheck.APPROVED_MODEL,
            CheckStatus.FAIL,
            f"no {registry_root} directory. In a container this usually means the "
            "registry was neither baked into the image (.dockerignore excludes it) "
            "nor mounted as a volume -- see deploy/docker-compose.yml.",
        )
    try:
        artifact = ModelRegistry(registry_root).load_current_approved()
    except NoApprovedModelError as exc:
        return CheckResult(PreflightCheck.APPROVED_MODEL, CheckStatus.FAIL, str(exc))
    except (OSError, ValueError, KeyError) as exc:
        return CheckResult(
            PreflightCheck.APPROVED_MODEL,
            CheckStatus.FAIL,
            f"approved model is present but did not load: {type(exc).__name__}: {exc}",
        )

    return CheckResult(
        PreflightCheck.APPROVED_MODEL,
        CheckStatus.PASS,
        f"model_id={artifact.model_id!r}, n_states={artifact.model.n_states}, "
        f"trained_at={artifact.created_at.date()}, features={artifact.feature_version!r} "
        "-- this check confirms the approved artifact loads, not that it is still "
        "appropriate for current market conditions; see the retrain schedule.",
    )


# --------------------------------------------------------------------------
# 13: risk limits
# --------------------------------------------------------------------------


def _check_risk_limits() -> CheckResult:
    """Schema validation (``config.models.RiskConfig``) already refuses to
    load a nonsensical value (leverage above 1.0, a negative threshold,
    warning above halt, etc.) -- so a successful ``load_settings()`` is
    itself most of this check. What is added here is a second, independent
    sanity ceiling on the two thresholds with the widest blast radius if
    misconfigured, and a full listing of every configured limit so a human
    reviewing this report can actually see what they are approving, not
    just that a call returned True.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        return CheckResult(
            PreflightCheck.RISK_LIMITS, CheckStatus.FAIL, f"configuration invalid: {exc}"
        )
    risk = settings.risk
    problems = []
    if risk.daily_loss_halt_pct > 0.10:
        problems.append(
            f"daily_loss_halt_pct={risk.daily_loss_halt_pct} exceeds the 10% sanity ceiling"
        )
    if risk.peak_to_trough_drawdown_halt_pct > 0.50:
        problems.append(
            f"peak_to_trough_drawdown_halt_pct={risk.peak_to_trough_drawdown_halt_pct} "
            "exceeds the 50% sanity ceiling"
        )
    if problems:
        return CheckResult(PreflightCheck.RISK_LIMITS, CheckStatus.FAIL, "; ".join(problems))
    return CheckResult(
        PreflightCheck.RISK_LIMITS,
        CheckStatus.PASS,
        f"max_gross_exposure={risk.max_gross_exposure}, max_leverage={risk.max_leverage}, "
        f"max_single_name_pct={risk.max_single_name_pct}, "
        f"daily_loss_halt_pct={risk.daily_loss_halt_pct}, "
        f"rolling_loss_halt_pct={risk.rolling_loss_halt_pct}, "
        f"peak_to_trough_drawdown_halt_pct={risk.peak_to_trough_drawdown_halt_pct}, "
        f"max_concurrent_positions={risk.max_concurrent_positions} "
        "-- review these values before enabling live trading; this check only confirms "
        "they loaded and are within a wide sanity ceiling, not that they are right for you.",
    )


# --------------------------------------------------------------------------
# 16: database backups
# --------------------------------------------------------------------------


def _check_database_backups(repo_root: Path) -> CheckResult:
    """Honest by construction: ``storage/database.py`` remains an
    unimplemented Phase 12 stub (see docs/ARCHITECTURE.md's phase plan),
    and no backup script exists in this repository. There is nothing to
    test yet, so this reports FAIL -- not skipped, not assumed -- until a
    real persistence layer and a tested backup/restore procedure exist.
    """
    database_module = repo_root / "storage" / "database.py"
    if not database_module.is_file():
        return CheckResult(
            PreflightCheck.DATABASE_BACKUPS, CheckStatus.FAIL, f"{database_module} does not exist"
        )
    text = database_module.read_text(encoding="utf-8")
    if "NotImplementedError" in text:
        return CheckResult(
            PreflightCheck.DATABASE_BACKUPS,
            CheckStatus.FAIL,
            "storage/database.py is still an unimplemented stub (Phase 12) -- no real "
            "database exists yet, so no backup/restore procedure can have been tested",
        )
    backup_scripts = list((repo_root / "scripts").glob("*backup*"))
    if not backup_scripts:
        return CheckResult(
            PreflightCheck.DATABASE_BACKUPS,
            CheckStatus.FAIL,
            "no backup script found under scripts/ (looked for scripts/*backup*)",
        )
    return CheckResult(
        PreflightCheck.DATABASE_BACKUPS,
        CheckStatus.PASS,
        f"found backup tooling: {[p.name for p in backup_scripts]}",
    )


# --------------------------------------------------------------------------
# 17-18: secrets
# --------------------------------------------------------------------------


def _git_tracked_files(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True
    )
    return [repo_root / line for line in result.stdout.splitlines() if line.strip()]


def _check_no_secrets_in_source(repo_root: Path) -> CheckResult:
    try:
        tracked = _git_tracked_files(repo_root)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return CheckResult(
            PreflightCheck.NO_SECRETS_IN_SOURCE,
            CheckStatus.FAIL,
            f"could not list git-tracked files: {exc}",
        )

    matches: list[str] = []
    for path in tracked:
        relative = path.relative_to(repo_root).as_posix()
        if relative in _ALLOWED_SECRET_SCAN_EXCLUSIONS or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern, description in _SECRET_PATTERNS:
            if re.search(pattern, text):
                matches.append(f"{relative}: looks like {description}")

    env_tracked = repo_root / ".env"
    if env_tracked in tracked:
        matches.append(".env is tracked by git -- it must only ever exist locally, untracked")

    if matches:
        return CheckResult(
            PreflightCheck.NO_SECRETS_IN_SOURCE, CheckStatus.FAIL, "; ".join(matches)
        )
    return CheckResult(
        PreflightCheck.NO_SECRETS_IN_SOURCE,
        CheckStatus.PASS,
        f"scanned {len(tracked)} git-tracked file(s), no embedded secrets found",
    )


def _check_no_live_credentials_in_dev(repo_root: Path) -> CheckResult:
    problems = []
    for name in ("BROKER_API_KEY", "BROKER_API_SECRET"):
        if os.environ.get(name):
            problems.append(f"{name} is set in this process's environment")

    env_file = repo_root / ".env"
    if env_file.is_file():
        text = env_file.read_text(encoding="utf-8")
        for name in ("BROKER_API_KEY", "BROKER_API_SECRET"):
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(f"{name}=") and stripped[len(name) + 1 :].strip():
                    problems.append(f"{name} has a non-empty value in .env")

    if problems:
        return CheckResult(
            PreflightCheck.NO_LIVE_CREDENTIALS_IN_DEV, CheckStatus.FAIL, "; ".join(problems)
        )
    return CheckResult(
        PreflightCheck.NO_LIVE_CREDENTIALS_IN_DEV,
        CheckStatus.PASS,
        "BROKER_API_KEY/BROKER_API_SECRET are unset in this process and (if present) empty in "
        ".env; config.models.BrokerConfig also has no field that could carry a credential -- "
        "settings.yaml is structurally incapable of holding one",
    )
