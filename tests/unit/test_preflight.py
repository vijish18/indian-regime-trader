"""Unit tests for ``live/preflight.py`` (Phase 22).

**Never points ``_run_pytest`` at this repository's own ``tests/unit``
directory.** ``PreflightCheck.UNIT_TESTS`` runs ``pytest tests/unit`` as a
subprocess -- since this very file lives under ``tests/unit``, doing that
from inside a test in this file would spawn another full run of this
entire suite (including this test), which would spawn another, forever.
The subprocess mechanics are instead proven against a small, disposable,
throwaway pytest project built fresh in ``tmp_path``. Every other check is
exercised directly, with ``config.loader.load_settings`` swapped for a
controlled ``Settings`` object via monkeypatching the name
``live.preflight`` imported it under -- no real file I/O, no dependence on
this machine's own ``config/settings.yaml`` content.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

import pytest

from config.loader import ConfigError, load_settings
from config.models import Settings
from live.preflight import (
    CheckResult,
    CheckStatus,
    PreflightCheck,
    PreflightReport,
    _check_api_configuration,
    _check_approved_model,
    _check_compliance_configuration,
    _check_database_backups,
    _check_no_live_credentials_in_dev,
    _check_no_secrets_in_source,
    _check_order_type_configuration,
    _check_paper_trading,
    _check_risk_limits,
    _check_static_ip_configuration,
    _run_pytest,
    run_preflight,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def base_settings() -> Settings:
    return load_settings()


def _with_valid_compliance(settings: Settings, **overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "broker_authorization_confirmed": True,
        "static_ip_primary": "203.0.113.10",
        "algo_identifier": "ALGO-TEST-0001",
    }
    defaults.update(overrides)
    updated_compliance = settings.compliance.model_copy(update=defaults)
    return settings.model_copy(update={"compliance": updated_compliance})


# --------------------------------------------------------------------------
# _run_pytest, against an isolated, disposable pytest project
# --------------------------------------------------------------------------


def _write_fake_project(tmp_path: Path, *, passing: bool) -> None:
    body = "def test_it() -> None:\n    assert True\n" if passing else (
        "def test_it() -> None:\n    assert False, 'deliberately failing'\n"
    )
    (tmp_path / "test_fake.py").write_text(body, encoding="utf-8")


def test_run_pytest_reports_pass_for_a_passing_suite(tmp_path: Path) -> None:
    _write_fake_project(tmp_path, passing=True)
    result = _run_pytest(tmp_path, ["test_fake.py"], PreflightCheck.UNIT_TESTS, skip=False)
    assert result.status is CheckStatus.PASS
    assert "passed" in result.detail


def test_run_pytest_reports_fail_for_a_failing_suite(tmp_path: Path) -> None:
    _write_fake_project(tmp_path, passing=False)
    result = _run_pytest(tmp_path, ["test_fake.py"], PreflightCheck.UNIT_TESTS, skip=False)
    assert result.status is CheckStatus.FAIL
    assert "failed" in result.detail


def test_run_pytest_reports_fail_when_skipped() -> None:
    result = _run_pytest(
        _REPO_ROOT, ["tests/unit/test_kill_switch.py"], PreflightCheck.UNIT_TESTS, skip=True
    )
    assert result.status is CheckStatus.FAIL
    assert "skipped" in result.detail


def test_run_pytest_carries_the_requested_check_label(tmp_path: Path) -> None:
    _write_fake_project(tmp_path, passing=True)
    result = _run_pytest(tmp_path, ["test_fake.py"], PreflightCheck.STRESS_TESTS, skip=False)
    assert result.check is PreflightCheck.STRESS_TESTS


# --------------------------------------------------------------------------
# 6: paper trading
# --------------------------------------------------------------------------


def test_paper_trading_fails_when_the_report_is_missing(tmp_path: Path) -> None:
    result = _check_paper_trading(tmp_path)
    assert result.status is CheckStatus.FAIL
    assert "does not exist" in result.detail


def test_paper_trading_fails_when_the_report_did_not_pass(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "validation_report.md").write_text(
        "# Report\n\n**Result: FAILED**\n", encoding="utf-8"
    )
    result = _check_paper_trading(tmp_path)
    assert result.status is CheckStatus.FAIL


def test_paper_trading_passes_when_the_report_says_passed(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "validation_report.md").write_text(
        "# Report\n\n**Result: PASSED**\n\n- Generated: 2024-01-01T00:00:00+00:00\n",
        encoding="utf-8",
    )
    result = _check_paper_trading(tmp_path)
    assert result.status is CheckStatus.PASS
    assert "2024-01-01" in result.detail


# --------------------------------------------------------------------------
# 9: compliance configuration
# --------------------------------------------------------------------------


def test_compliance_check_fails_against_the_default_placeholder(
    base_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("live.preflight.load_settings", lambda: base_settings)
    result = _check_compliance_configuration()
    assert result.status is CheckStatus.FAIL
    assert "broker authorization" in result.detail


def test_compliance_check_passes_once_fully_configured(
    base_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "live.preflight.load_settings", lambda: _with_valid_compliance(base_settings)
    )
    result = _check_compliance_configuration()
    assert result.status is CheckStatus.PASS


def test_compliance_check_fails_closed_on_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> Settings:
        raise ConfigError("bad config")

    monkeypatch.setattr("live.preflight.load_settings", _raise)
    result = _check_compliance_configuration()
    assert result.status is CheckStatus.FAIL
    assert "bad config" in result.detail


# --------------------------------------------------------------------------
# 10: static IP
# --------------------------------------------------------------------------


def test_static_ip_check_fails_against_the_placeholder(
    base_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("live.preflight.load_settings", lambda: base_settings)
    result = _check_static_ip_configuration()
    assert result.status is CheckStatus.FAIL
    assert "0.0.0.0" in result.detail


def test_static_ip_check_passes_once_configured(
    base_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "live.preflight.load_settings", lambda: _with_valid_compliance(base_settings)
    )
    result = _check_static_ip_configuration()
    assert result.status is CheckStatus.PASS


def test_static_ip_check_is_not_applicable_when_not_required(
    base_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    not_required = base_settings.model_copy(
        update={"broker": base_settings.broker.model_copy(update={"static_ip_required": False})}
    )
    monkeypatch.setattr("live.preflight.load_settings", lambda: not_required)
    result = _check_static_ip_configuration()
    assert result.status is CheckStatus.PASS
    assert "not applicable" in result.detail


# --------------------------------------------------------------------------
# 11: order-type configuration
# --------------------------------------------------------------------------


def test_order_type_check_passes_against_default_settings(
    base_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """settings.yaml's own defaults already configure a valid order type
    -- this is deliberately not gated behind compliance sign-off, since
    it is a structural configuration property, not an attestation."""
    monkeypatch.setattr("live.preflight.load_settings", lambda: base_settings)
    result = _check_order_type_configuration()
    assert result.status is CheckStatus.PASS


def test_order_type_check_fails_when_the_configured_type_is_not_allowed(
    base_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    narrowed = base_settings.model_copy(
        update={
            "compliance": base_settings.compliance.model_copy(
                update={"allowed_order_types": ("SL",)}
            )
        }
    )
    monkeypatch.setattr("live.preflight.load_settings", lambda: narrowed)
    result = _check_order_type_configuration()
    assert result.status is CheckStatus.FAIL
    assert "not in" in result.detail


def test_order_type_check_fails_when_market_fallback_is_permitted(
    base_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    permissive = base_settings.model_copy(
        update={
            "execution": base_settings.execution.model_copy(
                update={"no_market_order_fallback": False}
            )
        }
    )
    monkeypatch.setattr("live.preflight.load_settings", lambda: permissive)
    result = _check_order_type_configuration()
    assert result.status is CheckStatus.FAIL
    assert "no_market_order_fallback" in result.detail


# --------------------------------------------------------------------------
# 12: risk limits
# --------------------------------------------------------------------------


def test_risk_limits_check_passes_against_default_settings(
    base_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("live.preflight.load_settings", lambda: base_settings)
    result = _check_risk_limits()
    assert result.status is CheckStatus.PASS
    assert "max_gross_exposure" in result.detail


def test_risk_limits_check_fails_above_the_sanity_ceiling(
    base_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    reckless = base_settings.model_copy(
        update={"risk": base_settings.risk.model_copy(update={"daily_loss_halt_pct": 0.5})}
    )
    monkeypatch.setattr("live.preflight.load_settings", lambda: reckless)
    result = _check_risk_limits()
    assert result.status is CheckStatus.FAIL
    assert "daily_loss_halt_pct" in result.detail


# --------------------------------------------------------------------------
# 16: database backups
# --------------------------------------------------------------------------


def test_database_backups_fails_when_the_module_is_missing(tmp_path: Path) -> None:
    result = _check_database_backups(tmp_path)
    assert result.status is CheckStatus.FAIL
    assert "does not exist" in result.detail


def test_database_backups_fails_while_storage_remains_a_stub(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "database.py").write_text(
        "def backup() -> None:\n    raise NotImplementedError\n", encoding="utf-8"
    )
    result = _check_database_backups(tmp_path)
    assert result.status is CheckStatus.FAIL
    assert "unimplemented stub" in result.detail


def test_database_backups_fails_without_a_backup_script(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "database.py").write_text(
        "def backup() -> None:\n    real_backup()\n", encoding="utf-8"
    )
    (tmp_path / "scripts").mkdir()
    result = _check_database_backups(tmp_path)
    assert result.status is CheckStatus.FAIL
    assert "no backup script" in result.detail


def test_database_backups_passes_with_a_real_implementation_and_a_backup_script(
    tmp_path: Path,
) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "database.py").write_text(
        "def backup() -> None:\n    real_backup()\n", encoding="utf-8"
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_backup.py").write_text("", encoding="utf-8")
    result = _check_database_backups(tmp_path)
    assert result.status is CheckStatus.PASS


# --------------------------------------------------------------------------
# 17: no secrets in source
# --------------------------------------------------------------------------


def _init_git_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)


def test_no_secrets_check_passes_on_a_clean_tree(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    result = _check_no_secrets_in_source(tmp_path)
    assert result.status is CheckStatus.PASS


def test_no_secrets_check_fails_on_an_embedded_private_key(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "leaky.py").write_text(
        "KEY = '''-----BEGIN PRIVATE KEY-----\\nMIIBogIBAAJBAK...\\n-----END PRIVATE KEY-----'''\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    result = _check_no_secrets_in_source(tmp_path)
    assert result.status is CheckStatus.FAIL
    assert "leaky.py" in result.detail


def test_no_secrets_check_fails_on_a_literal_api_key_assignment(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "config.py").write_text(
        'api_key = "sk-live-abcdefghijklmnop"\n', encoding="utf-8"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    result = _check_no_secrets_in_source(tmp_path)
    assert result.status is CheckStatus.FAIL


def test_no_secrets_check_excludes_dot_env_example(tmp_path: Path) -> None:
    """.env.example documents the shape of a secret (a name followed by an
    empty value) and is excluded by name -- it must never itself be
    flagged as containing one."""
    _init_git_repo(tmp_path)
    (tmp_path / ".env.example").write_text(
        "BROKER_API_KEY=\nBROKER_API_SECRET=\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    result = _check_no_secrets_in_source(tmp_path)
    assert result.status is CheckStatus.PASS


def test_no_secrets_check_pattern_ignores_an_empty_placeholder_elsewhere(
    tmp_path: Path,
) -> None:
    """The pattern itself (not just the .env.example exclusion) must not
    flag a name-only placeholder with no value assigned."""
    _init_git_repo(tmp_path)
    (tmp_path / "settings_placeholder.py").write_text(
        'api_key = ""\n', encoding="utf-8"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    result = _check_no_secrets_in_source(tmp_path)
    assert result.status is CheckStatus.PASS


def test_no_secrets_check_flags_a_tracked_env_file(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / ".env").write_text("BROKER_API_KEY=real-value\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", ".env"], cwd=tmp_path, check=True)
    result = _check_no_secrets_in_source(tmp_path)
    assert result.status is CheckStatus.FAIL
    assert ".env is tracked" in result.detail


# --------------------------------------------------------------------------
# 18: no live credentials in dev
# --------------------------------------------------------------------------


def test_no_live_credentials_check_passes_with_a_clean_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BROKER_API_KEY", raising=False)
    monkeypatch.delenv("BROKER_API_SECRET", raising=False)
    result = _check_no_live_credentials_in_dev(tmp_path)
    assert result.status is CheckStatus.PASS


def test_no_live_credentials_check_fails_if_the_env_var_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BROKER_API_KEY", "leaked-value")
    result = _check_no_live_credentials_in_dev(tmp_path)
    assert result.status is CheckStatus.FAIL
    assert "BROKER_API_KEY" in result.detail


def test_no_live_credentials_check_fails_if_dot_env_has_a_real_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BROKER_API_KEY", raising=False)
    monkeypatch.delenv("BROKER_API_SECRET", raising=False)
    (tmp_path / ".env").write_text("BROKER_API_SECRET=this-should-not-be-here\n", encoding="utf-8")
    result = _check_no_live_credentials_in_dev(tmp_path)
    assert result.status is CheckStatus.FAIL
    assert "BROKER_API_SECRET" in result.detail


def test_no_live_credentials_check_passes_with_an_empty_dot_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BROKER_API_KEY", raising=False)
    monkeypatch.delenv("BROKER_API_SECRET", raising=False)
    (tmp_path / ".env").write_text("BROKER_API_KEY=\nBROKER_API_SECRET=\n", encoding="utf-8")
    result = _check_no_live_credentials_in_dev(tmp_path)
    assert result.status is CheckStatus.PASS


# --------------------------------------------------------------------------
# API configuration -- structural part only (the adapter-test subset is
# exercised via run_test_suites=False below, and for real by the CLI)
# --------------------------------------------------------------------------


def test_api_configuration_fails_for_an_unsupported_provider(
    base_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsupported = base_settings.model_copy(
        update={"broker": base_settings.broker.model_copy(update={"provider": "not-a-broker"})}
    )
    monkeypatch.setattr("live.preflight.load_settings", lambda: unsupported)
    result = _check_api_configuration(_REPO_ROOT, run_test_suites=False)
    assert result.status is CheckStatus.FAIL
    assert "no adapter" in result.detail


# --------------------------------------------------------------------------
# run_preflight as a whole
# --------------------------------------------------------------------------


def test_run_preflight_covers_every_defined_check(tmp_path: Path) -> None:
    report = run_preflight(tmp_path, run_test_suites=False)
    assert [r.check for r in report.results] == list(PreflightCheck)


def test_run_preflight_with_test_suites_skipped_never_passes_overall(tmp_path: Path) -> None:
    report = run_preflight(tmp_path, run_test_suites=False)
    assert report.passed is False
    skipped_reasons = [r for r in report.failed_checks if "skipped" in r.detail]
    # Checks 1-5, 7, 13-15 call _run_pytest directly; check 8 (API
    # configuration) also calls it internally once its provider check
    # passes -- ten in total.
    assert len(skipped_reasons) == 10


def test_preflight_report_failed_checks_lists_only_failures() -> None:
    report = PreflightReport(
        generated_at=dt.datetime.now(dt.UTC),
        results=(
            CheckResult(PreflightCheck.UNIT_TESTS, CheckStatus.PASS, "ok"),
            CheckResult(PreflightCheck.STRESS_TESTS, CheckStatus.FAIL, "broke"),
        ),
    )
    assert report.failed_checks == (report.results[1],)
    assert report.passed is False  # also incomplete: only 2 of 18 checks present


def test_preflight_report_passed_requires_every_check_present() -> None:
    report = PreflightReport(
        generated_at=dt.datetime.now(dt.UTC),
        results=(CheckResult(PreflightCheck.UNIT_TESTS, CheckStatus.PASS, "ok"),),
    )
    assert report.passed is False


# --------------------------------------------------------------------------
# An approved model must exist and load
# --------------------------------------------------------------------------


def test_a_missing_registry_directory_fails_the_check(tmp_path: Path) -> None:
    """The container case, and the reason this check exists.

    ``model_registry/`` is in both .gitignore and .dockerignore, and
    docker-compose mounts state, logs and data_cache but not the registry.
    So an image built from a clean checkout has no artifact at all -- and
    before this check, every other one passed and the failure surfaced at
    the first regime call of the first live session.
    """
    result = _check_approved_model(tmp_path)

    assert result.status is CheckStatus.FAIL
    assert "model_registry" in result.detail


def test_an_empty_registry_fails_rather_than_finding_nothing_quietly(
    tmp_path: Path,
) -> None:
    (tmp_path / "model_registry").mkdir()

    result = _check_approved_model(tmp_path)

    assert result.status is CheckStatus.FAIL
    assert "no approved model" in result.detail.lower()


def test_an_approved_id_with_no_artifact_on_disk_fails(tmp_path: Path) -> None:
    """Approval is a pointer. A pointer to a file somebody deleted is the
    failure mode an existence check on the directory would miss."""
    registry = tmp_path / "model_registry"
    registry.mkdir()
    (registry / "approved.json").write_text('{"model_id": "hmm_gone"}', encoding="utf-8")

    result = _check_approved_model(tmp_path)

    assert result.status is CheckStatus.FAIL
    assert "hmm_gone" in result.detail


def test_a_corrupt_artifact_fails_rather_than_passing_on_existence(
    tmp_path: Path,
) -> None:
    """Why this loads the artifact instead of stat-ing the file. A truncated
    or hand-edited JSON survives an existence check and fails at 09:15."""
    registry = tmp_path / "model_registry"
    registry.mkdir()
    (registry / "approved.json").write_text('{"model_id": "hmm_broken"}', encoding="utf-8")
    (registry / "hmm_broken.json").write_text('{"model_id": "hmm_broken"', encoding="utf-8")

    result = _check_approved_model(tmp_path)

    assert result.status is CheckStatus.FAIL


def test_the_shipped_registry_passes_and_names_the_model() -> None:
    """A reviewer approving a live deployment has to be able to see *which*
    model they are approving, not just that a call returned True."""
    result = _check_approved_model(Path(__file__).resolve().parents[2])

    assert result.status is CheckStatus.PASS
    assert "model_id=" in result.detail
    assert "n_states=" in result.detail


def test_the_approved_model_check_is_part_of_the_checklist() -> None:
    """A check that exists but is never assembled into the report protects
    nothing."""
    assert PreflightCheck.APPROVED_MODEL in set(PreflightCheck)
