"""Unit tests for ``app/cli.py`` (Phase 22): ``python -m app.cli preflight``
prints a PASS/FAIL report with reasons, writes the Markdown report, and
returns the right exit code -- proven against a stubbed
``live.preflight.run_preflight`` so this file never runs the real
(multi-minute) test-suite-backed checklist itself.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from app.cli import main
from live.preflight import CheckResult, CheckStatus, PreflightCheck, PreflightReport


def _all_pass_report() -> PreflightReport:
    return PreflightReport(
        generated_at=dt.datetime(2024, 6, 3, 10, 0, tzinfo=dt.UTC),
        results=tuple(
            CheckResult(check, CheckStatus.PASS, f"{check.value}: ok") for check in PreflightCheck
        ),
    )


def _one_failure_report() -> PreflightReport:
    results = [
        CheckResult(check, CheckStatus.PASS, f"{check.value}: ok") for check in PreflightCheck
    ]
    results[0] = CheckResult(PreflightCheck.UNIT_TESTS, CheckStatus.FAIL, "3 failed, 40 passed")
    return PreflightReport(
        generated_at=dt.datetime(2024, 6, 3, 10, 0, tzinfo=dt.UTC), results=tuple(results)
    )


def test_preflight_command_exits_zero_when_everything_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("app.cli.run_preflight", lambda **kwargs: _all_pass_report())
    report_path = tmp_path / "preflight_report.md"
    exit_code = main(["app.cli", "preflight", "--report", str(report_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "OVERALL: PASS" in captured.out
    assert report_path.is_file()
    assert "**Result: PASS**" in report_path.read_text(encoding="utf-8")


def test_preflight_command_exits_nonzero_on_any_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("app.cli.run_preflight", lambda **kwargs: _one_failure_report())
    report_path = tmp_path / "preflight_report.md"
    exit_code = main(["app.cli", "preflight", "--report", str(report_path)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "OVERALL: FAIL" in captured.out
    assert "unit tests pass" in captured.out


def test_preflight_command_prints_every_check_with_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("app.cli.run_preflight", lambda **kwargs: _one_failure_report())
    main(["app.cli", "preflight", "--report", str(tmp_path / "report.md")])
    captured = capsys.readouterr()
    for check in PreflightCheck:
        assert check.value in captured.out


def test_skip_test_suites_flag_is_passed_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: dict[str, object] = {}

    def _fake_run_preflight(**kwargs: object) -> PreflightReport:
        received.update(kwargs)
        return _all_pass_report()

    monkeypatch.setattr("app.cli.run_preflight", _fake_run_preflight)
    main(["app.cli", "preflight", "--skip-test-suites", "--report", str(tmp_path / "r.md")])
    assert received["run_test_suites"] is False


def test_default_invocation_runs_the_test_suites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: dict[str, object] = {}

    def _fake_run_preflight(**kwargs: object) -> PreflightReport:
        received.update(kwargs)
        return _all_pass_report()

    monkeypatch.setattr("app.cli.run_preflight", _fake_run_preflight)
    main(["app.cli", "preflight", "--report", str(tmp_path / "r.md")])
    assert received["run_test_suites"] is True


def test_no_command_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["app.cli"])


def test_an_unknown_command_is_an_error() -> None:
    with pytest.raises(SystemExit):
        main(["app.cli", "not-a-real-command"])
