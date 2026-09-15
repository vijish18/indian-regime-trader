"""The end-to-end paper-trading validation itself (Phase 21), run as a
pytest test so it is part of the normal suite rather than a script no one
runs. This is deliberately the slowest test in the repository -- it wires
the *entire* system (real ``config/settings.yaml``, a real fitted and
approved HMM, a real ``PaperBroker``) and runs a full session through it,
including all eight required failure injections -- because that
end-to-end claim is exactly what a collection of per-layer unit tests
cannot make on its own.

The session is built and run **once** (module-scoped fixture) and every
test below asserts against that one result -- building the environment
alone fits a full production-config HMM (~10s); running it a second or
third time for separate assertions would multiply that for no benefit,
since nothing here mutates the report after it is produced.

Individual pieces (``validation.paper_feed``, ``validation.invariants``,
``validation.synthetic_market``) have their own, fast, focused unit tests
elsewhere; this file exists to prove the assembled whole.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from validation.harness import ValidationEnvironment, build_validation_environment
from validation.invariants import Invariant, InvariantStatus
from validation.scenario import SessionReport, StageStatus, run_end_to_end_validation

_REQUIRED_STAGES = {
    "market-data ingestion",
    "feature calculation + HMM",
    "stock ranking",
    "portfolio construction",
    "risk management",
    "paper execution",
    "fills",
    "portfolio accounting",
    "monitoring",
    "shutdown",
    "reconciliation",
}

_REQUIRED_FAILURE_INJECTIONS = {
    "lost WebSocket",
    "delayed market data",
    "broker API timeout",
    "rejected order",
    "partial fill",
    "duplicate event",
    "application crash",
    "database restart",
}


@pytest.fixture(scope="module")
def environment(tmp_path_factory: pytest.TempPathFactory) -> ValidationEnvironment:
    root = tmp_path_factory.mktemp("e2e_validation")
    return build_validation_environment(root, sessions=1100, train_end_index=900)


@pytest.fixture(scope="module")
def session(environment: ValidationEnvironment) -> SessionReport:
    return run_end_to_end_validation(environment)


def test_the_full_session_completes_with_every_stage_and_invariant_passing(
    session: SessionReport,
) -> None:
    failures = [(s.name, s.detail) for s in session.failed_stages]
    assert failures == [], f"stage(s) failed: {failures}"

    failed_invariants = [
        (r.invariant.value, r.detail)
        for r in session.final_invariants
        if r.status is InvariantStatus.FAIL
    ]
    assert failed_invariants == [], f"final invariant(s) failed: {failed_invariants}"
    assert session.all_ok is True


def test_every_required_narrative_stage_ran(session: SessionReport) -> None:
    ran = {stage.name for stage in session.stages if stage.category == "stage"}
    missing = _REQUIRED_STAGES - ran
    assert missing == set(), f"required stage(s) never ran: {missing}"


def test_every_required_failure_injection_ran(session: SessionReport) -> None:
    ran = {stage.name for stage in session.stages if stage.category == "failure_injection"}
    missing = _REQUIRED_FAILURE_INJECTIONS - ran
    assert missing == set(), f"required failure injection(s) never ran: {missing}"


def test_every_required_invariant_was_evaluated_and_passed_at_the_end(
    session: SessionReport,
) -> None:
    # RESTART_IS_SAFE is not a point-in-time check (check_invariants
    # evaluates a single moment; "is a restart safe" is a property of a
    # *sequence* -- crash, then correct refusal to trade, then recovery).
    # It is proven by the application-crash/reconciliation stage pair
    # instead -- see test_the_application_crash_scenario_refuses_to_trade_until_recovered.
    checkable = set(Invariant) - {Invariant.RESTART_IS_SAFE}
    evaluated = {result.invariant for result in session.final_invariants}
    assert evaluated == checkable
    for result in session.final_invariants:
        assert result.status is not InvariantStatus.SKIPPED, (
            f"{result.invariant.value} was skipped, not proven -- {result.detail}"
        )
        assert result.ok, f"{result.invariant.value} failed: {result.detail}"


def test_no_live_credentials_are_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structural, not just behavioral: even if the environment happened
    to have real broker credentials set, this validation must never read
    them. Deliberately its own, separately-built environment (not the
    shared fixture) so the credential check applies to construction
    itself, not to a session already built without them.
    """
    monkeypatch.setenv("BROKER_API_KEY", "should-never-be-read")
    monkeypatch.setenv("BROKER_API_SECRET", "should-never-be-read")
    env = build_validation_environment(tmp_path, sessions=1100, train_end_index=900)
    assert env.settings.execution.mode == "paper"
    assert type(env.broker._inner).__name__ == "PaperBroker"  # noqa: SLF001


def test_the_broker_api_timeout_scenario_never_resubmits_the_order(
    session: SessionReport,
) -> None:
    """The one scenario with a direct historical precedent worth naming
    explicitly (Phase 17's own CRITICAL scenario): an ambiguous broker
    response must resolve by querying the broker, never by retrying the
    write.
    """
    timeout_stage = next(s for s in session.stages if s.name == "broker API timeout")
    assert timeout_stage.status == StageStatus.PASS
    assert "never resubmitted" in timeout_stage.detail


def test_the_application_crash_scenario_refuses_to_trade_until_recovered(
    session: SessionReport,
) -> None:
    crash_stage = next(s for s in session.stages if s.name == "application crash")
    reconciliation_stage = next(s for s in session.stages if s.name == "reconciliation")
    assert crash_stage.status == StageStatus.PASS
    assert reconciliation_stage.status == StageStatus.PASS
    assert "permit_strategy_execution=True" in reconciliation_stage.detail
