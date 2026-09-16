"""Kite session caching and the 6 AM IST expiry.

The expiry is the whole substance of this module. Zerodha invalidates
every access token at 6 AM IST daily as a regulatory requirement, so a
long-running process *will* cross that boundary, and a cached token is
worthless if the cache is willing to hand back a dead one. These tests
are mostly about the boundary and about refusing.

Timezone care matters more than usual here: the boundary is defined in
IST, the system stores timestamps in UTC, and 6 AM IST is 00:30 UTC. An
off-by-one-timezone bug would either expire tokens five and a half hours
early (annoying) or five and a half hours late (silently using a dead
credential mid-backfill).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from broker.zerodha.kite_session import (
    IST,
    CachedSession,
    KiteSessionError,
    load_session,
    next_expiry,
    save_session,
    session_path,
)


def _ist(year: int, month: int, day: int, hour: int, minute: int = 0) -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, tzinfo=IST)


def _session(obtained: dt.datetime, api_key: str = "test-key") -> CachedSession:
    return CachedSession(
        access_token="secret-token",
        api_key=api_key,
        user_id="AB1234",
        obtained_at=obtained,
        expires_at=next_expiry(obtained),
    )


# ---------------------------------------------------------------------------
# The 6 AM boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("obtained", "expected_expiry"),
    [
        # Logged in during the trading day -> dies 6 AM tomorrow.
        (_ist(2026, 9, 16, 9, 30), _ist(2026, 9, 17, 6)),
        # Logged in late at night -> still 6 AM tomorrow morning.
        (_ist(2026, 9, 16, 23, 55), _ist(2026, 9, 17, 6)),
        # Logged in before 6 AM -> dies at 6 AM *today*, hours away.
        (_ist(2026, 9, 16, 5, 30), _ist(2026, 9, 16, 6)),
        # Exactly 6 AM counts as the new window.
        (_ist(2026, 9, 16, 6, 0), _ist(2026, 9, 17, 6)),
    ],
)
def test_expiry_is_the_next_six_am_ist(
    obtained: dt.datetime, expected_expiry: dt.datetime
) -> None:
    """The pre-dawn case is the one worth spelling out: a token obtained
    at 05:30 is valid for thirty minutes, not for a day."""
    assert next_expiry(obtained) == expected_expiry


def test_expiry_is_computed_in_ist_regardless_of_the_input_timezone() -> None:
    """6 AM IST is 00:30 UTC. A session stamped in UTC must expire at the
    same instant as one stamped in IST, or the boundary moves by five and
    a half hours depending on who wrote the file."""
    same_moment_utc = _ist(2026, 9, 16, 9, 30).astimezone(dt.UTC)
    assert next_expiry(same_moment_utc) == next_expiry(_ist(2026, 9, 16, 9, 30))


def test_a_session_is_invalid_from_its_expiry_instant_onward() -> None:
    session = _session(_ist(2026, 9, 16, 9, 30))
    assert session.is_valid_at(_ist(2026, 9, 17, 5, 59)) is True
    assert session.is_valid_at(_ist(2026, 9, 17, 6, 0)) is False
    assert session.is_valid_at(_ist(2026, 9, 17, 6, 1)) is False


# ---------------------------------------------------------------------------
# Round trip and refusals
# ---------------------------------------------------------------------------


def test_a_saved_session_loads_back_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "kite_session.json"
    original = _session(_ist(2026, 9, 16, 9, 30))
    save_session(original, path=path)

    loaded = load_session(path=path, now=_ist(2026, 9, 16, 15, 0), api_key="test-key")
    assert loaded == original


def test_an_expired_session_is_refused_rather_than_returned(tmp_path: Path) -> None:
    """The point of the cache. Returning an expired token for the caller
    to check is how a dead credential gets used."""
    path = tmp_path / "kite_session.json"
    save_session(_session(_ist(2026, 9, 16, 9, 30)), path=path)

    with pytest.raises(KiteSessionError, match="expired"):
        load_session(path=path, now=_ist(2026, 9, 17, 7, 0))


def test_a_session_for_a_different_api_key_is_refused(tmp_path: Path) -> None:
    """Rotating the API secret, or switching apps, leaves a cached token
    that authenticates as something else. Silently using it would produce
    a confusing 403 far from the cause."""
    path = tmp_path / "kite_session.json"
    save_session(_session(_ist(2026, 9, 16, 9, 30), api_key="old-key"), path=path)

    with pytest.raises(KiteSessionError, match="different API key"):
        load_session(path=path, now=_ist(2026, 9, 16, 15, 0), api_key="new-key")


def test_a_missing_session_says_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(KiteSessionError, match="kite_login"):
        load_session(path=tmp_path / "absent.json")


def test_a_corrupt_session_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "kite_session.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(KiteSessionError):
        load_session(path=path, now=_ist(2026, 9, 16, 15, 0))


def test_a_session_missing_a_field_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "kite_session.json"
    path.write_text(json.dumps({"access_token": "x"}), encoding="utf-8")
    with pytest.raises(KiteSessionError, match="malformed"):
        load_session(path=path, now=_ist(2026, 9, 16, 15, 0))


# ---------------------------------------------------------------------------
# Where it lives
# ---------------------------------------------------------------------------


def test_the_default_session_path_is_inside_gitignored_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The file holds a live credential. ``state/`` is gitignored, so it
    cannot be committed by accident."""
    monkeypatch.delenv("KITE_SESSION_FILE", raising=False)
    assert session_path().parts[0] == "state"


def test_the_session_path_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KITE_SESSION_FILE", "/tmp/elsewhere.json")
    assert session_path() == Path("/tmp/elsewhere.json")


def test_the_token_is_never_included_in_an_error_message(tmp_path: Path) -> None:
    """Errors from this module get printed to terminals and logs. None of
    them should carry the credential."""
    path = tmp_path / "kite_session.json"
    save_session(_session(_ist(2026, 9, 16, 9, 30)), path=path)

    with pytest.raises(KiteSessionError) as excinfo:
        load_session(path=path, now=_ist(2026, 9, 17, 7, 0))
    assert "secret-token" not in str(excinfo.value)
