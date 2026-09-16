"""Storage for a Kite access token between processes.

A Kite access token expires at **6 AM the following day** -- Zerodha
describes this as a regulatory requirement, not a configurable session
length -- and getting a new one requires a human to complete a browser
login. So a token is worth caching (the login happens once a morning, not
once per script invocation) and worth expiring honestly (a backfill that
starts at 05:55 must not assume it can keep reading at 06:05).

The expiry rule below is deliberately *conservative* about the boundary:
:meth:`CachedSession.is_valid_at` treats the token as dead from 06:00 IST
onward, so the last few minutes before expiry are given up rather than
gambled on. Losing five minutes of backfill is free; discovering the
expiry mid-write is not.

The cache file holds a live credential. It is written under ``state/``,
which is gitignored, and this module never logs the token itself.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

TOKEN_EXPIRY_HOUR_IST = 6
"""Kite invalidates access tokens at 6 AM IST daily."""

DEFAULT_SESSION_PATH = Path("state/kite_session.json")
SESSION_PATH_ENV_VAR = "KITE_SESSION_FILE"


class KiteSessionError(RuntimeError):
    """The cached session is missing, unreadable, or expired.

    One error type for all three because the operator's next action is
    identical in every case -- run ``scripts/kite_login.py`` -- and
    distinguishing them would only invite a caller to handle one by
    carrying on.
    """


def session_path(environ: dict[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    configured = env.get(SESSION_PATH_ENV_VAR, "").strip()
    return Path(configured) if configured else DEFAULT_SESSION_PATH


def next_expiry(now: dt.datetime) -> dt.datetime:
    """The next 06:00 IST strictly after ``now``."""
    local = now.astimezone(IST)
    today_six = local.replace(
        hour=TOKEN_EXPIRY_HOUR_IST, minute=0, second=0, microsecond=0
    )
    return today_six if local < today_six else today_six + dt.timedelta(days=1)


@dataclass(frozen=True, slots=True)
class CachedSession:
    access_token: str
    api_key: str
    user_id: str
    obtained_at: dt.datetime
    expires_at: dt.datetime

    def is_valid_at(self, moment: dt.datetime) -> bool:
        return moment < self.expires_at

    def to_dict(self) -> dict[str, str]:
        return {
            "access_token": self.access_token,
            "api_key": self.api_key,
            "user_id": self.user_id,
            "obtained_at": self.obtained_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }

    @staticmethod
    def from_dict(payload: dict[str, str]) -> CachedSession:
        try:
            return CachedSession(
                access_token=payload["access_token"],
                api_key=payload["api_key"],
                user_id=payload.get("user_id", ""),
                obtained_at=dt.datetime.fromisoformat(payload["obtained_at"]),
                expires_at=dt.datetime.fromisoformat(payload["expires_at"]),
            )
        except (KeyError, ValueError) as exc:
            raise KiteSessionError(f"cached session is malformed: {exc}") from exc


def save_session(
    session: CachedSession, *, path: Path | None = None
) -> Path:
    target = path or session_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
    try:
        target.chmod(0o600)
    except OSError:
        # No-op on Windows/NTFS, where permissions are ACL-based. Not
        # fatal -- but it does mean the file's protection is whatever the
        # directory grants, which docs/KITE_DATA.md says out loud rather
        # than letting a chmod call imply a guarantee it did not deliver.
        pass
    return target


def load_session(
    *, path: Path | None = None, now: dt.datetime | None = None, api_key: str | None = None
) -> CachedSession:
    """The cached session, or :class:`KiteSessionError` explaining what to do.

    Never returns an expired or mismatched session. Handing one back for
    the caller to check is how an expired credential ends up being used.
    """
    target = path or session_path()
    moment = now or dt.datetime.now(dt.UTC)

    if not target.is_file():
        raise KiteSessionError(
            f"no Kite session at {target}. Run: python scripts/kite_login.py"
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KiteSessionError(f"could not read {target}: {exc}") from exc
    if not isinstance(payload, dict):
        raise KiteSessionError(f"{target} does not contain a JSON object")

    session = CachedSession.from_dict(payload)

    if api_key is not None and session.api_key != api_key:
        raise KiteSessionError(
            "the cached session belongs to a different API key than the one "
            "configured now. Run: python scripts/kite_login.py"
        )
    if not session.is_valid_at(moment):
        raise KiteSessionError(
            f"the Kite session expired at {session.expires_at.astimezone(IST):%Y-%m-%d %H:%M %Z}. "
            "Kite tokens die at 6 AM IST daily. Run: python scripts/kite_login.py"
        )
    return session
