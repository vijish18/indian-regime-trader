"""Obtain a Kite access token. Run this once each trading morning.

    python scripts/kite_login.py

It prints a login URL, waits on ``http://127.0.0.1:5000/kite/callback``
for Zerodha's redirect, exchanges the returned ``request_token`` for an
``access_token``, and caches it under ``state/`` until it expires at 6 AM
IST tomorrow.

Why a human has to be here: Kite's login requires your Zerodha password
and 2FA on Zerodha's own domain. That is the point -- it is what stands
between an API key and your account, and nothing in this repository tries
to automate around it.

If the loopback listener is inconvenient (running over SSH, port in use,
redirect URL registered differently), use ``--paste``: complete the login
in any browser and paste the whole redirected URL, or just the
``request_token`` from it. The redirect target never has to actually
serve anything -- the token is in the address bar even when the page
fails to load.

    python scripts/kite_login.py --paste

Credentials come from the environment, or from ``deploy/app.env``. They
are deliberately NOT read from the repository-root ``.env``: preflight
check 18 ("live credentials are not present in development
configuration") asserts that file stays clean, and this script must not
be the reason it stops being true.
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.server
import os
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from broker.errors import BrokerError  # noqa: E402
from broker.zerodha.kite_historical import KiteHistoricalClient  # noqa: E402
from broker.zerodha.kite_session import (  # noqa: E402
    IST,
    CachedSession,
    next_expiry,
    save_session,
)

DEFAULT_ENV_FILE = REPO_ROOT / "deploy" / "app.env"
DEFAULT_PORT = 5000
DEFAULT_CALLBACK_PATH = "/kite/callback"

_SUCCESS_PAGE = b"""<!doctype html><meta charset=utf-8>
<title>Kite login complete</title>
<body style="font-family:system-ui;padding:3rem;max-width:34rem">
<h2>Login captured</h2>
<p>You can close this tab and return to the terminal.</p>
<p style="color:#666">This token expires at 6 AM IST tomorrow.</p>
"""

_FAILURE_PAGE = b"""<!doctype html><meta charset=utf-8>
<title>Kite login failed</title>
<body style="font-family:system-ui;padding:3rem;max-width:34rem">
<h2>No request_token in the redirect</h2>
<p>Check the terminal for details.</p>
"""


def load_env_file(path: Path) -> dict[str, str]:
    """Minimal ``KEY=value`` reader.

    Deliberately not ``dotenv.load_dotenv``: that would populate
    ``os.environ`` for the whole process, and this script only needs two
    values in one place. Nothing here prints or logs a value.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


def resolve_credentials(env_file: Path) -> tuple[str, str]:
    from_file = load_env_file(env_file)
    api_key = os.environ.get("BROKER_API_KEY") or from_file.get("BROKER_API_KEY", "")
    api_secret = os.environ.get("BROKER_API_SECRET") or from_file.get("BROKER_API_SECRET", "")

    missing = [
        name
        for name, value in (("BROKER_API_KEY", api_key), ("BROKER_API_SECRET", api_secret))
        if not value
    ]
    if missing:
        raise SystemExit(
            f"missing {', '.join(missing)}.\n"
            f"Set them in {env_file} (see deploy/app.env.example) or in the environment."
        )
    return api_key, api_secret


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    request_token: str | None = None
    callback_path: str = DEFAULT_CALLBACK_PATH

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        token = (params.get("request_token") or [""])[0]

        if parsed.path != type(self).callback_path or not token:
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_FAILURE_PAGE)
            return

        type(self).request_token = token
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_SUCCESS_PAGE)

    def log_message(self, format: str, *args: object) -> None:
        """Silence the default stderr access log -- its request line
        contains the request_token."""


def wait_for_redirect(port: int, callback_path: str, timeout_seconds: float) -> str:
    _CallbackHandler.request_token = None
    _CallbackHandler.callback_path = callback_path

    server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
    server.timeout = 1.0
    thread = threading.Thread(target=_serve_until_token, args=(server, timeout_seconds))
    thread.start()
    thread.join()
    server.server_close()

    token = _CallbackHandler.request_token
    if not token:
        raise SystemExit(
            f"timed out after {timeout_seconds:.0f}s with no redirect to "
            f"127.0.0.1:{port}{callback_path}.\n"
            "Check the Redirect URL registered in the Kite developer console matches, "
            "or re-run with --paste."
        )
    return token


def _serve_until_token(server: http.server.HTTPServer, timeout_seconds: float) -> None:
    deadline = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=timeout_seconds)
    while _CallbackHandler.request_token is None and dt.datetime.now(dt.UTC) < deadline:
        server.handle_request()


def read_pasted_token() -> str:
    print("Paste the full redirected URL (or just the request_token), then press Enter:")
    raw = input("> ").strip()
    if not raw:
        raise SystemExit("nothing pasted")
    if "request_token=" in raw:
        parsed = urllib.parse.urlparse(raw)
        token = (urllib.parse.parse_qs(parsed.query).get("request_token") or [""])[0]
        if not token:
            raise SystemExit("could not find request_token in that URL")
        return token
    return raw


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Obtain and cache a Kite access token.")
    parser.add_argument(
        "--paste",
        action="store_true",
        help="paste the redirect URL instead of listening on loopback",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--callback-path", default=DEFAULT_CALLBACK_PATH)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv[1:])

    api_key, api_secret = resolve_credentials(args.env_file)
    client = KiteHistoricalClient(api_key)
    url = client.login_url()

    print("Open this URL and complete the Zerodha login:\n")
    print(f"    {url}\n")

    if not args.paste:
        print(
            f"Waiting for the redirect to http://127.0.0.1:{args.port}{args.callback_path} "
            f"(up to {args.timeout:.0f}s)..."
        )
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - a headless host simply has no browser
            pass

    token = read_pasted_token() if args.paste else wait_for_redirect(
        args.port, args.callback_path, args.timeout
    )

    try:
        access_token = client.exchange_request_token(token, api_secret)
    except BrokerError as exc:
        raise SystemExit(f"could not exchange the request_token: {exc}") from exc

    now = dt.datetime.now(dt.UTC)
    written = save_session(
        CachedSession(
            access_token=access_token,
            api_key=api_key,
            user_id="",
            obtained_at=now,
            expires_at=next_expiry(now),
        )
    )

    expiry = next_expiry(now).astimezone(IST)
    print("\nLogged in.")
    print(f"  session cached at {written}")
    print(f"  valid until       {expiry:%Y-%m-%d %H:%M} IST")
    print("\nThe token is not printed here on purpose. Next:")
    print("  python scripts/ingest_kite_history.py --help")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
