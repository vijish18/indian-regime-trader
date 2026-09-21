"""Read-only, loopback dashboard. Never imports a broker or serves the state directory."""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ASSETS = Path(__file__).with_name("web")
PUBLIC_FIELDS = frozenset({
    "generated_at", "regime_now", "hmm", "live_book", "realized", "ticks",
    "selection", "strategies", "equity_curves",
})


def read_snapshot(state_dir: Path) -> bytes:
    """Reject partial/non-finite snapshots; callers retain their last good display."""
    def invalid(value: str) -> None:
        raise ValueError(f"Non-finite JSON: {value}")

    raw = json.loads(
        (state_dir / "dashboard_data.json").read_text(encoding="utf-8"),
        parse_constant=invalid,
    )
    if not isinstance(raw, dict):
        raise ValueError("Snapshot must be an object")
    data = {key: value for key, value in raw.items() if key in PUBLIC_FIELDS}
    # The independent quote recorder can update more frequently than the book.
    # Preserve separate timestamps: newer ticks do not make an old book fresh.
    try:
        tick_data = json.loads(
            (state_dir / "live_ticks.json").read_text(encoding="utf-8"),
            parse_constant=invalid,
        )
        ticks = tick_data.get("ticks", [])
        if ticks and ticks[-1]["t"] > (data.get("ticks", {}).get("updated_at") or ""):
            data["ticks"] = {"updated_at": ticks[-1]["t"], "points": ticks[-840:]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return json.dumps(data, allow_nan=False, separators=(",", ":")).encode()


class DashboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args: object, state_dir: Path, **kwargs: object) -> None:
        self.state_dir = state_dir
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        route = urlsplit(self.path).path
        routes = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        try:
            if route == "/api/snapshot":
                body, content_type = read_snapshot(self.state_dir), "application/json"
            elif route in routes:
                filename, content_type = routes[route]
                body = (ASSETS / filename).read_bytes()
            else:
                self.send_error(404)
                return
        except (OSError, ValueError, TypeError):
            self.send_error(503, "Snapshot unavailable; waiting for a complete data file")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", type=Path, default=Path("state"))
    args = parser.parse_args()
    handler = partial(DashboardHandler, state_dir=args.state_dir.resolve())
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Regime Desk: http://127.0.0.1:{args.port} (read-only)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
