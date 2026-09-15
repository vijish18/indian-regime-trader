"""Minimal HTTP transport for the Kite Connect REST API.

Standard-library only (``urllib.request``) -- no new runtime dependency
for the REST side. Every request ``KiteBroker`` makes goes through
:class:`HttpTransport`, which is exactly the one seam every test in this
package replaces with an in-memory fake instead of touching a real
socket (docs/connect/v3/'s conventions: form-encoded request parameters,
JSON responses -- see ``kite_broker.py`` for the envelope/header format).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    body: bytes

    def json(self) -> object:
        return json.loads(self.body.decode("utf-8"))


class HttpTransport(Protocol):
    """Duck-typed HTTP transport. ``KiteBroker`` depends only on this
    protocol, never on ``urllib`` directly, so a test can inject a fake
    that returns scripted responses with no network access at all.
    """

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, object] | None = None,
    ) -> HttpResponse: ...


class UrllibHttpTransport:
    """The only place this package touches a real socket.

    docs/connect/v3/response-structure/ (verified against the published
    docs, not guessed): "All GET and DELETE request parameters go as
    query parameters, and POST and PUT parameters as form-encoded
    (``application/x-www-form-urlencoded``) parameters." This transport
    applies that convention uniformly rather than leaving it to callers.
    """

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, object] | None = None,
    ) -> HttpResponse:
        params = params or {}
        data: bytes | None = None
        request_url = url
        if method in ("GET", "DELETE"):
            if params:
                request_url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
        else:
            data = urllib.parse.urlencode(params, doseq=True).encode("utf-8")

        request = urllib.request.Request(request_url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return HttpResponse(status_code=response.status, body=response.read())
        except urllib.error.HTTPError as exc:
            return HttpResponse(status_code=exc.code, body=exc.read())
