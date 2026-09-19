"""Getting an alert off the machine.

Nothing here touches the network. A test that needed a real webhook would
only run where someone had configured one, which is exactly the condition
under which alerting quietly stops being tested.
"""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from monitoring.alert_channels import (
    AlertChannelError,
    LogChannel,
    WebhookChannel,
    build_channels,
    deliver_to_all,
    redact,
)


class _Response:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _Recorder:
    """Stands in for urlopen, capturing what would have been sent."""

    def __init__(self, status: int = 200, raises: Exception | None = None) -> None:
        self.status = status
        self.raises = raises
        self.requests: list[Any] = []

    def __call__(self, request: Any, timeout: float | None = None) -> _Response:
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        return _Response(self.status)

    @property
    def payload(self) -> dict[str, Any]:
        decoded: dict[str, Any] = json.loads(self.requests[-1].data.decode("utf-8"))
        return decoded


def _webhook(recorder: _Recorder, **kwargs: Any) -> WebhookChannel:
    return WebhookChannel(url="https://hooks.example.com/abc/def", opener=recorder, **kwargs)


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------


def test_an_alert_is_posted_as_json_to_the_configured_field() -> None:
    recorder = _Recorder()
    channel = _webhook(recorder)

    channel.deliver(message="halted", severity="critical", alert_type="halt", subject="portfolio")

    assert recorder.payload == {"text": "halted"}
    assert recorder.requests[-1].method == "POST"
    assert recorder.requests[-1].headers["Content-type"] == "application/json"


def test_static_fields_ride_along_so_one_channel_reaches_any_service() -> None:
    """Slack wants ``text``, Discord wants ``content``, Telegram wants
    ``text`` plus a ``chat_id``. One implementation covers all of them only
    if both the key and the extra fields are configuration."""
    recorder = _Recorder()
    channel = _webhook(recorder, message_field="text", static_fields={"chat_id": "12345"})

    channel.deliver(message="halted", severity="critical", alert_type="halt", subject="p")

    assert recorder.payload == {"chat_id": "12345", "text": "halted"}


def test_an_http_error_status_is_an_error_not_a_silent_success() -> None:
    """A 404 from a revoked Slack hook returns a response, not an exception.
    Treating that as delivered is how alerting dies without anyone noticing."""
    channel = _webhook(_Recorder(status=404))

    with pytest.raises(AlertChannelError, match="HTTP 404"):
        channel.deliver(message="m", severity="critical", alert_type="t", subject="s")


# ---------------------------------------------------------------------------
# Refusing to be insecure or misconfigured
# ---------------------------------------------------------------------------


def test_a_plain_http_webhook_is_refused() -> None:
    """Alerts carry position sizes, P&L and halt reasons, and the URL is
    itself the credential."""
    with pytest.raises(AlertChannelError, match="must be https"):
        WebhookChannel(url="http://hooks.example.com/abc")


def test_configuring_webhook_without_the_environment_variable_fails_at_startup() -> None:
    """The failure has to happen at startup. A deployment that lists
    'webhook' believes it can page; finding out otherwise during the
    incident is finding out at the worst possible moment."""
    with pytest.raises(AlertChannelError, match="ALERT_WEBHOOK_URL is not set"):
        build_channels(["log", "webhook"], environ={})


def test_an_unknown_channel_name_is_refused() -> None:
    with pytest.raises(AlertChannelError, match="unsupported alert channel"):
        build_channels(["log", "carrier_pigeon"], environ={})


def test_log_only_configuration_builds_without_any_environment() -> None:
    channels = build_channels(["log"], environ={})
    assert [c.name for c in channels] == ["log"]


def test_webhook_is_built_from_the_environment_not_from_config() -> None:
    channels = build_channels(
        ["log", "webhook"], environ={"ALERT_WEBHOOK_URL": "https://hooks.example.com/x"}
    )
    assert [c.name for c in channels] == ["log", "webhook"]


# ---------------------------------------------------------------------------
# The URL is a secret
# ---------------------------------------------------------------------------


def test_redaction_keeps_the_host_and_drops_the_path() -> None:
    """A Telegram URL embeds the bot token in its path and a Slack hook's
    whole secret is the path -- and a delivery failure is exactly when
    something would try to log the URL."""
    redacted = redact("https://api.telegram.org/bot123:AAErq/sendMessage")

    assert "api.telegram.org" in redacted
    assert "123:AAErq" not in redacted
    assert "sendMessage" not in redacted


def test_a_refused_url_is_not_echoed_in_the_error() -> None:
    with pytest.raises(AlertChannelError) as caught:
        WebhookChannel(url="http://hooks.example.com/super/secret/path")

    assert "secret" not in str(caught.value)


# ---------------------------------------------------------------------------
# A failing pager must not take down the trader
# ---------------------------------------------------------------------------


def test_a_dead_webhook_does_not_propagate() -> None:
    """An unreachable paging service must not be able to kill the process it
    is watching. The alert is already in the log; only its delivery is lost.
    """
    channel = _webhook(_Recorder(raises=urllib.error.URLError("connection refused")))

    delivered = deliver_to_all(
        [channel], message="m", severity="critical", alert_type="t", subject="s"
    )

    assert delivered == 0


def test_one_dead_channel_does_not_stop_the_others() -> None:
    dead = _webhook(_Recorder(raises=urllib.error.URLError("down")))
    alive = _webhook(_Recorder())

    delivered = deliver_to_all(
        [dead, alive, LogChannel()], message="m", severity="critical", alert_type="t", subject="s"
    )

    assert delivered == 2


def test_delivery_count_distinguishes_paged_from_tried_and_failed() -> None:
    """Zero pages because nothing was wrong, and zero pages because every
    channel is broken, look identical without this."""
    assert (
        deliver_to_all([LogChannel()], message="m", severity="info", alert_type="t", subject="s")
        == 1
    )
