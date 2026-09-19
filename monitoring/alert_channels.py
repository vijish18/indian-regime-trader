"""Where an alert goes once it has survived rate limiting.

Until now there was one destination -- the process's own log file -- which
means a system that halts at 09:20 has told nobody. The log is a record for
afterwards, not a notification: reading it requires already knowing to look,
which is precisely what an alert exists to tell you.

**One generic webhook rather than a family of vendor integrations.** Slack,
Discord, Telegram, Google Chat, ntfy and PagerDuty all accept an HTTPS POST
carrying JSON; they differ only in which key holds the text and what else
must ride along. So the channel posts a JSON object whose message key and
extra fields are configuration, and one implementation reaches all of them.
A Telegram bot, for instance:

    url             https://api.telegram.org/bot<TOKEN>/sendMessage
    message_field   text
    static_fields   {"chat_id": "<id>"}

**The URL is a credential and never appears in configuration.** Anyone
holding a Slack or Telegram webhook URL can post as you, and a Telegram URL
embeds the bot token outright. So config names an *environment variable* and
the value is read from the process environment, which keeps it out of Git in
the same way broker credentials are -- and every log line here redacts it,
because a delivery failure is exactly when a URL would otherwise be printed.

**Delivery is synchronous, with a short timeout.** A background thread would
keep the trading loop moving, at the cost of never knowing whether the page
arrived; for the one process whose job is to stop safely, knowing is worth
more than the seconds. The timeout is what bounds the risk: a webhook that
hangs delays a daily cycle by at most ``timeout_seconds``, and a webhook
that fails never propagates -- an unreachable paging service must not be
able to take down the trading process it is watching.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Protocol

from monitoring.logger import get_logger

logger = get_logger("monitoring.alert_channels")

LOG = "log"
WEBHOOK = "webhook"

SUPPORTED_CHANNELS: frozenset[str] = frozenset({LOG, WEBHOOK})


class AlertChannelError(RuntimeError):
    """A channel could not be constructed -- misconfiguration, not a
    delivery failure. Raised at startup so a deployment that believes it
    can page finds out before it needs to."""


def redact(url: str) -> str:
    """A URL reduced to what is safe to log: scheme, host, and the shape of
    the path. Telegram embeds the bot token in the path and Slack's whole
    secret is the path, so neither may be logged whole -- and a delivery
    failure is exactly the moment something would try.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return "<unparseable url>"
    depth = len([segment for segment in parts.path.split("/") if segment])
    return f"{parts.scheme}://{parts.netloc}/<{depth} path segment(s) redacted>"


class AlertChannel(Protocol):
    """Somewhere an alert can be delivered."""

    name: str

    def deliver(self, *, message: str, severity: str, alert_type: str, subject: str) -> None:
        """Send it, or raise. Callers isolate failures; channels do not
        need to."""


class LogChannel:
    """The original destination, kept as a channel so it is never the
    *only* one by accident -- a configuration naming only ``log`` now says
    so explicitly rather than by omission."""

    name = LOG

    def deliver(self, *, message: str, severity: str, alert_type: str, subject: str) -> None:
        # Deliberately a no-op: AlertManager already writes every alert to
        # the log with full structured fields before dispatching to
        # channels. Emitting again here would double every line.
        return None


class WebhookChannel:
    """An HTTPS POST carrying JSON, which is what every chat and paging
    service accepts."""

    name = WEBHOOK

    def __init__(
        self,
        *,
        url: str,
        message_field: str = "text",
        static_fields: dict[str, str] | None = None,
        timeout_seconds: float = 10.0,
        opener: object | None = None,
    ) -> None:
        if not url:
            raise AlertChannelError("webhook url is empty")
        scheme = urllib.parse.urlsplit(url).scheme
        if scheme != "https":
            # Alerts carry position sizes, P&L and halt reasons. Over plain
            # HTTP those are readable by anything on the path, and the URL
            # itself -- the credential -- goes with them.
            raise AlertChannelError(
                f"webhook url must be https, got {scheme or 'no'} scheme ({redact(url)})"
            )
        if not message_field:
            raise AlertChannelError("webhook message_field must not be empty")
        self.url = url
        self.message_field = message_field
        self.static_fields = dict(static_fields or {})
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    def deliver(self, *, message: str, severity: str, alert_type: str, subject: str) -> None:
        payload: dict[str, object] = dict(self.static_fields)
        payload[self.message_field] = message
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = self._opener or urllib.request.urlopen
        with opener(request, timeout=self.timeout_seconds) as response:  # type: ignore[operator]
            status = getattr(response, "status", 200)
            if status >= 400:
                raise AlertChannelError(f"webhook returned HTTP {status}")


def build_channels(
    names: list[str],
    *,
    webhook_url_env: str = "ALERT_WEBHOOK_URL",
    webhook_message_field: str = "text",
    webhook_static_fields: dict[str, str] | None = None,
    webhook_timeout_seconds: float = 10.0,
    environ: dict[str, str] | None = None,
) -> list[AlertChannel]:
    """Construct the configured channels, or refuse at startup.

    Refusing here rather than at first use is the point. A deployment that
    lists ``webhook`` but never set the environment variable believes it can
    page; discovering otherwise during the incident is discovering it at the
    worst possible time.
    """
    env = environ if environ is not None else dict(os.environ)
    unsupported = sorted(set(names) - SUPPORTED_CHANNELS)
    if unsupported:
        raise AlertChannelError(
            f"unsupported alert channel(s): {unsupported}; this system can deliver to "
            f"{sorted(SUPPORTED_CHANNELS)} only."
        )

    channels: list[AlertChannel] = []
    for name in names:
        if name == LOG:
            channels.append(LogChannel())
        elif name == WEBHOOK:
            url = env.get(webhook_url_env, "").strip()
            if not url:
                raise AlertChannelError(
                    f"alert channel 'webhook' is configured but {webhook_url_env} is not set "
                    "in the environment. The URL is a credential and is deliberately not "
                    "read from the config file; set it in the environment (deploy/app.env) "
                    "or remove 'webhook' from monitoring.alert_channels."
                )
            channels.append(
                WebhookChannel(
                    url=url,
                    message_field=webhook_message_field,
                    static_fields=webhook_static_fields,
                    timeout_seconds=webhook_timeout_seconds,
                )
            )
    return channels


def deliver_to_all(
    channels: list[AlertChannel],
    *,
    message: str,
    severity: str,
    alert_type: str,
    subject: str,
) -> int:
    """Deliver to every channel, counting successes.

    Failures are logged and swallowed, one channel at a time. A paging
    service that is down, rate-limiting, or misconfigured must not be able
    to take down the trading process that is trying to report to it -- the
    alert has already been written to the log, so the information is not
    lost, only its delivery.

    Returns how many channels accepted it, so a caller can tell "paged" from
    "tried to page and could not".
    """
    delivered = 0
    for channel in channels:
        try:
            channel.deliver(
                message=message, severity=severity, alert_type=alert_type, subject=subject
            )
        except (AlertChannelError, urllib.error.URLError, OSError, ValueError) as exc:
            logger.critical(
                "alert delivery FAILED on channel %s: %s",
                channel.name,
                exc,
                extra={
                    "extra_fields": {
                        "event": "alert_delivery_failed",
                        "channel": channel.name,
                        "alert_type": alert_type,
                        "severity": severity,
                    }
                },
            )
            continue
        delivered += 1
    return delivered
