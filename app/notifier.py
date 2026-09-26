"""Messages from the bot to its owner's phone.

Built on ``monitoring.alert_channels``: the channels (log, webhook) and the
webhook's shape come from ``monitoring`` in settings.yaml, and the webhook
URL -- a credential -- from the environment. A Telegram bot needs

    monitoring.alert_channels: [log, webhook]
    monitoring.alert_webhook_static_fields: {chat_id: "<your chat id>"}
    ALERT_WEBHOOK_URL=https://api.telegram.org/bot<TOKEN>/sendMessage   (deploy/app.env)

A failed delivery is logged and swallowed: an unreachable phone must never
stop the bot from trading or from failing safe.
"""

from __future__ import annotations

import logging

from config.models import Settings
from monitoring.alert_channels import AlertChannel, build_channels

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self, channels: list[AlertChannel]) -> None:
        self.channels = channels

    @classmethod
    def from_settings(cls, settings: Settings) -> Notifier:
        m = settings.monitoring
        return cls(
            build_channels(
                list(m.alert_channels) or ["log"],
                webhook_url_env=m.alert_webhook_url_env,
                webhook_message_field=m.alert_webhook_message_field,
                webhook_static_fields=dict(m.alert_webhook_static_fields),
                webhook_timeout_seconds=m.alert_webhook_timeout_seconds,
            )
        )

    def send(self, subject: str, message: str, *, severity: str = "info") -> None:
        text = f"{subject}\n{message}" if message else subject
        logger.info("notify [%s] %s", severity, text.replace("\n", " | "))
        print(f"[{severity}] {text}", flush=True)  # the scheduler's journal keeps stdout
        for channel in self.channels:
            try:
                channel.deliver(message=text, severity=severity, alert_type="bot", subject=subject)
            except Exception as exc:  # noqa: BLE001 - delivery must never propagate
                logger.warning("notification via %s failed: %s", type(channel).__name__, exc)
