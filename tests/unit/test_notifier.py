from __future__ import annotations

from app.notifier import Notifier


class _Recorder:
    name = "recorder"

    def __init__(self) -> None:
        self.sent: list[dict[str, str]] = []

    def deliver(self, *, message: str, severity: str, alert_type: str, subject: str) -> None:
        self.sent.append({"message": message, "severity": severity, "subject": subject})


class _Broken:
    name = "broken"

    def deliver(self, *, message: str, severity: str, alert_type: str, subject: str) -> None:
        raise OSError("network down")


def test_every_channel_gets_the_subject_and_body() -> None:
    rec = _Recorder()
    Notifier([rec]).send("Rebalance", "Top 10: A, B", severity="warning")
    assert rec.sent == [
        {"message": "Rebalance\nTop 10: A, B", "severity": "warning", "subject": "Rebalance"}
    ]


def test_a_failing_channel_never_raises_or_blocks_the_others() -> None:
    rec = _Recorder()
    Notifier([_Broken(), rec]).send("Halted", "")
    assert rec.sent[0]["message"] == "Halted"
