"""The terminal dashboard (Phase 20): renders a
``monitoring.snapshot.MonitoringSnapshot`` as plain text.

This module is deliberately inert. It queries nothing, computes nothing,
and holds no state -- ``render_dashboard`` is a pure function from a
snapshot to a string, which is what makes every field on screen testable
by assertion rather than by looking at it. Anything that needed
calculating was calculated by ``SnapshotCollector``; anything that needs
refreshing is the caller's loop.

Plain ASCII, no colour, no cursor control, no third-party terminal
library: this has to be readable over ssh, in a Windows console, and in a
CI log, and a monitoring surface that itself fails to render is worse than
no monitoring surface.
"""

from __future__ import annotations

import datetime as dt
from typing import TextIO

from monitoring.health import ComponentHealth
from monitoring.snapshot import (
    ExecutionPanel,
    MonitoringSnapshot,
    PortfolioPanel,
    RegimePanel,
    RiskPanel,
    SystemPanel,
)
from risk.circuit_breaker import CircuitState

DEFAULT_WIDTH = 80
"""The classic terminal width: the whole frame fits an 80-column console
exactly, with no wrapping."""

_LABEL_WIDTH = 24


def render_dashboard(snapshot: MonitoringSnapshot, width: int = DEFAULT_WIDTH) -> str:
    """The whole dashboard as one string, newline-separated, no trailing
    newline. ``width`` is the inside width of the frame.
    """
    width = max(width, 48)
    lines: list[str] = [
        _rule(width),
        _title(f"INDIAN REGIME TRADER  {_timestamp(snapshot.as_of)}", width),
    ]
    for heading, rows in (
        ("SYSTEM", _system_rows(snapshot.system)),
        ("PORTFOLIO", _portfolio_rows(snapshot.portfolio)),
        ("REGIME", _regime_rows(snapshot.regime)),
        ("EXECUTION", _execution_rows(snapshot.execution)),
        ("RISK", _risk_rows(snapshot.risk)),
    ):
        lines.append(_section(heading, width))
        lines.extend(_row(label, value, width) for label, value in rows)
    lines.append(_rule(width))
    return "\n".join(lines)


class TerminalDashboard:
    """Writes :func:`render_dashboard`'s output to a stream. Separate from
    the rendering itself so tests can assert on the text without capturing
    stdout, and so a caller can send the same frame somewhere other than a
    terminal.
    """

    def __init__(self, stream: TextIO, width: int = DEFAULT_WIDTH) -> None:
        self.stream = stream
        self.width = width

    def render(self, snapshot: MonitoringSnapshot) -> str:
        return render_dashboard(snapshot, self.width)

    def display(self, snapshot: MonitoringSnapshot) -> None:
        self.stream.write(self.render(snapshot) + "\n")
        self.stream.flush()


# -- sections ------------------------------------------------------------


def _system_rows(panel: SystemPanel) -> list[tuple[str, str]]:
    return [
        ("status", panel.status.upper()),
        ("uptime", _duration(panel.uptime)),
        (
            "trading session",
            f"{panel.session_date} {panel.session_type.value} ({panel.session_detail})",
        ),
        ("data feed", _health(panel.data_feed, panel.data_feed_detail)),
        ("broker", _connectivity(panel.broker_connected, panel.broker_detail)),
        ("model version", panel.model_version or "NONE APPROVED"),
    ]


def _portfolio_rows(panel: PortfolioPanel) -> list[tuple[str, str]]:
    return [
        ("equity", _money(panel.equity)),
        ("cash", _money(panel.cash)),
        ("exposure", _pct(panel.gross_exposure_pct)),
        ("daily P&L", f"{_signed_money(panel.daily_pnl)} ({_signed_pct(panel.daily_pnl_pct)})"),
        ("drawdown", _pct(panel.drawdown_pct)),
        ("positions", str(panel.position_count)),
    ]


def _regime_rows(panel: RegimePanel) -> list[tuple[str, str]]:
    regime = panel.regime.upper() if panel.regime else "UNKNOWN"
    if panel.label:
        regime = f"{regime} (label: {panel.label})"
    return [
        ("regime", regime),
        ("probability", _optional_pct(panel.probability)),
        ("confidence", _optional_pct(panel.confidence)),
        ("persistence", _optional_pct(panel.persistence)),
        ("India VIX", _level_and_change(panel.india_vix, panel.india_vix_change)),
        ("NIFTY 50", _level_and_pct(panel.nifty_close, panel.nifty_change_pct)),
    ]


def _execution_rows(panel: ExecutionPanel) -> list[tuple[str, str]]:
    return [
        ("orders submitted", str(panel.orders_submitted)),
        ("fills", str(panel.fills)),
        ("rejected orders", str(panel.rejected_orders)),
        ("open orders", str(panel.open_orders)),
        ("pending reconciliation", _pending(panel)),
    ]


def _risk_rows(panel: RiskPanel) -> list[tuple[str, str]]:
    rows = [
        ("risk mode", panel.risk_mode.value.upper()),
        ("circuit breaker", _circuit(panel)),
    ]
    # The trip reason is free text of unbounded length, so it gets a
    # continuation row of its own rather than competing for space with the
    # structured part of the line above it.
    if panel.risk_mode is not CircuitState.NORMAL and panel.circuit_reason:
        rows.append(("", panel.circuit_reason))
    rows.extend(
        [
            ("concentration", _concentration(panel)),
            ("turnover", _pct(panel.daily_turnover_pct)),
        ]
    )
    return rows


# -- field formatting ----------------------------------------------------


def _health(status: ComponentHealth, detail: str) -> str:
    return f"{status.value.upper()} ({detail})" if detail else status.value.upper()


def _connectivity(connected: bool, detail: str) -> str:
    label = "CONNECTED" if connected else "DISCONNECTED"
    return f"{label} ({detail})" if detail else label


def _circuit(panel: RiskPanel) -> str:
    """Most actionable first. A long reason will be clipped to fit the
    frame, so "MANUAL RESET REQUIRED" -- the only part that tells an
    operator to do something -- goes before it, not after.
    """
    if panel.risk_mode is CircuitState.NORMAL:
        return "none tripped"
    parts = [panel.risk_mode.value.upper()]
    if panel.requires_manual_recovery:
        parts.append("MANUAL RESET REQUIRED")
    if panel.circuit_triggered_by:
        parts.append(f"by {panel.circuit_triggered_by}")
    return " ".join(parts)


def _concentration(panel: RiskPanel) -> str:
    if panel.concentration_instrument is None:
        return "no positions"
    return f"{_pct(panel.max_concentration_pct)} ({panel.concentration_instrument})"


def _pending(panel: ExecutionPanel) -> str:
    total = panel.pending_reconciliation
    if total == 0:
        return "0"
    return (
        f"{total} ({panel.orders_in_unknown_state} unknown order(s), "
        f"{len(panel.reconciliation_mismatches)} position mismatch(es))"
    )


def _timestamp(moment: dt.datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def _duration(delta: dt.timedelta) -> str:
    total = int(delta.total_seconds())
    if total < 0:
        total = 0
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"{days}d {hours:02d}h {minutes:02d}m"
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"


def _money(value: float) -> str:
    return f"INR {value:,.2f}"


def _signed_money(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}INR {abs(value):,.2f}"


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _signed_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _optional_pct(value: float | None) -> str:
    return _pct(value) if value is not None else "n/a"


def _level_and_change(level: float | None, change: float | None) -> str:
    if level is None:
        return "n/a"
    if change is None:
        return f"{level:,.2f}"
    return f"{level:,.2f} ({change:+.2f})"


def _level_and_pct(level: float | None, change_pct: float | None) -> str:
    if level is None:
        return "n/a"
    if change_pct is None:
        return f"{level:,.2f}"
    return f"{level:,.2f} ({_signed_pct(change_pct)})"


# -- frame -----------------------------------------------------------------


def _rule(width: int) -> str:
    return f"+{'=' * (width - 2)}+"


def _title(text: str, width: int) -> str:
    return f"|{_clip(text, width - 2).center(width - 2)}|"


def _section(heading: str, width: int) -> str:
    inner = width - 2
    dashes = max(inner - len(heading) - 3, 0)
    return f"|{('-- ' + heading + ' ' + '-' * dashes)[:inner].ljust(inner)}|"


def _row(label: str, value: str, width: int) -> str:
    inner = width - 4
    text = f"{label:<{_LABEL_WIDTH}}{value}"
    return f"| {_clip(text, inner).ljust(inner)} |"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."
