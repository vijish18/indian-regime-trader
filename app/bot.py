"""The weekly trading bot: what runs on a schedule, and what it tells you.

    python -m app.bot evening [--as-of YYYY-MM-DD]   # after the nightly data update
    python -m app.bot morning                        # rebalance mornings, ~08:30 IST
    python -m app.bot trade                          # rebalance mornings, from 09:14 IST
    python -m app.bot status

The strategy is whatever ``config/settings.yaml`` says -- ``bot.exposure``,
``bot.rebalance``, ``selection`` and ``portfolio`` -- run through
``app.paper_runtime.PaperRuntime`` (the orchestrator with a paper broker
that fills against real Kite bid/ask). This module only decides *when*
each step runs and *what the owner is told*.

**The one thing only a human can do** is log in to Kite: Zerodha documents
no automated login, and this bot does not script around it. So on the
evening before a rebalance, and again at ~08:30 on the day, it sends the
login link; ``morning`` then waits for the redirect and caches the session.
No login by the time ``trade`` starts means no trading that week -- and a
message saying so -- never a guess.

Every command exits 0 when it did its job (including "nothing to do
today"), non-zero when it could not, so a scheduler can alert on failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from app.notifier import Notifier
from app.paper_market import IST
from broker.zerodha.kite_session import KiteSessionError, load_session
from config.loader import load_environment, load_settings
from config.models import Settings
from data.calendar import NSETradingCalendar
from orchestration.rebalance_schedule import is_rebalance_session, next_rebalance_session
from scripts.run_walk_forward import REPO_ROOT

if TYPE_CHECKING:
    from app.paper_runtime import PaperRuntime

STATE_DIR = REPO_ROOT / "state" / "bot"
LOGIN_WAIT_UNTIL_IST = dt.time(9, 10)


def _calendar() -> NSETradingCalendar:
    from scripts.run_walk_forward import HOLIDAY_FILE

    return NSETradingCalendar.from_file(HOLIDAY_FILE)


def _last_session_on_or_before(calendar: NSETradingCalendar, day: dt.date) -> dt.date:
    return day if calendar.is_trading_day(day) else calendar.previous_trading_day(day)


def _session_ok(now: dt.datetime) -> tuple[bool, str]:
    try:
        session = load_session(now=now)
    except KiteSessionError as exc:
        return False, str(exc)
    return True, f"valid until {session.expires_at.astimezone(IST):%d %b %H:%M IST}"


def _login_url() -> str:
    from broker.zerodha.kite_historical import KiteHistoricalClient
    from scripts.kite_login import DEFAULT_ENV_FILE, resolve_credentials

    api_key, _ = resolve_credentials(DEFAULT_ENV_FILE)
    return KiteHistoricalClient(api_key).login_url()


def _index_last_date(symbol: str) -> dt.date | None:
    from scripts.run_walk_forward import DATA_CACHE

    path = DATA_CACHE / "raw" / "index" / f"{symbol}.csv"
    if not path.is_file():
        return None
    last = path.read_text(encoding="utf-8").strip().splitlines()[-1].split(",")
    try:
        return dt.date.fromisoformat(last[1])
    except (IndexError, ValueError):
        return None


def _runtime(settings: Settings, as_of: dt.date) -> PaperRuntime:
    from app.paper_runtime import PaperRuntime

    return PaperRuntime(as_of, STATE_DIR, settings.bot.capital)


def _plan_text(runtime: PaperRuntime) -> str:
    picks = [c.symbol for c in runtime.candidates[: runtime.settings.selection.max_holdings]]
    held = {
        p.instrument_id.removeprefix("NSE:")
        for p in runtime.orchestrator.position_tracker.current_positions()
    }
    buys = [s for s in picks if s not in held]
    sells = sorted(held - set(picks))
    lines = [f"Top {len(picks)}: {', '.join(picks)}"]
    lines.append(f"Expected buys: {', '.join(buys) or 'none'}")
    lines.append(f"Expected sells: {', '.join(sells) or 'none'}")
    return "\n".join(lines)


REGIME_FIELDS = [
    "as_of",
    "label",
    "state_id",
    "confidence",
    "sizes_book",
    "model_id",
    "probabilities",
]


def record_regime(published: Path, history: Path) -> bool:
    """Append the published regime to the day-by-day log, once per session.
    The log is the as-recorded history: each row is what the model in force
    that evening said, unlike a path re-filtered later with a newer model."""
    import csv
    import json

    regime = json.loads(published.read_text(encoding="utf-8")).get("regime_now", {})
    if not regime.get("available"):
        return False
    seen: set[str] = set()
    if history.exists():
        with history.open(encoding="utf-8", newline="") as handle:
            seen = {row["as_of"] for row in csv.DictReader(handle)}
    if regime["as_of"] in seen:
        return False
    new = not history.exists()
    with history.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGIME_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(
            {
                "as_of": regime["as_of"],
                "label": regime["label"],
                "state_id": regime.get("state_id", ""),
                "confidence": f"{regime['confidence']:.4f}",
                "sizes_book": regime.get("sizes_book", ""),
                "model_id": regime.get("model_id", ""),
                "probabilities": json.dumps([round(x, 4) for x in regime["probabilities"]]),
            }
        )
    return True


def cmd_evening(
    settings: Settings, notifier: Notifier, as_of: dt.date | None, *, final: bool = True
) -> int:
    """``final=False`` is an early attempt with a later retry scheduled: its
    failures are printed, not sent, so a slow NSE publish does not page."""
    calendar = _calendar()
    today = dt.datetime.now(IST).date()
    as_of = as_of or _last_session_on_or_before(calendar, today)
    marker = STATE_DIR / f"evening_done_{as_of}"
    if marker.exists():
        return 0

    def fail(subject: str, body: str) -> int:
        if final:
            notifier.send(subject, body, severity="error")
        else:
            print(f"[not final] {subject}: {body}", flush=True)
        return 1

    nifty = _index_last_date("NIFTY50")
    if nifty != as_of:
        return fail(
            "Bot: data not ready",
            f"NIFTY 50 data ends {nifty}, expected {as_of}. Tonight's ranking was skipped.",
        )
    try:
        runtime = _runtime(settings, as_of)
        runtime.publish(STATE_DIR / "dashboard.json")
    except Exception as exc:  # noqa: BLE001 - report, then fail the scheduler job
        return fail("Bot: ranking failed", f"{type(exc).__name__}: {exc}")
    record_regime(STATE_DIR / "dashboard.json", STATE_DIR / "regime_history.csv")
    marker.touch()
    nxt = calendar.next_trading_day(as_of)
    if is_rebalance_session(settings.bot.rebalance, calendar, nxt):
        notifier.send(
            f"Rebalance next session ({nxt:%a %d %b})",
            f"{_plan_text(runtime)}\n\nLog in to Kite before 09:10 IST:\n{_login_url()}",
        )
    return 0


def cmd_morning(settings: Settings, notifier: Notifier) -> int:
    calendar = _calendar()
    now = dt.datetime.now(IST)
    if not is_rebalance_session(settings.bot.rebalance, calendar, now.date()):
        return 0
    ok, detail = _session_ok(now)
    if ok:
        return 0
    deadline = dt.datetime.combine(now.date(), LOGIN_WAIT_UNTIL_IST, tzinfo=IST)
    seconds = max(60.0, (deadline - now).total_seconds())
    notifier.send(
        "Rebalance today: please log in to Kite",
        f"Tap and log in before 09:10 IST:\n{_login_url()}",
        severity="warning",
    )
    from scripts import kite_login

    try:
        kite_login.main(["kite_login", "--no-browser", "--timeout", str(int(seconds))])
    except SystemExit as exc:
        if exc.code not in (0, None):
            notifier.send("Kite login not received", str(exc.code), severity="error")
            return 1
    ok, detail = _session_ok(dt.datetime.now(IST))
    notifier.send("Kite login received" if ok else "Kite login failed", detail)
    return 0 if ok else 1


def _orders_text(runtime: PaperRuntime, day: dt.date) -> str:
    rows = [
        o
        for o in runtime.orchestrator.order_manager.all_orders()
        if o.created_at.astimezone(IST).date() == day
    ]
    if not rows:
        return "No orders were needed."
    out = []
    for o in sorted(rows, key=lambda r: (r.side != "sell", r.instrument_id)):
        price = f" @ {o.avg_fill_price:,.2f}" if o.avg_fill_price else ""
        out.append(
            f"{o.side.upper()} {o.instrument_id.removeprefix('NSE:')} "
            f"{o.filled_quantity}/{o.quantity}{price} [{o.state.value}]"
            + (f" {o.reject_reason}" if o.reject_reason else "")
        )
    return "\n".join(out)


def cmd_trade(settings: Settings, notifier: Notifier) -> int:
    calendar = _calendar()
    now = dt.datetime.now(IST)
    today = now.date()
    if not is_rebalance_session(settings.bot.rebalance, calendar, today):
        return 0
    ok, detail = _session_ok(now)
    if not ok:
        notifier.send(
            "No rebalance this week",
            f"No valid Kite login ({detail}). Positions are held unchanged; "
            f"next rebalance {next_rebalance_session(settings.bot.rebalance, calendar, today)}.",
            severity="error",
        )
        return 1
    as_of = calendar.previous_trading_day(today)
    try:
        runtime = _runtime(settings, as_of)
    except Exception as exc:  # noqa: BLE001
        notifier.send(
            "Bot: could not start trading", f"{type(exc).__name__}: {exc}", severity="error"
        )
        return 1
    hh, mm = (int(x) for x in settings.bot.trade_until_ist.split(":"))
    until = dt.datetime.combine(today, dt.time(hh, mm), tzinfo=IST)
    output = STATE_DIR / "dashboard.json"
    errors: dict[str, int] = {}
    while dt.datetime.now(IST) < until:
        try:
            runtime.tick(dt.datetime.now(dt.UTC))
        except Exception as exc:  # noqa: BLE001 - keep trying until the window closes
            key = f"{type(exc).__name__}: {exc}"
            errors[key] = errors.get(key, 0) + 1
            runtime.status, runtime.detail = "blocked", key
            if errors[key] == 1:
                notifier.send("Bot: trading step failed (will retry)", key, severity="warning")
        runtime.publish(output)
        time.sleep(30)
    positions = runtime.orchestrator.position_tracker.current_positions()
    cash = runtime.broker.cash
    value = sum(p.quantity * p.current_price for p in positions)
    equity = cash + value
    notifier.send(
        f"Rebalance {today:%d %b}: {runtime.status}",
        f"{_orders_text(runtime, today)}\n\nEquity Rs {equity:,.0f} "
        f"({equity / settings.bot.capital - 1:+.2%} since start), cash Rs {cash:,.0f}, "
        f"{len(positions)} positions.\nDetail: {runtime.detail}",
    )
    return 0


def cmd_status(settings: Settings) -> int:
    calendar = _calendar()
    now = dt.datetime.now(IST)
    ok, detail = _session_ok(now)
    print(
        f"mode={settings.execution.mode} exposure={settings.bot.exposure} "
        f"rebalance={settings.bot.rebalance} capital={settings.bot.capital:,.0f}"
    )
    mode = settings.bot.rebalance
    print(f"today {now.date()} rebalance={is_rebalance_session(mode, calendar, now.date())}")
    print(f"next rebalance {next_rebalance_session(mode, calendar, now.date())}")
    print(f"kite session: {'OK ' if ok else 'MISSING '}{detail}")
    print(f"NIFTY50 data through {_index_last_date('NIFTY50')}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evening = sub.add_parser("evening")
    evening.add_argument("--as-of", type=dt.date.fromisoformat, default=None)
    evening.add_argument(
        "--not-final", action="store_true", help="a retry follows: print failures, do not send"
    )
    sub.add_parser("morning")
    sub.add_parser("trade")
    sub.add_parser("status")
    args = parser.parse_args(argv[1:])

    load_environment()
    settings = load_settings()
    if settings.execution.mode != "paper":
        raise SystemExit("app.bot runs paper mode only until the live runtime is built")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        return cmd_status(settings)
    notifier = Notifier.from_settings(settings)
    if args.command == "evening":
        return cmd_evening(settings, notifier, args.as_of, final=not args.not_final)
    if args.command == "morning":
        return cmd_morning(settings, notifier)
    return cmd_trade(settings, notifier)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
