"""Paper-only composition root. Closed-session signals, current quotes, real risk vetoes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, cast

from app.paper_market import IST, PaperMarket
from app.paper_state import paper_identity, restore_paper, save_paper
from backtest.costs import TradeSide
from broker.adapters.paper_broker import PaperBroker
from config.loader import load_settings
from core.features.feature_engineering import FeaturePipeline, feature_set_version
from core.regime.model_registry import ModelArtifact, ModelRegistry, NoApprovedModelError
from execution.order_manager import OrderManager
from execution.position_tracker import PositionTracker
from execution.system_state import SystemStateStore
from monitoring.health import HealthChecker
from orchestration.model_validation import validate_model_metadata
from orchestration.orchestrator import Orchestrator
from orchestration.rebalance_schedule import rebalance_due
from orchestration.regime_computation import RegimeComputer, full_exposure_target
from risk.circuit_breaker import CircuitBreaker, CircuitState
from risk.risk_manager import RiskManager
from risk.stop_loss import StopLossPolicy, evaluate
from scripts.run_walk_forward import REPO_ROOT, build_validator
from storage.atomic import atomic_write
from universe.stock_selector import StockSelector


class _SignalDaySelector:
    """The ranking computed at startup, reused for the signal date.

    Ranking the whole universe takes ~20 s on the bot's VM. The orchestrator
    ranks again inside the daily cycle, between the quote fetch and the
    orders; reusing the startup ranking keeps that gap short. Any other date
    goes to the real selector."""

    def __init__(self, inner: Any, as_of: dt.date, candidates: list[Any]) -> None:
        self.inner, self.as_of, self.candidates = inner, as_of, candidates

    def select(self, as_of: dt.date) -> list[Any]:
        return list(self.candidates) if as_of == self.as_of else self.inner.select(as_of)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class PaperRuntime:
    def __init__(self, as_of: dt.date, state_dir: Path, budget: float) -> None:
        self.settings = load_settings()
        if self.settings.execution.mode != "paper":
            raise ValueError("This entry point only supports paper mode")
        self.as_of, self.state_dir, self.budget = as_of, state_dir, budget
        self.validator = build_validator(as_of, state_dir / "research", frame_cache=0)
        self.calendar = self.validator.calendar
        definitions = tuple(
            d
            for d in self.validator.feature_pipeline.definitions
            if d.name != "nifty_volume_stress_20d"
        )
        self.validator.feature_pipeline = FeaturePipeline(definitions)
        registry = ModelRegistry(state_dir / "models")

        def fit_and_approve(reason: str) -> ModelArtifact:
            train_start = self.calendar.sessions_offset(
                as_of, -(self.settings.hmm.training_window_days - 1)
            )
            print(f"Fitting paper model ({reason}): {train_start} .. {as_of}", flush=True)
            model, scaler, _, model_id = self.validator._fit_fold(train_start, as_of)
            fitted = ModelArtifact(
                model_id,
                dt.datetime.now(dt.UTC),
                model,
                scaler,
                feature_set_version(definitions),
                notes="Paper-only model; approval is not evidence of profitability",
            )
            registry.save(fitted)
            registry.approve(model_id)
            return fitted

        try:
            artifact = registry.load_current_approved()
        except NoApprovedModelError:
            artifact = fit_and_approve("no approved model")
        else:
            # Retrain on schedule (hmm.retrain_interval_sessions) rather than
            # let the model age out: the orchestrator refuses a stale model
            # when the regime sizes the book, and reports it as stale otherwise.
            if validate_model_metadata(
                artifact,
                [d.name for d in definitions],
                feature_set_version(definitions),
                as_of=as_of,
                calendar=self.calendar,
                max_age_sessions=self.settings.hmm.retrain_interval_sessions,
            ):
                artifact = fit_and_approve("approved model is due for retraining")
        self.artifact = artifact
        print(f"Ranking equities as of {as_of} ...", flush=True)
        self.candidates = self.validator.engine.stock_selector.select(as_of)
        if not self.candidates:
            raise ValueError("No eligible candidates in the fresh snapshot")
        if any(
            (span := self.validator.market_data.available_range(c.instrument_id)) is None
            or span[1] != as_of
            for c in self.candidates
        ):
            raise ValueError("Candidate bars do not reach the signal date")
        # The date is fixed for this process. The next session needs a newly
        # prepared snapshot, rather than silently reusing an old ranking.
        selector = self.validator.engine.stock_selector
        self.market = PaperMarket(self.validator.market_data, REPO_ROOT / "state/kite_session.json")
        broker = PaperBroker(
            self.market,
            self.validator.engine.cost_model,
            self.settings.execution,
            self.settings.paper_trading,
            PositionTracker(),
            budget,
        )
        self.broker = broker
        tracker, orders = PositionTracker(), OrderManager()
        breaker = CircuitBreaker(self.settings.risk, state_dir / "circuit_breaker.json")
        health = HealthChecker(
            broker,
            self.market,
            selector.universe_provider,
            registry,
            self.calendar,
            [c.instrument_id for c in self.candidates],
            max_stale_sessions=0,
            model_max_age_sessions=self.settings.hmm.retrain_interval_sessions,
        )
        self.orchestrator = Orchestrator(
            calendar=self.calendar,
            market_data=self.market,
            stock_selector=cast(
                StockSelector, _SignalDaySelector(selector, as_of, self.candidates)
            ),
            regime_computer=RegimeComputer(
                self.market,
                self.settings.hmm,
                self.settings.allocation,
                self.validator.regime_policy,
                self.validator.feature_pipeline,
                "NIFTY50",
                "INDIAVIX",
                600,
                exposure=self.settings.bot.exposure,
            ),
            portfolio_constructor=self.validator.engine.portfolio_constructor,
            risk_manager=RiskManager(self.settings.risk, breaker),
            circuit_breaker=breaker,
            model_registry=registry,
            position_tracker=tracker,
            order_manager=orders,
            broker=broker,
            state_store=SystemStateStore(state_dir / "system_state.json"),
            health_checker=health,
            execution_config=self.settings.execution,
            strategy_version=(
                f"paper-orchestrated-v1:{self.settings.bot.exposure}:{self.settings.bot.rebalance}"
            ),
            settings_loader=lambda: self.settings,
        )
        self.identity = paper_identity(self.settings, budget)
        self.metadata = restore_paper(state_dir / "ledger.json", self.identity, self.orchestrator)
        self.policy = StopLossPolicy.from_mapping(self.settings.risk.stop_loss.model_dump())
        self.status = "prepared"
        self.detail = "Waiting for the next trading session and a fresh Kite login"
        self.last_quote_at: str | None = None
        self.ticks: list[dict[str, Any]] = []
        self.persist()

    def persist(self) -> None:
        save_paper(self.state_dir / "ledger.json", self.identity, self.orchestrator, self.metadata)

    def tick(self, now: dt.datetime) -> None:
        today = now.astimezone(IST).date()
        session = self.calendar.session(today)
        if not session.is_trading_day or not session.contains(now):
            self.status, self.detail = "waiting", "Outside the configured exchange session"
            return
        if self.calendar.previous_trading_day(today) != self.as_of:
            raise ValueError("Stale signal snapshot: rebuild through the previous trading session")
        ids = sorted(
            {c.instrument_id for c in self.candidates}
            | {p.instrument_id for p in self.orchestrator.position_tracker.current_positions()}
        )
        self.market.refresh(ids, dt.datetime.now(dt.UTC))
        self.last_quote_at = min(q.as_of for q in self.market.quotes.values()).isoformat()
        self.ticks.append(
            {
                "hhmm": now.astimezone(IST).strftime("%H:%M:%S"),
                "px": {k: float(q.last_price) for k, q in self.market.quotes.items()},
            }
        )
        self.ticks = self.ticks[-840:]
        day_key = today.isoformat()
        completed = [dt.date.fromisoformat(d) for d in self.metadata["completed_days"]]
        if day_key in self.metadata["completed_days"]:
            pass  # rebalanced earlier today; keep working its resting orders
        elif not rebalance_due(self.settings.bot.rebalance, self.calendar, today, completed):
            self.status, self.detail = "holding", "Not a rebalance session; positions held"
        else:
            report = self.orchestrator.run_daily_cycle(self.as_of)
            self.detail = " | ".join(report.messages[-4:])
            if report.permit_trading:
                self.metadata["completed_days"].append(day_key)
            self.status = report.state.value
        self.broker.process_resting_orders()
        self.orchestrator.fill_tracker.poll(self.orchestrator.broker)
        self.orchestrator._monitor_and_reconcile_once()
        self.apply_stops(now)
        self.persist()

    def apply_stops(self, now: dt.datetime) -> None:
        orch = self.orchestrator
        if orch.circuit_breaker.current_status().state is CircuitState.HALTED:
            return
        for position in orch.position_tracker.current_positions():
            quote = self.market.get_quote(position.instrument_id)
            price = float(quote.last_price)
            key = f"{now.astimezone(IST).date()}:{position.instrument_id}"
            extremes = self.metadata["extremes"].setdefault(key, {"high": price, "low": price})
            extremes["high"] = max(extremes["high"], price)
            extremes["low"] = min(extremes["low"], price)

            def proceeds(
                mark: float, p: Any = position, spread: float = float(quote.spread_bps)
            ) -> float:
                return self.validator.engine.cost_model.estimate_execution_cost(
                    p.instrument_id,
                    TradeSide.SELL,
                    p.quantity,
                    mark,
                    now.astimezone(IST).date(),
                    spread_bps=spread,
                    avg_daily_value=0,
                    volatility=0,
                ).net_value

            breach = evaluate(
                position.instrument_id,
                entry_price=position.avg_price,
                cost_basis=position.avg_price * position.quantity,
                session_high=extremes["high"],
                session_low=extremes["low"],
                reference_price=price,
                net_sale_value=proceeds,
                policy=self.policy,
            )
            if breach is None:
                continue
            token = f"stop:{key}"
            created = orch.order_manager.create(
                position.instrument_id,
                "sell",
                position.quantity,
                "limit",
                float(quote.bid),
                token,
                signal_id=token,
                risk_decision_id=f"{token}:{breach.reason.value}",
            )
            if not created.was_duplicate:
                orch.order_manager.submit(created.order.client_order_id, orch.broker)
                orch.fill_tracker.poll(orch.broker)

    def publish(self, output: Path) -> None:
        orch = self.orchestrator
        # No quote is invented for an unobserved position. A disconnected
        # feed retains the last mark with its original timestamp.
        positions = orch.position_tracker.current_positions()
        cash = self.broker.cash
        market_value = sum(p.quantity * p.current_price for p in positions)
        equity = cash + market_value
        regime_now: dict[str, Any]
        try:
            allocation, regime = orch.regime_computer.compute_today(self.artifact, self.as_of)
            path = orch.regime_computer.recent_states(self.artifact, self.as_of, 30)
            stats = sorted(self.artifact.model.statistics, key=lambda st: st.state_id)
            regime_now = {
                "available": True,
                "as_of": str(self.as_of),
                "label": regime.label.value,
                "state_id": regime.state_id,
                "confidence": regime.confidence,
                "min_confidence": self.settings.hmm.min_confidence,
                "confident": regime.confidence >= self.settings.hmm.min_confidence,
                "probabilities": list(regime.probabilities),
                "expected_volatility": regime.expected_volatility,
                "expected_return": regime.expected_return,
                "persistence": regime.persistence,
                "state_labels": [st.label.value for st in stats],
                "model_id": self.artifact.model_id,
                "sizes_book": orch.regime_sizes_book,
                "recent": [
                    {"as_of": str(st.as_of), "label": st.label.value, "confidence": st.confidence}
                    for st in path
                ],
            }
        except Exception as exc:  # noqa: BLE001 - a report must not take the job down
            allocation = orch.last_allocation_target or full_exposure_target(
                self.as_of, orch.regime_computer.regime_policy, 0.0
            )
            regime_now = {"available": False, "reason": f"regime not computed: {exc}"}
        payload = {
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "paper_execution": {
                "mode": "orchestrated",
                "status": self.status,
                "detail": self.detail,
                "signal_date": str(self.as_of),
                "allocation_target": allocation.target_gross_exposure,
                "allow_new_positions": allocation.allow_new_positions,
            },
            "selection": {
                "available": True,
                "as_of": str(self.as_of),
                "picks": [
                    {
                        "instrument_id": c.instrument_id,
                        "symbol": c.symbol,
                        "score": c.score,
                        "rank": c.rank,
                    }
                    for c in self.candidates
                ],
            },
            "regime_now": regime_now,
            "hmm": {
                "states": [
                    {
                        "state_id": s.state_id,
                        "label": s.label.value,
                        "expected_return": s.expected_return,
                        "expected_volatility": s.expected_volatility,
                        "expected_duration": s.expected_duration,
                        "occupancy": s.occupancy,
                    }
                    for s in sorted(self.artifact.model.statistics, key=lambda st: st.state_id)
                ]
            },
            "live_book": {
                "available": True,
                "hypothetical": False,
                "fetched_at": self.last_quote_at,
                "budget": self.budget,
                "cash": cash,
                "equity": equity,
                "equity_pct": equity / self.budget - 1,
                "market_value": market_value,
                "pnl": sum(p.unrealized_pnl for p in positions),
                "positions": [
                    {
                        "symbol": p.instrument_id.removeprefix("NSE:"),
                        "shares": p.quantity,
                        "entry": p.avg_price,
                        "last": p.current_price,
                        "value": p.quantity * p.current_price,
                        "pnl": p.unrealized_pnl,
                        "day_pct": self.market.rows.get(p.instrument_id, {}).get("day_pct"),
                        "stop": {
                            "hard_level": p.avg_price * (1 - self.policy.hard_stop_pct),
                            "hard_distance_pct": p.current_price
                            / (p.avg_price * (1 - self.policy.hard_stop_pct))
                            - 1,
                        },
                    }
                    for p in positions
                ],
            },
            "ticks": {"updated_at": self.last_quote_at, "points": self.ticks},
            "realized": {"available": False, "reason": "See durable fill ledger for costs"},
        }
        atomic_write(output, json.dumps(payload, allow_nan=False, default=str))
        atomic_write(self.state_dir / "status.json", json.dumps(payload["paper_execution"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--state-dir", type=Path, default=REPO_ROOT / "state/paper_live")
    parser.add_argument("--budget", type=float, default=100000)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    runtime = PaperRuntime(args.as_of, args.state_dir, args.budget)
    output = REPO_ROOT / "state/dashboard_data.json"
    runtime.publish(output)
    if args.prepare:
        print("Prepared paper account and model; no orders submitted", flush=True)
        return
    while True:
        try:
            runtime.tick(dt.datetime.now(dt.UTC))
        except Exception as exc:
            runtime.status, runtime.detail = "blocked", f"{type(exc).__name__}: {exc}"
            print(runtime.detail, flush=True)
        runtime.publish(output)
        if args.once:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
