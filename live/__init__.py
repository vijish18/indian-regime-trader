"""Phase 22: the live-trading safety gate.

Nothing in this package enables live order submission -- it exists to
make sure nothing *else* can, casually. ``live.kill_switch.KillSwitch``
gives an operator an explicit, immediate way to stop trading;
``live.preflight`` runs (and reports on) the formal pre-live checklist in
``docs/PRE_LIVE_CHECKLIST.md``. ``broker.factory.build_broker`` requires a
fresh, passing preflight result -- not a remembered one -- as one of its
own independent confirmations before it will construct a live-capable
broker at all.
"""
