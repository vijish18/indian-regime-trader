"""Phase 21: end-to-end paper-trading validation.

This package runs the whole system in paper mode -- from ingesting vendor
files through to shutdown, restart and reconciliation -- and reports what
held and what did not. It exists so that "the system works end to end" is
a claim backed by a run, with a written report, rather than by the fact
that every layer's own unit tests pass.

Nothing here reaches a live account: the only broker it constructs is
``broker.adapters.paper_broker.PaperBroker``, and no credentials are read
or required.
"""
