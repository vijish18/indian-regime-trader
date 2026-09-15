"""Phase 19: the application lifecycle and daily workflow that wires every
other layer together. This package contains no strategy mathematics of its
own -- every decision (regime, selection, portfolio, risk, sizing) is
delegated to the dedicated module that already owns it; ``orchestrator.py``
only sequences calls and reacts to their results. See
docs/ARCHITECTURE.md's "Application lifecycle and daily workflow" section.
"""
