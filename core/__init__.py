"""Core market-understanding layer: regime detection and feature engineering.

``core`` must never import from ``universe``, ``portfolio``, ``risk``,
``execution``, or ``broker``. It answers "what is the state of the market?",
not "what should we own?" or "how much should we risk?" -- keeping that
dependency direction one-way is what stops the regime model from turning into
a stock-selection or execution component by accident.
"""
