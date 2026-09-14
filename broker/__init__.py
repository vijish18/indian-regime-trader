"""Broker-neutral execution interface. Strategy, risk, and execution code
depend only on ``broker.base.Broker`` -- never on a concrete adapter --
so a broker can be swapped without touching upstream layers.
See docs/SPECIFICATION.md section 12.
"""
