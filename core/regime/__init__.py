"""Market-regime detection: the HMM engine, regime-to-exposure policy, and
model artifact registry. Operates only on market-level features (NIFTY 50,
India VIX, breadth) -- never on individual stock series. See
docs/SPECIFICATION.md section 6 and docs/ARCHITECTURE.md.
"""
