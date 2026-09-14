"""Portfolio construction: turns ranked candidates and a regime exposure
target into proposed target weights. Proposals from this layer are not
final -- risk/risk_manager.py always has veto authority over the output.
See docs/SPECIFICATION.md section 7.2.
"""
