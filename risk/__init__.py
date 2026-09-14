"""Independent risk management with veto authority over every other layer.

NON-NEGOTIABLE (docs/SPECIFICATION.md section 8): a valid signal from
regime/selection/portfolio construction can still be rejected or scaled down
for exposure, liquidity, drawdown, stale data, instrument status, pending
orders, broker state, or compliance reasons. Nothing upstream of this
package may bypass it -- ``risk`` must not import from ``core.regime``,
``universe``, or ``portfolio`` for anything other than the proposed
weights/candidates it is asked to evaluate.
"""
