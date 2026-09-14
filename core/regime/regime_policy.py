"""Maps an allocation-tier classification to a configured gross-exposure band.

This is deliberately the smallest possible piece: a pure lookup from
``AllocationRegime`` (``core/regime/allocation.py`` -- LOW_RISK, NORMAL_RISK,
HIGH_RISK, UNCERTAIN) to the band configured for it in
``config.settings.yaml``'s ``regime_policy`` section. It holds no state, does
not see raw ``RegimeState`` objects, and makes no decision about *which* tier
applies -- that classification (confidence handling, confirmation, flicker,
volatility thresholds) is ``RegimeAllocationEngine``'s job
(``core/regime/allocation.py``). Keeping the two separate means the exposure
numbers can be changed in config without touching any decision logic, and the
decision logic can be tested without caring what the configured numbers are.

The bands are intentionally not the HMM's own calm/normal/elevated/crisis
regime labels (``core/regime/hmm_engine.py::RegimeLabel``) -- see
docs/ARCHITECTURE.md, "Resolved specification ambiguities" (B6), for why
these are two deliberately separate vocabularies: the HMM's labels are a
*relative* ranking of states within one fitted model, purely for reporting;
these four tiers are an *absolute*, confidence-aware classification that
actually governs exposure. Confusing the two would make a dashboard string
load-bearing again, exactly what Phase 5 exists to prevent.
"""

from __future__ import annotations

from config.models import ExposureBand, RegimePolicyConfig
from core.regime.allocation import AllocationRegime


class RegimePolicy:
    """Looks up the configured exposure band for an allocation tier."""

    def __init__(self, config: RegimePolicyConfig) -> None:
        self.config = config

    def band_for(self, regime: AllocationRegime) -> ExposureBand:
        """The configured ``[min, max]`` gross-exposure band for ``regime``."""
        match regime:
            case AllocationRegime.LOW_RISK:
                return self.config.low_risk
            case AllocationRegime.NORMAL_RISK:
                return self.config.normal_risk
            case AllocationRegime.HIGH_RISK:
                return self.config.high_risk
            case AllocationRegime.UNCERTAIN:
                return self.config.uncertain
