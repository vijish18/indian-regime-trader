"""Maps a classified market regime to a portfolio gross-exposure band.

This is the only module allowed to translate "what state is the market in"
into "how much risk should the portfolio take". It must not see individual
stock candidates or scores (that is universe/stock_selector.py's job) and its
output is a target band, not a final weight -- risk/risk_manager.py always
has the last word. See docs/SPECIFICATION.md section 7.

Not implemented yet (Phase 6).
"""

from __future__ import annotations

from dataclasses import dataclass

from config.models import RegimePolicyConfig
from core.regime.hmm_engine import RegimeLabel


@dataclass(frozen=True)
class ExposureTarget:
    """A gross-exposure band proposed for the current regime. Not yet a
    final target -- risk/risk_manager.py may narrow this further.
    """

    label: RegimeLabel
    min_gross_exposure: float
    max_gross_exposure: float
    confidence: float


class RegimePolicy:
    """Translates a ``RegimeState`` into an ``ExposureTarget`` using the
    configured exposure bands per regime label.
    """

    def __init__(self, config: RegimePolicyConfig) -> None:
        self.config = config

    def exposure_for(self, label: RegimeLabel, confidence: float) -> ExposureTarget:
        """Return the configured exposure band for ``label``.

        Implementations must fall back to the most conservative band
        (crisis) when confidence is below the configured minimum or the
        regime is unconfirmed/flickering (docs/SPECIFICATION.md section 6),
        rather than defaulting to an optimistic exposure.
        """
        raise NotImplementedError("Phase 6: regime policy is not implemented yet.")
