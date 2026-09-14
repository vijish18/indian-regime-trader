"""Maps a classified market regime to a portfolio gross-exposure band.

This is the only module allowed to translate "what state is the market in"
into "how much risk should the portfolio take". It must not see individual
stock candidates or scores (that is universe/stock_selector.py's job) and its
output is a target band, not a final weight -- risk/risk_manager.py always
has the last word. See docs/SPECIFICATION.md section 7.

**Behavior is driven by measured statistics, not by regime names.**
``exposure_for`` takes a whole ``RegimeState``, which carries the fitted
state's measured ``expected_volatility``, ``expected_return`` and
``persistence`` alongside its label. The label is a reporting convenience: it
is assigned by ranking states on measured volatility, and renaming it must
not change a single order. An implementation that switched on
``state.label`` would make a dashboard string load-bearing, and would break
silently the moment a refit produced a different number of states.

Not implemented yet (Phase 6).
"""

from __future__ import annotations

from dataclasses import dataclass

from config.models import RegimePolicyConfig
from core.regime.hmm_engine import RegimeLabel, RegimeState, StateStatistics


@dataclass(frozen=True)
class ExposureTarget:
    """A gross-exposure band proposed for the current regime. Not yet a
    final target -- risk/risk_manager.py may narrow this further.
    """

    label: RegimeLabel
    min_gross_exposure: float
    max_gross_exposure: float
    confidence: float
    driving_volatility: float
    """The measured expected volatility this target was derived from, so an
    exposure decision can be audited against the number that produced it
    rather than against a name."""


class RegimePolicy:
    """Translates a ``RegimeState`` into an ``ExposureTarget`` using the
    configured exposure bands.
    """

    def __init__(self, config: RegimePolicyConfig) -> None:
        self.config = config

    def exposure_for(
        self, state: RegimeState, all_states: tuple[StateStatistics, ...]
    ) -> ExposureTarget:
        """Return the exposure band implied by ``state``'s measured risk.

        ``all_states`` provides the fitted model's full set of state
        statistics so the implementation can place this state's measured
        volatility *relative* to the others, rather than comparing it to a
        hardcoded absolute threshold that would silently mean something
        different after a retrain.

        Implementations must fall back to the most conservative band when
        confidence is below the configured minimum or the regime is
        unconfirmed/flickering (docs/SPECIFICATION.md section 6), rather than
        defaulting to an optimistic exposure.
        """
        raise NotImplementedError("Phase 6: regime policy is not implemented yet.")
