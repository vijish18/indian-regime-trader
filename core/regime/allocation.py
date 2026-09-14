"""Regime-aware portfolio allocation: the HMM's answer to "how much risk"
turned into a gross-exposure target.

This is the one arrow from docs/SPECIFICATION.md's brief, made concrete:

    RegimeState (expected_volatility, confidence)
        -> AllocationRegime (LOW_RISK / NORMAL_RISK / HIGH_RISK / UNCERTAIN)
        -> AllocationTarget (a gross-exposure band and a point target within it)

It must never choose individual stocks -- it has no access to any security's
price, score, or candidacy, only the market-level regime. Security selection
lives entirely in ``universe/stock_selector.py``, which this module has no
dependency on.

## The classification is not the HMM's own regime label

``core/regime/hmm_engine.py`` already assigns ``RegimeLabel`` (calm, normal,
elevated, crisis) by ranking a fitted model's states against *each other* on
measured volatility -- a relative, per-model, reporting-only label. The
``AllocationRegime`` this module computes is deliberately a different thing:
an *absolute*, confidence-aware classification, driven by two configured
volatility thresholds (``config.allocation.low_risk_volatility_threshold``,
``high_risk_volatility_threshold``) that mean the same thing across every
retrain, plus a fourth category -- UNCERTAIN -- that has no volatility-level
equivalent at all, because it is not about how risky the market looks; it is
about whether the *classification itself* should be trusted right now.

``RegimeAllocationEngine`` therefore never reads ``RegimeState.label``. Every
decision here comes from ``expected_volatility`` and ``confidence``, exactly
as docs/SPECIFICATION.md's brief requires ("based primarily on estimated
regime volatility and confidence") and exactly the property Phase 5 already
established for the HMM engine itself: regime *names* must never determine
behavior.

## Confirmation and flicker, not just a single reading

A single filtered observation can be a noisy blip. So this module does not
act on the latest ``RegimeState`` in isolation -- ``evaluate`` takes a
trailing *history* of states and:

1. Classifies every state in the history into a raw volatility tier.
2. Requires ``hmm.confirmation_bars`` consecutive observations agreeing on a
   tier before treating it as confirmed (or a single observation at
   ``allocation.extreme_confidence_threshold`` or above --
   docs/SPECIFICATION.md section 6, "2 consecutive observations unless
   confidence is extreme"). While a candidate transition is unconfirmed, the
   *previous* confirmed tier is held rather than acted on early.
3. Counts confirmed-tier changes within the trailing
   ``hmm.flicker_window_sessions``; too many forces UNCERTAIN regardless of
   what the latest reading says, because a regime that keeps flipping is not
   a regime anyone should be sizing a portfolio against.
4. Falls back to UNCERTAIN outright if the latest confidence is below
   ``hmm.min_confidence``, or if no tier has ever been confirmed yet (not
   enough history).

Within whichever tier survives all of that, the final exposure target is a
linear function of confidence across the tier's configured band -- not a
single fixed percentage. No leverage is possible by construction: every band
is a pydantic ``Percent`` field bounded to ``[0, 1]``
(``config/models.py::ExposureBand``), and ``AllocationTarget`` itself
re-validates the same bound as defense in depth.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from config.models import AllocationConfig, HMMConfig
from core.regime.hmm_engine import RegimeState

if TYPE_CHECKING:
    # Deferred to break the import cycle: regime_policy.py imports
    # AllocationRegime from this module, so this module cannot import
    # RegimePolicy at runtime. `from __future__ import annotations` (above)
    # means this is fine -- the annotation below is never evaluated.
    from core.regime.regime_policy import RegimePolicy


class AllocationRegime(StrEnum):
    """The allocation-facing risk tier -- what ``RegimeAllocationEngine``
    actually acts on. Deliberately not the HMM's own ``RegimeLabel``; see
    this module's docstring.
    """

    LOW_RISK = "low_risk"
    NORMAL_RISK = "normal_risk"
    HIGH_RISK = "high_risk"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class AllocationTarget:
    """A gross-exposure decision for one session: a band (for reference and
    for downstream risk checks) and a single point target within it.
    """

    as_of: dt.date
    regime: AllocationRegime
    target_gross_exposure: float
    min_gross_exposure: float
    max_gross_exposure: float
    allow_new_positions: bool
    """False only for UNCERTAIN: existing positions may still be trimmed
    toward the band, but no new entries should be opened."""

    confidence: float
    expected_volatility: float
    reason: str
    """Human-readable audit trail: which gate produced this target
    (e.g. "confirmed", "low confidence", "flickering", "unconfirmed")."""

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_gross_exposure <= self.max_gross_exposure <= 1.0):
            raise ValueError(
                f"exposure band [{self.min_gross_exposure}, {self.max_gross_exposure}] "
                "must satisfy 0 <= min <= max <= 1 (no leverage)"
            )
        if not (self.min_gross_exposure <= self.target_gross_exposure <= self.max_gross_exposure):
            raise ValueError(
                f"target_gross_exposure {self.target_gross_exposure} is outside the band "
                f"[{self.min_gross_exposure}, {self.max_gross_exposure}]"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence {self.confidence} must be within [0, 1]")
        if self.expected_volatility < 0:
            raise ValueError(f"expected_volatility {self.expected_volatility} must be >= 0")


def volatility_tier(expected_volatility: float, config: AllocationConfig) -> AllocationRegime:
    """Classify a single annualized volatility number into a risk tier.

    Pure function of the number and the configured thresholds -- never of a
    label, a state ID, or anything else. Never returns UNCERTAIN: that
    category is about trust in the classification, which this function has
    no way to assess from a bare number.
    """
    if expected_volatility < 0:
        raise ValueError(f"expected_volatility {expected_volatility} must be >= 0")
    if expected_volatility < config.low_risk_volatility_threshold:
        return AllocationRegime.LOW_RISK
    if expected_volatility >= config.high_risk_volatility_threshold:
        return AllocationRegime.HIGH_RISK
    return AllocationRegime.NORMAL_RISK


def confirmed_tier_sequence(
    raw_tiers: list[AllocationRegime],
    confidences: list[float],
    confirmation_bars: int,
    extreme_confidence_threshold: float,
) -> list[AllocationRegime | None]:
    """For each position, the most recently *confirmed* tier as of that
    point -- ``None`` before anything has ever been confirmed.

    A candidate tier confirms once it has been observed for
    ``confirmation_bars`` consecutive positions, or immediately if a single
    observation's confidence is at or above ``extreme_confidence_threshold``.
    Until a new candidate confirms, the previous confirmed tier is carried
    forward -- an unconfirmed transition does not change the answer.

    Pure function of parallel arrays (not ``RegimeState``), so it is directly
    testable without constructing a fitted model.
    """
    if len(raw_tiers) != len(confidences):
        raise ValueError(
            f"raw_tiers has {len(raw_tiers)} entries but confidences has {len(confidences)}"
        )
    if confirmation_bars < 1:
        raise ValueError(f"confirmation_bars must be >= 1, got {confirmation_bars}")

    confirmed: list[AllocationRegime | None] = []
    last_confirmed: AllocationRegime | None = None
    pending_tier: AllocationRegime | None = None
    pending_count = 0

    for tier, confidence in zip(raw_tiers, confidences, strict=True):
        if confidence >= extreme_confidence_threshold:
            last_confirmed = tier
            pending_tier = tier
            pending_count = confirmation_bars
        else:
            if tier == pending_tier:
                pending_count += 1
            else:
                pending_tier = tier
                pending_count = 1
            if pending_count >= confirmation_bars:
                last_confirmed = tier
        confirmed.append(last_confirmed)

    return confirmed


def count_transitions(tiers: list[AllocationRegime | None]) -> int:
    """Number of times consecutive non-``None`` entries differ. ``None``
    entries (not-yet-confirmed history) are skipped rather than counted as a
    transition into or out of "unknown".
    """
    known = [tier for tier in tiers if tier is not None]
    return sum(1 for earlier, later in zip(known, known[1:], strict=False) if earlier != later)


def _confidence_scale(confidence: float, min_confidence: float) -> float:
    """Map ``confidence`` linearly onto ``[0, 1]`` using ``min_confidence``
    as the zero point and full confidence (1.0) as the ceiling. Clipped, so a
    confidence at or below ``min_confidence`` never produces a negative
    scale and one at or above 1.0 never exceeds 1.
    """
    if min_confidence >= 1.0:
        return 1.0 if confidence >= min_confidence else 0.0
    scale = (confidence - min_confidence) / (1.0 - min_confidence)
    return min(max(scale, 0.0), 1.0)


class RegimeAllocationEngine:
    """Turns a trailing history of filtered ``RegimeState`` observations into
    an ``AllocationTarget`` -- confidence-gated, confirmed, and
    flicker-checked before a single stock-selection or risk module ever sees
    it.
    """

    def __init__(
        self,
        hmm_config: HMMConfig,
        allocation_config: AllocationConfig,
        policy: RegimePolicy,
    ) -> None:
        self.hmm_config = hmm_config
        self.allocation_config = allocation_config
        self.policy = policy

    def evaluate(self, history: list[RegimeState]) -> AllocationTarget:
        """The allocation for the most recent state in ``history``.

        ``history`` must be ascending by ``as_of`` and end at the session
        being decided for; it should cover at least
        ``max(hmm.confirmation_bars, hmm.flicker_window_sessions)`` sessions
        for the confirmation and flicker checks to have anything to work
        with -- a shorter history degrades gracefully to UNCERTAIN rather
        than raising, since "not enough history yet" is itself a valid,
        conservative answer.

        Raises:
            ValueError: if ``history`` is empty.
        """
        if not history:
            raise ValueError("cannot evaluate an allocation from empty regime history")

        latest = history[-1]
        raw_tiers = [
            volatility_tier(state.expected_volatility, self.allocation_config)
            for state in history
        ]
        confidences = [state.confidence for state in history]
        confirmed = confirmed_tier_sequence(
            raw_tiers,
            confidences,
            self.hmm_config.confirmation_bars,
            self.allocation_config.extreme_confidence_threshold,
        )

        if latest.confidence < self.hmm_config.min_confidence:
            return self._build(
                AllocationRegime.UNCERTAIN,
                latest,
                reason=(
                    f"confidence {latest.confidence:.3f} is below the minimum "
                    f"{self.hmm_config.min_confidence:.3f}"
                ),
            )

        latest_confirmed = confirmed[-1]
        if latest_confirmed is None:
            return self._build(
                AllocationRegime.UNCERTAIN,
                latest,
                reason="not enough history to confirm a regime yet",
            )

        window = confirmed[-self.hmm_config.flicker_window_sessions :]
        transitions = count_transitions(window)
        if transitions > self.allocation_config.max_flicker_transitions:
            return self._build(
                AllocationRegime.UNCERTAIN,
                latest,
                reason=(
                    f"{transitions} confirmed-regime changes in the trailing "
                    f"{len(window)} sessions exceeds the flicker limit of "
                    f"{self.allocation_config.max_flicker_transitions}"
                ),
            )

        return self._build(latest_confirmed, latest, reason="confirmed")

    def _build(
        self, regime: AllocationRegime, state: RegimeState, reason: str
    ) -> AllocationTarget:
        band = self.policy.band_for(regime)
        scale = _confidence_scale(state.confidence, self.hmm_config.min_confidence)
        raw_target = band.min_gross_exposure + (
            band.max_gross_exposure - band.min_gross_exposure
        ) * scale
        # Clamped defensively: `scale` is mathematically within [0, 1], but
        # floating-point arithmetic can still land a hair outside the band
        # (e.g. 0.45000000000000007), which AllocationTarget's own bound
        # check would otherwise reject.
        target = min(max(raw_target, band.min_gross_exposure), band.max_gross_exposure)
        return AllocationTarget(
            as_of=state.as_of,
            regime=regime,
            target_gross_exposure=target,
            min_gross_exposure=band.min_gross_exposure,
            max_gross_exposure=band.max_gross_exposure,
            allow_new_positions=regime is not AllocationRegime.UNCERTAIN,
            confidence=state.confidence,
            expected_volatility=state.expected_volatility,
            reason=reason,
        )
