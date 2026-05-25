"""
core/creep.py — Larson-Miller creep evaluation.

Phase 2 of the lifecycle / creep / fatigue rollout. This module
answers a single engineering question:

    Given material M held at wall temperature T for t hours, will its
    rupture stress at that combined exposure exceed the required stress
    sigma_required?

The math is the standard Larson-Miller parameter (LMP):

    LMP = T * (C + log10(t))

where T is in kelvin, t is in hours, and C is a material-specific
constant in the 13-25 range (lower for aluminium, higher for nickel
superalloys). Each material's LMP curve in ``materials_db.py`` is a
piecewise-linear (LMP, rupture_stress_MPa) interpolation table; this
module looks up the rupture stress at the queried (T, t) point and
reports a margin against the required stress.

Outputs match the conventions in ``matching_engine.py``: a four-stage
status string (``pass`` / ``marginal`` / ``fail`` / ``unknown`` /
``not_applicable``) plus a margin fraction for the score function.

Cross-cutting rules
-------------------

* **No physics-engine modifications.** This module consumes
  ``MaterialEntry`` directly; it does not touch the rest of the
  pipeline. Phase 3 wires it into ``matching_engine.py``.
* **Margin convention matches the structural stage.** A material is
  "pass" only when ``creep_margin_fraction >= CREEP_MARGIN_FRACTION``
  (0.20, mirroring ``MARGINAL_STRUCTURAL_FRACTION``).
* **Audience: small-rocket / university / startup.** Edge cases are
  flagged in plain English on the ``CreepEvaluation.notes`` field so
  a non-creep-specialist reading the report knows what was extrapolated
  vs. interpolated.

References
----------
Larson, F. R. and Miller, J. (1952). "A time-temperature relationship
for rupture and creep stresses." Trans. ASME 74, 765-775.

Reed, R. C. (2006). The Superalloys: Fundamentals and Applications.
Cambridge University Press. Chapter 6 covers LMP fitting and the
typical C values for nickel-superalloy classes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .materials_db import MaterialEntry


# ---------------------------------------------------------------------------
# Module constants — mirror the conventions used in matching_engine.py.
# ---------------------------------------------------------------------------

CREEP_MARGIN_FRACTION = 0.20
"""Threshold between ``pass`` and ``marginal`` for the creep-margin
fraction. Mirrors ``matching_engine.MARGINAL_STRUCTURAL_FRACTION``."""

CREEP_NOT_APPLICABLE_T_FRAC = 0.5
"""Fraction of melting point below which classical creep is
negligible. Materials with ``creep_data_status="unknown"`` AND
``T_K / melting_point_K < CREEP_NOT_APPLICABLE_T_FRAC`` get treated
as ``not_applicable`` rather than ``unknown`` — there is no physically
meaningful creep at that homologous temperature so the unknown flag
would be misleading."""


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class CreepEvaluation:
    """Verdict of evaluating one material at one (T, t, sigma_required) point.

    ``status`` semantics mirror the existing matching-engine stages:

    * ``"pass"``           — margin_fraction >= CREEP_MARGIN_FRACTION
    * ``"marginal"``       — 0 <= margin_fraction < CREEP_MARGIN_FRACTION
    * ``"fail"``           — margin_fraction < 0 (rupture stress
                             below sigma_required)
    * ``"not_applicable"`` — material does not classically creep at
                             relevant temperatures (TPS ablators,
                             polymer composites, ceramics below
                             ~0.5*Tm, etc.). Treated as a pass-through
                             in the matching engine.
    * ``"unknown"``        — no LMP data and no category rule applies.
                             Surfaced as a flag in the UI; doesn't
                             auto-reject.
    """
    status: str
    rupture_stress_MPa: float | None = None
    margin_fraction: float | None = None
    lmp_value: float | None = None
    data_source: str = ""
    notes: str = ""
    extrapolated: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def larson_miller_parameter(T_K: float, t_hours: float, C: float) -> float:
    """Compute LMP = T * (C + log10(t)).

    T is in kelvin; t is in hours (must be > 0); C is the
    material-specific Larson-Miller constant.

    Raises ``ValueError`` for non-positive temperatures or times — both
    are physical impossibilities that should fail loudly rather than
    silently produce log10(0) = -inf.
    """
    if T_K <= 0.0:
        raise ValueError(f"T_K must be > 0, got {T_K}")
    if t_hours <= 0.0:
        raise ValueError(f"t_hours must be > 0, got {t_hours}")
    return T_K * (C + math.log10(t_hours))


def creep_rupture_strength_MPa(
    material: MaterialEntry,
    T_K: float,
    t_hours: float,
) -> tuple[float | None, bool]:
    """Look up the rupture stress for ``material`` at (T, t).

    Returns ``(stress_MPa, extrapolated)`` where ``extrapolated`` is
    ``True`` when the queried LMP fell outside the curve range and a
    log-stress linear extrapolation was used.

    Returns ``(None, False)`` for materials with status ``"unknown"``
    or ``"not_applicable"`` (no curve to query).
    """
    if material.creep_data_status in ("unknown", "not_applicable"):
        return (None, False)
    if not material.lmp_curve or material.larson_miller_C is None:
        return (None, False)

    lmp = larson_miller_parameter(T_K, t_hours, material.larson_miller_C)
    curve = material.lmp_curve

    # Below the curve range: use the lowest-LMP point's stress
    # directly (extrapolating to a *cooler / shorter* condition is
    # almost always safe — the material is at least as strong as the
    # first data point. Flag as extrapolated for transparency.)
    if lmp <= curve[0][0]:
        return (curve[0][1], True if lmp < curve[0][0] else False)

    # Above the curve range: linear extrapolation in log10(stress)
    # vs. LMP, using the last two points. Flag as extrapolated.
    if lmp >= curve[-1][0]:
        if len(curve) >= 2:
            (lmp_a, s_a), (lmp_b, s_b) = curve[-2], curve[-1]
            # Avoid log(0); clamp to a tiny positive floor before log.
            log_s_a = math.log10(max(s_a, 1e-3))
            log_s_b = math.log10(max(s_b, 1e-3))
            slope = (log_s_b - log_s_a) / (lmp_b - lmp_a)
            log_s = log_s_b + slope * (lmp - lmp_b)
            return (max(10 ** log_s, 0.0), True)
        return (curve[-1][1], True)

    # In-range: piecewise-linear interpolation in σ space (engineering
    # convention; log-σ interpolation is an option but produces
    # essentially the same answer over the small intervals here).
    for i in range(1, len(curve)):
        lmp_a, s_a = curve[i - 1]
        lmp_b, s_b = curve[i]
        if lmp_a <= lmp <= lmp_b:
            f = (lmp - lmp_a) / (lmp_b - lmp_a) if lmp_b != lmp_a else 0.0
            return (s_a + f * (s_b - s_a), False)

    # Should be unreachable given the bracket checks above.
    return (curve[-1][1], True)


def evaluate_creep(
    material: MaterialEntry,
    T_K: float,
    t_hours: float,
    sigma_required_MPa: float,
) -> CreepEvaluation:
    """Full creep verdict for a material at a (T, t, sigma_required) point.

    Decision tree:

    1. ``t_hours <= 0`` → raises ``ValueError``. Caller should clamp
       to a positive lifetime before calling.
    2. ``creep_data_status == "not_applicable"`` → return
       ``status="not_applicable"`` (TPS, ceramics, polymers).
    3. ``creep_data_status == "unknown"`` AND
       ``T_K / melting_point_K < CREEP_NOT_APPLICABLE_T_FRAC`` →
       return ``status="not_applicable"`` with note "below creep
       regime". Avoids spuriously flagging refractory metals or
       cobalt alloys that simply aren't hot enough to creep.
    4. ``creep_data_status == "unknown"`` AND in creep regime → return
       ``status="unknown"`` with a warning. Don't auto-reject.
    5. Sourced or estimated curve → look up rupture stress, compute
       margin, return pass / marginal / fail.
    """
    if t_hours <= 0.0:
        raise ValueError(f"t_hours must be > 0, got {t_hours}")
    if T_K <= 0.0:
        raise ValueError(f"T_K must be > 0, got {T_K}")

    # Category-rule materials: TPS, polymers, ceramics, UHTC, carbon.
    if material.creep_data_status == "not_applicable":
        return CreepEvaluation(
            status="not_applicable",
            data_source=material.creep_data_source,
            notes=(
                "Material category does not classically creep at "
                "relevant service temperatures."
            ),
        )

    # Unknown materials with no LMP curve. Decide whether the
    # operative T even reaches the creep regime.
    if material.creep_data_status == "unknown":
        T_melt = float(material.melting_point_K)
        if T_melt > 0.0 and (T_K / T_melt) < CREEP_NOT_APPLICABLE_T_FRAC:
            return CreepEvaluation(
                status="not_applicable",
                notes=(
                    f"No LMP data sourced for this material, but "
                    f"T_wall = {T_K:.0f} K is only "
                    f"{100.0 * T_K / T_melt:.0f}% of the melting point "
                    f"({T_melt:.0f} K) — below the classical creep "
                    f"regime threshold ({CREEP_NOT_APPLICABLE_T_FRAC * 100:.0f}%)."
                ),
            )
        return CreepEvaluation(
            status="unknown",
            notes=(
                "No Larson-Miller data sourced for this material; "
                "creep behaviour at the queried (T, t) point cannot "
                "be verified. Treat with caution for sustained-load "
                "applications."
            ),
        )

    # Sourced or estimated: full evaluation via the LMP curve.
    rupture_stress, extrapolated = creep_rupture_strength_MPa(
        material, T_K, t_hours
    )
    if rupture_stress is None:
        # Defensive: shouldn't happen because we filtered above.
        return CreepEvaluation(status="unknown")

    if material.larson_miller_C is None:
        # Defensive: same as above.
        return CreepEvaluation(status="unknown")

    lmp = larson_miller_parameter(
        T_K, t_hours, material.larson_miller_C
    )

    # Margin fraction: (rupture - required) / required. Convention
    # matches the structural stage in matching_engine.py.
    if sigma_required_MPa <= 0.0:
        # Required stress is zero or negative — physically meaningless
        # but the evaluation should still report. Treat as "infinite
        # margin" → pass.
        return CreepEvaluation(
            status="pass",
            rupture_stress_MPa=rupture_stress,
            margin_fraction=float("inf"),
            lmp_value=lmp,
            data_source=material.creep_data_source,
            notes=(
                "sigma_required <= 0 (physically meaningless); "
                "treating as unconstrained pass."
            ),
            extrapolated=extrapolated,
        )

    margin = (rupture_stress - sigma_required_MPa) / sigma_required_MPa

    if margin < 0.0:
        status = "fail"
    elif margin < CREEP_MARGIN_FRACTION:
        status = "marginal"
    else:
        status = "pass"

    notes_parts: list[str] = []
    if material.creep_data_status == "estimated":
        notes_parts.append(
            "LMP curve is engineering-estimated, not from a primary "
            "reference; treat the margin as indicative."
        )
    if extrapolated:
        notes_parts.append(
            "LMP value falls outside the sourced curve range — "
            "stress was log-linearly extrapolated."
        )

    return CreepEvaluation(
        status=status,
        rupture_stress_MPa=rupture_stress,
        margin_fraction=margin,
        lmp_value=lmp,
        data_source=material.creep_data_source,
        notes=" ".join(notes_parts),
        extrapolated=extrapolated,
    )
