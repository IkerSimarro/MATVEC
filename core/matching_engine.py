"""
MATVEC Matching Engine — Step 3
Applies a three-stage filter+ranking pipeline to the 56-entry materials database
given a PhysicsResult from the physics engine.

Pipeline:
  Stage 1: Regime filter — discard materials not rated for the flight regime
  Stage 2: Thermal filter — service_temp_air_K and oxidation_max_temp_K vs T_wall
  Stage 3: Structural filter — temperature-adjusted strength vs sigma_tensile_required

Output: MatchResult with viable / marginal / not_viable / regime_rejected lists,
all ranked by composite score. Feeds directly into the Step 4 Streamlit UI.

Dependencies: materials_db, physics_engine (project modules); dataclasses (stdlib).
"""

__version__ = "1.0.0"

import math
from dataclasses import dataclass, field

from .materials_db import (
    MATERIALS_DB,
    MaterialEntry,
    get_materials_by_regime,
    get_strength_at_temperature,
)
from .physics_engine import PhysicsResult, E_REF_MPA, ALPHA_REF, THERMAL_RELIEF_FACTOR
from .creep import (
    CREEP_MARGIN_FRACTION,
    evaluate_creep,
)
from .transient_heat import integrate_panel


# ---------------------------------------------------------------------------
# Transient-heat integration thresholds
# ---------------------------------------------------------------------------
#
# When ``flight_duration_s`` is short relative to a typical aerospace
# panel's thermal time (~30-100 s for thin-skin metals), the steady-
# state T_wall is overly conservative and the 1D transient solver
# should be used instead. The threshold below 300 s captures all
# boost-coast trajectories (sounding rockets, missiles, single-mission
# reentry capsules) while leaving sustained-flight evaluations
# untouched.
#
# Above 300 s the steady-state thermal stage converges to the right
# answer anyway (surface and back-face both reach recovery
# temperature), so running the transient solver would just duplicate
# the static result and add cost.

TRANSIENT_DURATION_THRESHOLD_S = 300.0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MARGINAL_STRUCTURAL_FRACTION = 0.20   # < 20% strength margin above required → "marginal"

# Large sentinel used when T_wall ≈ 0 (no real thermal load)
_LARGE_SCORE = 1e6

# Physics-driven thresholds (these replace category-only gating)
TPS_UNLOCK_TEMP_K             = 1200.0   # T_wall above which TPS materials unlock for ANY category
                                          # (public constant — also imported by app.py and latex_export.py
                                          # so the threshold is defined exactly once)
_ABLATIVE_SUBSTRATE_T_FLOOR_K = 400.0    # Backside soak temperature floor for substrate-mode evaluation
_ABLATIVE_DENSITY_CEILING_K   = 8500.0   # Density ceiling raised to this when ablative-unlock is active

_CMC_SUPERSONIC_MACH_THRESHOLD = 5.0     # Below this Mach, CMCs are penalised in non-turbine categories
_CMC_SUPERSONIC_PENALTY        = 5.0     # Negative score adjustment for CMCs in supersonic regime

# Metal categories eligible for substructure-under-ablative second-pass evaluation
_SUBSTRATE_METAL_CATEGORIES = frozenset({"titanium", "steel", "aluminum", "nickel", "cobalt"})


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MaterialCandidate:
    material: MaterialEntry
    thermal_ceiling_K: float           # min(service_temp_air_K, oxidation_max_temp_K)
    thermal_margin_K: float            # ceiling - T_wall_K  (negative = fail)
    strength_at_T_wall_MPa: float      # get_strength_at_temperature(mat, T_wall_K)
    sigma_req_material_MPa: float      # material-specific structural requirement
    structural_margin_fraction: float  # strength / sigma_req_material - 1.0  (negative = fail)
    thermal_status: str                # "pass" / "marginal" / "fail"
    structural_status: str             # "pass" / "marginal" / "fail"
    overall_status: str                # "viable" / "marginal" / "not_viable"
    score: float                       # thermal_margin_fraction (primary ranking key)
    notes: list                        # per-material warnings/observations
    evaluation_mode: str = "direct"    # "direct" (T_wall) or "substrate" (T_soak under ablative coating)
    # ---- Creep / lifecycle fields (Phase 3 of the lifecycle rollout) ----
    # Populated by ``_evaluate_creep_for_candidate`` when ``match_materials``
    # is called with ``design_lifetime_hours``. At the default 1.0 h
    # lifetime the creep stage is a no-op for nearly all materials
    # (homologous T low or t_hours short), so most candidates carry
    # ``creep_status="not_applicable"`` or ``"pass"`` with a large margin.
    creep_status: str = "not_applicable"
    """One of: ``"pass"`` (margin >= CREEP_MARGIN_FRACTION),
    ``"marginal"`` (0 <= margin < CREEP_MARGIN_FRACTION),
    ``"fail"`` (margin < 0), ``"unknown"`` (no LMP data and material is
    in creep regime — flagged but not rejected), or
    ``"not_applicable"`` (TPS / ceramic / polymer category, OR
    short-lifetime / cool-temperature where creep is irrelevant)."""
    creep_rupture_stress_MPa: float | None = None
    creep_margin_fraction: float | None = None
    creep_lmp_value: float | None = None
    creep_data_source: str = ""
    creep_extrapolated: bool = False

    # ---- Transient-heat fields (Phase 7 of the lifecycle rollout) ----
    # Populated when ``match_materials`` is called with a
    # ``flight_profile`` (or ``flight_duration_s`` short enough to
    # trigger transient evaluation). At long sustained flights the
    # transient stage is a near-no-op: peak_backface_K converges to
    # the steady-state surface temperature and the thermal stage's
    # static T_wall check already covers the screening.
    transient_status: str = "not_applicable"
    """One of: ``"applied"`` (1D transient solve ran, peak temperatures
    populated), ``"not_applicable"`` (TPS / polymer / no-c_p material,
    or sustained flight where the static T_wall check is sufficient),
    or ``"unknown"`` (c_p not sourced — surfaced as a flag, not an
    auto-reject)."""
    transient_peak_surface_K: float | None = None
    transient_peak_backface_K: float | None = None
    transient_time_at_peak_backface_s: float | None = None
    transient_method: str = ""


@dataclass
class MatchResult:
    physics: PhysicsResult
    vehicle_category: str              # category key passed to match_materials
    viable: list                       # MaterialCandidate — both pass, score desc
    marginal: list                     # MaterialCandidate — no fail, ≥1 marginal, score desc
    not_viable: list                   # MaterialCandidate — ≥1 fail, score desc (nearest-miss first)
    regime_rejected: list              # MaterialEntry — not rated for this regime
    no_material_viable: bool           # True when viable is empty
    impossible: bool                   # True when viable AND marginal both empty
    diagnosis: str                     # non-empty when no_material_viable=True
    warnings: list                     # physics.warnings + matching-level warnings
    tps_coatings: list = field(default_factory=list)
    # MaterialCandidate — non-load-bearing TPS/ablators surfaced as a paired
    # "Required Coating Layer" recommendation when ablative-unlock is active.
    # NEVER ranked against metals on structural margin — sorted by thermal
    # margin descending within their own list. Empty when T_wall < TPS_UNLOCK_TEMP_K.


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _thermal_status(
    material: MaterialEntry,
    T_wall_K: float,
    T_wall_max_K: float,
    T_wall_SG_uncapped_K: float,
    vehicle_category: str,
    uses_recovery_model: bool = False,
) -> tuple:
    """
    Returns (status, ceiling_K, margin_K, notes).

    Thermal ceiling = max(bare_ceiling, coated_max_temp_K)
       bare_ceiling = min(service_temp_air_K, oxidation_max_temp_K)
    Materials with a mature protective coating (silicide/aluminide/HfC) publish
    a coated service temperature that exceeds the bare oxidation limit. When
    present, the coated value wins — otherwise the bare value is the ceiling.

    For the "aircraft" category, when the recovery cap is active the uncapped
    Sutton-Graves worst-case is not physically meaningful — the recovery
    temperature IS the upper bound.  Use T_wall_K as the effective worst-case.
    """
    bare_ceiling   = min(material.service_temp_air_K, material.oxidation_max_temp_K)
    coated_ceiling = getattr(material, "coated_max_temp_K", 0.0) or 0.0
    coating_active = coated_ceiling > bare_ceiling
    ceiling        = coated_ceiling if coating_active else bare_ceiling
    margin         = ceiling - T_wall_K
    notes: list    = []

    cap_active = (
        not uses_recovery_model
        and T_wall_K < T_wall_SG_uncapped_K - 0.5
    )
    if vehicle_category == "aircraft" and cap_active:
        effective_T_wall_max = T_wall_K   # recovery temperature IS the hard upper bound
    else:
        effective_T_wall_max = T_wall_max_K

    if ceiling >= effective_T_wall_max:
        status = "pass"
    elif ceiling >= T_wall_K:
        status = "marginal"
        notes.append(
            f"Thermal ceiling {ceiling:.0f} K clears nominal T_wall ({T_wall_K:.0f} K) "
            f"but not worst-case ({effective_T_wall_max:.0f} K)"
        )
    else:
        status = "fail"
        notes.append(
            f"Thermal ceiling {ceiling:.0f} K below T_wall {T_wall_K:.0f} K"
        )

    # Note which limit is binding
    if coating_active:
        notes.append(
            f"Protective coating required — coated ceiling {ceiling:.0f} K "
            f"vs. bare {bare_ceiling:.0f} K (oxidation limit)"
        )
    elif material.oxidation_max_temp_K < material.service_temp_air_K:
        notes.append("Oxidation limit is binding (not service temperature)")

    return status, ceiling, margin, notes


def _structural_status(
    material: MaterialEntry,
    T_wall_K: float,
    T_ambient_K: float,
    sigma_combined_MPa: float,
    vehicle_category: str = "general",
) -> tuple:
    """
    Returns (status, strength_MPa, sigma_req_material_MPa, margin_fraction, notes).

    Uses material-specific E and alpha (from the MaterialEntry) to compute the
    thermal stress component for most materials. Exception: polymer matrix composites
    (composite_polymer) use reference steel constants (E=200 GPa, alpha=12e-6) because
    their unidirectional fiber CTE is unrealistically low for laminate analysis — matrix
    cracking, biaxial laminate effects, and interlaminar failure dominate before the
    classical CTE-based thermal stress limit. Using reference constants prevents
    inflated structural margins for CFRP/PMC materials while preserving material-
    specific behaviour for metals, ceramics, and UHTCs.

    sigma_req_material = sigma_combined + E_thermal * alpha_thermal * delta_T
    where E_thermal/alpha_thermal = material-specific for non-composites,
                                    reference steel for composite_polymer.

    For "turbine" category, strength is derated by 0.6× as a creep proxy for
    sustained high-temperature loading under centrifugal stress.
    """
    notes = []
    strength = get_strength_at_temperature(material, T_wall_K)

    if vehicle_category == "turbine":
        strength = strength * 0.6
        notes.append("Turbine: strength derated 40% as creep proxy")

    delta_T = max(0.0, T_wall_K - T_ambient_K)
    E_mpa = material.youngs_modulus_GPa * 1000.0

    if material.category == "composite_polymer":
        # Use reference steel constants: unidirectional-fiber CTE gives unrealistically
        # low thermal stress for real laminates (biaxial loading, matrix cracking).
        sigma_thermal_material = THERMAL_RELIEF_FACTOR * E_REF_MPA * ALPHA_REF * delta_T
    else:
        sigma_thermal_material = THERMAL_RELIEF_FACTOR * E_mpa * material.thermal_expansion_1K * delta_T
    sigma_req_material = sigma_combined_MPa + sigma_thermal_material

    # Edge case: no structural load
    if sigma_req_material <= 0.0:
        return "pass", strength, 0.0, _LARGE_SCORE, notes

    margin = strength / sigma_req_material - 1.0

    if margin >= MARGINAL_STRUCTURAL_FRACTION:
        status = "pass"
    elif margin >= 0.0:
        status = "marginal"
        notes.append(
            f"Strength at T_wall ({strength:.0f} MPa) meets material-specific requirement "
            f"({sigma_req_material:.0f} MPa) but margin is only {margin * 100:.1f}% "
            f"(threshold {MARGINAL_STRUCTURAL_FRACTION * 100:.0f}%)"
        )
    else:
        status = "fail"
        notes.append(
            f"Strength at T_wall ({strength:.0f} MPa) below material-specific requirement "
            f"({sigma_req_material:.0f} MPa)"
        )

    return status, strength, sigma_req_material, margin, notes


def _score(thermal_ceiling_K: float, T_wall_K: float, structural_margin_fraction: float) -> float:
    """
    Ranking score = min(thermal_margin_fraction, structural_margin_fraction).
    For viable/marginal: smallest positive value = minimum adequate (sorted ascending).
    For not_viable: least-negative value = nearest miss (sorted descending).
    """
    if T_wall_K > 0.0:
        thermal_frac = (thermal_ceiling_K - T_wall_K) / T_wall_K
    else:
        # Defensive guard, not a live code path. T_wall_K is computed from
        # ISA temperatures (always > 0) or radiation-equilibrium of a positive
        # heat flux (always > 0 for Mach >= 5 with non-vacuum atmosphere). The
        # only way to land here is an upstream physics bug or epsilon == 0.
        # We keep the branch as cheap insurance against future regressions.
        thermal_frac = _LARGE_SCORE
    return min(thermal_frac, structural_margin_fraction)


_SPECIFIC_STRENGTH_REF_DENSITY = {"aircraft": 2500.0, "hypersonic_aircraft": 2500.0, "hypersonic_missile": 4000.0}
_SPECIFIC_STRENGTH_WEIGHT      = {"aircraft": 0.4,    "hypersonic_aircraft": 0.4,    "hypersonic_missile": 0.6}


def _evaluate_material(
    material: MaterialEntry,
    physics: PhysicsResult,
    vehicle_category: str = "general",
    evaluation_mode: str = "direct",
    design_lifetime_hours: float = 1.0,
    panel_thickness_m: float = 0.002,
    flight_profile: tuple = (),
    _skip_transient: bool = False,
) -> MaterialCandidate:
    """
    Evaluate a single material candidate.

    evaluation_mode:
      "direct"    — Material is exposed to T_wall (peak stagnation/recovery temp).
                    Standard physics, current behavior.
      "substrate" — Material is the metallic substructure under an ablative
                    coating; it sees only the soak-through temperature
                    T_soak = max(T_ambient, _ABLATIVE_SUBSTRATE_T_FLOOR_K).
                    Used for the second-pass ablative-substructure recommendation.

    design_lifetime_hours:
      Total airframe / component design lifetime (hours). Drives the
      creep evaluation stage. Default 1.0 h (single-flight) makes
      creep a near-no-op so behaviour matches pre-Phase-3 results
      for any caller that hasn't yet threaded a real lifetime.
    """
    T_ambient      = physics.thermal.T_ambient_K
    sigma_combined = physics.structural.sigma_combined_MPa

    if evaluation_mode == "substrate":
        T_eval     = max(T_ambient, _ABLATIVE_SUBSTRATE_T_FLOOR_K)
        T_eval_max = T_eval                      # no uncertainty band on soak temperature
        T_eval_SG  = T_eval                      # disables cap_active branch
        uses_recov = True                        # treat as recovery-style (capless) for thermal logic
    else:
        T_eval     = physics.thermal.T_wall_K
        T_eval_max = physics.thermal.T_wall_max_K
        T_eval_SG  = physics.thermal.T_wall_SG_uncapped_K
        uses_recov = physics.thermal.uses_recovery_model

    t_status, ceiling, t_margin, t_notes = _thermal_status(
        material, T_eval, T_eval_max, T_eval_SG, vehicle_category, uses_recov,
    )

    # TPS materials are never load-bearing structural members — bypass the
    # structural check entirely (regardless of category). Margin is set to 0.0
    # rather than a sentinel because TPS candidates are partitioned into
    # MatchResult.tps_coatings before primary sorting; this value is not used
    # for ranking. (Previously _LARGE_SCORE leaked into the UI as "+1e8% margin".)
    if material.category == "tps":
        s_status = "pass"
        strength = get_strength_at_temperature(material, T_eval)
        sigma_req_mat = 0.0
        s_margin = 0.0
        s_notes: list = []
    else:
        s_status, strength, sigma_req_mat, s_margin, s_notes = _structural_status(
            material, T_eval, T_ambient, sigma_combined, vehicle_category
        )

    # ----- Transient-heat stage (Phase 7) -----
    # Runs the 1D heat solver when (a) the flight is short enough
    # that steady-state is overly conservative AND (b) the material
    # has a sourced c_p. Replaces the per-candidate thermal-status
    # input with peak_backface_K so the structural/creep stages see
    # the realistic internal soak. For sustained flights or
    # materials without c_p, this stage is a no-op and the static
    # T_wall is used as before.
    transient_status_local = "not_applicable"
    transient_peak_surface = None
    transient_peak_backface = None
    transient_t_peak_back = None
    transient_method_used = ""

    flight_duration = float(physics.flight_duration_s)
    _has_profile = bool(flight_profile)
    transient_triggered = (
        not _skip_transient
        and evaluation_mode == "direct"
        and material.category != "tps"
        and (flight_duration < TRANSIENT_DURATION_THRESHOLD_S or _has_profile)
        and material.cp_data_status not in ("not_applicable", "unknown")
        and material.specific_heat_J_kgK is not None
    )

    if transient_triggered:
        try:
            t_result = integrate_panel(
                material=material,
                panel_thickness_m=float(panel_thickness_m),
                flight_profile=flight_profile,
                R_n_m=float(physics.thermal.R_n_m) if hasattr(
                    physics.thermal, "R_n_m"
                ) else 0.30,
                wall_emissivity=0.85,
                flight_duration_s=flight_duration,
                fallback_mach=float(physics.peak_mach),
                fallback_alt_km=float(physics.atmosphere.altitude_km),
            )
            transient_status_local = t_result.status
            transient_peak_surface = t_result.peak_surface_K
            transient_peak_backface = t_result.peak_backface_K
            transient_t_peak_back = t_result.time_at_peak_backface_s
            transient_method_used = t_result.method_used
            # Sanity-clamp the solver output: peak_backface must be a
            # finite temperature between ambient and the envelope's
            # T_wall. Outliers (numerical artefacts at extreme panel
            # thicknesses or material combinations) fall back to the
            # static T_wall so the matching never propagates -inf /
            # NaN downstream. T_eval at entry is the static T_wall.
            _backface = t_result.peak_backface_K
            _physically_sensible = (
                t_result.status == "applied"
                and _backface is not None
                and math.isfinite(_backface)
                and _backface >= T_ambient - 5.0
                and _backface <= T_eval + 5.0
            )
            if _physically_sensible and _backface < T_eval:
                # Use the transient backface peak as the operative
                # thermal screening temperature. Re-evaluate thermal +
                # structural at this lower temperature.
                T_eval = _backface
                T_eval_max = T_eval
                T_eval_SG = T_eval
                uses_recov = True
                t_status, ceiling, t_margin, t_notes = _thermal_status(
                    material, T_eval, T_eval_max, T_eval_SG,
                    vehicle_category, uses_recov,
                )
            elif _backface is not None and not _physically_sensible:
                # Solver produced an unphysical value; flag and skip.
                transient_status_local = "unknown"
        except Exception:
            # Solver failures (numerical edge cases) fall back to the
            # static thermal evaluation and surface the issue as
            # ``transient_status="unknown"``. Never let the transient
            # stage break the whole match.
            transient_status_local = "unknown"

    # ----- Creep / lifecycle stage (Phase 3) -----
    # TPS materials are non-load-bearing — skip the creep check
    # entirely and report not_applicable with a zero margin so
    # downstream sorting / scoring is not affected.
    if material.category == "tps":
        creep_verdict = None
        creep_status = "not_applicable"
        creep_rupture: float | None = None
        creep_margin: float | None = None
        creep_lmp: float | None = None
        creep_source = ""
        creep_extrap = False
    else:
        creep_verdict = evaluate_creep(
            material,
            T_K=T_eval,
            t_hours=design_lifetime_hours,
            sigma_required_MPa=max(sigma_req_mat, 0.0),
        )
        creep_status = creep_verdict.status
        creep_rupture = creep_verdict.rupture_stress_MPa
        creep_margin = creep_verdict.margin_fraction
        creep_lmp = creep_verdict.lmp_value
        creep_source = creep_verdict.data_source
        creep_extrap = creep_verdict.extrapolated
        if creep_verdict.notes:
            # Surface creep-evaluation notes (estimated curve,
            # extrapolated value) so the user sees provenance.
            t_notes = list(t_notes) + [creep_verdict.notes]

    # Overall status: a candidate is viable only if thermal AND
    # structural AND creep all pass-or-pass-through. Creep "unknown"
    # and "not_applicable" are pass-through (don't reject), but a
    # creep "marginal" downgrades viable -> marginal and a creep
    # "fail" sends the candidate to not_viable.
    if t_status == "fail" or s_status == "fail" or creep_status == "fail":
        overall = "not_viable"
    elif (
        t_status == "marginal" or s_status == "marginal"
        or creep_status == "marginal"
    ):
        overall = "marginal"
    else:
        overall = "viable"

    # Scoring: specific-strength-weighted for aircraft/missile, standard otherwise
    if vehicle_category in _SPECIFIC_STRENGTH_REF_DENSITY and sigma_req_mat > 0.0:
        ref_rho = _SPECIFIC_STRENGTH_REF_DENSITY[vehicle_category]
        w       = _SPECIFIC_STRENGTH_WEIGHT[vehicle_category]
        sp_margin = (strength * ref_rho) / (material.density_kgm3 * sigma_req_mat) - 1.0
        min_margin = _score(ceiling, T_eval, s_margin)
        candidate_score = (1.0 - w) * min_margin + w * sp_margin
    else:
        candidate_score = _score(ceiling, T_eval, s_margin)

    all_notes = t_notes + s_notes

    # CMC supersonic deprioritization (physics-driven, not category-driven).
    # Below Mach 5, CMCs (C/SiC, SiC/SiC, Oxide/Oxide) are penalized to prevent
    # them from out-ranking titanium/steel for primary structure on supersonic
    # vehicles, where their manufacturing cost and brittleness are not justified
    # by the modest thermal advantage. Turbine category is exempt — CMCs are the
    # canonical replacement for nickel superalloys in HPT hot-section.
    if (
        material.category == "composite_ceramic"
        and physics.peak_mach < _CMC_SUPERSONIC_MACH_THRESHOLD
        and vehicle_category != "turbine"
    ):
        candidate_score -= _CMC_SUPERSONIC_PENALTY
        all_notes.insert(
            0,
            f"CMC deprioritized for Mach {physics.peak_mach:.1f} regime: "
            "manufacturing complexity and brittleness outweigh the thermal-margin "
            "advantage; titanium or high-temperature steel preferred for primary structure",
        )
        if overall == "viable":
            overall = "marginal"

    # Substrate-mode tag: prepend a clear note so the user understands this
    # candidate represents the metal beneath an ablative coating, not direct exposure.
    if evaluation_mode == "substrate":
        all_notes.insert(
            0,
            f"Substructure under ablative coating "
            f"(T_soak ≈ {T_eval:.0f} K, evaluated below the ablator backside)",
        )

    return MaterialCandidate(
        material=material,
        thermal_ceiling_K=ceiling,
        thermal_margin_K=t_margin,
        strength_at_T_wall_MPa=strength,
        sigma_req_material_MPa=sigma_req_mat,
        structural_margin_fraction=s_margin,
        thermal_status=t_status,
        structural_status=s_status,
        overall_status=overall,
        score=candidate_score,
        notes=all_notes,
        evaluation_mode=evaluation_mode,
        creep_status=creep_status,
        creep_rupture_stress_MPa=creep_rupture,
        creep_margin_fraction=creep_margin,
        creep_lmp_value=creep_lmp,
        creep_data_source=creep_source,
        creep_extrapolated=creep_extrap,
        transient_status=transient_status_local,
        transient_peak_surface_K=transient_peak_surface,
        transient_peak_backface_K=transient_peak_backface,
        transient_time_at_peak_backface_s=transient_t_peak_back,
        transient_method=transient_method_used,
    )


def _build_diagnosis(physics: PhysicsResult, evaluated: list) -> str:
    """Build a human-readable diagnosis string when no material is viable."""
    if not evaluated:
        return (
            f"No materials are rated for the '{physics.flight_regime}' regime."
        )

    T_wall = physics.thermal.T_wall_K
    sigma_req = physics.structural.sigma_tensile_required_MPa
    regime = physics.flight_regime

    n_thermal_fail = sum(1 for c in evaluated if c.thermal_status == "fail")
    n_structural_fail = sum(1 for c in evaluated if c.structural_status == "fail")
    n_total = len(evaluated)

    parts = []

    if n_thermal_fail == n_total:
        parts.append(
            f"Thermal requirement (T_wall = {T_wall:.0f} K) exceeds the service ceiling "
            f"of all {n_total} candidates in the '{regime}' regime."
        )
    elif n_thermal_fail > 0:
        parts.append(
            f"Thermal requirement (T_wall = {T_wall:.0f} K) eliminated "
            f"{n_thermal_fail} of {n_total} candidates in the '{regime}' regime."
        )

    if n_structural_fail == n_total:
        parts.append(
            f"Reference structural requirement ({sigma_req:.0f} MPa at T_wall = {T_wall:.0f} K) "
            f"exceeds temperature-corrected strength of all candidates. "
            f"Note: per-material checks use each material's own E and \u03b1 with a "
            f"thermal relief factor; the reference value above is shown for context only."
        )
    elif n_structural_fail > 0:
        parts.append(
            f"Reference structural requirement ({sigma_req:.0f} MPa) eliminated "
            f"{n_structural_fail} of {n_total} candidates "
            f"(actual checks are per-material, with thermal relief)."
        )

    if not parts:
        parts.append(
            f"No single material passed both thermal and structural filters for the "
            f"'{regime}' regime (T_wall = {T_wall:.0f} K, reference \u03c3_req = {sigma_req:.0f} MPa)."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Materials excluded from evaluation by vehicle category (category = material.category field)
_CATEGORY_EXCLUSIONS: dict = {
    # TPS removed: TPS unlock is now physics-driven (T_wall ≥ 1200 K) — see
    # ablative_unlock_active in match_materials().
    # composite_ceramic removed: CMCs are kept visible but penalised below
    # Mach 5 (see _evaluate_material's CMC supersonic penalty).
    # Refractories and UHTCs remain excluded from aircraft — they would never
    # be used as airframe structural materials.
    "aircraft":            {"refractory", "uhtc"},
    "hypersonic_aircraft": set(),
    "reentry":             set(),
    # composite_polymer matrices (CFRP/PMC) are not viable for expendable missile
    # structures at elevated supersonic temperatures — exclude from missile category.
    "hypersonic_missile":  {"composite_polymer"},
    "turbine":             {"tps", "aluminum", "composite_polymer", "general_engineering"},
    "general":             set(),
}


def _get_category_exclusions(vehicle_category: str, physics) -> set[str]:
    """Return material categories to exclude, considering Mach-dependent rules."""
    base = set(_CATEGORY_EXCLUSIONS.get(vehicle_category, set()))
    # Polymer matrix composites are excluded for aircraft at Mach >= 2.0:
    # matrix glass-transition temperatures are exceeded, and near-zero fiber-direction
    # CTE produces artificially favorable thermal stress that displaces titanium.
    if vehicle_category == "aircraft" and physics.peak_mach >= 2.0:
        base.add("composite_polymer")
    return base


# Hard density ceiling by vehicle category.
# Materials above the threshold are non-starters due to weight budget constraints.
# Aircraft: 5000 kg/m³ keeps all Ti alloys (≤4760); excludes all Ni superalloys (≥8190).
# Hypersonic aircraft: 8500 kg/m³ admits Inconel-class hot structure (X-15 used Inconel X-750 ~8280).
# Missile:  8500 kg/m³ retains Inconel 718 (8190) for high-temperature missile
#   hot sections while excluding Mo (10280), W (19300), Re (21020). Matches
#   aircraft ablative-unlock ceiling — weight criticality difference between
#   aircraft and expendable missiles is addressed by the specific-strength
#   scoring weight (60% for missiles vs 40% for aircraft) rather than a
#   different density hard cutoff.
MAX_DENSITY_KGM3: dict = {
    "aircraft":            5000.0,
    "hypersonic_aircraft": 8500.0,
    "hypersonic_missile":  8500.0,
}


def match_materials(
    physics: PhysicsResult,
    vehicle_category: str = "general",
    *,
    design_lifetime_hours: float = 1.0,
    panel_thickness_m: float = 0.002,
    flight_profile: tuple = (),
    _skip_transient: bool = False,
) -> MatchResult:
    """
    Filter and rank all materials against the given PhysicsResult.

    vehicle_category is now mostly a preference dial (density ceiling,
    specific-strength weighting, irrelevant-class exclusions).  The major
    decisions — TPS unlock, CMC penalty, density-ceiling lift, substructure
    second pass — are driven by physics (T_wall, peak_mach), not category:

      • TPS unlock        : T_wall ≥ TPS_UNLOCK_TEMP_K  OR  category in {reentry,
                            hypersonic_aircraft}
      • CMC penalty       : peak_mach < 5     AND  category != turbine
      • Density lift      : ablative-unlock active → ceiling raised to 8500 kg/m³
      • Substrate pass    : ablative-unlock active → metals re-evaluated at
                            T_soak = max(T_ambient, 400 K)

    ``design_lifetime_hours`` (keyword-only; default 1.0) drives the
    Phase 3 creep evaluation stage. At single-flight lifetimes (~1 h)
    the creep stage is a near-no-op so existing callers without a
    real lifetime see behaviour identical to pre-Phase-3. Wire a
    realistic value (Concorde 25,000; SR-71 3,000; CFM56 25,000)
    through ``core/api.run_session`` to opt into lifecycle screening.

    Returns a MatchResult with four partitioned lists (viable, marginal,
    not_viable, regime_rejected), each sorted by composite score descending.
    """
    # Defensive clamp: zero / negative lifetime is physically meaningless
    # but we'd rather return a sensible "single-flight" result than
    # crash the entire pipeline. ``evaluate_creep`` would raise ValueError
    # if we passed 0 or negative through. The clamp threshold mirrors the
    # SessionSchema default.
    if design_lifetime_hours <= 0.0:
        design_lifetime_hours = 1.0

    regime = physics.flight_regime

    # Master physics-driven flag: is the vehicle hot enough to require an
    # external ablative coating?  Reentry and hypersonic_aircraft (legacy)
    # categories also force this on for backward compatibility.
    ablative_unlock_active = (
        physics.thermal.T_wall_K >= TPS_UNLOCK_TEMP_K
        or vehicle_category in ("reentry", "hypersonic_aircraft")
    )

    # Stage 1a — regime filter
    regime_candidates = get_materials_by_regime(regime)
    regime_candidate_names = {m.name for m in regime_candidates}
    regime_rejected = [m for m in MATERIALS_DB if m.name not in regime_candidate_names]

    # Stage 1b — category applicability filter
    excluded_mat_categories = _get_category_exclusions(vehicle_category, physics)
    category_excluded = [m for m in regime_candidates if m.category in excluded_mat_categories]
    regime_candidates  = [m for m in regime_candidates if m.category not in excluded_mat_categories]
    regime_rejected    = regime_rejected + category_excluded

    # Stage 1c — TPS unlock (physics-driven)
    # When ablative-unlock is active, pull TPS materials (which the regime
    # classifier puts in regime_rejected for non-reentry conditions) back into
    # the candidate pool so they can be recommended as ablative coatings.
    if ablative_unlock_active:
        tps_from_rejected  = [m for m in regime_rejected if m.category == "tps"]
        regime_candidates  = regime_candidates + tps_from_rejected
        regime_rejected    = [m for m in regime_rejected if m.category != "tps"]

    # Stage 1d — density ceiling filter (lifted under ablative-unlock)
    max_rho = MAX_DENSITY_KGM3.get(vehicle_category)
    if max_rho is not None and ablative_unlock_active:
        # When the vehicle needs ablative coating, weight optimization is
        # secondary to thermal survival — admit Inconel-class hot structure.
        max_rho = max(max_rho, _ABLATIVE_DENSITY_CEILING_K)
    if max_rho is not None:
        density_excluded = [m for m in regime_candidates if m.density_kgm3 > max_rho]
        regime_candidates = [m for m in regime_candidates if m.density_kgm3 <= max_rho]
        density_not_viable = []
        for m in density_excluded:
            cand = _evaluate_material(
                m, physics, vehicle_category,
                design_lifetime_hours=design_lifetime_hours,
                panel_thickness_m=panel_thickness_m,
                flight_profile=flight_profile,
                _skip_transient=_skip_transient,
            )
            cand.notes.insert(
                0,
                f"Density {m.density_kgm3:.0f} kg/m³ exceeds {vehicle_category} "
                f"ceiling ({max_rho:.0f} kg/m³)",
            )
            cand.overall_status = "not_viable"
            density_not_viable.append(cand)
    else:
        density_not_viable = []

    # Stages 2, 3, 4 & 5 — direct-mode evaluation (T_wall) + creep
    # + transient-heat stages
    evaluated = [
        _evaluate_material(
            m, physics, vehicle_category,
            evaluation_mode="direct",
            design_lifetime_hours=design_lifetime_hours,
            panel_thickness_m=panel_thickness_m,
            flight_profile=flight_profile,
            _skip_transient=_skip_transient,
        )
        for m in regime_candidates
    ]

    # Substrate-mode second pass: metal substructure beneath an ablative coating.
    # Each metal candidate is re-evaluated at T_soak = max(T_ambient, 400 K) so the
    # display layer can present the combined "ablator + metal substructure"
    # recommendation that real spaceplanes (X-15, Shuttle Orbiter) actually use.
    if ablative_unlock_active:
        substrate_candidates_in = [
            m for m in regime_candidates
            if m.category in _SUBSTRATE_METAL_CATEGORIES
        ]
        substrate_evaluated = [
            _evaluate_material(
                m, physics, vehicle_category,
                evaluation_mode="substrate",
                design_lifetime_hours=design_lifetime_hours,
                # Substrate mode evaluates at the back-face soak
                # temperature with the static thermal logic; the
                # transient stage is a no-op there so we don't pass
                # the profile through.
            )
            for m in substrate_candidates_in
        ]
        evaluated.extend(substrate_evaluated)

    # Stage 4 — separate TPS materials from primary structural ranking.
    # TPS materials are non-load-bearing: they cannot be compared against metals
    # on structural margin and must never appear at the top of the viable list.
    # When ablative-unlock is active they are surfaced as a paired "Required
    # Coating Layer" recommendation alongside the highest-scoring metal substrate.
    tps_evaluated = [c for c in evaluated if c.material.category == "tps"]
    evaluated     = [c for c in evaluated if c.material.category != "tps"]

    # Sort TPS coatings by thermal margin descending (best protection first),
    # grouped by status: viable first, then marginal, then not_viable.
    _tps_status_rank = {"viable": 0, "marginal": 1, "not_viable": 2}
    tps_coatings = sorted(
        tps_evaluated,
        key=lambda c: (
            _tps_status_rank.get(c.overall_status, 3),
            -(c.thermal_ceiling_K - physics.thermal.T_wall_K),  # neg → descending
        ),
    )

    # Partition by overall_status
    # mass-dominated categories (aircraft, hypersonic_aircraft, hypersonic_missile):
    #   viable/marginal sorted by density ascending (lightest adequate material first)
    # all others: viable/marginal ascending by score (minimum adequate first)
    # not_viable: descending by score (nearest miss first)
    if vehicle_category in ("aircraft", "hypersonic_aircraft", "hypersonic_missile"):
        viable = sorted(
            [c for c in evaluated if c.overall_status == "viable"],
            key=lambda c: c.material.density_kgm3,
        )
        marginal = sorted(
            [c for c in evaluated if c.overall_status == "marginal"],
            key=lambda c: c.material.density_kgm3,
        )
    else:
        viable = sorted(
            [c for c in evaluated if c.overall_status == "viable"],
            key=lambda c: c.score,
        )
        marginal = sorted(
            [c for c in evaluated if c.overall_status == "marginal"],
            key=lambda c: c.score,
        )
    not_viable = sorted(
        [c for c in evaluated if c.overall_status == "not_viable"] + density_not_viable,
        key=lambda c: c.score, reverse=True,
    )

    no_material_viable = len(viable) == 0
    impossible = no_material_viable and len(marginal) == 0

    diagnosis = _build_diagnosis(physics, evaluated) if no_material_viable else ""

    # Collect warnings: start from physics, add matching-level notes
    warnings = list(physics.warnings)
    if impossible:
        warnings.append(
            f"No viable or marginal materials found for '{regime}' regime. "
            "See diagnosis for details."
        )

    return MatchResult(
        physics=physics,
        vehicle_category=vehicle_category,
        viable=viable,
        marginal=marginal,
        not_viable=not_viable,
        regime_rejected=regime_rejected,
        no_material_viable=no_material_viable,
        impossible=impossible,
        diagnosis=diagnosis,
        warnings=warnings,
        tps_coatings=tps_coatings,
    )
