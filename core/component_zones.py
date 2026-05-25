"""
core/component_zones.py — per-zone material recommendations.

Item 6 of the six-improvement plan. The whole-vehicle MatchResult that
match_materials() returns represents the *worst* zone of the vehicle —
the leading edge, stagnation point, or hot-section blade tip — because
that is where T_wall and sigma_req peak. But a real airframe is built
out of zones with very different thermal and structural demands: an
internal spar 10 cm aft of the nose sees a fraction of the leading-edge
temperature and a different stress distribution. Showing a single
"this material survives the whole vehicle" list hides this and forces
the user to either over-spec everything or spend a separate session
estimating each zone by hand.

This module decomposes a vehicle into 3-5 named zones per category and
runs match_materials against each zone's locally-scaled physics view.
The output is a list of ZoneMatchResult, one per zone, in the order
defined by CATEGORY_ZONES[vehicle_category].

Multiplier semantics
--------------------

* ``t_wall_multiplier`` scales the temperature *rise* above ambient,
  not T_wall itself. Local recovery factor varies with surface angle:
  the leading edge recovers ~100% of the stagnation rise, an oblique
  fuselage panel recovers a smaller fraction, and a shaded leeward
  surface recovers very little. Defining the multiplier on the rise
  means a multiplier of 0 yields T_amb (correct floor) rather than
  zero kelvin (unphysical):

      T_wall_zone = T_amb + t_wall_multiplier * (T_wall_whole - T_amb)

* ``sigma_req_multiplier`` scales sigma_req directly. A multiplier
  > 1 means the zone sees a larger structural demand than the
  whole-vehicle reference (e.g. a turbine disk root carries much
  higher centrifugal stress than the blade airfoil); a multiplier
  < 1 represents membrane vs. bending stress relief on internal
  panels, etc.

Multipliers are engineering estimates documented per category below.
They are NOT preset-tuned — they are category-level constants
defensible from heat-transfer and stress-distribution principles.
If calibration shifts during use, multipliers move in one place.

Cross-cutting design notes
--------------------------

* No physics-engine constants are touched. Zone scaling rebuilds
  ThermalResults / StructuralResults / PhysicsResult via
  ``dataclasses.replace`` and feeds the result back into the
  unmodified ``match_materials``. This keeps the settled physics
  layer (HANDOFF.md §6) immutable.
* Audience: small-rocket / university / startup-concept users.
  Zone descriptions are plain-English ("nose tip", "fuselage skin")
  rather than CFD jargon.
"""

import dataclasses
from dataclasses import dataclass, field

from .physics_engine import PhysicsResult, ThermalResults, StructuralResults
from .matching_engine import match_materials, MatchResult


@dataclass(frozen=True)
class ComponentZone:
    """A named region of the vehicle with locally-scaled thermal/structural demands."""
    name: str
    description: str
    t_wall_multiplier: float       # multiplies the temperature *rise* above ambient
    sigma_req_multiplier: float    # multiplies sigma_req directly


@dataclass
class ZoneMatchResult:
    """A zone plus the locally-derived T/σ values and the match it produced."""
    zone: ComponentZone
    T_wall_zone_K: float
    sigma_req_zone_MPa: float
    match: MatchResult


# ---------------------------------------------------------------------------
# Per-category zone catalog
# ---------------------------------------------------------------------------
#
# Multiplier ranges below are engineering estimates. Sources of variation:
#   - Recovery factor varies with local flow angle (leading edge vs panel).
#   - Radiative cooling lowers leeward / aft surface temperatures relative
#     to the windward stagnation region (see Anderson, Hypersonic & High-
#     Temperature Gas Dynamics, ch. 3 for the basic picture).
#   - Internal substructure sees conduction-bounded soak temperatures, well
#     below the surface boundary-layer recovery temperature.
#   - Bending vs membrane stress relief: an internal spar carries 1.0–1.4×
#     the membrane tensile stress on a thin panel (factor depends on the
#     stiffness ratio); a panel face carries less than the spar but more
#     than a free-floating skin.
#   - Turbine: blade root sees ~1.5× the airfoil's centrifugal stress
#     (concentration at the fir-tree); disk hub sees ~2× (carries the full
#     blade-mass × ω² load); NGV is non-rotating so ~0.6× the blade load.

CATEGORY_ZONES = {
    "aircraft": (
        ComponentZone("Leading edge / nose",
                      "Stagnation-point heating and the primary thermal driver.",
                      1.00, 1.00),
        ComponentZone("Lower fuselage skin",
                      "Windward panels see most of the recovery temperature; load combines "
                      "membrane tension and pressure.",
                      0.75, 0.60),
        ComponentZone("Upper fuselage skin",
                      "Leeward / shaded panels recover less of the stagnation rise.",
                      0.55, 0.40),
        ComponentZone("Internal structure",
                      "Spars and longerons soak to a much lower temperature but carry "
                      "concentrated bending loads.",
                      0.40, 1.20),
    ),
    "hypersonic_aircraft": (
        ComponentZone("Stagnation / nose tip",
                      "Peak convective and radiative heating; sets the upper-bound material requirement.",
                      1.00, 1.00),
        ComponentZone("Wing leading edge",
                      "Sharp-edge heating only slightly below the nose tip; structurally lighter.",
                      0.95, 0.70),
        ComponentZone("Windward skin",
                      "Lower fuselage panels under direct shock heating.",
                      0.80, 0.50),
        ComponentZone("Leeward skin",
                      "Wake-side panels see substantially relieved heating.",
                      0.55, 0.40),
        ComponentZone("Internal substructure",
                      "Spars / bulkheads soak well below the surface temperature; carry concentrated loads.",
                      0.40, 1.30),
    ),
    "hypersonic_missile": (
        ComponentZone("Nose tip",
                      "Peak stagnation heating; small-radius blunt or pointed nose.",
                      1.00, 1.00),
        ComponentZone("Fin leading edge",
                      "Sharp control-surface edges run hot but carry less load than the body.",
                      0.85, 0.60),
        ComponentZone("Body skin",
                      "Cylindrical body panels — recovery temperature, modest membrane stress.",
                      0.70, 0.50),
        ComponentZone("Internal structure",
                      "Frames / motor case interface — cooler soak, concentrated bending.",
                      0.40, 1.20),
    ),
    "reentry": (
        ComponentZone("Stagnation point",
                      "Peak heat flux on the windward heat shield centerline.",
                      1.00, 1.00),
        ComponentZone("Windward heat shield",
                      "Off-center forebody panels still see severe heating but lower stress than the apex.",
                      0.90, 0.50),
        ComponentZone("Shoulder / corner",
                      "High curvature region with locally elevated heating but moderate load.",
                      0.80, 0.70),
        ComponentZone("Backshell / leeward",
                      "Wake-side surfaces — much lower heating, modest load.",
                      0.45, 0.30),
        ComponentZone("Internal substructure",
                      "Pressure shell / equipment frame inside the TPS; soak temperature, "
                      "concentrated bending under deceleration g-load.",
                      0.30, 1.40),
    ),
    "turbine": (
        ComponentZone("Blade leading edge / airfoil",
                      "Hot-section gas-stream temperature; primary creep + oxidation driver.",
                      1.00, 0.80),
        ComponentZone("Blade trailing edge",
                      "Slightly relieved gas temperature, thinner section, higher local stress.",
                      0.85, 0.70),
        ComponentZone("Blade root / fir-tree",
                      "Cooler than the airfoil but carries the full centrifugal load — "
                      "stress concentration at the attachment lobes.",
                      0.60, 1.50),
        ComponentZone("Disk hub",
                      "Coolest part of the rotating assembly but carries the integrated "
                      "blade-mass centrifugal load.",
                      0.50, 2.00),
        ComponentZone("Nozzle guide vane (NGV)",
                      "Stationary; sees the same gas temperature as the blade but does not rotate.",
                      1.00, 0.60),
    ),
    "general": (
        ComponentZone("Panel face",
                      "The reference design point: surface temperature, design stress.",
                      1.00, 1.00),
        ComponentZone("Panel edge / fastener",
                      "Joints and fastener boundaries — local stress concentrations.",
                      0.90, 1.30),
        ComponentZone("Internal substructure",
                      "Cooler soak; carries concentrated bending and shear.",
                      0.70, 0.80),
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_zones(physics, vehicle_category):
    """Run match_materials against each zone of the given vehicle category.

    Parameters
    ----------
    physics : PhysicsResult
        The whole-vehicle physics result (the existing pipeline output).
    vehicle_category : str
        One of the keys in CATEGORY_ZONES. Unknown categories return [].

    Returns
    -------
    list[ZoneMatchResult]
        One result per zone in the order defined by CATEGORY_ZONES.
        Empty list if vehicle_category is not in CATEGORY_ZONES.
    """
    zones = CATEGORY_ZONES.get(vehicle_category)
    if not zones:
        return []
    out = []
    for zone in zones:
        scaled = _scale_physics_for_zone(physics, zone)
        # Per-zone analysis uses the static multipliers, which already
        # approximate transient lag at internal zones via the smaller
        # t_wall_multiplier values (e.g. internal_structure = 0.40 ×
        # surface). Running the 1D transient solver on top would
        # double-correct AND cost N_materials × N_zones runs of an
        # expensive solver — skip it here.
        zone_match = match_materials(
            scaled, vehicle_category=vehicle_category,
            _skip_transient=True,
        )
        out.append(ZoneMatchResult(
            zone=zone,
            T_wall_zone_K=scaled.thermal.T_wall_K,
            sigma_req_zone_MPa=scaled.structural.sigma_tensile_required_MPa,
            match=zone_match,
        ))
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _scale_physics_for_zone(physics, zone):
    """Return a new PhysicsResult whose thermal/structural fields are
    scaled to the zone's local demands. Zero physics-engine modification —
    we build new dataclass instances via dataclasses.replace and leave
    the original physics object untouched."""
    th = physics.thermal
    st = physics.structural
    T_amb = th.T_ambient_K

    # Scale temperatures on the *rise* above ambient (see module docstring).
    def scale_T(T):
        return T_amb + zone.t_wall_multiplier * (T - T_amb)

    scaled_thermal = dataclasses.replace(
        th,
        T_wall_K=scale_T(th.T_wall_K),
        T_wall_min_K=scale_T(th.T_wall_min_K),
        T_wall_max_K=scale_T(th.T_wall_max_K),
        T_wall_sealevel_K=scale_T(th.T_wall_sealevel_K),
        # T_stag, T_amb, q_*, plasma_*, uses_recovery_model, thermal_source
        # all unchanged — those are whole-vehicle/Mach properties.
    )

    # Scale structural demands. sigma_combined and sigma_tensile_required
    # are the two values the matching engine consumes.
    scaled_structural = dataclasses.replace(
        st,
        sigma_combined_MPa=zone.sigma_req_multiplier * st.sigma_combined_MPa,
        sigma_tensile_required_MPa=zone.sigma_req_multiplier * st.sigma_tensile_required_MPa,
        # F_inertial, q_dyn, A_ref, sigma_inertial, sigma_thermal_ref,
        # characteristic_length unchanged — they describe the whole vehicle.
    )

    return dataclasses.replace(
        physics,
        thermal=scaled_thermal,
        structural=scaled_structural,
    )
