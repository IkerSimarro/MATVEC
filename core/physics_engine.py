"""
MATVEC Physics Engine — Step 2
Computes aerodynamic heating, structural loads, propulsion energy, and EM
signature for a given flight envelope. Output feeds the Step 3 matching engine.

stdlib only: math, dataclasses
"""

__version__ = "1.0.0"

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GAMMA_AIR = 1.4
R_AIR = 287.058           # J/(kg·K)
G0 = 9.80665              # m/s²
SIGMA_SB = 5.670374419e-8 # W/(m²·K⁴)

# Sutton-Graves convective heating constant (SI, V in m/s, q in W/m²)
C_SUTTON_GRAVES = 1.7415e-4

# Tauber-Sutton radiative heating (V in km/s, q in W/cm²; multiply by 1e4 → W/m²)
C_TAUBER_RAD = 4.736e-4
A_TAUBER_RAD = 1.072
B_TAUBER_RAD = 3.5
V_RAD_THRESHOLD = 6000.0  # m/s

SUTTON_GRAVES_UNCERTAINTY = 0.20   # ±20%
RECOVERY_TEMP_UNCERTAINTY = 0.15   # ±15% for recovery temperature model (M < 5)

# Real-gas threshold for the calorically perfect stagnation-temperature formula.
# T_stag = T_amb·(1 + (γ-1)/2·M²) assumes γ = 1.4 is constant. Above ~3000 K in air,
# O2 dissociates and N2 vibrational modes unfreeze; the energy absorbed by these
# processes is no longer available as kinetic temperature. The calorically perfect
# value then overestimates the real T_stag by 2–5× (e.g. CPG gives ~17,600 K at
# Mach 20 vs. measured ~5,500 K). We flag values above this threshold so the
# report can disclose the limitation rather than silently presenting nonsense.
REAL_GAS_T_STAG_THRESHOLD_K = 3000.0
# Recovery temperature cap: wall temperature cannot physically exceed the adiabatic
# wall temperature (recovery temperature). Sutton-Graves overestimates for M < ~5
# because it is calibrated for hypersonic flow; the cap prevents unphysical T_wall
# values (e.g. 860 K at M=1.8 where the true recovery temperature is ~336 K).
RECOVERY_FACTOR = 0.85             # turbulent flat-plate recovery factor
STRUCTURAL_SAFETY_FACTOR = 1.5    # MIL-HDBK-5
E_REF_MPA = 200_000.0             # 200 GPa reference Young's modulus
ALPHA_REF = 12.0e-6               # reference CTE (1/K)
# Thermal expansion relief factor — aerospace structures mitigate thermal stress
# via slip joints, floating panels, segmented skins, blade-root slots, etc. The
# raw E×α×ΔT product assumes a fully-constrained isotropic block, which is never
# how real airframes are built. 0.4 is a reasonable engineering fraction (literature
# gives 0.2–0.5 depending on detail design); applied uniformly to the reference
# thermal stress here AND the per-material thermal stress in matching_engine.
THERMAL_RELIEF_FACTOR = 0.4
RHO_VEHICLE_REF = 2700.0          # kg/m³ for characteristic length estimation

ENERGY_DENSITY_KEROSENE = 4.32e7  # J/kg  (Jet-A / kerosene)
ENERGY_DENSITY_LH2      = 1.42e8  # J/kg  (liquid hydrogen)

Q_DYN_WARN_PA = 200_000.0         # Pa

# ---------------------------------------------------------------------------
# ISA atmosphere — 7-layer model, 0–86 km
# ---------------------------------------------------------------------------
_ISA_LAYERS = (
    # (h_lo_km, h_hi_km, T_base_K, lapse_K_per_km)
    ( 0.0,  11.0, 288.15, -6.5),
    (11.0,  20.0, 216.65,  0.0),
    (20.0,  32.0, 216.65, +1.0),
    (32.0,  47.0, 228.65, +2.8),
    (47.0,  51.0, 270.65,  0.0),
    (51.0,  71.0, 270.65, -2.8),
    (71.0,  86.0, 214.65, -2.0),
)

_P0 = 101325.0  # Pa, sea-level pressure


def _build_isa_pressure_bases() -> tuple:
    """Integrate pressure bases for each ISA layer bottom programmatically."""
    bases = [_P0]
    P = _P0
    for i, (h_lo, h_hi, T_base, lapse_km) in enumerate(_ISA_LAYERS):
        if i == len(_ISA_LAYERS) - 1:
            break  # no next layer
        h_lo_next, _, _, _ = _ISA_LAYERS[i + 1]
        lapse_m = lapse_km / 1000.0  # K/m
        dh_m = (h_lo_next - h_lo) * 1000.0
        if abs(lapse_m) < 1e-12:  # isothermal
            P = P * math.exp(-G0 * dh_m / (R_AIR * T_base))
        else:                     # gradient layer
            T_top = T_base + lapse_km * (h_lo_next - h_lo)
            P = P * (T_top / T_base) ** (-G0 / (R_AIR * lapse_m))
        bases.append(P)
    return tuple(bases)


_ISA_P_BASES: tuple = _build_isa_pressure_bases()


def _isa_atmosphere(altitude_km: float) -> tuple:
    """
    Return (T_K, P_Pa, rho_kg_m3) for the given altitude.
    Clamps: below 0 → 0 km; above 86 → 86 km.
    """
    h = max(0.0, min(86.0, altitude_km))
    for i, (h_lo, h_hi, T_base, lapse_km) in enumerate(_ISA_LAYERS):
        if h_lo <= h < h_hi or (i == len(_ISA_LAYERS) - 1 and h >= h_lo):
            P_base = _ISA_P_BASES[i]
            lapse_m = lapse_km / 1000.0
            dh_m = (h - h_lo) * 1000.0
            if abs(lapse_m) < 1e-12:  # isothermal
                T = T_base
                P = P_base * math.exp(-G0 * dh_m / (R_AIR * T_base))
            else:                     # gradient
                T = T_base + lapse_km * (h - h_lo)
                P = P_base * (T / T_base) ** (-G0 / (R_AIR * lapse_m))
            rho = P / (R_AIR * T)
            return T, P, rho
    # Should never reach here; return 86 km values as fallback
    return _isa_atmosphere(86.0)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AtmosphericConditions:
    altitude_km: float
    temperature_K: float
    pressure_Pa: float
    density_kgm3: float


@dataclass
class ThermalResults:
    velocity_ms: float
    q_conv_Wm2: float
    q_rad_Wm2: float            # 0.0 if V < 6000 m/s, or 0.0 when uses_recovery_model=True
    q_total_Wm2: float
    T_wall_K: float             # recovery temperature (M<5) or radiation-equilibrium (M≥5)
    T_wall_min_K: float         # lower bound: −15% (recovery) or −20% (SG)
    T_wall_max_K: float         # upper bound: +15% (recovery) or +20% (SG)
    T_wall_SG_uncapped_K: float # Sutton-Graves radiation-eq (SG regime); 0.0 sentinel when uses_recovery_model=True
    T_stag_K: float             # stagnation temperature (context only)
    T_ambient_K: float
    q_total_sealevel_Wm2: float
    T_wall_sealevel_K: float
    plasma_sheath: bool
    uses_recovery_model: bool   # True when M < 5 (recovery temp model); False when M >= 5 (SG)
    T_stag_real_gas_suspect: bool = False  # True when T_stag > REAL_GAS_T_STAG_THRESHOLD_K
                                           # (calorically perfect value overstates real T_stag)
    plasma_threshold_slender: bool = False # True when slender-body threshold (M>6) triggered
                                           # instead of the blunt-body threshold (M>10).
                                           # Used by the LaTeX layer to display the correct
                                           # threshold string in §4 and §7.
    thermal_source: str = "aerodynamic"    # "aerodynamic" (default) | "turbine_inlet_override"
                                           # When set to "turbine_inlet_override", T_wall_K has
                                           # been replaced by a hot-section (turbine inlet)
                                           # temperature. Downstream consumers (LaTeX §4, §6)
                                           # branch on this to produce correct narrative.


@dataclass
class StructuralResults:
    F_inertial_N: float
    q_dyn_Pa: float
    A_ref_m2: float
    sigma_inertial_MPa: float
    sigma_combined_MPa: float
    sigma_thermal_ref_MPa: float
    sigma_tensile_required_MPa: float
    characteristic_length_m: float


@dataclass
class PropulsionResults:
    KE_J: float
    P_peak_W: float
    E_total_J: float
    fuel_mass_kerosene_kg: float   # Jet-A equivalent for energy context
    fuel_mass_LH2_kg: float        # LH2 equivalent for energy context


@dataclass
class EMResults:
    P_rad_W: float
    lambda_peak_um: float
    emission_band: str          # "near-IR" / "mid-wave IR" / "long-wave IR" / "far-IR"
    plasma_sheath: bool
    plasma_threshold_slender: bool = False  # mirrors ThermalResults.plasma_threshold_slender
                                            # so LaTeX §7 can render the correct threshold
                                            # string without re-deriving it


@dataclass
class PhysicsResult:
    # Inputs echoed back
    peak_mach: float
    cruise_altitude_km: float
    vehicle_mass_kg: float
    nose_radius_m: float
    peak_g_load: float
    characteristic_length_m: float
    flight_duration_s: float
    wall_emissivity: float
    # Derived
    atmosphere: AtmosphericConditions
    thermal: ThermalResults
    structural: StructuralResults
    propulsion: PropulsionResults
    em: EMResults
    flight_regime: str
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sutton_graves(rho: float, V: float, R_n: float) -> float:
    """Sutton-Graves convective heat flux [W/m²]. Returns 0 if V=0."""
    if V <= 0.0 or R_n <= 0.0:
        return 0.0
    return C_SUTTON_GRAVES * math.sqrt(rho / R_n) * V ** 3


def _tauber_sutton_rad(rho: float, V: float, R_n: float) -> float:
    """
    Tauber-Sutton radiative heat flux [W/m²].
    Returns 0.0 if V < V_RAD_THRESHOLD (6000 m/s).
    V must be in m/s; internally converted to km/s per Tauber & Sutton (1991).
    The published constant C = 4.736e-4 produces q in W/cm²; multiply by 1e4.
    """
    if V < V_RAD_THRESHOLD or R_n <= 0.0:
        return 0.0
    V_kms = V / 1000.0
    q_Wcm2 = C_TAUBER_RAD * (rho ** A_TAUBER_RAD) * (V_kms ** B_TAUBER_RAD) * R_n
    return q_Wcm2 * 1e4  # W/cm² → W/m²


def _radiation_equilibrium_wall_temperature(q: float, epsilon: float) -> float:
    """Radiation-equilibrium wall temperature [K]. Returns 0.0 if q ≤ 0."""
    if q <= 0.0 or epsilon <= 0.0:
        return 0.0
    return (q / (epsilon * SIGMA_SB)) ** 0.25


def _compute_thermal(
    mach: float,
    alt_km: float,
    R_n: float,
    epsilon: float,
    atm: AtmosphericConditions,
) -> ThermalResults:
    T_amb = atm.temperature_K
    rho = atm.density_kgm3
    V = mach * math.sqrt(GAMMA_AIR * R_AIR * T_amb)

    # Stagnation temperature (context only — not a material requirement)
    T_stag = T_amb * (1.0 + (GAMMA_AIR - 1.0) / 2.0 * mach ** 2)

    # Plasma sheath — two-tier threshold based on body slenderness.
    #
    # Blunt bodies (R_n ≥ 0.5 m) form a thick shock layer that only ionises
    # strongly at M > ~10 — the classical "radio blackout" entry regime
    # (Apollo, Shuttle, Soyuz).
    #
    # Slender bodies (R_n < 0.5 m) — missiles, X-15, pointed hypersonic
    # airframes — show measurable plasma-related effects (radio attenuation,
    # UV glow) from M ~ 6 because the attached-shock boundary layer reaches
    # dissociation temperatures at lower Mach. The X-15 flew through exactly
    # this regime and documented it; the old M>10 threshold hid that fact.
    slender = R_n < 0.5
    if slender:
        plasma_sheath = mach > 6.0 and alt_km < 80.0
    else:
        plasma_sheath = mach > 10.0 and alt_km < 80.0
    plasma_threshold_slender = slender and plasma_sheath

    if mach < 5.0:
        # --- Recovery Temperature Model (M < 5) ---
        # For sustained supersonic cruise, the adiabatic wall (recovery) temperature
        # is the correct primary thermal framework. Sutton-Graves is calibrated for
        # hypersonic blunt-body entry and is not valid here.
        T_wall    = T_amb * (1.0 + RECOVERY_FACTOR * (GAMMA_AIR - 1.0) / 2.0 * mach ** 2)
        T_wall_min = T_wall * (1.0 - RECOVERY_TEMP_UNCERTAINTY)
        T_wall_max = T_wall * (1.0 + RECOVERY_TEMP_UNCERTAINTY)
        q_conv = 0.0
        q_rad  = 0.0
        q_total = 0.0
        T_wall_SG_raw = 0.0   # sentinel — SG not computed in this branch
        q_total_sl = 0.0

        # Sea-level worst case: recovery temperature at sea-level ambient temperature
        T_sl, _, _ = _isa_atmosphere(0.0)
        T_wall_sl = T_sl * (1.0 + RECOVERY_FACTOR * (GAMMA_AIR - 1.0) / 2.0 * mach ** 2)

        uses_recovery = True

    else:
        # --- Sutton-Graves + Tauber-Sutton Model (M >= 5) ---
        # SG convective correlation is valid for hypersonic stagnation-point heating.
        # No recovery cap needed — SG is the appropriate model in this regime.
        q_conv = _sutton_graves(rho, V, R_n)
        q_rad  = _tauber_sutton_rad(rho, V, R_n)
        q_total = q_conv + q_rad

        T_wall_SG_raw = _radiation_equilibrium_wall_temperature(q_total, epsilon)
        T_wall     = T_wall_SG_raw
        T_wall_min = _radiation_equilibrium_wall_temperature(
            q_total * (1.0 - SUTTON_GRAVES_UNCERTAINTY), epsilon
        )
        T_wall_max = _radiation_equilibrium_wall_temperature(
            q_total * (1.0 + SUTTON_GRAVES_UNCERTAINTY), epsilon
        )

        # Sea-level worst case (same velocity, sea-level density)
        T_sl, P_sl, rho_sl = _isa_atmosphere(0.0)
        q_conv_sl  = _sutton_graves(rho_sl, V, R_n)
        q_rad_sl   = _tauber_sutton_rad(rho_sl, V, R_n)
        q_total_sl = q_conv_sl + q_rad_sl
        T_wall_sl  = _radiation_equilibrium_wall_temperature(q_total_sl, epsilon)

        uses_recovery = False

    return ThermalResults(
        velocity_ms=V,
        q_conv_Wm2=q_conv,
        q_rad_Wm2=q_rad,
        q_total_Wm2=q_total,
        T_wall_K=T_wall,
        T_wall_min_K=T_wall_min,
        T_wall_max_K=T_wall_max,
        T_wall_SG_uncapped_K=T_wall_SG_raw,
        T_stag_K=T_stag,
        T_ambient_K=T_amb,
        q_total_sealevel_Wm2=q_total_sl,
        T_wall_sealevel_K=T_wall_sl,
        plasma_sheath=plasma_sheath,
        uses_recovery_model=uses_recovery,
        T_stag_real_gas_suspect=T_stag > REAL_GAS_T_STAG_THRESHOLD_K,
        plasma_threshold_slender=plasma_threshold_slender,
    )


def _estimate_characteristic_length(mass_kg: float) -> float:
    """Estimate vehicle length from mass assuming cylindrical body with ref density."""
    vol = mass_kg / RHO_VEHICLE_REF
    return (vol / (math.pi / 4.0)) ** (1.0 / 3.0)


def _compute_structural(
    mass: float,
    g_load: float,
    rho_cruise: float,
    V: float,
    T_wall_K: float,
    T_ambient_K: float,
    L: float,
) -> StructuralResults:
    A_ref = math.pi * (L / 2.0) ** 2
    F_inertial = mass * g_load * G0
    q_dyn = 0.5 * rho_cruise * V ** 2

    sigma_inertial_MPa = (F_inertial / A_ref) / 1e6
    sigma_combined_MPa = (sigma_inertial_MPa + q_dyn / 1e6) * STRUCTURAL_SAFETY_FACTOR

    delta_T = max(0.0, T_wall_K - T_ambient_K)
    sigma_thermal_ref_MPa = THERMAL_RELIEF_FACTOR * E_REF_MPA * ALPHA_REF * delta_T

    sigma_tensile_required_MPa = sigma_combined_MPa + sigma_thermal_ref_MPa

    return StructuralResults(
        F_inertial_N=F_inertial,
        q_dyn_Pa=q_dyn,
        A_ref_m2=A_ref,
        sigma_inertial_MPa=sigma_inertial_MPa,
        sigma_combined_MPa=sigma_combined_MPa,
        sigma_thermal_ref_MPa=sigma_thermal_ref_MPa,
        sigma_tensile_required_MPa=sigma_tensile_required_MPa,
        characteristic_length_m=L,
    )


def _compute_propulsion(
    mass: float,
    g_load: float,
    V: float,
    duration: float,
) -> PropulsionResults:
    KE = 0.5 * mass * V ** 2
    F_thrust = mass * g_load * G0
    P_peak = F_thrust * V
    E_total = P_peak * duration

    return PropulsionResults(
        KE_J=KE,
        P_peak_W=P_peak,
        E_total_J=E_total,
        fuel_mass_kerosene_kg=E_total / ENERGY_DENSITY_KEROSENE,
        fuel_mass_LH2_kg=E_total / ENERGY_DENSITY_LH2,
    )


def _compute_em(R_n: float, T_wall: float, epsilon: float, plasma: bool,
                plasma_slender: bool = False) -> EMResults:
    A_surface = 4.0 * math.pi * R_n ** 2
    P_rad = epsilon * SIGMA_SB * A_surface * T_wall ** 4

    if T_wall > 0.0:
        lambda_peak_um = 2897.8 / T_wall
    else:
        lambda_peak_um = float("inf")

    if lambda_peak_um < 3.0:
        band = "near-IR"
    elif lambda_peak_um < 5.0:
        band = "mid-wave IR"
    elif lambda_peak_um < 12.0:
        band = "long-wave IR"
    else:
        band = "far-IR"

    return EMResults(
        P_rad_W=P_rad,
        lambda_peak_um=lambda_peak_um,
        emission_band=band,
        plasma_sheath=plasma,
        plasma_threshold_slender=plasma_slender,
    )


def _validate_inputs(
    mach: float,
    alt_km: float,
    R_n: float,
    q_dyn: float,
) -> list:
    warnings = []
    if q_dyn > Q_DYN_WARN_PA:
        warnings.append("Dynamic pressure exceeds known structural limits")
    if mach > 25.0:
        warnings.append("Outside validated range")
    if alt_km > 86.0:
        warnings.append("ISA model not valid above 86 km")
    if R_n < 0.001:
        warnings.append("Below manufacturing minimum nose radius")
    return warnings


def _classify_regime(mach: float, alt_km: float) -> str:
    if alt_km > 80.0 and mach > 15.0:
        return "reentry"
    if mach > 25.0:
        return "reentry"
    if mach >= 5.0:
        return "hypersonic"
    if mach >= 0.8:
        return "supersonic"
    return "subsonic"


# ---------------------------------------------------------------------------
# Quick read-only envelope summary (UI helper)
# ---------------------------------------------------------------------------

def envelope_summary(
    mach: float,
    alt_km: float,
    *,
    R_n_m: float = 0.30,
    epsilon: float = 0.85,
) -> dict:
    """Compute a small set of envelope-derived quantities for the UI.

    Pure read-only — calls ``_isa_atmosphere`` and (for M ≥ 5) the same
    Sutton-Graves convective + Tauber-Sutton radiative + radiation-
    equilibrium chain that ``run_analysis`` uses. Numbers are
    guaranteed consistent with the full pipeline.

    Used by the Streamlit form's right-hand "Computed" panel to give
    the engineer a real-time read on the envelope without paying for
    the full thermal+structural+EM pipeline on every keystroke.

    Parameters
    ----------
    mach, alt_km : float
        Flight envelope.
    R_n_m : float
        Nose-radius — needed for Sutton-Graves at M ≥ 5. Ignored at M < 5.
    epsilon : float
        Wall emissivity — needed for radiation-equilibrium at M ≥ 5.
        Ignored at M < 5.

    Returns a dict with:
      * ``a_ms``                 — speed of sound at altitude (m/s)
      * ``velocity_ms``          — freestream velocity (m/s)
      * ``T_ambient_K``          — ISA ambient temperature (K)
      * ``rho_kgm3``             — ISA ambient density (kg/m³)
      * ``q_inf_Pa``             — freestream dynamic pressure ½ρV² (Pa)
      * ``T_wall_K``             — predicted wall temperature using the
                                   same model as ``run_analysis``: turbulent
                                   recovery for M < 5; Sutton-Graves
                                   radiation-equilibrium for M ≥ 5. This
                                   is the number to display in the UI.
      * ``T_wall_model``         — "recovery" or "sutton_graves" — string
                                   tag identifying which model produced
                                   ``T_wall_K``, suitable for a label.
      * ``T_recovery_K``         — turbulent flat-plate recovery
                                   temperature (provided regardless of
                                   regime as a reference).
      * ``T_stagnation_K``       — adiabatic stagnation temperature (the
                                   theoretical maximum)
      * ``flight_regime``        — one of "subsonic" / "supersonic" /
                                   "hypersonic" / "reentry"
    """
    m = float(mach)
    a = float(alt_km)
    R_n = float(R_n_m)
    eps = float(epsilon)
    T_amb, _P, rho = _isa_atmosphere(a)
    a_ms = math.sqrt(GAMMA_AIR * R_AIR * T_amb)
    V = m * a_ms
    q_inf = 0.5 * rho * V * V
    T_stag = T_amb * (1.0 + (GAMMA_AIR - 1.0) / 2.0 * m * m)
    T_rec = T_amb * (1.0 + RECOVERY_FACTOR * (GAMMA_AIR - 1.0) / 2.0 * m * m)

    # M < 5 → recovery temperature is the realistic wall T (the run_analysis
    # branch). M ≥ 5 → recovery formula explodes (T_amb * 83 at M=22 = 18,000 K
    # which is non-physical); switch to Sutton-Graves convective + Tauber-
    # Sutton radiative + radiation-equilibrium, same as run_analysis's
    # hypersonic branch. R_n and epsilon now matter — caller passes them.
    if m < 5.0:
        T_wall = T_rec
        T_wall_model = "recovery"
    else:
        q_conv = _sutton_graves(rho, V, R_n)
        q_rad  = _tauber_sutton_rad(rho, V, R_n)
        T_wall = _radiation_equilibrium_wall_temperature(q_conv + q_rad, eps)
        T_wall_model = "sutton_graves"

    return {
        "a_ms":           a_ms,
        "velocity_ms":    V,
        "T_ambient_K":    T_amb,
        "rho_kgm3":       rho,
        "q_inf_Pa":       q_inf,
        "T_wall_K":       T_wall,
        "T_wall_model":   T_wall_model,
        "T_recovery_K":   T_rec,
        "T_stagnation_K": T_stag,
        "flight_regime":  _classify_regime(m, a),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_analysis(
    peak_mach: float,
    cruise_altitude_km: float,
    vehicle_mass_kg: float,
    nose_radius_m: float,
    *,
    peak_g_load: float = 1.0,
    characteristic_length_m: float | None = None,
    flight_duration_s: float = 600.0,
    wall_emissivity: float = 0.85,
) -> PhysicsResult:
    """
    Run the full MATVEC physics analysis for the given flight envelope.

    Parameters
    ----------
    peak_mach              : Peak flight Mach number
    cruise_altitude_km     : Cruise altitude in kilometres
    vehicle_mass_kg        : Vehicle mass in kg
    nose_radius_m          : Nose/leading-edge radius in metres
    peak_g_load            : Peak structural load factor (default 1 G)
    characteristic_length_m: Vehicle body length; estimated from mass if None
    flight_duration_s      : Mission duration in seconds (default 10 min)
    wall_emissivity        : Wall thermal emissivity (default 0.85)

    Returns
    -------
    PhysicsResult with atmosphere, thermal, structural, propulsion, EM sub-results
    """
    # --- atmosphere at cruise altitude ---
    T_K, P_Pa, rho = _isa_atmosphere(cruise_altitude_km)
    atm = AtmosphericConditions(
        altitude_km=cruise_altitude_km,
        temperature_K=T_K,
        pressure_Pa=P_Pa,
        density_kgm3=rho,
    )

    # --- thermal branch ---
    thermal = _compute_thermal(peak_mach, cruise_altitude_km, nose_radius_m, wall_emissivity, atm)

    # --- characteristic length ---
    L = characteristic_length_m if characteristic_length_m is not None else \
        _estimate_characteristic_length(vehicle_mass_kg)

    # --- structural branch ---
    structural = _compute_structural(
        vehicle_mass_kg,
        peak_g_load,
        rho,
        thermal.velocity_ms,
        thermal.T_wall_K,
        T_K,
        L,
    )

    # --- propulsion branch ---
    propulsion = _compute_propulsion(
        vehicle_mass_kg,
        peak_g_load,
        thermal.velocity_ms,
        flight_duration_s,
    )

    # --- EM branch ---
    em = _compute_em(
        nose_radius_m,
        thermal.T_wall_K,
        wall_emissivity,
        thermal.plasma_sheath,
        plasma_slender=thermal.plasma_threshold_slender,
    )

    # --- regime + validation ---
    regime = _classify_regime(peak_mach, cruise_altitude_km)
    warnings = _validate_inputs(peak_mach, cruise_altitude_km, nose_radius_m, structural.q_dyn_Pa)
    if thermal.T_stag_real_gas_suspect:
        warnings.append(
            f"T_stag = {thermal.T_stag_K:.0f} K exceeds {REAL_GAS_T_STAG_THRESHOLD_K:.0f} K — "
            f"calorically perfect gas assumption invalid (dissociation/ionization absorb energy); "
            f"true stagnation temperature is substantially lower"
        )

    return PhysicsResult(
        peak_mach=peak_mach,
        cruise_altitude_km=cruise_altitude_km,
        vehicle_mass_kg=vehicle_mass_kg,
        nose_radius_m=nose_radius_m,
        peak_g_load=peak_g_load,
        characteristic_length_m=L,
        flight_duration_s=flight_duration_s,
        wall_emissivity=wall_emissivity,
        atmosphere=atm,
        thermal=thermal,
        structural=structural,
        propulsion=propulsion,
        em=em,
        flight_regime=regime,
        warnings=warnings,
    )
