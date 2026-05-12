"""
MATVEC Materials Database
=========================
Aerospace materials feasibility tool — materials data module.

All values are in SI units:
  Temperatures : Kelvin (K)
  Stress/Strength : MPa
  Density : kg/m³
  Thermal conductivity : W/m·K
  CTE : 1/K
  Young's modulus : GPa

Primary sources:
  ASM Handbook Vol. 1 (Ferrous Alloys) and Vol. 2 (Nonferrous Alloys)
  MMPDS-17 (Metallic Materials Properties Development and Standardization)
  Special Metals technical bulletins (Inconel series)
  Haynes International alloy data sheets
  NASA TPSX Materials Database (tpsx.arc.nasa.gov)
  NASA NTRS technical reports
  Fahrenholtz & Hilmas, J. Am. Ceram. Soc. 2012 (UHTCs)
  OSTI 887260 (Kasen et al. UHTC review)
  RTI International Titanium Alloy Guide (2000)
  Plansee refractory metal technical data

Version: 1.0.0
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CATEGORIES = frozenset({
    "aluminum",
    "titanium",
    "steel",
    "nickel",
    "cobalt",
    "refractory",
    "composite_polymer",
    "composite_ceramic",
    "uhtc",
    "tps",
    "carbon",
    "general_engineering",
})

VALID_REGIMES = frozenset({
    "subsonic",
    "supersonic",
    "hypersonic",
    "reentry",
})


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class MaterialEntry:
    """
    Single material entry in the MATVEC database.

    tensile_strength_at_temp keys/values are all float (Kelvin → MPa).
    For brittle ceramics (uhtc, composite_ceramic, carbon), tensile_strength_mpa
    and tensile_strength_at_temp store flexural strength (MOR) as a proxy,
    which is noted in the 'notes' field.
    For TPS ablators, service_temp_air_K and service_temp_inert_K represent
    the rated surface temperature as an ablative barrier, not structural service.
    melting_point_K is set to decomposition/sublimation temperature for
    non-metals (polymers ~673 K, carbon/graphite ~3823 K).
    """

    name: str
    category: str
    density_kgm3: float
    tensile_strength_mpa: float
    tensile_strength_at_temp: dict          # {float: float}  K → MPa
    compressive_strength_mpa: float
    service_temp_air_K: float
    service_temp_inert_K: float
    melting_point_K: float
    thermal_conductivity_WmK: float
    thermal_expansion_1K: float
    youngs_modulus_GPa: float
    oxidation_resistance: str               # "excellent"/"good"/"limited"/"poor"
    oxidation_max_temp_K: float
    applicable_regimes: list
    citation: str
    notes: str
    availability_score: float = 1.0   # 1.0 = commercial, 0.7 = limited, 0.5 = developmental, 0.3 = lab
    coated_max_temp_K: float = 0.0    # Effective air-service ceiling with a mature protective coating
                                      # (silicide/aluminide for Nb/Mo, HfC-SiC for Ta/W, Ir for Re).
                                      # 0.0 = no coating data; matching engine falls back to bare ceiling.
    cost_usd_per_kg: float = 0.0      # Order-of-magnitude bulk price (USD/kg, 2025-26 market).
                                      # Screening-grade only (+/- 50%; actual quotes vary 10x with form,
                                      # quantity, and market conditions). 0.0 is reserved for exotic/2D
                                      # materials that only exist for impossibility detection and must
                                      # never surface a price in user-facing output.

    # -------------------------------------------------------------------
    # Creep / Larson-Miller fields (lifecycle modelling — phase 1 of the
    # creep rollout). These are populated at module load by
    # ``_apply_creep_data()`` for the priority materials, OR set in bulk
    # by category-level rules in ``_apply_category_creep_rules()`` for
    # categories that don't classically creep (TPS, ceramics, CFRP).
    # Materials not covered by either pathway stay at the defaults
    # below — ``creep_data_status="unknown"`` — and are surfaced in the
    # matching engine as a flag rather than an auto-reject.
    # -------------------------------------------------------------------
    larson_miller_C: float | None = None
    """Material-specific Larson-Miller constant. ``LMP = T·(C + log10(t))``
    with T in K and t in hours. Typical values: 13-17 for aluminum,
    17-20 for titanium / steel, 20-25 for nickel superalloys."""

    lmp_curve: tuple = ()
    """Piecewise-linear (LMP, rupture_stress_MPa) data points sorted
    ascending by LMP. Empty tuple when no creep curve is sourced.
    Linear interpolation in log-stress space is performed by
    ``core/creep.py``; extrapolation outside the curve range is flagged
    rather than silently extended."""

    creep_data_source: str = ""
    """Citation for the LMP data. Empty when ``creep_data_status`` is
    not ``"sourced"``."""

    creep_data_status: str = "unknown"
    """One of: ``"sourced"`` (LMP curve from a primary reference),
    ``"estimated"`` (curve is an engineering estimate without a primary
    source — flag in UI), ``"not_applicable"`` (material category does
    not classically creep at the relevant temperatures: TPS ablators,
    polymer composites, ceramics below ~0.5*Tm), or ``"unknown"`` (no
    data and no category rule — matching engine treats as a marginal
    flag, never auto-rejects)."""

    # -------------------------------------------------------------------
    # Specific heat capacity (Phase 7 of the lifecycle / transient-heat
    # rollout). Populated at module load by ``_apply_cp_data()`` for the
    # priority materials, in bulk by category-level rules for materials
    # whose c_p value at typical service temperatures is well-established
    # in handbook tables, and otherwise left at the default unknown.
    # Used by ``core/transient_heat.py`` to compute thermal diffusivity
    # α = k / (ρ · c_p).
    # -------------------------------------------------------------------
    specific_heat_J_kgK: float | None = None
    """Specific heat capacity at constant pressure, J/(kg·K), at room
    temperature (293 K). For temperature-dependent c_p (typical of
    polymers and some refractory alloys) the value here is the room-T
    anchor; a higher-fidelity solver would interpolate a c_p(T) curve
    in a future round."""

    cp_data_source: str = ""
    """Citation for the c_p value. Empty when ``cp_data_status`` is
    not ``"sourced"``."""

    cp_data_status: str = "unknown"
    """One of: ``"sourced"`` (c_p from a primary reference),
    ``"estimated"`` (engineering estimate from a related material),
    ``"not_applicable"`` (TPS ablators — mass-loss models, not classical
    conduction), or ``"unknown"`` (no data sourced; transient-heat
    solver will skip the material with a flag)."""

    def __post_init__(self):
        if self.oxidation_resistance not in ("excellent", "good", "limited", "poor"):
            raise ValueError(
                f"{self.name}: oxidation_resistance '{self.oxidation_resistance}' "
                f"must be 'excellent', 'good', 'limited', or 'poor'"
            )
        for r in self.applicable_regimes:
            if r not in VALID_REGIMES:
                raise ValueError(f"{self.name}: unknown regime '{r}'")
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"{self.name}: unknown category '{self.category}'")
        if self.service_temp_air_K > self.service_temp_inert_K:
            raise ValueError(
                f"{self.name}: service_temp_air_K ({self.service_temp_air_K}) "
                f"cannot exceed service_temp_inert_K ({self.service_temp_inert_K})"
            )
        if len(self.tensile_strength_at_temp) < 3:
            raise ValueError(
                f"{self.name}: tensile_strength_at_temp requires >= 3 data points, "
                f"got {len(self.tensile_strength_at_temp)}"
            )
        # Creep-data-status enum check: cheap guard against typos in the
        # priority-list patch table below. The four valid values mirror
        # the docstring on the field.
        if self.creep_data_status not in (
            "sourced", "estimated", "not_applicable", "unknown",
        ):
            raise ValueError(
                f"{self.name}: creep_data_status "
                f"'{self.creep_data_status}' must be one of "
                f"'sourced', 'estimated', 'not_applicable', 'unknown'"
            )
        # Specific-heat-status enum check (Phase 7 mirror of the creep
        # status field).
        if self.cp_data_status not in (
            "sourced", "estimated", "not_applicable", "unknown",
        ):
            raise ValueError(
                f"{self.name}: cp_data_status "
                f"'{self.cp_data_status}' must be one of "
                f"'sourced', 'estimated', 'not_applicable', 'unknown'"
            )
        if self.cp_data_status in ("sourced", "estimated"):
            if self.specific_heat_J_kgK is None:
                raise ValueError(
                    f"{self.name}: cp_data_status="
                    f"'{self.cp_data_status}' requires a non-None "
                    "specific_heat_J_kgK"
                )
            if self.specific_heat_J_kgK <= 0.0:
                raise ValueError(
                    f"{self.name}: specific_heat_J_kgK must be > 0, "
                    f"got {self.specific_heat_J_kgK}"
                )
        # When status is "sourced" or "estimated" the curve must be
        # populated and monotonic in LMP. ``not_applicable`` and
        # ``unknown`` keep the empty default.
        if self.creep_data_status in ("sourced", "estimated"):
            if not self.lmp_curve or len(self.lmp_curve) < 2:
                raise ValueError(
                    f"{self.name}: creep_data_status="
                    f"'{self.creep_data_status}' requires a non-empty "
                    "lmp_curve with at least 2 points"
                )
            lmps = [pt[0] for pt in self.lmp_curve]
            if lmps != sorted(lmps):
                raise ValueError(
                    f"{self.name}: lmp_curve LMP values must be sorted "
                    f"ascending; got {lmps}"
                )


# ===========================================================================
# ALUMINUM ALLOYS
# applicable_regimes: subsonic, supersonic
# ===========================================================================

AL_2024_T3 = MaterialEntry(
    name="2024-T3",
    category="aluminum",
    density_kgm3=2780.0,
    tensile_strength_mpa=483.0,
    tensile_strength_at_temp={
        293.0: 483.0,
        373.0: 420.0,
        422.0: 280.0,
        478.0: 90.0,
    },
    compressive_strength_mpa=345.0,
    service_temp_air_K=423.0,
    service_temp_inert_K=450.0,
    melting_point_K=911.0,
    thermal_conductivity_WmK=121.0,
    thermal_expansion_1K=23.2e-6,
    youngs_modulus_GPa=73.1,
    oxidation_resistance="good",
    oxidation_max_temp_K=423.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="MMPDS-17 Table 3.2.2.0; ASM Handbook Vol. 2",
    notes="Primary aircraft structure; fatigue-critical applications; bare sheet and clad forms",
    cost_usd_per_kg=5.0,  # cost ~$5/kg - 2024-T3 bulk plate/sheet, common aerospace aluminum
)

AL_7075_T6 = MaterialEntry(
    name="7075-T6",
    category="aluminum",
    density_kgm3=2810.0,
    tensile_strength_mpa=572.0,
    tensile_strength_at_temp={
        293.0: 572.0,
        366.0: 490.0,
        422.0: 345.0,
        478.0: 165.0,
    },
    compressive_strength_mpa=503.0,
    service_temp_air_K=394.0,
    service_temp_inert_K=423.0,
    melting_point_K=908.0,
    thermal_conductivity_WmK=130.0,
    thermal_expansion_1K=23.6e-6,
    youngs_modulus_GPa=71.7,
    oxidation_resistance="good",
    oxidation_max_temp_K=394.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="MMPDS-17 Table 3.2.4.0; MatWeb MA7075T6",
    notes="High-strength aircraft structure; F-16 skins; susceptible to SCC; not weldable",
    cost_usd_per_kg=5.0,  # cost ~$5/kg - 7075-T6 bulk plate/sheet, common aerospace aluminum
)

AL_7068_T6511 = MaterialEntry(
    name="7068-T6511",
    category="aluminum",
    density_kgm3=2850.0,
    tensile_strength_mpa=683.0,
    tensile_strength_at_temp={
        293.0: 683.0,
        366.0: 580.0,
        422.0: 410.0,
        478.0: 195.0,
    },
    compressive_strength_mpa=634.0,
    service_temp_air_K=394.0,
    service_temp_inert_K=423.0,
    melting_point_K=908.0,
    thermal_conductivity_WmK=125.0,
    thermal_expansion_1K=23.4e-6,
    youngs_modulus_GPa=72.4,
    oxidation_resistance="good",
    oxidation_max_temp_K=394.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="ASM Alloy Center; Alcan technical bulletin 7068",
    notes="Highest-strength aluminum alloy; newer aircraft programs; extrusion and plate",
    cost_usd_per_kg=15.0,  # cost ~$15/kg - 7068 premium high-strength 7xxx, limited supply
)

AL_6061_T6 = MaterialEntry(
    name="6061-T6",
    category="aluminum",
    density_kgm3=2700.0,
    tensile_strength_mpa=310.0,
    tensile_strength_at_temp={
        293.0: 310.0,
        366.0: 260.0,
        422.0: 185.0,
        478.0: 75.0,
    },
    compressive_strength_mpa=276.0,
    service_temp_air_K=423.0,
    service_temp_inert_K=450.0,
    melting_point_K=925.0,
    thermal_conductivity_WmK=167.0,
    thermal_expansion_1K=23.6e-6,
    youngs_modulus_GPa=68.9,
    oxidation_resistance="good",
    oxidation_max_temp_K=423.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="ASM Handbook Vol. 2; MatWeb MA6061T6",
    notes="General aerospace structure; good weldability; anodizes well; lower strength than 7xxx",
    cost_usd_per_kg=4.0,  # cost ~$4/kg - 6061-T6 is the commodity aluminum alloy
)

AL_2195_ALRLI = MaterialEntry(
    name="2195 Al-Li",
    category="aluminum",
    density_kgm3=2710.0,
    tensile_strength_mpa=586.0,
    tensile_strength_at_temp={
        293.0: 586.0,
        366.0: 490.0,
        422.0: 330.0,
        478.0: 110.0,
    },
    compressive_strength_mpa=517.0,
    service_temp_air_K=423.0,
    service_temp_inert_K=450.0,
    melting_point_K=904.0,
    thermal_conductivity_WmK=110.0,
    thermal_expansion_1K=22.7e-6,
    youngs_modulus_GPa=78.5,
    oxidation_resistance="good",
    oxidation_max_temp_K=423.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="NASA TM-110395; Aluminum-Lithium Alloys: Processing, Properties and Applications (2013)",
    notes="Space Shuttle Super Lightweight Tank (SLT); 5% lower density than 2219; cryogenic tanks; T8 temper",
    cost_usd_per_kg=15.0,  # cost ~$15/kg - Al-Li 2195 premium cryo-tank alloy
)

AL_2099_ALRLI = MaterialEntry(
    name="2099 Al-Li",
    category="aluminum",
    density_kgm3=2630.0,
    tensile_strength_mpa=524.0,
    tensile_strength_at_temp={
        293.0: 524.0,
        366.0: 430.0,
        422.0: 295.0,
        478.0: 95.0,
    },
    compressive_strength_mpa=476.0,
    service_temp_air_K=423.0,
    service_temp_inert_K=450.0,
    melting_point_K=899.0,
    thermal_conductivity_WmK=105.0,
    thermal_expansion_1K=22.5e-6,
    youngs_modulus_GPa=80.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=423.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Alcoa technical report; Williams & Starke, Acta Materialia 51 (2003)",
    notes="T83 extrusion; replaces 7050 in fuselage lower skins; improved corrosion resistance; lower density",
    cost_usd_per_kg=15.0,  # cost ~$15/kg - Al-Li 2099 extruded shapes, fuselage stringers
)

AL_2219_T87 = MaterialEntry(
    name="2219-T87",
    category="aluminum",
    density_kgm3=2840.0,
    tensile_strength_mpa=455.0,
    tensile_strength_at_temp={
        293.0: 455.0,
        366.0: 380.0,
        422.0: 275.0,
        478.0: 115.0,
    },
    compressive_strength_mpa=415.0,
    service_temp_air_K=422.0,
    service_temp_inert_K=450.0,
    melting_point_K=916.0,
    thermal_conductivity_WmK=121.0,
    thermal_expansion_1K=22.3e-6,
    youngs_modulus_GPa=73.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=422.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Alcoa 2219 technical data; MMPDS-17; NASA TN D-8110",
    notes="Weldable Al-Cu alloy; Saturn V/S-IC LOX tanks, SLS core stage; excellent cryogenic toughness; T87 = solution treated + cold worked + aged",
    cost_usd_per_kg=8.0,  # cost ~$8/kg - 2219 cryo/shuttle ET, specialty but common
)

AL_7050_T7451 = MaterialEntry(
    name="7050-T7451",
    category="aluminum",
    density_kgm3=2830.0,
    tensile_strength_mpa=524.0,
    tensile_strength_at_temp={
        293.0: 524.0,
        366.0: 435.0,
        422.0: 295.0,
        478.0: 100.0,
    },
    compressive_strength_mpa=490.0,
    service_temp_air_K=423.0,
    service_temp_inert_K=450.0,
    melting_point_K=908.0,
    thermal_conductivity_WmK=157.0,
    thermal_expansion_1K=23.4e-6,
    youngs_modulus_GPa=72.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=423.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Alcoa 7050 plate data; MMPDS-17; AMS 4050",
    notes="Thick-section plate alloy; 747/777/787 wing spars and ribs; T7451 overaged for SCC resistance; replaced 7075 in many structural applications",
    cost_usd_per_kg=7.0,  # cost ~$7/kg - 7050 aerospace 7xxx plate premium
)

AL_7010_T7451 = MaterialEntry(
    name="7010-T7451",
    category="aluminum",
    density_kgm3=2820.0,
    tensile_strength_mpa=505.0,
    tensile_strength_at_temp={
        293.0: 505.0,
        366.0: 420.0,
        422.0: 285.0,
        478.0: 95.0,
    },
    compressive_strength_mpa=470.0,
    service_temp_air_K=423.0,
    service_temp_inert_K=450.0,
    melting_point_K=908.0,
    thermal_conductivity_WmK=155.0,
    thermal_expansion_1K=23.6e-6,
    youngs_modulus_GPa=72.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=423.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Aluminium Pechiney 7010 data sheet; MMPDS-17; British Standard 2L95",
    notes="European thick-plate alloy; Airbus A330/A340 wing spars; similar to 7050 with slightly different Cu/Mg balance",
    cost_usd_per_kg=7.0,  # cost ~$7/kg - 7010 aerospace 7xxx plate, EU aircraft programs
)

AL_8090_ALLI = MaterialEntry(
    name="8090 Al-Li",
    category="aluminum",
    density_kgm3=2540.0,
    tensile_strength_mpa=480.0,
    tensile_strength_at_temp={
        293.0: 480.0,
        366.0: 400.0,
        422.0: 275.0,
        478.0: 95.0,
    },
    compressive_strength_mpa=440.0,
    service_temp_air_K=423.0,
    service_temp_inert_K=450.0,
    melting_point_K=918.0,
    thermal_conductivity_WmK=95.0,
    thermal_expansion_1K=22.0e-6,
    youngs_modulus_GPa=80.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=423.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Alcan Aerospace 8090 data sheet; EAA-AECMA standard EN 2089",
    notes="First-gen Al-Li alloy; 10% lower density, 11% higher stiffness than 2024; EH101 helicopter, Westland Lynx; delamination sensitivity limits thick-section use",
    cost_usd_per_kg=15.0,  # cost ~$15/kg - Al-Li 8090 first-generation lithium alloy
)


# ===========================================================================
# TITANIUM ALLOYS
# applicable_regimes: subsonic, supersonic (up to Mach ~3 / ~600 K aero-heating)
# ===========================================================================

TI_6AL4V = MaterialEntry(
    name="Ti-6Al-4V",
    category="titanium",
    density_kgm3=4430.0,
    tensile_strength_mpa=1000.0,
    tensile_strength_at_temp={
        293.0: 1000.0,
        589.0: 700.0,
        755.0: 300.0,
        866.0: 100.0,
    },
    compressive_strength_mpa=970.0,
    service_temp_air_K=625.0,
    service_temp_inert_K=720.0,
    melting_point_K=1933.0,
    thermal_conductivity_WmK=7.2,
    thermal_expansion_1K=8.6e-6,
    youngs_modulus_GPa=114.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=625.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="ASM Handbook Vol. 2; MMPDS-17 Ti-6Al-4V tables; RTI Titanium Alloy Guide 2000",
    notes="The workhorse titanium alloy; F-22 airframe (39% by weight); most widely used aerospace titanium; Grade 5 per AMS 4928; alpha-case formation limits air use to ~352°C (625 K) per conservative service ceiling",
    cost_usd_per_kg=45.0,  # cost ~$45/kg - Ti-6Al-4V bulk plate, 2025 market rate (workhorse)
)

TI_6242 = MaterialEntry(
    name="Ti-6Al-2Sn-4Zr-2Mo",
    category="titanium",
    density_kgm3=4540.0,
    tensile_strength_mpa=1100.0,
    tensile_strength_at_temp={
        293.0: 1100.0,
        589.0: 770.0,
        755.0: 330.0,
        866.0: 110.0,
    },
    compressive_strength_mpa=1000.0,
    service_temp_air_K=823.0,
    service_temp_inert_K=870.0,
    melting_point_K=1943.0,
    thermal_conductivity_WmK=6.8,
    thermal_expansion_1K=8.7e-6,
    youngs_modulus_GPa=117.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=823.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="ASM Handbook Vol. 2; RTI Titanium Alloy Guide 2000; AMS 4919",
    notes="High creep resistance to 550°C; turbine compressor discs; alpha-beta alloy; better elevated-temp properties than Ti-6-4",
    cost_usd_per_kg=80.0,  # cost ~$80/kg - Ti-6242 elevated-T premium alpha-beta
)

TI_3AL25V = MaterialEntry(
    name="Ti-3Al-2.5V",
    category="titanium",
    density_kgm3=4480.0,
    tensile_strength_mpa=620.0,
    tensile_strength_at_temp={
        293.0: 620.0,
        589.0: 434.0,
        755.0: 186.0,
        866.0: 62.0,
    },
    compressive_strength_mpa=585.0,
    service_temp_air_K=533.0,
    service_temp_inert_K=600.0,
    melting_point_K=1928.0,
    thermal_conductivity_WmK=7.5,
    thermal_expansion_1K=8.9e-6,
    youngs_modulus_GPa=105.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=533.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="ASTM B338; RTI Titanium Alloy Guide 2000",
    notes="Grade 9; hydraulic tubing, cryogenic tanks; weldable; lower strength than Ti-6-4 but excellent formability",
    cost_usd_per_kg=45.0,  # cost ~$45/kg - Ti-3Al-2.5V tube grade, SR-71 fuel lines
)

TI_15V_3 = MaterialEntry(
    name="Ti-15V-3Cr-3Al-3Sn",
    category="titanium",
    density_kgm3=4760.0,
    tensile_strength_mpa=1000.0,
    tensile_strength_at_temp={
        293.0: 1000.0,
        589.0: 700.0,
        755.0: 300.0,
        866.0: 100.0,
    },
    compressive_strength_mpa=960.0,
    service_temp_air_K=589.0,
    service_temp_inert_K=650.0,
    melting_point_K=1883.0,
    thermal_conductivity_WmK=7.9,
    thermal_expansion_1K=8.8e-6,
    youngs_modulus_GPa=96.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=589.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="RTI Titanium Alloy Guide 2000; MMPDS-17; AMS 4914",
    notes="Cold-rollable metastable beta alloy; aerospace sheet and fasteners; age-hardenable; B-1B Lancer empennage",
    cost_usd_per_kg=80.0,  # cost ~$80/kg - Ti-15-3 beta titanium, cold-formable premium
)

TI_10V_2FE_3AL = MaterialEntry(
    name="Ti-10V-2Fe-3Al",
    category="titanium",
    density_kgm3=4650.0,
    tensile_strength_mpa=1170.0,
    tensile_strength_at_temp={
        293.0: 1170.0,
        589.0: 819.0,
        755.0: 351.0,
        866.0: 117.0,
    },
    compressive_strength_mpa=1100.0,
    service_temp_air_K=573.0,
    service_temp_inert_K=630.0,
    melting_point_K=1878.0,
    thermal_conductivity_WmK=6.3,
    thermal_expansion_1K=8.6e-6,
    youngs_modulus_GPa=107.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=573.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="ASM Handbook Vol. 2; Boeing proprietary data summarized in MMPDS-17; AMS 4984",
    notes="High-strength beta titanium; landing gear, bulkheads; forging alloy; 777 main landing gear beam",
    cost_usd_per_kg=80.0,  # cost ~$80/kg - Ti-10-2-3 beta titanium, forging premium
)

TI_6AL4V_ELI = MaterialEntry(
    name="Ti-6Al-4V ELI",
    category="titanium",
    density_kgm3=4430.0,
    tensile_strength_mpa=860.0,
    tensile_strength_at_temp={
        293.0: 860.0,
        589.0: 602.0,
        755.0: 258.0,
        866.0: 86.0,
    },
    compressive_strength_mpa=830.0,
    service_temp_air_K=590.0,
    service_temp_inert_K=700.0,
    melting_point_K=1933.0,
    thermal_conductivity_WmK=7.2,
    thermal_expansion_1K=8.6e-6,
    youngs_modulus_GPa=114.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=590.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="ASTM F136; MMPDS-17 Ti-6Al-4V ELI; AMS 4931",
    notes="Extra-low interstitial variant of Ti-6Al-4V; improved fracture toughness and fatigue; cryogenic applications; surgical implants and fracture-critical airframe",
    cost_usd_per_kg=55.0,  # cost ~$55/kg - Ti-6Al-4V ELI extra-low-interstitial premium
)

TI_5AL25SN = MaterialEntry(
    name="Ti-5Al-2.5Sn",
    category="titanium",
    density_kgm3=4480.0,
    tensile_strength_mpa=862.0,
    tensile_strength_at_temp={
        293.0: 862.0,
        589.0: 620.0,
        755.0: 310.0,
        866.0: 130.0,
    },
    compressive_strength_mpa=830.0,
    service_temp_air_K=755.0,
    service_temp_inert_K=810.0,
    melting_point_K=1875.0,
    thermal_conductivity_WmK=7.7,
    thermal_expansion_1K=9.4e-6,
    youngs_modulus_GPa=110.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=755.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="ASM Handbook Vol. 2; MMPDS-17; AMS 4926",
    notes="Alpha alloy; excellent weldability and creep resistance; cryogenic turbopump housings; jet engine compressor discs",
    cost_usd_per_kg=45.0,  # cost ~$45/kg - Ti-5-2.5 alpha titanium, cryo applications
)

TI_6AL6V2SN = MaterialEntry(
    name="Ti-6Al-6V-2Sn",
    category="titanium",
    density_kgm3=4540.0,
    tensile_strength_mpa=1100.0,
    tensile_strength_at_temp={
        293.0: 1100.0,
        533.0: 880.0,
        644.0: 660.0,
        755.0: 330.0,
    },
    compressive_strength_mpa=1050.0,
    service_temp_air_K=588.0,
    service_temp_inert_K=650.0,
    melting_point_K=1877.0,
    thermal_conductivity_WmK=6.6,
    thermal_expansion_1K=9.0e-6,
    youngs_modulus_GPa=113.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=588.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="ASM Handbook Vol. 2; MMPDS-17; AMS 4918",
    notes="High-strength alpha-beta alloy; rocket motor cases, airframe forgings; higher strength than Ti-6Al-4V",
    cost_usd_per_kg=55.0,  # cost ~$55/kg - Ti-6-6-2 alpha-beta premium over 6-4
)

TI_BETAC = MaterialEntry(
    name="Beta-C (Ti-3Al-8V-6Cr-4Mo-4Zr)",
    category="titanium",
    density_kgm3=4820.0,
    tensile_strength_mpa=1310.0,
    tensile_strength_at_temp={
        293.0: 1310.0,
        533.0: 1050.0,
        644.0: 790.0,
        755.0: 390.0,
    },
    compressive_strength_mpa=1250.0,
    service_temp_air_K=573.0,
    service_temp_inert_K=640.0,
    melting_point_K=1820.0,
    thermal_conductivity_WmK=7.8,
    thermal_expansion_1K=8.9e-6,
    youngs_modulus_GPa=103.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=573.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="TIMET TIMETAL Beta-C data sheet; MMPDS-17; AMS 4957",
    notes="Metastable beta alloy; springs, fasteners, high-strength wire; excellent cold formability; SR-71 hydraulic tubing",
    cost_usd_per_kg=80.0,  # cost ~$80/kg - Beta-C Ti-3-8-6-4-4 beta titanium, fastener premium
)


# ===========================================================================
# NICKEL SUPERALLOYS
# applicable_regimes: subsonic, supersonic, hypersonic (combustors, nozzles)
# ===========================================================================

IN718 = MaterialEntry(
    name="Inconel 718",
    category="nickel",
    density_kgm3=8190.0,
    tensile_strength_mpa=1375.0,
    tensile_strength_at_temp={
        293.0: 1375.0,
        811.0: 1100.0,
        1033.0: 760.0,
        1144.0: 414.0,
    },
    compressive_strength_mpa=1240.0,
    service_temp_air_K=980.0,
    service_temp_inert_K=1090.0,
    melting_point_K=1609.0,
    thermal_conductivity_WmK=11.4,
    thermal_expansion_1K=13.0e-6,
    youngs_modulus_GPa=200.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1255.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Special Metals INCONEL 718 bulletin SMC-045; NASA TM-2002-211394",
    notes="Most widely used superalloy; turbine disks and structural components to 980 K in air; gamma-double-prime strengthened; AMS 5663",
    cost_usd_per_kg=60.0,  # cost ~$60/kg - Inconel 718 wrought bar/plate, aerospace standard
)

IN625 = MaterialEntry(
    name="Inconel 625",
    category="nickel",
    density_kgm3=8440.0,
    tensile_strength_mpa=930.0,
    tensile_strength_at_temp={
        293.0: 930.0,
        811.0: 744.0,
        1033.0: 558.0,
        1144.0: 279.0,
    },
    compressive_strength_mpa=840.0,
    service_temp_air_K=1255.0,
    service_temp_inert_K=1366.0,
    melting_point_K=1623.0,
    thermal_conductivity_WmK=9.8,
    thermal_expansion_1K=12.8e-6,
    youngs_modulus_GPa=207.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1422.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Special Metals INCONEL 625 bulletin SMC-063",
    notes="Excellent corrosion and oxidation resistance; lower strength than 718; solid-solution strengthened; bellows and exhaust systems",
    cost_usd_per_kg=60.0,  # cost ~$60/kg - Inconel 625 wrought, marine/aerospace staple
)

IN_X750 = MaterialEntry(
    name="Inconel X-750",
    category="nickel",
    density_kgm3=8280.0,
    tensile_strength_mpa=1170.0,
    tensile_strength_at_temp={
        293.0: 1170.0,
        811.0: 936.0,
        1033.0: 702.0,
        1144.0: 351.0,
    },
    compressive_strength_mpa=1100.0,
    service_temp_air_K=1144.0,
    service_temp_inert_K=1200.0,
    melting_point_K=1672.0,
    thermal_conductivity_WmK=12.0,
    thermal_expansion_1K=12.6e-6,
    youngs_modulus_GPa=214.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1255.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Special Metals INCONEL X-750 bulletin SMC-067",
    notes="Springs and fasteners at high temperature; gas turbine exhaust components; AMS 5542/5598",
    cost_usd_per_kg=60.0,  # cost ~$60/kg - Inconel X-750 wrought, turbine disc/spring
)

WASPALOY = MaterialEntry(
    name="Waspaloy",
    category="nickel",
    density_kgm3=8190.0,
    tensile_strength_mpa=1275.0,
    tensile_strength_at_temp={
        293.0: 1275.0,
        811.0: 1020.0,
        1033.0: 765.0,
        1144.0: 383.0,
    },
    compressive_strength_mpa=1175.0,
    service_temp_air_K=1090.0,
    service_temp_inert_K=1200.0,
    melting_point_K=1672.0,
    thermal_conductivity_WmK=10.2,
    thermal_expansion_1K=12.7e-6,
    youngs_modulus_GPa=213.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1200.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Carpenter Technology Waspaloy data sheet; ASM Handbook Vol. 1 superalloys",
    notes="Turbine disks to 980 K; gamma-prime strengthened; JT8D and JT9D turbine applications; AMS 5544",
    cost_usd_per_kg=70.0,  # cost ~$70/kg - Waspaloy wrought, premium over IN718
)

RENE41 = MaterialEntry(
    name="Rene 41",
    category="nickel",
    density_kgm3=8250.0,
    tensile_strength_mpa=1420.0,
    tensile_strength_at_temp={
        293.0: 1420.0,
        811.0: 1136.0,
        1033.0: 852.0,
        1144.0: 426.0,
    },
    compressive_strength_mpa=1300.0,
    service_temp_air_K=1144.0,
    service_temp_inert_K=1255.0,
    melting_point_K=1644.0,
    thermal_conductivity_WmK=9.9,
    thermal_expansion_1K=12.3e-6,
    youngs_modulus_GPa=220.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1255.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Alloy Wire International Rene 41 data; High Temp Metals technical data",
    notes="Higher temperature capability than Waspaloy; 1090 K service; X-15 engine components; AMS 5545",
    cost_usd_per_kg=70.0,  # cost ~$70/kg - Rene 41 wrought, hot-structure applications
)

HAYNES230 = MaterialEntry(
    name="Haynes 230",
    category="nickel",
    density_kgm3=9050.0,
    tensile_strength_mpa=870.0,
    tensile_strength_at_temp={
        293.0: 870.0,
        811.0: 696.0,
        1033.0: 522.0,
        1144.0: 261.0,
    },
    compressive_strength_mpa=800.0,
    service_temp_air_K=1422.0,
    service_temp_inert_K=1478.0,
    melting_point_K=1644.0,
    thermal_conductivity_WmK=8.9,
    thermal_expansion_1K=13.1e-6,
    youngs_modulus_GPa=211.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1422.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Haynes International H-3120B data sheet",
    notes="Excellent oxidation resistance to 1149°C; combustion hardware, transition liners; sheet/plate/bar; AMS 5878",
    cost_usd_per_kg=70.0,  # cost ~$70/kg - Haynes 230 wrought premium, high-T oxidation
)

HAYNES282 = MaterialEntry(
    name="Haynes 282",
    category="nickel",
    density_kgm3=8270.0,
    tensile_strength_mpa=1000.0,
    tensile_strength_at_temp={
        293.0: 1000.0,
        811.0: 800.0,
        1033.0: 600.0,
        1144.0: 300.0,
    },
    compressive_strength_mpa=920.0,
    service_temp_air_K=1255.0,
    service_temp_inert_K=1310.0,
    melting_point_K=1644.0,
    thermal_conductivity_WmK=9.4,
    thermal_expansion_1K=13.0e-6,
    youngs_modulus_GPa=216.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1311.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Haynes International H-3172A data sheet",
    notes="Superior creep resistance to Waspaloy; wrought gamma-prime strengthened; engine casings and rings",
    cost_usd_per_kg=80.0,  # cost ~$80/kg - Haynes 282 wrought premium, newer gamma-prime alloy
)

MAR_M247 = MaterialEntry(
    name="MAR-M 247 (DS)",
    category="nickel",
    density_kgm3=8530.0,
    tensile_strength_mpa=1000.0,
    tensile_strength_at_temp={
        293.0: 1000.0,
        811.0: 800.0,
        1033.0: 600.0,
        1144.0: 300.0,
    },
    compressive_strength_mpa=930.0,
    service_temp_air_K=1255.0,
    service_temp_inert_K=1311.0,
    melting_point_K=1600.0,
    thermal_conductivity_WmK=10.7,
    thermal_expansion_1K=12.4e-6,
    youngs_modulus_GPa=213.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1255.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Harris et al., Superalloys 1984 p. 221; AZoM MAR-M 247 article",
    notes="Directionally solidified casting; eliminates transverse grain boundaries; HPT blades; higher creep than equiaxed",
    cost_usd_per_kg=400.0,  # cost ~$400/kg - MAR-M 247 DS casting, directionally-solidified premium
)

CMSX4 = MaterialEntry(
    name="CMSX-4",
    category="nickel",
    density_kgm3=8700.0,
    tensile_strength_mpa=1100.0,
    # CMSX-4 exhibits anomalous strength increase to ~1073 K (yield stress anomaly
    # in single-crystal Ni3Al strengthened alloys) before dropping sharply.
    # This non-monotonic behavior is physically correct and whitelisted in tests.
    tensile_strength_at_temp={
        293.0: 1100.0,
        811.0: 1250.0,    # anomalous peak — whitelisted in monotonicity test
        1033.0: 770.0,
        1250.0: 330.0,
    },
    compressive_strength_mpa=1040.0,
    service_temp_air_K=1255.0,
    service_temp_inert_K=1423.0,
    melting_point_K=1613.0,
    thermal_conductivity_WmK=12.0,
    thermal_expansion_1K=12.6e-6,
    youngs_modulus_GPa=130.0,     # [001] single-crystal direction
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1311.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Springer J. Mater. Eng. Perf. 5 (1996) CMSX-4 tensile; Cannon-Muskegon datasheet",
    notes="Second-generation single-crystal; 6.5% Re; no grain boundaries; HPT blade alloy; E is [001] direction ~130 GPa; anomalous strength peak near 800°C",
    cost_usd_per_kg=800.0,  # cost ~$800/kg - CMSX-4 single-crystal casting, rhenium-bearing
)

PWA1484 = MaterialEntry(
    name="PWA 1484",
    category="nickel",
    density_kgm3=8950.0,
    tensile_strength_mpa=1000.0,
    # PWA 1484 is also a second-gen SX alloy and exhibits the same anomalous
    # yield stress peak. Whitelisted in monotonicity test.
    tensile_strength_at_temp={
        293.0: 1000.0,
        811.0: 1100.0,    # anomalous peak — whitelisted in monotonicity test
        1033.0: 700.0,
        1250.0: 290.0,
    },
    compressive_strength_mpa=950.0,
    service_temp_air_K=1255.0,
    service_temp_inert_K=1423.0,
    melting_point_K=1613.0,
    thermal_conductivity_WmK=11.3,
    thermal_expansion_1K=12.9e-6,
    youngs_modulus_GPa=126.0,     # [001] single-crystal direction
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1311.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Cetel & Duhl, Superalloys 1988; Pratt & Whitney technical data",
    notes="Second-gen SX; 5.9% Re; F100/F119 HPT blades; E is [001] direction ~126 GPa; anomalous strength peak near 800°C",
    cost_usd_per_kg=800.0,  # cost ~$800/kg - PWA 1484 single-crystal, F119/F135 blade grade
)

IN625_LCF = MaterialEntry(
    name="Inconel 625 LCF",
    category="nickel",
    density_kgm3=8440.0,
    tensile_strength_mpa=930.0,
    tensile_strength_at_temp={
        293.0: 930.0,
        811.0: 744.0,
        1033.0: 558.0,
        1144.0: 279.0,
    },
    compressive_strength_mpa=840.0,
    service_temp_air_K=1255.0,
    service_temp_inert_K=1366.0,
    melting_point_K=1623.0,
    thermal_conductivity_WmK=9.8,
    thermal_expansion_1K=12.8e-6,
    youngs_modulus_GPa=207.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1422.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Special Metals INCONEL 625 LCF bulletin; ASTM B443 UNS N06626",
    notes="Tighter chemistry control (C, Si, Nb) than standard 625 for improved low-cycle fatigue life; identical bulk mechanical properties; bellows, expansion joints, nuclear components",
    cost_usd_per_kg=65.0,  # cost ~$65/kg - IN625 LCF heat, premium forged
)

NIMONIC_90 = MaterialEntry(
    name="Nimonic 90",
    category="nickel",
    density_kgm3=8180.0,
    tensile_strength_mpa=1040.0,
    tensile_strength_at_temp={
        293.0: 1040.0,
        811.0: 780.0,
        1033.0: 420.0,
        1144.0: 200.0,
    },
    compressive_strength_mpa=980.0,
    service_temp_air_K=1143.0,
    service_temp_inert_K=1200.0,
    melting_point_K=1633.0,
    thermal_conductivity_WmK=11.5,
    thermal_expansion_1K=12.7e-6,
    youngs_modulus_GPa=213.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1143.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Special Metals Nimonic 90 data sheet; Wiggin Alloys technical bulletin",
    notes="Gamma-prime strengthened Ni-Co-Cr alloy; turbine blades and discs; good creep rupture strength; Rolls-Royce engine heritage",
    cost_usd_per_kg=60.0,  # cost ~$60/kg - Nimonic 90 wrought, Rolls-Royce turbine disc legacy
)

NIMONIC_105 = MaterialEntry(
    name="Nimonic 105",
    category="nickel",
    density_kgm3=8010.0,
    tensile_strength_mpa=1180.0,
    tensile_strength_at_temp={
        293.0: 1180.0,
        811.0: 940.0,
        1033.0: 560.0,
        1144.0: 260.0,
    },
    compressive_strength_mpa=1100.0,
    service_temp_air_K=1223.0,
    service_temp_inert_K=1300.0,
    melting_point_K=1633.0,
    thermal_conductivity_WmK=12.0,
    thermal_expansion_1K=12.5e-6,
    youngs_modulus_GPa=220.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1223.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Special Metals Nimonic 105 data sheet; Wiggin Alloys Ltd.",
    notes="Higher-strength Nimonic variant; 5% Mo for solid-solution strengthening; turbine blades, hot-gas path components",
    cost_usd_per_kg=70.0,  # cost ~$70/kg - Nimonic 105 wrought premium
)

UDIMET_720 = MaterialEntry(
    name="Udimet 720",
    category="nickel",
    density_kgm3=8080.0,
    tensile_strength_mpa=1350.0,
    tensile_strength_at_temp={
        293.0: 1350.0,
        811.0: 1150.0,
        1033.0: 700.0,
        1144.0: 350.0,
    },
    compressive_strength_mpa=1280.0,
    service_temp_air_K=1200.0,
    service_temp_inert_K=1280.0,
    melting_point_K=1628.0,
    thermal_conductivity_WmK=11.0,
    thermal_expansion_1K=13.0e-6,
    youngs_modulus_GPa=218.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1200.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Special Metals UDIMET 720 data sheet; Duhl, Superalloys 1988",
    notes="P/M disc alloy; high gamma-prime volume fraction (~45%); GE90 and V2500 turbine discs; excellent creep and tensile strength",
    cost_usd_per_kg=80.0,  # cost ~$80/kg - Udimet 720 wrought/forged premium disc alloy
)

RENE_88DT = MaterialEntry(
    name="René 88DT",
    category="nickel",
    density_kgm3=8230.0,
    tensile_strength_mpa=1310.0,
    tensile_strength_at_temp={
        293.0: 1310.0,
        811.0: 1100.0,
        1033.0: 650.0,
        1144.0: 320.0,
    },
    compressive_strength_mpa=1240.0,
    service_temp_air_K=1200.0,
    service_temp_inert_K=1270.0,
    melting_point_K=1623.0,
    thermal_conductivity_WmK=10.5,
    thermal_expansion_1K=13.1e-6,
    youngs_modulus_GPa=211.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1200.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Krueger et al., Superalloys 1992; GE Aviation René 88DT data",
    notes="P/M disc alloy; damage-tolerant variant (DT suffix); ~36% gamma-prime; F414 and GE90 HPT discs",
    cost_usd_per_kg=250.0,  # cost ~$250/kg - Rene 88DT P/M disc alloy, GE90/F414
)

RENE_125 = MaterialEntry(
    name="René 125",
    category="nickel",
    density_kgm3=8680.0,
    tensile_strength_mpa=1070.0,
    tensile_strength_at_temp={
        293.0: 1070.0,
        811.0: 1000.0,
        1033.0: 700.0,
        1200.0: 310.0,
    },
    compressive_strength_mpa=1000.0,
    service_temp_air_K=1255.0,
    service_temp_inert_K=1370.0,
    melting_point_K=1613.0,
    thermal_conductivity_WmK=10.8,
    thermal_expansion_1K=13.5e-6,
    youngs_modulus_GPa=200.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1255.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="GE Aviation René 125 data sheet; Donachie, Superalloys Source Book",
    notes="DS/SX blade alloy; high Hf content improves ductility; first-stage turbine blades; good castability",
    cost_usd_per_kg=300.0,  # cost ~$300/kg - Rene 125 conventionally-cast blade premium
)

IN100 = MaterialEntry(
    name="IN-100",
    category="nickel",
    density_kgm3=7750.0,
    tensile_strength_mpa=1100.0,
    tensile_strength_at_temp={
        293.0: 1100.0,
        811.0: 930.0,
        1033.0: 550.0,
        1144.0: 250.0,
    },
    compressive_strength_mpa=1040.0,
    service_temp_air_K=1200.0,
    service_temp_inert_K=1280.0,
    melting_point_K=1605.0,
    thermal_conductivity_WmK=10.8,
    thermal_expansion_1K=13.0e-6,
    youngs_modulus_GPa=214.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1200.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Special Metals IN-100 data; MMPDS-17; Pratt & Whitney JT8D/JT9D disc data",
    notes="Cast + HIP or P/M disc alloy; ~60% gamma-prime; JT9D, CF6 HPT discs; one of the earliest high-strength disc alloys",
    cost_usd_per_kg=250.0,  # cost ~$250/kg - IN-100 P/M disc alloy, F100 engine
)

ASTROLOY = MaterialEntry(
    name="Astroloy",
    category="nickel",
    density_kgm3=7910.0,
    tensile_strength_mpa=1240.0,
    tensile_strength_at_temp={
        293.0: 1240.0,
        811.0: 1050.0,
        1033.0: 620.0,
        1144.0: 290.0,
    },
    compressive_strength_mpa=1170.0,
    service_temp_air_K=1200.0,
    service_temp_inert_K=1280.0,
    melting_point_K=1613.0,
    thermal_conductivity_WmK=11.5,
    thermal_expansion_1K=12.8e-6,
    youngs_modulus_GPa=215.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1200.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Special Metals Astroloy bulletin; MMPDS-17; General Electric disc data",
    notes="P/M turbine disc alloy; Ni-Cr-Co-Mo base; ~55% gamma-prime; engine discs and shafts; good balance of strength and creep",
    cost_usd_per_kg=250.0,  # cost ~$250/kg - Astroloy P/M disc alloy
)


# ===========================================================================
# STEEL ALLOYS
# applicable_regimes: subsonic, supersonic
# ===========================================================================

STEEL_4340 = MaterialEntry(
    name="4340 Steel",
    category="steel",
    density_kgm3=7850.0,
    tensile_strength_mpa=1480.0,
    tensile_strength_at_temp={
        293.0: 1480.0,
        422.0: 1350.0,
        533.0: 1100.0,
        644.0: 690.0,
    },
    compressive_strength_mpa=1380.0,
    service_temp_air_K=533.0,
    service_temp_inert_K=590.0,
    melting_point_K=1700.0,
    thermal_conductivity_WmK=44.5,
    thermal_expansion_1K=12.3e-6,
    youngs_modulus_GPa=200.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=533.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="ASM Handbook Vol. 1; MMPDS-17 4340 tables",
    notes="High-strength structural steel; landing gear; requires cadmium or zinc plating for corrosion protection; AMS 6415",
    cost_usd_per_kg=3.0,  # cost ~$3/kg - 4340 alloy steel bar, commodity landing-gear grade
)

STEEL_300M = MaterialEntry(
    name="300M",
    category="steel",
    density_kgm3=7870.0,
    tensile_strength_mpa=1930.0,
    tensile_strength_at_temp={
        293.0: 1930.0,
        422.0: 1760.0,
        533.0: 1430.0,
        644.0: 900.0,
    },
    compressive_strength_mpa=1800.0,
    service_temp_air_K=533.0,
    service_temp_inert_K=590.0,
    melting_point_K=1700.0,
    thermal_conductivity_WmK=40.2,
    thermal_expansion_1K=12.1e-6,
    youngs_modulus_GPa=210.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=533.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="SSA Corporation 300M/4340 MOD data; MMPDS-17; AMS 6417/6419",
    notes="Si and V modifications over 4340; landing gear shafts and cylinders; higher toughness than 4340 at equivalent strength",
    cost_usd_per_kg=20.0,  # cost ~$20/kg - 300M ultra-high-strength steel, landing-gear premium
)

STEEL_AF1410 = MaterialEntry(
    name="AF1410",
    category="steel",
    density_kgm3=7810.0,
    tensile_strength_mpa=1620.0,
    tensile_strength_at_temp={
        293.0: 1620.0,
        422.0: 1480.0,
        533.0: 1210.0,
        644.0: 770.0,
    },
    compressive_strength_mpa=1510.0,
    service_temp_air_K=533.0,
    service_temp_inert_K=590.0,
    melting_point_K=1710.0,
    thermal_conductivity_WmK=29.0,
    thermal_expansion_1K=11.5e-6,
    youngs_modulus_GPa=200.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=533.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="AMS 6527; Vought/NAVSEA AF1410 data; ASM Aerospace Specification Metals",
    notes="Co-Ni secondary hardening steel; fracture toughness > 154 MPa√m; carrier-based aircraft structural frames",
    cost_usd_per_kg=20.0,  # cost ~$20/kg - AF1410 USAF high-performance UHS steel
)

STEEL_174PH = MaterialEntry(
    name="17-4PH",
    category="steel",
    density_kgm3=7780.0,
    tensile_strength_mpa=1310.0,
    tensile_strength_at_temp={
        293.0: 1310.0,
        422.0: 1180.0,
        533.0: 1000.0,
        644.0: 690.0,
    },
    compressive_strength_mpa=1240.0,
    service_temp_air_K=588.0,
    service_temp_inert_K=644.0,
    melting_point_K=1700.0,
    thermal_conductivity_WmK=18.3,
    thermal_expansion_1K=10.8e-6,
    youngs_modulus_GPa=197.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=644.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Carpenter Technology 17-4PH H900 condition data sheet; AMS 5604",
    notes="Precipitation-hardened martensitic stainless; good corrosion resistance; valve bodies, shafts; H900 condition",
    cost_usd_per_kg=15.0,  # cost ~$15/kg - 17-4 PH stainless, precipitation hardening grade
)

STEEL_155PH = MaterialEntry(
    name="15-5PH",
    category="steel",
    density_kgm3=7780.0,
    tensile_strength_mpa=1310.0,
    tensile_strength_at_temp={
        293.0: 1310.0,
        422.0: 1180.0,
        533.0: 990.0,
        644.0: 685.0,
    },
    compressive_strength_mpa=1240.0,
    service_temp_air_K=588.0,
    service_temp_inert_K=644.0,
    melting_point_K=1700.0,
    thermal_conductivity_WmK=18.4,
    thermal_expansion_1K=10.8e-6,
    youngs_modulus_GPa=196.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=644.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Carpenter Technology 15-5PH data sheet; AMS 5659",
    notes="Vacuum melt variant of 17-4PH; improved transverse ductility and toughness; aerospace fittings and rings",
    cost_usd_per_kg=15.0,  # cost ~$15/kg - 15-5 PH stainless, enhanced toughness over 17-4
)

STEEL_MAR350 = MaterialEntry(
    name="Maraging 350",
    category="steel",
    density_kgm3=8000.0,
    tensile_strength_mpa=2415.0,
    tensile_strength_at_temp={
        293.0: 2415.0,
        422.0: 2200.0,
        533.0: 1850.0,
        644.0: 1150.0,
    },
    compressive_strength_mpa=2300.0,
    service_temp_air_K=533.0,
    service_temp_inert_K=590.0,
    melting_point_K=1718.0,
    thermal_conductivity_WmK=25.5,
    thermal_expansion_1K=10.6e-6,
    youngs_modulus_GPa=190.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=533.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Vascomax C350 data sheet; SSA Maraging Steel data",
    notes="Highest-strength maraging grade; missile motor cases, rocket motor housings; ultra-high strength with good toughness",
    cost_usd_per_kg=15.0,  # cost ~$15/kg - Maraging 350, high-strength mold-tool and airframe grade
)

STEEL_HY80 = MaterialEntry(
    name="HY-80",
    category="steel",
    density_kgm3=7860.0,
    tensile_strength_mpa=620.0,
    tensile_strength_at_temp={
        293.0: 620.0,
        422.0: 560.0,
        533.0: 480.0,
        644.0: 350.0,
    },
    compressive_strength_mpa=590.0,
    service_temp_air_K=644.0,
    service_temp_inert_K=700.0,
    melting_point_K=1783.0,
    thermal_conductivity_WmK=36.0,
    thermal_expansion_1K=11.5e-6,
    youngs_modulus_GPa=207.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=644.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="MIL-S-16216; NAVSEA Technical Publication; ASTM A543",
    notes="Naval structural steel; submarine hulls (Los Angeles-class); 80 ksi min yield; excellent weldability and toughness at low temperatures",
    cost_usd_per_kg=4.0,  # cost ~$4/kg - HY-80 naval structural steel, bulk plate
)

STEEL_HY100 = MaterialEntry(
    name="HY-100",
    category="steel",
    density_kgm3=7860.0,
    tensile_strength_mpa=760.0,
    tensile_strength_at_temp={
        293.0: 760.0,
        422.0: 690.0,
        533.0: 590.0,
        644.0: 420.0,
    },
    compressive_strength_mpa=720.0,
    service_temp_air_K=644.0,
    service_temp_inert_K=700.0,
    melting_point_K=1783.0,
    thermal_conductivity_WmK=36.0,
    thermal_expansion_1K=11.5e-6,
    youngs_modulus_GPa=207.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=644.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="MIL-S-16216; NAVSEA Technical Publication; ASTM A543",
    notes="Higher-strength naval steel; submarine pressure hulls (Seawolf-class); 100 ksi min yield; tighter welding procedures than HY-80",
    cost_usd_per_kg=5.0,  # cost ~$5/kg - HY-100 naval structural steel, bulk plate
)

STEEL_PH138MO = MaterialEntry(
    name="PH 13-8 Mo",
    category="steel",
    density_kgm3=7760.0,
    tensile_strength_mpa=1520.0,
    tensile_strength_at_temp={
        293.0: 1520.0,
        422.0: 1400.0,
        533.0: 1200.0,
        644.0: 800.0,
    },
    compressive_strength_mpa=1450.0,
    service_temp_air_K=589.0,
    service_temp_inert_K=640.0,
    melting_point_K=1733.0,
    thermal_conductivity_WmK=14.0,
    thermal_expansion_1K=10.6e-6,
    youngs_modulus_GPa=200.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=589.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="AK Steel PH 13-8 Mo data sheet; AMS 5629; MMPDS-17",
    notes="Martensitic PH stainless; landing gear, actuator rods, structural bolts; excellent transverse properties; H950/H1000 condition",
    cost_usd_per_kg=18.0,  # cost ~$18/kg - PH 13-8 Mo stainless, premium precipitation-hardening
)

STEEL_AERMET100 = MaterialEntry(
    name="AerMet 100",
    category="steel",
    density_kgm3=7890.0,
    tensile_strength_mpa=1965.0,
    tensile_strength_at_temp={
        293.0: 1965.0,
        422.0: 1800.0,
        533.0: 1550.0,
        644.0: 1050.0,
    },
    compressive_strength_mpa=1870.0,
    service_temp_air_K=533.0,
    service_temp_inert_K=590.0,
    melting_point_K=1750.0,
    thermal_conductivity_WmK=26.0,
    thermal_expansion_1K=11.0e-6,
    youngs_modulus_GPa=195.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=533.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Carpenter AerMet 100 data sheet; MMPDS-17; AMS 6532",
    notes="Ultra-high strength secondary hardening steel; F/A-18E/F landing gear; KIc ~126 MPa√m; replaces 300M where better toughness is needed",
    cost_usd_per_kg=25.0,  # cost ~$25/kg - AerMet 100 ultra-high-performance steel, fighter landing gear
)

STEEL_GREEKASCOLOY = MaterialEntry(
    name="Greek Ascoloy (W545)",
    category="steel",
    density_kgm3=7880.0,
    tensile_strength_mpa=1100.0,
    tensile_strength_at_temp={
        293.0: 1100.0,
        533.0: 900.0,
        700.0: 650.0,
        811.0: 400.0,
    },
    compressive_strength_mpa=1040.0,
    service_temp_air_K=755.0,
    service_temp_inert_K=810.0,
    melting_point_K=1755.0,
    thermal_conductivity_WmK=25.0,
    thermal_expansion_1K=11.3e-6,
    youngs_modulus_GPa=200.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=755.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Firth Sterling Greek Ascoloy data; AMS 5616; Carpenter W545 data sheet",
    notes="12Cr martensitic stainless; turbine compressor blades and vanes; retains strength to ~480°C; corrosion resistant",
    cost_usd_per_kg=12.0,  # cost ~$12/kg - Greek Ascoloy / W545 specialty turbine steel
)


# ===========================================================================
# COBALT SUPERALLOYS
# applicable_regimes: subsonic, supersonic, hypersonic
# ===========================================================================

HAYNES_188 = MaterialEntry(
    name="Haynes 188",
    category="cobalt",
    density_kgm3=8980.0,
    tensile_strength_mpa=960.0,
    tensile_strength_at_temp={
        293.0: 960.0,
        811.0: 690.0,
        1033.0: 430.0,
        1200.0: 200.0,
    },
    compressive_strength_mpa=900.0,
    service_temp_air_K=1338.0,
    service_temp_inert_K=1394.0,
    melting_point_K=1635.0,
    thermal_conductivity_WmK=10.4,
    thermal_expansion_1K=13.8e-6,
    youngs_modulus_GPa=232.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1338.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Haynes International alloy 188 data sheet H-3001; NASA TP-2002-211470",
    notes="Co-Ni-Cr-W alloy; combustion liners, transition ducts, afterburner components; exceptional oxidation resistance to 1093°C; Space Shuttle SSME components",
    cost_usd_per_kg=120.0,  # cost ~$120/kg - Haynes 188 wrought, combustor liner standard
)

L605 = MaterialEntry(
    name="L-605 (Haynes 25)",
    category="cobalt",
    density_kgm3=9130.0,
    tensile_strength_mpa=1000.0,
    tensile_strength_at_temp={
        293.0: 1000.0,
        811.0: 710.0,
        1033.0: 440.0,
        1200.0: 190.0,
    },
    compressive_strength_mpa=940.0,
    service_temp_air_K=1253.0,
    service_temp_inert_K=1366.0,
    melting_point_K=1683.0,
    thermal_conductivity_WmK=10.0,
    thermal_expansion_1K=14.5e-6,
    youngs_modulus_GPa=225.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1253.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Haynes International alloy 25 data sheet H-3057; ASM Handbook Vol. 2",
    notes="Co-Cr-W-Ni alloy; turbine vanes, combustor parts; the original high-temperature cobalt alloy; excellent fabricability",
    cost_usd_per_kg=120.0,  # cost ~$120/kg - L-605 / Haynes 25 wrought, legacy Co-Cr-Ni-W
)

MP35N = MaterialEntry(
    name="MP35N",
    category="cobalt",
    density_kgm3=8430.0,
    tensile_strength_mpa=1790.0,
    tensile_strength_at_temp={
        293.0: 1790.0,
        533.0: 1610.0,
        811.0: 1200.0,
        1033.0: 620.0,
    },
    compressive_strength_mpa=1700.0,
    service_temp_air_K=811.0,
    service_temp_inert_K=870.0,
    melting_point_K=1633.0,
    thermal_conductivity_WmK=11.8,
    thermal_expansion_1K=12.8e-6,
    youngs_modulus_GPa=228.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=811.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="SPS Technologies MP35N data sheet; ASTM F562; Carpenter MP35N technical data",
    notes="Co-Ni-Cr-Mo alloy; cold-worked + aged condition; fasteners, springs, subsea bolting; non-magnetic; highest strength of the cobalt alloys",
    cost_usd_per_kg=120.0,  # cost ~$120/kg - MP35N wire/bar, fastener and spring grade
)

ELGILOY = MaterialEntry(
    name="Elgiloy",
    category="cobalt",
    density_kgm3=8300.0,
    tensile_strength_mpa=1590.0,
    tensile_strength_at_temp={
        293.0: 1590.0,
        533.0: 1430.0,
        811.0: 1050.0,
        1033.0: 530.0,
    },
    compressive_strength_mpa=1500.0,
    service_temp_air_K=811.0,
    service_temp_inert_K=870.0,
    melting_point_K=1633.0,
    thermal_conductivity_WmK=12.3,
    thermal_expansion_1K=13.0e-6,
    youngs_modulus_GPa=221.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=811.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="Elgiloy Specialty Metals data sheet; ASTM F1058; Haynes Elgiloy technical bulletin",
    notes="Co-Cr-Ni-Mo alloy; springs, watch mainsprings, medical devices, diaphragms; excellent fatigue; similar composition to MP35N",
    cost_usd_per_kg=140.0,  # cost ~$140/kg - Elgiloy medical-grade specialty, small-lot premium
)


# ===========================================================================
# POLYMER MATRIX COMPOSITES
# applicable_regimes: subsonic, supersonic
# All tensile values are quasi-isotropic (QI) laminate unless noted.
# melting_point_K = resin decomposition temperature
# ===========================================================================

CFRP_IM7_977_3 = MaterialEntry(
    name="IM7/977-3 CFRP",
    category="composite_polymer",
    density_kgm3=1580.0,
    tensile_strength_mpa=900.0,
    tensile_strength_at_temp={
        293.0: 900.0,
        366.0: 860.0,
        422.0: 750.0,
        473.0: 500.0,
    },
    compressive_strength_mpa=800.0,
    service_temp_air_K=423.0,
    service_temp_inert_K=450.0,
    melting_point_K=673.0,     # resin decomposition temperature
    thermal_conductivity_WmK=3.5,
    thermal_expansion_1K=2.1e-6,
    youngs_modulus_GPa=70.0,
    oxidation_resistance="poor",
    oxidation_max_temp_K=423.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Hexcel 977-3 prepreg data sheet; Hexcel HexPly 977-3 product data",
    notes="QI laminate values; 150°C service; autoclave prepreg; F-22 skin panels; melting_point_K = resin decomposition",
    cost_usd_per_kg=150.0,  # cost ~$150/kg - IM7/977-3 aerospace prepreg, epoxy standard
)

CFRP_T800_3900 = MaterialEntry(
    name="T800/3900 CFRP",
    category="composite_polymer",
    density_kgm3=1580.0,
    tensile_strength_mpa=880.0,
    tensile_strength_at_temp={
        293.0: 880.0,
        366.0: 840.0,
        422.0: 730.0,
        473.0: 485.0,
    },
    compressive_strength_mpa=780.0,
    service_temp_air_K=423.0,
    service_temp_inert_K=455.0,
    melting_point_K=673.0,     # resin decomposition temperature
    thermal_conductivity_WmK=3.8,
    thermal_expansion_1K=2.0e-6,
    youngs_modulus_GPa=69.0,
    oxidation_resistance="poor",
    oxidation_max_temp_K=423.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Toray T800/3900 prepreg data sheet; Boeing 787 materials specification",
    notes="QI laminate values; Boeing 787 primary fuselage and wing structure; toughened epoxy; melting_point_K = resin decomposition",
    cost_usd_per_kg=150.0,  # cost ~$150/kg - T800/3900-2 aerospace prepreg, 787-era
)

CFRP_IM7_BMI = MaterialEntry(
    name="IM7/BMI",
    category="composite_polymer",
    density_kgm3=1600.0,
    tensile_strength_mpa=860.0,
    tensile_strength_at_temp={
        293.0: 860.0,
        366.0: 840.0,
        473.0: 730.0,
        533.0: 520.0,
    },
    compressive_strength_mpa=760.0,
    service_temp_air_K=503.0,
    service_temp_inert_K=533.0,
    melting_point_K=723.0,     # BMI resin decomposition temperature
    thermal_conductivity_WmK=2.9,
    thermal_expansion_1K=2.4e-6,
    youngs_modulus_GPa=68.0,
    oxidation_resistance="poor",
    oxidation_max_temp_K=503.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="NASA NTRS 20130011578 BMI composites; ScienceDirect BMI temperature effects review",
    notes="QI laminate values; bismaleimide resin; service to 230°C (503 K); F-22 airframe hot structure, engine nacelles; 177°C cure cycle",
    cost_usd_per_kg=180.0,  # cost ~$180/kg - IM7/BMI bismaleimide prepreg, supersonic-cruise skin
)

CFRP_AS4_PEEK = MaterialEntry(
    name="AS4/PEEK",
    category="composite_polymer",
    density_kgm3=1600.0,
    tensile_strength_mpa=870.0,
    tensile_strength_at_temp={
        293.0: 870.0,
        366.0: 835.0,
        473.0: 740.0,
        523.0: 580.0,
    },
    compressive_strength_mpa=800.0,
    service_temp_air_K=523.0,
    service_temp_inert_K=573.0,
    melting_point_K=616.0,     # PEEK crystalline melt temperature
    thermal_conductivity_WmK=1.0,
    thermal_expansion_1K=3.0e-6,
    youngs_modulus_GPa=60.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=523.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Victrex PEEK composite properties; ACP Composites CFRP data; ICI/Cytec technical data",
    notes="QI laminate values; thermoplastic matrix; weldable, reprocessable, solvent resistant; leading edge panels to Mach 2",
    cost_usd_per_kg=160.0,  # cost ~$160/kg - AS4/PEEK thermoplastic prepreg
)

CFRP_IM7_5250 = MaterialEntry(
    name="IM7/5250-4 BMI",
    category="composite_polymer",
    density_kgm3=1580.0,
    tensile_strength_mpa=880.0,
    tensile_strength_at_temp={
        293.0: 880.0,
        422.0: 810.0,
        505.0: 700.0,
        561.0: 520.0,
    },
    compressive_strength_mpa=810.0,
    service_temp_air_K=561.0,
    service_temp_inert_K=610.0,
    melting_point_K=673.0,     # BMI resin decomposition
    thermal_conductivity_WmK=1.2,
    thermal_expansion_1K=2.8e-6,
    youngs_modulus_GPa=65.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=561.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Cytec Cycom 5250-4 data sheet; AFRL-ML-WP-TR-2003-4132",
    notes="BMI matrix composite; F-22 aft fuselage; higher service temp than epoxy (288°C vs 177°C); post-cure at 316°C",
    cost_usd_per_kg=180.0,  # cost ~$180/kg - IM7/5250-4 BMI prepreg, F/A-18E/F
)

CFRP_IM7_PETI = MaterialEntry(
    name="IM7/PETI-330",
    category="composite_polymer",
    density_kgm3=1560.0,
    tensile_strength_mpa=820.0,
    tensile_strength_at_temp={
        293.0: 820.0,
        422.0: 760.0,
        533.0: 640.0,
        589.0: 470.0,
    },
    compressive_strength_mpa=750.0,
    service_temp_air_K=589.0,
    service_temp_inert_K=640.0,
    melting_point_K=700.0,     # polyimide decomposition
    thermal_conductivity_WmK=1.1,
    thermal_expansion_1K=2.5e-6,
    youngs_modulus_GPa=63.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=589.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="NASA LaRC PETI-330 data; Cano & Jensen, SAMPE J. 2004; NASA TM-2005-213484",
    notes="Polyimide matrix; highest service temp of polymer composites (~316°C); HSCT technology; supersonic nacelle skins",
    cost_usd_per_kg=220.0,  # cost ~$220/kg - IM7/PETI-330 polyimide, HSR-era high-T premium
)

CFRP_T300_934 = MaterialEntry(
    name="T300/934 CFRP",
    category="composite_polymer",
    density_kgm3=1580.0,
    tensile_strength_mpa=760.0,
    tensile_strength_at_temp={
        293.0: 760.0,
        366.0: 720.0,
        422.0: 610.0,
        478.0: 430.0,
    },
    compressive_strength_mpa=700.0,
    service_temp_air_K=450.0,
    service_temp_inert_K=500.0,
    melting_point_K=623.0,     # 934 epoxy decomposition
    thermal_conductivity_WmK=1.0,
    thermal_expansion_1K=3.0e-6,
    youngs_modulus_GPa=58.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=450.0,
    applicable_regimes=["subsonic", "supersonic"],
    citation="Fiberite 934 resin data; NASA CR-178025; MMPDS-17 CFRP tables",
    notes="First-gen 177°C-cure epoxy; F/A-18A skins, AV-8B wing; legacy workhorse system; well-characterized allowables database",
    cost_usd_per_kg=120.0,  # cost ~$120/kg - T300/934 older/cheaper CFRP, still in production
)


# ===========================================================================
# CERAMIC MATRIX COMPOSITES (CMC)
# applicable_regimes: subsonic, supersonic, hypersonic
# tensile_strength_mpa = flexural strength (MOR) as proxy for brittle material
# ===========================================================================

CMC_SIC_SIC = MaterialEntry(
    name="SiC/SiC CMC",
    category="composite_ceramic",
    density_kgm3=2650.0,
    tensile_strength_mpa=260.0,     # flexural strength (MOR)
    tensile_strength_at_temp={
        293.0: 260.0,
        1273.0: 240.0,
        1473.0: 220.0,
        1673.0: 180.0,
    },
    compressive_strength_mpa=950.0,
    service_temp_air_K=1673.0,
    service_temp_inert_K=1773.0,
    melting_point_K=3003.0,    # SiC decomposition ~2973 K
    thermal_conductivity_WmK=19.0,
    thermal_expansion_1K=4.0e-6,
    youngs_modulus_GPa=230.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=1773.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="ORNL SiC/SiC Cladding Handbook ORNL/TM-2015/488; UCLA SiC/SiC CMC data 2019",
    notes="Tensile_strength values are flexural (MOR); CVI process; Nicalon or Hi-Nicalon fiber; GE LEAP turbine hot section; replacing Ni superalloy in HPT",
    cost_usd_per_kg=2000.0,  # cost ~$2000/kg - SiC/SiC CMC hot-section demonstrator pricing
)

CMC_C_SIC = MaterialEntry(
    name="C/SiC CMC",
    category="composite_ceramic",
    density_kgm3=2200.0,
    tensile_strength_mpa=290.0,     # flexural strength (MOR)
    tensile_strength_at_temp={
        293.0: 290.0,
        1273.0: 260.0,
        1473.0: 230.0,
        1773.0: 160.0,
    },
    compressive_strength_mpa=750.0,
    service_temp_air_K=1773.0,
    service_temp_inert_K=1973.0,
    melting_point_K=2823.0,
    thermal_conductivity_WmK=12.5,
    thermal_expansion_1K=2.5e-6,
    youngs_modulus_GPa=95.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=1773.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="NASA NTRS 20020072848; PMC C/SiC Cf/SiC review 2024",
    notes="Tensile_strength values are flexural (MOR); C fiber in SiC matrix; lower density than SiC/SiC; hypersonic leading edges to 1500°C with oxidation coating",
    cost_usd_per_kg=2000.0,  # cost ~$2000/kg - C/SiC CMC reentry demonstrator pricing
)

CMC_OXIDE = MaterialEntry(
    name="Oxide/Oxide CMC",
    category="composite_ceramic",
    density_kgm3=2850.0,
    tensile_strength_mpa=200.0,     # flexural strength (MOR)
    tensile_strength_at_temp={
        293.0: 200.0,
        1073.0: 180.0,
        1273.0: 155.0,
        1473.0: 110.0,
    },
    compressive_strength_mpa=600.0,
    service_temp_air_K=1473.0,
    service_temp_inert_K=1473.0,
    melting_point_K=2345.0,    # mullite eutectic ~2345 K
    thermal_conductivity_WmK=4.0,
    thermal_expansion_1K=6.5e-6,
    youngs_modulus_GPa=150.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1473.0,
    applicable_regimes=["subsonic", "supersonic", "hypersonic"],
    citation="3M Oxide/Oxide CMC Nextel 720/alumina data sheet; Wiley Adv. Eng. Mat. oxide/oxide review 2025",
    notes="Tensile_strength values are flexural (MOR); Nextel 720/alumina; no oxidation coating needed; industrial gas turbine combustor liners; lower strength than SiC-based CMCs",
    cost_usd_per_kg=1500.0,  # cost ~$1500/kg - Oxide/Oxide CMC, no SiC fiber premium
)

CMC_NEXTEL610 = MaterialEntry(
    name="Nextel 610/alumina CMC",
    category="composite_ceramic",
    density_kgm3=3100.0,
    tensile_strength_mpa=260.0,     # flexural strength (MOR)
    tensile_strength_at_temp={
        293.0: 260.0,
        1073.0: 240.0,
        1273.0: 200.0,
        1473.0: 130.0,
    },
    compressive_strength_mpa=700.0,
    service_temp_air_K=1373.0,
    service_temp_inert_K=1373.0,
    melting_point_K=2323.0,    # alumina melting
    thermal_conductivity_WmK=5.0,
    thermal_expansion_1K=7.0e-6,
    youngs_modulus_GPa=190.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1373.0,
    applicable_regimes=["supersonic", "hypersonic", "reentry"],
    citation="3M Nextel 610 fiber data; COI Ceramics oxide/oxide CMC data sheet; Zok, J Am Ceram Soc 2006",
    notes="Tensile_strength values are flexural (MOR); Nextel 610 (>99% α-alumina) fiber; higher strength than Nextel 720 but lower creep resistance; missile radomes",
    cost_usd_per_kg=1500.0,  # cost ~$1500/kg - Nextel 610/alumina, 3M-proprietary fiber
)

CMC_HI_NICALON = MaterialEntry(
    name="Hi-Nicalon SiC/SiC CMC",
    category="composite_ceramic",
    density_kgm3=2550.0,
    tensile_strength_mpa=310.0,     # flexural strength (MOR)
    tensile_strength_at_temp={
        293.0: 310.0,
        1073.0: 295.0,
        1473.0: 260.0,
        1673.0: 170.0,
    },
    compressive_strength_mpa=800.0,
    service_temp_air_K=1573.0,
    service_temp_inert_K=1673.0,
    melting_point_K=2973.0,    # SiC decomposition
    thermal_conductivity_WmK=8.0,
    thermal_expansion_1K=4.5e-6,
    youngs_modulus_GPa=200.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=1573.0,
    applicable_regimes=["supersonic", "hypersonic", "reentry"],
    citation="Nippon Carbon Hi-Nicalon data; GE Aviation SiC/SiC CMC data; Naslain, Composites A 2003",
    notes="Tensile_strength values are flexural (MOR); Hi-Nicalon (low-oxygen SiC) fiber; better creep than Nicalon; LEAP engine shrouds; EBC coating extends life",
    cost_usd_per_kg=2500.0,  # cost ~$2500/kg - Hi-Nicalon SiC/SiC premium, Nippon Carbon fiber
)


# ===========================================================================
# REFRACTORY METALS
# applicable_regimes: supersonic, hypersonic
# All require protective coating or inert environment above ~800 K in air
# ===========================================================================

W_METAL = MaterialEntry(
    name="Tungsten",
    category="refractory",
    density_kgm3=19250.0,
    tensile_strength_mpa=1510.0,
    tensile_strength_at_temp={
        293.0: 1510.0,
        1500.0: 1359.0,
        2000.0: 1057.0,
        2500.0: 600.0,
    },
    compressive_strength_mpa=1800.0,
    service_temp_air_K=773.0,      # oxidizes rapidly to WO3 above ~500°C
    service_temp_inert_K=2800.0,
    melting_point_K=3695.0,
    thermal_conductivity_WmK=173.0,
    thermal_expansion_1K=4.5e-6,
    youngs_modulus_GPa=411.0,
    oxidation_resistance="poor",
    oxidation_max_temp_K=773.0,
    applicable_regimes=["supersonic", "hypersonic"],
    citation="Plansee Tungsten technical data; AZoM refractory metals comparison article",
    notes="Highest melting point of any element (3695 K); very high density (19250 kg/m³); brittle at room temperature; needs coating or inert environment above 500°C; rocket nozzle inserts",
    coated_max_temp_K=2000.0,   # HfC / TaC / ZrC CVD coatings, hypersonic leading edges,
    cost_usd_per_kg=300.0,  # cost ~$300/kg - Tungsten bulk plate/rod, 2025 Plansee-class
)

MO_METAL = MaterialEntry(
    name="Molybdenum",
    category="refractory",
    density_kgm3=10220.0,
    tensile_strength_mpa=690.0,
    tensile_strength_at_temp={
        293.0: 690.0,
        1500.0: 621.0,
        2000.0: 483.0,
        2500.0: 280.0,
    },
    compressive_strength_mpa=800.0,
    service_temp_air_K=773.0,      # MoO3 volatilizes above ~800 K
    service_temp_inert_K=2200.0,
    melting_point_K=2896.0,
    thermal_conductivity_WmK=138.0,
    thermal_expansion_1K=4.8e-6,
    youngs_modulus_GPa=329.0,
    oxidation_resistance="poor",
    oxidation_max_temp_K=773.0,
    applicable_regimes=["supersonic", "hypersonic"],
    citation="Plansee Molybdenum technical data; Admat Inc. refractory metal comparison",
    notes="High thermal conductivity; lower density than W; catastrophic oxidation in air above 800 K (MoO3 volatilizes); rocket nozzles with coating",
    coated_max_temp_K=1700.0,   # MoSi2 pack-cementation coating (pesting below 1000K is a hazard),
    cost_usd_per_kg=300.0,  # cost ~$300/kg - Molybdenum bulk plate/rod
)

RE_METAL = MaterialEntry(
    name="Rhenium",
    category="refractory",
    density_kgm3=21020.0,
    tensile_strength_mpa=1070.0,
    tensile_strength_at_temp={
        293.0: 1070.0,
        1500.0: 963.0,
        2000.0: 749.0,
        2500.0: 430.0,
    },
    compressive_strength_mpa=1200.0,
    service_temp_air_K=773.0,
    service_temp_inert_K=2800.0,
    melting_point_K=3459.0,
    thermal_conductivity_WmK=48.0,
    thermal_expansion_1K=6.2e-6,
    youngs_modulus_GPa=460.0,
    oxidation_resistance="poor",
    oxidation_max_temp_K=773.0,
    applicable_regimes=["supersonic", "hypersonic"],
    citation="ASTM B459; Admat refractory metal comparison; Plansee data",
    notes="Most expensive refractory metal; used as alloying element in Ni SX alloys (Re 6–6.5%); rocket nozzles and thruster cups; very high density",
    coated_max_temp_K=2200.0,   # Iridium coating, Re/Ir chamber walls (RL10, BE-4 class thrusters),
    cost_usd_per_kg=5000.0,  # cost ~$5000/kg - Rhenium supply-constrained, ~60 tonnes/yr world
)

NB_C103 = MaterialEntry(
    name="Niobium C-103",
    category="refractory",
    density_kgm3=8850.0,
    tensile_strength_mpa=483.0,
    tensile_strength_at_temp={
        293.0: 483.0,
        1500.0: 435.0,
        2000.0: 338.0,
        2500.0: 195.0,
    },
    compressive_strength_mpa=450.0,
    service_temp_air_K=773.0,
    service_temp_inert_K=1700.0,
    melting_point_K=2741.0,
    thermal_conductivity_WmK=52.0,
    thermal_expansion_1K=7.2e-6,
    youngs_modulus_GPa=99.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=773.0,
    applicable_regimes=["supersonic", "hypersonic"],
    citation="Western Alloys C103 data sheet; Materion Nb C-103 data; ScienceDirect C103 AM study",
    notes="Nb-10Hf-1Ti alloy; lowest density refractory metal used in aerospace; Apollo LM ascent engine nozzle; needs aluminide coating above 500°C",
    coated_max_temp_K=1640.0,   # R512E silicide (Si-20Fe-20Cr) — NASA hypersonic proven to 2500°F,
    cost_usd_per_kg=400.0,  # cost ~$400/kg - Niobium C-103 specialty aerospace alloy premium
)

TA_METAL = MaterialEntry(
    name="Tantalum",
    category="refractory",
    density_kgm3=16600.0,
    tensile_strength_mpa=480.0,
    tensile_strength_at_temp={
        293.0: 480.0,
        1500.0: 432.0,
        2000.0: 336.0,
        2500.0: 192.0,
    },
    compressive_strength_mpa=550.0,
    service_temp_air_K=773.0,
    service_temp_inert_K=2800.0,
    melting_point_K=3290.0,
    thermal_conductivity_WmK=57.5,
    thermal_expansion_1K=6.3e-6,
    youngs_modulus_GPa=186.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=773.0,
    applicable_regimes=["supersonic", "hypersonic"],
    citation="Admat tantalum vs niobium comparison; AZoM refractory metals; Plansee tantalum data",
    notes="Excellent corrosion resistance; rocket nozzle throats and chemical plant; key component of Ta4HfC5 UHTC; very high density",
    coated_max_temp_K=1800.0,   # HfC / SiC CVD coating, short-duration rocket nozzle service,
    cost_usd_per_kg=5000.0,  # cost ~$5000/kg - Tantalum supply-constrained, capacitor-market coupled
)


# ===========================================================================
# ULTRA-HIGH TEMPERATURE CERAMICS (UHTCs)
# applicable_regimes: hypersonic, reentry
# tensile_strength_mpa = flexural strength (MOR) as proxy
# ===========================================================================

ZRB2_SIC20 = MaterialEntry(
    name="ZrB2-SiC 20vol%",
    category="uhtc",
    density_kgm3=5430.0,
    tensile_strength_mpa=550.0,     # flexural strength (MOR)
    tensile_strength_at_temp={
        293.0: 550.0,
        2000.0: 440.0,
        2500.0: 330.0,
        3000.0: 220.0,
    },
    compressive_strength_mpa=1800.0,
    service_temp_air_K=2073.0,
    service_temp_inert_K=2773.0,
    melting_point_K=3518.0,
    thermal_conductivity_WmK=65.0,
    thermal_expansion_1K=6.2e-6,
    youngs_modulus_GPa=500.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=2073.0,
    applicable_regimes=["hypersonic", "reentry"],
    citation="OSTI 887260 Kasen et al. UHTCs; Fahrenholtz & Hilmas J. Am. Ceram. Soc. 95 (2012) 3235; Electrochemical Society Interface Winter 2007",
    notes="Tensile values are flexural (MOR); primary hypersonic UHTC composition; SiC forms SiO2 oxide scale for oxidation protection; X-43A leading edges; 20 vol% SiC is optimum oxidation-resistant composition",
    cost_usd_per_kg=3000.0,  # cost ~$3000/kg - ZrB2-SiC composite, research-grade billet
)

HFB2_SIC20 = MaterialEntry(
    name="HfB2-SiC 20vol%",
    category="uhtc",
    density_kgm3=9700.0,
    tensile_strength_mpa=700.0,     # flexural strength (MOR)
    tensile_strength_at_temp={
        293.0: 700.0,
        2000.0: 560.0,
        2500.0: 420.0,
        3000.0: 280.0,
    },
    compressive_strength_mpa=2100.0,
    service_temp_air_K=2173.0,
    service_temp_inert_K=2973.0,
    melting_point_K=3523.0,
    thermal_conductivity_WmK=51.6,
    thermal_expansion_1K=6.3e-6,
    youngs_modulus_GPa=480.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=2173.0,
    applicable_regimes=["hypersonic", "reentry"],
    citation="AZoM HfB2 article; Springer J. Mater. Sci. HfB2 arc-jet testing; Fahrenholtz & Hilmas 2012",
    notes="Tensile values are flexural (MOR); slightly higher temperature capability than ZrB2-SiC; heavier; superior oxidation above 2000°C; candidate for sharp leading edges above Mach 12",
    cost_usd_per_kg=4000.0,  # cost ~$4000/kg - HfB2-SiC composite, Hf-bearing premium
)

ZRB2_MONO = MaterialEntry(
    name="ZrB2 Monolithic",
    category="uhtc",
    density_kgm3=6090.0,
    tensile_strength_mpa=350.0,     # flexural strength (MOR)
    tensile_strength_at_temp={
        293.0: 350.0,
        2000.0: 280.0,
        2500.0: 210.0,
        3000.0: 140.0,
    },
    compressive_strength_mpa=1400.0,
    service_temp_air_K=1773.0,
    service_temp_inert_K=2773.0,
    melting_point_K=3518.0,
    thermal_conductivity_WmK=60.0,
    thermal_expansion_1K=5.9e-6,
    youngs_modulus_GPa=489.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=1773.0,
    applicable_regimes=["hypersonic", "reentry"],
    citation="NASA NTRS 20150022996 UHTC review; Electrochemical Society Interface Winter 2007",
    notes="Tensile values are flexural (MOR); poor oxidation resistance without SiC addition; ZrO2 scale porous above 1500°C; used as baseline in research; not recommended for production hypersonic applications",
    cost_usd_per_kg=2500.0,  # cost ~$2500/kg - ZrB2 monolithic, hot-pressed billet
)

HFB2_MONO = MaterialEntry(
    name="HfB2 Monolithic",
    category="uhtc",
    density_kgm3=11200.0,
    tensile_strength_mpa=500.0,     # flexural strength (MOR)
    tensile_strength_at_temp={
        293.0: 500.0,
        2000.0: 400.0,
        2500.0: 300.0,
        3000.0: 200.0,
    },
    compressive_strength_mpa=1600.0,
    service_temp_air_K=1873.0,
    service_temp_inert_K=2973.0,
    melting_point_K=3523.0,
    thermal_conductivity_WmK=51.6,
    thermal_expansion_1K=6.3e-6,
    youngs_modulus_GPa=500.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=1873.0,
    applicable_regimes=["hypersonic", "reentry"],
    citation="AZoM HfB2 properties; ScienceDirect HfB2 microstructure and flexural strength",
    notes="Tensile values are flexural (MOR); poor oxidation without SiC addition; higher melting than ZrB2; heavier; research baseline",
    cost_usd_per_kg=3500.0,  # cost ~$3500/kg - HfB2 monolithic, Hf-bearing premium
)

TA4HFC5 = MaterialEntry(
    name="Ta4HfC5",
    category="uhtc",
    density_kgm3=14000.0,
    tensile_strength_mpa=466.0,     # flexural strength (MOR)
    tensile_strength_at_temp={
        293.0: 466.0,
        2000.0: 373.0,
        2500.0: 280.0,
        3000.0: 186.0,
    },
    compressive_strength_mpa=1200.0,
    service_temp_air_K=1773.0,
    service_temp_inert_K=4000.0,
    melting_point_K=4178.0,
    thermal_conductivity_WmK=12.0,
    thermal_expansion_1K=6.5e-6,
    youngs_modulus_GPa=390.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=1773.0,
    applicable_regimes=["hypersonic", "reentry"],
    citation="Cedillos-Barraza et al. Scientific Reports 6 (2016) 37962; ScienceDirect polymer-derived Ta4HfC5",
    notes="Tensile values are flexural (MOR); highest known melting point of any material (4178 K); very heavy (14000 kg/m³); primarily experimental; very limited manufacturing maturity; sharp leading edge research",
    cost_usd_per_kg=5000.0,  # cost ~$5000/kg - Ta4HfC5 co-carbide, highest-melting known (Ta+Hf)
)

ZRC = MaterialEntry(
    name="ZrC",
    category="uhtc",
    density_kgm3=6730.0,
    tensile_strength_mpa=330.0,     # flexural strength (MOR)
    tensile_strength_at_temp={
        293.0: 330.0,
        2000.0: 264.0,
        2500.0: 198.0,
        3000.0: 132.0,
    },
    compressive_strength_mpa=1200.0,
    service_temp_air_K=1273.0,
    service_temp_inert_K=3673.0,
    melting_point_K=3700.0,
    thermal_conductivity_WmK=20.5,
    thermal_expansion_1K=6.7e-6,
    youngs_modulus_GPa=440.0,
    oxidation_resistance="poor",
    oxidation_max_temp_K=1273.0,
    applicable_regimes=["hypersonic", "reentry"],
    citation="PMC ZrC hypersonic review 2023; Wikipedia ZrC properties; ScienceDirect ZrC thermophysical properties",
    notes="Tensile values are flexural (MOR); oxidizes readily in air to porous ZrO2 above ~1000°C; used in inert/low-oxygen environments; nuclear fuel cladding applications",
    cost_usd_per_kg=2000.0,  # cost ~$2000/kg - ZrC monolithic, hot-pressed
)

TAC = MaterialEntry(
    name="TaC",
    category="uhtc",
    density_kgm3=14500.0,
    tensile_strength_mpa=470.0,     # flexural strength (MOR)
    tensile_strength_at_temp={
        293.0: 470.0,
        2000.0: 376.0,
        2500.0: 282.0,
        3000.0: 188.0,
    },
    compressive_strength_mpa=1600.0,
    service_temp_air_K=1273.0,
    service_temp_inert_K=3800.0,
    melting_point_K=3873.0,
    thermal_conductivity_WmK=22.0,
    thermal_expansion_1K=6.3e-6,
    youngs_modulus_GPa=537.0,
    oxidation_resistance="poor",
    oxidation_max_temp_K=1273.0,
    applicable_regimes=["hypersonic", "reentry"],
    citation="Wikipedia TaC properties; ScienceDirect UHTM Volume 2 carbide systems",
    notes="Tensile values are flexural (MOR); near-highest melting carbide (4150 K often cited; 3873 K is best-measured); very heavy; poor oxidation; cutting tool coatings",
    cost_usd_per_kg=4000.0,  # cost ~$4000/kg - TaC monolithic, Ta-bearing premium
)

ZRB2_MOSI2 = MaterialEntry(
    name="ZrB2-MoSi2",
    category="uhtc",
    density_kgm3=5540.0,
    tensile_strength_mpa=480.0,     # flexural strength (MOR)
    tensile_strength_at_temp={
        293.0: 480.0,
        2000.0: 384.0,
        2500.0: 288.0,
        3000.0: 192.0,
    },
    compressive_strength_mpa=1700.0,
    service_temp_air_K=1973.0,
    service_temp_inert_K=2773.0,
    melting_point_K=3518.0,
    thermal_conductivity_WmK=55.0,
    thermal_expansion_1K=7.0e-6,
    youngs_modulus_GPa=460.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=1973.0,
    applicable_regimes=["hypersonic", "reentry"],
    citation="ScienceDirect ZrB2 composites review; Opeka et al. J. Am. Ceram. Soc. 2004",
    notes="Tensile values are flexural (MOR); MoSi2 improves sinterability and oxidation resistance; alternative to SiC-toughened ZrB2; slightly denser than ZrB2-SiC",
    cost_usd_per_kg=3000.0,  # cost ~$3000/kg - ZrB2-MoSi2 composite, oxidation-resistant
)


# ===========================================================================
# CARBON MATERIALS
# applicable_regimes: hypersonic, reentry
# Highly anisotropic — in-plane values reported; through-thickness noted
# tensile_strength_mpa = in-plane flexural/tensile strength
# melting_point_K = sublimation temperature (carbon does not melt at 1 atm)
# ===========================================================================

CARBON_CARBON = MaterialEntry(
    name="Carbon-Carbon Composite",
    category="carbon",
    density_kgm3=1900.0,
    tensile_strength_mpa=280.0,     # in-plane tensile
    # C/C composites retain or slightly increase strength to ~1000°C then hold;
    # carbon sublimes rather than softens, so strength is maintained until very
    # high temperature in inert environment.
    tensile_strength_at_temp={
        293.0: 280.0,
        1273.0: 300.0,
        2073.0: 280.0,
        2773.0: 200.0,
    },
    compressive_strength_mpa=350.0,
    service_temp_air_K=1773.0,    # SiC-coated operational form (X-43A, Space Shuttle RCC)
    service_temp_inert_K=3073.0,
    melting_point_K=3823.0,       # carbon sublimation temperature
    thermal_conductivity_WmK=130.0,   # in-plane (through-thickness ~10 W/m·K)
    thermal_expansion_1K=1.0e-6,      # in-plane (near-zero or slightly negative)
    youngs_modulus_GPa=100.0,
    oxidation_resistance="limited",   # SiC coating provides oxidation protection but can crack
    oxidation_max_temp_K=1773.0,  # SiC coating oxidation limit (~1500°C)
    applicable_regimes=["hypersonic", "reentry"],
    citation="NASA TM-2002-211892 C/C composites; Frontiers in Materials 2024 C/C review; cfccarbon.com properties",
    notes="In-plane values; through-thickness k ~10 W/m·K; SiC-coated form used in aerospace (X-43A nose/leading edges, Space Shuttle RCC); uncoated base oxidizes above ~500°C; service_temp_air = SiC coating limit; melting_point_K = sublimation",
    cost_usd_per_kg=1500.0,  # cost ~$1500/kg - Carbon-Carbon 2D/3D weave, legacy shuttle-era pricing
)

PYROLYTIC_GRAPHITE = MaterialEntry(
    name="Pyrolytic Graphite",
    category="carbon",
    density_kgm3=2220.0,
    tensile_strength_mpa=80.0,      # in-plane
    tensile_strength_at_temp={
        293.0: 80.0,
        1273.0: 85.0,
        2073.0: 75.0,
        2773.0: 55.0,
    },
    compressive_strength_mpa=120.0,
    service_temp_air_K=773.0,
    service_temp_inert_K=3073.0,
    melting_point_K=3823.0,        # carbon sublimation temperature
    thermal_conductivity_WmK=1700.0,   # in-plane (through-thickness ~5 W/m·K)
    thermal_expansion_1K=0.5e-6,       # in-plane
    youngs_modulus_GPa=50.0,
    oxidation_resistance="poor",
    oxidation_max_temp_K=773.0,
    applicable_regimes=["hypersonic", "reentry"],
    citation="Minerals Technology Pyroid HT data sheet; DTIC AD0437144; Wikipedia pyrolytic graphite",
    notes="Extremely anisotropic; in-plane k=1700 W/m·K vs through-thickness ~5 W/m·K; rocket nozzle ablator liners; thermal management for hypersonic seekers; melting_point_K = sublimation",
    cost_usd_per_kg=8000.0,  # cost ~$8000/kg - Pyrolytic graphite anisotropic CVD, rocket nozzle grade
)

DIAMOND_CVD = MaterialEntry(
    name="Diamond CVD",
    category="carbon",
    density_kgm3=3520.0,
    tensile_strength_mpa=2800.0,   # flexural/fracture strength — brittle
    tensile_strength_at_temp={
        293.0: 2800.0,
        873.0: 2520.0,
        1073.0: 1960.0,
        1800.0: 560.0,
    },
    compressive_strength_mpa=8900.0,
    service_temp_air_K=973.0,     # combustion to CO2 above ~700°C; conservative air limit
    service_temp_inert_K=2073.0,
    melting_point_K=3823.0,       # converts to graphite ~1700 K; sublimation 3823 K
    thermal_conductivity_WmK=2000.0,
    thermal_expansion_1K=1.1e-6,
    youngs_modulus_GPa=1050.0,
    oxidation_resistance="poor",
    oxidation_max_temp_K=973.0,
    applicable_regimes=["hypersonic", "reentry"],
    citation="Element Six CVD Diamond Handbook; Seki Diamond CVD properties; ScienceDirect CVD diamond material properties",
    notes="Tensile values are flexural/fracture strength; highest thermal conductivity of any material (2000 W/m·K); coatings and small components only; dome windows for hypersonic IR seekers; burns in air above 800°C; melting_point_K = graphite conversion/sublimation",
    cost_usd_per_kg=8000.0,  # cost ~$8000/kg - CVD diamond plate, optical/thermal management grade
)


# ===========================================================================
# THERMAL PROTECTION SYSTEMS (TPS) — ABLATORS
# applicable_regimes: reentry ONLY
# service_temp_air_K and service_temp_inert_K = rated surface temperature as
# ablative barrier, NOT structural service temperature
# melting_point_K = carbon sublimation for carbon-based ablators
# ===========================================================================

PICA = MaterialEntry(
    name="PICA",
    category="tps",
    density_kgm3=270.0,
    tensile_strength_mpa=1.5,
    tensile_strength_at_temp={
        293.0: 1.5,
        500.0: 1.2,
        760.0: 0.8,
        1200.0: 0.3,
    },
    compressive_strength_mpa=2.1,
    service_temp_air_K=3000.0,    # rated ablative surface temperature, not structural
    service_temp_inert_K=3200.0,
    melting_point_K=3823.0,       # carbon sublimation
    thermal_conductivity_WmK=0.22,
    thermal_expansion_1K=3.5e-6,
    youngs_modulus_GPa=0.30,
    oxidation_resistance="excellent",   # ablates rather than oxidizes in reentry boundary layer
    oxidation_max_temp_K=3000.0,
    applicable_regimes=["reentry"],
    citation="NASA TN-D-7120 Tran et al.; NASA science.nasa.gov PICA page; AIAA 2011-3913",
    notes="Phenolic Impregnated Carbon Ablator; 270 kg/m³; service_temp ratings are ablative surface limits, not structural; Stardust capsule, Dragon capsule, Mars missions; not for non-reentry structures",
    cost_usd_per_kg=400.0,  # cost ~$400/kg - PICA ablator, MSL/Orion heritage pricing
)

PICA_X = MaterialEntry(
    name="PICA-X",
    category="tps",
    density_kgm3=240.0,
    tensile_strength_mpa=1.3,
    tensile_strength_at_temp={
        293.0: 1.3,
        500.0: 1.0,
        760.0: 0.7,
        1200.0: 0.25,
    },
    compressive_strength_mpa=1.8,
    service_temp_air_K=3000.0,
    service_temp_inert_K=3200.0,
    melting_point_K=3823.0,
    thermal_conductivity_WmK=0.20,
    thermal_expansion_1K=3.2e-6,
    youngs_modulus_GPa=0.28,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=3000.0,
    applicable_regimes=["reentry"],
    citation="SpaceX AIAA 2011 PICA-X paper; NASA science page",
    notes="SpaceX proprietary PICA variant; lower density than NASA PICA (240 vs 270 kg/m³); Dragon capsule heat shield; service_temp ratings are ablative surface limits, not structural",
    cost_usd_per_kg=600.0,  # cost ~$600/kg - PICA-X SpaceX proprietary, Dragon heritage
)

AVCOAT = MaterialEntry(
    name="AVCOAT",
    category="tps",
    density_kgm3=529.0,
    tensile_strength_mpa=0.5,
    tensile_strength_at_temp={
        293.0: 0.5,
        500.0: 0.4,
        760.0: 0.25,
        1200.0: 0.08,
    },
    compressive_strength_mpa=1.0,
    service_temp_air_K=3500.0,
    service_temp_inert_K=3600.0,
    melting_point_K=3823.0,
    thermal_conductivity_WmK=0.242,
    thermal_expansion_1K=9.0e-6,
    youngs_modulus_GPa=0.20,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=3500.0,
    applicable_regimes=["reentry"],
    citation="Wikipedia AVCOAT; Apollo/Orion MPCV program specifications; NASA NTRS Orion AVCOAT",
    notes="Epoxy novolac + glass microballoons in fiberglass honeycomb; Apollo Command Module heat shield; Orion MPCV; service_temp ratings are ablative surface limits, not structural",
    cost_usd_per_kg=400.0,  # cost ~$400/kg - AVCOAT ablator, Apollo/Orion heritage
)

RCC = MaterialEntry(
    name="RCC",
    category="tps",
    density_kgm3=1600.0,
    tensile_strength_mpa=95.0,
    tensile_strength_at_temp={
        293.0: 95.0,
        1273.0: 100.0,
        2073.0: 92.0,
        2773.0: 65.0,
    },
    compressive_strength_mpa=200.0,
    service_temp_air_K=1783.0,    # SiC-coated RCC; uncoated oxidizes above 500°C
    service_temp_inert_K=3073.0,
    melting_point_K=3823.0,       # carbon sublimation
    thermal_conductivity_WmK=6.3,
    thermal_expansion_1K=1.5e-6,
    youngs_modulus_GPa=55.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=1783.0,   # with SiC coating
    applicable_regimes=["reentry"],
    citation="Wikipedia Space Shuttle TPS; NASA Fandom TPS article; NTRS thermal protection review",
    notes="Reinforced Carbon-Carbon; SiC-coated C/C; Space Shuttle nose cap and wing leading edges; reusable to 1510°C; oxidation_resistance 'good' reflects the operational SiC-coated form; uncoated C/C would be 'poor'",
    cost_usd_per_kg=3000.0,  # cost ~$3000/kg - RCC Shuttle-grade reinforced carbon-carbon
)

LI900 = MaterialEntry(
    name="LI-900",
    category="tps",
    density_kgm3=144.0,
    tensile_strength_mpa=0.5,
    tensile_strength_at_temp={
        293.0: 0.5,
        811.0: 0.45,
        1144.0: 0.35,
        1533.0: 0.20,
    },
    compressive_strength_mpa=0.8,
    service_temp_air_K=1533.0,
    service_temp_inert_K=1700.0,
    melting_point_K=1983.0,   # amorphous silica glass transition/devitrification
    thermal_conductivity_WmK=0.066,
    thermal_expansion_1K=0.45e-6,
    youngs_modulus_GPa=0.06,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1533.0,
    applicable_regimes=["reentry"],
    citation="NASA TPSX Database Material ID 1 LI-900; Wikipedia LI-900; NASA CR-174235",
    notes="99.9% amorphous silica; 94% air by volume; 144 kg/m³; Space Shuttle lower surface tiles; very fragile; service_temp is operational surface temperature limit",
    cost_usd_per_kg=500.0,  # cost ~$500/kg - LI-900 shuttle silica tile
)

SLA561V = MaterialEntry(
    name="SLA-561V",
    category="tps",
    density_kgm3=256.0,
    tensile_strength_mpa=0.5,
    tensile_strength_at_temp={
        293.0: 0.5,
        500.0: 0.4,
        760.0: 0.25,
        1200.0: 0.08,
    },
    compressive_strength_mpa=0.7,
    service_temp_air_K=3000.0,
    service_temp_inert_K=3000.0,
    melting_point_K=3823.0,
    thermal_conductivity_WmK=0.14,
    thermal_expansion_1K=30.0e-6,
    youngs_modulus_GPa=0.25,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=3000.0,
    applicable_regimes=["reentry"],
    citation="Lockheed Martin SLA-561 Product Information sheet; NASA TM-110402; NTRS 20100033684",
    notes="Silicone elastomer/silica microballoon foam; Mars Viking, Pathfinder, MER, MSL, InSight aeroshells; rated 225 W/cm² heat flux; service_temp ratings are ablative surface limits",
    cost_usd_per_kg=400.0,  # cost ~$400/kg - SLA-561V Viking/MSL heritage ablator
)

TUFI = MaterialEntry(
    name="TUFI",
    category="tps",
    density_kgm3=192.0,
    tensile_strength_mpa=0.3,
    tensile_strength_at_temp={
        293.0: 0.3,
        500.0: 0.25,
        760.0: 0.15,
        1200.0: 0.05,
    },
    compressive_strength_mpa=0.6,
    service_temp_air_K=1533.0,
    service_temp_inert_K=1533.0,
    melting_point_K=1983.0,    # silica fiber softening
    thermal_conductivity_WmK=0.06,
    thermal_expansion_1K=0.5e-6,
    youngs_modulus_GPa=0.04,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1533.0,
    applicable_regimes=["reentry"],
    citation="NASA Ames TUFI data; Leiser et al., NASA TM-112158",
    notes="Toughened Uni-piece Fibrous Insulation; alumina-borosilicate fiber tile with reaction-cured glass coating; X-37B leeward TPS; handles higher surface pressure than LI-900",
    cost_usd_per_kg=600.0,  # cost ~$600/kg - TUFI toughened unipiece fibrous insulation coating
)

AETB8 = MaterialEntry(
    name="AETB-8",
    category="tps",
    density_kgm3=128.0,
    tensile_strength_mpa=0.2,
    tensile_strength_at_temp={
        293.0: 0.2,
        500.0: 0.17,
        760.0: 0.10,
        1200.0: 0.04,
    },
    compressive_strength_mpa=0.4,
    service_temp_air_K=1644.0,
    service_temp_inert_K=1644.0,
    melting_point_K=1983.0,    # silica fiber softening
    thermal_conductivity_WmK=0.05,
    thermal_expansion_1K=0.5e-6,
    youngs_modulus_GPa=0.03,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1644.0,
    applicable_regimes=["reentry"],
    citation="NASA Ames AETB data; Goldstein et al., NASA TM-4713",
    notes="Alumina Enhanced Thermal Barrier, 8 pcf density; silica + alumina + aluminoborosilicate fiber mix; higher-temperature successor to LI-900; Space Shuttle upgrade studies",
    cost_usd_per_kg=500.0,  # cost ~$500/kg - AETB-8 advanced enhanced thermal blanket
)

SIRCA = MaterialEntry(
    name="SIRCA",
    category="tps",
    density_kgm3=256.0,
    tensile_strength_mpa=0.6,
    tensile_strength_at_temp={
        293.0: 0.6,
        500.0: 0.5,
        760.0: 0.3,
        1200.0: 0.1,
    },
    compressive_strength_mpa=1.0,
    service_temp_air_K=1900.0,
    service_temp_inert_K=1900.0,
    melting_point_K=3100.0,     # silicone impregnant pyrolysis + silica backbone
    thermal_conductivity_WmK=0.12,
    thermal_expansion_1K=1.5e-6,
    youngs_modulus_GPa=0.08,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1900.0,
    applicable_regimes=["reentry"],
    citation="NASA Ames SIRCA data; Congdon et al., AIAA-2003-2165",
    notes="Silicone Impregnated Reusable Ceramic Ablator; silicone-filled fibrous tile; Mars Pathfinder backshell patches; machineable to complex shapes",
    cost_usd_per_kg=400.0,  # cost ~$400/kg - SIRCA silicone-impregnated ceramic ablator
)

CARBON_PHENOLIC = MaterialEntry(
    name="Carbon phenolic",
    category="tps",
    density_kgm3=1440.0,
    tensile_strength_mpa=3.0,
    tensile_strength_at_temp={
        293.0: 3.0,
        500.0: 2.5,
        1000.0: 1.5,
        2000.0: 0.5,
    },
    compressive_strength_mpa=5.0,
    service_temp_air_K=3500.0,
    service_temp_inert_K=3500.0,
    melting_point_K=3923.0,    # carbon sublimation
    thermal_conductivity_WmK=0.5,
    thermal_expansion_1K=1.0e-6,
    youngs_modulus_GPa=3.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=3500.0,
    applicable_regimes=["reentry"],
    citation="Raytheon MK-12A RV data; NASA CR-195059; Aerojet ablative data",
    notes="Chopped carbon fiber in phenolic resin; highest heat flux TPS material; ICBM reentry vehicles, Pioneer/Galileo probes; service_temp is ablative limit not structural",
    cost_usd_per_kg=400.0,  # cost ~$400/kg - Carbon phenolic, baseline ablator for heat shields
)


# ── General Engineering Materials ──────────────────────────────────────────
# Subsonic-only category for accessible, affordable materials used by
# university teams, drone startups, and small hardware companies.
# Not suitable for supersonic or higher regimes due to thermal/structural limits.

GE_A36 = MaterialEntry(
    name="Structural Steel A36",
    category="general_engineering",
    density_kgm3=7850.0,
    tensile_strength_mpa=400.0,
    tensile_strength_at_temp={293.0: 400.0, 423.0: 368.0, 533.0: 308.0, 673.0: 200.0},
    compressive_strength_mpa=400.0,
    service_temp_air_K=673.0,
    service_temp_inert_K=750.0,
    melting_point_K=1811.0,
    thermal_conductivity_WmK=50.0,
    thermal_expansion_1K=12.0e-6,
    youngs_modulus_GPa=200.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=773.0,
    applicable_regimes=["subsonic"],
    citation="ASTM A36/A36M standard",
    notes="Most widely used structural steel; low cost; good weldability; not suitable above subsonic regime due to strength degradation at elevated temperature",
    cost_usd_per_kg=2.0,  # cost ~$2/kg - A36 structural steel, commodity construction grade
)

GE_SS304 = MaterialEntry(
    name="Stainless Steel 304",
    category="general_engineering",
    density_kgm3=8000.0,
    tensile_strength_mpa=515.0,
    tensile_strength_at_temp={293.0: 515.0, 533.0: 450.0, 755.0: 380.0, 1033.0: 180.0},
    compressive_strength_mpa=515.0,
    service_temp_air_K=1033.0,
    service_temp_inert_K=1100.0,
    melting_point_K=1727.0,
    thermal_conductivity_WmK=16.0,
    thermal_expansion_1K=17.2e-6,
    youngs_modulus_GPa=193.0,
    oxidation_resistance="excellent",
    oxidation_max_temp_K=1033.0,
    applicable_regimes=["subsonic"],
    citation="ASM Handbook Volume 1",
    notes="Excellent corrosion resistance; widely available; used in structural and pressure vessel applications; subsonic only — austenitic steel not rated for elevated-speed aerodynamic heating",
    cost_usd_per_kg=5.0,  # cost ~$5/kg - 304 stainless bulk sheet/plate
)

GE_MILD1020 = MaterialEntry(
    name="Mild Steel 1020",
    category="general_engineering",
    density_kgm3=7870.0,
    tensile_strength_mpa=395.0,
    tensile_strength_at_temp={293.0: 395.0, 423.0: 364.0, 533.0: 304.0, 573.0: 200.0},
    compressive_strength_mpa=395.0,
    service_temp_air_K=573.0,     # capped to oxidation_max_temp_K; oxidation is the binding limit
    service_temp_inert_K=650.0,
    melting_point_K=1793.0,
    thermal_conductivity_WmK=51.0,
    thermal_expansion_1K=11.7e-6,
    youngs_modulus_GPa=200.0,
    oxidation_resistance="limited",
    oxidation_max_temp_K=573.0,
    applicable_regimes=["subsonic"],
    citation="ASM Handbook Volume 1",
    notes="Baseline carbon steel reference; low cost; easy to machine and weld; limited temperature capability; service_temp_air capped to oxidation limit (573 K, not the 673 K structural limit)",
    cost_usd_per_kg=2.0,  # cost ~$2/kg - 1020 mild steel, commodity carbon steel
)

GE_ABS = MaterialEntry(
    name="ABS Plastic",
    category="general_engineering",
    density_kgm3=1050.0,
    tensile_strength_mpa=40.0,
    tensile_strength_at_temp={293.0: 40.0, 323.0: 32.0, 353.0: 20.0, 373.0: 1.0},
    compressive_strength_mpa=40.0,
    service_temp_air_K=373.0,
    service_temp_inert_K=373.0,
    melting_point_K=543.0,        # decomposition temperature
    thermal_conductivity_WmK=0.17,
    thermal_expansion_1K=90.0e-6,
    youngs_modulus_GPa=2.3,
    oxidation_resistance="good",
    oxidation_max_temp_K=373.0,
    applicable_regimes=["subsonic"],
    citation="Matweb ABS generic",
    notes="Widely used for prototyping and low-load structural parts; not suitable for elevated temperature; service_temp is continuous use limit not decomposition; melting_point_K = decomposition temperature 543 K",
    cost_usd_per_kg=3.0,  # cost ~$3/kg - ABS plastic injection-grade resin
)

GE_NYLON66 = MaterialEntry(
    name="Nylon 66",
    category="general_engineering",
    density_kgm3=1140.0,
    tensile_strength_mpa=80.0,
    tensile_strength_at_temp={293.0: 80.0, 323.0: 64.0, 353.0: 40.0, 383.0: 8.0, 393.0: 1.0},
    compressive_strength_mpa=80.0,
    service_temp_air_K=393.0,
    service_temp_inert_K=393.0,
    melting_point_K=533.0,
    thermal_conductivity_WmK=0.25,
    thermal_expansion_1K=80.0e-6,
    youngs_modulus_GPa=2.7,
    oxidation_resistance="good",
    oxidation_max_temp_K=393.0,
    applicable_regimes=["subsonic"],
    citation="Matweb Nylon 66 generic",
    notes="Good wear resistance and mechanical properties for low-temperature applications; absorbs moisture which affects strength; melting point is true crystalline melt",
    cost_usd_per_kg=5.0,  # cost ~$5/kg - Nylon 66 engineering resin
)

GE_GFRP = MaterialEntry(
    name="GFRP Generic",
    category="general_engineering",
    density_kgm3=1800.0,
    tensile_strength_mpa=300.0,
    tensile_strength_at_temp={293.0: 300.0, 323.0: 240.0, 353.0: 150.0, 383.0: 30.0, 423.0: 3.0},
    compressive_strength_mpa=250.0,
    service_temp_air_K=423.0,
    service_temp_inert_K=423.0,
    melting_point_K=573.0,        # resin decomposition
    thermal_conductivity_WmK=0.3,
    thermal_expansion_1K=14.0e-6,
    youngs_modulus_GPa=25.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=423.0,
    applicable_regimes=["subsonic"],
    citation="ASM Engineered Materials Handbook Volume 1 Composites",
    notes="Woven E-glass fiber in epoxy matrix; widely available and affordable; tensile value is in-plane; stored as flexural proxy; melting_point_K = resin decomposition",
    cost_usd_per_kg=20.0,  # cost ~$20/kg - Generic GFRP, glass-epoxy consumer/industrial grade
)

GE_CFRP = MaterialEntry(
    name="CFRP Generic",
    category="general_engineering",
    density_kgm3=1550.0,
    tensile_strength_mpa=600.0,
    tensile_strength_at_temp={293.0: 600.0, 323.0: 480.0, 353.0: 300.0, 383.0: 60.0, 423.0: 6.0},
    compressive_strength_mpa=500.0,
    service_temp_air_K=423.0,
    service_temp_inert_K=450.0,
    melting_point_K=673.0,        # resin decomposition
    thermal_conductivity_WmK=5.0,
    thermal_expansion_1K=3.0e-6,
    youngs_modulus_GPa=70.0,
    oxidation_resistance="good",
    oxidation_max_temp_K=423.0,
    applicable_regimes=["subsonic"],
    citation="ASM Engineered Materials Handbook Volume 1 Composites",
    notes="Generic woven CFRP in epoxy matrix; represents accessible commercial grades, not the aerospace unidirectional prepreg grades already in the database; tensile value is in-plane; stored as flexural proxy; melting_point_K = resin decomposition",
    cost_usd_per_kg=80.0,  # cost ~$80/kg - Generic CFRP, commercial woven-epoxy, not aerospace prepreg
)


# ===========================================================================
# Master database list
# ===========================================================================

MATERIALS_DB: list = [
    # Aluminum alloys
    AL_2024_T3,
    AL_7075_T6,
    AL_7068_T6511,
    AL_6061_T6,
    AL_2195_ALRLI,
    AL_2099_ALRLI,
    AL_2219_T87,
    AL_7050_T7451,
    AL_7010_T7451,
    AL_8090_ALLI,
    # Titanium alloys
    TI_6AL4V,
    TI_6242,
    TI_3AL25V,
    TI_15V_3,
    TI_10V_2FE_3AL,
    TI_6AL4V_ELI,
    TI_5AL25SN,
    TI_6AL6V2SN,
    TI_BETAC,
    # Nickel superalloys
    IN718,
    IN625,
    IN625_LCF,
    IN_X750,
    WASPALOY,
    RENE41,
    HAYNES230,
    HAYNES282,
    MAR_M247,
    CMSX4,
    PWA1484,
    NIMONIC_90,
    NIMONIC_105,
    UDIMET_720,
    RENE_88DT,
    RENE_125,
    IN100,
    ASTROLOY,
    # Steel alloys
    STEEL_4340,
    STEEL_300M,
    STEEL_AF1410,
    STEEL_174PH,
    STEEL_155PH,
    STEEL_MAR350,
    STEEL_HY80,
    STEEL_HY100,
    STEEL_PH138MO,
    STEEL_AERMET100,
    STEEL_GREEKASCOLOY,
    # Cobalt superalloys
    HAYNES_188,
    L605,
    MP35N,
    ELGILOY,
    # Polymer matrix composites
    CFRP_IM7_977_3,
    CFRP_T800_3900,
    CFRP_IM7_BMI,
    CFRP_AS4_PEEK,
    CFRP_IM7_5250,
    CFRP_IM7_PETI,
    CFRP_T300_934,
    # Ceramic matrix composites
    CMC_SIC_SIC,
    CMC_C_SIC,
    CMC_OXIDE,
    CMC_NEXTEL610,
    CMC_HI_NICALON,
    # Refractory metals
    W_METAL,
    MO_METAL,
    RE_METAL,
    NB_C103,
    TA_METAL,
    # Ultra-high temperature ceramics
    ZRB2_SIC20,
    HFB2_SIC20,
    ZRB2_MONO,
    HFB2_MONO,
    TA4HFC5,
    ZRC,
    TAC,
    ZRB2_MOSI2,
    # Carbon materials
    CARBON_CARBON,
    PYROLYTIC_GRAPHITE,
    DIAMOND_CVD,
    # TPS ablators
    PICA,
    PICA_X,
    AVCOAT,
    RCC,
    LI900,
    SLA561V,
    TUFI,
    AETB8,
    SIRCA,
    CARBON_PHENOLIC,
    # General engineering materials (subsonic only)
    GE_A36,
    GE_SS304,
    GE_MILD1020,
    GE_ABS,
    GE_NYLON66,
    GE_GFRP,
    GE_CFRP,
]


# ===========================================================================
# Availability scoring
# ===========================================================================
# Scores reflect commercial availability of each material:
#   1.0  — commercially available from multiple suppliers
#   0.7  — limited production (one or two specialized manufacturers)
#   0.5  — developmental (advanced development or limited military production)
#   0.3  — laboratory scale (characterized in research, not commercially produced)
#
# Materials not listed default to 1.0 (commercially available).

_AVAILABILITY_OVERRIDES: dict[str, float] = {
    # Single-crystal / DS nickel superalloys — specialized foundries only
    "CMSX-4":              0.7,
    "PWA 1484":            0.7,
    "MAR-M 247 (DS)":     0.7,
    "René 125":            0.7,
    # P/M disc alloys — specialized powder metallurgy supply chain
    "René 88DT":           0.7,
    "Udimet 720":          0.7,
    "IN-100":              0.7,
    "Astroloy":            0.7,
    # Advanced CMCs — limited suppliers (GE, Safran, COI Ceramics)
    "SiC/SiC CMC":         0.7,
    "C/SiC CMC":           0.7,
    "Hi-Nicalon SiC/SiC CMC": 0.7,
    "Nextel 610/alumina CMC":  0.7,
    # UHTCs — most are lab-scale or developmental
    "Ta4HfC5":             0.3,
    "ZrC":                 0.5,
    "TaC":                 0.5,
    "ZrB2-MoSi2":          0.5,
    "ZrB2 Monolithic":     0.5,
    "HfB2 Monolithic":     0.5,
    "ZrB2-SiC 20vol%":     0.5,
    "HfB2-SiC 20vol%":     0.5,
    # CVD diamond — specialized deposition, not structural-scale
    "Diamond CVD":         0.3,
    # Developmental TPS
    "SIRCA":               0.5,
    "TUFI":                0.7,
    "AETB-8":              0.7,
}

# Apply overrides to the database
for _mat in MATERIALS_DB:
    if _mat.name in _AVAILABILITY_OVERRIDES:
        object.__setattr__(_mat, "availability_score", _AVAILABILITY_OVERRIDES[_mat.name])


# ===========================================================================
# Larson-Miller creep data (lifecycle modelling — phase 1)
# ===========================================================================
#
# The dictionary below carries (T_K, t_hours, rupture_stress_MPa) data
# points for the priority materials listed in the lifecycle plan. The
# helper ``_apply_creep_data`` then computes the Larson-Miller parameter
# LMP = T(C + log10(t)) for each point and writes the resulting
# (LMP, sigma) curve onto the corresponding MaterialEntry.
#
# Data convention
# ---------------
#
# * ``status="sourced"`` means the rupture-stress points are taken
#   from a primary reference (manufacturer datasheet or MMPDS-17 /
#   ASM Handbook chapter cited in the ``source`` field). The
#   numbers below are rounded to 2-3 significant figures, which is
#   the practical accuracy of stress-rupture screening.
# * ``status="estimated"`` means the curve is an engineering
#   extrapolation from a closely-related alloy whose data IS
#   sourced. The ``source`` field calls this out explicitly so a
#   user sees the provenance in the report. Estimated curves are
#   used as a flag in the matching engine (still applied, but
#   surfaced with a note).
# * Larson-Miller constant ``C``: 13-17 for aluminium (lower
#   because Al creeps faster relative to its melting point), 17-20
#   for titanium and steel, 20-22 for nickel superalloys.
#
# Sources used
# ------------
#
# Special Metals: Inconel 718 (SMC-045), Inconel 625 (SMC-063),
# Inconel X-750 (SMC-067), Waspaloy (SMC-011). These manufacturer
# datasheets are publicly downloadable and quote stress-rupture
# tables in the (T, t) format reproduced below.
#
# ASM Specialty Handbook: Heat-Resistant Materials (1997) — Tables
# 6.6 through 6.18 cover most of the cast / wrought / single-crystal
# nickel superalloys (CMSX-4, MAR-M-247, IN-100, Astroloy, Udimet
# 720, René series, Nimonic series).
#
# MMPDS-17: Tables 3.x.x.0 (aluminum) and 5.x.x.0 (titanium). The
# stress-rupture values reproduced below are the elevated-temperature
# rows of those tables, transcribed at the (T, t) coordinates the
# matching engine will look up.
#
# RTI International "Titanium Alloy Guide" (2000) and Boyer (1994)
# "Materials Properties Handbook: Titanium Alloys" — Ti-6Al-4V and
# the elevated-temperature alpha-beta grades.
# ---------------------------------------------------------------------------

import math

_CREEP_DATA: dict[str, dict] = {
    # -------------------------------------------------------------------
    # NICKEL SUPERALLOYS — the highest-leverage creep data because every
    # turbine-category result depends on whether (CMSX-4, IN-100,
    # Inconel 718, ...) survive sustained T_wall ~1400 K x ~25,000 h.
    # -------------------------------------------------------------------
    "Inconel 718": {
        "C": 20.0,
        "points": [
            # (T_K, t_hours, rupture_stress_MPa)
            (922,   100,  770),   # 650 °C, 100 h
            (922,  1000,  620),   # 650 °C, 1,000 h
            (922, 10000,  470),   # 650 °C, 10,000 h
            (977,  1000,  370),   # 704 °C, 1,000 h
            (977, 10000,  250),   # 704 °C, 10,000 h
            (1033, 1000,  190),   # 760 °C, 1,000 h
            (1033, 10000, 105),   # 760 °C, 10,000 h
        ],
        "source": "Special Metals Inconel 718 datasheet (SMC-045) Table 9; ASM Handbook Vol 1 Table 6.6",
        "status": "sourced",
    },
    "Inconel 625": {
        "C": 20.0,
        "points": [
            (922,  1000,  330),
            (922, 10000,  250),
            (977,  1000,  200),
            (977, 10000,  140),
            (1033, 1000,  120),
            (1089, 1000,   75),   # 816 °C, 1,000 h
        ],
        "source": "Special Metals Inconel 625 datasheet (SMC-063) Table 7",
        "status": "sourced",
    },
    "Inconel X-750": {
        "C": 20.0,
        "points": [
            (922,   100,  620),
            (922,  1000,  480),
            (922, 10000,  330),
            (977,  1000,  270),
            (977, 10000,  150),
            (1033, 1000,  130),
        ],
        "source": "Special Metals Inconel X-750 datasheet (SMC-067) Table 5",
        "status": "sourced",
    },
    "Waspaloy": {
        "C": 20.0,
        "points": [
            (922,  1000,  620),
            (977,  1000,  440),
            (1033, 1000,  270),
            (1089, 1000,  140),
        ],
        "source": "Special Metals Waspaloy datasheet (SMC-011); ASM Specialty Handbook Heat-Resistant Materials Table 6.10",
        "status": "sourced",
    },
    "CMSX-4": {
        # Single-crystal nickel: the highest-temperature commercial
        # turbine blade alloy. Curve covers 1273-1473 K (1000-1200 °C),
        # which is exactly where the CFM56 HPT-blade preset operates.
        "C": 20.0,
        "points": [
            (1273,  100,  280),
            (1273, 1000,  190),
            (1373,  100,  155),
            (1373, 1000,   95),
            (1473,  100,   70),
        ],
        "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.18; Reed, *The Superalloys* (2006) Fig. 6.21",
        "status": "sourced",
    },
    "PWA 1484": {
        # 2nd-generation single-crystal Ni — slightly stronger than
        # CMSX-4 at high T. Estimated by scaling CMSX-4 by ~+10%.
        "C": 20.0,
        "points": [
            (1273,  100,  310),
            (1273, 1000,  210),
            (1373,  100,  170),
            (1373, 1000,  105),
            (1473,  100,   77),
        ],
        "source": "Estimated from CMSX-4 with +10% scaling per Pratt & Whitney 1st-vs-2nd-gen SC published comparisons",
        "status": "estimated",
    },
    "MAR-M 247 (DS)": {
        # Directionally-solidified nickel superalloy — between
        # conventional cast and single-crystal in capability.
        "C": 20.0,
        "points": [
            (1173,  100,  480),
            (1173, 1000,  330),
            (1273,  100,  280),
            (1273, 1000,  180),
            (1373, 1000,   85),
        ],
        "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.16",
        "status": "sourced",
    },
    "IN-100": {
        "C": 20.0,
        "points": [
            (977,  1000,  620),
            (1033, 1000,  480),
            (1089, 1000,  330),
            (1144, 1000,  210),
        ],
        "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.12",
        "status": "sourced",
    },
    "Astroloy": {
        "C": 20.0,
        "points": [
            (977,  1000,  580),
            (1033, 1000,  440),
            (1089, 1000,  290),
        ],
        "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.13",
        "status": "sourced",
    },
    "Nimonic 90": {
        "C": 20.0,
        "points": [
            (922,  1000,  310),
            (977,  1000,  210),
            (1033, 1000,  140),
        ],
        "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.14",
        "status": "sourced",
    },
    "Nimonic 105": {
        "C": 20.0,
        "points": [
            (922,  1000,  440),
            (977,  1000,  310),
            (1033, 1000,  210),
            (1089, 1000,  140),
        ],
        "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.14",
        "status": "sourced",
    },
    "Udimet 720": {
        "C": 20.0,
        "points": [
            (977,  1000,  620),
            (1033, 1000,  440),
            (1089, 1000,  290),
        ],
        "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.15",
        "status": "sourced",
    },
    "Rene 41": {
        "C": 20.0,
        "points": [
            (922,  1000,  480),
            (977,  1000,  330),
            (1033, 1000,  210),
        ],
        "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.11",
        "status": "sourced",
    },
    "René 88DT": {
        "C": 20.0,
        "points": [
            (977,  1000,  620),
            (1033, 1000,  440),
            (1089, 1000,  280),
        ],
        "source": "Estimated from related P/M disc-alloy literature (similar to Udimet 720)",
        "status": "estimated",
    },
    "René 125": {
        "C": 20.0,
        "points": [
            (1173, 1000,  310),
            (1273, 1000,  170),
            (1373, 1000,   85),
        ],
        "source": "Estimated from MAR-M-247 (DS) bracketing — similar conventionally-cast Ni superalloy",
        "status": "estimated",
    },
    "Haynes 230": {
        # Solid-solution-strengthened — designed for very-high-T
        # service but lower stress capability than precipitation-
        # hardened gamma-prime alloys.
        "C": 20.0,
        "points": [
            (977,  1000,  140),
            (1033, 1000,   95),
            (1089, 1000,   62),
            (1144, 1000,   41),
            (1255, 1000,   22),   # 982 °C, 1,000 h
        ],
        "source": "Haynes International alloy 230 brochure H-3000H Table 8",
        "status": "sourced",
    },
    "Haynes 282": {
        "C": 20.0,
        "points": [
            (977,  1000,  310),
            (1033, 1000,  210),
            (1089, 1000,  120),
        ],
        "source": "Haynes International alloy 282 brochure H-3173 Table 6",
        "status": "sourced",
    },

    # -------------------------------------------------------------------
    # TITANIUM ALLOYS — Concorde / SR-71 territory. Operative
    # temperature range 600-900 K. Most Ti alloys creep beyond ~700 K
    # (430 °C), with Ti-6Al-4V the workhorse and Ti-6242 designed for
    # higher T.
    # -------------------------------------------------------------------
    "Ti-6Al-4V": {
        "C": 20.0,
        "points": [
            (700, 1000, 620),    # 427 °C, 1,000 h
            (755,  100, 480),    # 482 °C, 100 h
            (755, 1000, 310),    # 482 °C, 1,000 h
            (811,  100, 310),    # 538 °C, 100 h
            (811, 1000, 140),    # 538 °C, 1,000 h
        ],
        "source": "RTI International Titanium Alloy Guide (2000) p. 22; Boyer (1994) Materials Properties Handbook: Titanium Alloys p. 506",
        "status": "sourced",
    },
    "Ti-6Al-4V ELI": {
        # Extra-low-interstitial grade — slightly lower σ, similar
        # creep resistance.
        "C": 20.0,
        "points": [
            (700, 1000, 580),
            (755,  100, 440),
            (755, 1000, 290),
            (811,  100, 290),
            (811, 1000, 130),
        ],
        "source": "Estimated from Ti-6Al-4V with -7% scaling for ELI grade",
        "status": "estimated",
    },
    "Ti-6Al-2Sn-4Zr-2Mo": {
        # Ti-6242 — designed for elevated-T service (engine compressor
        # discs). Best Ti for sustained 700-900 K range.
        "C": 20.0,
        "points": [
            (755, 1000, 410),
            (811, 1000, 270),
            (866, 1000, 140),    # 593 °C, 1,000 h
        ],
        "source": "Boyer (1994) Materials Properties Handbook: Titanium Alloys p. 583",
        "status": "sourced",
    },
    "Ti-5Al-2.5Sn": {
        # All-alpha alloy, lower strength than Ti-6Al-4V at
        # room T but holds up reasonably at elevated T.
        "C": 20.0,
        "points": [
            (700, 1000, 480),
            (755, 1000, 290),
            (811, 1000, 140),
        ],
        "source": "Boyer (1994) Materials Properties Handbook: Titanium Alloys p. 367",
        "status": "sourced",
    },
    "Ti-3Al-2.5V": {
        # Near-alpha tube alloy.
        "C": 20.0,
        "points": [
            (700, 1000, 350),
            (755, 1000, 210),
        ],
        "source": "Estimated from Ti-5Al-2.5Sn — same near-alpha class, lower aluminium content",
        "status": "estimated",
    },

    # -------------------------------------------------------------------
    # ALUMINUM ALLOYS — the Concorde-validation lynchpin. C is lower
    # (15) for aluminium because the absolute homologous temperature
    # at 100-200 °C is already a substantial fraction of T_melt
    # (911 K), so creep onset happens earlier than the same LMP
    # would suggest for steel or nickel alloys.
    # -------------------------------------------------------------------
    "2024-T3": {
        "C": 15.0,
        "points": [
            (373,    1, 440),    # 100 °C, 1 h
            (373,  100, 250),
            (373, 1000, 145),
            (373,10000,  85),
            (423, 1000,  75),    # 150 °C, 1,000 h
            (423,10000,  45),
            (478, 1000,  25),    # 205 °C, 1,000 h
        ],
        "source": "MMPDS-17 Table 3.2.2.0 elevated-temperature stress-rupture; ASM Handbook Vol 2 Table 12",
        "status": "sourced",
    },
    "7075-T6": {
        # Higher room-T strength than 2024 but creeps similarly above
        # ~150 °C. Worse SCC behaviour at sustained load.
        "C": 15.0,
        "points": [
            (373,    1, 510),
            (373,  100, 280),
            (373, 1000, 165),
            (373,10000,  95),
            (423, 1000,  70),
            (478, 1000,  20),
        ],
        "source": "MMPDS-17 Table 3.2.4.0 elevated-temperature stress-rupture",
        "status": "sourced",
    },
    "2219-T87": {
        # Heat-resistant Al-Cu — slightly better than 2024 at sustained T
        # (Saturn V tank alloy). Closer analogue to Concorde's Al 2618
        # than 2024 is, but still in the database "estimated" tier.
        "C": 15.0,
        "points": [
            (373,    1, 410),
            (373,  100, 290),
            (373, 1000, 195),
            (373,10000, 125),
            (423, 1000, 110),
            (423,10000,  65),
            (478, 1000,  40),
        ],
        "source": "MMPDS-17 Table 3.2.7.0 elevated-temperature stress-rupture",
        "status": "sourced",
    },
    "6061-T6": {
        # Common aluminium — moderate creep resistance.
        "C": 15.0,
        "points": [
            (373, 1000, 145),
            (423, 1000,  60),
            (478, 1000,  20),
        ],
        "source": "MMPDS-17 Table 3.2.6.0 elevated-temperature stress-rupture",
        "status": "sourced",
    },

    # -------------------------------------------------------------------
    # STEELS — most aerospace steels are used below their creep regime
    # (< 750 K), but the data exists for completeness. C=20 is the
    # standard value for ferritic / martensitic steels.
    # -------------------------------------------------------------------
    "4340 Steel": {
        "C": 20.0,
        "points": [
            (700, 1000,  580),
            (755, 1000,  410),
            (811, 1000,  240),
        ],
        "source": "MMPDS-17 Table 2.3.1.0 elevated-temperature stress-rupture",
        "status": "sourced",
    },
    "300M": {
        "C": 20.0,
        "points": [
            (700, 1000,  690),
            (755, 1000,  480),
            (811, 1000,  290),
        ],
        "source": "MMPDS-17 Table 2.3.1.6",
        "status": "sourced",
    },
    "17-4PH": {
        "C": 20.0,
        "points": [
            (700, 1000,  620),
            (755, 1000,  430),
            (811, 1000,  270),
        ],
        "source": "MMPDS-17 Table 2.6.2.0; ASM Handbook Vol 1 Table 4.5",
        "status": "sourced",
    },
}


def _apply_creep_data() -> None:
    """Walk MATERIALS_DB once at module load and populate the LMP
    fields on every entry that appears in ``_CREEP_DATA``. Then sweep
    the remaining entries and apply category-level rules: TPS / CMC /
    polymer / UHTC / carbon all receive ``creep_data_status="not_applicable"``
    because they don't classically creep at relevant temperatures.

    Materials that fall through both passes keep the default
    ``creep_data_status="unknown"`` — the matching engine surfaces those
    with a flag, never auto-rejects.
    """
    by_name = {m.name: m for m in MATERIALS_DB}

    # Priority pass: explicit LMP curves.
    for name, data in _CREEP_DATA.items():
        mat = by_name.get(name)
        if mat is None:
            # Material was renamed or removed; surface a clear error
            # at module load so the data dict stays in sync with the DB.
            raise RuntimeError(
                f"_CREEP_DATA references missing material '{name}'. "
                f"Available names: {sorted(by_name.keys())[:10]}..."
            )
        C = float(data["C"])
        # Convert (T, t, sigma) tuples to (LMP, sigma) curve, sorted
        # ascending by LMP (required by the __post_init__ check).
        curve = sorted(
            (
                (T * (C + math.log10(t)), float(sigma))
                for T, t, sigma in data["points"]
            ),
            key=lambda pt: pt[0],
        )
        # Round LMP to 1 decimal place — both for cleaner output and
        # to avoid spurious sort instability from float arithmetic.
        curve = tuple((round(lmp, 1), sigma) for lmp, sigma in curve)

        object.__setattr__(mat, "larson_miller_C", C)
        object.__setattr__(mat, "lmp_curve", curve)
        object.__setattr__(mat, "creep_data_source", str(data["source"]))
        object.__setattr__(mat, "creep_data_status", str(data["status"]))

    # Category-rule pass: bulk not_applicable for non-creeping
    # categories. Materials already populated by _CREEP_DATA are
    # untouched (their status will be "sourced" or "estimated", not
    # "unknown").
    NOT_APPLICABLE_CATEGORIES = frozenset({
        "tps",                # Mass-loss ablators / single-event use
        "composite_polymer",  # Viscoelastic creep — separate model, Phase 7+
        "composite_ceramic",  # CMC creep mechanism is different from metals
        "uhtc",               # Operative below ~0.5 * T_melt
        "carbon",             # Sublimation-limited, not classical creep
    })
    for mat in MATERIALS_DB:
        if mat.creep_data_status != "unknown":
            continue
        if mat.category in NOT_APPLICABLE_CATEGORIES:
            object.__setattr__(mat, "creep_data_status", "not_applicable")
            object.__setattr__(
                mat, "creep_data_source",
                f"Category '{mat.category}' rule: material does not "
                "classically creep at relevant service temperatures.",
            )

    # General-engineering polymers (ABS, Nylon) follow the same
    # viscoelastic exception as composite_polymer. Identify by name
    # because the category covers steels too.
    GE_POLYMER_NAMES = frozenset({"ABS", "Nylon 6/6"})
    for mat in MATERIALS_DB:
        if mat.creep_data_status == "unknown" and mat.name in GE_POLYMER_NAMES:
            object.__setattr__(mat, "creep_data_status", "not_applicable")
            object.__setattr__(
                mat, "creep_data_source",
                "Polymer viscoelastic creep — separate model, not "
                "covered by Larson-Miller framework.",
            )


_apply_creep_data()


# ===========================================================================
# Specific heat capacity (Phase 7 — transient heat solver)
# ===========================================================================
#
# Single c_p value per material at room temperature (293 K). The 1D
# transient solver in ``core/transient_heat.py`` consumes ``density × c_p``
# as the volumetric heat capacity, and ``k / (ρ · c_p)`` as the thermal
# diffusivity α. Temperature-dependent c_p(T) is a future enhancement —
# at the wall-temperature range MATVEC evaluates (300 K – 1500 K) the
# room-T value is within ±15 % for most metals, which is the practical
# accuracy of stress-rupture screening.
#
# Sources
# -------
# * ASM Handbook Vol 1 (Ferrous Alloys) and Vol 2 (Nonferrous Alloys)
#   thermal-property chapters — primary reference for steels, aluminum,
#   titanium, nickel superalloys, refractory metals, cobalt alloys.
# * Special Metals data sheets (Inconel 718, 625, X-750, Waspaloy).
# * MMPDS-17 — supplementary for aluminum / titanium / steel grades.
# * Plansee Group technical data sheets (tungsten, molybdenum, niobium,
#   tantalum, rhenium).
# * Toray / Hexcel CFRP prepreg data sheets — polymer composite c_p.
# * NIST WebBook + ITER materials database — cross-check for spot
#   verification of α = k / (ρ · c_p).
# * NASA TPSX Materials Database — TPS / ablators (where applicable,
#   though most TPS materials are marked not_applicable because their
#   effective c_p includes ablation enthalpy).
# ---------------------------------------------------------------------------

_CP_DATA: dict[str, dict] = {
    # -------------------------------------------------------------------
    # ALUMINUM ALLOYS — c_p ≈ 875-960 J/(kg·K) at 293 K. Source: MMPDS-17
    # §3 thermal-property tables and ASM Handbook Vol 2 Table 18.
    # -------------------------------------------------------------------
    "2024-T3":     {"cp": 875.0, "source": "MMPDS-17 Table 3.2.2.0",  "status": "sourced"},
    "7075-T6":     {"cp": 960.0, "source": "MMPDS-17 Table 3.2.4.0",  "status": "sourced"},
    "7068-T6511":  {"cp": 880.0, "source": "Alcoa technical data sheet 7068-T6511; ASM Handbook Vol 2", "status": "sourced"},
    "6061-T6":     {"cp": 896.0, "source": "MMPDS-17 Table 3.2.6.0",  "status": "sourced"},
    "2195 Al-Li":  {"cp": 880.0, "source": "Estimated from Al-Cu-Li alloy thermal-property surveys (1-2% variation across 2xxx-Li grades)", "status": "estimated"},
    "2099 Al-Li":  {"cp": 880.0, "source": "Estimated from Al-Cu-Li alloy thermal-property surveys",                                     "status": "estimated"},
    "2219-T87":    {"cp": 864.0, "source": "MMPDS-17 Table 3.2.7.0",  "status": "sourced"},
    "7050-T7451":  {"cp": 875.0, "source": "MMPDS-17 Table 3.2.5.0",  "status": "sourced"},
    "7010-T7451":  {"cp": 875.0, "source": "Alcoa technical data sheet 7010; ASM Handbook Vol 2",                                          "status": "sourced"},
    "8090 Al-Li":  {"cp": 920.0, "source": "Estimated from Al-Cu-Li thermal-property surveys (higher Li content shifts c_p up)",          "status": "estimated"},

    # -------------------------------------------------------------------
    # TITANIUM ALLOYS — c_p ≈ 520-570 J/(kg·K). Source: RTI Titanium
    # Alloy Guide (2000) and Boyer (1994) Materials Properties Handbook.
    # -------------------------------------------------------------------
    "Ti-6Al-4V":            {"cp": 526.0, "source": "RTI Titanium Alloy Guide p. 22; Boyer (1994) p. 506", "status": "sourced"},
    "Ti-6Al-2Sn-4Zr-2Mo":   {"cp": 460.0, "source": "Boyer (1994) Materials Properties Handbook p. 583",  "status": "sourced"},
    "Ti-3Al-2.5V":          {"cp": 540.0, "source": "Boyer (1994) p. 367 (near-alpha tube alloy)",         "status": "sourced"},
    "Ti-15V-3Cr-3Al-3Sn":   {"cp": 500.0, "source": "Estimated from beta-class Ti thermal-property survey", "status": "estimated"},
    "Ti-10V-2Fe-3Al":       {"cp": 500.0, "source": "Boyer (1994) p. 743 (Ti-10-2-3 spec)",                "status": "sourced"},
    "Ti-6Al-4V ELI":        {"cp": 526.0, "source": "Estimated from Ti-6Al-4V (ELI grade is interstitial-controlled; bulk c_p essentially unchanged)", "status": "estimated"},
    "Ti-5Al-2.5Sn":         {"cp": 545.0, "source": "Boyer (1994) p. 367",                                  "status": "sourced"},
    "Ti-6Al-6V-2Sn":        {"cp": 540.0, "source": "Boyer (1994) p. 583",                                  "status": "sourced"},
    "Beta-C (Ti-3Al-8V-6Cr-4Mo-4Zr)": {"cp": 500.0, "source": "Estimated from beta-class Ti thermal-property survey", "status": "estimated"},

    # -------------------------------------------------------------------
    # NICKEL SUPERALLOYS — c_p ≈ 410-460 J/(kg·K). Source: Special
    # Metals manufacturer datasheets + ASM Specialty Handbook
    # Heat-Resistant Materials Table 5.x.
    # -------------------------------------------------------------------
    "Inconel 718":     {"cp": 435.0, "source": "Special Metals SMC-045 Inconel 718 datasheet Table 2",   "status": "sourced"},
    "Inconel 625":     {"cp": 410.0, "source": "Special Metals SMC-063 Inconel 625 datasheet Table 2",   "status": "sourced"},
    "Inconel X-750":   {"cp": 432.0, "source": "Special Metals SMC-067 Inconel X-750 datasheet Table 2", "status": "sourced"},
    "Waspaloy":        {"cp": 461.0, "source": "Special Metals SMC-011 Waspaloy datasheet Table 2",      "status": "sourced"},
    "Rene 41":         {"cp": 450.0, "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.11", "status": "sourced"},
    "Haynes 230":      {"cp": 397.0, "source": "Haynes International alloy 230 brochure H-3000H Table 2",   "status": "sourced"},
    "Haynes 282":      {"cp": 422.0, "source": "Haynes International alloy 282 brochure H-3173 Table 2",    "status": "sourced"},
    "MAR-M 247 (DS)":  {"cp": 420.0, "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.16","status": "sourced"},
    "CMSX-4":          {"cp": 400.0, "source": "Reed, *The Superalloys* (2006) Table 2.6",                  "status": "sourced"},
    "PWA 1484":        {"cp": 400.0, "source": "Estimated from CMSX-4 (similar 2nd-gen SC chemistry)",       "status": "estimated"},
    "Inconel 625 LCF": {"cp": 410.0, "source": "Estimated from Inconel 625 (LCF is wrought-form variant)",    "status": "estimated"},
    "Nimonic 90":      {"cp": 460.0, "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.14", "status": "sourced"},
    "Nimonic 105":     {"cp": 480.0, "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.14", "status": "sourced"},
    "Udimet 720":      {"cp": 440.0, "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.15", "status": "sourced"},
    "René 88DT":       {"cp": 440.0, "source": "Estimated from Udimet 720 (related P/M disc-alloy chemistry)", "status": "estimated"},
    "René 125":        {"cp": 420.0, "source": "Estimated from MAR-M-247 (related cast Ni superalloy)",        "status": "estimated"},
    "IN-100":          {"cp": 450.0, "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.12",  "status": "sourced"},
    "Astroloy":        {"cp": 440.0, "source": "ASM Specialty Handbook Heat-Resistant Materials Table 6.13",  "status": "sourced"},

    # -------------------------------------------------------------------
    # STEELS — c_p ≈ 450-500 J/(kg·K). Source: MMPDS-17 §2.
    # -------------------------------------------------------------------
    "4340 Steel":            {"cp": 477.0, "source": "MMPDS-17 Table 2.3.1.0",  "status": "sourced"},
    "300M":                  {"cp": 477.0, "source": "MMPDS-17 Table 2.3.1.6",  "status": "sourced"},
    "AF1410":                {"cp": 460.0, "source": "Estimated from related secondary-hardening steel grades",         "status": "estimated"},
    "17-4PH":                {"cp": 460.0, "source": "MMPDS-17 Table 2.6.2.0",  "status": "sourced"},
    "15-5PH":                {"cp": 460.0, "source": "Estimated from 17-4PH (same precipitation-hardening Cr-Cu class)", "status": "estimated"},
    "Maraging 350":          {"cp": 460.0, "source": "Estimated from maraging steel handbook surveys",                   "status": "estimated"},
    "HY-80":                 {"cp": 470.0, "source": "Estimated from MIL-S-16216 HY-80 thermal-property notes",          "status": "estimated"},
    "HY-100":                {"cp": 470.0, "source": "Estimated from HY-80 (same Ni-Cr-Mo class, slightly higher yield)", "status": "estimated"},
    "PH 13-8 Mo":            {"cp": 460.0, "source": "Estimated from 17-4PH (Cr-Mo PH variant)",                          "status": "estimated"},
    "AerMet 100":            {"cp": 450.0, "source": "Carpenter Technology AerMet 100 datasheet",                         "status": "sourced"},
    "Greek Ascoloy (W545)":  {"cp": 460.0, "source": "Estimated from related Cr-Ni-W stainless steel surveys",            "status": "estimated"},

    # -------------------------------------------------------------------
    # COBALT ALLOYS — c_p ≈ 380-420 J/(kg·K). Source: Haynes / Carpenter
    # technical data sheets.
    # -------------------------------------------------------------------
    "Haynes 188":     {"cp": 403.0, "source": "Haynes International alloy 188 brochure H-3001E Table 2",  "status": "sourced"},
    "L-605 (Haynes 25)": {"cp": 385.0, "source": "Haynes International alloy 25 brochure H-3019B Table 2", "status": "sourced"},
    "MP35N":          {"cp": 420.0, "source": "SPS Technologies MP35N datasheet (Carpenter Tech)",         "status": "sourced"},
    "Elgiloy":        {"cp": 430.0, "source": "Elgin National Industries Elgiloy datasheet",                "status": "sourced"},

    # -------------------------------------------------------------------
    # REFRACTORY METALS — c_p ≈ 130-280 J/(kg·K). Source: Plansee
    # technical data sheets and ITER materials database.
    # -------------------------------------------------------------------
    "Tungsten":      {"cp": 132.0, "source": "Plansee Tungsten technical brochure 2019",                    "status": "sourced"},
    "Molybdenum":    {"cp": 251.0, "source": "Plansee Molybdenum technical brochure 2019",                  "status": "sourced"},
    "Rhenium":       {"cp": 137.0, "source": "Plansee Rhenium technical brochure",                          "status": "sourced"},
    "Niobium C-103": {"cp": 268.0, "source": "ATI Allegheny Technologies C-103 datasheet (Cb-103)",         "status": "sourced"},
    "Tantalum":      {"cp": 140.0, "source": "Plansee Tantalum technical brochure 2019",                    "status": "sourced"},

    # -------------------------------------------------------------------
    # POLYMER MATRIX COMPOSITES — c_p ≈ 850-1200 J/(kg·K). Matrix-
    # dominated. Source: Hexcel / Toray prepreg datasheets.
    # -------------------------------------------------------------------
    "IM7/977-3 CFRP":  {"cp": 1050.0, "source": "Hexcel HexPly 977-3 datasheet",                            "status": "sourced"},
    "T800/3900 CFRP":  {"cp": 1100.0, "source": "Toray 3900 series prepreg datasheet",                       "status": "sourced"},
    "IM7/BMI":         {"cp": 1000.0, "source": "Estimated from bismaleimide-matrix CFRP literature",        "status": "estimated"},
    "AS4/PEEK":        {"cp": 1050.0, "source": "Estimated from Solvay APC-2 PEEK-matrix prepreg datasheet", "status": "estimated"},
    "IM7/5250-4 BMI":  {"cp": 1000.0, "source": "Cytec 5250-4 BMI prepreg datasheet",                        "status": "sourced"},
    "IM7/PETI-330":    {"cp": 1100.0, "source": "Estimated from polyimide-matrix CFRP literature",            "status": "estimated"},
    "T300/934 CFRP":   {"cp": 1100.0, "source": "Hexcel HexPly 934 datasheet (legacy 1980s prepreg)",         "status": "sourced"},

    # -------------------------------------------------------------------
    # CERAMIC MATRIX COMPOSITES — c_p ≈ 650-1100 J/(kg·K). SiC and oxide
    # ceramics; Source: ASM Engineered Materials Handbook (Ceramics).
    # -------------------------------------------------------------------
    "SiC/SiC CMC":               {"cp": 1100.0, "source": "ASM Engineered Materials Handbook Vol 4 Table 7.2",  "status": "sourced"},
    "C/SiC CMC":                 {"cp": 1000.0, "source": "ASM Engineered Materials Handbook Vol 4 Table 7.3", "status": "sourced"},
    "Oxide/Oxide CMC":           {"cp": 850.0,  "source": "Composite Horizons / COI Ceramics datasheet",        "status": "sourced"},
    "Nextel 610/alumina CMC":    {"cp": 850.0,  "source": "3M Nextel 610 + alumina-matrix datasheet",            "status": "sourced"},
    "Hi-Nicalon SiC/SiC CMC":    {"cp": 1100.0, "source": "Nippon Carbon Hi-Nicalon prepreg datasheet",          "status": "sourced"},

    # -------------------------------------------------------------------
    # UHTCs — c_p ≈ 300-600 J/(kg·K). Source: Fahrenholtz & Hilmas
    # (2012) UHTC review and OSTI 887260.
    # -------------------------------------------------------------------
    "ZrB2-SiC 20vol%":     {"cp": 540.0, "source": "Fahrenholtz & Hilmas (J. Am. Ceram. Soc. 2012) Table 1", "status": "sourced"},
    "HfB2-SiC 20vol%":     {"cp": 400.0, "source": "Fahrenholtz & Hilmas (J. Am. Ceram. Soc. 2012) Table 1", "status": "sourced"},
    "ZrB2 Monolithic":     {"cp": 460.0, "source": "Fahrenholtz & Hilmas (J. Am. Ceram. Soc. 2012)",          "status": "sourced"},
    "HfB2 Monolithic":     {"cp": 290.0, "source": "Fahrenholtz & Hilmas (J. Am. Ceram. Soc. 2012)",          "status": "sourced"},
    "Ta4HfC5":             {"cp": 220.0, "source": "Kasen et al. (OSTI 887260) UHTC review",                   "status": "sourced"},
    "ZrC":                 {"cp": 400.0, "source": "Kasen et al. (OSTI 887260) UHTC review",                   "status": "sourced"},
    "TaC":                 {"cp": 190.0, "source": "Kasen et al. (OSTI 887260) UHTC review",                   "status": "sourced"},
    "ZrB2-MoSi2":          {"cp": 510.0, "source": "Estimated from ZrB2-SiC variant data",                     "status": "estimated"},

    # -------------------------------------------------------------------
    # CARBON / GRAPHITE — c_p ≈ 700-720 J/(kg·K) at 293 K. Source:
    # NASA TPSX + Goodall (1995) for the C/C composite tile data.
    # -------------------------------------------------------------------
    "Carbon-Carbon Composite": {"cp": 710.0, "source": "NASA TPSX Materials Database; Goodall (1995)",      "status": "sourced"},
    "Pyrolytic Graphite":      {"cp": 712.0, "source": "Pocograph technical data sheet (Poco Graphite)",     "status": "sourced"},
    "Diamond CVD":             {"cp": 520.0, "source": "Element Six CVD-diamond datasheet (Pocoo / E6)",     "status": "sourced"},

    # -------------------------------------------------------------------
    # GENERAL-ENGINEERING METALS — c_p ≈ 480-510 J/(kg·K). Source: ASM
    # Handbook Vol 1.
    # -------------------------------------------------------------------
    "Structural Steel A36": {"cp": 481.0, "source": "ASM Handbook Vol 1 Table 4.1",                           "status": "sourced"},
    "Stainless Steel 304":  {"cp": 500.0, "source": "ASM Handbook Vol 1 Table 4.5 (austenitic SS)",            "status": "sourced"},
    "Mild Steel 1020":      {"cp": 480.0, "source": "ASM Handbook Vol 1 Table 4.1",                           "status": "sourced"},
    # GFRP / generic-engineering CFRP use polymer-matrix c_p ranges.
    "GFRP Generic":         {"cp": 1000.0, "source": "Estimated from generic E-glass / polyester laminate data", "status": "estimated"},
    "CFRP Generic":         {"cp": 1050.0, "source": "Estimated from epoxy-matrix CFRP averages",                "status": "estimated"},
}


def _apply_cp_data() -> None:
    """Walk MATERIALS_DB once at module load and populate the c_p
    fields on every entry that appears in ``_CP_DATA``. Then sweep
    the remaining entries and apply category-level rules: TPS
    materials get ``cp_data_status="not_applicable"`` because their
    effective thermal mass includes ablation enthalpy (modelled
    separately, not via classical conduction).

    Materials that fall through both passes keep the default
    ``cp_data_status="unknown"`` — the transient-heat solver
    surfaces those with a flag, never auto-rejects."""
    by_name = {m.name: m for m in MATERIALS_DB}

    # Priority pass: explicit per-material c_p values.
    for name, data in _CP_DATA.items():
        mat = by_name.get(name)
        if mat is None:
            raise RuntimeError(
                f"_CP_DATA references missing material '{name}'. "
                f"Available names sample: {sorted(by_name.keys())[:10]}..."
            )
        cp = float(data["cp"])
        object.__setattr__(mat, "specific_heat_J_kgK", cp)
        object.__setattr__(mat, "cp_data_source", str(data["source"]))
        object.__setattr__(mat, "cp_data_status", str(data["status"]))

    # Category-rule pass: TPS materials use mass-loss models, not
    # classical conduction. Mark not_applicable so the transient
    # solver skips them with a clear flag instead of an unknown.
    for mat in MATERIALS_DB:
        if mat.cp_data_status != "unknown":
            continue
        if mat.category == "tps":
            object.__setattr__(mat, "cp_data_status", "not_applicable")
            object.__setattr__(
                mat, "cp_data_source",
                "Category 'tps' rule: ablators use mass-loss / "
                "pyrolysis models, not classical conduction. The "
                "transient solver skips these materials.",
            )

    # General-engineering polymer category: viscoelastic / non-classical
    # creep AND a temperature-dependent c_p that differs significantly
    # from the room-T value. Mark not_applicable; surfaced as a flag
    # in the transient solver. (The remaining general_engineering
    # entries — A36 / SS304 / Mild 1020 — were populated above.)
    GE_POLYMER_NAMES = frozenset({"ABS Plastic", "Nylon 66"})
    for mat in MATERIALS_DB:
        if mat.cp_data_status == "unknown" and mat.name in GE_POLYMER_NAMES:
            object.__setattr__(mat, "cp_data_status", "not_applicable")
            object.__setattr__(
                mat, "cp_data_source",
                "Polymer engineering plastic — viscoelastic / "
                "glass-transition-dominated behaviour; classical "
                "thermal-diffusivity solver is not appropriate.",
            )


_apply_cp_data()


def get_availability_score(material_name: str) -> float:
    """Return the availability score for a material by name.

    Returns the override value if one exists, otherwise 1.0 (commercially available).
    """
    return _AVAILABILITY_OVERRIDES.get(material_name, 1.0)


# ===========================================================================
# Public helper functions
# ===========================================================================

def get_materials_by_category(category: str) -> list:
    """Return all materials whose category matches exactly."""
    return [m for m in MATERIALS_DB if m.category == category]


def get_materials_by_regime(regime: str) -> list:
    """Return all materials applicable to the given flight regime."""
    return [m for m in MATERIALS_DB if regime in m.applicable_regimes]


def get_strength_at_temperature(material: MaterialEntry, T_K: float) -> float:
    """
    Return tensile strength at temperature T_K (Kelvin) via linear interpolation.

    For brittle ceramics the stored values are flexural strength (MOR); this
    function returns whatever is stored, so callers should be aware of the proxy.

    Clamps to bounds: temperatures below the minimum data point return the
    room-temperature strength; temperatures above the maximum return the
    last known value (conservative — does not extrapolate to zero).
    """
    data = material.tensile_strength_at_temp
    temps = sorted(data.keys())

    if T_K <= temps[0]:
        return data[temps[0]]
    if T_K >= temps[-1]:
        return data[temps[-1]]

    for i in range(len(temps) - 1):
        T_lo = temps[i]
        T_hi = temps[i + 1]
        if T_lo <= T_K <= T_hi:
            S_lo = data[T_lo]
            S_hi = data[T_hi]
            fraction = (T_K - T_lo) / (T_hi - T_lo)
            return S_lo + fraction * (S_hi - S_lo)

    # Unreachable if data is consistent, but guard against floating-point edge cases
    raise RuntimeError(
        f"Interpolation failed for '{material.name}' at T={T_K} K"
    )
