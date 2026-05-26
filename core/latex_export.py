"""
MATVEC LaTeX PDF Export Module
================================
Step 5: Generates a professional 13-section technical report as LaTeX/PDF.
"""

import math
import datetime
import subprocess
import tempfile
import os

from .physics_engine import (
    __version__ as _MATVEC_VERSION,
    C_SUTTON_GRAVES,
    SIGMA_SB,
    G0,
    R_AIR,
    RECOVERY_FACTOR,
    GAMMA_AIR,
    STRUCTURAL_SAFETY_FACTOR,
)
from .matching_engine import (
    MatchResult,
    TPS_UNLOCK_TEMP_K,
    _ABLATIVE_SUBSTRATE_T_FLOOR_K,
)
from .materials_db import MATERIALS_DB
from .pareto import compute_pareto, generate_pareto_chart_b64
from .surrogate import find_nearest_candidates, get_model_version


# ---------------------------------------------------------------------------
# Category labels (duplicated from app.py to avoid circular import)
# ---------------------------------------------------------------------------

_CATEGORY_LABELS = {
    "aluminum":            "Aluminum",
    "titanium":            "Titanium",
    "steel":               "Steel",
    "nickel":              "Nickel SA",
    "cobalt":              "Cobalt SA",
    "refractory":          "Refractory",
    "composite_polymer":   "CFRP/PMC",
    "composite_ceramic":   "CMC",
    "uhtc":                "UHTC",
    "tps":                 "TPS",
    "carbon":              "Carbon",
    "general_engineering": "Gen. Eng.",
}

_VEHICLE_CATEGORY_LABELS = {
    "general":             "General Structure",
    "aircraft":            "Aircraft / Airframe",
    "hypersonic_aircraft": "Hypersonic Aircraft / Spaceplane",
    "reentry":             "Reentry Vehicle",
    "hypersonic_missile":  "High-Speed Missile",
    "turbine":             "Turbine Component",
}


# ---------------------------------------------------------------------------
# Per-vehicle reference fuel (Item 5 of the six-improvement plan)
# ---------------------------------------------------------------------------
# Hoists the previously-inline fuel labels out of _sec_propulsion so each
# category's reference fuel is a single readable line, easy to override or
# extend. SR-71 used JP-7 (not Jet-A); tactical missiles use JP-10; X-15-class
# hypersonic aircraft are rocket-propelled (NH3 + LOX). Reentry has no fuel
# (handled by an early return in _sec_propulsion). Turbine omits the table
# (also early return) since blade selection is driven by hot-section temp,
# not by a vehicle-level fuel load.

class _FuelRef:
    """A reference fuel entry for the energy-equivalent mass table."""
    __slots__ = ("label", "energy_density_J_per_kg", "energy_density_tex",
                 "mass_attr", "use_kinetic_energy_basis")

    def __init__(self, label, energy_density_J_per_kg, energy_density_tex,
                 mass_attr=None, use_kinetic_energy_basis=False):
        self.label = label
        self.energy_density_J_per_kg = energy_density_J_per_kg
        self.energy_density_tex = energy_density_tex
        # mass_attr: name of attribute on PropulsionResults to read mass from.
        # If None, mass is computed inline from KE_J / energy_density (rocket basis).
        self.mass_attr = mass_attr
        self.use_kinetic_energy_basis = use_kinetic_energy_basis


_PRIMARY_FUEL_BY_CATEGORY = {
    "aircraft":            _FuelRef("Kerosene / JP-7 (ref.)",
                                    4.32e7, r"$4.32 \times 10^{7}$",
                                    mass_attr="fuel_mass_kerosene_kg"),
    "hypersonic_aircraft": _FuelRef("Ammonia + LOX (rocket ref., KE basis)",
                                    8.0e6, r"$8.0 \times 10^{6}$",
                                    use_kinetic_energy_basis=True),
    "hypersonic_missile":  _FuelRef("Kerosene / JP-10 (ref.)",
                                    4.32e7, r"$4.32 \times 10^{7}$",
                                    mass_attr="fuel_mass_kerosene_kg"),
    "general":             _FuelRef("Kerosene / JP-7 (ref.)",
                                    4.32e7, r"$4.32 \times 10^{7}$",
                                    mass_attr="fuel_mass_kerosene_kg"),
    # "reentry" and "turbine" intentionally absent — those branches early-return
    # in _sec_propulsion with a "no fuel" / "engine-internal" explanation.
}


def _resolve_primary_fuel(vehicle_category, peak_mach):
    """Return the _FuelRef appropriate for this vehicle.

    Backward-compatibility shim: an aircraft preset at Mach >= 5 (X-15-class)
    is rocket-propelled in practice — air-breathing turbines aren't viable —
    so it is re-routed to the hypersonic_aircraft fuel entry. This preserves
    the previous behavior while making explicit hypersonic_aircraft callers
    use the same entry without needing to also bump Mach.
    """
    if vehicle_category == "aircraft" and peak_mach >= 5.0:
        return _PRIMARY_FUEL_BY_CATEGORY["hypersonic_aircraft"]
    return _PRIMARY_FUEL_BY_CATEGORY.get(
        vehicle_category, _PRIMARY_FUEL_BY_CATEGORY["general"]
    )


def _cat_label(cat: str) -> str:
    return _CATEGORY_LABELS.get(cat, cat)


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _tex_escape(s: str) -> str:
    """Escape LaTeX special characters in an arbitrary string."""
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]:
        s = s.replace(old, new)
    return s


def _fmt_sci(val: float, sig: int = 4) -> str:
    """Format as LaTeX \\times 10^{n} scientific notation."""
    if val == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    mantissa = val / (10 ** exp)
    decimals = max(0, sig - 1)
    m_str = f"{mantissa:.{decimals}f}"
    return rf"{m_str} \times 10^{{{exp}}}"


def _fmt(val: float, fmt_spec: str = ".4g") -> str:
    """Format a float for LaTeX, converting Python e-notation to \\times 10^{}."""
    s = format(val, fmt_spec)
    if "e" in s:
        m, e = s.split("e")
        return rf"{m} \times 10^{{{int(e)}}}"
    return s


def _format_cost_usd(usd: float) -> str:
    """Render a USD figure as a compact, scannable string for tables.

    Returns ``"---"`` for zero cost (the sentinel used for exotic/2D entries
    that must never display a price in user-facing output --- see the
    ``cost_usd_per_kg`` docstring on ``MaterialEntry``). Otherwise picks a
    SI-style suffix so 200-row tables stay readable: $42 / $1.2k / $34k /
    $1.2M / $24M. Sub-dollar costs round to ``<$1`` rather than printing
    ``$0`` (which would be confused with the exotic-sentinel display).
    """
    if usd is None or usd <= 0.0:
        return "---"
    if usd < 1.0:
        return r"\textless\$1"
    if usd < 1e3:
        return f"\\${usd:.0f}"
    if usd < 1e6:
        return f"\\${usd / 1e3:.1f}k"
    if usd < 1e9:
        return f"\\${usd / 1e6:.1f}M"
    return f"\\${usd / 1e9:.1f}B"


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------

def _verdict_str(match: MatchResult) -> str:
    if match.impossible:
        return "BEYOND KNOWN MATERIALS"
    if match.no_material_viable:
        return "NO SINGLE MATERIAL SATISFIES ALL REQUIREMENTS"
    return "FEASIBLE"


def _verdict_color(match: MatchResult) -> str:
    if match.impossible:
        return "orange!40"
    if match.no_material_viable:
        return "red!25"
    return "green!25"


# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------

_CITE_KEYWORDS = [
    ("Fahrenholtz",       "fahrenholtz2017"),
    ("Cedillos",          "cedillos2016"),
    ("NASA TPSX",         "nasatpsx"),
    ("NTRS",              "nasatpsx"),
    ("Plansee",           "plansee"),
    ("RTI International", "rti2000"),
    ("Special Metals",    "specialmetals"),
    ("Haynes",            "haynes"),
    ("MMPDS",             "milhdbk5"),
    ("MIL-HDBK",          "milhdbk5"),
    ("ASM Engineered",    "asmhandbook"),
    ("ASM Handbook",      "asmhandbook"),
    ("ASTM",              "asmhandbook"),
    ("Matweb",            "matweb"),
]

_ALL_BIBITEMS = {
    "sutton1971":
        r"\bibitem{sutton1971} Sutton, K.\ \& Graves, R.A.\ (1971). "
        r"\textit{A General Stagnation-Point Convective Heating Equation for Arbitrary Gas Mixtures}. "
        r"NASA TR-R-376.",
    "tauber1989":
        r"\bibitem{tauber1989} Tauber, M.E.\ \& Sutton, K.\ (1991). "
        r"Stagnation-point radiative heating relations for Earth and Mars entries. "
        r"\textit{Journal of Spacecraft and Rockets}, 28(1), 40--42.",
    "icao7488":
        r"\bibitem{icao7488} International Civil Aviation Organization (1993). "
        r"\textit{Manual of the ICAO Standard Atmosphere}, 3rd ed. ICAO Doc 7488.",
    "milhdbk5":
        r"\bibitem{milhdbk5} U.S.\ Department of Defense (2003). "
        r"\textit{Metallic Materials Properties Development and Standardization (MMPDS-17)}, "
        r"formerly MIL-HDBK-5.",
    "asmhandbook":
        r"\bibitem{asmhandbook} ASM International (1990--2022). "
        r"\textit{ASM Handbook}, Vols.\ 1 \& 2. Materials Park, OH.",
    "cedillos2016":
        r"\bibitem{cedillos2016} Cedillos-Barraza, O.\ et al.\ (2016). "
        r"Investigating the highest melting temperature materials: A laser melting study of the TaC--HfC system. "
        r"\textit{Scientific Reports}, 6, 37962.",
    "fahrenholtz2017":
        r"\bibitem{fahrenholtz2017} Fahrenholtz, W.G.\ \& Hilmas, G.E.\ (2017). "
        r"Oxidation of ultra-high temperature transition metal diboride ceramics. "
        r"\textit{International Materials Reviews}, 57(1), 61--72.",
    "nasatpsx":
        r"\bibitem{nasatpsx} NASA Ames Research Center. "
        r"\textit{TPSX Materials Properties Database}. Available: tpsx.arc.nasa.gov.",
    "rti2000":
        r"\bibitem{rti2000} RTI International (2000). "
        r"\textit{Titanium: A Technical Guide}, 2nd ed. ASM International.",
    "specialmetals":
        r"\bibitem{specialmetals} Special Metals Corporation. "
        r"\textit{Inconel Alloy Technical Bulletins}. Available: specialmetals.com.",
    "haynes":
        r"\bibitem{haynes} Haynes International. "
        r"\textit{High-Temperature Alloy Technical Data Sheets}. Available: haynesintl.com.",
    "plansee":
        r"\bibitem{plansee} Plansee SE. "
        r"\textit{Refractory Metals Technical Data}. Available: plansee.com.",
    "matweb":
        r"\bibitem{matweb} MatWeb LLC. "
        r"\textit{Material Property Data}. Available: matweb.com.",
}

_ALWAYS_CITE = {"sutton1971", "tauber1989", "icao7488", "milhdbk5",
                "asmhandbook", "cedillos2016", "fahrenholtz2017"}


def _cite_key(citation: str) -> str:
    for keyword, key in _CITE_KEYWORDS:
        if keyword.lower() in citation.lower():
            return key
    return "asmhandbook"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _preamble(label_esc: str) -> str:
    return "\n".join([
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{physics}",
        r"\usepackage{siunitx}",
        r"\usepackage{booktabs}",
        r"\usepackage[a4paper,margin=2.5cm]{geometry}",
        r"\usepackage[hidelinks,colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue]{hyperref}",
        r"\usepackage[table,dvipsnames]{xcolor}",
        r"\usepackage{graphicx}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{fancyhdr}",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        f"\\rhead{{\\small MATVEC v{_MATVEC_VERSION}}}",
        f"\\lhead{{\\small {label_esc}}}",
        r"\cfoot{\thepage}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.5em}",
        r"\begin{document}",
    ])


def _title_block(label_esc: str, ts: str) -> str:
    return "\n".join([
        r"\begin{center}",
        r"{\Huge\bfseries MATVEC Materials Feasibility Analysis}\\[0.4em]",
        f"{{\\Large {label_esc}}}\\\\[0.3em]",
        f"\\texttt{{{_tex_escape(ts)}}}\\\\[0.2em]",
        f"{{\\small Version {_tex_escape(_MATVEC_VERSION)}}}\\\\[0.8em]",
        r"\hrule\\[0.5em]",
        r"{\small Automated aerospace materials feasibility analysis based on flight envelope inputs.}",
        r"\end{center}",
        r"\vspace{0.5em}",
        r"\tableofcontents",
        r"\newpage",
    ])


def _sec_executive_summary(physics, match: MatchResult) -> str:
    th = physics.thermal
    v = _verdict_str(match)

    top_str = ""
    top_candidates = list(match.viable) or list(match.marginal)
    if top_candidates:
        top = top_candidates[0]
        label = "viable" if match.viable else "marginal"
        top_str = (
            f" The highest-scoring {label} material is "
            f"\\textbf{{{_tex_escape(top.material.name)}}} "
            f"(min. margin: {top.score:.3f})."
        )
    elif match.not_viable:
        top = match.not_viable[0]
        top_str = (
            f" The nearest candidate is "
            f"\\textbf{{{_tex_escape(top.material.name)}}}, "
            f"which misses the thermal requirement by {abs(top.thermal_margin_K):.0f}\\,K."
        )

    diag_str = ""
    if match.diagnosis:
        diag_str = " " + _tex_escape(match.diagnosis)

    # Ablative-coating advisory: physics-driven, surfaces whenever T_wall
    # exceeds the thermal ceiling of standard structural metals (~1200 K).
    ablative_str = ""
    if th.T_wall_K >= TPS_UNLOCK_TEMP_K:
        T_soak = max(th.T_ambient_K, 400.0)
        ablative_str = (
            r" \textbf{Ablative coating recommended:} external thermal load "
            f"({th.T_wall_K:.0f}\\,K) exceeds the ceiling of monolithic metallic "
            "airframes; recommendations include an ablative hot-face layer "
            f"over a metallic substructure evaluated at $T_{{\\text{{soak}}}}"
            f" \\approx {T_soak:.0f}$\\,K. Substrate-mode candidates are "
            r"tagged \textit{(substrate)} in the materials table."
        )

    para = (
        f"\\textbf{{Verdict: {_tex_escape(v)}.}} "
        f"Flight regime: \\textit{{{_tex_escape(physics.flight_regime)}}}. "
        f"Peak wall temperature: {th.T_wall_K:.0f}\\,K "
        f"(uncertainty range {th.T_wall_min_K:.0f}--{th.T_wall_max_K:.0f}\\,K, "
        f"$\\pm{'15' if th.uses_recovery_model else '20'}\\%$)."
        + top_str + diag_str + ablative_str
    )

    return "\n".join([
        r"\section{Executive Summary}",
        para,
    ])


def _sec_input_parameters(
    physics,
    vehicle_category: str = "general",
    *,
    design_lifetime_hours: float | None = None,
    panel_thickness_m: float | None = None,
) -> str:
    th = physics.thermal
    vel_kmh = th.velocity_ms * 3.6

    default_eps = abs(physics.wall_emissivity - 0.85) < 1e-9
    eps_str = f"{physics.wall_emissivity:.2f}" + (r" \textit{(default)}" if default_eps else "")

    cat_label = _VEHICLE_CATEGORY_LABELS.get(vehicle_category, vehicle_category)

    rows = [
        ("Vehicle Category",            _tex_escape(cat_label)),
        ("Peak Mach",                   f"{physics.peak_mach:.3f}"),
        ("Velocity",                    f"{vel_kmh:,.0f}\\,km/h \\;({th.velocity_ms:,.1f}\\,m/s)"),
        ("Cruise Altitude",             f"{physics.cruise_altitude_km:.1f}\\,km"),
        ("Vehicle Mass",                f"{physics.vehicle_mass_kg:,.0f}\\,kg"),
        ("Leading Edge Radius",          f"{physics.nose_radius_m:.3f}\\,m"),
        ("Peak G-load",                 f"{physics.peak_g_load:.1f}\\,g"),
        ("Characteristic Length",       f"{physics.characteristic_length_m:.1f}\\,m"),
        ("Flight Duration",             f"{physics.flight_duration_s:.0f}\\,s"),
    ]
    # Phase-7 schema additions: surface design_lifetime_hours and
    # panel_thickness_m so a reader can verify what values drove the
    # creep + transient screening sections. Both are session-level
    # fields (not on PhysicsResult); they reach this function via the
    # ``generate_tex_source`` kwargs and are optional so older
    # callers that don't pass them keep working.
    if design_lifetime_hours is not None:
        rows.append(
            ("Design Lifetime", f"{float(design_lifetime_hours):,.0f}\\,h")
        )
    if panel_thickness_m is not None:
        rows.append(
            ("Panel Thickness", f"{float(panel_thickness_m) * 1000.0:.1f}\\,mm")
        )
    rows.append(("Wall Emissivity $\\varepsilon$", eps_str))

    # Parameter names are author-controlled LaTeX strings (may contain math like $\varepsilon$).
    # Only the value column is user-derived and requires escaping.
    table_rows = "\n".join(f"  {param} & {val} \\\\" for param, val in rows)

    return "\n".join([
        r"\section{Input Parameters}",
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
        r"Parameter & Value \\",
        r"\midrule",
        table_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def _sec_atmospheric(physics) -> str:
    atm = physics.atmosphere

    rows = [
        ("Altitude",                          f"{atm.altitude_km:.1f}\\,km"),
        ("Ambient Temperature $T_\\infty$",   f"{atm.temperature_K:.2f}\\,K"),
        ("Static Pressure $p$",               f"{atm.pressure_Pa:,.1f}\\,Pa"),
        ("Density $\\rho$",                   f"{atm.density_kgm3:.5f}\\,kg/m$^3$"),
    ]

    table_rows = "\n".join(f"  {param} & {val} \\\\" for param, val in rows)

    return "\n".join([
        r"\section{Atmospheric Conditions}",
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        table_rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"Atmospheric properties computed using the International Standard Atmosphere "
        r"(ISA) model, ICAO Doc 7488 \cite{icao7488}, valid 0--86\,km.",
    ])


def _t_stag_row(th):
    """Return the (label, value) tuple for the T_stag row in the thermal table.

    When the calorically perfect value is physically invalid (dissociation
    regime), flag it inline so the number is not read as a credible wall
    temperature. The follow-up caveat block renders the full explanation.
    """
    if th.T_stag_real_gas_suspect:
        label = (r"$T_{\text{stag}}$ \textit{(CPG, real-gas invalid --- see below)}")
    else:
        label = r"$T_{\text{stag}}$ (total) \textit{--- context only}"
    return (label, f"{th.T_stag_K:.1f}\\,K")


def _plasma_sheath_display(th) -> str:
    """Return the LaTeX snippet for the plasma-sheath row.

    The physics engine now applies a two-tier threshold: blunt bodies
    (nose radius >= 0.5 m) use the classic Mach > 10 threshold, while
    slender bodies (nose radius < 0.5 m) trip the plasma flag at Mach > 6
    to capture the Mach 6-8 attenuation effects documented on the X-15
    (radio blackout and UV glow at M ~ 6.7). The display string must match
    whichever branch actually fired so that engineers do not see "Yes"
    against a threshold line that contradicts their input Mach.
    """
    if not th.plasma_sheath:
        return "No"
    if getattr(th, "plasma_threshold_slender", False):
        return r"Yes ($M > 6.0$, slender body, alt $< 80$\,km)"
    return r"Yes ($M > 10.0$, alt $< 80$\,km)"


def _sec_thermal(physics) -> str:
    th = physics.thermal
    atm = physics.atmosphere

    lines = [r"\section{Thermal Analysis}\label{sec:thermal}"]

    # --- Turbine hot-section override -----------------------------------
    # For turbine components the wall temperature is driven by turbine
    # inlet temperature (TIT) and film-cooling physics, not by aerodynamic
    # heating. Recovery temperature at Mach 0.5 / sea level is ~300 K,
    # which is meaningless for HPT blade analysis. When the caller has
    # injected a hot-section temperature (thermal_source ==
    # "turbine_inlet_override") we replace the Sutton-Graves / recovery
    # derivation entirely with a short framing paragraph and a compact
    # results table that omits q_conv, q_rad, epsilon (those are computed
    # from the aerodynamic stub for back-compat but carry no information
    # in this branch).
    thermal_source = getattr(th, "thermal_source", "aerodynamic")
    if thermal_source == "turbine_inlet_override":
        lines += [
            r"\textbf{Hot-section override active.} "
            r"Wall temperature is set by the turbine inlet / hot-section environment "
            r"rather than external aerodynamic heating. "
            f"$T_{{\\text{{wall}}}} = {th.T_wall_K:.0f}$\\,K represents a modern cooled "
            r"HPT blade metal-face temperature (turbine inlet $\approx "
            f"{th.T_wall_K + 300.0:.0f}$\\,K with a $\\sim 300$\\,K film-cooling delta). "
            r"The Sutton-Graves and recovery-temperature derivations are "
            r"omitted from this branch --- they do not apply to a component "
            r"embedded inside an engine.",
            "",
            r"\subsection{Thermal Results Summary}",
            r"\begin{tabular}{@{}ll@{}}",
            r"\toprule",
            r"Quantity & Value \\",
            r"\midrule",
        ]
        rows = [
            (r"$T_{\text{wall}}$ nominal (hot-section override)",
             f"{th.T_wall_K:.1f}\\,K"),
            (r"$T_{\text{wall}}$ minimum ($-5\%$)",
             f"{th.T_wall_min_K:.1f}\\,K"),
            (r"$T_{\text{wall}}$ maximum ($+5\%$)",
             f"{th.T_wall_max_K:.1f}\\,K"),
            ("Thermal source", _tex_escape(thermal_source)),
        ]
        lines += [f"  {param} & {val} \\\\" for param, val in rows]
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    if th.uses_recovery_model:
        # --- Recovery Temperature Model (M < 5) ---
        # Item 3 of the six-improvement plan: positive, model-selection framing
        # — say what the tool uses, not what it is "not applicable" for.
        lines += [
            r"\textbf{Model scope.} "
            r"MATVEC selects the thermal model by Mach number: adiabatic wall (recovery) "
            r"temperature for sustained flight at $M < 5$, and Sutton-Graves convective "
            r"heating with the Tauber-Sutton radiative addition for $M \geq 5$. "
            r"This envelope is in the recovery regime --- $T_{\text{wall}}$ is computed "
            r"directly from the recovery formula with a turbulent recovery factor of "
            r"$r = 0.85$. Each correlation is applied within its calibrated range. "
            r"$T_{\text{wall}}$ represents the stagnation-point temperature at the nose tip / "
            r"leading edge; bulk fuselage temperatures are substantially lower "
            r"(Section~\ref{sec:component_zones} reports per-zone $T$ and $\sigma$ "
            r"with locally scaled multipliers). "
            r"Ablative coatings, thermal barrier coatings, and active cooling are "
            r"\textbf{not} modelled --- they would relax the constraint.",
            r"\subsection{Aerodynamic Heating --- Recovery Temperature Model}",
            r"\begin{align}",
            r"T_{\text{wall}} &= T_{\text{amb}}\!\left(1 + r\,\frac{\gamma-1}{2}\,M^2\right),"
            r"\quad r = 0.85 \text{ (turbulent)} \\",
            (f"&= {atm.temperature_K:.2f}"
             f"\\!\\left(1 + 0.85 \\times \\frac{{0.4}}{{2}} \\times {physics.peak_mach:.3f}^2\\right) \\\\"),
            f"&= {th.T_wall_K:.1f}\\;\\text{{K}}",
            r"\end{align}",
            r"The recovery temperature model is the appropriate thermal framework for sustained "
            r"flight at Mach~$<$~5, where aerodynamic heating flux is below the threshold where "
            r"radiation equilibrium becomes the limiting constraint. "
            r"Uncertainty bounds: $\pm 15\%$ on $T_{\text{wall}}$.",
        ]

        plasma_str = _plasma_sheath_display(th)

        rows = [
            (r"$T_{\text{wall}}$ nominal (recovery temperature)",
             f"{th.T_wall_K:.1f}\\,K"),
            (r"$T_{\text{wall}}$ minimum ($-15\%$)",
             f"{th.T_wall_min_K:.1f}\\,K"),
            (r"$T_{\text{wall}}$ maximum ($+15\%$)",
             f"{th.T_wall_max_K:.1f}\\,K"),
            _t_stag_row(th),
            (r"$T_{\text{wall}}$, sea-level worst case",
             f"{th.T_wall_sealevel_K:.1f}\\,K"),
            ("Plasma sheath", plasma_str),
        ]

    else:
        # --- Sutton-Graves + Tauber-Sutton Model (M >= 5) ---
        lines += [
            r"\textbf{Model scope and limitations.} "
            r"The heating calculation below uses the Sutton-Graves convective correlation, "
            r"which was calibrated for \emph{blunt-body stagnation-point} heating on "
            r"reentry capsules (Mach~$\geq$~5). "
            r"The result, $T_{\text{wall}}$, is therefore the radiation-equilibrium temperature "
            r"\emph{at the stagnation point} (nose tip / leading edge) only --- "
            r"it is \textbf{not} the bulk fuselage or wing-panel temperature. "
            r"For winged vehicles and lifting bodies, fuselage bulk temperatures are "
            r"substantially lower than the stagnation value; structural material selection "
            r"for panels away from the nose should use vehicle-specific aeroheating analysis. "
            r"Section~\ref{sec:component_zones} reports per-zone $T$ and $\sigma$ values "
            r"using category-level multipliers as a first-order approximation. "
            r"Ablative coatings, thermal barrier coatings, and active cooling are \textbf{not} modelled.",
        ]

        # 1. Sutton-Graves
        lines += [
            r"\subsection{Aerodynamic Heating --- Stagnation-Point Heating (Sutton-Graves, Mach~$\geq$~5)}",
            r"\begin{align}",
            r"q_{\text{conv}} &= C \sqrt{\frac{\rho}{R_n}} V^3 \\",
            (f"&= 1.7415 \\times 10^{{-4}}"
             f"\\sqrt{{\\frac{{{atm.density_kgm3:.5f}}}{{{physics.nose_radius_m:.3f}}}}}"
             f" \\times {th.velocity_ms:.1f}^3 \\\\"),
            f"&= {_fmt_sci(th.q_conv_Wm2, 4)}\\;\\text{{W/m}}^2",
            r"\end{align}",
            (f"where $C = 1.7415 \\times 10^{{-4}}$, "
             f"$\\rho = {atm.density_kgm3:.5f}$\\,kg/m$^3$, "
             f"$R_n = {physics.nose_radius_m:.3f}$\\,m, "
             f"$V = {th.velocity_ms:.1f}$\\,m/s. "
             r"Source: Sutton \& Graves (1971) \cite{sutton1971}."),
        ]

        # 2. Radiative heating
        lines.append(r"\subsection{Radiative Heating}")
        if th.q_rad_Wm2 > 0:
            lines += [
                r"Radiative heating is significant at this velocity. "
                r"Tauber-Sutton correlation \cite{tauber1989}:",
                r"\begin{align}",
                r"q_{\text{rad}} &= C_{\text{rad}}\, \rho^{1.22}\, R_n^{0.5}\, V^{3.5} \\",
                f"&= {_fmt_sci(th.q_rad_Wm2, 4)}\\;\\text{{W/m}}^2",
                r"\end{align}",
            ]
        else:
            lines.append(
                f"Radiative heating negligible at "
                f"$V = {th.velocity_ms:.0f}$\\,m/s $< 6000$\\,m/s."
            )

        # 3. Radiation-equilibrium wall temperature
        lines += [
            r"\subsection{Radiation-Equilibrium Temperature at Stagnation Point}",
            r"\begin{align}",
            r"T_{\text{wall}} &= \left(\frac{q_{\text{total}}}{\varepsilon\,\sigma_{SB}}\right)^{1/4} \\",
            (f"&= \\left(\\frac{{{_fmt_sci(th.q_total_Wm2, 4)}}}"
             f"{{{physics.wall_emissivity:.2f} \\times 5.670 \\times 10^{{-8}}}}\\right)^{{1/4}} \\\\"),
            f"&= {th.T_wall_K:.1f}\\;\\text{{K}}",
            r"\end{align}",
        ]

        plasma_str = _plasma_sheath_display(th)

        rows = [
            (r"$T_{\text{wall}}$ stagnation pt.\ nominal",
             f"{th.T_wall_K:.1f}\\,K"),
            (r"$T_{\text{wall}}$ stagnation pt.\ minimum ($-20\%$)",
             f"{th.T_wall_min_K:.1f}\\,K"),
            (r"$T_{\text{wall}}$ stagnation pt.\ maximum ($+20\%$)",
             f"{th.T_wall_max_K:.1f}\\,K"),
            _t_stag_row(th),
            (r"$q_{\text{conv}}$",
             f"{_fmt_sci(th.q_conv_Wm2, 3)}\\,W/m$^2$"),
            (r"$q_{\text{rad}}$",
             f"{_fmt_sci(th.q_rad_Wm2, 3)}\\,W/m$^2$"),
            (r"$q_{\text{total}}$",
             f"{_fmt_sci(th.q_total_Wm2, 3)}\\,W/m$^2$"),
            (r"$T_{\text{wall}}$, sea-level worst case",
             f"{th.T_wall_sealevel_K:.1f}\\,K"),
            ("Plasma sheath", plasma_str),
        ]

    table_rows = "\n".join(f"  {param} & {val} \\\\" for param, val in rows)

    lines += [
        r"\subsection{Thermal Results Summary}",
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        table_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]

    if th.T_stag_real_gas_suspect:
        lines += [
            "",
            r"\textbf{Real-gas caveat.} "
            r"The stagnation temperature above is computed from the calorically "
            r"perfect gas relation "
            r"$T_{\text{stag}} = T_{\text{amb}}(1 + \tfrac{\gamma-1}{2} M^2)$ "
            r"with $\gamma = 1.4$. "
            r"Above $\sim 3000$\,K, O$_2$ dissociates, N$_2$ vibrational modes unfreeze, "
            r"and ionization begins. These endothermic processes absorb energy, so the "
            r"\emph{true} stagnation temperature is substantially lower (typically 2--5$\times$ "
            r"smaller at $M \geq 15$). The CPG value is reported for completeness but "
            r"\emph{is not used as a material requirement} --- the radiation-equilibrium "
            r"wall temperature $T_{\text{wall}}$ (which incorporates the actual heat flux) "
            r"remains the feasibility driver.",
        ]

    return "\n".join(lines)


def _fmt_mpa_sci(v: float) -> str:
    """Format MPa value; use scientific notation when v < 0.1 to avoid '0.00'."""
    if v < 0.1:
        mantissa, exp = f"{v:.2e}".split("e")
        return f"{mantissa} \\times 10^{{{int(exp)}}}\\,\\text{{MPa}}"
    return f"{v:.2f}\\,\\text{{MPa}}"


def _sec_structural(physics, vehicle_category: str = "general") -> str:
    s = physics.structural
    q_dyn_MPa = s.q_dyn_Pa / 1e6

    # --- TPS-protected variant --------------------------------------------
    # For vehicles that could plausibly use a thermal protection system
    # (ablator, TBC, active cooling) the "binding" structural requirement
    # on the substructure is *not* σ_req at T_wall — it is σ_req at the
    # soak-through temperature below the hot face. We present both values
    # so the engineer sees the correct number whether their design is
    # monolithic or TPS-layered; §8 carries the candidate list under the
    # substrate mode assumption.
    show_tps_variant = (
        physics.thermal.T_wall_K >= TPS_UNLOCK_TEMP_K
        or vehicle_category in ("reentry", "hypersonic_aircraft")
    )
    T_ambient = physics.thermal.T_ambient_K
    T_soak_floor = max(T_ambient, _ABLATIVE_SUBSTRATE_T_FLOOR_K)
    delta_T_tps = max(0.0, T_soak_floor - T_ambient)
    sigma_th_tps = 0.4 * 200_000.0 * 12e-6 * delta_T_tps   # MPa, same constants as §3
    sigma_req_tps = s.sigma_combined_MPa + sigma_th_tps

    lines = [
        r"\section{Structural Analysis}",
        r"\subsection{Load Components}",
        "",
        r"\textbf{1.\ Inertial Load:}",
        r"\begin{align}",
        (f"F_{{\\text{{inertial}}}} &= m \\times G \\times g_0 "
         f"= {physics.vehicle_mass_kg:,.0f} \\times {physics.peak_g_load:.1f} \\times 9.807 "
         f"= {s.F_inertial_N:,.1f}\\;\\text{{N}} \\\\"),
        (f"\\sigma_{{\\text{{inertial}}}} &= F_{{\\text{{inertial}}}} / A_{{\\text{{ref}}}} "
         f"= {s.F_inertial_N:,.1f} / {s.A_ref_m2:.4f} "
         f"= {_fmt_mpa_sci(s.sigma_inertial_MPa)}"),
        r"\end{align}",
        "",
        r"\textbf{2.\ Dynamic Pressure:}",
        r"\begin{align}",
        (f"q_{{\\text{{dyn}}}} &= \\tfrac{{1}}{{2}}\\rho V^2 "
         f"= \\tfrac{{1}}{{2}} \\times {physics.atmosphere.density_kgm3:.5f} "
         f"\\times {physics.thermal.velocity_ms:.1f}^2 "
         f"= {s.q_dyn_Pa:,.0f}\\;\\text{{Pa}}"),
        r"\end{align}",
        f"Dynamic pressure contribution (before safety factor): "
        f"$q_{{\\text{{dyn}}}} / 10^6 = {q_dyn_MPa:.4f}$\\,MPa.",
        "",
        (r"\textbf{3.\ Reference Thermal Stress} "
         r"($E_{\text{ref}} = 200$\,GPa, $\alpha_{\text{ref}} = 12 \times 10^{-6}$\,K$^{-1}$, "
         r"relief factor $k_{\text{relief}} = 0.4$):"),
        r"\begin{align}",
        (f"\\sigma_{{\\text{{th,ref}}}} &= k_{{\\text{{relief}}}}\\,E_{{\\text{{ref}}}}\\,\\alpha_{{\\text{{ref}}}}\\,\\Delta T "
         f"= {s.sigma_thermal_ref_MPa:.1f}\\;\\text{{MPa}}"),
        r"\end{align}",
        "",
        (r"The relief factor $k_{\text{relief}} = 0.4$ accounts for slip joints, "
         r"floating panels, segmented skins, and blade-root slots that mitigate "
         r"thermal stress in real aerospace structures. Applied uniformly to the "
         r"reference thermal stress and to the per-material thermal stress in the "
         r"feasibility screen."),
        "",
        r"\textbf{Combined Requirement (MIL-HDBK-5 \cite{milhdbk5}, safety factor $1.5\times$):}",
        r"\begin{align}",
        (f"\\sigma_{{\\text{{req}}}} &= 1.5 \\times "
         f"(\\sigma_{{\\text{{inertial}}}} + \\sigma_q) + \\sigma_{{\\text{{th,ref}}}} "
         f"= {s.sigma_tensile_required_MPa:.1f}\\;\\text{{MPa}}"),
        r"\end{align}",
        "",
        r"\subsection{Structural Results Summary}",
    ]

    rows = [
        (r"Inertial stress $\sigma_{\text{inertial}}$",
         _fmt_mpa_sci(s.sigma_inertial_MPa)),
        (r"Dynamic pressure $q_{\text{dyn}} / 10^6$ (raw)",
         f"{q_dyn_MPa:.4f}\\,MPa"),
        (r"Combined $\times 1.5$ (MIL-HDBK-5) $\sigma_{\text{comb}}$",
         f"{s.sigma_combined_MPa:.1f}\\,MPa"),
        (r"Reference thermal stress $\sigma_{\text{th,ref}}$",
         f"{s.sigma_thermal_ref_MPa:.1f}\\,MPa"),
    ]

    # Primary σ_req (always shown): the monolithic / exposed-primary case.
    if show_tps_variant:
        rows.append((
            r"\textbf{Required tensile strength} "
            r"(primary structure exposed, $T_{\text{wall}}$)",
            f"\\textbf{{{s.sigma_tensile_required_MPa:.1f}\\,MPa}}",
        ))
        rows.append((
            r"\textbf{Required tensile strength} "
            r"(primary structure under TPS, $T_{\text{soak}} \approx "
            f"{T_soak_floor:.0f}$\\,K)",
            f"\\textbf{{{sigma_req_tps:.1f}\\,MPa}}",
        ))
    else:
        rows.append((
            r"\textbf{Required tensile strength} $\sigma_{\text{tensile,req}}$",
            f"\\textbf{{{s.sigma_tensile_required_MPa:.1f}\\,MPa}}",
        ))

    table_rows = "\n".join(f"  {param} & {val} \\\\" for param, val in rows)

    lines += [
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
        r"Component & Value \\",
        r"\midrule",
        table_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]

    # Dual-requirement explainer: appears only when both rows were shown,
    # so bare-airframe subsonic cases are not burdened with a TPS disclaimer
    # that has no bearing on their design.
    if show_tps_variant:
        lines += [
            "",
            (r"\textbf{Two structural requirements.} If your design has no "
             r"thermal protection layer, the \emph{exposed} value above is "
             r"the binding requirement on the primary structure. If you are "
             r"using an ablator, thermal barrier coating, or active cooling, "
             r"the \emph{TPS-protected} value is binding on the substructure "
             r"beneath the hot face; see the materials selection and trade-off "
             r"sections for substrate-mode candidates evaluated on that basis. "
             r"For vehicles below the TPS-unlock regime "
             r"($T_{\text{wall}} < 1200$\,K and non-reentry / non-hypersonic-aircraft "
             r"categories), only the exposed value is shown --- TPS is not a "
             r"design axis at that temperature."),
        ]

    return "\n".join(lines)


def _sec_propulsion(physics, vehicle_category: str = "general") -> str:
    p = physics.propulsion
    KE_GJ = p.KE_J / 1e9
    P_GW  = p.P_peak_W / 1e9
    E_GJ  = p.E_total_J / 1e9

    # Energy budget — relabel the former "Total energy" row to make its
    # scaffold nature explicit. The underlying formula (P_peak * duration)
    # is a worst-case propulsion-system sizing upper bound, not a mission
    # energy integral; the row used to read as the latter.
    energy_rows = [
        ("Kinetic energy",  f"{KE_GJ:.4g}\\,GJ"),
        ("Peak power",      f"{P_GW:.4g}\\,GW"),
        (r"Peak-power $\times$ duration (worst-case scaffold)",
         f"{E_GJ:.4g}\\,GJ"),
    ]
    er = "\n".join(f"  {param} & {val} \\\\" for param, val in energy_rows)

    # Disclaimer printed immediately after the Energy Budget table in every
    # branch below (including reentry and turbine, whose fuel tables are
    # omitted). The scaffold disclaimer applies to the E_total row; the
    # fuel-mass disclaimer applies to the fuel-comparison table.
    energy_scaffold_note = (
        r"\textbf{Scaffold, not integral.} "
        r"The \emph{peak-power $\times$ duration} value assumes sustained peak "
        r"thrust at peak velocity for the entire mission duration --- it is an "
        r"upper bound for propulsion-system sizing, not an actual mission energy "
        r"integral. Real missions consume a fraction (typically 1/3 to 1/30) of "
        r"this value depending on thrust profile. Treat it as an order-of-magnitude "
        r"ceiling."
    )
    fuel_interpretation_note = (
        r"\textbf{Interpretation.} "
        r"These masses are the fuel quantity whose chemical energy equals the "
        r"reference mechanical energy budget listed above. They are \emph{not} "
        r"mission fuel estimates --- real missions require substantially less "
        r"(propellant-fraction effects, duty cycle, and engine efficiency all "
        r"reduce actual fuel load). Use these as order-of-magnitude scale checks, "
        r"not design sizing."
    )
    # Cost-axis caveat (Cost-Axis-on-Pareto-Front feature). Lives at the bottom of
    # the propulsion section because (a) it sits next to the other "this number is
    # informational, not a constraint" disclaimers, and (b) it applies uniformly
    # to all three branches (reentry, turbine, propulsive) since the per-row cost
    # column in Section~\ref{sec:materials} appears regardless of category.
    cost_caveat = (
        r"\textbf{Cost caveat.} "
        r"Material cost figures elsewhere in this report are order-of-magnitude "
        r"estimates from bulk-pricing literature ($\pm$50\%). Actual quotes vary "
        r"$10\times$ with form (sheet vs.\ bar vs.\ powder), quantity (1\,kg vs.\ "
        r"1000\,kg), market conditions, and certification level (commercial vs.\ "
        r"aerospace-grade). Use the cost column in Section~\ref{sec:materials} to "
        r"rank affordability classes (steel \textit{vs.}\ titanium \textit{vs.}\ "
        r"single-crystal nickel \textit{vs.}\ ceramic-matrix composite), not to "
        r"build a bill of materials."
    )

    # --- Reentry: aerodynamic deceleration, not propulsion ---
    if vehicle_category == "reentry":
        return "\n".join([
            r"\section{Propulsion Energy Context}",
            r"\textit{Informational --- not a material constraint.}",
            "",
            r"\subsection{Energy Budget (to be dissipated)}",
            r"\begin{tabular}{@{}ll@{}}",
            r"\toprule",
            r"Quantity & Value \\",
            r"\midrule",
            er,
            r"\bottomrule",
            r"\end{tabular}",
            "",
            energy_scaffold_note,
            "",
            r"\subsection{Propulsion Requirement}",
            r"\textbf{No propulsion fuel is required.} "
            r"Reentry vehicles enter the atmosphere ballistically (or after a deorbit "
            r"burn performed on-orbit) and decelerate through \emph{aerodynamic drag}. "
            r"The kinetic energy listed above is the amount that must be \emph{dissipated} "
            r"as heat into the thermal protection system and the atmospheric boundary "
            r"layer --- it is not an energy that the vehicle must supply. "
            r"Retro-propulsion (e.g.\ SpaceX Starship catch, Mars EDL) is a mission-specific "
            r"enhancement and is outside the scope of this tool. "
            r"The fuel-mass comparison table is therefore omitted for the reentry category.",
            "",
            cost_caveat,
        ])

    # --- Turbine: embedded in an engine, no vehicle-level fuel load ---
    if vehicle_category == "turbine":
        return "\n".join([
            r"\section{Propulsion Energy Context}",
            r"\textit{Informational --- not a material constraint. "
            r"Values refer to the aerodynamic scaffold only; turbine components "
            r"operate inside an engine whose propulsion budget is a system-level "
            r"choice.}",
            "",
            r"\subsection{Energy Budget (aerodynamic scaffold)}",
            r"\begin{tabular}{@{}ll@{}}",
            r"\toprule",
            r"Quantity & Value \\",
            r"\midrule",
            er,
            r"\bottomrule",
            r"\end{tabular}",
            "",
            energy_scaffold_note,
            "",
            r"\subsection{Propulsion Requirement}",
            r"\textbf{Turbine components do not carry their own fuel.} "
            r"They operate inside an engine whose fuel load is a system-level "
            r"choice outside the scope of this materials analysis. The KE / "
            r"peak-power / scaffold values above are computed from the ambient "
            r"Mach number for consistency with the other sections but do not "
            r"drive blade material selection --- hot-section temperature, creep "
            r"resistance, and oxidation margin do. "
            r"The fuel-mass comparison table is therefore omitted for the "
            r"turbine category.",
            "",
            cost_caveat,
        ])

    # --- Propulsive vehicles: fuel-mass table applies ---
    # Primary fuel row comes from _PRIMARY_FUEL_BY_CATEGORY (Item 5 of the
    # six-improvement plan); secondary rows (HTPB for missiles, LH2 for
    # general/comparison) are appended inline because their roles are
    # category-specific context, not a vehicle-level "primary" fuel.
    primary = _resolve_primary_fuel(vehicle_category, physics.peak_mach)
    if primary.use_kinetic_energy_basis:
        # Rocket-propelled vehicles: propellant scales with KE delivered, not
        # with sustained-thrust energy over the full flight duration. Using
        # E_total here (as we do for kerosene in air-breathing cruise) would
        # overstate X-15-class propellant load by ~30x. KE is the correct
        # first-order proxy. ~8 MJ/kg for NH3+LOX (NH3 LHV 18.6 MJ/kg,
        # O2/NH3 ~1.41).
        primary_mass = p.KE_J / primary.energy_density_J_per_kg
    else:
        primary_mass = getattr(p, primary.mass_attr)
    fuel_data = [(primary.label, primary.energy_density_tex, primary_mass)]

    if vehicle_category == "hypersonic_missile":
        htpb_mass = p.E_total_J / 5.0e6   # HTPB ~5 MJ/kg solid rocket propellant
        fuel_data.append(
            (r"Solid rocket (HTPB, ${\sim}5{\times}10^6$\,J/kg)",
             r"$5.0 \times 10^{6}$", htpb_mass)
        )
    elif vehicle_category not in ("aircraft", "hypersonic_aircraft"):
        # general (and any future propulsive category that lacks an explicit
        # entry): include LH2 as a high-specific-energy comparison reference.
        fuel_data.append(
            ("Liquid Hydrogen", r"$1.42 \times 10^{8}$", p.fuel_mass_LH2_kg)
        )
    fr = "\n".join(
        f"  {name} & {dens} & {_fmt(mass, '.3g')} \\\\"
        for name, dens, mass in fuel_data
    )

    missile_note = ""
    if vehicle_category == "hypersonic_missile":
        missile_note = (
            "\n\\footnotetext{\\textbf{Missile application note:} Cryogenic (LH$_2$) and "
            r"high-specific-energy fuel comparisons are reference context only. "
            "Tactical missiles are stored ready-to-fire and typically use solid rocket "
            "motors (e.g.\\ HTPB propellant) or stable storable liquid fuels (e.g.\\ JP-10). "
            "Cryogenic fuels are not operationally viable for expendable missile applications.}"
        )

    return "\n".join([
        r"\section{Propulsion Energy Context}",
        r"\textit{Informational --- not a material constraint. "
        r"Values give context for propulsion system scale.}",
        "",
        r"\subsection{Energy Budget}",
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        er,
        r"\bottomrule",
        r"\end{tabular}",
        "",
        energy_scaffold_note,
        "",
        fuel_interpretation_note,
        "",
        r"\subsection{Fuel Mass Comparison}",
        r"\begin{tabular}{@{}llr@{}}",
        r"\toprule",
        r"Fuel & Energy density (J/kg) & Energy-equivalent mass (kg) \\",
        r"\midrule",
        fr,
        r"\bottomrule",
        r"\end{tabular}",
        missile_note,
        "",
        cost_caveat,
    ])


def _sec_em(physics) -> str:
    em = physics.em
    P_kW = em.P_rad_W / 1e3
    # Mirror §4's plasma display so the threshold shown here matches the
    # actual code path (slender-body Mach 6 vs blunt-body Mach 10). The
    # EM dataclass carries the same plasma_threshold_slender flag.
    plasma_str = _plasma_sheath_display(em)

    rows = [
        ("Peak radiated power",      f"{P_kW:.3f}\\,kW"),
        ("Peak emission wavelength", f"{em.lambda_peak_um:.3f}\\,$\\mu$m"),
        ("Emission band",            _tex_escape(em.emission_band)),
        ("Plasma sheath",            plasma_str),
    ]
    table_rows = "\n".join(f"  {param} & {val} \\\\" for param, val in rows)

    return "\n".join([
        r"\section{Electromagnetic Signature}",
        r"\subsection{Peak Emission Wavelength --- Wien Displacement Law}",
        r"\begin{align}",
        r"\lambda_{\text{peak}} &= \frac{b}{T_{\text{wall}}} "
        r"= \frac{2.898 \times 10^{-3}\;\text{m\,K}}{T_{\text{wall}}} \\",
        (f"&= \\frac{{2.898 \\times 10^{{-3}}}}{{{physics.thermal.T_wall_K:.1f}}} "
         f"= {em.lambda_peak_um:.3f}\\;\\mu\\text{{m}}"),
        r"\end{align}",
        "",
        r"\subsection{EM Results Summary}",
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        table_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def _sec_materials(
    physics,
    match: MatchResult,
    cost_ceiling_usd: float = 0.0,
) -> str:
    v  = _verdict_str(match)
    vc = _verdict_color(match)

    lines = [
        r"\section{Materials Feasibility Results}\label{sec:materials}",
        "",
        r"\textit{Screening basis: the stagnation-point wall temperature "
        r"($T_{\text{wall}}$, Section~\ref{sec:thermal}) is used as a conservative "
        r"upper bound for thermal viability. "
        r"For blunt reentry vehicles this drives heat-shield design. "
        r"For winged vehicles the fuselage bulk temperature is substantially lower, "
        r"so materials shown here as marginal or not-viable at the stagnation point "
        r"may be perfectly viable for structural panels away from the nose. "
        r"Ablative coatings, thermal barrier coatings, and active cooling are not "
        r"accounted for --- historical engineering solutions using these techniques "
        r"may not appear in the viable list.}",
        "",
        r"\textbf{On the two $\sigma_{\text{req}}$ values in this report.} "
        r"Section~5 (Structural Analysis) reports a single reference "
        r"$\sigma_{\text{req}}$ computed with steel-like constants "
        r"($E_{\text{ref}} = 200$\,GPa, $\alpha_{\text{ref}} = 12\times10^{-6}$\,K$^{-1}$). "
        r"This is the \emph{screening} requirement --- a material-agnostic benchmark "
        r"used to rank materials on a common basis. "
        r"The per-material table below, however, evaluates each candidate against "
        r"its own material-specific structural requirement "
        r"$\sigma_{\text{req,mat}} = \sigma_{\text{comb}} + k_{\text{relief}}\,E_{\text{mat}}\,\alpha_{\text{mat}}\,\Delta T$ "
        r"(polymer-matrix composites use the reference constants because their "
        r"fiber-direction CTE is unrepresentative of laminate behaviour). "
        r"So a material with low $E \alpha$ (e.g.\ Inconel 718, C-C composite) "
        r"may pass even when its strength at $T_{\text{wall}}$ is below the "
        r"Section~5 reference $\sigma_{\text{req}}$ --- this is expected, not a "
        r"contradiction.",
        "",
        (f"\\begin{{center}}"
         f"\\colorbox{{{vc}}}{{\\parbox{{0.85\\textwidth}}"
         f"{{\\centering\\Large\\bfseries {_tex_escape(v)}}}}}"
         f"\\end{{center}}"),
        "",
    ]

    all_candidates = list(match.viable) + list(match.marginal)
    if all_candidates:
        lines.append(r"\subsection{Viable and Marginal Materials}")
        lines += _longtable_candidates(
            all_candidates,
            len(match.viable),
            vehicle_mass_kg=float(getattr(physics, "vehicle_mass_kg", 0.0)),
            cost_ceiling_usd=float(cost_ceiling_usd or 0.0),
        )

    if match.tps_coatings:
        lines.append(r"\subsection{Required Coating Layer}")
        lines.append(
            r"\textit{The following TPS / ablator materials are non-load-bearing "
            r"and must be paired with a metallic substructure from the table above. "
            r"They are sorted by thermal protection (ceiling minus $T_{\text{wall}}$, "
            r"highest first) and intentionally not ranked against primary metals on "
            r"structural margin --- they protect the substructure, not carry loads.}"
        )
        lines.append("")
        lines += _tps_coatings_table(match.tps_coatings)

    if match.no_material_viable and match.not_viable:
        lines.append(r"\subsection{Nearest Misses (Top 10)}")
        lines += _nearest_misses_table(match.not_viable[:10])

    if match.diagnosis:
        lines += [
            r"\subsection*{Diagnosis}",
            f"\\textit{{{_tex_escape(match.diagnosis)}}}",
        ]

    return "\n".join(lines)


def _sec_sensitivity(sensitivity, aux_files: dict) -> str:
    """Sensitivity-analysis section (Uncertainty & Sensitivity feature).

    Renders only when ``sensitivity`` is a populated
    ``core.sensitivity.SensitivityResult``. When ``sensitivity`` is
    ``None`` (user did not run the analysis) or has an empty
    ``materials`` list (no nominal viable material), returns an empty
    string so the section is silently omitted from the PDF.

    The section is inserted between the materials longtable
    (Section~\\ref{sec:materials}) and the component-zone refinement,
    so a reader who just finished reviewing the viable list immediately
    sees "...and here is how that list holds up under realistic input
    uncertainty" before drilling into per-zone detail.

    The tornado PNG is written to ``aux_files`` under a stable name
    (``sensitivity_tornado.png``) — the same mechanism the Pareto
    section uses. ``\\includegraphics`` then references the extension-
    stripped basename.
    """
    if sensitivity is None:
        return ""
    if not sensitivity.materials:
        # Nominal had no viable material — the impossibility diagnosis in
        # the materials section has already told the engineer what to do.
        return ""

    # Stash the tornado PNG under a predictable name. The compiler helper
    # writes every aux_files entry into the temp build dir, so the
    # \includegraphics line finds the file by basename alone.
    chart_filename = "sensitivity_tornado.png"
    if sensitivity.chart_png:
        aux_files[chart_filename] = sensitivity.chart_png
    include_name = chart_filename.rsplit(".", 1)[0]

    # Lazy import: keeps matplotlib (transitively pulled by core.sensitivity)
    # off the import path for PDFs that don't include this section.
    from core.sensitivity import _INPUT_DISPLAY_NAMES

    spec = sensitivity.spec
    lines = [
        r"\section{Sensitivity Analysis}\label{sec:sensitivity}",
        "",
        r"Materials that are \emph{just} viable at the nominal flight envelope "
        r"can drop out of the viable list under realistic input uncertainty --- "
        r"a knife-edge pick is a poor bet for a \$4M wind-tunnel campaign. "
        r"This section sweeps each of four envelope inputs about its nominal "
        r"value and counts, for every material in the nominal viable list, "
        r"the fraction of perturbed scenarios in which that material remained "
        r"viable.",
        "",
        r"\textbf{Sweep configuration.} "
        f"Mach was swept $\\pm {spec.mach_delta_frac*100:.0f}\\%$, "
        f"vehicle mass $\\pm {spec.mass_delta_frac*100:.0f}\\%$, "
        f"nose radius $\\pm {spec.R_n_delta_frac*100:.0f}\\%$, "
        f"peak g-load $\\pm {spec.g_load_delta_frac*100:.0f}\\%$, "
        f"with {spec.n_samples} equally-spaced samples per input "
        f"({4 * spec.n_samples} total perturbed scenarios).",
        "",
        r"\textbf{Robustness labels.} "
        r"A material is \emph{robust} when it survives at least 90\,\% of "
        r"perturbed scenarios, \emph{borderline} between 50\,\% and 90\,\%, "
        r"and \emph{knife-edge} below 50\,\%. The \emph{critical input} is "
        r"whichever swept input dropped the material from viable most often.",
        "",
    ]

    # --- Tornado chart ---
    if sensitivity.chart_png:
        lines.append(r"\begin{center}")
        lines.append(
            r"\includegraphics[width=0.78\textwidth]{" + include_name + r"}"
        )
        lines.append(r"\end{center}")
        lines.append("")
        lines.append(
            r"\textit{Each bar shows how much each input's uncertainty range "
            r"erodes the nominal safety margin (in percentage points) for "
            r"the top-ranked material. Longer bars indicate inputs that must "
            r"be measured more precisely before committing to this material; "
            r"bars labelled `negligible' indicate inputs the material is "
            r"insensitive to.}"
        )
        lines.append("")

    # --- Robustness table ---
    # Sort materials by fraction descending so the bullet-proof picks read
    # first. This is the LaTeX twin of the Streamlit _show_sensitivity
    # sort — same ordering in the PDF and the UI.
    rows = sorted(
        sensitivity.materials,
        key=lambda m: m.robustness_fraction,
        reverse=True,
    )
    lines.append(r"\textbf{Per-material robustness.}")
    lines.append("")
    lines.append(r"\begin{tabular}{l l r l}")
    lines.append(r"\toprule")
    lines.append(r"Material & Label & Scenarios & Critical input \\")
    lines.append(r"\midrule")
    for r in rows:
        name  = _tex_escape(r.material_name)
        label = _tex_escape(r.robustness_label)
        cell  = f"{r.n_scenarios_viable} / {r.n_scenarios_total}"
        crit  = _tex_escape(
            _INPUT_DISPLAY_NAMES.get(r.critical_input, r.critical_input)
        )
        lines.append(f"{name} & {label} & {cell} & {crit} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")

    # --- Knife-edge and borderline narratives ---
    knife_edges = [m for m in sensitivity.materials
                   if m.robustness_label == "knife-edge"]
    borderlines = [m for m in sensitivity.materials
                   if m.robustness_label == "borderline"]

    delta_pct_for = {
        "mach":   spec.mach_delta_frac,
        "mass":   spec.mass_delta_frac,
        "R_n":    spec.R_n_delta_frac,
        "g_load": spec.g_load_delta_frac,
    }

    if knife_edges or borderlines:
        lines.append(r"\textbf{Risk notes.}")
        lines.append(r"\begin{itemize}")
        for m in knife_edges:
            n_drops   = m.n_scenarios_total - m.n_scenarios_viable
            crit      = _tex_escape(
                _INPUT_DISPLAY_NAMES.get(m.critical_input, m.critical_input)
            )
            pct       = delta_pct_for.get(m.critical_input, 0.10) * 100
            name_esc  = _tex_escape(m.material_name)
            lines.append(
                r"\item \textbf{" + name_esc + r"} is a \emph{knife-edge} "
                f"pick --- it drops out of viable in {n_drops} of "
                f"{m.n_scenarios_total} perturbed scenarios. "
                f"Most fragile to \\textbf{{{crit}}} "
                f"(swept $\\pm {pct:.0f}\\%$)."
            )
        for m in borderlines:
            crit      = _tex_escape(
                _INPUT_DISPLAY_NAMES.get(m.critical_input, m.critical_input)
            )
            name_esc  = _tex_escape(m.material_name)
            frac_pct  = m.robustness_fraction * 100
            lines.append(
                r"\item \textbf{" + name_esc + r"} is \emph{borderline} "
                f"--- viable in {frac_pct:.0f}\\% of scenarios, "
                f"most sensitive to \\textbf{{{crit}}}."
            )
        lines.append(r"\end{itemize}")
        lines.append("")

    return "\n".join(lines)


def _sec_component_zones(physics, match: MatchResult) -> str:
    """Per-zone material recommendations (Item 6 of the six-improvement plan).

    The whole-vehicle MatchResult rendered in Section~\\ref{sec:materials}
    represents the *worst* zone of the vehicle --- the leading edge or
    stagnation point where $T_{\\text{wall}}$ and $\\sigma_{\\text{req}}$
    peak. A real airframe is built out of zones with very different thermal
    and structural demands. This section decomposes the vehicle into named
    zones with locally scaled multipliers (see ``core/component_zones.py``)
    and re-runs the matching pipeline against each zone separately, so a
    user can see what materials become viable at internal / leeward / aft
    locations the whole-vehicle screen rejected.
    """
    # Local import keeps the dependency one-way and avoids a circular import
    # if the zone module ever needs anything from latex_export in the future.
    from core.component_zones import evaluate_zones

    zone_results = evaluate_zones(physics, match.vehicle_category)
    if not zone_results:
        return ""  # unknown category — silently omit the section

    vc_label = _VEHICLE_CATEGORY_LABELS.get(
        match.vehicle_category, match.vehicle_category
    )

    lines = [
        r"\section{Per-Zone Material Recommendations}\label{sec:component_zones}",
        "",
        r"\textit{The whole-vehicle results in Section~\ref{sec:materials} "
        r"represent the \emph{worst} zone of the vehicle --- the leading "
        r"edge or stagnation point where $T_{\text{wall}}$ and "
        r"$\sigma_{\text{req}}$ peak. A real airframe is built out of zones "
        r"with very different thermal and structural demands: an internal "
        r"spar 10\,cm aft of the nose sees a fraction of the leading-edge "
        r"temperature. This section decomposes the vehicle into named zones "
        r"using per-category thermal and structural multipliers (engineering "
        r"estimates documented in \texttt{core/component\_zones.py}, derived "
        r"from recovery-factor variation with local flow angle and "
        r"membrane-vs-bending stress distribution) and re-runs the matching "
        r"pipeline against each zone separately. The first zone in each "
        r"category is the calibration anchor: its multipliers are unity, so "
        r"its viable list reproduces the whole-vehicle result.}",
        "",
        f"Vehicle category: \\textbf{{{_tex_escape(vc_label)}}}. "
        f"Zones evaluated: {len(zone_results)}.",
        "",
    ]

    for i, zr in enumerate(zone_results, start=1):
        z = zr.zone
        lines += [
            f"\\subsection{{Zone {i}: {_tex_escape(z.name)}}}",
            f"\\textit{{{_tex_escape(z.description)}}}",
            "",
            r"\begin{tabular}{@{}lr@{}}",
            r"\toprule",
            r"Quantity & Value \\",
            r"\midrule",
            f"  $T_{{\\text{{wall, zone}}}}$ & {zr.T_wall_zone_K:.1f}\\,K \\\\",
            f"  $\\sigma_{{\\text{{req, zone}}}}$ & {zr.sigma_req_zone_MPa:.2f}\\,MPa \\\\",
            f"  Thermal multiplier (rise basis) & {z.t_wall_multiplier:.2f} \\\\",
            f"  Structural multiplier & {z.sigma_req_multiplier:.2f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]

        viable = list(zr.match.viable)
        marginal = list(zr.match.marginal)
        if viable:
            lines.append(r"\textbf{Top viable materials for this zone:}")
            lines.append(r"\begin{itemize}")
            for c in viable[:3]:
                lines.append(
                    f"  \\item {_tex_escape(c.material.name)} "
                    f"({_tex_escape(_cat_label(c.material.category))}, "
                    f"$\\rho = {c.material.density_kgm3:.0f}$\\,kg/m$^3$, "
                    f"thermal margin {c.thermal_margin_K:+.0f}\\,K)"
                )
            lines.append(r"\end{itemize}")
        elif marginal:
            lines.append(r"\textbf{No viable material at this zone; "
                         r"closest marginal candidates:}")
            lines.append(r"\begin{itemize}")
            for c in marginal[:3]:
                lines.append(
                    f"  \\item {_tex_escape(c.material.name)} "
                    f"({_tex_escape(_cat_label(c.material.category))}, "
                    f"thermal margin {c.thermal_margin_K:+.0f}\\,K)"
                )
            lines.append(r"\end{itemize}")
        else:
            lines.append(
                r"\textit{No single material satisfies this zone's "
                r"requirements; refer to the whole-vehicle nearest-miss "
                r"diagnostics in Section~\ref{sec:materials}.}"
            )
        lines.append("")

    return "\n".join(lines)


def _tps_coatings_table(coatings: list) -> list:
    """Dedicated LaTeX table for non-load-bearing TPS / ablator coatings."""
    lines = [
        r"\rowcolors{2}{gray!8}{white}",
        r"\begin{tabular}{p{4.0cm}lrrrl}",
        r"\toprule",
        r"Coating Material & Cat. & $\rho$\,kg/m$^3$ & Ceil.\,K & Margin\,K & Th. \\",
        r"\midrule",
    ]
    for c in coatings:
        lines.append(
            f"{_tex_escape(c.material.name)} & "
            f"{_tex_escape(_cat_label(c.material.category))} & "
            f"{c.material.density_kgm3:.0f} & "
            f"{c.thermal_ceiling_K:.0f} & "
            f"{c.thermal_margin_K:+.0f} & "
            f"{_tex_escape(c.thermal_status)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def _longtable_candidates(
    candidates: list,
    n_viable: int,
    vehicle_mass_kg: float = 0.0,
    cost_ceiling_usd: float = 0.0,
) -> list:
    """Render the viable+marginal materials longtable.

    The optional ``vehicle_mass_kg`` enables the "Est. material cost" column
    (Cost-Axis-on-Pareto-Front feature): per-row cost = ``cost_usd_per_kg
    \u00d7 vehicle_mass_kg``, formatted via ``_format_cost_usd``. The
    column is included whenever ``vehicle_mass_kg > 0``; legacy callers that
    don't pass it (none in the current tree, but defensive) get the original
    9-column layout.

    ``cost_ceiling_usd`` is informational only here --- per-row red-flag
    highlighting lives in the Streamlit table; in the LaTeX longtable we
    keep the row colours alternating gray/white so over-ceiling rows are
    flagged by the cost column itself, not by row colour. We accept the
    parameter so the caller doesn't have to special-case its absence.
    """
    show_cost = vehicle_mass_kg and vehicle_mass_kg > 0.0
    if show_cost:
        col_spec   = r"\begin{longtable}{p{3.0cm}lrrlrrrrr}"
        header_row = (
            r"Material & Cat. & Ceil.\,K & Margin\,K & Th. & "
            r"Str@T\,MPa & $\sigma_r$\,MPa & SM\% & Min.\,Margin & Est.~Cost \\"
        )
        n_cols = 10
    else:
        col_spec   = r"\begin{longtable}{p{3.0cm}lrrlrrrr}"
        header_row = (
            r"Material & Cat. & Ceil.\,K & Margin\,K & Th. & "
            r"Str@T\,MPa & $\sigma_r$\,MPa & SM\% & Min.\,Margin \\"
        )
        n_cols = 9

    lines = [
        r"\rowcolors{2}{gray!8}{white}",
        col_spec,
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{{n_cols}}}{{c}}{{\small\itshape (continued from previous page)}}\\",
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{n_cols}}}{{r}}{{\small\itshape (continued on next page)}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    sm_flag_needed = False
    for i, c in enumerate(candidates):
        if i == n_viable and n_viable > 0 and len(candidates) > n_viable:
            lines += [
                r"\midrule",
                rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{--- Marginal candidates below ---}}}} \\",
                r"\midrule",
            ]
        sm_pct = c.structural_margin_fraction * 100.0
        if sm_pct > 500.0:
            sm_str = f"{sm_pct:+.1f}$^{{\\dag}}$"
            sm_flag_needed = True
        else:
            sm_str = f"{sm_pct:+.1f}"
        # Substrate-mode tag: distinguishes "metal under ablative coating" from
        # "direct exposure to T_wall" in the materials table.
        if getattr(c, "evaluation_mode", "direct") == "substrate":
            name_cell = (
                f"{_tex_escape(c.material.name)}"
                r" \textsubscript{\textit{(substrate)}}"
            )
        else:
            name_cell = _tex_escape(c.material.name)
        if show_cost:
            cost_per_kg = float(getattr(c.material, "cost_usd_per_kg", 0.0))
            row_cost_usd = cost_per_kg * vehicle_mass_kg
            cost_cell = _format_cost_usd(row_cost_usd)
            # Bold the cost cell when it crosses the ceiling --- the visual
            # equivalent of the Streamlit table's red-row treatment.
            if (
                cost_ceiling_usd
                and cost_ceiling_usd > 0.0
                and row_cost_usd > cost_ceiling_usd
            ):
                cost_cell = rf"\textbf{{{cost_cell}}}"
            lines.append(
                f"{name_cell} & "
                f"{_tex_escape(_cat_label(c.material.category))} & "
                f"{c.thermal_ceiling_K:.0f} & "
                f"{c.thermal_margin_K:+.0f} & "
                f"{_tex_escape(c.thermal_status)} & "
                f"{c.strength_at_T_wall_MPa:.0f} & "
                f"{c.sigma_req_material_MPa:.0f} & "
                f"{sm_str} & "
                f"{c.score:.3f} & "
                f"{cost_cell} \\\\"
            )
        else:
            lines.append(
                f"{name_cell} & "
                f"{_tex_escape(_cat_label(c.material.category))} & "
                f"{c.thermal_ceiling_K:.0f} & "
                f"{c.thermal_margin_K:+.0f} & "
                f"{_tex_escape(c.thermal_status)} & "
                f"{c.strength_at_T_wall_MPa:.0f} & "
                f"{c.sigma_req_material_MPa:.0f} & "
                f"{sm_str} & "
                f"{c.score:.3f} \\\\"
            )

    lines.append(r"\end{longtable}")
    if sm_flag_needed:
        lines.append(
            r"\footnotetext[$\dag$]{Structural model may overstate margin for this "
            r"material class. Per-material checks use each material's own $E$ and "
            r"$\alpha$ with a thermal expansion relief factor (0.4) applied uniformly; "
            r"polymer-matrix composites still use reference steel constants because "
            r"unidirectional-fiber CTE understates real laminate thermal stress. "
            r"Engineering judgement required.}"
        )
    return lines


def _nearest_misses_table(candidates: list) -> list:
    lines = [
        r"\begin{tabular}{p{3.0cm}lrrlr}",
        r"\toprule",
        r"Material & Cat. & Ceil.\,K & Margin\,K & Failed & Deficit \\",
        r"\midrule",
    ]

    for c in candidates:
        if c.thermal_status == "fail":
            failed  = "Thermal"
            deficit = f"{c.thermal_margin_K:.0f}\\,K"
        else:
            sm_pct  = c.structural_margin_fraction * 100.0
            failed  = "Structural"
            deficit = f"{sm_pct:.1f}\\%"

        lines.append(
            f"{_tex_escape(c.material.name)} & "
            f"{_tex_escape(_cat_label(c.material.category))} & "
            f"{c.thermal_ceiling_K:.0f} & "
            f"{c.thermal_margin_K:+.0f} & "
            f"{failed} & "
            f"{deficit} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def _pareto_partition_block(
    heading: str,
    framing: str,
    partition_candidates: list,
    objective_values,
    pareto_mask,
    trade_offs: list,
    chart_filename: str,
    chart_b64,
    aux_files: dict,
) -> list:
    """Render one Pareto partition (direct or substrate) as a LaTeX subsection.

    Returns a list of LaTeX lines. Callers decide whether to invoke this
    function at all (the partition-emptiness check happens upstream so the
    top-level section can skip empty partitions cleanly).
    """
    import base64 as _b64
    lines = [r"\subsection*{" + heading + "}"]
    if framing:
        lines.append(framing)
        lines.append("")

    if chart_b64 is not None:
        try:
            aux_files[chart_filename] = _b64.b64decode(chart_b64)
            include_name = chart_filename.rsplit(".", 1)[0]
            lines.append(r"\begin{center}")
            lines.append(
                r"\includegraphics[width=0.85\textwidth]{" + include_name + r"}"
            )
            lines.append(r"\end{center}")
            lines.append("")
        except Exception:
            lines.append(r"\emph{Chart generation unavailable for this partition.}")
            lines.append("")

    if trade_offs:
        lines.append(r"\textbf{Trade-off summary.}")
        lines.append(r"\begin{enumerate}")
        for desc in trade_offs:
            lines.append(r"\item " + _tex_escape(desc))
        lines.append(r"\end{enumerate}")

    lines.append(r"\textbf{Pareto front members.}")
    lines.append("")
    lines.append(r"\begin{tabular}{l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Material & Weight & Thermal & Structural & Availability \\")
    lines.append(r"\midrule")
    for i, is_front in enumerate(pareto_mask):
        if is_front:
            name = _tex_escape(partition_candidates[i].material.name)
            vals = " & ".join(f"{objective_values[i, j]:.3f}" for j in range(4))
            lines.append(f"{name} & {vals} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return lines


def _sec_transient_heat(
    physics, match: MatchResult, panel_thickness_m: float
) -> str:
    """Section: 1D transient heat solver results per material.

    Renders a longtable of every direct-mode candidate whose
    ``transient_status == "applied"`` with its peak surface,
    midpoint (via the back-face by proxy), and back-face
    temperatures. Header paragraph documents the 1D solver
    methodology so a reader knows what physics produced the
    Soak@Life numbers in the materials table.

    Omitted entirely when no candidate triggered the transient
    stage (sustained-flight envelopes — the static T_wall check
    is sufficient and reported in the materials section).
    """
    cands = [
        c for c in (
            list(match.viable) + list(match.marginal)
            + list(match.not_viable)
        )
        if getattr(c, "transient_status", "") == "applied"
        and c.material.category != "tps"
    ]
    if not cands:
        return ""

    head = [
        r"\section{Transient Heat / Soak Evaluation}"
        r"\label{sec:transient}",
        "",
        r"\textit{Materials in Section~\ref{sec:materials} were "
        r"screened using the static $T_{\text{wall}}$ recovery "
        r"temperature. For short-duration / boost-coast trajectories "
        r"the airframe never reaches that steady-state value: the "
        r"surface flashes briefly to the recovery temperature while "
        r"the internal substructure (back face) lags significantly. "
        r"This section reports the peak temperatures reached during "
        r"the actual flight, computed by a 1D finite-difference "
        r"transient heat solver integrating the heat equation "
        r"$\partial T / \partial t = \alpha \, \partial^2 T / \partial x^2$ "
        r"through the panel with convective + radiative surface "
        r"boundary condition and insulated back face (worst-case "
        r"internal soak).}",
        "",
        f"Panel thickness: \\textbf{{{panel_thickness_m * 1000.0:.1f}\\,mm}}. "
        f"Flight duration: \\textbf{{{physics.flight_duration_s:.0f}\\,s}}. "
        f"Materials evaluated: {len(cands)}.",
        "",
        r"\begin{longtable}{@{}lrrrl@{}}",
        r"\toprule",
        r"Material & $T_{\text{surf, peak}}$ (K) & "
        r"$T_{\text{back, peak}}$ (K) & "
        r"$t_{\text{at peak back}}$ (s) & Method \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Material & $T_{\text{surf, peak}}$ (K) & "
        r"$T_{\text{back, peak}}$ (K) & "
        r"$t_{\text{at peak back}}$ (s) & Method \\",
        r"\midrule",
        r"\endhead",
    ]

    rows = []
    # Sort by back-face peak ascending (coolest first — most viable
    # for short-duration vehicles).
    for c in sorted(
        cands, key=lambda c: c.transient_peak_backface_K or 1e9
    ):
        name = _tex_escape(c.material.name)
        rows.append(
            f"  {name} & "
            f"{c.transient_peak_surface_K:.0f} & "
            f"{c.transient_peak_backface_K:.0f} & "
            f"{(c.transient_time_at_peak_backface_s or 0.0):.1f} & "
            f"{_tex_escape(c.transient_method)} \\\\"
        )

    rows.append(r"\bottomrule")
    rows.append(r"\end{longtable}")

    return "\n".join(head + rows)


def _sec_creep_evaluation(
    physics, match: MatchResult, design_lifetime_hours: float
) -> str:
    """Section: lifecycle / creep evaluation per material.

    Renders a longtable of every direct-mode candidate with its LMP
    value, rupture stress at (T_wall, lifetime), creep margin, and
    pass/marginal/fail/unknown/not_applicable verdict. The header
    paragraph documents the Larson-Miller framework so a reader who
    has not seen MATVEC before knows what is being computed.

    Returns an empty string when ``design_lifetime_hours`` is below
    the 1000-hour threshold — the creep stage is essentially a no-op
    there and the section would clutter the report. Mirrors the UI
    banner in ``app.py`` which uses the same threshold.
    """
    if design_lifetime_hours < 1000.0:
        return ""

    # Pull every direct-mode (non-substrate, non-tps) candidate so the
    # section reports the same population the materials longtable does.
    cands = (
        list(match.viable)
        + list(match.marginal)
        + list(match.not_viable)
    )
    cands = [
        c for c in cands
        if getattr(c, "evaluation_mode", "direct") == "direct"
        and c.material.category != "tps"
    ]
    if not cands:
        return ""

    T_wall = physics.thermal.T_wall_K

    head = [
        r"\section{Lifecycle / Creep Evaluation}"
        r"\label{sec:creep}",
        "",
        r"\textit{Materials in Section~\ref{sec:materials} were "
        r"screened against an instantaneous thermal+structural "
        r"snapshot. This section adds the lifecycle dimension: at "
        r"sustained service over thousands of hours, metals creep "
        r"under combined temperature and stress and can rupture at "
        r"loads they would survive for a single mission. The "
        r"Larson-Miller parameter "
        r"$\mathrm{LMP} = T \cdot (C + \log_{10} t)$ collapses (T, t) "
        r"pairs onto a single coordinate; each material's published "
        r"stress-rupture data is fit as a piecewise-linear "
        r"$(\mathrm{LMP}, \sigma_r)$ curve. A material passes when "
        r"its rupture stress at the queried (T, t) point exceeds "
        r"$\sigma_{\text{req}}$ by at least 20\%.}",
        "",
        f"Design lifetime: \\textbf{{{design_lifetime_hours:,.0f}\\,h}}. "
        f"$T_{{\\text{{wall}}}}$: \\textbf{{{T_wall:.0f}\\,K}}. "
        f"Materials evaluated: {len(cands)}.",
        "",
        r"\begin{longtable}{@{}lrrrrl@{}}",
        r"\toprule",
        r"Material & LMP & $\sigma_r$ (MPa) & $\sigma_{\text{req}}$ "
        r"(MPa) & Margin & Status \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Material & LMP & $\sigma_r$ (MPa) & $\sigma_{\text{req}}$ "
        r"(MPa) & Margin & Status \\",
        r"\midrule",
        r"\endhead",
    ]

    n_pass = n_marg = n_fail = n_unk = n_na = 0
    rows = []
    # Group by status for readability — pass first, then marginal,
    # then fail, then unknown / n/a at the bottom.
    status_order = {"pass": 0, "marginal": 1, "fail": 2,
                    "unknown": 3, "not_applicable": 4}
    for c in sorted(cands, key=lambda c: status_order.get(c.creep_status, 5)):
        st = c.creep_status
        if st == "pass":      n_pass += 1
        elif st == "marginal":n_marg += 1
        elif st == "fail":    n_fail += 1
        elif st == "unknown": n_unk  += 1
        else:                 n_na   += 1

        name = _tex_escape(c.material.name)
        sigma_req = c.sigma_req_material_MPa

        if st in ("pass", "marginal", "fail") and c.creep_rupture_stress_MPa is not None:
            lmp_str = f"{c.creep_lmp_value:,.0f}" if c.creep_lmp_value else "--"
            sigma_r_str = f"{c.creep_rupture_stress_MPa:.0f}"
            # ``\%`` is the LaTeX-escaped percent sign. The double
            # backslash in the f-string literal collapses to one
            # backslash in the rendered string, giving ``\%`` to LaTeX.
            margin_str = (
                f"{(c.creep_margin_fraction or 0.0) * 100:+.1f}\\%"
            )
            status_str = {
                "pass": r"\textcolor{black}{pass}",
                "marginal": r"\textcolor{black}{marginal}",
                "fail": r"\textcolor{black}{\textbf{fail}}",
            }[st]
        elif st == "unknown":
            lmp_str = sigma_r_str = margin_str = "--"
            status_str = r"\textit{unknown}"
        else:
            lmp_str = sigma_r_str = margin_str = "--"
            status_str = r"\textit{n/a}"

        rows.append(
            f"  {name} & {lmp_str} & {sigma_r_str} & "
            f"{sigma_req:.0f} & {margin_str} & {status_str} \\\\"
        )

    rows.append(r"\bottomrule")
    rows.append(r"\end{longtable}")

    # Summary line below the table for at-a-glance totals.
    summary = (
        f"\\par\\noindent Summary: "
        f"\\textbf{{{n_pass}}} pass, "
        f"\\textbf{{{n_marg}}} marginal, "
        f"\\textbf{{{n_fail}}} fail, "
        f"{n_unk} unknown (no LMP data), "
        f"{n_na} not applicable (TPS / ceramic / cool-soak metal)."
    )

    return "\n".join(head + rows + ["", summary])


def _sec_pareto(physics, match: MatchResult, aux_files: dict) -> str:
    """Multi-objective Pareto trade-off section with embedded chart.

    Candidates are partitioned by ``evaluation_mode`` inside ``compute_pareto``.
    Direct-exposure candidates (primary structure sees T_wall) and substrate-
    mode candidates (primary structure sits under TPS and sees T_soak) are
    Pareto-ranked separately and rendered as two subsections. For monolithic-
    airframe vehicles below the TPS-unlock regime, the substrate partition is
    empty and only the direct subsection is shown --- matching the pre-split
    behavior for SR-71-class cases.
    """
    candidates = list(match.viable) + list(match.marginal)
    if len(candidates) < 3:
        return ""

    pareto = compute_pareto(candidates, physics, match.vehicle_category)
    has_direct = bool(pareto.pareto_front)
    has_substrate = bool(pareto.pareto_front_substrate)
    if not (has_direct or has_substrate):
        return ""

    # Generate chart PNGs per-partition (may be None on failure).
    direct_chart_b64 = None
    substrate_chart_b64 = None
    try:
        if has_direct:
            direct_chart_b64 = generate_pareto_chart_b64(pareto, is_substrate=False)
    except Exception:
        direct_chart_b64 = None
    try:
        if has_substrate:
            substrate_chart_b64 = generate_pareto_chart_b64(pareto, is_substrate=True)
    except Exception:
        substrate_chart_b64 = None

    lines = [
        r"\section{Multi-Objective Trade-off Analysis}",
        r"Four minimization objectives define the Pareto front: "
        r"weight penalty ($\rho / 5000$), "
        r"thermal deficit ($\max(0, T_{\text{ref}} - T_{\text{service}}) / T_{\text{ref}}$), "
        r"structural deficit ($\max(0, \sigma_{\text{req}} - \sigma_{\text{mat}}) / \sigma_{\text{req}}$), "
        r"and availability penalty ($1 - a$). "
        r"Exact dominance ($O(n^2)$) identifies the non-dominated set. "
        r"Candidates evaluated in direct-exposure mode and in substrate-under-TPS "
        r"mode are Pareto-ranked separately so that a primary-structure titanium "
        r"is not compared head-to-head against a nickel superalloy evaluated only "
        r"at a 400\,K soak temperature.",
        "",
    ]

    if has_direct:
        direct_framing = (
            r"\emph{Direct-exposure candidates.} "
            r"These materials form the primary structure and see "
            f"$T_{{\\text{{wall}}}} = {physics.thermal.T_wall_K:.0f}$\\,K directly. "
            r"Numbered markers on the chart correspond to the front-member table below."
        )
        lines += _pareto_partition_block(
            heading=r"Direct-exposure materials (primary structure sees $T_{wall}$)",
            framing=direct_framing,
            partition_candidates=pareto.candidates_direct,
            objective_values=pareto.objective_values,
            pareto_mask=pareto.pareto_mask,
            trade_offs=pareto.trade_off_descriptions,
            chart_filename="pareto_chart.png",
            chart_b64=direct_chart_b64,
            aux_files=aux_files,
        )
        lines.append("")

    if has_substrate:
        substrate_framing = (
            r"\emph{Substrate-mode candidates.} "
            r"These materials sit beneath a thermal protection layer (ablator, "
            r"TBC, or active cooling) and are evaluated at the soak-through "
            f"temperature $T_{{\\text{{soak}}}} \\approx {pareto.T_substrate_K:.0f}$\\,K, "
            r"not at $T_{\text{wall}}$. Mixing this list with the direct-exposure "
            r"list would compare apples to oranges; the split keeps both design "
            r"axes legible. Appearance here does \emph{not} mean the material "
            r"is viable against the full stagnation temperature --- it means the "
            r"material is viable as substructure when a TPS layer is present."
        )
        lines += _pareto_partition_block(
            heading=r"Substrate-mode materials (primary structure under TPS, $T_{soak}$)",
            framing=substrate_framing,
            partition_candidates=pareto.candidates_substrate,
            objective_values=pareto.objective_values_substrate,
            pareto_mask=pareto.pareto_mask_substrate,
            trade_offs=pareto.trade_off_descriptions_substrate,
            chart_filename="pareto_chart_substrate.png",
            chart_b64=substrate_chart_b64,
            aux_files=aux_files,
        )

    return "\n".join(lines)


def _sec_surrogate(physics, match: MatchResult) -> str:
    """Materials property space analysis via k-NN surrogate."""
    surr = find_nearest_candidates(
        physics, match.vehicle_category, match_result=match, k=5,
    )
    if not surr.candidates:
        return ""

    agr = surr.agreement_with_margin_ranking
    if agr >= 0.6:
        agr_text = "high"
    elif agr >= 0.4:
        agr_text = "moderate"
    else:
        agr_text = "low"

    viable_names = {c.material.name for c in list(match.viable) + list(match.marginal)}

    lines = [
        r"\section{Materials Property Space Analysis}",
        r"A k-nearest-neighbor surrogate model in 7-dimensional normalized property space "
        r"(density, tensile strength, service temperature, melting point, thermal conductivity, "
        r"Young's modulus, CTE) identifies the materials closest to the operational requirements. "
        # NOTE: ``%`` in LaTeX starts a comment, so the format spec
        # cannot end with a literal percent sign. Convert the fraction
        # to a number-times-100 and append an escaped ``\%`` so the
        # parenthetical reads cleanly in the PDF, e.g. "(14\%)".
        f"Agreement with the margin-based ranking is {agr_text} ({agr*100:.0f}\\%).",
        "",
        r"\begin{tabular}{r l l r l}",
        r"\toprule",
        r"\# & Material & Category & Distance & In Viable/Marginal \\",
        r"\midrule",
    ]

    for idx, (mat, dist) in enumerate(zip(surr.candidates, surr.distances), 1):
        name = _tex_escape(mat.name)
        cat = _tex_escape(mat.category)
        in_match = "Yes" if mat.name in viable_names else "---"
        lines.append(f"{idx} & {name} & {cat} & {dist:.3f} & {in_match} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def _sec_falsifiable(physics, match: MatchResult) -> str:
    th = physics.thermal
    s  = physics.structural

    candidates = list(match.viable) + list(match.marginal)
    if not candidates:
        candidates = list(match.not_viable)

    # Partition: direct-mode (stagnation-point) vs substrate-mode (under ablative)
    direct_candidates = [c for c in candidates if getattr(c, "evaluation_mode", "direct") == "direct"]
    substrate_candidates = [c for c in candidates if getattr(c, "evaluation_mode", "direct") == "substrate"]
    top_direct = direct_candidates[:5]
    top_substrate = substrate_candidates[:5]

    # Substrate soak temperature — matches _ABLATIVE_SUBSTRATE_T_FLOOR_K in matching_engine.py
    T_soak = max(th.T_ambient_K, 400.0)

    lines = [
        r"\section{Minimum Adequate Material Candidates}",
        r"The following quantified claims cover the materials with the smallest positive margin---"
        r"the minimum adequate choices. These are directly verifiable against the cited sources.",
    ]

    # Primary numbered list: direct-mode candidates only
    if top_direct:
        lines.append(r"\begin{enumerate}")
        for c in top_direct:
            t_pass = c.thermal_status != "fail"
            margin = c.thermal_margin_K
            ceil   = c.thermal_ceiling_K

            if t_pass:
                ratio        = ceil / th.T_wall_K if th.T_wall_K > 0 else 0.0
                verdict_word = r"\textbf{passes}"
                detail       = f"by a margin of {margin:.0f}\\,K (factor {ratio:.2f})"
            else:
                ratio        = th.T_wall_K / ceil if ceil > 0 else 0.0
                verdict_word = r"\textbf{fails}"
                detail       = f"by {abs(margin):.0f}\\,K (exceeds service limit by factor {ratio:.2f})"

            ck = _cite_key(c.material.citation)
            lines.append(
                f"\\item At the computed peak wall temperature of {th.T_wall_K:.0f}\\,K, "
                f"\\textbf{{{_tex_escape(c.material.name)}}} "
                f"(service limit {ceil:.0f}\\,K) {verdict_word} the thermal requirement "
                + detail + f". Source: \\cite{{{ck}}}."
            )

        if match.impossible and match.not_viable:
            best = match.not_viable[0]
            gap  = th.T_wall_K - best.thermal_ceiling_K
            lines.append(
                f"\\item No material in the {len(MATERIALS_DB)}-entry database meets the combined "
                f"thermal ({th.T_wall_K:.0f}\\,K) and structural "
                f"({s.sigma_tensile_required_MPa:.0f}\\,MPa) requirements. "
                f"The nearest thermal candidate, "
                f"\\textbf{{{_tex_escape(best.material.name)}}}, "
                f"falls short by {gap:.0f}\\,K."
            )

        lines.append(r"\end{enumerate}")
    elif match.impossible and match.not_viable:
        # No direct candidates at all — show impossible message standalone
        best = match.not_viable[0]
        gap  = th.T_wall_K - best.thermal_ceiling_K
        lines.append(
            f"No material in the {len(MATERIALS_DB)}-entry database meets the combined "
            f"thermal ({th.T_wall_K:.0f}\\,K) and structural "
            f"({s.sigma_tensile_required_MPa:.0f}\\,MPa) requirements. "
            f"The nearest thermal candidate, "
            f"\\textbf{{{_tex_escape(best.material.name)}}}, "
            f"falls short by {gap:.0f}\\,K."
        )

    # Substrate-mode candidates: separate section with T_soak-based factors
    if top_substrate:
        lines.append("")
        lines.append(
            r"\subsection*{Substrate-mode candidates "
            f"(evaluated at $T_{{\\text{{soak}}}} \\approx {T_soak:.0f}$\\,K, "
            r"requires TPS hot-face layer)}"
        )
        lines.append(r"\begin{enumerate}")
        for c in top_substrate:
            ceil   = c.thermal_ceiling_K
            margin = c.thermal_margin_K
            ratio  = ceil / T_soak if T_soak > 0 else 0.0
            ck = _cite_key(c.material.citation)
            mat_name = _tex_escape(c.material.name)
            lines.append(
                f"\\item \\textbf{{{mat_name}}} "
                f"(service limit {ceil:.0f}\\,K) is evaluated as a metallic substructure "
                f"at $T_{{\\text{{soak}}}} \\approx {T_soak:.0f}$\\,K, where it "
                f"\\textbf{{passes}} thermal requirements with a margin of "
                f"{margin:.0f}\\,K (factor {ratio:.2f} at $T_{{\\text{{soak}}}}$). "
                f"Note: {mat_name} cannot survive direct stagnation-point exposure "
                f"at {th.T_wall_K:.0f}\\,K---it requires an ablative TPS layer "
                f"as the hot-face material. Source: \\cite{{{ck}}}."
            )
        lines.append(r"\end{enumerate}")

    return "\n".join(lines)


def _sec_methodology(physics) -> str:
    if physics.thermal.uses_recovery_model:
        thermal_subsection = "\n".join([
            r"\subsection{Thermal Model --- Recovery Temperature Model}",
            r"The wall temperature is computed directly from the adiabatic wall (recovery) "
            r"temperature formula: "
            r"$T_{\text{wall}} = T_{\text{amb}}(1 + r\frac{\gamma-1}{2}M^2)$, $r = 0.85$ "
            r"(turbulent flat-plate recovery factor). "
            r"This is the physically correct primary model for sustained flight at Mach~$<$~5, "
            r"where aerodynamic heating flux is below the threshold at which radiation equilibrium "
            r"becomes the limiting constraint. "
            r"The Sutton-Graves stagnation-point correlation is not applied in this regime --- "
            r"it is calibrated for hypersonic blunt-body reentry (Mach~$\geq$~5) and "
            r"over-predicts heating at lower Mach numbers. "
            r"Uncertainty: $\pm15\%$ on $T_{\text{wall}}$.",
        ])
    else:
        thermal_subsection = "\n".join([
            r"\subsection{Thermal Model --- Sutton-Graves Correlation (Mach~$\geq$~5)}",
            r"The stagnation-point convective heating flux is computed using the Sutton-Graves "
            r"correlation (NASA TR-R-376 \cite{sutton1971}), building on the Detra-Hidalgo framework. "
            r"Valid for hypersonic blunt-body stagnation-point heating; accuracy $\pm20\%$ for "
            r"$M \geq 5$. "
            r"Tauber-Sutton radiative heating \cite{tauber1989} is added for $V \geq 6000$\,m/s. "
            r"Wall temperature is the radiation-equilibrium value $T_{\text{wall}} = "
            r"(q_{\text{total}} / \varepsilon\sigma_{SB})^{1/4}$.",
        ])

    thermal_limitation = (
        r"\medskip\noindent\textbf{Note:} $T_{\text{wall}}$ represents the worst-case "
        r"stagnation-point temperature at the nose tip or leading edge. "
        r"For slender winged vehicles such as aircraft and missiles, fuselage bulk skin "
        r"temperatures are significantly lower than the stagnation-point value --- "
        r"typically 60 to 80 percent of $T_{\text{wall}}$ for supersonic cruise vehicles. "
        r"The material recommendations reflect stagnation-point requirements and are "
        r"conservative for fuselage structure. Leading edge and nose cap components "
        r"should be evaluated at $T_{\text{wall}}$ while primary airframe structure "
        r"operates at substantially lower temperatures."
    )

    return "\n".join([
        r"\section{Methodology}",
        "",
        thermal_subsection,
        "",
        thermal_limitation,
        "",
        r"\subsection{Structural Model}",
        r"Simplified beam theory with the MIL-HDBK-5 structural safety factor of "
        r"$1.5\times$ \cite{milhdbk5}. "
        r"Uniform stress distribution assumption yields a conservative lower bound on peak stress "
        r"for non-uniform geometries. "
        r"Reference constants $E_{\text{ref}} = 200$\,GPa and "
        r"$\alpha_{\text{ref}} = 12 \times 10^{-6}$\,K$^{-1}$ define the structural requirement "
        r"and are used directly for polymer matrix composites; "
        r"material-specific values are used for metals, ceramics, and UHTCs.",
        "",
        r"\subsection{Propulsion Energy Model}",
        r"Work-energy theorem with a simplified flight-envelope energy integral. "
        r"Fuel mass comparisons are for reference context only --- not a propulsion design tool.",
        "",
        r"\subsection{Electromagnetic Signature Model}",
        r"Stefan-Boltzmann blackbody emission integrated over the nose cap area at $T_{\text{wall}}$. "
        r"Peak emission wavelength via Wien's displacement law: "
        r"$\lambda_{\text{peak}} = b/T$, $b = 2.898 \times 10^{-3}$\,m\,K. "
        r"Plasma sheath onset uses a two-tier threshold that depends on nose radius: "
        r"blunt bodies ($R_n \geq 0.5$\,m) trip the flag at $M > 10.0$ and altitude $< 80$\,km, "
        r"while slender bodies ($R_n < 0.5$\,m) trip at $M > 6.0$ and altitude $< 80$\,km "
        r"to capture the partial-plasma / radio-attenuation effects documented on slender "
        r"hypersonic vehicles at Mach 6--8 (e.g.\ X-15 flight records at $M \approx 6.7$).",
    ])


def _sec_reproducibility(physics, ts: str) -> str:
    db_count = len(MATERIALS_DB)
    block = (
        f"MATVEC Version    : {_MATVEC_VERSION}\n"
        f"Materials DB      : {db_count} entries\n"
        f"Surrogate model   : {get_model_version()}\n"
        f"C_sutton_graves   : {C_SUTTON_GRAVES:.4e}\n"
        f"sigma_SB          : {SIGMA_SB:.9e} W/(m^2 K^4)\n"
        f"G0                : {G0} m/s^2\n"
        f"R_AIR             : {R_AIR} J/(kg K)\n"
        f"Analysis timestamp: {ts}"
    )
    return "\n".join([
        r"\section{Reproducibility}",
        r"The following block uniquely identifies the software state and constants used. "
        r"The same inputs should yield identical results.",
        r"\begin{verbatim}",
        block,
        r"\end{verbatim}",
    ])


def _sec_references(match: MatchResult) -> str:
    keys = set(_ALWAYS_CITE)
    all_mats = list(match.viable) + list(match.marginal) + list(match.not_viable[:10])
    for c in all_mats:
        keys.add(_cite_key(c.material.citation))

    bibitems = [_ALL_BIBITEMS[k] for k in sorted(keys) if k in _ALL_BIBITEMS]

    return "\n".join([
        r"\begin{thebibliography}{99}",
        *bibitems,
        r"\end{thebibliography}",
    ])


# ---------------------------------------------------------------------------
# pdflatex compilation
# ---------------------------------------------------------------------------

def _compile_tex(tex_source: str, aux_files: dict | None = None) -> bytes | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write auxiliary files (e.g., chart PNGs)
        if aux_files:
            for fname, data in aux_files.items():
                with open(os.path.join(tmpdir, fname), "wb") as f:
                    f.write(data)
        tex_path = os.path.join(tmpdir, "report.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_source)
        result = None
        for _ in range(2):  # two passes for cross-references / TOC
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode",
                 "-output-directory", tmpdir, tex_path],
                capture_output=True,
                timeout=60,
            )
        pdf_path = os.path.join(tmpdir, "report.pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                return f.read()
        # Surface the pdflatex log so callers can show a useful error message.
        log = ""
        if result is not None:
            log = (result.stdout or b"").decode("utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"pdflatex produced no PDF.\n\n{log}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_tex_source(
    physics,
    match: MatchResult,
    system_label: str = "Custom Analysis",
    *,
    cost_ceiling_usd: float = 1_000_000.0,
    sensitivity=None,
    design_lifetime_hours: float = 1.0,
    panel_thickness_m: float = 0.002,
) -> tuple[str, dict]:
    """Return (LaTeX document source string, auxiliary files dict).

    ``cost_ceiling_usd`` flows into the materials longtable so rows whose
    (cost/kg \u00d7 vehicle mass) exceeds it are bolded in-line. Matches the
    ``compute_pareto`` / session.options convention --- it's a reporting
    parameter, not a physics input.

    ``sensitivity`` is an optional ``core.sensitivity.SensitivityResult``.
    When supplied, a dedicated Sensitivity Analysis section is inserted
    between the materials longtable and the component-zone refinement.
    When ``None`` (the default), the section is silently omitted --- this
    keeps the headless CLI and regression-snapshot tests backwards-
    compatible, since neither asks for sensitivity data.
    """
    now = datetime.datetime.utcnow()
    ts  = now.isoformat(timespec="seconds") + "Z"

    label_esc = _tex_escape(system_label)

    aux_files: dict = {}

    sections = [
        _preamble(label_esc),
        _title_block(label_esc, ts),
        _sec_executive_summary(physics, match),
        _sec_input_parameters(
            physics, match.vehicle_category,
            design_lifetime_hours=design_lifetime_hours,
            panel_thickness_m=panel_thickness_m,
        ),
        _sec_atmospheric(physics),
        _sec_thermal(physics),
        _sec_structural(physics, match.vehicle_category),
        _sec_propulsion(physics, match.vehicle_category),
        _sec_em(physics),
        _sec_materials(physics, match, cost_ceiling_usd=cost_ceiling_usd),
        _sec_transient_heat(physics, match, panel_thickness_m),
        _sec_creep_evaluation(physics, match, design_lifetime_hours),
        _sec_sensitivity(sensitivity, aux_files),
        _sec_component_zones(physics, match),
        _sec_pareto(physics, match, aux_files),
        _sec_surrogate(physics, match),
        _sec_falsifiable(physics, match),
        _sec_methodology(physics),
        _sec_reproducibility(physics, ts),
        _sec_references(match),
        r"\end{document}",
    ]
    tex = "\n\n".join(s for s in sections if s)
    return tex, aux_files


def generate_report(
    physics,
    match: MatchResult,
    system_label: str = "Custom Analysis",
    *,
    cost_ceiling_usd: float = 1_000_000.0,
    sensitivity=None,
    design_lifetime_hours: float = 1.0,
    panel_thickness_m: float = 0.002,
) -> bytes | None:
    """Compile to PDF via pdflatex. Returns PDF bytes or None on failure.

    The optional ``sensitivity`` argument is passed straight through to
    ``generate_tex_source`` --- see that function's docstring for how
    the resulting section is rendered.

    ``design_lifetime_hours`` enables the Section~\\ref{sec:creep}
    Lifecycle / Creep Evaluation block. Default 1.0 h (single-flight)
    omits the section entirely; ≥1000 h emits the longtable + summary.

    ``panel_thickness_m`` is reported alongside the Section~\\ref{sec:transient}
    Transient Heat / Soak Evaluation table; the section appears only
    when at least one direct-mode candidate triggered the 1D solver
    (short-duration flights).
    """
    try:
        tex_source, aux_files = generate_tex_source(
            physics, match, system_label,
            cost_ceiling_usd=cost_ceiling_usd,
            sensitivity=sensitivity,
            design_lifetime_hours=design_lifetime_hours,
            panel_thickness_m=panel_thickness_m,
        )
        return _compile_tex(tex_source, aux_files)
    except FileNotFoundError:
        # pdflatex not installed on this machine.
        return None
    except (subprocess.TimeoutExpired, OSError):
        return None
    except RuntimeError:
        # pdflatex was found but failed to produce a PDF — re-raise so the
        # caller can surface the log to the user for debugging.
        raise
