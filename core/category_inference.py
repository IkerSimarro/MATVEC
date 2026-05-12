"""infer_category — derive vehicle category from a flight envelope.

Used by the Streamlit form's right-hand "Computed" panel and (optionally)
by the CLI as a default when --category is omitted. Pure function,
no Streamlit, no I/O — testable headlessly.

Turbine is intentionally never inferred. The binding parameter for the
turbine category is hot-section temperature, not the aerodynamic
envelope; a Mach 0.5 / sea-level point is indistinguishable from a
subsonic generic-structure analysis at the envelope level. The user
must override to "turbine" to opt into the hot-section temperature
input and the turbine matching-engine derate.
"""

VALID_CATEGORIES: tuple[str, ...] = (
    "general",
    "aircraft",
    "hypersonic_aircraft",
    "reentry",
    "hypersonic_missile",
    "turbine",
)


def infer_category(mach: float, alt_km: float, mass_kg: float) -> str:
    """Return one of VALID_CATEGORIES (never "turbine").

    Decision tree, evaluated top-down (first match wins):

      1. alt >= 60 km OR Mach >= 12       → reentry
      2. Mach >= 5  AND mass <  3000 kg   → hypersonic_missile
      3. Mach >= 5  AND mass >= 3000 kg   → hypersonic_aircraft
      4. Mach >= 2  AND mass <  3000 kg   → hypersonic_missile  (M>2 missile)
      5. Mach >= 0.4 AND mass >= 1000 kg  → aircraft
      6. otherwise                         → general

    The 60 km altitude boundary captures atmospheric-entry physics — once
    you're above the mesopause the dominant materials concern is ablative
    heat-shield response, regardless of whether the vehicle is a 5-tonne
    capsule or a 500-kg sample-return canister.

    The Mach 12 boundary catches re-entry trajectories that haven't yet
    descended to 60 km (e.g., upper-trajectory sample). Combined, the
    two upper boundaries cover both ballistic and lifting reentries.

    The Mach 5 boundary is the conventional hypersonic threshold. Mass
    splits the regime between expendable missiles (TPS / polymer
    composites excluded; specific strength weighted 60%) and crewed /
    semi-reusable aircraft (TPS hot-face options included; hot-structure
    alloys up to 8500 kg/m³ allowed).

    The Mach 2 / mass<3000 fork covers supersonic missiles that aren't
    fully hypersonic — per CATEGORY_DESCRIPTIONS in app.py the
    hypersonic_missile category covers "high-speed (M > 2) expendable
    missile body structure".

    The Mach 0.4 / mass>=1000 fork covers everything from subsonic
    transports to Concorde-class supersonic cruise. The mass floor
    keeps small low-speed structural panels in "general".
    """
    m = float(mach)
    a = float(alt_km)
    w = float(mass_kg)

    if a >= 60.0 or m >= 12.0:
        return "reentry"
    if m >= 5.0:
        return "hypersonic_missile" if w < 3000.0 else "hypersonic_aircraft"
    if m >= 2.0 and w < 3000.0:
        return "hypersonic_missile"
    if m >= 0.4 and w >= 1000.0:
        return "aircraft"
    return "general"
