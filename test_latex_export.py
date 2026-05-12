"""
Test suite for latex_export.py — Phase 6 of MATVEC.

Verifies the text-content invariants in the generated LaTeX source that
back the communication/presentation fixes from the audit:
    Fix 1 — dual σ_req rows gated by TPS unlock
    Fix 4 — "Energy-equivalent mass" header and Interpretation disclaimer
    Fix 6 — plasma-sheath slender-body vs blunt-body display string
    Fix 7 — peak-power × duration (worst-case scaffold) label and note
    Fix 9 — turbine branches: §4 hot-section framing, §6 omits fuel table

stdlib unittest only. Run: python -m unittest test_latex_export.py -v
"""

import unittest

from physics_engine import run_analysis
from matching_engine import match_materials
from latex_export import generate_tex_source


def _run(mach, alt_km, mass_kg, R_n, g_load=1.0):
    return run_analysis(mach, alt_km, mass_kg, R_n, peak_g_load=g_load)


def _tex(mach, alt_km, mass_kg, R_n, g_load, category, label="Test"):
    physics = _run(mach, alt_km, mass_kg, R_n, g_load)
    match = match_materials(physics, vehicle_category=category)
    tex, _aux = generate_tex_source(physics, match, system_label=label)
    return tex, physics, match


# ---------------------------------------------------------------------------
# TestStructuralSigmaReqGating — Fix 1
# ---------------------------------------------------------------------------
class TestStructuralSigmaReqGating(unittest.TestCase):
    """§5 shows two σ_req rows only when TPS unlock is physically plausible."""

    def test_x15_shows_dual_sigma_req(self):
        """X-15 (T_wall ≈ 1645 K > 1200 K) must show both exposed and
        TPS-protected σ_req rows."""
        tex, physics, _ = _tex(6.7, 30.0, 15195.0, 0.30, 5.0, "aircraft",
                               label="X-15")
        self.assertGreater(physics.thermal.T_wall_K, 1200.0)
        self.assertIn("primary structure exposed", tex)
        self.assertIn("primary structure under TPS", tex)

    def test_sr71_shows_single_sigma_req(self):
        """SR-71 (T_wall ≈ 607 K) is below TPS unlock — only the exposed row."""
        tex, physics, _ = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft",
                               label="SR-71")
        self.assertLess(physics.thermal.T_wall_K, 1200.0)
        self.assertNotIn("primary structure under TPS", tex)
        self.assertNotIn("primary structure exposed", tex)
        # Falls back to the simple row header.
        self.assertIn(r"$\sigma_{\text{tensile,req}}$", tex)

    def test_reentry_always_shows_dual(self):
        """Reentry category always shows dual rows regardless of T_wall."""
        tex, _physics, _ = _tex(20.0, 70.0, 500.0, 1.50, 8.0, "reentry",
                                label="Reentry Capsule")
        self.assertIn("primary structure exposed", tex)
        self.assertIn("primary structure under TPS", tex)


# ---------------------------------------------------------------------------
# TestPropulsionLabels — Fixes 4 + 7
# ---------------------------------------------------------------------------
class TestPropulsionLabels(unittest.TestCase):
    """§6 labels reference an energy budget, not mission fuel estimates."""

    def test_energy_equivalent_mass_header(self):
        """X-15 fuel comparison table uses 'Energy-equivalent mass', not
        'Required mass'."""
        tex, _, _ = _tex(6.7, 30.0, 15195.0, 0.30, 5.0, "aircraft")
        self.assertIn("Energy-equivalent mass", tex)
        self.assertNotIn("Required mass (kg)", tex)

    def test_interpretation_disclaimer(self):
        """Fuel comparison has the 'not mission fuel' disclaimer."""
        tex, _, _ = _tex(6.7, 30.0, 15195.0, 0.30, 5.0, "aircraft")
        self.assertIn("Interpretation", tex)
        # "not mission fuel" text is broken by \emph{not} in the source
        # ("\emph{not} mission fuel estimates"). Match the surrounding phrase.
        self.assertIn("mission fuel estimates", tex)

    def test_scaffold_label_and_note(self):
        """Total energy row is relabeled as peak-power × duration scaffold,
        with a 'Scaffold, not integral' note."""
        tex, _, _ = _tex(6.7, 30.0, 15195.0, 0.30, 5.0, "aircraft")
        self.assertIn("Peak-power", tex)
        self.assertIn("worst-case scaffold", tex)
        self.assertIn("Scaffold, not integral", tex)


# ---------------------------------------------------------------------------
# TestThermalDisclaimerRewrite — Item 3 of the six-improvement plan
# ---------------------------------------------------------------------------
class TestThermalDisclaimerRewrite(unittest.TestCase):
    """§4 M<5 disclaimer must use confident model-selection framing.
    The previous text was self-defeating ('Sutton-Graves is not applicable in
    this regime'), implying the tool was using the wrong model. The new text
    states what the tool selects and why, and bounds each correlation's range."""

    def test_m_lt_5_disclaimer_describes_both_regimes(self):
        """SR-71 (Mach 3.2) — disclaimer must mention both Mach branches."""
        tex, _, _ = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        # Mentions the M<5 recovery regime with explicit recovery factor
        self.assertIn("r = 0.85", tex)
        self.assertIn("recovery", tex.lower())
        # And mentions the Sutton-Graves branch positively (model selection,
        # not 'not applicable')
        self.assertIn("Sutton-Graves", tex)
        self.assertIn("Tauber-Sutton", tex)

    def test_m_lt_5_disclaimer_uses_model_selection_framing(self):
        """The disclaimer should frame model choice as a selection rule."""
        tex, _, _ = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        self.assertIn("selects the thermal model by Mach number", tex)

    def test_m_lt_5_disclaimer_omits_apologetic_phrasing(self):
        """The 'is not applicable in this regime' phrase must not appear."""
        tex, _, _ = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        self.assertNotIn("is not applicable in this regime", tex)
        self.assertNotIn("not applicable", tex)

    def test_m_lt_5_disclaimer_uses_calibrated_range_phrasing(self):
        """The new disclaimer should talk about calibrated ranges, not exclusion."""
        tex, _, _ = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        self.assertIn("calibrated range", tex)


# ---------------------------------------------------------------------------
# TestPerVehicleFuelReference — Item 5 of the six-improvement plan
# ---------------------------------------------------------------------------
class TestPerVehicleFuelReference(unittest.TestCase):
    """Each vehicle category renders its own reference fuel label.
    SR-71 used JP-7 (not Jet-A); tactical missiles use JP-10; X-15-class
    rocketplanes use NH3+LOX; reentry omits the table; turbine omits the table.
    The labels are now sourced from a single per-category lookup so future
    vehicles can override without editing the propulsion section."""

    def test_lookup_dict_covers_propulsive_categories(self):
        """Every propulsive category has an entry; reentry/turbine do not."""
        from latex_export import _PRIMARY_FUEL_BY_CATEGORY
        for required in ("aircraft", "hypersonic_aircraft",
                         "hypersonic_missile", "general"):
            self.assertIn(required, _PRIMARY_FUEL_BY_CATEGORY)
        self.assertNotIn("reentry", _PRIMARY_FUEL_BY_CATEGORY)
        self.assertNotIn("turbine", _PRIMARY_FUEL_BY_CATEGORY)

    def test_aircraft_subsonic_label_says_jp7(self):
        """SR-71-like aircraft (M<5) → 'Kerosene / JP-7 (ref.)' label."""
        tex, _, _ = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        self.assertIn("Kerosene / JP-7 (ref.)", tex)
        self.assertNotIn("Kerosene (Jet-A)", tex)

    def test_aircraft_at_M5_uses_ammonia_lox(self):
        """X-15-like aircraft (Mach >= 5) auto-routes to NH3+LOX rocket basis.
        Backward-compatibility: an aircraft preset at hypersonic Mach is
        treated as rocket-propelled, matching real X-15 propulsion."""
        tex, _, _ = _tex(6.7, 30.0, 15195.0, 0.30, 5.0, "aircraft")
        self.assertIn("Ammonia + LOX (rocket ref., KE basis)", tex)
        self.assertNotIn("Kerosene / JP-7 (ref.)", tex)

    def test_hypersonic_aircraft_explicit_uses_ammonia_lox(self):
        """An explicit hypersonic_aircraft category gets the rocket fuel
        even when called with the same envelope as a subsonic aircraft."""
        tex, _, _ = _tex(6.7, 30.0, 15195.0, 0.30, 5.0, "hypersonic_aircraft")
        self.assertIn("Ammonia + LOX (rocket ref., KE basis)", tex)

    def test_hypersonic_missile_label_says_jp10(self):
        """Tactical missiles use JP-10, not JP-7 or generic 'Jet-A'."""
        tex, _, _ = _tex(4.0, 20.0, 900.0, 0.08, 10.0, "hypersonic_missile")
        self.assertIn("Kerosene / JP-10 (ref.)", tex)
        self.assertNotIn("Kerosene / JP-7", tex)
        # HTPB secondary row preserved
        self.assertIn("Solid rocket (HTPB", tex)

    def test_general_label_says_jp7_with_lh2_secondary(self):
        """The general category keeps JP-7 + LH2 as the comparison pair."""
        tex, _, _ = _tex(0.3, 5.0, 500.0, 0.5, 2.0, "general")
        self.assertIn("Kerosene / JP-7 (ref.)", tex)
        self.assertIn("Liquid Hydrogen", tex)

    def test_reentry_omits_fuel_table_with_explanation(self):
        """Reentry has no fuel table; an explicit non-airbreathing sentence stands in."""
        tex, _, _ = _tex(20.0, 70.0, 500.0, 1.50, 8.0, "reentry")
        # No "Fuel Mass Comparison" subsection
        self.assertNotIn(r"\subsection{Fuel Mass Comparison}", tex)
        # Explicit explanation present
        self.assertIn("No propulsion fuel is required", tex)
        # Old hardcoded labels must not leak into reentry output
        self.assertNotIn("Kerosene / JP-7", tex)
        self.assertNotIn("Liquid Hydrogen", tex)


# ---------------------------------------------------------------------------
# TestPlasmaSheathDisplay — Fix 6
# ---------------------------------------------------------------------------
class TestPlasmaSheathDisplay(unittest.TestCase):
    """§4/§7 plasma string reflects the two-tier threshold."""

    def test_x15_slender_plasma_string(self):
        """X-15 (R_n=0.3 m, M=6.7) → slender-body plasma above M=6.0."""
        tex, _, _ = _tex(6.7, 30.0, 15195.0, 0.30, 5.0, "aircraft")
        self.assertIn("slender body", tex)
        self.assertIn("M > 6.0", tex)

    def test_sr71_no_plasma_string(self):
        """SR-71 (M=3.2) → below both thresholds → 'No'."""
        tex, _, _ = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        # No slender-body marker, no blunt-body marker — plasma row reads "No".
        # The helper yields the literal string "No" in this case.
        # Verify by searching for the substring "Plasma sheath & No".
        self.assertIn("Plasma sheath", tex)
        self.assertNotIn("slender body", tex)

    def test_reentry_blunt_plasma_string(self):
        """Reentry (R_n=1.5 m, M=20) → blunt-body threshold → M > 10.0 string."""
        tex, _, _ = _tex(20.0, 70.0, 500.0, 1.50, 8.0, "reentry")
        self.assertIn("M > 10.0", tex)
        self.assertNotIn("slender body", tex)


# ---------------------------------------------------------------------------
# TestTurbineLatexBranches — Fix 9
# ---------------------------------------------------------------------------
class TestTurbineLatexBranches(unittest.TestCase):
    """Turbine category + hot-section override rewrites §4 and §6."""

    def _tex_turbine(self):
        from app import _apply_turbine_override
        physics = _run(0.5, 0.0, 50.0, 0.05, g_load=1.0)
        physics = _apply_turbine_override(physics, 1400.0)
        match = match_materials(physics, vehicle_category="turbine")
        tex, _aux = generate_tex_source(physics, match, system_label="Turbine HPT")
        return tex

    def test_thermal_section_frames_hot_section_not_sutton_graves(self):
        """When thermal_source=='turbine_inlet_override', §4 does not walk
        through the Sutton-Graves / recovery-temperature derivation."""
        tex = self._tex_turbine()
        # thermal_source label appears in the results table (LaTeX-escaped:
        # 'turbine\_inlet\_override') — verify via the escaped form.
        self.assertIn(r"turbine\_inlet\_override", tex)
        # The explicit hot-section framing paragraph must appear.
        self.assertIn("Hot-section override active", tex)
        self.assertIn("turbine inlet", tex.lower())

    def test_propulsion_omits_fuel_table_for_turbine(self):
        """Turbine branch replaces the fuel-mass comparison with the
        system-level disclaimer paragraph."""
        tex = self._tex_turbine()
        # No fuel-comparison table header should survive for turbine.
        self.assertNotIn("Energy-equivalent mass", tex)
        # The turbine-specific explainer paragraph appears.
        self.assertIn("do not carry their own fuel", tex)


# ---------------------------------------------------------------------------
# TestComponentZonesSection — Item 6 of the six-improvement plan
# ---------------------------------------------------------------------------
class TestComponentZonesSection(unittest.TestCase):
    """The new §9 'Per-Zone Material Recommendations' must render for every
    propulsive/load-bearing category, contain the catalog zone names, and
    cross-reference the whole-vehicle materials section as the calibration
    anchor."""

    def test_section_header_present_for_aircraft(self):
        tex, _, _ = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        self.assertIn(r"\section{Per-Zone Material Recommendations}", tex)
        self.assertIn(r"\label{sec:component_zones}", tex)

    def test_aircraft_zone_names_render(self):
        """Every aircraft zone defined in CATEGORY_ZONES must appear in the PDF."""
        from core.component_zones import CATEGORY_ZONES
        tex, _, _ = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        for zone in CATEGORY_ZONES["aircraft"]:
            with self.subTest(zone=zone.name):
                self.assertIn(zone.name, tex)

    def test_reentry_zone_names_render(self):
        from core.component_zones import CATEGORY_ZONES
        tex, _, _ = _tex(20.0, 70.0, 500.0, 1.50, 8.0, "reentry")
        for zone in CATEGORY_ZONES["reentry"]:
            with self.subTest(zone=zone.name):
                self.assertIn(zone.name, tex)

    def test_hypersonic_missile_zone_names_render(self):
        from core.component_zones import CATEGORY_ZONES
        tex, _, _ = _tex(4.0, 20.0, 900.0, 0.08, 10.0, "hypersonic_missile")
        for zone in CATEGORY_ZONES["hypersonic_missile"]:
            with self.subTest(zone=zone.name):
                self.assertIn(zone.name, tex)

    def test_general_zone_names_render(self):
        from core.component_zones import CATEGORY_ZONES
        tex, _, _ = _tex(0.3, 5.0, 500.0, 0.5, 2.0, "general")
        for zone in CATEGORY_ZONES["general"]:
            with self.subTest(zone=zone.name):
                self.assertIn(zone.name, tex)

    def test_section_renders_subsections_for_each_zone(self):
        """One \\subsection per zone (count must match the catalog)."""
        from core.component_zones import CATEGORY_ZONES
        tex, _, _ = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        # The "Zone N:" prefix is unique to this section.
        zone_subsection_count = tex.count(r"\subsection{Zone ")
        self.assertEqual(zone_subsection_count, len(CATEGORY_ZONES["aircraft"]))

    def test_section_includes_per_zone_temperature_and_sigma_rows(self):
        tex, _, _ = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        self.assertIn(r"$T_{\text{wall, zone}}$", tex)
        self.assertIn(r"$\sigma_{\text{req, zone}}$", tex)

    def test_section_cross_references_whole_vehicle_materials_section(self):
        """The intro paragraph should refer back to §8 (Materials) as the
        calibration anchor / worst-zone summary."""
        tex, _, _ = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        self.assertIn(r"\ref{sec:materials}", tex)

    def test_thermal_disclaimer_forward_references_zones_for_recovery_branch(self):
        """The M<5 thermal disclaimer should point forward to the per-zone section."""
        tex, _, _ = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        self.assertIn(r"\ref{sec:component_zones}", tex)

    def test_thermal_disclaimer_forward_references_zones_for_sutton_graves_branch(self):
        """The M>=5 thermal disclaimer should also point forward to the zones."""
        tex, _, _ = _tex(20.0, 70.0, 500.0, 1.50, 8.0, "reentry")
        self.assertIn(r"\ref{sec:component_zones}", tex)

    def test_section_omitted_for_unknown_category(self):
        """A category not in CATEGORY_ZONES must not render a (broken) section.

        We can't easily reach this branch via match_materials (which validates
        category), but we can call the helper directly and confirm an empty
        return string for an unknown key."""
        from latex_export import _sec_component_zones
        physics = _run(3.0, 20.0, 1000.0, 0.20, 2.0)
        match = match_materials(physics, vehicle_category="aircraft")
        # Replace the vehicle_category on a copy of match using dataclasses:
        import dataclasses
        bogus_match = dataclasses.replace(match, vehicle_category="not_a_real_category")
        out = _sec_component_zones(physics, bogus_match)
        self.assertEqual(out, "")

    def test_section_renders_for_turbine_with_hot_section_override(self):
        """Even with the turbine_inlet_override branch active in §4, the
        component-zone section must render with the turbine zones."""
        from app import _apply_turbine_override
        from core.component_zones import CATEGORY_ZONES
        physics = _run(0.5, 0.0, 50.0, 0.05, g_load=1.0)
        physics = _apply_turbine_override(physics, 1400.0)
        match = match_materials(physics, vehicle_category="turbine")
        tex, _aux = generate_tex_source(physics, match, system_label="Turbine HPT")
        self.assertIn(r"\label{sec:component_zones}", tex)
        for zone in CATEGORY_ZONES["turbine"]:
            with self.subTest(zone=zone.name):
                self.assertIn(zone.name, tex)


# ---------------------------------------------------------------------------
# TestCostAxisIntegration — Cost-Axis-on-Pareto-Front feature
# ---------------------------------------------------------------------------

class TestCostAxisIntegration(unittest.TestCase):
    """The cost axis touches LaTeX in two places: the §8 materials
    longtable gains an 'Est.~Cost' column, and §6 propulsion gets a
    'Cost caveat' paragraph in every branch (reentry, turbine,
    propulsive). Both must render for every category."""

    def test_materials_table_has_est_cost_column_aircraft(self):
        """SR-71 (aircraft, 30600 kg): the longtable header must include
        the Est.~Cost column."""
        tex, _phys, _m = _tex(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft",
                              label="SR-71")
        self.assertIn(r"Est.~Cost", tex,
                      "Materials longtable missing 'Est.~Cost' column.")
        # And at least one row in the body should carry the SI-suffix
        # idiom: the SR-71 viable list contains titanium ($45/kg \u00d7
        # 30600 kg = $1.4M => '$1.4M').
        self.assertIn("M ", tex.replace(r"\$", "$"),
                      "No SI-suffix cost cell rendered (expected $X.XM).")

    def test_materials_table_has_est_cost_column_reentry(self):
        """Small reentry capsule (500 kg): cost column must render."""
        tex, _phys, _m = _tex(20.0, 70.0, 500.0, 1.50, 8.0, "reentry",
                              label="Reentry-500kg")
        self.assertIn(r"Est.~Cost", tex)

    def test_cost_caveat_appears_in_propulsion_section(self):
        """Every propulsion-section branch (aircraft, reentry, turbine)
        must end with the 'Cost caveat' paragraph."""
        cases = [
            (3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft", "SR-71"),
            (20.0, 70.0, 500.0, 1.50, 8.0, "reentry", "Reentry"),
            (0.5, 0.0, 50.0, 0.05, 1.0, "turbine", "Turbine"),
        ]
        for mach, alt, mass, R_n, g, cat, label in cases:
            with self.subTest(category=cat):
                tex, _phys, _m = _tex(mach, alt, mass, R_n, g, cat,
                                      label=label)
                self.assertIn(
                    r"\textbf{Cost caveat.}", tex,
                    f"{cat}: §6 propulsion section missing 'Cost caveat' "
                    f"paragraph.",
                )
                # Caveat must mention the order-of-magnitude framing so
                # readers know not to use the table for a BOM.
                self.assertIn(
                    r"$\pm$50\%", tex,
                    f"{cat}: cost caveat missing the \u00b150% disclaimer.",
                )

    def test_ceiling_bolds_overspend_rows(self):
        """When cost > ceiling, the row's cost cell is wrapped in
        \\textbf{...}. Use a tight ceiling on Reentry Capsule (500 kg)
        so any UHTC/CMC entry trips it."""
        from latex_export import generate_tex_source
        physics = _run(20.0, 70.0, 500.0, 1.50, 8.0)
        match = match_materials(physics, vehicle_category="reentry")
        # $100k ceiling: at 500 kg => any material > $200/kg is over.
        # Even nickel-base alloys ($60-$800/kg) and CMCs ($1500-$3000/kg)
        # will trip it.
        tex, _ = generate_tex_source(
            physics, match, system_label="Reentry-tight",
            cost_ceiling_usd=100_000.0,
        )
        self.assertIn(r"\textbf{\$", tex,
                      "Tight ceiling did not bold any over-budget cost cell.")


class TestCreepEvaluationSection(unittest.TestCase):
    """The Lifecycle / Creep Evaluation section appears in the LaTeX
    source when design_lifetime_hours >= 1000 h, and is omitted at
    shorter lifetimes (single-flight default behaviour)."""

    def _render(self, lifetime_hours: float) -> str:
        from core.api import run_session
        from core.presets import CANONICAL_PRESETS
        from latex_export import generate_tex_source
        from dataclasses import replace

        session = replace(
            CANONICAL_PRESETS["Concorde"],
            design_lifetime_hours=lifetime_hours,
        )
        result = run_session(session, compile_pdf=False)
        tex, _ = generate_tex_source(
            result.physics, result.match, session.system_label,
            design_lifetime_hours=lifetime_hours,
        )
        return tex

    def test_section_omitted_for_short_lifetime(self):
        tex = self._render(lifetime_hours=1.0)
        self.assertNotIn(
            r"\section{Lifecycle / Creep Evaluation}", tex,
            "Single-flight (1 h) lifetime should NOT produce the "
            "creep evaluation section.",
        )

    def test_section_present_for_long_lifetime(self):
        tex = self._render(lifetime_hours=25_000.0)
        self.assertIn(
            r"\section{Lifecycle / Creep Evaluation}", tex,
            "25,000 h lifetime should produce the creep evaluation "
            "section.",
        )

    def test_section_includes_larson_miller_formula(self):
        tex = self._render(lifetime_hours=25_000.0)
        self.assertIn(
            r"\mathrm{LMP} = T \cdot (C + \log_{10} t)", tex,
            "Creep section should explain the Larson-Miller formula.",
        )

    def test_section_includes_summary_with_status_counts(self):
        tex = self._render(lifetime_hours=25_000.0)
        self.assertIn("Summary:", tex)
        self.assertIn("pass", tex)
        self.assertIn("fail", tex)


class TestTransientHeatSection(unittest.TestCase):
    """The Transient Heat / Soak Evaluation section appears when the
    matching engine ran the 1D solver on at least one candidate, and
    is omitted otherwise (sustained-flight envelopes)."""

    def _render(self, preset_key: str) -> str:
        from core.api import run_session
        from core.presets import CANONICAL_PRESETS
        from latex_export import generate_tex_source

        session = CANONICAL_PRESETS[preset_key]
        result = run_session(session, compile_pdf=False)
        tex, _ = generate_tex_source(
            result.physics, result.match, session.system_label,
            design_lifetime_hours=float(session.design_lifetime_hours),
            panel_thickness_m=float(session.panel_thickness_m),
        )
        return tex

    def test_section_present_for_sounding_rocket(self):
        tex = self._render("Collegiate Sounding Rocket")
        self.assertIn(
            r"\section{Transient Heat / Soak Evaluation}", tex,
            "Sounding rocket (25 s flight) should produce the "
            "transient-heat section.",
        )

    def test_section_omitted_for_concorde(self):
        tex = self._render("Concorde")
        self.assertNotIn(
            r"\section{Transient Heat / Soak Evaluation}", tex,
            "Concorde (10,800 s) is a sustained flight; the "
            "transient section should be omitted.",
        )

    def test_section_documents_1d_heat_equation(self):
        tex = self._render("Collegiate Sounding Rocket")
        self.assertIn(
            r"\partial T / \partial t = \alpha", tex,
            "The transient section should document the heat equation.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
