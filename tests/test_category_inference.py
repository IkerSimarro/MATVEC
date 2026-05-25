"""
Tests for core.category_inference.infer_category.

The boundary table is the contract: every preset envelope in
CANONICAL_PRESETS has a deterministic inferred category, and a small
set of deliberate boundary points pin the decision-tree edges so a
future tweak to the heuristic surfaces here instead of as a silent
behaviour drift in the Streamlit UI.
"""

import unittest

from core.category_inference import VALID_CATEGORIES, infer_category


class TestInferCategoryPresets(unittest.TestCase):
    """Every canonical preset's envelope has a deterministic verdict."""

    PRESET_BOUNDARIES = [
        # (label, mach, alt_km, mass_kg, expected)
        # Note: X-15's stored category is "aircraft" for historical
        # schema reasons, but the inference correctly identifies it as
        # hypersonic_aircraft per the category descriptions. This
        # divergence is intentional — preset loading overrides the
        # inferred default, but a user typing X-15-like envelope by
        # hand gets the better answer.
        ("SR-71 Blackbird",            3.2, 25.0, 30600.0, "aircraft"),
        ("X-15",                       6.7, 30.0, 15195.0, "hypersonic_aircraft"),
        ("Generic Hypersonic Aircraft", 6.0, 30.0, 12000.0, "hypersonic_aircraft"),
        ("Mach 4 Tactical Missile",    4.0, 20.0,   900.0, "hypersonic_missile"),
        ("Supersonic Cruise",          2.5, 18.0, 45000.0, "aircraft"),
        ("Small Reentry Capsule",     20.0, 70.0,   500.0, "reentry"),
        ("Turbine HPT Blade",          0.5,  0.0,    50.0, "general"),  # never turbine
        ("General Structure Panel",    0.3,  5.0,   500.0, "general"),
    ]

    def test_each_preset_envelope_infers_expected(self):
        for label, mach, alt, mass, expected in self.PRESET_BOUNDARIES:
            with self.subTest(preset=label):
                got = infer_category(mach, alt, mass)
                self.assertEqual(
                    got, expected,
                    f"{label} ({mach=}, {alt=}, {mass=}): expected "
                    f"{expected!r}, got {got!r}",
                )


class TestInferCategoryDecisionEdges(unittest.TestCase):
    """Pin the decision-tree boundaries so tweaks surface here, not in UI."""

    EDGE_CASES = [
        # Reentry boundary on altitude (use mass>=3000 to actually
        # exercise the alt boundary — at mass<3000 the M>=5 missile fork
        # would fire first and mask the alt boundary).
        ("alt just below 60 km",       6.0, 59.99, 5000.0, "hypersonic_aircraft"),
        ("alt exactly 60 km",          0.5, 60.00,  500.0, "reentry"),
        ("alt above 60 km regardless of M", 0.5, 75.00, 500.0, "reentry"),

        # Reentry boundary on Mach (mass>=3000 so we actually test the
        # M>=12 boundary rather than the M>=5/mass<3000 missile fork).
        ("Mach 11.99 at low alt",     11.99, 30.0, 5000.0, "hypersonic_aircraft"),
        ("Mach 12 at low alt",        12.00, 30.0, 5000.0, "reentry"),
        # Mach 12 also catches small-mass craft (the alt OR Mach guard
        # fires before the M>=5 mass split).
        ("Mach 12 small mass",        12.00, 30.0,  500.0, "reentry"),

        # Hypersonic mass split
        ("Mach 5, mass 2999",          5.0, 25.0, 2999.0, "hypersonic_missile"),
        ("Mach 5, mass 3000",          5.0, 25.0, 3000.0, "hypersonic_aircraft"),

        # Supersonic-missile fork
        ("Mach 2, mass 2999",          2.0, 20.0, 2999.0, "hypersonic_missile"),
        ("Mach 2, mass 3000",          2.0, 20.0, 3000.0, "aircraft"),
        ("Mach 1.99, mass 500",        1.99, 20.0,  500.0, "general"),  # below M2 fork; mass<1000 fails aircraft fork

        # Aircraft fork
        ("Mach 0.4, mass 1000",        0.4,  5.0, 1000.0, "aircraft"),
        ("Mach 0.4, mass 999",         0.4,  5.0,  999.0, "general"),
        ("Mach 0.39, mass 5000",       0.39, 5.0, 5000.0, "general"),

        # Default
        ("subsonic small",             0.3,  2.0,   100.0, "general"),
    ]

    def test_decision_tree_edges(self):
        for label, mach, alt, mass, expected in self.EDGE_CASES:
            with self.subTest(case=label):
                got = infer_category(mach, alt, mass)
                self.assertEqual(
                    got, expected,
                    f"{label} ({mach=}, {alt=}, {mass=}): expected "
                    f"{expected!r}, got {got!r}",
                )


class TestInferCategoryNeverReturnsTurbine(unittest.TestCase):
    """Turbine is the user-override-only category. Across the realistic
    envelope grid, infer_category must never return "turbine"."""

    def test_grid_sweep_excludes_turbine(self):
        # Coarse but exhaustive grid covering subsonic → orbital re-entry
        # at masses from a panel (50 kg) to a heavy lifter (500,000 kg).
        for mach in (0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.2, 5.0, 7.0,
                     10.0, 15.0, 20.0, 25.0):
            for alt_km in (0.0, 5.0, 10.0, 18.0, 25.0, 40.0, 60.0, 80.0):
                for mass_kg in (50.0, 500.0, 1000.0, 5000.0, 30000.0,
                                100000.0, 500000.0):
                    cat = infer_category(mach, alt_km, mass_kg)
                    self.assertNotEqual(
                        cat, "turbine",
                        f"infer_category({mach=}, {alt_km=}, {mass_kg=}) "
                        f"returned turbine; turbine must be user-override only.",
                    )

    def test_returns_only_valid_categories(self):
        for mach in (0.05, 0.5, 2.0, 5.0, 12.0, 25.0):
            for alt in (0.0, 30.0, 60.0, 86.0):
                for mass in (10.0, 1000.0, 1_000_000.0):
                    cat = infer_category(mach, alt, mass)
                    self.assertIn(
                        cat, VALID_CATEGORIES,
                        f"infer_category returned out-of-domain {cat!r}",
                    )


class TestInferCategoryPureness(unittest.TestCase):
    """Pure function — no side effects, no module state mutation."""

    def test_repeated_calls_are_identical(self):
        for _ in range(5):
            self.assertEqual(infer_category(3.2, 25.0, 30600.0), "aircraft")

    def test_accepts_int_inputs(self):
        # Streamlit number_input may pass ints when step is integer-typed.
        self.assertEqual(infer_category(3, 25, 30000), "aircraft")


if __name__ == "__main__":
    unittest.main()
