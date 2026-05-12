"""
Tests for core/creep.py — Larson-Miller creep evaluation.
"""

import math
import unittest

from materials_db import MATERIALS_DB
from core.creep import (
    CREEP_MARGIN_FRACTION,
    CREEP_NOT_APPLICABLE_T_FRAC,
    CreepEvaluation,
    creep_rupture_strength_MPa,
    evaluate_creep,
    larson_miller_parameter,
)


def _by_name(name: str):
    return next(m for m in MATERIALS_DB if m.name == name)


# ---------------------------------------------------------------------------
# Larson-Miller parameter formula
# ---------------------------------------------------------------------------
class TestLarsonMillerFormula(unittest.TestCase):
    """LMP = T * (C + log10(t)). Sanity points + edge handling."""

    def test_basic_round_trip(self):
        # 1000 K, 1000 h, C=20 -> LMP = 1000 * (20 + 3) = 23000
        self.assertAlmostEqual(
            larson_miller_parameter(1000, 1000, 20.0), 23000.0, places=3,
        )

    def test_one_hour_collapses_to_T_times_C(self):
        # log10(1) = 0 -> LMP = T * C
        self.assertAlmostEqual(
            larson_miller_parameter(800.0, 1.0, 20.0), 16000.0, places=3,
        )

    def test_higher_temp_gives_higher_lmp(self):
        a = larson_miller_parameter(900.0, 1000.0, 20.0)
        b = larson_miller_parameter(1000.0, 1000.0, 20.0)
        self.assertLess(a, b)

    def test_higher_time_gives_higher_lmp(self):
        a = larson_miller_parameter(900.0, 100.0, 20.0)
        b = larson_miller_parameter(900.0, 100000.0, 20.0)
        self.assertLess(a, b)

    def test_zero_or_negative_t_raises(self):
        with self.assertRaises(ValueError):
            larson_miller_parameter(900.0, 0.0, 20.0)
        with self.assertRaises(ValueError):
            larson_miller_parameter(900.0, -5.0, 20.0)

    def test_zero_or_negative_T_raises(self):
        with self.assertRaises(ValueError):
            larson_miller_parameter(0.0, 1000.0, 20.0)
        with self.assertRaises(ValueError):
            larson_miller_parameter(-100.0, 1000.0, 20.0)


# ---------------------------------------------------------------------------
# Rupture-stress lookup vs. published reference points
# ---------------------------------------------------------------------------
class TestRuptureStressLookup(unittest.TestCase):
    """Spot-check the curve lookup against the data points themselves —
    a sourced material's published (T, t, sigma) point must round-trip
    through the LMP machinery."""

    def test_inconel_718_at_650C_1000h_recovers_published_point(self):
        # Special Metals datasheet: 922 K (650 C) / 1000 h -> 620 MPa
        in718 = _by_name("Inconel 718")
        sigma, extrap = creep_rupture_strength_MPa(in718, 922.0, 1000.0)
        self.assertIsNotNone(sigma)
        self.assertFalse(extrap)
        # The exact data point should round-trip within the
        # interpolation tolerance.
        self.assertAlmostEqual(sigma, 620.0, delta=10.0)

    def test_cmsx4_at_1273_1000h_recovers_published_point(self):
        cmsx4 = _by_name("CMSX-4")
        sigma, extrap = creep_rupture_strength_MPa(cmsx4, 1273.0, 1000.0)
        self.assertIsNotNone(sigma)
        self.assertFalse(extrap)
        self.assertAlmostEqual(sigma, 190.0, delta=10.0)

    def test_ti_6al_4v_at_755_100h_recovers_published_point(self):
        ti = _by_name("Ti-6Al-4V")
        sigma, extrap = creep_rupture_strength_MPa(ti, 755.0, 100.0)
        self.assertIsNotNone(sigma)
        self.assertFalse(extrap)
        self.assertAlmostEqual(sigma, 480.0, delta=10.0)

    def test_al_2024_at_373_25000h_concorde_envelope(self):
        """Concorde validation lynchpin: 100 C * 25,000 h on Al 2024-T3
        must give a rupture stress well below typical airframe
        sigma_required (~117 MPa). Historically, this is exactly why
        Hiduminium RR58 / Al 2618 was developed."""
        al = _by_name("2024-T3")
        sigma, _extrap = creep_rupture_strength_MPa(al, 373.0, 25000.0)
        self.assertIsNotNone(sigma)
        self.assertLess(
            sigma, 100.0,
            f"Al 2024-T3 at Concorde envelope gave {sigma:.0f} MPa; "
            f"expected <100 MPa so the creep stage correctly rejects "
            f"it.",
        )

    def test_rupture_stress_decreases_with_time(self):
        in718 = _by_name("Inconel 718")
        s_short, _ = creep_rupture_strength_MPa(in718, 922.0, 100.0)
        s_long, _ = creep_rupture_strength_MPa(in718, 922.0, 10000.0)
        self.assertGreater(s_short, s_long)

    def test_rupture_stress_decreases_with_temperature(self):
        in718 = _by_name("Inconel 718")
        s_cool, _ = creep_rupture_strength_MPa(in718, 922.0, 1000.0)
        s_hot, _ = creep_rupture_strength_MPa(in718, 1033.0, 1000.0)
        self.assertGreater(s_cool, s_hot)

    def test_unknown_material_returns_none(self):
        # Pick any material with creep_data_status="unknown"
        unknown_mat = next(
            m for m in MATERIALS_DB if m.creep_data_status == "unknown"
        )
        sigma, extrap = creep_rupture_strength_MPa(
            unknown_mat, 800.0, 1000.0
        )
        self.assertIsNone(sigma)

    def test_not_applicable_material_returns_none(self):
        pica = _by_name("PICA")
        sigma, _ = creep_rupture_strength_MPa(pica, 1500.0, 100.0)
        self.assertIsNone(sigma)

    def test_above_curve_range_extrapolates(self):
        cmsx4 = _by_name("CMSX-4")
        # 1500 K * 100,000 h is well above the 1473 K * 100 h data.
        sigma, extrap = creep_rupture_strength_MPa(cmsx4, 1500.0, 100000.0)
        self.assertIsNotNone(sigma)
        self.assertTrue(extrap)
        # Extrapolated stress should be small but non-negative.
        self.assertGreaterEqual(sigma, 0.0)
        self.assertLess(sigma, 60.0)

    def test_below_curve_range_returns_first_point(self):
        cmsx4 = _by_name("CMSX-4")
        # 800 K * 1 h is well below the 1273 K * 100 h data.
        sigma, extrap = creep_rupture_strength_MPa(cmsx4, 800.0, 1.0)
        self.assertIsNotNone(sigma)
        self.assertTrue(extrap)
        # Cooler / shorter than the curve — should report the first
        # (highest-stress) data point.
        self.assertEqual(sigma, cmsx4.lmp_curve[0][1])


# ---------------------------------------------------------------------------
# evaluate_creep: full verdict logic
# ---------------------------------------------------------------------------
class TestEvaluateCreep(unittest.TestCase):
    def test_pass_when_rupture_well_above_required(self):
        in718 = _by_name("Inconel 718")
        # 922 K * 1000 h gives ~620 MPa. Required 100 MPa -> margin ~5.2.
        verdict = evaluate_creep(in718, 922.0, 1000.0, 100.0)
        self.assertEqual(verdict.status, "pass")
        self.assertGreater(verdict.margin_fraction, CREEP_MARGIN_FRACTION)
        self.assertIsNotNone(verdict.lmp_value)
        self.assertIn("Special Metals", verdict.data_source)

    def test_marginal_band(self):
        in718 = _by_name("Inconel 718")
        # Required ~ rupture stress * 0.9 -> margin ~0.11 (in marginal band).
        sigma_r, _ = creep_rupture_strength_MPa(in718, 922.0, 1000.0)
        verdict = evaluate_creep(in718, 922.0, 1000.0, sigma_r * 0.9)
        self.assertEqual(verdict.status, "marginal")
        self.assertGreaterEqual(verdict.margin_fraction, 0.0)
        self.assertLess(verdict.margin_fraction, CREEP_MARGIN_FRACTION)

    def test_fail_when_rupture_below_required(self):
        al = _by_name("2024-T3")
        # 100 C * 25000 h * required 200 MPa -> rupture ~80 -> fail.
        verdict = evaluate_creep(al, 373.0, 25000.0, 200.0)
        self.assertEqual(verdict.status, "fail")
        self.assertLess(verdict.margin_fraction, 0.0)

    def test_concorde_al2024_fails_creep(self):
        """Validation lynchpin: at Concorde envelope (100 C * 25000 h)
        with a typical airframe sigma_required (~117 MPa), Al 2024-T3
        must fail the creep stage."""
        al = _by_name("2024-T3")
        verdict = evaluate_creep(al, 373.0, 25000.0, 117.0)
        self.assertEqual(verdict.status, "fail")

    def test_cmsx4_passes_at_1273_1000h_typical_blade_load(self):
        """At 1000 C * 1000 h with a representative blade
        centrifugal stress of ~80 MPa, CMSX-4 must pass cleanly.
        (At 1100 C the same alloy goes marginal — CMSX-4's
        published rupture data shows it near its limit there,
        which is why real engines run it at 1000-1050 C for
        sustained operation, not 1100.)"""
        cmsx4 = _by_name("CMSX-4")
        verdict = evaluate_creep(cmsx4, 1273.0, 1000.0, 80.0)
        self.assertEqual(verdict.status, "pass")

    def test_short_lifetime_makes_aluminum_pass(self):
        """At single-flight lifetime (1 h), even Al 2024 at 100 C
        should pass typical airframe stresses easily — the entire
        point of the schema-default 1.0 h preserving pre-creep
        behaviour."""
        al = _by_name("2024-T3")
        verdict = evaluate_creep(al, 373.0, 1.0, 117.0)
        self.assertEqual(verdict.status, "pass")

    def test_tps_material_returns_not_applicable(self):
        avcoat = _by_name("AVCOAT")
        verdict = evaluate_creep(avcoat, 1500.0, 100.0, 50.0)
        self.assertEqual(verdict.status, "not_applicable")

    def test_polymer_composite_returns_not_applicable(self):
        cfrp = _by_name("IM7/977-3 CFRP")
        verdict = evaluate_creep(cfrp, 380.0, 1000.0, 50.0)
        self.assertEqual(verdict.status, "not_applicable")

    def test_unknown_material_below_creep_regime_returns_not_applicable(self):
        """A refractory metal (unknown LMP) at room temperature is
        far below 0.5 * Tm — should be flagged not_applicable, not
        unknown."""
        # Tungsten has Tm ~3680 K. At 300 K, T/Tm = 0.08 << 0.5.
        w = _by_name("Tungsten")
        if w.creep_data_status != "unknown":
            self.skipTest("Tungsten status changed — adjust this test")
        verdict = evaluate_creep(w, 300.0, 1000.0, 100.0)
        self.assertEqual(verdict.status, "not_applicable")
        self.assertIn("below", verdict.notes.lower())

    def test_unknown_material_in_creep_regime_returns_unknown(self):
        """A refractory metal (unknown LMP) ABOVE 0.5 * Tm — flag as
        unknown so the matching engine surfaces the gap."""
        w = _by_name("Tungsten")
        if w.creep_data_status != "unknown":
            self.skipTest("Tungsten status changed — adjust this test")
        # Tungsten Tm ~3680 K -> 0.5 * Tm ~ 1840 K. Use 2500 K.
        verdict = evaluate_creep(w, 2500.0, 1000.0, 100.0)
        self.assertEqual(verdict.status, "unknown")

    def test_estimated_status_carries_warning_note(self):
        """Materials marked status="estimated" must surface that
        provenance in the verdict's notes field."""
        # PWA 1484 is estimated in the Phase 1 data block.
        pwa = _by_name("PWA 1484")
        verdict = evaluate_creep(pwa, 1273.0, 1000.0, 100.0)
        self.assertIn("estimated", verdict.notes.lower())

    def test_extrapolated_lookup_carries_warning_note(self):
        cmsx4 = _by_name("CMSX-4")
        # Far above curve range -> extrapolation flag should appear in notes.
        verdict = evaluate_creep(cmsx4, 1600.0, 100000.0, 10.0)
        self.assertTrue(verdict.extrapolated)
        self.assertIn("extrapolat", verdict.notes.lower())

    def test_zero_required_stress_treats_as_pass(self):
        in718 = _by_name("Inconel 718")
        verdict = evaluate_creep(in718, 922.0, 1000.0, 0.0)
        self.assertEqual(verdict.status, "pass")
        self.assertEqual(verdict.margin_fraction, float("inf"))

    def test_negative_lifetime_raises(self):
        in718 = _by_name("Inconel 718")
        with self.assertRaises(ValueError):
            evaluate_creep(in718, 922.0, -10.0, 100.0)

    def test_zero_lifetime_raises(self):
        in718 = _by_name("Inconel 718")
        with self.assertRaises(ValueError):
            evaluate_creep(in718, 922.0, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Margin convention parity with matching_engine
# ---------------------------------------------------------------------------
class TestMarginConvention(unittest.TestCase):
    def test_creep_margin_fraction_constant_matches_structural(self):
        """The margin threshold for creep should match the structural
        threshold so the user's mental model stays consistent across
        stages."""
        from matching_engine import MARGINAL_STRUCTURAL_FRACTION
        self.assertEqual(CREEP_MARGIN_FRACTION, MARGINAL_STRUCTURAL_FRACTION)

    def test_creep_not_applicable_t_frac_is_sane(self):
        """0.5 * Tm is the canonical homologous-T threshold for
        creep relevance. Document any future change in this constant
        loudly."""
        self.assertEqual(CREEP_NOT_APPLICABLE_T_FRAC, 0.5)


# ---------------------------------------------------------------------------
# CreepEvaluation dataclass
# ---------------------------------------------------------------------------
class TestCreepEvaluationDataclass(unittest.TestCase):
    def test_default_construction_produces_minimal_object(self):
        cv = CreepEvaluation(status="unknown")
        self.assertEqual(cv.status, "unknown")
        self.assertIsNone(cv.rupture_stress_MPa)
        self.assertIsNone(cv.margin_fraction)
        self.assertEqual(cv.notes, "")
        self.assertFalse(cv.extrapolated)

    def test_full_construction_preserves_all_fields(self):
        cv = CreepEvaluation(
            status="pass",
            rupture_stress_MPa=500.0,
            margin_fraction=0.42,
            lmp_value=22000.0,
            data_source="MMPDS-17",
            notes="all good",
            extrapolated=False,
        )
        self.assertEqual(cv.status, "pass")
        self.assertEqual(cv.rupture_stress_MPa, 500.0)
        self.assertAlmostEqual(cv.margin_fraction, 0.42)
        self.assertEqual(cv.lmp_value, 22000.0)
        self.assertEqual(cv.data_source, "MMPDS-17")
        self.assertEqual(cv.notes, "all good")


if __name__ == "__main__":
    unittest.main(verbosity=2)
