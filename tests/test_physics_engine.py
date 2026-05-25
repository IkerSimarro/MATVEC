"""
Test suite for physics_engine.py — Step 2 of MATVEC.
stdlib unittest only. Run: python -m unittest test_physics_engine.py -v
"""

import math
import unittest

from core.physics_engine import (
    _isa_atmosphere,
    _sutton_graves,
    _tauber_sutton_rad,
    _radiation_equilibrium_wall_temperature,
    _classify_regime,
    run_analysis,
    SIGMA_SB,
)


# ---------------------------------------------------------------------------
# 1. ISA Atmosphere
# ---------------------------------------------------------------------------
class TestISAAtmosphere(unittest.TestCase):
    """Verify temperature and pressure at 7 known ISA check-points."""

    # (alt_km, expected_T_K, expected_P_Pa)
    CHECKPOINTS = [
        (0.0,  288.15, 101325.0),
        (11.0, 216.65,  22632.0),
        (20.0, 216.65,   5475.0),
        (32.0, 228.65,    868.0),
        (47.0, 270.65,    110.9),
        (51.0, 270.65,     66.9),
        (71.0, 214.65,      3.96),
    ]

    def test_temperature_at_checkpoints(self):
        for alt, T_expected, _ in self.CHECKPOINTS:
            with self.subTest(alt=alt):
                T, P, rho = _isa_atmosphere(alt)
                self.assertAlmostEqual(T, T_expected, delta=0.1,
                    msg=f"T at {alt} km: expected {T_expected} K, got {T:.3f} K")

    def test_pressure_at_checkpoints(self):
        for alt, _, P_expected in self.CHECKPOINTS:
            with self.subTest(alt=alt):
                T, P, rho = _isa_atmosphere(alt)
                rel_err = abs(P - P_expected) / P_expected
                self.assertLess(rel_err, 0.01,
                    msg=f"P at {alt} km: expected ~{P_expected} Pa, got {P:.2f} Pa "
                        f"(rel error {rel_err:.4f})")

    def test_density_positive_at_all_checkpoints(self):
        for alt, _, _ in self.CHECKPOINTS:
            T, P, rho = _isa_atmosphere(alt)
            self.assertGreater(rho, 0.0)

    def test_clamp_below_zero(self):
        T_neg, P_neg, rho_neg = _isa_atmosphere(-1.0)
        T_sl, P_sl, rho_sl = _isa_atmosphere(0.0)
        self.assertAlmostEqual(T_neg, T_sl, delta=1e-9)
        self.assertAlmostEqual(P_neg, P_sl, delta=1e-6)

    def test_clamp_above_86(self):
        T_hi, P_hi, rho_hi = _isa_atmosphere(90.0)
        T_86, P_86, rho_86 = _isa_atmosphere(86.0)
        self.assertAlmostEqual(T_hi, T_86, delta=1e-9)
        self.assertAlmostEqual(P_hi, P_86, delta=1e-6)

    def test_density_ideal_gas_consistency(self):
        """ρ = P / (R_air × T) should hold at every layer."""
        R_AIR = 287.058
        for alt, _, _ in self.CHECKPOINTS:
            T, P, rho = _isa_atmosphere(alt)
            rho_check = P / (R_AIR * T)
            self.assertAlmostEqual(rho, rho_check, delta=rho * 1e-9)


# ---------------------------------------------------------------------------
# 2. Sutton-Graves Heat Flux
# ---------------------------------------------------------------------------
class TestSuttonGravesHeatFlux(unittest.TestCase):
    """Unit tests for the _sutton_graves scaling law."""

    def test_known_value(self):
        # rho=1.0, V=1000 m/s, R_n=1.0: q = 1.7415e-4 × 1 × 1e9 = 174150 W/m²
        q = _sutton_graves(1.0, 1000.0, 1.0)
        self.assertAlmostEqual(q, 174150.0, delta=1.0)

    def test_v_cubed_scaling(self):
        """Doubling V should produce 8× flux."""
        q1 = _sutton_graves(1.0, 1000.0, 1.0)
        q2 = _sutton_graves(1.0, 2000.0, 1.0)
        self.assertAlmostEqual(q2 / q1, 8.0, places=6)

    def test_Rn_inverse_sqrt_scaling(self):
        """Quadrupling R_n should produce 0.5× flux."""
        q1 = _sutton_graves(1.0, 1000.0, 1.0)
        q2 = _sutton_graves(1.0, 1000.0, 4.0)
        self.assertAlmostEqual(q2 / q1, 0.5, places=6)

    def test_zero_velocity(self):
        self.assertEqual(_sutton_graves(1.0, 0.0, 1.0), 0.0)

    def test_zero_density(self):
        self.assertEqual(_sutton_graves(0.0, 1000.0, 1.0), 0.0)

    def test_positive_flux_for_positive_inputs(self):
        q = _sutton_graves(0.5, 3000.0, 0.1)
        self.assertGreater(q, 0.0)


# ---------------------------------------------------------------------------
# 3. Wall Temperature
# ---------------------------------------------------------------------------
class TestWallTemperature(unittest.TestCase):
    """Unit tests for radiation-equilibrium wall temperature."""

    def test_round_trip(self):
        """Compute q from T=1000K then recover T from q."""
        epsilon = 0.85
        T_known = 1000.0
        q = epsilon * SIGMA_SB * T_known ** 4
        T_calc = _radiation_equilibrium_wall_temperature(q, epsilon)
        self.assertAlmostEqual(T_calc, T_known, delta=0.01)

    def test_zero_input_returns_zero(self):
        self.assertEqual(_radiation_equilibrium_wall_temperature(0.0, 0.85), 0.0)

    def test_negative_q_returns_zero(self):
        self.assertEqual(_radiation_equilibrium_wall_temperature(-100.0, 0.85), 0.0)

    def test_bounds_ordering(self):
        """T_wall_max >= T_wall >= T_wall_min for any valid run.
        Uses M=9.6 (hypersonic) where Sutton-Graves is valid and T_recovery >> T_wall,
        so the recovery temperature cap does not apply and the ±20% uncertainty
        produces strictly ordered bounds.
        """
        result = run_analysis(9.6, 33.5, 1400.0, 0.05, peak_g_load=5.0)
        self.assertGreater(result.thermal.T_wall_max_K, result.thermal.T_wall_K)
        self.assertGreater(result.thermal.T_wall_K, result.thermal.T_wall_min_K)

    def test_uses_recovery_model_below_mach5(self):
        """M=1.8 is below the Mach 5 threshold — must use recovery temperature model."""
        result = run_analysis(1.8, 12.0, 19700.0, 0.30, peak_g_load=9.0)
        self.assertTrue(result.thermal.uses_recovery_model)
        self.assertGreater(result.thermal.T_wall_max_K, result.thermal.T_wall_K)
        # Heat fluxes must be zero in recovery model
        self.assertEqual(result.thermal.q_conv_Wm2, 0.0)
        self.assertEqual(result.thermal.q_total_Wm2, 0.0)
        # SG uncapped is sentinel 0.0
        self.assertEqual(result.thermal.T_wall_SG_uncapped_K, 0.0)

    def test_uses_recovery_model_sr71(self):
        """SR-71 (M=3.2, 25 km) must use recovery model with calibrated T_wall."""
        result = run_analysis(3.2, 25.0, 77000.0, 0.15, peak_g_load=2.5)
        self.assertTrue(result.thermal.uses_recovery_model)
        # SR-71 recovery temp calibration: ~607–620 K
        self.assertGreater(result.thermal.T_wall_K, 580.0)
        self.assertLess(result.thermal.T_wall_K, 650.0)

    def test_uses_sg_model_x15(self):
        """X-15 (M=6.7) is above Mach 5 threshold — must use SG model."""
        result = run_analysis(6.7, 30.0, 6804.0, 0.30, peak_g_load=5.0)
        self.assertFalse(result.thermal.uses_recovery_model)
        self.assertGreater(result.thermal.q_conv_Wm2, 0.0)

    def test_uses_sg_model_reentry(self):
        """Reentry vehicle (M=20) must use SG model."""
        result = run_analysis(20.0, 70.0, 5800.0, 0.30)
        self.assertFalse(result.thermal.uses_recovery_model)

    def test_recovery_model_uncertainty_bounds(self):
        """Recovery model uses ±15% uncertainty bounds."""
        result = run_analysis(2.5, 18.0, 186000.0, 0.20, peak_g_load=1.5)
        th = result.thermal
        self.assertTrue(th.uses_recovery_model)
        self.assertAlmostEqual(th.T_wall_min_K, th.T_wall_K * 0.85, delta=1.0)
        self.assertAlmostEqual(th.T_wall_max_K, th.T_wall_K * 1.15, delta=1.0)


# ---------------------------------------------------------------------------
# 4. Reference Vehicles
# ---------------------------------------------------------------------------
class TestReferenceVehicles(unittest.TestCase):
    """
    Physical ordering and bounds tests on four reference vehicles.
    Note: Sutton-Graves is a hypersonic formula that overestimates at M < 5.
    F-22 test checks bounds and ordering only, not absolute spec targets.
    """

    def test_shuttle_proxy(self):
        # M=25, 70 km, 100000 kg, R_n=0.3 m
        r = run_analysis(25.0, 70.0, 100_000.0, 0.3, peak_g_load=3.0)
        self.assertGreater(r.thermal.q_conv_Wm2, 500_000.0,
            "Shuttle q_conv should exceed 500 kW/m²")
        self.assertGreater(r.thermal.T_wall_K, 2100.0,
            "Shuttle T_wall should exceed 2100 K")
        self.assertTrue(r.thermal.plasma_sheath,
            "Shuttle should have plasma sheath at M=25, 70 km")
        # M=25, alt=70 km → not > 80 km, so regime is hypersonic (M=25 is on boundary)
        self.assertIn(r.flight_regime, ("hypersonic", "reentry"))

    def test_apollo_proxy(self):
        # M=36, 80 km, 5900 kg, R_n=4.7 m (blunt capsule)
        r_apollo = run_analysis(36.0, 80.0, 5900.0, 4.7, peak_g_load=5.0)
        r_shuttle = run_analysis(25.0, 70.0, 100_000.0, 0.3, peak_g_load=3.0)
        # Blunt Apollo body should have lower wall temperature than sharp Shuttle nose
        self.assertLess(r_apollo.thermal.T_wall_K, r_shuttle.thermal.T_wall_K,
            "Apollo (blunt, R_n=4.7) T_wall should be less than Shuttle (R_n=0.3)")
        self.assertEqual(r_apollo.flight_regime, "reentry",
            "Apollo M=36, 80 km → reentry")

    def test_x43a_proxy(self):
        # M=9.6, 33.5 km, 1400 kg, R_n=0.05 m (very sharp leading edge)
        r_x43 = run_analysis(9.6, 33.5, 1400.0, 0.05, peak_g_load=5.0)
        r_apollo = run_analysis(36.0, 80.0, 5900.0, 4.7, peak_g_load=5.0)
        # Sharp nose → higher local heat flux than blunt Apollo nose
        self.assertGreater(r_x43.thermal.q_conv_Wm2, r_apollo.thermal.q_conv_Wm2,
            "X-43A (sharp R_n=0.05) q_conv should exceed Apollo (blunt R_n=4.7)")
        self.assertGreater(r_x43.thermal.T_wall_K, 2000.0,
            "X-43A T_wall should exceed 2000 K")
        self.assertEqual(r_x43.flight_regime, "hypersonic")

    def test_f22_proxy(self):
        # M=1.8, 12 km, 19700 kg, R_n=0.3 m — supersonic fighter
        # Sutton-Graves overestimates at M<5; test bounds and ordering only.
        r = run_analysis(1.8, 12.0, 19_700.0, 0.3, peak_g_load=9.0)
        self.assertGreater(r.thermal.T_wall_K, r.thermal.T_ambient_K,
            "T_wall must exceed ambient (aero heating adds energy)")
        self.assertLess(r.thermal.T_wall_K, 1500.0,
            "F-22 T_wall should be below 1500 K (Sutton-Graves overestimates M<5)")
        self.assertFalse(r.thermal.plasma_sheath,
            "No plasma sheath at M=1.8")
        self.assertEqual(r.flight_regime, "supersonic")
        self.assertNotIn("Dynamic pressure exceeds known structural limits",
            r.warnings, "F-22 at 12 km should not trigger q_dyn warning")


# ---------------------------------------------------------------------------
# 5. Structural Branch
# ---------------------------------------------------------------------------
class TestStructuralBranch(unittest.TestCase):
    """Verify structural mechanics ordering and geometry."""

    def setUp(self):
        # F-22 proxy
        self.r = run_analysis(1.8, 12.0, 19_700.0, 0.3, peak_g_load=9.0)

    def test_combined_exceeds_inertial(self):
        s = self.r.structural
        self.assertGreater(s.sigma_combined_MPa, s.sigma_inertial_MPa)

    def test_tensile_required_exceeds_combined(self):
        s = self.r.structural
        self.assertGreater(s.sigma_tensile_required_MPa, s.sigma_combined_MPa)

    def test_q_dyn_below_warning_threshold(self):
        self.assertLess(self.r.structural.q_dyn_Pa, 200_000.0)

    def test_characteristic_length_positive(self):
        self.assertGreater(self.r.structural.characteristic_length_m, 0.0)

    def test_area_matches_characteristic_length(self):
        s = self.r.structural
        L = s.characteristic_length_m
        A_expected = math.pi * (L / 2.0) ** 2
        self.assertAlmostEqual(s.A_ref_m2, A_expected, places=10)

    def test_explicit_characteristic_length_respected(self):
        r = run_analysis(1.8, 12.0, 19_700.0, 0.3, characteristic_length_m=15.0)
        self.assertAlmostEqual(r.structural.characteristic_length_m, 15.0, places=10)
        self.assertAlmostEqual(r.characteristic_length_m, 15.0, places=10)


# ---------------------------------------------------------------------------
# 6. Input Validation
# ---------------------------------------------------------------------------
class TestInputValidation(unittest.TestCase):
    """Verify non-blocking warning generation for out-of-range inputs."""

    def test_high_dynamic_pressure_warning(self):
        # M=25 at sea level → enormous q_dyn
        r = run_analysis(25.0, 0.0, 50_000.0, 0.3)
        self.assertTrue(
            any("Dynamic pressure" in w for w in r.warnings),
            f"Expected dynamic pressure warning; got: {r.warnings}"
        )

    def test_mach_range_warning(self):
        r = run_analysis(30.0, 70.0, 1000.0, 0.3)
        self.assertTrue(
            any("Outside validated range" in w for w in r.warnings),
            f"Expected Mach range warning; got: {r.warnings}"
        )

    def test_altitude_warning(self):
        r = run_analysis(15.0, 90.0, 1000.0, 0.3)
        self.assertTrue(
            any("ISA model not valid" in w for w in r.warnings),
            f"Expected altitude warning; got: {r.warnings}"
        )

    def test_nose_radius_warning(self):
        r = run_analysis(5.0, 30.0, 1000.0, 0.0005)
        self.assertTrue(
            any("Below manufacturing minimum" in w for w in r.warnings),
            f"Expected nose radius warning; got: {r.warnings}"
        )

    def test_no_warnings_for_nominal_inputs(self):
        # F-22-like: benign flight conditions
        r = run_analysis(1.8, 12.0, 19_700.0, 0.3, peak_g_load=9.0)
        self.assertEqual(r.warnings, [],
            f"Expected no warnings for F-22-like inputs; got: {r.warnings}")

    def test_analysis_runs_despite_warnings(self):
        """Warnings must be non-blocking — result must still be returned."""
        r = run_analysis(30.0, 90.0, 1000.0, 0.0005)
        self.assertIsNotNone(r)
        self.assertGreater(len(r.warnings), 0)


# ---------------------------------------------------------------------------
# 7. Regime Classification
# ---------------------------------------------------------------------------
class TestRegimeClassification(unittest.TestCase):
    """Twelve boundary cases covering all regime transitions."""

    def _regime(self, mach, alt=10.0):
        return _classify_regime(mach, alt)

    def test_subsonic_below_0_8(self):
        self.assertEqual(self._regime(0.79), "subsonic")

    def test_supersonic_at_0_8(self):
        self.assertEqual(self._regime(0.8), "supersonic")

    def test_supersonic_at_4_9(self):
        self.assertEqual(self._regime(4.9), "supersonic")

    def test_hypersonic_at_5_0(self):
        self.assertEqual(self._regime(5.0), "hypersonic")

    def test_hypersonic_at_24_9(self):
        self.assertEqual(self._regime(24.9), "hypersonic")

    def test_hypersonic_at_25_0(self):
        # 25.0 is exactly the boundary; M > 25 triggers reentry, so 25.0 is hypersonic
        self.assertEqual(self._regime(25.0), "hypersonic")

    def test_reentry_above_25(self):
        self.assertEqual(self._regime(25.1), "reentry")

    def test_reentry_high_alt_and_high_mach(self):
        # alt > 80 AND mach > 15 → reentry
        self.assertEqual(_classify_regime(16.0, 81.0), "reentry")

    def test_not_reentry_alt_80_exact(self):
        # alt = 80 is not > 80; must be strictly greater
        self.assertEqual(_classify_regime(16.0, 80.0), "hypersonic")

    def test_not_reentry_mach_15_exact(self):
        # mach = 15 is not > 15; must be strictly greater
        self.assertEqual(_classify_regime(15.0, 81.0), "hypersonic")

    def test_hypersonic_just_below_mach_threshold(self):
        # alt > 80 but mach = 14.9 (not > 15) → hypersonic, not reentry
        self.assertEqual(_classify_regime(14.9, 81.0), "hypersonic")

    def test_subsonic_zero_mach(self):
        self.assertEqual(self._regime(0.0), "subsonic")

    def test_run_analysis_regime_matches_classifier(self):
        """run_analysis flight_regime must equal _classify_regime directly."""
        cases = [
            (0.5, 5.0), (1.8, 12.0), (9.6, 33.5), (25.0, 70.0), (36.0, 80.0),
        ]
        for mach, alt in cases:
            with self.subTest(mach=mach, alt=alt):
                r = run_analysis(mach, alt, 5000.0, 0.3)
                expected = _classify_regime(mach, alt)
                self.assertEqual(r.flight_regime, expected)


# ---------------------------------------------------------------------------
# 8. Output Completeness
# ---------------------------------------------------------------------------
class TestOutputCompleteness(unittest.TestCase):
    """Every field of every nested dataclass must be non-None and typed correctly."""

    def setUp(self):
        self.r = run_analysis(9.6, 33.5, 1400.0, 0.05, peak_g_load=5.0)

    def _assert_float(self, val, name):
        self.assertIsNotNone(val, msg=f"{name} is None")
        self.assertIsInstance(val, float, msg=f"{name} is {type(val)}, expected float")
        self.assertFalse(math.isnan(val), msg=f"{name} is NaN")
        self.assertFalse(math.isinf(val), msg=f"{name} is Inf")

    def test_top_level_echo_fields(self):
        r = self.r
        self._assert_float(r.peak_mach, "peak_mach")
        self._assert_float(r.cruise_altitude_km, "cruise_altitude_km")
        self._assert_float(r.vehicle_mass_kg, "vehicle_mass_kg")
        self._assert_float(r.nose_radius_m, "nose_radius_m")
        self._assert_float(r.peak_g_load, "peak_g_load")
        self._assert_float(r.characteristic_length_m, "characteristic_length_m")
        self._assert_float(r.flight_duration_s, "flight_duration_s")
        self._assert_float(r.wall_emissivity, "wall_emissivity")

    def test_atmosphere_fields(self):
        a = self.r.atmosphere
        self._assert_float(a.altitude_km, "atmosphere.altitude_km")
        self._assert_float(a.temperature_K, "atmosphere.temperature_K")
        self._assert_float(a.pressure_Pa, "atmosphere.pressure_Pa")
        self._assert_float(a.density_kgm3, "atmosphere.density_kgm3")

    def test_thermal_fields(self):
        t = self.r.thermal
        float_fields = [
            "velocity_ms", "q_conv_Wm2", "q_rad_Wm2", "q_total_Wm2",
            "T_wall_K", "T_wall_min_K", "T_wall_max_K",
            "T_stag_K", "T_ambient_K",
            "q_total_sealevel_Wm2", "T_wall_sealevel_K",
        ]
        for f in float_fields:
            self._assert_float(getattr(t, f), f"thermal.{f}")
        self.assertIsInstance(t.plasma_sheath, bool)

    def test_structural_fields(self):
        s = self.r.structural
        float_fields = [
            "F_inertial_N", "q_dyn_Pa", "A_ref_m2",
            "sigma_inertial_MPa", "sigma_combined_MPa",
            "sigma_thermal_ref_MPa", "sigma_tensile_required_MPa",
            "characteristic_length_m",
        ]
        for f in float_fields:
            self._assert_float(getattr(s, f), f"structural.{f}")

    def test_propulsion_fields(self):
        p = self.r.propulsion
        float_fields = [
            "KE_J", "P_peak_W", "E_total_J",
            "fuel_mass_kerosene_kg", "fuel_mass_LH2_kg",
        ]
        for f in float_fields:
            self._assert_float(getattr(p, f), f"propulsion.{f}")

    def test_em_fields(self):
        e = self.r.em
        self._assert_float(e.P_rad_W, "em.P_rad_W")
        self._assert_float(e.lambda_peak_um, "em.lambda_peak_um")
        self.assertIsInstance(e.emission_band, str)
        self.assertIn(e.emission_band, ("near-IR", "mid-wave IR", "long-wave IR", "far-IR"))
        self.assertIsInstance(e.plasma_sheath, bool)

    def test_warnings_is_list(self):
        self.assertIsInstance(self.r.warnings, list)

    def test_flight_regime_is_string(self):
        self.assertIsInstance(self.r.flight_regime, str)
        self.assertIn(self.r.flight_regime, ("subsonic", "supersonic", "hypersonic", "reentry"))

    def test_nested_objects_not_none(self):
        r = self.r
        self.assertIsNotNone(r.atmosphere)
        self.assertIsNotNone(r.thermal)
        self.assertIsNotNone(r.structural)
        self.assertIsNotNone(r.propulsion)
        self.assertIsNotNone(r.em)


class TestPlasmaSheathTwoTier(unittest.TestCase):
    """Two-tier plasma-sheath threshold.

    Blunt bodies (R_n >= 0.5 m) use the classic Mach > 10 threshold.
    Slender bodies (R_n < 0.5 m) trip plasma at Mach > 6 to capture the
    Mach 6-8 attenuation effects documented on the X-15 (radio blackout
    and UV glow at M ~ 6.7).
    """

    def test_x15_slender_body_trips_plasma_at_mach_6_7(self):
        """X-15 (R_n=0.3 m, M=6.7) is slender and should register plasma."""
        r = run_analysis(6.7, 30.0, 15195.0, 0.30, peak_g_load=5.0)
        self.assertTrue(r.thermal.plasma_sheath,
                        "X-15 at Mach 6.7 should trip slender-body plasma threshold")
        self.assertTrue(r.thermal.plasma_threshold_slender,
                        "Slender-body flag should be set for R_n < 0.5")
        # EM dataclass mirrors the thermal flag.
        self.assertTrue(r.em.plasma_sheath)
        self.assertTrue(r.em.plasma_threshold_slender)

    def test_blunt_body_mach_8_no_plasma(self):
        """Blunt body (R_n=1.5 m) at Mach 8 must not trip plasma (needs M > 10)."""
        r = run_analysis(8.0, 40.0, 5000.0, 1.50, peak_g_load=5.0)
        self.assertFalse(r.thermal.plasma_sheath,
                         "Blunt body at Mach 8 should not trip plasma")
        self.assertFalse(r.thermal.plasma_threshold_slender)

    def test_reentry_capsule_mach_20_plasma(self):
        """Reentry capsule (R_n=1.5 m, M=20) trips blunt-body plasma."""
        r = run_analysis(20.0, 70.0, 500.0, 1.50, peak_g_load=8.0)
        self.assertTrue(r.thermal.plasma_sheath,
                        "Reentry capsule at Mach 20 should trip plasma")
        self.assertFalse(r.thermal.plasma_threshold_slender,
                         "Blunt body (R_n >= 0.5) should not use slender threshold")

    def test_subsonic_no_plasma(self):
        """Subsonic flight does not trip plasma regardless of nose radius."""
        r = run_analysis(0.8, 10.0, 50000.0, 0.3, peak_g_load=2.0)
        self.assertFalse(r.thermal.plasma_sheath)

    def test_slender_body_mach_5_below_threshold(self):
        """Slender body (R_n=0.1 m) at Mach 5 is below slender threshold (>6)."""
        r = run_analysis(5.0, 25.0, 1000.0, 0.10, peak_g_load=5.0)
        self.assertFalse(r.thermal.plasma_sheath,
                         "Slender body at Mach 5 is below the 6.0 threshold")


class TestThermalSourceField(unittest.TestCase):
    """ThermalResults carries a thermal_source label for downstream LaTeX export."""

    def test_aerodynamic_source_default(self):
        """Standard aerodynamic run reports thermal_source='aerodynamic'."""
        r = run_analysis(3.2, 25.0, 30600.0, 0.15, peak_g_load=2.5)
        self.assertEqual(r.thermal.thermal_source, "aerodynamic")


if __name__ == "__main__":
    unittest.main(verbosity=2)
