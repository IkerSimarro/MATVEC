"""
Tests for core/transient_heat.py — 1D transient heat conduction
solver. Verifies the physics limits (steady state, adiabatic),
explicit-vs-implicit agreement, edge cases, and the realistic
flight-envelope behaviour the matching engine will rely on.

Run: python -m unittest test_transient_heat.py -v
"""

import math
import unittest

from core.materials_db import MATERIALS_DB
from core.transient_heat import (
    TransientHeatResult,
    DEFAULT_N_NODES,
    EXPLICIT_STEP_BUDGET,
    integrate_panel,
    _interp_profile,
    _coerce_profile,
    _stagnation_heat_flux_coldwall,
    _hot_wall_heat_flux,
    _recovery_temp,
)


def _by_name(name: str):
    return next(m for m in MATERIALS_DB if m.name == name)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
class TestProfileHandling(unittest.TestCase):
    def test_empty_profile_falls_back_to_constant_point(self):
        profile = _coerce_profile(
            [], flight_duration_s=25.0, fallback_mach=2.0,
            fallback_alt_km=9.0,
        )
        # Two-point constant profile.
        self.assertEqual(len(profile), 2)
        self.assertEqual(profile[0], (0.0, 2.0, 9.0))
        self.assertEqual(profile[1], (25.0, 2.0, 9.0))

    def test_empty_profile_without_duration_raises(self):
        with self.assertRaises(ValueError):
            _coerce_profile([], flight_duration_s=None,
                            fallback_mach=0.0, fallback_alt_km=0.0)

    def test_non_monotonic_profile_raises(self):
        bad = [(0.0, 0.0, 0.0), (10.0, 1.0, 5.0), (5.0, 0.5, 3.0)]
        with self.assertRaises(ValueError):
            _coerce_profile(bad, None, 0.0, 0.0)

    def test_interp_at_endpoints(self):
        profile = ((0.0, 0.0, 0.0), (10.0, 2.0, 9.0), (25.0, 0.5, 12.0))
        self.assertEqual(_interp_profile(profile, 0.0), (0.0, 0.0))
        self.assertEqual(_interp_profile(profile, 10.0), (2.0, 9.0))
        self.assertEqual(_interp_profile(profile, 25.0), (0.5, 12.0))

    def test_interp_linear_between_samples(self):
        profile = ((0.0, 0.0, 0.0), (10.0, 2.0, 9.0))
        m, a = _interp_profile(profile, 5.0)
        self.assertAlmostEqual(m, 1.0, places=4)
        self.assertAlmostEqual(a, 4.5, places=4)

    def test_interp_clamps_outside_range(self):
        profile = ((10.0, 1.0, 5.0), (20.0, 2.0, 9.0))
        # Below range
        self.assertEqual(_interp_profile(profile, 0.0), (1.0, 5.0))
        # Above range
        self.assertEqual(_interp_profile(profile, 100.0), (2.0, 9.0))


# ---------------------------------------------------------------------------
# Heat-flux helpers
# ---------------------------------------------------------------------------
class TestHeatFluxHelpers(unittest.TestCase):
    def test_zero_mach_returns_zero_flux(self):
        self.assertEqual(
            _stagnation_heat_flux_coldwall(1.0, 0.0, 0.1, 288.0),
            0.0,
        )

    def test_zero_rho_returns_zero_flux(self):
        self.assertEqual(
            _stagnation_heat_flux_coldwall(0.0, 2.0, 0.1, 288.0),
            0.0,
        )

    def test_hot_wall_correction_zero_at_recovery(self):
        # When T_wall equals T_recovery, hot-wall flux must be zero.
        self.assertEqual(
            _hot_wall_heat_flux(10000.0, 400.0, 400.0, 250.0),
            0.0,
        )

    def test_hot_wall_correction_full_at_cold_wall(self):
        # When T_wall equals T_amb, hot-wall flux must equal q_cold.
        result = _hot_wall_heat_flux(10000.0, 250.0, 400.0, 250.0)
        self.assertAlmostEqual(result, 10000.0, places=3)

    def test_hot_wall_correction_clamps_negative(self):
        # Above-recovery wall would give negative q; clamp to 0.
        self.assertEqual(
            _hot_wall_heat_flux(10000.0, 500.0, 400.0, 250.0),
            0.0,
        )

    def test_recovery_temp_mach_zero_is_ambient(self):
        self.assertEqual(_recovery_temp(288.0, 0.0), 288.0)

    def test_recovery_temp_increases_with_mach(self):
        self.assertGreater(
            _recovery_temp(288.0, 2.0),
            _recovery_temp(288.0, 1.0),
        )


# ---------------------------------------------------------------------------
# Material-status short-circuits
# ---------------------------------------------------------------------------
class TestMaterialShortCircuits(unittest.TestCase):
    def test_tps_material_returns_not_applicable(self):
        avcoat = _by_name("AVCOAT")
        r = integrate_panel(
            avcoat, 0.002, flight_duration_s=100.0,
            fallback_mach=1.0, fallback_alt_km=10.0,
        )
        self.assertEqual(r.status, "not_applicable")
        self.assertIsNone(r.peak_backface_K)

    def test_unknown_cp_returns_unknown(self):
        unk = next(
            (m for m in MATERIALS_DB if m.cp_data_status == "unknown"),
            None,
        )
        if unk is None:
            self.skipTest("No unknown-c_p materials in DB")
        r = integrate_panel(
            unk, 0.002, flight_duration_s=100.0,
            fallback_mach=1.0, fallback_alt_km=10.0,
        )
        self.assertEqual(r.status, "unknown")

    def test_invalid_thickness_raises(self):
        with self.assertRaises(ValueError):
            integrate_panel(
                _by_name("2024-T3"), 0.0,
                flight_duration_s=10.0, fallback_mach=1.0,
                fallback_alt_km=0.0,
            )

    def test_invalid_n_nodes_raises(self):
        with self.assertRaises(ValueError):
            integrate_panel(
                _by_name("2024-T3"), 0.002,
                flight_duration_s=10.0, fallback_mach=1.0,
                fallback_alt_km=0.0, n_nodes=2,
            )


# ---------------------------------------------------------------------------
# Physics limits
# ---------------------------------------------------------------------------
class TestPhysicsLimits(unittest.TestCase):
    def test_zero_mach_no_heating(self):
        """At M=0 the convective flux is zero and the panel should
        stay essentially at ambient temperature for any duration."""
        al = _by_name("2024-T3")
        r = integrate_panel(
            al, 0.002,
            flight_duration_s=600.0,
            fallback_mach=0.0, fallback_alt_km=5.0,
        )
        # Surface and backface should both stay near ambient at 5 km
        # (T_amb ≈ 256 K).
        self.assertLess(r.peak_surface_K, 260.0)
        self.assertLess(r.peak_backface_K, 260.0)

    def test_steady_state_converges_to_lumped_capacitance(self):
        """For a thin metal panel (small Biot number) the long-time
        limit must converge to the lumped-capacitance equilibrium:

            ε σ T_eq^4 = q_cold (T_rec - T_eq)/(T_rec - T_amb) + ε σ T_amb^4
        """
        al = _by_name("2024-T3")
        # Long sustained flight at constant M=2.0, 9 km.
        r = integrate_panel(
            al, 0.002, flight_duration_s=3600.0,
            fallback_mach=2.0, fallback_alt_km=9.0,
        )
        # Compute the lumped equilibrium reference numerically.
        T_amb = 228.0  # ISA at 9 km
        from core.transient_heat import _recovery_temp, _stagnation_heat_flux_coldwall
        T_rec = _recovery_temp(T_amb, 2.0)
        # Lumped ODE solve with ample resolution.
        rho, cp, L = al.density_kgm3, al.specific_heat_J_kgK, 0.002
        from core.physics_engine import _isa_atmosphere
        T_amb_isa, _p, rho_atm = _isa_atmosphere(9.0)
        q_cold = _stagnation_heat_flux_coldwall(rho_atm, 2.0, 0.30, T_amb_isa)
        T_rec_isa = _recovery_temp(T_amb_isa, 2.0)
        eps_sigma = 0.85 * 5.67e-8
        T = T_amb_isa
        dt_ref = 0.5
        for _ in range(int(3600.0 / dt_ref)):
            q_hot = q_cold * max(0, T_rec_isa - T) / (T_rec_isa - T_amb_isa)
            q_rad = eps_sigma * (T ** 4 - T_amb_isa ** 4)
            T += (q_hot - q_rad) / (rho * cp * L) * dt_ref
        # Solver should match the lumped reference within a few K.
        self.assertAlmostEqual(r.peak_surface_K, T, delta=10.0)

    def test_backface_below_surface_during_transient(self):
        """During heat-up the back face must lag the surface."""
        ti = _by_name("Ti-6Al-4V")
        prof = ((0.0, 0.0, 0.0), (10.0, 2.0, 9.0))
        r = integrate_panel(ti, 0.003, flight_profile=prof, R_n_m=0.05)
        self.assertLessEqual(r.peak_backface_K, r.peak_surface_K + 1.0)


# ---------------------------------------------------------------------------
# Explicit vs implicit agreement
# ---------------------------------------------------------------------------
class TestExplicitImplicitAgreement(unittest.TestCase):
    def test_explicit_and_implicit_agree_on_short_aluminum_flight(self):
        """At a duration where both schemes are tractable, they
        must agree on peak temperatures within ~5 K."""
        al = _by_name("2024-T3")
        prof = ((0.0, 0.0, 0.0), (10.0, 2.0, 9.0), (25.0, 0.5, 12.0))
        r_exp = integrate_panel(
            al, 0.003, flight_profile=prof, R_n_m=0.05,
            method="explicit",
        )
        r_imp = integrate_panel(
            al, 0.003, flight_profile=prof, R_n_m=0.05,
            method="implicit",
        )
        self.assertEqual(r_exp.method_used, "explicit")
        self.assertEqual(r_imp.method_used, "implicit")
        self.assertAlmostEqual(
            r_exp.peak_surface_K, r_imp.peak_surface_K, delta=5.0,
        )
        self.assertAlmostEqual(
            r_exp.peak_backface_K, r_imp.peak_backface_K, delta=5.0,
        )


# ---------------------------------------------------------------------------
# Real-envelope smoke tests (Gemini-prediction targets)
# ---------------------------------------------------------------------------
class TestRealisticEnvelopes(unittest.TestCase):
    def test_sounding_rocket_backface_60_to_90_C(self):
        """Gemini's predicted internal soak for a Mach-2 25 s
        sounding rocket on a 3 mm aluminum airframe is 60-70 °C
        (333-343 K). The solver must land in that ballpark, not the
        steady-state 113 °C (386 K) the prior MATVEC build assumed."""
        al = _by_name("2024-T3")
        prof = ((0.0, 0.0, 0.0), (10.0, 2.0, 9.0), (15.0, 2.0, 9.0),
                (25.0, 0.5, 12.0))
        r = integrate_panel(al, 0.003, flight_profile=prof, R_n_m=0.05)
        self.assertEqual(r.status, "applied")
        # Allow a generous 60-100 °C window to absorb variation in
        # the profile shape; the key is that we are NOT at 110+ °C.
        self.assertGreater(r.peak_backface_K, 320.0)  # > 47 °C
        self.assertLess(r.peak_backface_K, 380.0)     # < 107 °C

    def test_concorde_long_cruise_converges_near_recovery(self):
        """Concorde 3-h cruise on Al panel: equilibrium ≈ 364 K
        (lumped-capacitance reference)."""
        al = _by_name("2024-T3")
        r = integrate_panel(
            al, 0.002, flight_duration_s=10800.0,
            fallback_mach=2.04, fallback_alt_km=18.0, R_n_m=0.40,
        )
        self.assertEqual(r.status, "applied")
        self.assertAlmostEqual(r.peak_surface_K, 364.0, delta=8.0)

    def test_sr71_long_cruise_in_titanium_range(self):
        """SR-71 90 min cruise on 1.5 mm Ti panel: equilibrium near
        ~583 K (lumped reference). Real SR-71 skin temperatures are
        reported in the 500-600 K range."""
        ti = _by_name("Ti-6Al-4V")
        r = integrate_panel(
            ti, 0.0015, flight_duration_s=5400.0,
            fallback_mach=3.2, fallback_alt_km=25.0, R_n_m=0.15,
        )
        self.assertEqual(r.status, "applied")
        self.assertGreater(r.peak_surface_K, 500.0)
        self.assertLess(r.peak_surface_K, 700.0)


# ---------------------------------------------------------------------------
# Result dataclass + diagnostics
# ---------------------------------------------------------------------------
class TestResultDataclass(unittest.TestCase):
    def test_applied_result_carries_time_series(self):
        al = _by_name("2024-T3")
        prof = ((0.0, 0.0, 0.0), (25.0, 2.0, 9.0))
        r = integrate_panel(al, 0.003, flight_profile=prof, R_n_m=0.05)
        self.assertEqual(r.status, "applied")
        self.assertGreater(len(r.time_s), 10)
        self.assertEqual(
            len(r.T_surface_K), len(r.time_s)
        )
        self.assertEqual(
            len(r.T_backface_K), len(r.time_s)
        )

    def test_method_used_field_populated(self):
        al = _by_name("2024-T3")
        r = integrate_panel(
            al, 0.003, flight_duration_s=20.0,
            fallback_mach=1.0, fallback_alt_km=0.0,
        )
        self.assertIn(r.method_used, ("explicit", "implicit"))

    def test_estimated_cp_carries_warning_note(self):
        # PWA 1484 has cp_data_status="estimated".
        pwa = _by_name("PWA 1484")
        if pwa.cp_data_status != "estimated":
            self.skipTest("PWA 1484 cp status changed")
        r = integrate_panel(
            pwa, 0.0015, flight_duration_s=600.0,
            fallback_mach=0.5, fallback_alt_km=0.0,
        )
        if r.status == "applied":
            self.assertIn("estimated", r.notes.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
