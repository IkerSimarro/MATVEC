"""
Test suite for matching_engine.py — Step 3 of MATVEC.
stdlib unittest only. Run: python -m unittest test_matching_engine.py -v
"""

import unittest

from core.materials_db import MATERIALS_DB, get_materials_by_regime, get_strength_at_temperature
from core.physics_engine import run_analysis
from core.matching_engine import (
    match_materials,
    MARGINAL_STRUCTURAL_FRACTION,
    MaterialCandidate,
    MatchResult,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _run(mach, alt_km, mass_kg, R_n, g_load=1.0):
    return run_analysis(mach, alt_km, mass_kg, R_n, peak_g_load=g_load)


def _all_evaluated(result: MatchResult) -> list:
    """Return all direct-mode candidates (excludes substrate-mode duplicates).

    Substrate-mode candidates are evaluated at T_soak rather than T_wall, so
    they break invariants that hold for direct-exposure evaluation (e.g.,
    thermal_margin = ceiling - T_wall, strength stored at T_wall). Tests that
    need substrate candidates should iterate result.viable + result.marginal
    directly with an explicit evaluation_mode filter.
    """
    return [
        c for c in result.viable + result.marginal + result.not_viable
        if getattr(c, "evaluation_mode", "direct") == "direct"
    ]


def _total_direct_count(result: MatchResult) -> int:
    """Total direct-mode candidates + tps_coatings + regime_rejected.

    Use this for accounting tests that expect the result to cover every
    material in MATERIALS_DB exactly once.
      - direct-mode structural candidates (metals/composites in viable/marginal/not_viable)
      - tps_coatings (ablators/TPS surfaced as paired coating layer)
      - regime_rejected (filtered out before evaluation)
    Substrate-mode candidates are intentional duplicates of metals (under
    hypothetical ablative coating) and must NOT be counted toward this invariant.
    """
    return (
        len(_all_evaluated(result))
        + len(result.tps_coatings)
        + len(result.regime_rejected)
    )


# ---------------------------------------------------------------------------
# 1. Regime Filter
# ---------------------------------------------------------------------------
class TestRegimeFilter(unittest.TestCase):

    def test_regime_rejected_excludes_inapplicable_materials(self):
        # Hypersonic run: polymer composites don't list "hypersonic"
        r = match_materials(_run(9.6, 33.5, 1400.0, 0.05, g_load=5.0))
        rejected_names = {m.name for m in r.regime_rejected}
        evaluated_names = {c.material.name for c in _all_evaluated(r)}
        coating_names = {c.material.name for c in r.tps_coatings}
        for m in MATERIALS_DB:
            if "hypersonic" not in m.applicable_regimes:
                # TPS materials get physics-pulled-back when T_wall ≥ 1200 K
                # (ablative unlock) and now live in r.tps_coatings (not the
                # primary lists). They are valid in either bucket.
                if m.category == "tps":
                    self.assertTrue(
                        m.name in rejected_names or m.name in coating_names,
                        f"{m.name} should be in regime_rejected or tps_coatings (TPS pull-back)")
                else:
                    self.assertIn(m.name, rejected_names,
                        f"{m.name} should be in regime_rejected for hypersonic")

    def test_evaluated_materials_all_have_regime(self):
        r = match_materials(_run(9.6, 33.5, 1400.0, 0.05, g_load=5.0))
        # Primary structural lists never contain TPS after Bug 2 fix; this loop
        # only sees metals/composites whose regime list must include "hypersonic".
        for candidate in _all_evaluated(r):
            self.assertIn("hypersonic", candidate.material.applicable_regimes,
                f"{candidate.material.name} in evaluated list but doesn't list 'hypersonic'")

    def test_no_material_in_both_rejected_and_evaluated(self):
        r = match_materials(_run(1.8, 12.0, 19700.0, 0.3, g_load=9.0))
        rejected_names = {m.name for m in r.regime_rejected}
        for candidate in _all_evaluated(r):
            self.assertNotIn(candidate.material.name, rejected_names,
                f"{candidate.material.name} appears in both evaluated and regime_rejected")

    def test_reentry_rejects_polymer_composites(self):
        r = match_materials(_run(36.0, 80.0, 5900.0, 4.7, g_load=5.0))
        rejected_names = {m.name for m in r.regime_rejected}
        polymer_names = {m.name for m in MATERIALS_DB if m.category == "composite_polymer"}
        for name in polymer_names:
            self.assertIn(name, rejected_names,
                f"Polymer composite {name} should be regime-rejected for reentry")

    def test_supersonic_all_materials_accounted(self):
        r = match_materials(_run(1.8, 12.0, 19700.0, 0.3, g_load=9.0))
        total = len(r.viable) + len(r.marginal) + len(r.not_viable) + len(r.regime_rejected)
        self.assertEqual(total, len(MATERIALS_DB))


# ---------------------------------------------------------------------------
# 2. Thermal Filter
# ---------------------------------------------------------------------------
class TestThermalFilter(unittest.TestCase):

    def setUp(self):
        # X-43A: T_wall ≈ 2541 K — well above most metals' service limits
        self.physics = _run(9.6, 33.5, 1400.0, 0.05, g_load=5.0)
        self.result = match_materials(self.physics)
        self.T_wall = self.physics.thermal.T_wall_K
        self.T_wall_max = self.physics.thermal.T_wall_max_K

    def _find_candidate(self, name):
        for c in _all_evaluated(self.result):
            if c.name_matches(name) if hasattr(c, 'name_matches') else c.material.name == name:
                return c
        return None

    def test_low_service_temp_materials_fail_thermal(self):
        # Aluminum 2024-T3 (service_temp_air ≈ 450K) must fail thermal at T_wall ≈ 2541K
        for candidate in _all_evaluated(self.result):
            m = candidate.material
            if m.category == "aluminum":
                self.assertEqual(candidate.thermal_status, "fail",
                    f"{m.name} (service {m.service_temp_air_K:.0f} K) should fail thermal "
                    f"at T_wall={self.T_wall:.0f} K")

    def test_high_service_temp_materials_pass_thermal(self):
        # ZrB2-SiC UHTC: service_temp_air typically > 2200K — should pass at T_wall_max
        for candidate in _all_evaluated(self.result):
            m = candidate.material
            ceiling = candidate.thermal_ceiling_K
            if ceiling >= self.T_wall_max:
                self.assertEqual(candidate.thermal_status, "pass",
                    f"{m.name} ceiling {ceiling:.0f} K >= T_wall_max {self.T_wall_max:.0f} K "
                    f"should be thermal 'pass'")

    def test_thermal_ceiling_is_min_of_service_and_oxidation(self):
        for candidate in _all_evaluated(self.result):
            m = candidate.material
            bare_ceiling   = min(m.service_temp_air_K, m.oxidation_max_temp_K)
            coated_ceiling = getattr(m, "coated_max_temp_K", 0.0) or 0.0
            # Coated refractories (C-103, Mo, Ta, W, Re) publish an effective
            # air-service ceiling above the bare oxidation limit.
            expected_ceiling = max(bare_ceiling, coated_ceiling)
            self.assertAlmostEqual(candidate.thermal_ceiling_K, expected_ceiling, places=6,
                msg=f"{m.name}: thermal_ceiling should be max(bare, coated)")

    def test_thermal_margin_consistent_with_ceiling(self):
        for candidate in _all_evaluated(self.result):
            expected_margin = candidate.thermal_ceiling_K - self.T_wall
            self.assertAlmostEqual(candidate.thermal_margin_K, expected_margin, places=6,
                msg=f"{candidate.material.name}: thermal_margin inconsistent")

    def test_marginal_thermal_status_in_correct_range(self):
        # Any marginal thermal material must have T_wall <= ceiling < T_wall_max
        for candidate in _all_evaluated(self.result):
            if candidate.thermal_status == "marginal":
                c = candidate.thermal_ceiling_K
                self.assertGreaterEqual(c, self.T_wall,
                    f"{candidate.material.name} thermal=marginal but ceiling < T_wall")
                self.assertLess(c, self.T_wall_max,
                    f"{candidate.material.name} thermal=marginal but ceiling >= T_wall_max")

    def test_thermal_fail_means_ceiling_below_T_wall(self):
        for candidate in _all_evaluated(self.result):
            if candidate.thermal_status == "fail":
                self.assertLess(candidate.thermal_ceiling_K, self.T_wall,
                    f"{candidate.material.name} thermal=fail but ceiling >= T_wall")


# ---------------------------------------------------------------------------
# 3. Structural Filter
# ---------------------------------------------------------------------------
class TestStructuralFilter(unittest.TestCase):

    def setUp(self):
        # Use a moderate hypersonic case
        self.physics = _run(5.0, 20.0, 5000.0, 0.3, g_load=3.0)
        self.result = match_materials(self.physics)
        self.sigma_req = self.physics.structural.sigma_tensile_required_MPa
        self.T_wall = self.physics.thermal.T_wall_K

    def test_strength_stored_matches_helper(self):
        for candidate in _all_evaluated(self.result):
            expected = get_strength_at_temperature(candidate.material, self.T_wall)
            self.assertAlmostEqual(candidate.strength_at_T_wall_MPa, expected, places=6,
                msg=f"{candidate.material.name}: stored strength doesn't match helper")

    def test_structural_pass_has_sufficient_margin(self):
        for candidate in _all_evaluated(self.result):
            if candidate.structural_status == "pass":
                self.assertGreaterEqual(
                    candidate.structural_margin_fraction,
                    MARGINAL_STRUCTURAL_FRACTION,
                    msg=f"{candidate.material.name} structural=pass but margin < threshold"
                )

    def test_structural_marginal_in_correct_range(self):
        for candidate in _all_evaluated(self.result):
            if candidate.structural_status == "marginal":
                self.assertGreaterEqual(candidate.structural_margin_fraction, 0.0,
                    f"{candidate.material.name} structural=marginal but margin < 0")
                self.assertLess(candidate.structural_margin_fraction,
                    MARGINAL_STRUCTURAL_FRACTION,
                    f"{candidate.material.name} structural=marginal but margin >= threshold")

    def test_structural_fail_has_negative_margin(self):
        for candidate in _all_evaluated(self.result):
            if candidate.structural_status == "fail":
                self.assertLess(candidate.structural_margin_fraction, 0.0,
                    f"{candidate.material.name} structural=fail but margin >= 0")

    def test_structural_margin_formula(self):
        for candidate in _all_evaluated(self.result):
            # material-specific sigma_req; skip edge case where sigma_req_material <= 0
            if candidate.sigma_req_material_MPa > 0:
                expected_margin = (candidate.strength_at_T_wall_MPa /
                                   candidate.sigma_req_material_MPa - 1.0)
                self.assertAlmostEqual(candidate.structural_margin_fraction, expected_margin,
                    places=6, msg=f"{candidate.material.name}: structural_margin formula wrong")


# ---------------------------------------------------------------------------
# 4. Marginal Zone
# ---------------------------------------------------------------------------
class TestMarginalZone(unittest.TestCase):

    def test_thermally_marginal_never_viable(self):
        for mach, alt, mass, R_n in [
            (5.0, 20.0, 5000.0, 0.3),
            (9.6, 33.5, 1400.0, 0.05),
        ]:
            r = match_materials(_run(mach, alt, mass, R_n, g_load=3.0))
            for c in r.viable:
                self.assertNotEqual(c.thermal_status, "marginal",
                    f"{c.material.name} in viable but thermal_status='marginal'")

    def test_structurally_marginal_never_viable(self):
        for mach, alt, mass, R_n in [(5.0, 20.0, 5000.0, 0.3)]:
            r = match_materials(_run(mach, alt, mass, R_n, g_load=3.0))
            for c in r.viable:
                self.assertNotEqual(c.structural_status, "marginal",
                    f"{c.material.name} in viable but structural_status='marginal'")

    def test_no_material_viable_consistent_with_viable_list(self):
        for mach, alt, mass, R_n, g in [
            (1.8, 12.0, 19700.0, 0.3, 9.0),
            (9.6, 33.5, 1400.0, 0.05, 5.0),
            (25.0, 70.0, 100000.0, 0.3, 3.0),
        ]:
            r = match_materials(_run(mach, alt, mass, R_n, g_load=g))
            self.assertEqual(r.no_material_viable, len(r.viable) == 0,
                f"no_material_viable inconsistent for M={mach}")

    def test_impossible_consistent_with_viable_and_marginal(self):
        for mach, alt, mass, R_n, g in [
            (9.6, 33.5, 1400.0, 0.05, 5.0),
            (36.0, 80.0, 5900.0, 4.7, 5.0),
        ]:
            r = match_materials(_run(mach, alt, mass, R_n, g_load=g))
            expected_impossible = (len(r.viable) == 0 and len(r.marginal) == 0)
            self.assertEqual(r.impossible, expected_impossible,
                f"impossible flag inconsistent for M={mach}")


# ---------------------------------------------------------------------------
# 5. Ranking
# ---------------------------------------------------------------------------
class TestRanking(unittest.TestCase):
    """
    viable/marginal: sorted ascending (smallest positive min-margin = minimum adequate first).
    not_viable: sorted descending (least-negative min-margin = nearest miss first).
    """

    def _assert_sorted_asc(self, candidates, label):
        for i in range(len(candidates) - 1):
            self.assertLessEqual(
                candidates[i].score, candidates[i + 1].score,
                msg=f"{label}: score not ascending at index {i} "
                    f"({candidates[i].material.name}={candidates[i].score:.4f} > "
                    f"{candidates[i+1].material.name}={candidates[i+1].score:.4f})"
            )

    def _assert_sorted_desc(self, candidates, label):
        for i in range(len(candidates) - 1):
            self.assertGreaterEqual(
                candidates[i].score, candidates[i + 1].score,
                msg=f"{label}: score not descending at index {i} "
                    f"({candidates[i].material.name}={candidates[i].score:.4f} < "
                    f"{candidates[i+1].material.name}={candidates[i+1].score:.4f})"
            )

    def test_viable_sorted_ascending(self):
        r = match_materials(_run(0.5, 5.0, 10000.0, 0.5, g_load=2.0))
        if len(r.viable) >= 2:
            self._assert_sorted_asc(r.viable, "viable")

    def test_marginal_sorted_ascending(self):
        r = match_materials(_run(1.8, 12.0, 19700.0, 0.3, g_load=9.0))
        if len(r.marginal) >= 2:
            self._assert_sorted_asc(r.marginal, "marginal")

    def test_not_viable_sorted_descending(self):
        r = match_materials(_run(9.6, 33.5, 1400.0, 0.05, g_load=5.0))
        if len(r.not_viable) >= 2:
            self._assert_sorted_desc(r.not_viable, "not_viable")

    def test_viable_have_no_fail_status(self):
        # Viable candidates must have both thermal and structural as "pass" — never "fail"
        r = match_materials(_run(0.5, 5.0, 10000.0, 0.5, g_load=2.0))
        for c in r.viable:
            self.assertNotEqual(c.thermal_status, "fail",
                f"{c.material.name} is viable but thermal_status=fail")
            self.assertNotEqual(c.structural_status, "fail",
                f"{c.material.name} is viable but structural_status=fail")
        for c in r.marginal:
            self.assertNotEqual(c.thermal_status, "fail",
                f"{c.material.name} is marginal but thermal_status=fail")
            self.assertNotEqual(c.structural_status, "fail",
                f"{c.material.name} is marginal but structural_status=fail")

    def test_all_regimes_produce_sorted_results(self):
        cases = [
            (0.3, 5.0, 5000.0, 0.5, 2.0),    # subsonic
            (1.8, 12.0, 19700.0, 0.3, 9.0),   # supersonic
            (5.0, 20.0, 5000.0, 0.3, 3.0),    # hypersonic
            (36.0, 80.0, 5900.0, 4.7, 5.0),   # reentry
        ]
        for mach, alt, mass, R_n, g in cases:
            r = match_materials(_run(mach, alt, mass, R_n, g_load=g))
            with self.subTest(mach=mach):
                if len(r.viable) >= 2:
                    self._assert_sorted_asc(r.viable, f"viable M={mach}")
                if len(r.marginal) >= 2:
                    self._assert_sorted_asc(r.marginal, f"marginal M={mach}")
                if len(r.not_viable) >= 2:
                    self._assert_sorted_desc(r.not_viable, f"not_viable M={mach}")


# ---------------------------------------------------------------------------
# 6. Coverage — All Materials Accounted For
# ---------------------------------------------------------------------------
class TestNotViableList(unittest.TestCase):

    def test_all_56_materials_accounted_for(self):
        cases = [
            (1.8, 12.0, 19700.0, 0.3, 9.0),
            (9.6, 33.5, 1400.0, 0.05, 5.0),
            (36.0, 80.0, 5900.0, 4.7, 5.0),
            (0.5, 5.0, 10000.0, 0.5, 2.0),
        ]
        for mach, alt, mass, R_n, g in cases:
            with self.subTest(mach=mach):
                r = match_materials(_run(mach, alt, mass, R_n, g_load=g))
                # Count direct-mode candidates only — substrate-mode duplicates
                # are intentional second-pass evaluations of metals under
                # ablative coating and must not inflate the accounting total.
                total = _total_direct_count(r)
                self.assertEqual(total, len(MATERIALS_DB),
                    f"M={mach}: {total} materials accounted for, expected {len(MATERIALS_DB)}")

    def test_no_duplicate_materials(self):
        r = match_materials(_run(9.6, 33.5, 1400.0, 0.05, g_load=5.0))
        # Direct-mode candidates + tps_coatings + regime_rejected — substrate-mode
        # candidates ARE intentional duplicates of metals (under hypothetical ablative
        # coating) and are excluded.
        all_names = (
            [c.material.name for c in _all_evaluated(r)] +
            [c.material.name for c in r.tps_coatings] +
            [m.name for m in r.regime_rejected]
        )
        self.assertEqual(len(all_names), len(set(all_names)),
            "Duplicate material found across result lists (direct-mode + tps_coatings)")

    def test_diagnosis_empty_when_viable_nonempty(self):
        # Low-speed, low-temp run — should have viable materials
        r = match_materials(_run(0.3, 2.0, 5000.0, 0.5, g_load=1.0))
        if r.viable:
            self.assertEqual(r.diagnosis, "",
                "diagnosis should be empty string when viable is non-empty")

    def test_diagnosis_nonempty_when_no_material_viable(self):
        # X-43A: almost certainly no viable materials
        r = match_materials(_run(9.6, 33.5, 1400.0, 0.05, g_load=5.0))
        if r.no_material_viable:
            self.assertGreater(len(r.diagnosis), 0,
                "diagnosis must be non-empty when no_material_viable=True")


# ---------------------------------------------------------------------------
# 7. Reference Vehicles
# ---------------------------------------------------------------------------
class TestReferenceVehicles(unittest.TestCase):

    # ── SR-71 Blackbird (M=3.2, 25 km) ──────────────────────────────────────

    def test_sr71_regime(self):
        r = match_materials(_run(3.2, 25.0, 30600.0, 0.15, g_load=2.5))
        self.assertEqual(r.physics.flight_regime, "supersonic")

    def test_sr71_total_is_97(self):
        r = match_materials(_run(3.2, 25.0, 30600.0, 0.15, g_load=2.5))
        total = len(r.viable) + len(r.marginal) + len(r.not_viable) + len(r.regime_rejected)
        self.assertEqual(total, 97)

    def test_sr71_in718_viable(self):
        # T_wall≈607K (recovery-capped); IN718 ceiling=980K → must be viable or marginal
        r = match_materials(_run(3.2, 25.0, 30600.0, 0.15, g_load=2.5))
        viable_names = {c.material.name for c in r.viable + r.marginal}
        self.assertIn("Inconel 718", viable_names,
            "SR-71: Inconel 718 (ceiling 980K) should be viable with T_wall=607K")

    def test_sr71_ti6al4v_marginal(self):
        # Ti-6Al-4V ceiling=625K; T_wall≈607K → ceiling clears nominal but not
        # uncapped SG T_wall_max (~1127K) → marginal in "general" category
        r = match_materials(_run(3.2, 25.0, 30600.0, 0.15, g_load=2.5))
        marginal_names = {c.material.name for c in r.marginal}
        self.assertIn("Ti-6Al-4V", marginal_names,
            "SR-71 general: Ti-6Al-4V (ceiling 625K) should be marginal with T_wall=607K < T_wall_max")

    # ── X-15 (M=6.7, 30 km) ─────────────────────────────────────────────────

    def test_x15_regime(self):
        r = match_materials(_run(6.7, 30.0, 15195.0, 0.30, g_load=5.0))
        self.assertEqual(r.physics.flight_regime, "hypersonic")

    def test_x15_total_is_97(self):
        r = match_materials(_run(6.7, 30.0, 15195.0, 0.30, g_load=5.0))
        # Direct-mode total only — substrate-mode metals are duplicates
        total = _total_direct_count(r)
        self.assertEqual(total, 97)

    def test_x15_cc_composite_viable_or_marginal(self):
        # T_wall≈1645K; C/C Composite ceiling=1773K → viable
        r = match_materials(_run(6.7, 30.0, 15195.0, 0.30, g_load=5.0))
        viable_names = {c.material.name for c in r.viable + r.marginal}
        self.assertIn("Carbon-Carbon Composite", viable_names,
            "X-15: C/C Composite (ceiling 1773K) should be viable at T_wall=1645K")

    def test_x15_inconel_x750_not_viable(self):
        # T_wall≈1645K; Inconel X-750 ceiling=1144K → not_viable
        r = match_materials(_run(6.7, 30.0, 15195.0, 0.30, g_load=5.0))
        not_viable_names = {c.material.name for c in r.not_viable}
        self.assertIn("Inconel X-750", not_viable_names,
            "X-15: Inconel X-750 (ceiling 1144K) should be not_viable at T_wall=1645K")

    def test_x15_result_well_formed(self):
        r = match_materials(_run(6.7, 30.0, 15195.0, 0.30, g_load=5.0))
        self.assertIsInstance(r.viable, list)
        self.assertIsInstance(r.marginal, list)
        self.assertIsInstance(r.not_viable, list)
        self.assertIsInstance(r.regime_rejected, list)
        self.assertIsInstance(r.diagnosis, str)
        self.assertIsInstance(r.warnings, list)
        total = _total_direct_count(r)
        self.assertEqual(total, 97)

    def test_x15_candidate_fields_nonnull(self):
        r = match_materials(_run(6.7, 30.0, 15195.0, 0.30, g_load=5.0))
        for candidate in _all_evaluated(r):
            self.assertIsNotNone(candidate.material)
            self.assertIsNotNone(candidate.thermal_ceiling_K)
            self.assertIsNotNone(candidate.strength_at_T_wall_MPa)
            self.assertIsNotNone(candidate.sigma_req_material_MPa)
            self.assertIsInstance(candidate.score, float)
            self.assertIsInstance(candidate.notes, list)

    def test_x15_impossible_flag(self):
        r = match_materials(_run(6.7, 30.0, 15195.0, 0.30, g_load=5.0))
        expected = (len(r.viable) == 0 and len(r.marginal) == 0)
        self.assertEqual(r.impossible, expected)

    # ── X-15 under "aircraft" category (physics-driven ablative unlock) ────

    def test_x15_aircraft_ablative_unlock_active(self):
        """X-15 at M=6.7 (T_wall ≈ 1645 K > 1200) must surface TPS recommendations
        in the dedicated tps_coatings list (NOT in primary viable/marginal/not_viable)."""
        r = match_materials(
            _run(6.7, 30.0, 15195.0, 0.30, g_load=5.0),
            vehicle_category="aircraft",
        )
        self.assertGreater(
            len(r.tps_coatings), 0,
            "X-15 (T_wall ≈ 1645 K) must trigger ablative-unlock and populate tps_coatings",
        )
        # TPS must NOT contaminate the primary structural lists
        for c in r.viable + r.marginal + r.not_viable:
            self.assertNotEqual(
                c.material.category, "tps",
                f"TPS material {c.material.name} leaked into primary structural ranking",
            )

    def test_x15_aircraft_ablative_in_tps_coatings(self):
        """At least one ablative TPS (PICA / AVCOAT / SLA-561V) should appear
        in tps_coatings as a viable or marginal coating layer."""
        r = match_materials(
            _run(6.7, 30.0, 15195.0, 0.30, g_load=5.0),
            vehicle_category="aircraft",
        )
        coating_names = {
            c.material.name for c in r.tps_coatings
            if c.overall_status in ("viable", "marginal")
        }
        ablatives = {"PICA", "PICA-X", "AVCOAT", "SLA-561V"}
        self.assertTrue(
            coating_names & ablatives,
            f"Expected at least one ablative TPS in tps_coatings; got {sorted(coating_names)}",
        )

    def test_x15_aircraft_density_ceiling_lifted(self):
        """X-15: density ceiling lifts to 8500 kg/m³ under ablative-unlock."""
        r = match_materials(
            _run(6.7, 30.0, 15195.0, 0.30, g_load=5.0),
            vehicle_category="aircraft",
        )
        above = [c for c in r.viable + r.marginal + r.not_viable
                 if c.material.density_kgm3 > 8500.0]
        # Refractories above 8500 (Mo, W, Re) should still be excluded.
        for c in above:
            self.assertIn(
                "exceeds aircraft ceiling",
                " ".join(c.notes),
                f"{c.material.name} above 8500 kg/m³ should be density-excluded",
            )
        # At least Inconel X-750 (8280) should now be admitted (was excluded under
        # the strict 5000 kg/m³ ceiling that applies to non-ablative aircraft).
        all_names = {
            c.material.name
            for c in r.viable + r.marginal + r.not_viable
            if c.material.density_kgm3 > 5000.0
        }
        self.assertTrue(
            len(all_names) > 0,
            "Density ceiling lift failed: no materials > 5000 kg/m³ in evaluation",
        )

    def test_x15_aircraft_tps_structural_bypassed(self):
        """TPS materials must bypass the structural check regardless of category.
        After Bug 2 fix, TPS lives in r.tps_coatings, not the primary lists."""
        r = match_materials(
            _run(6.7, 30.0, 15195.0, 0.30, g_load=5.0),
            vehicle_category="aircraft",
        )
        for c in r.tps_coatings:
            self.assertEqual(
                c.structural_status, "pass",
                f"TPS {c.material.name} structural check should be bypassed",
            )
            self.assertEqual(
                c.sigma_req_material_MPa, 0.0,
                f"TPS {c.material.name} sigma_req should be 0",
            )
            self.assertEqual(
                c.structural_margin_fraction, 0.0,
                f"TPS {c.material.name} structural_margin should be 0.0 "
                f"(not the +1e6 sentinel that leaked +100,000,000% to the UI)",
            )

    def test_x15_aircraft_substrate_metal_passes(self):
        """X-15: at least one metallic substrate candidate should appear (Ti or Inconel)."""
        r = match_materials(
            _run(6.7, 30.0, 15195.0, 0.30, g_load=5.0),
            vehicle_category="aircraft",
        )
        substrate_candidates = [
            c for c in r.viable + r.marginal
            if c.evaluation_mode == "substrate"
        ]
        self.assertTrue(
            len(substrate_candidates) > 0,
            "X-15 should have at least one substrate-mode metal in viable/marginal",
        )
        # Substrate candidates should be metals
        for c in substrate_candidates:
            self.assertIn(
                c.material.category,
                ("titanium", "steel", "aluminum", "nickel", "cobalt"),
                f"Substrate candidate {c.material.name} is not a metal",
            )

    # ── CMC supersonic deprioritization ────────────────────────────────────

    def test_cmc_penalized_below_mach5_aircraft(self):
        """Mach-3 aircraft: CMCs must NOT appear in viable; should be in marginal with note."""
        r = match_materials(
            _run(3.0, 20.0, 30000.0, 0.30, g_load=2.5),
            vehicle_category="aircraft",
        )
        cmc_in_viable = [c for c in r.viable if c.material.category == "composite_ceramic"]
        self.assertEqual(
            len(cmc_in_viable), 0,
            f"CMCs must not be in viable for Mach-3 aircraft; "
            f"found {[c.material.name for c in cmc_in_viable]}",
        )
        cmc_present = [
            c for c in r.viable + r.marginal + r.not_viable
            if c.material.category == "composite_ceramic"
        ]
        self.assertTrue(
            len(cmc_present) > 0,
            "CMCs should still be evaluated, just demoted",
        )
        for c in cmc_present:
            self.assertTrue(
                any("deprioritized" in n.lower() for n in c.notes),
                f"{c.material.name} should have CMC deprioritization note",
            )

    def test_cmc_not_penalized_above_mach5(self):
        """X-15 (Mach 6.7): CMCs should NOT be penalized."""
        r = match_materials(
            _run(6.7, 30.0, 15195.0, 0.30, g_load=5.0),
            vehicle_category="aircraft",
        )
        for c in r.viable + r.marginal + r.not_viable:
            if c.material.category == "composite_ceramic":
                self.assertFalse(
                    any("deprioritized" in n.lower() for n in c.notes),
                    f"{c.material.name} should not be penalized at Mach 6.7",
                )

    def test_mach4_missile_not_topped_by_cmc(self):
        """Mach-4 missile: top viable should NOT be a CMC (Ti/steel preferred)."""
        r = match_materials(
            _run(4.0, 20.0, 900.0, 0.08, g_load=10.0),
            vehicle_category="hypersonic_missile",
        )
        if r.viable:
            top = r.viable[0].material
            self.assertNotEqual(
                top.category, "composite_ceramic",
                f"Top viable for Mach-4 missile is CMC ({top.name}); should be Ti/steel",
            )

    # ── TPS unlock threshold ───────────────────────────────────────────────

    def test_sr71_no_tps_below_threshold(self):
        """SR-71 (T_wall < 1200 K): TPS must NOT be unlocked.
        Both the primary lists AND tps_coatings must be free of TPS."""
        r = match_materials(
            _run(3.2, 25.0, 30600.0, 0.15, g_load=2.5),
            vehicle_category="aircraft",
        )
        tps_in_primary = any(
            c.material.category == "tps"
            for c in r.viable + r.marginal + r.not_viable
        )
        self.assertFalse(
            tps_in_primary,
            "SR-71 (T_wall well below 1200 K) must not surface TPS in primary lists",
        )
        self.assertEqual(
            len(r.tps_coatings), 0,
            "SR-71 (T_wall well below 1200 K) must not populate tps_coatings",
        )

    def test_sr71_no_substrate_candidates(self):
        """SR-71: no substrate-mode candidates should appear."""
        r = match_materials(
            _run(3.2, 25.0, 30600.0, 0.15, g_load=2.5),
            vehicle_category="aircraft",
        )
        substrate_candidates = [
            c for c in r.viable + r.marginal + r.not_viable
            if c.evaluation_mode == "substrate"
        ]
        self.assertEqual(
            len(substrate_candidates), 0,
            "SR-71 should have no substrate-mode candidates",
        )

    # ── Bug 2: TPS coating partition ───────────────────────────────────────

    def test_tps_not_in_viable_list(self):
        """No vehicle / regime should ever place TPS in r.viable / .marginal / .not_viable."""
        scenarios = [
            (6.7, 30.0, 15195.0, 0.30, 5.0,  "aircraft"),           # X-15 hot
            (3.2, 25.0, 30600.0, 0.15, 2.5,  "aircraft"),           # SR-71 cool
            (4.0, 20.0,   900.0, 0.08, 10.0, "hypersonic_missile"), # Mach-4 missile
            (9.6, 33.5,  1400.0, 0.05, 5.0,  "hypersonic_missile"), # X-43A
        ]
        for mach, alt, mass, R_n, g, cat in scenarios:
            r = match_materials(_run(mach, alt, mass, R_n, g_load=g), vehicle_category=cat)
            for c in r.viable + r.marginal + r.not_viable:
                self.assertNotEqual(
                    c.material.category, "tps",
                    f"[{cat} M={mach}] TPS material {c.material.name} leaked into "
                    f"primary structural ranking — Bug 2 partition failed",
                )

    def test_tps_coatings_populated_when_hot(self):
        """T_wall ≥ TPS_UNLOCK_TEMP_K must populate tps_coatings."""
        from core.matching_engine import TPS_UNLOCK_TEMP_K
        r = match_materials(
            _run(6.7, 30.0, 15195.0, 0.30, g_load=5.0),
            vehicle_category="aircraft",
        )
        self.assertGreaterEqual(
            r.physics.thermal.T_wall_K, TPS_UNLOCK_TEMP_K,
            "X-15 sanity: T_wall must be above TPS unlock threshold",
        )
        self.assertGreater(
            len(r.tps_coatings), 0,
            "T_wall ≥ TPS_UNLOCK_TEMP_K must populate tps_coatings",
        )
        # Each entry must be a TPS-category candidate with sigma_req=0
        for c in r.tps_coatings:
            self.assertEqual(c.material.category, "tps")
            self.assertEqual(c.sigma_req_material_MPa, 0.0)
            self.assertEqual(c.structural_margin_fraction, 0.0)

    def test_tps_coatings_empty_when_cool(self):
        """T_wall < TPS_UNLOCK_TEMP_K (and category not in {reentry, hypersonic_aircraft})
        must leave tps_coatings empty."""
        from core.matching_engine import TPS_UNLOCK_TEMP_K
        r = match_materials(
            _run(3.2, 25.0, 30600.0, 0.15, g_load=2.5),
            vehicle_category="aircraft",
        )
        self.assertLess(
            r.physics.thermal.T_wall_K, TPS_UNLOCK_TEMP_K,
            "SR-71 sanity: T_wall must be below TPS unlock threshold",
        )
        self.assertEqual(len(r.tps_coatings), 0)

    def test_mach4_missile_thermal_relief_effective(self):
        """Bug 1 sanity: Mach-4 missile thermal stress must be realistic after the
        thermal relief factor fix. Pre-fix: sigma_thermal_ref = 1414 MPa (crushed
        everything). Post-fix: relieved by 0.4 factor → ~566 MPa reference.

        Ti-6Al-2Sn-4Zr-2Mo should have a per-material sigma_req < 300 MPa (not the
        old 600+ MPa that made every metal impossible). The scenario itself is still
        demanding (T_wall ≈ 806 K blocks most Ti thermally, Ni alloys exceed the
        missile density ceiling), but the structural numbers must be realistic."""
        r = match_materials(
            _run(4.0, 20.0, 900.0, 0.08, g_load=10.0),
            vehicle_category="hypersonic_missile",
        )
        # Reference sigma should be relieved (was 1414, now ~566)
        ref_sigma = r.physics.structural.sigma_tensile_required_MPa
        self.assertLess(
            ref_sigma, 700.0,
            f"Reference sigma_tensile_required = {ref_sigma:.0f} MPa is still too high; "
            "thermal relief factor may not be applied in _compute_structural",
        )
        # Ti-6Al-2Sn-4Zr-2Mo per-material sigma_req should be well below old value
        ti_candidates = [
            c for c in r.viable + r.marginal + r.not_viable
            if c.material.name == "Ti-6Al-2Sn-4Zr-2Mo"
        ]
        self.assertEqual(len(ti_candidates), 1)
        sigma_req = ti_candidates[0].sigma_req_material_MPa
        self.assertLess(
            sigma_req, 300.0,
            f"Ti-6Al-2Sn-4Zr-2Mo sigma_req = {sigma_req:.0f} MPa; should be < 300 "
            "with per-material E/α and 0.4 relief factor (was 600+ pre-fix)",
        )


# ---------------------------------------------------------------------------
# 8. Output Completeness
# ---------------------------------------------------------------------------
class TestOutputCompleteness(unittest.TestCase):

    def setUp(self):
        self.result = match_materials(_run(9.6, 33.5, 1400.0, 0.05, g_load=5.0))

    def test_result_fields_present(self):
        r = self.result
        self.assertIsNotNone(r.physics)
        self.assertIsInstance(r.vehicle_category, str)
        self.assertIsInstance(r.viable, list)
        self.assertIsInstance(r.marginal, list)
        self.assertIsInstance(r.not_viable, list)
        self.assertIsInstance(r.regime_rejected, list)
        self.assertIsInstance(r.no_material_viable, bool)
        self.assertIsInstance(r.impossible, bool)
        self.assertIsInstance(r.diagnosis, str)
        self.assertIsInstance(r.warnings, list)
        self.assertIsInstance(r.tps_coatings, list)

    def test_no_material_viable_consistent(self):
        r = self.result
        self.assertEqual(r.no_material_viable, len(r.viable) == 0)

    def test_impossible_consistent(self):
        r = self.result
        self.assertEqual(r.impossible, len(r.viable) == 0 and len(r.marginal) == 0)

    def test_overall_status_values_valid(self):
        valid = {"viable", "marginal", "not_viable"}
        for candidate in _all_evaluated(self.result):
            self.assertIn(candidate.overall_status, valid,
                f"{candidate.material.name}: invalid overall_status '{candidate.overall_status}'")

    def test_status_fields_valid(self):
        valid = {"pass", "marginal", "fail"}
        for candidate in _all_evaluated(self.result):
            self.assertIn(candidate.thermal_status, valid,
                f"{candidate.material.name}: invalid thermal_status")
            self.assertIn(candidate.structural_status, valid,
                f"{candidate.material.name}: invalid structural_status")

    def test_score_is_float(self):
        for candidate in _all_evaluated(self.result):
            self.assertIsInstance(candidate.score, float,
                f"{candidate.material.name}: score is not float")

    def test_notes_is_list(self):
        for candidate in _all_evaluated(self.result):
            self.assertIsInstance(candidate.notes, list,
                f"{candidate.material.name}: notes is not list")

    def test_viable_candidates_match_overall_status(self):
        for c in self.result.viable:
            self.assertEqual(c.overall_status, "viable",
                f"{c.material.name} in viable list but overall_status='{c.overall_status}'")

    def test_marginal_candidates_match_overall_status(self):
        for c in self.result.marginal:
            self.assertEqual(c.overall_status, "marginal",
                f"{c.material.name} in marginal list but overall_status='{c.overall_status}'")

    def test_not_viable_candidates_match_overall_status(self):
        for c in self.result.not_viable:
            self.assertEqual(c.overall_status, "not_viable",
                f"{c.material.name} in not_viable list but overall_status='{c.overall_status}'")

    def test_consistency_across_regimes(self):
        cases = [
            (0.3, 5.0, 5000.0, 0.5, 1.0),
            (1.8, 12.0, 19700.0, 0.3, 9.0),
            (5.0, 20.0, 5000.0, 0.3, 3.0),
            (36.0, 80.0, 5900.0, 4.7, 5.0),
        ]
        for mach, alt, mass, R_n, g in cases:
            with self.subTest(mach=mach):
                r = match_materials(_run(mach, alt, mass, R_n, g_load=g))
                self.assertEqual(r.no_material_viable, len(r.viable) == 0)
                self.assertEqual(r.impossible, len(r.viable) == 0 and len(r.marginal) == 0)
                # Direct-mode count only (substrate-mode duplicates excluded)
                total = _total_direct_count(r)
                self.assertEqual(total, len(MATERIALS_DB))


# ---------------------------------------------------------------------------
# 9. Vehicle Category
# ---------------------------------------------------------------------------
class TestVehicleCategory(unittest.TestCase):

    def test_default_category_is_general(self):
        r = match_materials(_run(3.2, 25.0, 30600.0, 0.15, g_load=2.5))
        self.assertEqual(r.vehicle_category, "general")

    def test_vehicle_category_stored_in_result(self):
        r = match_materials(_run(3.2, 25.0, 30600.0, 0.15, g_load=2.5), vehicle_category="aircraft")
        self.assertEqual(r.vehicle_category, "aircraft")

    def test_aircraft_sr71_ti6al4v_marginal(self):
        """SR-71 aircraft: Ti-6Al-4V (ceiling 625 K) is marginal because T_wall_max (+15%) ~698 K exceeds ceiling."""
        r = match_materials(_run(3.2, 25.0, 30600.0, 0.15, g_load=2.5), vehicle_category="aircraft")
        marginal_names = {c.material.name for c in r.marginal}
        not_viable_names = {c.material.name for c in r.not_viable}
        self.assertIn("Ti-6Al-4V", marginal_names,
            "Aircraft SR-71: Ti-6Al-4V (ceiling 625 K) should be marginal — "
            "passes nominal T_wall (~607 K) but T_wall_max (~698 K) exceeds its ceiling")
        self.assertNotIn("Ti-6Al-4V", not_viable_names,
            "Ti-6Al-4V must not be not_viable — it passes the nominal thermal check")

    def test_aircraft_excludes_tps(self):
        r = match_materials(_run(3.2, 25.0, 30600.0, 0.15, g_load=2.5), vehicle_category="aircraft")
        all_eval = _all_evaluated(r)
        tps_names = {c.material.name for c in all_eval if c.material.category == "tps"}
        self.assertEqual(len(tps_names), 0, "Aircraft category must not evaluate TPS materials")

    def test_aircraft_supersonic_cmc_not_in_viable(self):
        """C/SiC and SiC/SiC CMCs must NOT appear in viable for supersonic aircraft (Mach < 5).

        CMCs are now physics-deprioritized below Mach 5: their score is penalised
        and any 'viable' classification is demoted to 'marginal' to prevent them
        from outranking titanium for SR-71-class supersonic airframes.
        """
        r = match_materials(_run(3.2, 25.0, 30600.0, 0.15, g_load=2.5), vehicle_category="aircraft")
        cmc_in_viable = [c for c in r.viable if c.material.category == "composite_ceramic"]
        self.assertEqual(len(cmc_in_viable), 0,
            "Aircraft Mach 3.2: CMCs must not appear in viable list "
            "(CMC supersonic penalty + viable→marginal demotion)")

    def test_missile_excludes_composite_polymer(self):
        """Polymer matrix composites are not viable for expendable missile bodies at elevated temps."""
        r = match_materials(_run(4.0, 20.0, 900.0, 0.08, g_load=10.0), vehicle_category="hypersonic_missile")
        all_eval = _all_evaluated(r)
        pmc_names = {c.material.name for c in all_eval if c.material.category == "composite_polymer"}
        self.assertEqual(len(pmc_names), 0,
            "Hypersonic missile category must not evaluate composite_polymer materials")

    def test_aircraft_total_is_full_db(self):
        r = match_materials(_run(3.2, 25.0, 30600.0, 0.15, g_load=2.5), vehicle_category="aircraft")
        total = len(r.viable) + len(r.marginal) + len(r.not_viable) + len(r.regime_rejected)
        self.assertEqual(total, len(MATERIALS_DB))

    def test_reentry_includes_tps(self):
        """Capsule at M=20, 70km classifies as HYPERSONIC; Reentry category must surface TPS
        in the dedicated tps_coatings list (after Bug 2 partition, not in primary lists)."""
        r = match_materials(_run(20.0, 70.0, 500.0, 1.50, g_load=8.0), vehicle_category="reentry")
        coating_names = {c.material.name for c in r.tps_coatings}
        self.assertIn("PICA", coating_names, "Reentry category must include PICA in tps_coatings")

    def test_reentry_total_is_full_db(self):
        """TPS pull-back must not double-count — all materials accounted for exactly once.

        Substrate-mode metals (added when ablative-unlock is active) are
        intentional duplicates and excluded from this accounting.
        """
        r = match_materials(_run(20.0, 70.0, 500.0, 1.50, g_load=8.0), vehicle_category="reentry")
        total = _total_direct_count(r)
        self.assertEqual(total, len(MATERIALS_DB),
            "Reentry TPS override must not double-count materials")

    def test_reentry_tps_bypass_structural(self):
        """TPS materials in reentry category should not fail due to low tensile strength.
        After Bug 2 fix, TPS lives in r.tps_coatings, not the primary lists."""
        r = match_materials(_run(20.0, 70.0, 500.0, 1.50, g_load=8.0), vehicle_category="reentry")
        pica_candidates = [c for c in r.tps_coatings if c.material.name == "PICA"]
        self.assertEqual(len(pica_candidates), 1,
            "PICA must appear in tps_coatings for reentry")
        self.assertEqual(pica_candidates[0].structural_status, "pass",
            "PICA in reentry category should have structural_status=pass (TPS structural bypass)")

    def test_turbine_excludes_aluminum(self):
        r = match_materials(_run(0.8, 11.0, 50000.0, 0.3, g_load=2.0), vehicle_category="turbine")
        all_eval = _all_evaluated(r)
        al_names = {c.material.name for c in all_eval if c.material.category == "aluminum"}
        self.assertEqual(len(al_names), 0, "Turbine category must not evaluate aluminum materials")

    def test_turbine_total_is_full_db(self):
        r = match_materials(_run(0.8, 11.0, 50000.0, 0.3, g_load=2.0), vehicle_category="turbine")
        total = len(r.viable) + len(r.marginal) + len(r.not_viable) + len(r.regime_rejected)
        self.assertEqual(total, len(MATERIALS_DB))

    def test_missile_total_is_full_db(self):
        r = match_materials(_run(4.0, 20.0, 900.0, 0.08, g_load=10.0), vehicle_category="hypersonic_missile")
        total = len(r.viable) + len(r.marginal) + len(r.not_viable) + len(r.regime_rejected)
        self.assertEqual(total, len(MATERIALS_DB))

    def test_aircraft_viable_sorted_by_density_ascending(self):
        """Aircraft viable list must be sorted lightest first (density ascending).
        Ti-6Al-4V (4430 kg/m3) must appear before Inconel 718 (8220 kg/m3) if both viable."""
        r = match_materials(_run(3.2, 25.0, 30600.0, 0.15, g_load=2.5), vehicle_category="aircraft")
        if len(r.viable) >= 2:
            for i in range(len(r.viable) - 1):
                self.assertLessEqual(
                    r.viable[i].material.density_kgm3,
                    r.viable[i + 1].material.density_kgm3,
                    f"Aircraft viable[{i}] ({r.viable[i].material.name}, "
                    f"{r.viable[i].material.density_kgm3} kg/m³) is denser than "
                    f"viable[{i+1}] ({r.viable[i+1].material.name}, "
                    f"{r.viable[i+1].material.density_kgm3} kg/m³)"
                )

    def test_aircraft_viable_first_lighter_than_inconel718(self):
        """First viable aircraft material must be lighter than Inconel 718 (8220 kg/m3)."""
        r = match_materials(_run(3.2, 25.0, 30600.0, 0.15, g_load=2.5), vehicle_category="aircraft")
        self.assertGreater(len(r.viable), 0, "SR-71 aircraft should have at least one viable material")
        first = r.viable[0]
        self.assertLess(
            first.material.density_kgm3, 8220.0,
            f"First aircraft viable material should be lighter than Inconel 718 (8220 kg/m³); "
            f"got {first.material.name} at {first.material.density_kgm3} kg/m³"
        )

    def test_hypersonic_missile_viable_sorted_by_density_ascending(self):
        """Hypersonic missile is mass-dominated like aircraft — viable list must be density ascending.
        Item 4 of the six-improvement plan: extend density-ascending sort to hypersonic_missile so
        small-margin dense superalloys do not falsely top the list when a lighter superalloy of the
        same family is also viable."""
        r = match_materials(_run(4.0, 20.0, 900.0, 0.08, g_load=10.0), vehicle_category="hypersonic_missile")
        if len(r.viable) >= 2:
            for i in range(len(r.viable) - 1):
                self.assertLessEqual(
                    r.viable[i].material.density_kgm3,
                    r.viable[i + 1].material.density_kgm3,
                    f"Missile viable[{i}] ({r.viable[i].material.name}, "
                    f"{r.viable[i].material.density_kgm3} kg/m³) is denser than "
                    f"viable[{i+1}] ({r.viable[i+1].material.name}, "
                    f"{r.viable[i+1].material.density_kgm3} kg/m³)"
                )
        if len(r.marginal) >= 2:
            for i in range(len(r.marginal) - 1):
                self.assertLessEqual(
                    r.marginal[i].material.density_kgm3,
                    r.marginal[i + 1].material.density_kgm3,
                    f"Missile marginal[{i}] is denser than marginal[{i+1}]",
                )

    def test_hypersonic_missile_first_viable_not_inconel718(self):
        """Item 4 SR-71-style sanity check for the missile: first viable should be the lightest
        nickel superalloy, not the densest. With score-ascending sort, Inconel 718 (8190 kg/m³)
        could top; with density-ascending sort, IN-100 (7750 kg/m³) tops instead."""
        r = match_materials(_run(4.0, 20.0, 900.0, 0.08, g_load=10.0), vehicle_category="hypersonic_missile")
        self.assertGreater(len(r.viable), 0, "Mach-4 missile should have at least one viable material")
        first = r.viable[0]
        # First viable cannot be denser than every other viable.
        all_densities = [c.material.density_kgm3 for c in r.viable]
        self.assertEqual(
            first.material.density_kgm3, min(all_densities),
            f"First missile viable should be the lightest viable; got {first.material.name} "
            f"at {first.material.density_kgm3} kg/m³ but min is {min(all_densities)} kg/m³"
        )

    def test_aircraft_density_ceiling(self):
        """No viable or marginal material may exceed 5000 kg/m³ for aircraft category."""
        r = match_materials(_run(3.2, 25.0, 77000.0, 0.15, g_load=2.5), vehicle_category="aircraft")
        above_threshold = [
            c for c in r.viable + r.marginal
            if c.material.density_kgm3 > 5000.0
        ]
        self.assertEqual(
            len(above_threshold), 0,
            f"Aircraft viable/marginal must not include materials > 5000 kg/m³; "
            f"found: {[c.material.name for c in above_threshold]}"
        )

    def test_missile_density_ceiling(self):
        """No viable or marginal material may exceed 8500 kg/m³ for hypersonic_missile category."""
        r = match_materials(_run(4.0, 20.0, 500.0, 0.05, g_load=20.0), vehicle_category="hypersonic_missile")
        above_threshold = [
            c for c in r.viable + r.marginal
            if c.material.density_kgm3 > 8500.0
        ]
        self.assertEqual(
            len(above_threshold), 0,
            f"Missile viable/marginal must not include materials > 8500 kg/m³; "
            f"found: {[c.material.name for c in above_threshold]}"
        )


    def test_aircraft_excludes_pmc_above_mach2(self):
        """Mach 2.5 aircraft: polymer composites excluded (matrix degradation + CTE artifact)."""
        r = match_materials(
            _run(2.5, 18.0, 45000.0, 0.20, g_load=3.0),
            vehicle_category="aircraft",
        )
        pmc_viable_marginal = [
            c for c in r.viable + r.marginal
            if c.material.category == "composite_polymer"
        ]
        self.assertEqual(len(pmc_viable_marginal), 0,
            f"Aircraft Mach 2.5: composite_polymer must not appear in viable/marginal; "
            f"found {[c.material.name for c in pmc_viable_marginal]}")

    def test_aircraft_includes_pmc_below_mach2(self):
        """Mach 0.85 aircraft: polymer composites should be available."""
        r = match_materials(
            _run(0.85, 10.0, 75000.0, 0.50, g_load=2.5),
            vehicle_category="aircraft",
        )
        all_eval = _all_evaluated(r)
        pmc_names = {c.material.name for c in all_eval if c.material.category == "composite_polymer"}
        self.assertGreater(len(pmc_names), 0,
            "Aircraft Mach 0.85: at least one composite_polymer should be evaluated")


# ---------------------------------------------------------------------------
# TestTurbinePreset — hot-section temperature override (Fix 9)
# ---------------------------------------------------------------------------
class TestTurbinePreset(unittest.TestCase):
    """Turbine HPT blade at Mach 0.5 / sea level: aerodynamic recovery
    temperature is ~300 K, but the blade actually sees ~1400 K (turbine
    inlet temperature minus film-cooling delta). The app-layer override
    rewrites T_wall and re-derives structural σ_req so the matching
    engine screens against hot-section conditions.
    """

    HOT_SECTION_K = 1400.0

    def _make_physics(self, override=True):
        from app import _apply_turbine_override
        physics = _run(0.5, 0.0, 50.0, 0.05, g_load=1.0)
        if override:
            physics = _apply_turbine_override(physics, self.HOT_SECTION_K)
        return physics

    def test_override_sets_T_wall_exactly(self):
        """Override sets T_wall to the supplied hot-section value."""
        physics = self._make_physics(override=True)
        self.assertEqual(physics.thermal.T_wall_K, self.HOT_SECTION_K)
        # Uncertainty band widened to ±5%
        self.assertAlmostEqual(
            physics.thermal.T_wall_min_K, self.HOT_SECTION_K * 0.95, delta=0.1
        )
        self.assertAlmostEqual(
            physics.thermal.T_wall_max_K, self.HOT_SECTION_K * 1.05, delta=0.1
        )

    def test_thermal_source_flag(self):
        """thermal_source reads 'turbine_inlet_override' when applied,
        'aerodynamic' otherwise."""
        default = self._make_physics(override=False)
        self.assertEqual(default.thermal.thermal_source, "aerodynamic")

        overridden = self._make_physics(override=True)
        self.assertEqual(
            overridden.thermal.thermal_source, "turbine_inlet_override"
        )

    def test_override_surfaces_high_temp_alloys(self):
        """After override, viable list contains nickel superalloys or CMCs
        (aluminum-grade materials must not dominate a 1400 K environment).
        """
        physics = self._make_physics(override=True)
        r = match_materials(physics, vehicle_category="turbine")
        all_surfaced = r.viable + r.marginal
        categories = {c.material.category for c in all_surfaced}
        # Must include at least one high-temp category
        self.assertTrue(
            categories & {"nickel_superalloy", "composite_ceramic",
                          "refractory", "carbon"},
            f"Turbine override at 1400 K must surface nickel superalloys or "
            f"CMCs; viable/marginal categories = {categories}",
        )
        # Aluminum ceiling (~700 K) would be overwhelmed at 1400 K → must NOT
        # appear in viable/marginal
        self.assertNotIn(
            "aluminum", categories,
            "Aluminum-grade materials must not survive 1400 K screening",
        )

    def test_structural_sigma_req_uses_override_delta_T(self):
        """σ_req recomputation uses ΔT = (T_hot − T_ambient) after override."""
        default = self._make_physics(override=False)
        overridden = self._make_physics(override=True)

        # Aerodynamic ΔT at M=0.5 / sea level is ~few K; override ΔT is ~1100 K.
        # So σ_thermal_ref (and therefore σ_tensile_required) must climb
        # markedly after the override.
        self.assertGreater(
            overridden.structural.sigma_thermal_ref_MPa,
            default.structural.sigma_thermal_ref_MPa + 100.0,
            "Override ΔT must raise σ_thermal_ref by at least 100 MPa "
            "over the aerodynamic-only baseline",
        )
        # Sanity: σ_thermal_ref ≈ 0.4 × 200 GPa × 12e-6 × (1400 − T_amb)
        delta_T = self.HOT_SECTION_K - overridden.thermal.T_ambient_K
        expected_sigma_th = 0.4 * 200_000.0 * 12e-6 * delta_T
        self.assertAlmostEqual(
            overridden.structural.sigma_thermal_ref_MPa,
            expected_sigma_th,
            delta=1.0,
        )


# ---------------------------------------------------------------------------
# TestGeneralStructurePreset — Fix 10 preset sanity check
# ---------------------------------------------------------------------------
class TestGeneralStructurePreset(unittest.TestCase):
    """General Structure Panel preset: Mach 0.3, 5 km, 500 kg, R_n=0.5,
    char_len=2.0, category='general'. This is a subsonic panel case with
    no exclusions — ordinary engineering alloys should dominate the
    viable list.
    """

    def test_general_structure_returns_common_alloys(self):
        r = match_materials(
            _run(0.3, 5.0, 500.0, 0.5, g_load=2.0),
            vehicle_category="general",
        )
        all_surfaced = {
            c.material.category for c in r.viable + r.marginal
        }
        # At subsonic / low-temp panel conditions, steels, aluminums and
        # titaniums must all be eligible candidates.
        self.assertTrue(
            {"aluminum", "titanium"} & all_surfaced or "steel" in all_surfaced,
            f"General structure viable list must include common engineering "
            f"alloys (steel / aluminum / titanium); saw {all_surfaced}",
        )

    def test_general_structure_no_regime_exclusions(self):
        """General category does not filter categories — every regime-eligible
        material reaches evaluation."""
        r = match_materials(
            _run(0.3, 5.0, 500.0, 0.5, g_load=2.0),
            vehicle_category="general",
        )
        # No category-exclusion suppression at subsonic conditions for
        # 'general' — the only filter is regime applicability.
        total = _total_direct_count(r)
        self.assertEqual(total, 97)


# ---------------------------------------------------------------------------
# TestCreepStage — Phase 3 lifecycle integration
# ---------------------------------------------------------------------------
class TestCreepStage(unittest.TestCase):
    """The creep stage runs after thermal+structural and gates the
    overall_status. At ``design_lifetime_hours=1.0`` (default,
    single-flight) the creep stage is a near-no-op so the viable
    list matches pre-Phase-3 behaviour. At long lifetimes the stage
    correctly downgrades materials whose rupture stress falls below
    sigma_required."""

    def _concorde_physics(self):
        # M=2.04, 18 km, 78,000 kg, R_n=0.40 — matches VALIDATION.md.
        return run_analysis(2.04, 18.0, 78000.0, 0.40, peak_g_load=2.0)

    def _turbine_physics(self):
        from core.api import apply_turbine_override
        p = run_analysis(0.5, 0.0, 50.0, 0.005, peak_g_load=1.0)
        return apply_turbine_override(p, 1400.0)

    def test_default_lifetime_does_not_change_viable_list_for_subsonic(self):
        """At lifetime=1.0 h (single-flight), running with creep
        should produce the same viable count as without explicitly
        passing the kwarg — because the default IS 1.0. This is the
        backward-compat regression guard."""
        physics = run_analysis(0.3, 5.0, 500.0, 0.5, peak_g_load=2.0)
        a = match_materials(physics, vehicle_category="general")
        b = match_materials(
            physics, vehicle_category="general",
            design_lifetime_hours=1.0,
        )
        self.assertEqual(len(a.viable), len(b.viable))
        self.assertEqual(len(a.marginal), len(b.marginal))

    def test_concorde_long_lifetime_demotes_aluminum(self):
        """At Concorde lifetime (25,000 h) Al 2024-T3 must NOT appear
        in the viable list — its rupture stress at 100 C * 25,000 h is
        below the sigma_required at the Concorde envelope. This is
        the historically-correct outcome that drove development of
        Hiduminium RR58 / Al 2618."""
        physics = self._concorde_physics()
        result = match_materials(
            physics, vehicle_category="aircraft",
            design_lifetime_hours=25000.0,
        )
        viable_names = {c.material.name for c in result.viable}
        self.assertNotIn(
            "2024-T3", viable_names,
            f"At Concorde lifetime (25,000 h) Al 2024-T3 should not "
            f"be viable. Viable list: {sorted(viable_names)}",
        )

    def test_concorde_short_lifetime_keeps_aluminum(self):
        """Same Concorde envelope but at 1 h — Al 2024-T3 should
        survive (or at worst be marginal). At single-flight the creep
        stage doesn't bite."""
        physics = self._concorde_physics()
        result = match_materials(
            physics, vehicle_category="aircraft",
            design_lifetime_hours=1.0,
        )
        viable_or_marginal = (
            [c.material.name for c in result.viable]
            + [c.material.name for c in result.marginal]
        )
        self.assertIn(
            "2024-T3", viable_or_marginal,
            "At single-flight lifetime Al 2024-T3 should survive at "
            "Concorde's 100 C wall temperature.",
        )

    def test_turbine_long_lifetime_keeps_cmsx4(self):
        """CFM56-class turbine at 25,000 h * 1400 K must keep
        CMSX-4 viable — it is the canonical commercial single-crystal
        nickel for this exact duty cycle."""
        physics = self._turbine_physics()
        result = match_materials(
            physics, vehicle_category="turbine",
            design_lifetime_hours=25000.0,
        )
        viable_or_marginal = (
            [c.material.name for c in result.viable]
            + [c.material.name for c in result.marginal]
        )
        self.assertIn(
            "CMSX-4", viable_or_marginal,
            f"CMSX-4 should survive 25,000 h at 1400 K (it's "
            f"specifically designed for this). Viable+marginal: "
            f"{viable_or_marginal[:10]}",
        )

    def test_creep_status_populated_on_every_candidate(self):
        physics = run_analysis(0.3, 5.0, 500.0, 0.5, peak_g_load=2.0)
        result = match_materials(
            physics, vehicle_category="general",
            design_lifetime_hours=10.0,
        )
        all_cands = result.viable + result.marginal + result.not_viable
        self.assertGreater(len(all_cands), 0)
        valid = {"pass", "marginal", "fail", "unknown", "not_applicable"}
        for c in all_cands:
            self.assertIn(c.creep_status, valid,
                f"{c.material.name}: creep_status={c.creep_status!r}")

    def test_unknown_creep_does_not_auto_reject(self):
        """A material with creep_data_status="unknown" that's
        otherwise viable on thermal+structural should stay in viable
        (just flagged in notes), not move to not_viable."""
        physics = run_analysis(0.3, 5.0, 500.0, 0.5, peak_g_load=2.0)
        result = match_materials(
            physics, vehicle_category="general",
            design_lifetime_hours=10.0,
        )
        # Look for any candidate with creep_status="unknown" in the
        # viable list — at this low envelope plenty of refractory /
        # cobalt / niche-aluminum entries should still survive.
        unknown_in_viable = [
            c for c in result.viable if c.creep_status == "unknown"
        ]
        # Not strict — depends on the database. The invariant is that
        # NO unknown-creep candidate ends up in not_viable purely
        # because of the unknown flag (would need a separate flag to
        # distinguish, which the API explicitly doesn't do — unknown
        # is pass-through).
        for c in result.not_viable:
            if c.creep_status == "unknown":
                # This is fine if thermal_status or structural_status
                # is "fail" — the unknown creep didn't cause it.
                self.assertTrue(
                    c.thermal_status == "fail"
                    or c.structural_status == "fail",
                    f"{c.material.name}: rejected with creep_status="
                    f"unknown but thermal={c.thermal_status} "
                    f"structural={c.structural_status} — unknown creep "
                    "should be pass-through, not auto-reject",
                )

    def test_transient_stage_runs_for_short_flight(self):
        """Sounding rocket envelope: peak_backface_K should be
        populated and lower than steady-state T_wall ≈ 386 K."""
        from dataclasses import replace
        physics = run_analysis(2.0, 9.0, 30.0, 0.05, peak_g_load=15.0)
        # Run_analysis defaults flight_duration to 600 s; override
        # to the realistic IREC boost-coast duration so the
        # transient trigger fires (< 300 s threshold).
        physics = replace(physics, flight_duration_s=25.0)
        result = match_materials(
            physics, vehicle_category="general",
            panel_thickness_m=0.003,
        )
        # Find an aluminum candidate to inspect.
        al_cands = [
            c for c in result.viable + result.marginal
            if "Al" in c.material.name or "2024" in c.material.name
            or "6061" in c.material.name
        ]
        self.assertGreater(
            len(al_cands), 0,
            "Expected aluminum candidates at sounding-rocket envelope.",
        )
        applied = [c for c in al_cands if c.transient_status == "applied"]
        self.assertGreater(
            len(applied), 0,
            "Expected at least one applied transient-status for aluminum "
            "at short-duration flight.",
        )
        for c in applied:
            self.assertIsNotNone(c.transient_peak_backface_K)
            # Backface peak should be well below 386 K (steady-state
            # recovery temperature at this envelope).
            self.assertLess(
                c.transient_peak_backface_K, 386.0,
                f"{c.material.name}: transient backface peak "
                f"{c.transient_peak_backface_K:.1f} K should be below "
                f"steady-state 386 K for 25 s flight.",
            )

    def test_transient_stage_skipped_for_sustained_flight(self):
        """SR-71 90 min cruise: transient stage should NOT trigger
        (flight_duration > 300 s threshold)."""
        physics = run_analysis(3.2, 25.0, 30600.0, 0.15, peak_g_load=2.5)
        # Override the default flight_duration_s in the physics result
        # via the engine's parameter.
        from dataclasses import replace
        physics = replace(physics, flight_duration_s=5400.0)
        result = match_materials(
            physics, vehicle_category="aircraft",
            panel_thickness_m=0.0015,
        )
        all_cands = (
            result.viable + result.marginal + result.not_viable
        )
        applied = [c for c in all_cands if c.transient_status == "applied"]
        self.assertEqual(
            len(applied), 0,
            "Transient stage should not run for sustained-flight "
            "(>300 s) envelopes; the static T_wall check is sufficient.",
        )

    def test_transient_stage_short_circuits_for_tps(self):
        physics = run_analysis(2.0, 9.0, 30.0, 0.05, peak_g_load=15.0)
        result = match_materials(
            physics, vehicle_category="reentry",
            panel_thickness_m=0.003,
        )
        for c in result.tps_coatings:
            self.assertEqual(
                c.transient_status, "not_applicable",
                f"TPS material {c.material.name}: expected transient "
                f"status='not_applicable'",
            )

    def test_zero_lifetime_clamps_silently(self):
        """A user passing lifetime=0 (typo, missing form value) must
        not crash the pipeline. The matching engine clamps to the
        single-flight default and runs through."""
        physics = self._concorde_physics()
        a = match_materials(
            physics, vehicle_category="aircraft",
            design_lifetime_hours=0.0,
        )
        b = match_materials(
            physics, vehicle_category="aircraft",
            design_lifetime_hours=1.0,
        )
        # Same number of viable + marginal at single-flight behaviour.
        self.assertEqual(len(a.viable), len(b.viable))
        self.assertEqual(len(a.marginal), len(b.marginal))

    def test_negative_lifetime_clamps_silently(self):
        physics = self._concorde_physics()
        result = match_materials(
            physics, vehicle_category="aircraft",
            design_lifetime_hours=-1000.0,
        )
        # Doesn't crash, produces a sensible result.
        self.assertGreaterEqual(len(result.viable), 0)

    def test_extreme_long_lifetime_does_not_crash(self):
        """At 1,000,000 h far beyond any real airframe, creep
        extrapolation should not blow up. Plenty of materials
        will fail creep, but the engine returns a coherent result."""
        physics = self._concorde_physics()
        result = match_materials(
            physics, vehicle_category="aircraft",
            design_lifetime_hours=1_000_000.0,
        )
        self.assertGreaterEqual(len(result.viable), 0)
        self.assertGreaterEqual(len(result.not_viable), 0)

    def test_tps_materials_carry_not_applicable_creep_status(self):
        """TPS coatings always get creep_status='not_applicable'
        regardless of lifetime — they're non-load-bearing."""
        # Reentry envelope unlocks TPS materials.
        physics = run_analysis(20.0, 70.0, 500.0, 1.5, peak_g_load=8.0)
        result = match_materials(
            physics, vehicle_category="reentry",
            design_lifetime_hours=25000.0,
        )
        for c in result.tps_coatings:
            self.assertEqual(
                c.creep_status, "not_applicable",
                f"TPS material {c.material.name} should always be "
                f"creep_status='not_applicable', got {c.creep_status}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
