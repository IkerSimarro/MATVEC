"""
Test suite for core/surrogate.py — Phase 6 of MATVEC.
stdlib unittest only. Run: python -m unittest test_surrogate.py -v
"""

import unittest
import copy

from materials_db import MATERIALS_DB, MaterialEntry
from physics_engine import run_analysis
from matching_engine import match_materials
from core.surrogate import (
    build_surrogate,
    find_nearest_candidates,
    SurrogateResult,
    get_model_version,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(mach, alt_km, mass_kg, R_n, g_load=1.0):
    return run_analysis(mach, alt_km, mass_kg, R_n, peak_g_load=g_load)


# ---------------------------------------------------------------------------
# TestSurrogateProperties — 5 tests
# ---------------------------------------------------------------------------

class TestSurrogateProperties(unittest.TestCase):
    """Structural invariants of the surrogate model."""

    def test_surrogate_builds_without_error(self):
        scaler, matrix, mats, version = build_surrogate(MATERIALS_DB)
        self.assertEqual(matrix.shape[0], len(MATERIALS_DB))
        self.assertEqual(matrix.shape[1], 7)

    def test_find_nearest_returns_k_results(self):
        physics = _run(3.2, 25.0, 30600.0, 0.15, 2.5)
        result = match_materials(physics, vehicle_category="aircraft")
        surr = find_nearest_candidates(physics, "aircraft", match_result=result, k=10)
        self.assertLessEqual(len(surr.candidates), 10)
        self.assertGreaterEqual(len(surr.candidates), 1)
        self.assertEqual(len(surr.candidates), len(surr.distances))

    def test_distances_are_non_negative(self):
        physics = _run(6.7, 30.0, 15195.0, 0.30, 5.0)
        surr = find_nearest_candidates(physics, "aircraft", k=10)
        for d in surr.distances:
            self.assertGreaterEqual(d, 0.0)

    def test_distances_sorted_ascending(self):
        physics = _run(6.7, 30.0, 15195.0, 0.30, 5.0)
        surr = find_nearest_candidates(physics, "aircraft", k=10)
        for i in range(len(surr.distances) - 1):
            self.assertLessEqual(
                surr.distances[i], surr.distances[i + 1],
                f"Distance at index {i} not <= distance at index {i+1}",
            )

    def test_model_version_is_hash(self):
        version = get_model_version()
        self.assertEqual(len(version), 64)
        # Must be valid hex
        int(version, 16)


# ---------------------------------------------------------------------------
# TestSurrogateReferenceVehicles — 4 tests
# ---------------------------------------------------------------------------

class TestSurrogateReferenceVehicles(unittest.TestCase):
    """Surrogate results for reference vehicle scenarios."""

    def test_sr71_surrogate_includes_titanium(self):
        physics = _run(3.2, 25.0, 30600.0, 0.15, 2.5)
        result = match_materials(physics, vehicle_category="aircraft")
        surr = find_nearest_candidates(physics, "aircraft", match_result=result, k=5)
        cats = {m.category for m in surr.candidates}
        self.assertIn("titanium", cats)

    def test_x15_surrogate_includes_high_temp_alloy(self):
        physics = _run(6.7, 30.0, 15195.0, 0.30, 5.0)
        result = match_materials(physics, vehicle_category="aircraft")
        surr = find_nearest_candidates(physics, "aircraft", match_result=result, k=5)
        max_service = max(m.service_temp_air_K for m in surr.candidates)
        self.assertGreater(
            max_service, 1100.0,
            "X-15 top-5 surrogate should include at least one material with "
            f"service_temp > 1100 K, best was {max_service} K",
        )

    def test_agreement_range(self):
        physics = _run(3.2, 25.0, 30600.0, 0.15, 2.5)
        result = match_materials(physics, vehicle_category="aircraft")
        surr = find_nearest_candidates(physics, "aircraft", match_result=result, k=10)
        self.assertGreaterEqual(surr.agreement_with_margin_ranking, 0.0)
        self.assertLessEqual(surr.agreement_with_margin_ranking, 1.0)

    def test_model_version_changes_with_database(self):
        """Modifying a material property changes the model version hash."""
        original_version = get_model_version()
        # Create a modified copy of the database
        modified_db = []
        for m in MATERIALS_DB:
            modified_db.append(copy.copy(m))
        # Change one property
        object.__setattr__(modified_db[0], "density_kgm3", 9999.0)
        _, _, _, new_version = build_surrogate(modified_db)
        self.assertNotEqual(original_version, new_version)


# ---------------------------------------------------------------------------
# TestSurrogateCategoryFilter — filter exotic/off-category classes
# ---------------------------------------------------------------------------

class TestSurrogateCategoryFilter(unittest.TestCase):
    """The surrogate must apply the same category-exclusion rules as the
    matching engine, so that an airliner query at Mach 2.5 does not surface
    Niobium C-103 or HY-80 submarine steel in its top-k nearest neighbors.

    These tests exercise `find_nearest_candidates` directly, using the
    matching-engine-derived exclusion set (not a frozen dict).
    """

    def test_aircraft_supersonic_excludes_refractory_and_polymer_composite(self):
        """Aircraft at Mach 2.5 must not return refractory or polymer composites.

        Polymer composites are excluded at Mach >= 2 by the Mach-dependent
        rule inside _get_category_exclusions; refractory is always excluded
        for aircraft.
        """
        physics = _run(2.5, 18.0, 45000.0, 0.4, g_load=2.0)
        result = match_materials(physics, vehicle_category="aircraft")
        surr = find_nearest_candidates(
            physics, "aircraft", match_result=result, k=10,
        )
        cats = {m.category for m in surr.candidates}
        self.assertNotIn("refractory", cats,
                         "Aircraft surrogate must not return refractory metals")
        self.assertNotIn("composite_polymer", cats,
                         "Aircraft at M>=2 must not return polymer composites")
        # General-engineering steels (submarine-grade HY-80 etc.) are also
        # excluded for aircraft — they are in _CATEGORY_EXCLUSIONS.
        self.assertNotIn("general_engineering", cats,
                         "Aircraft surrogate must not return general-engineering steels")

    def test_turbine_excludes_aluminum_and_polymer(self):
        """Turbine category must not return aluminum / polymer composites / TPS."""
        physics = _run(0.5, 0.0, 50.0, 0.05, g_load=1.0)
        result = match_materials(physics, vehicle_category="turbine")
        surr = find_nearest_candidates(
            physics, "turbine", match_result=result, k=10,
        )
        cats = {m.category for m in surr.candidates}
        for excluded in ("aluminum", "composite_polymer", "tps",
                         "general_engineering"):
            self.assertNotIn(
                excluded, cats,
                f"Turbine surrogate must not return {excluded}",
            )

    def test_reentry_allows_refractory_and_uhtc(self):
        """Reentry exclusion set is empty — refractory and UHTC must stay eligible."""
        physics = _run(20.0, 70.0, 500.0, 1.50, g_load=8.0)
        result = match_materials(physics, vehicle_category="reentry")
        surr = find_nearest_candidates(
            physics, "reentry", match_result=result, k=10,
        )
        # Can't guarantee refractory appears in top-k (distance-dependent),
        # but the suppression count must be 0 — the reentry exclusion set is empty.
        self.assertEqual(
            surr.suppressed_count, 0,
            "Reentry must not suppress any regime-eligible categories",
        )

    def test_suppressed_count_is_positive_for_filtered_category(self):
        """Aircraft at supersonic Mach must report suppressed_count > 0."""
        physics = _run(2.5, 18.0, 45000.0, 0.4, g_load=2.0)
        result = match_materials(physics, vehicle_category="aircraft")
        surr = find_nearest_candidates(
            physics, "aircraft", match_result=result, k=10,
        )
        self.assertGreater(
            surr.suppressed_count, 0,
            "Aircraft surrogate should suppress refractory/submarine-grade "
            "materials from the regime pool",
        )

    def test_fallback_flag_default_false(self):
        """Well-populated vehicle categories should not trigger the fallback."""
        physics = _run(2.5, 18.0, 45000.0, 0.4, g_load=2.0)
        result = match_materials(physics, vehicle_category="aircraft")
        surr = find_nearest_candidates(
            physics, "aircraft", match_result=result, k=10,
        )
        self.assertFalse(
            surr.fallback_used,
            "Aircraft at M=2.5 has plenty of eligible materials --- no fallback",
        )
        self.assertGreater(len(surr.candidates), 0)


if __name__ == "__main__":
    unittest.main()
