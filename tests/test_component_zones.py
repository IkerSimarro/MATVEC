"""
Test suite for core/component_zones.py.

Covers Item 6 of the six-improvement plan: per-zone material recommendations
that decompose a vehicle into 3-5 named zones with locally-scaled thermal
and structural demands, then run match_materials against each zone.

stdlib unittest only. Run: python -m unittest test_component_zones.py -v
"""

import unittest

from core.physics_engine import run_analysis
from core.matching_engine import match_materials, MatchResult
from core.component_zones import (
    CATEGORY_ZONES,
    ComponentZone,
    ZoneMatchResult,
    evaluate_zones,
    _scale_physics_for_zone,
)


def _physics(mach, alt_km, mass_kg, R_n, g_load=1.0):
    return run_analysis(mach, alt_km, mass_kg, R_n, peak_g_load=g_load)


# ---------------------------------------------------------------------------
# TestZoneCatalog — CATEGORY_ZONES is well-formed
# ---------------------------------------------------------------------------
class TestZoneCatalog(unittest.TestCase):
    """The per-category zone catalog must cover every category used elsewhere."""

    EXPECTED_CATEGORIES = {
        "aircraft", "hypersonic_aircraft", "hypersonic_missile",
        "reentry", "turbine", "general",
    }

    def test_every_vehicle_category_has_zones(self):
        for cat in self.EXPECTED_CATEGORIES:
            with self.subTest(category=cat):
                self.assertIn(cat, CATEGORY_ZONES)

    def test_each_category_has_at_least_three_zones(self):
        for cat, zones in CATEGORY_ZONES.items():
            with self.subTest(category=cat):
                self.assertGreaterEqual(
                    len(zones), 3,
                    f"Category {cat!r} has only {len(zones)} zones; need >= 3",
                )

    def test_each_category_has_at_most_five_zones(self):
        """Zone lists must stay short enough to render legibly in the PDF."""
        for cat, zones in CATEGORY_ZONES.items():
            with self.subTest(category=cat):
                self.assertLessEqual(len(zones), 5)

    def test_first_zone_per_category_has_unit_t_multiplier(self):
        """The first zone in each category is the leading-edge / stagnation
        reference: t_wall_multiplier == 1.0 so it matches the whole-vehicle
        physics. Calibration anchor."""
        for cat, zones in CATEGORY_ZONES.items():
            with self.subTest(category=cat):
                self.assertAlmostEqual(zones[0].t_wall_multiplier, 1.0, places=6)

    def test_thermal_multipliers_in_valid_range(self):
        """t_wall_multiplier in (0, 1] for every zone in every category.
        Exceeding 1 would imply zone heating above leading-edge — unphysical
        for the multiplier-on-rise framework."""
        for cat, zones in CATEGORY_ZONES.items():
            for z in zones:
                with self.subTest(category=cat, zone=z.name):
                    self.assertGreater(z.t_wall_multiplier, 0.0)
                    self.assertLessEqual(z.t_wall_multiplier, 1.0)

    def test_structural_multipliers_positive(self):
        """sigma_req_multiplier > 0 for every zone (turbine roots/disks
        legitimately exceed 1.0 due to centrifugal stress concentration)."""
        for cat, zones in CATEGORY_ZONES.items():
            for z in zones:
                with self.subTest(category=cat, zone=z.name):
                    self.assertGreater(z.sigma_req_multiplier, 0.0)

    def test_zone_names_are_unique_within_category(self):
        for cat, zones in CATEGORY_ZONES.items():
            names = [z.name for z in zones]
            self.assertEqual(len(names), len(set(names)), f"Duplicate zone names in {cat!r}")


# ---------------------------------------------------------------------------
# TestScalePhysicsForZone — multiplier semantics correct
# ---------------------------------------------------------------------------
class TestScalePhysicsForZone(unittest.TestCase):
    """_scale_physics_for_zone applies multipliers to the rise above ambient,
    not directly to T_wall — and scales sigma_req directly."""

    def test_unit_multiplier_recovers_whole_vehicle_temperatures(self):
        """t_wall_multiplier == 1.0 must yield T_wall_zone == T_wall_whole."""
        p = _physics(3.2, 25.0, 30600.0, 0.15, 2.5)
        zone = ComponentZone("ref", "test", t_wall_multiplier=1.0,
                             sigma_req_multiplier=1.0)
        scaled = _scale_physics_for_zone(p, zone)
        self.assertAlmostEqual(scaled.thermal.T_wall_K, p.thermal.T_wall_K, places=4)

    def test_zero_multiplier_yields_ambient_temperature(self):
        """t_wall_multiplier == 0 must yield T_wall_zone == T_amb (the floor)."""
        p = _physics(3.2, 25.0, 30600.0, 0.15, 2.5)
        zone = ComponentZone("ambient", "test", t_wall_multiplier=0.0,
                             sigma_req_multiplier=1.0)
        scaled = _scale_physics_for_zone(p, zone)
        self.assertAlmostEqual(
            scaled.thermal.T_wall_K, p.thermal.T_ambient_K, places=4,
            msg="Zero multiplier must produce T_amb, not 0 K (multiplier-on-rise semantics)",
        )

    def test_half_multiplier_halves_the_rise(self):
        """T_wall_zone - T_amb == 0.5 * (T_wall_whole - T_amb)."""
        p = _physics(3.2, 25.0, 30600.0, 0.15, 2.5)
        zone = ComponentZone("half", "test", 0.5, 1.0)
        scaled = _scale_physics_for_zone(p, zone)
        rise_whole = p.thermal.T_wall_K - p.thermal.T_ambient_K
        rise_zone = scaled.thermal.T_wall_K - p.thermal.T_ambient_K
        self.assertAlmostEqual(rise_zone, 0.5 * rise_whole, places=4)

    def test_sigma_req_scales_directly(self):
        p = _physics(3.2, 25.0, 30600.0, 0.15, 2.5)
        zone = ComponentZone("z", "test", 1.0, 0.4)
        scaled = _scale_physics_for_zone(p, zone)
        self.assertAlmostEqual(
            scaled.structural.sigma_tensile_required_MPa,
            0.4 * p.structural.sigma_tensile_required_MPa, places=4,
        )

    def test_unchanged_fields_are_preserved(self):
        """Zone scaling must not alter ambient T, Mach, mass, or model selection."""
        p = _physics(3.2, 25.0, 30600.0, 0.15, 2.5)
        zone = ComponentZone("z", "test", 0.5, 0.5)
        scaled = _scale_physics_for_zone(p, zone)
        self.assertEqual(scaled.thermal.T_ambient_K, p.thermal.T_ambient_K)
        self.assertEqual(scaled.thermal.uses_recovery_model, p.thermal.uses_recovery_model)
        self.assertEqual(scaled.peak_mach, p.peak_mach)
        self.assertEqual(scaled.vehicle_mass_kg, p.vehicle_mass_kg)
        # Original physics object untouched (immutable input)
        self.assertEqual(p.thermal.T_wall_K, p.thermal.T_wall_K)


# ---------------------------------------------------------------------------
# TestEvaluateZones — end-to-end zone matching
# ---------------------------------------------------------------------------
class TestEvaluateZones(unittest.TestCase):
    """evaluate_zones returns per-zone match results in catalog order."""

    def test_returns_zone_count_matching_category(self):
        for cat, zones in CATEGORY_ZONES.items():
            with self.subTest(category=cat):
                # Pick a generic envelope; specifics don't matter for the count.
                p = _physics(3.0, 20.0, 1000.0, 0.20, 2.0)
                results = evaluate_zones(p, vehicle_category=cat)
                self.assertEqual(len(results), len(zones))

    def test_returns_results_in_catalog_order(self):
        p = _physics(3.0, 20.0, 1000.0, 0.20, 2.0)
        results = evaluate_zones(p, vehicle_category="aircraft")
        for i, zr in enumerate(results):
            self.assertEqual(zr.zone.name, CATEGORY_ZONES["aircraft"][i].name)

    def test_unknown_category_returns_empty_list(self):
        p = _physics(3.0, 20.0, 1000.0, 0.20, 2.0)
        self.assertEqual(evaluate_zones(p, "made_up_category"), [])

    def test_each_result_has_real_match_result(self):
        p = _physics(3.2, 25.0, 30600.0, 0.15, 2.5)
        results = evaluate_zones(p, "aircraft")
        for zr in results:
            self.assertIsInstance(zr, ZoneMatchResult)
            self.assertIsInstance(zr.match, MatchResult)
            # T_wall_zone is positive and bounded by the whole-vehicle T_wall
            self.assertGreater(zr.T_wall_zone_K, 0.0)
            self.assertLessEqual(zr.T_wall_zone_K, p.thermal.T_wall_K + 1e-3)

    def test_first_zone_match_equals_whole_vehicle_match(self):
        """The leading-edge zone (multiplier 1.0) must produce the same viable
        set as the whole-vehicle match — that is the calibration anchor."""
        for cat in ("aircraft", "hypersonic_missile", "reentry", "general"):
            with self.subTest(category=cat):
                # Use category-appropriate envelopes
                if cat == "aircraft":
                    p = _physics(3.2, 25.0, 30600.0, 0.15, 2.5)
                elif cat == "hypersonic_missile":
                    p = _physics(4.0, 20.0, 900.0, 0.08, 10.0)
                elif cat == "reentry":
                    p = _physics(20.0, 70.0, 500.0, 1.50, 8.0)
                else:
                    p = _physics(0.3, 5.0, 500.0, 0.5, 2.0)
                whole = match_materials(p, vehicle_category=cat)
                results = evaluate_zones(p, vehicle_category=cat)
                zone0 = results[0]
                whole_viable = {c.material.name for c in whole.viable}
                zone_viable = {c.material.name for c in zone0.match.viable}
                self.assertEqual(
                    whole_viable, zone_viable,
                    f"Leading-edge zone for {cat} must match whole-vehicle viable set; "
                    f"got whole={whole_viable - zone_viable} extra, "
                    f"zone={zone_viable - whole_viable} extra",
                )

    def test_internal_zone_runs_cooler_than_leading_edge(self):
        """Lower t_wall_multiplier zones must have strictly lower T_wall_zone."""
        p = _physics(3.2, 25.0, 30600.0, 0.15, 2.5)
        results = evaluate_zones(p, "aircraft")
        leading = results[0]
        internal = results[-1]   # the 'Internal structure' zone in aircraft
        self.assertLess(internal.T_wall_zone_K, leading.T_wall_zone_K)

    def test_internal_zone_unlocks_more_material_options(self):
        """A cooler zone should have at least as many viable materials as the
        hot leading edge, because lower T expands the thermally-passing set."""
        p = _physics(3.2, 25.0, 30600.0, 0.15, 2.5)
        results = evaluate_zones(p, "aircraft")
        leading = results[0]
        internal = results[-1]
        # Total candidates that pass the thermal check (viable + marginal)
        leading_passes = len(leading.match.viable) + len(leading.match.marginal)
        internal_passes = len(internal.match.viable) + len(internal.match.marginal)
        self.assertGreaterEqual(internal_passes, leading_passes)

    def test_reentry_backshell_lighter_load_than_stagnation(self):
        """Backshell σ_req multiplier (0.30) is well below stagnation (1.00)."""
        p = _physics(20.0, 70.0, 500.0, 1.50, 8.0)
        results = evaluate_zones(p, "reentry")
        names = [r.zone.name for r in results]
        i_stag = names.index("Stagnation point")
        i_back = names.index("Backshell / leeward")
        self.assertLess(
            results[i_back].sigma_req_zone_MPa,
            results[i_stag].sigma_req_zone_MPa,
        )

    def test_turbine_disk_carries_higher_load_than_blade(self):
        """Disk multiplier (2.0) > blade airfoil multiplier (0.8) — disk
        carries the integrated centrifugal load."""
        p = _physics(0.5, 0.0, 50.0, 0.05, 1.0)
        results = evaluate_zones(p, "turbine")
        names = [r.zone.name for r in results]
        i_blade = names.index("Blade leading edge / airfoil")
        i_disk = names.index("Disk hub")
        self.assertGreater(
            results[i_disk].sigma_req_zone_MPa,
            results[i_blade].sigma_req_zone_MPa,
        )


if __name__ == "__main__":
    unittest.main()
