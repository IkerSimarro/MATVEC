"""
Test suite for materials_db.py
================================
Runs with stdlib unittest only — no external dependencies required.

Usage:
    python -m unittest test_materials_db.py -v

All 7 test classes must pass before proceeding to Step 2 (physics engine).
If any test fails, fix the data in materials_db.py — do not adjust tests
to pass incorrect data.
"""

import math
import unittest

import core.materials_db as db
from core.materials_db import (
    MATERIALS_DB,
    VALID_CATEGORIES,
    VALID_REGIMES,
    MaterialEntry,
    get_materials_by_category,
    get_materials_by_regime,
    get_strength_at_temperature,
)

# ---------------------------------------------------------------------------
# Expected counts per category
# ---------------------------------------------------------------------------
EXPECTED_COUNTS = {
    "aluminum": 10,
    "titanium": 9,
    "nickel": 18,
    "steel": 11,
    "cobalt": 4,
    "composite_polymer": 7,
    "composite_ceramic": 5,
    "refractory": 5,
    "uhtc": 8,
    "carbon": 3,
    "tps": 10,
    "general_engineering": 7,
}

# Single-crystal nickel alloys exhibit an anomalous yield-stress peak near
# 800°C before dropping sharply — physically correct, whitelisted here.
# Carbon-carbon and pyrolytic graphite also strengthen slightly to ~1000°C
# before declining — documented behavior in graphitic carbon microstructure.
MONOTONICITY_EXCEPTIONS = {
    "CMSX-4",
    "PWA 1484",
    "Carbon-Carbon Composite",
    "Pyrolytic Graphite",
    "RCC",   # SiC-coated C/C shares the graphitic strengthening behavior to ~1000°C
}


# ===========================================================================
# 1. Database Completeness
# ===========================================================================

class TestDatabaseCompleteness(unittest.TestCase):

    def test_total_entry_count(self):
        """Database must contain exactly 97 entries."""
        self.assertEqual(len(MATERIALS_DB), 97,
            f"Expected 97 entries, got {len(MATERIALS_DB)}")

    def test_category_counts(self):
        """Each category must contain the expected number of entries."""
        for cat, expected in EXPECTED_COUNTS.items():
            actual = len(get_materials_by_category(cat))
            self.assertEqual(actual, expected,
                f"Category '{cat}': expected {expected}, got {actual}")

    def test_no_none_fields(self):
        """All fields on every entry must be non-None — except for
        the optional creep-data field ``larson_miller_C``, which is
        deliberately None for materials whose creep_data_status is
        ``"unknown"`` or ``"not_applicable"`` (no LMP curve to fit
        a constant against).
        """
        OPTIONAL_NULLABLE = {"larson_miller_C", "specific_heat_J_kgK"}
        for mat in MATERIALS_DB:
            for field in mat.__dataclass_fields__:
                if field in OPTIONAL_NULLABLE:
                    continue
                value = getattr(mat, field)
                self.assertIsNotNone(value,
                    f"{mat.name}: field '{field}' is None")

    def test_no_empty_strings(self):
        """String fields must not be empty."""
        string_fields = ("name", "category", "oxidation_resistance",
                         "citation", "notes")
        for mat in MATERIALS_DB:
            for field in string_fields:
                value = getattr(mat, field)
                self.assertIsInstance(value, str,
                    f"{mat.name}: field '{field}' is not a string")
                self.assertGreater(len(value.strip()), 0,
                    f"{mat.name}: field '{field}' is empty or whitespace-only")

    def test_minimum_temp_data_points(self):
        """Every material must have at least 3 temperature data points."""
        for mat in MATERIALS_DB:
            n = len(mat.tensile_strength_at_temp)
            self.assertGreaterEqual(n, 3,
                f"{mat.name}: tensile_strength_at_temp has only {n} points (need >= 3)")

    def test_no_nan_or_inf_in_numeric_fields(self):
        """No numeric field may be NaN or infinite."""
        float_fields = (
            "density_kgm3", "tensile_strength_mpa", "compressive_strength_mpa",
            "service_temp_air_K", "service_temp_inert_K", "melting_point_K",
            "thermal_conductivity_WmK", "thermal_expansion_1K",
            "youngs_modulus_GPa", "oxidation_max_temp_K",
        )
        for mat in MATERIALS_DB:
            for field in float_fields:
                v = getattr(mat, field)
                self.assertFalse(math.isnan(v),
                    f"{mat.name}: field '{field}' is NaN")
                self.assertFalse(math.isinf(v),
                    f"{mat.name}: field '{field}' is infinite")
            for T, S in mat.tensile_strength_at_temp.items():
                self.assertFalse(math.isnan(T) or math.isnan(S),
                    f"{mat.name}: NaN in tensile_strength_at_temp")
                self.assertFalse(math.isinf(T) or math.isinf(S),
                    f"{mat.name}: Inf in tensile_strength_at_temp")

    def test_density_sanity_bounds(self):
        """
        All densities must be physically plausible:
        above 50 kg/m³ (above aerogel; LI-900 is 144 kg/m³) and
        below 25000 kg/m³ (below osmium at 22590 kg/m³, with margin for Ta4HfC5).
        """
        for mat in MATERIALS_DB:
            self.assertGreater(mat.density_kgm3, 50.0,
                f"{mat.name}: density {mat.density_kgm3} kg/m³ seems too low")
            self.assertLess(mat.density_kgm3, 25000.0,
                f"{mat.name}: density {mat.density_kgm3} kg/m³ exceeds osmium")

    def test_positive_numeric_properties(self):
        """Core mechanical and thermal properties must be positive."""
        pos_fields = (
            "density_kgm3", "tensile_strength_mpa", "compressive_strength_mpa",
            "melting_point_K", "thermal_conductivity_WmK",
            "youngs_modulus_GPa",
        )
        for mat in MATERIALS_DB:
            for field in pos_fields:
                v = getattr(mat, field)
                self.assertGreater(v, 0.0,
                    f"{mat.name}: field '{field}' = {v} must be > 0")

    def test_applicable_regimes_not_empty(self):
        """Every material must have at least one applicable regime."""
        for mat in MATERIALS_DB:
            self.assertGreater(len(mat.applicable_regimes), 0,
                f"{mat.name}: applicable_regimes is empty")

    def test_unique_names(self):
        """All material names must be unique."""
        names = [m.name for m in MATERIALS_DB]
        self.assertEqual(len(names), len(set(names)),
            "Duplicate material names found: "
            f"{[n for n in names if names.count(n) > 1]}")


# ===========================================================================
# 2. Monotonic Strength (temperature knockdown)
# ===========================================================================

class TestMonotonicStrength(unittest.TestCase):

    def test_strength_decreases_monotonically_with_temperature(self):
        """
        Tensile (or flexural) strength must not increase with temperature,
        except for single-crystal nickel alloys which exhibit an anomalous
        yield-stress peak near 800°C (MONOTONICITY_EXCEPTIONS whitelist).
        """
        for mat in MATERIALS_DB:
            if mat.name in MONOTONICITY_EXCEPTIONS:
                continue
            temps = sorted(mat.tensile_strength_at_temp.keys())
            strengths = [mat.tensile_strength_at_temp[t] for t in temps]
            for i in range(len(strengths) - 1):
                self.assertGreaterEqual(
                    strengths[i], strengths[i + 1],
                    msg=(
                        f"{mat.name}: strength increased from "
                        f"{temps[i]} K ({strengths[i]:.1f} MPa) to "
                        f"{temps[i+1]} K ({strengths[i+1]:.1f} MPa) — "
                        f"expected monotonic decrease"
                    )
                )

    def test_whitelisted_alloys_have_anomalous_peak(self):
        """
        CMSX-4 and PWA 1484 must actually have the expected non-monotonic
        peak (verify they are whitelisted for the right reason, not by accident).
        """
        for name in MONOTONICITY_EXCEPTIONS:
            matches = [m for m in MATERIALS_DB if m.name == name]
            self.assertEqual(len(matches), 1,
                f"Whitelisted alloy '{name}' not found in database")
            mat = matches[0]
            temps = sorted(mat.tensile_strength_at_temp.keys())
            strengths = [mat.tensile_strength_at_temp[t] for t in temps]
            # There must be at least one increase in the sequence
            has_increase = any(
                strengths[i + 1] > strengths[i]
                for i in range(len(strengths) - 1)
            )
            self.assertTrue(has_increase,
                f"{name} is whitelisted for anomalous strength peak but "
                f"no peak found in data: {dict(zip(temps, strengths))}")

    def test_all_materials_above_zero_at_all_temps(self):
        """Strength must remain positive at all data points."""
        for mat in MATERIALS_DB:
            for T, S in mat.tensile_strength_at_temp.items():
                self.assertGreater(S, 0.0,
                    f"{mat.name}: strength at {T} K = {S} MPa must be > 0")


# ===========================================================================
# 3. Applicability Rules
# ===========================================================================

class TestApplicabilityRules(unittest.TestCase):

    # Category → regimes that must NOT appear
    FORBIDDEN = {
        "aluminum":          {"hypersonic", "reentry"},
        "titanium":          {"hypersonic", "reentry"},
        "steel":             {"hypersonic", "reentry"},
        "composite_polymer": {"hypersonic", "reentry"},
    }

    # TPS must be reentry-only (no other regime)
    def test_tps_reentry_only(self):
        """TPS ablators must only appear in 'reentry' regime."""
        for mat in get_materials_by_category("tps"):
            self.assertEqual(
                set(mat.applicable_regimes), {"reentry"},
                f"{mat.name}: TPS must have applicable_regimes == ['reentry'], "
                f"got {mat.applicable_regimes}"
            )

    def test_aluminum_not_in_high_speed_regimes(self):
        """Aluminum alloys must not be marked applicable to hypersonic or reentry."""
        for mat in get_materials_by_category("aluminum"):
            for regime in ("hypersonic", "reentry"):
                self.assertNotIn(regime, mat.applicable_regimes,
                    f"{mat.name}: aluminum should not be applicable to '{regime}'")

    def test_titanium_not_in_high_speed_regimes(self):
        """Titanium alloys must not be marked applicable to hypersonic or reentry."""
        for mat in get_materials_by_category("titanium"):
            for regime in ("hypersonic", "reentry"):
                self.assertNotIn(regime, mat.applicable_regimes,
                    f"{mat.name}: titanium should not be applicable to '{regime}'")

    def test_steel_not_in_high_speed_regimes(self):
        """Steel alloys must not be marked applicable to hypersonic or reentry."""
        for mat in get_materials_by_category("steel"):
            for regime in ("hypersonic", "reentry"):
                self.assertNotIn(regime, mat.applicable_regimes,
                    f"{mat.name}: steel should not be applicable to '{regime}'")

    def test_polymer_composites_not_in_high_speed_regimes(self):
        """Polymer composites must not be applicable to hypersonic or reentry."""
        for mat in get_materials_by_category("composite_polymer"):
            for regime in ("hypersonic", "reentry"):
                self.assertNotIn(regime, mat.applicable_regimes,
                    f"{mat.name}: composite_polymer should not be applicable to '{regime}'")

    def test_uhtc_includes_hypersonic_or_reentry(self):
        """UHTCs must be applicable to hypersonic and/or reentry."""
        for mat in get_materials_by_category("uhtc"):
            has_high_speed = (
                "hypersonic" in mat.applicable_regimes
                or "reentry" in mat.applicable_regimes
            )
            self.assertTrue(has_high_speed,
                f"{mat.name}: UHTC must include 'hypersonic' or 'reentry' in applicable_regimes")

    def test_uhtc_not_in_subsonic(self):
        """UHTCs should not be marked for subsonic use."""
        for mat in get_materials_by_category("uhtc"):
            self.assertNotIn("subsonic", mat.applicable_regimes,
                f"{mat.name}: UHTC should not be applicable to 'subsonic'")

    def test_carbon_materials_not_in_low_speed_regimes(self):
        """Carbon materials (C/C, PG, CVD diamond) should be hypersonic/reentry only."""
        for mat in get_materials_by_category("carbon"):
            for regime in ("subsonic", "supersonic"):
                self.assertNotIn(regime, mat.applicable_regimes,
                    f"{mat.name}: carbon material should not be applicable to '{regime}'")

    def test_refractory_not_in_subsonic(self):
        """Refractory metals are not used in subsonic structures."""
        for mat in get_materials_by_category("refractory"):
            self.assertNotIn("subsonic", mat.applicable_regimes,
                f"{mat.name}: refractory metal should not be applicable to 'subsonic'")

    def test_oxidation_max_temp_at_least_service_temp_air(self):
        """
        oxidation_max_temp_K must be >= service_temp_air_K: a material
        cannot be structurally usable in air up to a temperature where its
        own oxidation has already failed. Note: oxidation_max_temp_K CAN
        exceed service_temp_inert_K because oxidation protection is a
        separate limit from creep/strength (e.g. IN718 has excellent
        oxidation to 1255 K but creep limits structural service to 1090 K).
        """
        for mat in MATERIALS_DB:
            self.assertGreaterEqual(
                mat.oxidation_max_temp_K, mat.service_temp_air_K,
                f"{mat.name}: oxidation_max_temp_K ({mat.oxidation_max_temp_K}) "
                f"< service_temp_air_K ({mat.service_temp_air_K})"
            )

    def test_general_engineering_subsonic_only(self):
        """general_engineering materials must be rated for subsonic only — never hypersonic or reentry."""
        for mat in get_materials_by_category("general_engineering"):
            self.assertEqual(
                set(mat.applicable_regimes), {"subsonic"},
                f"{mat.name}: general_engineering must have applicable_regimes == ['subsonic'], "
                f"got {mat.applicable_regimes}"
            )


# ===========================================================================
# 4. Oxidation Ratings Spot-Check
# ===========================================================================

class TestOxidationRatings(unittest.TestCase):

    def test_steel_not_excellent_oxidation(self):
        """Steel alloys corrode; none should be rated 'excellent' for oxidation
        except the stainless PH steels which have 'good' at best."""
        for mat in get_materials_by_category("steel"):
            self.assertNotEqual(mat.oxidation_resistance, "excellent",
                f"{mat.name}: steel should not have 'excellent' oxidation resistance")

    def test_nickel_not_poor_oxidation(self):
        """Nickel superalloys are developed for oxidation resistance; none should be 'poor'."""
        for mat in get_materials_by_category("nickel"):
            self.assertNotEqual(mat.oxidation_resistance, "poor",
                f"{mat.name}: nickel superalloy should not have 'poor' oxidation resistance")

    def test_tps_not_poor_oxidation(self):
        """TPS ablators ablate in the boundary layer — rated 'excellent' or 'good',
        never 'poor' or 'limited'."""
        for mat in get_materials_by_category("tps"):
            self.assertIn(mat.oxidation_resistance, ("excellent", "good"),
                f"{mat.name}: TPS should have 'excellent' or 'good' oxidation_resistance")

    def test_carbon_materials_poor_oxidation(self):
        """Carbon materials burn in air — must be rated 'poor' or 'limited' (SiC-coated form)."""
        for mat in get_materials_by_category("carbon"):
            self.assertIn(mat.oxidation_resistance, ("poor", "limited"),
                f"{mat.name}: carbon material should have 'poor' or 'limited' oxidation resistance")

    def test_zrb2_sic_good_oxidation(self):
        """ZrB2-SiC 20vol% must be 'good' — the SiC addition provides oxidation protection."""
        mat = next((m for m in MATERIALS_DB if m.name == "ZrB2-SiC 20vol%"), None)
        self.assertIsNotNone(mat, "ZrB2-SiC 20vol% not found in database")
        self.assertEqual(mat.oxidation_resistance, "good",
            "ZrB2-SiC 20vol% must have 'good' oxidation resistance")

    def test_zrc_poor_oxidation(self):
        """ZrC oxidizes readily in air — must be 'poor'."""
        mat = next((m for m in MATERIALS_DB if m.name == "ZrC"), None)
        self.assertIsNotNone(mat, "ZrC not found in database")
        self.assertEqual(mat.oxidation_resistance, "poor",
            "ZrC must have 'poor' oxidation resistance")

    def test_aluminum_good_or_better_oxidation(self):
        """Aluminum alloys form protective Al2O3 — should be 'good' or 'excellent'."""
        for mat in get_materials_by_category("aluminum"):
            self.assertIn(mat.oxidation_resistance, ("excellent", "good"),
                f"{mat.name}: aluminum should have 'good' or 'excellent' oxidation resistance")

    def test_tungsten_molybdenum_poor_oxidation(self):
        """W and Mo form volatile oxides in air — both must be 'poor'."""
        for name in ("Tungsten", "Molybdenum"):
            mat = next((m for m in MATERIALS_DB if m.name == name), None)
            self.assertIsNotNone(mat, f"{name} not found in database")
            self.assertEqual(mat.oxidation_resistance, "poor",
                f"{name} must have 'poor' oxidation resistance")

    def test_general_engineering_oxidation_spot_checks(self):
        """Spot-check oxidation ratings for general engineering materials."""
        checks = {
            "Structural Steel A36": "good",
            "Stainless Steel 304":  "excellent",
            "Mild Steel 1020":      "limited",
            "ABS Plastic":          "good",
            "Nylon 66":             "good",
            "GFRP Generic":         "good",
            "CFRP Generic":         "good",
        }
        name_map = {m.name: m for m in MATERIALS_DB}
        for name, expected_rating in checks.items():
            mat = name_map.get(name)
            self.assertIsNotNone(mat, f"{name} not found in MATERIALS_DB")
            self.assertEqual(mat.oxidation_resistance, expected_rating,
                f"{name}: expected oxidation_resistance='{expected_rating}', "
                f"got '{mat.oxidation_resistance}'")


# ===========================================================================
# 5. Service Temperature Ordering
# ===========================================================================

class TestServiceTemperatureOrdering(unittest.TestCase):

    def test_service_temp_air_lte_inert(self):
        """
        Air service temperature must be <= inert/vacuum service temperature
        for every material (enforced by __post_init__ but also tested explicitly).
        """
        for mat in MATERIALS_DB:
            self.assertLessEqual(
                mat.service_temp_air_K, mat.service_temp_inert_K,
                f"{mat.name}: service_temp_air_K ({mat.service_temp_air_K}) "
                f"> service_temp_inert_K ({mat.service_temp_inert_K})"
            )

    def test_service_temps_below_melting_point(self):
        """Service temperatures (inert) must be below the melting/sublimation point."""
        for mat in MATERIALS_DB:
            self.assertLessEqual(
                mat.service_temp_inert_K, mat.melting_point_K,
                f"{mat.name}: service_temp_inert_K ({mat.service_temp_inert_K}) "
                f">= melting_point_K ({mat.melting_point_K})"
            )

    def test_service_temps_above_room_temperature(self):
        """Service temperatures must be above room temperature (293 K)."""
        for mat in MATERIALS_DB:
            self.assertGreater(mat.service_temp_air_K, 293.0,
                f"{mat.name}: service_temp_air_K ({mat.service_temp_air_K}) <= 293 K")

    def test_melting_points_above_300k(self):
        """All melting/decomposition temperatures must be above 300 K."""
        for mat in MATERIALS_DB:
            self.assertGreater(mat.melting_point_K, 300.0,
                f"{mat.name}: melting_point_K ({mat.melting_point_K}) <= 300 K")


# ===========================================================================
# 6. Interpolation Function
# ===========================================================================

class TestInterpolationFunction(unittest.TestCase):

    def _make_test_material(self, temp_strength_dict: dict) -> MaterialEntry:
        """Helper: create a minimal valid MaterialEntry for interpolation testing."""
        return MaterialEntry(
            name="Test Material",
            category="aluminum",
            density_kgm3=2700.0,
            tensile_strength_mpa=temp_strength_dict[min(temp_strength_dict)],
            tensile_strength_at_temp=temp_strength_dict,
            compressive_strength_mpa=200.0,
            service_temp_air_K=500.0,
            service_temp_inert_K=600.0,
            melting_point_K=900.0,
            thermal_conductivity_WmK=100.0,
            thermal_expansion_1K=10e-6,
            youngs_modulus_GPa=70.0,
            oxidation_resistance="good",
            oxidation_max_temp_K=490.0,
            applicable_regimes=["subsonic", "supersonic"],
            citation="test",
            notes="test",
        )

    def test_exact_key_returns_exact_value(self):
        """Query at an exact dictionary key must return the exact value."""
        mat = self._make_test_material({
            293.0: 500.0,
            500.0: 300.0,
            800.0: 100.0,
        })
        self.assertAlmostEqual(get_strength_at_temperature(mat, 293.0), 500.0)
        self.assertAlmostEqual(get_strength_at_temperature(mat, 500.0), 300.0)
        self.assertAlmostEqual(get_strength_at_temperature(mat, 800.0), 100.0)

    def test_interpolation_midpoint(self):
        """Midpoint between two data points should return linear interpolation."""
        mat = self._make_test_material({
            293.0: 500.0,
            500.0: 300.0,
            800.0: 100.0,
        })
        # Midpoint of 293-500: T = 396.5, expected = 400 MPa
        T_mid = (293.0 + 500.0) / 2.0
        expected = (500.0 + 300.0) / 2.0
        result = get_strength_at_temperature(mat, T_mid)
        self.assertAlmostEqual(result, expected, places=5)

    def test_clamp_below_minimum(self):
        """Temperature below minimum data point returns room-temperature strength."""
        mat = self._make_test_material({
            293.0: 500.0,
            500.0: 300.0,
            800.0: 100.0,
        })
        self.assertAlmostEqual(get_strength_at_temperature(mat, 0.0), 500.0)
        self.assertAlmostEqual(get_strength_at_temperature(mat, 100.0), 500.0)
        self.assertAlmostEqual(get_strength_at_temperature(mat, 292.9), 500.0)

    def test_clamp_above_maximum(self):
        """Temperature above maximum data point returns last known strength."""
        mat = self._make_test_material({
            293.0: 500.0,
            500.0: 300.0,
            800.0: 100.0,
        })
        self.assertAlmostEqual(get_strength_at_temperature(mat, 900.0), 100.0)
        self.assertAlmostEqual(get_strength_at_temperature(mat, 5000.0), 100.0)

    def test_interpolation_in_upper_interval(self):
        """Interpolation works correctly in the upper temperature interval."""
        mat = self._make_test_material({
            293.0: 500.0,
            500.0: 300.0,
            800.0: 100.0,
        })
        # T = 650 is (650-500)/(800-500) = 0.5 through [500, 800]
        # Expected: 300 + 0.5 * (100 - 300) = 300 - 100 = 200 MPa
        result = get_strength_at_temperature(mat, 650.0)
        self.assertAlmostEqual(result, 200.0, places=5)

    def test_returns_float(self):
        """Function must always return a float."""
        for mat in MATERIALS_DB:
            for T in (293.0, 800.0, 1500.0):
                result = get_strength_at_temperature(mat, T)
                self.assertIsInstance(result, float,
                    f"{mat.name} at {T} K: expected float, got {type(result)}")

    def test_all_materials_at_standard_query_temps(self):
        """Function must run without error for every material at 293, 800, 1500 K."""
        for mat in MATERIALS_DB:
            for T in (293.0, 800.0, 1500.0):
                try:
                    result = get_strength_at_temperature(mat, T)
                    self.assertGreater(result, 0.0,
                        f"{mat.name} at {T} K: strength must be > 0, got {result}")
                except Exception as e:
                    self.fail(f"{mat.name} at {T} K raised exception: {e}")

    def test_strength_never_negative(self):
        """Interpolated strength must never be negative."""
        test_temps = [200.0, 293.0, 500.0, 800.0, 1000.0,
                      1500.0, 2000.0, 2500.0, 3000.0, 5000.0]
        for mat in MATERIALS_DB:
            for T in test_temps:
                result = get_strength_at_temperature(mat, T)
                self.assertGreaterEqual(result, 0.0,
                    f"{mat.name} at {T} K: strength = {result} MPa < 0")


# ===========================================================================
# 7. Category and Regime String Consistency
# ===========================================================================

class TestCategoryConsistency(unittest.TestCase):

    def test_all_categories_in_valid_set(self):
        """Every material's category must be in VALID_CATEGORIES."""
        for mat in MATERIALS_DB:
            self.assertIn(mat.category, VALID_CATEGORIES,
                f"{mat.name}: category '{mat.category}' not in VALID_CATEGORIES")

    def test_all_regimes_in_valid_set(self):
        """Every regime string must be in VALID_REGIMES."""
        for mat in MATERIALS_DB:
            for regime in mat.applicable_regimes:
                self.assertIn(regime, VALID_REGIMES,
                    f"{mat.name}: regime '{regime}' not in VALID_REGIMES")

    def test_query_by_category_completeness(self):
        """Sum of materials across all categories must equal total database size."""
        total = sum(
            len(get_materials_by_category(cat))
            for cat in VALID_CATEGORIES
        )
        self.assertEqual(total, len(MATERIALS_DB),
            "Sum of category queries does not match total database size "
            "(possible duplicate category assignment)")

    def test_regime_query_returns_subset(self):
        """get_materials_by_regime must return a strict subset of MATERIALS_DB."""
        for regime in VALID_REGIMES:
            results = get_materials_by_regime(regime)
            for mat in results:
                self.assertIn(mat, MATERIALS_DB,
                    f"get_materials_by_regime('{regime}') returned a material "
                    f"not in MATERIALS_DB: {mat.name}")
                self.assertIn(regime, mat.applicable_regimes,
                    f"get_materials_by_regime('{regime}') returned {mat.name} "
                    f"which does not have '{regime}' in applicable_regimes")

    def test_hypersonic_regime_has_expected_categories(self):
        """
        Hypersonic-applicable materials must only come from categories that
        are physically appropriate at hypersonic speeds.
        """
        allowed_hypersonic_cats = {
            "nickel", "cobalt", "composite_ceramic", "refractory", "uhtc", "carbon"
        }
        for mat in get_materials_by_regime("hypersonic"):
            self.assertIn(mat.category, allowed_hypersonic_cats,
                f"{mat.name} (category='{mat.category}') is marked hypersonic "
                f"but is not in the allowed categories for that regime: "
                f"{allowed_hypersonic_cats}")

    def test_reentry_regime_has_expected_categories(self):
        """
        Reentry-applicable materials must only come from categories that
        survive orbital reentry conditions.
        """
        allowed_reentry_cats = {"composite_ceramic", "uhtc", "carbon", "tps"}
        for mat in get_materials_by_regime("reentry"):
            self.assertIn(mat.category, allowed_reentry_cats,
                f"{mat.name} (category='{mat.category}') is marked reentry "
                f"but is not in the allowed categories for that regime: "
                f"{allowed_reentry_cats}")

    def test_subsonic_only_materials_not_in_hypersonic(self):
        """
        Materials that are only applicable to subsonic/supersonic should
        not appear in a hypersonic query.
        """
        hypersonic_materials = get_materials_by_regime("hypersonic")
        hypersonic_names = {m.name for m in hypersonic_materials}

        for mat in get_materials_by_category("aluminum"):
            self.assertNotIn(mat.name, hypersonic_names,
                f"Aluminum {mat.name} should not appear in hypersonic query")
        for mat in get_materials_by_category("titanium"):
            self.assertNotIn(mat.name, hypersonic_names,
                f"Titanium {mat.name} should not appear in hypersonic query")


class TestAvailabilityScores(unittest.TestCase):

    def test_availability_scores_in_range(self):
        """All materials must have availability_score between 0.0 and 1.0."""
        for mat in MATERIALS_DB:
            self.assertGreaterEqual(mat.availability_score, 0.0,
                f"{mat.name}: availability_score {mat.availability_score} < 0.0")
            self.assertLessEqual(mat.availability_score, 1.0,
                f"{mat.name}: availability_score {mat.availability_score} > 1.0")

    def test_availability_default_is_commercial(self):
        """Materials without explicit override must default to 1.0."""
        from core.materials_db import _AVAILABILITY_OVERRIDES
        for mat in MATERIALS_DB:
            if mat.name not in _AVAILABILITY_OVERRIDES:
                self.assertEqual(mat.availability_score, 1.0,
                    f"{mat.name}: expected default 1.0, got {mat.availability_score}")


# ===========================================================================
# 9. Material Cost Field (Cost-Axis-on-Pareto-Front feature)
# ===========================================================================

class TestMaterialCostField(unittest.TestCase):
    """Order-of-magnitude bulk price (USD/kg, 2025-26 market) on every entry.

    cost_usd_per_kg is the data hook for the Pareto cost axis and the
    "Est. Cost" column in the materials table. The values are screening-
    grade only (\u00b150% per the docstring on MaterialEntry); these tests
    pin the qualitative shape, not the absolute numbers.

    Sentinel value 0.0 is reserved for exotic / 2D materials that exist
    only for impossibility detection. The current materials_db has no such
    entries, so every row must report a strictly positive price.
    """

    def test_every_material_has_positive_cost(self):
        """Catches a future contributor adding a row without a cost."""
        for mat in MATERIALS_DB:
            self.assertGreater(
                mat.cost_usd_per_kg, 0.0,
                f"{mat.name}: cost_usd_per_kg must be > 0 (got "
                f"{mat.cost_usd_per_kg}). Use 0.0 only for exotic/2D "
                f"sentinels — none exist in this DB.",
            )

    def test_cost_field_has_sensible_upper_bound(self):
        """Bulk price ceiling: $10000/kg flags any unit error or typo
        (e.g. someone writing the per-gram price into the per-kg field).
        Today's most expensive entry is Pyrolytic Graphite / Diamond CVD
        at $8000/kg, well under the cap."""
        for mat in MATERIALS_DB:
            self.assertLess(
                mat.cost_usd_per_kg, 10_000.0,
                f"{mat.name}: cost {mat.cost_usd_per_kg} exceeds $10k/kg "
                f"sanity ceiling — likely a unit error.",
            )

    def test_category_average_cost_ordering(self):
        """Bulk-price hierarchy that any practising aerospace engineer
        would recognise: steel < titanium < nickel-base alloys
        < ceramic-matrix composites < ultra-high-temperature ceramics.
        Tests average within category so a single anomalous entry
        (e.g. CMSX-4 at $800 in nickel) doesn't dominate the order."""
        from collections import defaultdict
        by_cat: dict = defaultdict(list)
        for mat in MATERIALS_DB:
            by_cat[mat.category].append(mat.cost_usd_per_kg)

        avg = {cat: sum(vs) / len(vs) for cat, vs in by_cat.items()}

        # Hard ordering — these gaps are wide enough to survive a 2x
        # revaluation of any single category.
        self.assertLess(
            avg["steel"], avg["titanium"],
            f"steel avg ${avg['steel']:.0f}/kg should be cheaper than "
            f"titanium avg ${avg['titanium']:.0f}/kg",
        )
        self.assertLess(
            avg["titanium"], avg["nickel"],
            f"titanium avg ${avg['titanium']:.0f}/kg should be cheaper than "
            f"nickel avg ${avg['nickel']:.0f}/kg",
        )
        self.assertLess(
            avg["nickel"], avg["composite_ceramic"],
            f"nickel avg ${avg['nickel']:.0f}/kg should be cheaper than "
            f"CMC avg ${avg['composite_ceramic']:.0f}/kg",
        )
        self.assertLess(
            avg["composite_ceramic"], avg["uhtc"],
            f"CMC avg ${avg['composite_ceramic']:.0f}/kg should be cheaper "
            f"than UHTC avg ${avg['uhtc']:.0f}/kg",
        )

    def test_aluminum_is_cheapest_category(self):
        """Aluminum should be the cheapest category on average — a sanity
        anchor: if this fails, the whole pricing table is upside down."""
        from collections import defaultdict
        by_cat: dict = defaultdict(list)
        for mat in MATERIALS_DB:
            by_cat[mat.category].append(mat.cost_usd_per_kg)
        avg = {cat: sum(vs) / len(vs) for cat, vs in by_cat.items()}
        cheapest = min(avg, key=lambda c: avg[c])
        # aluminum or steel could plausibly tie for cheapest in a future
        # revaluation; assert "one of the two" rather than aluminum-only.
        self.assertIn(
            cheapest, ("aluminum", "steel"),
            f"Cheapest category is {cheapest} (avg ${avg[cheapest]:.0f}/kg) "
            f"— expected aluminum or steel.",
        )


# ---------------------------------------------------------------------------
# Specific-heat-capacity fields (Phase 7.1, transient-heat rollout)
# ---------------------------------------------------------------------------
class TestSpecificHeatCapacity(unittest.TestCase):
    """Phase 7.1 deliverable: every material has a non-default
    cp_data_status, at least 30 materials are sourced, and the
    computed thermal diffusivity α = k / (ρ × c_p) for the key
    priority materials lands within ±15 % of published values."""

    def test_every_material_has_cp_data_status(self):
        valid = {"sourced", "estimated", "not_applicable", "unknown"}
        for mat in MATERIALS_DB:
            with self.subTest(material=mat.name):
                self.assertIn(
                    mat.cp_data_status, valid,
                    f"{mat.name}: cp_data_status "
                    f"{mat.cp_data_status!r} not in {valid}",
                )

    def test_at_least_30_materials_are_sourced(self):
        sourced = [
            m for m in MATERIALS_DB if m.cp_data_status == "sourced"
        ]
        self.assertGreaterEqual(
            len(sourced), 30,
            f"Only {len(sourced)} sourced c_p values; Phase 7.1 "
            f"acceptance criterion is >=30. Got: "
            f"{[m.name for m in sourced]}",
        )

    def test_sourced_materials_have_non_default_cp(self):
        for mat in MATERIALS_DB:
            if mat.cp_data_status in ("sourced", "estimated"):
                with self.subTest(material=mat.name):
                    self.assertIsNotNone(
                        mat.specific_heat_J_kgK,
                        f"{mat.name}: cp_data_status="
                        f"{mat.cp_data_status} but c_p is None",
                    )
                    self.assertGreater(
                        mat.specific_heat_J_kgK, 0.0,
                        f"{mat.name}: c_p must be > 0",
                    )
                    self.assertNotEqual(
                        mat.cp_data_source, "",
                        f"{mat.name}: cp_data_source is empty",
                    )

    def test_tps_materials_marked_not_applicable(self):
        for mat in MATERIALS_DB:
            if mat.category == "tps":
                with self.subTest(material=mat.name):
                    self.assertEqual(
                        mat.cp_data_status, "not_applicable",
                        f"{mat.name} (tps): expected "
                        f"cp_data_status='not_applicable'",
                    )

    def test_thermal_diffusivity_within_published_tolerance(self):
        """α = k / (ρ × c_p) for Inconel 718, Ti-6Al-4V, Al 2024-T3
        within ±15 % of published NIST / ITER values."""
        cases = [
            ("Inconel 718", 3.1e-6),
            ("Ti-6Al-4V",   2.9e-6),
            ("2024-T3",     5.0e-5),
        ]
        for name, alpha_published in cases:
            mat = next(m for m in MATERIALS_DB if m.name == name)
            with self.subTest(material=name):
                alpha = (
                    mat.thermal_conductivity_WmK
                    / (mat.density_kgm3 * mat.specific_heat_J_kgK)
                )
                self.assertAlmostEqual(
                    alpha, alpha_published,
                    delta=0.15 * alpha_published,
                    msg=(
                        f"{name}: computed α={alpha:.3e}, "
                        f"published ~{alpha_published:.1e} ({alpha_published*1e6:.1f}e-6 m^2/s); "
                        f"exceeds ±15% tolerance."
                    ),
                )


# ---------------------------------------------------------------------------
# Larson-Miller / creep-data fields (lifecycle modelling, Phase 1)
# ---------------------------------------------------------------------------
class TestCreepData(unittest.TestCase):
    """Every material in MATERIALS_DB must carry a non-default
    ``creep_data_status`` after module load. ``_apply_creep_data()``
    populates the priority materials with sourced / estimated curves
    and the category-rule pass marks not_applicable categories;
    materials that fall through both keep status="unknown" but they
    must do so deliberately, not by oversight."""

    def test_every_material_has_creep_data_status(self):
        """No silent unknowns — every entry's status must be one of
        the four valid values, set explicitly by either the priority
        list or the category rule."""
        valid = {"sourced", "estimated", "not_applicable", "unknown"}
        for mat in MATERIALS_DB:
            with self.subTest(material=mat.name):
                self.assertIn(
                    mat.creep_data_status, valid,
                    f"{mat.name}: creep_data_status "
                    f"{mat.creep_data_status!r} not in {valid}",
                )

    def test_at_least_18_materials_are_sourced(self):
        """Phase 1 acceptance: ≥18 materials must have curves drawn
        from a primary reference (Special Metals datasheets, MMPDS,
        ASM Handbook, RTI Titanium Guide). The plan's priority list
        targets ~20; we set the floor at 18 so a small renaming
        accident doesn't mask a real regression."""
        sourced = [m for m in MATERIALS_DB if m.creep_data_status == "sourced"]
        self.assertGreaterEqual(
            len(sourced), 18,
            f"Only {len(sourced)} sourced materials — Phase 1 plan "
            f"targets at least 18. Got: "
            f"{[m.name for m in sourced]}",
        )

    def test_sourced_materials_have_complete_curve_and_citation(self):
        """If status="sourced" the curve must be non-empty and the
        source citation must not be the default empty string. Same
        applies to estimated entries (whose source must explain WHAT
        was extrapolated and from what)."""
        for mat in MATERIALS_DB:
            if mat.creep_data_status not in ("sourced", "estimated"):
                continue
            with self.subTest(material=mat.name):
                self.assertGreater(
                    len(mat.lmp_curve), 1,
                    f"{mat.name}: lmp_curve has < 2 points",
                )
                self.assertNotEqual(
                    mat.creep_data_source, "",
                    f"{mat.name}: creep_data_source is empty",
                )
                self.assertIsNotNone(
                    mat.larson_miller_C,
                    f"{mat.name}: larson_miller_C is None",
                )

    def test_lmp_curves_are_sorted_ascending(self):
        """Linear interpolation in core/creep.py assumes the curve is
        sorted by LMP. Sort order is also enforced in
        MaterialEntry.__post_init__, so this is a regression guard
        for any future hand-edits."""
        for mat in MATERIALS_DB:
            if not mat.lmp_curve:
                continue
            lmps = [pt[0] for pt in mat.lmp_curve]
            with self.subTest(material=mat.name):
                self.assertEqual(
                    lmps, sorted(lmps),
                    f"{mat.name}: lmp_curve LMP values not ascending "
                    f"({lmps})",
                )

    def test_lmp_curves_have_decreasing_stress(self):
        """At higher LMP (hotter or longer time), rupture stress
        must drop — physics requirement, not a data convention. A
        non-monotonic stress would mean a material gets *stronger*
        at higher T, which would only happen for a transcription
        error."""
        for mat in MATERIALS_DB:
            if not mat.lmp_curve or len(mat.lmp_curve) < 2:
                continue
            stresses = [pt[1] for pt in mat.lmp_curve]
            for i in range(1, len(stresses)):
                with self.subTest(material=mat.name, point=i):
                    self.assertLessEqual(
                        stresses[i], stresses[i - 1] + 1.0,  # tiny tolerance
                        f"{mat.name}: rupture stress increased between "
                        f"points {i-1} and {i}: {stresses[i-1]} -> "
                        f"{stresses[i]} MPa",
                    )

    def test_concorde_envelope_al2024_creep_limited(self):
        """Spot-check that confirms the Concorde validation re-pass
        will work as planned: at 100 °C × 25,000 h, Al 2024-T3
        rupture stress must be below typical airframe sigma_required
        (~100-150 MPa). This is the historically-correct outcome
        that drove development of Hiduminium RR58 / Al 2618."""
        al = next(m for m in MATERIALS_DB if m.name == "2024-T3")
        T_K = 373.0       # 100 °C
        t_hours = 25_000.0
        C = al.larson_miller_C
        lmp = T_K * (C + math.log10(t_hours))
        # Linear interpolation in σ for a quick spot-check (the
        # full creep module will land in Phase 2).
        curve = al.lmp_curve
        sigma = None
        for i in range(1, len(curve)):
            if curve[i - 1][0] <= lmp <= curve[i][0]:
                f = (lmp - curve[i - 1][0]) / (curve[i][0] - curve[i - 1][0])
                sigma = curve[i - 1][1] + f * (curve[i][1] - curve[i - 1][1])
                break
        self.assertIsNotNone(
            sigma,
            f"Concorde envelope LMP={lmp:.0f} fell outside Al 2024-T3 "
            f"curve {curve}",
        )
        self.assertLess(
            sigma, 100.0,
            f"Al 2024-T3 at Concorde envelope (100°C, 25000h) gave "
            f"{sigma:.0f} MPa rupture stress — should be < 100 MPa "
            f"so the creep stage correctly fails it (sigma_required "
            f"~117 MPa per the validation envelope).",
        )

    def test_tps_and_polymer_composites_are_not_applicable(self):
        """Category-rule check: every TPS material and every polymer
        composite must have status="not_applicable" (single-event
        ablators don't classically creep; CFRP creep is a separate
        viscoelastic model out-of-scope for Larson-Miller)."""
        for mat in MATERIALS_DB:
            if mat.category in ("tps", "composite_polymer"):
                with self.subTest(material=mat.name):
                    self.assertEqual(
                        mat.creep_data_status, "not_applicable",
                        f"{mat.name} ({mat.category}): expected "
                        f"creep_data_status='not_applicable', got "
                        f"{mat.creep_data_status!r}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
