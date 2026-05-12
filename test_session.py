"""
Unit tests for core/session.py — the JSON round-trip schema.

Scope:
  * SessionSchema ↔ dict ↔ JSON round-trip equality.
  * Forward-compatibility: unknown top-level fields dropped silently;
    unknown options preserved; newer matvec_version warns but loads.
  * Error handling: missing required fields raise with field name;
    invalid types / malformed JSON raise ValueError.

No streamlit, no physics — this file intentionally tests the schema in
isolation so a breakage in the physics / matching layer doesn't masquerade
as a schema regression (and vice versa).
"""

import json
import unittest
import warnings

from core import MATVEC_VERSION
from core.session import (
    SessionSchema,
    session_to_dict,
    dict_to_session,
    session_to_json,
    json_to_session,
    _REQUIRED_ENVELOPE_FIELDS,
)


def _minimal_session() -> SessionSchema:
    """A plausible envelope used as the baseline fixture."""
    return SessionSchema(
        mach=3.2,
        alt_km=25.0,
        mass_kg=30600.0,
        R_n_m=0.15,
        g_load=2.5,
        char_len_m=32.7,
        vehicle_category="aircraft",
        system_label="Test SR-71",
    )


class TestRoundTrip(unittest.TestCase):
    """Schema → dict → Schema and Schema → JSON → Schema must preserve
    every field byte-for-byte."""

    def test_dict_roundtrip_preserves_all_fields(self):
        s1 = _minimal_session()
        s2 = dict_to_session(session_to_dict(s1))
        self.assertEqual(s1, s2)

    def test_json_roundtrip_preserves_all_fields(self):
        s1 = _minimal_session()
        s2 = json_to_session(session_to_json(s1))
        self.assertEqual(s1, s2)

    def test_roundtrip_preserves_options_dict(self):
        s1 = SessionSchema(
            mach=0.5, alt_km=0.0, mass_kg=50.0, R_n_m=0.05,
            g_load=1.0, char_len_m=0.1,
            vehicle_category="turbine",
            system_label="Turbine HPT Blade",
            options={"hot_section_temp_K": 1400.0},
        )
        s2 = json_to_session(session_to_json(s1))
        self.assertEqual(s2.options, {"hot_section_temp_K": 1400.0})

    def test_json_is_sorted_and_indented(self):
        """Committed preset JSONs must produce stable diffs."""
        js = session_to_json(_minimal_session())
        lines = js.splitlines()
        # 2-space indent on inner keys
        self.assertTrue(any(line.startswith("  \"R_n_m\"") for line in lines))
        # Keys sorted alphabetically
        parsed = json.loads(js)
        self.assertEqual(list(parsed.keys()), sorted(parsed.keys()))


class TestForwardCompat(unittest.TestCase):
    """Schema loader is liberal in what it accepts (future runtimes that
    add fields shouldn't break older loaders)."""

    def test_unknown_top_level_field_is_dropped_silently(self):
        """A newer file carrying an unknown field loads cleanly."""
        d = session_to_dict(_minimal_session())
        d["future_toggle"] = "ignore_me"
        s = dict_to_session(d)
        # No attribute added to the dataclass
        self.assertFalse(hasattr(s, "future_toggle"))
        # Required fields still intact
        self.assertEqual(s.mach, 3.2)

    def test_unknown_option_key_survives_roundtrip(self):
        """options is the designated "future features" dict — unknown
        keys must pass through untouched."""
        d = session_to_dict(_minimal_session())
        d["options"] = {"hot_section_temp_K": 1400.0, "future_toggle": True}
        s = dict_to_session(d)
        self.assertEqual(s.options["future_toggle"], True)
        # Roundtrip again: the unknown key still survives
        s2 = json_to_session(session_to_json(s))
        self.assertEqual(s2.options["future_toggle"], True)

    def test_newer_matvec_version_warns_but_loads(self):
        d = session_to_dict(_minimal_session())
        d["matvec_version"] = "99.99.99"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            s = dict_to_session(d)
            # Still constructs successfully
            self.assertEqual(s.matvec_version, "99.99.99")
            self.assertEqual(s.mach, 3.2)
            # And emitted a UserWarning
            self.assertTrue(
                any(issubclass(w.category, UserWarning) for w in caught),
                "Expected a UserWarning about matvec_version mismatch.",
            )

    def test_same_matvec_version_does_not_warn(self):
        d = session_to_dict(_minimal_session())
        d["matvec_version"] = MATVEC_VERSION
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dict_to_session(d)
            self.assertFalse(
                any(issubclass(w.category, UserWarning) for w in caught),
                "Did not expect any UserWarning when versions match.",
            )


class TestErrorHandling(unittest.TestCase):
    """Missing required fields / invalid types must raise ValueError
    with a message that names the offending field."""

    def test_missing_required_envelope_field_raises_with_field_name(self):
        for missing in _REQUIRED_ENVELOPE_FIELDS:
            with self.subTest(missing=missing):
                d = session_to_dict(_minimal_session())
                d.pop(missing)
                with self.assertRaises(ValueError) as ctx:
                    dict_to_session(d)
                # Error message must name the field so the user can fix it.
                self.assertIn(missing, str(ctx.exception))

    def test_missing_vehicle_category_raises(self):
        d = session_to_dict(_minimal_session())
        d.pop("vehicle_category")
        with self.assertRaises(ValueError) as ctx:
            dict_to_session(d)
        self.assertIn("vehicle_category", str(ctx.exception))

    def test_missing_system_label_raises(self):
        d = session_to_dict(_minimal_session())
        d.pop("system_label")
        with self.assertRaises(ValueError) as ctx:
            dict_to_session(d)
        self.assertIn("system_label", str(ctx.exception))

    def test_non_dict_payload_raises(self):
        with self.assertRaises(ValueError):
            dict_to_session([1, 2, 3])

    def test_options_must_be_a_dict(self):
        d = session_to_dict(_minimal_session())
        d["options"] = "not a dict"
        with self.assertRaises(ValueError) as ctx:
            dict_to_session(d)
        self.assertIn("options", str(ctx.exception))

    def test_invalid_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            json_to_session("{ not valid json")

    def test_integer_values_coerce_to_float(self):
        """A hand-authored JSON with integer literals (mass_kg: 500) must
        not bleed int arithmetic downstream."""
        d = session_to_dict(_minimal_session())
        d["mass_kg"] = 500              # int, not float
        s = dict_to_session(d)
        self.assertIsInstance(s.mass_kg, float)
        self.assertEqual(s.mass_kg, 500.0)


class TestDefaults(unittest.TestCase):
    """Default values for optional envelope fields mirror the physics
    engine's defaults — otherwise downloading / uploading a JSON would
    silently shift the flight_duration_s or wall_emissivity."""

    def test_default_flight_duration_matches_physics_engine(self):
        s = _minimal_session()
        self.assertEqual(s.flight_duration_s, 600.0)

    def test_default_wall_emissivity_matches_physics_engine(self):
        s = _minimal_session()
        self.assertEqual(s.wall_emissivity, 0.85)

    def test_default_matvec_version_matches_core(self):
        s = _minimal_session()
        self.assertEqual(s.matvec_version, MATVEC_VERSION)

    def test_default_design_lifetime_is_one_hour(self):
        """Default 1 h preserves pre-creep-feature behaviour: the
        creep evaluation stage is a no-op for nearly all materials at
        single-flight lifetimes, so existing session JSONs without the
        field load and analyse identically to before."""
        s = _minimal_session()
        self.assertEqual(s.design_lifetime_hours, 1.0)


class TestDesignLifetimeRoundTrip(unittest.TestCase):
    """``design_lifetime_hours`` was added in the lifecycle / creep
    rollout. It must round-trip through dict and JSON, default cleanly
    when absent, and coerce integer literals to float (same convention
    as the rest of the envelope).
    """

    def test_round_trip_preserves_lifetime(self):
        from core.session import (
            session_to_dict,
            dict_to_session,
            session_to_json,
            json_to_session,
        )
        s = _minimal_session()
        # Override the default to confirm the value survives serialisation.
        from dataclasses import replace
        s = replace(s, design_lifetime_hours=25000.0)
        self.assertEqual(
            dict_to_session(session_to_dict(s)).design_lifetime_hours,
            25000.0,
        )
        self.assertEqual(
            json_to_session(session_to_json(s)).design_lifetime_hours,
            25000.0,
        )

    def test_legacy_json_without_lifetime_loads_with_default(self):
        """A session JSON written by an older runtime (pre-creep) will
        not include ``design_lifetime_hours``. Loading must succeed and
        fall back to the 1.0 h default — no exception, no warning."""
        from core.session import json_to_session
        legacy = json.dumps({
            "mach": 1.5, "alt_km": 12.0, "mass_kg": 1000.0,
            "R_n_m": 0.30, "g_load": 5.0, "char_len_m": 5.0,
            "flight_duration_s": 600.0, "wall_emissivity": 0.85,
            "vehicle_category": "general",
            "system_label": "legacy-no-lifetime",
        })
        s = json_to_session(legacy)
        self.assertEqual(s.design_lifetime_hours, 1.0)

    def test_integer_lifetime_coerces_to_float(self):
        """Hand-authored JSONs sometimes use integer literals; the
        envelope already coerces to float and the new field must
        follow the same convention."""
        from core.session import json_to_session
        payload = json.dumps({
            "mach": 1.5, "alt_km": 12.0, "mass_kg": 1000.0,
            "R_n_m": 0.30, "g_load": 5.0, "char_len_m": 5.0,
            "flight_duration_s": 600.0, "wall_emissivity": 0.85,
            "design_lifetime_hours": 25000,  # integer literal
            "vehicle_category": "general",
            "system_label": "int-lifetime",
        })
        s = json_to_session(payload)
        self.assertIsInstance(s.design_lifetime_hours, float)
        self.assertEqual(s.design_lifetime_hours, 25000.0)


class TestPanelThicknessAndProfileRoundTrip(unittest.TestCase):
    """Phase 7.0 schema additions: ``panel_thickness_m`` and
    ``flight_profile``. Both are optional (defaulted) so older
    session JSONs without them load with the schema defaults.
    """

    def test_defaults_match_phase_7_plan(self):
        s = _minimal_session()
        self.assertEqual(s.panel_thickness_m, 0.002)
        self.assertEqual(s.flight_profile, ())

    def test_panel_thickness_round_trip(self):
        from dataclasses import replace
        from core.session import (
            session_to_dict, dict_to_session,
            session_to_json, json_to_session,
        )
        s = replace(_minimal_session(), panel_thickness_m=0.0015)
        self.assertEqual(
            dict_to_session(session_to_dict(s)).panel_thickness_m,
            0.0015,
        )
        self.assertEqual(
            json_to_session(session_to_json(s)).panel_thickness_m,
            0.0015,
        )

    def test_flight_profile_round_trip(self):
        from dataclasses import replace
        from core.session import session_to_json, json_to_session
        # Triplets are (t_s, mach, alt_km). JSON serialises tuple as
        # list, so the round-trip must re-coerce to tuple-of-tuples
        # on the way back in.
        profile = (
            (0.0, 0.0, 0.0),
            (10.0, 2.0, 9.0),
            (25.0, 0.5, 12.0),
        )
        s = replace(_minimal_session(), flight_profile=profile)
        round_tripped = json_to_session(session_to_json(s))
        self.assertEqual(round_tripped.flight_profile, profile)
        # Each element must be a plain tuple of floats so downstream
        # consumers don't accidentally mutate the profile.
        for sample in round_tripped.flight_profile:
            self.assertIsInstance(sample, tuple)
            for v in sample:
                self.assertIsInstance(v, float)

    def test_legacy_json_without_new_fields_loads_with_defaults(self):
        from core.session import json_to_session
        legacy = json.dumps({
            "mach": 1.5, "alt_km": 12.0, "mass_kg": 1000.0,
            "R_n_m": 0.30, "g_load": 5.0, "char_len_m": 5.0,
            "flight_duration_s": 600.0, "wall_emissivity": 0.85,
            "vehicle_category": "general",
            "system_label": "legacy-pre-phase-7",
        })
        s = json_to_session(legacy)
        self.assertEqual(s.panel_thickness_m, 0.002)
        self.assertEqual(s.flight_profile, ())

    def test_flight_profile_non_sequence_raises(self):
        from core.session import dict_to_session
        bad = {
            "mach": 1.5, "alt_km": 12.0, "mass_kg": 1000.0,
            "R_n_m": 0.30, "g_load": 5.0, "char_len_m": 5.0,
            "flight_duration_s": 600.0, "wall_emissivity": 0.85,
            "vehicle_category": "general",
            "system_label": "bad-profile",
            "flight_profile": "not a sequence",
        }
        with self.assertRaises(ValueError):
            dict_to_session(bad)


class TestPresetParity(unittest.TestCase):
    """Every bundled presets/*.json file must (a) parse cleanly via
    json_to_session, and (b) have a matching (vehicle_category, mach)
    entry in core.presets.CANONICAL_PRESETS.

    Repointed from the deleted app.EXAMPLES dict to the bundled JSON
    folder when the in-sidebar preset dropdown was removed. Catches
    the case where someone hand-edits a bundled JSON file or forgets
    to re-run scripts/generate_example_presets.py after editing
    CANONICAL_PRESETS in core/presets.py.
    """

    def test_every_bundled_json_has_canonical_match(self):
        from pathlib import Path
        from core.presets import CANONICAL_PRESETS
        from core.session import json_to_session

        presets_dir = Path(__file__).resolve().parent / "presets"
        if not presets_dir.is_dir():
            self.skipTest(
                f"{presets_dir} not present -- run "
                "scripts/generate_example_presets.py to populate it."
            )

        canonical_index = {
            (s.vehicle_category, round(s.mach, 2)): name
            for name, s in CANONICAL_PRESETS.items()
        }

        json_files = sorted(presets_dir.glob("*.json"))
        self.assertGreaterEqual(
            len(json_files), 1,
            f"{presets_dir} contains no .json files -- bundled examples "
            "are missing.",
        )

        for path in json_files:
            with self.subTest(file=path.name):
                schema = json_to_session(path.read_text(encoding="utf-8"))
                lookup = (schema.vehicle_category, round(schema.mach, 2))
                self.assertIn(
                    lookup, canonical_index,
                    f"Bundled preset {path.name} (category="
                    f"{schema.vehicle_category}, mach={schema.mach}) "
                    "has no corresponding entry in CANONICAL_PRESETS. "
                    "Re-run scripts/generate_example_presets.py or "
                    "update CANONICAL_PRESETS in core/presets.py.",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
