"""
Live wiring tests for app.py via Streamlit's AppTest harness.

After the bundled-JSON-presets rebuild:
  * The sidebar preset dropdown is gone. Bundled examples are loaded
    via the "Load bundled example" dropdown inside the Session I/O
    expander, sourced from presets/*.json files.
  * EXAMPLES, _PRESET_NOTES, VEHICLE_IMG_LOCAL/URL,
    _build_sectioned_preset_options, _is_preset_separator are all
    deleted from app.py.
  * The category_override selectbox is on the main canvas (Setup tab).
  * Inputs (Mach / altitude / mass / R_n / g / hot-section) are on the
    main canvas (Setup tab), NOT the sidebar.

Tests below guard the live wiring of the new bundled-example dropdown,
the staging chain that propagates loaded values, and the tabs / form
hierarchy.

Run: python -m unittest test_app_live.py -v
"""

import os
import unittest
from pathlib import Path

os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

# app.py lives at the project root, one level above tests/
_APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

from streamlit.testing.v1 import AppTest  # noqa: E402

import app  # noqa: E402
from core.session import json_to_session  # noqa: E402


def _safe_get(state, key, default=None):
    """AppTest's session_state raises KeyError on missing keys instead of
    supporting .get(). Tiny helper so individual asserts read cleanly."""
    try:
        return state[key]
    except (KeyError, AttributeError):
        return default


def _run_default():
    at = AppTest.from_file(_APP_PATH,default_timeout=30)
    at.run()
    return at


def _run_with_category_override(category):
    """Pin the category override (skipping the auto-inference branch)."""
    at = AppTest.from_file(_APP_PATH,default_timeout=30)
    at.session_state["category_override"] = category
    at.run()
    return at


def _run_with_bundled_example(display_label):
    """Drive the new 'Load bundled example' dropdown directly by label."""
    at = AppTest.from_file(_APP_PATH,default_timeout=30)
    at.session_state["bundled_example_select"] = display_label
    at.run()
    return at


# ---------------------------------------------------------------------------
# TestSidebarSmoke — every override renders without exceptions
# ---------------------------------------------------------------------------
class TestSidebarSmoke(unittest.TestCase):
    """Default render and per-category-override render must not raise."""

    def test_default_render_no_exceptions(self):
        at = _run_default()
        self.assertEqual(list(at.exception), [])

    def test_every_category_override_renders_without_exceptions(self):
        for cat in ("__auto__", "general", "aircraft", "hypersonic_aircraft",
                    "reentry", "hypersonic_missile", "turbine"):
            with self.subTest(category=cat):
                at = _run_with_category_override(cat)
                self.assertEqual(list(at.exception), [])

    def test_no_preset_dropdown_at_top_level_of_sidebar(self):
        """The legacy 'CALIBRATION PRESETS' dropdown was removed from
        the sidebar. The bundled-example loader now lives INSIDE the
        Session I/O expander instead."""
        at = _run_default()
        sidebar_selectboxes = [
            s for s in at.sidebar.selectbox
            if s.key in ("preset_select", "example_select")
        ]
        self.assertEqual(
            len(sidebar_selectboxes), 0,
            "Legacy preset selectbox still present in sidebar — should "
            "have been removed when bundled JSON files replaced it.",
        )


# ---------------------------------------------------------------------------
# TestBundledExampleLoadingLive — selecting a bundled JSON populates state
# ---------------------------------------------------------------------------
class TestBundledExampleLoadingLive(unittest.TestCase):
    """Selecting a bundled example from the Session I/O dropdown must
    propagate the envelope into alt_km / mass_kg / mach / R_n / peak_g
    AND set category_override to the loaded JSON's vehicle_category.

    Each row of PRESETS is the system_label inside a presets/*.json file
    (which the dropdown surfaces as its display label). Updates to the
    bundled set should regenerate via:
        python scripts/generate_example_presets.py
    and the corresponding row here updated to match."""

    PRESETS = [
        # (display_label, expected_alt, expected_mass, expected_category)
        # Two validation-anchored (SR-71, Concorde) + one audience-
        # relevant (Collegiate Sounding Rocket, NOT in VALIDATION.md).
        ("SR-71 Blackbird",                25.0, 30600.0, "aircraft"),
        ("Concorde",                       18.0, 78000.0, "aircraft"),
        ("Collegiate Sounding Rocket",      9.0,    30.0, "general"),
    ]

    def test_each_bundled_example_propagates(self):
        for label, expected_alt, expected_mass, expected_cat in self.PRESETS:
            with self.subTest(preset=label):
                at = _run_with_bundled_example(label)
                self.assertEqual(list(at.exception), [])
                self.assertEqual(
                    _safe_get(at.session_state, "alt_km"), expected_alt,
                    f"{label!r}: alt_km did not propagate from bundled JSON",
                )
                self.assertEqual(
                    _safe_get(at.session_state, "mass_kg"), expected_mass,
                    f"{label!r}: mass_kg did not propagate from bundled JSON",
                )
                self.assertEqual(
                    _safe_get(at.session_state, "category_override"),
                    expected_cat,
                    f"{label!r}: category_override should be {expected_cat!r}",
                )
                self.assertEqual(
                    _safe_get(at.session_state, "_last_bundled_loaded"),
                    label,
                    "_last_bundled_loaded marker did not advance after load",
                )

    def test_system_label_propagates_for_sr71(self):
        """Loading the SR-71 bundled preset should set system_label
        to 'SR-71 Blackbird' (the JSON's system_label field)."""
        at = _run_with_bundled_example("SR-71 Blackbird")
        self.assertEqual(
            _safe_get(at.session_state, "system_label"),
            "SR-71 Blackbird",
        )


# ---------------------------------------------------------------------------
# TestBundledFolderInvariants — the presets/ folder is well-formed
# ---------------------------------------------------------------------------
class TestBundledFolderInvariants(unittest.TestCase):
    """Every presets/*.json file must round-trip through json_to_session
    and surface a non-empty system_label. Catches a hand-edit gone wrong
    (or a partial run of the generation script) before it bites a user."""

    def test_presets_dir_exists(self):
        self.assertTrue(
            app._PRESETS_DIR.is_dir(),
            f"{app._PRESETS_DIR} does not exist — run "
            "scripts/generate_example_presets.py to populate it.",
        )

    def test_every_bundled_json_round_trips(self):
        for path in sorted(app._PRESETS_DIR.glob("*.json")):
            with self.subTest(file=path.name):
                schema = json_to_session(path.read_text(encoding="utf-8"))
                self.assertTrue(
                    schema.system_label,
                    f"{path.name}: system_label is empty",
                )

    def test_at_least_one_bundled_example(self):
        """If the folder is empty the dropdown silently disappears —
        better to fail fast in CI."""
        files = list(app._PRESETS_DIR.glob("*.json"))
        self.assertGreaterEqual(
            len(files), 1,
            f"{app._PRESETS_DIR} contains no .json files — bundled "
            "examples are missing.",
        )


# ---------------------------------------------------------------------------
# TestMainCanvasFormHierarchy — Flight Envelope form is on the main canvas
# inside the Setup tab; the page is split into 3 tabs.
# ---------------------------------------------------------------------------
class TestMainCanvasFormHierarchy(unittest.TestCase):
    """After the tabs/multi-page rebuild, the input widgets live inside
    the Setup tab on the main canvas, NOT in the sidebar. The page also
    has three top-level tabs (Setup / Results / Trade-offs)."""

    EXPECTED_TAB_LABELS = ["Setup", "Results", "Trade-offs"]

    def test_three_top_level_tabs_present(self):
        at = _run_default()
        labels = [t.label for t in at.tabs]
        for expected in self.EXPECTED_TAB_LABELS:
            self.assertIn(
                expected, labels,
                f"Top-level tab {expected!r} missing. Got: {labels}",
            )

    def test_alt_km_input_is_on_main_canvas_not_sidebar(self):
        at = _run_default()
        sidebar_alt_inputs = [
            n for n in at.sidebar.number_input
            if getattr(n, "label", "") == "CRUISE ALTITUDE (km)"
        ]
        self.assertEqual(
            len(sidebar_alt_inputs), 0,
            "CRUISE ALTITUDE input still in sidebar — should be on main canvas.",
        )
        main_alt_inputs = [
            n for n in at.number_input
            if getattr(n, "label", "") == "CRUISE ALTITUDE (km)"
        ]
        self.assertGreaterEqual(
            len(main_alt_inputs), 1,
            "CRUISE ALTITUDE input not found on main canvas after rebuild.",
        )

    def test_category_override_widget_is_on_main_canvas(self):
        at = _run_default()
        sidebar_cat = [s for s in at.sidebar.selectbox if s.key == "category_override"]
        self.assertEqual(
            len(sidebar_cat), 0,
            "category_override selectbox should not be in the sidebar.",
        )
        main_cat = [s for s in at.selectbox if s.key == "category_override"]
        self.assertGreaterEqual(
            len(main_cat), 1,
            "category_override selectbox not found on main canvas.",
        )

    def test_turbine_metric_strip_replaces_freestream_metrics(self):
        """When the user picks the turbine category, the metric strip
        should show the hot-section override + creep-derate + reference-
        frame note INSTEAD of the misleading freestream metrics."""
        at = AppTest.from_file(_APP_PATH,default_timeout=30)
        at.session_state["category_override"] = "turbine"
        at.run()
        labels = [m.label for m in at.metric]
        self.assertIn("Hot-section metal-face T", labels)
        self.assertIn("Strength derate", labels)
        for forbidden in ("Recovery T_wall", "Wall T (SG model)", "Dynamic pressure", "Flight regime"):
            self.assertNotIn(
                forbidden, labels,
                f"freestream metric {forbidden!r} leaked into turbine strip",
            )

    def test_envelope_chip_renders_above_tabs(self):
        at = _run_default()
        regime_tokens = ("SUBSONIC", "SUPERSONIC", "HYPERSONIC", "REENTRY")
        markdown_blobs = [m.value for m in at.markdown]
        has_chip = any(
            any(tok in (m or "") for tok in regime_tokens)
            for m in markdown_blobs
        )
        self.assertTrue(
            has_chip,
            "Persistent envelope chip not found in main markdown.",
        )


# ---------------------------------------------------------------------------
# TestComponentZonesLive — the per-zone section renders inside the Results
# tab when the matching engine produces zone results for the active
# vehicle category. The headline materials table represents the worst
# zone of the vehicle; the per-zone breakdown surfaces how the
# recommendation shifts across leading edge / fuselage / internal
# substructure.
# ---------------------------------------------------------------------------
class TestComponentZonesLive(unittest.TestCase):
    """Surfacing the existing core/component_zones.py module into the
    Streamlit Results tab. The PDF report already shows this section
    via latex_export._sec_component_zones; the UI was the missing half."""

    def _run_sr71_envelope(self):
        at = AppTest.from_file(_APP_PATH,default_timeout=30)
        at.session_state["mach"]              = 3.2
        at.session_state["alt_km"]            = 25.0
        at.session_state["mass_kg"]           = 30600.0
        at.session_state["R_n"]               = 0.15
        at.session_state["peak_g"]            = 2.5
        at.session_state["category_override"] = "aircraft"
        at.run()
        return at

    def test_per_zone_section_renders_for_supersonic_envelope(self):
        """Loading an SR-71-class envelope (aircraft category, 4 zones)
        must surface the 'Per-Zone Material Recommendations' header in
        the Results tab without raising."""
        at = self._run_sr71_envelope()
        self.assertEqual(list(at.exception), [])
        markdown_blobs = [m.value for m in at.markdown if m.value]
        self.assertTrue(
            any("Per-Zone Material" in m for m in markdown_blobs),
            "Per-zone section header did not render for SR-71 envelope. "
            "Markdown blobs that did appear: "
            f"{[m[:60] for m in markdown_blobs[-10:]]}",
        )

    def test_per_zone_renders_zone_titles_and_local_t_wall(self):
        """Each zone card emits its zone name (from CATEGORY_ZONES) AND
        a 'T_wall local' label. Aircraft category has 4 zones — at least
        the leading-edge title + one local-T_wall label must appear."""
        from core.component_zones import CATEGORY_ZONES

        expected_zone_names = [z.name for z in CATEGORY_ZONES["aircraft"]]
        at = self._run_sr71_envelope()
        markdown_blobs = [m.value for m in at.markdown if m.value]
        joined = "\n".join(markdown_blobs)

        # All 4 aircraft-category zone titles should appear somewhere
        # in the rendered markdown (each is the title of its own
        # _instrument_card).
        for name in expected_zone_names:
            self.assertIn(
                name, joined,
                f"Zone {name!r} title missing from rendered markdown.",
            )

        # The local-T_wall label is shared across all zone cards.
        self.assertIn(
            "T_wall local", joined,
            "Per-zone cards should label the primary value as 'T_wall local'.",
        )


# ---------------------------------------------------------------------------
# TestCategoryModeBadgeLive — the Results tab surfaces a single-line
# disclosure of what the active category does to the matching engine.
# Default is "general" (no category-specific exclusions); switching
# categories must update the badge text.
# ---------------------------------------------------------------------------
class TestCategoryModeBadgeLive(unittest.TestCase):
    """The category-mode badge is the user-facing trace of what the
    active category does. Without it, switching categories silently
    changes the recommendations and the user has no way to see why."""

    def test_default_category_renders_general_mode_badge(self):
        """First-load default is now 'general' (was '__auto__')."""
        at = _run_default()
        self.assertEqual(list(at.exception), [])
        blobs = "\n".join(m.value for m in at.markdown if m.value)
        self.assertIn(
            "General mode", blobs,
            "Default render should surface the General mode badge "
            "(category_override now defaults to 'general').",
        )

    def test_aircraft_category_renders_aircraft_mode_badge(self):
        at = _run_with_category_override("aircraft")
        self.assertEqual(list(at.exception), [])
        blobs = "\n".join(m.value for m in at.markdown if m.value)
        self.assertIn(
            "Aircraft mode", blobs,
            "Aircraft category override should surface the Aircraft "
            "mode badge.",
        )


class TestCreepBannerLive(unittest.TestCase):
    """The lifecycle / creep banner appears on the Results tab when
    design_lifetime_hours >= 1000 h. Default lifetime is 1.0 h
    (preserves pre-Phase-4 behaviour) so the banner is hidden until
    the user loads a long-lifetime preset or types a real lifetime
    in the Advanced expander."""

    def test_default_lifetime_hides_creep_banner(self):
        at = _run_default()
        self.assertEqual(list(at.exception), [])
        blobs = "\n".join(m.value for m in at.markdown if m.value)
        self.assertNotIn(
            "Lifecycle screening active", blobs,
            "Default render (lifetime=1 h) should NOT show the "
            "creep-screening banner.",
        )

    def test_concorde_preset_shows_creep_banner(self):
        """Loading the Concorde bundled preset auto-fills lifetime
        to 25,000 h, which crosses the 1000 h threshold and triggers
        the banner."""
        at = _run_with_bundled_example("Concorde")
        self.assertEqual(list(at.exception), [])
        blobs = "\n".join(m.value for m in at.markdown if m.value)
        self.assertIn(
            "Lifecycle screening active", blobs,
            "Concorde preset (25,000 h lifetime) should surface the "
            "creep-screening banner.",
        )
        self.assertIn(
            "Creep@Life", blobs,
            "The Creep@Life column header should appear in the "
            "materials table when creep is active.",
        )

    def test_collegiate_sounding_rocket_does_not_show_creep_banner(self):
        """Sounding rocket has design_lifetime_hours=1.0, which is
        below the 1000 h threshold for triggering creep screening.
        The banner must stay hidden so the materials list reflects
        single-flight (no creep) behaviour."""
        at = _run_with_bundled_example("Collegiate Sounding Rocket")
        self.assertEqual(list(at.exception), [])
        blobs = "\n".join(m.value for m in at.markdown if m.value)
        self.assertNotIn(
            "Lifecycle screening active", blobs,
            "1-h lifetime should NOT trigger the creep-screening "
            "banner --- the screening is silent at single-flight "
            "lifetimes.",
        )

    def test_collegiate_sounding_rocket_shows_transient_banner(self):
        """Sounding rocket flight_duration_s=25 triggers the 1D
        transient heat solver. The banner must appear AND the
        Soak@Life column header must be in the materials table."""
        at = _run_with_bundled_example("Collegiate Sounding Rocket")
        self.assertEqual(list(at.exception), [])
        blobs = "\n".join(m.value for m in at.markdown if m.value)
        self.assertIn(
            "Transient screening active", blobs,
            "Sounding rocket (25 s flight) should surface the "
            "transient-heat banner.",
        )
        self.assertIn(
            "Soak@Life", blobs,
            "The Soak@Life column header should appear in the "
            "materials table when the transient solver is active.",
        )

    def test_concorde_does_not_show_transient_banner(self):
        """Concorde is 10800 s (sustained) — transient solver does
        not run and the banner stays hidden. Static T_wall check
        does the screening."""
        at = _run_with_bundled_example("Concorde")
        self.assertEqual(list(at.exception), [])
        blobs = "\n".join(m.value for m in at.markdown if m.value)
        self.assertNotIn(
            "Transient screening active", blobs,
            "Concorde (10,800 s) should NOT trigger the transient "
            "banner --- sustained-flight envelopes use the static "
            "T_wall check.",
        )


class TestBundledPresetDirectStaging(unittest.TestCase):
    """Regression: selecting a bundled preset must propagate the
    Phase-7 widget values (``design_lifetime_hours`` and
    ``panel_thickness_mm``) on the FIRST rerun, with no Reload button
    or extra user action required.

    The earlier implementation routed bundled-preset values through
    the same pending+rerun bridge used by the file uploader. That
    two-rerun dance was observed to drop these two specific keys in
    live Streamlit sessions even though it round-tripped cleanly
    under headless AppTest. The fix moved the bundled-preset stage
    step to the very TOP of ``_sidebar()`` (before any widget renders)
    so the values land directly into ``session_state`` without
    bouncing through ``_pending_session_values``.
    """

    def test_sr71_lifetime_and_thickness_stage_without_reload(self):
        # Simulate the dropdown-pick selecting SR-71 by pre-seeding
        # the selectbox's session key. The top-of-sidebar stager
        # should observe the choice, find the matching JSON, and
        # write design_lifetime_hours + panel_thickness_mm directly.
        at = AppTest.from_file(_APP_PATH,default_timeout=60)
        at.session_state["bundled_example_select"] = "SR-71 Blackbird"
        at.run()
        self.assertEqual(
            _safe_get(at.session_state, "design_lifetime_hours"),
            3000.0,
            "SR-71 lifetime (3000 h) must stage on the first rerun "
            "after picking the bundled preset — no Reload click.",
        )
        self.assertEqual(
            _safe_get(at.session_state, "panel_thickness_mm"),
            1.5,
            "SR-71 panel thickness (1.5 mm) must stage on the first "
            "rerun after picking the bundled preset.",
        )
        self.assertEqual(
            _safe_get(at.session_state, "flight_duration_s"),
            5400.0,
            "SR-71 flight duration should also stage on the first rerun.",
        )

    def test_gate_prevents_restage_after_manual_edit(self):
        """Once ``_last_bundled_loaded`` matches the dropdown choice,
        the stager must NOT re-apply preset values — otherwise manual
        widget edits would be clobbered on the next sidebar repaint."""
        at = AppTest.from_file(_APP_PATH,default_timeout=60)
        at.session_state["bundled_example_select"] = "SR-71 Blackbird"
        at.session_state["_last_bundled_loaded"] = "SR-71 Blackbird"
        # Simulate the user having edited the lifetime AFTER the
        # preset loaded; the gate should preserve their edit.
        at.session_state["design_lifetime_hours"] = 42.0
        at.run()
        self.assertEqual(
            _safe_get(at.session_state, "design_lifetime_hours"),
            42.0,
            "Stager must respect _last_bundled_loaded gate so manual "
            "edits to design_lifetime_hours survive a sidebar repaint.",
        )


class TestExportPlumbsPhase7Kwargs(unittest.TestCase):
    """Regression: clicking the PDF / .tex export buttons must call
    ``generate_report`` / ``generate_tex_source`` with the live
    ``design_lifetime_hours`` and ``panel_thickness_m`` values from the
    active SessionSchema. An earlier bug had ``_show_export`` omitting
    these kwargs entirely, so exported PDFs always reported the
    function-signature defaults (1 h / 2 mm) regardless of what the
    user picked or what the on-screen materials table showed.
    """

    def test_show_export_signature_accepts_phase7_kwargs(self):
        import inspect
        sig = inspect.signature(app._show_export)
        self.assertIn("design_lifetime_hours", sig.parameters)
        self.assertIn("panel_thickness_m", sig.parameters)

    def test_main_passes_phase7_kwargs_to_show_export(self):
        """Static check: main() must pass design_lifetime_hours and
        panel_thickness_m into _show_export. Without this the PDF
        export silently uses the function-signature defaults."""
        from pathlib import Path
        src = Path(_APP_PATH).read_text(encoding="utf-8")
        # Skip the "def _show_export(" line and find the call site.
        idx = 0
        while True:
            idx = src.find("_show_export(", idx)
            if idx == -1:
                self.fail("Could not find _show_export(...) call site")
            # Look at the text just before the match: a "def " preface
            # marks the function definition, anything else is a call.
            prefix_start = max(0, idx - 4)
            if src[prefix_start:idx] == "def ":
                idx += len("_show_export(")
                continue
            # Walk forward to the matching close-paren.
            depth = 0
            j = idx
            while j < len(src):
                if src[j] == "(":
                    depth += 1
                elif src[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            call = src[idx:j + 1]
            self.assertIn("design_lifetime_hours=", call)
            self.assertIn("panel_thickness_m=", call)
            return


if __name__ == "__main__":
    unittest.main(verbosity=2)
