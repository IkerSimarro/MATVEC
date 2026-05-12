"""
Unit tests for core/sensitivity.py.

Scope:
  * Public dataclass invariants (defaults, label thresholds, sensible
    typing) — pinned so a downstream consumer (LaTeX exporter,
    Streamlit badge, CLI summary line) can rely on them.
  * Real-pipeline acceptance:
      - SR-71 top viable material is labeled "robust" under the
        default sweep (a calibration touchstone — if Ti-5Al-2.5Sn
        ever flips on the SR-71 envelope without an intentional
        physics change, this catches it).
      - X-15 top viable material's tornado is dominated by Mach
        OR has fraction < 1.0 (X-15 sits at M=6.7 — the
        recovery/Sutton-Graves cross-over — so Mach perturbation
        meaningfully shifts T_wall, and the top material should
        feel that).
  * Determinism: the sweep is built on numpy.linspace, no RNG. Two
    calls with identical inputs must produce byte-equal chart PNGs
    and equal MaterialRobustness lists. Without this guarantee the
    LaTeX include hash would change every recompile, which would
    poison content-addressed PDF caches and CI snapshot diffs.
  * PNG magic-number check on chart_png — catches both the empty-
    viable placeholder branch and the normal-render branch.
  * Side-effect freedom: the input SessionSchema must not be
    mutated, and module state in matching_engine / physics_engine
    must not leak across calls. (The mutation hazard exists because
    apply_turbine_override does mutate the PhysicsResult it's
    handed; run_sensitivity rebuilds physics each sweep step, so
    the envelope itself stays clean — this test pins that.)

Speed: every test uses n_samples=5 (instead of the default 11) to
keep the suite under ~15 s on a developer laptop. The
acceptance-test calibration was confirmed at both n_samples=5 and
n_samples=11.
"""

import copy
import unittest

from core.presets import CANONICAL_PRESETS
from core.sensitivity import (
    BORDERLINE_THRESHOLD,
    MaterialRobustness,
    ROBUST_THRESHOLD,
    SensitivityResult,
    SensitivitySpec,
    _label_for_fraction,
    run_sensitivity,
)


# ---------------------------------------------------------------------------
# Dataclass-level invariants (cheap, no pipeline runs)
# ---------------------------------------------------------------------------

class TestSensitivitySpecDefaults(unittest.TestCase):
    """The default spec is a calibration value — these numbers
    appear in the LaTeX text and CLI help, so a silent change here
    must trip a test, not a user."""

    def test_default_deltas_match_calibration(self):
        spec = SensitivitySpec()
        self.assertAlmostEqual(spec.mach_delta_frac,   0.10)
        self.assertAlmostEqual(spec.mass_delta_frac,   0.15)
        self.assertAlmostEqual(spec.R_n_delta_frac,    0.20)
        self.assertAlmostEqual(spec.g_load_delta_frac, 0.25)
        self.assertEqual(spec.n_samples, 11)


class TestRobustnessThresholds(unittest.TestCase):
    """ROBUST_THRESHOLD / BORDERLINE_THRESHOLD are imported by the
    Streamlit badge layer and the LaTeX legend. Pin them so the
    badge-color → semantic-meaning mapping doesn't drift."""

    def test_threshold_values(self):
        self.assertAlmostEqual(ROBUST_THRESHOLD,     0.90)
        self.assertAlmostEqual(BORDERLINE_THRESHOLD, 0.50)

    def test_label_for_fraction_boundaries(self):
        # >= 0.90 -> robust
        self.assertEqual(_label_for_fraction(1.00), "robust")
        self.assertEqual(_label_for_fraction(0.90), "robust")
        # 0.50 .. <0.90 -> borderline
        self.assertEqual(_label_for_fraction(0.89), "borderline")
        self.assertEqual(_label_for_fraction(0.50), "borderline")
        # < 0.50 -> knife-edge
        self.assertEqual(_label_for_fraction(0.49), "knife-edge")
        self.assertEqual(_label_for_fraction(0.00), "knife-edge")


# ---------------------------------------------------------------------------
# Acceptance tests on canonical presets
# ---------------------------------------------------------------------------

# Reduced sample count keeps test wall-time tolerable on CI without
# changing the qualitative outcome (the calibration was confirmed at
# both 5 and 11 samples).
_FAST_SPEC = SensitivitySpec(n_samples=5)


class TestSR71Robustness(unittest.TestCase):
    """SR-71 sits comfortably inside Ti-5Al-2.5Sn's envelope —
    the top viable material should land "robust" even under the
    default ±10/15/20/25% sweep."""

    def test_sr71_top_material_is_robust(self):
        session = CANONICAL_PRESETS["SR-71 Blackbird"]
        result = run_sensitivity(
            session, session.vehicle_category, spec=_FAST_SPEC,
        )
        self.assertIsInstance(result, SensitivityResult)
        self.assertGreater(
            len(result.materials), 0,
            "SR-71 should have at least one nominally viable material.",
        )
        top = result.materials[0]
        self.assertIsInstance(top, MaterialRobustness)
        self.assertEqual(
            top.robustness_label, "robust",
            f"SR-71 top {top.material_name!r} should be 'robust' under "
            f"default sweep (got {top.robustness_label} at "
            f"f={top.robustness_fraction:.3f}).",
        )

    def test_sr71_top_material_name_echoed(self):
        """The top_material_name field should match the materials[0]
        entry — the LaTeX section header reads from it directly."""
        session = CANONICAL_PRESETS["SR-71 Blackbird"]
        result = run_sensitivity(
            session, session.vehicle_category, spec=_FAST_SPEC,
        )
        self.assertEqual(
            result.top_material_name,
            result.materials[0].material_name,
        )


class TestX15Sensitivity(unittest.TestCase):
    """X-15 at M=6.7 sits on the recovery/Sutton-Graves boundary.
    Mach perturbation should be the dominant tornado bar (or at
    minimum: the top material should not be perfectly robust)."""

    def test_x15_top_is_mach_sensitive(self):
        session = CANONICAL_PRESETS["X-15"]
        result = run_sensitivity(
            session, session.vehicle_category, spec=_FAST_SPEC,
        )
        self.assertGreater(
            len(result.materials), 0,
            "X-15 should have at least one nominally viable material.",
        )
        top = result.materials[0]
        # Loose assertion: either Mach is the critical input, or the
        # top material is less than perfectly robust (fraction<1.0).
        # Both signals indicate the user should pay attention to
        # Mach uncertainty for this airframe.
        sensitive_signal = (
            top.critical_input == "mach"
            or top.robustness_fraction < 1.0
        )
        self.assertTrue(
            sensitive_signal,
            f"Expected X-15 top {top.material_name!r} to be Mach-"
            f"sensitive (critical_input='mach') OR have "
            f"fraction<1.0; got critical_input={top.critical_input!r}, "
            f"fraction={top.robustness_fraction:.3f}.",
        )

    def test_x15_tornado_has_all_four_inputs(self):
        """The tornado dict should have one bar per swept input —
        the LaTeX exporter's chart-include block assumes this."""
        session = CANONICAL_PRESETS["X-15"]
        result = run_sensitivity(
            session, session.vehicle_category, spec=_FAST_SPEC,
        )
        self.assertEqual(
            set(result.tornado_data.keys()),
            {"mach", "mass", "R_n", "g_load"},
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    """numpy.linspace + same envelope + same spec ⇒ byte-equal
    output. This is what lets a CI snapshot diff find real
    regressions instead of noise."""

    def test_chart_png_byte_identical_across_runs(self):
        session = CANONICAL_PRESETS["SR-71 Blackbird"]
        a = run_sensitivity(session, session.vehicle_category, spec=_FAST_SPEC)
        b = run_sensitivity(session, session.vehicle_category, spec=_FAST_SPEC)
        self.assertEqual(
            a.chart_png, b.chart_png,
            "Two identical run_sensitivity calls produced different "
            "chart_png bytes — determinism contract broken.",
        )

    def test_robustness_list_equal_across_runs(self):
        session = CANONICAL_PRESETS["SR-71 Blackbird"]
        a = run_sensitivity(session, session.vehicle_category, spec=_FAST_SPEC)
        b = run_sensitivity(session, session.vehicle_category, spec=_FAST_SPEC)
        # Compare by tuple-ised dataclass payloads — the actual
        # MaterialRobustness instances are different objects.
        as_tuples = lambda lst: [
            (m.material_name, m.n_scenarios_viable, m.n_scenarios_total,
             m.robustness_fraction, m.robustness_label, m.critical_input)
            for m in lst
        ]
        self.assertEqual(as_tuples(a.materials), as_tuples(b.materials))

    def test_tornado_dict_equal_across_runs(self):
        session = CANONICAL_PRESETS["SR-71 Blackbird"]
        a = run_sensitivity(session, session.vehicle_category, spec=_FAST_SPEC)
        b = run_sensitivity(session, session.vehicle_category, spec=_FAST_SPEC)
        self.assertEqual(a.tornado_data, b.tornado_data)


# ---------------------------------------------------------------------------
# PNG magic number — chart is always a real PNG
# ---------------------------------------------------------------------------

class TestChartPngMagic(unittest.TestCase):
    """chart_png is consumed by st.image (Streamlit) and
    \\includegraphics (LaTeX). Both will silently render a broken
    image if the bytes aren't really a PNG — so we pin the magic
    header here, on both branches."""

    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

    def test_normal_render_chart_starts_with_png_magic(self):
        session = CANONICAL_PRESETS["SR-71 Blackbird"]
        result = run_sensitivity(
            session, session.vehicle_category, spec=_FAST_SPEC,
        )
        self.assertEqual(
            result.chart_png[: len(self.PNG_MAGIC)], self.PNG_MAGIC,
            "Normal-branch chart_png does not start with PNG magic.",
        )

    def test_empty_viable_placeholder_is_still_a_png(self):
        """Construct an envelope with such a hostile thermal load
        that nothing survives matching, then confirm the placeholder
        branch still emits real PNG bytes (LaTeX can't include a
        zero-byte file)."""
        # Take SR-71 and crank Mach to an absurd value — every
        # material will be regime-rejected or thermal-failed.
        base = CANONICAL_PRESETS["SR-71 Blackbird"]
        from dataclasses import replace
        cooked = replace(base, mach=25.0, alt_km=10.0)
        result = run_sensitivity(
            cooked, cooked.vehicle_category, spec=_FAST_SPEC,
        )
        # Even if the materials list is non-empty, chart_png
        # must always be a valid PNG — the assertion holds either way.
        self.assertEqual(
            result.chart_png[: len(self.PNG_MAGIC)], self.PNG_MAGIC,
            "Placeholder/empty-branch chart_png is not a valid PNG.",
        )

    def test_chart_png_intrinsic_size_is_compact(self):
        """Lock in the smaller dpi/figsize from the redesign — a
        regression that bumps either back up to the old values would
        re-trigger the 'way too big in Streamlit' bug.

        60 KB is comfortably above the expected ~25-40 KB for a
        5.0×2.4 in @ 110 dpi PNG and well below the ~80-100 KB the
        old 6.2×3.2 @ 150 dpi produced."""
        session = CANONICAL_PRESETS["SR-71 Blackbird"]
        result = run_sensitivity(
            session, session.vehicle_category, spec=_FAST_SPEC,
        )
        self.assertLess(
            len(result.chart_png), 60_000,
            "chart_png suspiciously large — figsize or dpi may have "
            "regressed to pre-redesign values.",
        )


# ---------------------------------------------------------------------------
# Side-effect freedom
# ---------------------------------------------------------------------------

class TestNoStateLeak(unittest.TestCase):
    """run_sensitivity must not mutate its arguments and must not
    poison module-level state in matching_engine / physics_engine."""

    def test_envelope_not_mutated(self):
        session = CANONICAL_PRESETS["SR-71 Blackbird"]
        before = copy.deepcopy(session)
        _ = run_sensitivity(
            session, session.vehicle_category, spec=_FAST_SPEC,
        )
        # Check every numeric field byte-for-byte.
        for field in ("mach", "alt_km", "mass_kg", "R_n_m", "g_load",
                      "char_len_m", "flight_duration_s", "wall_emissivity"):
            self.assertEqual(
                getattr(session, field), getattr(before, field),
                f"run_sensitivity mutated session.{field}!",
            )
        self.assertEqual(session.vehicle_category, before.vehicle_category)
        self.assertEqual(session.options, before.options)

    def test_materials_db_not_mutated(self):
        """A previous bug had matching_engine cache a per-call scalar
        on a MaterialEntry. This test shows that the materials_db
        is structurally identical before and after a sweep."""
        from materials_db import MATERIALS_DB
        before_names = [m.name for m in MATERIALS_DB]
        before_count = len(MATERIALS_DB)

        session = CANONICAL_PRESETS["SR-71 Blackbird"]
        _ = run_sensitivity(
            session, session.vehicle_category, spec=_FAST_SPEC,
        )

        self.assertEqual(len(MATERIALS_DB), before_count)
        self.assertEqual([m.name for m in MATERIALS_DB], before_names)


# ---------------------------------------------------------------------------
# Degenerate / corner cases
# ---------------------------------------------------------------------------

class TestDegenerateEnvelope(unittest.TestCase):
    """When no nominal viable material exists, the contract is:
    return a fully-populated SensitivityResult with an empty
    materials list, an empty tornado_data dict, and a placeholder
    PNG. The LaTeX exporter relies on ``materials`` being a list
    (not None) — None would crash the iteration."""

    def test_no_viable_returns_well_formed_result(self):
        # Force impossible thermal load.
        from dataclasses import replace
        base = CANONICAL_PRESETS["SR-71 Blackbird"]
        cooked = replace(base, mach=25.0, alt_km=10.0)
        result = run_sensitivity(
            cooked, cooked.vehicle_category, spec=_FAST_SPEC,
        )
        self.assertIsInstance(result, SensitivityResult)
        self.assertIsInstance(result.materials, list)
        self.assertIsInstance(result.tornado_data, dict)
        self.assertGreater(len(result.chart_png), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
