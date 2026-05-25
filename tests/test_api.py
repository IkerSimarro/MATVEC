"""
Unit tests for core/api.py and the matvec CLI.

Scope:
  * ``run_session()`` on every CANONICAL_PRESETS entry produces a
    populated SessionResult: physics / match / tex_source non-empty.
  * One end-to-end pdflatex compile on a representative preset
    (skipped cleanly if pdflatex is not on PATH — CI without TeX must
    still exercise the data-path checks).
  * ``python -m matvec run <preset>.json --out <pdf>`` subprocess
    smoke test (skipped if pdflatex missing).
  * Streamlit-free invariant: the CLI entry point, the pipeline, and
    the schema module must import cleanly when streamlit is stubbed
    out with ``sys.modules["streamlit"] = None``.
  * Static check: no ``import streamlit`` / ``from streamlit`` in the
    three files that must stay headless.

The "populated MatchResult" bar is deliberately low (≥0 viable, ≥0
marginal, ≥0 not_viable but total > 0) — this file asserts the
*pipeline is wired*, not *the physics answer is correct*. Physics
validation lives in test_physics_engine.py / test_matching_engine.py.
"""

import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from core import MATVEC_VERSION
from core.api import run_session, SessionResult, apply_turbine_override
from core.presets import CANONICAL_PRESETS
from core.session import SessionSchema, session_to_json


_REPO_ROOT = Path(__file__).resolve().parent.parent
_PDFLATEX_AVAILABLE = shutil.which("pdflatex") is not None


# ---------------------------------------------------------------------------
# run_session on every preset (fast — compile_pdf=False)
# ---------------------------------------------------------------------------

class TestRunSessionAllPresets(unittest.TestCase):
    """Every preset must flow through the full (non-PDF) pipeline
    without raising and produce a non-degenerate SessionResult."""

    def test_every_preset_produces_populated_result(self):
        for name, session in CANONICAL_PRESETS.items():
            with self.subTest(preset=name):
                result = run_session(session, compile_pdf=False)
                self.assertIsInstance(result, SessionResult)

                # Physics populated
                self.assertIsNotNone(result.physics)
                self.assertGreater(result.physics.thermal.T_wall_K, 0.0)
                self.assertGreater(
                    result.physics.structural.sigma_tensile_required_MPa, 0.0
                )

                # Match populated (total count > 0 — materials_db has 56
                # entries, every run sees every entry).
                total = (
                    len(result.match.viable)
                    + len(result.match.marginal)
                    + len(result.match.not_viable)
                )
                self.assertGreater(
                    total, 0,
                    f"{name}: match pipeline produced zero candidates.",
                )

                # LaTeX source written
                self.assertIsInstance(result.tex_source, str)
                self.assertGreater(
                    len(result.tex_source), 500,
                    f"{name}: tex_source suspiciously short.",
                )

                # PDF explicitly skipped in this fast path
                self.assertIsNone(result.pdf_bytes)

    def test_turbine_preset_applies_hot_section_override(self):
        """The turbine preset carries hot_section_temp_K=1400 — the
        override should raise T_wall above what the aerodynamic
        recovery temperature would give at M=0.5/sea-level (~305 K)."""
        session = CANONICAL_PRESETS["Turbine HPT Blade"]
        self.assertEqual(session.vehicle_category, "turbine")
        self.assertEqual(
            session.options.get("hot_section_temp_K"), 1400.0,
        )
        result = run_session(session, compile_pdf=False)
        self.assertAlmostEqual(
            result.physics.thermal.T_wall_K, 1400.0, delta=1e-3,
        )
        self.assertEqual(
            result.physics.thermal.thermal_source,
            "turbine_inlet_override",
        )


# ---------------------------------------------------------------------------
# End-to-end PDF compile on one representative preset
# ---------------------------------------------------------------------------

class TestRunSessionPdfCompile(unittest.TestCase):
    """The slow path — compile one real PDF so we catch pdflatex /
    latex_export integration breakage that the fast path misses."""

    @unittest.skipUnless(
        _PDFLATEX_AVAILABLE,
        "pdflatex not on PATH — skipping real PDF compile.",
    )
    def test_sr71_preset_produces_non_empty_pdf(self):
        session = CANONICAL_PRESETS["SR-71 Blackbird"]
        result = run_session(session, compile_pdf=True)
        self.assertIsNotNone(result.pdf_bytes)
        self.assertGreater(
            len(result.pdf_bytes), 10_000,
            "PDF bytes suspiciously small — compile may have produced a "
            "stub.",
        )
        self.assertTrue(
            result.pdf_bytes.startswith(b"%PDF-"),
            "Payload does not start with the %PDF- magic header.",
        )


# ---------------------------------------------------------------------------
# CLI subprocess smoke test
# ---------------------------------------------------------------------------

class TestCliSubprocess(unittest.TestCase):
    """Run the CLI as a real subprocess, exactly as a user would.

    This catches import-time failures the in-process tests miss
    (e.g. a stray ``import streamlit`` inside __main__.py).
    """

    def _run_cli(self, *args, cwd=None, timeout=120):
        """Invoke ``python -m matvec ...`` with the current interpreter."""
        return subprocess.run(
            [sys.executable, "-m", "matvec", *args],
            cwd=str(cwd or _REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_version_flag(self):
        """--version is the cheapest end-to-end check: argparse wires
        up, imports succeed, no streamlit pulled in by default."""
        proc = self._run_cli("--version")
        self.assertEqual(
            proc.returncode, 0,
            f"CLI --version failed.\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}",
        )
        self.assertIn(MATVEC_VERSION, proc.stdout)

    def test_help_flag_lists_all_subcommands(self):
        proc = self._run_cli("--help")
        self.assertEqual(proc.returncode, 0)
        for sub in ("run", "validate", "batch"):
            self.assertIn(sub, proc.stdout)

    @unittest.skipUnless(
        _PDFLATEX_AVAILABLE,
        "pdflatex not on PATH — skipping CLI PDF smoke test.",
    )
    def test_run_subcommand_produces_pdf_on_disk(self):
        """End-to-end: write a preset JSON, run the CLI, confirm the
        PDF appeared with the expected magic header."""
        tmp_dir = _REPO_ROOT / "_test_api_cli_tmp"
        tmp_dir.mkdir(exist_ok=True)
        try:
            json_path = tmp_dir / "sr71.json"
            pdf_path = tmp_dir / "sr71.pdf"
            json_path.write_text(
                session_to_json(CANONICAL_PRESETS["SR-71 Blackbird"]),
                encoding="utf-8",
            )
            proc = self._run_cli(
                "run", str(json_path), "--out", str(pdf_path),
                timeout=300,
            )
            self.assertEqual(
                proc.returncode, 0,
                f"CLI run failed.\nstdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}",
            )
            self.assertTrue(pdf_path.is_file())
            self.assertGreater(pdf_path.stat().st_size, 10_000)
            self.assertTrue(
                pdf_path.read_bytes()[:5] == b"%PDF-",
            )
        finally:
            # Best-effort cleanup — test isolation trumps leaving
            # artefacts behind for debugging.
            for f in tmp_dir.glob("*"):
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                tmp_dir.rmdir()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Streamlit-free invariants
# ---------------------------------------------------------------------------

_HEADLESS_FILES = (
    _REPO_ROOT / "core" / "api.py",
    _REPO_ROOT / "core" / "session.py",
    _REPO_ROOT / "matvec" / "__main__.py",
    _REPO_ROOT / "matvec" / "__init__.py",
    _REPO_ROOT / "core" / "presets.py",
    _REPO_ROOT / "core" / "sensitivity.py",
)


class TestStreamlitFreeCLI(unittest.TestCase):
    """The whole point of the core/ + matvec/ split is to let MATVEC
    run headlessly. Any regression that re-introduces a streamlit
    import into these files must trip a test, not a user."""

    def test_no_streamlit_imports_in_headless_files(self):
        """Static source scan — cheap and catches the 90% case
        (someone adds ``import streamlit as st`` at the top of
        core/api.py by reflex)."""
        pat = re.compile(
            r"^\s*(?:import\s+streamlit|from\s+streamlit)\b",
            re.MULTILINE,
        )
        for path in _HEADLESS_FILES:
            with self.subTest(path=str(path.relative_to(_REPO_ROOT))):
                self.assertTrue(
                    path.is_file(),
                    f"Headless file missing from tree: {path}",
                )
                src = path.read_text(encoding="utf-8")
                self.assertIsNone(
                    pat.search(src),
                    f"{path} contains a streamlit import — that breaks "
                    f"the CLI-in-headless-env contract.",
                )

    def test_headless_modules_import_without_streamlit(self):
        """Functional check: stub streamlit out of sys.modules, force
        reimport of every headless module, confirm no ImportError.

        This catches *transitive* streamlit imports — e.g. if someone
        made core/api.py import app.py, the static scan above would
        still pass but this test would fail."""
        # Identify the modules we care about. They may already be
        # imported (from this test file's imports), so we force-evict
        # them and reimport under the streamlit-stub.
        headless_modules = [
            "core",
            "core.api",
            "core.session",
            "core.presets",
            "core.sensitivity",
            "matvec",
            "matvec.__main__",
        ]

        # Snapshot & remove the real modules so the reimport is honest.
        saved = {
            name: sys.modules[name]
            for name in headless_modules
            if name in sys.modules
        }
        # Also evict streamlit itself so our stub wins the race.
        saved_streamlit = sys.modules.pop("streamlit", None)

        try:
            # Block any attempt to import streamlit — setting the
            # value to None makes ``import streamlit`` raise
            # ImportError, which would fail this test loudly if any
            # headless module reaches for it.
            sys.modules["streamlit"] = None  # type: ignore[assignment]
            for name in headless_modules:
                sys.modules.pop(name, None)

            import importlib
            for name in headless_modules:
                try:
                    importlib.import_module(name)
                except ImportError as exc:
                    self.fail(
                        f"Headless module {name!r} failed to import "
                        f"with streamlit stubbed out: {exc}"
                    )
        finally:
            # Restore real state so the rest of the suite is unaffected.
            sys.modules.pop("streamlit", None)
            if saved_streamlit is not None:
                sys.modules["streamlit"] = saved_streamlit
            for name, mod in saved.items():
                sys.modules[name] = mod


# ---------------------------------------------------------------------------
# apply_turbine_override unit semantics (moved from app.py)
# ---------------------------------------------------------------------------

class TestTurbineOverride(unittest.TestCase):
    """``apply_turbine_override`` moved from app.py to core.api in the
    CLI refactor. A direct unit test here pins the observable contract
    so a future reorganization of physics_engine doesn't silently
    change what ``run_session`` does for the turbine branch."""

    def test_override_rewrites_t_wall_and_band(self):
        session = CANONICAL_PRESETS["Turbine HPT Blade"]
        # Get a pre-override physics object (without the override).
        from core.physics_engine import run_analysis
        physics = run_analysis(
            session.mach, session.alt_km, session.mass_kg,
            session.R_n_m,
            peak_g_load=session.g_load,
            wall_emissivity=session.wall_emissivity,
            characteristic_length_m=session.char_len_m,
            flight_duration_s=session.flight_duration_s,
        )
        # Sanity: aerodynamic T_wall at M=0.5/SL should be modest.
        self.assertLess(physics.thermal.T_wall_K, 400.0)

        result = apply_turbine_override(physics, 1400.0)
        self.assertAlmostEqual(result.thermal.T_wall_K, 1400.0)
        self.assertAlmostEqual(
            result.thermal.T_wall_min_K, 1400.0 * 0.95,
        )
        self.assertAlmostEqual(
            result.thermal.T_wall_max_K, 1400.0 * 1.05,
        )
        self.assertEqual(
            result.thermal.thermal_source, "turbine_inlet_override",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
