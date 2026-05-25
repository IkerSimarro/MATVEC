"""Pin the historical-match results from VALIDATION.md so future
matching-engine tweaks cannot silently break the credibility document.

Each test reuses the corresponding ``ValidationCase`` from
``scripts/run_validation.py`` and asserts that one of the
``expected_materials`` appears as a *strong* match (viable list, OR
tps_coatings bucket with positive thermal margin) for vehicles that
historically used TPS, OR in the viable / marginal list otherwise.

The bar is deliberately tolerant — we are pinning that the right
material *family* survives at the right envelope, not that ranking
order or score values are stable. If a future change to the matching
engine moves a hit from "viable" to "marginal" we want the test to
pass; if a change makes the historical material disappear from all
three buckets we want the test to fail loudly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Make the script-side VALIDATION_CASES importable without putting
# scripts/ on the package path. The script is self-contained and re-
# importing it just runs its module body once (no side effects beyond
# defining VALIDATION_CASES + helpers).
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_validation import VALIDATION_CASES, evaluate_case  # noqa: E402


class TestHistoricalValidation(unittest.TestCase):
    """One test per validation case; the test name embeds the vehicle
    so a failure in CI points at exactly which historical match broke."""

    def _assert_strong_match(self, case_index: int) -> None:
        case = VALIDATION_CASES[case_index]
        result = evaluate_case(case)
        # "viable" and "tps" both count as strong; "marginal" counts as
        # acceptable for vehicles where the DB lacks the actual grade
        # (Concorde / Al 2618). None counts as a regression.
        self.assertIsNotNone(
            result.matched_material,
            f"{case.vehicle}: no expected material from {case.expected_materials} "
            f"appeared in viable, tps_coatings (T_marg>0), or marginal. "
            f"Top-5 viable was: {result.top_5_viable}",
        )

    def test_sr71_titanium(self):
        # SR-71: viable hit on a Ti-6Al-4V family member.
        self._assert_strong_match(0)

    def test_x15_inconel(self):
        # X-15: viable hit on the Inconel family.
        self._assert_strong_match(1)

    def test_concorde_2xxx_aluminum(self):
        # Concorde: marginal-list hit on a 2xxx aluminum analogue
        # (Al 2618 itself is not in the DB).
        self._assert_strong_match(2)

    def test_apollo_ablator(self):
        # Apollo CM: TPS-bucket hit with positive thermal margin on
        # AVCOAT / PICA / Carbon phenolic.
        self._assert_strong_match(3)

    def test_shuttle_tps(self):
        # Shuttle Orbiter: TPS-bucket hit with positive margin on RCC,
        # LI-900, AETB-8, or PICA.
        self._assert_strong_match(4)

    def test_cfm56_single_crystal_nickel(self):
        # CFM56-class HPT blade: viable hit on CMSX-4 / PWA 1484 etc.
        self._assert_strong_match(5)


class TestValidationCoverage(unittest.TestCase):
    """The validation suite must continue to cover all 6 reference
    vehicles. If someone adds or removes a case in run_validation.py
    without updating this file, fail fast."""

    def test_six_validation_cases_present(self):
        self.assertEqual(
            len(VALIDATION_CASES), 6,
            "VALIDATION_CASES must contain exactly 6 entries — update "
            "test_validation.py if intentionally changing the count.",
        )


if __name__ == "__main__":
    unittest.main()
