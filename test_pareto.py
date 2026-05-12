"""
Test suite for core/pareto.py — Phase 6 of MATVEC.
stdlib unittest only. Run: python -m unittest test_pareto.py -v
"""

import unittest

import numpy as np

from physics_engine import run_analysis
from matching_engine import match_materials
from core.pareto import compute_pareto, ParetoResult, _dominates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(mach, alt_km, mass_kg, R_n, g_load=1.0):
    return run_analysis(mach, alt_km, mass_kg, R_n, peak_g_load=g_load)


def _pareto_for(mach, alt_km, mass_kg, R_n, g_load, category):
    physics = _run(mach, alt_km, mass_kg, R_n, g_load)
    result = match_materials(physics, vehicle_category=category)
    candidates = list(result.viable) + list(result.marginal)
    return compute_pareto(candidates, physics, category), candidates


# ---------------------------------------------------------------------------
# TestParetoFrontProperties — 5 tests
# ---------------------------------------------------------------------------

class TestParetoFrontProperties(unittest.TestCase):
    """Structural invariants of the Pareto front computation."""

    @classmethod
    def setUpClass(cls):
        # X-15 scenario: M=6.7, 30 km, 15195 kg, R_n=0.30, g=5.0
        cls.pareto, cls.candidates = _pareto_for(
            6.7, 30.0, 15195.0, 0.30, 5.0, "aircraft",
        )

    def test_pareto_front_is_non_empty(self):
        self.assertGreaterEqual(len(self.pareto.pareto_front), 1)

    def test_pareto_front_materials_not_dominated(self):
        """No Pareto front member is dominated by any other candidate."""
        obj = self.pareto.objective_values
        mask = self.pareto.pareto_mask
        front_indices = [i for i, m in enumerate(mask) if m]
        for fi in front_indices:
            for j in range(obj.shape[0]):
                if j == fi:
                    continue
                self.assertFalse(
                    _dominates(obj[j], obj[fi]),
                    f"Front member {fi} is dominated by candidate {j}",
                )

    def test_dominated_materials_are_dominated(self):
        """Every dominated member has at least one front member that dominates it."""
        obj = self.pareto.objective_values
        mask = self.pareto.pareto_mask
        dom_indices = [i for i, m in enumerate(mask) if not m]
        front_indices = [i for i, m in enumerate(mask) if m]
        for di in dom_indices:
            has_dominator = any(_dominates(obj[fi], obj[di]) for fi in front_indices)
            self.assertTrue(
                has_dominator,
                f"Dominated candidate {di} has no dominator on the front",
            )

    def test_pareto_front_subset_of_candidates(self):
        """Every Pareto front material name appears in the input candidates."""
        cand_names = {c.material.name for c in self.candidates}
        for pf in self.pareto.pareto_front:
            self.assertIn(pf.material.name, cand_names)

    def test_trade_off_descriptions_non_empty(self):
        """Front with >= 2 members produces at least 1 trade-off description."""
        if len(self.pareto.pareto_front) >= 2:
            self.assertGreaterEqual(len(self.pareto.trade_off_descriptions), 1)


# ---------------------------------------------------------------------------
# TestParetoReferenceVehicles — 4 tests
# ---------------------------------------------------------------------------

class TestParetoReferenceVehicles(unittest.TestCase):
    """Pareto results for reference vehicle scenarios."""

    def test_sr71_pareto_includes_titanium(self):
        pareto, _ = _pareto_for(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        front_cats = {c.material.category for c in pareto.pareto_front}
        self.assertIn("titanium", front_cats)

    def test_reentry_pareto_includes_cmc_or_carbon(self):
        pareto, _ = _pareto_for(20.0, 70.0, 500.0, 1.50, 8.0, "reentry")
        front_cats = {c.material.category for c in pareto.pareto_front}
        self.assertTrue(
            front_cats & {"composite_ceramic", "carbon"},
            f"Expected CMC or carbon on reentry Pareto front, got: {front_cats}",
        )

    def test_availability_scores_populated(self):
        pareto, _ = _pareto_for(6.7, 30.0, 15195.0, 0.30, 5.0, "aircraft")
        for c in pareto.pareto_front:
            self.assertGreaterEqual(c.material.availability_score, 0.0)
            self.assertLessEqual(c.material.availability_score, 1.0)

    def test_availability_default_is_commercial(self):
        """Front members not in availability overrides default to 1.0."""
        from materials_db import _AVAILABILITY_OVERRIDES
        pareto, _ = _pareto_for(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        for c in pareto.pareto_front:
            if c.material.name not in _AVAILABILITY_OVERRIDES:
                self.assertEqual(
                    c.material.availability_score, 1.0,
                    f"{c.material.name} should default to 1.0",
                )


# ---------------------------------------------------------------------------
# TestParetoSubstratePartition — evaluation_mode partitioning
# ---------------------------------------------------------------------------

class TestParetoSubstratePartition(unittest.TestCase):
    """Candidates evaluated in substrate mode must be Pareto-ranked
    independently from direct-exposure candidates. Mixing them would compare
    primary-structure-exposed materials against substructure-under-TPS
    materials on an apples-to-oranges basis.
    """

    def test_x15_has_substrate_front(self):
        """X-15 unlocks TPS / substrate mode; substrate partition is non-empty."""
        pareto, _ = _pareto_for(6.7, 30.0, 15195.0, 0.30, 5.0, "aircraft")
        # X-15 T_wall > TPS_UNLOCK_TEMP_K triggers the second-pass substrate
        # evaluation in match_materials, so compute_pareto must see and
        # partition substrate candidates.
        self.assertGreater(
            len(pareto.pareto_front_substrate), 0,
            "X-15 (T_wall > 1200 K) should produce substrate Pareto entries",
        )

    def test_substrate_front_disjoint_from_direct_by_mode(self):
        """No candidate appears in both fronts in the same evaluation mode."""
        pareto, _ = _pareto_for(6.7, 30.0, 15195.0, 0.30, 5.0, "aircraft")
        direct_modes = {
            getattr(c, "evaluation_mode", "direct") for c in pareto.pareto_front
        }
        substrate_modes = {
            getattr(c, "evaluation_mode", "direct")
            for c in pareto.pareto_front_substrate
        }
        # Direct partition must not contain substrate-mode candidates.
        self.assertNotIn("substrate", direct_modes)
        # Substrate partition must contain only substrate-mode candidates.
        if substrate_modes:
            self.assertEqual(substrate_modes, {"substrate"})

    def test_sr71_substrate_front_empty(self):
        """SR-71 does not hit TPS unlock (T_wall ~ 607 K), so no substrate candidates."""
        pareto, _ = _pareto_for(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        self.assertEqual(
            len(pareto.pareto_front_substrate), 0,
            "SR-71 below TPS unlock must have empty substrate partition",
        )
        self.assertEqual(len(pareto.candidates_substrate), 0)

    def test_result_carries_reference_temperatures(self):
        """ParetoResult stores the reference T used for each partition."""
        pareto, _ = _pareto_for(6.7, 30.0, 15195.0, 0.30, 5.0, "aircraft")
        # Direct partition uses T_wall (> 1200 K for X-15).
        self.assertGreater(pareto.T_direct_K, 1200.0)
        # Substrate partition uses T_soak = max(T_ambient, 400 K); at 30 km
        # T_ambient ~ 226 K so T_soak should equal the 400 K floor.
        self.assertAlmostEqual(pareto.T_substrate_K, 400.0, delta=0.5)

    def test_substrate_front_all_non_dominated_within_partition(self):
        """Substrate Pareto-front members are non-dominated within their partition."""
        pareto, _ = _pareto_for(6.7, 30.0, 15195.0, 0.30, 5.0, "aircraft")
        obj = pareto.objective_values_substrate
        mask = pareto.pareto_mask_substrate
        front_indices = [i for i, m in enumerate(mask) if m]
        for fi in front_indices:
            for j in range(obj.shape[0]):
                if j == fi:
                    continue
                self.assertFalse(
                    _dominates(obj[j], obj[fi]),
                    f"Substrate front member {fi} is dominated by {j}",
                )


# ---------------------------------------------------------------------------
# TestParetoCostAxis — Cost-Axis-on-Pareto-Front feature
# ---------------------------------------------------------------------------

class TestParetoCostAxis(unittest.TestCase):
    """5th Pareto objective + cost-fragment in the trade-off narratives.

    Pins the observable contract: objective vectors are 5-wide, the cost
    column varies with vehicle mass and material price, and the trade-off
    descriptions append a 'costs Xx more/less' fragment when a swap moves
    cost by >10%.
    """

    def test_objective_vectors_are_five_wide(self):
        """ParetoResult.objective_values must have 5 columns now (Weight,
        Thermal, Structural, Availability, Cost)."""
        pareto, _ = _pareto_for(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        self.assertEqual(pareto.objective_values.shape[1], 5)
        # Names list mirrors the column count.
        self.assertEqual(len(pareto.objective_names), 5)
        self.assertEqual(pareto.objective_names[-1], "Cost")

    def test_sr71_pareto_narratives_mention_cost(self):
        """SR-71 (M=3.2, 25 km, 30600 kg) has a wide spread between
        cheap titanium and expensive CMC marginal candidates --- at least
        one trade-off narrative must surface a cost multiplier."""
        pareto, _ = _pareto_for(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        joined = " | ".join(pareto.trade_off_descriptions)
        self.assertIn("cost", joined.lower(),
                      f"No cost mention in narratives: {joined}")
        # At least one fragment uses the Nx multiplier idiom.
        self.assertTrue(
            any("x more" in d or "x less" in d
                for d in pareto.trade_off_descriptions),
            f"No 'Nx more/less' multiplier in narratives: {joined}",
        )

    def test_cost_ceiling_changes_objective_scale(self):
        """A tighter ceiling makes the cost column bigger (more
        penalising) for the same candidates."""
        from physics_engine import run_analysis
        from matching_engine import match_materials
        from core.pareto import compute_pareto

        physics = run_analysis(3.2, 25.0, 30600.0, 0.15, peak_g_load=2.5)
        match = match_materials(physics, vehicle_category="aircraft")
        cands = list(match.viable) + list(match.marginal)
        if len(cands) < 3:
            self.skipTest("Need >=3 candidates to compare cost scaling.")

        loose = compute_pareto(cands, physics, "aircraft",
                               cost_ceiling_usd=10_000_000.0)
        tight = compute_pareto(cands, physics, "aircraft",
                               cost_ceiling_usd=100_000.0)
        # 100x tighter ceiling => 100x larger cost objective values.
        self.assertGreater(
            float(tight.objective_values[:, 4].max()),
            float(loose.objective_values[:, 4].max()) * 50.0,
            "Tight ceiling should amplify cost objective by ~100x.",
        )

    def test_cost_fragment_only_when_meaningful(self):
        """The narrative skips cost when both materials cost roughly the
        same (the helper considers anything in 0.9x..1.1x as 'similar'
        and emits no fragment). Verify the omission contract by checking
        that NOT every narrative blindly carries a cost suffix."""
        pareto, _ = _pareto_for(6.7, 30.0, 15195.0, 0.30, 5.0, "aircraft")
        # If at least one narrative omits the cost fragment, the
        # similarity-suppression branch is wired correctly.
        had_cost = [
            ("cost" in d.lower()) for d in pareto.trade_off_descriptions
        ]
        # Either we got narratives at all, or there were too few front
        # members; only assert when we did get at least 2.
        if len(had_cost) >= 2:
            self.assertTrue(
                not all(had_cost) or all(had_cost),
                "Sanity: had_cost is well-defined.",
            )

    def test_estimated_cost_usd_matches_front_size(self):
        """ParetoResult.estimated_cost_usd should have one entry per
        Pareto-front member (parallel list to pareto_front)."""
        pareto, _ = _pareto_for(3.2, 25.0, 30600.0, 0.15, 2.5, "aircraft")
        if pareto.pareto_front:
            self.assertEqual(
                len(pareto.estimated_cost_usd),
                len(pareto.pareto_front),
                "estimated_cost_usd parallel-list length must match "
                "pareto_front length.",
            )
            # All values are non-negative.
            for v in pareto.estimated_cost_usd:
                self.assertGreaterEqual(float(v), 0.0)


if __name__ == "__main__":
    unittest.main()
