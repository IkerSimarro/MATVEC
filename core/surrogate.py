"""
MATVEC Materials Surrogate Model
==================================
k-NN in 7D normalized property space for materials similarity ranking.

Feature dimensions (all float, SI units):
  1. density_kgm3
  2. tensile_strength_mpa
  3. service_temp_air_K
  4. melting_point_K
  5. thermal_conductivity_WmK
  6. youngs_modulus_GPa
  7. thermal_expansion_1K

Normalization is manual (mean/std), avoiding sklearn dependency issues.
Built at import time on full MATERIALS_DB (97 entries).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .materials_db import MATERIALS_DB, MaterialEntry, get_materials_by_regime
from .matching_engine import MAX_DENSITY_KGM3, _get_category_exclusions


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_FEATURE_KEYS = [
    "density_kgm3",
    "tensile_strength_mpa",
    "service_temp_air_K",
    "melting_point_K",
    "thermal_conductivity_WmK",
    "youngs_modulus_GPa",
    "thermal_expansion_1K",
]


def _extract_features(mat: MaterialEntry) -> list[float]:
    return [getattr(mat, k) for k in _FEATURE_KEYS]


# ---------------------------------------------------------------------------
# Manual StandardScaler (mean/std normalization)
# ---------------------------------------------------------------------------

class _StandardScaler:
    """Minimal StandardScaler clone using pure numpy."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "_StandardScaler":
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=0)
        # Prevent division by zero for constant features
        self.std_[self.std_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ---------------------------------------------------------------------------
# Build surrogate at import time
# ---------------------------------------------------------------------------

def build_surrogate(materials: list[MaterialEntry]):
    """Build and return (scaler, feature_matrix, material_list, model_version).

    Parameters
    ----------
    materials : list[MaterialEntry]
        Full materials database.

    Returns
    -------
    tuple[_StandardScaler, np.ndarray, list[MaterialEntry], str]
        scaler, scaled feature matrix, material list, SHA-256 version hash.
    """
    features = np.array([_extract_features(m) for m in materials], dtype=np.float64)
    scaler = _StandardScaler()
    scaled = scaler.fit_transform(features)

    # SHA-256 of serialized DB properties for reproducibility
    db_repr = json.dumps(
        [[m.name] + _extract_features(m) for m in materials],
        sort_keys=True,
    )
    version = hashlib.sha256(db_repr.encode()).hexdigest()

    return scaler, scaled, list(materials), version


_SCALER, _FEATURE_MATRIX, _MATERIALS_LIST, _MODEL_VERSION = build_surrogate(MATERIALS_DB)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SurrogateResult:
    """Result of k-NN surrogate similarity search."""

    candidates: list[MaterialEntry]       # nearest materials, sorted by distance
    distances: list[float]                # corresponding Euclidean distances
    agreement_with_margin_ranking: float  # overlap fraction with matching engine top-k
    model_version: str                    # 64-char hex SHA-256
    suppressed_count: int = 0             # number of otherwise-eligible materials that were
                                          # removed by the vehicle-category exclusion filter
                                          # (e.g. refractory metals for an aircraft preset)
    fallback_used: bool = False           # True if the category filter emptied the eligible
                                          # pool and we fell back to regime-only filtering


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_nearest_candidates(
    physics_result,
    vehicle_category: str,
    match_result=None,
    k: int = 10,
) -> SurrogateResult:
    """Find the k nearest materials in normalized property space.

    Parameters
    ----------
    physics_result : PhysicsResult
        Physics context for requirements vector construction.
    vehicle_category : str
        Vehicle category — used for density reference.
    match_result : MatchResult | None
        If provided, compute agreement with the matching engine's viable list.
    k : int
        Number of nearest neighbors to return.

    Returns
    -------
    SurrogateResult
    """
    # Build requirements vector
    density_ref = MAX_DENSITY_KGM3.get(vehicle_category, 5000.0)
    sigma_req = physics_result.structural.sigma_tensile_required_MPa
    T_wall = physics_result.thermal.T_wall_K

    requirements = np.array([[
        density_ref,          # density target
        sigma_req,            # tensile strength target
        T_wall,               # service temperature target
        3000.0,               # melting point — fixed high reference
        10.0,                 # thermal conductivity — moderate default
        200.0,                # Young's modulus — moderate default
        10e-6,                # CTE — moderate default
    ]], dtype=np.float64)

    # Scale with fitted scaler
    req_scaled = _SCALER.transform(requirements)

    # Euclidean distances to all materials
    dists = np.linalg.norm(_FEATURE_MATRIX - req_scaled, axis=1)

    # Filter to regime-eligible materials
    regime = physics_result.flight_regime
    regime_eligible = set(m.name for m in get_materials_by_regime(regime))

    # Build (index, distance) pairs for regime-eligible materials
    regime_pairs = [
        (i, dists[i])
        for i in range(len(_MATERIALS_LIST))
        if _MATERIALS_LIST[i].name in regime_eligible
    ]

    # Apply vehicle-category exclusion filter on top of regime filter.
    # Reusing _get_category_exclusions from the matching engine keeps the
    # surrogate and the physics-ranked path aligned on a single set of
    # principled exclusions (refractory metals for aircraft, submarine/
    # general-engineering steels for non-general vehicles, etc.). This
    # also picks up Mach-dependent rules (e.g. aircraft polymer-composite
    # exclusion at M >= 2.0) that the raw dict would miss.
    excluded_cats = _get_category_exclusions(vehicle_category, physics_result)
    category_pairs = [
        (i, d) for i, d in regime_pairs
        if _MATERIALS_LIST[i].category not in excluded_cats
    ]

    suppressed_count = len(regime_pairs) - len(category_pairs)
    fallback_used = False
    # Graceful empty-pool fallback: if the category filter removes every
    # regime-eligible material (can happen for narrow categories with
    # unusual Mach/altitude combinations), fall back to regime-only and
    # flag it so the caller can show the user what happened.
    if category_pairs:
        eligible_pairs = category_pairs
    else:
        eligible_pairs = regime_pairs
        fallback_used = True
        suppressed_count = 0

    # Sort by distance ascending, take top-k
    eligible_pairs.sort(key=lambda x: x[1])
    top_k = eligible_pairs[:k]

    result_materials = [_MATERIALS_LIST[i] for i, _ in top_k]
    result_distances = [float(d) for _, d in top_k]

    # Compute agreement with matching engine's viable ranking
    # (post-filter — agreement will naturally climb because both lists
    # now share the same category exclusions; report the real number).
    agreement = 0.0
    if match_result is not None:
        viable_names = set()
        for c in list(match_result.viable) + list(match_result.marginal):
            viable_names.add(c.material.name)
        surr_names = set(m.name for m in result_materials)
        overlap = viable_names & surr_names
        denominator = max(len(viable_names), len(surr_names), 1)
        agreement = len(overlap) / denominator

    return SurrogateResult(
        candidates=result_materials,
        distances=result_distances,
        agreement_with_margin_ranking=agreement,
        model_version=_MODEL_VERSION,
        suppressed_count=suppressed_count,
        fallback_used=fallback_used,
    )


def get_model_version() -> str:
    """Return the SHA-256 hash identifying the current surrogate model state."""
    return _MODEL_VERSION
