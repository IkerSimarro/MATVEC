"""
MATVEC Pareto Optimization Module
===================================
Multi-objective Pareto front computation for materials trade-off analysis.

Five minimization objectives:
  f1 — Weight penalty:      density / 5000  (normalized)
  f2 — Thermal deficit:     max(0, T_ref - service_temp) / T_ref
  f3 — Structural deficit:  max(0, sigma_req - strength@T_ref) / sigma_req
  f4 — Availability penalty: 1.0 - availability_score
  f5 — Cost penalty:        (cost_usd_per_kg * vehicle_mass_kg) / cost_ceiling

The cost axis lets a human read the Pareto front as a decision-support
artefact rather than a pure physics trade-off: "choosing CMC over
titanium adds 16% weight AND costs 12× more". Cost figures are order-of-
magnitude screening estimates (+/- 50%) — see the §6 "Cost caveat" in
latex_export.py for the user-facing disclosure.

Two partitions are evaluated independently:
  - Direct-exposure candidates (evaluation_mode != "substrate") → T_ref = T_wall.
  - Substrate-mode candidates (evaluation_mode == "substrate") → T_ref = T_soak
    (the temperature seen below an ablator / TBC / active-cooling layer).

Each partition has its own Pareto front, dominated set, trade-off narrative,
and chart. Mixing the two would compare primary-structure-exposed materials
against substrate-under-TPS materials on an apples-to-oranges basis.

The Pareto front is computed via exact dominance (O(n²)) since the
candidate count is always < 200. NSGA-II is not used — see docstring
on compute_pareto for rationale.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

import numpy as np

from .materials_db import get_strength_at_temperature


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParetoResult:
    """Result of a multi-objective Pareto front computation.

    The *direct-exposure* partition occupies the historical field names
    (``pareto_front``, ``dominated``, ``trade_off_descriptions``,
    ``objective_values``, ``pareto_mask``) so that callers written before
    the substrate split continue to work unchanged.

    The *substrate-mode* partition lives on the ``_substrate`` suffixed
    fields and defaults to empty when no substrate candidates are present
    (e.g. monolithic-airframe vehicles below the TPS unlock temperature).

    ``estimated_cost_usd`` / ``estimated_cost_usd_substrate`` are parallel
    lists to ``pareto_front`` / ``pareto_front_substrate`` storing the
    ``cost_usd_per_kg * vehicle_mass_kg`` product for each front member.
    Written by ``compute_pareto`` so the LaTeX exporter and UI can print
    a USD figure without re-multiplying.
    """

    # Direct-exposure partition (primary structure sees T_wall)
    pareto_front: list            # MaterialCandidates on the direct-exposure front
    dominated: list               # direct-exposure dominated set
    trade_off_descriptions: list  # direct-exposure trade-off narratives
    objective_names: list = field(
        default_factory=lambda: [
            "Weight", "Thermal", "Structural", "Availability", "Cost",
        ]
    )
    objective_values: np.ndarray = field(default_factory=lambda: np.empty((0, 5)))
    pareto_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))

    # Substrate-mode partition (primary structure sits beneath an ablator / TBC
    # / active cooling and sees T_soak, not T_wall). Empty when no candidates
    # were evaluated in substrate mode — that is the correct state for vehicles
    # below the TPS unlock regime and the caller should treat "empty" as
    # "substrate mode is not a design axis here".
    pareto_front_substrate: list = field(default_factory=list)
    dominated_substrate: list = field(default_factory=list)
    trade_off_descriptions_substrate: list = field(default_factory=list)
    objective_values_substrate: np.ndarray = field(
        default_factory=lambda: np.empty((0, 5))
    )
    pareto_mask_substrate: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool)
    )

    # Candidate lists per partition — stored so chart generation can be
    # driven from the result alone without re-partitioning upstream.
    candidates_direct: list = field(default_factory=list)
    candidates_substrate: list = field(default_factory=list)

    # Reference temperature used for each partition's objective computation.
    # Needed by the chart so the thermal-margin axis is labelled against the
    # temperature the material actually sees.
    T_direct_K: float = 0.0
    T_substrate_K: float = 0.0

    # Estimated material cost (USD) for each Pareto-front member in the
    # corresponding partition. Same ordering as ``pareto_front`` /
    # ``pareto_front_substrate``. Empty when the front is empty.
    estimated_cost_usd: list = field(default_factory=list)
    estimated_cost_usd_substrate: list = field(default_factory=list)

    # Cost ceiling used to normalise the cost-penalty objective. Echoed
    # back on the result so reporters can display "Budget: $Xk" without
    # re-plumbing the parameter.
    cost_ceiling_usd: float = 1_000_000.0


# ---------------------------------------------------------------------------
# Objective computation
# ---------------------------------------------------------------------------

_DENSITY_REF = 5000.0   # kg/m³ normalization reference
_DEFAULT_COST_CEILING_USD = 1_000_000.0  # $1M screening-grade budget


def _compute_objectives(
    candidate,
    T_ref_K: float,
    sigma_req_MPa: float,
    vehicle_mass_kg: float,
    cost_ceiling_usd: float,
) -> np.ndarray:
    """Return 5-element objective vector (all minimization) for one candidate.

    ``T_ref_K`` is the temperature the material actually sees — ``T_wall`` for
    direct-exposure candidates, ``T_soak`` for substrate-mode candidates.

    f5 (cost penalty) is normalised by the user-supplied budget ceiling so
    that a value of 1.0 means "one full budget of this material" and
    values >> 1 flag hopelessly-expensive candidates. A candidate carrying
    ``cost_usd_per_kg == 0`` (reserved for exotic / 2D impossibility-only
    entries) contributes zero cost penalty — that branch is unreachable
    under the standard matching-engine exclusion, but kept defensive.
    """
    mat = candidate.material

    f1 = mat.density_kgm3 / _DENSITY_REF

    if T_ref_K > 0:
        f2 = max(0.0, T_ref_K - mat.service_temp_air_K) / T_ref_K
    else:
        f2 = 0.0

    strength = get_strength_at_temperature(mat, T_ref_K)
    if sigma_req_MPa > 0:
        f3 = max(0.0, sigma_req_MPa - strength) / sigma_req_MPa
    else:
        f3 = 0.0

    f4 = 1.0 - mat.availability_score

    cost_per_kg = float(getattr(mat, "cost_usd_per_kg", 0.0))
    if cost_ceiling_usd > 0:
        f5 = (cost_per_kg * vehicle_mass_kg) / cost_ceiling_usd
    else:
        f5 = 0.0

    return np.array([f1, f2, f3, f4, f5])


# ---------------------------------------------------------------------------
# Exact Pareto dominance
# ---------------------------------------------------------------------------

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a dominates b (a <= b on all, a < b on at least one).

    Applies component-wise to whatever vector length the caller supplies —
    now 5-element (weight, thermal, structural, availability, cost). The
    cost axis means high-USD-per-kg materials are dominated by cheaper
    equivalents at the same performance margin, matching the intent in
    HANDOFF.md §2 of making the front a decision-support artefact.
    """
    return bool(np.all(a <= b) and np.any(a < b))


def _compute_pareto_mask(obj_values: np.ndarray) -> np.ndarray:
    """Return boolean mask: True for non-dominated (Pareto front) rows."""
    n = obj_values.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            if _dominates(obj_values[j], obj_values[i]):
                mask[i] = False
                break
    return mask


# ---------------------------------------------------------------------------
# Trade-off descriptions
# ---------------------------------------------------------------------------

_OBJ_LABELS = [
    "weight", "thermal capability", "structural capability",
    "availability", "cost",
]
_OBJ_UNITS = [
    "% density", "K thermal margin", "% structural margin",
    "availability", "cost multiplier",
]


def _format_cost_ratio(cost_a: float, cost_b: float) -> str | None:
    """Return a human-readable 'costs Xx more/less' fragment, or None
    when the comparison is not informative.

    Returns ``None`` when either side is zero (exotic/2D unpriced entry
    slipped through), or when the ratio is within [0.9, 1.1] — noise-
    level differences shouldn't clutter the narrative.
    """
    if cost_a <= 0 or cost_b <= 0:
        return None
    ratio = cost_b / cost_a
    if 0.9 <= ratio <= 1.1:
        return None
    if ratio > 1.0:
        return f"costs {ratio:.1f}x more"
    return f"costs {1.0 / ratio:.1f}x less"


def _generate_trade_offs(
    front_candidates: list,
    front_objectives: np.ndarray,
    T_ref_K: float,
    sigma_req_MPa: float,
    vehicle_mass_kg: float = 0.0,
) -> list[str]:
    """Generate plain-language trade-off descriptions between adjacent Pareto points.

    Adjacent means sorted by f1 (weight) ascending. Every narrative now
    carries a cost-delta tail ("...and costs 4.2x more.") when the two
    candidates have materially different cost_usd_per_kg — that's the
    "decision-support artefact" framing from HANDOFF.md's cost-axis brief.

    ``vehicle_mass_kg`` is accepted for forward compatibility (so we can
    later say "$120k more for the same margin") but the narrative itself
    quotes a ratio, which is robust to mass scaling.
    """
    if len(front_candidates) < 2:
        return []

    # Sort by f1 (weight) ascending
    order = np.argsort(front_objectives[:, 0])
    sorted_cands = [front_candidates[i] for i in order]
    sorted_objs = front_objectives[order]

    descriptions: list[str] = []
    for k in range(len(sorted_cands) - 1):
        a_name = sorted_cands[k].material.name
        b_name = sorted_cands[k + 1].material.name

        a_mat = sorted_cands[k].material
        b_mat = sorted_cands[k + 1].material

        # Compute meaningful deltas
        weight_delta_pct = (b_mat.density_kgm3 - a_mat.density_kgm3) / a_mat.density_kgm3 * 100
        thermal_delta_K = b_mat.service_temp_air_K - a_mat.service_temp_air_K

        parts: list[str] = []
        if abs(weight_delta_pct) >= 5.0:
            direction = "adds" if weight_delta_pct > 0 else "saves"
            parts.append(f"{direction} {abs(weight_delta_pct):.1f}% weight")

        if abs(thermal_delta_K) >= 20.0:
            direction = "gains" if thermal_delta_K > 0 else "loses"
            parts.append(f"{direction} {abs(thermal_delta_K):.0f} K of thermal margin")

        str_a = get_strength_at_temperature(a_mat, T_ref_K)
        str_b = get_strength_at_temperature(b_mat, T_ref_K)
        if str_a > 0:
            str_delta_pct = (str_b - str_a) / str_a * 100
            if abs(str_delta_pct) >= 5.0:
                direction = "gains" if str_delta_pct > 0 else "loses"
                parts.append(f"{direction} {abs(str_delta_pct):.0f}% structural strength")

        # Cost delta — always appended when non-trivial so users can read
        # the front as a decision artefact, not just a physics ranking.
        cost_a = float(getattr(a_mat, "cost_usd_per_kg", 0.0))
        cost_b = float(getattr(b_mat, "cost_usd_per_kg", 0.0))
        cost_fragment = _format_cost_ratio(cost_a, cost_b)
        if cost_fragment is not None:
            parts.append(cost_fragment)

        if parts:
            if len(parts) == 1:
                desc = f"Choosing {b_name} over {a_name} {parts[0]}"
            elif len(parts) == 2:
                desc = (
                    f"Choosing {b_name} over {a_name} "
                    f"{parts[0]} but {parts[1]}"
                )
            else:
                # 3+ parts: first two joined with "but", remainder with "and".
                head = f"{parts[0]} but {parts[1]}"
                tail = " and ".join(parts[2:])
                desc = f"Choosing {b_name} over {a_name} {head} and {tail}"
            descriptions.append(desc + ".")
        else:
            descriptions.append(
                f"{b_name} and {a_name} have similar trade-off profiles."
            )

    return descriptions


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def _generate_pareto_chart(
    candidates: list,
    obj_values: np.ndarray,
    pareto_mask: np.ndarray,
    T_ref_K: float,
    title_suffix: str = "",
) -> str:
    """Generate Pareto front scatter plot, return as base64 PNG string.

    Pareto-front members are drawn with numbered markers (1, 2, 3, ...) and
    a side legend resolves numbers to material names. This replaces direct
    ``ax.annotate`` labels which collided at high density (several reentry
    TPS candidates clustered at ρ ≈ 2000 kg/m³ with near-identical thermal
    margins were unreadable with fixed 6-pixel offsets).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A slightly wider figure leaves room for the side legend without
    # squashing the scatter area.
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    # Empty-input guard: return a minimal "no data" placeholder chart
    # rather than crashing. This path is hit for monolithic vehicles on
    # the substrate chart.
    if len(candidates) == 0 or obj_values.shape[0] == 0:
        ax.text(
            0.5, 0.5, "No candidates in this partition",
            ha="center", va="center", color="#8b949e", fontsize=10,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    # X-axis: weight penalty (f1 = density / 5000)
    # Y-axis: thermal margin (service_temp - T_ref, can be negative)
    x_all = obj_values[:, 0]
    y_all = np.array([c.material.service_temp_air_K - T_ref_K for c in candidates])

    # Dominated points
    dom_mask = ~pareto_mask
    if np.any(dom_mask):
        ax.scatter(
            x_all[dom_mask], y_all[dom_mask],
            c="#484f58", s=30, alpha=0.6, zorder=2, label="Dominated",
        )

    # Pareto-front points — plain dots first (so numbers sit on top).
    if np.any(pareto_mask):
        ax.scatter(
            x_all[pareto_mask], y_all[pareto_mask],
            c="#58a6ff", s=110, zorder=3, label="Pareto front",
            edgecolors="#c9d1d9", linewidths=0.9,
        )
        # Numbered markers at each Pareto-front point.  Front members are
        # numbered in ascending f1 (weight) order so the legend reads
        # naturally from lightest to heaviest candidate.
        front_indices = [i for i, m in enumerate(pareto_mask) if m]
        front_x = x_all[pareto_mask]
        order = np.argsort(front_x)
        ordered_indices = [front_indices[k] for k in order]
        legend_entries: list[str] = []
        for display_num, i in enumerate(ordered_indices, start=1):
            ax.text(
                x_all[i], y_all[i], str(display_num),
                ha="center", va="center",
                fontsize=7.0, color="#0d1117", fontweight="bold", zorder=4,
            )
            legend_entries.append(
                f"{display_num} — {candidates[i].material.name}"
            )

    ax.axhline(0, color="#f85149", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Weight Penalty (density / 5000)", color="#c9d1d9", fontsize=9)
    ax.set_ylabel("Thermal Margin (K)", color="#c9d1d9", fontsize=9)
    title = "Pareto Front — Weight vs. Thermal Trade-off"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title, color="#c9d1d9", fontsize=10)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    # Primary legend (dots vs Pareto) in the upper-left of the axes.
    primary_legend = ax.legend(
        facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
        fontsize=7.5, loc="upper left",
    )
    ax.add_artist(primary_legend)

    # Material-number legend outside the plot on the right — collision-free
    # and linearly scannable.
    if np.any(pareto_mask):
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor="#58a6ff", edgecolor="#c9d1d9", label=entry)
            for entry in legend_entries
        ]
        ax.legend(
            handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
            facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
            fontsize=7.0, title="Front members", title_fontsize=7.5,
            handlelength=1.0, borderpad=0.6,
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Partition helpers
# ---------------------------------------------------------------------------

def _compute_partition(
    candidates: list,
    T_ref_K: float,
    sigma_req_MPa: float,
    vehicle_mass_kg: float,
    cost_ceiling_usd: float,
) -> dict:
    """Run Pareto dominance + trade-off generation for a single partition.

    Returns a dict with keys: front, dominated, trade_offs, obj_values,
    mask, front_cost_usd. Handles the empty-input case by returning
    empty structures.
    """
    if not candidates:
        return {
            "front": [],
            "dominated": [],
            "trade_offs": [],
            "obj_values": np.empty((0, 5)),
            "mask": np.empty(0, dtype=bool),
            "front_cost_usd": [],
        }

    obj_rows = [
        _compute_objectives(
            c, T_ref_K, sigma_req_MPa, vehicle_mass_kg, cost_ceiling_usd,
        )
        for c in candidates
    ]
    obj_values = np.vstack(obj_rows)
    pareto_mask = _compute_pareto_mask(obj_values)

    front = [c for c, m in zip(candidates, pareto_mask) if m]
    dominated = [c for c, m in zip(candidates, pareto_mask) if not m]
    front_objs = obj_values[pareto_mask]
    trade_offs = _generate_trade_offs(
        front, front_objs, T_ref_K, sigma_req_MPa, vehicle_mass_kg,
    )

    # Estimated USD cost per front member: cost_per_kg * vehicle_mass_kg.
    # Sorted implicitly in the same order as ``front``, which is the
    # candidates' original order filtered by pareto_mask.
    front_cost_usd = [
        float(getattr(c.material, "cost_usd_per_kg", 0.0)) * float(vehicle_mass_kg)
        for c in front
    ]

    return {
        "front": front,
        "dominated": dominated,
        "trade_offs": trade_offs,
        "obj_values": obj_values,
        "mask": pareto_mask,
        "front_cost_usd": front_cost_usd,
    }


def _substrate_reference_conditions(physics_result) -> tuple[float, float]:
    """Return (T_soak_K, sigma_req_tps_MPa) for the substrate partition.

    Mirrors the matching engine's substrate evaluation logic so Pareto
    objectives are computed against the same operating point at which the
    candidates were scored. Using T_wall/sigma_req here would re-rank
    substrate candidates as if they were facing direct exposure, defeating
    the purpose of partitioning.
    """
    # Local import avoids a circular dependency at module import time.
    from .matching_engine import _ABLATIVE_SUBSTRATE_T_FLOOR_K

    T_ambient = physics_result.thermal.T_ambient_K
    T_soak = max(T_ambient, _ABLATIVE_SUBSTRATE_T_FLOOR_K)

    # Thermal-stress component at T_soak, using the same k_relief=0.4,
    # E_ref=200 GPa, alpha_ref=12e-6/K constants that the structural
    # block uses. Mechanical (sigma_combined) is unchanged — g-load and
    # pressure loading still apply to the substructure.
    delta_T_tps = max(0.0, T_soak - T_ambient)
    sigma_th_tps = 0.4 * 200_000.0 * 12e-6 * delta_T_tps
    sigma_req_tps = physics_result.structural.sigma_combined_MPa + sigma_th_tps
    return T_soak, sigma_req_tps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pareto(
    candidates: list,
    physics_result,
    vehicle_category: str,
    *,
    cost_ceiling_usd: float = _DEFAULT_COST_CEILING_USD,
) -> ParetoResult:
    """Compute Pareto fronts across five objectives for the given candidates.

    Candidates are partitioned by ``evaluation_mode`` before dominance
    analysis. Direct-exposure candidates (``evaluation_mode != "substrate"``)
    are evaluated against ``T_wall`` and the original ``sigma_req``.
    Substrate-mode candidates are evaluated against ``T_soak`` and the
    TPS-protected ``sigma_req_tps`` (mechanical + thermal stress from the
    ambient→soak temperature rise). Each partition gets its own Pareto
    front, dominated set, trade-off narrative, and chart inputs.

    Uses exact dominance (O(n²)) rather than NSGA-II because the per-partition
    candidate count is always well under 200. Exact dominance is deterministic,
    faster, and produces the true Pareto front rather than an approximation.
    Switch to NSGA-II only if candidate counts exceed 200.

    Parameters
    ----------
    candidates : list of MaterialCandidate
        Combined viable + marginal candidates from the matching engine.
        May contain both direct-exposure and substrate-mode entries.
    physics_result : PhysicsResult
        Physics context for the evaluation. ``vehicle_mass_kg`` is read
        from here to compute the cost-penalty axis.
    vehicle_category : str
        Vehicle category string (unused in this function but accepted for
        API symmetry with the matching engine; may drive partition policy
        in a future refactor).
    cost_ceiling_usd : float, keyword-only
        Budget ceiling that normalises the cost-penalty objective to O(1).
        Defaults to $1M — representative of a university / startup concept
        programme. Streamlit exposes this as a sidebar input so users can
        dial it to their actual budget and re-read the front as a go / no-go
        economic gate rather than a pure physics trade-off.

    Returns
    -------
    ParetoResult
    """
    if not candidates:
        return ParetoResult(
            pareto_front=[],
            dominated=[],
            trade_off_descriptions=[],
            objective_values=np.empty((0, 5)),
            pareto_mask=np.empty(0, dtype=bool),
            cost_ceiling_usd=float(cost_ceiling_usd),
        )

    # Partition by evaluation_mode. The getattr fallback keeps us safe
    # if an older MaterialCandidate (no evaluation_mode field) ever slips
    # through — default to "direct" so the backward-compatible fields stay
    # populated.
    direct_cands = [
        c for c in candidates
        if getattr(c, "evaluation_mode", "direct") != "substrate"
    ]
    substrate_cands = [
        c for c in candidates
        if getattr(c, "evaluation_mode", "direct") == "substrate"
    ]

    vehicle_mass_kg = float(physics_result.vehicle_mass_kg)

    # Direct partition: operates against T_wall / sigma_req as before.
    T_wall_K = physics_result.thermal.T_wall_K
    sigma_req = physics_result.structural.sigma_tensile_required_MPa
    direct_results = _compute_partition(
        direct_cands, T_wall_K, sigma_req,
        vehicle_mass_kg, cost_ceiling_usd,
    )

    # Substrate partition: operates against T_soak / sigma_req_tps so that
    # "thermal margin" and "structural margin" on the chart reflect the
    # temperature the substrate actually experiences beneath the TPS.
    T_soak_K, sigma_req_tps = _substrate_reference_conditions(physics_result)
    substrate_results = _compute_partition(
        substrate_cands, T_soak_K, sigma_req_tps,
        vehicle_mass_kg, cost_ceiling_usd,
    )

    return ParetoResult(
        # Direct (backward-compat fields)
        pareto_front=direct_results["front"],
        dominated=direct_results["dominated"],
        trade_off_descriptions=direct_results["trade_offs"],
        objective_values=direct_results["obj_values"],
        pareto_mask=direct_results["mask"],
        # Substrate
        pareto_front_substrate=substrate_results["front"],
        dominated_substrate=substrate_results["dominated"],
        trade_off_descriptions_substrate=substrate_results["trade_offs"],
        objective_values_substrate=substrate_results["obj_values"],
        pareto_mask_substrate=substrate_results["mask"],
        # Partition inputs + reference temperatures (for chart regeneration)
        candidates_direct=direct_cands,
        candidates_substrate=substrate_cands,
        T_direct_K=T_wall_K,
        T_substrate_K=T_soak_K,
        # Cost outputs
        estimated_cost_usd=direct_results["front_cost_usd"],
        estimated_cost_usd_substrate=substrate_results["front_cost_usd"],
        cost_ceiling_usd=float(cost_ceiling_usd),
    )


def generate_pareto_chart_b64(
    pareto_result: ParetoResult,
    candidates: list | None = None,
    T_wall_K: float | None = None,
    *,
    is_substrate: bool = False,
) -> str:
    """Generate the Pareto front chart for a partition, return base64 PNG.

    Parameters
    ----------
    pareto_result : ParetoResult
        The partitioned result returned by ``compute_pareto``.
    candidates : list | None
        Deprecated — retained for backward compatibility with the original
        ``(result, candidates, T_wall_K)`` signature. If provided together
        with ``T_wall_K`` and ``is_substrate=False``, these override the
        partition data stored on ``pareto_result`` (this is the exact path
        the pre-partition LaTeX code took).
    T_wall_K : float | None
        See ``candidates``.
    is_substrate : bool, keyword-only
        If ``True``, chart the substrate-mode partition. Defaults to ``False``
        (direct-exposure partition, historical behavior).
    """
    if is_substrate:
        return _generate_pareto_chart(
            pareto_result.candidates_substrate,
            pareto_result.objective_values_substrate,
            pareto_result.pareto_mask_substrate,
            pareto_result.T_substrate_K,
            title_suffix="substrate under TPS",
        )

    # Direct-exposure path.  If the caller passed legacy args, use those
    # (preserves the old signature contract exactly); otherwise use the
    # partition data on the result object.
    if candidates is not None and T_wall_K is not None:
        return _generate_pareto_chart(
            candidates,
            pareto_result.objective_values,
            pareto_result.pareto_mask,
            T_wall_K,
        )
    return _generate_pareto_chart(
        pareto_result.candidates_direct,
        pareto_result.objective_values,
        pareto_result.pareto_mask,
        pareto_result.T_direct_K,
        title_suffix="direct exposure",
    )
