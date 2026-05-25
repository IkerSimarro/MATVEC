"""
MATVEC — Programmatic pipeline entry point.

``run_session()`` is the single function both the CLI and the Streamlit
UI call. It wires physics → match → Pareto → surrogate → LaTeX → PDF in
one place, so UI changes never drift out of sync with CLI output.

Hard constraint: this module does NOT import ``streamlit``. That rule
is what keeps the headless CLI runnable in environments without a
Streamlit install (Docker, CI, academic venvs). If you find yourself
reaching for ``st.warning`` or ``st.spinner`` here — push that concern
back into ``app.py``.
"""

from dataclasses import dataclass

from .matching_engine import match_materials, MatchResult
from .physics_engine import run_analysis, PhysicsResult
from .latex_export import generate_report, generate_tex_source
from .pareto import compute_pareto, ParetoResult
from .surrogate import find_nearest_candidates, SurrogateResult
from .session import SessionSchema


# ---------------------------------------------------------------------------
# Result bundle
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    """Complete pipeline output for one SessionSchema.

    Every field is always populated, though ``pareto`` may be a
    degenerate (empty) ParetoResult for runs with <3 candidates, and
    ``pdf_bytes`` is None when pdflatex is unavailable OR when
    ``compile_pdf=False`` was passed to ``run_session``.
    """
    physics: PhysicsResult
    match: MatchResult
    pareto: ParetoResult
    surrogate: SurrogateResult
    tex_source: str
    pdf_bytes: bytes | None


# ---------------------------------------------------------------------------
# Turbine hot-section override
# ---------------------------------------------------------------------------

def apply_turbine_override(
    physics: PhysicsResult, hot_section_temp_K: float
) -> PhysicsResult:
    """Overwrite aerodynamic T_wall with turbine hot-section temperature.

    At Mach 0.5 / sea level the aerodynamic recovery temperature is
    ~300 K. That value is correct for a freestream airflow but
    meaningless for a turbine blade: the blade's wall temperature is
    set by turbine inlet temperature and film-cooling physics, not by
    external aero-heating. This helper replaces ``T_wall`` with the
    caller-supplied hot-section value (typically ~1400 K for a modern
    cooled HPT blade metal face; TIT ~1700 K minus ~300 K film-cooling
    delta), widens the uncertainty band to ±5%, and recomputes
    structural σ_req against the new ΔT so downstream material
    screening sees the right thermal-stress load.

    A ``thermal_source`` flag is written to ``ThermalResults`` so the
    LaTeX exporter knows to swap the Sutton-Graves / recovery
    derivation block for a short override-framing paragraph in §4.

    Moved here from ``app.py`` so the CLI pipeline can apply the same
    turbine branch without importing Streamlit. The original
    in-place mutation of ``physics.thermal`` is preserved — historical
    callers (tests, app.py's main loop) rely on it — and the returned
    PhysicsResult is the same object passed in.
    """
    from .physics_engine import _compute_structural

    T_target = float(hot_section_temp_K)
    physics.thermal.T_wall_K          = T_target
    physics.thermal.T_wall_min_K      = T_target * 0.95
    physics.thermal.T_wall_max_K      = T_target * 1.05
    physics.thermal.T_wall_sealevel_K = T_target
    physics.thermal.thermal_source    = "turbine_inlet_override"

    physics.structural = _compute_structural(
        physics.vehicle_mass_kg,
        physics.peak_g_load,
        physics.atmosphere.density_kgm3,
        physics.thermal.velocity_ms,
        T_target,
        physics.thermal.T_ambient_K,
        physics.structural.characteristic_length_m,
    )
    return physics


# ---------------------------------------------------------------------------
# Pareto helper
# ---------------------------------------------------------------------------

def _empty_pareto() -> ParetoResult:
    """Construct a degenerate ParetoResult for runs with <3 candidates.

    ParetoResult's ``default_factory`` fields take care of the numpy
    arrays; we only need to supply the required list positionals.
    This mirrors the "skip Pareto quietly" branch in the historical
    ``_show_pareto`` in app.py, but as a real dataclass so downstream
    code (LaTeX exporter, tests) can read the same shape unconditionally.
    """
    return ParetoResult(
        pareto_front=[],
        dominated=[],
        trade_off_descriptions=[],
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_session(
    session: SessionSchema,
    *,
    compile_pdf: bool = True,
    sensitivity=None,
) -> SessionResult:
    """Execute the full MATVEC pipeline for a single SessionSchema.

    Parameters
    ----------
    session : SessionSchema
        Flight envelope + vehicle category + label + options.
    compile_pdf : bool
        If False, skip the pdflatex subprocess and leave
        ``pdf_bytes`` as None. Useful for fast smoke tests and for
        Streamlit flows that render the PDF lazily on button-click.
    sensitivity : SensitivityResult | None
        Optional pre-computed sensitivity bundle. When provided, the
        LaTeX exporter inserts a ``\\section{Sensitivity Analysis}``
        between Materials (§8) and Per-Zone (§9). Computed externally
        because the sweep is heavy (~44 pipeline runs by default) and
        opt-in — callers (CLI ``--sensitivity`` flag, Streamlit
        sidebar checkbox) decide when to pay that cost. Duck-typed
        here to avoid an import cycle with ``core.sensitivity``,
        which already imports ``apply_turbine_override`` from this
        module.

    Returns
    -------
    SessionResult
        All pipeline outputs in one record.
    """
    # --- 1. physics ---
    physics = run_analysis(
        session.mach,
        session.alt_km,
        session.mass_kg,
        session.R_n_m,
        peak_g_load=session.g_load,
        wall_emissivity=session.wall_emissivity,
        characteristic_length_m=session.char_len_m,
        flight_duration_s=session.flight_duration_s,
    )

    # --- 2. turbine override (only for the turbine category) ---
    # Opt-in, not implicit — a user with category="turbine" but no
    # hot_section_temp_K in options gets the default aerodynamic T_wall,
    # which is what the physics engine would report anyway.
    if session.vehicle_category == "turbine":
        hot_K = session.options.get("hot_section_temp_K")
        if hot_K is not None:
            physics = apply_turbine_override(physics, float(hot_K))

    # --- 3. matching ---
    match = match_materials(
        physics, vehicle_category=session.vehicle_category,
        design_lifetime_hours=float(session.design_lifetime_hours),
        panel_thickness_m=float(session.panel_thickness_m),
        flight_profile=tuple(session.flight_profile),
    )

    # --- 4. Pareto (skip gracefully for tiny candidate sets) ---
    # Cost ceiling is carried on the SessionSchema options dict (rather
    # than a first-class field) because it is a reporting parameter, not
    # a physics input — a user can re-run the same envelope with a
    # different ceiling to ask "does this fit a $500k budget instead of
    # $1M?" without touching the physics layer.
    cost_ceiling_usd = float(
        session.options.get("cost_ceiling_usd", 1_000_000.0)
    )
    candidates = list(match.viable) + list(match.marginal)
    if len(candidates) >= 3:
        pareto = compute_pareto(
            candidates, physics, session.vehicle_category,
            cost_ceiling_usd=cost_ceiling_usd,
        )
    else:
        pareto = _empty_pareto()
        # Keep cost_ceiling echo on the empty result for downstream display.
        pareto.cost_ceiling_usd = cost_ceiling_usd

    # --- 5. surrogate ---
    surrogate = find_nearest_candidates(
        physics,
        session.vehicle_category,
        match_result=match,
        k=10,
    )

    # --- 6. LaTeX source (always — it's fast and useful on pdflatex failure) ---
    # cost_ceiling_usd flows from session.options into both the materials
    # table (bolds rows whose row-cost exceeds the ceiling) and the Pareto
    # objective (already plumbed in step 4). Same ceiling for both keeps
    # the PDF self-consistent: a row bolded in §8 is the same row penalised
    # on the Pareto front in §10.
    tex_source, _aux = generate_tex_source(
        physics, match, session.system_label,
        cost_ceiling_usd=cost_ceiling_usd,
        sensitivity=sensitivity,
        design_lifetime_hours=float(session.design_lifetime_hours),
        panel_thickness_m=float(session.panel_thickness_m),
    )

    # --- 7. PDF (optional) ---
    if compile_pdf:
        pdf_bytes = generate_report(
            physics, match, session.system_label,
            cost_ceiling_usd=cost_ceiling_usd,
            sensitivity=sensitivity,
            design_lifetime_hours=float(session.design_lifetime_hours),
            panel_thickness_m=float(session.panel_thickness_m),
        )
    else:
        pdf_bytes = None

    return SessionResult(
        physics=physics,
        match=match,
        pareto=pareto,
        surrogate=surrogate,
        tex_source=tex_source,
        pdf_bytes=pdf_bytes,
    )
