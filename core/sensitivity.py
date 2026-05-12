"""
MATVEC — Uncertainty & Sensitivity Analysis.

Wraps the existing physics + matching pipeline to answer one question
the calibration table cannot: *"how robust is each viable material to
realistic uncertainty in the flight envelope inputs?"*

The motivation is the screening-before-$4M-test workflow. A material
that is viable at the nominal envelope but drops out the moment Mach
nudges +5% is a knife-edge pick — the engineer needs to know that
before signing off on a test campaign.

Algorithm
---------
1. Run the nominal pipeline (``run_analysis`` + ``match_materials``)
   to capture the baseline viable list.
2. For each input in {mach, mass, R_n, g_load}, sweep ``n_samples``
   linspace points over ``(1 - delta) * nominal`` to ``(1 + delta) *
   nominal`` and re-run the pipeline at each point. The default
   spec produces ``4 * 11 = 44`` perturbed scenarios.
3. For every material in the nominal viable list, count the
   scenarios where it remained viable. The fraction maps to a
   discrete robustness label:

   ===============  =====================
   Fraction         Label
   ===============  =====================
   >= 0.90          ``"robust"``
   0.50 <= f < 0.90 ``"borderline"``
   < 0.50           ``"knife-edge"``
   ===============  =====================

   ``critical_input`` is the input whose sweep dropped the material
   from ``viable`` the most often (tie-break by INPUT order:
   mach > mass > R_n > g_load).

4. For the top-ranked nominal material, build a tornado dictionary:
   ``{input_name: max_abs_change_in_min_margin}``. The accompanying
   chart is rendered with a headless matplotlib (``Agg`` backend)
   and returned as raw PNG bytes — callers (LaTeX, Streamlit) decide
   whether to base64-encode or write to a file.

Hard rules
----------
* No ``streamlit`` import — this module must remain importable in
  the headless CLI environment (enforced by
  ``test_api.TestStreamlitFreeCLI``).
* No filesystem writes. The chart is returned in-memory.
* No physics constants are read or modified — only the public
  ``run_analysis`` / ``match_materials`` entry points are called.
* Deterministic: ``numpy.linspace`` provides a fixed sweep grid,
  no random sampling.

The ``apply_turbine_override`` step is mirrored from
``core.api.run_session`` so that turbine envelopes (M=0.5/SL with a
hot-section temperature override) are evaluated against the actual
metal-face temperature, not the meaningless aerodynamic recovery
value.
"""

import io
from dataclasses import dataclass, field

import numpy as np

from physics_engine import run_analysis, PhysicsResult
from matching_engine import match_materials, MatchResult
from core.session import SessionSchema
from core.api import apply_turbine_override


# ---------------------------------------------------------------------------
# Robustness label thresholds — encoded once so tests and downstream
# consumers (LaTeX narrative, Streamlit badges) can import them rather
# than re-encode the magic numbers.
# ---------------------------------------------------------------------------

ROBUST_THRESHOLD     = 0.90    # fraction >= this → "robust"
BORDERLINE_THRESHOLD = 0.50    # fraction >= this AND < ROBUST_THRESHOLD → "borderline"
                                # fraction <  this → "knife-edge"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SensitivitySpec:
    """Per-input perturbation magnitudes and sweep granularity.

    Defaults reflect the realistic uncertainty bands a screening
    engineer faces *before* a wind-tunnel campaign: Mach is the
    best-known input (autopilot or trajectory simulation); g-load
    is the least-known (mission profile is preliminary).
    """
    mach_delta_frac: float    = 0.10   # +/-10% on Mach
    mass_delta_frac: float    = 0.15   # +/-15% on vehicle mass
    R_n_delta_frac: float     = 0.20   # +/-20% on nose radius
    g_load_delta_frac: float  = 0.25   # +/-25% on peak g-load
    n_samples: int            = 11     # sweep points per input (incl. endpoints)


@dataclass
class MaterialRobustness:
    """One row of the robustness table: how often a nominally-viable
    material survives the perturbation sweep.

    ``critical_input`` is the single input whose sweep contributed
    the most drops. ``"none"`` when the material survived every
    perturbation (the happy case for genuinely robust picks).
    """
    material_name: str
    n_scenarios_viable: int
    n_scenarios_total: int
    robustness_fraction: float
    robustness_label: str       # "robust" / "borderline" / "knife-edge"
    critical_input: str         # "mach" / "mass" / "R_n" / "g_load" / "none"


@dataclass
class SensitivityResult:
    """Bundle of everything ``run_sensitivity`` produces.

    ``nominal_physics`` and ``nominal_match`` echo the baseline run
    so the LaTeX and Streamlit layers can show "the table you see in
    section 8" without having to re-run the pipeline.

    ``tornado_data`` maps input-name → max absolute change in the
    top-nominal material's min-margin across that input's sweep.
    Always non-negative. Empty dict when the nominal viable list is
    empty.

    ``chart_png`` is raw PNG bytes — never base64-encoded at this
    layer. Empty placeholder bytes when the nominal viable list is
    empty (the chart still renders, but with a "no candidates"
    message rather than crashing the LaTeX include).
    """
    spec: SensitivitySpec
    nominal_physics: PhysicsResult
    nominal_match: MatchResult
    materials: list                          # list[MaterialRobustness]
    tornado_data: dict                       # dict[str, float]
    chart_png: bytes
    top_material_name: str = ""              # name of the material the tornado describes;
                                             # empty when no nominal viable material exists
    baseline_min_margin: float = 0.0         # min-margin of the top material at the
                                             # nominal envelope. The Streamlit st.metric
                                             # next to the chart reads this directly so
                                             # the operator gets a glanceable KPI without
                                             # the chart having to carry it in its title.


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Canonical sweep order. Used everywhere a deterministic input
# enumeration matters — tornado bar order, critical-input tie-break,
# tornado-dict insertion order. Listed mach→mass→R_n→g_load roughly
# in "engineer trust" order: Mach is best known, g_load is least.
_INPUT_ORDER = ("mach", "mass", "R_n", "g_load")


# Plain-English translations of the four sweep-input keys. Single
# source of truth — imported by app.py for the chart's Y-axis labels,
# the materials-table "Critical input" column, and the risk-notes
# block; rendered into the matplotlib chart by _render_tornado. Adding
# a fifth sweep input would mean adding it here AND to _INPUT_ORDER
# and the four lookup helpers below — keep the two structures aligned.
_INPUT_DISPLAY_NAMES = {
    "mach":   "Peak Mach",
    "mass":   "Vehicle mass",
    "R_n":    "Nose radius",
    "g_load": "Peak g-load",
}

# Margin-erosion values below this many percentage points are rendered
# as dimmed grey bars annotated "negligible". Anchored in pp because
# the tornado now plots margin-erosion in pp (see _render_tornado).
# 0.5 pp ≈ 0.005 fractional shift — well inside the round-off floor of
# any responsible material screening decision.
_NEG_THRESHOLD_PP = 0.5


def _envelope_to_kwargs(envelope):
    """Project a SessionSchema (or any duck-typed envelope) into the
    kwargs ``run_analysis`` expects.

    Kept as a thin helper so the per-sweep loop can ``dict.update``
    the perturbed input without rebuilding the whole bundle. Order
    of keys is irrelevant — ``run_analysis`` consumes them by name."""
    return {
        "peak_mach":               float(envelope.mach),
        "cruise_altitude_km":      float(envelope.alt_km),
        "vehicle_mass_kg":         float(envelope.mass_kg),
        "nose_radius_m":           float(envelope.R_n_m),
        "peak_g_load":             float(envelope.g_load),
        "characteristic_length_m": float(envelope.char_len_m),
        "flight_duration_s":       float(envelope.flight_duration_s),
        "wall_emissivity":         float(envelope.wall_emissivity),
    }


def _maybe_apply_turbine(physics, envelope, vehicle_category):
    """Mirror the turbine override that ``core.api.run_session``
    applies at line ~159 — without it, sweep scenarios for the
    turbine category would re-evaluate against the meaningless
    aerodynamic T_wall (~300 K at M=0.5/SL) rather than the
    hot-section metal-face temperature.

    Opt-in: a turbine envelope without ``hot_section_temp_K`` in
    options gets the default aerodynamic physics, which matches
    what ``run_session`` would produce for the same envelope.
    """
    if vehicle_category != "turbine":
        return physics
    options = getattr(envelope, "options", None) or {}
    hot_K = options.get("hot_section_temp_K")
    if hot_K is None:
        return physics
    return apply_turbine_override(physics, float(hot_K))


def _run_pipeline(envelope, vehicle_category, **input_overrides):
    """Run ``run_analysis`` then ``match_materials``, optionally
    overriding one or more envelope inputs.

    Returns ``(physics, match)``. The turbine override is applied
    in between so the sweep sees the same composite physics that a
    real ``run_session`` would have produced.
    """
    kwargs = _envelope_to_kwargs(envelope)
    kwargs.update(input_overrides)
    physics = run_analysis(**kwargs)
    physics = _maybe_apply_turbine(physics, envelope, vehicle_category)
    match   = match_materials(physics, vehicle_category=vehicle_category)
    return physics, match


def _find_candidate(match, name):
    """Locate a candidate by material name across viable / marginal
    / not_viable. Returns None if the material was regime-rejected
    (no MaterialCandidate produced — only a MaterialEntry stub on
    the regime_rejected list)."""
    for bucket in (match.viable, match.marginal, match.not_viable):
        for c in bucket:
            if c.material.name == name:
                return c
    return None


def _candidate_min_margin(candidate, T_wall_K):
    """Min-margin metric for the tornado. Independent of the
    category-specific specific-strength weighting baked into
    ``candidate.score`` — this gives a tornado that means the same
    thing across aircraft / reentry / turbine.

    Formula matches the canonical ``min(thermal_margin_fraction,
    structural_margin_fraction)`` used by ``_score`` in
    matching_engine.

    Returns -1.0 sentinel for materials that disappeared from the
    candidate set entirely (regime-rejected after a perturbation).
    """
    if candidate is None:
        return -1.0
    if T_wall_K > 0.0:
        thermal_frac = candidate.thermal_margin_K / T_wall_K
    else:
        thermal_frac = 1e9
    return min(thermal_frac, candidate.structural_margin_fraction)


def _label_for_fraction(fraction):
    """Classify a robustness fraction into one of three labels.
    Thresholds live as module-level constants so tests can import
    them rather than hard-code 0.90 / 0.50 in three places."""
    if fraction >= ROBUST_THRESHOLD:
        return "robust"
    if fraction >= BORDERLINE_THRESHOLD:
        return "borderline"
    return "knife-edge"


def _input_value_for(envelope, input_name):
    """Map our short input name onto the matching SessionSchema
    attribute. Centralised so the sweep loop and chart annotation
    use the same lookup."""
    return {
        "mach":   envelope.mach,
        "mass":   envelope.mass_kg,
        "R_n":    envelope.R_n_m,
        "g_load": envelope.g_load,
    }[input_name]


def _sweep_kwarg_for(input_name, value):
    """Given the short input name and a perturbed scalar, return
    the kwarg dict to splat into ``run_analysis``."""
    return {
        "mach":   {"peak_mach":       float(value)},
        "mass":   {"vehicle_mass_kg": float(value)},
        "R_n":    {"nose_radius_m":   float(value)},
        "g_load": {"peak_g_load":     float(value)},
    }[input_name]


def _delta_frac_for(spec, input_name):
    return {
        "mach":   spec.mach_delta_frac,
        "mass":   spec.mass_delta_frac,
        "R_n":    spec.R_n_delta_frac,
        "g_load": spec.g_load_delta_frac,
    }[input_name]


# ---------------------------------------------------------------------------
# Tornado chart rendering
# ---------------------------------------------------------------------------

# Chart palette — matches the dark theme used by core/pareto.py so
# Streamlit / LaTeX renders share a visual style.
_CHART_BG      = "#0d1117"
_AXES_BG       = "#161b22"
_FG_TEXT       = "#c9d1d9"
_FG_AXIS       = "#8b949e"
_FG_SPINE      = "#30363d"
_BAR_COLOR     = "#58a6ff"
_BAR_EDGE      = "#c9d1d9"
_FAIL_LINE     = "#f85149"


def _render_tornado(top_material_name, tornado_data, baseline_min_margin):
    """Render a horizontal-bar tornado chart of the per-input
    safety-margin erosion for the top-nominal material.

    Design notes
    ------------
    * Plot units are *percentage points* of safety margin (a 0.085
      fractional shift in min-margin is rendered as 8.5 pp). This
      reads more concretely than raw fractions and matches the
      "robustness" framing on the materials table next to the chart.
    * Bars below ``_NEG_THRESHOLD_PP`` are rendered dimmed and
      annotated "negligible" rather than showing a near-zero stub
      that looks like a render bug. This is the visual fix for the
      common SR-71-class case where only Mach moves the needle.
    * Title is a single line ("Which inputs threaten viability of
      <material>?") — the baseline-margin number is now surfaced
      separately as a Streamlit ``st.metric`` next to the chart, so
      the chart itself doesn't have to carry that weight.
    * Y-axis labels run through ``_INPUT_DISPLAY_NAMES`` so the chart
      matches the "Critical input" column wording elsewhere in the UI.
    * figsize/dpi are tuned so the source PNG width (~550 px) is the
      LaTeX render width (~5.07 in @ 0.78 textwidth on A4) at native
      resolution — zero upscale, zero aliasing in the PDF.

    Always returns valid PNG bytes. When ``tornado_data`` is empty
    (no nominal viable material), renders a placeholder so the LaTeX
    include never breaks the build.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    fig.patch.set_facecolor(_CHART_BG)
    ax.set_facecolor(_AXES_BG)

    if not tornado_data:
        ax.text(
            0.5, 0.5, "Sensitivity not computed (no viable material)",
            ha="center", va="center", color=_FG_AXIS, fontsize=10,
            transform=ax.transAxes,
        )
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(_FG_SPINE)
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", dpi=110, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    # Sort by magnitude descending — largest swing on top. Convert
    # raw fractional shifts to percentage points for display.
    items     = sorted(tornado_data.items(), key=lambda kv: kv[1], reverse=True)
    labels    = [_INPUT_DISPLAY_NAMES.get(k, k) for k, _ in items]
    values_pp = [v * 100.0 for _, v in items]

    y = np.arange(len(items))

    # Two-pass barh — significant bars in the saturated brand color,
    # negligible bars greyed-out at low alpha. This is what turns the
    # "looks broken" four-bar chart into "one bar matters, the rest
    # don't, and I can see at a glance which is which".
    sig_y = [yi for yi, v in zip(y, values_pp) if v >= _NEG_THRESHOLD_PP]
    sig_v = [v  for      v in values_pp        if v >= _NEG_THRESHOLD_PP]
    neg_y = [yi for yi, v in zip(y, values_pp) if v <  _NEG_THRESHOLD_PP]
    neg_v = [v  for      v in values_pp        if v <  _NEG_THRESHOLD_PP]

    if sig_v:
        ax.barh(
            sig_y, sig_v, color=_BAR_COLOR, edgecolor=_BAR_EDGE,
            linewidth=0.7, height=0.55, zorder=3,
        )
    if neg_v:
        # alpha=0.35 + grey color makes "we did sweep this and nothing
        # happened" read as a deliberate finding rather than a missing bar.
        ax.barh(
            neg_y, neg_v, color=_FG_AXIS, edgecolor=_FG_SPINE,
            linewidth=0.5, height=0.55, alpha=0.35, zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=_FG_TEXT, fontsize=9)
    ax.invert_yaxis()                        # largest bar at top

    # Per-bar text annotation — value in pp for significant bars,
    # the literal word "negligible" for sub-threshold bars. Color
    # follows the bar color so the dim bars' annotations don't shout
    # louder than the bars themselves.
    for yi, v in zip(y, values_pp):
        if v >= _NEG_THRESHOLD_PP:
            label = f"  {v:.1f} pp"
            color = _FG_TEXT
        else:
            label = "  negligible"
            color = _FG_AXIS
        ax.text(v, yi, label, va="center", ha="left",
                color=color, fontsize=8)

    ax.set_xlabel(
        "Worst-case erosion of safety margin (pp)",
        color=_FG_TEXT, fontsize=9,
    )
    ax.set_title(
        f"Which inputs threaten viability of {top_material_name}?",
        color=_FG_TEXT, fontsize=10,
    )
    ax.tick_params(colors=_FG_AXIS)
    for spine in ax.spines.values():
        spine.set_color(_FG_SPINE)

    # A vertical zero reference makes "tiny but non-zero" shifts
    # visually distinguishable from the y-axis.
    ax.axvline(0, color=_FAIL_LINE, linewidth=0.8, linestyle="--", alpha=0.5)

    # Right-padding for the per-bar annotations under bbox_inches="tight".
    # 1.30 (vs the previous 1.25) gives "negligible" enough room for its
    # 11-character footprint at fontsize 8.
    if values_pp:
        xmax = max(values_pp)
        ax.set_xlim(0.0, xmax * 1.30 if xmax > 0 else 1.0)

    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=110, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_sensitivity(envelope, vehicle_category, spec=None):
    """Sweep four envelope inputs and report per-material robustness.

    Parameters
    ----------
    envelope : SessionSchema (or duck-typed equivalent)
        Carries ``mach``, ``alt_km``, ``mass_kg``, ``R_n_m``,
        ``g_load``, ``char_len_m``, ``flight_duration_s``,
        ``wall_emissivity``, and an ``options`` dict (used only for
        the turbine ``hot_section_temp_K`` override).
    vehicle_category : str
        One of the matching-engine category keys
        (``"aircraft"`` / ``"hypersonic_aircraft"`` /
        ``"hypersonic_missile"`` / ``"reentry"`` / ``"turbine"`` /
        ``"general"``). Passed straight through to
        ``match_materials``.
    spec : SensitivitySpec, optional
        Per-input perturbation deltas and sample count. Defaults
        to the calibration values on ``SensitivitySpec``.

    Returns
    -------
    SensitivityResult
        See dataclass docstring. Even for a degenerate envelope
        with no viable material, every field is populated — the
        ``materials`` list and ``tornado_data`` dict are simply
        empty, and ``chart_png`` carries a placeholder PNG.

    Notes
    -----
    Determinism: the sweep grid is ``numpy.linspace((1-d)*x, (1+d)*x,
    n)`` for each input. Two calls with the same envelope, category,
    and spec produce byte-identical ``chart_png`` and equal
    ``materials`` lists. This is enforced by the determinism test in
    Deliverable E.

    Side-effect freedom: ``run_sensitivity`` does not mutate
    ``envelope``, does not touch ``materials_db``, and does not
    write to disk. The only output is the returned
    ``SensitivityResult``.
    """
    if spec is None:
        spec = SensitivitySpec()

    # ---- Nominal pipeline ----
    nom_physics, nom_match = _run_pipeline(envelope, vehicle_category)
    nom_T_wall  = nom_physics.thermal.T_wall_K
    nominal_viable_names = [c.material.name for c in nom_match.viable]

    # ---- Degenerate case: no viable nominal materials ----
    if not nominal_viable_names:
        chart = _render_tornado("(no viable material)", {}, 0.0)
        return SensitivityResult(
            spec=spec,
            nominal_physics=nom_physics,
            nominal_match=nom_match,
            materials=[],
            tornado_data={},
            chart_png=chart,
            top_material_name="",
        )

    # ---- Per-input sweep ----
    # sweeps[input_name] = list of (sample_value, MatchResult, T_wall_K)
    sweeps = {}
    for input_name in _INPUT_ORDER:
        nom_val = _input_value_for(envelope, input_name)
        delta   = _delta_frac_for(spec, input_name)
        lo      = (1.0 - delta) * nom_val
        hi      = (1.0 + delta) * nom_val
        samples = np.linspace(lo, hi, spec.n_samples)
        sweeps[input_name] = []
        for s in samples:
            override = _sweep_kwarg_for(input_name, float(s))
            phys, mres = _run_pipeline(envelope, vehicle_category, **override)
            sweeps[input_name].append((float(s), mres, float(phys.thermal.T_wall_K)))

    # ---- Robustness scoring per nominal-viable material ----
    materials = []
    for name in nominal_viable_names:
        n_total       = 0
        n_viable      = 0
        per_input_drops = {}
        for input_name in _INPUT_ORDER:
            n_in     = 0
            n_in_vbl = 0
            for _s_val, mres, _T in sweeps[input_name]:
                n_in += 1
                if any(c.material.name == name for c in mres.viable):
                    n_in_vbl += 1
            per_input_drops[input_name] = n_in - n_in_vbl
            n_total  += n_in
            n_viable += n_in_vbl

        # Critical input = whichever sweep produced the most drops.
        # When everything held, critical_input is "none" — that's the
        # honest signal for a genuinely robust pick.
        max_drops = max(per_input_drops.values())
        if max_drops == 0:
            critical = "none"
        else:
            # Deterministic tie-break: walk _INPUT_ORDER and pick first.
            critical = next(
                inp for inp in _INPUT_ORDER
                if per_input_drops[inp] == max_drops
            )

        frac = n_viable / n_total if n_total > 0 else 0.0
        materials.append(MaterialRobustness(
            material_name        = name,
            n_scenarios_viable   = n_viable,
            n_scenarios_total    = n_total,
            robustness_fraction  = frac,
            robustness_label     = _label_for_fraction(frac),
            critical_input       = critical,
        ))

    # ---- Tornado for the top-nominal material ----
    # The "top" is whichever the matching engine ranked first in viable.
    # For aircraft/missile categories that's a specific-strength-weighted
    # pick; for everyone else it's pure min-margin descending. Either way
    # it's the material the user is most likely to act on — so it's the
    # right one to characterise the sensitivity of.
    top_name      = nominal_viable_names[0]
    top_nom_cand  = _find_candidate(nom_match, top_name)
    baseline_mm   = _candidate_min_margin(top_nom_cand, nom_T_wall)

    tornado_data = {}
    for input_name in _INPUT_ORDER:
        max_delta = 0.0
        for _s_val, mres, T in sweeps[input_name]:
            cand    = _find_candidate(mres, top_name)
            mm      = _candidate_min_margin(cand, T)
            delta   = abs(mm - baseline_mm)
            if delta > max_delta:
                max_delta = delta
        tornado_data[input_name] = max_delta

    chart_png = _render_tornado(top_name, tornado_data, baseline_mm)

    return SensitivityResult(
        spec                = spec,
        nominal_physics     = nom_physics,
        nominal_match       = nom_match,
        materials           = materials,
        tornado_data        = tornado_data,
        chart_png           = chart_png,
        top_material_name   = top_name,
        baseline_min_margin = baseline_mm,
    )
