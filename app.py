"""
MATVEC — Aerospace Materials Feasibility Tool
Step 4 Streamlit UI: professional scientific instrument aesthetic.

Run: streamlit run app.py
"""

import math
import re
import datetime
from pathlib import Path
import streamlit as st

from matching_engine import match_materials, TPS_UNLOCK_TEMP_K
from physics_engine import run_analysis, envelope_summary
from core.category_inference import infer_category
from latex_export import generate_report, generate_tex_source
from core.pareto import compute_pareto
from core.surrogate import find_nearest_candidates
# Canonical pipeline + JSON schema live in core.* so the CLI and tests
# can reuse them without importing Streamlit. app.py is a thin UI shell
# that builds a SessionSchema from widget state and renders whatever
# run_session() returns.
from core.api import run_session, apply_turbine_override as _apply_turbine_override  # noqa: F401
from core.session import (
    SessionSchema,
    session_to_json,
    json_to_session,
)
# Sensitivity analysis is opt-in (sidebar checkbox), so we import the module
# unconditionally but only call into it when the user enables the feature.
# Pure module — no streamlit dependency, headless-safe, returns PNG bytes.
from core.sensitivity import (
    SensitivitySpec,
    SensitivityResult,
    _INPUT_DISPLAY_NAMES,
    run_sensitivity,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MATVEC | Materials Feasibility",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants matching physics_engine.py
# ---------------------------------------------------------------------------
_GAMMA = 1.4
_R_AIR = 287.05
_RECOVERY_FACTOR = 0.85
_SAFETY_FACTOR = 1.5   # MIL-HDBK-5

# NOTE: the in-source EXAMPLES dict, _PRESET_NOTES dict, and image-binding
# (VEHICLE_IMG_LOCAL/URL) maps were removed when the preset dropdown was
# replaced by the bundled-JSON-files approach (presets/*.json + the
# "Load bundled example" dropdown inside the Session I/O expander).
# core.presets.CANONICAL_PRESETS remains the single source of truth for
# tests + CLI; presets/*.json are mechanically generated from it via
# scripts/generate_example_presets.py.


VEHICLE_CATEGORIES = {
    "general":             "General Structure",
    "aircraft":            "Aircraft / Airframe",
    "hypersonic_aircraft": "Hypersonic Aircraft / Spaceplane",
    "reentry":             "Reentry Vehicle",
    "hypersonic_missile":  "High-Speed Missile",
    "turbine":             "Turbine Component",
}

CATEGORY_DESCRIPTIONS = {
    "general":             "No category-specific filtering. All regime-appropriate materials evaluated with standard thermal/structural criteria.",
    "aircraft":            "Airframe and structural panels. TPS, exotic refractories, and ceramic matrix composites excluded. Recovery-temperature cap treated as hard upper bound for thermal check. Specific strength weighted 40% in score to penalise dense alloys.",
    "hypersonic_aircraft": "Rocket-powered hypersonic aircraft and spaceplanes (X-15, Space Shuttle Orbiter, X-37). TPS materials (ablative and reusable) included as hot-face options; hot-structure alloys up to 8500 kg/m³ allowed for primary structure. Structural check bypassed for TPS category. Specific strength weighted 40% in score — weight still matters, but hot-structure density is relaxed vs. sustained-cruise airframes.",
    "reentry":             "Capsule heat shield and TPS panels. Ablative TPS materials included regardless of altitude-based regime classifier. Structural check bypassed for TPS category (ablators are not load-bearing).",
    "hypersonic_missile":  "High-speed (M > 2) expendable missile body structure. TPS and polymer composites excluded. Specific strength weighted 60% to penalise heavy alloys for expendable applications.",
    "turbine":             "High-pressure turbine blades and vanes. Aluminium, polymer composites, and general engineering materials excluded. Strength derated 40% as creep proxy for sustained high-temperature loading.",
}

# Bundled-example presets directory — the UI's "Load bundled example"
# dropdown (inside the Session I/O expander) globs *.json files here.
# Files are mechanically generated from core.presets.CANONICAL_PRESETS
# via scripts/generate_example_presets.py. Adding a new bundled
# example: edit the script, re-run, commit the resulting *.json.
_PRESETS_DIR = Path(__file__).resolve().parent / "presets"
_BUNDLED_PLACEHOLDER = "— Select a bundled example —"


# ---------------------------------------------------------------------------
# Category display labels
# ---------------------------------------------------------------------------
_CATEGORY_LABELS = {
    "aluminum":            "Aluminum",
    "titanium":            "Titanium",
    "steel":               "Steel",
    "nickel":              "Nickel Superalloy",
    "cobalt":              "Cobalt Superalloy",
    "refractory":          "Refractory Metal",
    "composite_polymer":   "Polymer Composite",
    "composite_ceramic":   "Ceramic Composite",
    "uhtc":                "UHTC",
    "tps":                 "TPS Ablator",
    "carbon":              "Carbon",
    "general_engineering": "General Engineering",
}

def _cat_label(cat: str) -> str:
    return _CATEGORY_LABELS.get(cat, cat)


def _make_slug(label: str) -> str:
    """Convert a system label to a filename-safe slug."""
    s = label.lower().replace(" ", "-")
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "analysis"


# ``_apply_turbine_override`` is imported from core.api near the top of
# this file — keeping the symbol available under the old name so the
# existing ``from app import _apply_turbine_override`` callsites in
# test_latex_export.py and test_matching_engine.py keep working.


# ---------------------------------------------------------------------------
# CSS injection — dark scientific instrument theme
# ---------------------------------------------------------------------------
_CSS = """
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: -apple-system, "Segoe UI", sans-serif;
    font-size: 14px;
}
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
}
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #c9d1d9 !important;
}
/* ── Main content area ── */
.block-container {
    background-color: #0d1117;
    padding-top: 3.5rem !important;
}
/* ── Inputs ── */
input[type="number"], input[type="text"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 4px !important;
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace !important;
}
input[type="number"]:focus, input[type="text"]:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}
.stNumberInput label, .stTextInput label, .stRadio label {
    color: #8b949e !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
/* ── Buttons (sharp corners) ── */
button[kind="primary"], button[kind="secondary"], .stButton > button {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 3px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 0.3rem 0.8rem !important;
    transition: border-color 0.15s, background-color 0.15s;
}
.stButton > button:hover {
    border-color: #58a6ff !important;
    background-color: #1f2937 !important;
    color: #58a6ff !important;
}
/* ── Selectbox ── */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 4px !important;
    color: #c9d1d9 !important;
}
/* ── Metrics ── */
[data-testid="metric-container"] {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 0.75rem 1rem;
}
[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace !important;
    font-size: 1.5rem !important;
    color: #58a6ff !important;
}
/* ── Tabs ── */
[data-testid="stTabs"] {
    background-color: #0d1117;
}
button[data-baseweb="tab"] {
    background-color: transparent !important;
    border-bottom: 2px solid transparent !important;
    color: #8b949e !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
}
/* ── Info / Warning / Error boxes ── */
.stAlert {
    border-radius: 3px !important;
    border-left-width: 3px !important;
}
/* ── Divider ── */
hr {
    border-color: #21262d !important;
    margin: 1rem 0 !important;
}
/* ── Expander ── */
details summary {
    color: #8b949e !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
/* ── Instrument card ── */
.instrument-card {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.card-title {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8b949e;
    margin-bottom: 0.4rem;
}
.card-primary-row {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    margin-bottom: 0.6rem;
}
.card-value {
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
    font-size: 1.75rem;
    font-weight: 700;
    color: #58a6ff;
    line-height: 1;
}
.card-unit {
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
    font-size: 0.9rem;
    color: #8b949e;
}
.card-range {
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
    font-size: 0.75rem;
    color: #8b949e;
    margin-bottom: 0.5rem;
}
.card-row {
    font-size: 13px;
    color: #c9d1d9;
    margin: 0.2rem 0;
    display: flex;
    gap: 0.5rem;
}
.card-row-label {
    color: #8b949e;
    min-width: 10rem;
    font-size: 12px;
}
.card-row-value {
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
    font-size: 13px;
    color: #c9d1d9;
}
.card-badge {
    display: inline-block;
    padding: 0.1rem 0.45rem;
    border-radius: 2px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.4rem;
}
.badge-warn {
    background-color: rgba(210,153,34,0.15);
    border: 1px solid #d29922;
    color: #d29922;
}
.badge-ok {
    background-color: rgba(35,134,54,0.15);
    border: 1px solid #238636;
    color: #3fb950;
}
/* ── Atmosphere panel ── */
.atm-panel {
    background-color: #0d1117;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
}
.atm-title {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8b949e;
    margin-bottom: 0.4rem;
}
.atm-row {
    font-size: 12px;
    color: #8b949e;
    display: flex;
    gap: 0.5rem;
    margin: 0.15rem 0;
}
.atm-val {
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
    color: #c9d1d9;
}
/* ── Material table ── */
.matvec-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    color: #c9d1d9;
    margin-top: 0.5rem;
}
.matvec-table thead tr {
    border-bottom: 1px solid #30363d;
}
.matvec-table th {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b949e;
    font-weight: 600;
    padding: 0.4rem 0.6rem;
    text-align: left;
    white-space: nowrap;
}
.matvec-table td {
    padding: 0.35rem 0.6rem;
    border-bottom: 1px solid #161b22;
    vertical-align: middle;
    white-space: nowrap;
}
.matvec-table .col-mono {
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
}
.matvec-table .col-notes {
    white-space: normal;
    font-size: 11px;
    color: #8b949e;
    max-width: 280px;
}
.matvec-table tr.row-viable > td:first-child {
    border-left: 3px solid #238636;
    padding-left: 0.5rem;
}
.matvec-table tr.row-marginal > td:first-child {
    border-left: 3px solid #d29922;
    padding-left: 0.5rem;
}
.matvec-table tr.row-fail > td:first-child {
    border-left: 3px solid #da3633;
    padding-left: 0.5rem;
}
.matvec-table tr.row-coating > td:first-child {
    border-left: 3px solid #f0883e;
    padding-left: 0.5rem;
}
.matvec-table tr:hover td {
    background-color: #1c2128;
}
/* ── Export section ── */
.export-section {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    margin-top: 1.5rem;
}
.export-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
    margin-bottom: 0.75rem;
}
.btn-export-primary {
    display: inline-block;
    background-color: #1f6feb;
    color: #ffffff !important;
    border: 1px solid #388bfd;
    border-radius: 3px;
    padding: 0.5rem 1.25rem;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none;
}
.btn-export-secondary {
    display: inline-block;
    background-color: #161b22;
    color: #c9d1d9 !important;
    border: 1px solid #30363d;
    border-radius: 3px;
    padding: 0.5rem 1.25rem;
    font-size: 13px;
    cursor: pointer;
}
/* ── Section headers ── */
.section-header {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #58a6ff;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}
/* ── Regime badge ── */
.regime-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 2px;
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.regime-subsonic   { background-color: rgba(88,166,255,0.1);  border:1px solid #58a6ff; color:#79c0ff; }
.regime-supersonic { background-color: rgba(210,153,34,0.1);  border:1px solid #d29922; color:#e3b341; }
.regime-hypersonic { background-color: rgba(218,55,51,0.1);   border:1px solid #da3633; color:#f85149; }
.regime-reentry    { background-color: rgba(188,103,252,0.1); border:1px solid #bc6ffc; color:#d2a8ff; }
/* ── Computed metric ── */
.computed-metric {
    font-size: 12px;
    color: #8b949e;
    margin-top: 0.25rem;
    margin-bottom: 0.6rem;
}
.computed-metric span {
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
    color: #79c0ff;
}
/* ── Equation section ── */
.eq-panel {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.eq-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
    margin-bottom: 0.25rem;
}
</style>
"""


# ---------------------------------------------------------------------------
# ISA speed-of-sound helper (for km/h ↔ Mach display only)
# ---------------------------------------------------------------------------

def _isa_speed_of_sound_ms(alt_km: float) -> float:
    """Speed of sound from ISA standard atmosphere (m/s), matching physics_engine layers."""
    if alt_km <= 11.0:
        T = 288.15 - 6.5 * alt_km
    elif alt_km <= 20.0:
        T = 216.65
    elif alt_km <= 32.0:
        T = 216.65 + 1.0 * (alt_km - 20.0)
    elif alt_km <= 47.0:
        T = 228.65 + 2.8 * (alt_km - 32.0)
    elif alt_km <= 51.0:
        T = 270.65
    elif alt_km <= 71.0:
        T = 270.65 - 2.8 * (alt_km - 51.0)
    else:
        T = max(214.65 - 2.0 * (alt_km - 71.0), 186.0)
    return math.sqrt(_GAMMA * _R_AIR * T)


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------

def _card_row(label: str, value: str, unit: str = "") -> str:
    unit_span = f" <span style='color:#8b949e;font-size:11px'>{unit}</span>" if unit else ""
    return (
        f'<div class="card-row">'
        f'<span class="card-row-label">{label}</span>'
        f'<span class="card-row-value">{value}{unit_span}</span>'
        f'</div>'
    )


def _instrument_card(title: str, primary_value: str, primary_unit: str,
                     body_html: str, extra_badges: str = "") -> str:
    return f"""
<div class="instrument-card">
  <div class="card-title">{title}</div>
  <div class="card-primary-row">
    <span class="card-value">{primary_value}</span>
    <span class="card-unit">{primary_unit}</span>
  </div>
  {body_html}
  {extra_badges}
</div>"""


def _format_cost_usd_html(usd: float) -> str:
    """Compact USD formatter for the materials table (HTML twin of the
    LaTeX `_format_cost_usd` helper).

    Returns a dash for zero (exotic/2D sentinel) and SI suffixes
    everywhere else so wide tables stay scannable.
    """
    if usd is None or usd <= 0.0:
        return "&mdash;"
    if usd < 1.0:
        return "&lt;$1"
    if usd < 1e3:
        return f"${usd:.0f}"
    if usd < 1e6:
        return f"${usd / 1e3:.1f}k"
    if usd < 1e9:
        return f"${usd / 1e6:.1f}M"
    return f"${usd / 1e9:.1f}B"


_ROBUSTNESS_BADGE_STYLE = {
    # Green / yellow / red badges keyed by sensitivity label. Centralised
    # so the colour story matches between the materials table and the
    # standalone Sensitivity card below it.
    "robust":     ("#238636", "#fff",   "ROBUST"),
    "borderline": ("#d29922", "#0d1117", "BORDERLINE"),
    "knife-edge": ("#f85149", "#fff",   "KNIFE-EDGE"),
}


def _robustness_badge_html(label: str, fraction: float) -> str:
    """Pill badge for the Robustness column. Falls back to a neutral
    grey badge if an unknown label sneaks in (defensive — shouldn't
    happen with the current `run_sensitivity` output)."""
    bg, fg, txt = _ROBUSTNESS_BADGE_STYLE.get(
        label, ("#30363d", "#c9d1d9", str(label).upper()),
    )
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
        f'background:{bg};color:{fg};font-size:11px;font-weight:700;'
        f'letter-spacing:0.05em">{txt}</span>'
        f'<span style="margin-left:6px;color:#8b949e;font-size:11px">'
        f'{fraction:.0%}</span>'
    )


def _tornado_row_html(label: str, value: float, value_max: float) -> str:
    """One row of the inline HTML tornado — mirrors the Pareto _bar_cell
    pattern at app.py:1916. ``value`` and ``value_max`` are fractional
    margin-erosions; bar width scales relative to ``value_max`` so the
    chart stays readable even when one input dominates.

    Significance tiers:
      * rel < 0.05  → grey bar + italic "negligible" tag (visually
        "present but trivial", not "broken/empty")
      * rel < 0.34  → cool brand-blue (#388bfd)
      * rel < 0.67  → mid brand-blue (#58a6ff)
      * rel ≥ 0.67  → hot brand-blue (#79c0ff) — the dominant input(s)
    """
    if value_max <= 0:
        rel = 0.0
        pct = 0.0
    else:
        rel = value / value_max
        pct = max(0.0, min(100.0, rel * 100.0))

    if rel < 0.05:
        bar_color = "#30363d"
        value_text = (
            f"<span style='color:#8b949e'>{value*100:.2f} pp</span>"
            "<span style='color:#8b949e;font-size:11px;margin-left:8px;"
            "font-style:italic'>negligible</span>"
        )
    else:
        if rel < 0.34:
            bar_color = "#388bfd"
        elif rel < 0.67:
            bar_color = "#58a6ff"
        else:
            bar_color = "#79c0ff"
        value_text = f"<span style='color:#c9d1d9'>{value*100:.2f} pp</span>"

    return (
        "<tr>"
        f"<td style='color:#c9d1d9;width:140px;padding:6px 10px'>{label}</td>"
        "<td style='position:relative;padding:6px 10px;height:22px'>"
        f"<div style='position:absolute;left:10px;top:50%;transform:translateY(-50%);"
        f"height:14px;width:calc({pct:.1f}% - 20px);background:{bar_color};"
        "opacity:0.55;border-radius:3px'></div>"
        f"<div style='position:absolute;left:10px;top:50%;transform:translateY(-50%);"
        f"height:14px;border-left:2px solid {bar_color};border-radius:1px'></div>"
        "</td>"
        f"<td class='col-mono' style='text-align:right;width:160px;padding:6px 10px'>"
        f"{value_text}</td>"
        "</tr>"
    )


def _candidates_table_html(
    candidates: list,
    row_class: str,
    *,
    vehicle_mass_kg: float = 0.0,
    cost_ceiling_usd: float = 0.0,
    robustness_by_name: dict | None = None,
) -> str:
    if not candidates:
        return ""
    show_cost = vehicle_mass_kg and vehicle_mass_kg > 0.0
    show_robustness = robustness_by_name is not None and len(robustness_by_name) > 0
    headers = [
        "Material", "Category", "Ceiling (K)", "T_wall Margin (K)",
        "Thermal", "Strength@T (MPa)", "σ_req (MPa)", "Struct. Margin",
        "Structural",
        # Creep / lifecycle column — added in Phase 4 of the lifecycle
        # rollout. Cell shows status icon + rupture stress (or "—" for
        # not_applicable / "?" for unknown). Header gets a tooltip via
        # the surrounding caption rather than a per-th title attribute
        # so existing CSS doesn't need to change.
        "Creep@Life",
        # Transient soak column — added in Phase 7. Shows the 1D-solver
        # peak back-face temperature (K) for short-duration flights;
        # empty cell otherwise. Same pass / fail visual language as
        # the creep column.
        "Soak@Life",
        "Min. Margin (fraction)",
    ]
    if show_cost:
        headers.append("Est. Cost")
    if show_robustness:
        headers.append("Robustness")
    n_cols = len(headers)
    _SM_NEGLIGIBLE_TOOLTIP = (
        "title='Required stress is a tiny fraction of this material&apos;s strength. "
        "The percentage margin is mathematically huge but operationally meaningless — "
        "structural design here is governed by minimum-gauge / handling concerns, "
        "not yield. CTE mismatch or thermal stress may still drive material choice.'"
    )
    rows_html = ""
    for c in candidates:
        m = c.material
        thermal_icon = {"pass": "✅", "marginal": "⚠️", "fail": "❌"}.get(c.thermal_status, "")
        struct_icon  = {"pass": "✅", "marginal": "⚠️", "fail": "❌"}.get(c.structural_status, "")
        # Creep cell: status icon + rupture stress or sentinel.
        creep_icon = {
            "pass": "✅", "marginal": "⚠️", "fail": "❌",
            "unknown": "❓", "not_applicable": "—",
        }.get(getattr(c, "creep_status", "not_applicable"), "—")
        creep_status_str = getattr(c, "creep_status", "not_applicable")
        creep_rupture = getattr(c, "creep_rupture_stress_MPa", None)
        if creep_status_str in ("pass", "marginal", "fail") and creep_rupture is not None:
            margin_frac = getattr(c, "creep_margin_fraction", None) or 0.0
            mcolor = (
                "#3fb950" if margin_frac >= 0.20
                else "#d29922" if margin_frac >= 0.0
                else "#f85149"
            )
            creep_cell_html = (
                f'{creep_icon} <span style="color:{mcolor}">'
                f'{creep_rupture:.0f} MPa</span>'
            )
        elif creep_status_str == "unknown":
            creep_cell_html = (
                f'{creep_icon} <span style="color:#8b949e" '
                f'title="No Larson-Miller data sourced for this '
                f'material; creep behaviour at the queried (T, t) '
                f'point cannot be verified.">unknown</span>'
            )
        else:
            # not_applicable
            creep_cell_html = (
                f'<span style="color:#8b949e" '
                f'title="Material category does not classically creep '
                f'at relevant service temperatures (TPS / ceramic / '
                f'polymer / cool-soak metal).">{creep_icon} n/a</span>'
            )

        # Transient soak cell: rendered only when the 1D solver
        # actually ran for this candidate. Otherwise show a neutral
        # "—" so the column doesn't look broken on long-duration
        # flights.
        transient_status_str = getattr(c, "transient_status", "not_applicable")
        transient_peak = getattr(c, "transient_peak_backface_K", None)
        if transient_status_str == "applied" and transient_peak is not None:
            # Colour: cool soak = green, warm = amber, hot = red.
            ceiling_for_color = c.thermal_ceiling_K or 0.0
            margin_to_ceiling = ceiling_for_color - transient_peak
            tcolor = (
                "#3fb950" if margin_to_ceiling > 100.0
                else "#d29922" if margin_to_ceiling > 0.0
                else "#f85149"
            )
            transient_cell_html = (
                f'<span style="color:{tcolor}" '
                f'title="1D transient solver peak back-face '
                f'temperature over the flight; method='
                f'{getattr(c, "transient_method", "")}.">'
                f'{transient_peak:.0f} K</span>'
            )
        elif transient_status_str == "unknown":
            transient_cell_html = (
                f'<span style="color:#8b949e" '
                f'title="Transient solver could not run for this '
                f'material (no c_p data).">❓ unknown</span>'
            )
        else:
            transient_cell_html = (
                f'<span style="color:#8b949e" '
                f'title="Transient solver not invoked: sustained-flight '
                f'envelope, or non-conducting category (TPS / polymer). '
                f'The static T_wall check is the operative screen.">—</span>'
            )
        sm_pct = c.structural_margin_fraction * 100
        # When the structural margin exceeds 500% the percentage stops
        # carrying useful information — it just means σ_req is a tiny
        # fraction of the material's strength. Replace the number with a
        # plain-English tag so the engineer reads "load doesn't drive
        # this material" instead of "+250,093.4%".
        if sm_pct > 500:
            sm_cell_html = (
                f'<span style="color:#8b949e;cursor:help" {_SM_NEGLIGIBLE_TOOLTIP}>'
                f'≫ load (negligible)</span>'
            )
        else:
            sm_color = "#3fb950" if c.structural_margin_fraction >= 0 else "#f85149"
            sm_cell_html = (
                f'<span style="color:{sm_color}">{sm_pct:+.1f}%</span>'
            )
        # Substrate-mode badge: this candidate represents a metal substructure
        # under an ablative coating, evaluated at backside soak temperature
        substrate_badge = ""
        name_suffix = ""
        if getattr(c, "evaluation_mode", "direct") == "substrate":
            substrate_badge = (
                ' <span style="background:#1f6feb;color:#fff;padding:1px 6px;'
                'border-radius:3px;font-size:11px;font-weight:normal">substrate</span>'
            )
            name_suffix = " (under ablative)"
        notes_html = (
            f'<tr class="{row_class}"><td colspan="{n_cols}" class="col-notes" '
            f'style="border-left:none;padding-left:3.5rem;color:#8b949e">'
            f'{"  ".join(c.notes)}</td></tr>'
        ) if c.notes else ""
        # Cost cell + over-ceiling row treatment. The "row turns red" UX from
        # the spec is implemented as a left-border + red text on the cost
        # cell rather than a full-row repaint, so the existing thermal/
        # structural colour cues stay readable for at-a-glance feasibility.
        cost_cell_html = ""
        row_style = ""
        if show_cost:
            cost_per_kg = float(getattr(m, "cost_usd_per_kg", 0.0))
            row_cost = cost_per_kg * vehicle_mass_kg
            over_ceiling = (
                cost_ceiling_usd
                and cost_ceiling_usd > 0.0
                and row_cost > cost_ceiling_usd
            )
            cost_color = "#f85149" if over_ceiling else "#c9d1d9"
            cost_weight = "700" if over_ceiling else "400"
            cost_cell_html = (
                f'<td class="col-mono" '
                f'style="color:{cost_color};font-weight:{cost_weight};text-align:right">'
                f'{_format_cost_usd_html(row_cost)}</td>'
            )
            if over_ceiling:
                row_style = ' style="border-left:3px solid #f85149"'
        # Optional Robustness cell — populated when the user ran the
        # sensitivity analysis. Only viable / marginal candidates are
        # in the dict; rows without a match get a neutral '—' cell.
        robustness_cell_html = ""
        if show_robustness:
            robust = robustness_by_name.get(m.name) if robustness_by_name else None
            if robust is not None:
                robustness_cell_html = (
                    f'<td style="text-align:left">'
                    f'{_robustness_badge_html(robust.robustness_label, robust.robustness_fraction)}'
                    f'</td>'
                )
            else:
                robustness_cell_html = (
                    '<td style="text-align:left;color:#8b949e">—</td>'
                )
        rows_html += f"""
<tr class="{row_class}"{row_style}>
  <td><strong>{m.name}</strong>{name_suffix}{substrate_badge}</td>
  <td style="color:#8b949e">{_cat_label(m.category)}</td>
  <td class="col-mono">{c.thermal_ceiling_K:.0f}</td>
  <td class="col-mono" style="color:{'#3fb950' if c.thermal_margin_K >= 0 else '#f85149'}">{c.thermal_margin_K:+.0f}</td>
  <td>{thermal_icon} {c.thermal_status}</td>
  <td class="col-mono">{c.strength_at_T_wall_MPa:.0f}</td>
  <td class="col-mono">{c.sigma_req_material_MPa:.0f}</td>
  <td class="col-mono">{sm_cell_html}</td>
  <td>{struct_icon} {c.structural_status}</td>
  <td class="col-mono">{creep_cell_html}</td>
  <td class="col-mono">{transient_cell_html}</td>
  <td class="col-mono">{c.score:.4f}</td>
  {cost_cell_html}
  {robustness_cell_html}
</tr>{notes_html}"""
    thead = "".join(f"<th>{h}</th>" for h in headers)
    return f"""
<div style="overflow-x:auto">
<table class="matvec-table">
<thead><tr>{thead}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>"""


def _tps_coatings_table_html(coatings: list) -> str:
    """Dedicated table for non-load-bearing TPS / ablator coatings.
    Sorted by thermal margin descending; structural columns are intentionally
    omitted because TPS materials are not ranked on structural margin."""
    if not coatings:
        return ""
    headers = [
        "Coating Material", "Category", "Density (kg/m³)",
        "Ceiling (K)", "T_wall Margin (K)", "Thermal",
    ]
    rows_html = ""
    for c in coatings:
        m = c.material
        thermal_icon = {"pass": "✅", "marginal": "⚠️", "fail": "❌"}.get(c.thermal_status, "")
        notes_html = (
            f'<tr class="row-coating"><td colspan="6" class="col-notes" '
            f'style="border-left:none;padding-left:3.5rem;color:#8b949e">'
            f'{"  ".join(c.notes)}</td></tr>'
        ) if c.notes else ""
        rows_html += f"""
<tr class="row-coating">
  <td><strong>{m.name}</strong></td>
  <td style="color:#8b949e">{_cat_label(m.category)}</td>
  <td class="col-mono">{m.density_kgm3:.0f}</td>
  <td class="col-mono">{c.thermal_ceiling_K:.0f}</td>
  <td class="col-mono" style="color:{'#3fb950' if c.thermal_margin_K >= 0 else '#f85149'}">{c.thermal_margin_K:+.0f}</td>
  <td>{thermal_icon} {c.thermal_status}</td>
</tr>{notes_html}"""
    thead = "".join(f"<th>{h}</th>" for h in headers)
    return f"""
<div style="overflow-x:auto">
<table class="matvec-table">
<thead><tr>{thead}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>"""


def _rejected_table_html(rejected: list) -> str:
    if not rejected:
        return ""
    rows_html = ""
    for m in rejected:
        rows_html += f"""
<tr>
  <td>{m.name}</td>
  <td style="color:#8b949e">{_cat_label(m.category)}</td>
  <td style="color:#8b949e">{", ".join(m.applicable_regimes)}</td>
</tr>"""
    return f"""
<div style="overflow-x:auto">
<table class="matvec-table">
<thead><tr><th>Material</th><th>Category</th><th>Rated For</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>"""


# ---------------------------------------------------------------------------
# Session JSON I/O — Deliverable D
#
# Download/upload helpers. The *download* path builds a SessionSchema
# eagerly from the current widget values so st.download_button has the
# bytes ready at render time (one click → one file). The *upload* path
# parses the uploaded bytes, validates via json_to_session, then stages
# every resulting widget value in "_pending_session_values" and
# triggers a rerun — the staging key is drained at the very top of
# _sidebar() BEFORE widgets instantiate, which is Streamlit's only
# legal way to programmatically set a widget's value.
# ---------------------------------------------------------------------------

def _build_session_from_widgets(
    *, mach: float, alt_km: float, mass_kg: float, R_n: float,
    peak_g: float, epsilon: float, char_len: float,
    flight_duration_s: float, vehicle_category: str,
    hot_section_temp_K: float | None,
    design_lifetime_hours: float = 1.0,
    panel_thickness_m: float = 0.002,
    cost_ceiling_usd: float | None = None,
) -> SessionSchema:
    """Project the current sidebar widget values into a SessionSchema."""
    options: dict = {}
    if vehicle_category == "turbine" and hot_section_temp_K is not None:
        options["hot_section_temp_K"] = float(hot_section_temp_K)
    if cost_ceiling_usd is not None:
        options["cost_ceiling_usd"] = float(cost_ceiling_usd)
    system_label = st.session_state.get("system_label") or "Custom Analysis"
    return SessionSchema(
        mach=float(mach),
        alt_km=float(alt_km),
        mass_kg=float(mass_kg),
        R_n_m=float(R_n),
        g_load=float(peak_g),
        char_len_m=float(char_len),
        flight_duration_s=float(flight_duration_s),
        wall_emissivity=float(epsilon),
        design_lifetime_hours=float(design_lifetime_hours),
        panel_thickness_m=float(panel_thickness_m),
        vehicle_category=str(vehicle_category),
        system_label=str(system_label),
        options=options,
    )


def _session_to_pending_state(session: SessionSchema) -> dict:
    """Map a SessionSchema to the session_state keys the sidebar widgets
    are bound to. Returns a dict suitable for stashing under
    ``_pending_session_values`` (drained on the next rerun).

    The velocity-mode conversion is handled here: if the current widget
    state is in km/h mode we translate Mach → speed_kmh using the ISA
    speed of sound at the loaded altitude. This keeps roundtrip correct
    in both velocity modes.

    ``flight_duration_s``, ``epsilon``, ``char_len`` are bound to the
    ADVANCED expander widgets; without exposing them we'd silently
    clip non-default values on upload.
    """
    pending: dict = {
        "alt_km":            float(session.alt_km),
        "mass_kg":           float(session.mass_kg),
        "R_n":               float(session.R_n_m),
        "peak_g":            float(session.g_load),
        "char_len":          float(session.char_len_m),
        "epsilon":           float(session.wall_emissivity),
        "flight_duration_s": float(session.flight_duration_s),
        "design_lifetime_hours": float(session.design_lifetime_hours),
        # Convert metres → millimetres for the widget. The widget is
        # bound to a mm-formatted number_input; storing metres there
        # would force the user to read 0.0015 instead of 1.5.
        "panel_thickness_mm": float(session.panel_thickness_m) * 1000.0,
        "system_label":      str(session.system_label),
        # Category override must go through the staging key (the
        # category_override selectbox is widget-bound on the main canvas
        # right column). Setting it pins the override to the loaded
        # session's category, which suppresses the auto-inference.
        "_pending_category_override": str(session.vehicle_category),
        # Loading a JSON replaces the active preset; clear the preset
        # selector so the sectioned dropdown re-initialises to placeholder.
        "_last_example": None,
    }
    # Velocity — respect the active input mode if one has been chosen.
    if st.session_state.get("vel_mode", "Mach") == "Mach":
        pending["mach"] = float(session.mach)
    else:
        a_ms = _isa_speed_of_sound_ms(float(session.alt_km))
        pending["speed_kmh"] = float(session.mach) * a_ms * 3.6

    hot_K = session.options.get("hot_section_temp_K")
    if hot_K is not None:
        pending["hot_section_temp_K"] = float(hot_K)
    # Cost ceiling round-trip: the widget lives in the ADVANCED expander,
    # so a JSON download → load cycle must restore it just like flight
    # duration / emissivity / char_len.
    cost_ceiling = session.options.get("cost_ceiling_usd")
    if cost_ceiling is not None:
        pending["cost_ceiling_usd"] = float(cost_ceiling)
    return pending


def _stage_bundled_preset_if_needed() -> None:
    """Direct-stage a bundled-preset's widget values into session_state.

    Runs at the very top of ``_sidebar()`` BEFORE any widget renders.
    Because no widget has instantiated yet, we can write directly to
    ``st.session_state[key]`` without Streamlit's "cannot modify a
    widget-bound key in the same run that creates the widget" error.

    This replaces the older pending+rerun staging dance for bundled
    presets, which depended on a two-rerun bridge (post-dropdown handler
    stashes ``_pending_session_values`` → ``st.rerun()`` → top-of-sidebar
    drain). That bridge was observed to drop a few specific keys
    (``design_lifetime_hours``, ``panel_thickness_mm``) in some live
    sessions even though headless AppTest reproduced the success path,
    likely a rerun-timing edge case. Writing directly here is robust
    against that timing because there is no rerun in between.

    Gate: ``_last_bundled_loaded == choice`` prevents re-staging on
    every rerun — without it, manual edits to staged widgets would be
    overwritten on the next sidebar repaint.
    """
    choice = st.session_state.get("bundled_example_select", _BUNDLED_PLACEHOLDER)
    if choice == _BUNDLED_PLACEHOLDER:
        return
    if st.session_state.get("_last_bundled_loaded") == choice:
        return
    if not _PRESETS_DIR.is_dir():
        return
    # Locate the JSON file whose system_label matches the dropdown choice.
    # Same lookup logic as the dropdown builder — we can't reuse the
    # ``label_to_path`` it constructs because the dropdown hasn't rendered
    # yet in this run.
    for p in sorted(_PRESETS_DIR.glob("*.json")):
        try:
            schema = json_to_session(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (schema.system_label or p.stem) != choice:
            continue
        pending = _session_to_pending_state(schema)
        cat_override = pending.pop("_pending_category_override", None)
        for k, v in pending.items():
            st.session_state[k] = v
        if cat_override is not None:
            st.session_state["category_override"] = cat_override
        st.session_state["_last_bundled_loaded"] = choice
        return


def _render_session_io() -> None:
    """Render the Session I/O expander (download / load JSON).

    Reads every input value from session_state — every form / advanced /
    sensitivity widget is widget-bound and the values are committed by
    the time this expander renders. This avoids passing 11 args through
    the function signature and removes a coupling point that broke in
    the tabs/multi-page rebuild (the form widgets moved out of _sidebar
    and aren't available as locals here)."""
    st.divider()
    with st.expander("Session I/O (download / load JSON)", expanded=False):
        st.caption(
            "Save the current flight envelope to a JSON file, or load "
            "one previously saved. Useful for sharing envelopes, "
            "CI regression snapshots, and the headless CLI "
            "(`python -m matvec run <file.json>`)."
        )

        # ---- Bundled examples ----
        # The presets/ folder ships a curated set of pre-built envelopes
        # (drone, sounding rocket, eVTOL, SR-71, etc). Same JSON format
        # as the file uploader below — this is just a 2-click shortcut
        # for the bundled set so users don't have to file-pick into the
        # project directory. Adding a new bundled example: drop a .json
        # file in presets/ and it appears here on the next Streamlit
        # restart, no code change required.
        bundled_paths = (
            sorted(_PRESETS_DIR.glob("*.json"))
            if _PRESETS_DIR.is_dir() else []
        )
        # Build a {display_label: filepath} map by reading each file's
        # system_label. Failures fall back to the filename stem so a
        # malformed JSON in the folder doesn't break the dropdown.
        label_to_path: dict[str, Path] = {}
        for p in bundled_paths:
            try:
                schema = json_to_session(p.read_text(encoding="utf-8"))
                label = schema.system_label or p.stem
            except Exception:
                label = p.stem
            label_to_path[label] = p
        if label_to_path:
            options = [_BUNDLED_PLACEHOLDER] + sorted(label_to_path.keys())
            choice = st.selectbox(
                "LOAD BUNDLED EXAMPLE",
                options=options,
                index=0,
                key="bundled_example_select",
                help=(
                    "Pre-built calibration envelopes shipped in "
                    "presets/. Pick one and it populates the form just "
                    "like the file uploader below — no need to browse "
                    "the project directory yourself."
                ),
            )
            # Bundled-preset staging happens at the TOP of _sidebar()
            # via _stage_bundled_preset_if_needed() — no post-dropdown
            # pending+rerun handler is needed here. We just show the
            # active example's notes (if any) below the dropdown so the
            # description lives with the selection.
            if choice != _BUNDLED_PLACEHOLDER:
                try:
                    schema = json_to_session(
                        label_to_path[choice].read_text(encoding="utf-8")
                    )
                    if schema.notes:
                        st.caption(schema.notes)
                except Exception:
                    pass

        # ---- Download ----
        # Read live values from session_state. Defaults match
        # _SIDEBAR_DEFAULTS (which has been seeded earlier in _sidebar).
        # Compute current vehicle_category from the override + inference
        # rules — same logic as _render_setup_tab.
        s = st.session_state
        _mach = float(s.get("mach", 1.8))
        _alt  = float(s.get("alt_km", 12.0))
        _mass = float(s.get("mass_kg", 19700.0))
        _cat_override = s.get("category_override", "general")
        if _cat_override == "__auto__":
            vehicle_category = infer_category(_mach, _alt, _mass)
        else:
            vehicle_category = _cat_override
        hot_K = float(s["hot_section_temp_K"]) if vehicle_category == "turbine" and "hot_section_temp_K" in s else None
        session = _build_session_from_widgets(
            mach=_mach, alt_km=_alt, mass_kg=_mass,
            R_n=float(s.get("R_n", 0.30)),
            peak_g=float(s.get("peak_g", 9.0)),
            epsilon=float(s.get("epsilon", 0.85)),
            char_len=float(s.get("char_len", 10.0)),
            flight_duration_s=float(s.get("flight_duration_s", 600.0)),
            design_lifetime_hours=float(s.get("design_lifetime_hours", 1.0)),
            panel_thickness_m=float(s.get("panel_thickness_mm", 2.0)) / 1000.0,
            vehicle_category=vehicle_category,
            hot_section_temp_K=hot_K,
            cost_ceiling_usd=float(s.get("cost_ceiling_usd", 1_000_000.0)),
        )
        slug = _make_slug(session.system_label)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_text = session_to_json(session)
        st.download_button(
            label="⬇  Download session JSON",
            data=json_text.encode("utf-8"),
            file_name=f"matvec_session_{slug}_{ts}.json",
            mime="application/json",
            key="dl_session_json",
            use_container_width=True,
        )

        # ---- Upload ----
        uploaded = st.file_uploader(
            "Load session JSON",
            type=["json"],
            key="session_json_upload",
            help=(
                "Pick a file previously saved from this sidebar or produced "
                "by the CLI. Values populate every widget above."
            ),
        )
        # Guard against re-processing the same upload on every rerun by
        # remembering the last filename+size we successfully loaded.
        # Streamlit keeps the file object in session_state until the user
        # clicks the 'x', so without this guard the sidebar would rerun
        # the staging write on every keystroke in an unrelated widget.
        if uploaded is not None:
            fingerprint = (uploaded.name, uploaded.size)
            if st.session_state.get("_last_uploaded_session") != fingerprint:
                try:
                    payload = uploaded.getvalue().decode("utf-8")
                    loaded = json_to_session(payload)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to parse JSON: {exc}")
                else:
                    # Same single-rerun pattern as the bundled-example
                    # loader — pop the override key out and write it
                    # directly so the drain at the top of _sidebar()
                    # catches it on the same rerun the envelope drains.
                    pending_up = _session_to_pending_state(loaded)
                    cat_override_up = pending_up.pop(
                        "_pending_category_override", None
                    )
                    st.session_state["_pending_session_values"] = pending_up
                    if cat_override_up is not None:
                        st.session_state["_pending_category_override"] = (
                            cat_override_up
                        )
                    label_up = pending_up.get("system_label")
                    if label_up:
                        st.session_state["system_label"] = label_up
                    st.session_state["_last_uploaded_session"] = fingerprint
                    st.success(
                        f"Loaded '{loaded.system_label}' "
                        f"({loaded.vehicle_category}). Reloading…"
                    )
                    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Widget defaults — seeded into st.session_state once per session.
#
# Why: every form widget binds via key= to a session_state slot. The preset
# loader and the session-JSON uploader stage values into _pending_*
# staging keys, drained at the top of _sidebar() into the live widget
# slots. This drain happens BEFORE the widgets render — Streamlit then
# warns when a widget also passes value= because both paths set the
# initial value, and Streamlit can't tell which the developer intended.
#
# The fix: never pass value= on widgets that participate in the staging
# pipeline. Use setdefault() once at the top of _sidebar() to seed the
# first-run default; from then on session_state holds the live value
# and the widget reads it via key= alone.
# ---------------------------------------------------------------------------
_SIDEBAR_DEFAULTS: dict = {
    # Form (rendered in _render_setup_tab on the main canvas)
    "vel_mode":            "Mach",
    "alt_km":              12.0,
    "mach":                1.8,
    "speed_kmh":           1944.0,
    "mass_kg":             19700.0,
    "R_n":                 0.30,
    "peak_g":              9.0,
    "hot_section_temp_K":  1400.0,
    # Default to General mode (no category-specific exclusions or
    # ranking penalties). Users can switch to a specific category to
    # opt into its behaviour, or to Auto for heuristic inference.
    # General-as-default avoids the misclassification failure mode
    # where the Auto heuristic picked hypersonic_missile for boost-
    # coast vehicles and forced sustained-flight assumptions onto
    # them.
    "category_override":   "general",
    # Sidebar — advanced expander
    "epsilon":             0.85,
    "char_len":            10.0,
    "flight_duration_s":   600.0,
    # Design lifetime default 1.0 h preserves pre-creep-feature
    # behaviour (creep evaluation is a no-op at single-flight
    # lifetimes for nearly all materials). Bundled presets override
    # this on load to their per-vehicle values (SR-71 → 3000,
    # Concorde → 25000, CFM56 → 25000).
    "design_lifetime_hours": 1.0,
    # Panel thickness default 2.0 mm models a typical aerospace
    # thin-skin panel. Bundled presets override on load (SR-71 1.5,
    # Concorde 2.0, Sounding Rocket 3.0, Turbine 1.0). Widget-
    # bound in millimetres for usability; converted to metres at
    # the SessionSchema boundary.
    "panel_thickness_mm":  2.0,
    "cost_ceiling_usd":    1_000_000.0,
    # Sidebar — sensitivity expander
    "run_sensitivity":     False,
    "sens_mach_delta_pct": 10,
    "sens_mass_delta_pct": 15,
    "sens_Rn_delta_pct":   20,
    "sens_g_delta_pct":    25,
    "sens_n_samples":      11,
}


def _seed_session_state_defaults() -> None:
    """Seed every widget-bound session_state slot with its default value.
    Called once at the top of _sidebar() before any widget renders."""
    for key, default in _SIDEBAR_DEFAULTS.items():
        st.session_state.setdefault(key, default)


# ---------------------------------------------------------------------------
# Main-canvas Flight Envelope form (rendered in the Setup tab)
# ---------------------------------------------------------------------------

def _render_setup_tab() -> dict:
    """Render the Flight Envelope form on the main canvas.

    Layout:
      Row 1 — Vehicle category override widget (full width). Defaults to
              "(Auto-detect)"; selecting a specific category pins a
              sticky override.
      Row 2 — Four horizontal metric chips: Speed of sound | Recovery
              T_wall | Dynamic pressure | Flight regime.
      Row 3 — Three-column input form:
                Velocity & altitude  | Geometry        | Loads & conditions
                (vel mode / alt /    | (mass / R_n)    | (peak g / hot-section
                 mach OR speed)      |                 |  if turbine)

    Returns a dict with keys: mach, alt_km, mass_kg, R_n, peak_g,
    vehicle_category, hot_section_temp_K, vel_mode. main() merges
    this with the sidebar's secondary inputs.
    """
    # Pre-resolve the effective category from the prior committed
    # session_state values so the "Auto: X" label and the conditional
    # hot-section input render correctly on the FIRST pass of this
    # rerun (before the form widgets re-bind their values).
    s = st.session_state
    _mach_prev = float(s.get("mach", 1.8))
    _alt_prev  = float(s.get("alt_km", 12.0))
    _mass_prev = float(s.get("mass_kg", 19700.0))
    _cat_override = s.get("category_override", "general")
    _inferred_cat_pre = infer_category(_mach_prev, _alt_prev, _mass_prev)
    vehicle_category = (
        _inferred_cat_pre if _cat_override == "__auto__" else _cat_override
    )

    # ── Row 1: VEHICLE CATEGORY override ──
    cat_options = ["__auto__"] + list(VEHICLE_CATEGORIES.keys())

    def _fmt_cat(opt: str) -> str:
        if opt == "__auto__":
            return f"Auto: {VEHICLE_CATEGORIES[_inferred_cat_pre]}"
        return VEHICLE_CATEGORIES[opt]

    st.selectbox(
        "VEHICLE CATEGORY",
        options=cat_options,
        format_func=_fmt_cat,
        key="category_override",
        help=(
            "Defaults to General (no category-specific exclusions or "
            "ranking penalties). Pick a specific category to apply that "
            "category's exclusions and ranking — e.g., Aircraft for "
            "specific-strength sorting, Turbine to opt into the hot-"
            "section temperature input. Auto uses a Mach + altitude + "
            "mass heuristic that can be wrong; explicit is safer."
        ),
    )
    st.markdown(
        f'<p style="font-size:11px;color:#8b949e;margin-top:-4px">'
        f'{CATEGORY_DESCRIPTIONS[vehicle_category]}</p>',
        unsafe_allow_html=True,
    )

    # ── Row 2: live envelope summary metrics (chip strip) ──
    # Turbine analysis is special: the flight envelope (Mach, altitude)
    # is a placeholder — there's no freestream flow at a turbine blade.
    # The binding parameter is the hot-section metal-face temperature,
    # which the materials pipeline applies via apply_turbine_override.
    # Showing freestream-derived metrics here is misleading (e.g. it'd
    # report Recovery T_wall = 300 K when the pipeline actually uses
    # 1100 K from the override). So for category=turbine we display a
    # turbine-specific strip with the override + creep-derate context.
    if vehicle_category == "turbine":
        _hot_K = float(s.get("hot_section_temp_K", 1400.0))
        m1, m2, m3 = st.columns([1.2, 1.0, 1.0])
        with m1:
            st.metric(
                "Hot-section metal-face T",
                f"{_hot_K:.0f} K",
                help=(
                    "Turbine blade metal-face temperature. This is the "
                    "binding parameter for material selection — it "
                    "overrides the aerodynamic recovery temperature via "
                    "apply_turbine_override. Modern cooled HPT blades: "
                    "1350–1500 K. Uncooled small turbojets: 1100–1250 K."
                ),
            )
        with m2:
            st.metric(
                "Strength derate",
                "0.4× (creep proxy)",
                help=(
                    "All material strengths are derated to 40% of their "
                    "room-temperature yield as a proxy for sustained "
                    "high-temperature creep loading. Aluminium, polymer "
                    "composites, and general-engineering materials are "
                    "filtered out for the turbine category."
                ),
            )
        with m3:
            st.metric(
                "Reference frame",
                "Compressor exit",
                help=(
                    "Mach 0.5 / sea-level placeholder — there is no "
                    "freestream flow at a turbine blade. Mass and "
                    "characteristic length are scaled to drive the "
                    "structural / thermal envelope; the binding number "
                    "is the hot-section temperature above."
                ),
            )
    else:
        # Non-turbine (default) metric strip — the freestream envelope
        # quantities are operationally meaningful here.
        # Pass R_n + epsilon so envelope_summary uses the same T_wall
        # model as run_analysis (Sutton-Graves at M≥5 instead of the
        # recovery formula, which over-predicts by ~10x at hypersonic).
        _R_n_prev    = float(s.get("R_n", 0.30))
        _epsilon_prev = float(s.get("epsilon", 0.85))
        summary_pre = envelope_summary(
            _mach_prev, _alt_prev,
            R_n_m=_R_n_prev, epsilon=_epsilon_prev,
        )
        # Wall-temperature label flips between "Recovery T_wall" and
        # "Wall T (SG model)" depending on which model produced the
        # value. This avoids the previous bug where the recovery
        # formula produced 18,000 K for M=22 reentry capsules.
        if summary_pre["T_wall_model"] == "recovery":
            twall_label = "Recovery T_wall"
        else:
            twall_label = "Wall T (SG model)"
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                f"Speed of sound at {_alt_prev:.1f} km",
                f"{summary_pre['a_ms']:.0f} m/s",
                help=(
                    "ISA standard-atmosphere speed of sound at the current "
                    "altitude. a = √(γRT); T drops with altitude in the "
                    "troposphere, then rises in the stratosphere — that's "
                    "why this number changes between presets."
                ),
            )
        with m2:
            st.metric(
                twall_label,
                f"{summary_pre['T_wall_K']:.0f} K",
                help=(
                    "Predicted leading-edge / nose-tip wall temperature. "
                    "M < 5: turbulent flat-plate recovery temperature. "
                    "M ≥ 5: Sutton-Graves convective + Tauber-Sutton radiative "
                    "heat flux balanced by surface re-radiation."
                ),
            )
        with m3:
            st.metric(
                "Dynamic pressure",
                f"{summary_pre['q_inf_Pa']:,.0f} Pa",
                help="Freestream dynamic pressure q∞ = ½ρV². Drives aerodynamic loads.",
            )
        with m4:
            st.metric(
                "Flight regime",
                summary_pre["flight_regime"].capitalize(),
            )

    st.divider()

    # ── Row 3: 3-column input form ──
    col_vel, col_geom, col_loads = st.columns(3)

    with col_vel:
        st.markdown(
            '<p style="font-size:11px;font-weight:600;color:#58a6ff;'
            'text-transform:uppercase;letter-spacing:0.08em;margin:0 0 0.5rem">'
            'Velocity &amp; altitude</p>',
            unsafe_allow_html=True,
        )
        vel_mode = st.radio(
            "vel_mode_radio",
            ["Mach", "km/h"],
            horizontal=True,
            key="vel_mode",
            label_visibility="collapsed",
        )
        alt_km = st.number_input(
            "CRUISE ALTITUDE (km)",
            min_value=0.0, max_value=86.0, step=0.5, format="%.1f",
            key="alt_km",
        )
        a_ms = _isa_speed_of_sound_ms(alt_km)
        if vel_mode == "Mach":
            mach = st.number_input(
                "MACH NUMBER",
                min_value=0.05, max_value=25.0, step=0.05, format="%.2f",
                key="mach",
            )
            v_kmh = mach * a_ms * 3.6
            st.markdown(
                f'<div class="computed-metric">≡ <span>{v_kmh:,.0f} km/h</span> at {alt_km:.1f} km</div>',
                unsafe_allow_html=True,
            )
        else:
            speed_kmh = st.number_input(
                "VELOCITY (km/h)",
                min_value=18.0, max_value=30000.0, step=10.0, format="%.0f",
                key="speed_kmh",
            )
            mach = (speed_kmh / 3.6) / a_ms
            st.markdown(
                f'<div class="computed-metric">≡ Mach <span>{mach:.3f}</span> at {alt_km:.1f} km</div>',
                unsafe_allow_html=True,
            )

    with col_geom:
        st.markdown(
            '<p style="font-size:11px;font-weight:600;color:#58a6ff;'
            'text-transform:uppercase;letter-spacing:0.08em;margin:0 0 0.5rem">'
            'Geometry</p>',
            unsafe_allow_html=True,
        )
        mass_kg = st.number_input(
            "VEHICLE MASS (kg)",
            # Min lowered to 0.1 to accept the new low-mass presets
            # (Consumer FPV Drone is 0.7 kg, Consumer Quadcopter 1 kg).
            # Step stays at 100 since most realistic envelopes are >>1 kg.
            min_value=0.1, max_value=1_000_000.0, step=100.0, format="%.1f",
            key="mass_kg",
        )
        R_n = st.number_input(
            "LEADING EDGE RADIUS (m)",
            min_value=0.001, max_value=20.0, step=0.005, format="%.3f",
            key="R_n",
            help=(
                "Radius of curvature at the stagnation point. "
                "Blunt bodies (capsules, rounded noses): 0.5–5 m. "
                "Fighter jets and moderate shapes: 0.1–0.5 m. "
                "Sharp hypersonic leading edges: 0.01–0.1 m."
            ),
        )

    with col_loads:
        st.markdown(
            '<p style="font-size:11px;font-weight:600;color:#58a6ff;'
            'text-transform:uppercase;letter-spacing:0.08em;margin:0 0 0.5rem">'
            'Loads &amp; conditions</p>',
            unsafe_allow_html=True,
        )
        peak_g = st.number_input(
            "PEAK STRUCTURAL LOAD (g)",
            min_value=0.1, max_value=30.0, step=0.5, format="%.1f",
            key="peak_g",
        )
        hot_section_temp_K: float | None = None
        if vehicle_category == "turbine":
            hot_section_temp_K = st.number_input(
                "HOT-SECTION TEMPERATURE (K)",
                min_value=500.0, max_value=2500.0, step=25.0, format="%.0f",
                key="hot_section_temp_K",
                help=(
                    "Turbine blade metal-face temperature (TIT minus film-cooling delta). "
                    "Modern cooled HPT blades: 1350–1500 K. "
                    "Uncooled industrial turbines: 1100–1250 K. "
                    "Research / next-gen cooled: up to ~1700 K."
                ),
            )

    return {
        "mach": mach,
        "alt_km": alt_km,
        "mass_kg": mass_kg,
        "R_n": R_n,
        "peak_g": peak_g,
        "vehicle_category": vehicle_category,
        "hot_section_temp_K": hot_section_temp_K,
        "vel_mode": vel_mode,
    }


# ---------------------------------------------------------------------------
# Persistent envelope-summary chip (rendered above the tabs)
# ---------------------------------------------------------------------------

def _render_envelope_chip(physics, vehicle_category: str) -> None:
    """One-line summary chip rendered at the top of every page (above the
    tabs) so the engineer always sees what scenario the results belong
    to, even from the Trade-offs tab."""
    regime = physics.flight_regime
    cat_label = VEHICLE_CATEGORIES.get(vehicle_category, vehicle_category)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:1rem;'
        f'margin:0 0 0.6rem;padding:6px 12px;background:#161b22;'
        f'border:1px solid #30363d;border-radius:6px">'
        f'<span class="regime-badge regime-{regime}">{regime.upper()}</span>'
        f'<span style="font-size:14px;font-weight:600;color:#c9d1d9">'
        f'Mach {physics.peak_mach:.2f} &nbsp;|&nbsp; '
        f'{physics.cruise_altitude_km:.1f} km &nbsp;|&nbsp; '
        f'{physics.vehicle_mass_kg:,.0f} kg &nbsp;|&nbsp; '
        f'LE radius = {physics.nose_radius_m:.3f} m &nbsp;|&nbsp; '
        f'{cat_label}'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _sidebar() -> dict:
    # -----------------------------------------------------------------------
    # Direct-stage bundled-preset widget values BEFORE any widget renders.
    # Replaces the older post-dropdown pending+rerun handler for bundled
    # presets — see _stage_bundled_preset_if_needed() docstring. JSON-file
    # uploads still use the pending+rerun bridge below because that path
    # runs AFTER widgets render and therefore can't avoid the rerun.
    # -----------------------------------------------------------------------
    _stage_bundled_preset_if_needed()

    # -----------------------------------------------------------------------
    # Apply any category-override change queued by the preset loader or by
    # the session-JSON uploader BEFORE the category_override widget
    # instantiates on the main canvas. Streamlit forbids modifying a
    # widget-bound key in the same run that creates the widget, so the
    # staging key "_pending_category_override" bridges across reruns.
    # -----------------------------------------------------------------------
    if "_pending_category_override" in st.session_state:
        st.session_state["category_override"] = st.session_state.pop(
            "_pending_category_override"
        )

    # Backward compat: legacy session JSONs (and any in-flight uploads
    # from a pre-rebuild client) may still write the old staging key
    # "_pending_vehicle_category". Translate to the new override slot
    # so existing JSON files load without surprise.
    if "_pending_vehicle_category" in st.session_state:
        st.session_state["category_override"] = st.session_state.pop(
            "_pending_vehicle_category"
        )

    # -----------------------------------------------------------------------
    # Apply any envelope values queued by the "Load Session JSON" uploader
    # (Deliverable D). The uploader runs AFTER the widgets in the current
    # rerun, so it stashes a dict of {widget_key: value} under
    # "_pending_session_values" and triggers a rerun; here, on the next
    # rerun, we drain that dict into session_state BEFORE the widgets
    # re-instantiate. This is the same staging pattern as the preset
    # loader — Streamlit forbids writing to a widget-bound key mid-run,
    # so we bridge across reruns.
    # -----------------------------------------------------------------------
    if "_pending_session_values" in st.session_state:
        pending = st.session_state.pop("_pending_session_values")
        for k, v in pending.items():
            st.session_state[k] = v

    # Seed first-run defaults into session_state. Every widget below
    # binds via key= alone (no value= argument) so that staged values
    # from the preset loader / session-JSON uploader take effect
    # without triggering Streamlit's "default + session_state" warning.
    _seed_session_state_defaults()

    st.sidebar.markdown('<p style="font-size:18px;font-weight:700;color:#58a6ff;margin:0">MATVEC</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<p style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.75rem">Materials Feasibility Tool</p>', unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # SIDEBAR (slim): Session I/O → Advanced → Sensitivity.
    #
    # The legacy "calibration preset" dropdown that used to live here
    # was removed when bundled examples moved to JSON files in the
    # presets/ folder. Loading a bundled example is now done via the
    # "Load bundled example" dropdown inside the Session I/O expander
    # below — same staging chain, identical downstream behaviour.
    #
    # The flight-envelope inputs and the vehicle-category control live
    # on the main canvas inside the Setup tab.
    # ------------------------------------------------------------------
    session_io_container = st.sidebar.container()
    advanced_container = st.sidebar.container()

    # ------------------------------------------------------------------
    # ADVANCED — secondary parameters in an expander at the very bottom.
    # ------------------------------------------------------------------
    with advanced_container:
        st.divider()
        with st.expander("ADVANCED PARAMETERS"):
            # Default values for these widgets are seeded into
            # session_state via _seed_session_state_defaults() at the
            # top of _sidebar(); we don't pass value= here so Streamlit
            # doesn't warn about "default + session_state" collision.
            epsilon = st.number_input(
                "WALL EMISSIVITY (ε)",
                min_value=0.05, max_value=1.0, step=0.01, format="%.2f",
                key="epsilon",
            )
            char_len = st.number_input(
                "CHARACTERISTIC LENGTH (m)",
                min_value=0.1, max_value=200.0, step=0.5, format="%.1f",
                key="char_len",
            )
            # Flight duration is also an advanced knob. Default 600 s (10 min)
            # mirrors physics_engine.run_analysis's default. Exposed here so
            # the Session JSON download/upload flow (Deliverable D) can
            # round-trip non-default durations — otherwise a user who loaded
            # a JSON with a 1800-s cruise would silently get clipped to 600.
            flight_duration_s = st.number_input(
                "FLIGHT DURATION (s)",
                min_value=1.0, max_value=86400.0, step=10.0, format="%.0f",
                key="flight_duration_s",
                help=(
                    "Mission duration. Drives the peak-power × duration scaffold "
                    "in §6; does not currently affect the thermal branch."
                ),
            )
            # Design lifetime (creep / lifecycle feature). Distinct from
            # flight_duration_s: this is total airframe / component
            # service life in hours, which drives the creep-evaluation
            # stage in the matching engine. SR-71 ~3,000 h; Concorde
            # ~25,000 h; CFM56 HPT blade overhaul ~25,000 h. Default
            # 1.0 h is single-flight, which makes creep evaluation
            # essentially a no-op (matches pre-feature behaviour).
            design_lifetime_hours = st.number_input(
                "DESIGN LIFETIME (h)",
                min_value=0.1, max_value=200_000.0, step=100.0, format="%.0f",
                key="design_lifetime_hours",
                help=(
                    "Total airframe / component service life in hours. "
                    "Drives the creep-rupture screening: at long lifetimes "
                    "(thousands of hours) materials must survive sustained "
                    "(T_wall, σ_req) without creeping out of tolerance. "
                    "Default 1 h is single-flight (creep no-op); set to "
                    "the bundled-preset value or your own design target."
                ),
            )
            # Panel thickness (Phase 7 transient-heat feature). Drives the
            # 1D heat-conduction integration through the panel; the
            # back-face peak temperature is the operative screening
            # value for short-duration boost-coast flights. Default
            # 2 mm models a typical aerospace thin-skin panel; per-
            # preset values override this on load (SR-71 1.5 mm,
            # Concorde 2 mm, Sounding Rocket 3 mm, Turbine 1 mm).
            panel_thickness_mm = st.number_input(
                "PANEL THICKNESS (mm)",
                min_value=0.1, max_value=50.0, step=0.1, format="%.1f",
                key="panel_thickness_mm",
                help=(
                    "Representative through-thickness gauge for the "
                    "1D transient heat solver. Typical aerospace thin-"
                    "skin panels are 1-3 mm. The solver integrates "
                    "convective heat flux from the surface through "
                    "this thickness to an insulated back face; the "
                    "back-face peak temperature drives the transient "
                    "thermal screen for short-duration flights."
                ),
            )
            # Material budget ceiling (Cost-Axis-on-Pareto-Front feature).
            # Carried on session.options (not a SessionSchema first-class
            # field) because it is a reporting parameter — it scales the
            # cost objective on the Pareto front and bolds rows in the
            # materials table, but does not change physics or matching.
            # Default $1M is a reasonable concept-study budget; min $10k
            # avoids divide-by-near-zero in the Pareto cost-penalty term.
            cost_ceiling_usd = st.number_input(
                "MATERIAL BUDGET CEILING (USD)",
                min_value=10_000.0, max_value=1_000_000_000.0,
                step=50_000.0, format="%.0f",
                key="cost_ceiling_usd",
                help=(
                    "Total material budget for the vehicle. Materials whose "
                    "estimated cost (cost/kg × vehicle mass) exceeds this "
                    "ceiling are flagged in red in the table below and "
                    "penalised on the Pareto front cost axis. Order-of-"
                    "magnitude estimates only — see Section 6 in the PDF."
                ),
            )

        # ------------------------------------------------------------------
        # SENSITIVITY ANALYSIS — opt-in expander (Uncertainty & Sensitivity).
        # The whole feature is pay-as-you-go: a fresh session does NOT pay
        # the 4 × n_samples sweep cost unless the user ticks the checkbox.
        # The deltas live next to the checkbox so the user can read the
        # uncertainty band that gets reported on the §N tornado.
        # ------------------------------------------------------------------
        with st.expander("SENSITIVITY ANALYSIS"):
            st.caption(
                "Sweep each input ±delta and count how often each viable "
                "material survives. Robust picks survive ≥90% of scenarios; "
                "knife-edge picks fail in >50%."
            )
            run_sensitivity_flag = st.checkbox(
                "Run sensitivity analysis",
                key="run_sensitivity",
                help=(
                    "Ticks on a 4-input × N-sample sweep (default 44 "
                    "scenarios). Adds ~2 s to the analysis. Renders a new "
                    "section below the materials table with a robustness "
                    "label per material and a tornado chart."
                ),
            )
            sens_mach_delta = st.slider(
                "Mach Δ (±%)",
                min_value=1, max_value=30, step=1,
                key="sens_mach_delta_pct",
                help="Sweep ±X% around nominal Mach.",
            )
            sens_mass_delta = st.slider(
                "Mass Δ (±%)",
                min_value=1, max_value=40, step=1,
                key="sens_mass_delta_pct",
            )
            sens_Rn_delta = st.slider(
                "Nose radius Δ (±%)",
                min_value=1, max_value=40, step=1,
                key="sens_Rn_delta_pct",
            )
            sens_g_delta = st.slider(
                "g-load Δ (±%)",
                min_value=1, max_value=50, step=1,
                key="sens_g_delta_pct",
            )
            sens_n_samples = st.slider(
                "Samples per input",
                min_value=3, max_value=21, step=2,
                key="sens_n_samples",
                help=(
                    "Number of sweep points per input (including endpoints). "
                    "Total scenarios = 4 × this value. Higher = smoother "
                    "tornado, slower run."
                ),
            )

    # ------------------------------------------------------------------
    # SESSION JSON I/O — Download / Load a canonical SessionSchema.
    # Rendered AFTER the inputs so session_state contains the current
    # widget values, which become the payload of the download.
    # ------------------------------------------------------------------
    with session_io_container:
        # _render_session_io now reads everything from session_state —
        # the form values live in session_state because the form
        # widgets in _render_setup_tab() bind via key=.
        _render_session_io()

    # Sidebar returns ONLY the secondary inputs (advanced + sensitivity).
    # Form values (mach, alt_km, mass_kg, R_n, peak_g, vehicle_category,
    # hot_section_temp_K) are returned by _render_setup_tab() and merged
    # into the full inputs dict in main().
    return dict(
        epsilon=epsilon,
        char_len=char_len,
        flight_duration_s=flight_duration_s,
        design_lifetime_hours=design_lifetime_hours,
        # Panel thickness is widget-bound in millimetres for usability;
        # convert to metres at the boundary so the matching pipeline /
        # solver sees SI-consistent units like the rest of the schema.
        panel_thickness_m=float(panel_thickness_mm) / 1000.0,
        cost_ceiling_usd=cost_ceiling_usd,
        # Sensitivity-analysis controls — opt-in flag + per-input deltas.
        # Carried through to main() which decides whether to call
        # run_sensitivity (the cost is borne only when the box is ticked).
        run_sensitivity=run_sensitivity_flag,
        sens_mach_delta_frac=sens_mach_delta / 100.0,
        sens_mass_delta_frac=sens_mass_delta / 100.0,
        sens_Rn_delta_frac=sens_Rn_delta / 100.0,
        sens_g_delta_frac=sens_g_delta / 100.0,
        sens_n_samples=int(sens_n_samples),
    )


# ---------------------------------------------------------------------------
# Physics cards
# ---------------------------------------------------------------------------

# Per-category disclosure messages: what the active category implies
# for the matching engine's exclusions and ranking. Surfaced as a small
# badge on the Results tab so users can see the consequence of their
# selection without reading the matching-engine source.
_CATEGORY_MODE_MESSAGES: dict[str, str] = {
    "general": (
        "<strong>General mode</strong> — no category-specific exclusions "
        "or ranking penalties. All compatible materials evaluated equally."
    ),
    "aircraft": (
        "<strong>Aircraft mode</strong> — polymer composites excluded "
        "above Mach 2; viable list sorted by specific strength "
        "(mass-driven, favours titanium over Inconel)."
    ),
    "hypersonic_aircraft": (
        "<strong>Hypersonic aircraft mode</strong> — slender-body plasma "
        "threshold (Mach > 6); polymer composites excluded; specific-"
        "strength ranking applied."
    ),
    "hypersonic_missile": (
        "<strong>Hypersonic missile mode</strong> — slender-body plasma "
        "threshold (Mach > 6); polymer composites excluded; specific-"
        "strength ranking applied."
    ),
    "reentry": (
        "<strong>Reentry mode</strong> — UHTCs and refractory metals "
        "unlocked; high-density alloys allowed; substrate-mode evaluation "
        "when T_wall &ge; 1200 K."
    ),
    "turbine": (
        "<strong>Turbine mode</strong> — aluminum and polymers excluded; "
        "T_wall driven by hot-section temperature input, not freestream."
    ),
}


def _show_category_mode_badge(vehicle_category: str) -> None:
    """One-line disclosure of what the active category does to the
    analysis. Renders just above the Physics Analysis section on the
    Results tab. Returns silently for unknown categories."""
    msg = _CATEGORY_MODE_MESSAGES.get(vehicle_category)
    if not msg:
        return
    st.markdown(
        f'<div style="border-left:3px solid #58a6ff;padding:6px 10px;'
        f'margin:0 0 8px;background:#161b22;font-size:12px;color:#c9d1d9;'
        f'border-radius:0 4px 4px 0">{msg}</div>',
        unsafe_allow_html=True,
    )


def _show_physics(physics, vehicle_category: str = "general") -> None:
    th     = physics.thermal
    struct = physics.structural
    prop   = physics.propulsion
    em     = physics.em
    atm    = physics.atmosphere

    col_left, col_right = st.columns(2)

    with col_left:
        # Thermal card — two branches based on which model was used
        uses_recovery = th.uses_recovery_model
        body = ""
        if uses_recovery:
            # Recovery temperature model (M < 5): heat flux rows are meaningless (zero)
            uncertainty_pct = "±15%"
            body += _card_row("T_wall range",          f"{th.T_wall_min_K:.0f} – {th.T_wall_max_K:.0f}", f"K  ({uncertainty_pct})")
            body += _card_row("Total stagnation temp",  f"{th.T_stag_K:.0f}",                              "K")
            body += _card_row("Sea-level worst case",   f"{th.T_wall_sealevel_K:.0f}",                     "K")
            thermal_badge = '<span class="card-badge badge-ok">Recovery temp model</span>'
        else:
            # Sutton-Graves model (M >= 5)
            uncertainty_pct = "±20%"
            body += _card_row("Total heat flux",        f"{th.q_total_Wm2 / 1e6:.4f}",                    "MW/m²")
            body += _card_row("↳ Convective (SG)",      f"{th.q_conv_Wm2 / 1e6:.4f}",                     "MW/m²")
            body += _card_row("↳ Radiative (TS)",       f"{th.q_rad_Wm2 / 1e6:.4f}",                      "MW/m²")
            body += _card_row("T_wall range",           f"{th.T_wall_min_K:.0f} – {th.T_wall_max_K:.0f}", f"K  ({uncertainty_pct})")
            body += _card_row("Total stagnation temp",  f"{th.T_stag_K:.0f}",                              "K")
            body += _card_row("Sea-level worst case",   f"{th.T_wall_sealevel_K:.0f}",                     "K")
            thermal_badge = '<span class="card-badge badge-ok">SG radiation-equilibrium</span>'
        st.markdown(
            _instrument_card("Wall Temperature", f"{th.T_wall_K:.0f}", "K", body, thermal_badge),
            unsafe_allow_html=True,
        )

        # Thermal limitation note for winged vehicles
        if vehicle_category in ("aircraft", "hypersonic_aircraft", "hypersonic_missile"):
            st.markdown(
                '<div style="border-left:3px solid #58a6ff;padding:8px 12px;margin:4px 0 12px;'
                'background:#161b22;font-size:13px;color:#8b949e;border-radius:0 4px 4px 0">'
                '<strong style="color:#58a6ff">ℹ Stagnation-point note</strong><br>'
                'T<sub>wall</sub> is the worst-case nose tip / leading edge temperature. '
                'Fuselage bulk skin temperatures are typically 60–80% of T<sub>wall</sub> '
                'for supersonic cruise vehicles. Material recommendations are conservative '
                'for primary airframe structure away from the nose.</div>',
                unsafe_allow_html=True,
            )

        # Ablative-coating advisory: physics-driven, surfaces whenever T_wall
        # exceeds the thermal ceiling of standard structural metals (~1200 K).
        if th.T_wall_K >= TPS_UNLOCK_TEMP_K:
            T_soak_display = max(physics.thermal.T_ambient_K, 400.0)
            st.markdown(
                '<div style="border-left:3px solid #f0883e;padding:8px 12px;margin:4px 0 12px;'
                'background:#161b22;font-size:13px;color:#c9d1d9;border-radius:0 4px 4px 0">'
                '<strong style="color:#f0883e">🔥 Ablative coating recommended</strong><br>'
                f'External T<sub>wall</sub> = {th.T_wall_K:.0f} K exceeds the thermal ceiling '
                'of standard structural metals (~1200 K). Materials below include both '
                '<strong>ablative hot-face options</strong> (PICA, AVCOAT, etc.) and '
                '<strong>metallic substructure candidates</strong> evaluated at the '
                f'backside soak temperature T<sub>soak</sub> ≈ {T_soak_display:.0f} K. '
                'Substrate candidates are tagged with a <em>substrate</em> badge.</div>',
                unsafe_allow_html=True,
            )

        # Propulsion card — fuel rows filtered by vehicle category and Mach regime
        body = ""
        body += _card_row("Peak power",   f"{prop.P_peak_W / 1e9:.3f}",  "GW")
        body += _card_row("Total energy", f"{prop.E_total_J / 1e12:.4f}", "TJ")

        # Fuel reference — branch by category and (for aircraft) by Mach regime.
        # Hypersonic airframes (M >= 5) cannot use air-breathing turbines; show
        # rocket propellant reference instead of JP-7.
        is_hypersonic_airframe = (
            vehicle_category == "aircraft" and physics.peak_mach >= 5.0
        )

        if vehicle_category == "reentry":
            # Reentry capsules decelerate aerodynamically — no propulsion fuel.
            body += _card_row("Propulsion fuel", "none", "(aerodynamic deceleration)")
        elif is_hypersonic_airframe:
            # X-15-class: Reaction Motors XLR99 used anhydrous NH3 + LOX.
            # ~8 MJ/kg of combined propellant (NH3 LHV 18.6 MJ/kg, O2/NH3 ~1.41).
            # Use KE_J (kinetic energy to deliver) rather than E_total_J (which
            # assumes sustained thrust over full duration) — rockets accelerate
            # and coast, they don't burn for the full cruise segment.
            nh3_lox_mass = prop.KE_J / 8.0e6
            body += _card_row("Ammonia + LOX (rocket, KE basis)", f"{nh3_lox_mass:.0f}", "kg")
        else:
            body += _card_row("Kerosene/JP-7 (ref.)", f"{prop.fuel_mass_kerosene_kg:.0f}", "kg")

        if vehicle_category == "hypersonic_missile":
            htpb_mass = prop.E_total_J / 5.0e6
            body += _card_row("Solid rocket (HTPB ~5 MJ/kg)", f"{htpb_mass:.0f}", "kg")
        elif vehicle_category not in ("aircraft", "hypersonic_missile", "reentry"):
            # general/turbine/hypersonic_aircraft (legacy) — LH₂ as energetic analogue
            body += _card_row("LH₂ equivalent", f"{prop.fuel_mass_LH2_kg:.0f}", "kg")
        st.markdown(
            _instrument_card("Kinetic Energy", f"{prop.KE_J / 1e9:.3f}", "GJ", body),
            unsafe_allow_html=True,
        )

    with col_right:
        # Structural card — three components before combined
        q_dyn_contrib = struct.q_dyn_Pa / 1e6   # dynamic pressure in MPa (raw, before safety factor)
        body = ""
        body += _card_row("1. Inertial stress",  f"{struct.sigma_inertial_MPa:.1f}", "MPa  (raw)")
        body += _card_row("2. Dynamic pressure", f"{q_dyn_contrib:.2f}",            "MPa  (raw, q_dyn / A_ref)")
        body += _card_row(f"  × safety factor",  f"× {_SAFETY_FACTOR}  →  {struct.sigma_combined_MPa:.1f}", "MPa")
        body += _card_row("3. Thermal ref (σ_th)", f"{struct.sigma_thermal_ref_MPa:.1f}", "MPa  (E=200 GPa, α=12e-6)")
        body += _card_row("Dynamic pressure",    f"{struct.q_dyn_Pa / 1000:.2f}",   "kPa")
        st.markdown(
            _instrument_card("Tensile Requirement", f"{struct.sigma_tensile_required_MPa:.0f}", "MPa", body),
            unsafe_allow_html=True,
        )

        # EM card
        plasma_badge = (
            '<span class="card-badge badge-warn">Plasma sheath — RF blackout</span>'
            if em.plasma_sheath else ""
        )
        body = ""
        body += _card_row("Peak wavelength", f"{em.lambda_peak_um:.3f}", "µm")
        body += _card_row("Emission band",   em.emission_band, "")
        body += _card_row("Plasma sheath",   "YES" if em.plasma_sheath else "no", "")
        st.markdown(
            _instrument_card("Radiated Power", f"{em.P_rad_W / 1e3:.2f}", "kW", body, plasma_badge),
            unsafe_allow_html=True,
        )

    # Atmospheric conditions panel
    st.markdown(f"""
<div class="atm-panel">
  <div class="atm-title">ISA Atmospheric Conditions — {atm.altitude_km:.1f} km</div>
  <div style="display:flex;gap:2.5rem;flex-wrap:wrap">
    <div class="atm-row">Temperature&nbsp;<span class="atm-val">{atm.temperature_K:.2f} K</span>&nbsp;({atm.temperature_K - 273.15:.1f} °C)</div>
    <div class="atm-row">Pressure&nbsp;<span class="atm-val">{atm.pressure_Pa:.0f} Pa</span>&nbsp;({atm.pressure_Pa / 101325:.4f} atm)</div>
    <div class="atm-row">Density&nbsp;<span class="atm-val">{atm.density_kgm3:.5f} kg/m³</span></div>
    <div class="atm-row">V&nbsp;<span class="atm-val">{th.velocity_ms:.1f} m/s</span>&nbsp;({th.velocity_ms * 3.6:.0f} km/h)</div>
  </div>
</div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Equations — always visible
# ---------------------------------------------------------------------------

def _show_equations() -> None:
    st.markdown('<div class="section-header">Governing Equations</div>', unsafe_allow_html=True)
    eq1, eq2 = st.columns(2)
    with eq1:
        st.markdown('<div class="eq-panel"><div class="eq-label">Sutton-Graves Convective Heating (ρ kg/m³, R_n m, V m/s → q W/m²)</div>', unsafe_allow_html=True)
        st.latex(r"q_{conv} = C \sqrt{\frac{\rho}{R_n}}\,V^3, \quad C = 1.7415 \times 10^{-4}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="eq-panel" style="margin-top:0.5rem"><div class="eq-label">Tauber-Sutton Radiative Heating (V > 6 km/s, V in km/s → q in W/m²)</div>', unsafe_allow_html=True)
        st.latex(r"q_{rad} = C_r\,\rho^{1.072}\,V^{3.5}\,R_n, \quad C_r = 4.736 \times 10^{-4} \times 10^4")
        st.markdown('</div>', unsafe_allow_html=True)

    with eq2:
        st.markdown('<div class="eq-panel"><div class="eq-label">Radiation-Equilibrium Wall Temperature</div>', unsafe_allow_html=True)
        st.latex(r"T_{wall} = \min\!\left[\left(\frac{q_{total}}{\varepsilon\,\sigma_{SB}}\right)^{1/4},\; T_{amb}\!\left(1 + r\frac{\gamma-1}{2}M^2\right)\right], \quad r=0.85")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="eq-panel" style="margin-top:0.5rem"><div class="eq-label">Structural Requirement (Material-Specific Thermal Stress)</div>', unsafe_allow_html=True)
        st.latex(r"\sigma_{req} = \underbrace{(\sigma_{inertial} + q_{dyn})\times 1.5}_{\sigma_{combined}} + E_{mat}\,\alpha_{mat}\,\Delta T")
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Materials table
# ---------------------------------------------------------------------------

def _show_materials(
    result, physics=None, vehicle_category: str = "general",
    *, cost_ceiling_usd: float = 0.0,
    sensitivity: SensitivityResult | None = None,
    design_lifetime_hours: float = 1.0,
    panel_thickness_m: float = 0.002,
) -> None:
    st.markdown('<div class="section-header">Materials Feasibility Analysis</div>', unsafe_allow_html=True)

    # Lifecycle banner: when the design lifetime is long enough that
    # creep matters (≥1000 h is the conventional threshold), surface
    # a one-line disclosure so the user knows the materials list has
    # been screened against rupture-stress at (T_wall, lifetime), not
    # just static thermal+structural.
    if design_lifetime_hours >= 1000.0:
        st.markdown(
            f'<div style="border-left:3px solid #d29922;padding:6px 10px;'
            f'margin:0 0 8px;background:#161b22;font-size:12px;color:#c9d1d9;'
            f'border-radius:0 4px 4px 0">'
            f'<strong>Lifecycle screening active</strong> — design '
            f'lifetime <strong>{design_lifetime_hours:,.0f} h</strong> '
            f'triggers Larson-Miller creep evaluation. Materials below '
            f'have been screened against rupture stress at '
            f'(T_wall, lifetime). The <em>Creep@Life</em> column shows '
            f'the rupture stress and pass/marginal/fail verdict.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Transient-heat banner: triggered when the matching engine ran
    # the 1D solver on at least one candidate. Surfaces the operative
    # peak-back-face temperature and the panel thickness assumption so
    # the user knows the table reflects internal soak, not steady-state.
    transient_active = any(
        getattr(c, "transient_status", "") == "applied"
        for c in (result.viable + result.marginal + result.not_viable)
    )
    if transient_active:
        thickness_mm = panel_thickness_m * 1000.0
        st.markdown(
            f'<div style="border-left:3px solid #1f6feb;padding:6px 10px;'
            f'margin:0 0 8px;background:#161b22;font-size:12px;color:#c9d1d9;'
            f'border-radius:0 4px 4px 0">'
            f'<strong>Transient screening active</strong> — short-duration '
            f'flight triggered the 1D heat solver. Materials are screened '
            f'against the peak back-face temperature reached during the '
            f'flight (through a <strong>{thickness_mm:.1f} mm</strong> panel), '
            f'not the steady-state surface envelope. The <em>Soak@Life</em> '
            f'column shows the peak internal temperature reached.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Vehicle mass drives the per-row "Est. Cost" column in every materials
    # tab; lifted here once so each tab call doesn't have to re-getattr.
    vehicle_mass_kg = float(getattr(physics, "vehicle_mass_kg", 0.0)) if physics else 0.0

    # When a SensitivityResult is supplied, project it into a name→robustness
    # lookup so the per-row Robustness column can render badges in O(1).
    # None when the user hasn't ticked the sensitivity checkbox — table
    # falls back to its pre-sensitivity 10/11-column shape.
    robustness_by_name: dict | None = None
    if sensitivity is not None and sensitivity.materials:
        robustness_by_name = {
            m.material_name: m for m in sensitivity.materials
        }

    viable       = result.viable
    marginal     = result.marginal
    not_viable   = result.not_viable[:15]
    rejected     = result.regime_rejected
    tps_coatings = result.tps_coatings

    # The "Required Coating" tab only appears when TPS materials were unlocked
    # by physics (T_wall ≥ TPS_UNLOCK_TEMP_K). Aircraft and missiles below that
    # threshold see the standard four-tab layout.
    tab_labels = [
        f"VIABLE ({len(viable)})",
        f"MARGINAL ({len(marginal)})",
        f"NOT VIABLE — TOP 15 of {len(result.not_viable)}",
        f"REGIME REJECTED ({len(rejected)})",
        "SURROGATE RANKING",
    ]
    if tps_coatings:
        tab_labels.insert(0, f"🔥 REQUIRED COATING ({len(tps_coatings)})")

    tabs = st.tabs(tab_labels)

    if tps_coatings:
        tab_c, tab_v, tab_m, tab_nv, tab_r, tab_surr = tabs
        with tab_c:
            st.markdown(
                '<div style="border-left:3px solid #f0883e;padding:8px 12px;margin:4px 0 12px;'
                'background:#161b22;font-size:13px;color:#c9d1d9;border-radius:0 4px 4px 0">'
                '<strong style="color:#f0883e">Required Coating Layer</strong><br>'
                'These TPS / ablator materials are <strong>not load-bearing</strong> and '
                'must be paired with a metallic substructure from the VIABLE list above. '
                'Sorted by thermal protection (highest ceiling first); structural columns '
                'are intentionally omitted because TPS materials are not ranked on '
                'structural margin against primary metals.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_tps_coatings_table_html(tps_coatings), unsafe_allow_html=True)
    else:
        tab_v, tab_m, tab_nv, tab_r, tab_surr = tabs

    with tab_v:
        if viable:
            st.markdown(
                _candidates_table_html(
                    viable, "row-viable",
                    vehicle_mass_kg=vehicle_mass_kg,
                    cost_ceiling_usd=cost_ceiling_usd,
                    robustness_by_name=robustness_by_name,
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info("No fully viable materials for this flight condition.")

    with tab_m:
        if marginal:
            st.markdown(
                _candidates_table_html(
                    marginal, "row-marginal",
                    vehicle_mass_kg=vehicle_mass_kg,
                    cost_ceiling_usd=cost_ceiling_usd,
                    robustness_by_name=robustness_by_name,
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info("No marginal materials for this flight condition.")

    with tab_nv:
        if not_viable:
            st.markdown(
                _candidates_table_html(
                    not_viable, "row-fail",
                    vehicle_mass_kg=vehicle_mass_kg,
                    cost_ceiling_usd=cost_ceiling_usd,
                    robustness_by_name=robustness_by_name,
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info("No evaluated materials in the not-viable category.")

    with tab_r:
        if rejected:
            st.markdown(_rejected_table_html(rejected), unsafe_allow_html=True)
        else:
            st.info("No materials were regime-rejected.")

    with tab_surr:
        if physics is None:
            st.info("Physics context required for surrogate ranking.")
        else:
            surr = find_nearest_candidates(physics, vehicle_category, match_result=result, k=10)
            # Agreement badge
            agr = surr.agreement_with_margin_ranking
            if agr >= 0.6:
                badge_color, badge_label = "#238636", "HIGH"
            elif agr >= 0.4:
                badge_color, badge_label = "#d29922", "MODERATE"
            else:
                badge_color, badge_label = "#f85149", "LOW"
            st.markdown(
                f'<div style="display:inline-block;padding:4px 10px;border-radius:4px;'
                f'background:{badge_color};color:#fff;font-size:12px;font-weight:700;'
                f'margin-bottom:8px">{badge_label} AGREEMENT ({agr:.0%})</div>',
                unsafe_allow_html=True,
            )
            # Surrogate ranking table
            viable_names = {c.material.name for c in result.viable + result.marginal}
            header = "| # | Material | Category | Distance | Also Viable/Marginal |"
            sep = "|---|---|---|---:|:---:|"
            rows = []
            for idx, (mat, dist) in enumerate(zip(surr.candidates, surr.distances), 1):
                in_match = "Yes" if mat.name in viable_names else "—"
                rows.append(f"| {idx} | {mat.name} | {mat.category} | {dist:.3f} | {in_match} |")
            st.markdown("\n".join([header, sep] + rows))
            st.caption(
                f"k-NN surrogate in 7D normalized property space. "
                f"Model version: {surr.model_version[:12]}…"
            )


# ---------------------------------------------------------------------------
# Sensitivity analysis card (Uncertainty & Sensitivity feature)
# ---------------------------------------------------------------------------

def _show_sensitivity(sensitivity: SensitivityResult | None) -> None:
    """Render the Sensitivity Analysis section between the materials
    table and the Pareto front.

    Skipped (returns immediately) when the user did not tick the
    sensitivity checkbox (sensitivity is None) OR when the nominal
    viable list was empty (sensitivity.materials is empty — there
    is nothing to robustness-rank).

    The intro paragraph is deliberately written for the pre-test-
    campaign engineering audience: a one-line plain explanation, a
    table + tornado, and a knife-edge narrative paragraph that flags
    materials the engineer should NOT bet a $4M test on.
    """
    if sensitivity is None:
        return
    if not sensitivity.materials:
        # Nominal had no viable material — nothing to robustness-rank.
        # The materials section already showed the impossibility diagnosis.
        return

    st.markdown(
        '<div class="section-header">Sensitivity Analysis</div>',
        unsafe_allow_html=True,
    )

    spec = sensitivity.spec
    with st.expander("How to read the sensitivity chart", expanded=False):
        st.markdown(
            "We perturb each envelope input one at a time across its "
            "uncertainty range and re-run the full pipeline. **Long bars** "
            "= inputs whose uncertainty meaningfully erodes the safety "
            "margin — pin these down before testing. **Short or "
            "'negligible' bars** = the top material is insensitive to that "
            "input. **Robustness labels** in the table count how often each "
            "material stays viable across all swept scenarios."
        )
    st.caption(
        f"Each input was swept ±{spec.mach_delta_frac:.0%} (Mach), "
        f"±{spec.mass_delta_frac:.0%} (mass), "
        f"±{spec.R_n_delta_frac:.0%} (nose radius), "
        f"±{spec.g_load_delta_frac:.0%} (g-load) over "
        f"{spec.n_samples} samples each = "
        f"{4 * spec.n_samples} perturbed scenarios. "
        "Robust ≥90%, borderline 50–90%, knife-edge <50%."
    )

    # ── Instrument card: top-material headline + nominal margin +
    # native HTML tornado. Mirrors the visual language of the Pareto
    # _bar_cell pattern (app.py:1916) so this section reads as a
    # sibling of the Pareto Trade-off section directly below it,
    # rather than an embedded image plate. The matplotlib tornado
    # (sensitivity.chart_png) is intentionally NOT used here — it
    # remains the canonical PDF rendering and is consumed by
    # latex_export.py:_sec_sensitivity untouched.
    tornado_items = sorted(
        sensitivity.tornado_data.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    v_max = max((v for _, v in tornado_items), default=0.0)
    margin_color = (
        "#3fb950" if sensitivity.baseline_min_margin >= 0 else "#f85149"
    )

    if not tornado_items:
        # Defensive: tornado_data is populated alongside materials, but
        # if a future code path ever ships an empty dict alongside a
        # non-empty materials list, we want a self-explanatory message
        # instead of a broken-looking empty table.
        inner = (
            "<em style='color:#8b949e'>"
            "No per-input sensitivity data available for this run."
            "</em>"
        )
    elif v_max < 1e-4:
        # All four inputs erode the margin by < 0.01 pp. Don't render
        # four indistinguishable zero-width bars — say so plainly.
        inner = (
            "<div style='color:#3fb950;font-size:13px;padding:8px 0'>"
            "All swept inputs are negligible at their nominal "
            "uncertainty ranges — this material is robust across the "
            "full envelope sweep."
            "</div>"
        )
    else:
        body = "".join(
            _tornado_row_html(
                _INPUT_DISPLAY_NAMES.get(k, k), v, v_max
            )
            for k, v in tornado_items
        )
        inner = (
            "<table style='width:100%;border-collapse:collapse'>"
            f"<tbody>{body}</tbody>"
            "</table>"
        )

    st.markdown(
        "<div style='background:#161b22;border:1px solid #30363d;"
        "border-radius:6px;padding:16px 20px;margin:12px 0'>"
          "<div style='font-size:11px;color:#8b949e;"
                  "text-transform:uppercase;letter-spacing:0.08em'>"
            f"Top-ranked material — {sensitivity.top_material_name}"
          "</div>"
          "<div style='font-size:24px;color:#c9d1d9;margin:6px 0 4px 0'>"
            "Nominal safety margin: "
            f"<strong style='color:{margin_color}'>"
            f"{sensitivity.baseline_min_margin:+.3f}</strong>"
          "</div>"
          "<div style='font-size:12px;color:#8b949e;margin-bottom:14px'>"
            "Each bar shows how much that input's uncertainty range "
            "erodes the safety margin (in percentage points). "
            "Longer bars = inputs to nail down before testing."
          "</div>"
          f"{inner}"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Materials robustness table — full width, below the card.
    # Sorted by fraction descending so the most bullet-proof picks lead
    # and the knife-edge picks land at the bottom where they're easiest
    # to spot.
    rows = sorted(
        sensitivity.materials,
        key=lambda m: m.robustness_fraction,
        reverse=True,
    )
    header = (
        "<table class='matvec-table'><thead><tr>"
        "<th>Material</th><th>Robustness</th>"
        "<th>Scenarios viable</th><th>Critical input</th>"
        "</tr></thead><tbody>"
    )
    body_rows = []
    for r in rows:
        body_rows.append(
            "<tr>"
            f"<td><strong>{r.material_name}</strong></td>"
            f"<td>{_robustness_badge_html(r.robustness_label, r.robustness_fraction)}</td>"
            f"<td class='col-mono'>{r.n_scenarios_viable} / {r.n_scenarios_total}</td>"
            f"<td style='color:#8b949e'>{_INPUT_DISPLAY_NAMES.get(r.critical_input, r.critical_input)}</td>"
            "</tr>"
        )
    st.markdown(
        header + "".join(body_rows) + "</tbody></table>",
        unsafe_allow_html=True,
    )

    # Knife-edge narrative — one sentence per knife-edge or borderline
    # material so the engineer can read "what would knock this out" at
    # a glance without parsing the table. Robust picks are silent.
    knife_edges = [m for m in sensitivity.materials if m.robustness_label == "knife-edge"]
    borderlines = [m for m in sensitivity.materials if m.robustness_label == "borderline"]
    if knife_edges or borderlines:
        st.markdown("**Risk notes**")
        for m in knife_edges:
            crit_pct = {
                "mach":   spec.mach_delta_frac,
                "mass":   spec.mass_delta_frac,
                "R_n":    spec.R_n_delta_frac,
                "g_load": spec.g_load_delta_frac,
            }.get(m.critical_input, 0.10) * 100
            st.markdown(
                f'<div style="border-left:3px solid #f85149;padding:6px 12px;'
                f'margin:4px 0;background:#161b22;font-size:13px;color:#c9d1d9;'
                f'border-radius:0 4px 4px 0">'
                f'<strong style="color:#f85149">Knife-edge: {m.material_name}</strong> '
                f'— drops out of viable in {m.n_scenarios_total - m.n_scenarios_viable} '
                f'of {m.n_scenarios_total} scenarios. Most fragile to '
                f'<strong>{_INPUT_DISPLAY_NAMES.get(m.critical_input, m.critical_input)}</strong> '
                f'(swept ±{crit_pct:.0f}%).</div>',
                unsafe_allow_html=True,
            )
        for m in borderlines:
            st.markdown(
                f'<div style="border-left:3px solid #d29922;padding:6px 12px;'
                f'margin:4px 0;background:#161b22;font-size:13px;color:#c9d1d9;'
                f'border-radius:0 4px 4px 0">'
                f'<strong style="color:#d29922">Borderline: {m.material_name}</strong> '
                f'— survives only {m.robustness_fraction:.0%} of the sweep; '
                f'most sensitive to <strong>{_INPUT_DISPLAY_NAMES.get(m.critical_input, m.critical_input)}</strong>.</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Pareto trade-off analysis
# ---------------------------------------------------------------------------

def _show_pareto(
    physics, result, vehicle_category: str,
    *, cost_ceiling_usd: float = 1_000_000.0,
) -> None:
    candidates = list(result.viable) + list(result.marginal)
    if len(candidates) < 3:
        return

    # Cost ceiling flows into the Pareto cost objective (5th dimension).
    # Same value the materials table uses for red-row highlighting, so
    # a row flagged in the table is the row penalised on the front.
    pareto = compute_pareto(
        candidates, physics, vehicle_category,
        cost_ceiling_usd=float(cost_ceiling_usd) or 1_000_000.0,
    )
    if not pareto.pareto_front:
        return

    st.markdown(
        '<div class="section-header">Pareto Trade-off Analysis</div>',
        unsafe_allow_html=True,
    )

    mask = pareto.pareto_mask
    obj = pareto.objective_values

    def _bar_color(v: float) -> str:
        if v <= 0.25:
            return "#238636"
        if v <= 0.55:
            return "#d29922"
        return "#f85149"

    def _bar_cell(v: float) -> str:
        c = _bar_color(v)
        pct = min(v * 100, 100)
        return (
            '<td class="col-mono" style="text-align:right;position:relative">'
            f'<div style="position:absolute;left:0;top:0;bottom:0;'
            f'width:{pct:.0f}%;background:{c};opacity:0.18;'
            f'border-radius:2px"></div>'
            f'<span style="position:relative;z-index:1">{v:.3f}</span>'
            '</td>'
        )

    # Pareto now carries 5 objectives (Weight, Thermal, Structural,
    # Availability, Cost). Render every column the result actually has —
    # `range(obj.shape[1])` is robust to a future 6th axis.
    n_obj = obj.shape[1] if obj.size else 5
    rows = []
    for i, is_front in enumerate(mask):
        if is_front:
            cand = candidates[i]
            name = cand.material.name
            cat = cand.material.category
            cells = "".join(_bar_cell(obj[i, j]) for j in range(n_obj))
            rows.append(
                f'<tr style="border-left:3px solid #58a6ff">'
                f'<td style="padding-left:0.5rem">{name}</td>'
                f'<td style="color:#8b949e;font-size:11px">{cat}</td>'
                f'{cells}</tr>'
            )

    obj_headers = ["Wt", "Th", "St", "Av", "$$"]
    hdr_obj = "".join(
        f'<th style="text-align:right">{obj_headers[j]}</th>'
        for j in range(n_obj)
    )
    hdr = (
        '<th>Material</th><th>Category</th>'
        + hdr_obj
    )

    st.markdown(
        '<div class="instrument-card" style="padding:0.6rem 0.8rem">'
        '<div class="card-title">Pareto Front Members</div>'
        '<div style="overflow-x:auto">'
        f'<table class="matvec-table" style="margin-top:0.3rem">'
        f'<thead><tr>{hdr}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    if pareto.trade_off_descriptions:
        items = "".join(
            f'<li style="margin-bottom:0.25rem">{d}</li>'
            for d in pareto.trade_off_descriptions
        )
        st.markdown(
            '<div class="instrument-card" style="margin-top:0.5rem;padding:0.6rem 0.8rem">'
            '<div class="card-title">Trade-off Summary</div>'
            f'<ol style="margin:0;padding-left:1.2rem;font-size:12px;color:#8b949e">'
            f'{items}</ol></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def _show_component_zones(physics, vehicle_category: str) -> None:
    """Per-zone material recommendations.

    The headline materials table above ranks materials against the
    vehicle's worst-case envelope (typically the leading edge or
    stagnation point). Real airframes are zoned: a fuselage panel sees
    a fraction of the leading-edge temperature; an internal spar sees a
    cooler soak temperature but concentrated bending stress. This
    section runs the matching engine separately for each zone with its
    own locally-scaled (T_wall, sigma_req), so a user can see how the
    recommendation shifts across the vehicle.

    Mirrors the LaTeX section emitted by ``latex_export._sec_component_zones``
    so the UI and PDF report show the same per-zone breakdown.
    """
    from core.component_zones import evaluate_zones

    zones = evaluate_zones(physics, vehicle_category)
    if not zones:
        return  # category has no per-zone catalog

    st.markdown(
        '<div class="section-header">Per-Zone Material Recommendations</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The materials table above ranks against the worst-zone envelope "
        "(usually the leading edge or stagnation point). Below, each "
        "geometric zone is re-evaluated with its own local thermal and "
        "structural demands — useful for hot-leading-edge / cool-fuselage "
        "compositions."
    )

    cols = st.columns(min(len(zones), 3))
    for idx, zr in enumerate(zones):
        with cols[idx % len(cols)]:
            # Pick top candidates, preferring viable but falling back to
            # marginal so a zone with no fully-viable material still
            # shows something useful (often the case for the hottest
            # zone of a TPS-protected vehicle).
            top = list(zr.match.viable[:3])
            if len(top) < 3:
                top.extend(zr.match.marginal[: 3 - len(top)])
            top_html = ""
            if top:
                rows = []
                for c in top:
                    cat = _cat_label(c.material.category)
                    badge = ""
                    if c in zr.match.marginal:
                        badge = (
                            ' <span style="color:#d29922;font-size:11px;'
                            'font-weight:600">marginal</span>'
                        )
                    rows.append(
                        f'<div class="card-row">'
                        f'<span class="card-row-label">{c.material.name}'
                        f'{badge}</span>'
                        f'<span class="card-row-value" '
                        f'style="font-size:11px;color:#8b949e">{cat}</span>'
                        f'</div>'
                    )
                top_html = "".join(rows)
            else:
                top_html = (
                    '<div class="card-row">'
                    '<span class="card-row-label" style="color:#8b949e">'
                    'no viable or marginal candidates</span>'
                    '<span class="card-row-value"></span>'
                    '</div>'
                )

            body = ""
            body += (
                f'<div style="font-size:12px;color:#8b949e;line-height:1.4;'
                f'margin:2px 0 8px">{zr.zone.description}</div>'
            )
            body += _card_row(
                "σ_req (local)",
                f"{zr.sigma_req_zone_MPa:.0f}",
                "MPa",
            )
            body += (
                '<div style="border-top:1px solid #21262d;margin-top:6px;'
                'padding-top:6px;color:#8b949e;font-size:11px;'
                'text-transform:uppercase;letter-spacing:0.5px">'
                'Top candidates</div>'
            )
            body += top_html

            st.markdown(
                _instrument_card(
                    zr.zone.name,
                    f"{zr.T_wall_zone_K:.0f}",
                    "K  (T_wall local)",
                    body,
                ),
                unsafe_allow_html=True,
            )


def _show_diagnosis(result) -> None:
    if result.no_material_viable:
        msg = result.diagnosis or "No viable materials for this flight condition."
        if result.impossible:
            st.error(f"**INFEASIBLE DESIGN SPACE** — {msg}")
        else:
            st.warning(f"**NO VIABLE MATERIALS** — {msg}")


# ---------------------------------------------------------------------------
# Export stub
# ---------------------------------------------------------------------------

def _show_export(
    physics, match_result, system_label: str,
    *, cost_ceiling_usd: float = 1_000_000.0,
    sensitivity: SensitivityResult | None = None,
    design_lifetime_hours: float = 1.0,
    panel_thickness_m: float = 0.002,
) -> None:
    st.markdown("""
<div class="export-section">
  <div class="export-title">Report Export</div>
  <p style="font-size:12px;color:#8b949e;margin:0 0 0.75rem">
    Generate a full technical report with equations, tables, and material assessments.
  </p>
</div>""", unsafe_allow_html=True)

    slug = _make_slug(system_label)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    # The PDF export must use the same cost ceiling as the on-screen Pareto
    # and materials table, otherwise a user who tightens the slider and
    # then exports gets a PDF with bolded rows that don't match what they
    # see in the browser.
    ceiling = float(cost_ceiling_usd) or 1_000_000.0

    col_pdf, col_tex, _ = st.columns([2, 2, 6])
    with col_pdf:
        if st.button("📄  Export Report (PDF)", use_container_width=True, key="btn_pdf"):
            with st.spinner("Compiling report with pdflatex…"):
                pdf_bytes = generate_report(
                    physics, match_result, system_label,
                    cost_ceiling_usd=ceiling,
                    sensitivity=sensitivity,
                    design_lifetime_hours=design_lifetime_hours,
                    panel_thickness_m=panel_thickness_m,
                )
            if pdf_bytes:
                st.download_button(
                    label="⬇  Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"matvec_{slug}_{ts}.pdf",
                    mime="application/pdf",
                    key="dl_pdf",
                )
            else:
                st.error(
                    "pdflatex compilation failed. "
                    "Download the LaTeX source and compile manually."
                )
                tex_src, _ = generate_tex_source(
                    physics, match_result, system_label,
                    cost_ceiling_usd=ceiling,
                    sensitivity=sensitivity,
                    design_lifetime_hours=design_lifetime_hours,
                    panel_thickness_m=panel_thickness_m,
                )
                st.download_button(
                    label="⬇  Download LaTeX Source (.tex)",
                    data=tex_src.encode("utf-8"),
                    file_name=f"matvec_{slug}_{ts}.tex",
                    mime="text/plain",
                    key="dl_tex_fallback",
                )
    with col_tex:
        if st.button("📋  Export LaTeX Source (.tex)", use_container_width=True, key="btn_tex"):
            tex_src, _ = generate_tex_source(
                physics, match_result, system_label,
                cost_ceiling_usd=ceiling,
                sensitivity=sensitivity,
                design_lifetime_hours=design_lifetime_hours,
                panel_thickness_m=panel_thickness_m,
            )
            st.download_button(
                label="⬇  Download .tex Source",
                data=tex_src.encode("utf-8"),
                file_name=f"matvec_{slug}_{ts}.tex",
                mime="text/plain",
                key="dl_tex",
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Sidebar (preset + advanced + sensitivity sliders + IO) ──
    sidebar_inputs = _sidebar()

    # ── Reserve the slot for the persistent envelope chip above the
    # tabs. Filled below once we have physics results.
    chip_slot = st.empty()

    # ── Three-tab top-level layout ──
    tab_setup, tab_results, tab_tradeoffs = st.tabs([
        "Setup", "Results", "Trade-offs",
    ])

    # ── Tab 1: Flight Envelope form ──
    with tab_setup:
        form_inputs = _render_setup_tab()

    # The Streamlit UI uses the SAME pipeline entry point as the CLI:
    # build a SessionSchema, hand it to run_session(). Every piece of
    # glue (turbine override, category filtering, match / Pareto /
    # surrogate wiring) lives in core/api.py. Keeping the UI as a thin
    # shell over that boundary is what unblocks the academic
    # Metric-Standard extraction described in HANDOFF.md §1 — the UI
    # never reaches past the SessionResult.
    # system_label propagates through the pipeline (PDF title page,
    # filename slug). It's set by the bundled-example loader (or by a
    # JSON upload), persists across reruns via session_state, and
    # falls back to "Custom Analysis" for envelopes the user typed
    # from scratch.
    system_label = st.session_state.get("system_label") or "Custom Analysis"

    vehicle_category = form_inputs["vehicle_category"]
    cost_ceiling_usd = float(sidebar_inputs.get("cost_ceiling_usd") or 0.0)

    session = _build_session_from_widgets(
        mach=form_inputs["mach"],
        alt_km=form_inputs["alt_km"],
        mass_kg=form_inputs["mass_kg"],
        R_n=form_inputs["R_n"],
        peak_g=form_inputs["peak_g"],
        epsilon=sidebar_inputs["epsilon"],
        char_len=sidebar_inputs["char_len"],
        flight_duration_s=sidebar_inputs["flight_duration_s"],
        design_lifetime_hours=sidebar_inputs["design_lifetime_hours"],
        panel_thickness_m=sidebar_inputs["panel_thickness_m"],
        vehicle_category=vehicle_category,
        hot_section_temp_K=form_inputs.get("hot_section_temp_K"),
        cost_ceiling_usd=cost_ceiling_usd,
    )
    # Use the (already-resolved) system_label for this render. The
    # widget-driven helper above falls back to "Custom Analysis" if
    # nothing is set; override to the resolved value so the PDF
    # title page matches what a user who didn't touch the label would
    # expect (e.g. "SR-71 Blackbird" after loading that preset).
    session.system_label = system_label

    # Compile PDF lazily: the Report Export button calls generate_report
    # on click rather than every rerun. `run_session(compile_pdf=False)`
    # keeps the main loop fast — ~50ms match vs ~2s pdflatex.
    session_result = run_session(session, compile_pdf=False)
    physics = session_result.physics
    result  = session_result.match

    # Now fill the envelope-chip slot above the tabs (visible from
    # every tab so the engineer always knows what scenario the
    # results belong to).
    with chip_slot.container():
        _render_envelope_chip(physics, vehicle_category)

    # Sensitivity is opt-in: run only when the sidebar checkbox is ticked.
    # Computed BEFORE the Results tab renders so the materials table
    # can include the Robustness column inline (single source of truth —
    # no second sweep).
    sensitivity_result: SensitivityResult | None = None
    if sidebar_inputs.get("run_sensitivity"):
        spec = SensitivitySpec(
            mach_delta_frac   = float(sidebar_inputs["sens_mach_delta_frac"]),
            mass_delta_frac   = float(sidebar_inputs["sens_mass_delta_frac"]),
            R_n_delta_frac    = float(sidebar_inputs["sens_Rn_delta_frac"]),
            g_load_delta_frac = float(sidebar_inputs["sens_g_delta_frac"]),
            n_samples         = int(sidebar_inputs["sens_n_samples"]),
        )
        with st.spinner(
            f"Running sensitivity sweep "
            f"({4 * spec.n_samples} perturbed scenarios)…"
        ):
            sensitivity_result = run_sensitivity(
                session, vehicle_category, spec=spec,
            )

    # ── Tab 2: Results (physics + materials + diagnosis + export) ──
    with tab_results:
        for w in result.warnings:
            st.warning(w)
        _show_category_mode_badge(vehicle_category)
        st.markdown(
            '<div class="section-header">Physics Analysis</div>',
            unsafe_allow_html=True,
        )
        _show_physics(physics, vehicle_category=vehicle_category)
        st.divider()
        _show_equations()
        st.divider()
        _show_materials(
            result, physics=physics,
            vehicle_category=vehicle_category,
            cost_ceiling_usd=cost_ceiling_usd,
            sensitivity=sensitivity_result,
            design_lifetime_hours=float(session.design_lifetime_hours),
            panel_thickness_m=float(session.panel_thickness_m),
        )
        st.divider()
        _show_component_zones(physics, vehicle_category)
        _show_diagnosis(result)
        _show_export(
            physics, result, session.system_label,
            cost_ceiling_usd=cost_ceiling_usd,
            sensitivity=sensitivity_result,
            design_lifetime_hours=float(session.design_lifetime_hours),
            panel_thickness_m=float(session.panel_thickness_m),
        )

    # ── Tab 3: Trade-offs (Pareto + Sensitivity tornado) ──
    with tab_tradeoffs:
        _show_pareto(
            physics, result, vehicle_category,
            cost_ceiling_usd=cost_ceiling_usd,
        )
        _show_sensitivity(sensitivity_result)


if __name__ == "__main__":
    main()
