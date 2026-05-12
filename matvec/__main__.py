"""
MATVEC CLI — headless entry point.

Usage:
    python -m matvec run envelope.json --out report.pdf
    python -m matvec run --mach 3.2 --alt 25 --mass 30600 --rn 0.15 \\
                          --g 2.5 --category aircraft --out report.pdf
    python -m matvec validate --out-dir reports/
    python -m matvec batch sweep.csv --out-dir reports/

Commands
--------
run      — one envelope → one PDF. Accepts a JSON file OR explicit
           flag arguments; the two modes are mutually exclusive.
validate — regenerates every canonical preset PDF into --out-dir.
           Non-zero exit if any PDF fails to render. CI hook.
batch    — reads a CSV whose columns are the envelope fields, one row
           per analysis; writes N PDFs into --out-dir.

HARD CONSTRAINT: this module does NOT import streamlit. The whole
point of the CLI boundary is to let MATVEC run in environments where
streamlit isn't installed (Docker, CI, academic venvs). The static
check in ``test_api.py::TestStreamlitFreeCLI`` enforces this.
"""

import argparse
import csv
import itertools
import math
import re
import sys
from dataclasses import replace as _dc_replace
from pathlib import Path

from core import MATVEC_VERSION
from core.api import run_session
from core.presets import CANONICAL_PRESETS
from core.sensitivity import SensitivitySpec, run_sensitivity
from core.session import SessionSchema, json_to_session


# ---------------------------------------------------------------------------
# Small helpers — each kept narrow so the command handlers below read
# like sentences.
# ---------------------------------------------------------------------------

def _slug(label: str) -> str:
    """Filename-safe slug — matches ``app.py::_make_slug`` semantics.

    Keeps CLI-generated and UI-generated filenames identical for the
    same system label, so a validate-command PDF and a Streamlit-
    exported PDF for the same preset have the same basename.
    """
    s = label.lower().replace(" ", "-")
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "analysis"


def _default_char_len(mass_kg: float) -> float:
    """Fallback characteristic length used when --char-len is omitted.

    Mirrors ``physics_engine._estimate_characteristic_length``: treat
    the vehicle as an aluminum sphere (ρ = 2700 kg/m³) and take the
    side of an equivalent cylinder as the characteristic length. Crude
    but documented as such — the user can always override with --char-len.
    """
    return (float(mass_kg) / (2700.0 * math.pi / 4.0)) ** (1.0 / 3.0)


def _write_pdf(pdf_bytes: bytes, out_path: Path) -> None:
    """Create parent dirs and write PDF bytes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pdf_bytes)


def _load_session_from_file(path: Path) -> SessionSchema:
    return json_to_session(path.read_text(encoding="utf-8"))


def _spec_from_flags(ns) -> SensitivitySpec:
    """Build a SensitivitySpec from the ``--sens-*`` flags on the
    namespace. Any flag left unset (None) inherits the SensitivitySpec
    dataclass default — so ``--sensitivity`` with no other knobs
    reproduces the canonical ±10/15/20/25%, 11-sample sweep that
    the Streamlit checkbox uses.

    Percentage flags (``--sens-mach-delta`` etc.) accept the
    *percentage* value (e.g. ``10`` for ±10%) because that is how
    the Streamlit slider exposes it; the spec stores fractions, so
    we divide by 100 here. ``--sens-samples`` is a raw integer.
    """
    defaults = SensitivitySpec()
    return SensitivitySpec(
        mach_delta_frac   = (ns.sens_mach_delta   / 100.0) if ns.sens_mach_delta   is not None else defaults.mach_delta_frac,
        mass_delta_frac   = (ns.sens_mass_delta   / 100.0) if ns.sens_mass_delta   is not None else defaults.mass_delta_frac,
        R_n_delta_frac    = (ns.sens_rn_delta     / 100.0) if ns.sens_rn_delta     is not None else defaults.R_n_delta_frac,
        g_load_delta_frac = (ns.sens_g_delta      / 100.0) if ns.sens_g_delta      is not None else defaults.g_load_delta_frac,
        n_samples         = int(ns.sens_samples)               if ns.sens_samples       is not None else defaults.n_samples,
    )


def _session_from_flags(ns) -> SessionSchema:
    """Build a SessionSchema from explicit CLI arguments.

    Optional flags inherit SessionSchema dataclass defaults. If
    --char-len is omitted, fall back to the mass-derived estimate
    (matching the Streamlit path's behaviour so CLI and UI produce
    identical numbers for the same inputs).
    """
    options = {}
    if ns.hot_section_K is not None:
        options["hot_section_temp_K"] = float(ns.hot_section_K)

    char_len = (
        float(ns.char_len)
        if ns.char_len is not None
        else _default_char_len(ns.mass)
    )

    return SessionSchema(
        mach=float(ns.mach),
        alt_km=float(ns.alt),
        mass_kg=float(ns.mass),
        R_n_m=float(ns.rn),
        g_load=float(ns.g),
        char_len_m=char_len,
        flight_duration_s=float(ns.duration),
        wall_emissivity=float(ns.emissivity),
        vehicle_category=ns.category,
        system_label=ns.label or "Custom Analysis",
        options=options,
    )


# ---------------------------------------------------------------------------
# Command: run — one envelope → one PDF
# ---------------------------------------------------------------------------

def cmd_run(ns) -> int:
    # Source-selection: --from/positional JSON is one mode; the explicit
    # envelope flags are the other. Reject the ambiguous combo loudly.
    explicit_flags = (ns.mach, ns.alt, ns.mass, ns.rn, ns.g)
    has_explicit = any(x is not None for x in explicit_flags)

    if ns.from_file and has_explicit:
        print(
            "ERROR: --from / positional JSON and explicit envelope flags "
            "are mutually exclusive.",
            file=sys.stderr,
        )
        return 2

    if ns.from_file:
        path = Path(ns.from_file)
        if not path.is_file():
            print(f"ERROR: JSON file not found: {path}", file=sys.stderr)
            return 2
        try:
            session = _load_session_from_file(path)
        except ValueError as exc:
            print(f"ERROR: could not parse {path}: {exc}", file=sys.stderr)
            return 2
    else:
        # Explicit-flag mode — argparse doesn't enforce the combination,
        # so check it here and produce a friendly message with every
        # missing flag listed at once.
        required = (
            ("--mach", ns.mach),
            ("--alt", ns.alt),
            ("--mass", ns.mass),
            ("--rn", ns.rn),
            ("--g", ns.g),
            ("--category", ns.category),
        )
        missing = [name for name, val in required if val is None]
        if missing:
            print(
                f"ERROR: missing required arguments: {', '.join(missing)}. "
                "Either pass a JSON envelope via --from / positional "
                "FILE.json, or supply all of --mach / --alt / --mass / "
                "--rn / --g / --category.",
                file=sys.stderr,
            )
            return 2
        session = _session_from_flags(ns)

    # Optional sensitivity sweep — opt-in via --sensitivity. Computed
    # before run_session so the LaTeX exporter can splice the
    # \section{Sensitivity Analysis} block between Materials (§8) and
    # Per-Zone (§9). Heavy: ~44 pipeline runs at default n_samples=11,
    # so we surface a one-line progress note instead of staying silent.
    sensitivity = None
    if ns.sensitivity:
        spec = _spec_from_flags(ns)
        n_total = 4 * spec.n_samples
        print(
            f"  Sensitivity sweep: 4 inputs x {spec.n_samples} samples "
            f"= {n_total} scenarios (this may take ~10-30 s)...",
            file=sys.stderr, flush=True,
        )
        sensitivity = run_sensitivity(
            session, session.vehicle_category, spec=spec,
        )

    result = run_session(session, sensitivity=sensitivity)

    if result.pdf_bytes is None:
        print(
            "ERROR: pdflatex produced no output — is pdflatex installed "
            "and on PATH?",
            file=sys.stderr,
        )
        if ns.tex_on_fail:
            tex_out = Path(ns.out).with_suffix(".tex")
            tex_out.parent.mkdir(parents=True, exist_ok=True)
            tex_out.write_text(result.tex_source, encoding="utf-8")
            print(f"  LaTeX source written to: {tex_out}", file=sys.stderr)
        return 1

    out_path = Path(ns.out)
    _write_pdf(result.pdf_bytes, out_path)
    size_kb = out_path.stat().st_size / 1024
    extra = ""
    if sensitivity is not None:
        # Echo top-material robustness so the operator gets a
        # one-glance summary without opening the PDF.
        if sensitivity.materials:
            top = sensitivity.materials[0]
            extra = (
                f" sensitivity={top.material_name}:{top.robustness_label} "
                f"({top.n_scenarios_viable}/{top.n_scenarios_total})"
            )
        else:
            extra = " sensitivity=no-viable-material"
    print(
        f"OK: {out_path} ({size_kb:.1f} KB); "
        f"viable={len(result.match.viable)} "
        f"marginal={len(result.match.marginal)}"
        f"{extra}"
    )
    return 0


# ---------------------------------------------------------------------------
# Command: validate — regenerate every preset PDF (CI hook)
# ---------------------------------------------------------------------------

def cmd_validate(ns) -> int:
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fail = 0
    total = len(CANONICAL_PRESETS)
    for name, session in CANONICAL_PRESETS.items():
        slug = _slug(session.system_label)
        pdf_path = out_dir / f"matvec_{slug}.pdf"
        print(f"[{name}] rendering…", end=" ", flush=True)
        try:
            result = run_session(session)
        except Exception as exc:  # noqa: BLE001 — we want any failure
            print(f"FAILED during run_session: {exc}")
            fail += 1
            continue
        if not result.pdf_bytes:
            print("FAILED (no PDF bytes)")
            fail += 1
            continue
        _write_pdf(result.pdf_bytes, pdf_path)
        size_kb = pdf_path.stat().st_size / 1024
        print(f"OK ({size_kb:.1f} KB) viable={len(result.match.viable)}")

    if fail:
        print(
            f"\n{fail}/{total} presets failed to render.",
            file=sys.stderr,
        )
        return 1
    print(f"\nAll {total} presets rendered to {out_dir}.")
    return 0


# ---------------------------------------------------------------------------
# Command: batch — CSV sweep
# ---------------------------------------------------------------------------

# Aliases are intentional: spreadsheet authors should not have to
# remember the exact schema field names. Each alias resolves to a
# SessionSchema field. Adding one here does not change the SessionSchema
# field set — it only eases CSV authoring.
_CSV_ALIASES = {
    "alt": "alt_km",
    "altitude_km": "alt_km",
    "mass": "mass_kg",
    "R_n": "R_n_m",
    "nose_radius_m": "R_n_m",
    "g": "g_load",
    "peak_g": "g_load",
    "char_len": "char_len_m",
    "characteristic_length_m": "char_len_m",
    "emissivity": "wall_emissivity",
    "duration_s": "flight_duration_s",
    "category": "vehicle_category",
    "label": "system_label",
}


def _row_to_session(row: dict) -> SessionSchema:
    """Convert one CSV row → SessionSchema.

    Accepts both SessionSchema field names and the aliases in
    ``_CSV_ALIASES``. Missing required columns raise ``KeyError`` so
    the batch command can skip that row and continue.
    """
    norm: dict[str, str] = {}
    for k, v in row.items():
        if v is None or v == "":
            continue
        key = _CSV_ALIASES.get(k, k)
        norm[key] = v

    required_env = ("mach", "alt_km", "mass_kg", "R_n_m", "g_load")
    for f in required_env:
        if f not in norm:
            raise KeyError(f"missing required column '{f}' (or an alias)")

    options = {}
    if "hot_section_temp_K" in norm:
        options["hot_section_temp_K"] = float(norm["hot_section_temp_K"])

    char_len = norm.get("char_len_m")
    if char_len is None:
        char_len = _default_char_len(float(norm["mass_kg"]))
    else:
        char_len = float(char_len)

    return SessionSchema(
        mach=float(norm["mach"]),
        alt_km=float(norm["alt_km"]),
        mass_kg=float(norm["mass_kg"]),
        R_n_m=float(norm["R_n_m"]),
        g_load=float(norm["g_load"]),
        char_len_m=char_len,
        flight_duration_s=float(norm.get("flight_duration_s", 600.0)),
        wall_emissivity=float(norm.get("wall_emissivity", 0.85)),
        vehicle_category=str(norm.get("vehicle_category", "general")),
        system_label=str(norm.get("system_label", "Custom Analysis")),
        options=options,
    )


def cmd_batch(ns) -> int:
    csv_path = Path(ns.csv)
    if not csv_path.is_file():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        return 2
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("ERROR: CSV has no data rows.", file=sys.stderr)
        return 2

    fail = 0
    for idx, row in enumerate(rows):
        try:
            session = _row_to_session(row)
        except (KeyError, ValueError) as exc:
            print(f"[row {idx}] parse error: {exc}", file=sys.stderr)
            fail += 1
            continue
        slug = _slug(session.system_label or f"row{idx}")
        pdf_path = out_dir / f"matvec_{slug}_{idx:03d}.pdf"
        print(
            f"[row {idx}: {session.system_label}] rendering…",
            end=" ", flush=True,
        )
        try:
            result = run_session(session)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}", file=sys.stderr)
            fail += 1
            continue
        if not result.pdf_bytes:
            print("FAILED (no PDF bytes)")
            fail += 1
            continue
        _write_pdf(result.pdf_bytes, pdf_path)
        print(f"OK → {pdf_path.name}")

    if fail:
        print(f"\n{fail}/{len(rows)} rows failed.", file=sys.stderr)
        return 1
    print(f"\nAll {len(rows)} rows rendered to {out_dir}.")
    return 0


# ---------------------------------------------------------------------------
# Command: sweep — vary one or more envelope inputs over a grid
# ---------------------------------------------------------------------------

# Keep the user-facing variable names short and stable. Each entry maps
# the public ``--vary`` token to the SessionSchema attribute it perturbs.
# Mirroring the four ``run_sensitivity`` knobs keeps the user-mental-model
# consistent across CLI subcommands ("the inputs you can perturb").
_SWEEP_FIELDS = {
    "mach":   "mach",
    "alt":    "alt_km",
    "mass":   "mass_kg",
    "rn":     "R_n_m",
    "g":      "g_load",
}


def _parse_vary_spec(spec: str) -> tuple[str, float, float]:
    """Parse one ``--vary VAR:LO-HI`` token.

    Examples
    --------
    ``mach:0.9-1.1``    → ("mach", 0.9, 1.1)
    ``mass:25000-35000`` → ("mass", 25000.0, 35000.0)

    The ``-`` between LO and HI is treated as the range separator;
    if either bound itself is negative the user must use a different
    syntax (none of the supported envelope inputs accept negatives,
    so this stays unambiguous).
    """
    if ":" not in spec:
        raise ValueError(
            f"--vary {spec!r}: expected NAME:LO-HI (e.g. mach:0.9-1.1)"
        )
    name, rng = spec.split(":", 1)
    name = name.strip().lower()
    if name not in _SWEEP_FIELDS:
        raise ValueError(
            f"--vary {spec!r}: unknown variable {name!r}; "
            f"valid: {', '.join(_SWEEP_FIELDS)}"
        )
    if "-" not in rng:
        raise ValueError(
            f"--vary {spec!r}: expected LO-HI (e.g. 0.9-1.1)"
        )
    lo_str, hi_str = rng.split("-", 1)
    try:
        lo = float(lo_str)
        hi = float(hi_str)
    except ValueError as exc:
        raise ValueError(
            f"--vary {spec!r}: LO/HI must be numeric ({exc})"
        ) from None
    if hi <= lo:
        raise ValueError(
            f"--vary {spec!r}: HI ({hi}) must be > LO ({lo})"
        )
    return name, lo, hi


def cmd_sweep(ns) -> int:
    # Source: same two-mode model as cmd_run — JSON envelope OR explicit flags.
    explicit_flags = (ns.mach, ns.alt, ns.mass, ns.rn, ns.g)
    has_explicit = any(x is not None for x in explicit_flags)

    if ns.from_file and has_explicit:
        print(
            "ERROR: --from and explicit envelope flags are mutually exclusive.",
            file=sys.stderr,
        )
        return 2

    if ns.from_file:
        path = Path(ns.from_file)
        if not path.is_file():
            print(f"ERROR: JSON file not found: {path}", file=sys.stderr)
            return 2
        try:
            base_session = _load_session_from_file(path)
        except ValueError as exc:
            print(f"ERROR: could not parse {path}: {exc}", file=sys.stderr)
            return 2
    else:
        required = (
            ("--mach", ns.mach), ("--alt", ns.alt), ("--mass", ns.mass),
            ("--rn", ns.rn), ("--g", ns.g), ("--category", ns.category),
        )
        missing = [name for name, val in required if val is None]
        if missing:
            print(
                f"ERROR: missing required arguments: {', '.join(missing)}. "
                "Either pass --from FILE.json or supply all of --mach / "
                "--alt / --mass / --rn / --g / --category.",
                file=sys.stderr,
            )
            return 2
        base_session = _session_from_flags(ns)

    if not ns.vary:
        print(
            "ERROR: at least one --vary NAME:LO-HI is required "
            "(e.g. --vary mach:0.9-1.1).",
            file=sys.stderr,
        )
        return 2

    # Parse every --vary spec up-front so a typo on the third one
    # doesn't waste compute on the first two.
    try:
        var_specs = [_parse_vary_spec(s) for s in ns.vary]
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Build the linspace grid for each varied input. Cartesian product
    # gives us the scenario set; cell labels are short slugs of the
    # form "mach=0p950_mass=27500" for filename use.
    n = int(ns.samples)
    if n < 2:
        print(
            f"ERROR: --samples must be >= 2 (got {n}).", file=sys.stderr,
        )
        return 2

    import numpy as np
    grids = []
    for name, lo, hi in var_specs:
        grids.append([(name, float(v)) for v in np.linspace(lo, hi, n)])

    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = list(itertools.product(*grids))
    fail = 0
    for idx, cell in enumerate(cells):
        # Apply each (name, value) onto a fresh copy of base_session.
        kwargs = {}
        slug_parts = []
        for name, val in cell:
            attr = _SWEEP_FIELDS[name]
            kwargs[attr] = val
            # Convert decimals to a filename-safe form: 0.95 → "0p950".
            slug_parts.append(f"{name}={val:.3g}".replace(".", "p"))
        scenario = _dc_replace(base_session, **kwargs)
        cell_slug = "_".join(slug_parts)

        base_slug = _slug(base_session.system_label or f"sweep{idx:03d}")
        pdf_path = out_dir / f"matvec_{base_slug}_{cell_slug}.pdf"
        print(
            f"[{idx + 1}/{len(cells)}: {cell_slug}] rendering...",
            end=" ", flush=True,
        )
        try:
            result = run_session(scenario)
        except Exception as exc:  # noqa: BLE001 — same loose policy as cmd_batch
            print(f"FAILED: {exc}", file=sys.stderr)
            fail += 1
            continue
        if not result.pdf_bytes:
            print("FAILED (no PDF bytes)")
            fail += 1
            continue
        _write_pdf(result.pdf_bytes, pdf_path)
        print(f"OK -> {pdf_path.name}")

    if fail:
        print(f"\n{fail}/{len(cells)} scenarios failed.", file=sys.stderr)
        return 1
    print(f"\nAll {len(cells)} scenarios rendered to {out_dir}.")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

_VALID_CATEGORIES = (
    "general", "aircraft", "hypersonic_aircraft",
    "reentry", "hypersonic_missile", "turbine",
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="matvec",
        description=(
            "MATVEC - Aerospace materials feasibility (headless CLI). "
            f"Version {MATVEC_VERSION}."
        ),
    )
    p.add_argument(
        "--version", action="version",
        version=f"matvec {MATVEC_VERSION}",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- run ----
    p_run = sub.add_parser(
        "run",
        help="Run one envelope -> one PDF.",
        description=(
            "Run a single MATVEC analysis. Pass an envelope JSON via "
            "--from / positional FILE.json, or supply all of --mach/--alt/"
            "--mass/--rn/--g/--category as explicit flags."
        ),
    )
    p_run.add_argument(
        "json_positional", nargs="?", default=None,
        help="JSON envelope file (same as --from).",
    )
    p_run.add_argument(
        "--from", dest="from_file",
        help="Load envelope from JSON file.",
    )
    p_run.add_argument("--mach", type=float,
                       help="Freestream Mach number.")
    p_run.add_argument("--alt", type=float,
                       help="Cruise altitude (km).")
    p_run.add_argument("--mass", type=float,
                       help="Vehicle mass (kg).")
    p_run.add_argument("--rn", type=float,
                       help="Nose / leading-edge radius (m).")
    p_run.add_argument("--g", type=float,
                       help="Peak structural g-load.")
    p_run.add_argument(
        "--category",
        choices=_VALID_CATEGORIES, default=None,
        help="Vehicle category.",
    )
    p_run.add_argument(
        "--char-len", dest="char_len", type=float, default=None,
        help="Characteristic length (m). Default: mass-derived estimate.",
    )
    p_run.add_argument(
        "--duration", type=float, default=600.0,
        help="Flight duration (s). Default: 600.",
    )
    p_run.add_argument(
        "--emissivity", type=float, default=0.85,
        help="Wall emissivity. Default: 0.85.",
    )
    p_run.add_argument(
        "--hot-section-K", dest="hot_section_K", type=float, default=None,
        help="Turbine hot-section metal-face temperature (K). "
             "Only meaningful for --category turbine.",
    )
    p_run.add_argument(
        "--label", default=None,
        help="System label (report title page & filename slug).",
    )
    p_run.add_argument(
        "--out", required=True,
        help="Output PDF path.",
    )
    p_run.add_argument(
        "--tex-on-fail", action="store_true",
        help="If pdflatex fails, write the .tex source alongside --out.",
    )

    # ---- run: sensitivity-sweep options ----
    # All flags here are opt-in: omitting --sensitivity gives the
    # historical "physics + materials only" PDF. When --sensitivity is
    # set, the four delta knobs and --sens-samples customise the
    # SensitivitySpec (omitted knobs inherit dataclass defaults).
    p_run.add_argument(
        "--sensitivity", action="store_true",
        help="Run a sensitivity-sweep on top of the nominal pipeline; "
             "inserts a Sensitivity Analysis section into the PDF.",
    )
    p_run.add_argument(
        "--sens-mach-delta", dest="sens_mach_delta", type=float, default=None,
        help="Sensitivity sweep: +/- %% on Mach (default 10).",
    )
    p_run.add_argument(
        "--sens-mass-delta", dest="sens_mass_delta", type=float, default=None,
        help="Sensitivity sweep: +/- %% on vehicle mass (default 15).",
    )
    p_run.add_argument(
        "--sens-rn-delta", dest="sens_rn_delta", type=float, default=None,
        help="Sensitivity sweep: +/- %% on nose radius (default 20).",
    )
    p_run.add_argument(
        "--sens-g-delta", dest="sens_g_delta", type=float, default=None,
        help="Sensitivity sweep: +/- %% on peak g-load (default 25).",
    )
    p_run.add_argument(
        "--sens-samples", dest="sens_samples", type=int, default=None,
        help="Sensitivity sweep: linspace samples per input (default 11).",
    )

    # ---- validate ----
    p_val = sub.add_parser(
        "validate",
        help="Regenerate every canonical preset PDF (CI hook).",
        description=(
            "Loop through every preset in core.presets.CANONICAL_PRESETS "
            "and render a PDF for each. Non-zero exit if any preset "
            "fails to render."
        ),
    )
    p_val.add_argument(
        "--out-dir", default="reports",
        help="Output directory. Default: reports/",
    )

    # ---- batch ----
    p_bat = sub.add_parser(
        "batch",
        help="Run a CSV sweep: one row -> one PDF.",
        description=(
            "Read a CSV whose columns are the SessionSchema envelope "
            "fields (or aliases). One PDF written per row into "
            "--out-dir, named by slugified system_label + row index."
        ),
    )
    p_bat.add_argument("csv", help="CSV file with envelope columns.")
    p_bat.add_argument(
        "--out-dir", default="reports",
        help="Output directory. Default: reports/",
    )

    # ---- sweep ----
    # Same envelope-source model as 'run' (--from JSON or explicit
    # flags), plus one or more --vary NAME:LO-HI knobs. Output is
    # one PDF per cell of the cartesian product, named by a slug
    # composed from the varied values so the filenames sort cleanly.
    p_swp = sub.add_parser(
        "sweep",
        help="Vary one or more envelope inputs over a grid; one PDF per cell.",
        description=(
            "Cartesian-product sweep over user-chosen envelope inputs. "
            "Pass --vary NAME:LO-HI once per varied input (e.g. "
            "--vary mach:0.9-1.1 --vary mass:25000-35000). The base "
            "envelope can come from --from envelope.json or from "
            "explicit --mach/--alt/--mass/--rn/--g/--category flags."
        ),
    )
    p_swp.add_argument(
        "--from", dest="from_file",
        help="Load base envelope from JSON file.",
    )
    p_swp.add_argument("--mach", type=float,
                       help="Base freestream Mach number.")
    p_swp.add_argument("--alt", type=float,
                       help="Base cruise altitude (km).")
    p_swp.add_argument("--mass", type=float,
                       help="Base vehicle mass (kg).")
    p_swp.add_argument("--rn", type=float,
                       help="Base nose / leading-edge radius (m).")
    p_swp.add_argument("--g", type=float,
                       help="Base peak structural g-load.")
    p_swp.add_argument(
        "--category",
        choices=_VALID_CATEGORIES, default=None,
        help="Vehicle category.",
    )
    p_swp.add_argument(
        "--char-len", dest="char_len", type=float, default=None,
        help="Characteristic length (m). Default: mass-derived estimate.",
    )
    p_swp.add_argument(
        "--duration", type=float, default=600.0,
        help="Flight duration (s). Default: 600.",
    )
    p_swp.add_argument(
        "--emissivity", type=float, default=0.85,
        help="Wall emissivity. Default: 0.85.",
    )
    p_swp.add_argument(
        "--hot-section-K", dest="hot_section_K", type=float, default=None,
        help="Turbine hot-section metal-face temperature (K).",
    )
    p_swp.add_argument(
        "--label", default=None,
        help="System label (used as the filename slug prefix).",
    )
    p_swp.add_argument(
        "--vary", action="append", default=[],
        metavar="NAME:LO-HI",
        help="Vary an envelope input over a linspace grid. "
             "NAME is one of: mach, alt, mass, rn, g. Repeatable.",
    )
    p_swp.add_argument(
        "--samples", type=int, default=5,
        help="Linspace samples per varied input. Default: 5.",
    )
    p_swp.add_argument(
        "--out-dir", default="reports",
        help="Output directory. Default: reports/",
    )

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)

    # Promote the positional JSON filename to --from for 'run'.
    if ns.cmd == "run" and getattr(ns, "json_positional", None):
        if ns.from_file:
            print(
                "ERROR: pass either a positional JSON path or --from, "
                "not both.",
                file=sys.stderr,
            )
            return 2
        ns.from_file = ns.json_positional

    if ns.cmd == "run":
        return cmd_run(ns)
    if ns.cmd == "validate":
        return cmd_validate(ns)
    if ns.cmd == "batch":
        return cmd_batch(ns)
    if ns.cmd == "sweep":
        return cmd_sweep(ns)
    parser.error(f"unknown command: {ns.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
