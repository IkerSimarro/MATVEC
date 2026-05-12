"""
MATVEC — Canonical session schema and JSON round-trip.

A SessionSchema is a plain-data record of everything the physics pipeline
needs to reproduce a MATVEC analysis: the flight envelope, the vehicle
category, the report's display label, and an open ``options`` dict for
category-specific toggles (e.g. turbine hot-section temperature).

Intended consumers:
  * CLI — ``python -m matvec run envelope.json``
  * Streamlit UI — Download / Load Session JSON buttons
  * CI regression harness — commit one JSON per preset, diff over time
  * Future Metric-Standard repo — physics-core library consumes
    SessionSchema directly without needing MATVEC's UI layer

Deliberately stdlib-only (no pydantic, no pandas, no streamlit) so the
schema module stays importable in lean / CLI-only / CI environments.

Forward-compat contract
-----------------------
* Loading a JSON with a newer ``matvec_version`` emits a ``UserWarning``
  but still attempts to construct the SessionSchema. A hard version gate
  would break more than it protects — most version bumps are semantic
  labels, not schema-breaking edits.
* Unknown top-level fields are silently dropped on load (a newer file
  carrying a field this build doesn't understand yet is not an error).
* Unknown keys inside ``options`` are preserved verbatim through
  round-trip, so a future toggle that only the newer runtime knows how
  to act on still survives a download → upload cycle under the older
  runtime.
* Missing *required* envelope fields raise ``ValueError`` with the
  offending field name — that's an authoring bug, not a version gap.
"""

import json
import warnings
from dataclasses import dataclass, field, asdict

from core import MATVEC_VERSION


# Canonical envelope field names. Every consumer (CLI, Streamlit,
# regression snapshots) must agree on this exact set — a tuple so that
# a typo at lookup time raises instead of silently missing.
_REQUIRED_ENVELOPE_FIELDS = (
    "mach",
    "alt_km",
    "mass_kg",
    "R_n_m",
    "g_load",
    "char_len_m",
    "flight_duration_s",
    "wall_emissivity",
)

# Top-level required keys beyond the envelope.
_REQUIRED_TOP_LEVEL = ("vehicle_category", "system_label")


@dataclass
class SessionSchema:
    """Canonical envelope + metadata for a MATVEC analysis.

    Envelope fields mirror ``physics_engine.run_analysis`` keyword
    arguments but use explicit unit-suffixed names (e.g. ``R_n_m``,
    ``char_len_m``) so downstream code never has to guess whether a
    number is in metres, millimetres, or km.

    ``options`` is reserved for category-specific inputs that only
    apply to a subset of analyses — currently only
    ``hot_section_temp_K`` (turbine category). Unknown option keys
    round-trip so future toggles do not require a schema version bump.
    """

    # ---- Flight envelope ----
    mach: float
    alt_km: float
    mass_kg: float
    R_n_m: float
    g_load: float
    char_len_m: float
    flight_duration_s: float = 600.0
    wall_emissivity: float = 0.85
    # Total airframe / component design lifetime in hours. Distinct from
    # ``flight_duration_s`` (per-mission cruise duration). Drives the
    # creep-evaluation stage in the matching engine: at lifetimes much
    # greater than a single flight, materials must survive sustained
    # (T_wall, sigma_required) without creep-rupture, not just the
    # static thermal+structural snapshot. Default 1.0 hour preserves
    # pre-creep-feature behavior (creep evaluation is essentially a
    # no-op below ~1000 hours for most materials).
    design_lifetime_hours: float = 1.0
    # Representative through-thickness panel thickness in metres for
    # the 1D transient heat solver (Phase 7). Default 2 mm models a
    # typical aerospace thin-skin panel (1-3 mm aluminium / titanium /
    # CFRP). The solver integrates the heat equation through this
    # thickness from the convectively-heated surface to the insulated
    # back face; the back-face peak temperature drives the transient
    # thermal screen for short-duration flights. Per-preset values
    # override the default in ``CANONICAL_PRESETS``.
    panel_thickness_m: float = 0.002
    # Optional time-series flight profile: ``((t_s, mach, alt_km), ...)``
    # sorted ascending by time. When non-empty, the transient heat
    # solver integrates the convective heat flux along the profile
    # rather than holding the steady-state design point for the full
    # ``flight_duration_s``. Empty tuple (the default) means "no
    # profile supplied — fall back to constant-condition exposure
    # at the design point." Profile sample format chosen to round-
    # trip cleanly through JSON (lists of length-3 lists).
    flight_profile: tuple = ()

    # ---- Metadata ----
    vehicle_category: str = "general"
    system_label: str = "Custom Analysis"
    # Optional human-readable description of what this envelope represents.
    # Surfaced in the UI as the "About this preset" caption when a bundled
    # example is loaded, and embedded in the bundled presets/*.json files
    # so the description lives with the data instead of in a separate Python
    # dict. Empty string is the default; older session JSONs without this
    # field load fine (it's added to the known-fields allowlist below).
    notes: str = ""
    options: dict = field(default_factory=dict)
    matvec_version: str = MATVEC_VERSION


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def session_to_dict(session: SessionSchema) -> dict:
    """Serialize a SessionSchema into a JSON-ready plain dict.

    ``dataclasses.asdict`` recurses into ``options``, so any nested
    structures (currently just scalars) survive intact.
    """
    return asdict(session)


def dict_to_session(d: dict) -> SessionSchema:
    """Deserialize a plain dict into a SessionSchema.

    * Missing required fields → ``ValueError`` naming the field.
    * Unknown top-level fields → dropped silently (forward-compat).
    * ``options`` → copied verbatim (unknown keys survive).
    * ``matvec_version`` mismatches → ``UserWarning``, no exception.
    """
    if not isinstance(d, dict):
        raise ValueError(
            f"SessionSchema payload must be a dict, got {type(d).__name__}"
        )

    for fname in _REQUIRED_ENVELOPE_FIELDS:
        if fname not in d:
            raise ValueError(
                f"SessionSchema missing required envelope field '{fname}'. "
                f"Expected envelope fields: {_REQUIRED_ENVELOPE_FIELDS}"
            )
    for fname in _REQUIRED_TOP_LEVEL:
        if fname not in d:
            raise ValueError(
                f"SessionSchema missing required field '{fname}'."
            )

    file_version = d.get("matvec_version", None)
    if file_version is not None and file_version != MATVEC_VERSION:
        warnings.warn(
            f"SessionSchema matvec_version={file_version!r} does not match "
            f"current MATVEC_VERSION={MATVEC_VERSION!r}; loading anyway. "
            "Results are comparable if physics constants are unchanged — "
            "see HANDOFF.md §5 for the list of settled constants.",
            UserWarning,
            stacklevel=2,
        )

    # Known-field allow-list — everything else (from a newer runtime)
    # is dropped silently rather than forwarded as surprise state.
    known_fields = (
        set(_REQUIRED_ENVELOPE_FIELDS)
        | set(_REQUIRED_TOP_LEVEL)
        | {
            "options", "matvec_version", "notes",
            "design_lifetime_hours",
            # Phase 7 transient-heat additions:
            "panel_thickness_m", "flight_profile",
        }
    )
    payload = {k: v for k, v in d.items() if k in known_fields}

    options = payload.get("options", {})
    if not isinstance(options, dict):
        raise ValueError(
            f"SessionSchema 'options' must be a dict, got "
            f"{type(options).__name__}"
        )

    # Coerce envelope fields to float so a hand-authored JSON that uses
    # integer literals ("mass_kg": 500) doesn't bleed int arithmetic into
    # the physics engine downstream.
    numeric = {f: float(payload[f]) for f in _REQUIRED_ENVELOPE_FIELDS}

    # Phase 7 transient-heat: panel_thickness_m and flight_profile are
    # both optional (defaulted). The profile is a sequence of
    # (t_s, mach, alt_km) samples; JSON round-trips it as list-of-lists
    # so we coerce back to a tuple-of-tuples to match the dataclass
    # annotation and to preserve immutability on the SessionSchema.
    raw_profile = payload.get("flight_profile", ())
    if raw_profile and not isinstance(raw_profile, (list, tuple)):
        raise ValueError(
            f"SessionSchema 'flight_profile' must be a sequence of "
            f"(t_s, mach, alt_km) triples, got {type(raw_profile).__name__}"
        )
    coerced_profile = tuple(
        (float(t), float(m), float(a))
        for sample in raw_profile
        for (t, m, a) in (tuple(sample),)
    ) if raw_profile else ()

    return SessionSchema(
        **numeric,
        vehicle_category=str(payload["vehicle_category"]),
        system_label=str(payload["system_label"]),
        notes=str(payload.get("notes", "")),
        # design_lifetime_hours is optional (defaulted) so older session
        # JSONs without the field load cleanly. Coerced to float to match
        # the envelope-field convention.
        design_lifetime_hours=float(
            payload.get("design_lifetime_hours", 1.0)
        ),
        panel_thickness_m=float(
            payload.get("panel_thickness_m", 0.002)
        ),
        flight_profile=coerced_profile,
        options=dict(options),
        matvec_version=str(payload.get("matvec_version", MATVEC_VERSION)),
    )


def session_to_json(session: SessionSchema) -> str:
    """Serialize to a pretty-printed, sorted, 2-space indented JSON string.

    Sorted keys make diffs stable across writes — important for the
    CI regression-snapshot workflow where one-character churn in a
    committed preset JSON would flag every run.
    """
    return json.dumps(session_to_dict(session), indent=2, sort_keys=True)


def json_to_session(s: str) -> SessionSchema:
    """Parse a JSON string into a SessionSchema.

    Wraps ``json.JSONDecodeError`` into a ``ValueError`` so callers only
    need one exception type for all malformed-input cases.
    """
    try:
        payload = json.loads(s)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    return dict_to_session(payload)
