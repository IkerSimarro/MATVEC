"""Generate the bundled example presets/*.json files.

Walks a curated subset of ``CANONICAL_PRESETS`` (the 9 audience-
friendly envelopes + 1 aerospace reference anchor), embeds a per-
preset description on the new ``SessionSchema.notes`` field, and
writes one JSON file per entry to ``presets/``.

Idempotent: re-running overwrites existing files with byte-identical
content (json.dumps uses sort_keys+indent=2 → stable diffs). Safe to
commit the generated JSON files; safe to re-run after editing
``CANONICAL_PRESETS`` (just commit the resulting diff).

Run from the project root:

    python scripts/generate_example_presets.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

# Allow running from anywhere — resolve project root from this script's path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.presets import CANONICAL_PRESETS  # noqa: E402
from core.session import session_to_json     # noqa: E402

PRESETS_DIR = _PROJECT_ROOT / "presets"

# Curated UI-facing subset. Each entry maps:
#   canonical_preset_key  -> (display_label, notes_text)
# The display_label becomes the SessionSchema.system_label written into
# the JSON (so the UI dropdown shows it as that label). canonical_preset_key
# is the key into CANONICAL_PRESETS — the envelope numbers come from there.
#
# Adding a new bundled example: add an entry to CANONICAL_PRESETS in
# core/presets.py (so the CLI / tests can reach it), then add an entry
# here, then re-run this script.
BUNDLED_PRESETS: dict[str, tuple[str, str]] = {
    # Three bundled presets: two validation-anchored (SR-71, Concorde
    # --- envelope numbers match VALIDATION_CASES in
    # scripts/run_validation.py byte-for-byte) plus one audience-
    # relevant (Collegiate Sounding Rocket --- IREC competition class,
    # included for university-team relevance rather than historical
    # calibration).
    "SR-71 Blackbird": (
        "SR-71 Blackbird",
        "Lockheed SR-71 reconnaissance aircraft. Sustained "
        "Mach 3.2 cruise at high altitude --- a classic high-speed "
        "airframe useful for seeing how the tool handles long-"
        "duration supersonic heating.",
    ),
    "Collegiate Sounding Rocket": (
        "Collegiate Sounding Rocket",
        "IREC / Spaceport America Cup competition rocket. Mach 2 "
        "boost-coast trajectory to about 9 km apogee, single-flight "
        "student airframe. Useful for amateur and university "
        "rocketry projects.",
    ),
    "Concorde": (
        "Concorde",
        "Aerospatiale-BAC Concorde supersonic airliner. Sustained "
        "Mach 2 trans-Atlantic cruise. Useful for seeing how a "
        "long-lifetime commercial-aviation envelope screens against "
        "sustained heating and creep.",
    ),
}


def slugify(name: str) -> str:
    """Turn 'Consumer FPV Drone' into 'consumer_fpv_drone' for filenames."""
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "/"):
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def main() -> int:
    if not PRESETS_DIR.exists():
        PRESETS_DIR.mkdir()
        print(f"created {PRESETS_DIR.relative_to(_PROJECT_ROOT)}/")

    written = 0
    skipped: list[str] = []
    for canonical_key, (display_label, notes_text) in BUNDLED_PRESETS.items():
        if canonical_key not in CANONICAL_PRESETS:
            skipped.append(canonical_key)
            continue
        schema = CANONICAL_PRESETS[canonical_key]
        # Override the display label + embed the notes. Using
        # dataclasses.replace so we don't mutate the canonical entry.
        bundled = replace(
            schema,
            system_label=display_label,
            notes=notes_text,
        )
        slug = slugify(display_label)
        path = PRESETS_DIR / f"{slug}.json"
        path.write_text(session_to_json(bundled), encoding="utf-8")
        print(f"  wrote presets/{path.name}")
        written += 1

    if skipped:
        print(
            f"WARNING: {len(skipped)} bundled-preset key(s) not found in "
            f"CANONICAL_PRESETS: {skipped}",
            file=sys.stderr,
        )
        return 1

    print(f"\n{written} preset file(s) written to {PRESETS_DIR.relative_to(_PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
