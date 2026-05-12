# MATVEC ML components

# Single source of truth for the MATVEC release version.
#
# This string is consumed by:
#   * core.session.SessionSchema — written into every JSON round-trip
#     record so forward/backward-compat loaders can warn on mismatch.
#   * core.api.run_session — included in the LaTeX report's repro block
#     (indirectly, via the SessionSchema -> report pipeline).
#   * matvec.__main__ — the CLI prints it on --version.
#
# Bump semantics (follow SemVer):
#   * PATCH (1.0.X) — docs / disclaimers / UX-only changes.
#   * MINOR (1.X.0) — new features, new physics branches, new outputs —
#     must stay backward-compatible with older JSON sessions.
#   * MAJOR (X.0.0) — physics-constant changes, schema field removal,
#     or any edit to the "settled" table in HANDOFF.md §5.
#
# The forward-compat contract in core.session treats matvec_version
# mismatches as a warning, not an error — a MINOR bump shipping while a
# user still has a JSON from last week is the expected case, not a
# failure mode.
MATVEC_VERSION = "1.0.0"
