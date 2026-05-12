"""
MATVEC programmatic entry point.

``python -m matvec ...`` dispatches to ``matvec/__main__.py``. Everything
importable here is a re-export of the stable API in ``core/`` so
third-party scripts can do:

    from matvec import run_session, SessionSchema, json_to_session
    session = json_to_session(Path("envelope.json").read_text())
    result  = run_session(session)

No streamlit imports — this package is intentionally lightweight so it
can be installed / imported in CLI-only environments.
"""

from core import MATVEC_VERSION
from core.api import run_session, SessionResult, apply_turbine_override
from core.session import (
    SessionSchema,
    session_to_dict,
    dict_to_session,
    session_to_json,
    json_to_session,
)
from core.presets import CANONICAL_PRESETS

__all__ = [
    "MATVEC_VERSION",
    "run_session",
    "SessionResult",
    "apply_turbine_override",
    "SessionSchema",
    "session_to_dict",
    "dict_to_session",
    "session_to_json",
    "json_to_session",
    "CANONICAL_PRESETS",
]
