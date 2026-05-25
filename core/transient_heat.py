"""
core/transient_heat.py — 1D transient heat conduction solver.

Phase 7.2 of the transient-heat rollout. Integrates the 1D heat
equation through a panel of given thickness, with a convective heat
flux applied to the hot face and an insulated cold face (worst-case
upper bound on internal soak). The solver answers the engineering
question this tool exists to answer:

    Given a material with thermal diffusivity α, panel thickness L,
    and a time-varying convective heat flux q(t) at the surface, what
    is the peak temperature reached at the back face (internal
    substructure) over the flight?

The peak back-face temperature is the screening value used for
short-duration flights, replacing the steady-state recovery
temperature that the existing thermal stage uses. For sustained
flights the peak back-face converges to the steady-state surface
temperature and the substitution is a no-op.

Math
----

The 1D heat equation through panel thickness x ∈ [0, L]:

    ∂T/∂t = α · ∂²T/∂x²

with:
  * **Hot face (x = 0)**: convective + radiative balance
      k · ∂T/∂x|_{x=0} = q_conv(t) - ε · σ · T(0,t)^4 + ε · σ · T_amb^4
    where q_conv(t) is the time-varying convective flux supplied by
    the upstream Sutton-Graves / recovery-temperature physics.
  * **Cold face (x = L)**: insulated (worst case)
      ∂T/∂x|_{x=L} = 0

Spatial discretisation
----------------------

Uniform grid with N nodes; node 0 is the hot face, node N-1 is the
cold face. Δx = L / (N - 1). The interior stencil is the standard
second-order central difference; the boundary nodes use ghost-point
elimination to incorporate the flux BCs without losing accuracy.

Time integration
----------------

Two schemes available, with automatic selection:

* **Explicit forward-Euler** (default for short flights, large Δx,
  diffusive materials): requires Fourier number Fo = α·Δt / Δx² ≤ 0.5
  for stability. Cheap per step, but the stability constraint forces
  millions of steps for long flights or thin panels.

* **Crank-Nicolson implicit** (default for long flights, fine grid,
  low-α materials): unconditionally stable, second-order accurate
  in time. Requires solving a tridiagonal system per step but allows
  arbitrary Δt.

`method="auto"` picks based on the estimated step count: stick with
explicit when it would take ≤ 10⁵ steps, fall back to implicit
otherwise.

Reference: Anderson, *Hypersonic and High-Temperature Gas Dynamics*,
2nd ed., Ch. 6 for the convective + radiative boundary condition;
Patankar, *Numerical Heat Transfer and Fluid Flow*, Ch. 4 for the
finite-difference formulation. Crank & Nicolson, *A practical method
for numerical evaluation of solutions of partial differential
equations of the heat-conduction type* (1947) for the implicit
scheme.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .materials_db import MaterialEntry
from .physics_engine import (
    GAMMA_AIR, R_AIR, SIGMA_SB, C_SUTTON_GRAVES, RECOVERY_FACTOR,
    _isa_atmosphere,
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

EXPLICIT_STEP_BUDGET = 100_000
"""Above this many estimated explicit time-steps the solver switches
to Crank-Nicolson. Below it, explicit forward-Euler is cheap enough
and avoids the tridiagonal-solve overhead."""

DEFAULT_N_NODES = 5
"""Default through-thickness discretisation. 5 nodes give 4 cells
across the panel — coarse but adequate for screening-grade peak-
temperature calculation, and ~6x faster than 11 nodes (explicit
step count scales with dx²). For higher fidelity, callers can
pass ``n_nodes=11`` or more."""

FOURIER_STABILITY_LIMIT = 0.45
"""Explicit forward-Euler stability requires Fo = α·Δt/Δx² < 0.5.
We use 0.45 for a 10 % safety margin against round-off accumulation."""


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TransientHeatResult:
    """Result of a 1D transient heat integration through a single panel.

    Time-series fields are stored as tuples so the result is safely
    immutable. ``peak_*`` fields are the screening values the matching
    engine reads. ``status`` indicates whether the integration ran or
    was skipped:

    * ``"applied"``        — full integration completed.
    * ``"not_applicable"`` — material category does not classically
                             conduct (TPS ablator, ABS / Nylon
                             polymer). The solver short-circuits with
                             the steady-state recovery temperature
                             so the matching engine has *something* to
                             screen against, but the result is flagged.
    * ``"unknown"``        — material has no sourced c_p. Solver
                             skips with a flag; matching engine
                             treats as pass-through.
    """
    status: str
    time_s: tuple = ()
    T_surface_K: tuple = ()
    T_midpoint_K: tuple = ()
    T_backface_K: tuple = ()
    peak_surface_K: float | None = None
    peak_midpoint_K: float | None = None
    peak_backface_K: float | None = None
    time_at_peak_backface_s: float | None = None
    method_used: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def integrate_panel(
    material: MaterialEntry,
    panel_thickness_m: float,
    flight_profile: Sequence[tuple[float, float, float]] = (),
    R_n_m: float = 0.30,
    wall_emissivity: float = 0.85,
    *,
    flight_duration_s: float | None = None,
    fallback_mach: float = 0.0,
    fallback_alt_km: float = 0.0,
    n_nodes: int = DEFAULT_N_NODES,
    method: str = "auto",
) -> TransientHeatResult:
    """Integrate the 1D heat equation through a panel under a
    time-varying convective heat flux dictated by the flight profile.

    Parameters
    ----------
    material : MaterialEntry
        Provides ρ (``density_kgm3``), k (``thermal_conductivity_WmK``),
        and c_p (``specific_heat_J_kgK``).
    panel_thickness_m : float
        Hot-face-to-cold-face thickness, in metres. Typical aerospace
        thin-skin panels are 1-3 mm.
    flight_profile : sequence of (t_s, mach, alt_km)
        Time-series sampled at user-chosen times. The solver
        interpolates linearly between samples. An empty sequence
        falls back to constant exposure at ``fallback_mach`` /
        ``fallback_alt_km`` for ``flight_duration_s``.
    R_n_m : float
        Stagnation-point nose radius, drives Sutton-Graves heat flux.
    wall_emissivity : float
        Surface emissivity for radiative re-emission at the hot face.
    flight_duration_s : float or None
        Required when the profile is empty. The simulation runs from
        t=0 to t=flight_duration_s with constant conditions.
    fallback_mach, fallback_alt_km : float
        Steady-state design point used when no profile is supplied.
    n_nodes : int
        Through-thickness discretisation. Default 11.
    method : str
        ``"explicit"`` / ``"implicit"`` / ``"auto"`` (default).
    """
    # Short-circuit: TPS / polymer materials don't run the solver.
    if material.cp_data_status == "not_applicable":
        return TransientHeatResult(
            status="not_applicable",
            notes=(
                "Material category (TPS / engineering polymer) does "
                "not use classical conduction screening. Skipping "
                "transient solve."
            ),
        )
    if (
        material.cp_data_status == "unknown"
        or material.specific_heat_J_kgK is None
    ):
        return TransientHeatResult(
            status="unknown",
            notes=(
                "No c_p sourced for this material; transient heat "
                "solver cannot run. Matching engine will pass through."
            ),
        )

    # Validate inputs.
    if panel_thickness_m <= 0.0:
        raise ValueError(
            f"panel_thickness_m must be > 0, got {panel_thickness_m}"
        )
    if n_nodes < 3:
        raise ValueError(
            f"n_nodes must be >= 3 (need at least surface / midpoint / "
            f"backface), got {n_nodes}"
        )

    # Build the working profile. Empty → flatten to a constant point
    # for ``flight_duration_s``.
    profile = _coerce_profile(
        flight_profile, flight_duration_s, fallback_mach, fallback_alt_km
    )
    t_total = profile[-1][0]
    if t_total <= 0.0:
        return TransientHeatResult(
            status="not_applicable",
            notes="Flight duration is zero; nothing to integrate.",
        )

    # Material thermal properties.
    rho = float(material.density_kgm3)
    k   = float(material.thermal_conductivity_WmK)
    cp  = float(material.specific_heat_J_kgK)
    alpha = k / (rho * cp)

    # Spatial discretisation.
    L = float(panel_thickness_m)
    N = int(n_nodes)
    dx = L / (N - 1)

    # Choose initial Δt and method. The explicit stability limit
    # gives Δt_max = Fo_max × dx² / α.
    dt_explicit = FOURIER_STABILITY_LIMIT * dx * dx / alpha
    n_steps_explicit = int(math.ceil(t_total / dt_explicit))

    if method == "auto":
        chosen_method = (
            "explicit" if n_steps_explicit <= EXPLICIT_STEP_BUDGET
            else "implicit"
        )
    else:
        chosen_method = method

    # Pick the actual Δt. Explicit must respect stability; implicit
    # picks a coarser Δt that resolves the surface flux variation
    # but trims step count.
    if chosen_method == "explicit":
        dt = dt_explicit
    else:
        # Implicit is unconditionally stable, but BE's accuracy
        # degrades when dt × panel_heat_rate is comparable to the
        # equilibrium ΔT (radiative linearisation breaks down). Cap
        # Δt to keep each step's expected temperature rise modest —
        # 2000 steps is a screening-grade compromise that runs in
        # well under a second per material per flight.
        dt = max(t_total / 2000.0, 0.05)

    # Initial condition: panel at ambient temperature at t=0.
    # _isa_atmosphere returns (T_K, P_Pa, rho_kg_m3).
    T_amb0 = _isa_atmosphere(profile[0][2])[0]
    T = [T_amb0] * N

    # Storage for time-series (downsample to ≤500 saved points so the
    # result dataclass stays light).
    history_t: list[float] = []
    history_surf: list[float] = []
    history_mid: list[float] = []
    history_back: list[float] = []
    save_every = max(1, int(math.ceil(t_total / dt / 500.0)))

    t = 0.0
    step = 0
    peak_surface = -math.inf
    peak_mid     = -math.inf
    peak_back    = -math.inf
    t_at_peak_back = 0.0

    mid_index = N // 2
    back_index = N - 1

    while t < t_total - 0.5 * dt:
        # Interpolate the flight profile to the current time.
        mach, alt_km = _interp_profile(profile, t)
        T_amb, _p_atm, rho_atm = _isa_atmosphere(alt_km)
        q_cold = _stagnation_heat_flux_coldwall(rho_atm, mach, R_n_m, T_amb)
        T_rec  = _recovery_temp(T_amb, mach)
        # Hot-wall correction: q_conv depends on current surface
        # temperature, so we recompute it each step against the
        # current T_surf.
        q_conv = _hot_wall_heat_flux(q_cold, T[0], T_rec, T_amb)

        # Time-step the panel by one Δt using the chosen scheme.
        if chosen_method == "explicit":
            T = _step_explicit(
                T, dt, dx, alpha, rho, cp, k,
                q_conv, T_amb, wall_emissivity,
            )
        else:
            T = _step_implicit(
                T, dt, dx, alpha, rho, cp, k,
                q_conv, T_amb, wall_emissivity,
            )

        t += dt
        step += 1

        # Track peaks every step.
        if T[0] > peak_surface:
            peak_surface = T[0]
        if T[mid_index] > peak_mid:
            peak_mid = T[mid_index]
        if T[back_index] > peak_back:
            peak_back = T[back_index]
            t_at_peak_back = t

        # Save downsampled history.
        if step % save_every == 0:
            history_t.append(t)
            history_surf.append(T[0])
            history_mid.append(T[mid_index])
            history_back.append(T[back_index])

    notes_parts: list[str] = []
    if chosen_method == "explicit" and n_steps_explicit > EXPLICIT_STEP_BUDGET:
        notes_parts.append(
            "Explicit method chosen despite high step count "
            f"({n_steps_explicit:,}); consider implicit for "
            "faster integration."
        )
    if material.cp_data_status == "estimated":
        notes_parts.append(
            "c_p value is engineering-estimated, not from a "
            "primary reference; treat peak temperatures as indicative."
        )

    return TransientHeatResult(
        status="applied",
        time_s=tuple(history_t),
        T_surface_K=tuple(history_surf),
        T_midpoint_K=tuple(history_mid),
        T_backface_K=tuple(history_back),
        peak_surface_K=peak_surface,
        peak_midpoint_K=peak_mid,
        peak_backface_K=peak_back,
        time_at_peak_backface_s=t_at_peak_back,
        method_used=chosen_method,
        notes=" ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Internals — profile handling, heat flux, step kernels
# ---------------------------------------------------------------------------

def _coerce_profile(
    flight_profile: Sequence[tuple[float, float, float]],
    flight_duration_s: float | None,
    fallback_mach: float,
    fallback_alt_km: float,
) -> tuple[tuple[float, float, float], ...]:
    """Return a sanitised (t, mach, alt) tuple-of-tuples. Validates
    that the time axis is non-decreasing and non-empty. When the
    caller passed no profile, fabricate a two-point constant profile
    from the fallback design point."""
    if not flight_profile:
        if flight_duration_s is None or flight_duration_s <= 0.0:
            raise ValueError(
                "Either flight_profile or flight_duration_s>0 is required."
            )
        return (
            (0.0, fallback_mach, fallback_alt_km),
            (float(flight_duration_s), fallback_mach, fallback_alt_km),
        )
    out = tuple(
        (float(t), float(m), float(a))
        for sample in flight_profile
        for (t, m, a) in (tuple(sample),)
    )
    times = [s[0] for s in out]
    if times != sorted(times):
        raise ValueError(
            f"flight_profile times must be non-decreasing; got {times}"
        )
    return out


def _interp_profile(
    profile: Sequence[tuple[float, float, float]], t: float
) -> tuple[float, float]:
    """Piecewise-linear interpolation of mach + alt at time t."""
    if t <= profile[0][0]:
        return profile[0][1], profile[0][2]
    if t >= profile[-1][0]:
        return profile[-1][1], profile[-1][2]
    for i in range(1, len(profile)):
        if t <= profile[i][0]:
            t0, m0, a0 = profile[i - 1]
            t1, m1, a1 = profile[i]
            f = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
            return m0 + f * (m1 - m0), a0 + f * (a1 - a0)
    return profile[-1][1], profile[-1][2]


def _stagnation_heat_flux_coldwall(
    rho_atm: float, mach: float, R_n_m: float, T_amb: float
) -> float:
    """Sutton-Graves convective stagnation heat flux for a cold wall
    (T_wall ≈ T_amb), W/m². The published Sutton-Graves correlation
    is the cold-wall limit; for a hot wall it must be corrected by
    ``q_hot = q_cold × (T_recovery - T_wall) / (T_recovery - T_amb)``
    so the flux goes to zero as the wall approaches T_recovery
    (otherwise the panel heats past T_recovery indefinitely)."""
    if mach <= 0.0 or rho_atm <= 0.0 or R_n_m <= 0.0:
        return 0.0
    a_ms = math.sqrt(GAMMA_AIR * R_AIR * T_amb)
    v_ms = mach * a_ms
    return C_SUTTON_GRAVES * math.sqrt(rho_atm / R_n_m) * v_ms ** 3


def _hot_wall_heat_flux(
    q_cold: float, T_wall: float, T_recovery: float, T_amb: float
) -> float:
    """Apply the hot-wall correction.

    Strictly this is an enthalpy ratio (h_recovery - h_wall) /
    (h_recovery - h_amb); for ideal gas with constant c_p that
    collapses to a simple temperature ratio. Required physics so
    the panel surface asymptotes to T_recovery rather than running
    away to the radiative-only equilibrium (which is significantly
    hotter at every Mach we care about)."""
    if T_recovery <= T_amb:
        return 0.0
    ratio = (T_recovery - T_wall) / (T_recovery - T_amb)
    if ratio <= 0.0:
        return 0.0
    return q_cold * ratio


def _recovery_temp(T_amb: float, mach: float) -> float:
    """Recovery / adiabatic-wall temperature with turbulent BL
    recovery factor r=0.85. Used as the asymptotic surface
    temperature the solver radiates toward."""
    if mach <= 0.0 or T_amb <= 0.0:
        return T_amb
    return T_amb * (
        1.0 + RECOVERY_FACTOR * (GAMMA_AIR - 1.0) / 2.0 * mach * mach
    )


def _step_explicit(
    T: list[float], dt: float, dx: float, alpha: float,
    rho: float, cp: float, k: float,
    q_conv: float, T_amb: float, epsilon: float,
) -> list[float]:
    """One forward-Euler step. Returns new temperature array.

    Hot face (node 0) uses a ghost-node Neumann BC: the surface
    boundary flux q_conv (after hot-wall correction by the caller)
    minus radiative loss is folded into the heat-equation stencil
    as a source term on the surface half-cell. Cold face (node N-1)
    is insulated (symmetric ghost node).
    """
    N = len(T)
    T_new = list(T)
    r = alpha * dt / (dx * dx)
    # Interior nodes: standard explicit FTCS.
    for i in range(1, N - 1):
        T_new[i] = T[i] + r * (T[i + 1] - 2.0 * T[i] + T[i - 1])
    # Hot face: ghost-node trick. The BC is:
    #   k ∂T/∂x|_{x=0} = ε σ T_0^4 - ε σ T_amb^4 - q_conv
    # (heat flux INTO the panel = q_conv - radiative losses)
    # The central-difference second derivative at node 0 with the
    # ghost-node substitution gives:
    #   ∂²T/∂x²|_0 = (2/dx²)(T_1 - T_0) + (2/(k·dx))(q_conv - ε σ T_0^4 + ε σ T_amb^4)
    # So the explicit update is:
    F_surf = q_conv - epsilon * SIGMA_SB * T[0] ** 4 + epsilon * SIGMA_SB * T_amb ** 4
    T_new[0] = T[0] + r * 2.0 * (T[1] - T[0]) + (2.0 * dt / (rho * cp * dx)) * F_surf
    # Cold face: insulated → symmetric ghost (T_{N} = T_{N-2}).
    T_new[N - 1] = T[N - 1] + r * 2.0 * (T[N - 2] - T[N - 1])
    return T_new


def _step_implicit(
    T: list[float], dt: float, dx: float, alpha: float,
    rho: float, cp: float, k: float,
    q_conv: float, T_amb: float, epsilon: float,
) -> list[float]:
    """One Backward-Euler step with surface as a fully-coupled node.

    All N nodes (surface, interior, cold face) are unknowns in a
    single tridiagonal system. The hot-face BC is a ghost-node
    Neumann condition with the convective + radiative surface flux
    folded in as a source term on the surface half-cell. The
    radiative term (T_0^4) is iteratively linearised around the
    current best estimate of T_new[0] via Picard iteration — a single
    Taylor-expansion around the OLD T[0] is too inaccurate when the
    surface heats substantially in one step, so we relinearise 3-4
    times until convergence.

    Backward-Euler chosen over Crank-Nicolson because BE is
    unconditionally L-stable: it damps high-frequency modes correctly
    at large Fourier numbers (Fo = α·dt/dx² >> 1), which CN does not.
    For long-duration flights with a fine grid, r can exceed 10⁴ and
    CN's second-order accuracy is dwarfed by oscillation / accuracy
    failures; BE is first-order but produces physically sensible
    results across the full r range.
    """
    N = len(T)
    r = alpha * dt / (dx * dx)
    s = 2.0 * dt / (rho * cp * dx)  # surface-forcing scaling

    # Picard outer loop on the nonlinear radiative term. Initial
    # linearisation point is T[0]^n; each iteration uses the latest
    # estimate of T_new[0]. 4 iterations are sufficient for the
    # screening accuracy we target.
    T_lin = T[0]
    T_new = list(T)
    for _picard in range(4):
        rad_slope = 4.0 * epsilon * SIGMA_SB * T_lin ** 3
        F_at_Tlin = (
            q_conv
            - epsilon * SIGMA_SB * T_lin ** 4
            + epsilon * SIGMA_SB * T_amb ** 4
        )
        # Surface flux: F(T_new) ≈ F_at_Tlin - rad_slope × (T_new - T_lin)
        # BE update:
        #   (T_new[0] - T[0]) = 2r(T_new[1] - T_new[0])
        #                    + s × [F_at_Tlin - rad_slope (T_new[0] - T_lin)]
        # =>  (1 + 2r + s·rad_slope) T_new[0] - 2r T_new[1]
        #     = T[0] + s × (F_at_Tlin + rad_slope × T_lin)

        a = [0.0] * N
        b = [0.0] * N
        c = [0.0] * N
        d = [0.0] * N

        a[0] = 0.0
        b[0] = 1.0 + 2.0 * r + s * rad_slope
        c[0] = -2.0 * r
        d[0] = T[0] + s * (F_at_Tlin + rad_slope * T_lin)

        for i in range(1, N - 1):
            a[i] = -r
            b[i] = 1.0 + 2.0 * r
            c[i] = -r
            d[i] = T[i]

        a[N - 1] = -2.0 * r
        b[N - 1] = 1.0 + 2.0 * r
        c[N - 1] = 0.0
        d[N - 1] = T[N - 1]

        for i in range(1, N):
            m = a[i] / b[i - 1]
            b[i] = b[i] - m * c[i - 1]
            d[i] = d[i] - m * d[i - 1]
        T_new = [0.0] * N
        T_new[-1] = d[-1] / b[-1]
        for i in range(N - 2, -1, -1):
            T_new[i] = (d[i] - c[i] * T_new[i + 1]) / b[i]

        # Picard convergence check on the surface temperature.
        if abs(T_new[0] - T_lin) < 0.5:
            break
        T_lin = T_new[0]

    return T_new
