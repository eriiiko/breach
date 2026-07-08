"""
Shared EOS substrate for the rung-A and rung-B solvers (spike, P0.5).

ONE source of truth for the air-density field `N` and the derived pressure
`P = C * N * T`, so BOTH rungs start from an IDENTICAL air state and the
A-vs-B comparison is fair. The control (semi-Lagrangian Stable-Fluids) is
incompressible / constant-density and does NOT use this module.

Deliberately tiny: only the fairness-critical shared state lives here. Each
solver owns its own advection / pressure-solve internals.
"""

import numpy as np

from state import State, AMBIENT_T

# Normalised ideal-gas constants. N is dimensionless (ambient air == N0).
N0 = np.float32(1.0)                 # ambient air particle/mass density
P_ATM = np.float32(101325.0)         # Pa, ambient pressure
# Choose C so ambient air (N=N0, T=AMBIENT_T) gives P = P_ATM under P = C*N*T.
C_EOS = np.float32(P_ATM / (N0 * AMBIENT_T))


def ensure_N(state: State) -> np.ndarray:
    """Idempotently attach the air-density field `N` to `state`: ambient N0 in
    open air, 0 in solid/vacuum. Call once at a solver's first `step`.

    (P0's State intentionally omits N — it is a rung-A/B concern — so both
    rungs create it the SAME way, right here.)
    """
    if getattr(state, "N", None) is None:
        state.N = np.where(state.open_air, N0, np.float32(0.0)).astype(np.float32)
    return state.N


def derive_pressure(state: State) -> np.ndarray:
    """Ideal-gas EOS reframe: P = C_EOS * N * T. Writes and returns state.P."""
    state.P = (C_EOS * state.N * state.T).astype(np.float32)
    return state.P
