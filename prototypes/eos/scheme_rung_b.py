"""
Rung B -- Kwatra semi-implicit compressible solver.

EOS Phase-1.2 visual prototype (PLAN.md; docs/eos_research_report.md §4
"Rung B"). Unlike rung A (prescribed-divergence incompressible: thermal
expansion only, report §4 "Rung A"), rung B carries REAL momentum -- gas
actually accelerates under a pressure gradient and can genuinely compress
or expand -- which is what gives it baroclinic curl (rolling vortices) and
shock steepening rung A cannot produce by construction. The trick that
makes this affordable (report §0/§3, Kwatra-Su-Gretarsson-Fedkiw 2009) is
splitting each tick into:

  1. an EXPLICIT advection stage, substepped at the `|u|`-limited CFL
     (NOT `|u|+c` -- avoiding the sound-speed CFL is the entire point: a
     2500 K core's ~250 explicit-acoustic substeps/tick collapses to ~25
     just from this split, report §3), and
  2. an IMPLICIT acoustic stage -- a Helmholtz solve (identity + a Poisson
     term) for a pressure correction `p`, fixed-count Gauss-Seidel, that
     folds the sound-speed stiffness in WITHOUT needing to substep for it.
     As `c_max -> inf` the identity term becomes negligible relative to the
     Laplacian term and this reduces to the same incompressible Poisson
     projection rung A/control already run (report §4) -- rung B is a
     strict superset: dial `c_max` down and it degenerates toward them.

State carried: velocity `vx, vy` + density `N` (eos_core.ensure_N) +
temperature `T` -- not the full conservative (N, N*u, N*E) triple the
report's pseudocode sketches; see "Simplifications" below.

`self.c_max` (m/s) is the tunable dial report Q1 asks for: game-capped
(default 120 m/s -- comfortably above the grid's own `dx/dt ~= 4 m/s`
advective scale, far below real air's ~340-1000 m/s) for affordability and
stability.

IMPORTANT, measured-not-assumed finding (see class docstring): in THIS
split, `self.last_substeps` (the advection substep count) is a function of
`max|u|` ONLY -- it does not depend on `c_max` at all, because decoupling
advection from the acoustic speed is Kwatra's whole point. `c_max` instead
trades off the Helmholtz solve's conditioning/sharpness at a FIXED sweep
cost, not the per-tick substep count. Reported explicitly rather than
silently assumed, per the "measure it honestly" brief.

Simplifications (throwaway spike; flagged, not hidden):
  - Momentum is advected as velocity `u` (self-advection), not as `N*u`
    with a divide-back -- the task's own "(N*u or u)" leaves this open.
    Chosen because dividing by `N` right where the flow is most dynamic
    (breach fronts, where N is small/near-vacuum) is exactly where a
    divide-by-small-N would be least stable; advecting velocity directly
    sidesteps that failure mode entirely. `N` and `T` are advected as
    scalar tracers on that same velocity, all via solver.py's existing
    bilinear semi-Lagrangian sampler (`_advect_semilagrangian`, the same
    one rung A/control use) -- not a from-scratch conservative
    finite-volume advection. Semi-Lagrangian is unconditionally stable
    regardless of CFL; the substep loop below exists ONLY to honestly
    reproduce Kwatra's real cost story (many small advection steps), not
    because this particular advection scheme would blow up without it.
  - The Helmholtz Laplacian's face coefficients (`c_max^2 / N`) use a
    plain arithmetic mean between neighboring cells (not a harmonic mean).
    At a `solid` face the wall-side coefficient is first mirrored to the
    interior cell's own value (ghost cell = interior material) before
    averaging -- see `_solve_helmholtz`'s docstring for why that is a
    float32-precision fix, not just a style choice.
  - Compression/expansion heating uses the standard linearized adiabatic
    relation `dT/T = -(gamma-1) div(u) dt` (gamma=1.4, the same air
    constant behind the report's `c = 20.05 sqrt(T)`), sourced from the
    PRE-correction divergence (the same one driving the pressure solve's
    rhs) -- not a full internal-energy budget. `T` is floored at 1 K as a
    cheap safety net (a fixed floor, not an adaptive/tolerance check).
  - `vx, vy` are zeroed everywhere outside `open_air` (not just `solid`)
    immediately after the acoustic correction. Rung A/control only zero
    velocity in `solid` because they never divide by `N`; rung B does
    (`u -= dt*grad(p)/N`), and a fluid cell's gradient divided by the
    near-vacuum floor can be large-but-finite right at a vacuum boundary.
    Zeroing it keeps that artifact from leaking into next tick's `max|u|`
    CFL count -- which would otherwise quietly inflate the very substep
    number this file exists to report honestly.

Found empirically during self-verify, fixed, and flagged here rather than
silently patched over: the Helmholtz solve's `rhs` as first written was
PURELY REACTIVE (`dt*c_max^2*div(u*)`) -- it only ever responds to
divergence a velocity field ALREADY has. A resting gas with no initial
kick (S1's post-blast T staying hot with no fresh momentum; S5's water
compressing air with nothing already moving) produced EXACTLY zero further
motion, because there was no SOURCE term at all, only a corrector. The fix
reuses rung A's own prescribed-divergence recipe (report §4 "Rung A": same
`K_EXPAND` thermal term, same `W_DISPLACE_GAIN` water term, same tuned
constants -- reusing rung A's already-validated magnitudes rather than
inventing new ones under time pressure) as an extra `div_target` folded
into rung B's rhs: `rhs = dt*c_max^2*(div(u*) - div_target)`. This keeps
the same dimensional consistency and c_max->inf limit derived above (that
limit now reduces to EXACTLY rung A's own projection equation, which is
the report's "reduces to a Poisson projection as c -> inf" point taken
one step further: same target, reached over time through the acoustic
dynamics instead of enforced instantly by a hard projection). The
`N *= free_h_before/free_h_after` state update (step 5) is kept as well --
it is what actually satisfies "raise N ... there"; `div_target` is what
makes that event dynamically visible given rhs's reactive-only gap.
"""

import numpy as np

from state import State, AMBIENT_T
from solver import Solver, _advect_semilagrangian
from eos_core import ensure_N, derive_pressure
from shallow_water import FreeHeightTracker


def _mirror_neighbors(field: np.ndarray, solid: np.ndarray):
    """Up/down/left/right neighbors of `field`, mirrored (Neumann, zero-
    gradient) at `solid` -- the same wall-mirror convention as solver.py's
    `_diffuse` / rung A's & control's own `_mirror_neighbors` /
    prototypes/smoke_sim.py:compute_laplacian_with_walls."""
    up = np.where(np.roll(solid, 1, axis=0), field, np.roll(field, 1, axis=0))
    down = np.where(np.roll(solid, -1, axis=0), field, np.roll(field, -1, axis=0))
    left = np.where(np.roll(solid, 1, axis=1), field, np.roll(field, 1, axis=1))
    right = np.where(np.roll(solid, -1, axis=1), field, np.roll(field, -1, axis=1))
    return up, down, left, right


def _divergence(vx: np.ndarray, vy: np.ndarray, dx: float) -> np.ndarray:
    """Central-difference divergence of (vx, vy). No wall-mirroring needed:
    the caller has already zeroed vx/vy outside open air (Dirichlet no-
    penetration, distinct from the Neumann/mirror BC used for pressure
    below); the domain's outer ring is always solid too (scenarios.py), so
    the periodic np.roll wraparound never mixes in live fluid from the far
    edge."""
    ddx = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2.0 * dx)
    ddy = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2.0 * dx)
    return (ddx + ddy).astype(np.float32)


def _solve_helmholtz(rhs: np.ndarray, coef: np.ndarray, solid: np.ndarray,
                      vacuum: np.ndarray, beta: float, sweeps: int) -> np.ndarray:
    """Fixed-count (NOT tolerance-based -- determinism, report risk #4)
    red-black Gauss-Seidel relaxation for the acoustic Helmholtz system

        p - beta * div(coef * grad p) = -rhs,      beta = dt^2 / dx^2

    (docs/eos_research_report.md §4 "Rung B":
    `(I - dt^2 div((c_max^2/N) grad)) p' = -rhs`; `coef` is `c_max^2/N`
    here, `p` is that report's `p'`). Red-black (checkerboard) ordering
    keeps this vectorized while staying true Gauss-Seidel -- each color's
    pass only ever reads neighbors of the OTHER color, just refreshed by
    the pass immediately before it.

    Face coefficients are the arithmetic mean of the two adjacent cells'
    `coef`, EXCEPT across a `solid` face, where the wall-side coefficient
    is first mirrored to the interior cell's own value (ghost cell =
    interior material -- the standard convention, and also what makes the
    Neumann trick below exact). This is not merely cosmetic: leaving a
    wall-side coefficient at its literal `c_max^2/N_FLOOR` value (`N_FLOOR`
    being a tiny, physically-arbitrary floor -- there is no real gas behind
    a wall) creates a large-magnitude term in both the diagonal and the
    neighbor sum that only cancels via a near-exact subtraction. Exact in
    real-number math (see the Neumann note below -- it holds for ANY
    coefficient value), but a float32 catastrophic-cancellation trap in
    practice at `c_max ~ 100` scales. Mirroring the coefficient too keeps
    every face value bounded by real open-air `N` and avoids that trap
    entirely.

    Neumann (mirror) at `solid`: mirroring a wall neighbor's `p` value to
    the center cell's own (not-yet-updated) value is algebraically exactly
    a zero-flux face for ANY coefficient on that face -- same ghost-cell
    trick as rung A/control's `_solve_pressure`, re-derived here for the
    variable-coefficient operator (the `beta*coef*p_center` term this
    mirroring adds to the numerator exactly cancels the same term implicit
    in the diagonal, leaving a reduced-diagonal update using only the real
    neighbors).

    Dirichlet p=0 at `vacuum`: vacuum cells are excluded from the update
    mask, so they stay at their zero-initialized value for the whole
    solve. Non-mirrored neighbor lookups mean a fluid cell next to vacuum
    reads that true zero, which is what creates the pressure gradient
    driving outrush into a breach.

    Unlike a pure Poisson solve, this system's diagonal
    `1 + beta*sum(face coef)` is STRICTLY diagonally dominant (the leading
    `1` from the identity term is extra margin on top of the Laplacian's
    own diagonal) -- Gauss-Seidel is guaranteed to converge here; the fixed
    sweep count is a cost cap, not a stability requirement.
    """
    p = np.zeros(rhs.shape, dtype=np.float32)

    wall_up = np.roll(solid, 1, axis=0)
    wall_down = np.roll(solid, -1, axis=0)
    wall_left = np.roll(solid, 1, axis=1)
    wall_right = np.roll(solid, -1, axis=1)

    coef_nbr_up = np.where(wall_up, coef, np.roll(coef, 1, axis=0))
    coef_nbr_down = np.where(wall_down, coef, np.roll(coef, -1, axis=0))
    coef_nbr_left = np.where(wall_left, coef, np.roll(coef, 1, axis=1))
    coef_nbr_right = np.where(wall_right, coef, np.roll(coef, -1, axis=1))
    coef_up = 0.5 * (coef + coef_nbr_up)
    coef_down = 0.5 * (coef + coef_nbr_down)
    coef_left = 0.5 * (coef + coef_nbr_left)
    coef_right = 0.5 * (coef + coef_nbr_right)
    diag = (1.0 + beta * (coef_up + coef_down + coef_left + coef_right)).astype(np.float32)

    yy, xx = np.mgrid[0:rhs.shape[0], 0:rhs.shape[1]]
    checker = (yy + xx) % 2 == 0
    fluid = ~solid & ~vacuum
    color_masks = (checker & fluid, ~checker & fluid)

    for _ in range(sweeps):
        for mask in color_masks:
            up = np.where(wall_up, p, np.roll(p, 1, axis=0))
            down = np.where(wall_down, p, np.roll(p, -1, axis=0))
            left = np.where(wall_left, p, np.roll(p, 1, axis=1))
            right = np.where(wall_right, p, np.roll(p, -1, axis=1))
            neighbor_sum = (coef_up * up + coef_down * down
                            + coef_left * left + coef_right * right)
            p_new = (-rhs + beta * neighbor_sum) / diag
            p = np.where(mask, p_new, p)

    return p.astype(np.float32)


class RungBSolver(Solver):
    """Kwatra semi-implicit compressible gas solver -- see module
    docstring. `self.c_max` is the tunable sound-speed dial (report Q1).

    Cost model, precisely (this is the number the spike exists to
    measure): `self.last_substeps` is set from step 1's explicit-advection
    substep count `n`, derived once per tick from `max|u|` at the tick's
    start via `dt_adv = CFL_ADV*dx/(max|u|+eps)`, `n = min(ceil(dt/dt_adv),
    MAX_SUBSTEPS)`. Step 2's implicit Helmholtz solve is a SEPARATE,
    always-fixed `PRESSURE_SWEEPS` (~40) Gauss-Seidel sweeps every tick --
    it does not scale with velocity (that is the entire point of making it
    implicit) and is therefore not folded into `last_substeps`; it shows up
    honestly in ms/tick instead, as a roughly-constant per-tick overhead.
    `c_max` does not appear in the advection CFL at all, so raising or
    lowering it changes neither `n` nor the sweep count -- only the
    Helmholtz operator's stiffness (`c_max^2/N`), i.e. how sharply that
    fixed sweep budget can resolve the acoustic response. Lower `c_max`
    softens/smooths that response (cheaper to resolve well within the same
    40 sweeps, closer to RSST's "capped wave speed" look); higher `c_max`
    sharpens it (closer to a true incompressible projection in the limit)
    but is more likely to be under-resolved at a fixed sweep count. That is
    the real cost/sharpness trade `c_max` offers here -- not a substep
    count -- and it is reported as such rather than forcing the two to
    appear coupled.
    """

    name = "rungB"

    CFL_ADV = np.float32(1.0)      # advection substep CFL number, |u|-limited (NOT |u|+c)
    MAX_SUBSTEPS = 64              # hard cap/tick regardless of CFL (honest cap, not adaptive)
    EPS_SPEED = np.float32(1e-3)   # m/s, floor under max|u| so a resting grid needs only n=1
    PRESSURE_SWEEPS = 40           # fixed Gauss-Seidel sweeps for the Helmholtz solve (determinism)
    N_FLOOR = np.float32(1e-3)     # dimensionless N floor for /N divisions (solid/vacuum safety)
    GAMMA = np.float32(1.4)        # air ratio of specific heats (report §3's c = 20.05*sqrt(T))
    RATIO_MIN = np.float32(0.5)    # W3 N-bump ratio cap (shallow_water.py: "ratio capped")
    RATIO_MAX = np.float32(2.0)
    # div_target source constants -- deliberately the SAME values rung A
    # (scheme_rung_a.py) uses for the identical physical terms: reusing
    # its already-tuned, already-verified-safe magnitudes rather than
    # re-tuning from scratch (module docstring's "Found empirically" note).
    K_EXPAND = np.float32(0.15)         # 1/s, thermal-expansion divergence gain
    W_DISPLACE_GAIN = np.float32(60.0)  # dimensionless, W3 water-displacement gain

    def __init__(self, c_max: float = 120.0):
        super().__init__()
        self.c_max = np.float32(c_max)   # m/s -- TUNABLE DIAL (report Q1); see class docstring
        self._free_h_tracker: FreeHeightTracker | None = None

    def step(self, state: State, dt: float) -> None:
        dx = state.tile_size_m
        solid = state.solid
        vacuum = state.vacuum
        air = state.open_air
        N = ensure_N(state)
        vx, vy, T = state.vx, state.vy, state.T

        # ---- 1. EXPLICIT advection, substepped at the |u|-CFL (NOT |u|+c:
        #         decoupling from the acoustic speed is Kwatra's whole
        #         point, report §0/§3). n equal substeps of size dt/n
        #         exactly cover the tick; n is derived once, from the
        #         velocity this tick STARTS with, not re-derived mid-loop
        #         -- fixed count (determinism), and the honest cost number
        #         this whole spike exists to report.
        speed = np.sqrt(vx * vx + vy * vy)
        max_speed = float(speed.max())
        dt_adv = self.CFL_ADV * dx / (max_speed + self.EPS_SPEED)
        n = max(1, min(int(np.ceil(dt / dt_adv)), self.MAX_SUBSTEPS))
        sub_dt = dt / n

        for _ in range(n):
            vx0, vy0 = vx, vy
            vx = _advect_semilagrangian(vx0, vx0, vy0, sub_dt, dx)
            vy = _advect_semilagrangian(vy0, vx0, vy0, sub_dt, dx)
            N = _advect_semilagrangian(N, vx0, vy0, sub_dt, dx)
            T = _advect_semilagrangian(T, vx0, vy0, sub_dt, dx)
            vx[solid] = 0.0
            vy[solid] = 0.0
            N[~air] = 0.0
            T[solid] = AMBIENT_T

        # cost signal: CFL-driven advection substeps only -- see class
        # docstring for why the (fixed-cost) Helmholtz sweeps are not
        # added in here.
        self.last_substeps = n

        # ---- 1.5. prescribed div_target source (thermal expansion + W3
        #           water displacement) -- rung A's own recipe (report §4
        #           "Rung A"), reused here as rung B's acoustic SOURCE
        #           rather than a hard projection target; see module
        #           docstring's "Found empirically" note for why this
        #           exists (without it the Helmholtz solve is purely
        #           reactive and a resting gas never starts moving).
        #           `self._free_h_tracker.update()` has side effects (it
        #           advances its own before/after snapshot) so it is
        #           called exactly ONCE per tick, here; step 5 below reuses
        #           free_h_before/after rather than updating again.
        div_target = np.zeros(state.shape, dtype=np.float32)
        excess = np.clip(T - AMBIENT_T, 0.0, None)
        div_target += np.where(air, self.K_EXPAND * excess / AMBIENT_T, 0.0)

        if self._free_h_tracker is None:
            self._free_h_tracker = FreeHeightTracker(state)
        free_h_before, free_h_after = self._free_h_tracker.update(state)
        water_div = -((free_h_after - free_h_before) / dt) / free_h_after
        div_target += np.where(air, self.W_DISPLACE_GAIN * water_div, 0.0)

        # ---- 2. IMPLICIT acoustic solve: (I - dt^2 div((c_max^2/N) grad)) p = -rhs
        #         Fixed-count red-black Gauss-Seidel (REUSE-shaped after
        #         rung A/control's `_solve_pressure`, report §4 "REUSE
        #         RB-GS kernel"), generalized with the variable coefficient
        #         c_max^2/N and the added identity term. rhs = dt*c_max^2*
        #         (div(u*) - div_target) is the choice that makes this
        #         reduce to rung A's OWN prescribed-divergence projection
        #         equation as c_max -> inf, matching `u -= dt*grad(p)/N`
        #         below reducing to the plain projection correction in
        #         that same limit.
        N_safe = np.maximum(N, self.N_FLOOR)
        coef = (self.c_max * self.c_max / N_safe).astype(np.float32)
        div_star = _divergence(vx, vy, dx)
        beta = float(dt * dt / (dx * dx))
        rhs = (dt * self.c_max * self.c_max * (div_star - div_target)).astype(np.float32)
        p = _solve_helmholtz(rhs, coef, solid, vacuum, beta, self.PRESSURE_SWEEPS)

        # ---- 3. velocity correction + compression/expansion heating ------
        up, down, left, right = _mirror_neighbors(p, solid)
        grad_x = (right - left) / (2.0 * dx)
        grad_y = (down - up) / (2.0 * dx)
        vx = (vx - dt * grad_x / N_safe).astype(np.float32)
        vy = (vy - dt * grad_y / N_safe).astype(np.float32)
        vx[~air] = 0.0   # not just solid -- see module docstring's last bullet
        vy[~air] = 0.0

        T = (T - (self.GAMMA - 1.0) * T * div_star * dt).astype(np.float32)
        np.maximum(T, 1.0, out=T)   # cheap positivity floor, fixed (not adaptive)
        T[solid] = AMBIENT_T

        # ---- 4. derive P = C*N*T; advect smoke on the final velocity ------
        state.N, state.T = N, T
        derive_pressure(state)
        smoke = _advect_semilagrangian(state.smoke, vx, vy, dt, dx)

        # ---- 5. W3 water/air coupling (S5 only; identically a ratio of
        #         ~1.0 -- a no-op -- on every scenario where water_depth
        #         stays 0, i.e. S1-S4). Rising water shrinks the free air
        #         column above it, compressing the trapped air:
        #         N *= free_h_before/free_h_after, shallow_water.py's own
        #         suggested W3 recipe, ratio capped per that module's
        #         docstring. Reuses the SAME free_h_before/after captured
        #         in step 1.5 (the tracker's `.update()` must run only
        #         once per tick).
        ratio = np.clip(free_h_before / free_h_after, self.RATIO_MIN, self.RATIO_MAX)
        N = (N * np.where(air, ratio, 1.0)).astype(np.float32)

        # ---- 6. wall / vacuum cleanup --------------------------------------
        vx[solid] = 0.0   # belt-and-suspenders: already ~air-zeroed in step 3
        vy[solid] = 0.0
        N[~air] = 0.0
        smoke[~air] = 0.0
        np.clip(smoke, 0.0, None, out=smoke)

        state.vx, state.vy = vx, vy
        state.smoke = smoke
        state.N = N
