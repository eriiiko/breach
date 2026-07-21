"""EOS refactor P3 design-gate microbenchmarks — MEASUREMENT ONLY, no solver changes.

Executes deliverables #2 and #3 of docs/eos_refactor_design.md §8 patch P3:

  M1 — shipped-engine baseline wall-clock at 160x160 (+ ~50x120 ship-scale
       reference), under a worst-case-ish load (staggered explosions, a
       breach to vacuum, water released). Pins the p99/max the "p99 <= 25%
       of the 83ms budget" gate is judged against.
  M2 — RB-GS per-sweep cost (the existing atmosphere `diffuse_solve` kernel,
       the one the new Helmholtz solve reuses) at 160x160, as a function of
       `gs_iters` (already a pybind-exposed AtmosphereSolver.gs_iters
       read-write attribute — NO new binding needed). Derives ms/sweep,
       projects the Helmholtz cost at S=8/16/24/40 (x1.5 wide-int64 factor
       per the design), and times the ~50-substep legacy wave core being
       deleted (one `wave_substep` call x 50) to check the napkin claim.
  M3 — substep-count distribution: using the CURRENT engine's fields as
       proxies for the future solver state (u ~ wind_x/wind_y; P ~
       atmosphere + wave_p; N_hat ~ atmosphere floored at 1e-3), computes
       the design's substep-count formula
           n = ceil(dt / (CFL_ADV*dx / (u_est + eps)))
           u_est = max|u| + (max|grad P| / N_hat) * dt
       at every tick of the M1 scenarios plus two "nasty" stress scenarios
       (simultaneous multi-explosion stack; a single-cell O2-tank-rupture
       pressure/temperature spike via direct field writes). Pins the
       N_SUB_MAX recommendation.

Design constants (task-specified, independent of the shipped engine's
1/24s physics tick — these are the FUTURE solver's design-time constants):
    dt = 0.083 s, dx = 1/3 m, CFL_ADV = 0.5

Underscore-prefixed -> a throwaway dev diagnostic (mirrors tests/_xarch_*.py,
tests/_s1_conservation_check.py). Not a pytest suite; run directly:

    C:/Users/steen/miniconda3/envs/data/python.exe tests/_eos_p3_bench.py

Writes nothing but stdout; the committed deliverable is
docs/eos_p3_microbench_results.md (hand-authored from this script's output).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                          # noqa: E402
from level_loader import LevelData                    # noqa: E402
from simulation import Simulation                     # noqa: E402
from simulation import atmosphere_fixed, wave_fixed, water_fixed  # noqa: E402
from simulation.physics import apply_explosion        # noqa: E402
from simulation.weapons import get_tables as weapon_tables  # noqa: E402

SEED = 20260710

# --- Design constants (task-specified; the FUTURE solver's own numbers,
#     NOT read from config.toml's ticks_per_second=24 -> 1/24s tick) -------
DT_DESIGN = 0.083
DX_DESIGN = 1.0 / 3.0
CFL_ADV = 0.5
EPS = 1e-6


# ===========================================================================
# Map builders — hull ring + a grid of interior rooms (v1 tilemap vocabulary:
# 1=hull wall, 4=interior air; see level_loader.materials_from_tilemap).
# ===========================================================================
def _room_lattice_level(h, w, room_size, door_w=3, seed=0, name="p3_bench"):
    tm = np.full((h, w), 4, dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    rng = np.random.default_rng(seed)
    row_walls = list(range(room_size, h - 1, room_size))
    col_walls = list(range(room_size, w - 1, room_size))
    for y in row_walls:
        tm[y, 1:-1] = 1
    for x in col_walls:
        tm[1:-1, x] = 1
    # Door gaps punched through every partition wall (deterministic RNG) so
    # rooms interconnect — the wind/pressure field actually has somewhere
    # to go, instead of N sealed boxes.
    for y in row_walls:
        n_doors = max(1, w // 30)
        for _ in range(n_doors):
            cx = int(rng.integers(2, w - 2))
            lo, hi = max(1, cx - door_w // 2), min(w - 1, cx + door_w // 2 + 1)
            tm[y, lo:hi] = 4
    for x in col_walls:
        n_doors = max(1, h // 30)
        for _ in range(n_doors):
            cy = int(rng.integers(2, h - 2))
            lo, hi = max(1, cy - door_w // 2), min(h - 1, cy + door_w // 2 + 1)
            tm[lo:hi, x] = 4
    return LevelData(name=name, version="1", path=Path("."),
                      tilemap=tm, tile_size_m=DX_DESIGN, diffuse_path=Path("."))


def _make_sim(h, w, room_size, seed=SEED):
    lvl = _room_lattice_level(h, w, room_size=room_size, seed=seed)
    sim = Simulation(lvl, seed=seed, breach_physics=bp, enable_recorder=False)
    sim.set_paused(False)
    return sim


# ===========================================================================
# M3 helper — the design's CFL substep-count estimate, computed from the
# CURRENT engine's fields as PROXIES for the future solver state.
# ===========================================================================
def compute_substep_count(g, dt=DT_DESIGN, dx=DX_DESIGN, cfl_adv=CFL_ADV, eps=EPS):
    """n = ceil(dt / (CFL_ADV*dx / (u_est+eps))); u_est = max|u| + (max|gradP|/N_hat)*dt.

    Proxies (task-specified): u ~ (wind_x, wind_y); P ~ atmosphere + wave_p;
    N_hat ~ atmosphere floored at 1e-3. All fields dequantized to real units
    via the shared Q16.16 FP_ONE=65536 scale (atmosphere_fixed.dequantize
    also covers wind — see atmosphere_fixed.py's module docstring).

    Wall-adjacency guard: the CURRENT engine hard-zeros `atmosphere` at
    solid/wall cells (diffuse_solve's BC pass) — a numerical artifact of
    reading the raw field, not physics. The design's real Helmholtz operator
    uses a NEUMANN MIRROR at solid (zero cross-wall flux), so a naive
    whole-grid np.gradient would manufacture a huge spurious gradient at
    every wall face and swamp the real signal. We exclude any cell touching
    a structural wall from the max|gradP|/N_hat reduction. TRUE vacuum cells
    (a breach) stay IN the domain — that venting gradient is real Dirichlet
    P=0 physics and is exactly the high-CFL case this measurement must catch.
    """
    wind_x = atmosphere_fixed.dequantize(g.wind_x)
    wind_y = atmosphere_fixed.dequantize(g.wind_y)
    atm = atmosphere_fixed.dequantize(g.atmosphere)
    wp = wave_fixed.dequantize(g.wave_p)
    P = atm + wp
    N_hat = np.maximum(atm, 1e-3)

    solid = g.solid   # structural walls (== the "is_wall" arg the solvers take)
    open_mask = ~solid
    wall_adjacent = solid.copy()
    wall_adjacent[1:, :] |= solid[:-1, :]
    wall_adjacent[:-1, :] |= solid[1:, :]
    wall_adjacent[:, 1:] |= solid[:, :-1]
    wall_adjacent[:, :-1] |= solid[:, 1:]
    valid = open_mask & ~wall_adjacent

    max_u = float(np.sqrt(np.max((wind_x * wind_x + wind_y * wind_y)[open_mask]))) \
        if open_mask.any() else 0.0
    gy, gx = np.gradient(P, dx)
    grad_mag = np.sqrt(gy * gy + gx * gx)
    # Per-cell (|gradP|/N_hat), then take the grid max over the valid
    # (non-wall-adjacent) domain — the conservative (worst-case) reading of
    # "max|gradP|/N_hat" for a substep-count cliff.
    accel_term = grad_mag / N_hat
    max_accel = float(np.max(accel_term[valid])) if valid.any() else float(np.max(accel_term))

    u_est = max_u + max_accel * dt
    dt_adv = cfl_adv * dx / (u_est + eps)
    n = int(np.ceil(dt / dt_adv))
    return n, u_est


# ===========================================================================
# M1 — shipped-engine baseline wall-clock, worst-case-ish load.
# ===========================================================================
def _explosion_schedule(h, w, n_events, seed):
    rng = np.random.default_rng(seed)
    ticks = np.linspace(0.06, 0.75, n_events)
    events = []
    for frac in ticks:
        y = int(rng.integers(h // 6, h - h // 6))
        x = int(rng.integers(w // 6, w - w // 6))
        events.append((frac, y, x))
    return events


def run_m1_scenario(h, w, room_size, n_ticks, n_explosions, seed, record_substeps):
    """Run the worst-case-ish scenario; return (tick_ms array, substep n list)."""
    sim = _make_sim(h, w, room_size, seed=seed)
    g = sim.gmap
    frag = weapon_tables().payload_for_ammo("grenade_frag")
    events = [(int(frac * n_ticks), y, x)
              for (frac, y, x) in _explosion_schedule(h, w, n_explosions, seed + 1)]
    breach_tick = int(n_ticks * 0.55)
    water_tick = int(n_ticks * 0.75)
    breach_y, breach_x = h // 2, 0        # edge hull tile -> true vacuum breach
    water_room = max(3, room_size - 6)
    water_y0, water_y1 = 2, 2 + water_room
    water_x0, water_x1 = w - room_size + 2, w - 2

    tick_ms = np.empty(n_ticks, dtype=np.float64)
    substep_n = [] if record_substeps else None

    for t in range(n_ticks):
        for (et, ey, ex) in events:
            if t == et:
                apply_explosion(g, sim.edit_queue, ey, ex,
                                 int(frag.radius * 2), float(frag.pressure * 3),
                                 float(frag.wall_damage))
        if t == breach_tick:
            g.destroy_wall(breach_y, breach_x)
        if t == water_tick:
            g.water_depth[water_y0:water_y1, water_x0:water_x1] = \
                water_fixed.quantize_scalar(0.4)

        if record_substeps:
            n, _u_est = compute_substep_count(g)
            substep_n.append(n)

        t0 = time.perf_counter()
        sim.step()
        tick_ms[t] = (time.perf_counter() - t0) * 1000.0

    return tick_ms, substep_n


def _pctl(arr, p):
    return float(np.percentile(arr, p))


def summarize_ms(tick_ms, warmup=10):
    """p50/p99/max over the post-warmup ticks (skip JIT/cache-cold first N)."""
    a = np.asarray(tick_ms[warmup:])
    return _pctl(a, 50), _pctl(a, 99), float(a.max())


# ===========================================================================
# M2 — RB-GS per-sweep cost + the legacy wave-substep cost it replaces.
# ===========================================================================
def _snapshot_atmos_state(g):
    return dict(
        atmosphere=g.atmosphere.copy(), wave_p=g.wave_p.copy(),
        wave_v=g.wave_v.copy(), wave_source=g.wave_source.copy(),
        wind_x=g.wind_x.copy(), wind_y=g.wind_y.copy(),
        obstacles=g.obstacles.copy(), is_wall=g.solid.copy(),
        is_vacuum=g.is_vacuum.copy(), permeability=g.dyn_permeability.copy(),
        wave_absorb=g.dyn_wave_absorb.copy(),
    )


def measure_diffuse_solve_cost(state, atmos_solver, dt, gs_iters_list, repeats=25):
    """ms per diffuse_solve() call at each gs_iters setting, from the SAME
    snapshotted state each repeat (restores copies -> no cross-call drift).
    Restores atmos_solver.gs_iters to the shipped default (8) on exit —
    measurement-only, must change nothing when unused."""
    default_gs_iters = atmos_solver.gs_iters
    out = {}
    try:
        for gs_iters in gs_iters_list:
            atmos_solver.gs_iters = gs_iters
            times = []
            for _ in range(repeats):
                atm = state["atmosphere"].copy()
                wp = state["wave_p"].copy()
                wv = state["wave_v"].copy()
                ws = state["wave_source"].copy()
                wx = state["wind_x"].copy()
                wy = state["wind_y"].copy()
                t0 = time.perf_counter()
                atmos_solver.diffuse_solve(
                    atm, wp, wv, ws, wx, wy,
                    state["obstacles"], state["is_wall"], state["is_vacuum"],
                    state["permeability"], dt)
                times.append((time.perf_counter() - t0) * 1000.0)
            out[gs_iters] = times
    finally:
        atmos_solver.gs_iters = default_gs_iters
    return out


def measure_wave_substep_cost(state, atmos_solver, dt, repeats=25):
    times = []
    for _ in range(repeats):
        wp = state["wave_p"].copy()
        wv = state["wave_v"].copy()
        ws = state["wave_source"].copy()
        atm = state["atmosphere"].copy()
        t0 = time.perf_counter()
        atmos_solver.wave_substep(
            wp, wv, ws, atm,
            state["obstacles"], state["is_wall"], state["is_vacuum"],
            state["permeability"], state["wave_absorb"], dt)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def linear_fit_ms_per_sweep(gs_results):
    """Least-squares fit ms = a + b*gs_iters over the medians; b = ms/sweep."""
    xs = np.array(sorted(gs_results.keys()), dtype=np.float64)
    ys = np.array([np.median(gs_results[int(x)]) for x in xs], dtype=np.float64)
    A = np.vstack([np.ones_like(xs), xs]).T
    (a, b), *_ = np.linalg.lstsq(A, ys, rcond=None)
    return a, b, xs, ys


# ===========================================================================
# M3 — nasty stress scenarios (beyond the M1 scenarios' own substep trace).
# ===========================================================================
def run_nasty_multiexplosion(h, w, room_size, n_ticks=80, seed=SEED + 7):
    sim = _make_sim(h, w, room_size, seed=seed)
    g = sim.gmap
    frag = weapon_tables().payload_for_ammo("grenade_frag")
    stack_tick = 20
    cy, cx = h // 2, w // 2
    offsets = [(-3, -3), (-3, 3), (3, -3), (3, 3), (0, 0),
               (-6, 0), (6, 0), (0, -6), (0, 6)]
    substep_n = []
    for t in range(n_ticks):
        if t == stack_tick:
            for (dy, dx) in offsets:
                apply_explosion(g, sim.edit_queue, cy + dy, cx + dx,
                                 int(frag.radius * 2), float(frag.pressure * 4),
                                 float(frag.wall_damage))
        n, _u_est = compute_substep_count(g)
        substep_n.append(n)
        sim.step()
    return np.array(substep_n)


def run_nasty_tank_rupture(h, w, room_size, n_ticks=60, seed=SEED + 9):
    sim = _make_sim(h, w, room_size, seed=seed)
    g = sim.gmap
    rupture_tick = 10
    cy, cx = h // 2, w // 2
    substep_n = []
    for t in range(n_ticks):
        if t == rupture_tick:
            # O2-tank rupture proxy: DIRECT field writes (not FieldEdit) — a
            # local ~200x-ambient pressure spike + a temperature spike over a
            # 3x3 patch, exactly the "big single-cell spike" the task asks for.
            spike_atm = atmosphere_fixed.quantize_scalar(200.0)
            g.atmosphere[cy - 1:cy + 2, cx - 1:cx + 2] = spike_atm
            temp_hi_q = int(2000.0 * 65536.0)   # ~2000 K, Q16.16
            g.temperature[cy - 1:cy + 2, cx - 1:cx + 2] = temp_hi_q
        n, _u_est = compute_substep_count(g)
        substep_n.append(n)
        sim.step()
    return np.array(substep_n)


def substep_summary(arr):
    a = np.asarray(arr)
    return _pctl(a, 50), _pctl(a, 99), int(a.max())


# ===========================================================================
# Main — run all three measurements, print the tables.
# ===========================================================================
def main():
    print("=" * 78)
    print("EOS refactor P3 design-gate microbenchmarks (measurement only)")
    print(f"design constants: dt={DT_DESIGN}s dx={DX_DESIGN:.6f}m CFL_ADV={CFL_ADV}")
    print("=" * 78)

    N_TICKS = 300

    # --- M1: 160x160 primary + ~50x120 ship-scale reference ---------------
    print("\n--- M1: shipped-engine baseline wall-clock ---")
    print("[160x160] running scenario...")
    ms_160, substeps_160 = run_m1_scenario(
        h=160, w=160, room_size=40, n_ticks=N_TICKS, n_explosions=5,
        seed=SEED, record_substeps=True)
    p50, p99, mx = summarize_ms(ms_160)
    print(f"  160x160  p50={p50:.3f}ms  p99={p99:.3f}ms  max={mx:.3f}ms "
          f"(n={len(ms_160)} ticks)")
    print(f"  25% of 83ms budget = {0.25 * 83.0:.3f}ms -> "
          f"{'PASS' if p99 <= 0.25 * 83.0 else 'FAIL'} (p99 vs budget)")

    print("[50x120] running ship-scale reference scenario...")
    ms_ship, substeps_ship = run_m1_scenario(
        h=50, w=120, room_size=20, n_ticks=N_TICKS, n_explosions=5,
        seed=SEED + 100, record_substeps=True)
    p50s, p99s, mxs = summarize_ms(ms_ship)
    print(f"  ~50x120  p50={p50s:.3f}ms  p99={p99s:.3f}ms  max={mxs:.3f}ms "
          f"(n={len(ms_ship)} ticks)")

    # --- M2: RB-GS per-sweep cost + legacy wave-substep cost ---------------
    print("\n--- M2: RB-GS per-sweep cost (160x160) ---")
    sim2 = _make_sim(160, 160, 40, seed=SEED + 2)
    g2 = sim2.gmap
    frag2 = weapon_tables().payload_for_ammo("grenade_frag")
    # Warm the field into a non-trivial turbulent state before snapshotting
    # (a flat-ambient field would under-cost the RB-GS work vs a real tick).
    apply_explosion(g2, sim2.edit_queue, 80, 80, int(frag2.radius * 2),
                     float(frag2.pressure * 3), float(frag2.wall_damage))
    for _ in range(15):
        sim2.step()
    state2 = _snapshot_atmos_state(g2)
    atmos_solver = sim2.physics_runner.atmos
    dt_engine = 1.0 / 24.0   # the shipped physics tick's actual dt (measurement realism)

    gs_list = [8, 16, 24, 40]
    gs_results = measure_diffuse_solve_cost(state2, atmos_solver, dt_engine, gs_list)
    for gi in gs_list:
        times = gs_results[gi]
        print(f"  gs_iters={gi:>3d}  median={np.median(times):.4f}ms  "
              f"p90={np.percentile(times,90):.4f}ms  min={min(times):.4f}ms")
    a_fit, b_fit, xs, ys = linear_fit_ms_per_sweep(gs_results)
    print(f"  linear fit: ms = {a_fit:.4f} + {b_fit:.5f}*gs_iters "
          f"-> ms/sweep @160x160 = {b_fit:.5f}")
    print(f"  (post-run) atmos_solver.gs_iters restored to default = "
          f"{atmos_solver.gs_iters}")

    wave_times = measure_wave_substep_cost(state2, atmos_solver, dt_engine)
    ms_per_wave_substep = float(np.median(wave_times))
    # The napkin's "~50 explicit substeps/tick" assumed the C++ DEFAULT wave
    # speed (c=300 tiles/s -> max_dt=0.5/300, n=ceil(0.083/max_dt)=50). The
    # SHIPPED config (config.toml wave_c=66.0) gives a different max_dt — the
    # actual configured substep count is computed here directly via
    # atmos_solver.max_dt(), not assumed, to VERIFY the napkin's count.
    n_wave_napkin = 50
    n_wave_actual = int(np.ceil(DT_DESIGN / atmos_solver.max_dt()))
    legacy_wave_core_napkin_ms = ms_per_wave_substep * n_wave_napkin
    legacy_wave_core_actual_ms = ms_per_wave_substep * n_wave_actual
    print(f"  wave_substep: median={ms_per_wave_substep:.4f}ms/call")
    print(f"  atmos_solver.c={atmos_solver.c} (shipped config) -> max_dt="
          f"{atmos_solver.max_dt():.6f}s -> n_wave at dt={DT_DESIGN}s = "
          f"{n_wave_actual} (napkin assumed {n_wave_napkin}, its c=300 default)")
    print(f"  legacy wave core @ napkin's assumed 50 substeps = "
          f"{legacy_wave_core_napkin_ms:.4f}ms")
    print(f"  legacy wave core @ ACTUAL configured {n_wave_actual} substeps = "
          f"{legacy_wave_core_actual_ms:.4f}ms")

    WIDE_INT64_FACTOR = 1.5
    print(f"  projected Helmholtz cost (ms/sweep x {WIDE_INT64_FACTOR} x S):")
    helm_costs = {}
    for S in (8, 16, 24, 40):
        c = b_fit * WIDE_INT64_FACTOR * S
        helm_costs[S] = c
        print(f"    S={S:>3d}: {c:.4f}ms/tick")

    diffuse_solve_8_ms = float(np.median(gs_results[8]))
    old_total_ms = legacy_wave_core_actual_ms + diffuse_solve_8_ms
    print(f"  old total atmosphere-group cost/tick (wave core [actual n] + "
          f"diffuse_solve@gs=8) = {legacy_wave_core_actual_ms:.4f} + "
          f"{diffuse_solve_8_ms:.4f} = {old_total_ms:.4f}ms")
    for S in (8, 16):
        ratio = old_total_ms / helm_costs[S]
        print(f"    vs new Helmholtz-only @ S={S}: {helm_costs[S]:.4f}ms "
              f"-> {ratio:.2f}x cheaper (new advection substeps not yet "
              f"built/measured; this ratio EXCLUDES them)")

    # --- M3: substep-count distribution -------------------------------
    print("\n--- M3: substep-count distribution (proxy fields) ---")
    p50n, p99n, maxn = substep_summary(substeps_160)
    print(f"  M1 160x160 scenario:  p50={p50n:.1f}  p99={p99n:.1f}  max={maxn}")
    p50ns, p99ns, maxns = substep_summary(substeps_ship)
    print(f"  M1 ~50x120 scenario:  p50={p50ns:.1f}  p99={p99ns:.1f}  max={maxns}")

    print("  running nasty scenario: simultaneous multi-explosion stack...")
    nasty_multi = run_nasty_multiexplosion(160, 160, 40)
    p50m, p99m, maxm = substep_summary(nasty_multi)
    print(f"  nasty multi-explosion: p50={p50m:.1f}  p99={p99m:.1f}  max={maxm}")

    print("  running nasty scenario: O2-tank rupture (direct field spike)...")
    nasty_tank = run_nasty_tank_rupture(160, 160, 40)
    p50t, p99t, maxt = substep_summary(nasty_tank)
    print(f"  nasty tank-rupture:    p50={p50t:.1f}  p99={p99t:.1f}  max={maxt}")

    overall_max = max(maxn, maxns, maxm, maxt)
    n_sub_max_rec = 1
    while n_sub_max_rec < overall_max * 2:   # power-of-two >= observed max, with margin
        n_sub_max_rec *= 2
    print(f"\n  overall observed max n = {overall_max}")
    print(f"  N_SUB_MAX recommendation (power of two >= max, 2x margin) = "
          f"{n_sub_max_rec}")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
