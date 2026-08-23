"""velocity_clamp_pv2_measure.py — P-V2 measurement (velocity-clamp arc).

Runs a scripted, deterministic headless blast on ``playground`` (the level
the arc-opening audit's human-play session used) that exercises the exact
spike path the audit named: a pressurised/heated pocket ("grenade-scale"
gas+heat deposit, mimicking an explosive without needing the full
weapon/combat stack) followed by ``destroy_wall`` events that vent it into
true vacuum through the level's known south-hull breach line (the same
``(67, 10)``/vacuum-at-``(68, 10)`` geometry ``tests/test_playground_level.py``
already asserts). Multiple such events are staged across one run, plus one
pure interior deposit with no wall break, so the run covers both mechanisms
the P-V2 task named ("destroy_wall events and/or grenade-scale deposits").

Per tick this script captures the RAW (undequantized, int32 Q16.16) engine
fields it needs directly from ``GameMap`` — not through ``PhysicsRecorder``,
so the core symptom count (own-cell supersonic violations) can be computed
in **exact int64 / Python-bignum arithmetic**, matching gate 1's own
``rad > cap2`` int64 test instead of a lossy float32 round-trip. It also
snapshots ``EOSSolver`` telemetry every tick (``dbg_last_c_local_q``,
``dbg_last_n_sub``, the nine rail counters, ``ke_drag_removed``) — the
counters are CUMULATIVE members (never reset by ``step()``; only
``ke_drag_removed`` and its P-E3 siblings reset per tick), so this script
diffs them itself.

It then runs the SAME symptom-table formulas (c_amb=300, T_amb=290, s_eos=1,
D1 ambient floor, D4 ambient-cap-for-ts is approximated by excluding solid
cells only — this script does not have a separate ``thermal_solid`` mask in
either dataset, a stated approximation, see the doc) against the pre-fix
seed dump ``debug_manual_20260818_194038_velocity_clamp_seed.npz`` side by
side, and prints both symptom tables plus the required-n_sub-vs-N_SUB_MAX-8
histogram and a best-effort clamp-energy estimate.

Usage:
    conda run -n data python tools/velocity_clamp_pv2_measure.py
    conda run -n data python tools/velocity_clamp_pv2_measure.py --pre-fix-dump PATH --n-ticks 420
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                          # noqa: E402
import level_loader                                   # noqa: E402
from simulation import Simulation                     # noqa: E402
from simulation.gases import O2, INERT_N2             # noqa: E402

FP_ONE = 65536.0
C_AMB = 300.0
T_AMB = 290.0
S_EOS = 1.0

COUNTER_ATTRS = ("u_clamp_hits", "u_max_hits", "work_clamp_hits",
                 "energy_floor_hits", "t_max_phys_hits")


# ===========================================================================
# Scenario construction
# ===========================================================================
def deposit_blast(gmap, cy, cx, radius, density_factor, temp_k):
    """A "grenade-scale" pressurise-and-heat deposit in a disc, mimicking an
    explosive payload without driving the weapon/combat stack: scale the
    bulk gas (+ atmosphere, so the pressure step sees it immediately) by
    ``density_factor`` and set temperature to ``temp_k`` over every open
    (non-solid, non-vacuum) cell within ``radius`` tiles of (cy, cx)."""
    h, w = gmap.solid.shape
    ys, xs = np.mgrid[0:h, 0:w]
    mask = ((ys - cy) ** 2 + (xs - cx) ** 2 <= radius * radius)
    mask &= ~gmap.solid & ~gmap.is_vacuum
    for g in (O2, INERT_N2):
        gmap.gas[g][mask] = (gmap.gas[g][mask].astype(np.int64)
                              * density_factor).astype(np.int32)
    gmap.atmosphere[mask] = (gmap.atmosphere[mask].astype(np.int64)
                              * density_factor).astype(np.int32)
    gmap.temperature[mask] = int(round(temp_k * FP_ONE))
    return int(mask.sum())


def build_scenario(seed=20260819):
    level = level_loader.load("playground")
    sim = Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False)
    sim.set_paused(False)
    return sim


# Events: (tick, kind, kwargs). Three breach events on the known south-hull
# vacuum line (verified openness/geometry against the live playground map
# before writing this list — see the P-V2 measurement doc), one pure
# interior deposit with no wall break.
EVENTS = [
    (2,   "deposit",      dict(cy=62, cx=10, radius=4, density_factor=9.0, temp_k=820.0)),
    (6,   "destroy_wall", dict(fy=67, fx=10)),
    (90,  "deposit",      dict(cy=62, cx=36, radius=4, density_factor=9.0, temp_k=780.0)),
    (94,  "destroy_wall", dict(fy=67, fx=36)),
    (180, "deposit",      dict(cy=62, cx=27, radius=4, density_factor=9.0, temp_k=900.0)),
    (184, "destroy_wall", dict(fy=67, fx=27)),
    (280, "deposit",      dict(cy=30, cx=50, radius=4, density_factor=9.0, temp_k=850.0)),
    # no wall break for this one — pure interior transport stress
]


def capture_raw(gmap):
    """RAW int32/bool field snapshot, direct from GameMap (no Recorder
    dequant round-trip) — kept exact so the symptom count can run in
    int64/bignum arithmetic."""
    return dict(
        wind_x=gmap.wind_x.copy(), wind_y=gmap.wind_y.copy(),
        atmosphere=gmap.atmosphere.copy(), temperature=gmap.temperature.copy(),
        gas_o2=gmap.gas[O2].copy(), inert_n2=gmap.gas[INERT_N2].copy(),
        solid=gmap.solid.copy(), is_vacuum=gmap.is_vacuum.copy(),
    )


def run_scenario(n_ticks, seed=20260819):
    sim = build_scenario(seed)
    gmap = sim.gmap
    eos = sim.physics_runner.eos
    dt = 1.0 / float(sim._tps)

    events_by_tick = {}
    for (tick, kind, kw) in EVENTS:
        events_by_tick.setdefault(tick, []).append((kind, kw))

    snaps = []            # list of capture_raw() dicts, index 0 = tick-0 entry
    telemetry = []         # list of dicts, index i = the tick that produced snaps[i+1] from snaps[i]
    event_log = []         # (tick, kind, detail)

    snaps.append(capture_raw(gmap))
    prev_counters = {a: getattr(eos, a) for a in COUNTER_ATTRS}

    for tick in range(1, n_ticks + 1):
        for (kind, kw) in events_by_tick.get(tick, []):
            if kind == "deposit":
                n_cells = deposit_blast(gmap, **kw)
                event_log.append((tick, "deposit", dict(kw, n_cells=n_cells)))
            elif kind == "destroy_wall":
                gmap.destroy_wall(kw["fy"], kw["fx"])
                event_log.append((tick, "destroy_wall", kw))

        entry = snaps[-1]   # tick-entry state (pre-step, post any event mutation above)
        sim.step()

        cur_counters = {a: getattr(eos, a) for a in COUNTER_ATTRS}
        deltas = {a: cur_counters[a] - prev_counters[a] for a in COUNTER_ATTRS}
        prev_counters = cur_counters

        telemetry.append(dict(
            tick=tick,
            c_local=float(eos.dbg_last_c_local_q) / FP_ONE,
            n_sub_used=int(eos.dbg_last_n_sub),
            ke_drag_removed_raw=int(eos.ke_drag_removed),
            **{f"d_{a}": deltas[a] for a in COUNTER_ATTRS},
        ))
        snaps.append(capture_raw(gmap))

    return dict(snaps=snaps, telemetry=telemetry, event_log=event_log,
                dt=dt, eos_config=dict(
                    c_max=eos.c_max, dx=eos.dx, adiabatic_index=eos.adiabatic_index,
                    CFL_ADV=eos.CFL_ADV, N_SUB_MAX=eos.N_SUB_MAX,
                    N_FLOOR_SOLVER=eos.N_FLOOR_SOLVER, U_MAX=eos.U_MAX,
                    k_drag=eos.k_drag, k_drag2=eos.k_drag2))


# ===========================================================================
# Required-n_sub reconstruction (eos_solver.cpp:470-536's formula, replayed
# in REAL (dequantized) units — no overflow risk exists in real-unit
# arithmetic; the int64/mul128_shr overflow the design doc warns about is
# specific to the C++ RAW Q16.16/Q32.32 pipeline, not this recompute).
# ===========================================================================
def _mirror_neighbor(field, solid, dy, dx):
    """field/solid at (y+dy, x+dx), Neumann-reflecting to self at a solid
    neighbour or the grid edge — eos_solver.cpp's mirror_idx, vectorized."""
    h, w = field.shape
    ys, xs = np.mgrid[0:h, 0:w]
    ny, nx = ys + dy, xs + dx
    valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
    ny_c, nx_c = np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)
    neigh_val = field[ny_c, nx_c]
    neigh_solid = np.zeros((h, w), dtype=bool)
    neigh_solid[valid] = solid[ny_c[valid], nx_c[valid]]
    use_self = (~valid) | neigh_solid
    return np.where(use_self, field, neigh_val)


def required_n_sub(entry, cfg, c_local, dt):
    """Uncapped required substep count for the tick that consumes ``entry``
    as its tick-entry state, per eos_solver.cpp:470-536's formula."""
    solid = entry["solid"]
    P = entry["atmosphere"].astype(np.float64) / FP_ONE
    n_total = (entry["gas_o2"].astype(np.int64)
               + entry["inert_n2"].astype(np.int64)).astype(np.float64) / FP_ONE
    wx = entry["wind_x"].astype(np.float64) / FP_ONE
    wy = entry["wind_y"].astype(np.float64) / FP_ONE

    K = cfg["c_max"] ** 2 / cfg["adiabatic_index"]
    Kdt = K * dt
    dx = cfg["dx"]

    P_l = _mirror_neighbor(P, solid, 0, -1)
    P_r = _mirror_neighbor(P, solid, 0, 1)
    P_u = _mirror_neighbor(P, solid, -1, 0)
    P_d = _mirror_neighbor(P, solid, 1, 0)
    gx = (P_r - P_l) / (2.0 * dx)
    gy = (P_d - P_u) / (2.0 * dx)
    gmag = np.maximum(np.abs(gx), np.abs(gy))

    n_hat = np.maximum(n_total, cfg["N_FLOOR_SOLVER"])
    du = Kdt * gmag / n_hat
    du = np.where(solid, 0.0, du)
    max_du = float(du.max()) if du.size else 0.0

    speed = np.hypot(wx, wy)
    max_u = float(speed.max()) if speed.size else 0.0

    u_est = max_u + max_du + 2.0 / FP_ONE
    u_est_cap = max(c_local, cfg["U_MAX"])
    u_est = min(u_est, u_est_cap)

    req = max(1, math.ceil((dt * u_est) / (cfg["CFL_ADV"] * dx) - 1e-12))
    return req, max_u, max_du


# ===========================================================================
# Own-cell cap symptom table (audit's formulas, D1/D2v2 fold)
# ===========================================================================
def cap2_raw_plane(temperature_raw, t_amb_q, s_eos_q, c_amb2_q32, u_max2_q32,
                    ratio_umax):
    """Exact int64/Python-bignum recompute of eos_solver.cpp:418-439's
    cap2_plane_ fold (D1 floor; D4's ts-routing is NOT modelled — neither
    dataset here carries a separate thermal_solid mask, an approximation
    stated in the P-V2 doc; ambient/solid cells are excluded by the caller's
    open_mask instead of routed to u_max2 here, which only matters for the
    excluded cells anyway)."""
    t_abs = ((s_eos_q * temperature_raw.astype(np.int64)) >> 16) + t_amb_q
    t_abs = np.maximum(t_abs, t_amb_q)                     # D1 floor
    ratio = (t_abs << 16) // t_amb_q                        # int64, exact
    rail = ratio >= ratio_umax
    ratio_obj = ratio.astype(object)
    mul = (c_amb2_q32 * ratio_obj) >> 16                    # Python bignum, exact
    cap2 = np.where(rail, u_max2_q32, mul.astype(np.int64))
    return cap2


def symptom_table_from_raw(snaps, label, deposit_tick_indices=()):
    """Own-cell supersonic count + peak ratio + P_min + worst-cell x-ambient
    + peak single-tick cell-gain, from a list of RAW capture_raw() dicts."""
    t_amb_q = int(round(T_AMB * FP_ONE))
    s_eos_q = int(round(S_EOS * FP_ONE))
    c_amb_q = int(round(C_AMB * FP_ONE))
    u_max_q = int(round(1000.0 * FP_ONE))
    c_amb2_q32 = c_amb_q * c_amb_q
    u_max2_q32 = u_max_q * u_max_q
    ru = u_max_q / c_amb_q
    ratio_umax = int(ru * ru * 65536.0) + 1

    n_snap = len(snaps)
    total_open_cellsnaps = 0
    viol_count = 0
    viol_snaps = set()
    max_ratio = 0.0
    max_ratio_loc = None
    p_min = float("inf")
    worst_n_ambient = 0.0
    peak_cell_gain = 0.0
    peak_cell_gain_loc = None
    prev_N = None

    for i, snap in enumerate(snaps):
        solid = snap["solid"]
        # The kick's own skip-set is solid||is_vacuum||ambient-ring (a strict
        # superset of the scan's solid||is_vacuum) — a cell that just joined
        # is_vacuum (a fresh destroy_wall breach) keeps whatever stale wind it
        # carried the instant before, and the kick never touches it again, so
        # counting it as a "violation" is a measurement artifact, not a clamp
        # miss. Exclude it when the capture has is_vacuum (our own post-fix
        # run); the pre-fix dump predates that field (DEFAULT_FIELDS has no
        # is_vacuum), so it stays solid-only there — same limitation the
        # original audit's own analysis had, kept for apples-to-apples parity
        # with its published numbers.
        open_mask = ~solid
        if "is_vacuum" in snap:
            open_mask = open_mask & ~snap["is_vacuum"]
        total_open_cellsnaps += int(open_mask.sum())

        cap2 = cap2_raw_plane(snap["temperature"], t_amb_q, s_eos_q,
                               c_amb2_q32, u_max2_q32, ratio_umax)
        rad = (snap["wind_x"].astype(np.int64) ** 2
               + snap["wind_y"].astype(np.int64) ** 2)
        cap_mag = np.floor(np.sqrt(cap2.astype(np.float64)))
        allowed = (cap_mag + 2.0) ** 2
        viol = (rad.astype(np.float64) > allowed) & open_mask
        nviol = int(viol.sum())
        if nviol:
            viol_count += nviol
            viol_snaps.add(i)

        cap_real = np.sqrt(np.maximum(cap2.astype(np.float64), 1.0)) / FP_ONE
        speed_real = np.sqrt(rad.astype(np.float64)) / FP_ONE
        ratio_field = np.where(open_mask & (cap_real > 0), speed_real / cap_real, 0.0)
        r = float(ratio_field.max()) if ratio_field.size else 0.0
        if r > max_ratio:
            max_ratio = r
            yx = np.unravel_index(int(np.argmax(ratio_field)), ratio_field.shape)
            max_ratio_loc = (i, int(yx[0]), int(yx[1]))

        P_real = snap["atmosphere"].astype(np.float64) / FP_ONE
        p_min = min(p_min, float(P_real[open_mask].min()) if open_mask.any() else p_min)

        N = (snap["gas_o2"].astype(np.int64)
             + snap["inert_n2"].astype(np.int64)).astype(np.float64) / FP_ONE
        worst_n_ambient = max(worst_n_ambient, float(N.max()))

        if prev_N is not None and i not in deposit_tick_indices:
            delta = N - prev_N
            dmax = float(delta.max())
            if dmax > peak_cell_gain:
                peak_cell_gain = dmax
                yx = np.unravel_index(int(np.argmax(delta)), delta.shape)
                peak_cell_gain_loc = (i, int(yx[0]), int(yx[1]))
        prev_N = N

    return dict(
        label=label, n_snapshots=n_snap,
        total_open_cellsnaps=total_open_cellsnaps,
        supersonic_violations=viol_count,
        supersonic_violation_snapshots=len(viol_snaps),
        max_ratio=max_ratio, max_ratio_loc=max_ratio_loc,
        p_min=p_min, worst_n_ambient=worst_n_ambient,
        peak_cell_gain=peak_cell_gain, peak_cell_gain_loc=peak_cell_gain_loc,
    )


# ===========================================================================
# Pre-fix seed dump loader (already dequantized float32 — Recorder schema)
# ===========================================================================
def load_prefix_dump_as_raw(path):
    """Adapt the pre-fix .npz (already dequantized float32/float, per
    PhysicsRecorder) into the same RAW-int32-ish dict shape
    ``symptom_table_from_raw``/``cap2_raw_plane`` expect, by re-quantizing.
    This is NOT free of precision loss (the dump already round-tripped
    through float32) — flagged explicitly in the doc; it is the best
    available fidelity for a dataset that predates this script."""
    d = np.load(path)
    # IMPORTANT: NpzFile.__getitem__ re-decompresses the array from the zip
    # on EVERY call (it does not cache) — indexing d["wind_x"][i] inside a
    # 775-iteration loop would decompress the full (775,70,100) array 775
    # times per field. Pull each field into memory ONCE, then slice.
    wind_x = d["wind_x"]; wind_y = d["wind_y"]
    atmosphere = d["atmosphere"]; temperature = d["temperature"]
    gas_o2 = d["gas_o2"]; inert_n2 = d["inert_n2"]; obstacles = d["obstacles"]
    n = atmosphere.shape[0]

    wind_x_raw = np.round(wind_x.astype(np.float64) * FP_ONE).astype(np.int64)
    wind_y_raw = np.round(wind_y.astype(np.float64) * FP_ONE).astype(np.int64)
    atmosphere_raw = np.round(atmosphere.astype(np.float64) * FP_ONE).astype(np.int64)
    temperature_raw = np.round(temperature.astype(np.float64) * FP_ONE).astype(np.int64)
    gas_o2_raw = np.round(gas_o2.astype(np.float64) * FP_ONE).astype(np.int64)
    inert_n2_raw = inert_n2.astype(np.int64)   # already raw per recorder.py
    solid = obstacles.astype(bool)

    snaps = []
    for i in range(n):
        snaps.append(dict(
            wind_x=wind_x_raw[i], wind_y=wind_y_raw[i],
            atmosphere=atmosphere_raw[i], temperature=temperature_raw[i],
            gas_o2=gas_o2_raw[i], inert_n2=inert_n2_raw[i], solid=solid[i],
        ))
    return snaps


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-ticks", type=int, default=420)
    ap.add_argument("--pre-fix-dump", type=str,
                     default=str(ROOT / "debug_manual_20260818_194038_velocity_clamp_seed.npz"))
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    print(f"=== Running post-fix scripted blast scenario ({args.n_ticks} ticks, "
          f"seed {args.seed}) ===")
    result = run_scenario(args.n_ticks, seed=args.seed)
    snaps = result["snaps"]
    telemetry = result["telemetry"]
    cfg = result["eos_config"]
    dt = result["dt"]

    print("\nevents:")
    for (tick, kind, detail) in result["event_log"]:
        print(f"  tick {tick:4d}  {kind:14s} {detail}")

    # required n_sub reconstruction, cross-checked against the engine's own
    # (capped) dbg_last_n_sub telemetry
    req_list = []
    mismatches = 0
    for i, tel in enumerate(telemetry):
        entry = snaps[i]
        req, max_u, max_du = required_n_sub(entry, cfg, tel["c_local"], dt)
        capped = min(req, cfg["N_SUB_MAX"])
        if capped != tel["n_sub_used"]:
            mismatches += 1
        req_list.append(dict(tick=tel["tick"], required=req, capped=capped,
                              engine_used=tel["n_sub_used"], max_u=max_u, max_du=max_du))

    print(f"\nrequired-n_sub reconstruction: {mismatches}/{len(req_list)} ticks where "
          f"min(required, N_SUB_MAX) != engine's dbg_last_n_sub (sanity check; small "
          f"counts are expected from the tick-entry-state alignment approximation)")

    reqs = np.array([r["required"] for r in req_list])
    print(f"required n_sub: min={reqs.min()} max={reqs.max()} mean={reqs.mean():.2f} "
          f"median={np.median(reqs):.1f}")
    rail_frac = float((reqs > cfg["N_SUB_MAX"]).mean())
    print(f"fraction of ticks where required > N_SUB_MAX={cfg['N_SUB_MAX']}: {rail_frac:.3f} "
          f"({int((reqs > cfg['N_SUB_MAX']).sum())}/{len(reqs)})")
    hist_edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 100000]
    hist, _ = np.histogram(reqs, bins=hist_edges)
    for lo, hi, c in zip(hist_edges[:-1], hist_edges[1:], hist):
        print(f"  [{lo:6d}, {hi:6d}) : {c}")

    # counters over the run
    d_u_clamp = sum(t["d_u_clamp_hits"] for t in telemetry)
    d_u_max = sum(t["d_u_max_hits"] for t in telemetry)
    d_work = sum(t["d_work_clamp_hits"] for t in telemetry)
    d_efloor = sum(t["d_energy_floor_hits"] for t in telemetry)
    d_tmaxphys = sum(t["d_t_max_phys_hits"] for t in telemetry)
    ke_drag_total_real = sum(t["ke_drag_removed_raw"] for t in telemetry) / (FP_ONE ** 2)
    print(f"\nsolver counters over the run: u_clamp_hits={d_u_clamp} u_max_hits={d_u_max} "
          f"work_clamp_hits={d_work} energy_floor_hits={d_efloor} "
          f"t_max_phys_hits={d_tmaxphys}")
    print(f"ke_drag_removed, summed over run (real units, Sigma n*Delta(u^2)): "
          f"{ke_drag_total_real:.3f}")

    # own-cell supersonic symptom table, post-fix
    deposit_idx = set()
    for (tick, kind, _kw) in result["event_log"]:
        if kind == "deposit":
            deposit_idx.add(tick)   # snaps[tick] is post-deposit-and-step
    post = symptom_table_from_raw(snaps, "POST-FIX (scripted scenario)",
                                   deposit_tick_indices=deposit_idx)
    print("\n=== POST-FIX symptom table (scripted scenario) ===")
    for k, v in post.items():
        print(f"  {k}: {v}")

    # pre-fix dump, same formulas
    print(f"\n=== Loading pre-fix seed dump: {args.pre_fix_dump} ===")
    pre_snaps = load_prefix_dump_as_raw(args.pre_fix_dump)
    pre = symptom_table_from_raw(pre_snaps, "PRE-FIX (seed dump)")
    print("\n=== PRE-FIX symptom table (seed dump, same formulas) ===")
    for k, v in pre.items():
        print(f"  {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
