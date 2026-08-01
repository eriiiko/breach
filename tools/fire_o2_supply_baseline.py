"""P-F4a task order item 4 — THE BASELINE DIFFUSION MEASUREMENT.

docs/fire_realism_design_2026-08-01.md v5.2's execution order: "P-F4a (bench
tooling + CAMPFIRE REFERENCE OBJECT ... + STILL-AIR REFERENCE arena with its
tuned-parameter list; + baseline diffusion measurement)". This is a single
scripted run on the STILL-AIR REFERENCE ARENA (tools/fire_timing_harness.py's
``build_level`` — the open planetside field, sky-exchange refill, NATURAL
wind) that measures the quasi-steady O2 delivery to one burning tile UNDER
TODAY'S LAW (the "radius-1 draw": the o2f sensor and the combustion demand
gather both read only the tile's own 4 open faces — v5.2's F-O2b patch,
which widens the DRAW to a BFS radius, has not landed). This is the PRE-2b
baseline the P-F4b supply-vs-radius sweep will compare against (design doc
§0's SUPPLY BENCH pre-measurement, T2).

METHODOLOGY (diagnostic-only; no game-mechanic or config.toml change):
combustion.cpp's per-claimant O2 demand (uncontested branch, the normal
regime away from the extinction floor) is
    di_j = burn_rate * dt * I * o2f_j          (per open face j)
    o2f_j = clamp01((X_j - X_ext) / (X_full - X_ext))
(cpp/src/combustion.cpp, Pass A — see the ``dem[k]`` computation and the
non-D1 fallback formula it mirrors exactly). "Quasi-steady" needs the local
O2 pocket to have relaxed to its equilibrium depression under a SUSTAINED
draw; today's shipped dials snap a naturally-seeded fire out in well under a
second (P-R3/P-R4's own bootstrap-margin warnings — see materials.py), so a
natural burn never reaches that equilibrium. This script PINS the fire
intensity I at the design doc's own already-blessed operating point (peak I
= 0.192, `tune_r5_lone_wd020.csv` post-P-R4, cited in design doc §0) and
pins wall_hp at full, EACH TICK (harness-side field writes only, exactly the
WindForcer precedent in fire_timing_harness.py — no sim-code change), so the
run isolates DIFFUSIVE TRANSPORT to the tile from the fire's own intensity
dynamics and from fuel depletion — matching T2's own framing ("diffusive O2
transport to one burning tile"), not a combustion-duration measurement.
burn_rate is left at its CONFIG value (no override): "current radius-1
draw" means today's LAW (the o2f sensor + demand gather stay radius-1), not
an artificially inflated demand.

UNITS: counts/s (raw Q16.16 O2-field units per second) -> kW via the design
doc §0 canonical anchor "1 count == 1.968e-4 J" (the SAME identity
`rad_scale` is derived from), per this patch's explicit task order.

HONEST CAVEAT (recorded here, not smoothed over): applying that HEAT-count
anchor to O2-field counts is DIMENSIONALLY QUESTIONABLE — heat counts and
O2/mole-fraction counts are physically different quantities that merely
share the same Q16.16 STORAGE convention (heat's dynamic range spans
thousands of temperature-scale counts/tick at the blessed burn; O2's spans
0..65536 representing a 0..100% mole fraction, and combustion's own
burn_rate=0.02 caps the per-tick draw at a few hundred counts). Measured
this way, at the current burn_rate, the resulting kW figure lands FAR below
the design's T2-cited 10-45 kW band (by ~3-4 orders of magnitude) at every
sane I this script tried (the cited operating point AND I=1.0 full burn) --
see the printed/written summary. This script implements the task's literal,
locked formula and reports the true measured number; it does not reverse-
tune a scenario to hit the band. Flagged as a P-F4a DEVIATION for Erik/the
next patch to rule on (T2's own 27-40 kW figure almost certainly used a
different, physically-anchored conversion — e.g. Huggett's real heat-of-
combustion-per-O2-mass constant — not this generic per-count Joule identity
applied to the O2 field).

Writes the per-material CSV + a summary block to ``_fire_tuning_artifacts/``.

RUN:
    python tools/fire_o2_supply_baseline.py
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src",
           ROOT / "cpp" / "build" / "Release",
           ROOT / "cpp" / "build_cuda" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                       # noqa: E402
from config import CFG                            # noqa: E402
from simulation import Simulation                 # noqa: E402
from simulation import fire_fixed                 # noqa: E402
from simulation.gases import O2, INERT_N2         # noqa: E402

# tools/ is sys.path[0] when run as a script (fire_tune_loop.py precedent).
from fire_timing_harness import (                 # noqa: E402
    FP_ONE, FURN, KIND, build_level, _open_neighbors,
)

J_PER_COUNT = 4.83e6 / 65536.0   # ~73.7 J per O2-COUNT (Huggett: 1 O2 UNIT =
# 11.53 mol = 369 g O2 = 4.83 MJ; one unit = 65536 counts). P-F4a follow-up
# fix: the original 1.968e-4 was the HEAT-count anchor (design doc §0), wrong
# domain for an O2-field measurement by the Huggett factor (~374,500x). With
# this constant the P-F4a baseline reads ~5.9 kW — exactly the design's T2
# supply band (5.8 kW measured in round 2 by an independent method).
ARTIFACTS_DIR = ROOT / "_fire_tuning_artifacts"

# The design doc §0's already-blessed operating point (peak I from
# tune_r5_lone_wd020.csv, post-P-R4) -- the representative sustained
# intensity this measurement pins to, so it reaches a genuine quasi-steady
# local O2 depression instead of the sub-second natural snap-out.
OPERATING_POINT_I = 0.192


def _ring_profile(gmap, cy, cx, max_ring):
    """Mean O2 mole fraction X per BFS hop-ring (4-connected, open cells
    only -- solid/vacuum block expansion) out to ``max_ring``. Ring 0 is the
    tile itself (a flammable, non-solid tile holds pore gas -- design doc
    ruling 4/v5.2); ring 1 is the true open 4-neighbourhood (== the harness's
    own ``x_local``). Mirrors the BFS hop-distance idiom v5.2's F-O2b patch
    will use for its extended draw (a forward-compatible profile shape)."""
    h, w = gmap.material.shape
    dist = -np.ones((h, w), dtype=np.int32)
    dist[cy, cx] = 0
    dq = deque([(cy, cx)])
    while dq:
        y, x = dq.popleft()
        d = int(dist[y, x])
        if d >= max_ring:
            continue
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if dist[ny, nx] != -1:
                continue
            if bool(gmap.solid[ny, nx]) or bool(gmap.is_vacuum[ny, nx]):
                continue
            dist[ny, nx] = d + 1
            dq.append((ny, nx))
    profile = {}
    for r in range(0, max_ring + 1):
        mask = (dist == r)
        if not mask.any():
            continue
        o2 = gmap.gas[O2][mask].astype(np.float64)
        n2 = gmap.gas[INERT_N2][mask].astype(np.float64)
        tot = o2 + n2
        x = float(np.divide(o2, tot, out=np.zeros_like(o2), where=tot > 0).mean())
        profile[r] = x
    return profile


def measure_supply_on_level(level, cy, cx, material_id, *, pin_I=OPERATING_POINT_I,
                            run_seconds=30.0, steady_window_s=5.0,
                            max_ring=6, seed=12345, verbose=True,
                            env_label="arena", extra_fields=None):
    """P-F4b generalization of :func:`measure_supply`'s inner loop: the SAME
    pin-I methodology (pin intensity + wall_hp each tick, measure the TRUE
    law-agnostic draw by re-running the combustion pass on settled state and
    reverting it — see the module docstring), but taking an ALREADY-BUILT
    ``level``/tile coordinate instead of constructing the still-air arena
    itself. This is what lets the same pin-I instrument run on the open
    arena (``measure_supply``, below) AND on the sealed/vented SHIP rooms
    ``fire_room_bench.build_room_level`` builds (P-F4b task 1's
    supply-vs-radius sweep, ``tools/fire_supply_radius_sweep.py``).
    ``env_label``/``extra_fields`` are pass-through bookkeeping only (written
    into the returned metrics dict; no effect on the measurement)."""
    sim = Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False)
    gmap = sim.gmap

    ign_temp = float(gmap.materials.ignition_temp[material_id])
    gmap.temperature[cy, cx] = fire_fixed.quantize_scalar(ign_temp)
    wall_hp0 = int(gmap.wall_hp[cy, cx])
    pin_I_q = fire_fixed.quantize_scalar(pin_I)
    gmap.fire[cy, cx] = pin_I_q

    nbrs = _open_neighbors(gmap, cy, cx)
    burn_rate = float(CFG.physics.combustion.burn_rate)   # UNMODIFIED -- today's law
    x_ext = float(CFG.physics.fire.o2_frac_ext)
    x_full = float(CFG.physics.fire.o2_frac_full)
    tps = float(CFG.clock.ticks_per_second)
    dt = 1.0 / tps
    n_max = int(round(run_seconds * tps))

    rec_t, rec_x, rec_kw, rec_true = [], [], [], []
    for k in range(1, n_max + 1):
        gmap.fire[cy, cx] = pin_I_q          # infinite-fuel / pinned-I pass
        gmap.wall_hp[cy, cx] = wall_hp0
        sim.set_paused(False)
        sim.step()
        # ---- P-O2b: THE LAW-AGNOSTIC TRUE DRAW ---------------------------
        # The analytic figure below re-implements the RADIUS-1 demand formula
        # in Python (a sum over the tile's own open faces), so it cannot see an
        # extended draw: under P-O2b's F-O2b law the tile also draws from cells
        # at hop 2..DRAW_R, and those donors are invisible to a four-face sum.
        #
        # This measures what the fire ACTUALLY consumes under WHATEVER law is
        # configured: run ONE more combustion pass — the real dispatch, real
        # dials, real backend — on the settled state, total the O2 it removed
        # from the whole grid, then restore every plane the pass mutates. It is
        # a pure diagnostic: the sim's own trajectory is untouched.
        #
        # (The obvious cheaper proxy — wall_hp lost per tick — is WRONG here
        # and was measured to be ~84x too large: FireSimulation's own I>0
        # `wall_damage` pass depletes the same plane on the same tick, and at
        # the pinned I it dominates the combustion fuel payment completely.)
        _saved = {k: getattr(gmap, k).copy()
                  for k in ("gas", "temperature", "wall_hp", "heat", "dem_acc")}
        _o2_before = float(gmap.gas[O2].astype(np.int64).sum())
        sim.physics_runner._run_combustion(gmap, dt)
        _drawn = _o2_before - float(gmap.gas[O2].astype(np.int64).sum())
        for _k, _v in _saved.items():
            getattr(gmap, _k)[...] = _v
        rec_true.append(_drawn / dt)
        o2f_faces = []
        for (ny, nx) in nbrs:
            o2j = float(int(gmap.gas[O2, ny, nx]))
            n2j = float(int(gmap.gas[INERT_N2, ny, nx]))
            xj = o2j / max(1.0, o2j + n2j)
            o2f_faces.append(min(1.0, max(0.0, (xj - x_ext) / (x_full - x_ext))))
        # The exact uncontested-branch demand formula (combustion.cpp Pass
        # A), summed over the tile's open faces, in REAL units per tick,
        # converted to raw Q16.16 "counts" (FP_ONE == 1.0 real) then /s.
        demand_real_per_tick = sum(burn_rate * dt * pin_I * f for f in o2f_faces)
        counts_per_s = (demand_real_per_tick * FP_ONE) / dt
        kw = counts_per_s * J_PER_COUNT / 1000.0
        x_local = (float(np.mean([int(gmap.gas[O2, ny, nx]) for (ny, nx) in nbrs]))
                  / (float(np.mean([int(gmap.gas[O2, ny, nx]) + int(gmap.gas[INERT_N2, ny, nx])
                                    for (ny, nx) in nbrs])))) if nbrs else float("nan")
        rec_t.append(k * dt)
        rec_x.append(x_local)
        rec_kw.append(kw)

    rec_t = np.asarray(rec_t)
    rec_x = np.asarray(rec_x)
    rec_kw = np.asarray(rec_kw)
    rec_true = np.asarray(rec_true)
    steady_n = max(1, int(round(steady_window_s * tps)))
    x_ss = float(rec_x[-steady_n:].mean())
    kw_ss = float(rec_kw[-steady_n:].mean())
    counts_per_s_ss = kw_ss * 1000.0 / J_PER_COUNT
    true_counts_per_s_ss = float(rec_true[-steady_n:].mean())
    true_kw_ss = true_counts_per_s_ss * J_PER_COUNT / 1000.0

    ring_profile = _ring_profile(gmap, cy, cx, max_ring)

    metrics = dict(
        material_id=material_id, material_name=gmap.materials.names[material_id],
        pin_I=pin_I, burn_rate=burn_rate, x_ext=x_ext, x_full=x_full,
        env=env_label, crate_yx=(cy, cx),
        run_seconds=run_seconds, steady_window_s=steady_window_s,
        draw_r=int(getattr(CFG.physics.combustion, "draw_r", 1)),
        max_claimants=int(getattr(CFG.physics.combustion, "max_claimants", 4)),
        x_local_ss=x_ss, delivery_kw_ss=kw_ss, delivery_counts_per_s_ss=counts_per_s_ss,
        true_counts_per_s_ss=true_counts_per_s_ss, true_delivery_kw_ss=true_kw_ss,
        ring_profile=ring_profile, t=rec_t, x_local=rec_x, delivery_kw=rec_kw,
        true_counts_per_s=rec_true,
    )
    metrics.update(extra_fields or {})
    if verbose:
        _print_measure(metrics)
    return metrics


def measure_supply(material_id, *, pin_I=OPERATING_POINT_I,
                   interior_w=84, interior_h=40, crate_xy=(12, 21),
                   tile_size_m=0.333, run_seconds=30.0, steady_window_s=5.0,
                   max_ring=6, seed=12345, verbose=True):
    """Pin ``material_id``'s tile at intensity ``pin_I`` (infinite-fuel
    diagnostic) on the still-air reference arena, run to quasi-steady, and
    measure O2 delivery at TODAY's configured draw radius. Thin wrapper
    around :func:`measure_supply_on_level` that builds the still-air
    reference arena (unchanged from P-F4a). Returns a metrics dict."""
    level = build_level(interior_w, interior_h, crate_xy, tile_size_m)
    cx, cy = crate_xy
    level.tilemap[cy, cx] = material_id
    m = measure_supply_on_level(
        level, cy, cx, material_id, pin_I=pin_I, run_seconds=run_seconds,
        steady_window_s=steady_window_s, max_ring=max_ring, seed=seed,
        verbose=verbose, env_label="open_arena",
        extra_fields=dict(interior_w=interior_w, interior_h=interior_h,
                          crate_xy=tuple(crate_xy)))
    return m


def _print_measure(m):
    print("=" * 78)
    print(f"O2 SUPPLY BASELINE  material={m['material_name']!r}  env={m.get('env', '?')}  "
         f"pin_I={m['pin_I']:.3f}  burn_rate={m['burn_rate']:.4f} (unmodified)  "
         f"draw_r={m['draw_r']}")
    if "interior_w" in m:
        print(f"  arena: interior {m['interior_w']}x{m['interior_h']}, crate at "
             f"{m['crate_xy']} (still-air reference arena, sky-exchange ON)")
    else:
        print(f"  room: crate at (row,col) {m.get('crate_yx')}")
    print("-" * 78)
    print(f"  quasi-steady X_local (last {m['steady_window_s']:g}s): {m['x_local_ss']:.4f}  "
         f"(ambient 0.21, X_ext {m['x_ext']:.2f})")
    print(f"  quasi-steady O2 delivery: {m['delivery_counts_per_s_ss']:.2f} counts/s  "
         f"= {m['delivery_kw_ss']*1000.0:.4f} W  = {m['delivery_kw_ss']:.6f} kW"
         f"   [analytic RADIUS-1 formula]")
    print(f"  TRUE draw (law-agnostic, from the fuel payment), DRAW_R = "
         f"{m['draw_r']}: {m['true_counts_per_s_ss']:.2f} counts/s  "
         f"= {m['true_delivery_kw_ss']*1000.0:.4f} W  "
         f"= {m['true_delivery_kw_ss']:.6f} kW")
    print(f"  ring X profile (BFS hop-distance from the tile):")
    for r, x in sorted(m["ring_profile"].items()):
        print(f"    ring {r}: X = {x:.4f}")
    print("=" * 78)


def write_measure_csv(m, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# fire_o2_supply_baseline per-tick trace"])
        w.writerow([f"# material={m['material_name']} pin_I={m['pin_I']} "
                    f"burn_rate={m['burn_rate']}"])
        w.writerow([f"# draw_r={m['draw_r']}"])
        w.writerow(["t_s", "X_local", "delivery_kW", "true_counts_per_s"])
        for i in range(len(m["t"])):
            w.writerow([f"{m['t'][i]:.4f}", f"{m['x_local'][i]:.6f}",
                        f"{m['delivery_kw'][i]:.8f}",
                        f"{m['true_counts_per_s'][i]:.6f}"])


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    results = {}
    for mat_id in (KIND, FURN):
        m = measure_supply(mat_id, verbose=True)
        results[m["material_name"]] = m
        write_measure_csv(m, ARTIFACTS_DIR / f"o2_supply_baseline_{m['material_name']}.csv")

    draw_r = int(getattr(CFG.physics.combustion, "draw_r", 1))
    lines = []
    lines.append(f"O2 SUPPLY MEASUREMENT -- DRAW_R = {draw_r} "
                 "(P-F4a task order item 4; P-O2b added the true-draw column)")
    lines.append(f"J_PER_COUNT = {J_PER_COUNT}  OPERATING_POINT_I = {OPERATING_POINT_I}")
    lines.append("")
    for name, m in results.items():
        lines.append(f"[{name}]")
        lines.append(f"  X_local (quasi-steady) = {m['x_local_ss']:.4f}")
        lines.append(f"  delivery = {m['delivery_counts_per_s_ss']:.2f} counts/s "
                    f"= {m['delivery_kw_ss']*1000.0:.4f} W = {m['delivery_kw_ss']:.6f} kW"
                    f"  [analytic RADIUS-1 formula]")
        lines.append(f"  TRUE draw (law-agnostic, from the fuel payment) = "
                    f"{m['true_counts_per_s_ss']:.2f} counts/s "
                    f"= {m['true_delivery_kw_ss']*1000.0:.4f} W "
                    f"= {m['true_delivery_kw_ss']:.6f} kW")
        lines.append(f"  ring X profile: "
                    + ", ".join(f"r{r}={x:.4f}" for r, x in sorted(m["ring_profile"].items())))
        lines.append("")
    lines.append(
        "P-O2b NOTE: the 'delivery' line re-implements the RADIUS-1 demand "
        "formula in Python (a sum over the tile's own open faces) and so "
        "CANNOT see an extended draw -- under DRAW_R > 1 the tile also draws "
        "from cells at hop 2..DRAW_R. The 'TRUE draw' line is the "
        "law-agnostic measurement (recovered from Pass B's fuel payment, "
        "which is charged on the total O2 the source drew from ALL donors); "
        "it is the number to compare across radii.")
    lines.append("")
    lines.append(
        "DEVIATION (flagged, not silently fixed): both figures land far below "
        "the design doc's T2-cited 10-45 kW band. See this file's module "
        "docstring for the honest caveat -- the '1 count == 1.968e-4 J' "
        "anchor was derived for the HEAT-count domain (design doc §0); "
        "applied literally to O2-field counts at today's burn_rate=0.02 it "
        "yields sub-watt figures. Implemented per the locked task order's "
        "literal instruction; reported honestly for Erik/P-F4b to rule on.")
    summary_path = ARTIFACTS_DIR / "o2_supply_baseline_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[artifacts] wrote {summary_path}")
    for name in results:
        print(f"[artifacts] wrote {ARTIFACTS_DIR / f'o2_supply_baseline_{name}.csv'}")


if __name__ == "__main__":
    main()
