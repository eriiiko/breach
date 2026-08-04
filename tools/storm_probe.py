"""storm_probe.py — atmosphere-disturbance instrument (ANALYSIS ONLY).

PROVENANCE: written 2026-08-03 for the storming investigation; every number in
`docs/fire_atmosphere_oscillation_analysis_2026-08-03.md` (on main) came from
this file. It is a one-off investigation harness, NOT a maintained bench — no
test coverage, no provenance headers on its artifacts (the audit's own P7
finding applies to it). Committed so the headline result stays reproducible;
treat it as evidence, not as infrastructure.

Depends on `tools/fire_timing_harness.py`, which lives on this branch only —
it does not exist on main.


Harness-level: builds levels, drives the shipped engine, reads mirrored planes.
NOTHING in cpp/ or src/ is modified; dials move via CFG overrides that are
restored after every run (fire_timing_harness.apply_overrides precedent).

Modes
  thermal   : sealed room, one-shot thermal impulse, NO fire  -> ring-down (T2)
  jet       : sealed room, one-shot velocity impulse, NO fire  -> vortical ring-down
  hotplate  : one tile pinned at dT every tick, NO fire        -> steady drive (T1)
  fire      : a real crate fire at chosen dials                -> reproduction (T0)

Geometry
  --geom room      one sealed room (hull ring)
  --geom tworoom   two rooms joined by a 1-tile door (Erik's B2 Helmholtz case)
  --geom arena     open planetside arena (SPACE ring + sky), the fire bench level

Metric v0 (per tick): kinetic energy over open cells, max |u|, the pressure
transient |P - P_prev| (mean/max), spatial std of P, probe traces, EOS rails.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release",
           ROOT / "cpp" / "build_cuda" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                      # noqa: E402
from config import CFG                           # noqa: E402
from level_loader import LevelData               # noqa: E402
from simulation import Simulation                # noqa: E402
from simulation import fire_fixed                # noqa: E402

from fire_timing_harness import (                # noqa: E402
    FP_ONE, AIR, HULL, SPACE, FURN,
    apply_overrides, restore_overrides,
)

# The P-F1b recalibration dials (origin/pf1b-recalibration config.toml).
PF1B = {
    "ignition_seed": "0.12",
    "k_grow": "2.0",
    "k_die": "0.008",
    "I_cap_per_avail": "14.0",
    "ignition_to_ext_delta": "200.0",
    "fire_T_span": "180.0",
    "wall_damage": "0.03",
    "T_emit_gate": "310.0",
    "physics.combustion.H_BED_M": "18125.0",
    "physics.combustion.H_BED_SHIFT": "4",
    "materials.furniture.cool_shift": "13",
    "materials.wood.cool_shift": "13",
    "materials.kindling.cool_shift": "13",
}


def build_room(interior_w, interior_h, tile_size_m, crate_xy=None):
    """Sealed ship room: 1-tile HULL ring, AIR interior, boundary=space."""
    h, w = interior_h + 2, interior_w + 2
    tm = np.full((h, w), AIR, dtype=np.int32)
    tm[0, :] = HULL
    tm[h - 1, :] = HULL
    tm[:, 0] = HULL
    tm[:, w - 1] = HULL
    if crate_xy is not None:
        tm[crate_xy[1], crate_xy[0]] = FURN
    return LevelData(name="storm_room", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=float(tile_size_m),
                     diffuse_path=Path("."), boundary="space")


def build_tworoom(interior_w, interior_h, tile_size_m, door_h=1, crate_xy=None):
    """Two sealed rooms sharing a HULL partition with a door gap — the B2
    Helmholtz geometry (gas inertia in the neck + compressibility either side)."""
    h = interior_h + 2
    w = 2 * interior_w + 3
    tm = np.full((h, w), AIR, dtype=np.int32)
    tm[0, :] = HULL
    tm[h - 1, :] = HULL
    tm[:, 0] = HULL
    tm[:, w - 1] = HULL
    mid = interior_w + 1
    tm[:, mid] = HULL
    start = 1 + max(0, (interior_h - door_h) // 2)
    for r in range(start, min(start + door_h, h - 1)):
        tm[r, mid] = AIR
    if crate_xy is not None:
        tm[crate_xy[1], crate_xy[0]] = FURN
    return LevelData(name="storm_tworoom", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=float(tile_size_m),
                     diffuse_path=Path("."), boundary="space")


def build_arena(interior_w, interior_h, tile_size_m, crate_xy):
    from fire_timing_harness import build_level
    return build_level(interior_w, interior_h, crate_xy, tile_size_m,
                       sky_tau_s=60.0, sponge_width=8)


def stamp_air_damping(gmap, value):
    """Set air's per-cell velocity damping (the shipped-but-inert lever:
    u *= 1 - wave_absorb*absorb_strength*dt, eos_solver.cpp:632-642).
    Stamped straight onto the mirrored planes — no config file is touched."""
    air = ~gmap.solid
    gmap.wave_absorb[air] = np.float32(value)
    gmap.dyn_wave_absorb[air] = np.float32(value)


def eos_counters(sim):
    out = {}
    for holder in (getattr(sim, "physics_runner", None),
                   getattr(getattr(sim, "physics_runner", None), "engine", None),
                   getattr(getattr(sim, "physics_runner", None), "eos", None)):
        if holder is None:
            continue
        for name in ("dbg_last_n_sub", "u_clamp_hits", "u_max_hits",
                     "work_clamp_hits", "energy_floor_hits", "t_max_phys_hits"):
            try:
                v = getattr(holder, name)
            except Exception:
                continue
            if isinstance(v, (int, float)):
                out[name] = v
    return out


def measure(gmap, open_mask, p_prev_raw, probe):
    wx = gmap.wind_x[open_mask].astype(np.float64) / FP_ONE
    wy = gmap.wind_y[open_mask].astype(np.float64) / FP_ONE
    ke = float(np.sum(wx * wx + wy * wy))
    umax = float(np.max(np.sqrt(wx * wx + wy * wy))) if wx.size else 0.0
    p = gmap.atmosphere[open_mask].astype(np.float64) / FP_ONE
    dp = np.abs(p - p_prev_raw)
    py, px = probe
    return dict(
        ke=ke, umax=umax,
        p_mean=float(np.mean(p)), p_std=float(np.std(p)),
        dp_mean=float(np.mean(dp)), dp_max=float(np.max(dp)),
        p_probe=float(int(gmap.atmosphere[py, px]) / FP_ONE),
        u_probe=float(int(gmap.wind_x[py, px]) / FP_ONE),
        T_probe=float(int(gmap.temperature[py, px]) / FP_ONE),
    )


def run(mode="thermal", geom="room", interior=12, tile=0.5, seconds=25.0,
        damp=0.0, dT=200.0, jet=5.0, patch=2, dials=None, verbose=True,
        impulse_at=1.0, period=10.0, amp=45.0, door=1):
    overrides = dict(dials or {})
    restore = apply_overrides(overrides) if overrides else []
    try:
        crate = None
        if mode == "fire":
            crate = (interior // 2 + 1, interior // 2 + 1)
        if geom == "room":
            level = build_room(interior, interior, tile, crate)
        elif geom == "tworoom":
            level = build_tworoom(interior, interior, tile, door_h=door,
                                  crate_xy=crate)
        elif geom == "arena":
            crate = crate or (interior // 2 + 1, interior // 2 + 1)
            level = build_arena(interior, interior, tile, crate)
        else:
            raise ValueError(geom)

        sim = Simulation(level, seed=12345, breach_physics=bp,
                         enable_recorder=False)
        gmap = sim.gmap
        if damp > 0.0:
            stamp_air_damping(gmap, damp)

        open_mask = (~gmap.solid)
        for attr in ("is_vacuum", "is_ambient"):
            m = getattr(gmap, attr, None)
            if m is not None:
                open_mask = open_mask & (~m)
        h, w = gmap.solid.shape
        cy, cx = h // 2, w // 2
        if geom == "tworoom":
            cx = (w // 4)
        probe = (cy, cx)

        if mode == "fire":
            fy, fx = crate[1], crate[0]
            seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.1))
            gmap.fire[fy, fx] = fire_fixed.quantize_scalar(seed_i)
            gmap.temperature[fy, fx] = fire_fixed.quantize_scalar(280.0)

        tps = float(CFG.clock.ticks_per_second)
        dt = 1.0 / tps
        n = int(round(seconds * tps))
        impulse_tick = max(1, int(round(impulse_at * tps)))

        rows = []
        p_prev = gmap.atmosphere[open_mask].astype(np.float64) / FP_ONE
        for k in range(1, n + 1):
            if k == impulse_tick and mode in ("thermal", "jet"):
                sl = (slice(cy - patch // 2, cy - patch // 2 + patch),
                      slice(cx - patch // 2, cx - patch // 2 + patch))
                if mode == "thermal":
                    gmap.temperature[sl] = fire_fixed.quantize_scalar(dT)
                else:
                    gmap.wind_x[sl] = int(round(jet * FP_ONE))
            if mode == "hotplate":
                gmap.temperature[cy, cx] = fire_fixed.quantize_scalar(dT)
            elif mode == "pulse":
                # A flicker stand-in: the burning tile's T swinging about a mean,
                # the shape P-F1b measured (T 210-331 on the furniture plateau).
                val = dT + amp * np.sin(2.0 * np.pi * (k * dt) / period)
                gmap.temperature[cy, cx] = fire_fixed.quantize_scalar(float(val))

            sim.set_paused(False)
            sim.step()

            m = measure(gmap, open_mask, p_prev, probe)
            p_prev = gmap.atmosphere[open_mask].astype(np.float64) / FP_ONE
            m["t"] = k * dt
            m["tick"] = k
            if mode == "fire":
                m["I"] = int(gmap.fire[crate[1], crate[0]]) / FP_ONE
                m["T_fire"] = int(gmap.temperature[crate[1], crate[0]]) / FP_ONE
            rows.append(m)

        counters = eos_counters(sim)
        return dict(rows=rows, counters=counters, dt=dt, probe=probe,
                    n_open=int(open_mask.sum()), mode=mode, geom=geom,
                    damp=damp, dT=dT, tile=tile, interior=interior)
    finally:
        restore_overrides(restore)


def analyse(res, settle_from=None):
    rows = res["rows"]
    t = np.array([r["t"] for r in rows])
    ke = np.array([r["ke"] for r in rows])
    dpm = np.array([r["dp_mean"] for r in rows])
    umax = np.array([r["umax"] for r in rows])
    out = {}
    out["ke_peak"] = float(ke.max())
    out["ke_final"] = float(ke[-1])
    out["ke_last_decile_mean"] = float(ke[int(0.9 * len(ke)):].mean())
    out["umax_peak"] = float(umax.max())
    out["umax_final"] = float(umax[-1])
    out["dp_mean_tail"] = float(dpm[int(0.5 * len(dpm)):].mean())
    out["dp_max_overall"] = float(max(r["dp_max"] for r in rows))
    if ke.max() > 0:
        out["ke_retention"] = float(out["ke_last_decile_mean"] / ke.max())
    # ring-down: log-linear fit of KE after the peak
    ipk = int(np.argmax(ke))
    seg_t, seg_ke = t[ipk:], ke[ipk:]
    good = seg_ke > (seg_ke.max() * 1e-6)
    if good.sum() > 10:
        sl = np.polyfit(seg_t[good], np.log(seg_ke[good]), 1)[0]
        out["ke_decay_rate_per_s"] = float(-sl)
        out["ke_efold_s"] = float(-1.0 / sl) if sl < 0 else float("inf")
    # dominant period of the probe pressure trace (after any impulse)
    p = np.array([r["p_probe"] for r in rows])
    seg = p[ipk:] - p[ipk:].mean()
    if seg.size > 16:
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        freqs = np.fft.rfftfreq(seg.size, d=res["dt"])
        spec[0] = 0.0
        j = int(np.argmax(spec))
        out["dom_freq_hz"] = float(freqs[j])
        out["dom_period_s"] = float(1.0 / freqs[j]) if freqs[j] > 0 else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="thermal",
                    choices=["thermal", "jet", "hotplate", "fire", "pulse"])
    ap.add_argument("--period", type=float, default=10.0)
    ap.add_argument("--amp", type=float, default=45.0)
    ap.add_argument("--geom", default="room",
                    choices=["room", "tworoom", "arena"])
    ap.add_argument("--interior", type=int, default=12)
    ap.add_argument("--tile", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--damp", type=float, default=0.0)
    ap.add_argument("--dT", type=float, default=200.0)
    ap.add_argument("--jet", type=float, default=5.0)
    ap.add_argument("--patch", type=int, default=2)
    ap.add_argument("--pf1b", action="store_true", help="apply P-F1b dials")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    dials = dict(PF1B) if a.pf1b else {}
    res = run(mode=a.mode, geom=a.geom, interior=a.interior, tile=a.tile,
              seconds=a.seconds, damp=a.damp, dT=a.dT, jet=a.jet,
              patch=a.patch, dials=dials, period=a.period, amp=a.amp)
    summ = analyse(res)
    head = (f"mode={a.mode} geom={a.geom} interior={a.interior} tile={a.tile} "
            f"damp={a.damp} dT={a.dT} open_cells={res['n_open']}")
    print(head)
    for k, v in summ.items():
        print(f"  {k:24s} {v}")
    if res["counters"]:
        print(f"  counters {res['counters']}")
    if a.csv:
        keys = list(res["rows"][0].keys())
        with open(a.csv, "w", encoding="utf-8") as f:
            f.write(f"# {head}\n")
            f.write(",".join(keys) + "\n")
            for r in res["rows"]:
                f.write(",".join(f"{r[k]:.8g}" for k in keys) + "\n")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"head": head, "summary": summ,
                       "counters": res["counters"]}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
