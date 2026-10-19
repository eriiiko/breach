"""tools/wind_drift_probe.py — the smoke plume's real drift vs the render wind.

Graphics arc G0 P2 / issue #51 (docs/graphics_direction_2026-10-18.md §8.1):
``renderer.gas_detail.tame_wind`` turns the EOS velocity planes into the
visual drift the smoke detail shader advects its noise by. This probe runs a
level headless and, per sim tick, measures on the SAME run:

  * the visible plume's CENTROID (weighted by the optical depth tau the gas
    medium draws -- ``gas_optical_depth`` over the trace gases with the light
    engine's own k_s) and its drift speed (finite difference over a window);
  * the plume-weighted mean of the HONEST velocity (``wind_tiles_per_tick``:
    the m/s planes / tile_size_m / tps) -- by continuity this is what the
    centroid follows wherever no source is adding smoke;
  * the plume-weighted mean of the TAMED velocity at the given v_ref /
    v_sens -- what the shader actually moves the noise by (x adv_gain).

The retune rule (§8.1): choose v_ref / v_sens so the visual drift matches the
plume's centroid drift. Analysis only: reads the sim, writes nothing to it
(the scripted detonation goes through the canonical payload executor, like
the studio's).

    C:/Users/steen/anaconda3/envs/data/python.exe tools/wind_drift_probe.py \\
        --level smoke_light_studio --ticks 168
    ... --level fire_studio --ticks 120 --detonate 24,16,24
    ... --v-ref 0.08 --v-sens 0.0625     (candidate taming dials)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from renderer.gas_detail import (WIND_V_REF, WIND_V_SENS, tame_wind,  # noqa: E402
                                 wind_tiles_per_tick)
from renderer.gas_medium import gas_optical_depth  # noqa: E402
from simulation import Simulation, gas_fixed  # noqa: E402
from simulation.gases import N_TRACE_GASES  # noqa: E402
from simulation.payloads import execute_payload  # noqa: E402


def _k_s(gas_table) -> np.ndarray:
    """The gas medium's per-gas extinction (GasMediumOverlay._bind_table): the
    light engine's own coefficient, mean over R/G/B."""
    return (np.asarray(gas_table.light_absorb_q16[:N_TRACE_GASES],
                       dtype=np.float64).mean(axis=1) / 65536.0)


def probe(level: str, ticks: int, detonate=None, v_ref=WIND_V_REF,
          v_sens=WIND_V_SENS, window: int = 6, start: int = 0):
    lvl = load_level(level)
    sim = Simulation(lvl, seed=42, breach_physics=bp, enable_recorder=False)
    sim.set_paused(False)
    tps = float(CFG.clock.ticks_per_second)
    tile = float(sim.gmap.tile_size_m)
    h, w = sim.gmap.solid.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    rows = []
    for t in range(ticks):
        if detonate is not None and t == detonate[2]:
            pay = sim.weapons_tables.payload_for_ammo("grenade_frag")
            execute_payload(sim.gmap, sim.edit_queue, sim.units, detonate[1],
                            detonate[0], pay, sim.rng, events=[],
                            kind="grenade")
        if sim.is_paused():
            sim.set_paused(False)
        sim.step()
        g = sim.gmap
        trace = gas_fixed.dequantize(g.gas[:N_TRACE_GASES])
        tau = gas_optical_depth(trace, _k_s(g.gases), base_absorb_scale=1.0,
                                plume_k_scale=1.0)
        m = float(tau.sum())
        if m <= 1e-9:
            rows.append(None)
            continue
        ux, uy = wind_tiles_per_tick(g.wind_x, g.wind_y, tile_size_m=tile,
                                     ticks_per_second=tps)
        tw = tame_wind(g.wind_x, g.wind_y, tile_size_m=tile,
                       ticks_per_second=tps, v_ref=v_ref, v_sens=v_sens)
        wsum = lambda a: float((tau * a).sum()) / m  # noqa: E731
        rows.append(dict(
            t=t + 1, m=m, cx=wsum(xx), cy=wsum(yy),
            honest=(wsum(ux), wsum(uy)), honest_speed=wsum(np.hypot(ux, uy)),
            tamed=(wsum(tw[..., 0]), wsum(tw[..., 1])),
            tamed_speed=wsum(np.hypot(tw[..., 0], tw[..., 1]))))
    return rows, dict(tile_size_m=tile, tps=tps)


def report(rows, window: int, start: int) -> dict:
    """Window-averaged drift: |dC|/dt from the centroid, vs the plume-weighted
    mean honest and tamed vectors over the same window. Windows whose plume
    mass changed by > 20 % are flagged (a source, not a drift, moves C)."""
    out = []
    for i in range(max(start, window), len(rows)):
        a, b = rows[i - window], rows[i]
        if a is None or b is None:
            continue
        drift = np.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"]) / window
        seg = [r for r in rows[i - window + 1:i + 1] if r is not None]
        hv = np.mean([r["honest"] for r in seg], axis=0)
        tv = np.mean([r["tamed"] for r in seg], axis=0)
        out.append(dict(t=b["t"], drift=drift, honest=float(np.hypot(*hv)),
                        tamed=float(np.hypot(*tv)),
                        honest_speed=float(np.mean([r["honest_speed"] for r in seg])),
                        tamed_speed=float(np.mean([r["tamed_speed"] for r in seg])),
                        dm=abs(b["m"] - a["m"]) / max(a["m"], 1e-9)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--level", default="smoke_light_studio")
    ap.add_argument("--ticks", type=int, default=168)
    ap.add_argument("--detonate", default=None, help="X,Y,TICK frag grenade")
    ap.add_argument("--v-ref", type=float, default=WIND_V_REF)
    ap.add_argument("--v-sens", type=float, default=WIND_V_SENS)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    det = tuple(int(v) for v in a.detonate.split(",")) if a.detonate else None
    rows, meta = probe(a.level, a.ticks, det, a.v_ref, a.v_sens)
    win = report(rows, a.window, a.start)
    print(f"[drift] {a.level}  tile {meta['tile_size_m']} m  {meta['tps']:.0f} tps"
          f"  v_ref {a.v_ref}  v_sens {a.v_sens} (tiles/tick)  window {a.window}")
    print("[drift]   tick  centroid  |mean u|  mean|u|  |mean tamed|  mean|tamed|  dM/M")
    for r in win[::a.window]:
        print(f"[drift] {r['t']:6d}  {r['drift']:8.4f}  {r['honest']:8.4f}  "
              f"{r['honest_speed']:7.4f}  {r['tamed']:11.4f}  {r['tamed_speed']:11.4f}"
              f"  {r['dm']:5.2f}")
    clean = [r for r in win if r["dm"] <= 0.2]
    if clean:
        md = lambda k: float(np.median([r[k] for r in clean]))  # noqa: E731
        print(f"[drift] SUMMARY over {len(clean)} source-free windows (median, "
              f"tiles/tick): centroid {md('drift'):.4f} | honest mean-vector "
              f"{md('honest'):.4f}, mean speed {md('honest_speed'):.4f} | tamed "
              f"mean-vector {md('tamed'):.4f}, mean speed {md('tamed_speed'):.4f}")


if __name__ == "__main__":
    main()
