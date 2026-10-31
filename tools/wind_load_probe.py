r"""tools/wind_load_probe.py -- what the air does to a body (#140).

The headless numbers behind the wind law (docs/wind_on_units_design_2026-10-31.md
§6.1): per tick, the wind LOAD at a named unit's footprint through THE read,
``GameMap.footprint_wind_load`` -- ``W = N_raw * (v^2 >> 16)`` -- as the
dynamic pressure ``q = rho_ref * W / 2^33`` in Pa (``rho_ref`` is the
``[materials.air]`` density row, the one owner), with the MEAN over the
footprint's gas tiles beside the max (the law reads the max; the mean shows
how much a one-tile jet over-reads a three-wide body) and the peak ``|v|``.

Scenes (each a ``Simulation`` built here, except ``match``, which watches
``tools/run_match.run_match`` through its ``on_tick`` hook -- never a second
match loop):

  doorway   ship_demo's zombie room raised by +0.44 atm (~1.44 atm, #139),
            every door outside it opened, then its west doors: a marine
            standing in the antechamber doorway. Ticks above q_slow / q_fall.
  match     ship_demo's match, seed 1 (the #139 stall, ticks ~600-1400): per
            tick the largest q at any living marine.
  grenade   the retired push's calibration grenade (radius 5, pressure
            10.0, 48x48 open room, 1 m tiles), then the SHIPPED frag_standard
            row through execute_payload at 0.333 m tiles: peak q at a marine
            by centre distance d = 2..12 -> the knockdown ring.
  fire      fire_smoke_studio's steady fire: a marine north of the fire row
            and one at its west end, from 10 s to 60 s.

Reads only; writes nothing. Run:
    C:/Users/steen/anaconda3/python.exe tools/wind_load_probe.py doorway
    C:/Users/steen/anaconda3/python.exe tools/wind_load_probe.py grenade fire
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "cpp" / "build" / "Release", ROOT, ROOT / "src",
           ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.atmosphere_fixed import FP_ONE, FP_SHIFT  # noqa: E402
from simulation.gases import INERT_N2, O2  # noqa: E402
from simulation.materials import MAT_AIR, MAT_DOOR  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.unit import Unit  # noqa: E402

TPS = int(CFG.clock.ticks_per_second)
Q_SLOW = float(getattr(CFG.exchange, "wind_q_slow_pa", 39.0))
Q_FALL = float(getattr(CFG.exchange, "wind_q_fall_pa", 245.0))


def rho_ref(gmap) -> float:
    return float(gmap.materials.density[MAT_AIR])


def q_of(gmap, w: int) -> float:
    return rho_ref(gmap) * w / float(1 << (2 * FP_SHIFT + 1))


def footprint_read(gmap, u):
    """(W max, W mean over gas tiles, peak |v| m/s) at u's footprint. The
    max is THE read (GameMap.footprint_wind_load); the mean and |v| are the
    same per-tile terms, for the report."""
    w_max = gmap.footprint_wind_load(u.tile_y, u.tile_x, u.footprint)
    ws, vmax = [], 0.0
    for (tx, ty) in u.occupied_tiles():
        if not (0 <= ty < gmap.solid.shape[0] and 0 <= tx < gmap.solid.shape[1]):
            continue
        if gmap.solid[ty, tx]:
            continue
        vx, vy = int(gmap.wind_x[ty, tx]), int(gmap.wind_y[ty, tx])
        n = int(gmap.gas[O2][ty, tx]) + int(gmap.gas[INERT_N2][ty, tx])
        ws.append(n * ((vx * vx + vy * vy) >> FP_SHIFT))
        vmax = max(vmax, float(np.hypot(vx, vy)) / FP_ONE)
    w_mean = sum(ws) / len(ws) if ws else 0.0
    return w_max, w_mean, vmax


def _band_summary(qs, label):
    qs = np.asarray(qs, dtype=float)
    def runs(mask):
        best = cur = 0
        for m in mask:
            cur = cur + 1 if m else 0
            best = max(best, cur)
        return best
    print(f"  [{label}] ticks {len(qs)}: max q {qs.max():.1f} Pa, mean q "
          f"{qs.mean():.2f} Pa; ticks >= {Q_SLOW:g} Pa: "
          f"{int((qs >= Q_SLOW).sum())} (longest run {runs(qs >= Q_SLOW)}); "
          f">= {Q_FALL:g} Pa: {int((qs >= Q_FALL).sum())} "
          f"(longest run {runs(qs >= Q_FALL)})")


# ---------------------------------------------------------------------------
def scene_doorway(vent_ticks=24 * 20):
    from simulation.field_edit import EditMode, FieldEdit, Region
    level = load_level("ship_demo")
    sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    g = sim.gmap
    sim.edit_queue.enqueue(FieldEdit("atmosphere", Region.RECT,
                                     (18, 41, 36, 58), 0.44, EditMode.ADD))
    sim.set_paused(False)
    for _ in range(30):
        sim.step()
    for (r, c) in np.argwhere(g.material == MAT_DOOR).tolist():
        if not (17 <= r <= 37 and 40 <= c <= 59):
            g.destroy_wall(r, c)
    for _ in range(24):
        sim.step()
    u = Unit("doorway", x=37.0, y=22.0, team=0)
    sim.add_unit(u)
    print(f"doorway: room at {int(g.atmosphere[27, 50]) / FP_ONE:.3f} atm, "
          f"marine footprint x 37..39 y 22..24 against the door (col 40, "
          f"rows 21..24); rho_ref {rho_ref(g)} kg/m3")
    for r in range(21, 25):
        g.destroy_wall(r, 40)
    print("  tick   t[s]   q_max[Pa]  q_mean[Pa]  |v|max[m/s]  P_room[atm]  status")
    qs = []
    for t in range(vent_ticks):
        sim.step()
        w, wm, v = footprint_read(g, u)
        q = q_of(g, w)
        qs.append(q)
        if t < 12 or t % 24 == 0:
            st = ",".join(str(s.kind) for s in u.statuses) or "-"
            print(f"  {t + 1:4d} {(t + 1) / TPS:6.2f} {q:10.1f} {q_of(g, wm):11.1f} "
                  f"{v:12.2f} {int(g.atmosphere[27, 50]) / FP_ONE:12.3f}  {st}")
    _band_summary(qs, "doorway")
    print(f"  marine position after the vent: ({u.x}, {u.y})")


def scene_match(t_lo=550, t_hi=1450):
    from run_match import run_match
    level = load_level("ship_demo")
    per_tick = []

    def hook(sim):
        if not (t_lo <= sim.tick <= t_hi):
            return
        g = sim.gmap
        best = (0.0, 0.0, 0.0, None)
        for u in sim.units:
            if not u.alive or u.is_zombie:
                continue
            w, wm, v = footprint_read(g, u)
            q = q_of(g, w)
            if q > best[0]:
                best = (q, q_of(g, wm), v, u.name)
        per_tick.append((sim.tick,) + best)

    out = run_match(level, seed=1, on_tick=hook)
    print(f"match ship_demo seed 1: {out.result.outcome} team {out.result.team} at tick {out.result.ticks}; the "
          f"largest q at any living marine, ticks {t_lo}..{t_hi}")
    print("  tick   q_max[Pa]  q_mean[Pa]  |v|max  unit")
    for (t, q, qm, v, name) in per_tick:
        if t % 48 == 0 or q >= Q_FALL:
            print(f"  {t:5d} {q:10.1f} {qm:11.1f} {v:7.2f}  {name}")
    _band_summary([p[1] for p in per_tick] or [0.0], "match")


def _blast_room(tile_size_m, h=48, w=48):
    from level_loader import LevelData
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    return LevelData(name="wind_ring", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=tile_size_m,
                     diffuse_path=Path("."))


def _ring(label, tile_size_m, detonate, dists, n_ticks):
    from simulation.status import KNOCKED_DOWN
    print(f"grenade [{label}]: 48x48 room at {tile_size_m} m tiles, blast at "
          f"(24, 12), marine centre d tiles along +x, peak over {n_ticks} ticks")
    print("   d   peak q_max[Pa]  peak q_mean[Pa]  peak |v|  tick  knocked (live law)")
    for d in dists:
        sim = Simulation(_blast_room(tile_size_m), seed=777, breach_physics=bp,
                         enable_recorder=False)
        cy, cx = 24, 12
        u = Unit("P", x=cx + d - 1, y=cy - 1, team=0)
        sim.add_unit(u)
        peak = (0.0, 0.0, 0.0, 0)
        down = False
        for t in range(n_ticks):
            if t == 1:
                detonate(sim, cy, cx)
            sim.set_paused(False)
            sim.step()
            w, wm, v = footprint_read(sim.gmap, u)
            q = q_of(sim.gmap, w)
            if q > peak[0]:
                peak = (q, q_of(sim.gmap, wm), v, t + 1)
            down = down or any(s.kind == KNOCKED_DOWN for s in u.statuses)
        print(f"  {d:2d} {peak[0]:15.1f} {peak[1]:16.1f} {peak[2]:9.2f} "
              f"{peak[3]:5d}  {'yes' if down else 'no'}")


def scene_grenade(dists=range(2, 13), n_ticks=40):
    """The [exchange] calibration grenade (radius 5, pressure 10.0, 1 m
    tiles -- the scenario the retired push was tuned on), then the SHIPPED
    frag_standard row through execute_payload at the game's 0.333 m tiles."""
    from simulation.physics import apply_explosion
    from simulation.payloads import execute_payload
    from simulation.weapons import get_tables, rebuild_tables

    def calibration(sim, cy, cx):
        apply_explosion(sim.gmap, sim.edit_queue, cy, cx, 5, 10.0, 200)
    _ring("calibration: radius 5, pressure 10.0", 1.0, calibration,
          dists, n_ticks)

    rebuild_tables(tile_size_m=0.333)
    frag = get_tables().payload_for_ammo("grenade_frag")

    def shipped(sim, cy, cx):
        execute_payload(sim.gmap, sim.edit_queue, sim.units, cy, cx, frag,
                        sim.rng)
    print()
    _ring(f"shipped frag_standard: radius {frag.radius}, pressure "
          f"{frag.pressure}, heat {frag.heat_amount}", 0.333, shipped,
          dists, n_ticks)


def scene_fire(seconds=60, from_s=10):
    level = load_level("fire_smoke_studio")
    sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    # The fire row is furniture at y 22, x 23..27 (lit at 0.5 s).
    north = Unit("north", x=24.0, y=19.0, team=0)   # rows 19..21, over x 24..26
    west = Unit("west", x=20.0, y=21.0, team=0)     # cols 20..22, beside x 23
    for u in (north, west):
        sim.add_unit(u)
    sim.set_paused(False)
    rows = {u.name: [] for u in (north, west)}
    for t in range(seconds * TPS):
        sim.set_paused(False)
        sim.step()
        if t + 1 < from_s * TPS:
            continue
        for u in (north, west):
            w, wm, v = footprint_read(sim.gmap, u)
            rows[u.name].append((q_of(sim.gmap, w), q_of(sim.gmap, wm), v))
    fire_q = int(np.count_nonzero(sim.gmap.fire[22, 23:28]))
    print(f"fire: fire_smoke_studio, {from_s}..{seconds} s; burning tiles on "
          f"the row at the end: {fire_q}/5")
    print("  unit    q_max: max / mean [Pa]   q_mean: max / mean [Pa]   |v| max")
    for name, r in rows.items():
        a = np.asarray(r)
        print(f"  {name:6s} {a[:, 0].max():10.2f} / {a[:, 0].mean():6.2f} "
              f"{a[:, 1].max():14.2f} / {a[:, 1].mean():6.2f} "
              f"{a[:, 2].max():10.2f}")
        _band_summary(a[:, 0], name)


SCENES = {"doorway": scene_doorway, "match": scene_match,
          "grenade": scene_grenade, "fire": scene_fire}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenes", nargs="*",
                    help=f"any of {', '.join(SCENES)} (default: all)")
    args = ap.parse_args()
    args.scenes = args.scenes or list(SCENES)
    for name in args.scenes:
        if name not in SCENES:
            ap.error(f"unknown scene {name!r}; choose from {list(SCENES)}")
    print(f"q_slow {Q_SLOW:g} Pa, q_fall {Q_FALL:g} Pa (stability 1.0)")
    for name in args.scenes:
        SCENES[name]()
        print()


if __name__ == "__main__":
    main()
