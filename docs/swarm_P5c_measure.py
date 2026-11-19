"""arc #63 P5c measurement (docs/swarm_P5c_impl.md): what it takes to destroy a
corpse. Headless, CPU path, the continuous ruleset; P5b's script is the
precedent (docs/swarm_P5b_measure.py).

FRAGS: in the larva studio's nursery (no larvae on it) a marine's corpse and a
turned zombie's corpse (killed at exactly their hit points, so each starts
whole at hp 0), then ``frag_standard`` detonations on the tile ``d`` east of
the corpse's centre tile until the body is gone (``corpse.destroyed``); per
distance, the frags it took and the hp each took off.

FIRE: a 3 x 3 furniture pile lit and burning for 20 s (the body-temperature
test's scene, ``tests/test_body_temp.py``), then the corpse is laid with its
footprint over the pile's east edge (as the marine of that test stands) or on
its middle; seconds until the body is gone (or none within the limit).

    python docs/swarm_P5c_measure.py [fire limit seconds (default 180)]
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tools"):
    sys.path.insert(0, str(p))
import os
os.chdir(ROOT)
import numpy as np
import breach_physics as bp
import level_loader
from level_loader import LevelData
from simulation import body_temp, corpse, unit_traits
from simulation.damage import KINETIC, DamagePacket, apply_packet
from simulation.materials import MAT_AIR, MAT_FURNITURE, MAT_HULL
from simulation.payloads import execute_payload
from simulation.ruleset import ContinuousRealtime
from simulation.simulation import Simulation
from simulation.unit import Unit

TPS = 24
ONE = 1 << 16


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


def _corpse(sim, name, x, y, team):
    u = Unit(name, x=float(x), y=float(y), team=team)
    sim.add_unit(u)
    _step(sim)
    apply_packet(u, DamagePacket(amount=u.current_hp, dtype=KINETIC), None,
                 source="explosion")
    assert not u.alive and u.hp_q == 0
    return u


def frags(team, d, seed=1):
    lv = level_loader.load("larva_studio", str(ROOT / "levels"))
    sim = Simulation(lv, seed=seed, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    u = _corpse(sim, "K", 7, 25, team)                 # the nursery, south-west
    row = sim.weapons_tables.payloads.by_name["frag_standard"]
    cy, cx = int(u.center_tile_y()), int(u.center_tile_x()) + d
    took = []
    while not corpse.destroyed(u) and len(took) < 40:
        hp0 = u.hp_q
        execute_payload(sim.gmap, sim.edit_queue, sim.units, cy, cx, row, sim.draws)
        _step(sim)
        took.append((hp0 - u.hp_q) / ONE)
        if took[-1] == 0 and len(took) > 2:
            break
    full = unit_traits.max_hp_q(u) / ONE
    return full, corpse.body_q(u) / ONE, took, corpse.destroyed(u)


def fire(team, where, limit_s):
    tm = np.full((40, 40), MAT_HULL, dtype=np.int32)
    tm[1:39, 1:39] = MAT_AIR
    tm[19:22, 19:22] = MAT_FURNITURE
    lv = LevelData(name="fire_corpse", version="2", path=Path("."), tilemap=tm,
                   tile_size_m=0.333, diffuse_path=Path("."), entities=[], wires=[])
    s = Simulation(lv, seed=3, breach_physics=bp, enable_recorder=False,
                   ruleset=ContinuousRealtime())
    _step(s)
    row = s.weapons_tables.payloads.by_name["studio_ignite"]
    for y in range(19, 22):
        for x in range(19, 22):
            execute_payload(s.gmap, s.edit_queue, s.units, y, x, row, s.draws)
    _step(s, 20 * TPS)
    x, y = {"edge": (21.5, 19.5), "middle": (19.5, 19.5)}[where]
    u = _corpse(s, "K", x, y, team)
    amb = body_temp.ambient_q()
    peak = 0
    for t in range(limit_s * TPS):
        _step(s)
        peak = max(peak, int(u.body_temp_q) - amb)
        if corpse.destroyed(u):
            return (t + 1) / TPS, peak / ONE, u.hp_q / ONE
    return None, peak / ONE, u.hp_q / ONE


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    for team, who in ((0, "marine"), (255, "zombie")):
        for d in (0, 1, 2):
            full, body, took, gone = frags(team, d)
            print(f"FRAG {who} corpse (max hp {full:g}, body {body:g}): frag at d = {d}: "
                  f"{len(took)} frags, hp off each {took[:3]}{'...' if len(took) > 3 else ''}, "
                  f"destroyed {gone}")
    for team, who in ((0, "marine"), (255, "zombie")):
        for where in ("edge", "middle"):
            ts, peak, hp = fire(team, where, limit)
            print(f"FIRE {who} corpse on the pile's {where}: destroyed after "
                  f"{ts} s (body peak +{peak:.0f} C over ambient, hp {hp:.1f})")


if __name__ == "__main__":
    main()
