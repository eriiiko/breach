"""arc #63 P5b measurement (docs/swarm_P5b_impl.md): corpses in the larva
studio. Headless, CPU path, the studio as checked in, the continuous ruleset
(the conversion door runs every tick), seeds 1..5.

RISE: a marine's corpse in the middle of the west brood and a zombie's corpse
in the middle of the east brood, killed at t = 0 (an uncredited packet: not a
zombie's kill, so only the larvae can raise them); per corpse the time to
rise (or none within the limit) and the mean count of larvae under its
footprint until then.

FRAG: a marine's corpse in the nursery (no larvae on it), a frag
(``frag_standard``) detonated on the tile beside its footprint: its integrity
and its hit points before and after, and the frag's hits on the corpse.

    python docs/swarm_P5b_measure.py [limit seconds (default 120)] [warm-up seconds]
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tools"):
    sys.path.insert(0, str(p))
import os
os.chdir(ROOT)
import breach_physics as bp
import level_loader
from simulation import exchange, swarm
from simulation.damage import KINETIC, DamagePacket, apply_packet
from simulation.events import UnitHitEvent
from simulation.payloads import execute_payload
from simulation.ruleset import ContinuousRealtime
from simulation.simulation import Simulation
from simulation.unit import Unit

TPS = 24
ONE = 1 << 16


def _sim(seed):
    lv = level_loader.load("larva_studio", str(ROOT / "levels"))
    return Simulation(lv, seed=seed, breach_physics=bp, enable_recorder=False,
                      ruleset=ContinuousRealtime())


def _corpse(sim, name, x, y, team):
    u = Unit(name, x=float(x), y=float(y), team=team)
    sim.add_unit(u)
    apply_packet(u, DamagePacket(amount=10_000.0, dtype=KINETIC), None,
                 source="explosion")
    return u


def rise(seed, limit_s, warm_s=0):
    sim = _sim(seed)
    for _ in range(warm_s * TPS):          # let the broods spread first
        sim.set_paused(False)
        sim.step()
    corpses = {"marine": _corpse(sim, "KM", 12, 16, 0),     # brood_west (9..17)
               "zombie": _corpse(sim, "KZ", 23, 16, 255)}  # brood_east (20..28)
    out = {k: [None, 0] for k in corpses}
    for t in range(limit_s * TPS):
        sim.set_paused(False)
        sim.step()
        plane = swarm.count_plane(sim.gmap)
        for k, u in corpses.items():
            if out[k][0] is not None:
                continue
            if u.alive:
                out[k][0] = (t + 1) / TPS
            else:
                out[k][1] += exchange.reduce_sum(plane, u.occupied_tiles())
    res = {}
    for k, (ts, nsum) in out.items():
        ticks = (ts * TPS) if ts is not None else limit_s * TPS
        res[k] = (ts, nsum / max(1, ticks))
    return res


def frag(seed):
    sim = _sim(seed)
    u = _corpse(sim, "KF", 7, 25, 0)                    # nursery, south-west, no brood
    sim.set_paused(False)
    sim.step()
    row = sim.weapons_tables.payloads.by_name["frag_standard"]
    before = (int(u.corpse_integrity_q), int(u.hp_q))
    events = []
    execute_payload(sim.gmap, sim.edit_queue, sim.units, 26, 10, row, sim.draws,
                    events=events)                  # the tile beside the footprint (7..9 x 25..27)
    for _ in range(5):
        sim.set_paused(False)
        sim.step()
    after = (int(u.corpse_integrity_q), int(u.hp_q))
    hits = sum(1 for e in events if isinstance(e, UnitHitEvent) and e.unit_id == u.id)
    return before, after, hits


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    warm = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(f"broods spread for {warm} s before the deaths")
    for seed in range(1, 6):
        r = rise(seed, limit, warm)
        print(f"seed {seed}: " + "; ".join(
            f"{k} corpse rose at {v[0] if v[0] is None else round(v[0], 2)} s "
            f"(mean {v[1]:.1f} larvae under it)" for k, v in r.items()))
    b, a, hits = frag(1)
    print(f"frag beside a corpse: integrity {b[0] / ONE:.1f} -> {a[0] / ONE:.1f} HP, "
          f"hp {b[1] / ONE:.1f} -> {a[1] / ONE:.1f}, hits on the corpse {hits}")


if __name__ == "__main__":
    main()
