"""arc #63 P5a measurement (docs/swarm_P5a_impl.md): hit points a marine will
lose to the larvae in the larva studio -- a walk across the nursery, a sprint
across, 10 s standing in the brood -- mean over seeds 1..5. Headless, CPU
path, the studio as checked in. The harm counted is the law's own: the bite
doses taken DURING the exposure x hp_per_unit (10 HP a unit, neutral suit),
arriving over the following half minute; the run stops when the exposure ends.

    python docs/swarm_P5a_measure.py [warm-up seconds before the marine enters]
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
from control_source import quantize_stick_direction
from simulation import exchange, poison, swarm
from simulation.ruleset import ContinuousRealtime
from simulation.simulation import Simulation
from simulation.unit import Unit

TPS = 24
Y = 16                       # anchor row: footprint rows 16..18, mid-nursery
X0, X1 = 6, 29               # anchor x: footprint 6..8 (west wall) .. 29..31 (east wall)
STAND_X = 17                 # footprint 17..19: between the two broods' centres
TOTAL_S = 60


def run(seed, mode, warm_s=0):
    lv = level_loader.load("larva_studio", str(ROOT / "levels"))
    sim = Simulation(lv, seed=seed, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    for _ in range(warm_s * TPS):          # let the broods spread first
        sim.set_paused(False)
        sim.step()
    start_x = STAND_X if mode == "stand" else X0
    u = Unit("M", x=float(start_x), y=float(Y), team=0)
    sim.add_unit(u)
    doses = []
    phase = [None]
    real = poison.dose

    def spy(unit, amount_q, source_id=-1, cause=None, __r=real):
        if unit is u and cause is not None and phase[0] is None:
            doses.append(int(amount_q))
        return __r(unit, amount_q, source_id, cause) if cause is not None else \
            __r(unit, amount_q, source_id)
    exchange.poison.dose = spy
    try:
        east = quantize_stick_direction(1.0, 0.0, 0.0)
        west = quantize_stick_direction(-1.0, 0.0, 0.0)
        thr = {"walk": 0, "sprint": 65536, "stand": 0}[mode]
        phase_ticks = None
        dens = []
        phase[0] = None
        for t in range(TOTAL_S * TPS):
            if mode in ("walk", "sprint"):
                if u.x < X1 and phase_ticks is None:
                    sim.set_move_dir(u.id, east[0], east[1], thr)
                elif phase_ticks is None:
                    phase_ticks = t
                    phase[0] = t
                    sim.clear_move_dir(u.id)
            else:
                if t < 10 * TPS:
                    sim.clear_move_dir(u.id)
                elif u.x > X0:
                    if phase_ticks is None:
                        phase_ticks = t
                        phase[0] = t
                    sim.set_move_dir(u.id, west[0], west[1], 65536)   # sprint out
                else:
                    sim.clear_move_dir(u.id)
            sim.set_paused(False)
            sim.step()
            if phase_ticks is None:
                tiles = u.occupied_tiles()
                dens.append(sum(swarm.count_on_tile(sim.gmap, "larva", tx, ty)
                                for tx, ty in tiles))
            if phase_ticks is not None:
                break                      # the exposure is over: its harm is H x dose
        return dict(hp_lost=10.0 * sum(doses) / 65536.0, alive=u.alive,
                    secs=(phase_ticks or 0) / TPS, units=sum(doses) / 65536.0,
                    under=float(np.mean(dens)) if dens else 0.0)
    finally:
        exchange.poison.dose = real


if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]
    warm = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for mode in ("walk", "sprint", "stand"):
        rs = [run(s, mode, warm) for s in seeds]
        print(f"warm {warm:3d} s {mode:6s} seeds {seeds}: hp lost {[round(r['hp_lost'], 1) for r in rs]} "
              f"mean {np.mean([r['hp_lost'] for r in rs]):.1f}; poison units "
              f"{np.mean([r['units'] for r in rs]):.2f}; phase {np.mean([r['secs'] for r in rs]):.2f} s; "
              f"larvae under the footprint (mean while in) {np.mean([r['under'] for r in rs]):.1f}",
              flush=True)
