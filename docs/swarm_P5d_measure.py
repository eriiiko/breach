"""arc #63 P5d measurement (docs/swarm_P5d_impl.md): headless bot matches on
generated freighters WITH broods against the same seed's ship WITHOUT them
(the same archetype with its [infestation] table removed: tile for tile the
ship, minus the broods and the opened doors). CPU path, the ship's own
match.toml (bots vs bots, the horde, 300 s), ``tools/run_match.run_match``.

Per match: the outcome and its length; the larvae alive at 0 / 30 / 60 / 120
/ 240 s and at the end; the bite dose the living marines took (Q16 units of
poison summed off the bite plane under their footprints, the plane the
coupling row reads) and how many marine-ticks were bitten; the bodies that
rose (a unit dead one tick and alive the next); the mean wall time per tick.

    python docs/swarm_P5d_measure.py <levels dir> [time limit s]

The ships are written into <levels dir> (they are not committed: generated
level folders, ruling 2026-11-06).
"""
import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tools"):
    sys.path.insert(0, str(p))
import os
os.chdir(ROOT)

import gen_ship as G  # noqa: E402
import level_loader  # noqa: E402
import ship_grammar as S  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import exchange, swarm  # noqa: E402

TPS = 24
MARKS_S = (0, 30, 60, 120, 240)
CASES = (("medium", 1), ("medium", 2), ("large", 1), ("large", 2))


def measure(level_name, levels_dir, limit_s):
    level = level_loader.load(level_name, str(levels_dir))
    spec = level.match
    if limit_s:
        spec = dataclasses.replace(spec, time_limit_s=float(limit_s))
    st = dict(larvae={}, dose=0, bitten=0, risen=0, alive_prev=None)

    def on_tick(sim):
        t = sim.tick
        if st["alive_prev"] is None:
            st["alive_prev"] = {u.id: u.alive for u in sim.units}
            st["larvae"][0] = sim.swarm.live_count("larva")
        if t % TPS == 0 and t // TPS in MARKS_S:
            st["larvae"][t // TPS] = sim.swarm.live_count("larva")
        st["end"] = sim.swarm.live_count("larva")
        plane = swarm.bite_plane(sim.gmap)
        for u in sim.units:
            was = st["alive_prev"].get(u.id, u.alive)
            if u.alive and not was:
                st["risen"] += 1
            st["alive_prev"][u.id] = u.alive
            if u.alive and not u.is_zombie and plane is not None:
                d = exchange.reduce_sum(plane, u.occupied_tiles())
                if d:
                    st["dose"] += d
                    st["bitten"] += 1

    t0 = time.perf_counter()
    o = run_match(level, spec, out=lambda *_: None, on_tick=on_tick)
    wall = time.perf_counter() - t0
    ticks = max(1, o.result.ticks)
    return dict(result=f"{o.result.outcome} team {o.result.team}",
                secs=o.result.ticks / TPS,
                larvae={**st["larvae"], "end": st.get("end")},
                dose=st["dose"] / 65536.0, bitten=st["bitten"],
                risen=st["risen"], converted=o.converted_alive,
                ms_per_tick=1000.0 * wall / ticks)


def main():
    levels_dir = Path(sys.argv[1])
    limit_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    levels_dir.mkdir(parents=True, exist_ok=True)
    arch = S.load_archetype("freighter")
    clean = dataclasses.replace(arch, infestation=())
    for size, seed in CASES:
        p = G.generate(seed, size, recipe=arch)
        c = G.generate(seed, size, recipe=clean)
        assert (p.g, p.attempt) == (c.g, c.attempt)
        G.write(p, levels_dir / p.name)
        G.write(c, levels_dir / (p.name + "_clean"))
        broods = [e.fields["count"] for e in p.entities
                  if e.class_name == "swarm_brood"]
        print(f"== {p.name}: {p.infestation}, broods in {list(p.broods)} "
              f"({len(broods)} broods, {sum(broods)} larvae)", flush=True)
        for tag, name in (("broods", p.name), ("clean", p.name + "_clean")):
            r = measure(name, levels_dir, limit_s)
            print(f"   {tag:6s} {r['result']:14s} {r['secs']:6.1f} s  larvae "
                  f"{r['larvae']}  bite dose {r['dose']:.2f} over "
                  f"{r['bitten']} marine-ticks  risen {r['risen']} "
                  f"(converted alive {r['converted']})  "
                  f"{r['ms_per_tick']:.2f} ms/tick", flush=True)


if __name__ == "__main__":
    main()
