r"""tools/blast_light_probe.py -- does a blast light its room, and does its own
smoke hide that light? (#94; G4 #75 P1, docs/explosion_look_design_2026-11-07.md
§3.6 and §6 B3).

Headless, no renderer. Runs ``levels/blast_final`` (four charges, one per room,
all firing at the same tick) with the room's lamps requested as the game does
(the level's ``[[light]]`` entries through THE light assembly,
``renderer/frame_lights``, no flashlight, no cursor lamp, the level's sky), and
reads the light through THE light accessor (``simulation.light_field``) at the
tiles 3 and 6 tiles east and west of each charge, at ticks +1, +2, +4, +8, +12
after the detonation and at the tick before it (the lamps-only baseline):

  (a) as shipped: the light the tick computed;
  (b) smoke off: a THROWAWAY sim run identically to the same tick, its smoke
      trace plane zeroed in place (``gmap.gas[SMOKE] = 0``, a probe fixture's
      write on a sim that is then discarded -- never a game path), relit
      (``Simulation.relight``) and read.

``scalar`` is ``LightView.scalar`` (the brightest channel, light units); the
RGB table is ``light_at``'s integer triple dequantized to light units. No units
are spawned: a body on a probe tile would absorb there.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe tools/blast_light_probe.py
    C:/Users/steen/anaconda3/envs/data/python.exe tools/blast_light_probe.py --out docs/explosion_look_design_2026-11-07/light_measurement.md
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
from level_lights import monotonic_total_tick, partition_lights  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from renderer import frame_lights  # noqa: E402
from simulation import Simulation, light_field, optics_fixed  # noqa: E402
from simulation.events import ExplosionEvent  # noqa: E402
from simulation.gases import SMOKE  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402

LEVEL = "blast_final"
# (row name, x, y) of each charge, from levels/blast_final/level.toml
CHARGES = (("frag_standard", 22, 19), ("breach_focus", 53, 19),
           ("magazine_cookoff", 22, 44), ("demolition_c4", 53, 44))
OFFSETS = (-6, -3, 3, 6)          # tiles along x (west negative)
AFTER = (1, 2, 4, 8, 12)          # ticks after the detonation


class Run:
    """One deterministic run of the level with the lamps requested every tick."""

    def __init__(self):
        self.level = load_level(LEVEL)
        self.sim = Simulation(self.level, seed=42, breach_physics=bp,
                              enable_recorder=False, ruleset=ContinuousRealtime())
        self.sim.set_paused(False)
        self.lights_in, _ = partition_lights(self.level.lights, self.level.width,
                                             self.level.height)
        self.sky = frame_lights.sky_for_level(
            getattr(self.level, "boundary", "space"), CFG, bp)
        self.spt = 1.0 / float(CFG.clock.ticks_per_second)
        self.tpr = int(CFG.clock.ticks_per_round)
        self.detonated_at = None

    def rows(self):
        s = self.sim
        total = monotonic_total_tick(s.turn_number, self.tpr, s.tick)
        specs = frame_lights.frame_light_specs(
            self.lights_in, total_tick=total, sim_time_per_tick=self.spt).specs
        return frame_lights.cone_rows(specs, self.level.width, self.level.height,
                                      int(bp.L_FINE_BITS))

    def step(self):
        s = self.sim
        s.set_light(True, self.rows(), self.sky)
        if s.is_paused():
            s.set_paused(False)
        s.step()
        if self.detonated_at is None and any(
                isinstance(e, ExplosionEvent) for e in s.tick_events):
            self.detonated_at = s.tick

    def step_to(self, tick):
        while self.sim.tick < tick:
            self.step()


def read(gmap):
    """{(name, dx): (scalar, (r, g, b))} at the probe tiles, in light units."""
    sc = light_field.read_light(gmap).scalar
    fb = optics_fixed.light_fine_bits(gmap)
    out = {}
    for name, x, y in CHARGES:
        for dx in OFFSETS:
            q = np.asarray(light_field.light_at(gmap, y, x + dx), dtype=np.int64)
            rgb = tuple(float(v) for v in optics_fixed.dequantize_light(q, fb))
            out[(name, dx)] = (float(sc[y, x + dx]), rgb)
    return out


def _fmt(v):
    return f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="also write the tables here (.md)")
    a = ap.parse_args()

    # (a): one run, lamps requested every tick, read after each tick.
    run_a = Run()
    while run_a.detonated_at is None:
        run_a.step()
        if run_a.sim.tick > 400:
            raise SystemExit("no detonation in 400 ticks")
    t0 = run_a.detonated_at
    base_run = Run()                    # the baseline: a run stopped at t0 - 1
    base_run.step_to(t0 - 1)
    assert base_run.detonated_at is None
    shipped = {-1: read(base_run.sim.gmap)}
    del base_run
    for k in AFTER:
        run_a.step_to(t0 + k)
        shipped[k] = read(run_a.sim.gmap)

    # (b): a throwaway run per tick, smoke zeroed after the fact, relit.
    smoke_off = {}
    for k in AFTER:
        r = Run()
        r.step_to(t0 + k)
        g = r.sim.gmap
        g.gas[SMOKE] = 0                    # probe fixture; this sim is discarded
        r.sim.set_light(True, r.rows(), r.sky)
        r.sim.relight()
        smoke_off[k] = read(g)
        del r

    lines = [f"# Blast light at the units' tiles, {LEVEL} (G4 #75 P1, #94)", "",
             f"Generated by `tools/blast_light_probe.py`; the volley detonates on "
             f"tick {t0}. Light units (`LightView.scalar`, the brightest channel); "
             "dx in tiles from the charge along its row, west negative. base = the "
             "tick before the detonation (lamps only). (a) as shipped; (b) the "
             "smoke trace plane zeroed on a throwaway sim at the same tick and "
             "relit. No units on the probe tiles.", ""]
    for name, x, y in CHARGES:
        lines += [f"## {name} at ({x}, {y})", ""]
        lines.append("| tick | " + " | ".join(f"dx {dx:+d} (a)" for dx in OFFSETS)
                     + " | " + " | ".join(f"dx {dx:+d} (b)" for dx in OFFSETS)
                     + " |")
        lines.append("|" + "---|" * (1 + 2 * len(OFFSETS)))
        base = shipped[-1]
        lines.append(f"| {t0 - 1} (base) | "
                     + " | ".join(_fmt(base[(name, dx)][0]) for dx in OFFSETS)
                     + " | " + " | ".join("-" for _ in OFFSETS) + " |")
        for k in AFTER:
            sa, sb = shipped[k], smoke_off[k]
            lines.append(f"| +{k} | "
                         + " | ".join(_fmt(sa[(name, dx)][0]) for dx in OFFSETS)
                         + " | "
                         + " | ".join(_fmt(sb[(name, dx)][0]) for dx in OFFSETS)
                         + " |")
        lines.append("")
    lines += ["RGB at dx +3, light units:", "",
              "| row | tick | (a) r, g, b | (b) r, g, b |", "|---|---|---|---|"]
    for name, _x, _y in CHARGES:
        lines.append(f"| {name} | base | "
                     + ", ".join(_fmt(v) for v in shipped[-1][(name, 3)][1])
                     + " | - |")
        for k in AFTER:
            ra = shipped[k][(name, 3)][1]
            rb = smoke_off[k][(name, 3)][1]
            lines.append(f"| {name} | +{k} | " + ", ".join(_fmt(v) for v in ra)
                         + " | " + ", ".join(_fmt(v) for v in rb) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
