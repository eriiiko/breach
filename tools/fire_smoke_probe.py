r"""tools/fire_smoke_probe.py — the headless numbers behind a fire + smoke
tuning session (#12, 2026-10-22; procedure: the `fire-smoke-tuning` skill).

Runs a level (default ``fire_smoke_studio``) without a window and prints what
the screen cannot tell you apart:

  * a TIME TABLE at one burning tile: fire intensity, fuel left, the tile's
    temperature, the hottest gas within 5 tiles, and the room's smoke total /
    densest tile -- "did the fire go out with fuel left?", "is the gas hot
    enough to glow (~800 K)?", "is smoke piling up or being removed?";
  * at the END, the smoke density and wind along one row -- "is there a trail
    to the vent that the renderer is simply not drawing?".

Reads the sim only through the GameMap's fields, through the Q16 / trace
boundary modules; changes nothing. Stepped with ContinuousRealtime and
unpaused, so the level's timed charges fire (a Simulation starts paused).

Run:
    C:/Users/steen/anaconda3/python.exe tools/fire_smoke_probe.py
    C:/Users/steen/anaconda3/python.exe tools/fire_smoke_probe.py --seconds 120 --no-vents
    C:/Users/steen/anaconda3/python.exe tools/fire_smoke_probe.py --tile 25,22 --row 22
    C:/Users/steen/anaconda3/python.exe tools/fire_smoke_probe.py --seconds 12 --dump tests/data/fire_smoke_studio_flame_inputs.npz --dump-at 4,8,12

``--dump`` (graphics arc G3 F0b, docs/g3_fire_design_2026-10-23.md §13.2)
also saves, at the ``--dump-at`` seconds, exactly what the flame VFX reads per
tick: fire intensity I, the game temperature (dequantized ΔT), the TAMED wind
(``renderer.gas_detail.tame_wind`` with the relief's dials -- the one wind
product, tiles/tick) and the solid / thermal_solid / flammable / vacuum /
ambient masks, cropped to the level's non-space box, into one compressed .npz.
The flame's numpy spec twin (``renderer/fire_vfx_ref.py``) is driven by it.
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
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation, gas_fixed  # noqa: E402
from simulation.gases import SMOKE  # noqa: E402
from simulation.materials import MAT_AIR  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from temperature_scale import load as load_temperature_scale  # noqa: E402
from simulation.fire_fixed import dequantize as dequantize_fire  # noqa: E402
from simulation.wall_fixed import dequantize as dequantize_hp  # noqa: E402

dequantize_q16 = gas_fixed.dequantize_q16      # temperature, wind: plain Q16


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--level", default="fire_smoke_studio")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--every", type=float, default=10.0, help="table step, s")
    ap.add_argument("--tile", default="25,22", help="x,y of the burning tile")
    ap.add_argument("--row", type=int, default=None,
                    help="row for the end-of-run smoke/wind profile (default: the tile's)")
    ap.add_argument("--no-vents", action="store_true",
                    help="drop every duct/vent entity (isolate the burn)")
    ap.add_argument("--units", action="store_true",
                    help="spawn the level's squad and report its HP")
    ap.add_argument("--dump", default=None,
                    help="save the flame VFX's per-tick inputs to this .npz")
    ap.add_argument("--dump-at", default="4,8,12",
                    help="seconds at which --dump captures (comma list)")
    a = ap.parse_args()

    tx, ty = (int(v) for v in a.tile.split(","))
    row = ty if a.row is None else a.row
    lvl = load_level(a.level)
    if a.no_vents:
        lvl.entities = [e for e in lvl.entities if e.class_name not in ("vent", "duct")]
    sim = Simulation(lvl, seed=42, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    if a.units:
        for s in lvl.spawns:
            sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team, footprint=s.footprint))
    sim.set_paused(False)
    g = sim.gmap
    from config import CFG
    tps = int(CFG.clock.ticks_per_second)
    ts = load_temperature_scale()
    kelvin = lambda q: ts.to_kelvin(float(dequantize_q16(q)))        # noqa: E731

    hp0 = max(float(dequantize_hp(np.asarray(g.wall_hp)[ty, tx])), 1e-9)
    y0, y1 = max(ty - 5, 0), ty + 6
    x0, x1 = max(tx - 5, 0), tx + 6
    print(f"{a.level}: tile ({tx},{ty}), {a.seconds:.0f} s"
          f"{', vents off' if a.no_vents else ''}")
    print("    t   fire  fuel   tile K  hot gas K  smoke total  densest" +
          ("  squad HP" if a.units else ""))
    n_ticks = int(a.seconds * tps)
    step = max(1, int(a.every * tps))
    dump_ticks = ({int(round(float(v) * tps)) for v in a.dump_at.split(",")}
                  if a.dump else set())
    dumps = []
    for t in range(n_ticks + 1):
        if t in dump_ticks:
            dumps.append((t, _flame_inputs(g, tps)))
        if t % step == 0:
            T = np.asarray(g.temperature)
            gas_win = np.asarray(g.material)[y0:y1, x0:x1] == MAT_AIR
            hot = T[y0:y1, x0:x1][gas_win].max() if gas_win.any() else 0
            sm = gas_fixed.dequantize_trace(np.asarray(g.gas)[SMOKE])
            fire = float(dequantize_fire(np.asarray(g.fire)[ty, tx]))
            fuel = float(dequantize_hp(np.asarray(g.wall_hp)[ty, tx])) / hp0
            line = (f"{t / tps:5.0f}  {fire:5.2f}  {fuel * 100:3.0f}%  "
                    f"{kelvin(T[ty, tx]):7.0f}  {kelvin(hot):9.0f}  "
                    f"{sm.sum():11.3f}  {sm.max():7.4f}")
            if a.units:
                line += "  " + " ".join(f"{u.current_hp:.0f}" for u in sim.units)
            print(line)
        if sim.is_paused():
            sim.set_paused(False)
        sim.step()

    sm = gas_fixed.dequantize_trace(np.asarray(g.gas)[SMOKE])
    wx = dequantize_q16(np.asarray(g.wind_x))
    xs = range(0, sm.shape[1], 3)
    print(f"\nrow {row} at the end (every 3rd tile):")
    print("  x     " + " ".join(f"{x:6d}" for x in xs))
    print("  smoke " + " ".join(f"{sm[row, x]:6.4f}" for x in xs))
    print("  wind  " + " ".join(f"{float(wx[row, x]):6.2f}" for x in xs) + "  (m/s, +east)")
    if a.dump:
        _save_dump(a.dump, a.level, tps, g, dumps)


def _flame_inputs(g, tps: int) -> dict:
    """One tick of what the flame VFX reads, dequantized at the read (the
    render convention; nothing is written back)."""
    from renderer.gas_detail import tame_wind, WIND_V_REF, WIND_V_SENS
    tamed = tame_wind(g.wind_x, g.wind_y, tile_size_m=g.tile_size_m,
                      ticks_per_second=float(tps),
                      v_ref=WIND_V_REF, v_sens=WIND_V_SENS)
    return {
        "fire": np.asarray(dequantize_fire(np.asarray(g.fire)), np.float32),
        "t_game": np.asarray(dequantize_q16(np.asarray(g.temperature)),
                             np.float32),
        "wind_tamed": np.asarray(tamed, np.float32),
    }


def _save_dump(path: str, level: str, tps: int, g, dumps: list) -> None:
    """Crop to the box of non-space tiles (the hull and what it holds) and
    save compressed. Masks are static in this level (nothing breaks)."""
    vac = np.asarray(g.is_vacuum, bool)
    ys, xs = np.nonzero(~vac)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = (slice(y0, y1), slice(x0, x1))
    out = {
        "level": np.array(level), "tps": np.array(tps),
        "tile_size_m": np.array(float(g.tile_size_m)),
        "origin_xy": np.array([x0, y0]),
        "ticks": np.array([t for t, _ in dumps]),
        "solid": np.asarray(g.solid, bool)[crop],
        "thermal_solid": np.asarray(g.thermal_solid, bool)[crop],
        "flammable": np.asarray(g.flammable, bool)[crop],
        "is_vacuum": vac[crop],
        "is_ambient": np.asarray(g.is_ambient, bool)[crop],
    }
    for key in ("fire", "t_game", "wind_tamed"):
        out[key] = np.stack([d[key][crop] for _, d in dumps])
    np.savez_compressed(path, **out)
    print(f"\nflame inputs at ticks {list(out['ticks'])} -> {path} "
          f"(crop x {x0}:{x1}, y {y0}:{y1})")


if __name__ == "__main__":
    main()
