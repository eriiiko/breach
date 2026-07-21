"""VERIFY P1 at REAL level dimensions — drive the real GameRenderer on
unhcr_vessel_2 (50x120 tiles, world RT 2400x5760) with use_3d_units=True, and
export the actual composited world RT (the same pixels the game shows), plus
crops around marines. Prior arc builds gave FALSE PASSES on tiny synthetic
scenes; this uses the real level + real LightingPass + real light sources.

A RED test light is dropped on the first marine so the "marine takes the
lamp's colour and matches the floor beside it" claim is checkable; the rest of
the roster is lit by the level's own [[light]] entries (or ambient in dark
rooms — occlusion parity).

Outputs (scratchpad/):
  verify_p1_world_3d_full.png     - full 2400x5760 world RT, 3D marines ON
  verify_p1_crop_lit.png          - crop around the RED-lit marine
  verify_p1_crop_ctx.png          - wider crop: marine(s) + surrounding ship
  verify_p1_world_sprite_full.png - same frame, 3D OFF (sprite path unchanged)
Run: conda run -n data python scratchpad/verify_p1_real.py
"""
import sys
from pathlib import Path

import numpy as np
import pyray as rl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import breach_physics as bp
from config import CFG
from level_loader import load as load_level
from level_lights import (light_source_params, monotonic_total_tick,
                          partition_lights)
from renderer import GameRenderer
from renderer.game_renderer import RenderConfig
from renderer.camera import Camera2D
from simulation import Simulation
from simulation.unit import Unit

OUT = ROOT / "scratchpad"


def build_sources(level, sim, add_red_at=None):
    sim_time_per_tick = 1.0 / float(CFG.clock.ticks_per_second)
    ticks_per_round = int(CFG.clock.ticks_per_round)

    def _mk(params):
        s = bp.LightSource()
        for k, v in params.items():
            setattr(s, k, v)
        return s

    lights_in, _off = partition_lights(level.lights, level.width, level.height)
    static = [_mk(light_source_params(e, 0, sim_time_per_tick))
              for e in lights_in if e.kind != "beacon"]
    beacons = [e for e in lights_in if e.kind == "beacon"]
    total_tick = monotonic_total_tick(sim.turn_number, ticks_per_round, sim.tick)
    src = list(static)
    src += [_mk(light_source_params(e, total_tick, sim_time_per_tick))
            for e in beacons]
    if add_red_at is not None:
        r = bp.LightSource()
        r.x = float(add_red_at[0])
        r.y = float(add_red_at[1])
        r.max_range = 22
        r.intensity = 2.6
        r.angle_spread = 6.283
        r.color = (1.0, 0.18, 0.12)   # strong red
        r.jitter = 0.0
        src.append(r)
    return src


def render_frames(renderer, sim, level, red_at, n=8):
    for _ in range(n):
        sources = build_sources(level, sim, add_red_at=red_at)
        renderer.upload_state(sim.gmap, light_sources=sources)
        renderer.begin_frame()
        renderer.compose_world(
            units_marines=sim.marines(),
            units_zombies=sim.zombies(),
            projectiles=sim.projectiles,
            current_phase=0,
            doors=sim._doors,
        )
        renderer.draw_background_to_screen()
        renderer.blit_world_to_screen()
        renderer.draw_panel(sim=sim)
        renderer.end_frame()


def export_rt(renderer, path):
    img = rl.load_image_from_texture(renderer.world.rt.texture)
    rl.image_flip_vertical(img)          # RT is y-up; flip to y-down top-left
    rl.export_image(img, str(path))
    return img                            # caller unloads


def crop(img, cx_px, cy_px, w, h, path):
    W, H = img.width, img.height
    x = int(max(0, min(W - w, cx_px - w / 2)))
    y = int(max(0, min(H - h, cy_px - h / 2)))
    c = rl.image_copy(img)
    rl.image_crop(c, rl.Rectangle(x, y, w, h))
    rl.export_image(c, str(path))
    rl.unload_image(c)


def main():
    level_name = "unhcr_vessel_2"
    level = load_level(level_name)
    print(f"level {level.name}: {level.width}x{level.height} tiles")
    sim = Simulation(level, seed=42, breach_physics=bp, enable_recorder=True)
    for s in level.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                          footprint=s.footprint))
    marines = sim.marines()
    zombies = sim.zombies()
    print(f"spawned marines={len(marines)} zombies={len(zombies)}")
    if not marines:
        print("NO MARINES in level — cannot verify; aborting")
        return

    wpt = float(getattr(CFG.rendering, "world_px_per_tile", 24.0))
    m0 = marines[0]
    red_at = (m0.x + getattr(m0, "footprint", 3) / 2.0,
              m0.y + getattr(m0, "footprint", 3) / 2.0)
    print(f"marine[0] at tile ({m0.x},{m0.y}); RED light at {red_at}; wpt={wpt}")

    # Window sized so the panel/map fit on screen; the world RT is full-size
    # (grid*wpt) regardless. Non-borderless so the run is bounded + closes.
    map_px_w, map_px_h, panel_px_w = 1000, 760, 280
    cfg = RenderConfig(
        map_px_w=map_px_w, map_px_h=map_px_h, panel_px_w=panel_px_w,
        grid_w=level.width, grid_h=level.height,
        world_px_per_tile=wpt, use_3d_units=True)
    cam = Camera2D(pos_tile_x=0.0, pos_tile_y=0.0,
                   zoom_px_per_tile=max(20.0, min(64.0, map_px_w / level.width)),
                   viewport_px_w=map_px_w, viewport_px_h=map_px_h,
                   world_size_tile_w=level.width, world_size_tile_h=level.height)
    renderer = GameRenderer(level, bp, cfg, initial_camera=cam, borderless=False)
    renderer.lighting.set_ambient((0.10, 0.10, 0.13))  # match main.py
    print(f"world RT = {renderer.world.world_px_w}x{renderer.world.world_px_h}, "
          f"unit_models.ready={renderer.unit_models.ready}, "
          f"marine_shader={renderer.unit_models._shader is not None}")

    # ---- 3D marines ON ----
    render_frames(renderer, sim, level, red_at, n=8)
    full = export_rt(renderer, OUT / "verify_p1_world_3d_full.png")
    cx = (m0.x + getattr(m0, "footprint", 3) / 2.0) * wpt
    cy = (m0.y + getattr(m0, "footprint", 3) / 2.0) * wpt
    crop(full, cx, cy, 700, 700, OUT / "verify_p1_crop_lit.png")
    crop(full, cx, cy, 1400, 1400, OUT / "verify_p1_crop_ctx.png")
    # Probe the actual field colour under the marine (data anchor for the PNG).
    gx = int(m0.x + getattr(m0, "footprint", 3) // 2)
    gy = int(m0.y + getattr(m0, "footprint", 3) // 2)
    print(f"red-lit marine foot tile ({gx},{gy}) "
          f"incoming_rgb={renderer.lighting.light_rgb[gy, gx]}")
    rl.unload_image(full)

    # ---- 3D marines OFF: sprite path must be unchanged ----
    renderer.cfg.use_3d_units = False
    render_frames(renderer, sim, level, red_at, n=2)
    spr = export_rt(renderer, OUT / "verify_p1_world_sprite_full.png")
    crop(spr, cx, cy, 700, 700, OUT / "verify_p1_crop_sprite.png")
    rl.unload_image(spr)

    renderer.shutdown()
    print("OK — wrote verify_p1_*.png to scratchpad/")


if __name__ == "__main__":
    main()
