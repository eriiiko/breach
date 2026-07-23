"""Breach main entry point (Raylib renderer + Simulation facade).

Wires the new :class:`simulation.Simulation` to the existing pyray
renderer. This is the canonical entry point — ``game.py`` (the pygame
legacy) is scheduled for deletion as step 13 of the migration.

Game loop (real time + pause):

  - Sim starts paused. Player plans all orders for the round across
    Phase 1 (preparation) and Phase 2 (engagement) — Tab switches
    which phase the next order belongs to.
  - Spacebar resumes execution; sim ticks at CFG.clock.ticks_per_second
    (12 Hz by default).
  - Sim runs the full round (120 ticks) in one go without pausing
    between phases — phase 1 and phase 2 play through smoothly like
    a movie. Auto-pause fires only at end of round, returning to
    planning for the next round.
  - Backspace undoes last order; Tab switches planning phase; Esc
    clears selection; Ctrl+R reloads config; F8 dumps physics .npz.

Input is a :class:`control_source.ControlSource`, chosen at startup by the
``--control`` flag (default ``wego`` -> :class:`input_handler.WEGOPlanningInput`,
today's keyboard/mouse WEGO planning input, unchanged — see
``control_source.py`` for the seam, control_modularity design §3b). The
renderer reads ``sim.get_state()`` and ``sim.tick_events`` each frame — it
never writes back into the sim.

Run:
    C:/Users/steen/anaconda3/python.exe main.py
    C:/Users/steen/anaconda3/python.exe main.py --level playground   # sandbox
    C:/Users/steen/anaconda3/python.exe main.py --control wego       # explicit default
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure we can import the C++ physics module + project modules.
# ``src/`` hosts the new ``simulation`` package — imported as
# ``from simulation import X`` everywhere (no ``src.`` prefix). See
# src/simulation/__init__.py for the convention.
ROOT = Path(__file__).resolve().parent

# GPU launch path (--cuda). breach_physics is imported at module load below, so
# the CUDA build must be put on sys.path + its backends flipped BEFORE that
# import. tools/run_on_cuda does exactly that and then calls main() — so when
# --cuda is present and we have NOT yet been routed through the wrapper, we hand
# off to it and never fall through to the CPU import. The default launch (no
# --cuda) is byte-for-byte unchanged.
if "--cuda" in sys.argv and "breach_physics" not in sys.modules:
    sys.path.insert(0, str(ROOT / "tools"))
    import run_on_cuda
    run_on_cuda.setup_cuda_import()
    import breach_physics as _bp_cuda
    run_on_cuda.enable_all_backends(_bp_cuda)

sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pyray as rl

import breach_physics as bp
from config import CFG
from level_loader import load as load_level
from level_lights import (light_source_params, monotonic_total_tick,
                          partition_lights)
from renderer import GameRenderer
from renderer.fire_lights import FireLightSelector
from renderer.frame_lights import build_frame_light_sources, build_light_source
from renderer.game_renderer import RenderConfig
from simulation import Simulation
from simulation.unit import Unit
from control_source import create_control_source

# Windows consoles default to cp1252, which can't encode unicode (arrows etc.)
# that creep into startup help text — force utf-8 so a stray glyph can never
# crash launch (errors='replace' is a final belt-and-suspenders).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _parse_level_override():
    """Read an optional ``--level NAME`` per-launch override from argv (P5).

    The game's standing level selection stays ``[display] level`` in
    config.toml (engine/12 §4); this flag only overrides ONE launch so the
    playground / a test map can be opened without editing config:

        C:/Users/steen/anaconda3/python.exe main.py --level playground

    Returns the folder name, or None when the flag is absent.
    """
    if "--level" not in sys.argv:
        return None
    i = sys.argv.index("--level")
    try:
        name = sys.argv[i + 1]
    except IndexError:
        raise SystemExit(
            "--level requires a level folder name, e.g. --level playground")
    if name.startswith("--"):
        raise SystemExit(
            f"--level requires a level folder name, got {name!r}")
    # Arc C7: `--level` also accepts the map editor's F5 scratch-play form,
    # `_editor_scratch/<name>` (one subpath level, editor doc §8) — but
    # never an absolute path or a `..` component, which would let a launch
    # argument escape levels/ entirely (level_loader.load's `here /
    # levels_dir / level_name` join follows pathlib's own rules for both:
    # an absolute right-hand side replaces the join outright, and `..`
    # walks back out of it). A plain single-component name is unaffected.
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        raise SystemExit(
            f"--level must be a level folder name under levels/ (optionally "
            f"'_editor_scratch/<name>'), got {name!r}")
    return name


def _parse_res_factor() -> int:
    """Read an optional ``--res N`` integer grid multiplier from argv.

    N=1 (default) leaves the level untouched. N>1 asks for a denser physics
    grid — see :func:`_upscale_level`. Returns the clamped integer factor.
    """
    if "--res" not in sys.argv:
        return 1
    i = sys.argv.index("--res")
    try:
        n = int(sys.argv[i + 1])
    except (IndexError, ValueError):
        raise SystemExit("--res requires an integer factor, e.g. --res 2")
    if n < 1:
        raise SystemExit(f"--res factor must be >= 1, got {n}")
    return n


def _parse_control_flag() -> str:
    """Read an optional ``--control NAME`` launch flag from argv (P2,
    control_modularity design §3b: the ``ControlSource`` seam). Defaults to
    ``"wego"`` — today's WEGO planning input, unchanged behavior. See
    ``control_source.create_control_source`` for the set of valid names
    (only ``"wego"`` exists until P3 adds ``GamepadDirect``).
    """
    if "--control" not in sys.argv:
        return "wego"
    i = sys.argv.index("--control")
    try:
        name = sys.argv[i + 1]
    except IndexError:
        raise SystemExit("--control requires a name, e.g. --control wego")
    return name


def _upscale_level(level, factor: int):
    """Nearest-neighbour upscale the physics grid by an integer ``factor``.

    The grid resolution in Breach is the tilemap CSV shape (GameMap reads
    ``level.tilemap.shape``), with no native scale knob — so this is the
    simplest additive way for Erik to experiment with resolution: replicate
    each tile ``factor``x``factor`` (more cells, same ship), shrink
    ``tile_size_m`` by 1/factor so the PHYSICAL size is preserved, and scale
    the spawn coords + footprints by ``factor`` so units land in the same
    place. The optional height_path heightmap is per-PIXEL art (not grid-
    sized) and is left as-is; the art-align px_per_tile recomputes from the
    new grid shape inside the renderer (art_w / grid_w), so the art still
    lines up. Materials are derived from the tilemap downstream, so upscaling
    the raw tilemap is sufficient and correct.

    Mutates and returns ``level`` (a dataclass instance).
    """
    if factor <= 1:
        return level
    import numpy as np
    from dataclasses import replace
    from level_loader import SpawnEntry

    # A6 / S1 (a6 doors design §3): record the base-resolution recovery
    # BEFORE mutating — meters-first entity consumers (door span
    # quantization) must see the PRE-scale tile size and replicate their
    # derived tile sets by the accumulated integer factor; a float
    # recompute (tile_size_m * factor) is only IEEE-exact for power-of-two
    # factors, so the base is CARRIED, never recomputed. Authored
    # [[entity]] fields are NOT mutated (they stay the authored record).
    if level.tile_size_m_base is None:
        level.tile_size_m_base = float(level.tile_size_m)
    level.res_factor = int(level.res_factor) * factor

    level.tilemap = np.repeat(
        np.repeat(level.tilemap, factor, axis=0), factor, axis=1)
    level.tile_size_m = float(level.tile_size_m) / float(factor)
    level.spawns = [
        SpawnEntry(name=s.name, team=s.team,
                   x=s.x * factor, y=s.y * factor,
                   footprint=max(1, s.footprint * factor))
        for s in level.spawns
    ]
    # [[light]] entities scale like spawns (P4 — the "units land in the same
    # place" contract): positions by ``factor``, and ``range`` too (it is
    # measured in tiles, footprint-style, so the PHYSICAL reach is preserved
    # when tile_size_m shrinks by 1/factor).
    level.lights = [
        replace(l, x=l.x * factor, y=l.y * factor, range=l.range * factor)
        for l in level.lights
    ]
    # [water] initial state scales like the tilemap (P5): the loader pinned
    # depth_map.shape == tilemap.shape, so the seed MUST follow the grid or
    # GameMap's masked seed write would shape-mismatch. Replicating the
    # per-tile depth (metres of standing water) preserves the physical
    # volume exactly: Σdepth·dx² is invariant (factor² more cells, dx²
    # smaller by factor²).
    if level.water_depth_q is not None:
        level.water_depth_q = np.repeat(
            np.repeat(level.water_depth_q, factor, axis=0), factor, axis=1)
    # zones.npy paint grid scales like the tilemap (editor design §5, A8):
    # the loader pinned zones.npy.shape == tilemap.shape, so the mask must
    # follow the grid or a --res run would shape-mismatch (or silently drop
    # zones). Nearest-neighbour replication keeps every painted id covering
    # the same PHYSICAL area — same zones, factor² more member tiles; the
    # zone [[entity]] instances (zone_id bindings, rosters) are untouched.
    if level.zone_grid is not None:
        level.zone_grid = np.repeat(
            np.repeat(level.zone_grid, factor, axis=0), factor, axis=1)
    # air_init.npy override scales like the tilemap too (A9): the loader
    # pinned air_init.shape == tilemap.shape, so the override must follow
    # the grid or GameMap's masked seed would shape-mismatch. Pressure is
    # INTENSIVE (atm per tile), so nearest-neighbour replication preserves
    # the physical field exactly — each finer cell keeps its tile's
    # pressure, and the room's total N (Σ P·dx² at ambient T) is invariant
    # (factor² more cells, dx² smaller by factor²) — the same argument as
    # water's per-tile depth. `boundary` is a scalar level property and
    # needs nothing here.
    if level.air_init_q is not None:
        level.air_init_q = np.repeat(
            np.repeat(level.air_init_q, factor, axis=0), factor, axis=1)
    # Drop any explicit art-align px_per_tile so the renderer recomputes it
    # from the new (denser) grid shape — otherwise the art would stretch.
    level.art_px_per_tile = None
    level.art_align_explicit = False
    return level


def main():
    # 1. Load level + build the simulation. --level overrides config for
    # one launch (playground / test maps); default = [display] level.
    level_name = (_parse_level_override()
                  or getattr(CFG.display, "level", "playground"))
    print(f"Loading level: {level_name}")
    level = load_level(level_name)
    res_factor = _parse_res_factor()
    if res_factor > 1:
        _upscale_level(level, res_factor)
        print(f"  --res {res_factor}: grid upscaled "
              f"{res_factor}x -> {level.width}x{level.height} tiles, "
              f"tile size now {level.tile_size_m:.5f} m")
    print(f"  {level.name} — {level.width}x{level.height} tiles, "
          f"tile size {level.tile_size_m} m")

    # Control source is chosen FIRST (control-modularity P2/P3, §3b): a control
    # scheme also picks the turn structure it runs under (Ruleset) and whether
    # the sim starts paused. `--control wego` -> None (default TwoPhaseWEGO),
    # starts paused (plan first); `--control gamepad` -> ContinuousRealtime,
    # starts running. Byte-identical to the pre-P3 default under `wego`.
    control_source = create_control_source(_parse_control_flag())

    sim = Simulation(level, seed=42, breach_physics=bp,
                     enable_recorder=True,
                     ruleset=control_source.initial_ruleset())
    sim.set_paused(control_source.starts_paused())

    # The old hardcoded demo scene (pre-breach + persistent smoke/fire
    # sources) is gone: it permanently vented the ship from an interior
    # vacuum pool and dragged all smoke toward it via the sink-pull.
    # Hazards are interactive now — I ignite, J gas, U pour water,
    # explosives breach — and will be level-defined later.

    # ----- Spawn units from level.toml [[spawn]] entries. Zero spawns is
    # legal: a unit-free physics-tuning sandbox (camera starts at 0,0;
    # marines()/zombies() handle empty rosters).
    if not level.spawns:
        print(f"  NOTE: level '{level.name}' has no [[spawn]] entries — "
              f"running as a unit-free physics sandbox")
    for s in level.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                          footprint=s.footprint))
    if level.spawns:
        print(f"  Spawned {len(level.spawns)} units from level.toml")

    # 2. Render config — borderless windowed at monitor resolution.
    from renderer.camera import Camera2D

    BORDERLESS = "--windowed" not in sys.argv
    if BORDERLESS:
        # Borderless windowed mode uses the actual monitor size, not the
        # numbers we pass to init_window. Open the window first so we can
        # query the real dimensions and lay the panel/map out to fit.
        from renderer import core as rcore
        rcore.init_window(0, 0, title=f"Breach — {level.name}",
                          borderless=True)
        screen_w, screen_h = rcore.get_monitor_size()
    else:
        screen_w, screen_h = 1280, 720

    panel_px_w = 280
    map_px_w   = screen_w - panel_px_w
    map_px_h   = screen_h
    cfg = RenderConfig(
        map_px_w=map_px_w, map_px_h=map_px_h,
        panel_px_w=panel_px_w,
        grid_w=level.width, grid_h=level.height,
        world_px_per_tile=float(getattr(CFG.rendering, "world_px_per_tile",
                                        24.0)),
    )
    print(f"  Window: {screen_w}x{screen_h} "
          f"(borderless={BORDERLESS}, world RT "
          f"{int(level.width*cfg.world_px_per_tile)}x"
          f"{int(level.height*cfg.world_px_per_tile)})")
    print(f"  WASD/arrows pan | Q/E or wheel zoom | "
          f"Space resume | Tab phase | Bksp undo | Ctrl+R reload")
    print(f"  DEBUG: T toggles temperature overlay (black-body heat ramp) | "
          f"I ignites the tile under the cursor")
    print(f"  DEBUG: J spawns the selected gas under the cursor | "
          f"K cycles the gas (white->black->poison->teargas->fuel)")
    print(f"  DEBUG: U pours water (0.2 m) under the cursor | "
          f"V toggles water overlay | P / Shift+P tilts the ship +/-2 deg")
    print(f"  DEBUG: O toggles the door under the cursor (A6 doors v0 — "
          f"dev-only latch)")
    print(f"  DEBUG: N cycles the selected unit's weapon through the armory "
          f"(W6 — the tuning key)")

    fit_w_zoom = map_px_w / max(level.width, 1)
    initial_zoom = max(20.0, min(64.0, fit_w_zoom))
    initial_camera = Camera2D(
        pos_tile_x=0.0, pos_tile_y=0.0,
        zoom_px_per_tile=initial_zoom,
        viewport_px_w=map_px_w, viewport_px_h=map_px_h,
        world_size_tile_w=level.width, world_size_tile_h=level.height,
    )
    renderer = GameRenderer(level, bp, cfg,
                            initial_camera=initial_camera,
                            borderless=BORDERLESS)
    renderer.lighting.set_ambient((0.10, 0.10, 0.13))

    # Level lights (P4): the [[light]] entities from level.toml (the old
    # hardcoded emergency lamps now live in the vessel/playground tomls).
    # Static sources are compiled structs built ONCE here; beacons are
    # rebuilt per frame from the SIM tick (they freeze with the sim —
    # src/level_lights.py owns the math; this stays a thin setattr loop).
    sim_time_per_tick = 1.0 / float(CFG.clock.ticks_per_second)
    ticks_per_round = int(CFG.clock.ticks_per_round)

    # The LightSource setattr builder + the per-frame statics/beacons/fire
    # assembly are extracted into renderer/frame_lights.py so main.py and
    # tools/lighting_demo.py build the SAME list with no drift (B2 P1).
    lights_in, lights_off = partition_lights(level.lights,
                                             level.width, level.height)
    if lights_off:
        # ONE warning at load (never per frame): e.g. vessel-authored lamp
        # coordinates on a smaller level.
        skipped = ", ".join(f"({l.x:g}, {l.y:g})" for l in lights_off)
        print(f"  WARNING: {len(lights_off)} [[light]] entries off-grid "
              f"for {level.width}x{level.height} — skipped: {skipped}")
    static_lights = [
        build_light_source(bp, light_source_params(e, 0, sim_time_per_tick))
        for e in lights_in if e.kind != "beacon"
    ]
    beacon_lights = [e for e in lights_in if e.kind == "beacon"]
    if level.lights:
        print(f"  Lights: {len(static_lights)} static + "
              f"{len(beacon_lights)} beacon from level.toml")

    # Fire light sources (Fire & Heat Beauty B1): the brightest-K hot tiles
    # become omni ray-traced lights each frame, colour + intensity from the
    # renderer's shared black-body ramp. RENDER-ONLY — they never write the
    # synced heat channel (see renderer/fire_lights.py). Built once from
    # [render.fire_lights]; queried per frame in the sources block below.
    fire_light_selector = FireLightSelector.from_config(CFG)

    # Entity registry (entity design §3b): apply the dev tuning overlay
    # (hard-errors on schema-in-TOML mistakes, like a bad config.toml), then
    # rewrite the editor's last-good fallback — a successful launch is the
    # freshness guarantee. Only the file write is soft-failed: a locked file
    # must not kill a play session.
    from simulation.entities import apply_tuning_overlay, export_registry_json
    apply_tuning_overlay()
    try:
        export_registry_json()
    except OSError as exc:
        print(f"  WARNING: entity_registry.json export failed: {exc}")

    # 3. Main loop.
    last_time = time.perf_counter()
    tick_accum = 0.0

    try:
        while not renderer.should_close():
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            # ----- Input first (may toggle pause / queue orders) -----
            renderer.poll_toggles()
            renderer.update_camera(dt)
            control_source.handle_frame(sim, renderer)

            # ----- Tick the simulation while not paused -----
            if not sim.is_paused():
                tick_accum += dt
                # Cap the per-frame catch-up to avoid spirals if a
                # background pause stalls the loop.
                max_catch_up = 5
                steps = 0
                while tick_accum >= sim_time_per_tick and steps < max_catch_up:
                    sim.step()
                    tick_accum -= sim_time_per_tick
                    steps += 1
                    # Sim may auto-pause mid-batch (phase boundary).
                    if sim.is_paused():
                        break

            # ----- Lights: level statics + beacons + fire (shared assembly) --
            # The statics/beacons/fire assembly lives in renderer/frame_lights.py
            # so main.py and tools/lighting_demo.py build the SAME per-frame list
            # with no drift (B2 P1). NOT an entity system — it only assembles
            # LightSource params from today's suppliers (level [[light]] rows
            # incl. the beacon, B1 fire tiles). total_tick = the MONOTONIC sim
            # tick on the SIM clock: beacons freeze on pause + replay exactly
            # (P4 §2.2), never the wall-clock frame dt. The mouse flashlight + W6
            # transient emitters stay caller-side below (they differ per caller).
            total_tick = monotonic_total_tick(
                sim.turn_number, ticks_per_round, sim.tick)
            frame = build_frame_light_sources(
                bp, static_lights, beacon_lights,
                total_tick=total_tick, sim_time_per_tick=sim_time_per_tick,
                fire_selector=fire_light_selector,
                temperature_field=sim.gmap.temperature,
                blackbody_ramp=renderer.blackbody_ramp,
                show_fire_lights=renderer.show_fire_lights)
            sources = frame.sources
            renderer.set_fire_light_stats(frame.fire_count, frame.fire_peaks,
                                          fire_light_selector.max_lights)
            mouse_f = renderer.mouse_to_tile_float()
            if mouse_f is not None:
                src = bp.LightSource()
                src.x = float(mouse_f[0])
                src.y = float(mouse_f[1])
                src.max_range = 25
                src.intensity = 2.5
                src.angle_spread = 6.283
                # Flashlight — cool white (profile: flashlight).
                src.color = (1.0, 1.0, 0.95)
                src.jitter = 0.0
                sources.append(src)

            # W6 transient emitters: flame/miasma jets + plasma bolts. The
            # renderer derives light specs from its own live effect queue
            # (SprayJetEvent / ProjectileGlowEvent visuals) — render-side
            # only, one frame behind the sim tick, never written back.
            for spec in renderer.transient_light_specs():
                src = bp.LightSource()
                src.x = float(spec["x"])
                src.y = float(spec["y"])
                src.max_range = int(spec["max_range"])
                src.intensity = float(spec["intensity"])
                src.angle_spread = 6.283
                src.color = spec["color"]
                src.jitter = 0.0
                sources.append(src)

            # ----- Upload + draw -----
            # total_tick (the MONOTONIC sim clock computed above) is the P3 smoke
            # detail clock too — the advected-noise crossfade rides the sim tick,
            # not wall time, so replays/spectators render identical smoke.
            renderer.upload_state(sim.gmap, light_sources=sources,
                                  sim_tick=total_tick)
            renderer.begin_frame()

            renderer.compose_world(
                units_marines=sim.marines(),
                units_zombies=sim.zombies(),
                projectiles=sim.projectiles,
                orders_phase1=sim.orders_for_phase(0),
                orders_phase2=sim.orders_for_phase(1),
                current_phase=control_source.planning_phase,
                doors=sim._doors,   # A6 dev door draw (render-read only)
            )
            renderer.draw_background_to_screen()
            renderer.blit_world_to_screen()
            renderer.draw_debug_hud(sim.gmap)

            # Pull tick events into the renderer's effect queue, then age
            # the queue. tick_events is cleared by the next sim.step(),
            # so we MUST consume here every frame.
            renderer.consume_events(sim.tick_events)
            renderer._advance_effects(dt)

            selected = sim.get_unit(control_source.selected_unit_id) \
                if control_source.selected_unit_id is not None else None
            renderer.draw_panel(
                sim=sim,
                selected_unit=selected,
                planning_phase=control_source.planning_phase,
                current_mode=control_source.current_mode,
            )
            renderer.end_frame()
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
