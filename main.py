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

import numpy as np
import pyray as rl

import breach_physics as bp
from config import CFG
from level_loader import load as load_level
from level_lights import monotonic_total_tick, partition_lights
from renderer import GameRenderer
from renderer import frame_lights
from renderer.game_renderer import RenderConfig
from renderer.static_props import placements_from_entities
from simulation import Simulation
from simulation.unit import Unit
from control_source import create_control_source
# The OnePhaseWEGO interface (onephase_wego design §16). Imported
# unconditionally — ui.model is pure/headless and ui.draw only needs the pyray
# already loaded above — and used only when that ruleset is running.
import ui
from ui import draw as ui_draw

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


def _parse_debug_flag() -> bool:
    """``--debug`` — re-arm the diagnostic keys (onephase_wego design §17 +
    Erik's kickoff ruling 2).

    §17 rules that in game mode the control scheme owns EVERY binding, so
    ``--control onephase`` ships with no diagnostic keys at all. This flag
    hands the whole current dev set back on the keys it already uses
    (I/J/K/U/N/O/P sim-side, F1-F10 + T/V/M/L/B/H/G renderer-side). The other
    control schemes are unaffected — they keep their keys either way.
    """
    return "--debug" in sys.argv


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


def _onephase_world_overlay(sim, control_source, hover_tile=None):
    """The OnePhaseWEGO world-space overlay, as a callback for
    ``compose_world`` (onephase_wego design §16).

    Drawn INSIDE the world render target so the camera transforms it like the
    ship: the teal plan viz for the selected marine, every marine's overwatch
    cone, the flashlight cones, and the team's marks. All of it reads sim
    state and writes nothing.
    """
    def _draw(world_px_per_tile):
        selected = (sim.get_unit(control_source.selected_unit_id)
                    if control_source.selected_unit_id is not None else None)
        # NOTE: the flashlight cones are NOT drawn as sectors here. They are
        # already real light (cone emitters on the radiation sweep), and
        # drawing a translucent wedge on top of the light they cast doubled the
        # cone and read as a UI overlay rather than as a torch (Erik, 2nd play
        # session: "the half cones … flashlights already kind of show this, and
        # it should be enough"). The lighting IS the facing indicator.
        #
        # Cover is intangible (a shape, not a tile), so the tileset never draws
        # it — without this the player is protected by something invisible.
        ui_draw.draw_cover(sim, world_px_per_tile)
        # An overwatch cone is not a "view" indicator, it is the player's own
        # target-control dial (§9), so it stays.
        for u in sim.marines():
            ui_draw.draw_overwatch_cone(u, world_px_per_tile)
        ui_draw.draw_marks(sim, 0, world_px_per_tile)
        ui_draw.draw_selected_marker(selected, world_px_per_tile)

        # EVERY marine's plan stays on screen (Erik: "i'd like everything to
        # stay on screen"), the unselected ones muted. Drawn first so the
        # selected marine's plan lands on top at full strength — you can read
        # the whole assault at once without losing which one you are steering.
        for u in sim.marines():
            if selected is not None and u.id == selected.id:
                continue
            ui_draw.draw_plan_overlay(ui.plan_overlay(sim, u),
                                      world_px_per_tile, dimmed=True)
        if selected is not None:
            ui_draw.draw_plan_overlay(
                ui.plan_overlay(sim, selected, hover_tile=hover_tile,
                                armed_action=control_source.armed_action),
                world_px_per_tile)
    return _draw


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
    debug_mode = _parse_debug_flag()
    control_source = create_control_source(_parse_control_flag(),
                                           debug=debug_mode)

    sim = Simulation(level, seed=42, breach_physics=bp,
                     enable_recorder=True,
                     ruleset=control_source.initial_ruleset())
    sim.set_paused(control_source.starts_paused())
    # Is the turn-formula redesign driving? The renderer/HUD branches below key
    # off the RULESET rather than the flag name, so the two halves of the
    # loadable game stay chosen together (control_modularity §3a/§3b).
    onephase = bool(getattr(sim.ruleset, "drives_units", False))

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
    if onephase:
        # §17: game mode owns every binding, so the help text IS the keymap.
        print(f"  OnePhaseWEGO — {sim.ticks_per_round} ticks/round "
              f"({CFG.clock.round_duration_seconds:g} s), one phase")
        print(f"  MOUSE: LMB select marine (or apply armed slot) | "
              f"RMB MOVE here | Shift+RMB queue waypoint | wheel zoom")
        print(f"  HOTBAR 1..0: move, shoot, move&shoot, overwatch, ambush, "
              f"hold, grenade, charge, +belt items")
        print(f"  KEYS: Q swap weapon | X mark | Space SUBMIT (run the round) "
              f"| Bksp undo | Esc cancel | Tab cycle marine")
        print(f"        I inventory (DS3 menu) | L flashlight variant | "
              f"WASD/arrows pan")
        if debug_mode:
            print(f"  --debug ON: the diagnostic keys are re-armed "
                  f"(F1-F10, T/V/M/B/H/G, I/J/K/U/N/O/P, Ctrl+R, F8)")
        else:
            print(f"  (no diagnostic keys in game mode — relaunch with "
                  f"--debug to re-arm them)")
    else:
        print(f"  WASD/arrows pan | Q/E or wheel zoom | "
              f"Space resume | Tab phase | Bksp undo | Ctrl+R reload")
        print(f"  DEBUG: T toggles temperature overlay (black-body heat ramp) "
              f"| I ignites the tile under the cursor")
        print(f"  DEBUG: J spawns the selected gas under the cursor | "
              f"K cycles the gas (white->black->poison->teargas->fuel)")
        print(f"  DEBUG: U pours water (0.2 m) under the cursor | "
              f"V toggles water overlay | P / Shift+P tilts the ship +/-2 deg")
        print(f"  DEBUG: O toggles the door under the cursor (A6 doors v0 — "
              f"dev-only latch)")
        print(f"  DEBUG: N cycles the selected unit's weapon through the "
              f"armory (W6 — the tuning key)")

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
    # Level lights (P4): the [[light]] entities from level.toml (the old
    # hardcoded emergency lamps now live in the vessel/playground tomls).
    # Beacons turn with the SIM tick (they freeze with the sim —
    # src/level_lights.py owns the math).
    sim_time_per_tick = 1.0 / float(CFG.clock.ticks_per_second)
    ticks_per_round = int(CFG.clock.ticks_per_round)

    # ray-engine-v2 P6b: THE ONE ASSEMBLY (renderer/frame_lights.py). Every
    # light is enumerated once as a spec, and the engine gets their
    # CONE-EMITTER rows once per sim tick (the sweep lights the world). The
    # flat term is the small [render.lighting] floor; the ambient light is the
    # SKY below.
    lights_in, lights_off = partition_lights(level.lights,
                                             level.width, level.height)
    if lights_off:
        # ONE warning at load (never per frame): e.g. vessel-authored lamp
        # coordinates on a smaller level.
        skipped = ", ".join(f"({l.x:g}, {l.y:g})" for l in lights_off)
        print(f"  WARNING: {len(lights_off)} [[light]] entries off-grid "
              f"for {level.width}x{level.height} — skipped: {skipped}")
    if level.lights:
        n_beacon = sum(1 for e in lights_in if e.kind == "beacon")
        print(f"  Lights: {len(lights_in) - n_beacon} static + "
              f"{n_beacon} beacon from level.toml")
    # P6b: THE SKY -- the light ring's per-ordinate RGB from the level's boundary
    # type ([light.sky.<boundary>]): space is dark, a planetside ("ambient")
    # boundary is open sky. None = the dark ring.
    light_sky = frame_lights.sky_for_level(getattr(level, "boundary", "space"), CFG, bp)
    light_fine_bits = int(bp.L_FINE_BITS)
    print(f"  Light: sweep; sky "
          f"{'dark' if light_sky is None else 'lit'} "
          f"(boundary '{getattr(level, 'boundary', 'space')}')")

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

    # ray-engine-v2 P6b: the per-tick light assembly -- every light as a spec,
    # once (frame_lights.frame_light_specs). The render-sourced lights (the
    # onephase flashlights at their lens, WEGO's cursor lamp, the W6 transient
    # emitters) are the caller's tail, exactly as before P6b.
    def _footprint_of(uid):
        u = sim.get_unit(uid)
        return int(getattr(u, "footprint", 3)) if u is not None else 3

    def assemble_lights():
        total = monotonic_total_tick(sim.turn_number, ticks_per_round, sim.tick)
        extra = []
        if onephase:
            # §8 removes the CURSOR flashlight outright: marines carry them, and
            # the light is the expression of their facing cone.
            extra += frame_lights.flashlight_specs(
                ui.flashlight_cones(
                    sim, team=0, mode=control_source.flashlight_mode,
                    selected_unit_id=control_source.selected_unit_id,
                    cursor_tile=renderer.mouse_to_tile(),
                    paused=sim.is_paused()),
                _footprint_of)
        else:
            cursor = frame_lights.cursor_lamp_spec(renderer.mouse_to_tile_float())
            if cursor is not None:
                extra.append(cursor)
        # W6 transient emitters: flame/miasma jets + plasma bolts, from the
        # renderer's own live effect queue — render-side only.
        extra += frame_lights.transient_specs(renderer.transient_light_specs())
        return frame_lights.frame_light_specs(
            lights_in, total_tick=total, sim_time_per_tick=sim_time_per_tick,
            extra=extra)

    def cone_rows_now():
        grid_h, grid_w = sim.gmap.solid.shape
        return frame_lights.cone_rows(assemble_lights().specs, grid_w, grid_h,
                                      light_fine_bits)

    # 3. Main loop.
    last_time = time.perf_counter()
    tick_accum = 0.0
    light_on = None          # P6b: is the sweep's light currently requested?
    lit_rows = None          # the cone rows the current light field was computed from
    last_relight = 0.0

    try:
        while not renderer.should_close():
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            # ----- Input first (may toggle pause / queue orders) -----
            # §17: the renderer's diagnostic keys are game-mode keys too. A
            # scheme that owns every binding (OnePhaseWEGO) only lets them
            # through under --debug; zoom is a view control, not a diagnostic,
            # so it stays live either way — with Q/E released, since Q is the
            # weapon swap there.
            if control_source.wants_renderer_toggles:
                renderer.poll_toggles()
            else:
                renderer.poll_camera_zoom(allow_qe=False)
            renderer.update_camera(dt)
            control_source.handle_frame(sim, renderer)

            # ----- ray-engine-v2 P6b: the light -----
            # The radiation sweep lights the world: the game REQUESTS light
            # while the lighting pass shows (F4), and the cone rows ride each
            # sim tick.
            new_light = bool(renderer.show_lighting)
            if new_light != light_on:
                if not new_light:
                    sim.set_light(False)
                light_on, lit_rows = new_light, None

            # ----- Tick the simulation while not paused -----
            stepped = False
            if not sim.is_paused():
                tick_accum += dt
                # Cap the per-frame catch-up to avoid spirals if a
                # background pause stalls the loop.
                max_catch_up = 5
                steps = 0
                while tick_accum >= sim_time_per_tick and steps < max_catch_up:
                    if new_light:
                        # the ONE assembly, once per sim tick, for the engine
                        lit_rows = cone_rows_now()
                        sim.set_light(True, lit_rows, light_sky)
                    sim.step()
                    stepped = True
                    tick_accum -= sim_time_per_tick
                    steps += 1
                    # Sim may auto-pause mid-batch (phase boundary).
                    if sim.is_paused():
                        break
            # PAUSED (planning): nothing ticks, so the light is RELIT on the
            # current state whenever its emitters moved (the cursor lamp, a
            # planning flashlight aimed at the cursor) -- and at 4 Hz anyway, so
            # a dev-key edit made while paused shows up. The relight writes only
            # the three light planes (PhysicsRunner.relight).
            if new_light and not stepped and sim.is_paused():
                rows = cone_rows_now()
                t_now = time.perf_counter()
                if (lit_rows is None or not np.array_equal(rows, lit_rows)
                        or t_now - last_relight > 0.25):
                    sim.set_light(True, rows, light_sky)
                    sim.relight()
                    lit_rows, last_relight = rows, t_now

            # total_tick = the MONOTONIC sim tick on the SIM clock: beacons
            # freeze on pause + replay exactly (P4 §2.2), and it is the P3 smoke
            # detail clock too (replays render identical smoke).
            total_tick = monotonic_total_tick(
                sim.turn_number, ticks_per_round, sim.tick)

            # ----- Upload + draw -----
            renderer.upload_state(sim.gmap, sim_tick=total_tick,
                                  light_serial=sim.light_serial if new_light else None)
            renderer.begin_frame()

            # Fog of war (§8) is visibility GATING: an enemy the team cannot
            # see is simply not drawn. The sim keeps simulating it in full.
            zombies = (ui.drawable_enemies(sim, 0) if onephase
                       else sim.zombies())
            renderer.compose_world(
                units_marines=sim.marines(),
                units_zombies=zombies,
                projectiles=sim.projectiles,
                # The teal planning viz replaces the phase waypoint lines
                # entirely under OnePhaseWEGO — there are no phases (§2).
                orders_phase1=None if onephase else sim.orders_for_phase(0),
                orders_phase2=None if onephase else sim.orders_for_phase(1),
                current_phase=control_source.planning_phase,
                doors=sim._doors,   # A6 dev door draw (render-read only)
                # Props & vegetation arc #60 P3: loaded prop entities flow
                # loader -> sim -> renderer as placement records; zero cost
                # when the level has none (design §4.3).
                props=placements_from_entities(sim.entities,
                                               renderer.cfg.world_px_per_tile),
                overlay_fn=(_onephase_world_overlay(
                    sim, control_source, renderer.mouse_to_tile())
                    if onephase else None),
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
            # Screen-space HUD (§16): hotbar, round banner, planning clock,
            # and the DS3 menu when it is open. After the panel so the menu
            # overlays everything, exactly as it does in DS.
            if onephase:
                ui_draw.draw_round_banner(sim, screen_w)
                ui_draw.draw_hotbar(
                    ui.hotbar(sim, selected, control_source.bindings),
                    screen_w, screen_h)
                ui_draw.draw_planning_clock(
                    ui.planning_clock(sim, control_source.planning_elapsed),
                    screen_w)
                armed = control_source.armed_action
                if armed:
                    dial = control_source.armed_dial_seconds(sim)
                    ui_draw.draw_armed_readout(
                        sim.actions_table.get(armed).label, dial,
                        screen_w, screen_h)
                if control_source.menu_open:
                    ui_draw.draw_ds3_menu(
                        ui.ds3_menu(sim, selected, control_source.menu_page),
                        screen_w, screen_h)
            renderer.end_frame()
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
