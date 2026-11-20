---
paths:
  - "renderer/game_renderer.py"
  - "renderer/world_composite.py"
  - "renderer/core.py"
  - "renderer/lighting.py"
  - "renderer/lit3d.py"
  - "renderer/frame_lights.py"
  - "renderer/light_curve.py"
  - "renderer/blackbody.py"
  - "renderer/hover_readout.py"
  - "renderer/tile_art.py"
  - "shaders/world_resolve.fs"
  - "shaders/lighting.fs"
  - "shaders/lighting.vs"
  - "shaders/pass_quad.vs"
  - "src/level_lights.py"
  - "ui/**"
  - "tools/look_shots.py"
  - "tools/look_shots.toml"
  - "tools/lighting_demo.py"
---

# Render core: the frame, the tone-map, light, the seams, the UI

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## One tone-map
- Every world layer writes LINEAR HDR into the RGBA16F scene RT; `resolve()` is the ONLY place ACES, the gamma encode and the dither run, and `u_light_gain` (`[render.lighting] master_gain`) is the only exposure. A layer that tone-maps itself is the bug (`tests/test_one_tonemap.py`).
- Display-referred draws (UI, debug colormaps, doors, waypoints, sprites, effects, `overlay_fn`) come AFTER the resolve; lit 3D enters through the units / props pass, never `overlay_fn`.
- ONE named exception: the legacy black-body heat layer (`blackbody.pack_emissive_rgba`) stays display-space, ACES-shaped through `blackbody.aces_curve`, drawn after the resolve by THE fire shader's display stage, through the smoke and under the flame's cover; no second one. G3 F3 measured its move into the linear scene failing the hue gate (`tools/fire_look_gate.py`) and kept it; its retirement is its own patch (start from `2fe0589`). The FLAME itself is linear, in the scene.

## Look shots
- Every graphics patch renders the shot set before and after with `tools/lighting_demo.py --shots`, and the implementer looks at the PNGs before claiming the patch works. A new scene is a row (`hide` isolates a layer): never a second screenshot harness.

## Wall-destruction render seam
- A topology change reaches the picture through this ONE seam, for every cause (burst, explosion, chew, burn-through, door): never per-cause redraw code, never a per-frame grid scan. A new topology writer calls `on_tile_changed` (#9).
- A changed tile shows the pristine art of the nearest tile loaded as its current material (no rubble art yet). An EXPOSED vacuum or sky tile (`is_vacuum` / `is_ambient` AND open) is punched through the vacuum mask; a SOLID tile keeps the flag as the sealed-hull state, so every reader that distinguishes a hole from a wall reads the flag with `solid`, never alone (#150: a closed hull door drawn as space).

## LightingPass
- A CONSUMER: it uploads the light accessor's field once per sim tick and never calls a producer. New lit passes sample `light_tex_a/b`, never a second raycast. The flat term `[render.lighting] floor` is never a substitute for the sky. The ONLY light producer is the sweep.

## Light curve
- The one GLSL copy (`lit3d._LIGHT_CURVE_GLSL`), applied per fragment before albedo and N.L on the floor, the units and the props: never a per-pass curve. `enabled = false` is the plain light bit for bit. The shipped curve is Erik's hand-tuned one (2026-11-07); its values are his to tune (the `light-curve` skill; `panel = false` before a commit).
- The editor owns the mouse through `GameRenderer.pointer_captured`: a new mouse read in the game checks it.
- ACCEPTED GAP: smoke reads the plain light. (The water takes the curve and the smoke's cast shadow since #97 W1, spliced at `shaders/water.fs`'s `// <SMOKE_SHADOW>` marker.)

## Frame lights
- Every light is enumerated ONCE as a `level_lights.LightSpec` (`frame_light_specs`), then `cone_rows`, the ONLY output row. A fire is never a row (its light is the sweep's thermal emission); a `[[light]]`'s `range` is parsed but ignored. Never a second assembly.
- Transient lights (muzzle flashes, blast flashes) ride ONE queue under ONE cap, `TRANSIENT_MAX_TOTAL`, through `cap_transient_group` (the muzzle flashes dim first); gen_ship's lamp budget is derived from it. Never a sim field.
- `EMITTER_INPUTS` names each light's input source (sim / render); `sky_for_level` is the sky's door; `emit_mult_rows` is the fire flicker's float → Q16 door.

## Tile inspector
- Its sweep rows show what the sweep READS, formed by the engine's own FP_HD one-cell functions (`RadiationSweep.gas_extinction_q16`, `fleck_f_gas_q24`): never a copy of the law. Light through `light_field.light_at`. A future resident gather swaps inside this seam.

## Ping-pong helper
- Passes are DIRECT GL inside one `DirectPasses` block (far cheaper than raylib's batch); `GLDirect` is THE ctypes GL loader. The pass path never issues a `glGet*` (~100 µs stall each).
- fp32 (fp16 fallback), no depth, blending OFF in the block, texels addressed by `gl_FragCoord` with GL row = data row (no flip anywhere on the pass path), samplers from unit 5; partial passes keep outside texels equal in both buffers (`clear_rects`). The traps are in its docstring.

## No BLAS in the frame path
- Incident: G3 2026-10-29, fire_smoke_studio's HUD upload took over 1000 ms per frame (MKL's thread pool turned a 0.03 ms species product into 150 to 1000 ms whenever another process kept the cores busy). Species-axis sums go through `species_sum`; other small products are outer products or row sums; a load-time use says `blas-exempt: <why>` on its line (`tests/test_no_blas_in_frame_path.py`).

## UI split
- The side panel's game-state block is data too (`ui.model.panel_state_lines` → `PanelLine` rows; `GameRenderer.draw_panel` only draws them). A turn-based ruleset shows round / phase / AP; `ContinuousRealtime` shows the tick and the unit's magazine, never turn text.

## Blackbody
- Retuning `[render.blackbody]` or the curves means `tools/gen_light_table.py --write` (the sweep's L° is baked from this ramp; `tests/test_light_table.py` enforces).

## Blackbody GPU LUT
- Shaders sample the LUT, never re-derive Planck. It is uploaded in `renderer/fire_vfx.py` so `renderer/blackbody.py` stays pyray-free; `lut_size` is not raised for it.
