# Water look: design for the build (#97)

2026-11-19. **Status: PROPOSED by the lead (graphics planning is Claude's,
`docs/presentation_graphics_plan_2026-11-15.md`); the look is Erik's to judge
by eye.** Not yet critiqued. Written so that a session on another PC can run
the build from this document alone.

This is the build design for track 2 of the presentation graphics plan. That
plan's section 4 holds the research and the reasons; this document does not
repeat them. It adds what a builder needs: the facts of today's code, a first
patch the plan did not have (a level to judge water on), each patch's scope,
dials, shots and gates, and the Systems section.

## 1. Facts of today's code (read 2026-11-19, main `0e646cc4`)

- **The pass:** `renderer/water.py::WaterPass` + `shaders/water.fs` (430 lines). One fragment pass, drawn inside `compose_world()` after the floor and before the floor smoke layer (keep that order, G2).
- **Its input:** one RGBA32F grid texture per tile, filled in `WaterPass.update`: R = `ripple` (m), G = `ripple_v` (m/s), B = `water_depth` (m), A = reserved for foam, zero today.
- **What the simulation offers and the pass ignores:** `gmap.flow_vx`, `gmap.flow_vy` (int32, the water's fixed-point currency, converted only through `src/simulation/water_fixed.py::dequantize_f32`). They are synced state. `ripple` and `ripple_v` are float32 and visual only, never synced.
- **The fine detail today:** `ambientSine` in the shader, three fixed sine waves scaled by `wave_scale` and `ambient_amp` in `[graphics.water]`. This is what reads "far too big" (#97).
- **Foam today:** a white mix where the ripple gradient or the depth gradient passes `foam_threshold`. No memory, no texture.
- **Light:** the pass samples the light textures itself and so does not receive the smoke's cast shadow (an accepted gap written in `.claude/rules/render-core.md`).
- **How water gets into a level:** `gmap.water_sources` and the debug pour (`src/debug_keys.py::debug_pour_water`). There is **no water studio level**: the only water shot is `steam_over_pool` on `fire_studio`.
- **Helpers that exist and must be used:** `renderer/core.py::PingPong` (the one double-buffered target helper), `renderer/particles.py` (the one particle system), `renderer/lit3d.py` (the shared light-field GLSL and the smoke shadow block), `renderer/noise_bake.py` (the precedent for a baked repeating texture), `tools/lighting_demo.py --shots` with `tools/look_shots.toml`.

## 2. The patches

All render-only. No file under `src/simulation/` or `cpp/` changes, the water
solver is untouched, no golden moves. Every patch renders the named shot set
before and after, and the builder looks at the pictures before reporting.

| Patch | What | Depends on |
|---|---|---|
| W0 | A water studio level and three look shots | none |
| W1 | A fine surface that flows | W0 |
| W2 | Foam with a life | W1 |
| W3 | Things in the water: wakes, splashes, spray | W2 |
| W4 | Sharp caustics from the fine surface | W1 |
| W5 | Wet floor | the marks layer (#180), so it waits for track 1 |

W1 and W2 are most of the gain. W4 may run before W3. W5 is last and optional.

### W0. The water studio

- **Why first:** every later patch is judged by eye, and there is nothing to look at. A level made for it also keeps the pinned levels untouched.
- **What:** `levels/water_studio`, written through `level_lib` by a small generator in `tools/` (the precedent is `tools/gen_explosion_studio.py`; never a hand-written `level.toml`). Tile size 0.333 m. Three areas:
  - a still pool about 12 x 8 tiles, 0.3 m deep, with one lamp that can be switched off;
  - a corridor about 3 x 25 tiles with a water source at one end and a drain or a lower basin at the other, so water runs along it for the whole shot;
  - a shallow wading lane, 0.15 m, beside a fuel pile for firelight.
- **Shots** added to `tools/look_shots.toml`: `still_pool_flashlight`, `flooding_corridor`, `wade_by_firelight`. The existing `steam_over_pool` stays.
- **To check before building:** how a level declares a standing depth at load and a continuous source (`water_sources`); if a level cannot hold either today, that is a finding to report, not something to add to the simulation in this track.
- **Gate:** the level loads, the airtight lint passes, the three shots render; a property test that the corridor's water is still moving at the shot's frame (flow speed above a floor), since a shot of still water proves nothing about W1.

### W1. A fine surface that flows

- **Upload:** a second grid texture (RG32F) with the flow velocity in metres per second, filled in `WaterPass.update` from `flow_vx` / `flow_vy` through `water_fixed.dequantize_f32` (a fresh float copy at the render read, never written back, never an inline divide).
- **The ripple texture:** a small repeating two-channel slope texture of fine waves, baked once at load. Put the bake in `renderer/noise_bake.py` if its tiling noise fits; a new noise shape is a function there, never an inline bake.
- **The flow method** (Vlachos 2010, cited in the shader header and archived under `docs/papers/`): two copies of the texture slid along the local flow, half a cycle apart, cross-faded. The flow is read with bilinear filtering and the phase is offset by a low-frequency noise, so the reset of a cycle never shows as a pulse.
- **Strength:** grows with flow speed and with the local agitation (`|ripple_v|`). Still water is nearly flat with a slow drift.
- **Replaces** `ambientSine`. `wave_scale` and `ambient_amp` are removed from `[graphics.water]`, the shader and the pass in the same patch (no dead dials).
- **Dials** (`[graphics.water]`): `detail_wavelength_m` (the answer to #97: wave size in metres, independent of the tile), `detail_strength_still`, `detail_strength_flow`, `flow_cycle_s`, `flow_speed_ref` (the speed at which the strength saturates).
- **Also in W1:** the pass reads the smoke's cast shadow through the lit3d block, and the accepted-gap line in `.claude/rules/render-core.md` loses its water half.
- **Shots:** `still_pool_flashlight` (one clean reflection, not a pattern), `flooding_corridor` (the surface visibly streams, no tile grid, no repetition).
- **Properties to test:** zero flow gives a time-steady normal apart from the drift term; reversing the flow reverses the slide; the result does not change when the grid is offset by a whole tile (no tile-locked pattern); a dry map costs nothing (the existing early-out is kept).

### W2. Foam with a life

- **State:** a foam amount in its own image through `PingPong`, four cells per tile to start (the one number the plan says can only be chosen on screen; a dial `foam_cells_per_tile`).
- **Each frame, one update pass:** carry the amount along the flow (read the previous image upstream by flow x frame time), add the sources, subtract the decay.
- **Sources,** each with its own gain dial: steep fine surface; fast water meeting a wall or running shallow (flow speed against depth); the advancing edge of a flood (depth appearing where there was none); where water pours in (a source tile); a strong ripple impulse (a blast, a body, from `|ripple_v|`).
- **The draw:** the amount is a threshold on a bubble texture at two sizes, both sliding with the flow (Parberry's point: foam breaks into lace, it does not fade). Lit by the room's light, a soft shadow on the floor beneath, thick foam hides the floor. The old gradient foam and its two dials (`foam_threshold`, `foam_intensity`) are removed.
- **Time:** the update runs on render frames with the frame's real time step, clamped. It is picture state only: never synced, never read by the simulation, reset when a level loads. A paused game holds it still.
- **Dials:** `foam_decay_s`, the five source gains, `foam_bubble_m` (two sizes), `foam_cover` (how much amount closes the sheet), `foam_shadow`.
- **Shots:** `flooding_corridor` (foam at the front and along the walls, travelling), `steam_over_pool` (unchanged where the water is calm).
- **Properties:** with no source the total amount only falls; with zero flow and zero decay the image is steady; the amount moves downstream at the flow's speed within a tolerance; the pass allocates nothing per frame; no BLAS contraction in the per-frame Python.

### W3. Things in the water

- **What:** a unit wading leaves a wake (a foam source and a ripple impulse along its path, from the unit's position and speed already uploaded for the 3D units); a round hitting water throws a small splash; a blast throws a ring of spray that falls back as foam.
- **Through:** `renderer/particles.py` (new kinds in its tables, never a second particle system) and W2's foam sources.
- **Reads only** events and state the renderer is already handed (the blast log's render seam, the shock front's inputs, unit rows). If a splash needs an event the renderer is not given, report it; do not add a render hook to the simulation.
- **Shot:** `wade_by_firelight`.

### W4. Sharp caustics

- **What:** the caustic term is recomputed from W1's fine surface with the area method (Wallace 2011, cited): how much the refracted light is squeezed together on the floor, scaled by depth and by the light that is there.
- **Replaces** the Laplacian caustic and the procedural detail layer (`caustic_scale` goes; `caustic_strength` stays).
- **Shot:** `still_pool_flashlight` with a thrown disturbance: a moving net of light under the lamp, none in the dark.
- **Property:** a flat surface gives exactly no caustic; total caustic light over a region is bounded by the incident light (it redistributes, it does not create).

### W5. Wet floor

Waits for the marks layer of track 1 (#180). Where water has been, the floor
stays dark and glossy and dries from the edges. Designed with the marks.

## 3. Rules every patch keeps

- Linear HDR into the world targets; `WorldComposite.resolve` is the only tone-map.
- Light comes from the light textures and the lit3d block; never a second raycast.
- Dequantize at the render read through the field's own `*_fixed.dequantize_f32`.
- No model or picture state on a simulation object; nothing written back.
- No BLAS contraction in per-frame Python (`renderer/gas_medium.py::species_sum` is the precedent).
- New switches and numbers are `config.toml` settings, never keys.
- A published technique carries its author and paper in the file header, and the paper is archived under `docs/papers/`.
- Tests assert properties, never snapshots; GL checks run in a subprocess.
- The frame cost is measured before and after with `tools/lighting_demo.py --perf` on the water studio and on `ship_demo` (dry), and reported in milliseconds. A dry ship must cost the same as today.

## 4. Gates and merging

- Per patch: the affected tests (`tools/affected_tests.py --run`), the render test files, `tests/test_claude_rules_paths.py`, the shot set before and after, the frame cost.
- The full suite stays the gate before a merge by the project rule. These patches touch no simulation or device file, so no CUDA gate applies.
- A green patch merges; Erik looks at the shots when he wants to (iron rule, 2026-11-13).

## 5. For Erik, by eye

Nothing blocks the start. These are the calls the pictures will ask for:

- How flat still water should be (W1's `detail_strength_still`).
- How much foam is right and how long it lives (W2).
- Whether a wading marine's wake should be visible from across a room (W3).
- The water colour and depth tint were tuned before the light curve of 2026-11-07; they may want another look once the surface is new.

## 6. Running it from another PC

- One implementation agent at a time, each patch in its own worktree off `main`, per `git-workflow` and `autonomous-patch-workflow`.
- The brief for each patch is its subsection above plus section 3 and section 4. W0 first.
- This design has had no critique pass yet. One critic (Opus) on sections 2 and 3 before W1 starts is the usual step; W0 does not need to wait for it.

## 7. Systems

**Existing systems this design must use:** GameRenderer + WorldComposite;
One tone-map; Look shots; LightingPass and the Lit-3D seam; Smoke cast
product; Ping-pong helper; Particle primitive; Billow noise bake (if its
tiling noise fits the ripple texture); Dequantize convention; No BLAS in the
frame path; Level data layer (`level_lib`) and the generator precedent; Config.

**New systems it creates,** each with its draft one-line rule (written into
the project instructions only when the code exists):

| System | Patch | Draft rule |
|---|---|---|
| Water studio | W0 | THE level for judging water, written by its generator through `level_lib`; never used by a test that pins play |
| Water flow detail | W1 | THE fine water surface: one ripple texture carried by the simulated flow; a new surface detail is a term here, never a second wave source |
| Water foam | W2 | THE foam: one amount image made, carried and lost per frame; a new cause of foam is a source term with its gain, never a second foam layer; picture state only |

## 8. Sources

As listed in `docs/presentation_graphics_plan_2026-11-15.md` (Vlachos 2010;
Chentanez and Müller 2010; Ojeda and Susín; Parberry; Wallace 2011) and
`docs/water_rendering_research.md`. The papers a patch implements are
archived under `docs/papers/` by that patch.
