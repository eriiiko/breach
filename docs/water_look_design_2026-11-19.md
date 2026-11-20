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
- **What:** `levels/water_studio`, written through `level_lib` by a small generator in `tools/` (the precedent is `tools/gen_explosion_studio.py`; never a hand-written `level.toml`). Tile size 0.333 m. Designed with Erik on 2026-11-20 (section 8 has the rulings). Three rooms:
  - **The pool hall,** about 30 x 20 tiles (10 x 6.7 m; resized by Erik on 2026-11-20 from 12 x 8: the rooms were too small in metres), flooded to 0.3 m, with one lamp that can be switched off. A stack of crates (`[materials.furniture]`) stands in the water and is set burning at the start, so there is firelight and smoke over the water. A periodic `timed_charge` goes off in the water every 20 s or so, for waves now and spray in W3. One corner is kept empty for the octopus (section 8): nothing is placed there and nothing is coded for it.
  - **The dam and the corridor,** a corridor 5 x 50 tiles (1.7 x 17 m; was 3 x 25) with its tank at the head and a basin at the far end. At its head is a deep glass tank (1.2 m, the `aquarium_demo` precedent and its safe band). A `timed_charge` breaks the glass a few seconds in, and the flood runs down the corridor into a basin room at the far end. This is one-shot by design: a running source does not exist in a level today and is not added here.
  - **The wading room,** about 24 x 18 tiles (8 x 6 m), flooded to 0.15 m. A `heat_source` entity (a stove or furnace) gives steady firelight along one side, and a fuel pile and a wood wall beside it can catch from it. A marine and a zombie fight here: a `match.toml` with one marine on the layered `bots` and one zombie, spawn rooms inside this room, watched live with `main.py --control match`.
- **Water today:** a standing depth at load is the level's `[water] depth_map` (`water_init.npy`, written through `level_lib`; the map editor and `aquarium_demo` are the precedent). There is no level declaration for a continuous source or a drain (`gmap.water_sources` is filled only by the debug pour and one bench). The floor is flat (`floor_height` is all zero), so each room's water levels out over the whole room: a "pool" is a flooded room sealed by its walls.
- **Fire today:** start it by delivering heat (the CLAUDE.md "Starting a fire" row); the heat source does this for the wading room. The fire code does not read water, so crates burn in the pool as on a dry floor (issue #219, milestone "Water, fire, ice v2"; accepted for the studio).
- **Shots** added to `tools/look_shots.toml`: `still_pool_flashlight`, `flooding_corridor`, `wade_by_firelight`. The existing `steam_over_pool` stays. Units in a shot stand still; no shot needs a unit walking (Erik, 2026-11-20). `wade_by_firelight` shows the marine standing in the wading room's water by the stove's light.
- **Gate:** the level loads, the airtight lint passes, the three shots render; a property test that the corridor's water is still moving at the `flooding_corridor` shot's frame (flow speed above a floor), since a shot of still water proves nothing about W1; the match plays headless through `tools/run_match.py` to a result.
- **As built (2026-11-20).** `tools/gen_water_studio.py` writes `levels/water_studio` (84 x 47, space band, every room sealed; diffuse + the smoke-light studio's seam normal map, no bake; `gen_smoke_light_studio.build_normal` now takes any grid, pixels unchanged for its two studios). Sizes as resized by Erik the same day. Pool hall 30 x 20 at 0.30 m, one lamp (intensity 2.0 for the 10 m room); a 3 x 3 furniture stack in its north-east lit at 0.5 s by one `timed_charge` per tile firing `[payloads.studio_ignite]` (the fire_smoke_studio igniter); the wave charge stands in the water more than its radius from every wall and fires `[payloads.engine_death]` at 8 s and every 20 s, because of the rows that break nothing there it moves the water most (headless: the frag grenade's `frag_standard` 0.02 m/s, this row 0.05 m/s and a depth swing of 0.282 to 0.300 m; furniture is not a blast wall). The octopus corner is the hall's south-west 7 x 7 (`OCTOPUS_CORNER`). Dam: an 8 x 10 tank (2.7 x 3.3 m) at 1.20 m, 96 tile-metres that settle at ~0.18 m over tank, corridor and basin (14 x 15); its front is glass above and below the corridor's mouth and a WOODEN sluice gate (five tiles) across it, blown at 3 s by `[payloads.breach_focus]` against the gate's middle (no hull within the 1.2 tiles where it could break any). Headless the gate opens on the charge's tick, the front runs at ~2.4 m/s at first, reaches a third of the corridor 2.5 s later and the corridor's far end ~20 s after the gate; the tank size was kept, since it floods the whole corridor. Wading room 24 x 18 at 0.15 m: the stove is a 2 x 2 steel block with a `heat_source` ON each tile (radius 0, `heat_per_s` 3200, `max_temp` 1000, about 1290 K), at temperature in ~10 s; at 600 a one-tile stove's light barely showed in the shot. Headless the wood wall catches at ~22 s and the fuel pile at ~50 s; the bot marine stays unhurt by the heat. The match: team 0 one rifleman on `bots`, team 1 one rifleman on `idle` in a sealed 5 x 5 booth, one zombie, 60 s; headless the bot kills the zombie and the match is a draw at the clock (~8 s wall). A `[[spawn]]` marine ("Wader") stands in the wading room's water for `main.py` and the shots (`lighting_demo` spawns `[[spawn]]` rows only). Shot frames: `still_pool_flashlight` 180 (7.5 s, before the first blast, the whole hall at zoom 38), `flooding_corridor` 132 (2.5 s after the gate, front about a third down, ~1.95 m/s), `wade_by_firelight` 300 (12.5 s, stove at temperature). Findings: (1) a blast cannot break glass today (glass is not in `physics._BLAST_WALL_MATERIALS`, the over-pressure burst is off, `burst_enabled = false`, #186), hence the wooden gate; a glass dam is Erik's call. (2) A match needs two marine teams (`parse_match` asks for two rows; `MatchRules` ends at once with one), hence the idle booth team (the bot tournament's `--horde` precedent). (3) The W6a ripple drifts to a uniform offset at its clamp (`k_amp` x depth: ~0.145 m in the pool, ~0.074 m in the wading room) within ~6 s of load and stays there, so a blast's splash barely shows in it (std 3 to 5 mm); W1 and W3 should know before they lean on `ripple`. (4) In today's water look the 0.6 m flood front reads as a pale milky tongue, blue only where shallow.

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
- **As built (2026-11-20).** The law is `renderer/water_surface.py` (pyray-free twin, the one refusing door `water_settings` read every frame, `flow_frame`, `detail_slope`, `wet_box`, `pack_flow`); `shaders/water.fs::detailSlope` / `hexSlope` mirror it and `tests/gl_water_flow_check.py` holds the two equal per pixel (p99 error 1.7 % of the strength, from the GPU's 8-bit bilinear weights). The flow map is `advected_noise`'s crossfade, extended there with `alpha` (sub-tick) and `offset` (per point), plus `layer_ages_weight_field`, `phase_offset_field` and `ADVECTION_GLSL` (spliced at `// <ADVECTION>`); the tick-stepped callers are pinned bit-identical against the frozen formula. The ripple texture is `noise_bake.bake_ripple_rgba` (96 plane waves, integer wave vectors, 16 dominant waves per period, RMS slope 1, slope + height, RGBA32F, REPEAT, mipmapped, trilinear). Added on Erik's word the same day: every lookup is hex-tiled (Mikkelsen 2022, JCGT 11(3); three lattice taps with a hashed offset and rotation, the slope rotated back with its tap, his `|s|`-luminance x `w^7` weights), with an integer hash (lowbias32) so the GPU and the twin draw the same bits; the drift is not wrapped (the hex lattice is not periodic). The slide is capped per layer and the flow fades in over `flow_edge_depth_m`; the strength rises from `detail_strength_still` to `detail_strength_flow` with the flow speed or `|ripple_v|` and fades out zoomed out (`heat_shimmer.zoom_fade`). The clock is the pass's own `fire_vfx.SimClock` (idle when paused); `u_time` (now that clock) survives only for the caustic's procedural detail, which W4 replaces. The second texture is RGBA16F `(vx, vy, solid, previous packed tick's depth)`. Smoke shadow + light curve are spliced at `// <SMOKE_SHADOW>` (lit3d's copy), the light exposed by `u_light_gain`; the render-core gap now names the smoke only. Dry: one `.any()` on the int depth plane, no dequantize, no pack, no draw (GL check: RT bit-identical); wet: packed on a sim-state change only (per tick; CRC-keyed while paused), drawn over view x wet box. `lighting_demo`'s copy is one `_WATER_DIALS` table pushed into `CFG.graphics.water`; the setters, `wave_scale`, `ambient_amp` and their sliders are gone. Dials (starting values, Erik's to tune): `detail_wavelength_m` 0.12, `detail_strength_still` 0.03, `detail_strength_flow` 0.3, `detail_drift_mps` 0.03, `detail_zoom_zero` / `_full` 6 / 14, `flow_cycle_s` 0.6, `flow_speed_ref` 1.0, `flow_max_offset_m` 0.5, `flow_edge_depth_m` 0.02, `flow_phase_noise_m` 3.0, `agitation_ref` 0.05, `debug_view` 0 (1 draws the slope). One fix outside the brief: the glint's light direction is renormalised after bilinear filtering -- between two light sources the filtered direction shortened, L turned vertical, and the now near-flat water lit a tile-stepped staircase of GGX glints (`wade_by_firelight`); the ambient sines had hidden it. Cost (`--perf`, water pass alone, GL_TIME_ELAPSED): water_studio GPU 0.86 -> 1.36 ms (the hex lookup ~0.2 of it), CPU update 0.08 -> 0.29 ms; ship_demo (dry) GPU 0.96 -> 0.00 ms. Properties (`tests/test_water_flow_detail.py`, 17): the crossfade bit-identical for old callers, alpha, field = scalar, the offset smooth and spread; the bake periodic, RMS 1, deterministic; the door refuses, the section holds exactly its dials; still water steady but for the drift; reversing the flow reverses the slide (downstream); the cap holds; the edge fade; the strength; no pulse across a phase reset (a frozen crossfade fails it); no spectral peak at the tile frequency (a tile-locked pattern fails it); no spectral peak at the ripple texture's own period (a plain lookup fails it); the hex lookup keeps 80 % of the contrast; the wet box. Findings: (1) still water shows no reflection of a lamp at all: the light field gives every source the one elevation `light_z` (0.5, ~27 deg), whose mirror direction is ~32 deg off the top-down view, so only slopes near 0.6 glint; `still_pool_flashlight` therefore shows a calm, faintly wobbling floor, not "one clean reflection". (2) The light curve now brightens the water as it does the floor (the pool in `steam_over_pool` reads lighter). (3) The pre-existing bright bands along every wall (the ripple's uniform offset meeting the wall's zero: a Laplacian caustic and the depth-gradient foam) are now curved too and read warm by firelight; W2 (foam) and W4 (caustic) replace both. (4) `refract_strength` is in UV units of the whole map, so the same slope warps the floor more on a larger map. Amended 2026-11-20 (in W2's branch): the strength's agitation is the RELATIVE one (`water_surface.relative_agitation`, ripple_v against its wet neighbours', packed in `u_water`'s A, shared with the foam), not raw |ripple_v|, because the sim's ripple field heaves whole rooms as one and left every still room at ~0.75 of full strength after a blast anywhere; `agitation_ref` 0.05 -> 0.1 m/s for the new input (a still room's frozen residue is <= 0.006 m/s at p99, a blast's crests 0.15-0.33 m/s near the charge).

### W2. Foam with a life

- **State:** a foam amount in its own image through `PingPong`, four cells per tile to start (the one number the plan says can only be chosen on screen; a dial `foam_cells_per_tile`).
- **Each frame, one update pass:** carry the amount along the flow (read the previous image upstream by flow x frame time), add the sources, subtract the decay.
- **Sources,** each with its own gain dial: steep fine surface; fast water meeting a wall or running shallow (flow speed against depth); the advancing edge of a flood (depth appearing where there was none); where water pours in (a source tile); a strong ripple impulse (a blast, a body, from `|ripple_v|`).
- **The draw:** the amount is a threshold on a bubble texture at two sizes, both sliding with the flow (Parberry's point: foam breaks into lace, it does not fade). Lit by the room's light, a soft shadow on the floor beneath, thick foam hides the floor. The old gradient foam and its two dials (`foam_threshold`, `foam_intensity`) are removed.
- **Time:** the update runs on render frames with the frame's real time step, clamped. It is picture state only: never synced, never read by the simulation, reset when a level loads. A paused game holds it still.
- **Dials:** `foam_decay_s`, the five source gains, `foam_bubble_m` (two sizes), `foam_cover` (how much amount closes the sheet), `foam_shadow`.
- **Shots:** `flooding_corridor` (foam at the front and along the walls, travelling), `steam_over_pool` (unchanged where the water is calm).
- **Properties:** with no source the total amount only falls; with zero flow and zero decay the image is steady; the amount moves downstream at the flow's speed within a tolerance; the pass allocates nothing per frame; no BLAS contraction in the per-frame Python.
- **As built (2026-11-20).** The law is `renderer/water_foam.py` (pyray-free twin); `shaders/water_foam.fs` is one SUBSTEP of it and `shaders/water.fs::foamCover` its draw, held to it by `tests/gl_water_foam_check.py` (the step equal per cell to 3e-6, the draw's amount and cover per pixel to p99 5e-3). STATE: an amount in [0, 1] per cell, `foam_cells_per_tile` (4) per tile, in a `core.PingPong` allocated on the first wet frame (a dry map never owns one), stepped by `WaterPass.simulate` (called from `GameRenderer.begin_frame` beside the flame's, outside the scene RT) over the wet box in cells only, the cells that leave the box cleared in both buffers (`water_foam.leaving_rects`), reset on a backward tick (a seek), when the map dries, and by a level load (a new pass). CLOCK (B2): the pass's one `SimClock`, now stepped in `foam_step_s` (1/48 s) on the sim clock, idle when paused, capped at 8 steps a frame; each step is `water_foam.substeps` substeps so none moves more than one cell, capped at 4 (past it the displacement is clamped). SCHEME (M10): MacCormack + clamp (Selle 2008, the flame's) with a tap on a blocked (dry or solid) cell reading the traced cell's own value, then the sources, the decay, the clamp and the kill on blocked cells. SOURCES (M11), all from the two per-tile textures read with the shader's own `texelFetch` bilinear: whitewater by Froude number `|v| / sqrt(g h)` past `foam_froude`; rough water from W1's drive (`water_surface.strength_ratio`) PAST full strength, squared; the advancing edge `(share of the depth that arrived this tick)^2` from the previous-depth channel (the first packed tick is its own previous, so a standing pool does not foam on load); fast water along walls from the bilinear solid channel x speed^2; no pour source. The agitation the rough-water source reads is RELATIVE (`water_foam.relative_agitation`: ripple_v against its wet neighbours', packed per tick in `u_water`'s A), because the sim's ripple field heaves whole rooms as one (finding 1). DRAW: a threshold on a rank-equalised Worley bubble raft (`noise_bake.bake_foam_rgba`) read at two sizes (`foam_bubble_m`) along W1's flow map (`water_surface.layer_slides`, refactored out of `layer_coords`; GLSL `flowLayers`), each flow-map layer thresholded on its own and the covers cross-faded, the threshold exact for the mean of two uniforms (`lace_threshold`), so the drawn share is the amount's (Yingst, Alford & Parberry 2011, archived); faded to its mean with W1's zoom fade; composited OVER the water (opaque where it covers; lit as the floor, through the light curve's albedo; the glint hidden), its shadow `1 - foam_shadow x f` on the refracted floor and its caustic. The gradient foam, `foam_threshold` / `foam_intensity`, the dead `WaterFieldOverlay`, its test and its `[display] water_*` keys are deleted; `lighting_demo`'s `_WATER_DIALS` carries the float foam dials. `tools/lighting_demo.py --water-perf [N]` is the water pass's own cost instrument (added). `flooding_corridor` re-framed to centre (30, 28.5): its bottom eighth was the space band. Dials (starting values, Erik's to tune): `foam_cells_per_tile` 4, `foam_step_s` 1/48, `foam_decay_s` 2.5, `foam_froude` 0.8, `foam_gain_froude` 3.0, `foam_gain_steep` 0.3, `foam_gain_edge` 10.0, `foam_gain_wall` 4.0, `foam_bubble_m` [0.05, 0.2], `foam_cover` 1.0, `foam_shadow` 0.5; `debug_view` 2 draws (amount, cover). Cost (`--water-perf`, GL_TIME_ELAPSED, same session): water_studio draw GPU 1.6 -> 2.1 ms (the foam's draw ~0.35 of it), foam step 0.09-0.11 ms GPU / 0.08 ms CPU, CPU update 0.29 -> 0.50 ms (the per-tick relative agitation and flow max); ship_demo (dry) 0.00 ms GPU, update 0.02 ms, step 0.001 ms. Properties (`tests/test_water_foam.py`, 35 with the parametrised cases): with no source the maximum never rises, and the revert is what holds it; still water with no decay is steady to the bit; foam moves downstream at the flow's speed and back; a substep moves at most one cell; dried / solid cells are killed; a wall neither eats nor takes foam; each source fires where it should; a room heaving as one makes none; a flood front makes foam that travels on; the leaving rects; the bake uniform, periodic, deterministic; the lace covers the amount's share and is lace, not a fade; the sheet closes, no foam draws nothing; zoomed out it is its mean; the bubbles ride the one flow map; the shadow; the door; the old dials gone. GL (`tests/gl_water_foam_check.py`, 6). Gates (Erik's 2026-11-20 ruling: no full suite per patch): `affected_tests` selects the FULL suite (a deleted test file, `config.toml`, `look_shots.toml`), so the narrower set ran -- the 123 test files `affected_tests.select` names for the patch's code files, every test file that reads water, `[graphics.water]`, `[display]` or the look shots, and `tests/test_claude_rules_paths.py` (177 files): 2479 passed, 45 skipped, 2 xfailed, 1 failed -- `test_blast_effect`'s source read of lighting_demo (the new `--water-perf` consumed tick events after a helper, not a literal `sim.step()`), fixed and green on rerun. Findings: (1) the sim's ripple field (W6a) heaves whole rooms as one: after the pool blast at 8 s the wading room's ripple_v jumps to ~0.84 m/s uniformly and then sits at a frozen 0.0375 m/s with the ripple pinned at its clamp (`k_amp x depth`); W1's strength read that raw |ripple_v|, so still rooms ran near 0.75 of full fine-surface strength after any blast anywhere; it now reads the same relative agitation (W1's amended As-built). (2) `steam_over_pool`'s pool is not calm: fire_studio's 0.4 m block of water is unwalled and collapses outward, so at frame 16 it foams at its spreading rim and laces inside (the old gradient foam drew a pale halo there). (3) The pale wall bands are gone with the gradient foam; the thin warm line that remains along the pool's walls is the Laplacian caustic's step (W4's). (4) The flood front in `flooding_corridor` reads as a closed white sheet for ~4 m behind the front, then lace and two foam lines along the walls; the length of the sheet is `foam_decay_s` and the edge gain.

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
- **As built (2026-11-20).** M14 resolved by the orchestrator: the analytic Jacobian in GATHER form, the term signed. The law is in the twin `renderer/water_surface.py`: `hex_surface` (W1's hex lookup, now also returning the Hessian: each tap's baked tensor rotated back on both indices, `R^T Hf R`, blended with the slope taps' own normalised weights -- the weights' gradient, confined to the `w^7` seams, is dropped, so the hex surface is a weighted mix of three honest surfaces), `detail_surface` (the slope bit for bit as `detail_slope`; the Hessian per layer x `k / P`, cross-faded with `w0`; the slide's, the strength's and the crossfade's own gradients, a tile or more in scale against a 12 cm wave, are dropped), `caustic_det` (`det J = 1 + a tr H + a^2 det H`, `a = depth x (1 - 1/1.333)`: exact for the map `x -> x + a grad h`, the only approximations the small-slope Snell law and a vertical light), `caustic_gain` (`(min(1/|det|, caustic_max) - 1) x caustic_strength`) and `caustic_term` (the floor's source light x the gain). Two choices beyond the brief: `|det|`, not `det` -- past a focus the beam is inverted, not dark (`E/|det|`), so the folds are the bright lines and the floor between keeps its light; the brief's `clamp(1/det, 0, max)` blacks out every folded patch -- and the caustic multiplies the floor's SOURCE light inside the floor's own lighting (`albedo x (ambient + floorL x (1 + gain))`), so the ambient is never modulated and the albedo shades it like any light (the Laplacian added light past the albedo). The curvature is a companion bake, `noise_bake.bake_ripple_curvature_rgba` (`ripple_field(second=True)`: the exact second derivatives of the same 96 waves, in the slope bake's scaling; five numbers do not fit one texel), RGBA32F, REPEAT, mipmapped + trilinear like the ripple, on unit 10, read by the same `textureGrad` taps; zoomed out the mips and W1's zoom fade take it to flat. `shaders/water.fs` mirrors it (`hexSurface`, `detailSurface`, `causticDet`, `causticGain`; the curvature is read only while `caustic_strength > 0` or debug view 3 draws `(hxx, hxy, hyy, gain)`); the depth is the per-pixel `depthPos`. DELETED: the Laplacian caustic, its procedural detail, `caustic_scale` and the shader's last `u_time` (W1 finding 6); the warm line along walls went with the Laplacian (W2 finding 3; `wade_by_firelight`'s orange rim around the stove and the wood wall, the pool's and the tank's pale rims). Dials (starting values, Erik's to tune): `caustic_strength` 1.0 (was 2.5 for the Laplacian; 1 is the area method's own contrast), `caustic_max` 3.0 (the door refuses <= 1); `lighting_demo`'s `_WATER_DIALS` in step. Shot: `still_pool_flashlight` moved from frame 180 to 240 (10 s, 2 s after the pool's first blast); the name stays. Cost (`--water-perf`, water_studio, world RT 4032 x 2256, same session): draw GPU 1.85 -> 2.42..2.77 ms (with the caustic off 2.01: the term is ~0.4-0.75 ms, six RGBA32F taps and the tensor algebra); ship_demo (dry) 0.001 ms. Properties (`tests/test_water_caustic.py`, 10): the curvature bake is the slope bake's exact derivative (central differences of the field; the texels in the slope's scaling); where one hex tap carries the blend the Hessian is the hex slope's derivative and the UNROTATED tensor is far off; the surface Hessian is the slope's derivative in metres (`k / P`; without it far off) and the slope is `detail_slope`'s bit for bit; a flat surface gives det == 1 and a caustic of exactly 0; no light, no caustic; over one full period of a small surface the signed mean is SECOND order (`a^2 <tr H^2>`: `<tr H>` and `<det H>` vanish exactly over a period), asserted |mean| <= 1.5 rms^2 and quartering when the amplitude halves (measured mean / rms^2 = 0.93 .. 0.99 at RMS slopes 0.03 .. 0.01, 0.3 m); the term never exceeds `(caustic_max - 1) x strength` or takes more than the light, folds and deep water included; `|det|` (inverted, not dark); deeper water focuses more, proportionally while small, and the clamp holds it; the door and the deletions. GL (`tests/gl_water_caustic_check.py`, 4, wrapper `test_gl_water_caustic.py`): the Hessian per pixel p99 2.1 % of its RMS and the gain p99 0.012 where |det| > 0.5 against the twin; a flat surface and an unlit floor bit-identical to the caustic off; under light both signs, the cap held. Gates (Erik's 2026-11-20 ruling: no full suite per patch): `affected_tests` selects 420 of 450 test files for `tools/look_shots.toml` alone (a data file its closure cannot see), so the narrower set ran -- the 125 files `affected_tests.select` names for the patch's code files, every test file that reads water, `[graphics.water]` or the look shots, and `tests/test_claude_rules_paths.py` (169 files): 2326 passed, 44 skipped, 1 xfailed, 1 failed -- the rules test, on a `## Water caustic` section with no CLAUDE.md index row; the caustic's rules were folded into the Water flow detail section and the test passes (74), the water + GL set rerun green (73). Findings: (1) the GATHER form is biased bright past small amplitudes: the pixels sample the surface uniformly while the focus sits in less floor area, so the mean gain is `+a^2 <tr H^2>` and grows (per unit strength at 0.3 m with `caustic_max` 3: +0.20 at RMS slope 0.1, +0.61 at 0.3, W1's full strength); past the first focus it turns dark instead (the 1.2 m tank at full strength: half of it folded, 1/|det| of ONE preimage while the floor point has three or more) -- `flooding_corridor`'s tank now reads as a busy darker net. The scatter form (M14's other option: a grid rasterised into a target, Wallace's own) conserves and is the fix if Erik wants it. (2) The pool blast's agitation is local and brief: the relative agitation saturates only on the few tiles round the charge (18, 14), for under a second, under the fireball (headless: hall max sat 1.0 at 8.2-8.75 s, 0.27 at 9.5 s, 0.15 from 10 s; the beam region's mean 0.02-0.03) -- so the shot's "net after a disturbance" is the still water's net at `detail_strength_still` (+-~13 % of the light at 0.3 m), drifting at `detail_drift_mps`; a visible disturbed net waits for W3's wakes or a sim ripple that answers (W0 finding 3). (3) A pre-existing test pinned the debug-view set (`test_water_foam`'s door case `debug_view = 3`); it now asks `max(DEBUG_VIEWS) + 1`. (4) The floor read's bend (`refract_strength`, map UV, the opposite sign to Snell; W1 finding 4) is not the caustic's law: with the same law the gather would be exact for one preimage (the floor seen through a point IS where that point's light lands); unifying them is a look change left to Erik. (5) The light is taken as vertical; the field's one elevation (`light_z`) does not shear the pattern (accepted, v1).

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
- Every animated term runs on the SIM clock, `(tick + alpha) / tps`, through `fire_vfx.SimClock` (idle when paused, re-anchored on a seek), never wall time (critique B2).
- Each law (the flow surface, the foam step, the caustic, the waterline tint) lands first in a numpy twin; the shader is GL-checked against it and the properties are asserted on the twin (critique M5).
- The water dials pass one refusing settings door, read every frame; `tools/lighting_demo.py`'s copy of them changes in the same patch (critique M6).
- A dry level renders bit-identically and costs nothing (critique M4).
- The frame cost is measured before and after with `tools/lighting_demo.py --perf` on the water studio and on `ship_demo` (dry), and reported in milliseconds. A dry ship must cost the same as today.

## 4. Gates and merging

- Per patch: the affected tests (`tools/affected_tests.py --run`), the render test files, `tests/test_claude_rules_paths.py`, the shot set before and after, the frame cost.
- **Erik, 2026-11-20: affected tests per patch, the full suite once at arc close.** Where `tools/affected_tests.py` selects the whole suite only because `config.toml` or a rules/CLAUDE.md file changed, the per-patch set is instead: the water and render test files, the GL checks, `tests/test_claude_rules_paths.py`, the tests that read the changed `config.toml` section, and `affected_tests.py` over the patch's code files. These patches touch no simulation or device file, so no CUDA gate applies.
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

**Existing systems this design must use:** the advected noise flow map (`renderer/advected_noise.py`, critique B1); GameRenderer + WorldComposite;
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
| Water caustic (a term of the Water flow detail row, no index row of its own) | W4 | THE caustic: the Jacobian of the fine surface's refraction map on the floor's source light (`water_surface.caustic_*`); never a second wave source or a caustic texture of its own |

## 8. Rulings with Erik, 2026-11-20

- The studio as section W0 describes it: room sizes as given; the dam break instead of a running source; a heat source for steady fire; the marine and the zombie fight; units stand still in shots.
- **Units in water:** units stand on the floor and must stand out of the water by the right amount for their height. Today the water is drawn before the units, so a unit is drawn whole on top of it. The waterline (the part below the surface seen through the water, and a foam ring at the surface) joins **W3**.
- **The octopus** is a simulation creature on the unmerged branches `33-octopus-concept` / `106-p6g-swim-steering`. It is not recreated or brought in for this arc; the pool hall keeps a corner free for it.
- **Not in this arc:** a water source and drain (Erik builds those); exploding barrels and cargo (#218); water with temperature, water putting out fire, ice, steam, floating, ground depth (milestone "Water, fire, ice v2", #219). Floor heights stay 0.
- **Tilt** (the water solver's tilt input, and a camera that shows it): may come before this arc or inside it. Open; not in W0.
- The marks layer (#180) is not built first; W5 waits for it.

## 9. Critique (one Opus critic, sections 2-3, against main 7cba57dd, 2026-11-20)

Verdict: two blockers, both resolved on paper below; twelve majors folded into the
patch briefs; minors accepted. The orchestrator spot-checked the blockers and the
gap fact against the code (advected_noise.py:21/169/185; game_renderer.py:500/1079;
render-core.md:47).

**Blockers, resolved:**
- B1 W1 would build a second Vlachos flow map: `renderer/advected_noise.py` already
  holds the two-phase crossfade (`advection_phase`, `layer_ages_weight`), shared by
  gas detail, gas relief, heat shimmer. RESOLVED: W1 uses and extends it (sub-tick
  alpha, per-pixel phase offset), never a copy in water code; added to section 7.
- B2 no defined clock; the pass runs on wall time (`perf_counter`), animating through
  pause and drifting from the sim at `sim_speed` != 1 and in shots. RESOLVED: new
  section 3 rule -- W1-W3 run on the sim clock `(tick + alpha) / tps` through
  `fire_vfx.SimClock` (idle when paused, re-anchored on a seek), fixed substeps.

**Majors, into the briefs:**
- M3 (fact fix) the render-core gap is the LIGHT CURVE, not the smoke shadow. W1
  splices `_SMOKE_SHADOW_GLSL + _LIGHT_CURVE_GLSL` at a `// <SMOKE_SHADOW>` marker
  (the `lighting.py` precedent) and only then removes the water half of the gap line.
- M4 dry cost: the draw runs every frame over the whole world RT and dequantizes
  depth twice before `.any()`. W1: `.any()` on the int plane first, pack on tick
  change only, skip the draw when dry, clip to view x wet bbox. "A dry map costs
  nothing" becomes true and is measured.
- M5 new section 3 rule: each law (flow surface, foam step, caustic, waterline tint)
  lands in a numpy twin first; the shader is GL-checked against it; properties on
  the twin.
- M6 one refusing `water_settings(cfg)` door read every frame; `lighting_demo`'s
  parallel copy of the water dials edited in the same patch.
- M7 W1 slide capped per half-cycle (`flow_max_offset_m`), flow weighted by a depth
  smoothstep at wet/dry edges; property: the cap holds.
- M8 W1 detail faded by screen px per tile (`heat_shimmer` zoom fade), ripple
  texture mipmapped + repeat.
- M9 W2 property replaced: with no source the MAXIMUM never rises (MacCormack +
  clamp), not "the total only falls".
- M10 W2 substeps (at most one cell per step), backtrace clamped to wet non-solid
  cells, scheme named (MacCormack + clamp, as fire_vfx).
- M11 W1's second texture is RGBA16F `(vx, vy, solid, previous depth)`; whitewater
  source by Froude number; steepness source from W1's strength term; the pour
  source dropped until sources exist.
- M12 W3 waterline: a `lit3d._WATERLINE_GLSL` Beer-Lambert tint over the water
  column above the fragment's height (water texture on a free material slot), the
  ring a W2 foam source under the unit; metres to drawn scale (marine 6 tiles from
  1.83 m); the lit3d byte-identity oracle re-baselined deliberately; property: a
  dry tile leaves a unit fragment bit-identical. Top-down camera: what shows is
  limbs past the silhouette, prone / dead bodies, and the ring. Erik, 2026-11-20:
  the straight-down camera may tilt a little later, so the waterline is a per-
  fragment law in world height (it then shows on the body as the camera tilts),
  never a screen-space trick that assumes top-down.
- M13 W3 wake is a W2 foam source (+ optionally a V-wake term inside W1's one
  surface function), never a render-side ripple field.
- M14 W4: choose scatter (grid mesh into a target, conserves) or the analytic
  Jacobian with a clamp; term signed `(1/det - 1)·light`; curvature baked into
  the ripple texture; conservation tested on the scatter form only.

**Minors, accepted:** the grid texture is RGBA16F (section 1 said 32F); W1's
tile-offset property becomes "no spectral peak at the tile frequency + bounded
change across a phase reset"; W3 reads blasts through the `"blast"` effect with a
`spray_seen` marker (the blast log is a sim seam); "a round hitting water" becomes
a splash at a wet wall-stop or is dropped; unit speed from position deltas per sim
tick; spray in the debris stage with an analytic parabola, landings handed to W2
as a source list; delete the dead `WaterFieldOverlay` and its `[display] water_*`
keys in W2; foam PingPong allocated lazily, reset on a backward tick; every patch
asserts a dry level renders bit-identically; W3's gate is properties plus a live
check (no wake shot, units stand still).

## 10. Sources

As listed in `docs/presentation_graphics_plan_2026-11-15.md` (Vlachos 2010;
Chentanez and Müller 2010; Ojeda and Susín; Parberry; Wallace 2011) and
`docs/water_rendering_research.md`. The papers a patch implements are
archived under `docs/papers/` by that patch.
