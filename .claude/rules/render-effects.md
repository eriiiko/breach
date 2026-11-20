---
paths:
  - "renderer/gas_medium.py"
  - "renderer/gas_relief.py"
  - "renderer/gas_detail.py"
  - "renderer/noise_bake.py"
  - "renderer/advected_noise.py"
  - "renderer/fire_vfx.py"
  - "renderer/fire_vfx_ref.py"
  - "renderer/fire_flicker.py"
  - "renderer/particles.py"
  - "renderer/heat_shimmer.py"
  - "renderer/shockwave.py"
  - "renderer/muzzle_flame.py"
  - "renderer/tracers.py"
  - "renderer/water.py"
  - "renderer/water_surface.py"
  - "renderer/water_foam.py"
  - "shaders/gas_*"
  - "shaders/fire_*"
  - "shaders/particle_*"
  - "shaders/muzzle_flame.*"
  - "shaders/water.fs"
  - "shaders/water_foam.fs"
  - "shaders/world_resolve.fs"
  - "tools/muzzle_flame_sheet.py"
  - "tools/wind_drift_probe.py"
---

# Render effects: smoke, fire, particles, distortion, shots

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.
Smoke look tuning: the `smoke-tuning` skill; fire and smoke: `fire-smoke-tuning`.

## Gas medium
- ONE tau (`gas_optical_depth` + `tau_curve`) and ONE per-gas optics binding (`trace_gas_optics`) that every smoke integrator reads. Its own layer is the B2 erosion look, the A/B fallback while `[render.gas_relief] enabled = false`.
- Its base absorption is `[smoke] smoke_absorb_scale`, read from config: one owner, the same key the gases door folds into the sweep's light columns. Its glow is the light accessor's.

## Smoke relief
- Drawn as two SPECIES LAYERS: the floor group (steam, teargas, poison, fuel vapour) on the water and below the units; the ceiling group (`ceiling_gases`, soot) after the units; both before the resolve (Erik's ruling 2).
- Its tau is the gas medium's, its light `light_tex_a/b` (+ `top_light_field`, a filtered read, never a second light), its wind `tame_wind`, its clock `advected_noise`'s.
- Dials are read EVERY FRAME through `relief_settings` and refused when missing or mistyped. A new smoke term goes in this shader, never a second smoke pass; its samplers bind on units from 5 up (raylib's batch holds 4).

## Smoke cast product
- THE smoke's occlusion and height as ONE texture: the relief shader's own product mode (the same C, h and alpha the layers composite), rendered before the scene RT.
- Every pass that receives the smoke's cast shadow (the floor's `shaders/lighting.fs` and the water's `shaders/water.fs` at their `// <SMOKE_SHADOW>` markers; the marine and prop shaders through `_COMMON_GLSL`) or is hidden by it (the heat layer's glow-through and the flame's compensation in `shaders/fire_glow.fs`) reads it through the lit3d block (`smokeCoverAt` / `smokeShadow`): never a second smoke model, never an inline copy (`tests/test_smoke_cast_shadow.py`).
- Glow-through and the own-light cut act only on the glow BELOW the smoke (`glow_below_fraction`; hot gas is #95's). The flame stands under the ceiling soot, hidden by `glow_through_ceiling` only.

## Tamed wind (render)
- `gmap.wind_x` / `wind_y` are the EOS velocity in m/s (Q16), converted honestly as `v / tile_size_m / ticks_per_second` (#51: never the pre-EOS `advection_rate`, never a constant), then smoothed and saturated, calibrated with `tools/wind_drift_probe.py`.
- `GameRenderer.upload_state` computes it ONCE per frame (`frame_tamed_wind`) and every consumer (relief, prop sway, flame VFX, particles) takes that array: never a second call with other dials.
- A consumer that scales it by the air's density (momentum flux, not velocity: Erik's ruling) reads the density through the prop sway's one read (`StaticPropRenderer.sample_gas_frac` / `static_props.gas_frac_at_tiles`): never the planes raw, never a re-derived taming.

## FireVFX pass
- ONE shader (`shaders/fire_glow.fs`), three stages: the FLAME in the linear scene under the units, its COMPENSATION after the ceiling soot, the display-space HEAT layer. A new fire-light term is a stage here, never a second fire pass. Emission A "mask" is shipped (Erik, 2026-10-29); the emitter cover never darkens a lit floor.
- A render-only 2D fluid: reads `writeable=False` dequantized copies, writes no synced state, never runs headless, colours the flame only through the blackbody GPU LUT; the GPU never sees absolute time.
- Its numpy twin `renderer/fire_vfx_ref.py` owns the source law and every stage: import, never re-implement. Its `_hash01` is THE fire's hash, never a second copy. Each GL pass is a named composition of twin stages (`PASSES`, asserted bit-exact) and GL-checked one pass at a time; a dynamics change lands in the twin first.
- `m_u` must equal `m` (laptop m = m_u = 4, accepted gap); `dt_s` 1/120 sits at CFL ~1: do not raise it. `[render.fire_vfx]` is the single source of its dials; budget `tools/lighting_demo.py --fire-vfx-perf`.

## Particle primitive
- Consumers EXTEND it, never build a second. CPU numpy state, CAPPED, every particle with a lifetime it never outlives; motion is drag toward THE frame's tamed wind plus the twin's curl noise; thermal particles cool in closed form and are coloured ONLY through the blackbody GPU LUT.
- Spawns come from the CPU (the flame's per-tick source list `FireVFXPass.pack`, never a GPU readback; the blast walk's seeds handed in by the renderer: `renderer/particles.py` never imports the sim).
- A DARK particle is drawn in `ParticleDraw`'s `"debris"` stage: PREMULTIPLIED, lit by the light field through the lit-3D block, never additive, never a third blend path (`tests/test_debris.py`). Debris drag scales with the tile's gas density through the prop sway's read (Erik, 2026-11-07); a chip that would enter a wall stops.
- The renderer's `_effects` queue is LEGACY display-space effects, not a second particle system, and is not migrated.

## Resolve distortion block
- EVERY screen-space distortion of the world is a TERM of `shaders/world_resolve.fs::distortion_px`: never a second pass over the scene. A zero summed offset reads the plain point fetch bit for bit; a shifted read does its own bilinear tap.
- Heat shimmer's spec is the numpy twin `shimmer_offset_px` (`renderer/heat_shimmer.py`); a pixel whose four nearest tiles are cold is EXACTLY unshifted; the cost follows the heat. Budget 0.3 ms GPU.

## Shock front
- Started from each `"blast"` effect as a term of the distortion block: never a second pass, never from the pressure or velocity planes (measured 2026-11-07: the projection has no travelling front). Its shape comes from the canonical walk (`blast_expansion.reach_depth`); aged on the SIM clock, held while paused; no live ring = the plain fetch bit for bit (`tests/gl_shockwave_check.py`). A new blast distortion is a change here, never a second ring system.

## Muzzle flame
- PROCEDURAL, new every shot, its radiance by ONE law (twin first, then the shader; `tests/gl_muzzle_flame_check.py`); coloured only through the LUT; additive linear HDR after the units, under the ceiling soot, with the compensation.
- WHERE: the held prop's `muzzle` through the SAME matrix that draws the prop. DIRECTION: the round's own, so flame and tracer agree. A shooter that is not drawn flashes nothing.
- A new gun's flash is a `kinds` row + a `weapons` entry + its prop's `muzzle`; a new shape is a term of the law: never a sprite sheet, a second flash pass or a key read. Tune a kind headless with `tools/muzzle_flame_sheet.py` first.
- ACCEPTED GAP: the shot's room LIGHT stays the Frame lights row's, not tied to the flame's seed.

## Tracers
- A streak flies the sim's own path at the round's OWN speed (`round_speeds` from `[ammo.*] muzzle_mps`, never a render copy) × the kind's `speed_scale` (0.20 shipped, Erik's play test 2026-11-12; 1 = honest, which he tried first).
- No shader of its own: a tracer is at most four sprites of the particle draw, drawn by a second `ParticleDraw` instance. A new round look is a `kinds` row + a `weapons` entry. The legacy display-space tracer line is DELETED (the effect stays in the queue: the muzzle light, the flame and the tracers key on it).
- ACCEPTED GAPS: a round that stops on a unit or a cover crate throws nothing; the laser beam is still the legacy line.

## Fire room light (gain + flicker)
- The fire's own brightness dial is `[render.fire_light] gain` (Erik 2026-10-29: "much more brightly"; `gain·(1 + amplitude) ≤ 16` refused): never `[render.blackbody]` (also the heat layer's glow and the light table's top) nor a global light gain.
- The flicker is a LIGHT-ONLY, mean-preserving, per-tile multiplier on the flame's VFX clock from its CPU pack (never a GPU readback), crossing only through `frame_lights.emit_mult_rows` into the sweep's `emit_mult`. Never heat, never a lamp; a new fire-light modulation extends this, never a second multiplier path (`tests/test_fire_flicker.py`).

## Water flow detail
- THE fine water surface (#97 W1): one ripple slope texture (`noise_bake.bake_ripple_rgba`) laid in METRES (`detail_wavelength_m`, never tiles) and carried by the simulated flow; a new surface detail is a term of `water_surface.detail_slope`, never a second wave source. Its law lands in that numpy twin first, then `shaders/water.fs`; `tests/gl_water_flow_check.py` holds them equal per pixel.
- The flow map is `advected_noise`'s two-phase crossfade (spliced at the shader's `// <ADVECTION>` marker), never a copy in water code; its tick-stepped callers stay bit-identical (`tests/test_water_flow_detail.py` pins the frozen formula). Every lookup is hex-tiled (Mikkelsen 2022) with the integer `hex_hash`, so the GPU and the twin draw the same bits; the drift is never wrapped (the hex lattice is not periodic, a wrap would pop it).
- The strength's agitation is the RELATIVE one (`water_surface.relative_agitation`: ripple_v against its wet neighbours', packed per tick in `u_water`'s A, the foam's too), never raw `|ripple_v|`: the sim's ripple field heaves whole rooms as one (W2's finding, 2026-11-20).
- The clock is the pass's own `fire_vfx.SimClock` (sim time, idle when paused), never wall time. Dials pass `water_surface.water_settings`, the one refusing door, every frame; `tools/lighting_demo.py` drives them through `CFG.graphics.water` from its one `_WATER_DIALS` table, never through setters.
- A dry map costs one `.any()` on the INT depth plane and draws nothing (bit-identical); a wet one packs on a sim-state change only and draws the view ∩ the wet box. Its samplers beyond raylib's four sit on units 5 (smoke product), 6 (flow), 7 (ripple), 10 (the ripple's curvature, W4).

## Water caustic
- THE caustic (#97 W4): the Jacobian of the refraction map `x -> x + depth (1 - 1/n) grad h` of W1's ONE fine surface (Wallace's area method, analytic, in gather form), `det J = 1 + a tr H + a^2 det H`, read on the floor's SOURCE light as `min(1/|det|, caustic_max) - 1` x `caustic_strength` (`water_surface.caustic_det` / `caustic_gain` / `caustic_term`): signed, `|det|` (a fold is a bright line, never a black patch), depth only through `a`, never the ambient, never a second wave source, a caustic texture or a time term. Its law lands in the twin first, then `shaders/water.fs`; `tests/gl_water_caustic_check.py` holds them equal per pixel.
- The Hessian is the curvature companion bake (`noise_bake.bake_ripple_curvature_rgba`: the exact second derivatives of the ripple's own waves, in the slope bake's scaling) read through the SAME hex-tiled, flow-mapped taps as the slope (`water_surface.hex_surface` / `detail_surface`): each tap's tensor rotated back on both indices (`R^T Hf R`), blended with the slope taps' weights, x `k / P`, cross-faded with `w0`. A change to the slope lookup changes the curvature read with it, or the caustic leaves the surface it is drawn on.
- A flat surface (or no light) leaves the picture bit-identical to the caustic off: keep it so (`tests/gl_water_caustic_check.py`). The gather form brightens past small slopes and darkens past the first focus (W4's As-built, finding 1): a conserving caustic is the scatter form (M14's other option), never a renormalisation of this one.

## Water foam
- THE foam (#97 W2): one amount image per cell (`[graphics.water] foam_cells_per_tile`), a `core.PingPong` allocated on the first wet frame and never on a dry map, stepped by `WaterPass.simulate` (outside any texture mode, beside the flame's) over the wet box in cells only; the cells that leave the box are cleared in BOTH buffers (`water_foam.leaving_rects`). Its law lands in the numpy twin `renderer/water_foam.py` first, then `shaders/water_foam.fs` (the step) and `shaders/water.fs` (the draw); `tests/gl_water_foam_check.py` holds them equal. A new cause of foam is a term of `water_foam.sources` with its own gain, never a second foam layer.
- PICTURE STATE: never synced, never read by the simulation, never written back. Reset when the sim tick goes back (a replay seek) and when the map dries; a level load builds a new pass. The clock is the pass's `SimClock` in fixed `foam_step_s` steps (idle when paused), never the frame's time step.
- The step is MacCormack + clamp (Selle 2008) in substeps of at most one cell (`water_foam.substeps`, capped; past the cap the displacement is clamped, never a jump); a tap on a blocked (dry or solid) cell reads the traced cell's own value, so a wall neither eats nor takes foam. With no source the maximum never rises: keep it so (`tests/test_water_foam.py`).
- The sources read only the two per-tile textures, through the shader's own `texelFetch` bilinear (the twin's `heat_shimmer.bilinear_tiles`), and W1's terms (`water_surface.edge_flow`, `strength_ratio`): never a second flow or strength. The agitation they read is the RELATIVE one (`water_surface.relative_agitation`, packed per tick in `u_water`'s A, W1's strength's too): the sim's ripple field heaves whole rooms as one, which raw `|ripple_v|` would foam over wall to wall. There is no pour source until water sources exist.
- The draw is a THRESHOLD on the bubble raft (`noise_bake.bake_foam_rgba`, rank-equalised) at two sizes riding W1's flow map (`water_surface.layer_slides`), never a second advector and never a fade (Parberry); `water_foam.lace_threshold` is exact for that bake, so the bake and the threshold change together. Zoomed out the lace fades to its mean with W1's zoom fade. The foam is drawn over the water, lit as the floor is; samplers 8 (foam) and 9 (bubbles) in the draw, 8-10 in the step's direct-GL block.
