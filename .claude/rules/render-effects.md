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
  - "shaders/gas_*"
  - "shaders/fire_*"
  - "shaders/particle_*"
  - "shaders/muzzle_flame.*"
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
- Every pass that receives the smoke's cast shadow (the floor's `shaders/lighting.fs` at its `// <SMOKE_SHADOW>` marker; the marine and prop shaders through `_COMMON_GLSL`) or is hidden by it (the heat layer's glow-through and the flame's compensation in `shaders/fire_glow.fs`) reads it through the lit3d block (`smokeCoverAt` / `smokeShadow`): never a second smoke model, never an inline copy (`tests/test_smoke_cast_shadow.py`).
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
