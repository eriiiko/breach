# Rendering

**Depends on:** [State](../state) (world arrays + material table, the `gmap.<field>` interface, the `Simulation` facade), [Ray engine](./08_ray_engine.md) (per-channel `light_atten`, the `light_rgb` / `heat` / `smoke_glow` outputs, ACES tone-mapping).

---

The renderer turns the simulation's world arrays into a lit, scrolling image. It is pure presentation: it reads `gmap` fields and the unit/projectile lists, casts the ray engine's light field, and composites — it never writes simulation state. This is the same boundary the rest of the engine respects: the deterministic `Simulation` facade owns the world; the renderer is one of its readers, exactly like a headless training loop is, except it draws instead of computing a reward.

The backend is **pyray** (Python bindings to the Raylib C library). The decision to use pyray over pygame is deliberate and load-bearing: a single language drives game logic, physics (via the `breach_physics` pybind module), and rendering, and the GPU path unlocks the normal-mapped, shader-lit look the game is built around. There is no pygame fallback — pyray is the only rendering path.

The whole design rests on a **resolution split**. The simulation runs at *physics-tile* resolution (a typical ship is around 50 x 120 tiles). The art runs at high resolution (a diffuse PNG of roughly 1000 x 700 px, plus a matching normal map). The light field is computed cheaply at physics resolution and *interpolated* across the high-resolution art on the GPU, where a per-pixel normal map gives it surface detail the coarse light field never had. Low-fidelity systems output, high-fidelity image in — that is the core trick.

## Module layout

The renderer is a package, not a monolith. A thin orchestrator calls into focused modules:

| Module | Responsibility |
|---|---|
| `renderer/game_renderer.py` | `GameRenderer` orchestrator: per-frame upload, compose, blit, panel, input toggles, coordinate conversions. |
| `renderer/core.py` | pyray surface: window lifecycle, texture loading, dynamic-texture create/upload (RGBA8 and RGBA16F), frame begin/end, shader load with fallback. |
| `renderer/camera.py` | `Camera2D` — pure-math view state (position, zoom, viewport) and screen↔world↔tile conversions. No GL. |
| `renderer/world_composite.py` | `WorldComposite` — the world-space render target and its Y-flip blit to screen. |
| `renderer/lighting.py` | `LightingPass` — owns the lighting shader, casts the ray engine, packs the light field into textures, draws the lit ship. |
| `renderer/overlays.py` | Smoke, fire, god-ray glow field overlays; unit/waypoint/grid/text/panel draw helpers. |
| `renderer/pressure_overlay.py` | `PressureOverlay` — per-tile pressure colormap from `atmosphere + wave_p`. |
| `renderer/sprites.py` | `UnitSprites` — loads the 8-directional marine sprites and the zombie variant pool once. |
| `renderer/coords.py` | Coordinate-space naming discipline (`_tile` / `_wpx` / `_spx` suffixes) and the tiny conversion functions. |
| `shaders/lighting.{vs,fs}` | The GLSL lighting shader. |

The split exists because the compose phase grows with every entity type. Keeping the GPU-resource owners (`LightingPass`, `WorldComposite`, the overlays) separate from the orchestrator means adding a layer is one more draw call, not a shader rewrite.

## The world render target

Everything that lives in the game world is drawn into a single render target sized to the *entire world*, not the screen. `WorldComposite` allocates an RT of `world_tile_w * world_px_per_tile` by `world_tile_h * world_px_per_tile` pixels (default `world_px_per_tile = 24`). For a 50 x 120 ship that is 1200 x 2880 RGBA8 — about 14 MB, trivial on any modern GPU.

This is the architectural keystone. Because the diffuse art and every overlay cover the *full* world RT, every shader and every overlay samples `fragTexCoord` running 0..1 over the world. The light field covers the same 0..1 space. They line up *by construction* — no shader ever needs camera UVs. This is why an entire class of bug ("the flashlight points at the wrong tile after the camera scrolls") cannot recur: the camera transform does not exist during compose. It happens later, exactly once.

```
compose phase (inside BeginTextureMode(world_rt), no camera math):
    lit ship  (diffuse x normal x light field, via lighting shader)
    smoke     (premultiplied alpha)
    god-ray glow (additive)
    fire      (additive)
    pressure colormap (premultiplied alpha)
    orders / units / projectiles / effects / grid

blit phase (camera applies here, exactly once):
    DrawTexturePro(world_rt -> screen map area, src = camera's visible world rect)
```

The camera enters only in `WorldComposite.blit_to_screen`, as a single `DrawTexturePro` from the world RT to the map area of the screen. The source rectangle is the camera's visible world region; Raylib scales it to fill the destination. Zoom, pan, and letterboxing are all expressed as source/destination rectangle math in this one blit.

**Y-flip.** Raylib render targets store their contents origin-at-bottom-left (OpenGL), while the screen is origin-top-left. The blit corrects this with the standard idiom — the source rectangle starts at the bottom edge (`world_px_h - (y + h)`) and uses a *negative* height to flip Y back. This negative-height trick lives in exactly one place; every consumer of the world RT goes through `blit_to_screen`.

**Letterboxing, not stretching.** When the camera is zoomed out past the world bounds, the visible region is clipped to the world, the destination rectangle is shrunk and offset proportionally, and the empty space becomes black bars. The renderer never stretches the edge tile to fill space.

### Why this future-proofs the renderer

The single-RT pattern is what makes the forward roadmap cheap rather than a rewrite:

- **Post-processing** (bloom, vignette, CRT) becomes one extra full-screen pass over the RT.
- **Multiple cameras** (a security-cam inset, split-screen, replay) re-blit the *same* RT with a second `Camera2D` — no second render.
- **HDR** is a constructor argument plus an `rlgl` call away (the RT defaults to RGBA8 today).
- **Adding a physics field as a visible layer** (heat shimmer, gas, lightning) is one draw call inside compose, because it samples the same 0..1 world UVs as everything else.

## The camera

`Camera2D` is a pure dataclass — no GL state, no pyray imports, trivially testable. It owns the viewport's top-left position in world-tile units (`pos_tile_x/y`, sub-tile precision via float), the zoom in screen-pixels-per-tile, the viewport size, and the world bounds for clamping. Raylib's built-in `Camera2D` + `BeginMode2D` were deliberately rejected: the project wants explicit world-tile units, its own screen↔world transforms, and easy multi-camera — none of which the built-in helper gives cleanly.

The camera exposes the conversions the rest of the renderer needs:

- `world_tile_to_screen_px` / `screen_px_to_world_tile` — the forward and inverse transforms, both accounting for the viewport anchor (so a non-origin viewport, e.g. a security-cam inset, converts mouse clicks correctly).
- `visible_world_rect_world_px` — the source rectangle the blit needs.
- `pan`, `set_zoom`, `clamp_to_world` — operations that keep the camera inside the world (zoom has a hard floor of 1 screen-px/tile).

`clamp_to_world` locks the camera at the origin when the world is smaller than the viewport; the renderer is then responsible for letterboxing (which the blit does), not the camera. Mouse-to-tile conversions (`mouse_to_tile`, `mouse_to_tile_float`) route through the camera so clicks land on the correct world tile at any zoom and pan.

## Lighting: the heart of the renderer

The look of Breach is dynamic, directional, normal-mapped light over otherwise simple art. The pipeline has two halves: a CPU-side cast at physics resolution, and a GPU-side per-pixel shade.

### Casting the light field

Each frame, `LightingPass.compute_light_field` casts every active `LightSource` through the C++ ray engine. The march reads the world's **per-channel dynamic attenuation field**, `gmap.dyn_light_atten` — an `(h, w, 3)` float field equal to the static material attenuation MAX'd with stamped-unit opacity, rebuilt each tick in `stamp_units`. Occlusion is per-channel: an opaque tile `[1,1,1]` kills the ray exactly like a hard wall stop, glass transmits a dimmed copy, and an unequal triple tints the surviving light. Units stamped into this field restore their shadows; smoke remains a separate live input (`gmap.smoke`) passed alongside.

The cast accumulates, per physics tile:

- `light_rgb` — the incoming light *colour* (RGB), so a red lamp lights surfaces red.
- `light_dx`, `light_dy` — the dominant 2D light *direction*, a weighted average over all rays passing through the tile. After all sources are cast, the direction is normalized **by vector magnitude** (`/sqrt(dx²+dy²)`), not by intensity. Normalizing by intensity is wrong: two opposing rays at a bright tile would cancel the direction to (0,0). Vector-magnitude normalization gives the correct dominant direction and a sensible fallback at pure-cancel tiles.

The march also writes two ray-engine output buffers in place: `gmap.heat` (the Q16.16 heat deposit, a simulation-affecting field the ray engine owns) and `gmap.smoke_glow` (the per-channel light the smoke *scattered* — the render-only god-ray field, see below).

A render-only scalar `light_map = max over RGB channels` is derived for the consumers that still want a single brightness value (unit sprite tinting).

> The cast currently lives in the renderer (`upload_state` calls it). The ray engine is designed to move the cast into the simulation step so headless runs get the same light/heat fields; until then, the heat and glow buffers are passed in optionally and the renderer owns the per-frame invocation.

### Packing the light field

The light field is uploaded as **two RGBA16F (half-float) textures** at physics resolution:

| Texture | RGB | Alpha |
|---|---|---|
| A | `light_rgb` (HDR light colour) | `light_dir.x` (signed) |
| B | `smoke_glow` (god-ray glow) | `light_dir.y` (signed) |

Half-float is chosen on purpose. It carries HDR colour (values above 1.0) into the tone-map stage, stores the *signed* direction directly with no 0.5-centered encode, and avoids the near-dark banding an 8-bit field shows at ambient ~0.01. (An earlier 8-bit RGBA light field quantized direction to 256 angles and banded on smooth surfaces; the 16F format retired that problem.)

The light textures default to bilinear filtering, which is what produces smooth light across the high-resolution art from a coarse field. A toggle (`B`) flips them to point/nearest for a crisp, blocky comparison.

### The lighting shader

`shaders/lighting.fs` runs once over the full world RT as the lit ship is drawn. For each art pixel it:

1. **Discards vacuum.** A per-tile vacuum mask (uploaded once at level load) marks tiles outside the ship; the shader `discard`s them so the screen-fixed background (stars, void) shows through the breach.
2. **Decodes sRGB.** The diffuse PNG is sRGB-encoded; lighting math must be linear, so the shader decodes (`pow 2.2`) on input and re-encodes on output. Skipping this biases midtones dark and shifts shadow hue. (Toggleable with `G` for comparison.)
3. **Samples the light field** at `fragTexCoord` — same 0..1 world space as the diffuse, so the coarse field lines up with the art for free. Reconstructs `incoming_rgb` and the signed 2D light direction.
4. **Applies the normal map.** Unpacks the per-pixel normal `[0,1]→[-1,1]`, optionally flips Y for DirectX-convention maps (Laigter output varies; toggle `H`), blends toward flat by `normal_strength`, builds a 3D light vector `(dir.x, dir.y, light_z)`, and computes `ndotl = max(dot(N, L), 0)`. `light_z` controls the apparent light height — low = grazing/high-relief, high = overhead/flat — and is live-tunable.
5. **Composites and tone-maps.** `lit = diffuse * (ambient + incoming_rgb * ndotl)` — a Lambertian model with a flat ambient floor and a directional, normal-modulated term. The HDR result passes through an **ACES filmic tone-map** (Narkowicz approximation) that compresses over-bright coloured light toward [0,1] while staying saturated, instead of per-channel clipping that hue-shifts bright warm light toward white.

The vertex shader is a standard pass-through matching Raylib's default. Shader loading falls back to Raylib's default shader if the GLSL fails to compile, so a shader error never crashes the game — it degrades to flat.

### God-rays (lit smoke)

Lit smoke is handled by an additive **god-ray glow** layer, not by tinting the smoke surface. The ray march deposits, into `gmap.smoke_glow`, exactly the per-channel light the smoke *removed* from each ray. The renderer draws that field additively (the `GlowOverlay`) over the smoke, before units. The result is energy-conserving by construction — a red beam through smoke casts a red shaft, and the shaft already terminates at walls because the march deposits no glow past opaque tiles.

This supersedes an earlier `light_modulation` approach that multiplied the smoke colour by the local light to fake lit smoke. That double-counted energy and was retired. Smoke is now a flat grey density medium (alpha driven by density alone); the glow layer adds the colour the smoke scattered. One mechanism, no double-count.

## Overlays and blend discipline

Smoke, fire, the god-ray glow, and the pressure colormap are uploaded as dynamic RGBA8 textures at physics resolution and drawn stretched across the world RT. Blend mode is not incidental — it is the difference between a correct image and "galaxies showing through the ship after a grenade":

- **Smoke and pressure** are packed with **premultiplied alpha** and drawn with `BLEND_ALPHA_PREMULTIPLY`. Raylib's default `BLEND_ALPHA` applies `SRC_ALPHA` to *both* colour and alpha, so drawing semi-transparent smoke over an opaque ship pixel *reduces* the destination alpha. When the world RT is later blitted to screen, that lowered alpha lets the background bleed through pixels that should be solid ship. Premultiplied alpha with the premultiply blend mode gives correct Porter-Duff "over" compositing: ship alpha stays at 1.0.
- **Fire and the god-ray glow** are drawn with `BLEND_ADDITIVE`. Additive blending raises RGB without touching destination alpha, so these layers are *not* premultiplied — additive light correctly brightens without making the ship transparent.
- **The final RT→screen blit** itself uses `BLEND_ALPHA_PREMULTIPLY`, because the RT contents are already in premultiplied form. Blitting with normal alpha would multiply by alpha a second time.

The **pressure colormap** (`PressureOverlay`) maps `gmap.atmosphere + gmap.wave_p` through configurable colour stops (`config.toml [rendering] pressure_stops`), masked to non-wall, non-vacuum tiles, point-filtered so the per-tile pressure cells read as cells rather than a soft blob. It defaults on in the main game so explosions look dramatic, and is toggleable (`F7`).

Units are drawn from the **physics-tile float positions** (`unit.x`, `unit.y`) so the renderer interpolates sub-tile motion smoothly even though game state is discrete. Marines use 8-directional sprites keyed by `facing_compass()`; zombies draw from a stable variant pool keyed by `unit.id`. Each sprite is tinted by the scalar `light_map` sampled at the unit's centre tile, lifted by the ambient floor so units in unlit rooms match the faintly-lit ship around them rather than going pitch-black. (This is a stopgap until units get their own normal maps and are lit by the shader like the ship.)

Short-lived **visual effects** (bullet tracers, explosion rings, hit splats) are spawned by reading the simulation's tick events (`consume_events`), held in a small decay list the renderer owns, and drawn into the world RT so the camera transforms them like everything else. The simulation emits events; it does not track fade — that is a presentation concern.

## Frame flow

`main.py` drives the loop. Per frame:

1. `upload_state(gmap, light_sources)` — cast the light field (or clear it when lighting is off), update the smoke / fire / glow / pressure textures, push the GPU uploads. All the per-frame numpy work is concentrated here.
2. `begin_frame()` — clear the screen.
3. `compose_world(...)` — draw every world-space layer into the world RT (order above).
4. `draw_background_to_screen()` — the screen-fixed level background (stars) behind the map area.
5. `blit_world_to_screen()` — the single camera blit from RT to the map area.
6. `draw_panel(...)` — the right-side info panel (state, selected unit, FPS, raycast/frame timings, toggle states, control cheat-sheets) drawn directly on the screen.
7. `end_frame()`.

Frame and raycast timings are sampled separately (`last_frame_ms`, `last_raycast_ms`) and shown on the panel — the per-frame instrumentation the renderer was built with.

## Tooling: the lighting demo

`tools/lighting_demo.py` is a standalone tuning tool that reuses the exact production pipeline — same `level_loader`, `Simulation`, and `GameRenderer` — but orchestrates differently: a live physics sim, a single mouse-flashlight light source, and a raygui slider panel. It exposes the renderer's public lighting setters (`set_ambient`, `set_light_z`, `set_normal_strength`, `set_use_normal`, `set_srgb_decode`) and the `FieldOverlay` tint/alpha knobs as sliders, adds click-to-spawn grenades parameterised live, the pressure colormap toggle, and a cursor HUD reading tile/material/pressure/smoke. Good values are saved to `tools/lighting_presets.toml` and promoted to config by hand. It exists because art-direction iteration through an edit-config-restart loop is too slow; sliders turn "is this value right?" from minutes into seconds.

## Levels and art layers

A level is a `levels/<name>/` folder loaded by `level_loader.py` into a `LevelData`: `tilemap.csv` is the source of truth for the physics grid; `diffuse.png` is required; `normal.png`, the emissive mask/bloom, the background, and a hand-painted wall mask are optional layers the renderer skips when absent. The wall/vacuum masks are derived from the CSV by default (`materials_from_tilemap`). The layered-art ambition — space / exterior / hull / floor / walls / overlays, with normal maps generated from height maps, and destruction revealing lower layers — is the production target for art assets; the shipped renderer consumes whatever layers a level provides.

---

## Implementation status

This audits the renderer against the shipped code in `renderer/` and `shaders/`, not against older plans.

**Implemented and shipped.**

- The full pyray pipeline: window/texture/frame management (`core.py`), the world render target with letterboxing and the Y-flip blit (`world_composite.py`), and the `Camera2D` math object with screen↔world↔tile conversions (`camera.py`).
- The lighting pipeline end to end: per-source cast through the C++ ray engine reading `gmap.dyn_light_atten` (per-channel occlusion, glass/unit shadows), vector-magnitude direction normalization, two **RGBA16F** packed light textures, and the lighting shader with sRGB decode, normal mapping (with Y-flip toggle and `light_z`), Lambertian composite, and **ACES** tone-mapping. The bilinear/point light-field toggle works.
- Overlays with correct premultiplied/additive blend discipline: smoke, fire, the additive **god-ray glow** fed by the ray engine's `smoke_glow` (the retired `light_modulation` surface-tint is gone), and the pressure colormap.
- Units (8-dir marine sprites + zombie variant pool, light-tinted), waypoint/order lines, in-flight projectiles, event-driven tracers/explosions/hit-splats, the debug grid, the debug coord HUD, and the full info panel with timings and toggles.
- The `tools/lighting_demo.py` tuning tool.

**Designed, not yet built.**

- **Desaturation in shadow.** The "darkness reads as colourless, not merely dim" effect (a saturation→grayscale lerp by light level in the shader) is a stated design goal that was lost in an earlier refactor and is not in the current shader. Listed in `docs/TODO.md`.
- **Ambient lighting + two light roles.** A dedicated ambient-floor pass distinct from the directional/raycast lights (reveal geometry vs. flashlights/fires/emergency) is designed but not yet a separate path; today there is one flat `ambient` uniform plus the cast field.
- **Destruction painting layer.** Scorch marks (normal-map dot-product directional burns), blood splats, and floor/rubble over destroyed walls — all on one edit-texture — are fully designed (`graphics_lighting_design.md` §7) but unimplemented. No destruction is painted today.
- **Fire as a short-range light source.** Burning tiles wired into the ray engine as `LightSource` instances; designed (`fire_design_notes.md`), lands with the next physics pass. Today fire is an additive overlay only, not an emitter into the light field.
- **Normal-mapped / shader-lit units.** Units are currently tinted by a scalar `light_map` sample; they do not have normal maps and are not lit by the lighting shader like the ship.
- **LOS-based visibility filtering.** Not drawing enemy units the player has no line-of-sight to. The `gmap.has_los` Bresenham check exists; the renderer-side visibility filter does not.
- **1-bounce reflection** with surface tint (cheap caustics for metal corridors) — noted as secondary priority, not built.

**Gaps and known weaknesses.**

- **The cast lives in the renderer**, not the simulation step. The heat/glow buffers are passed in optionally and the renderer owns the per-frame invocation. Headless runs therefore do not produce the light/heat fields. Moving the cast into the sim is the planned boundary correction.
- **Smoke does not occlude units.** Compose order draws units after smoke, so a unit inside a smoke cloud renders in front of it. Matches the legacy convention but is visually backwards; whether smoke should occlude is an open design question.
- **`world_px_per_tile` is a silent constant (24).** If it does not match `diffuse_width / world_tile_w`, the art is up- or down-scaled into the RT before the camera blit re-scales it, costing sharpness. It is neither computed from the asset nor asserted against it.
- **Window resize is not wired.** The window is treated as fixed-size; `RenderConfig.map_px_w/h` and the camera viewport do not react to a resize. The borderless-windowed option sidesteps this by matching the monitor.
- **`renderer/coords.py` is effectively unused** outside its own naming-discipline role — the overlay helpers take `world_px_per_tile` directly rather than routing through it.
- **Camera niceties** (zoom-around-cursor focal preservation, shake, smooth follow) are not yet on `Camera2D`; the first explosion/effect that wants them will need them added.
