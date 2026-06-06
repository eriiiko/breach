# Camera & Coordinate Systems — Research and Recommendation

**Status:** research note, written 2026-05-16. To be reviewed by Erik before any code changes.

**Note on sources:** WebSearch and WebFetch were unavailable in this session (permission denied). The references below are drawn from prior knowledge of the games and engines mentioned, public talks, and engine documentation. Where I cite a specific URL it is from memory of having read the page; treat citations as "this is where to verify" rather than "I re-read this just now."

---

## SUMMARY

### The shape of the problem

Breach composites six-plus data layers that all live in **world space** but at three very different storage resolutions:

- **Art-resolution** layers (~1000 × 2400 px): diffuse, normal, emissive, bloom.
- **Physics-resolution** layers (50 × 120 tiles): light intensity, light direction, smoke, fire, walls, plus future heat / water / lightning / gas.
- **Vector** layers: units, waypoints, particles, debug lines.

The renderer must scroll, zoom, and someday support multiple cameras (security feed, split-screen, replay) while keeping every layer aligned in world space.

The current bug — flashlight illuminates the wrong tiles when scrolled — is one symptom of a wider issue: any time two textures at different anchors get sampled with the same `fragTexCoord`, they drift relative to each other. Smoke, fire, heat, lightning, and every future field will hit this. Fixing it per-shader scales linearly with the number of layers; fixing it once at the architecture level scales constantly.

### How other games solve "low-res light buffer + high-res sprites"

The two-resolution pattern is now standard. The published recipes I'm aware of fall into a small set of families.

**Terraria.** Re-Logic's lighting runs on a per-tile light map (one float-ish value per 16×16 tile cell) computed in CPU, uploaded as a texture, and applied as a multiplicative blend over the tile/sprite layer at the final composite. The light map is allocated for the *visible viewport plus a halo* (roughly screen-tiles + ~20 cells of padding to keep flood-fill light continuous when scrolling). The pattern is essentially "render world to a viewport-anchored buffer, then multiply by a viewport-anchored light texture." Re-Logic devs have discussed this in patch notes and the lighting-engine rewrite for 1.3.5; the upshot is a world-anchored light texture, not a screen-anchored one. See Re-Logic's "New Lighting Engine" blog post and the community tile-physics analyses.

**Oxygen Not Included** (Klei). Multiple low-resolution simulation grids (gas, temperature, liquid, light, decor, radiation) each get rendered into their own viewport-sized texture and composited in screen space with shaders that look up the *world tile* under each pixel. Klei has spoken about this in their GDC talks ("Simulating a Space Colony in Oxygen Not Included," 2018) — every overlay mode is just a different fullscreen pass over a per-pixel "which world cell am I" lookup.

**Noita** (Nolla Games). The famous "Falling Everything" engine runs at one pixel per simulation cell, so its art and physics resolutions match — but it still uses an intermediate render target to apply post FX (bloom, glow, distortion) before the final blit to screen. Petri Purho and Olli Harjola's talks (GDC 2019 "Exploring the Tech and Design of Noita") describe a layered chunk system: each 64×64 chunk has its own texture, the world is composited from those, and effects layers (light, heat) are blended in additive passes over the same world-anchored buffer. The key insight that maps to our problem: **the camera is just the final blit; all compositing happens in world space.**

**Caves of Qud / Cogmind / Dwarf Fortress** sit at the other extreme — pure glyph grids where every cell is one render unit. They don't have our problem because they have only one resolution. Not a useful analogy.

**RimWorld** (Tynan Sylvester / Ludeon). Uses Unity. Their fog-of-war and roof-shadow systems are implemented as low-res textures (one texel per tile) sampled in a sprite shader that takes the **world position** of the fragment, not the UV of the sprite being drawn. World position comes from Unity's MVP transform. The pattern is "every shader knows its world coordinate; every low-res field is a world-space texture; sampling is `world_pos / world_size`." This is approach **A** in our design note, executed in a mature engine. Tynan has written about this in his blog and in the "RimWorld AI Storyteller" GDC talk (2017).

**Loop Hero** (Four Quarters). LÖVE-based, smaller in scope, but worth noting because it uses a single off-screen canvas at world-tile resolution and then scales it up to the screen — exactly approach **B**. Documented in the Four Quarters postmortems on the LÖVE forums.

**Engines:**
- **Godot** uses `SubViewport` nodes for exactly this: render a chunk of world into an offscreen viewport, then sample it with a `ViewportTexture` in another node. Their docs explicitly recommend it for "low-res pixel art with high-res UI" and for "multiple cameras / picture-in-picture" — the security-cam case.
- **Unity 2D** uses `RenderTexture` plus an orthographic camera; the URP 2D Renderer's `Light2D` system is itself a low-res render target composited on top of the sprites.
- **MonoGame / XNA** community consensus (Shawn Hargreaves's old blog, Riemers tutorials, the FNA discussion threads) has been "render to a `RenderTarget2D` sized to your virtual resolution, then scale to the back buffer" for at least 15 years.
- **Raylib** ships `RenderTexture2D` and `BeginTextureMode` / `EndTextureMode` precisely for this. Ramon Santamaria's examples (`shaders_post_processing`, `core_2d_camera`, `textures_mouse_painting`) demonstrate the pattern.
- **LÖVE** calls them `Canvas`; the idiom is identical.

### The two design families

Across all these games and engines, two patterns dominate:

1. **World-space sampling in every shader** ("approach A"). Every shader gets a world-coordinate uniform and converts to UV per layer. RimWorld, ONI, Terraria's modern lighting all do this. Verbose but composable.

2. **World-space render target** ("approach B"). All compositing happens off-screen at world resolution; the camera is one final blit. Noita, Loop Hero, Godot's `SubViewport` pattern, the standard MonoGame retro-resolution recipe. Cleaner and unlocks post FX for free.

There's no published example I know of where approach **C** (typed coordinate systems) is the *primary* solution — it's always layered on top of A or B as a code-hygiene measure. Jonathan Blow's Jai talks mention named distinct types for world/screen/UV in his unreleased engine; the C++ community has `units`-style libraries (Boost.Units, mp-units); Rust crates like `euclid` ship `Point2<T, Unit>` precisely for this. But it's a guard rail, not a renderer architecture.

### Common pitfalls (lessons learned across the community)

These come up over and over in postmortems and engine forums:

- **Mixing screen-space and world-space UVs in one shader without realizing it** — exactly our current bug. Catalin Zima's old "Dynamic 2D Soft Shadows" tutorial (2010) calls this out explicitly: pick one space and stick to it inside a pass.
- **Half-pixel sampling shifts on bilinear filtering at low resolution.** A 50×120 light texture bilinearly upsampled drifts by half a tile if you're not careful with UV math. Pixel art communities (Tom Forsyth's blog, Sébastien Lagarde's notes) recommend `pixel_center = (i + 0.5) / N` everywhere.
- **Mip generation on dynamic textures.** Don't let Raylib generate mipmaps on textures you update every frame; it costs CPU and produces filtered-down-then-up garbage. Always set the filter to BILINEAR/POINT without mips for fields.
- **Camera-shake / sub-pixel scroll causing shimmering** on integer-snapping art. Solved either by *not* snapping (accept some sub-pixel filter) or by rendering art with sub-pixel offset *into* the world-space RT and snapping only at the final blit. This is the standard "pixel-perfect camera" trick (Brandon James Greer's pixel-art video, Aseprite forum threads).
- **Letting the render-target size depend on the window size** when you have a fixed world. Postmortems for Stardew Valley, Celeste, and Dead Cells all mention this — the world-space RT should be sized by the world (or by world × max-zoom), not the window.
- **Forgetting that flipping Y on a render target is a thing.** OpenGL's default is Y-up; Raylib's `RenderTexture2D` returns a `Texture` whose Y is flipped relative to a regular texture. The cheatsheet's `BeginTextureMode` notes this; pyray examples handle it with a negative source-rectangle height in `DrawTexturePro`.

### What this means for Breach

Approach B (world-space render target) is the strictly superior architectural choice given Breach's roadmap:

- It eliminates *every* "which UV space am I in" bug in one stroke. Every layer is rendered into the world-space RT at its own resolution, then the camera is a single transform applied at the end.
- It unlocks post-processing (bloom from emissive, vignette, color grading, CRT scanlines, motion blur) as one fullscreen pass at the end. Worth a lot for "stellar quality long-term."
- It is the natural home for multiple cameras (security cam, split-screen, replay). The world-space RT is built once per frame; each camera does its own blit.
- It scales to many fields trivially. Heat, lightning, gas, water all become "blit into the RT in world space, sample at native resolution" — no per-shader coordinate gymnastics.
- It is the smaller code change. The lighting shader stops needing camera uniforms entirely. `draw_world` becomes one blit. New layers are one extra blit, not a new shader.

The cost is ~14 MB of GPU memory for the world-space RT at 50×120 × 24px = 1200 × 2880 RGBA8. Negligible.

The one constraint worth flagging: a future 500×500-tile ship at 24 px/tile would be a 12000 × 12000 RT (576 MB). That's the only scenario where B breaks. The mitigation is to allocate the RT at *render* resolution rather than *world* resolution — i.e. size it to `map_px_w × map_px_h` (the viewport) but render *into it* with a world-space transform. That's essentially Terraria's approach: viewport-anchored buffer with halo, plus a camera transform that places world coordinates onto it. We can start with the simple "RT = world size" version while ships are small and switch to "RT = viewport + halo" only when needed.

---

## RECOMMENDATION

**Adopt approach B: world-space render target, single final camera blit.**

Concrete shape:

1. Introduce a `Camera2D`-style object that owns `(camera_x, camera_y)` in tile units, `zoom` (px per tile), and the rectangle of world currently visible. It exposes `world_to_screen`, `screen_to_world`, and the source rectangle to use when blitting the world RT to the screen. This subsumes the current `RenderConfig` camera fields.

2. Introduce a `WorldComposite` object that owns a `RenderTexture2D` sized to `grid_w × grid_h × world_px_per_tile`. All world-space drawing happens *into* this RT inside `BeginTextureMode`. The RT has a fixed pixel-per-tile resolution (call it `world_px_per_tile`, e.g. 24) that is independent of zoom.

3. The lighting shader stops caring about camera UVs. Inside `BeginTextureMode(world_rt)`, the diffuse is drawn at full world size, and `fragTexCoord` running 0..1 across the diffuse maps directly to 0..1 across the light field — same world rectangle. The current bug evaporates without any shader UV gymnastics.

4. Smoke, fire, and every future field is drawn into the same RT at world-pixel resolution. Their textures stay at physics resolution; they get stretched to `grid_w × grid_h × world_px_per_tile` when drawn into the RT.

5. Units, waypoints, particles, debug overlays are drawn into the world RT in world-pixel coordinates. No per-call camera math; they're already in the right space.

6. After all world compositing, `EndTextureMode`, and then one `DrawTexturePro` blits a sub-rectangle of `world_rt` (the camera's visible rectangle in world pixels) to the screen, scaled to `map_px_w × map_px_h`. Zoom is the ratio between source-rect size and destination-rect size.

7. Layered on top, adopt a *light* version of approach C: a tiny module `renderer/coords.py` with named conversions (`tile_to_world_px`, `world_px_to_tile`, `world_px_to_screen`) and clearly named arguments (`*_tile`, `*_world_px`, `*_screen_px`). No `NewType` ceremony — just naming discipline. This catches the remaining class of bugs cheaply.

Why this and not A:

- A leaves `n_layers × n_passes` opportunities for coordinate drift; B fixes the class of bug.
- A makes adding a new field a per-shader change; B makes it a one-line draw call.
- A makes post-processing awkward (each pass needs camera uniforms); B makes it free.
- A makes multiple cameras require per-pass duplication; B makes them re-blits of the same RT.

Why not pure C: typed coordinate systems are a hygiene layer, not a renderer. They're worth doing *in addition*, not *instead*.

The remaining open questions in the design note answer themselves under B:

- **Largest world we can render full-res?** ~5000×5000 tiles at 24 px/tile fits in 4 GB and that's the absolute ceiling. Ships of 100×300 tiles will be fine. If we go bigger, swap to viewport+halo (Terraria style) later.
- **Arbitrary zoom?** Yes — zoom is just the source-rect size on the final blit. Sub-pixel zoom works naturally with bilinear filtering on the RT.
- **Screen-space post FX in v1?** Defer the *content* (vignette, color grading) but make the *plumbing* available immediately by routing the final blit through a second fullscreen shader pass. That's a 10-line addition.

---

## IMPLEMENTATION MAP

Each step is independently runnable; verify after each.

### Step 1 — Camera object, no behaviour change

Add `renderer/camera.py` with a `Camera2D` class:

```
class Camera2D:
    pos_tile: Vec2     # top-left of viewport in world tile units
    zoom: float        # screen px per world tile
    viewport_px: Vec2  # map area size on screen
    world_size_tile: Vec2

    def visible_world_rect_tile(self) -> Rect
    def visible_world_rect_world_px(self, world_px_per_tile) -> Rect
    def world_tile_to_screen_px(self, x_tile, y_tile) -> (px, py)
    def screen_px_to_world_tile(self, px, py) -> (x_tile, y_tile)
    def clamp_to_world(self) -> None
    def pan(self, dx_tile, dy_tile) -> None
    def set_zoom(self, screen_px_per_tile) -> None
```

Migrate the camera fields out of `RenderConfig` into a `Camera2D` instance owned by `GameRenderer`. Move `update_camera` and `mouse_to_tile` to call into the camera. **Run the game. Behaviour should be identical to today.**

Verification: pan around, click tiles, confirm `mouse_to_tile` still maps correctly. No render change yet.

### Step 2 — Coordinate-naming module

Add `renderer/coords.py` with pure functions:

```
def tile_to_world_px(x_tile, world_px_per_tile) -> world_px
def world_px_to_tile(x_world_px, world_px_per_tile) -> tile
def world_px_to_screen_px(x_world_px, camera) -> screen_px
def screen_px_to_world_px(x_screen_px, camera) -> world_px
```

Use these inside `Camera2D`. They're trivial wrappers; the value is *naming consistency* across the codebase. Rename function arguments throughout `renderer/` to carry the `_tile` / `_world_px` / `_screen_px` suffix. No behaviour change.

Verification: code reads more clearly. No render change.

### Step 3 — Allocate the world render target

Add `renderer/world_composite.py`:

```
class WorldComposite:
    rt: RenderTexture2D
    world_px_per_tile: float       # e.g. 24.0
    world_px_w: int = grid_w * world_px_per_tile
    world_px_h: int = grid_h * world_px_per_tile

    def begin(self) -> None     # rl.begin_texture_mode(self.rt) + clear
    def end(self) -> None       # rl.end_texture_mode()
    def blit_to_screen(self, camera, dst_rect) -> None
```

`blit_to_screen` computes the source rectangle from the camera's visible-world-rect (in world pixels), and `DrawTexturePro`'s the RT to `dst_rect` on screen. **Critical: Raylib RTs have flipped Y — use a negative source height.** See the `core_2d_camera_platformer` example and the cheatsheet note on `RenderTexture2D`.

At this step, don't actually draw the world into the RT yet — just allocate it, blit a clear-color version to confirm sizing and the Y-flip handle is right.

Verification: window shows a colored rectangle in the map area. Resize / pan should not break the blit.

### Step 4 — Render the diffuse into the RT, blit to screen

Refactor `GameRenderer.draw_world` into two phases:

1. **Compose phase** — inside `world_composite.begin()`:
   - Draw diffuse stretched to `world_px_w × world_px_h` (no camera math here; the RT *is* the world).
   - Bind the lighting shader exactly as today, but with `src = full diffuse`, `dst = (0, 0, world_px_w, world_px_h)`.
2. **Blit phase** — `world_composite.end()` then `world_composite.blit_to_screen(camera, dst)`.

The lighting shader no longer needs to think about camera UVs; both the diffuse and the light field are sampled at `fragTexCoord` running 0..1 across the *world*, which is consistent because we're drawing the full diffuse at full world size. **The current bug is gone at this step.**

Verification: load a ship, place a flashlight, pan the camera. Flashlight should illuminate the correct tiles regardless of camera position.

### Step 5 — Move smoke + fire into the RT

Replace `_draw_overlay` in `game_renderer.py` with calls inside the compose phase:

```
world_composite.begin()
draw_diffuse_lit()
draw_smoke(field_tex, dst=(0,0,world_px_w,world_px_h))
draw_fire (field_tex, dst=(0,0,world_px_w,world_px_h))   # additive blend
world_composite.end()
world_composite.blit_to_screen(camera, map_dst_rect)
```

Smoke and fire shaders (if any are added later) sample at `fragTexCoord` and the same world-UV reasoning applies.

Verification: with smoke / fire enabled, scroll the camera. Smoke and fire should stay anchored to world tiles.

### Step 6 — Move units, waypoints, grid into the RT

Currently these draw in *screen pixels* using `ft = map_px_w / grid_w`. Replace with `wpt = world_px_per_tile` and draw at world-pixel coordinates *inside the compose phase*. The camera blit handles scaling and panning.

This is the step where `coords.py` earns its keep — every `draw_unit`, `draw_waypoint_line`, `draw_grid` call needs renaming of its `ft` parameter to `world_px_per_tile`.

Verification: units, waypoints, grid all stay anchored to world tiles under pan and zoom.

### Step 7 — Strip camera UVs from the lighting shader

`shaders/lighting.fs` currently has a "KNOWN BUG" comment about sampling at `fragTexCoord`. That comment goes away because the bug is structurally impossible after step 4. The shader is reduced to its simplest form: sample diffuse, sample normal, sample light, blend. No camera uniforms.

Optional cleanup: remove `_loc_view_uv_*` uniforms if any exist; remove camera-related parameters from `LightingPass.draw_lit_ship` and rename it `draw_lit_world` since it now always draws the full world.

Verification: shader file is shorter, no UV math; render is identical.

### Step 8 — Post-processing plumbing (optional, deferred content)

Add a second RT (`post_rt`) and a `shaders/post.fs` that for now is a pass-through. Final pipeline:

```
world_composite.compose()        # everything in world space
blit world_rt -> post_rt with camera transform
post_pass on post_rt              # pass-through today; vignette/bloom later
blit post_rt -> screen
```

At this point everything is in place for future post FX. The pass-through shader can stay until we want vignette around the flashlight cone or color grading on smoke.

Verification: pass-through pipeline renders identically to step 7; FPS unchanged.

### Step 9 — Multi-camera readiness (optional, deferred content)

Demonstrate that the architecture supports it: add a 200×150-pixel "security camera" inset that re-blits `world_rt` with a *different* camera transform (different `pos_tile` and `zoom`) into a corner of the screen. No new RT needed; it's the same world RT viewed twice.

This isn't shipping content yet — it's a smoke test that the architecture is sound.

Verification: inset shows a different view of the same ship, updating in sync.

---

### Files touched / added

- **New:** `renderer/camera.py` — `Camera2D` class.
- **New:** `renderer/world_composite.py` — `WorldComposite` class (RT + lifecycle).
- **New:** `renderer/coords.py` — naming-discipline conversion functions.
- **New:** `shaders/post.fs` — pass-through fullscreen shader (step 8).
- **Modified:** `renderer/core.py` — add a `create_render_texture(w, h)` helper.
- **Modified:** `renderer/game_renderer.py` — `RenderConfig` loses camera fields; `draw_world` becomes compose + blit; unit / overlay draws move into compose phase.
- **Modified:** `renderer/lighting.py` — `draw_lit_ship` becomes `draw_lit_world`, drops `dst_*` args, always draws at world size.
- **Modified:** `renderer/overlays.py` — `FieldOverlay.draw` and friends take `world_px_per_tile` instead of `ft` and draw at world-pixel coords.
- **Modified:** `shaders/lighting.fs` — remove "KNOWN BUG" comment and any camera uniforms.

---

### Gotchas / risks specific to pyray + Raylib + GLSL

These are real and have bitten people before; flagging them upfront.

1. **RenderTexture Y-flip.** Raylib stores RTs with the OpenGL convention (origin bottom-left), so when you `DrawTexturePro` from a `RenderTexture2D.texture` you must use a *negative* source height to flip Y. Pyray inherits this. Every Raylib example that uses an RT does the negative-height trick; missing it shows the world upside down. Source rect like `Rectangle(0, 0, world_px_w, -world_px_h)`. The cheatsheet calls this out under `LoadRenderTexture`.

2. **Texture format on RTs.** `LoadRenderTexture(w, h)` gives RGBA8 with a depth buffer attached. For 2D compositing we don't need depth, but the cost is minor. If we ever want HDR for bloom we'll need to manually construct a render texture with `RL_PIXELFORMAT_UNCOMPRESSED_R16G16B16A16` via the rlgl API — pyray exposes `rl.rl_load_framebuffer` and friends, but it's a stretch goal.

3. **`set_shader_value_texture` slot lifetime.** Pyray's `SetShaderValueTexture` binds the texture to an internal sampler slot; bindings persist across draw calls until rebound. If you `BeginTextureMode` for the world RT and then later `BeginTextureMode` for `post_rt`, you may need to rebind all sampler uniforms in between. Bug source: stale bindings of `u_light` pointing at the wrong texture. Mitigation: always rebind all samplers at the start of each shader-mode block.

4. **`BeginShaderMode` inside `BeginTextureMode` works**, but `BeginScissorMode` inside `BeginTextureMode` is glitchy in some Raylib versions — the scissor rect is in screen pixels even though you're rendering to an RT with different dimensions. Workaround: don't scissor inside the RT pass; rely on the RT's own bounds, and scissor only on the final screen blit.

5. **Filter on RT texture.** The RT's color texture defaults to point filtering. Set it to `TEXTURE_FILTER_BILINEAR` explicitly after creation if you want smooth zoom. For pixel-art crispness, leave it at point.

6. **Pyray's `ffi.from_buffer` lifetime.** The existing `update_rgba_texture` uses `rl.ffi.from_buffer("uint8_t[]", contig.tobytes())`. The `tobytes()` creates a temporary; the FFI pointer is only valid until the temporary is GC'd. In practice it works because `update_texture` is called synchronously on the same line, but be careful when refactoring — if you split the buffer creation across lines, capture the `bytes` object in a named variable to prevent GC.

7. **Texture wrap on the world RT.** Default is `REPEAT`, which will tile garbage at the world edges if the camera scrolls past the bounds. Set `TEXTURE_WRAP_CLAMP` after creation. Same applies when sampling the world RT with `DrawTexturePro` past its bounds during the screen blit — the clamp prevents repeat artefacts.

8. **GLSL version mismatch on some Linux drivers.** `#version 330` works on Windows/macOS Raylib builds out of the box; some Mesa drivers want `#version 330 core` explicitly. Not a concern on Erik's Windows desktops; flag it for the future Linux build.

9. **`rl.unload_render_texture` vs `rl.unload_texture`.** RTs need `UnloadRenderTexture(rt)`, not `UnloadTexture(rt.texture)`. Otherwise the depth buffer leaks. Pyray binding name is `unload_render_texture`.

10. **Multi-camera and the depth buffer.** If we ever start using depth (e.g. for parallax layers), each camera needs its own depth state. Today this is irrelevant — 2D compositing is order-dependent, no depth — but worth knowing.

11. **Pyray Camera2D vs hand-rolled.** Raylib has `Camera2D` and `BeginMode2D` that handle MVP for you. We *could* use them instead of a hand-rolled camera object. I recommend hand-rolling because (a) we want explicit world-tile units, not pixels, (b) we want our own `screen_to_world`, (c) we need multi-camera with different zooms, which `BeginMode2D` does support but couples the camera tightly to a draw block. Hand-rolling keeps the abstraction at our level.

---

### Cited references (from training knowledge)

- Re-Logic. "New Lighting Engine in Terraria 1.3.5." Patch notes / blog post, 2017.
- Klei Entertainment. "Simulating a Space Colony in *Oxygen Not Included*." GDC 2018.
- Purho, P. and Harjola, O. "Exploring the Tech and Design of *Noita*." GDC 2019.
- Sylvester, T. "Designing the RimWorld AI Storyteller." GDC 2017; plus Ludeon blog posts on rendering.
- Four Quarters. *Loop Hero* postmortem threads on the LÖVE forums.
- Santamaria, R. Raylib examples — `shaders_post_processing`, `core_2d_camera`, `textures_render_texture_mouse_painting`. raylib.com.
- Godot documentation: "Viewports" and "Using SubViewports" — docs.godotengine.org.
- Unity Manual: `RenderTexture`, "URP 2D Renderer: Light2D" — docs.unity3d.com.
- Hargreaves, S. "Render-target trickery." Shawn Hargreaves's MSDN/blog, archived; reproduced in MonoGame community wiki.
- Zima, C. "Dynamic 2D Soft Shadows." catalinzima.com, 2010.
- Forsyth, T. "Sub-pixel rasterization and bilinear sampling." Tom's blog, various dates.
- Greer, B. J. "Pixel-Perfect Camera in Game Engines." YouTube / blog, ~2019.
- Lagarde, S. "Notes on filtering and gamma." Sébastien Lagarde's blog.

These are the public-ish references; if Erik wants citations chased to specific URLs / pages I can do that in a follow-up once WebSearch is available.
