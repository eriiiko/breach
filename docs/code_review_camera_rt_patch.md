# Code Review — Camera + World-Space RT Patch

**Reviewer:** Senior C++ / graphics programmer pass
**Date:** 2026-05-16
**Scope:** `renderer/camera.py`, `renderer/coords.py`, `renderer/world_composite.py`, `renderer/game_renderer.py`, `renderer/lighting.py`, `shaders/lighting.fs`, `main.py`, `tests/test_renderer_smoke.py`
**Plan reference:** `docs/design_camera_and_coordinate_systems_research.md`

## Verdict

The refactor lands the right shape. World-space RT, single camera blit, lighting shader stripped of camera UVs — this matches the recommendation in the research note and the original flashlight-misalignment bug is structurally impossible now. The module split (`Camera2D` / `WorldComposite` / coords) is clean. However, the patch ships at least two real correctness bugs (mouse-to-tile ignores world coordinates of the camera blit, panel `draw_panel` runs *outside* `BeginDrawing` because `end_frame` already ended it — wait, let me re-verify: `begin_frame` calls `begin_drawing`, then compose happens *inside* `BeginTextureMode` which is nested inside `BeginDrawing`. That part is fine.) There is one genuine logic bug in `_draw_overlay_to_world` for fire/smoke (camera viewport never enters the equation for the overlay — also fine, by design — but stretching a Y-up RT-sourced texture into the world RT will *not* flip Y because the overlay tex is a regular texture, not an RT — fine). The real issues are the panel-area blit gap, the `set_zoom` viewport coupling, the bilinear-on-pixel-art question, and the redundant per-blit scissor. Nothing is on fire, but four items should land before this is locked in.

## Critical (must fix)

### C1 — `mouse_to_tile` is correct only when zoom equals the default fit (`renderer/game_renderer.py:302-310`, `renderer/camera.py:60-63`)

`mouse_to_tile` calls `self.camera.screen_px_to_world_tile(mx, my)`, which computes `pos_tile + spx / zoom_px_per_tile`. Mathematically that's right *if* the camera's blit is anchored to screen `(0, 0)` and fills exactly `viewport_px_w × viewport_px_h`, which is the case today (`blit_world_to_screen` passes `0, 0, map_px_w, map_px_h`). So in the current main loop the math is correct.

The fragility: `Camera2D` does not know where on the screen its viewport is anchored. If a future change ever blits the world RT at a screen offset (security-cam inset is in the plan! see research note step 9), or scales it to a non-default destination rectangle, `mouse_to_tile` returns garbage with no warning. Make the inverse mapping explicit: store `viewport_screen_origin_x/y` on `Camera2D`, or have `mouse_to_tile` route through a shared helper that knows the destination rect. Today: no bug observable. Tomorrow: the multi-camera step lands and clicks register on the wrong tile.

### C2 — Camera viewport never updates when zoom changes (`renderer/camera.py:72-74`)

`set_zoom` mutates `zoom_px_per_tile` and re-clamps, but `viewport_px_w/h` are baked at construction. If the window is later resized — `FLAG_WINDOW_RESIZABLE` is set in `core.init_window` (`renderer/core.py:25`) — and code at some point recomputes `cfg.map_px_w`, nothing propagates to `Camera2D.viewport_px_w`. `visible_tiles()` then returns stale tile counts, the clamp goes wrong, and the final blit shows garbage at the world edge or hides part of the world. Same H4 from the v1 review is back in a new shape. Fix: have `Camera2D` re-read viewport from the renderer config each frame, or expose a `set_viewport(w, h)` that the renderer calls on `IsWindowResized()`. Even without resize support, this is one decision deferred, not avoided.

### C3 — `set_shader_value_texture` binds before `BeginShaderMode`, same hazard as v1 H5 (`renderer/lighting.py:142-145`)

The order today is: `set_shader_value_texture(u_normal)` → `set_shader_value_texture(u_light)` → `BeginShaderMode` → `DrawTexturePro` → `EndShaderMode`. In raylib 4.5+ / pyray 5.5 this works because the sampler bindings are recorded against the shader and flushed at draw time. But:

1. It's the same pattern v1 H5 flagged and it didn't get fixed in the patch — the comment in the research note (point 3, "rebind all samplers at the start of each shader-mode block") was specifically about this.
2. The compose phase will gain a second shader (smoke, fire, post FX) shortly. The moment two shaders share the same sampler slot, you have a stale-binding bug that surfaces as "the smoke shader reads the light texture and looks fine — until you toggle lighting off and it reads garbage." Pyray's slot allocator is internal and not documented to be stable across raylib versions.

Fix: move both `set_shader_value_texture` calls to *after* `BeginShaderMode`. Or wrap shader-mode use in a helper that handles bind order. Either way, do not ship the post-FX step (research note step 8) with this pattern still in place.

## High (should fix soon)

### H1 — Scissor inside `blit_world_to_screen` is redundant and risks an over-cull when the panel renders (`renderer/game_renderer.py:218-223`)

`begin_scissor_mode(0, 0, map_px_w, map_px_h)` wraps a single `DrawTexturePro` whose destination is exactly `(0, 0, map_px_w, map_px_h)`. The scissor cannot cull a single pixel of that blit — the destination rectangle is the scissor rectangle. So the scissor accomplishes nothing today.

The hazard: the research note (gotcha 4) explicitly warns that `BeginScissorMode` inside `BeginTextureMode` is glitchy in some raylib versions. Today the scissor is *outside* the RT pass — fine. But it leaves a pattern that someone (probably future-you) will copy into the compose phase later, and there it *will* misbehave. Drop the scissor here entirely; if you eventually need to clip past the panel for a partial blit, do it then.

A second concern: scissor state in raylib persists until `end_scissor_mode`. The code does end it. Good. But if a future post-FX pass forgets to, the panel render after will be silently clipped to the map area. Belt-and-braces: don't introduce scissor state you don't need.

### H2 — `Camera2D.clamp_to_world` produces an undersize visible-rect blit when the world is smaller than the viewport (`renderer/camera.py:76-84`, `renderer/world_composite.py:61-84`)

When `world_size_tile < visible_tiles`, `max_x` clamps to 0 and `pos_tile_x` is forced to 0 — fine, but `visible_tiles()` still returns `viewport / zoom`, which is *larger than the world*. Then `visible_world_rect_world_px` returns `(0, 0, w_wpx > world_px_w, h_wpx > world_px_h)`. The source rect in `blit_to_screen` ends up sampling past the RT's right and bottom edges. The wrap mode is set to `CLAMP` (`renderer/world_composite.py:45-46`), so what shows is the last column / row of the RT stretched out. That's not garbage but it's not what you want either — the ship looks like it has a smear along its right and bottom edges.

Two reasonable fixes:
1. Clamp the visible-rect *size* to the world size, and adjust the destination rect to match aspect — i.e. letterbox.
2. Clamp the visible-rect size to the world size and stretch the destination rect over the full viewport — accepts non-uniform scaling.

Either way, the camera shouldn't pretend the world is bigger than it is. Today this only bites when zoom is set very small relative to viewport (e.g. `world_size_tile = 50, zoom = 4 spx/tile, viewport = 1000 spx` → asks for 250 tiles of horizontal world). With the default `zoom = 24` and a 50-wide world this is borderline — `viewport_px_w / zoom = 1000/24 = 41.67 tiles`, so it doesn't trigger. But `set_zoom` exists, the user will press `-` someday, and the visible region will pop past the world.

### H3 — `set_zoom` raising the floor to 1.0 spx/tile is too low (`renderer/camera.py:72-74`)

`max(1.0, zoom_px_per_tile)` means a fully-zoomed-out view at `zoom = 1` puts 1000 tiles across the viewport but the world is 50 wide. Combined with H2, that's the worst case. Either pick a floor that prevents `viewport / zoom > world_size`, or implement the letterbox.

### H4 — Bilinear filter on the world RT will fuzz tile boundaries and pixel-art edges when zoom is fractional (`renderer/world_composite.py:42-43`)

The research note (gotcha 5) explicitly says "for pixel-art crispness, leave it at point." The code picks `BILINEAR` for "smooth zoom." For *art-resolution* diffuse this looks fine. For the smoke/fire overlays, which are drawn at physics resolution (50×120) stretched 24× into the RT (1200×2880), bilinear at the *RT-creation* step blurs them only at the screen blit, not at the RT-internal upscale (the RT upscale uses the *smoke/fire texture's* filter — which is also bilinear, set in `core.create_dynamic_rgba_texture`, `renderer/core.py:103`). So you have two bilinear stages stacked. The smoke will look soft-edged regardless of what you do at the RT.

The compromise that pixel-art projects converge on:
- POINT on the diffuse and overlay textures (preserves crisp tile edges).
- BILINEAR on the RT (smooths fractional-zoom panning).
- Snap the camera blit to integer source pixels when zoom is at an integer multiple, fall back to bilinear interpolation off-integer.

For Breach specifically, the art is not really pixel-art (it's painted PSD output), so POINT on the diffuse is probably wrong. But you should *make this choice deliberately per texture*, not by accident. At minimum, document the decision somewhere (in the file or the design doc) so the next person reading this knows it was considered.

### H5 — `RenderConfig.world_px_per_tile` and `world_composite`'s internal `world_px_per_tile` can drift (`renderer/game_renderer.py:74-77`, `renderer/world_composite.py:36`)

`WorldComposite.__init__` takes `world_px_per_tile` as a constructor arg and stores it. `GameRenderer.__init__` also stores `cfg.world_px_per_tile` indirectly via `cfg`. Both are immutable today, but if you ever expose "render at higher world resolution for closer zoom," one source of truth needs to win. Make `WorldComposite` read from `cfg` once and don't store it, or have `cfg` not duplicate it. Today: identical values, no bug. Tomorrow: easy to get wrong.

## Medium (worth doing)

### M1 — `_draw_overlay_to_world` does no Y-flip (and shouldn't), but draws into a Y-flipped framebuffer (`renderer/game_renderer.py:182-187`)

This is correct, but it's a footgun explanation that's missing from the comments. Inside `BeginTextureMode`, raylib flips the projection matrix Y so that drawing at `(0, 0, w, h)` puts pixels in the conceptually-correct place (top-left = top-left). The user-facing experience is "draw normally inside `BeginTextureMode`." The Y-flip only matters when you later *sample* the RT's texture from screen-space. The `_draw_overlay_to_world` code is right, but a one-line comment explaining "no Y-flip needed here — the RT projection handles it; the flip is only on the final blit" would save the next reader 20 minutes.

### M2 — `compose_world` always clears the RT to black, then draws the diffuse over the full world (`renderer/game_renderer.py:154-163`)

`world.begin(clear_color=(0, 0, 0, 255))` followed immediately by drawing the diffuse over `(0, 0, world_px_w, world_px_h)` makes the clear redundant — every pixel is overwritten. The clear costs one fullscreen rasterize at 1200×2880 = 3.5 Mpx per frame. Not free, not catastrophic (~0.1 ms on integrated GPU), but eliminable: only clear when `self.textures.diffuse` is None. Or rely on `clear_background` being a fast fixed-function path and stop worrying.

### M3 — `LightingPass.draw_lit_world` reads `self.shader` after binding sampler textures, but the diffuse is bound *implicitly* by Raylib via `DrawTexturePro` (`renderer/lighting.py:142-148`)

Raylib's convention: the texture passed to `DrawTexturePro` becomes `texture0`, which the shader picks up as `u_diffuse` (the shader's `uniform sampler2D u_diffuse` is bound to slot 0 by the default `loc_map_diffuse`). This works because Raylib's shader-loading path auto-resolves the name `texture0` / `texture1` etc. — but the shader here uses `u_diffuse`, which is *not* the default name. The fact that it works is because raylib also recognizes the prefix `u_` in shader uniform discovery, or because `set_shader_value_texture` is doing the bind for normal and light, and the diffuse falls through to slot 0 by `DrawTexturePro`'s implicit binding to `texture0`.

Verify by toggling normal-map off and confirming the diffuse still appears. If it works today, fine — but document the implicit binding in the shader file. Right now line 18 of `shaders/lighting.fs` says "Pyray sends texture0 as u_diffuse implicitly (Raylib convention)" which is the right comment but the convention is fragile. Future raylib release could enforce stricter name matching.

### M4 — `clamp_to_world` is called from `pan` and `set_zoom`, but not from `__init__` or after a manual `pos_tile_*` assignment (`renderer/camera.py:67-74`, `main.py:158-163`)

`main.py` constructs a `Camera2D` with `pos_tile_y = 20.0` directly. If a level is loaded where `world_size_tile_h < 20 + visible_tiles_h`, the camera starts past the world edge. `clamp_to_world` is never called on the initial state. Fix: clamp in `__post_init__` (dataclass hook).

### M5 — `coords.py` exists but is unused (`renderer/coords.py`, all of it)

The module ships, exports four helpers, and `grep` finds zero imports of it from `renderer/` or `main.py`. The conversion math is duplicated inline in `Camera2D.world_tile_to_screen_px` and `Camera2D.screen_px_to_world_tile`. Either route the camera methods through `coords.world_px_to_screen_px` etc., or delete `coords.py` and admit the camera methods are the canonical conversions. The current state — module exists, none of its functions are called — is dead code that will rot.

### M6 — `RenderConfig.world_px_per_tile` defaults to 24.0 but never gets validated against world size (`renderer/game_renderer.py:44`)

For a 50×120 world at 24 wpx/tile, the RT is 1200×2880, ~14 MB at RGBA8. Fine. For a hypothetical 500×500 ship the RT is 12000×12000, 576 MB, which will blow GPU memory budget on integrated graphics. The research note flags this explicitly ("the only scenario where B breaks"). Add an assert or warning: `assert world_px_w * world_px_h * 4 < 256 * 1024 * 1024, "RT too large; switch to viewport+halo"`. Cheap insurance.

## Nice-to-have

### N1 — `WorldComposite.unload` doesn't reset `self.rt` (`renderer/world_composite.py:88-89`)

After `unload`, `self.rt` is a dangling raylib handle. Calling any method on `WorldComposite` post-unload will crash inside raylib. Set `self.rt = None` after unload, or add an `_unloaded` flag and assert in `begin`/`blit_to_screen`.

### N2 — `Camera2D` is a dataclass with 7 required positional args (`renderer/camera.py:24-32`)

Construction at the call site is verbose and easy to mis-order. `viewport_px_w` and `world_size_tile_w` are both ints; swapping them silently breaks clamping. Either group them (`viewport: Vec2`, `world_size_tile: Vec2`) or `kw_only=True` on the dataclass.

### N3 — `world_composite.py` line 76 comment about Y-flip is correct but reads as English-arithmetic (`renderer/world_composite.py:76-79`)

```python
float(self.world_px_h - y_wpx),   # Y flip: top of viewport is
                                  # high in world-RT Y space
float(w_wpx),
-float(h_wpx),                    # negative height = vertical flip
```

The math is right; the comment "top of viewport is high in world-RT Y space" is the most concise way to say "OpenGL has Y-up, screen has Y-down, RT stores Y-up, we sample with Y-flip." Consider a one-line ASCII diagram in the docstring of `blit_to_screen` showing the four corners. Diagrams age better than prose.

### N4 — `test_renderer_smoke.py` does not exercise the camera (`tests/test_renderer_smoke.py:64-90`)

The test runs the renderer with the default camera (top-left, default zoom) and never pans or zooms. If `clamp_to_world` regresses, the smoke test won't catch it. Add a `renderer.camera.pan(10, 10)` between frames or pan automatically.

### N5 — `tests/test_renderer_smoke.py:50` configures the test viewport at 8 wpx/tile but the renderer config uses `world_px_per_tile=8.0` — and these are independent of the screen viewport (`tests/test_renderer_smoke.py:49-57`)

The test mixes "screen px per tile" (8 = 400/50) with "world RT px per tile" (8.0) at the same value. The default `Camera2D` constructed in `GameRenderer.__init__` picks `zoom = map_px_w / grid_w = 400/50 = 8.0`. So screen px per tile and world px per tile coincide, which makes the camera blit a 1:1 sample. Convenient but accidental. If anyone tweaks `MAP_PX_W` or `world_px_per_tile` independently, the test becomes a fractional-zoom test, which will reveal whatever fractional-zoom bugs exist in `blit_to_screen`. Worth a comment.

## Things done well

- **Clean module boundary.** `Camera2D` owns transforms; `WorldComposite` owns the RT lifecycle; `GameRenderer` orchestrates. The compose-then-blit phase split (`renderer/game_renderer.py:145-180` vs `:215-223`) is exactly the right shape, and matches the research note step-by-step.
- **The shader rewrite is the right shape.** Sampling `u_light` at `fragTexCoord` inside `BeginTextureMode(world_rt)` with the diffuse drawn at full world size makes the light-field UV correct by construction. The "KNOWN BUG" comment is structurally impossible to reintroduce now (`shaders/lighting.fs:49-54`). This is the win the patch was for.
- **Y-flip is encapsulated in one place.** `blit_to_screen` is the only code that has to know about Raylib RT Y-up convention; every other caller draws into the RT with normal coords. That's the correct factoring of a footgun.
- **`unload_render_texture` is the right binding** for releasing both color and depth attachments (`renderer/world_composite.py:89`). The research note flagged this (gotcha 9) and the patch handles it correctly.
- **`set_texture_wrap(CLAMP)` on the RT** (`renderer/world_composite.py:45-46`) is the right defensive choice — without it, sampling past world bounds would tile garbage. Good catch from the research note (gotcha 7).
- **`compose_world` keyword-argument structure** (`renderer/game_renderer.py:145-180`) makes it obvious what's being drawn into the world. The unit/order/grid draws all happening in the compose phase means they automatically inherit camera scaling and panning — exactly the win the world-space RT was supposed to deliver.
- **`mouse_to_tile` correctly returns `None`** when the cursor is over the panel (`renderer/game_renderer.py:307-308`). Small detail; easy to forget.
- **`RenderConfig` made immutable and camera state moved off it** (`renderer/game_renderer.py:34-44`). Addresses the v1 review M1 (`fine_tile_px` vs `map_px_w / grid_w` disagreement) by removing the duplication entirely. Good cleanup.
- **`draw_lit_world` rename and signature reduction** (`renderer/lighting.py:133-149`). The old `draw_lit_ship` had `dst_*` args that varied per call; the new version always draws at world size and the camera handles the rest. The shader and the lighting class are both simpler.
