# Architecture Review — Camera + World Render-Target Patch

**Reviewer:** Senior architect pass
**Date:** 2026-05-16
**Scope:** `renderer/camera.py`, `renderer/coords.py`, `renderer/world_composite.py`, `renderer/game_renderer.py`, `renderer/lighting.py`, `shaders/lighting.fs`, `main.py`, `tests/test_renderer_smoke.py`
**Reference docs:** `docs/design_camera_and_coordinate_systems_research.md` (9-step plan), `docs/architecture.md`, `docs/code_review_renderer_v1.md`.

## Verdict

This is the right architectural move executed almost exactly as specified. Steps 1-7 of the 9-step plan are present, the camera lives outside `RenderConfig`, the world RT exists with proper Y-flip handling, and the lighting shader is now structurally bug-free for the "flashlight aligned to wrong tile under scroll" class of bug. The implementation lands the win the research note promised — adding the next physics field (heat, gas, lightning) is now genuinely one extra draw call rather than a per-shader coordinate rewrite. Where the patch falls short is at the seams: a stale `__init__.py` docstring, a scissor inside the compose phase (the research note explicitly flagged this as risky), a `LightingPass.draw_lit_world` call ordering that swaps a known-working bug for a subtler "stale sampler binding" hazard, units / waypoint / grid drawing using a parameter called `ft` while supplied `world_px_per_tile` (the coords.py discipline doesn't reach into `overlays.py`), and a smoke test that doesn't exercise the scroll/blit path it was supposed to verify. None of these are blocking; all of them are cheap to fix before piling on marines and orders. The most important medium-term concern is that compose_world() is going to grow unbounded as entities arrive — splitting it now is cheaper than splitting it later.

## Spec adherence (per-step audit)

**Step 1 — Camera2D, no behaviour change.** Done. `renderer/camera.py:24-84` matches the spec shape: `pos_tile_x/y`, `zoom_px_per_tile`, `viewport_px_w/h`, `world_size_tile_w/h`. `pan`, `set_zoom`, `clamp_to_world`, `visible_world_rect_tile`, `visible_world_rect_world_px`, `world_tile_to_screen_px`, `screen_px_to_world_tile` all present. Camera state correctly moved off `RenderConfig` (game_renderer.py:34-44). One deviation: the spec listed `pos_tile: Vec2` but the implementation uses two scalars (`pos_tile_x/y`); fine, both are equivalent and the two-scalar form avoids a Vec2 dependency. **Spec match: ~95%.**

**Step 2 — coords.py.** File exists (`renderer/coords.py`) with the four named conversions. **But the discipline is not enforced.** Grep across the codebase shows zero imports of any function from `coords.py` outside its own file. `overlays.draw_unit`, `draw_waypoint_line`, `draw_grid` still take a parameter literally named `ft` (overlays.py:87, 99, 108); the callers in `_draw_units_world`, `_draw_orders_world`, `_draw_grid_world` (game_renderer.py:189-211) pass `wpt = self.world.world_px_per_tile` into a parameter called `ft`. The naming-discipline value the research note promised is not yet realized. **Spec match: file present, intent unfulfilled.**

**Step 3 — Allocate the world RT.** Done correctly. `WorldComposite.__init__` (world_composite.py:32-46) sets bilinear + clamp filters as the gotcha section recommended (research note items 5, 7). `unload` uses `unload_render_texture` (item 9). **Spec match: 100%.**

**Step 4 — Diffuse into RT, blit to screen, fix the bug.** Done. `compose_world` (game_renderer.py:145-180) calls `draw_lit_world` first inside `world.begin()/end()`; the lighting shader samples `fragTexCoord` directly because the diffuse covers the full world RT (lighting.py:133-149). Shader (lighting.fs:49-54) has explanatory comments about why `fragTexCoord` works now. **Spec match: 100%.**

**Step 5 — Move smoke + fire into the RT.** Done. `compose_world` (game_renderer.py:166-171) draws smoke at full world bounds, then fire with `BLEND_ADDITIVE`. One minor deviation: `FireOverlay.draw` (overlays.py:77-80) still wraps its own draw in additive blend, while the new orchestrator wraps the call externally — slight double-wrap risk if `FireOverlay.draw` is ever reused. **Spec match: 100% with stale duplicate.**

**Step 6 — Units, waypoints, grid into the RT.** Done. `_draw_units_world`, `_draw_orders_world`, `_draw_grid_world` (game_renderer.py:189-211) draw inside the compose phase. The `ft` -> `world_px_per_tile` rename did not happen in `overlays.py` — see Step 2 above. **Spec match: ~80% (functional, not stylistic).**

**Step 7 — Strip camera UVs from the lighting shader.** Done. `shaders/lighting.fs` has no camera uniforms, no `_loc_view_uv_*` (none ever existed in this codebase), no "KNOWN BUG" comment. `LightingPass.draw_lit_world` (lighting.py:133-149) takes only `(diffuse, normal, world_px_w, world_px_h)`. The architectural class of bug is gone. **Spec match: 100%.**

**Steps 8 and 9 — Post-FX plumbing, multi-camera smoke test.** Not started, as advertised. The architecture supports them but the second RT, the `shaders/post.fs`, and the security-cam inset don't exist yet. **Acceptable — these were marked optional/deferred.**

## Critical

### C1 — Scissor inside compose ignored, but scissor on screen blit is correct
Actually a non-issue — the scissor is on the *final blit* (game_renderer.py:218), not inside compose. Withdrawn. **(See M3 below for the related camera-blit-overdraw nuance.)**

### C2 — `renderer/__init__.py` docstring describes an API that no longer exists
`renderer/__init__.py:1-12` documents `renderer.draw_world()`, `renderer.draw_units(units)`, `renderer.draw_overlays(state)` — none of which exist in the new `GameRenderer`. The real API is `compose_world`, `blit_world_to_screen`, `draw_panel`. A developer reading the package import will be sent the wrong way on minute one. Also, `__all__` still lists only `GameRenderer` despite `RenderConfig` being a public symbol that `main.py:29` and `tests/test_renderer_smoke.py:25` both import from a private path. This is the same `__all__` issue raised as N1 in `code_review_renderer_v1.md` and it survived the patch. Fix: rewrite the docstring against the new flow; add `RenderConfig` to `__all__` and re-export it.

## High

### H1 — `coords.py` is a dead file
Zero imports outside the module. The whole point per the research note (line 101) was naming discipline at the call site. Either:
- (a) Push the rename through `overlays.py` so `draw_unit`, `draw_waypoint_line`, `draw_grid` take `world_px_per_tile` and use the `coords` functions internally, or
- (b) Delete `coords.py` and stop pretending we have the discipline.

Option (a) is correct and cheap (a dozen line changes). Doing neither leaves the codebase with a file that exists only as a documentation artifact, which is worse than not having it.

A lightweight test or lint rule that fails CI on naked `* ft` / `/ ft` arithmetic in `renderer/` modules is a reasonable next-step guard rail.

### H2 — Sampler binding order risk in `draw_lit_world`
`lighting.py:142-143` calls `set_shader_value_texture` *before* `begin_shader_mode` on line 145. The research note item 3 (Gotchas) flagged that pyray's sampler bindings persist across draw calls and can be stale, and explicitly recommended rebinding all samplers at the start of each shader-mode block — meaning *inside* `BeginShaderMode`. The patch documents but doesn't follow its own warning. Today this works because nothing else uses `u_normal` / `u_light` between frames. The moment a post-FX pass arrives (Step 8), or a second shader (e.g. smoke god-rays) gets bound, this becomes a "why is the flashlight pink" bug. Move the `set_shader_value_texture` calls to after `begin_shader_mode`.

### H3 — `compose_world` is going to balloon
The current API takes `units_marines`, `units_zombies`, `orders_per_unit`. The architecture roadmap (architecture.md sections 9-13) adds: projectiles (grenades in flight), shot tracers, particles (debris, sparks, blood), animated sprite atlases, decals, debug HUDs (raycast lines, A* visualization), heat shimmer, lightning bolts, fluid surface, post-FX hooks. If every new entity type extends this method signature, the signature collapses under its own weight inside two months.

Recommend splitting now while the cost is small:
```
renderer.compose_world_begin()
renderer.draw_terrain()      # lit ship + smoke + fire (the static stuff)
renderer.draw_entities(scene)  # units, projectiles, particles
renderer.draw_overlays(scene)  # waypoints, orders, debug HUDs
renderer.draw_grid()         # opt-in
renderer.compose_world_end()
```
or accept a single `Scene` object and let it iterate. Either way, the v1 surface that takes individual `marines`/`zombies`/`orders` collections will need breaking changes within weeks of porting `game.py`.

### H4 — Smoke test does not exercise the camera/RT path
`tests/test_renderer_smoke.py` opens a window, accepts mouse input, runs until close. It is an *interactive demo*, not an automated test. It does not:
- Assert anything (pixel content, frame count without `--auto`, error-free shutdown).
- Pan the camera.
- Verify the flashlight follows the cursor in world coordinates as the camera moves.
- Run headlessly in CI.

The single most valuable test the research note implied (and the prior code review explicitly asked for) is: pan camera to (10, 10), place a light source at world tile (12, 12), render to RT, blit, read back pixel at screen-equivalent position, assert non-black. That would have caught a Y-flip sign error, a clamp_to_world off-by-one, and a `screen_px_to_world_tile` inverse-transform bug in one test.

The current smoke test also imports `os` (unused), `math` (unused), and lives outside any test framework — no pytest discovery, no assertions, no exit code on failure. The print at the end says "OK" regardless of what happened.

## Medium

### M1 — Y-flip math is right but undocumented at the math level
`world_composite.py:74-79` builds the source rect:
```
src = Rectangle(x_wpx, world_px_h - y_wpx, w_wpx, -h_wpx)
```
The Y offset (`world_px_h - y_wpx`) plus negative height is correct, but the comment says only "Y flip: top of viewport is high in world-RT Y space." A reader new to OpenGL RT Y-flip will not derive this from the comment. Add: "Raylib's RT.texture has its origin at the bottom-left. To blit a top-down sub-rectangle: start at the bottom (`H - y`) and draw upward (negative height) to flip back to screen Y-down." Same comment in the `WorldComposite` module docstring is good but could repeat at the call site for the inevitable copy-paste.

### M2 — `mouse_to_tile` does not handle the panel half-pixel
`game_renderer.py:302-310`: the guard is `mx >= self.cfg.map_px_w`. If `mx == map_px_w - 1` the call returns a valid tile; if `mx == map_px_w` it returns `None`. Correct for clicks. But the camera+zoom math `tx = pos_tile_x + mx / zoom_px_per_tile` will yield a sub-tile float that gets `int()`'d (truncated toward zero), giving the wrong tile for negative `pos_tile_x` (impossible today because `clamp_to_world` floors at 0, but if you ever add a margin/offset you'll regret the truncation). Use `math.floor` for forward compatibility.

### M3 — Camera does not yet expose a zoom-around-cursor helper
Mouse-wheel zoom is the imminent feature. The current `set_zoom` re-clamps but doesn't preserve a focal point. The standard idiom is: convert cursor-screen-px to world-tile *before* the zoom change, change zoom, then translate camera so the same world-tile is back under the cursor. That's three lines and very awkward to retrofit if forgotten. Add `Camera2D.zoom_at(zoom_new, focal_screen_px_xy)` now.

### M4 — Camera shake and lerp are reachable but not modeled
Adding a `camera_shake_offset_tile_xy` term and an interpolator (`pan_to_tile_smooth`) won't break the API — but they're not yet there, and someone implementing the first explosion will write them inline in `main.py`. Add stubs:
```
def add_shake(self, magnitude_tile: float, duration_s: float)
def follow(self, target_tile_xy, lerp_alpha: float)
```
These are 20 lines and prevent a future scatter of camera-state code across `main.py`.

### M5 — `compose_world` render order has a real problem with units in smoke
Current order (game_renderer.py:154-178): lit ship -> smoke -> fire -> orders -> units -> grid. Units are drawn *after* smoke. So units appear *in front of* smoke even when they are physically inside a smoke cloud. The expected visual is the opposite: smoke obscures units. The fix is to interleave or to draw smoke twice (once additive-light-only behind units, once attenuating in front of units), but at minimum the current order should swap units and smoke if the design goal is "smoke hides units." (See architecture.md §13 step order for the pygame reference: orders -> projectiles -> units -> shots -> ui. There smoke is below units too, so this matches existing convention — but the conversation about whether smoke should occlude is open, not closed.)

Also relevant: the flashlight-on-a-unit case. Static lights are emitted from configured tile positions; a flashlight on a marine should be a `LightSource` placed at that marine's tile and re-built each frame in `upload_state`. Today the only flashlight is at the mouse cursor (main.py:226-234). When a marine carries a torch the LightSource must be added by something — `upload_state` is the natural home, but it currently takes `light_sources` as an explicit argument from the caller. So the marine-carries-flashlight glue lives in `main.py`. Fine for now; flag for the marine port.

### M6 — `RenderConfig.world_px_per_tile` is a magic 24 with no validation
24 wpx/tile is a reasonable default but the choice is silent. If the level diffuse art is 1000x720, the natural choice is `1000 / level.width`, which for a 50-tile-wide ship is 20, not 24 — meaning the diffuse is upscaled 1.2x into the RT before being downscaled by the camera blit. Visible blur. Either: (a) compute `world_px_per_tile` from the diffuse asset dimensions, or (b) document why 24 is the right pixel-art tile size and assert it matches the asset. Today it's neither.

### M7 — `LightingPass.shader` is unloaded by `GameRenderer.shutdown` (game_renderer.py:316), reaching into another class's private resource
Same pattern as H6 in code_review_renderer_v1.md — the outer class reaches into the inner class to free a GPU handle. Move the unload to `LightingPass.unload()`. Mirror for `smoke_overlay`, `fire_overlay`, and the dynamic light texture. This is the moment to give every renderer subsystem an `unload()` method and have `GameRenderer.shutdown()` iterate.

## Future-proofing concerns

**Multiple render targets (color + emissive + bloom + heat).** `WorldComposite` owns a single RT. The natural extension for bloom is a second RT for emissive, an HDR pass, then a downsample-blur composite. The current `WorldComposite` would need to either (a) become a multi-RT container, (b) be subclassed, or (c) be paralleled by a `PostComposite`. The cleanest pattern is a small `RenderTarget` abstraction (size + format + filter + wrap + Y-flip blit) and have `WorldComposite` own a *list* of them. Today: 1 RT, no abstraction; if heat shimmer or HDR lands first, the class will be rewritten.

**HDR.** `LoadRenderTexture` defaults to RGBA8. Research note item 2 flagged this and noted the `rlgl` path for `R16G16B16A16`. The current `WorldComposite.__init__` doesn't take a `format` parameter — so HDR is one constructor argument plus an `rlgl` call away from being supported. Cost of adding now: 5 minutes. Cost of adding once `WorldComposite` has callers: a refactor.

**Depth buffer for parallax layers.** `load_render_texture` already attaches one. The Y-flip and 2D blits make it useless until someone uses it. Architecture note: if parallax layers arrive (a background starfield through a porthole at depth 1.0, foreground sprites at depth 0.5), the world RT will need depth-write enabled in `compose_world` and the blit will need to respect depth. None of this is in the way today, but the assumption "every compose is order-dependent 2D" should be challenged before it ossifies.

**Multi-camera (security cam, split screen, replay).** The architecture supports this trivially — `world.blit_to_screen(camera_A, ...)` then `world.blit_to_screen(camera_B, ...)`. But `GameRenderer.blit_world_to_screen` hardcodes `self.camera` and the map area's dst rect. Generalize to `blit_view(camera, dst_x, dst_y, dst_w, dst_h)` so the security-cam inset doesn't need a second renderer.

**Pixel-perfect snapping.** Bilinear filter is set on the RT (world_composite.py:42-43). For crisp pixel art there should be a `pixel_perfect: bool` flag that flips the filter to `POINT` and snaps `Camera2D.pos_tile_x/y` to the nearest sub-tile increment that lands on an integer pixel. Reachable from the current API but currently absent.

**Window resize.** `FLAG_WINDOW_RESIZABLE` is set in `core.init_window` (core.py:25) but neither `RenderConfig.map_px_w/h` nor `Camera2D.viewport_px_w/h` reacts to resize. Same issue as H4 in the prior code review; still not addressed. Drop the resize flag or wire `IsWindowResized()` -> update camera viewport + RenderConfig + scissor.

## Things done well

- **The bug class is gone, and gone for the right reason.** Per Step 4: the lighting shader samples `fragTexCoord` because the diffuse covers the full world RT — same UV space as the light texture by construction. The shader file (lighting.fs:49-54) calls this out explicitly. Future renderer developers reading those four lines will understand *why* the old bug can't recur, which is worth more than the bug fix itself.
- **Y-flip is centralized.** Every consumer of the world RT must go through `WorldComposite.blit_to_screen` (world_composite.py:61-84). The negative-height trick lives in exactly one place. The module docstring (lines 17-21) advertises this clearly.
- **`Camera2D` is a clean data object.** No GL state, no pyray imports, pure math, dataclass-friendly, easy to test in isolation (the absent test would be one of the easiest to write). The clamp logic (camera.py:76-84) handles the smaller-than-viewport case correctly.
- **`RenderConfig` is now immutable-ish.** Camera state moved off, docstring states the intent. This closes N4 from the prior code review.
- **The toggle/diagnostic panel matured well.** `draw_panel` (game_renderer.py:227-262) now shows camera pos and zoom alongside the framerate/raycast timings. F1-F5 + B/G/H toggle the right things. The "force-multiplier for debugging" praise from the prior review still applies and is even more relevant now that there's a transform between input and output.
- **`begin_scissor_mode` is on the screen blit, not inside the RT pass.** Research note Gotcha 4 explicitly warned about scissor-inside-RT being glitchy. The implementation respects this — scissor wraps `blit_to_screen` (game_renderer.py:218-223) only.
- **Default ambient color (`0.18, 0.18, 0.22`) is set in two places** (game_renderer.py:101 and 167). The second override in `main.py:167` is fine for the demo scene but should probably be documented as "scene-level override of renderer default." Worth noting that the redundancy exists; not a bug.
- **The `_lookup` wrapper warns on `-1` shader locations** (lighting.py:61-65). This closes M3 from the prior code review. Small thing; correct thing.

---

**Total findings:** 1 critical (docstring + `__all__`), 4 high (dead `coords.py`, sampler bind order, compose_world will balloon, smoke test doesn't test), 7 medium, 5 future-proofing concerns, 7 things done well. The patch is good. Ship it after C2, H1, and H4; address H2 before Step 8.
