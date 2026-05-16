# Code Review — Renderer V1

**Reviewer:** Senior engineer pass
**Date:** 2026-05-16
**Scope:** `renderer/` package, `shaders/lighting.fs`, `main.py`, `level_loader.py`, `levels/unhcr_vessel/level.toml`, `cpp/src/raycaster.cpp` directional additions
**Out of scope:** the known camera/world-UV coordinate bug (see `docs/design_camera_and_coordinate_systems.md`).

## Verdict

A well-scoped first pass that does what the patch plan promised: thin pyray surface, a tight orchestrator class, working directional light field, and a clean Lambertian shader. The module split (`core` / `lighting` / `overlays` / `game_renderer`) is the right shape for where this is going. There are real correctness landmines in the dynamic-texture upload path (FFI lifetime, byte-order) and a small pile of resource leaks waiting to bite once levels reload or the window resizes. The shader is approximately right but treats the diffuse as linear despite almost certainly being sRGB, and the normal map convention has not been audited. The overlay/units API will not stretch gracefully to particles, animated sprites, or multi-light cones — those will require a non-trivial refactor. None of this is on fire; all of it should be addressed before piling on the next features.

## Critical

### C1 — `update_rgba_texture` FFI lifetime is fragile (renderer/core.py:117-118)

```python
buf = rl.ffi.from_buffer("uint8_t[]", contig.tobytes())
rl.update_texture(tex, rl.ffi.cast("void *", buf))
```

`contig.tobytes()` allocates a fresh bytes object; `rl.ffi.from_buffer(...)` wraps a pointer into that bytes object; the `cast` returns a new cdata pointing at the same memory. Lifetime is held only by the local `buf` for the duration of the call, so today it works — but it is doing the slow, allocating, copying thing on every frame (tobytes copies the whole array), and the pattern is exactly the kind that breaks silently if anyone ever lifts the call into a helper that returns the cdata. Two fixes:

1. Drop `tobytes()` entirely. `rl.ffi.from_buffer("uint8_t[]", contig)` accepts the numpy array directly (cffi understands the buffer protocol), and zero-copies. The numpy array is contiguous (you assert it) and lives in `self.packed` for the life of the pass.
2. Drop the explicit `cast`. `update_texture` takes a cdata pointer; cffi's argument coercion handles the `uint8_t[]` → `void *` conversion implicitly. The explicit cast is noise.

Net: one less per-frame allocation (180×50×4 = 36 KB plus a copy, three times per frame for smoke/fire/light), and the failure mode goes away.

### C2 — Shader treats sRGB diffuse as linear (shaders/lighting.fs:33, 58)

PNG diffuse art exported from ChatGPT / image editors is sRGB. The shader samples it (`texture(u_diffuse, ...)`) and multiplies by `(ambient + intensity * ndotl)` then writes to the default framebuffer. With no `set_texture_filter`-equivalent linear flag and no `pow(c, 2.2)` decode, you are doing lighting math in gamma space. Result: midtones bias too dark, shadowed colors shift hue, normal-mapped highlights look chalky. The patch plan expert review flagged this and the file comment acknowledges it ("treated linear here") but no fix landed. The cleanest fix is to load the diffuse as an sRGB texture (raylib supports `PIXELFORMAT_UNCOMPRESSED_R8G8B8A8` + a flag, or you decode in the shader: `diffuse = pow(diffuse, vec3(2.2));` after sampling and `finalColor.rgb = pow(lit, vec3(1.0/2.2));` before writing). Pick one, document it, move on.

### C3 — Normal map orientation/encoding unaudited (shaders/lighting.fs:47)

`N = texture(u_normal, fragTexCoord).rgb * 2.0 - 1.0;` assumes (a) the normal map is linear (Laigter typically writes linear, but verify), (b) it uses OpenGL convention (Y up), and (c) the texture sampler is not flipping Y. If Laigter exports DirectX convention (Y down) the lighting will be inverted on horizontal surfaces and you will spend an afternoon thinking the shader is buggy. Add a runtime toggle / a uniform sign flip and check at integration. Same review item flagged in the patch plan; not yet closed.

## High

### H1 — Smoke overlay alpha mixed with light intensity, not modulated by it (renderer/overlays.py:46-50)

```python
mod = 0.15 + 0.85 * mod
self.packed[..., 3] = (v * mod * self.max_alpha).astype(np.uint8)
```

The intent — "smoke only visible where light reaches it" — is correct. But the multiplication happens in alpha, not in color, and the texture is then drawn with `BLEND_ALPHA` (default) over the already-lit ship. The result is: in dark areas, smoke is invisible (good); in lit areas, smoke is fully gray and **occludes** the lit detail beneath it (bad — smoke should *scatter* light, not paint over it). The physically motivated thing is to draw smoke with multiplicative blending against the lit ship (darkens what's behind) plus an additive gray term scaled by `intensity` (scattered light into the eye). For v1, at minimum, lower `max_alpha` significantly or premultiply the gray tint by `mod` so the smoke color in unlit zones is near-black instead of mid-gray with low alpha. Right now bright lit zones look more obscured by smoke than dark zones — opposite of god-rays.

### H2 — `FireOverlay` re-builds RGB every frame even though tint is constant (renderer/overlays.py:67-75)

Minor but visible: every frame, the fire overlay computes `(140 + (255 - 140) * v * 0.5)` for all H×W pixels just to apply a "hotter = whiter" tint that ranges 140-197 (it never reaches white because of the 0.5). At 50×120 that is 6000 ops per channel × 3 channels = 18k ops in numpy; fine. At 100×400 (later) it is 120k ops per frame just in the fire tint. If the tint is supposed to be a gradient from cool-orange to hot-white, do it once in a lookup table indexed by `(v * 255).astype(uint8)` and assign with fancy indexing — or move the tint into the shader (sample fire as a single-channel mask, do the color map in GLSL). The latter is also the future-proof answer for adding heat shimmer.

### H3 — Resource leaks on shutdown path (renderer/game_renderer.py:296-303)

`shutdown()` unloads the shader, the light texture, and the two overlay textures, then calls `core.shutdown()` (which closes the window). It does **not** unload the `self.textures` set via the existing `TextureSet.unload_all()` — wait, it does, at line 297. OK, that's fine. But:

- If `__init__` raises after some textures load (e.g., shader fails late, normal map missing surprises), nothing cleans up. Wrap construction in try/except and unload on failure, or move resource init into a method that owns its rollback.
- `core.shutdown()` is the *last* thing called, but raylib requires textures be unloaded **before** `CloseWindow`. Currently the order is correct only because the explicit unloads precede `core.shutdown()`. The smoke/fire/light textures are unloaded explicitly; the diffuse/normal/emissive go through `TextureSet.unload_all()`. Fine — but it's brittle. Centralize: have `GameRenderer` own a list of `(name, gpu_handle, unload_fn)` and iterate. When you add post-processing render targets, animated sprite atlases, particle textures, etc., you do not want to be patching `shutdown()` in five places.
- `TextureSet.unload_all()` (renderer/core.py:52-59) iterates `_loaded.values()` but never resets the `diffuse`/`normal`/etc. attributes to fresh `Texture` objects on reload — if you ever call `load_level_textures` twice on the same `TextureSet`, you leak. Currently nothing does this, but level switching is on the roadmap.

### H4 — Light field texture stays at construction size; window resize will desync (renderer/core.py:25 + renderer/lighting.py:36)

`FLAG_WINDOW_RESIZABLE` is set in `init_window`. The light texture is 50×120 (or whatever physics res), which is fine — that doesn't change. But the diffuse is sampled using `src` rectangle math built from `cfg.map_px_w / fine_tile_px` (game_renderer.py:137-138). If the user resizes the window, `map_px_w` stays at its initial value (`RenderConfig` is mutated only by camera pan), so half the new window goes unrendered or the map gets clipped. Either drop `FLAG_WINDOW_RESIZABLE` for v1, or hook `IsWindowResized()` + `GetScreenWidth/Height()` and update `cfg.map_px_w/h` each frame.

### H5 — `set_shader_value_texture` called outside `BeginShaderMode` (game_renderer.py:157-161, lighting.py:121-122)

Raylib's `SetShaderValueTexture` records the binding, but the actual GL bind happens inside `BeginShaderMode`/draw. In raylib 4.5+ this works as long as you call it before `BeginShaderMode` on the same shader; in earlier versions and in some pyray builds it can be a no-op if the shader isn't current. Today: probably fine. Future hazard: if you ever have two shaders bound across the frame, the bindings will leak between them. Defensively, call `set_shader_value_texture` *after* `BeginShaderMode` (raylib supports either order; "after" is safer in older versions). Or, more robustly, wrap shader use in a helper that handles bind order for you. This is the kind of bug that surfaces when you upgrade raylib and spend a day bisecting.

### H6 — `draw_world` duplicates the lit-ship draw logic from `LightingPass.draw_lit_ship` (game_renderer.py:144-164 vs lighting.py:110-130)

`LightingPass.draw_lit_ship` exists, has the binding logic, and is dead code: nothing calls it. `draw_world` reimplements the same shader-mode + draw_texture_pro sequence inline, accessing `self.lighting._loc_normal_tex` and `self.lighting._loc_light_tex` (underscore-prefixed; private). Either route `draw_world` through `draw_lit_ship` (passing the source rect), or remove the unused method. Right now: confusing to read, and the API boundary between `LightingPass` and `GameRenderer` is broken in the worst direction — the outer class reaches into the inner one's internals.

## Medium

### M1 — `RenderConfig.fine_tile_px` and `cfg.map_px_w / cfg.grid_w` disagree (game_renderer.py:189, 203)

`draw_units` and `draw_orders` compute `ft = self.cfg.map_px_w / self.cfg.grid_w` (full-map fit), while `draw_world` uses `cfg.fine_tile_px` (zoomed pan). The two will differ as soon as the camera is meaningful. Result: units render at one scale, the ship art renders at another — units float on top of a differently-zoomed background. This is partly the known camera-system bug, but the inconsistency between methods inside the same class is its own smell. Pick one scale (`cfg.fine_tile_px` is the right one) and use it everywhere; or compute a `tiles_to_pixels(fx, fy) -> (px, py)` helper and never write the arithmetic inline.

### M2 — `draw_unit` hardcodes 3×3 footprint offset (renderer/overlays.py:91-92)

```python
cx = (fx + 1.5) * ft
cy = (fy + 1.5) * ft
```

The `+1.5` assumes every unit is a 3×3 marine. Zombies, future projectiles, items, decals all want this to be configurable. Add a `footprint_tiles` parameter and default it to 3. Today the function takes `radius_tiles` but not footprint — those are different concepts and should both be explicit.

### M3 — Shader uniform locations not validated (renderer/lighting.py:45-49)

`get_shader_location` returns `-1` for missing uniforms; raylib accepts -1 in `set_shader_value` as a no-op. Combined with the fallback-default-shader path in `load_shader_with_fallback`, this means a totally broken shader gets you a silent fail — black screen with no log. Either log a warning when any of the five locations come back -1, or fail loud. Same suggestion the architect review made for shader compilation: extend it to uniform lookup.

### M4 — Light field encoding loses precision and direction sign at high res (renderer/lighting.py:101-103)

The expert review in the patch plan recommended a float texture for the light field, explicitly to avoid quantizing direction to 256 angles. The implementation went with RGBA8. At 50×120 with bilinear sampling on a 1000×720 viewport, the quantization shows up as visible banding in the lighting direction across smooth surfaces. The fix is small: `PIXELFORMAT_UNCOMPRESSED_R32G32B32A32` in `create_dynamic_rgba_texture` (parameterize the format), and upload float arrays directly. Pyray supports this. Cost: 4× the upload bandwidth, still negligible at 50×120.

### M5 — `load_shader_with_fallback` cannot detect actual compile failures (renderer/core.py:142-160)

The comment admits this: "raylib doesn't expose a direct compile-error check." Raylib 4.x **does** — `IsShaderReady(shader)` returns false if compile failed. Check it after `load_shader`, log, fall back. Right now a broken shader compiles to id=0 and you get an invisible scene with no warning.

### M6 — `gen_image_color` allocates a CPU image then immediately unloads it (renderer/core.py:99-102)

`create_dynamic_rgba_texture` builds a CPU image, formats it, uploads, frees. The intermediate `Image` is unnecessary — raylib has `LoadTextureFromImage` because it's the GL upload path, but for a blank dynamic texture you can use `GenImageColor` directly (you do) or skip the image entirely with `rlLoadTexture(NULL, w, h, format, 1)` via `rl.rl_load_texture`. Minor; only matters if you ever create render targets in a hot path.

### M7 — `PhysicsRunner.step` returns `destroyed` but `main` ignores it (main.py:118-123 + 212)

The C++ fire step returns destroyed wall coords; nothing consumes them. The renderer should be told so it can spawn smoke puffs, debris particles, or at least invalidate cached light bakes when walls disappear. Hook this up before you forget the return value exists.

## Nice-to-have

### N1 — `__all__` in `renderer/__init__.py` exposes only `GameRenderer` but the docstring example imports `RenderConfig` indirectly (game_renderer.py is imported from main.py for `RenderConfig`). Add `RenderConfig` to the package-level export so `from renderer import RenderConfig` works.

### N2 — `draw_grid` uses `step=3` to draw coarse tiles every 3 fine tiles — but `step=3` is the coarse-tile concept the patch plan says is being retired (see "What this patch does NOT cover"). Document or remove.

### N3 — `lighting.py` imports `math` but never uses it (renderer/lighting.py:7). `overlays.py` and `game_renderer.py` likewise have stale `math` imports.

### N4 — `RenderConfig` is a mutable dataclass and `update_camera` mutates `cfg.camera_x` directly. Works, but pattern is fragile — anyone passing the same `cfg` to two renderers gets surprises. Prefer giving `GameRenderer` its own `camera_x/y` state and let `RenderConfig` stay immutable construction-time settings.

### N5 — `main.py` constructs `LightSource` objects each frame inside the loop (main.py:222-228). For the cursor flashlight that's fine; for the five static lights it's done once outside the loop, good. Note in code that `LightSource` is a value type — anyone seeing this pattern will assume there's state.

### N6 — `level_loader.py` validates that optional asset files exist if declared but never validates image dimensions match between diffuse/normal/etc. A mismatched normal map will silently sample wrong. Add a `PIL.Image.open(path).size` cross-check.

### N7 — `levels/unhcr_vessel/level.toml` has commented-out optional fields. Once the schema solidifies, move these to a documented schema file rather than as comments in every level. Future levels will diverge in subtle ways.

## Future-proofing walls (asked in prompt 7)

Concrete things that will bite when you add features:

- **Multiple lights with cones**: Current `cast_source_directional` accumulates `deposit * (-dx, -dy)` per ray. Two opposing cones at the same tile cancel directionally, leaving `(0,0)` — which the shader interprets as "no light direction," falling back to ambient. The unit Z=0.5 trick masks this somewhat, but the lit ship will have "dead spots" where two lights meet. Plan to switch to per-tile dominant-direction (running max) or to a 2-light deferred-style pass before you ship multiple emergency lights.
- **Particles (explosions, blood, sparks)**: `FieldOverlay` is the wrong abstraction. Particles need per-instance position/rotation/scale + lifetime, not a CPU-side field. Add a `ParticleSystem` class that owns a single `RenderTexture2D` (or batched quads via `rlgl`); do not extend `FieldOverlay`.
- **Animated sprites**: `draw_unit` draws a circle. Replace with a sprite-atlas helper that takes `(texture, frame_index, flip_x)` before you have 12 marines each blinking at different phases. Sprite batching matters once you have 30+ units; raylib's `DrawTexturePro` per-unit will hit pyray FFI overhead quickly. Pre-batch via `rlgl.rl_set_texture` + manual quad emission, or accept a CPU bottleneck up to ~200 sprites.
- **Click-and-drag orders**: `mouse_to_tile()` is fine; the missing piece is a `MouseState` struct (down/up/dragging, start_tile, current_tile). Plan that next to the camera redesign — both touch the same "screen → world" pipeline.
- **Multiple light cones + flashlight on every unit**: 30 units × 1 flashlight each × 100+ rays = 3000+ rays/frame just for unit lights. The existing raycaster handles ~80k iterations/frame (per the patch plan estimate), so you have headroom. But the per-source `march_ray_directional` walks the whole map in worst case; consider a 2x2 light-tile coarse early-out when no `is_wall` blocks a tile, or a tile-binned spatial index. Not yet.

## Performance smells

- **Triple numpy clip+cast per frame in `LightingPass.compute_light_field` (lighting.py:101-103)**: At 50×120 this is microseconds. At 100×400 (a future bigger ship) it's still fine. No action.
- **`np.ascontiguousarray` + `tobytes` in `update_rgba_texture` (core.py:116-117)**: 36 KB copy three times per frame. See C1. Fixing C1 makes this disappear.
- **Per-frame raycaster cost scales linearly with light count × max_range**. The flashlight at `max_range=25` is ~157 rays, each marching up to ~25 tiles = ~4k iterations. Five static lights at `max_range=18` × ~113 rays × ~18 tiles = ~10k each = 50k. Plus flashlight = 54k. Within the patch plan budget.
- **No frame-timing histogram**: `last_frame_ms` is a single sample. You will want a rolling window once perf tuning matters. Easy add.

## Test/validation gaps

- **No headless render smoke test** (architect review promised one). Add a `tests/test_renderer_smoke.py` that initializes the renderer with a fake 8×8 level, draws one frame to an offscreen render target, and asserts pixel(0,0) is not pure black after a light source touches it. Catches "shader compiled but uniforms wrong" failures.
- **No test for the level loader's optional-asset error paths** — what if `normal.png` is declared but missing? `level_loader.py:96-100` raises `ValueError`; verify with a unit test.
- **No test that `update_rgba_texture` actually round-trips a known pattern** — generate a checkerboard, upload, render, read back. This would have caught C1 if it ever broke.
- **No test that the directional raycaster's direction vector points the right way.** Single source, single tile, check `(light_dx, light_dy)` is approximately `(src - tile) / |src - tile|`. Trivial to write, would catch sign flips in future refactors.
- **No fuzz on `mouse_to_tile` near map edges + during camera pan** — off-by-one when `mx == cfg.map_px_w - 1` returns a valid tile, when `mx == cfg.map_px_w` returns None. Probably fine; verify.

## Things done well

- **Module split lands cleanly.** `core` is pure utility, `lighting` owns the light-field GPU resource and shader, `overlays` is stateless function-style + the field-overlay class for stateful textures, `game_renderer` is a thin orchestrator. The architect-review three-module suggestion (`renderer_core`, `renderer_lighting`, `renderer_overlays`) is exactly what landed, with the additional `game_renderer.py` orchestrator on top. Keep this structure.
- **`TextureSet` is the right abstraction for level-owned static textures** (`renderer/core.py:43-59`). The named slots plus the `_loaded` dict-for-cleanup pattern is simple and correct. Generalize this to other resource bundles.
- **`load_shader_with_fallback` exists** (renderer/core.py:142-160). The fallback path isn't perfect (see M5) but the *intent* — never crash the game on a missing shader — is exactly right and rare to see implemented.
- **Directional raycaster math is correct** (`raycaster.cpp:154-160`): light direction stored as "vector toward the source," accumulated weighted, normalized by magnitude afterward. The comment about why it's `-dx, -dy` is the kind of thing future-you will thank present-you for.
- **Vector-magnitude normalization in `normalize_directions`** (raycaster.cpp:235-250) follows the expert review fix — not the patch plan's original (wrong) formulation. Visible learning loop from the review.
- **Toggle keys + on-screen state panel** (`renderer/game_renderer.py:241-254`, `draw_panel` 210-238): the F1-F5 toggles + the panel that shows which are on is a force-multiplier for debugging. Keep adding to this.
- **Frame timing instrumentation** (`game_renderer.py:79-80, 90-115`): `last_frame_ms` and `last_raycast_ms` already separated. Architect review asked for it; it's there.
- **Level loader fails loud** (`level_loader.py`): every missing-required-field path raises `ValueError` with a useful message. The `opt()` helper that raises on "declared but missing" (vs. silent on "not declared") is exactly the right ergonomic for optional assets.
- **`level.toml` reserves `floor_id` and comments-out future fields** (level.toml:21-26): forward-compatible schema work done up-front.
- **`PhysicsRunner.step` does sub-step iteration based on `atmos.max_dt()`** (main.py:101-117): correct CFL handling; one of those things that's easy to skip and painful to add later.
