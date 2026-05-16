# Patch Plan: Level Pipeline V1

**Date:** 2026-05-16
**Goal:** First real level loaded end-to-end, with proper graphics and the existing physics, so we can return focus to physics development.

---

## Decisions locked in this session

1. **Rendering: pyray** (Python bindings to Raylib C library). Single-language game code; rendering, physics, and game logic all callable from Python.
2. **Physics + raycaster stay in C++** (already exist as `breach_physics.pyd`).
3. **Game logic stays in Python** (`game.py`, ~2600 lines). No port to C++ until subsystems stabilize.
4. **Pygame retired.** All draw calls move to pyray.
5. **Lighting: low-resolution light field, per-pixel response via normal maps.** The raycaster outputs intensity + direction at physics tile resolution; the GPU shader combines that with the per-pixel normal map to produce smooth, directional lighting across the high-resolution art.
6. **Bilinear interpolation of light field: togglable.** Crisp/blocky vs smooth, for visual comparison.
7. **Reflections deferred** to v2.
8. **Optional art layers** (emissive mask, bloom) — if a level doesn't have them, the renderer skips that pass.
9. **Canonical level format:** `levels/<level_name>/` folder with `tilemap.csv` as source of truth and individual asset files inside (`diffuse.png`, `normal.png`, etc.) — no level name redundancy in filenames.
10. **Wall mask derived from CSV** by default; can be overridden with a hand-painted `wall_mask.png` if needed.
11. **Tile size remains 1/3 m** for now. Will be tuned later if needed.
12. **Fluid sim** clarification added to architecture.md: experimental "cheap high-res trick" runs pipe model at physics resolution, uses heightmap for pixel-level visual flooding.

---

## Architecture: how it fits together

```
                  ┌────────────────────────────┐
                  │     game.py  (Python)      │
                  │   game logic, orders,      │
                  │   turn system, input       │
                  └──────┬─────────────┬───────┘
                         │             │
                         │ (pybind11)  │ (pyray)
                         │             │
                  ┌──────▼─────┐ ┌─────▼──────┐
                  │ breach_    │ │  Raylib    │
                  │ physics.pyd│ │  (C lib)   │
                  │            │ │            │
                  │ AtmosSolver│ │ DrawTex,   │
                  │ Smoke      │ │ Shaders,   │
                  │ Fire       │ │ BlendModes,│
                  │ Raycaster  │ │ Textures   │
                  └────────────┘ └────────────┘
```

The Python game tick:
1. Read input
2. Apply orders → unit state changes
3. Call physics step (C++)
4. Call raycaster (C++ via physics module) — produces `light_intensity[h][w]` and `light_direction[h][w]`
5. Render (pyray):
   a. Upload textures (diffuse, normal, light field) — once or as they change
   b. Draw with shader: dark base layer revealed by light field, modulated by normal map
   c. Draw overlays (smoke, fire, emissive, units)
6. Swap buffers

---

## File-by-file changes

### NEW files

| Path | Purpose |
|------|---------|
| `levels/unhcr_vessel/tilemap.csv` | Physics tile grid, 120×50, source of truth |
| `levels/unhcr_vessel/diffuse.png` | Art image (renamed from ChatGPT export) |
| `levels/unhcr_vessel/normal.png` | Normal map (renamed from Laigter `_n.png`) |
| `levels/unhcr_vessel/level.toml` | Metadata: name, tile_size_m, optional asset paths |
| `level_loader.py` | Reads level.toml + tilemap.csv, returns level data |
| `renderer.py` | All pyray draw calls (replaces pygame in game.py) |
| `shaders/lighting.fs` | GLSL fragment shader: diffuse × light × (normal · light_dir) |

### MODIFIED files

| Path | Change |
|------|--------|
| `game.py` | Remove pygame imports/draw calls. Call `renderer.draw_frame(state)`. Replace `_load_from_csv` with `level_loader.load(level_name)`. |
| `config.toml` | Remove hardcoded `tilemap_csv` path. Add `level = "unhcr_vessel"`. |
| `config.py` | Load level name from config, pass to loader. |

### MOVED/RETIRED files

| Path | Action |
|------|--------|
| `prototypes/space_ship_gpt_pipeline1/` | Stays as source/build folder. The CSV and image originate here. |
| `prototypes/raylib_cpp/flashlight.cpp` | Keep as reference for the rendering passes we'll re-implement in pyray. |
| `prototypes/fluid_test.py`, `fluid_sandbox.py`, `fluid_scenarios.py`, `fluid_tilted_ship.py` | **Move** to `prototypes/archive/` (dead-end exploration, kept for reference). |
| `prototypes/generate_*.py`, `generate_with_segmentation.py`, `make_normal_map.py`, `ship_to_level.py`, `render_test.py`, `raylib_test.py` | **Move** to `prototypes/archive/`. |
| `prototypes/diagonal_cost_circles.py`, `shockwave_viz.py`, `wind_test*.py`, `explosion_sim.py`, `smoke_sim.py` | Keep — useful physics references. |

---

## Step-by-step execution plan

### Step 1: Set up the levels folder (15 min)

- Create `levels/unhcr_vessel/`
- Copy `tilemap.csv` from prototype folder
- Copy ChatGPT image as `diffuse.png`
- Copy Laigter `_n.png` as `normal.png`
- Write `level.toml`:
  ```toml
  name = "UNHCR Vessel"
  tilemap = "tilemap.csv"
  diffuse = "diffuse.png"
  normal = "normal.png"          # optional
  emissive_mask = "emissive_mask.png"  # optional, skip if missing
  emissive_bloom = "emissive_bloom.png" # optional, skip if missing
  wall_mask = "wall_mask.png"    # optional, derived from CSV if missing
  tile_size_m = 0.333
  ```

**Checkpoint:** Just file moves. Game still runs with old pygame renderer.

---

### Step 2: Write `level_loader.py` (30 min)

Reads `level.toml`, validates files, returns a `LevelData` object with:
- `tilemap: np.ndarray` (the CSV)
- `diffuse_path: Path`
- `normal_path: Path | None`
- `emissive_mask_path: Path | None`
- `bloom_path: Path | None`
- `wall_mask_path: Path | None`
- `tile_size_m: float`

Replace `game.py`'s `_load_from_csv` to use this loader. Game still renders with pygame (no rendering changes yet).

**Checkpoint:** Game loads from `levels/unhcr_vessel/`. Old gameplay still works.

---

### Step 3: Install pyray, write `renderer.py` skeleton (1 hour)

- `pip install raylib` (provides `pyray`)
- Create `renderer.py`:
  - `init(width, height, title)` — sets up pyray window
  - `load_textures(level_data)` — loads diffuse, normal, etc. once
  - `draw_frame(state)` — called every frame; for now, just draws diffuse texture full-screen
  - `shutdown()` — cleanup

In `game.py`, replace pygame init/loop with renderer calls. Strip out all pygame drawing code into a `legacy_pygame_renderer.py` (kept for reference, not called).

**Checkpoint:** Game opens a Raylib window. Diffuse texture visible. No lighting yet, no units, no overlays — just the ship art.

---

### Step 4: Light field from raycaster (1-2 hours)

**Raycaster status:** Currently outputs intensity only (`light_map[h][w]`). We need to extend it to also output direction.

**The extension (~15 lines of C++):**
- Add `light_dx[h][w]` and `light_dy[h][w]` fields
- In `march_ray`, deposit `(intensity * dist_atten, intensity * dist_atten * cos(angle), intensity * dist_atten * sin(angle))` into the three fields per visited tile
- After all rays cast, normalize: `(light_dx[i], light_dy[i]) /= max(light_map[i], epsilon)` so direction is a unit-ish vector
- Update Python bindings (`bindings.cpp`) to expose the new outputs

**How rays accumulate:**
Each tile collects contributions from all rays passing through. Direction is a weighted average — multiple rays from different angles produce a "dominant" direction (or near-zero if they cancel). Smoke and distance both reduce the intensity weight as the ray travels.

**Existing falloff and smoke absorption stay:**
- Distance: quadratic-ish (`1/(1 + d² × 0.01)`)
- Smoke: ray loses fraction proportional to local smoke density
- Walls: hard stop

**Texture upload:**
- Pack as one RGBA texture (50×120): R=intensity, G=dir_x (normalized 0-1), B=dir_y (normalized 0-1), A=unused/reserved
- Upload every frame after raycaster runs

**Frame budget:** Run raycaster every frame for v1. Optimization (cached static lights, dirty regions, frame-skipping for explosions) deferred to v2.

**Checkpoint:** Light field is computed and uploaded. Not yet visible (no shader yet).

---

### Step 5: Lighting shader (2-3 hours)

Write `shaders/lighting.fs` (GLSL):

```glsl
uniform sampler2D u_diffuse;
uniform sampler2D u_normal;
uniform sampler2D u_light_field;   // R = intensity, G/B = direction (encoded 0-1)
uniform vec3 u_ambient;
uniform bool u_bilinear_light;     // toggle

void main() {
    vec3 diffuse = texture(u_diffuse, uv).rgb;
    vec3 normal = texture(u_normal, uv).rgb * 2.0 - 1.0;

    vec4 light_sample = u_bilinear_light
        ? texture(u_light_field, uv)            // bilinear (default Raylib)
        : texelFetch(u_light_field, ivec2(uv * light_size), 0);  // nearest

    float intensity = light_sample.r;
    vec2 light_dir_2d = light_sample.gb * 2.0 - 1.0;
    vec3 light_dir = normalize(vec3(light_dir_2d, 0.5));

    float ndotl = max(dot(normal, light_dir), 0.0);
    vec3 lit = diffuse * (u_ambient + intensity * ndotl);

    gl_FragColor = vec4(lit, 1.0);
}
```

Bind in `renderer.py`, draw textured quad with shader. Toggle key for `u_bilinear_light`.

**Checkpoint:** Lighting visible. Move mouse → flashlight follows → walls cast shadows → normal map adds per-pixel detail.

---

### Step 6: Re-add game overlays (1-2 hours)

In `renderer.py`, port from the flashlight demo's approach:
- Smoke layer (gray semitransparent, sampled from `gmap.smoke`)
- Fire layer (orange glow, sampled from `gmap.fire`)
- Atmosphere overlay (debug pressure colormap, togglable)
- Unit sprites (colored rectangles per unit, with orientation)
- Order placement UI (waypoints, throw arc, etc.)
- Bullet tracers, explosions

Most of these are simple `draw_texture` or `draw_rectangle` calls. The diffuse + lighting is the hard part; everything else is straightforward.

**Checkpoint:** Game is fully playable in Raylib. Smoke, fire, units, orders all visible.

---

### Step 6.5: Smoke + light interaction (already in the raycaster — just visible now)

Smoke absorbs ray intensity (`smoke_absorption = 0.8`). When we render the lit scene with smoke in the air, we get:
- **God-rays** through doorways with smoke in them
- **Dim shadows** behind smoke clouds
- **Explosions** create transient dark zones until smoke clears

No extra code needed — this falls out for free once Steps 4 and 5 are done. Worth a screenshot/celebration when we first see it.

### Step 7: Optional emissive + bloom (1 hour, if assets exist)

For the UNHCR vessel, we don't have emissive/bloom yet. Skip for now. But the renderer should support them when assets are present:
- Emissive: drawn additively on top of lit ship
- Bloom: blurred halos drawn additively on top of emissive

When you later paint these for unhcr_vessel, the renderer picks them up automatically (level.toml lists them).

**Checkpoint:** Full pipeline ready. Emissive/bloom are optional add-ons per level.

---

### Step 8: Cleanup (30 min)

- Delete `legacy_pygame_renderer.py` once we're satisfied
- Move dead-end prototypes to `prototypes/archive/`
- Update README / dev_setup.md to mention pyray dependency
- Commit and push

**Checkpoint:** Clean repo, single canonical rendering path, one working level.

---

## Expert review feedback

### C++/Graphics reviewer (Sonnet, 2026-05-16)

**Critical fix — direction normalization (Step 4):**
The plan's normalization `(light_dx, light_dy) /= max(light_map, epsilon)` is **wrong**. Opposing rays would cancel direction to (0,0) at high-intensity tiles. Correct approach:

```cpp
float len = sqrt(light_dx[i]*light_dx[i] + light_dy[i]*light_dy[i]) + epsilon;
light_dx[i] /= len;
light_dy[i] /= len;
```

Normalize by vector magnitude, not intensity. Or alternatively skip normalization entirely and let the shader's `normalize()` handle it. At pure-cancel tiles (ambient-only), fall back to a default direction (e.g., upward).

**Texture format:**
Use floating-point texture (`R32G32B32A32` or `R32F`) instead of 8-bit RGBA8 to avoid quantizing direction to 256 discrete angles. The GPU overhead is negligible.

**GLSL shader refinements:**
- Light direction Z-component (`vec3(dir, 0.5)`) is a placeholder. For v1 it's fine. Future: compute from tile distance, or scale by intensity (brighter → steeper angle).
- Lighting model `diffuse * (ambient + intensity * ndotl)` is standard Lambertian — keep it simple for v1.
- Ambient term should be dim (~0.15 grey) so scene isn't fully lit without light sources.

**Color space / gamma:**
- Confirm Laigter normal map output is linear (not sRGB). If sRGB, pre-decode before unpacking.
- Confirm diffuse texture is sRGB (standard for color art). The shader treats it as linear — if it's sRGB the final image will be too dark; need explicit decode.

**Existing raycaster (`cpp/src/raycaster.cpp`) — verified sound:**
- DDA march correct, wall/smoke handling correct
- Per-frame cost estimated reasonable for v1 (~40-80k ray-march iterations per frame)
- Measure before optimizing

**Pyray overhead:**
- Single texture per layer + one draw call: negligible
- `UpdateTexture()` on mutable texture each frame is fine
- Shader uniforms must be set before drawing

**Decisions applied to plan:**
1. Direction normalization → fix to vector-magnitude normalization in raycaster
2. Texture format → use float texture for light field
3. Z-component → keep at 0.5 placeholder for v1, note for v2
4. sRGB/gamma → audit at integration time, fix if needed

### Software architect reviewer (Sonnet, 2026-05-16)

**Approved with revisions.** Key changes adopted:

1. **Split renderer into three modules** instead of single file:
   - `renderer_core.py` — window, texture loading, frame setup
   - `renderer_lighting.py` — light field upload, shader, lit-scene draw
   - `renderer_overlays.py` — smoke, fire, units, orders, debug HUDs
   - `renderer.py` — thin orchestrator that calls the above

2. **Introduce `GameRenderer` interface** between game.py and rendering. Keeps game.py from accumulating render concerns:
   ```python
   renderer = GameRenderer(level_data)
   renderer.upload_state(gmap)       # light, smoke, fire
   renderer.draw_frame(units, orders) # composites everything
   ```

3. **Aspect ratio**: deferral concern is valid. Mitigation: separate agent is solving alignment **in parallel** (see `tools/align_ship_art.py`). If alignment succeeds, use the aligned diffuse/normal. If it fails, accept stretch for v1 prototype. Either way, no manual work in this patch.

4. **Level loader validation**: add to spec
   - Validate CSV shape matches expected
   - Validate required textures exist (diffuse is required; others optional)
   - Add `version = "1"` field to level.toml for future schema migrations
   - Fail fast with clear errors

5. **Multi-floor / teleporter**: defer to v2, but reserve schema space in level.toml:
   ```toml
   floor_id = 0                    # for multi-floor v2
   # teleporters = [...]            # for v2
   ```

6. **Keep `prototypes/raylib_cpp/flashlight.cpp`** as reference (rename folder or add README). Don't delete — it's the C++ ground truth when pyray renderer behavior is unclear.

7. **Add to success criteria**:
   - Grid overlay toggle (F3) for alignment debugging
   - Frame stepping (single-tick advance) for debugging
   - Game state preserves after switching to new renderer (no behavioral regression)

8. **Performance budget**: instrument render and raycaster timings. Set v1 budget: `raycaster < 10ms`, `render < 8ms`, total tick < 16ms (60 FPS target). Log per-frame.

9. **Shader compilation error handling**: wrap shader load in try/except. Fall back to a trivial flat-color shader so the game still runs if GLSL fails.

10. **Hot-reload F5**: extend existing config hot-reload to also reload level assets (textures, CSV). Useful during level iteration.

**Validation tests added to plan:**
- `tests/test_level_loader.py`: load each `levels/*/`, verify CSV + texture parsing, no exceptions
- Smoke test on each `python game.py` startup: render one frame headless, save to `/tmp/last_frame.png`, no crash = pass

---

## Open questions — resolved

1. **Does the C++ raycaster currently output light direction?** **No.** It only outputs intensity. We will extend it in Step 4 (~15 lines of C++). Approach: each ray deposits `(intensity, intensity*cos(angle), intensity*sin(angle))` per visited tile. Normalized after all rays cast.
2. **Aspect-ratio mismatch (972×1619 image vs 50×120 tilemap):** **Defer.** Accept stretching for v1 prototype. Erik will manually align the diffuse to tilemap bounds later (or we automate it post-patch). The misalignment is not blocking — gameplay tests on the physics grid, rendering just visualizes it.
3. **Frame budget for lighting:** **Defer.** Run raycaster every frame in v1. v2 of raycaster will address cached static lights, dirty regions, per-source frame budgets.
4. **Coordinate convention:** Pyray uses top-left origin (Y down) — same as pygame. No flip needed.
5. **Level loader CSV validation:** Yes — verify CSV shape matches game grid, fail fast on mismatch.

---

## What this patch does NOT cover

- Removing the coarse-tile concept from `game.py` (~66 references). Some are leftover, some are needed for unit footprints. Separate dedicated cleanup later.
- Full normal-map quality tuning (stronger normals, tilted wings).
- Heat radiation, weapons with raycaster (already in architecture, separate feature work).
- Multi-floor ships, level teleporters.
- Level editor tool.
- Performance optimization (only address if we hit a bottleneck).

---

## Success criteria

When done:
1. `python game.py` opens a Raylib window with the unhcr_vessel level loaded
2. The ChatGPT spaceship art is visible, lit by a flashlight following the cursor
3. Walls cast proper shadows from the raycaster
4. Normal map gives per-pixel surface detail
5. Smoke, fire, units, orders all render correctly
6. Pressing `B` (bilinear toggle) switches between crisp and smooth light
7. Pygame is completely gone from the codebase

Total estimated effort: **8-12 hours**, doable in 2-3 sessions.
