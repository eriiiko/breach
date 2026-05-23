# Patch Plan — Lighting Demo Tool

> **Status:** Plan locked, implementation pending.
> **Drafted:** 2026-05-24.
> **Author intent:** Erik. Execution: implementation agent (sonnet, background).
> **Goal:** A standalone tuning tool — `tools/lighting_demo.py` — where Erik can
> dial in lighting / smoke / grenade values with live sliders while a real
> physics sim runs, and persist good values to a preset file.

## 1. Why this tool

Visual parameters in Breach (ambient, light height, smoke tint + alpha, grenade pressure/smoke output, etc.) currently require an edit-config / restart loop to tweak. That kills the iteration speed for art-direction work. The lighting demo replaces that loop with **live sliders + a save button**, so the inner loop of "is this value right?" goes from minutes to seconds.

It's a **tool**, not gameplay. Lives alongside the main game; reuses the level loader, simulation, renderer, lighting pass; orchestrates differently. Erik will use it across multiple sessions for base lighting, then for explosion visuals, flame settings, smoke settings, etc.

## 2. Locked decisions (Erik, 2026-05-23/24)

These are NOT open for re-deliberation. The agent follows them.

1. **Language: Python.** Standalone script at `tools/lighting_demo.py`.
2. **Reuse infrastructure.** Loads UNHCR Vessel via existing `level_loader`. Builds a `Simulation` (so physics ticks for smoke/wind/pressure behaviour). Builds a `GameRenderer` (so the lighting pipeline is identical to the game). Spawns the 7 level.toml units so we see normal-mapped shapes lit; **AI doesn't tick** (zombies stand still).
3. **Only one light source: the mouse flashlight.** No static emergency lights. (Static lights stay in `main.py` for the actual game — only the demo skips them.)
4. **GUI library: raygui** (built into raylib, exposed by pyray as `rl.gui_*`). No new dependencies.
5. **Save destination: separate file** at `tools/lighting_presets.toml` (Option b). Doesn't touch `config.toml`. Erik manually promotes values to config when satisfied.
6. **Visual-only sliders.** Anything that affects sim numerical stability (smoke diffusion D, wave c, fire spread rate, etc.) is OUT. Config-only.
7. **Grenade-spawn-on-click is a first-class feature** of the tool, with its own sliders for tuning blast/smoke/fuse.
8. **Pressure colormap overlay** (toggleable) — port the existing colormap from `config.toml [rendering] pressure_stops` and the legacy code at `game.py:2095-2149`. Used for visualizing the explosion's pressure field.
9. **Cursor-hover readouts** — always-on HUD shows tile coords + pressure + smoke density at the mouse position. (Erik's request 2026-05-24.)

## 3. File layout

- **New file:** `tools/lighting_demo.py` — the standalone script.
- **New file:** `tools/lighting_presets.toml` — created on first Save click. Initial contents = empty TOML; the Save button writes named presets into sections.
- **No changes to existing modules.** All hooks already exist:
  - `LightingPass.set_ambient`, `set_normal_strength`, `set_use_normal`, `set_srgb_decode`, `set_light_z` — all already public setters on the lighting pass
  - `FieldOverlay` has `self.tint_r/g/b` and `self.max_alpha` — directly mutable
  - `simulation.physics.apply_explosion` + `add_explosion_smoke` — already importable

## 4. Sliders panel (right side, ~340 px wide)

Sections, top to bottom:

### 4.1 Ambient
- **Ambient R** slider 0.0..1.0 (default 0.10)
- **Ambient G** slider 0.0..1.0 (default 0.10)
- **Ambient B** slider 0.0..1.0 (default 0.13)
- Calls `renderer.lighting.set_ambient((r, g, b))` on change

### 4.2 Lighting
- **Light Z** slider 0.0..1.5 (default 0.5) — calls `set_light_z`
- **Normal strength** slider 0.0..2.0 (default 1.0) — calls `set_normal_strength`
- **Use normal map** checkbox (default ON) — calls `set_use_normal`
- **sRGB decode** checkbox (default ON) — calls `set_srgb_decode`

### 4.3 Mouse flashlight
- **Max range** slider 5..40 (default 25)
- **Intensity** slider 0.0..5.0 (default 2.5)
- **Angle spread** slider 0..6.283 (default 6.283 = full circle, lower = cone)
- These values rebuild the `bp.LightSource` each frame at mouse position — straightforward inline

### 4.4 Smoke overlay
- **Tint R** slider 0..255 (default 190)
- **Tint G** slider 0..255 (default 195)
- **Tint B** slider 0..255 (default 210)
- **Max alpha** slider 0..255 (default 180)
- Mutate `renderer.smoke_overlay.tint_r/g/b` and `.max_alpha` directly each frame

### 4.5 Pressure overlay
- **Show pressure** checkbox (default OFF) — toggles whether the pressure colormap overlay is drawn
- **Pressure scale** slider 0.5..10.0 (default 2.0; matches `CFG.rendering.pressure_scale`)
- Optional v2: in-tool editor for colormap stops (deferred — for now read stops from `config.toml` at startup)

### 4.6 Grenade tuning (used by click-to-spawn)
- **Blast radius** slider 1..15 (default 6)
- **Pressure (blast)** slider 1..30 (default 10)
- **Wall damage** slider 0..1000 (default 200)
- **Unit damage** slider 0..200 (default 60)
- **Fuse seconds** slider 0.0..5.0 (default 0.0 — detonate on placement)
- **Smoke amount** slider 0.0..2.0 (default 1.0) — multiplier on `add_explosion_smoke` deposit

### 4.7 Save / Load
- **Save preset** button + text input for preset name (default "default")
- **Load preset** dropdown listing presets in `lighting_presets.toml`
- **Reset to defaults** button

## 5. Click-to-spawn grenade

- **Spawn grenade mode** toggle (button OR a key, e.g. `G`)
- When ON, mouse cursor shows a small crosshair
- Left-click on a map tile → spawn an explosion immediately at that tile, parameterised by the current §4.6 sliders. This calls `simulation.physics.apply_explosion(...)` directly (same code path as the game's grenade detonation) plus `simulation.physics.add_explosion_smoke(...)` scaled by the smoke-amount slider
- After click, mode stays ON until toggled off — multiple grenades per session

## 6. HUD readouts (top-left, always visible)

Yellow text on translucent black background, mirroring the F6 coords HUD style:
```
tile (x, y) — material
pressure: P.PP  (atmosphere + wave_p)
smoke: S.SS
```
- Reads `gmap.atmosphere[y, x] + gmap.wave_p[y, x]` for pressure (matches the colormap input formula)
- Reads `gmap.smoke[y, x]`
- Shows `—` when cursor is outside the map

## 7. Pressure colormap (port from legacy)

Source: `game.py:2095-2149`. Logic:
1. Load `CFG.rendering.pressure_stops` once at startup → numpy float32 array shape (N, 5) where each row is `[pressure, R, G, B, alpha]`
2. Per frame, if pressure overlay enabled:
   - Compute `total = gmap.atmosphere + gmap.wave_p`
   - Scale: `p = total * (10.0 / pressure_scale_slider)`
   - For each tile, find segment in stops array, linear interpolate RGBA
   - Pack into RGBA8 numpy array shape (H, W, 4), upload to dynamic texture
3. Mask: only show on non-wall non-vacuum tiles
4. Draw the colormap texture as another overlay in `compose_world` (same pattern as smoke/fire overlays)

Implementer note: the legacy code uses `pygame.image.frombuffer`. Port to pyray's `update_rgba_texture` from `renderer/core.py`. Same `FieldOverlay`-style pattern.

## 8. Save format (`tools/lighting_presets.toml`)

```toml
# Auto-managed by tools/lighting_demo.py — do not hand-edit while the demo is open.

[default]
ambient = [0.10, 0.10, 0.13]
light_z = 0.5
normal_strength = 1.0
use_normal = true
srgb_decode = true
flashlight = { max_range = 25, intensity = 2.5, angle_spread = 6.283 }
smoke_tint = [190, 195, 210]
smoke_max_alpha = 180
pressure_scale = 2.0
grenade = { blast_radius = 6, pressure = 10.0, wall_damage = 200, unit_damage = 60, fuse_seconds = 0.0, smoke_amount = 1.0 }
```

- "Save preset" with name "horror_dark" → adds/updates `[horror_dark]` section
- "Load preset" reads any section from the file and applies all sliders
- "Reset to defaults" applies the hardcoded fallback defaults (NOT a section read — the literal defaults)

Use `tomllib` for read, `tomli-w` for write (Python 3.11+ ships `tomllib` read-only; `pip install tomli-w` for write). **If `tomli-w` isn't installed, the agent should add it to requirements OR fall back to writing a hand-formatted TOML string** (small, the schema is fixed). Hand-formatted is fine for the foundation pass — preserves layout, no external dep.

## 9. Sim/render orchestration

```
load level
build sim (seed=42, enable_recorder=False  # no need to record)
spawn the 7 units from level.toml [[spawn]]   # for visual interest
# Demo-specific hazards: SKIP the SMOKE_SOURCE / FIRE_SOURCE / BREACH the
# main game adds. The demo starts with a clean ship so smoke/pressure
# visualizations show only what grenades emit.
build renderer
   - same as main.py
   - skip the 5 static emergency lights
   - load lighting_presets.toml if exists, apply [default] preset OR built-in defaults
main loop:
   - poll input (raygui handles slider clicks)
   - if not paused (always running unless user pauses):
     - tick sim: sim.step() in dt-accumulator pattern like main.py
   - build mouse flashlight LightSource each frame from the §4.3 sliders + mouse pos
   - upload_state(gmap, [mouse_flashlight])
   - begin_frame; compose_world(units, [], projectiles); draw_background; blit_world_to_screen
   - draw the pressure colormap overlay over the map (if enabled)
   - draw_debug_hud_extended(gmap)  # tile + pressure + smoke at cursor
   - draw the raygui panel
   - end_frame
on close: shutdown renderer
```

Pausing: tools don't normally need pause, but for grenade-spawn experiments Erik might want to freeze a moment. Bind **Space** to pause/resume. Sim ticks while unpaused.

## 10. Step-by-step execution (for the agent)

1. **Snapshot baseline.** `git status` clean; `pytest tests/` → 46 passing. Note SHA.
2. **Create `tools/lighting_demo.py`** with the skeleton: imports, level load, sim build, renderer build, no static lights, mouse-flashlight-only, main loop. Verify it runs and shows the level with the mouse light. (No sliders yet.) **Commit.**
3. **Add the HUD readouts** (§6). Verify by hovering the mouse, seeing tile/pressure/smoke values update. **Commit.**
4. **Add the raygui slider panel** (§4) with the four most important sections first: Ambient, Lighting, Mouse flashlight, Smoke. No save/load yet. Verify each slider moves the right uniform. **Commit.**
5. **Add the click-to-spawn grenade** (§5) including the grenade sliders (§4.6). Verify by toggling spawn mode, clicking, seeing a real explosion with smoke + pressure spread. **Commit.**
6. **Add the pressure colormap overlay** (§7). Verify the toggle, verify it shows the explosion pressure wave. **Commit.**
7. **Add save / load preset** (§8) including the file format and the hand-written TOML writer. **Commit.**
8. **Polish + verification.** Tighten layout, fix anything sticky. Run the full test suite (should still pass — this is an additive tool, no production code touched). **Commit.**
9. **Push to remote** after each verified commit. Match the established pattern.

## 11. Verification gates

1. `pytest tests/` — 46 passing throughout. No production code touched, so this should never break.
2. `C:/Users/steen/anaconda3/python.exe tools/lighting_demo.py` — opens a window showing UNHCR Vessel with the 7 units, mouse flashlight working, sliders responsive, click-to-spawn grenades visible, pressure overlay toggleable, HUD readouts updating.
3. `tools/lighting_presets.toml` written on Save click, parseable, sections survive a Load click round-trip.
4. No new dependencies (tomli-w is OK to skip in favor of hand-written TOML).

## 12. Out of scope (do NOT add — separate sessions)

- Per-area ambient texture (future)
- 1-bounce rays with surface tint (future)
- Per-source light_z (future)
- Fire as a short-range light source (future, with physics pass)
- Sliders for physics constants (smoke_d, wave_c, etc.) — config.toml only
- Touching `main.py`, `simulation/`, `renderer/`, `level_loader.py` — additive tool only
- Multiple light sources beyond the mouse flashlight
- Per-light-source color (raycaster output is scalar intensity for now)

## 13. Final report format

```
## Lighting demo tool — complete

### Commits landed
- <hash> <subject>
- ...

### Pushed: yes/no

### Decisions made by the agent during work
- raygui layout choices (sizes, spacing, scroll vs no-scroll)
- Anything else worth flagging.

### What works
- Sliders for: ambient, lighting, flashlight, smoke, pressure scale, grenade
- Click-to-spawn grenade with live params
- Pressure colormap overlay (toggle)
- Cursor HUD: tile + material + pressure + smoke
- Save/Load presets to tools/lighting_presets.toml

### Test results
- pytest: 46/46 passing

### Deviations from plan
<any divergences>

### Anything weird
<surprises, residual concerns>

### Pending for Erik
- Open the tool, dial in values, save a preset, confirm round-trip
```

## 14. Style notes

- Comments only where the WHY is non-obvious
- Type hints on new functions
- No new docs unless explicitly required
- Match existing code style (read `renderer/game_renderer.py` and `main.py` first)
- If raygui has a wart that requires a workaround, document it inline

---

When Erik (or whoever) picks this up:
- The patch plan is the implementation contract for v1
- Future v2 features go in their own patch plans (per-area ambient, etc.)
- The tool is meant to grow over time as new visual systems land — keep the slider layout extensible (a list of sections, not hard-coded)
