# Configuration & Hot-Reload

**Depends on:** (none)

Breach keeps its tunable numbers out of the code. Game balance, physics
coefficients, material properties, and weapon stats all live in a single
`config.toml` file at the repository root, loaded once at startup into a global
configuration object. A separate per-level descriptor (`levels/<name>/level.toml`)
declares which map, art, and unit spawns a session uses. This chapter describes
both files, the attribute-access wrapper that exposes them, the hot-reload path,
and the deliberate boundary between *config* values and *visual* values.

---

## 1. Why config is data, not code

The simulation is a deterministic numerical engine: a few dozen coefficients
decide how fast a shockwave travels, how readily wood ignites, how much an
explosion damages a wall. Tuning those numbers is the bulk of the iteration
work, and recompiling — or even re-editing source — for every tweak is too slow.
Pulling the numbers into `config.toml` makes the knobs visible in one place,
keeps them out of the C++ and Python logic, and lets a designer change behaviour
without touching code.

TOML is chosen because it is human-readable, supports nested tables and typed
arrays natively, and ships with Python's standard library (`tomllib`, read-only,
3.11+). No third-party parser is required to read config.

---

## 2. The config object: `CFG`

`config.py` defines a `GameConfig` class and exposes one process-global instance:

```python
from config import CFG
print(CFG.clock.ticks_per_second)      # 12
print(CFG.physics.wave_c)              # 66.0
print(CFG.weapons.grenade.blast_radius)  # 5
```

Loading walks the parsed TOML and wraps every nested table in a `Namespace` — a
thin object that turns dict keys into attributes recursively. The result is that
`[weapons.grenade]` in the file becomes `CFG.weapons.grenade` in code, with each
key an attribute. This is the only access pattern; nothing reads the raw dict.

`Namespace` is intentionally trivial — attribute access, a `__repr__`, nothing
else. It is not a schema. There is no validation of which keys exist or what
types they hold; a typo in a key name surfaces as an `AttributeError` at the
call site, and code that wants to tolerate a missing key uses
`getattr(CFG.section, "key", default)` (several call sites do exactly this for
parameters added after their first consumers shipped).

### Derived values

A few quantities are products of other config values and are computed once after
each load rather than stored redundantly in the file:

| Derived value | Formula |
|---|---|
| `CFG.clock.ticks_per_phase` | `ticks_per_second * phase_duration_seconds` |
| `CFG.clock.ticks_per_round` | `ticks_per_phase * phases_per_round` |

Keeping these in `_load()` means the file holds only the independent inputs, and
the dependent values can never drift out of sync with them.

---

## 3. What lives in `config.toml`

The file is organised into top-level sections, one per subsystem. The current
shape:

| Section | Holds | Example keys |
|---|---|---|
| `[clock]` | turn/tick timing | `ticks_per_second`, `phases_per_round`, `phase_duration_seconds`, `ap_per_phase` |
| `[movement]` | per-mode move costs | `marine_sprint_ticks_per_tile`, `xeno_sprint_ticks_per_tile` |
| `[physics]` | atmosphere/smoke/wave solver coefficients | `wave_c`, `d_atm`, `breach_rate`, `physics_dt`, `physics_substeps` |
| `[display]` | window + active level | `panel_width`, `level` |
| `[materials.<name>]` | one table per material | `hp`, `flammable`, `mobility`, `light_atten`, `conductivity`, … |
| `[weapons.<name>]` | per-weapon stats | rifle/grenade/door-explosive damage, radius, AP cost |
| `[zombie]`, `[marine]` | unit stats | `hp`, `melee_damage`, `trigger_radius` |
| `[rendering]` | pressure-overlay colormap | `pressure_scale`, `pressure_stops` |
| `[combat]` | shared combat constants | `blast_damage_threshold`, `unit_absorption` |

The `[materials.*]` block is the data-driven heart of the material system: each
`[materials.<name>]` row is a full property record (structural HP, flammability,
walkability, per-channel optical attenuation, thermal conductivity, ignition
temperature, acoustic reflect/absorb, blast resistance). These rows are read into
the `MaterialTable` (see the Material System chapter) as per-id numpy columns.
Several columns — the optics, acoustic, and thermal entries — are *stored but not
yet consumed* by every system that will eventually read them; they are the data
foundation for passes that land later. Their values are illustrative and are
tuned in the demos.

---

## 4. Levels: `level.toml` vs `config.toml`

A clean separation exists between **what the engine is tuned like** (one
`config.toml`, global) and **which content a session loads** (one `level.toml`
per level folder). `config.toml` does not name a map file directly — it names a
*level* by folder:

```toml
[display]
level = "unhcr_vessel"   # folder under levels/ to load on startup
```

At startup `main.py` reads `CFG.display.level` and hands it to the level loader,
which opens `levels/<name>/level.toml`. That descriptor is parsed by
`level_loader.py` into a `LevelData` dataclass:

| `level.toml` field | Meaning | Required |
|---|---|---|
| `version` | schema version (`"1"`) | yes |
| `name` | display name | no (defaults to folder) |
| `tilemap` | CSV file of physics tile codes | yes |
| `diffuse` | base-colour art texture | yes |
| `tile_size_m` | metres per tile | no (default `0.333`) |
| `normal` | Laigter normal map | no |
| `emissive_mask`, `emissive_bloom` | glow + halo masks | no |
| `wall_mask` | overrides CSV-derived walls | no |
| `background` | screen-fixed backdrop | no |
| `floor_id` | floor decoration id | no (default `0`) |
| `[[spawn]]` | unit spawn entries (name/team/x/y/footprint) | no |

The CSV tilemap is the source of truth for physics; art assets are render-only.
The loader validates the version, requires the mandatory fields, and raises a
clear `ValueError` if a declared optional asset is missing. CSV tile *codes* are
distinct from material *ids* — the loader's `materials_from_tilemap` maps codes
(0 vacuum, 1 hull, 2 wood, 3 door, 4–8 interior air) onto material ids and a
vacuum mask. The unused material ids (steel, glass) exist in the table but have
no CSV code yet; one is assigned when a level first needs them.

This split keeps `config.toml` about *engine behaviour* and `level.toml` about
*this map's content* — swapping levels is a one-line config change, and shipping
a new map is a new folder, never a code edit.

---

## 5. Hot-reload

Reloading config without restarting is essential to the tuning workflow. The
reload is bound to **Ctrl+R**, polled in `InputHandler.handle_frame`:

```python
if ctrl_held and rl.is_key_pressed(K.KEY_R):
    CFG.reload()
```

`CFG.reload()` simply re-runs `_load()`: it re-reads `config.toml` from disk,
re-wraps every section, and recomputes the derived clock values, mutating the
existing global `CFG` in place so all existing imports see the new numbers.

### Why Ctrl+R and not F5

F5 is the renderer's **normal-map toggle** (`GameRenderer.poll_toggles`), one of
a bank of F1–F7 debug-overlay keys. Config reload was deliberately moved off F5
onto Ctrl+R to avoid that collision — the chord is unambiguous and leaves the
single-key debug toggles intact.

### What a reload does and does not pick up

Reload is **not** uniform across subsystems. What it refreshes depends on whether
a consumer reads `CFG` live each time, or copied the value once at construction:

- **Picked up immediately** — anything read from `CFG` on demand. The pressure
  overlay reads `CFG.rendering.pressure_scale`/`pressure_stops` when built; clock
  timing is read per-frame in the main loop.
- **Material table — designed but not wired on Ctrl+R.** `GameMap` has a
  `reload_material_table()` method that rebuilds the `MaterialTable` from `CFG`
  and recomputes the table-derived caches (HP, conductivity, optical
  attenuation) while preserving the live atmosphere/obstacle state. It is the
  correct hook, but the Ctrl+R path currently calls only `CFG.reload()` and does
  **not** call `gmap.reload_material_table()`. So editing a `[materials.*]` row
  and pressing Ctrl+R updates `CFG` but not the running material caches.
- **Physics coefficients — not picked up.** `PhysicsRunner` copies every
  `CFG.physics.*` value onto the C++ solver objects (`AtmosphereSolver`,
  `SmokeDynamics`, `FireSimulation`) **once, at construction**. There is no
  re-bind on reload, so a physics tweak needs a restart to take effect.

This is the practical shape of hot-reload today: fast iteration on values that
are read live, restart-required for values cached into C++ solvers or the
material table.

---

## 6. The config / visual boundary

A clear rule governs *where* a tunable lives, driven by what breaks if it is
wrong:

- **Numerical-stability parameters stay in `config.toml`.** Anything that, set
  badly, makes the simulation diverge or behave non-physically — smoke diffusion
  `d_smoke`, wave speed `wave_c`, breach relaxation `breach_rate`, fire spread,
  substep counts — is config-only. These are *engine* values, not art knobs.
- **Pure-visual parameters are tuned with live sliders, saved to a preset
  file.** Ambient colour, light height, normal-map strength, smoke tint and
  alpha, grenade visual output — values that only affect *appearance* — are
  dialled in the standalone lighting/tuning tool (`tools/lighting_demo.py`) and
  persisted to `tools/lighting_presets.toml`, a file separate from `config.toml`.
  The tool reuses the real level loader, simulation, and renderer so the lit
  result is identical to the game, then writes named preset sections.

The two files do meet at a small, deliberate seam: the tuning tool reads the
pressure colormap (`CFG.rendering.pressure_stops`) and the default
`CFG.rendering.pressure_scale` from `config.toml` so its pressure-field overlay
matches the game's, while exposing the scale as a live slider. The preset file is
the designer's scratchpad; promoting a good visual value into `config.toml` (or
into level/material data) is a deliberate manual step, never an automatic
write-back. This keeps `config.toml` authoritative and reviewable, and keeps the
fast art-direction loop in its own file.

---

## 7. C++ config strategy

The simulation's hot inner loops are C++ (pybind11). Config still lives in TOML
on the Python side; the values are pushed across the binding into the solver
objects as plain attribute assignments (the pattern `PhysicsRunner` already
uses). The alternative — parsing TOML directly in C++ — is not used; keeping a
single Python-side parser avoids two sources of truth. The open work here is
making that push happen on *reload*, not only at construction, so physics tuning
gets the same live iteration the read-on-demand values already enjoy.

---

## Implementation status

Audited against `config.py`, `config.toml`, `level_loader.py`,
`src/input_handler.py`, `src/simulation/physics_runner.py`,
`src/simulation/gamemap.py`, `src/simulation/materials.py`,
`renderer/pressure_overlay.py`, `renderer/game_renderer.py`, and
`tools/lighting_demo.py`.

**Built and working**

- `GameConfig` / `Namespace` loader with recursive attribute access; global
  `CFG`. Derived clock values (`ticks_per_phase`, `ticks_per_round`) computed on
  load.
- `config.toml` with all sections described above, including the full named-key
  `[materials.*]` table and the `[rendering]` pressure colormap.
- `CFG.reload()` re-reading the file; bound to **Ctrl+R** in `InputHandler`
  (deliberately off F5, which remains the renderer's normal-map toggle).
- Level pipeline: `CFG.display.level` selects a folder; `level_loader.load`
  parses `level.toml` (version-checked, required-field-validated) into
  `LevelData`, including `[[spawn]]` entries and optional art assets.
- Physics coefficients sourced from `CFG.physics.*` and bound onto the C++
  solvers at construction — `wave_c`, `damping`, `transfer`, `d_atm`,
  `feed_rate`, `breach_rate`, `max_source_per_step`, and the smoke parameters.
  (This resolves most of the historical "physics constants live in code" issue.)
- `MaterialTable.from_config(CFG)` building per-id numpy columns; pressure
  overlay reading `CFG.rendering.pressure_stops`/`pressure_scale` with fallbacks.
- Lighting/tuning tool persisting visual presets to a *separate*
  `tools/lighting_presets.toml`, reading the pressure colormap from config.

**Designed but not wired**

- `GameMap.reload_material_table()` exists and is correct, but the Ctrl+R path
  does **not** call it. Editing a `[materials.*]` row and reloading updates `CFG`
  but not the running material caches — material hot-reload is effectively a
  no-op in the game loop until that call is added.
- C++ solver values are not re-pushed on reload; `PhysicsRunner` binds them only
  in `__init__`. Physics-coefficient hot-reload requires a restart.

**Gaps / known issues**

- **No schema or type validation.** `Namespace` accepts any keys; a misspelled
  section or key fails only at the consuming call site. Several consumers guard
  with `getattr(..., default)`, which masks missing keys rather than reporting
  them.
- **Residual fallback constants.** Fire-simulation parameters now bind from
  `[physics.fire]` in `config.toml` (`PhysicsRunner.__init__`); the `FIRE_*`
  module-level constants in `physics_runner.py` survive only as fallbacks used
  when a config key is absent. A handful of inline magic numbers (e.g. explosion
  smoke/atmosphere deposit factors) still remain outside config.
- **Door flammability mismatch.** `[materials.door]` ships `flammable = false`
  to preserve existing behaviour, even though the material-system design treats
  doors as flammable; a comment in `config.toml` flags this for a future flip.
- **Level reload is not part of Ctrl+R.** Switching the active level still
  requires a restart; hot-reloading level assets is a planned extension of the
  reload path, not yet built.
