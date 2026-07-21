# P4 impl — `[[light]]` entities: static lights + rotating beacons (levels-w1)

> **P4 patch record** — design-gated 2026-07-07 (1 blocker + 5 majors resolved across two independent critiques); built as specified, annotations binding.

**Status:** design-gate draft v1 (2026-07-07) — for adversarial critique, then build.
**Spec:** `docs/architecture/engine/15_level_authoring.md` §2.2. **Scout:** ray-engine seam report 2026-07-07.
**Erik's locked calls (2026-07-07):** (1) lights are **render-only** in P4 — no synced-state entry;
(2) beacons **freeze with the sim** (angle = pure function of sim tick); (3) **port** the five
hardcoded `main.py` emergency lamps into `unhcr_vessel/level.toml`.

## 1. What exists (scout facts, binding)

- `cpp/src/raycaster.h:53-71` `LightSource`: float `x,y,max_range,angle_center,angle_spread,
  intensity,heat,jitter,color[3]` + int `ray_count`. **Cone emission already exists**
  (`angle_spread < 2π`). CORRECTED (integration critique M2): `falloff` is NOT in the Python
  bindings (`bindings.cpp:847-865`) — every Python-built source is UNIFORM, including the
  actual flashlight (spread 6.283, `main.py:305`). `LightEntry` therefore has NO falloff field
  (parity by necessity); builder must NEVER write `src.falloff` (pybind class, no dynamic
  attrs → AttributeError). Also: `bp.LightSource` has only `py::init<>()` — apply params via
  a setattr loop, never `bp.LightSource(**d)`.
- Perf (integration critique, verified): `compute_light_field` already runs unconditionally
  every frame with no caching (`game_renderer.py:253,273-279`; `lighting.py:268-291`) — a
  rotating beacon adds nothing structural, and at beam 30°/range 12 costs ~8 rays vs 114/lamp.
  Per-frame construction of ~10 pybind objects is negligible (optional nicety: cache static
  sources, rebuild only beacons).
- Lights today: 5 hardcoded lamps built ONCE pre-loop (`main.py:249-263`); per frame,
  `sources = list(static_lights)` + mouse flashlight (`main.py:296-309`) →
  `renderer.upload_state(..., light_sources=...)` → `lighting.compute_light_field` →
  `cast_source_directional` per source. **No sim-side list.** (P4 moves LightSource
  construction per-frame for beacons — parameter cost only, see integration critique.)
- Determinism: render channels are ingress-exempt (engine/14 §synced-vs-local; renderer is free
  territory; lint scans `src/simulation/` only). `heat` is the only synced ray output — we do
  NOT set `heat` on level lights (stays 0.0).

## 2. Design

### 2.1 Loader (`level_loader.py` — outside the gated tree)
- `LightEntry` dataclass beside `SpawnEntry`: `x: float, y: float` (tile coords),
  `color: tuple[float,float,float]` (0–255 ints in toml, normalized 0–1 at parse),
  `intensity: float = 1.0`, `range: float = 12.0` (tiles), `kind: str = "static"`
  (`"static" | "beacon"`), `period_s: float = 2.0`, `beam_deg: float = 30.0`,
  `phase: float = 0.0` (fraction of a turn, 0–1).
- Parse loop mirrors `[[spawn]]` (`level_loader.py:309-323`): `raw.get("light", [])` →
  validate (kind whitelist; `period_s > 0`; `0 < beam_deg <= 360`; finite floats; error message
  carries entry index + required-fields hint) → `LevelData.lights: list[LightEntry]`.
- **No Q16.16 snap in P4** (simplest honest design): values are render-local floats, same class
  as `light_rgb`. The migration note in engine/15 §2.2 already states that moving lights
  sim-side later requires door-2 snap at load + integer trig (kit exists: `sin_q16/cos_q16`) —
  an accepted, documented gap, not machinery built now.

### 2.2 Beacon angle — pure function, no accumulated state
New pure module `src/level_lights.py` (importable without `breach_physics`, headless-testable;
NOT under `src/simulation/` — render-side helper):

```python
def beacon_angle(total_tick: int, tick_dt_s: float, period_s: float, phase: float) -> float:
    """Facing angle in radians. Pure function of the sim tick -> frozen when the sim
    is paused, exact under replay, no drift (never += per frame)."""
    return math.tau * (phase + (total_tick * tick_dt_s) / period_s)  # caller may mod 2pi

def light_source_params(entry, total_tick, tick_dt_s) -> dict:
    """LightEntry -> kwargs for bp.LightSource (angle_center/angle_spread for beacons,
    spread=2pi for static). ALWAYS emits heat=0.0, jitter=0.0 (structural, see below).
    Pure; main.py maps the dict onto the compiled struct."""
```
- **`tick_dt_s` is `sim_time_per_tick` (`main.py:267`, = 1/CFG.clock.ticks_per_second), NEVER
  the wall-clock frame `dt` of `main.py:272-274`** (critique B1: wall-clock dt = wobble +
  no pause-freeze + no replay). §2.5 gains an integration-shaped test: building the params
  twice at a fixed tick with the production arguments yields identical angles.
- **`total_tick` is monotonic across rounds** (critique M1): `sim.tick` rewinds each round
  (`simulation.py:241`, `_end_round`), which would snap beacons to phase-0 at round
  boundaries. Use `turn_counter * ticks_per_round + tick` (builder verifies the turn-counter
  attribute name); still a pure function of sim state, replay-exact, no visual snap.
- **heat=0.0 / jitter=0.0 are STRUCTURAL, not convention** (critique M2): the render loop holds
  a live write handle into golden-hashed `gmap.heat` (upload_state → compute_light_field), and
  headless goldens never execute that path — a leak would silently diverge interactive sessions
  from their replays. Therefore: (a) `light_source_params` hard-pins both, (b) a test asserts
  heat==0.0 and jitter==0.0 for every kind, (c) the loader REJECTS `heat`/`jitter` keys in
  `[[light]]` toml with a clear error.
- Beacon stepping granularity is the tick rate (24 Hz → 7.5°/step at period 2 s): accepted and
  recorded — do NOT "smooth" with wall-clock interpolation later; that breaks freeze/replay
  (critique N2).
- Cop-car pair = two `[[light]]` beacon entries, phases 0.0 / 0.5 (chapter example).
- Known multi-light direction-cancel dead spots (docs/TODO.md:301) are pre-existing and NOT
  P4's to fix; accepted gap, noted in the patch record.

### 2.3 `main.py` consumption + lamp port
- Delete the hardcoded `static_lights` block (`main.py:249-263`); per frame build
  static sources once at load + beacon sources per frame from `level.lights` (setattr loop,
  §1). Off-grid entries: skipped with ONE warning at load (not per frame). Mouse flashlight
  unchanged. `_upscale_level` (`--res N`, `main.py:120-155`) scales light positions like
  spawns (integration critique minor 6 — keeps the "units land in the same place" contract).
- **Lamp port covers EVERY level that has the lamps in-grid today (integration critique B1 —
  the hardcode lights ALL levels and the DEFAULT launch level is `unhcr_vessel_2` per
  config.toml:280):** all 5 lamps → `unhcr_vessel/level.toml` AND `unhcr_vessel_2/level.toml`
  (both 50×120, all 5 in-grid); the 3 in-bounds lamps (y=10/30/55) → `playground/level.toml`.
  No level gets darker by this patch. `tools/lighting_demo.py` keeps its own sources (out of
  scope).

### 2.4 Editor LIGHT mode (`tools/map_editor.py`, extends P3)
- P3 shipped: modes = TAB/Shift+TAB + F1–F5 (LIGHT becomes F6); number row is materials in
  EVERY mode; +/- taken (corridor width). **Free keys per P3's report: L, B, C, P, R, X, Y,
  H, E, Shift+wheel.** Pinned here: `B` toggles static↔beacon on the hovered light;
  remaining parameter keys (intensity/range/period) chosen at build time from the free set —
  NOT Shift+wheel (integration critique minor 1: wheel zoom is unconditional at
  `map_editor.py:745-749`; +/- are CORRIDOR-scoped so they're free inside LIGHT mode).
  Place (LMB), move (drag), delete (RMB) — the SPAWN-mode interaction template.
- **Editor plumbing the builder must wire (integration critique M3 + minors 2-4):**
  `MODE_HINTS["LIGHT"]` entry + the hardcoded HUD string "TAB/F1-F5" → F6
  (`map_editor.py:527-537`, `:1181-1183`); `dirty_lights` joined into `dirty_any`, the Esc
  unsaved guard, Ctrl+S reset, and exit print (`:704-714`, `:1023`, `:1216-1218`); a LIGHT
  branch in the mode-scoped Ctrl+Z dispatch (`:985-1008`); lights writeback runs after
  `write_spawns` sharing the once-per-SESSION `.bak` contract (`write_bak=False` /
  `toml_bak_written` — `:619-620`, `:1010-1019`). **Scope limitation on record:** the editor
  refuses levels without `[bake]` (`:568-578`), so the vessel/playground lamp entries are
  loader-consumed but hand-edited — LIGHT mode authors tiled-path levels only.
- P3 integration facts (from its report): mode dispatch is a flat elif chain in `run_editor`
  (SPAWN is the template); `[[light]]` gets its own managed block via the `write_spawns`
  pattern (`_SPAWN_HEADER_RE` generalizes) + its own small ring (`SpawnRing` is copy-generic);
  writeback call goes into the Ctrl+S block BEFORE `bake_level` (`.bak` ordering comment
  there); `run_editor` already loads `lvl.raw_toml`; light markers draw right after spawn
  markers in the world pass (same `to_screen`/`preview_ppt` transform).
- Markers: color-filled circle + range ring; beacons additionally a beam wedge at
  `beacon_angle(preview_tick, ...)` animated in the editor's own clock (editor is not the sim).
- Writeback: `[[light]]` managed block in `level.toml`, same mechanism as P3's spawn block,
  `.bak` once per save.

### 2.5 Tests (all headless, no `breach_physics`)
- Loader: parse/defaults/validation errors (bad kind, zero period, color range), round-trip.
- `beacon_angle`: pinned values; same tick → same angle (freeze semantics); phase pair 0/0.5
  opposite; period scaling.
- Editor: light writeback round-trip preserving unrelated toml bytes; marker hit-testing.
- Vessel: `unhcr_vessel/level.toml` AND `unhcr_vessel_2/level.toml` parse with exactly 5
  lights, `playground` with 3, coordinates pinned to the old constants; **colors pinned with
  ±1/255 tolerance** (integration critique minor 5: lamp color 0.1 → toml int 26 → 0.10196 —
  exact equality is impossible under the 0-255 int schema).
- Regression: full arc suites + loader/format suites stay green.

## 3. Out of scope (accepted gaps, on record)
Sim-visible light (AI/ML observation) — engine/15 §2.2 migration note; multi-light direction
cancel dead spots (TODO.md:301); per-source `light_z`; fire-as-light (TODO.md:29); light UI in
game (editor-only authoring). Golden signatures: untouched (render-only, no synced writes).

## 4. Build order inside P4
1. `LightEntry` + loader + tests → 2. `src/level_lights.py` pure helpers + tests →
3. `main.py` swap + vessel lamp port + pin test → 4. editor LIGHT mode + writeback tests →
5. patch record (`docs/patch_levels_p4_lights.md`) + arc suites green.
