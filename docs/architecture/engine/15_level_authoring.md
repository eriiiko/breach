# 15 — Level Format & Authoring (painted + tiled)

**Depends on:** 01 (grid & coordinates), 03 (material system), 07 (fluid & water — initial water),
08 (ray engine — light entities), 09 (rendering — art layers).

**Status:** format v2 core ✅ (shipped with `unhcr_vessel_2`) · v2.1 extensions 📝 · tiled
authoring path 📝 (this chapter's build order, agreed with Erik 2026-07-07).

> Locked 2026-07-07 (Erik + Claude). This chapter canonizes the level format (graduating
> `docs/level_editor_and_format_v2_proposal.md`, GO'd 2026-06-10) and adds the **second
> authoring path**: deliberate tile-based maps, painted in a standalone editor and baked to
> ordinary v2 levels. It will be refined as the tools land — the *decisions* below are settled;
> parameter values marked (tunable) are config, not canon.

---

## 0. Two authoring paths, one runtime format

A level is a folder `levels/<name>/` consumed by `level_loader.py`. There is exactly **one
runtime format** (v2). What varies is how a level is *authored*:

| Path | Flow | Tool | For |
|---|---|---|---|
| **Painted** ("one big image") | AI/hand art → align → paint materials over it | `tools/align_level_art.py` (exists) | hero levels, unique ships (`unhcr_vessel_2`) |
| **Tiled** (this chapter) | paint materials first → art is *baked* from a tileset | `tools/map_editor.py` + `tools/bake_level_art.py` (to build) | combat testbeds, ML-training maps, fast iteration |

The two tools stay **separate programs** (decided 2026-07-07): the align/refine tool keeps its
narrow job on painted levels; the map editor owns tiled levels. They share pure helpers
(brushes, undo ring, CSV IO) via a common module — shared logic, separate UIs.

**The key architectural fact** making the tiled path cheap: physics truth (`tilemap.csv`) and
art (full-level PNGs + `[art.align]`) are already decoupled. The tiled path adds a *baker* that
composes the art PNGs from the CSV + a tileset. The engine, loader, and renderer do not know or
care whether a level's art was painted by an AI, a human, or the baker. A baked level remains
hand-retouchable, and a painted level can never be broken by tiled-path code.

```
            map_editor.py (paint materials, place lights/water/spawns)
                 │ writes                              ▲ live preview
                 ▼                                     │
   tilemap.csv + level.toml  ──► bake_level_art.py ────┘
   (physics truth)               │  tileset + adjacency rules
                                 ▼
                 diffuse.png + normal.png (+ emissive_mask.png)
                                 │
                                 ▼
                 ordinary format-v2 level — loader/renderer unchanged
```

Procgen seam (ml/01): any script that emits a `tilemap.csv` can call the baker and get a
playable, lit, textured level — the intended path for generated RL-training maps.

---

## 1. Level format v2 — canon (as shipped)

- **`tilemap.csv`** — 2D grid of integer codes, one per physics tile (1 tile = ⅓ m). Codes ARE
  canon material ids from `src/simulation/materials.py` (`MAT_AIR=0, MAT_HULL=1, MAT_WOOD=2,
  MAT_DOOR=3, MAT_STEEL=4, MAT_GLASS=5, MAT_FURNITURE=6`) plus the one reserved non-material
  code `9` = SPACE (air + `is_vacuum`). Unknown codes are a hard load error.
  **Rule: no tool may carry its own material vocabulary.** Palettes, bakers, and validators
  derive their material set from `materials.py` / `config.toml` at runtime, so a new material
  (one config row) appears everywhere automatically. (This retires the last of the old
  generator/ChatGPT vocabulary for good.)
- **`level.toml`** — `version = "2"`, `name`, `tilemap`, `tile_size_m`, `floor_id`,
  `[[spawn]]` entries, and the `[art]` block: `background` + per-state layers
  (`[art.bare]` / `[art.furniture]` / `[art.destroyed]`, each `diffuse`/`normal`/optional
  `specular`/`height`) + optional `emissive_mask`.
- **`[art.align]`** — `offset_px` + `px_per_tile` (scalar or `[x, y]`): the single
  non-destructive transform (`tile_to_art_px`) through which renderer and editors sample art.
- Doors are `MAT_DOOR` tiles (wall-like material). The doors-v1 state machine (sliding slabs,
  proposal §3) is **not yet built**; see §6 upgrade paths.

## 2. Format v2.1 — three additive extensions

All three are optional keys — every existing v2 level loads unchanged. They serve *both*
authoring paths (a painted level wants lights and water too).

### 2.1 `[bake]` — tiled-level provenance

Presence of this table marks a level as *re-bakeable*; the editor and baker read it, the
loader ignores it.

```toml
[bake]
tileset = "art/tilesets/greybox"   # repo-relative tileset dir
px_per_tile = 64                    # bake output resolution (tunable)
seed = 1234                         # deterministic variant selection
```

### 2.2 `[[light]]` — light entities (static + rotating beacon)

Dynamic lights finally enter the format (they were explicitly out of scope in the v1 editor).
Loader → ray engine (ch. 08) as `LightSource` instances; the renderer owns nothing.

```toml
[[light]]
pos = [12.5, 40.0]          # tile coords (float, tile centers at .5)
color = [255, 200, 150]
intensity = 1.0
range = 12.0                # tiles
kind = "static"             # "static" | "beacon"

[[light]]                   # a rotating beacon (cop-car / heavy-machinery light)
pos = [30.5, 8.5]
color = [255, 40, 40]
intensity = 1.5
range = 16.0
kind = "beacon"
period_s = 1.5              # one full rotation
beam_deg = 30.0             # cone width
phase = 0.0                 # 0..1; a red/blue pair = two beacons, phase 0.0 / 0.5
```

A beacon is a directional cone whose direction advances `2π · dt / period_s` per tick — sim
state, deterministic, no renderer special-case. *Seam to verify at patch start:* the ray
engine's current directional-cone support; if Tier-1 casts omnidirectionally, the beacon patch
adds the cone mask there (it belongs in ch. 08 anyway — fire-as-light already wants it).

### 2.3 `[water]` — initial water state *(amended by P5, 2026-07-08)*

Aquariums, flooded compartments, coolant pools. An aquarium is *nothing special*: a
glass-enclosed region whose tiles start with water depth — the water system (ch. 07) does the
rest, including draining through the hole you shoot in the glass.

```toml
[water]
depth_map = "water_init.npy"   # int32 Q16.16 metres, shape == tilemap (H, W); 0 = dry
```

The file IS the field. The original 8-bit PNG + `max_depth_m` carrier was dropped in the P5
design gate (critique record, `docs/patch_levels_p5_water.md`): the auto-scaling 8-bit
quantization made edits **non-local** — deepening one pool re-quantized every other pool's
golden-pinned integers — and PNG *decoding* would have added the runtime imaging dependency
`level_loader.py` deliberately avoids. The `.npy` round-trips by identity (the editor writes
Q16.16 ints via `water_fixed.quantize`; the loader `np.load`s them verbatim, hard-validating
shape/dtype/sign; `GameMap.__init__` seeds `water_depth` masked to `(~solid) & (~is_vacuum)` —
the solver zeroes depth on solid, a mass sink). Trade-off on record: not hand-paintable in an
image editor — the map editor's WATER mode (§5) is the author; FieldEdit remains the runtime
write path. A level without a `[water]` key loads bit-identically to before the key existed
(water dormancy).

## 3. Tilesets

A tileset is a directory under `art/tilesets/<name>/` with a `tileset.toml` manifest and one
PNG strip (diffuse + normal, optional emissive/height) **per material appearance**:

- **Resolution (decided 2026-07-07):** tileset source art is authored at **128 px/tile**; the
  default bake output is **64 px/tile** (2× supersampled downscale — crisp edges). Both ends
  configurable; nothing in the pipeline assumes a number. Rationale: at 64 px/tile a
  256-tile-wide ship still fits a 16 384 px GPU texture and the full layer stack stays ~4×
  cheaper in VRAM than 128, while authoring at 128 means we can re-bake sharper any time
  without re-authoring art. (The old 24 px/tile was an artifact of the first vessel's art, not
  a decision.)
- **Autotiling: 16-case edge bitmask** (N/E/S/W neighbor-same → piece index) per wall-family
  material, v1. The 47-case blob set (corner-aware) is a drop-in upgrade later — the manifest
  declares which scheme a strip uses.
- **Connectivity groups**, not per-material islands: `hull/steel/wood/door/glass` form the
  *wall family*. A glass tile's bitmask counts any wall-family neighbor as connected, so a
  window reads as glass-in-a-frame continuing the wall line; doors likewise get frames. Groups
  are declared in `tileset.toml`, not code.
- **Floors:** `MAT_AIR` is walkable interior — it renders as deck plating from N floor
  variants, chosen deterministically from `[bake].seed` (no ML-visible pattern, no bake
  nondeterminism). `SPACE` bakes transparent; the screen-fixed `background` starfield shows
  through, exactly as in painted levels.
- **Normals ship with every piece** — beveled wall edges respond to ray-engine lighting from
  day 1. This also quietly subsumes most of the "bent tiles" idea (notes 2026-07-05): when the
  tileset later gains 45° wall pieces, their *baked normal maps* carry the true diagonal
  normal, so lighting and future impact decals read a smooth surface while physics keeps the
  tile staircase. No per-tile normal field needed.
- **v1 tileset = `greybox`:** generated procedurally (numpy — flat material colors, beveled
  edges, derived normals) by `tools/make_tileset.py`. Zero SD dependency; good enough to read
  the level instantly. AI-generated styled tilesets (via `tools/content_aware_tiles` etc.,
  re-cloned by the setup script) swap in later without touching the pipeline.

## 4. The baker — `tools/bake_level_art.py`

`tilemap.csv + tileset (+ seed) → diffuse.png + normal.png (+ emissive_mask.png)`.

- Pure function of its inputs — same inputs, byte-identical outputs (golden-image tested).
- **Region re-bake API**: bake only a tile rectangle (the editor's live preview repaints just
  the stroked region; full-level bake stays a save-time/CLI operation).
- CLI: `python tools/bake_level_art.py <level> [--tileset X] [--px-per-tile N]` — the procgen
  entry point. Writes the `[art]` + `[bake]` blocks into `level.toml` (with `.bak`, same
  byte-preserving convention as the align tool).
- Emits `emissive_mask.png` only if the tileset declares emissive pieces (greybox: none).

## 5. The map editor — `tools/map_editor.py`

Standalone pyray tool (same stack as the game and the align tool), **new-level-first**:

- **NEW** — create `levels/<name>/` from scratch: dimensions prompt → hull-shell scaffold
  (space border, hull ring, air interior) + minimal `level.toml`.
- **PAINT** — material brush; palette **generated from the material table** (§1 rule), brush
  sizes, line tool, eyedropper, undo ring — reusing the align tool's tested pure helpers via
  the shared module.
- **ROOM** — drag a rectangle → wall perimeter + floor interior (wall material selectable;
  overlap-aware: rooms sharing an edge share the wall).
- **CORRIDOR** — drag a path → floor of width w (tunable, default 3 tiles) with walls along
  the sides where it cuts through solid/space.
- **DOOR** — one-click on a wall tile → `MAT_DOOR` snapped into the wall run (refuses
  non-wall placement).
- **LIGHT** — place/move/delete `[[light]]` entities; static vs beacon toggle; parameter
  nudge keys; rendered live in the preview.
- **WATER** — bucket-fill an enclosed region to a depth → writes `water_init.npy` (§2.3).
  (Paint a glass box, fill it: that's an aquarium.)
- **SPAWN** — place/move/delete `[[spawn]]` entries (team, name) — a combat testbed needs
  marines and zombies without hand-editing TOML.
- **Live baked preview** — strokes trigger region re-bakes (§4) so you see the textured,
  normal-mapped result as you paint, not colored rectangles.
- **SAVE** — Ctrl+S: `tilemap.csv`, `level.toml` (lights/water/spawns/bake blocks),
  `water_init.npy`, full bake. `.bak` convention throughout.

Editor v1 explicitly does **not** do: entity kinds beyond lights/spawns, decals, multi-floor,
texture editing, in-game editing (see §6).

## 6. Upgrade paths (nothing below is blocked by v1 decisions)

- **Doors v1** (sliding slabs + open/close state machine, old proposal §3/F6): tiled levels
  store doors as `MAT_DOOR` tiles exactly like painted levels, so when door objects land they
  apply to both paths for free. The editor gains open/closed initial-state then.
- **In-game edit mode**: the old proposal's target. The standalone editor's pure helpers are
  the part that migrates; deferred until doors/entities settle the sim-side seams.
- **47-case blob autotiling, 45° pieces, styled AI tilesets, per-room floor themes**: tileset
  manifest upgrades — baker/editor logic untouched.
- **Impact decals reading baked normals** ("bent tiles", second half): a decal pass samples
  the baked normal map for orientation — lands with the decal layer (graphics backlog).

## 7. Build order (agreed 2026-07-07; autonomous patches, gate before P4/P5)

| # | Patch | Contents | Test spine |
|---|---|---|---|
| P1 | greybox tileset | `make_tileset.py`, `tileset.toml` schema, 128 px pieces + normals | unit: manifest, bitmask coverage, determinism |
| P2 | baker | `bake_level_art.py`: 16-case compose, region re-bake, CLI, toml writeback | golden small-map image; determinism; loader round-trip |
| P3 | editor core | `map_editor.py`: NEW/PAINT/ROOM/CORRIDOR/DOOR + live preview + SAVE; shared-helpers refactor out of align tool (its tests must stay green) | pure-helper units (raylib has no input injection) |
| P4 | lights | `[[light]]` schema + loader + ray-engine seam (cone/beacon) + LIGHT mode | schema/loader units; beacon determinism |
| P5 | water init | `[water]` schema + loader→WaterSolver seed + WATER fill mode | fill-volume conservation; aquarium demo |
| P6 | (later) AI tilesets | `tools/setup_ai_tools` script (the one `.gitignore` references but never existed), styled tileset via content_aware_tiles | — |
| P7 | (later) diagonal pieces + decal-normal sampling | — | — |

**Acceptance demo (end of P5):** `levels/combat_testbed_1` — rooms, corridors, doors, a
window wall, a static light + a red/blue beacon pair, one aquarium — painted start-to-finish
in the editor in under 15 minutes, running every physics system.

---

## Implementation status (honest, updated as patches land)

- Format v2 core: ✅ shipped (`unhcr_vessel_2`, loader `SUPPORTED_VERSIONS = {"1","2"}`).
- Painted-path tool `align_level_art.py`: ✅ ALIGN + PAINT + SAVE, unit-tested.
- Everything else in this chapter (v2.1 keys, tileset, baker, map editor): 📝 designed, not
  built. P1 is the first patch.
- Known seams to verify before their patch: ray-engine directional cones (P4), WaterSolver
  seeding (P5), `[[spawn]]` schema reuse in the editor (P3).
