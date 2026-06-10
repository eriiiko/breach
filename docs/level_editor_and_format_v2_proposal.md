# Level Format v2 + Level Editor v1 — proposal

**Status:** proposal for Erik's review — NOT canon. Written 2026-06-10 (evening) in answer to
`levels/unhcr_vessel_2/explanation_to_claude.txt` + the mission-1 conversation. On approval the
format half graduates into the architecture chapters (the level format touches grid/materials/
rendering canon); the editor half becomes its own chapter.

**Goal:** stitch the new three-layer UNHCR Vessel art into a playable level, make the CSV speak
canon material ids, render destruction visibly (furniture burns away, walls show damage), give
doors a body, make screens glow — and build the small in-game editor that makes all of this (and
every future level) minutes instead of nights.

---

## 1. Level format v2

### 1.1 Canon CSV — codes ARE material ids

The generator's room-type vocabulary (0=space, 2=floor, 4..8=room variants) dies at the level
boundary. A v2 `tilemap.csv` stores **canon material ids** (`src/simulation/materials.py`):

| code | meaning |
|------|---------|
| 0 | MAT_AIR (interior air) |
| 1 | MAT_HULL |
| 2 | MAT_WOOD |
| 3 | MAT_DOOR |
| 4 | MAT_STEEL |
| 5 | MAT_GLASS |
| 9 | **SPACE** — reserved non-material code: MAT_AIR + `is_vacuum` (the one thing that isn't a material) |

`level.toml` gains `version = "2"`; the loader keeps the old `materials_from_tilemap` mapping for
`version = "1"` levels and reads v2 CSVs literally (space code 9 → air + vacuum). A 20-line
migration script (`tools/migrate_tilemap_v2.py`) converts v1 → v2 (0→9, 1→1, 3→3, 2/4..8→0) so
both vessel folders can flip the same day and the old generator vocabulary never confuses anyone
again. The code-2 landmine (generator floor vs MAT_WOOD wall) dies with it.

### 1.2 Layered art

```toml
version = "2"
[art]
background = "background.png"          # screen-fixed backdrop (unchanged)
[art.bare]                              # the empty ship — always present
diffuse  = "ship_bare.png"
normal   = "ship_bare_n.png"
specular = "ship_bare_s.png"            # stored now, consumed when lighting learns specular (§2.3)
[art.furniture]                         # optional overlay state
diffuse  = "ship_w_furniture.png"
normal   = "ship_w_furniture_n.png"
specular = "ship_w_furniture_s.png"
[art.destroyed]                         # optional overlay state
diffuse  = "ship_destroyed.png"
normal   = "ship_destroyed_n.png"
specular = "ship_destroyed_s.png"
emissive_mask = "emissive_mask.png"     # painted in the editor (§4); screens glow in the dark
[art.align]                             # non-destructive alignment (§1.3)
offset_px = [0, 0]
px_per_tile = 24.0
```

### 1.3 Alignment without cropping

The new art is uncropped; the grid mapping lives in `[art.align]` (`offset_px`, `px_per_tile`)
rather than in destructive PNG crops. The renderer samples art through this transform; the editor's
ALIGN mode (§4) sets it by eye. Cropping stays possible (the editor can export cropped PNGs once
alignment is final) but is no longer required to ship the level.

### 1.4 The per-tile art-state mask (destruction made visible)

One `art_state` byte per tile, **derived at load + updated by events**, never hand-painted:

- `BARE` — tile shows the bare layer.
- `FURNISHED` — tile shows the furniture layer. Initialised wherever the editor painted a
  *furnishing material* (§4: furniture tags as wood — flammable, destructible; this also finally
  gives fire something to burn: the current vessel has zero flammable tiles).
- `DESTROYED` — tile shows the destroyed layer. Set by `destroy_wall` / burn-through / burst on
  that tile (the existing `on_tile_changed` seam), and by fire consuming a furnished tile — Erik's
  "fire could burn away furniture" and "grenades start to draw the destroyed version" fall out of
  the same event hook. Visible wall damage is the point: the ship should wear its scars.

Runtime-only state (recomputable from material + damage events; the recorder can snapshot it).

## 2. Renderer

### 2.1 Layer compose — CPU, on-change only

Keep one *active* diffuse (+normal) texture pair, assembled per tile from the three layers by
`art_state`, rebuilt **only for tiles whose state changed that tick** (a handful of texture-region
updates on a destruction event; zero per-frame cost). The lighting pass keeps consuming a single
diffuse+normal exactly as today — no shader changes for layering.

### 2.2 Emissive

`emissive_mask` pixels keep their diffuse brightness regardless of light (add `mask · diffuse` after
the lighting multiply, before ACES). Cheap, exactly the "pixels that always keep their brightness"
Erik described — and the loader has carried optional emissive slots since v1, unused until now.
**Authoring (decided 2026-06-10): automate the bulk, human eyes refine.** The editor seeds the mask
automatically — a luminance/saturation threshold over the dim-lit diffuse finds screens and panels
(they're the bright saturated pixels in otherwise dim art) — then the EMISSIVE paint mode adds and
removes by hand. Bright *dynamic* sources (the cargo-bay warning lamp) are NOT emissive pixels —
they're LightEmitter entities (out of scope here, mission-1 entity pass).

### 2.3 Specular — stored now, consumed later

The `_s` maps ship in the format immediately (they exist; storing is free). Consuming them is a
small lighting-shader feature (per-pixel specular term on the existing light direction) — its own
later step, after the level stands, so art never has to be re-delivered.

## 3. Doors v1 — sim-rendered sliding slabs

The art removed doors **deliberately** — right call: a door is sim state, so the renderer draws it
from state and art can never contradict it.

- **Render:** a dark slab over the door tiles (per-material tint later); *sliding* = draw the slab
  offset by `open_t · tile` into its frame — a one-float animation that reads as a sliding door.
- **State:** `closed ⇄ opening ⇄ open ⇄ closing` with `open_t ∈ [0,1]`. Closed = today's door
  material (movement-passable, flow-sealed, blocks LoS). Open = swap to a `door_open` material row
  (perm 1, light passes, passable) — LoS/flow/light all follow automatically from the material
  system; no special-case code.
- **Trigger v1:** proximity — any unit within ~2 tiles opens it; closes after a short empty delay.
  (Hissing open in a pitch-black corridor is free horror.) Locked/powered doors are later gameplay.
- Movement-while-closed becomes *blocked* once doors can open (today's walk-through-closed-doors is
  a stopgap) — approved 2026-06-10.
- **Atmosphere through open doors is free** (Erik's ask, confirmed cheap): the open-state material
  swap carries `permeability 1`, so pressure, smoke, and gas pour through the moment it opens — no
  extra code. The payoff scene works day one: grenade a room into overpressure, walk up, the door
  slides open, and the smoke billows out at you.
- **Doors are objects, tiles stay materials** (decided 2026-06-10): a small `Door` object on the
  Simulation owns the state machine (`open_t`, timers, its tile span) and drives the material swap +
  slab render; the *physics* presence remains purely material-driven. Whether `Door` later merges
  into a general entity system is open — v1 keeps it a minimal object list.

## 4. Editor v1 — an in-game mode

Built on what exists: cursor→tile resolution (I/J/U keys), the material table, `on_tile_changed`
live cache patching, FieldOverlay previews, the CSV/toml loader. Toggle into EDIT MODE in the
running game (physics keeps running — painting hull while smoke flows *is* the test):

- **ALIGN** — art offset/scale nudge keys + grid overlay → writes `[art.align]`.
- **MATERIAL PAINT** — palette (air, hull, wood/furniture, door, steel, glass, SPACE) on number
  keys, brush size on scroll, click-drag paints material ids; `on_tile_changed` updates the live
  sim per stroke. Furniture is just painting wood-family material over furnished art regions.
- **EMISSIVE PAINT** — auto-seed the mask from the diffuse (threshold pass, §2.2), then the same
  brush adds/removes by hand (screens, panels, strip lights).
- **SAVE** — `tilemap.csv` (canon v2 codes) + `level.toml` + `emissive_mask.png`, with a `.bak` of
  whatever it overwrites. **Undo** — a small snapshot ring of the paint targets (one keypress back).

Explicitly NOT v1: entity placement (vents/fans/lamps/spawns — the mission-1 entity pass), decal
painting, texture editing, multi-floor.

## 5. Asset pipeline for unhcr_vessel_2 (order matters)

1. **Upscale** the ChatGPT layers first (2–4×): Real-ESRGAN / 4x-UltraSharp via the ComfyUI stack
   already in `tools/`. Low-res in → everything downstream inherits it; upscale before normals.
2. **Align** in the editor (§4) — no destructive crop needed.
3. **Normals**: regenerate from the upscaled layers. Laigter with the saved preset
   (`prototypes/space_ship_gpt_pipeline1/laugter_normal_map.laigter`) is the baseline; the ML
   experiment worth one evening: monocular depth (Depth-Anything-class, runs in ComfyUI) → height
   map → import into Laigter (it accepts height input) → normals. Judge both in the lighting demo.
   OpenGL green-channel convention either way (docs/TODO.md item).
4. **Emissive mask** painted in the editor; **materials** painted in the editor; airtight tool +
   240-tick conservation probe certify the result.

## 6. Out of scope here (mission-1 backlog, for the record)

Vents/fans (the ch.04 §4 face-flux primitive — Erik's "gentle smoke removal" + interesting flow),
the flashing-orange warning lamp + screens-as-lights (LightEmitter entity), blood/decal layer
(destruction paint, canon ch.07 §6), rival-faction scripting + bombs, the EVA space band (escape
route + the Gray's exit), laser pointers (trivial ray-engine source), lights-on reveal state.

## 7. Build order (after review, W-style gated steps)

1. **F1** — CSV v2 + migration script + loader version gate (+ both vessels migrated, tests).
2. **F2** — `[art]` schema + align transform in loader/renderer (single bare layer first = the new
   level becomes playable immediately).
3. **F3** — layer mask + on-change compose + destruction/burn hooks (furniture burns away).
4. **F4** — editor mode: ALIGN + MATERIAL PAINT + SAVE (the minimum that replaces hand-editing).
5. **F5** — EMISSIVE paint + render term.
6. **F6** — doors v1 (slab render + state machine + proximity).
7. (later) specular consumption; entity pass; decals.

## Open questions — ANSWERED (Erik, 2026-06-10)

1. SPACE code = **9**. ✓
2. Furniture = **dedicated material row** (lower hp, partial permeability so smoke drifts past
   crates, flammable). ✓
3. Doors block movement when closed — **approved**. (Plain-language version of the question: today
   units walk straight through closed doors as if they weren't there — a stopgap. With doors v1, a
   closed door actually stops you until it opens. Changes pathing feel; Erik OK'd it.)
4. Destroyed-layer reveal: **per tile, only where damage happened**. ✓
5. Editor: **in-game mode** (live physics while painting). ✓

Proposal is GO — build order §7 starts at F1 next working session, after the asset-pipeline
session (upscale + normals, §5) which Erik wants to do hands-on together.
