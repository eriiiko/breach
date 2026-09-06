# Props & vegetation (3D)

> **Depends on:** engine/16 (entity system), engine/15 (level format & authoring),
> engine/06 (temperature & fire), engine/03 (material system), engine/09 (rendering),
> the lit-3D-in-world-RT seam (today: `renderer/marine_shader.py` +
> `renderer/unit_model_renderer.py`; extracted to `renderer/lit3d.py` in P1).
>
> **Status: 🧪 prototype-only, doc v2.1 — IMPLEMENTATION GO** (spike
> `prototypes/prop_spike/`, 2026-09-06 — perspective look blessed; v2
> 2026-09-07: adversarial critique folded in — 32 findings, 3 blockers, all
> addressed on paper; **v2.1: P0 top-down verdict PASSED by Erik + final §6.1
> rulings — walk-through foliage, 1×1 flammable stamp, charred → v2**).
> Indexed in `graphics/README.md` with the arc's first commit.

## 1. Vision

Fill interior spaces — above all the exotic-garden rooms — with stylized 3D
vegetation and, later, other 3D props. Reference look (Erik, 2026-09-06): soft
rounded canopy lobes with bright leaf-cluster detail over a darker mass
(mobile-game screenshot / Elden Ring foliage family). Atmosphere first;
gameplay coupling minimal but real (burn, block). Props also ease level
generation: rooms furnish themselves from a palette of placeable assets.

The tree is deliberately the PILOT for the wider "static 3D geometry in the
world RT, lit by the baked light field" seam that 3D walls need next — and the
critique sharpened this: the pilot's first patch IS the extraction of that
seam (§4.3), so walls later extend a real seam, not a metaphor.

## 2. What the spike established (2026-09-06/07)

- **Generated beats packs.** `prototypes/prop_spike/treegen.py` (pure numpy →
  triangle arrays): recursive skeleton (depth 3), smooth icosphere-blob canopy
  (ellipsoid normals), per-lobe + canopy-wide brightness gradients, dense leaf
  tufts (flutter weight baked in vertex-color ALPHA), `generate_palm`
  (15–20 serrated fronds), decor (`flowers` / `fruit`), palettes
  `green`/`autumn`/`exotic`. Seeded → deterministic. Sway prototyped
  (vertex-shader bend-by-height² + alpha-weighted flutter).
- **Tri budget — MEASURED at P2 (2026-09-07, this box, 16 seeds per row;
  supersedes the F26 estimate):**

  | generator (h) | tris (min–max, median) | VRAM/model | gen time (median) |
  |---|---|---|---|
  | `tree` smooth, no decor (2.2 m) | 6.4k–15.5k, **11.6k** | ~950 KiB | **314 ms** |
  | `tree` smooth + flowers (2.2 m) | 6.6k–15.8k, 11.8k | ~970 KiB | 330 ms |
  | `tree` smooth + fruit (2.2 m) | 6.5k–16.9k, 12.6k | ~1.04 MB | 358 ms |
  | `tree` faceted (2.2 m) | 1.0k–2.3k, 1.7k | ~140 KiB | 61 ms |
  | `palm` (2.8 m) | 0.6k–0.8k, 0.7k | ~55 KiB | **9 ms** |

  VRAM is exact, not estimated: non-indexed position+normal+colour =
  **84 bytes per triangle** (3 × (12 + 12 + 4)). The 12-look demo garden in
  `tools/lighting_demo.py` totals **114k tris / 9.1 MB / 3.2 s** to generate —
  i.e. a room's worth of distinct looks costs a **~3 s one-off load hitch**,
  which is why `StaticPropRenderer`'s cache is warmed up front, not on first
  draw. P2 already took the cheap 40% (hand-rolled `_cross3`/`_norm3` instead
  of `np.cross`/`np.linalg.norm` in the per-tuft hot loop) and cut fruit to a
  20-face primitive. If it still bites the ladder is unchanged: index the
  mesh, instance the tufts, or cache generated arrays to disk.
- **Crown size correction (P2):** at the shipped 48 world-px/tile and
  0.333 m/tile (**144 px/m**), a 2.4 m tree is **~7 tiles tall with a ~5-tile
  crown** — NOT the "~3×3 crown" §4.1 assumed (that would be a ~1 m shrub).
  Nothing in the design breaks (the stamp is 1×1 and the crown is visual
  only), but placement doctrine (#50) and the P5 dressed garden should plan
  around ~8-tile spacing, which is what the demo garden uses.
- **P0 top-down truth (critique F20/21 — the game camera is straight-down
  ORTHO, `unit_model_renderer.py:21-32`, with ACES and NO MSAA):** re-rendered
  through that geometry + tone-map in the spike. Result: Kenney trees read as
  flat discs (dead); **treegen trees survive** — per-lobe shading + tufts keep
  volumetric read; palms excel; a 20° mesh tilt adds a trunk hint (optional
  dial); decor reads as confetti from above (cluster larger before P2's feel
  gate). **Erik's verdict on the top-down look = the P0 gate.**
  *Decor confetti FIXED at P2:* `_emit_decor` now places 0–2 CLUSTERS per
  canopy blob (a tree has 8–27 blobs, so most carry none) of 3–7 elements
  each, ~2× the old element size, blossoms hugging the surface and fruit
  sitting proud of it at 1.12× the blob radius with a relaxed upward bias —
  a strictly lower-biased fruit tree read *bare* from straight above.
  Verified by re-rendering top-down and in perspective.
- **Kenney Nature Kit** (CC0) at `assets/models/props/kenney_nature_kit/`
  (license file included) stays useful for NON-vegetation props. GOTCHA:
  raylib 5.5 cgltf rejects its 2020-era GLBs — use `OBJ format/` variants.
- No asset files needed for generated props: meshes built from numpy at load.
  **Mesh-ownership contract (F22): `static_props.py` copies vertex data into
  raylib-owned memory before `upload_mesh` (or never unloads) — the spike's
  ffi.from_buffer-into-numpy trick corrupts the heap if `unload_model` runs.**

## 3. Design decisions (Erik's rulings, 2026-09-06)

1. Props are for looks, with minimal but real sim coupling: they burn and
   block. Nothing else v1 (wind-blocking a maybe-later).
2. Burning rides the existing fire path; the RENDERER derives the look.
3. Blocking rides existing tile flags — no new collision system.
4. Reusable, extensible prop system — never one-off code per prop.
5. (2026-09-06 late) No shadows/light occlusion yet; no vision interaction yet
   (future: aliens-hide-in-bushes via a "covered" tile mask); multi-tile → but
   see F17 revision in §4.1; wind sway IS wanted; 2.5D smoke constrains nothing.

## 4. Architecture (v2 — critique-hardened)

### 4.1 Prop = an entity type (placement & data)

A `prop` entity in the engine/16 registry (registry-in-code, `[[entity]]`,
written only through `level_lib`, one serializer). Fields — kinds chosen for
digest hygiene (F10: **a prop's look is not digest material; its footprint
is**):

```
x, y          KIND_INT (tile anchor — the trunk tile, like door.py) [synced]
material      KIND_ENUM over [materials.*] row names             [synced]
stamp_tiles   KIND_INT square stamp side, v1 = 1 (trunk only);
              2×2/3×3 etc. are the designed extension            [synced]
kind          KIND_STR  "generated" | "model"                    [not synced]
generator     KIND_STR  "tree" | "palm"                          [not synced]
seed          KIND_STR  (art-only; parsed int)                   [not synced]
palette       KIND_STR                                           [not synced]
style         KIND_STR  "smooth" | "faceted"                     [not synced]
decor         KIND_STR  "" | "flowers" | "fruit"                 [not synced]
height_m      meters, RENDER-ONLY (non-synced — nothing sim-side
              depends on it since foliage is walk-through);
              loader-capped (F24: ortho cam budget ≈ 20 tiles)   [not synced]
model         KIND_STR relative path under assets/models/props/ ;
              loader validates existence + extension (F18)       [not synced]
```

- **One `material` row replaces the earlier `blocking` + `fuel` fields (F7 —
  BLOCKER fix):** both are columns of a `[materials.*]` row. v1 adds an
  APPENDED `foliage` row — **Erik's spec (2026-09-07): fully walkable (no
  collision), no wind/vision interaction (permeability 1.0), flammable, fuel ≈
  2× the furniture row** (furniture ≈ 30–35 hp per Erik's bonfire tests →
  foliage ≈ 60–70; verify the actual row at implementation). Material ids are
  positional and validated contiguous — **rows are append-only** or every
  level re-materializes.
- **Footprint (F17, revised by Erik 2026-09-07):** the STAMP is 1×1 — the
  trunk tile is the only flammable/material tile. The ~3×3 crown is purely
  visual (the model just draws bigger than its tile). `stamp_tiles` exists so
  2×2+ stamps are a value change, not new machinery. The renderer always
  draws ONE model per prop.
- **Height (F16):** meters-first, quantized/validated once at load — never
  "tiles" (resolution-dependent), never raw Q16.16 in the file.
- Registry `choices` tuples are **append-only** (enum digests hash the index).
- **Ordinal stability (F11):** editor/`level_lib` prop writeback APPENDS to
  the `[[entity]]` array; an ordinal-stability test joins the P3 gate.
- Registering the class changes `registry_content_hash()` but no golden (F12
  — recorded so nobody over-rebaselines).
- P6 editor support is NOT free (F5): palette icon (C8 SVG→PNG pipeline) +
  in-editor preview + footprint ghost are real deliverables.

### 4.2 Sim coupling (load-time stamps, zero per-tick logic)

- **The stamp primitive is named (F9):** `stamp_prop_tiles()` beside
  `stamp_door_tiles()` (`door_system.py:70-118`), called in the same
  `GameMap.__init__` slot — between tilemap fill and `_update_caches()` — so
  field seeding runs post-stamp and t=0 conservation is trivial. Same
  validation shape (OOB / overlap / no vacuum / permitted base material);
  props may not overlap door spans; ordering: doors stamp first.
  (`seal_tiles` is the WRONG tool — it rejects non-solid materials and
  evacuates gas.)
- **Burn visuals: DEFERRED to v2 entirely (Erik 2026-09-07).** v1 props do
  not change appearance when their tile burns — the existing fire/smoke
  overlays render over them, which already reads as "the tree is on fire".
  The v2 upgrade is patchable and render-only (F8's analysis stands for it:
  `destroy_wall` erases tile memory, so charred needs a render-side monotonic
  latch — Erik's sketch: grayscale trunk, no leaves; resets on reload =
  accepted gap unless promoted to a synced flag, the door-pattern day).
- Props multiply burn-through-mintable tiles (F13): bounded, not cyclable;
  noted for the #54 accounting arc.
- **Airtight & pathfinding: NON-ISSUES in v1** — foliage is fully walkable
  and permeability-1.0, so reachability, sealing, and A* are untouched (F14
  collapses); a P3 test still asserts the stamp exists and the tile burns.
- Baker non-interaction (F6): props are not in `tilemap.csv`; the baker paints
  floor art under them; the editor's baked preview shows nothing where a prop
  stands — expected, documented.

### 4.3 Render: lit3d extraction + StaticPropRenderer

- **P1 extracts the seam (F1 — BLOCKER fix):** `renderer/lit3d.py` takes the
  shared GLSL (`_COMMON_GLSL`, light-field unpack/composite/tone-map), the
  `LightFieldCtx` dataclass, and `make_camera` out of
  `marine_shader.py`/`unit_model_renderer.py`; both marines and props consume
  it. Gate: the marine's composed shader source is byte-identical
  (string-equality test) + existing renderer tests. The top-down ortho camera
  moves here and is built UNCONDITIONALLY (F2) — no longer gated on
  `use_3d_units`.
  **P2 addition:** the light-field SAMPLE/UNPACK block (world UV → `tex_a` /
  `tex_b` → `incoming_rgb`, `light_dir_2d`, `L`, `N`) also moved into lit3d as
  `_FIELD_SAMPLE_GLSL`, the prop shader being its second consumer. Same
  discipline: the block is the marine's own text, placeholder-substituted, so
  `MARINE_FS` stays byte-identical and the P1 gate still holds.
- `renderer/propgen.py` — the promoted generator. **Public signature frozen at
  promotion (F27)**: generators normalized to height 1.0 with draw-time
  uniform scaling **if** tuft counts can be made scale-stable, else `height`
  joins the cache key; registry fields = exactly the generator parameters.
- `renderer/static_props.py` — model cache keyed
  `(generator, seed, palette, style, decor, height_bucket, burn_state)` +
  draws **in the units' shared `begin_mode_3d` pass** (F23/F25): the slot is
  immediately with `_draw_units_world`, one 3D pass for units + props (one
  batch flush; shared depth so occlusion is consistent). `compose_world`
  grows a `props=` keyword like `doors=` — loader→sim→renderer hand-off is
  real plumbing, listed in P3.
- **Canopy-over-marine policy (F23):** v1 accepts canopies occluding units
  beneath them (they are volumetric obstacles); if play shows it hurts
  readability, the follow-up dial is a canopy alpha-dither when a unit stands
  under the footprint — decided then, not built now.
- Burning glow arrives FREE via `fire_lights` (F4) — a burning prop tile is a
  fire tile; tune there, never add a prop light source. `frame_lights` stays
  the assembly seam; no new light kinds.
- Sway (P4): the shader from the spike; wind sampled from the **TAMED** wind
  (F3) — `gas_detail.py::pack_dynamics`' smoothed/gain-limited product (reuse
  the helper or the texture), NEVER raw `gmap.wind_x/wind_y` (fire-spiked
  `-grad(P)`, unusable as velocity). Pack models draw with `u_sway = 0`.
  **P4r (Erik's ruling 6.1/5, 2026-09-07): REAL WIND ONLY.** `idle_wind`
  defaults to **0**, so calm air draws the rigid P2 mesh and any motion the
  player sees is a true statement about the atmosphere. `lighting_demo`'s
  `demo_breeze` (+ its `--no-demo-wind` opt-out) is deleted; the tool makes
  wind the way the game does — a detonation (R + left click, or the headless
  `--detonate-at-tick X,Y,F`), routed through the canonical payload executor
  (`simulation.payloads.execute_payload` with `grenade_frag` →
  `[payloads.frag_standard]`, writing through the sim's `edit_queue`), whose
  pressure spike tames into a wind gust that bends the canopy and decays.
- Aliasing risk (F21): no MSAA in the world RT; tuft crawl under camera motion
  is checked at P2's HUMAN-TEST; mitigation ladder if it bites: fewer/larger
  tufts at authored density → tuft fan geometry → (last) RT supersampling.

## 5. Phasing (v2 — critique's split adopted)

| P | Content | Gate | Mode |
|---|---------|------|------|
| P0 | top-down truth render through real camera + tone-map | **PASSED (Erik, 2026-09-07)** — straight-down works; no mesh tilt; a tilted *world view* is a future whole-renderer question, out of scope | done |
| P1 | `renderer/lit3d.py` extraction (GLSL + LightFieldCtx + make_camera), marines byte-identical | shader-equality test + renderer tests green → auto-merge | subagent, Sonnet 5 |
| P2 | `propgen.py` promotion (frozen signature, owned-memory upload, measured budget) + `static_props.py` in the shared 3D pass, exercised via `tools/lighting_demo.py`; decor clustered for top-down | HUMAN-TEST (Erik: look in real RT/lighting) | subagent, Opus 4.8 |
| P3 | prop entity row + `foliage` material row + `stamp_prop_tiles` (1×1) + `level_lib` append + ordinal-stability test + burn test | digest/golden + new tests green → auto-merge | subagent, Sonnet 5 |
| P4 | wind sway on TAMED wind | HUMAN-TEST (feel) | subagent, Opus 4.8 |
| P5 | editor UX (icon, preview, stamp ghost) + dressed exotic garden | HUMAN-TEST | subagent, Opus 4.8 |
| v2 | burn/charred visuals (render latch) · 2×2+ stamps · covered-tile hiding · shadows/occlusion | — | future arc |

## 6. Rulings (Erik, 2026-09-06) — unchanged from v1

No shadows yet · no vision yet (future "covered" mask for hiding) · multi-tile
deferred **as a mechanic** (v1's square footprint is a stamp shape, not
multi-tile entity machinery) · wind sway IN (§4.3) · 2.5D smoke unconstrained.

### 6.1 Final rulings (Erik, 2026-09-07) — all resolved, implementation GO
1. **P0 PASSED**: straight-down look approved as-is (no mesh tilt; a slightly
   tilted world view is a future renderer-wide question — "moving toward 3D
   makes such choices easier").
2. **`foliage` row**: fully walkable, no wind/vision/movement interaction,
   flammable, fuel ≈ 2× furniture (Erik: furniture ~30–35 hp from the bonfire
   tests → foliage ~60–70; verify at implementation).
3. **Charred/burn visuals → v2 wholesale** (patchable, render-only; sketch:
   grayscale trunk, no leaves).
4. **Stamp 1×1** (trunk tile, the only flammable); ~3×3 crown is visual only;
   `stamp_tiles` keeps 2×2+ a value change later.
5. **Sway is a SIGNAL, never decoration** (2026-09-07, after the P4 HUMAN-TEST
   — supersedes P4's "a sealed room should still breathe" choice): *"We're in
   a spaceship — leaves should be TOTALLY STILL unless there is actual wind."*
   Shipped `[render.props] idle_wind = 0` (the dial survives for planetside /
   debug experiments); `lighting_demo.demo_breeze` and `--no-demo-wind`
   deleted. Moving foliage now means the air is genuinely moving there — a
   blast front, a hull breach, a running vent — which makes the canopy a free
   *readout* of the atmosphere solver rather than ambient noise. Verified
   headlessly at P4r: two frames of a calm garden are pixel-identical in the
   canopy, and frames straddling a scripted detonation show the crowns
   displace and then settle back.

## 7. Systems

**Existing canonical systems this design must use:** entity system (engine/16),
material table (one appended `foliage` row — never per-prop ifs), the
`stamp_door_tiles` load-slot pattern (sibling stamp), level data layer
(level_lib append / level_loader validate), A* mobility view, airtight lint
(post-stamp), WorldComposite/`compose_world`, the (extracted) lit3d seam,
`fire_lights` + `frame_lights`, `gas_detail` tamed wind, tile inspector
(prop debug rides `pack_hover_readout`), map-editor pattern + undo seam,
`tools/lighting_demo.py` as the P2 instrument, baker (non-interaction noted).

**New systems (draft rules → project CLAUDE.md at implementation):**

- *Lit-3D seam* — `renderer/lit3d.py`: THE shared light-field GLSL
  (`_COMMON_GLSL` + `_FIELD_SAMPLE_GLSL`), `LightFieldCtx`, and top-down
  `Camera3D` for everything 3D drawn in the world RT (marines, props, future
  walls); a second copy of any of them is the bug.
- *Prop generator* — `renderer/propgen.py`: THE procedural prop/vegetation
  geometry source. Pure numpy in, triangle arrays out; imports nothing from
  `simulation` and is never imported by it; seeded, render-only (Q16.16
  render exemption) — output never reaches sim state. New flora = a generator
  function + a `PALETTES` row here, never inline mesh code elsewhere.
- *Static props* — `renderer/static_props.py`: the ONLY path that draws
  placed 3D props (model cache + draw in the shared 3D pass inside
  `compose_world`). No prop render state ever lands on a sim entity (digest!).
- *Prop tile stamps* — `stamp_prop_tiles()` beside `stamp_door_tiles()` in
  the `GameMap.__init__` load slot: a prop's blocking + fuel are ONE
  `[materials.*]` row (never a per-prop if, never a new per-tile flag); the
  stamp is load-time only, props never tick, and a prop's look fields are
  non-synced kinds so art edits never move a digest.
- *Prop assets* — `assets/models/props/<pack>/`: every pack ships its license
  file (CC0 or equally clean); OBJ preferred (raylib 5.5 cgltf rejects
  2020-era GLBs); `model` paths validated at load, not first draw.
