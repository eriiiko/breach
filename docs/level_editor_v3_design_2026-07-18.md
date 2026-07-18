# Level Editor v3 — design v2 (LOCKED)

**Status: LOCKED 2026-07-18** — Erik's final read approved; critique folded.
`entity_system_design_2026-07-18.md` is the canonical MODEL; this doc is the
authoring VIEW: workflows, tools, UI, file mechanics, quantization.
**Build per entity doc §10 (arc plan); Arc A is cleared to start.**

Historical note: v1 of this doc carried its own registry/wiring/patch-plan
sections; those moved to the entity doc after the Factorio pivot. Superseded
text is deleted, not banner'd — git has the archaeology.

---

## 1. Goal — and the level that defines "done"

Unchanged: **Erik authors the Contested Chip level end-to-end without
hand-editing TOML.** The acceptance level (two teams, two breach sites, the
chip + extraction zones, pens with critters, aquarium + octopus, zombie
crew, doors/sensors/airlock) — full description in v1 of this doc (git) and
mission notes. De_dust analogy stands. With the arc split, the level is
*authorable* at the end of Arc C and *fully runnable* at the end of Arc B.

## 2. Lessons from the great editors (unchanged)

Hammer's FGD → registry-driven UI · LDtk → typed fields build the inspector
· UnrealEd → preview through the real engine · WC3 → power = exposed data
(triggers deferred) · TrenchBroom → direct manipulation + undo everything ·
Mario Maker → instant edit/play flip. Sources in v1 (git).

## 3. Design pillars (amended by critique)

1. **The registry is the editor — and the registry is CODE** (entity §3b):
   palettes, panes, inspectors generate from the imported entity module;
   graceful fallback to last-good `entity_registry.json` + red banner when
   the import breaks.
2. Instances are class + overrides, addressed by **mandatory ids** (entity
   §3a).
3. See it to design it: live baked preview + entity sprites + play-from-
   editor (§8).
4. The editor writes data; the game implements behavior. **`level_lib` is
   the single data layer** (entity §3c) — the editor is its client; the
   bespoke writeback paths are ported onto it and deleted.
5. Keyboard-first survives the panes.
6. **Undo is a single transaction log** of compound operations (grid delta +
   entity delta per user action) — replacing the per-domain rings. Every
   operation class joins it in the patch that introduces it: placement,
   moves, paints, zone paints, wires, tags, inspector field edits.

## 4. Units of measure — the exact quantization rule (critique-hardened)

Author-facing lengths are meters; tiles are derived. The rule, made exact:

- Resolution is stored as **integer `tiles_per_m`** (base = 3), not the
  float `tile_size_m` (the shipped `0.333` ≠ ⅓ bug: 3 × 0.333 = 0.999 m
  made the 1 m-door-vs-1 m-marine comparison flip by conversion direction).
  Migration: existing `tile_size_m = 0.333` levels load as exactly
  `tiles_per_m = 3`; the loader keeps accepting `tile_size_m` for other
  values, converting through Fraction.
- **`tiles = floor(length_m · tiles_per_m + ½)`** evaluated in exact
  arithmetic (Fraction or scaled ints) — round-half-up, never Python's
  banker's `round`, never a float division whose 15th digit decides a tie.
- **Quantize once at base resolution, then replicate**: `--res N` scales
  already-quantized tile values by the integer factor (matching
  `_upscale_level`'s existing semantics for spawns/footprints/lights).
  Door spans, sensor radii, and area discs all follow this order — never
  re-derived from meters at the scaled resolution (the two orders disagree:
  round(0.5 m · 6) = 3 ≠ round(0.5 m · 3) · 2 = 4).
- Registry `length_m` fields display meters; the inspector shows the
  snapped tile result (same pattern as the filter node's snapped τ).

## 5. Zones — matter-first, now with the binding specced

Two zone classes only (breach_site, extraction_zone), painted masks in
`zones.npy` — unchanged. Critique additions:

- **Binding:** each zone is an `[[entity]]` instance (id, class, faction,
  fields) carrying `zone_id` = its integer paint id. The npy grid holds
  paint ids; the instance holds everything else. Validators: every painted
  id has exactly one instance; every zone instance has ≥ 1 painted tile
  (warn); duplicate `zone_id` is a load error.
- Deleting a zone instance prompts to clear its paint; orphaned paint ids
  are a validator warning, not a crash. Painting id A over id B just
  shrinks B (warn at 0 tiles).
- Zone paints are transactions in the undo log (§3.6).
- `zones.npy` joins `_upscale_level` replication (like water) — `--res`
  must not shape-mismatch or drop zones.
- **Breach-site roster (format home, critique):** breach_site instances
  carry `roster = [[unit_type, count], ...]` where `unit_type` is the
  *unit-system* vocabulary (units are not registry entities — entity §3e);
  spawn positions randomize inside the zone from the level seed (the ML
  variation hook).

## 6. Doors, sensors, logic — see the entity doc

Canonical fields, semantics, tick behavior: entity doc §§4–7. Editor-side
obligations only:

- **DOOR tool:** places a door entity on a wall run (snap, default 1.0 m,
  drag to resize; width warnings per footprint, no hard minimum). The
  entity is authoritative; its `MAT_DOOR` tiles are written to the grid
  **immediately on placement** (not at save) so the live preview and
  physics-adjacent validators see them; the compound op is one undo
  transaction.
- **Sensor placement** sets the body tile and the `sample_tile` offset
  (rendered as a small arrow); the editor refuses a solid sample tile at
  placement (it may become solid later in play — that's physics).
- **Legacy migration is explicit, never a save side effect:** opening an
  old level with `[[spawn]]`/`[[light]]`/painted doors keeps them as-is;
  the loader hard-errors on *mixed* legacy+new forms in one file; the
  one-shot migration tool (Arc A) converts a level in place (including
  grouping painted MAT_DOOR runs into door entities) with a `.bak`.
  Ctrl+S on an unmigrated level saves in its legacy form.
- Managed-block writeback (via level_lib): multi-family replace is atomic
  (write temp + rename), so a crash mid-save can't leave both forms.

## 7. Painting power (unchanged from v1)

Magic wand (enclosure fill + same-code select) for materials, AIR, vacuum,
zones · `air_init.npy` with the hull-leak validator (fill escaping to the
border = leaky room, warn don't paint) · `boundary = "space" | "ambient"`
level field set in the level-properties pane.

## 8. The UI — panes + the critique's UX specs

Layout as mocked (top bar / tool rail / canvas / tabbed palette / inspector
/ status bar), with these now-explicit specs:

- **Multi-select is load-bearing** (tags need it): box select + shift-click
  add + select-by-class; "assign tag to selection"; **clump copy/paste**
  preserving internal wires (re-id on paste, external wires dropped) — the
  poor man's prefab, enough for several airlocks.
- **Wire tool: two-click** (click source, navigate freely — pan/zoom stay
  live — click target, Esc cancels). Never a drag gesture. The LOGIC
  overlay defaults to wires touching the current selection with a show-all
  toggle; wires to `tag:` targets render to a tag badge, not fanned to
  every member.
- **Bulk placement** (30 zombies): official answer is a `level_lib` snippet
  (scatter helpers ship with it); the editor offers only place-one.
- **PLAY (F5), fully specced:** saves everything (tilemap, entities, zones,
  air/water npy) to `levels/_editor_scratch/<name>/` (gitignored), reusing
  the existing baked PNGs when the grid is clean (no full re-bake per F5);
  launches `[sys.executable, "main.py", "--level", "_editor_scratch/<name>"]`
  (never bare `python` — the documented machine footgun); the scratch
  folder is deleted on subprocess exit and editor quit. Loader accepts the
  `_editor_scratch/` path form.
- **Icons:** SVG sources in `art/entities/icons/` + **committed PNGs**
  rasterized by `tools/rasterize_icons.py` (its rasterizer dependency is
  dev-only, needed only when icons change; a test asserts PNG freshness).
  A class without an icon renders a generated color chip + class initial —
  permanent fallback, never an error.
- Status bar: mode · cursor tile · validator summary · unsaved dot ·
  registry-import banner slot (entity §3b).

## 9. Patch plan — moved

The build plan is the entity doc §10 arc split (Arc A foundation → physics
close-out → Arc B logic → Arc C editor UX), decided by Erik at critique.
This doc's obligations land as: format/loader/migration/level_lib in Arc A;
wire tool + LOGIC overlay alongside Arc B; everything in §8 in Arc C.

## 10. Explicitly not v3 (unchanged)

Multi-floor · trigger scripting · AI tilesets (levels-w1 P6) · sliding-door
animation arc · in-editor embedded sim · decals · campaign structure · full
prefab library (clump copy/paste ships instead).

## 11. Decisions log

Rounds 1–2 (2026-07-18): matter-first zones · entity doors, width warnings
not errors · rules re-homed to stack 2 · both spawn models · meters-first
(exact rule now in §4) · doors-as-entities.
Critique round (2026-07-18): everything in this revision + the four Erik
rulings recorded in entity doc §9 (arc split; fail-deadly + alive idiom;
v1 signal-only; units out).
