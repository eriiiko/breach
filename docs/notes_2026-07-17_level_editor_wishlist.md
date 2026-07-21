# Notes dump — 2026-07-17 (evening): level-editor wishlist

Raw capture of Erik's editor direction, same convention as the topic-backlog
notes: capture only, not canon. Home for the eventual design:
`level_editor_and_format_v2_proposal.md` + engine/15. The AIR paint mode
already has an agreed feel (2026-07-12 session); the items below extend it.

---

## 1. Entities as classes + instances (the OOP model of level making)

> "I'd like to treat level making the same way we think about object oriented
> programming: we have classes of stuff we can place, and we can possibly
> modify the instances to our likings." — Erik, 2026-07-17

Direction: menus/palettes of placeable **entity classes**; placing one creates
an **instance** whose properties can be overridden per-placement.

What we already have (and should generalize rather than replace):
- `[[spawn]]`, `[[light]]` (incl. beacons), `[water]` in level.toml are
  already instances with per-instance properties — but each got its own
  bespoke editor mode (SPAWN, F6, F7) and its own writeback helper.
- Weapons/units/materials are class-defined in config.toml — that IS the
  class registry pattern, just not exposed to the editor.

Open design questions for the future doc:
- **Where do entity classes live?** A prefab/archetype registry (TOML) that
  both the game and the editor read — e.g. `entities.toml` defining classes
  (light kinds, spawn archetypes incl. zombie variants, future: doors,
  animal-pen, furniture props) with default properties; level.toml instances
  reference a class + overrides. (Mirrors materials/weapons config exactly.)
- **One generic PLACE mode** driven by the registry (menu of classes,
  per-instance property panel) instead of a new hotkey mode per entity type.
- Instance property editing UI: select-placed-entity → edit panel; keep the
  managed-block writeback (one table per instance, class + overrides only).
- Ties to the unit-variants design (`unit_variants_design_brainstorm.md`) —
  runner/brute zombies are exactly "class + stat overrides".

## 2. Magic-wand / bucket-fill painting (Photoshop language)

> A tool that paints ALL tiles within a constrained area with whatever we're
> painting with — and for air specifically, fill all empty space within a
> selection or perimeter/enclosure. Same for vacuum.

- The primitive exists: WATER F7 already bucket-fills an enclosed region
  (`flood-fill` on the tile grid). Generalize it: **magic wand = flood fill
  bounded by solid tiles (or by same-tile-code region), payload = whatever
  the active brush paints** (material code, air, vacuum, water depth).
- Two selection flavors worth distinguishing in the design:
  1. *Enclosure fill* (flood from a seed point, stops at walls) — the
     water-mode behavior, right one for AIR/vacuum.
  2. *Same-code select* (contiguous tiles of the code under the cursor) —
     classic magic wand, right one for repainting a floor region's material.
- For AIR: seed-click inside a sealed hull → whole room gets air; the
  hull-leak validator (2026-07-12 idea) falls out for free — if the fill
  escapes to the map border, the room leaks; warn instead of paint.

## 3. AIR paint mode (previously agreed, restated for one design pass)

Agreed feel (2026-07-12): paint an initial air-mask mirroring WATER mode;
DECOUPLE floor-visual from atmosphere; default vacuum, air only inside sealed
hull. `air_init.npy` seed mirroring `water_init.npy`; engine seeds ambient
gas N inside the mask, 0 outside. Bonus: O2-rich pockets, vacuum rooms,
pre-pressurized scenarios.

**Design implication of tonight's additions:** AIR mode should ship WITH the
enclosure-fill tool (painting air tile-by-tile is the wrong UX), and the
level-format rev should anticipate the entity-registry (§1) and the
boundary-condition field (notes_2026-07-17_topics_backlog.md Topic 4) so the
format is revved once, not three times.

## Sequencing note

One design doc covering §1–§3 + the BC level-format field, adversarially
critiqued, then patches — an arc, not a chat. The wall-burst fix + tiny
editor fixes (zero-spawn sandbox, mojibake) landed separately tonight.
