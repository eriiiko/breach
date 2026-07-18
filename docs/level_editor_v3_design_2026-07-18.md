# Level Editor v3 — design (DRAFT, in design phase)

**Status: DESIGN PHASE — not approved for build.** We iterate on this doc until
Erik explicitly says we're happy and ready to build. Then: adversarial critique →
patch plan locked → build.

Companions: `notes_2026-07-17_level_editor_wishlist.md` (raw capture this doc
absorbs) · engine/15 (shipped editor v1/format v2 canon) ·
`notes_2026-07-17_topics_backlog.md` Topic 4 (boundary conditions).

---

## 1. Goal — and the level that defines "done"

The editor determines the ceiling on level complexity, and therefore on mission
design and on how rich the ML training distribution can be. v3's goal is not
"more tools"; it is: **Erik can author the first ML-track mission level,
end-to-end, without hand-editing TOML.**

### The acceptance level: "Contested Chip" (Erik, 2026-07-17)

A large derelict ship carrying a **data chip**. Two opposing teams board
simultaneously at **two different breach sites**; the objective is contested —
secure the chip and carry it to your own **extraction point**. Aboard:

- **Animal pens** holding critters of varying aggression; pens can open by plan
  or by damage (chaos injection).
- An **aquarium with an octopus** (stretch — rides the procedural-animation
  track, the editor just needs to place it).
- The ship's original crew, now **zombies** — a third faction hostile to both
  teams and the wildlife.
- **Infection rule:** any unit (soldier or animal) killed BY a zombie rises as
  a zombie. Killed by anything else stays dead.

The de_dust analogy: two spawns, authored geometry, lights, doors, one
objective, competitive symmetry-of-opportunity (not necessarily of layout).
When this level can be built in the editor and played (by humans or agents),
v3 is done.

## 2. What the great editors teach us

Surveyed 2026-07-18 (sources in §12). The ones that earned lasting praise, and
the transferable lesson from each:

| Editor | Why it was loved | The lesson for us |
|---|---|---|
| **Valve Hammer / Worldcraft** (Half-Life, CS — de_dust itself) | The **FGD entity system**: game ships a declarative file defining entity *classes* + typed key-values; editor renders palettes & property sheets from it. Game and editor never drift. | Erik's "OOP level making" is exactly FGD. Our `entities.toml` = the FGD. Editor UI is *generated from the registry*, never hand-coded per entity type. |
| **LDtk** (Dead Cells' designer; the best modern 2D take) | Entity definitions with **typed custom fields** (ints, enums, paths, entity-refs) + constraints; **auto-tiling rules**; aggressive UX polish. | Typed fields with defaults/ranges in the registry → the inspector pane builds itself. Our autotile baker already follows its spirit. |
| **UnrealEd** | Rendered the level **with the game engine itself** — WYSIWYG was exact, iteration real-time. | We already share pyray + the baker with the game; keep hard-committing to "the preview IS the game's view". |
| **Warcraft III World Editor** | Data + **trigger** editing accessible to non-programmers → DotA. | The power ceiling comes from *exposing game data*, not editor features. (Triggers themselves: explicitly NOT v3 — noted in §10 as the far-future ceiling-raiser.) |
| **TrenchBroom** (modern Quake) | Praised for one thing: **direct manipulation + undo everything** — vs the Radiant clones' mode-heavy UX. | Every new tool must join the existing undo ring from day one; selection/move/delete must feel physical. |
| **Super Mario Maker** | The **instant play/edit flip** (and publish-gated-on-completion). | One-key play-from-editor (§8) is v3's killer feature for a physics game: paint air, press play, watch it vent. |

## 3. Design pillars

1. **The registry is the editor.** One `entities.toml` defines every placeable
   class; palettes, panes, inspectors, and writebacks derive from it. Adding an
   entity class = data edit, zero editor code (Hammer/LDtk lesson; extends the
   engine/15 §1 "no tool carries its own vocabulary" rule from materials to
   entities).
2. **Instances are class + overrides.** A placed entity stores its class id +
   only the fields that differ from class defaults (small diffs, readable TOML,
   retunable classes ripple into every level).
3. **See it to design it.** Live baked preview stays; entities render as their
   in-game look where cheap (lights already do), sprite/icon otherwise;
   one-key play-from-editor closes the loop.
4. **The editor writes data, the game implements behavior.** Pens, infection,
   objectives are *game* systems; the editor only places/configures them. v3
   ships the format + UI for all of §1's level, even where the game system
   lands in a later patch (chip logic yes; octopus no).
5. **Keyboard-first survives the panes.** Current hotkey modes stay; panes are
   for discovery, properties, and browsing — not a mouse-only rewrite.

## 4. The entity registry — `entities.toml`

New repo-level file (beside `config.toml`, hot-reloadable same way), read by
game AND editor:

```toml
[entity.marine]                      # class id
category = "units"                   # palette pane grouping
sprite   = "art/entities/marine.png" # editor icon / in-game sprite hook
[entity.marine.fields]               # typed fields -> inspector rows
team     = { type = "faction", default = "team_a" }
name     = { type = "string",  default = "" }
footprint= { type = "int",     default = 3, min = 1, max = 6 }

[entity.critter_snapper]
category = "creatures"
[entity.critter_snapper.fields]
aggression = { type = "enum", options = ["docile", "territorial", "feral"],
               default = "territorial" }
infectable = { type = "bool", default = true }

[entity.data_chip]
category = "objectives"
[entity.data_chip.fields]
score_value = { type = "int", default = 1 }
```

Field types v3 needs: `int`, `float`, `bool`, `string`, `enum`, `faction`,
`tile_ref` (a coordinate), `entity_ref` (link to another instance — pens
reference their gate). Constraints (`min`/`max`/`options`) drive inspector
widgets and load-time validation.

**Instances in level.toml:**

```toml
[[entity]]
class = "critter_snapper"
x = 41
y = 17
aggression = "feral"        # override; docile stays class-default elsewhere
```

**Legacy tables.** `[[spawn]]` and `[[light]]` become *aliases*: the loader
maps them into the same runtime instance model (`spawn` → class per team,
`light`/beacon → light classes). Existing levels load unchanged; the editor
WRITES the new form; a one-shot migration tool converts old levels at leisure.
`[water]` stays as-is (it's a field seed, not an entity — same for air, §7).

## 5. Zones — region entities

Some things are areas, not points. A **zone** is an entity whose "position" is
a painted tile mask (stored like `water_init.npy`: one small `zones.npy` id
grid, or per-zone RLE in level.toml — decide at critique). Zone classes v3
ships:

- `breach_site` (faction field) — where a team boards; spawn-cluster anchor.
  ("Two opposing teams from two different breach sites" = two of these.)
- `extraction_zone` (faction field) — carry the chip here to win.
- `animal_pen` (fields: `gate = entity_ref` to its door, `auto_open_on_damage
  = bool`) — the pen region groups its creatures; the gate opening (by plan or
  damage) releases them.
- Future, format-ready but no v3 UI: trigger regions, patrol areas, ambient
  sound zones.

Zones paint with the same brush/wand tools as materials — one interaction
model everywhere.

## 6. Objectives + factions — the game-side contract (architectural prep)

The editor places these; the *rules* are two small game patches that can land
independently (sequenced against the priority ledger — physics close-out still
owns stack position 1):

- **Objective v1 — carry & extract:** `data_chip` spawns at its tile; walking
  over it picks it up; carrier death drops it; a carrier inside their faction's
  `extraction_zone` scores/wins. Deliberately minimal — it is also exactly the
  ML reward function for the first training mission (`get_reward` hook,
  TODO.md AI-scaffolding item).
- **Factions as data:** a `[factions]` table (in entities.toml) with a
  hostility matrix — `team_a`, `team_b`, `zombies`, `wildlife` — replacing the
  `team != team` check (already a TODO deferral). Erik hasn't themed the
  factions yet; names are data, so theming can wait indefinitely.
- **Infection:** `infectable` flag (registry) + kill attribution: a kill whose
  attacker faction is `zombies` converts the victim (soldier or animal) to a
  zombie unit at death position. Rides the faction patch.

## 7. Field-seed painting: AIR (and the wand)

From the 2026-07-12 agreement + 2026-07-17 wishlist, unchanged in substance:

- **AIR paint** — `air_init.npy` mask mirroring `water_init.npy`; default
  vacuum, ambient gas N seeded inside the mask; floor-visual decoupled from
  atmosphere. Vacuum decks, O₂ pockets, pre-pressurized rooms.
- **Magic wand / enclosure fill** — two flavors: *enclosure fill* (flood from
  seed, bounded by solid — the WATER-mode primitive, generalized to any brush
  payload: material, air, vacuum, zone id) and *same-code select* (contiguous
  same-material region → repaint). The **hull-leak validator** falls out free:
  an air-fill that reaches the map border = leaky room → warn, don't paint.
- **Boundary conditions** — per-map `boundary = "space" | "ambient"` level.toml
  field (the AMBIENT border-ring tile from the backlog Topic 4 survey), set in
  a level-properties pane. Format lands here; the physics lands with the
  BC/residency work in stack item 1.

## 8. The UI — panes (see mockup)

Mockup artifact accompanies this doc. Layout:

```
+------------------------------------------------------------------+
| top bar: level name | boundary | bake | PLAY (F5) | save        |
+---+---------------------------------------------------+----------+
| t |                                                   | PALETTE  |
| o |                                                   | [Tiles]  |
| o |                CANVAS                             | [Units]  |
| l |         (live baked preview,                      | [Creat.] |
|   |          entities as sprites/icons,               | [Lights] |
| r |          zone tint overlays)                      | [Zones]  |
| a |                                                   | [Object.]|
| i |                                                   +----------+
| l |                                                   |INSPECTOR |
|   |                                                   | class:   |
|   |                                                   |  fields  |
|   |                                                   |  (typed) |
+---+---------------------------------------------------+----------+
| status: mode | tile under cursor | validators | unsaved changes  |
+------------------------------------------------------------------+
```

- **Tool rail (left):** select/move · paint · wand/fill · room · corridor ·
  door · place-entity · zone-paint. (Current hotkeys preserved; the rail
  displays the active mode.)
- **Palette (right, tabbed):** one tab per registry `category` — Erik's
  "different panes for different types". Tabs are DATA (categories found in
  entities.toml + the material table), not code.
- **Inspector (right, below):** selected instance's fields, widgets from field
  types; overridden fields highlighted vs class default; "reset to class".
- **PLAY (F5):** launches the game on the working level (saved to a temp copy)
  as a subprocess — v1 of play-from-editor is *cheap* (main.py already takes a
  level); embedded in-editor sim is a later upgrade, unlocked by the shared
  pyray stack, not blocked by anything in v3.

## 9. Patch plan (draft — locked only after critique)

Fewest patches that each land green and playable:

- **P0 — registry + format** (no UI): entities.toml loader, `[[entity]]` +
  legacy aliasing, zones storage, `air_init.npy` seeding, `boundary` field,
  validation. Tests: load/save round-trip, legacy levels byte-identical.
- **P1 — editor shell**: panes layout (top bar, rail, palette, inspector,
  status), registry-driven palette/inspector, generic place/select/move/delete
  in the undo ring. Existing modes keep working underneath.
- **P2 — painting power**: wand/enclosure fill (materials + zones + AIR),
  hull-leak validator, zone tint overlays.
- **P3 — play-from-editor** (F5 subprocess) + level-properties pane (boundary,
  name) + polish pass on the acceptance level's authoring flow.
- **P4 (game-side, separately gated): objective carry&extract + factions +
  infection** — the Contested Chip level becomes *playable*, and the ML reward
  hook exists.

Gates: P0 mechanical (digest-style tests). P1–P3 are editor UX → Erik drives
them hands-on (HUMAN-TEST equivalents). P4 is feel-adjacent game logic → full
HUMAN-TEST gate.

## 10. Explicitly NOT v3 (ceiling-raisers, format-safe to defer)

Multi-floor/decks · trigger/event scripting (the WC3 ceiling-raiser — needs
its own design arc when missions demand it) · AI-styled tilesets (levels-w1
P6, still owed) · doors-v1 sliding-slab state machine (own arc; pens v3 can
gate on a breakable `MAT_DOOR`/glass tile until then) · in-editor embedded
sim · decal/texture painting · campaign/meta structure.

## 11. Open questions (Erik input wanted before we lock)

1. **Zone storage**: one `zones.npy` id-grid (simple, one zone per tile) vs
   per-zone masks (overlapping zones possible). Overlap seems unneeded for v3
   — id-grid unless you object.
2. **Pen gates before doors-v1**: OK that a pen's "gate" is v3 = a breakable
   tile (glass/door material), with the real sliding-door state machine
   arriving in the doors arc?
3. **P4 sequencing**: objective/faction/infection are game patches riding an
   *editor* arc — comfortable, or do you want them re-homed under stack item 2
   (weapons/units/roster) and the editor arc kept pure-editor?
4. **Spawn model**: keep individual `[[spawn]]`-style placed units AND add
   breach-site zones (zone spawns its faction's roster at play start)? Or
   zones only for mission levels? (I lean: both — placed units for testbeds,
   zones for missions.)

## 12. Research sources

- [PC Gamer — Unreal's free editor as the true game-changer](https://www.pcgamer.com/gaming-industry/game-development/1998s-unreal-was-a-big-deal-but-its-free-editing-tool-was-the-true-game-changerand-the-origin-of-countless-careers/)
- [Tim Sweeney retrospective on UnrealEd (Game Developer)](https://www.gamedeveloper.com/design/classic-tools-retrospective-tim-sweeney-on-the-first-version-of-the-unreal-editor)
- [Valve Hammer Editor (Valve Developer Community)](https://developer.valvesoftware.com/wiki/User:Cvoxalury/Valve_Hammer_Editor)
- [Warcraft III World Editor (Warcraft Wiki)](https://warcraft.wiki.gg/wiki/World_Editor)
- [TrenchBroom](https://trenchbroom.github.io/)
- [LDtk](https://ldtk.io/) · [LDtk entity docs](https://ldtk.io/docs/general/editor-components/entities/)
- [The Level Design Book — tools appendix](https://book.leveldesignbook.com/appendix/tools)
