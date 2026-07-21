# A7 re-baseline rationale — the arc's single sanctioned golden event (2026-07-19)

Arc A patch A7 (plan: `arc_a_patch_plan_2026-07-18.md` A7 row, rulings 2/3;
standing constraint: "existing goldens re-baseline exactly once (A7)").
This document is the written rationale that ruling 2 requires. Tool:
`tools/migrate_level_entities.py` (commit 1 of A7); tests:
`tests/test_migration.py`.

## 1. What the migration changes, and why that is digest-visible

Two changes, both A7-scoped by prior decision:

1. **Painted `MAT_DOOR` → closed door entities (`MAT_DOOR_CLOSED`).**
   The legacy painted door is the walkable-but-flow-solid HYBRID
   (`MAT_DOOR.mobility = 1000`, config.toml; a6 doors design §0/§1). A
   migrated door is an entity whose CLOSED state is FULLY solid — flow
   AND movement. "A door standing there" honestly reads as a closed
   door, so every migrated door is `initial_state = "closed"` and its
   tiles become truly solid. **This is a real behavior change** (units
   can no longer walk through those tiles; sight/path/blast interactions
   follow the wall seams) and it is exactly the change ruling 2
   sanctions. Digest consequence: the `material` grid changes (3 → 7)
   AND the level gains entities.
2. **`[[light]]` → entity lights.** Render-identical (the A3 alias
   contract makes both forms yield the same LightEntry list), but entity
   lights trip the A4 presence rule, so the digest gains the
   `__entity__`/`__signals__` sections — the A4 impl note pinned this as
   A7-scoped, inside this same single event.

Presence rule (a4_digest_impl_note_2026-07-18.md): a migrated level's
`tick_digest` byte stream gains the entity sections; an unmigrated
level's is bit-identical to before. Entity-present digests are only
comparable at equal `registry_content_hash()`.

## 2. Level inventory and the migration decision

Every repo level, what loads it, and the ruling-2 call:

| level | painted doors | legacy lights | loaded by | decision |
|---|---|---|---|---|
| `test_level` | 15 tiles | 8 blocks | no suite loads it (dev/test level; `test_migration.py`'s entity-form check now pins it) | **MIGRATED** |
| `unhcr_vessel` | 21 | 5 | ~20 behavior suites (materials, fire, water, mobility, the legacy door-stamp-leak guard...) | stays legacy — ruling 2 names it (shipped level; Erik chooses, likely Arc C) |
| `unhcr_vessel_2` | 30 | 5 | test_level_art_v2 (skipped WIP art) | stays legacy — same ruling |
| `playground` | 24 | 3 | test_playground_level (run-vs-run determinism, no committed golden), test_level_lights | stays legacy — showcase level; migrating would flip 24 walkable hybrid doors to solid under its behavior tests |
| `aquarium_demo` | 0 | 0 | water suites | nothing to migrate (tool is a no-op on it) |
| `bake_demo` | 1 | 0 | test_bake_level_art loader round-trip; its committed baked PNGs were baked from the current tilemap | stays legacy — art-pipeline fixture; a 3→7 rewrite would desync tilemap ↔ committed baked art for zero digest benefit |
| `door_test` | 0 | 0 | test_a6_doors (already pure `[[entity]]` form) | no-op by construction |

**Decision: migrate `test_level` only.** It is the repo's designated
test level, carries both legacy forms, and is not a shipped/showcase
level. All other levels stay legacy per ruling 2 (revisit at Arc C when
the editor handles entities).

**Lights-only flips (the A4-note-required list): NONE.** No level was
migrated for lights alone — `test_level` flipped for doors AND lights in
one event; `aquarium_demo` (the only other candidate test level) has no
lights and no doors. If a lights-only level is ever migrated later, that
is a NEW digest event needing its own rationale.

## 3. What the golden/digest suites actually use — the empty re-baseline

The step-5 inventory finding: **every committed golden/digest artifact
derives from synthetic content, not from any level folder.**

- The canonical x-arch scenario is SYNTHETIC:
  `tests/field_ab_harness.py::_scenario_level` builds a 16×16 hull box
  with carved air in memory (tile codes 1 and 4 only — no painted doors,
  no lights, no `[[entity]]`), so `capture_trajectory` records
  `n_entities = 0` every tick and the A4 fold appends nothing.
- The `GOLDEN` constants in `tests/cuda_*_check.py` /
  `tests/_xarch_perfield_digest.py`
  (`98d3dd7eaf3d574d6e562513cd95f3b5ac077b7c69b1d0b024db931261735473`)
  are that synthetic scenario's aggregate trajectory digest.
- `tests/data/bake16_golden_*.png` bake from the in-test synthetic
  `GOLDEN_MAP`, not from a level folder.

**Therefore ZERO committed golden/digest artifacts change in this
re-baseline.** The sanctioned event is real (a level's digest identity
flipped — §4) but its committed-artifact footprint is empty. Verified
post-migration, full suite green with these artifacts byte-identical
(git blob hashes, before == after):

| artifact | git blob sha1 (unchanged) |
|---|---|
| `tests/digest_erik_lenovo_cpu_cpu.txt` | `989d52f24e6f6f23637bc9273ae0d594f255e1e0` |
| `tests/_xarch_liveheat_DESKTOP-0E98HUV.txt` | `8f5d87ad757cf41b4275da4edd19b560decfd933` |
| `tests/_xarch_perfield_DESKTOP-0E98HUV.txt` | `e1b3befcd895c42f5da6f5d44cddfb314bd6429f` |
| `tests/_xarch_perfield_erik_lenovo.txt` | `cfbacfaa3a96e2d6fdc452dc635cf126d09e1deb` |
| `tests/data/bake16_golden_diffuse.png` | `75fb017230061c89d265de8d6da42e782543f386` |
| `tests/data/bake16_golden_normal.png` | `e2668725fd75482b2f383b2c56d44d09483ba378` |

The x-arch cross-machine artifacts CANNOT change from this migration —
the scenario never touches a level folder — and did not.

## 4. The digest values that DID flip (the evidence of record)

`test_level` through the canonical capture path
(`field_ab_harness.capture_trajectory`, seed 1234, `breach_physics`
bound, 5 steps; desktop DESKTOP-0E98HUV, CPU backend):

| quantity | before (fb103eb / 010abf9) | after (this commit) |
|---|---|---|
| `n_entities` | 0 | 13 (5 doors + 8 lights) |
| tick-0 `tick_digest` | `aa298687d6932297599d591b2bd8660e9d101063f58d8f0f067afafb57239361` | `4b20af2f89be0d5e5c08d6fce279344125d23a540a1a210e0317e77f9e2cca03` |
| 5-step `trajectory_digest` | `570bf8b5124cf50308730b1aaf022c41f0c85190d25b10d64cde3a7dba66c457` | `8bf061f578e1ec2a475bf20ab54fe585fb0a903a05ab8c26241e66990e846872` |
| `registry_content_hash` | n/a (entity-free) | `95de3c9343c2a9967664f85920d57d358eab1a79dbeaf3480036f332f0cfee7c` |

Both the field half (`material` 3→7 on 15 tiles) and the entity fold
(13 instances, door runtime rows included) contribute. These values are
recorded here — not as committed golden files — because no suite pins
`test_level` digests; the entity-form regression guard is
`tests/test_migration.py::test_repo_test_level_is_entity_form` (armed by
this commit: 5 doors / 8 lights / 15-tile span coverage / GameMap
constructs).

## 5. What was migrated in `test_level`

5 doors (ids in anchor scan order; all `initial_state = "closed"`,
tiles 3 → 7): `door_1` h@(79,244)×3 · `door_2` h@(84,244)×3 · `door_3`
h@(79,245)×3 (the double-thick pair with door_1, honestly two parallel
panels) · `door_4` v@(87,249)×3 · `door_5` v@(104,249)×3 — all
`length_m = 1.0` (3 tiles at 3 tiles/m). 8 lights (`light_1..8`, file
order, authored values verbatim). `[[spawn]]` untouched. No painted
door tile carried water and none is vacuum-adjacent (checked before
migration). `.bak` files of both touched files left in the level folder
(untracked, per the tool contract).

## 6. Gate

`pytest tests -q` (conda env `data`) full-suite green on the final tree;
the only working-tree changes beyond the tool/tests commit are
`levels/test_level/level.toml`, `levels/test_level/tilemap.csv`, and
this document. Erik's blessing of this re-baseline is the A7 merge gate
(ruling 3 — no auto-merge).
