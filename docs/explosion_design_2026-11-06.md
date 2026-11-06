# The standard explosion (#31)

Date: 2026-11-06. Status: approved in substance by Erik in session, built on branch
`31-div-source`, merged only after he has watched the result (HUMAN-TEST).
Background and sources: `docs/research/explosion_litsearch_2026-11-06.md` (branch
`31-explosion-design`), and `levels/blast_quad`, `blast_div`, `blast_smoke`.

## 1. What Erik ruled

| Ruling | Value |
|---|---|
| The shock is real air the pressure solve delivers, respecting geometry | a divergence source (Feldman 2003), no one-tick rule |
| As little added mass as possible | no mass at all: no puff (Claude's recommendation, accepted) |
| Delivery | one tile, one tick |
| Smoke | `[blast_air] smoke_scale = 0.25`; the air read stays (P and O2 at detonation) |
| One strength number per explosive, grenade = 1 | section 2 |
| Per-explosive individual damage | `unit_damage`, `wall_damage`, `radius` stay row values |
| Do not tune now | knockdown and HP are placeholders until matches are watched |
| Multi-tick delivery | a future special explosive (#174), not this arc |

Accepted by Erik through "go with your recommendations": expansion stays 40 m3 at
strength 1 (real marine knocked down to 2.5 m); heat about 10 MJ over a 2-tile radius
in one tick (measured below); heat and smoke stop at walls with a fixed total.

Measured, one-tile source, 40 m3, large room, real marine, `unit_damage` 0:

| Heat delivery | MJ landed | Peak game T | HP at 1 / 1.5 / 2 / 2.5 / 3 m |
|---|---|---|---|
| one tile, 1 tick | 3.85 (rail) | 16000 | 77 85 93 95 97 |
| one tile, 2 ticks | 4.10 | 16000 | 70 80 92 94 97 |
| radius 2, 1 tick, heat 80 | 10.35 | 10110 | 54 63 84 89 94 |
| radius 2, 1 tick, heat 140 | 17.7 | 16000 | 12 27 67 77 87 |

Heat over ticks on one tile does not work because the source has emptied the tile.
Wind load is independent of the heat row: 1684 / 1007 / 458 / 337 / 195 Pa.

## 2. The rule

A payload row gains `strength` (float >= 0, 0 = no generic blast). A row with
`strength > 0` is a generic explosion. `[blast]`, all PLACEHOLDER:

    expansion_m3_per_strength = 40.0
    heat_mj_per_strength      = 10.0
    grenade_kg                = 0.184

- **Expansion**: `40 * strength` m3 in one tick, flat over the tiles reached from the
  charge within `max(1, sqrt(strength))` tiles.
- **Heat**: `10 * strength` MJ in one tick over the tiles reached within
  `2 * sqrt(max(strength, 1))` tiles, with the shipped LINEAR falloff, normalised so
  the total is fixed however many tiles the walk reached. Then the existing air read
  scales it.
- **Smoke**: `explosive_kg = strength * grenade_kg`, so smoke = that mass x
  `soot_yield` (or the `min_soot_yield` floor) x `smoke_scale`, over the heat's tiles,
  the air read scaling it per tile. A row no longer authors `explosive_kg` when it has
  `strength`; the row door refuses both.
- **Reach** is the walk `blast_expansion.reach_tiles`: 4-neighbour along open floor, so
  a wall stops it and a door gap does not. Fixed total means a blast in a corridor is
  hotter and shoves harder than the same blast in the open.
- `radius`, `unit_damage`, `wall_damage` stay the row's direct damage, with today's
  disc and falloff. `pressure`, `heat_amount`, `heat_radius`, `expansion_*` must be
  absent on a strength row; the door refuses them by name.
- Rows without `strength` behave exactly as today (`frag_hot_mass`, `frag_heat`, the
  gas and plasma rows). Nothing is removed.

Rows converted, all PLACEHOLDER:

| Row | strength | Note |
|---|---|---|
| `frag_standard` | 1.0 | radius 2, unit 60, wall 200 unchanged |
| `breach_focus` | 0.1 | C4. Wall 500 and radius 3 unchanged, `unit_damage` 0: the planter is unhurt by the row itself |
| `demolition_c4` | 12.0 | radius 8, unit 150, wall 800 unchanged |
| `magazine_cookoff` | 8.0 | radius 7, unit 120, wall 400 unchanged |
| `engine_death` | 3.0 | must still carry at least IGNITE_HEAT to 2 tiles ("Starting a fire") |

Left alone: `frag_heat`, `frag_hot_mass`, the `studio_*` rows (a tuning level), plasma
splashes, gas grenades, `quad_*` and `div_*` test rows.

## 3. Engine work

1. **Source on the GPU.** `div_source` reaches `cuda_eos_step` and `cuda_eos_resident`
   (`K_div_u`), integer for integer, GPU == CPU at tol 0. An `(N, h, w)`-born launch
   per RL-batch habits. Uploaded only on a tick with a source.
2. **Source state.** `gmap.expansions` holds state across a tick boundary when a
   detonation lands after the solve. It joins the digest, with the spec version bump
   and golden regeneration in one commit. A source is integer on entry: quantized once,
   and refused at the door if `S` would pass int32 at the row's reach.
3. **Fixed-total deposits.** Through FieldEdit, never inline. Heat and smoke share one
   reach. Same RNG draw order discipline as `add_explosion_smoke`.
4. **Python tick cost.** The walk runs at detonation only, never per tick.
5. **Direct-damage reach** is not touched.

## 4. Verification (properties)

- No gas added; energy closure exact through a blast (exists).
- A wall stops heat, smoke and source; a gap passes them (extend the existing test).
- The total heat and smoke are the same in a room and a corridor.
- Strength scales: double strength doubles the delivered heat and smoke, and wind grows.
- GPU == CPU at tol 0 on a blast tick, per-call and resident (`cuda_*_check` convention).
- A digest covers a pending source: a snapshot taken between detonation and solve differs.
- Determinism: replay of a match with grenades clears every checkpoint.
- The row door refuses a strength row carrying `pressure` or `heat_amount`.
- `engine_death` still lights a room; a breach charge still breaches a door, and the
  planter at the shipped placement distance loses no HP.
- Pre-existing tests that pin today's behaviour on purpose are re-pinned once, with a
  written rationale: `test_zombie_control` ship_demo digests, `test_payloads`
  `frag_and_breach`, and any other. Anything else failing is reported, never bent.

## 5. Patches

| Patch | Content | Mode | Gate |
|---|---|---|---|
| A | The rule, the door, fixed-total walk, converted rows, digest, re-pins, CLAUDE.md rows | Opus agent | suite + the properties above |
| B | GPU source, per-call and resident | same agent | CUDA lockstep at tol 0 |
| C | `levels/blast_final`: the real rows in four rooms; bot-match before and after over seeds 1-8 | same agent | HUMAN-TEST: Erik watches and OKs |

ACCEPTED GAP: heat on a wall tile. A blast no longer deposits directly on solids; walls
heat by conduction from the air beside them.

ACCEPTED GAP: no wind damage. Erik raised it and left it open; direct `unit_damage` is
the blast's impulse until matches say otherwise.

ACCEPTED GAP: knockdown, HP and every strength value are placeholders.

## 6. Systems

**Existing, must be used:** Payload executor (the one entry), FieldEdit (the only field
write), Gas energy seam and closure identity, Interior drag neighbour stages in EOS,
Fixed-point kits, Q16 boundary modules, Config, Field digest + GOLDEN_AGGREGATE, CUDA
harness, RL-batch habits, Starting a fire, Credited cause (`deposit_heat`'s claim must
survive), Look shots, E2E driver, `blast_log` seam.

**New, draft rules for CLAUDE.md at implementation:**

- *Generic explosion*: a payload row with `strength` is the one explosion: expansion,
  heat and smoke derive from `[blast]`, the row keeps only direct damage. A new
  explosive is a strength and an explosive name, never a new blast shape.
- *Divergence source*: `EOSSolver.div_source`, written only by the blast expansion
  module: THE way to make air move outward with no mass. It conserves N and E exactly;
  pair it with heat or the gas springs back.
- *Blast reach*: `blast_expansion.reach_tiles`: THE walk for what a blast touches.
- *Blast air read*: `blast_air.read_blast_air`: a charge reads P and O2 once, before
  any of its edits land. `smoke_scale` is a look dial on top of the cited `soot_yield`.
