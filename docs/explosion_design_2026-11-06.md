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

- **Expansion**: `40 * strength` m3 in one tick, flat over the tiles reached within
  `r_s = max(1, floor(sqrt(strength)))` tiles. Integer steps, so strength 1 up to just
  under 4 stays on one tile: a 1 % change of strength never moves S 5-fold.
- **Heat**: `10 * strength` MJ in one tick over the tiles reached within
  `r_h = 2 * r_s` tiles, with the shipped LINEAR falloff, normalised so the ENQUEUED
  total is fixed however many tiles the walk reached. Then the air read scales it. One
  derived constant at the door converts MJ to `heat` units (c_v, tile volume,
  `n_floor`), never a literal. The temperature rail still binds: an ambient tile holds
  3.85 MJ, so a blast in a one-tile-wide corridor LANDS less than it asks for (about
  8.8 of 10 MJ at strength 1). That is physics and stays.
- **Smoke**: `explosive_kg = strength * grenade_kg`, derived BEFORE the existing kg
  check, so smoke = that mass x `soot_yield` (or the `min_soot_yield` floor) x
  `smoke_scale`, over the heat's tiles, the air read scaling it per tile. The noise is
  kept, so the total is fixed in expectation, not per blast. A row no longer authors
  `explosive_kg` when it has `strength`; the row door refuses both, and requires
  `explosive` and `emit_blast_smoke`.
- **Reach** is the walk `blast_expansion.reach_tiles`: 4-neighbour along open floor,
  on the solve's own mask (never solid, vacuum or the ambient ring; passable fuel tiles
  stay in it), so a wall stops it and a door gap does not, and no share of S is silently
  lost. The radii are integer
  r2 fixed at the door, so walk and credit agree. The tile list is sorted row-major
  before any edit. The air read (`read_blast_air`) and the heat claim (`TickCredit`)
  take this same tile list, not a disc. Fixed total means a blast in a corridor is
  hotter and shoves harder than the same blast in the open.
- **A charge centred on a solid tile** (a round stops ON the wall; a door charge on a
  door) walks after `apply_explosion`, so a tile the blast opened is open. If the centre
  is still solid, the walk seeds from the APPROACH side: the tile the round came from,
  or the planter's side for a placed charge, and distance is measured from the centre.
  With no known side (a timed charge authored onto a wall) it seeds from every open
  4-neighbour in sorted order. CLAUDE'S CALL, Erik may veto at the HUMAN-TEST.
- **No fire-starting rule (Erik's ruling, 2026-11-06).** A fire starts only when a
  material reaches its own ignition temperature through the engine's ordinary heat.
  This is a core principle of Breach: situations fall out of the physics. The blast's
  fixed-total heat lands on every walked tile, passable fuel tiles included, and
  `apply_temperature_ignition` lights whatever reaches its `ignition_temp`. Nothing
  raises, biases or special-cases heat on fuel, and a strength row refuses
  `ignite_radius` and `ignite_intensity`. `ignite_ring` is unchanged for rows without
  `strength`. If a row no longer lights a room, that is the accepted result.
- `radius`, `unit_damage`, `wall_damage` stay the row's direct damage, with today's
  disc and falloff; the door requires `radius > 0`, or the row would skip `blast_log`. `pressure`, `heat_amount`, `heat_radius`, `expansion_*` must be
  ABSENT on a strength row; the door tests key presence, not value, and refuses them
  by name (`frag_standard`'s explicit `pressure = 0.0` is deleted).
- Rows without `strength` behave exactly as today (`frag_hot_mass`, `frag_heat`, the
  gas and plasma rows). Nothing is removed.

Rows converted, all PLACEHOLDER:

| Row | strength | Note |
|---|---|---|
| `frag_standard` | 1.0 | radius 2, unit 60, wall 200 unchanged |
| `breach_focus` | 0.1 | C4. Wall 500 and radius 3 unchanged, `unit_damage` 0: the planter is unhurt by the row itself. Its smoke falls to about 0.09 g, effectively none: accepted, and the "door charge keeps its smoke" comments are reworded |
| `demolition_c4` | 12.0 | radius 8, unit 150, wall 800 unchanged |
| `magazine_cookoff` | 8.0 | radius 7, unit 120, wall 400 unchanged |
| `engine_death` | 3.0 | radius 3, unit 80, wall 300 unchanged |

Left alone: `frag_heat`, `frag_hot_mass`, the `studio_*` rows (a tuning level), plasma
splashes, gas grenades, `quad_*` and `div_*` test rows.

## 3. Engine work

1. **Source on the GPU.** `div_source` reaches `cuda_eos_step` and `cuda_eos_resident`
   (`K_div_u`), integer for integer, GPU == CPU at tol 0. An `(N, h, w)`-born launch
   per RL-batch habits. Uploaded only on a tick with a source.
2. **Source state.** A detonation after the solve leaves its source for the next
   tick, paired with its queued heat and smoke. The pending source becomes a dense
   `(h, w)` int32 plane in DIGEST_FIELDS and SIM_FIELDS, zeroed on consumption, with the
   spec version bump and golden regeneration in one commit, hashed only when non-zero
   if the digest spec allows it (the ENTITY_SECT precedent), which spares every golden
   without a blast. The `EditQueue` itself is not digested, so a slot-9e blast's pending heat and smoke stay invisible to a
   snapshot, as every queued edit does today.
3. **Source range.** S depends on the walk, which is level geometry, so no row door
   can bound it, and strength 12 in a one-tile pocket crashes or wraps the prototype.
   Sources accumulate in int64 at detonation and saturate at a named `S_MAX`, counted.
   The subtraction from `div_u` widens and clamps identically in all three twins.
4. **Fixed-total deposits.** Through FieldEdit, never inline, on ONE new tile-list
   region (`Region` gains it; explicit per-tile weights, row-major), and one
   fixed-total deposit over it, never twins of `deposit_heat`, `_claim_disc` and
   `blast_smoke_peak`. Heat and smoke share one reach. Same RNG draw order discipline
   as `add_explosion_smoke`.
5. **Python tick cost.** The walk runs at detonation only, never per tick.
6. **Direct-damage reach** is not touched.
7. **Clients.** `tools/lighting_demo.py` copies a `PayloadDef` (`copy.copy` plus
   overrides), never rebuilds it column by column, or look shots lose the blast.
   Its pressure slider goes. `tools/wind_load_probe.py` stops printing stale columns.

## 4. Verification (properties)

- No gas added; energy closure exact through a blast (exists).
- A wall stops heat, smoke, source and the air read; a gap passes them.
- A charge on a wall tile delivers on the approach side and not behind the wall.
- The ENQUEUED heat total is the same in a room and a corridor (within n/2 LSB, before
  noise and rail); smoke's mean over seeds likewise.
- Strength scales: double strength doubles the enqueued heat and smoke, and wind at a
  fixed range beyond the rail (8 tiles) grows. Peak wind does not: it rails near 78 m/s.
- A source past `S_MAX` in a one-tile pocket saturates, counted, and never wraps.
- GPU == CPU at tol 0 on a blast tick, per-call and resident (`cuda_*_check` convention).
- A digest covers a pending source: a snapshot taken between detonation and solve differs.
- Determinism: replay of a match with grenades clears every checkpoint.
- The row door refuses a strength row carrying `pressure` or `heat_amount`.
- A fuel tile the blast's ordinary heat takes to its `ignition_temp` catches, and one
  that stays below does not. Nothing writes `fire`, and nothing adds heat beyond the
  fixed total, with fuel in the walk or without it.
- A breach charge still breaches a door, and the planter at the shipped placement
  distance loses no HP and is not knocked down.
- Pre-existing tests that pin today's behaviour on purpose are re-pinned once, with a
  written rationale: `test_zombie_control` ship_demo digests, `test_payloads`
  `frag_and_breach` (:102, :187), `test_weapons_tables` (:108, :110), and
  `test_heat_source` (:272, :433, to what the engine now does, no fire required of
  `engine_death`; its `_blast` fixture gets its own
  heat-free row, because `breach_focus` now carries heat). Anything else failing is
  reported, never bent.

## 5. Patches

| Patch | Content | Mode | Gate |
|---|---|---|---|
| A | The rule, the door, fixed-total walk, converted rows, digest, re-pins, CLAUDE.md rows | Opus agent | suite + the properties above |
| B | GPU source, per-call and resident. Merges with A, never apart: until B the CUDA paths drop the source | same agent | CUDA lockstep at tol 0 |
| C | `levels/blast_final`: the real rows in four rooms; bot-match before and after over seeds 1-8, reporting results, deaths by cause, and whether corridor knockdown passes the bots' `danger_radius` | same agent | HUMAN-TEST: Erik watches and OKs |

ACCEPTED GAP: heat on a wall tile. A blast no longer deposits directly on solids; walls
heat by conduction from the air beside them.

ACCEPTED GAP: no wind damage. Erik raised it and left it open; direct `unit_damage` is
the blast's impulse until matches say otherwise.

ACCEPTED GAP: knockdown, HP and every strength value are placeholders.

## 6. Critique

| Lens | Verdict | Outcome |
|---|---|---|
| Determinism and engine correctness (Opus, on v1) | 2 blockers, 5 majors, 5 minors; verified against the code, none rejected | v2: solid centre, ignition, `S_MAX`, integer reach steps, walk-based air read, row-door rules, digest plane, clients |
| Systems reuse and gameplay regression (Opus, on v1) | 1 blocker, 3 majors (all three already in v2), 7 minors; verified, none rejected | v3: ignition rule by material, one tile-list region and deposit, digest only when present, the fuller re-pin list, stale text |

Corrected by the second review: v2 said a frag lights only the exact fuel tile. It
lit a crate one tile off, so v3 added a derived ignition rule to preserve that.

Overruled by Erik, 2026-11-06: that critic's ignition fix is removed. No blast
fire-starting rule; fires fall out of the physics (section 2).

Sound by that review: both GPU consumers of `div_u` are named and complete, the mg RHS
is safe, tick order keeps a late source paired with its edits, closure holds.

## 7. Systems

**Existing, must be used:** Payload executor (the one entry), FieldEdit (the only field
write), Gas energy seam and closure identity, Interior drag neighbour stages in EOS,
Fixed-point kits, Q16 boundary modules, Config, Field digest + GOLDEN_AGGREGATE, CUDA
harness, RL-batch habits, Starting a fire, Credited cause (`deposit_heat`'s claim must
survive, on the walk's tiles), Look shots, E2E driver, `blast_log` seam, `PayloadTable`
(the row door), Wind load on a body, Timed charge,
Heat source, benches `tools/wind_load_probe.py` and `tests/_blast_bench.py`.

**Stale text to update at implementation:** CLAUDE.md Heat source row; the
`engine_death` comment in `config.toml`; `payloads.py` and `weapons.py` docstrings and
the IGNITE_HEAT comment; `.claude/skills/fire-smoke-tuning/SKILL.md`.

**New, draft rules for CLAUDE.md at implementation:**

- *Generic explosion*: a payload row with `strength` is the one explosion: expansion,
  heat and smoke derive from `[blast]`, read by a refusing settings door; the row keeps
  its explosive name and its direct damage. A new explosive is a
  row of those, never a new blast shape.
- *Divergence source*: `EOSSolver.div_source`, written only by the blast expansion
  module: THE way to make air move outward with no mass. It conserves N and E exactly;
  pair it with heat or the gas springs back.
- *Blast reach*: `blast_expansion.reach_tiles`: THE walk for what a blast touches.
- *Blast air read*: `blast_air.read_blast_air`: a charge reads P and O2 once, over the
  tiles it reached, after its wall damage and before its field edits land.
  `smoke_scale` is a look dial on top of the cited `soot_yield`.
- *Fixed-total tile deposit*: a FieldEdit tile-list region plus one deposit: THE way to
  land a total over a walked set of tiles.
