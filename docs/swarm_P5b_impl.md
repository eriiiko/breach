# Swarm units P5b -- corpse infestation (as built, 2026-11-19)

Arc #63, patch P5b. Branch `63-p5b-corpse-infestation` off `63-swarm-units`
at `63758421`. The shape is P5a's (`docs/swarm_P5a_impl.md`) and Erik's rulings
of 2026-11-14 (`docs/unit_design_2026-11-14.md` section 7.4): larvae on a
corpse raise it, through a per-tile field the swarm writes and a coupling row
the unit side reads; one level per corpse, decaying; the ONE conversion door;
any corpse, any number of times; a corpse-integrity pool; the columns are
unit-schema rows. Rules: `.claude/rules/swarm.md` (Count plane),
`.claude/rules/units.md` (Corpse law), `.claude/rules/horde.md` (Conversion
door).

## The census: which damage reaches a dead unit today

Read from the code and measured on main's two pinned ship_demo plays (`bots`,
`bots_v1`) with a read-only wrapper around `damage.apply_packet`.

| path | reaches a corpse? | what it does there today |
|---|---|---|
| blast (`exchange.apply_blast_damage`: grenades, charges, payload rounds) | NO: `if not u.alive: continue` | nothing. Measured in the larva studio: a `frag_standard` on the tile beside a corpse's footprint leaves its hp and integrity as they were, no hit event |
| heat row (9c, fire, tile heat, radiation) | NO: alive gate | nothing |
| statuses (2b DoT) | NO: alive gate ("frozen on corpses") | nothing |
| poison law (9c3'), gas poison and bite doses | NO: cleared at death; the rows gate on alive | nothing |
| spray (4b) | NO: it deposits gas / heat into cells; units are hurt only through the rows above | nothing |
| rounds in flight (slot 2', later ticks) | NO: the march skips the dead | nothing |
| attack slot (4c resolve: rounds marched this tick, beams, melee) | YES, only a packet decided against the slot's START state that lands after an earlier effect of the same resolve killed the target | the hp drops further below 0, a `UnitHitEvent` is emitted, no second kill |
| zombie slot (5, `update_zombies_tick`) | YES, the prey list is built at the slot's start: a second zombie's bite in the same pass lands on the marine the first one killed | as above (melee, `killed_by_zombie` already set) |

Measured in main's pinned plays: `bots` -- no packet on a corpse;
`bots_v1` -- one: zombie 11's 34 HP bite on marine 6, killed by zombie 11's
neighbour in the same slot-5 pass (the step at sim tick 270; the corpse converts at that
tick's end). `test_exposure`'s play (k_expose 0): one 10 HP round on marine 0's
corpse in the step at sim tick 269, from the same resolve that killed it.

What it would take to reach a corpse where nothing reaches today: a blast --
drop `apply_blast_damage`'s alive gate for corpses and route the corpse's
packet (below); fire -- the heat row's band now reads the BODY temperature
(#215), which a corpse no longer has (cleared at death), so a corpse heating
up is a new law (a corpse body temperature, or the felt temperature straight
into a band of its own): reported, not built (the brief's rule). Neither of
main's two plays leaves a corpse on a burning tile or under a heat deposit
(measured: 0 corpse-ticks of each).

## What was built

### The swarm side: the count plane
- `gmap.swarm_count`, `(h, w)` int32, a second swarm output plane beside the
  bite plane (`swarm.COUNT_PLANE_ATTR`, read accessor `swarm.count_plane`): the
  number of ACTIVE swarm units on each tile this tick. `swarm::active_on`
  (`cpp/src/swarm.h`) is the one predicate (valid, not dead, not in coma);
  `bites_on` now calls it, so the bite plane is the count plane times the bite
  row, and a later biting rule changes `bites_on` alone.
- Built in the SAME pass as the bites: the host builder `output_planes_host`
  (renamed from `bite_plane_host`, both planes, either may be null) after each
  species' law in `swarm::step_all`, and the resident `k_planes` (renamed from
  `k_bites`; integer `atomicAdd`, outside the law). Both planes are zeroed every
  non-dormant tick; the resident exit download carries both while a swarm is
  present (`SWARM_PLANE_NAMES`). Not digested; not in the swarm section.
- Every species counts 1 per active unit (ACCEPTED GAP: with a second species
  its patch decides whether it infests -- a species row, never a second plane).

### The unit side: the corpse law
- `src/simulation/corpse.py`, the ONE writer of two new unit-schema columns
  (group S, "health" block, regenerated device store W 185):
  `infest_q` (Q16, default 0) and `corpse_integrity_q` (Q16 hit points,
  default `[corpse] integrity`). Both are reset at every death
  (`damage.apply_packet`'s death branch, `corpse.on_death`) and every
  conversion (`corpse.clear`) to their DEFAULTS, and each is hashed only while
  it differs from its default -- so a match without larvae hashes exactly as
  before.
- THE ROW: `exchange.apply_corpse_infestation`, the `swarm_count | sum`
  coupling row, at slot 9c3 right after the bite row: for each DEAD unit, `n`
  = the count plane's sum over its footprint (`occupied_tiles`, the existing
  footprint helper; two corpses on a tile each read it in full); lazy (a
  corpse at level 0 with nobody on it is skipped; an all-zero plane is one
  `.any()`).
- THE LAW (`corpse.infest`), explicit Euler in integers:
  `level += n x rise_q - drain(level, decay_a)`, the drain on the level
  BEFORE the rise, through THE clearing step (`clearing.drain` at
  `clearing.RATE_SHIFT`, 2^-24 a tick; no flush: its floor of one count brings
  a level back to exactly 0). A level or a rise past `corpse.LEVEL_MAX` (2^38)
  is refused by name (`CorpseLevelRange`), never clamped. A destroyed corpse
  (integrity <= 0) holds its level at 0.
- A new door in `clearing`: `rate_per_tick(rate, tps)` -- a first-order rate
  per second as the per-tick fraction, the authored float as an exact rational
  rounded once (the decay is authored as a rate, the brief's number; a
  half-life keeps `half_life_rate`).
- THE DOOR: `ai_zombie.rises_as_zombie(u)` is the conversion's trigger, ONE
  predicate: `not alive and not destroyed and ((not is_zombie and
  killed_by_zombie) or corpse.rises(u))`, `corpse.rises` = level >=
  `infest_threshold`. `convert_marines_to_zombies` (the one conversion path,
  where it is called today: the ruleset's `on_tick_end`) converts exactly as
  before -- the zombie class's hit points, the clears -- plus `corpse.clear`
  (level 0, integrity full). A zombie's corpse rises the same way (it is not
  renamed a second time: `Z-` once). Nothing counts the rises. The door emits
  no event today, and a rise emits none: no `Cause` row (the agents'
  `hit_cause` input stays 11 wide).
- TICK ORDER (the conductor comment at 9c3 and the docstring): the count plane
  is written in slot 7 (the swarm step), the level moves in 9c3, the corpse
  rises at the tick's end (slot 10, the ruleset's tick end).
- Config `[corpse]`, construction-bound through its refusing door
  (`corpse.settings_from_config`; `Simulation.__init__` rebuilds it).

### The numbers (first values, the lead's; Erik tunes)

| `[corpse]` | value | in the law | why |
|---|---|---|---|
| `infest_rate` | 0.02 / larva / s | `rise_q` = 55 (Q16 a tick) | a lone larva must never raise the dead, a handful must within a fight |
| `infest_decay` | 0.05 / s | `decay_a` = 34 953 (2^-24 a tick; half-life 13.9 s) | larvae that leave a corpse lose their claim within a quarter minute |
| `infest_threshold` | 1.0 | `threshold_q` = 65 536 | steady level = 0.4 n: n = 3 is the smallest crowd that rises |
| `integrity` | 200 HP | 13 107 200 | two marines' worth of hit points: a corpse is harder to destroy than to kill |

Measured through the law (`tests/test_corpse_infestation.py`, and the sim in a
room the corpse fills, same ticks):

| larvae on the corpse | steady level | time to rise | the continuous law |
|---|---|---|---|
| 1 | 0.403 | never | 0.4, never |
| 2 | 0.806 | never | 0.8, never |
| 3 | 1.209 | 34.8 s (836 ticks) | 35.8 s |
| 4 | 1.611 | 19.3 s | 19.4 s |
| 5 | 2.014 | 13.7 s | 13.9 s |
| 10 | 4.028 | 5.7 s (137 ticks) | 5.75 s |
| 20 | 8.057 | 2.7 s | 2.7 s |

The integer law runs about 0.7 % hot in the rise (55 for 54.6 counts) and its
truncating drain about 1 % short, so three larvae rise a second sooner than the
continuous law says; both are inside one rounding of the dials.

### The integrity: built to the column, STOPPED at the damage
Built: the column, set at death (to its default) and reset at conversion, the
"destroyed never rises by either trigger" rule (the door and the level), host
and device, and their tests (the integrity written by a fixture).
NOT built: corpse damage wearing it. The census found ONE reach (the same-slot
packets above), and routing it through `apply_packet` to the integrity
instead of the hp was built and run against main's pins: it moves FIVE of the
eight pinned-match tests (`test_exposure` pre-G4p, `test_horde_behaviours`
stage 1, `test_turn_rate` pre-G1p, `test_turning_law` re-pin 1,
`test_unit_impulse` re-pin 1): outcome, ticks and field digest unchanged, the
unit stream and the tick digest move. First divergence (`test_exposure`'s play,
k_expose 0): the step at sim tick 269, marine 0 -- killed earlier in the same
attack-slot resolve -- takes a 10 HP round on its corpse; routed, its hashed hp
stays at the killing blow's value and `corpse_integrity_q` = 190 HP enters its
record. (`bots_v1`'s one corpse packet does not move it: that corpse converts
the same tick and the conversion resets both columns.) No re-baseline is
approved, so the routing was reverted and the integrity part stops at the
column; `tests/test_corpse_infestation.py::test_a_packet_on_a_corpse_leaves_its_integrity_whole`
pins the gap. The routing itself is two lines in each `apply_packet` (host and
device: the corpse's packet writes the integrity, a heal capped at full) and
the lead's call with Erik's re-pin.

### Device (Class T, in this patch, no debt)
- Slot 9c3's fused kernel (`cpp/src/cuda_unit_exchange.cu` `k_gas_rows`): the
  corpse row after the bite (the count plane a new declared input,
  `unit_kernels.GAS_FIELDS` + `swarm_count`; the early return keeps an
  uninfested corpse with nobody on it untouched); the same refusal past 2^38.
- The device `apply_packet`'s death branch (`cuda_unit_damage.cuh`): level 0,
  integrity full.
- `k_convert_zombies` (`cuda_unit_conversion.cu`): the trigger
  `rises_as_zombie`, and `corpse.clear`.
- Four device table rows (`corpse_rise_q`, `corpse_decay_a`,
  `corpse_threshold_q`, `corpse_integrity_q`), layouts regenerated.
- The swarm planes: `k_planes` on the resident path, `output_planes_host` on
  the host and per-call paths (Class P, the swarm twin).
- `apply_corpse_infestation` joined the `9c3 gas rows` profiler label, so the
  chained harness replaces it with the kernel.

### The look: left out
The dot view draws swarm units from the store's host mirror in
`GameRenderer.upload_state`, which is handed the map and not the units, and
its sprites share one radius (a uniform): a ring that fills toward the
threshold needs the unit list plumbed into that seam and a second sprite size
-- more than a few lines through the existing seams, so it was left out (the
brief's rule). A cheap later form: the tile inspector's line.

## Measured: the larva studio (`docs/swarm_P5b_measure.py`)
Headless, CPU path, the continuous ruleset, seeds 1..5. A marine's corpse in
the middle of the west brood, a zombie's in the middle of the east brood,
killed at t = 0 by an uncredited packet (so only the larvae can raise them).

| when the bodies fall | marine's corpse rises (s; mean larvae under it) | zombie's corpse rises |
|---|---|---|
| fresh broods (t = 0) | 1.9, 2.3, 2.0, 2.4, 3.1 (17 to 27 larvae) | 2.6, 2.5, 1.8, 2.7, 3.2 (17 to 28) |
| after 60 s of spreading | 7.0, 6.8, 5.5, 4.2, 4.5 (8 to 13) | 15.1, 9.3, 5.6, 7.8, 9.8 (5 to 10) |

A body that falls in a fresh brood stands up again in two or three seconds; in
a brood that has spread for a minute, in five to fifteen. A frag on the tile
beside a corpse: integrity 200 -> 200 HP, hp unchanged, no hit (the census).

## Tests (each names its property and the change that breaks it)
- `tests/test_corpse_infestation.py` (new, 20): the count plane counts living
  non-coma larvae per tile, order-free, garbage gone, equal to the bite plane /
  bite row; equal to a host scan every tick in a sim and never written on a
  swarm-free map; resident and not digested; the dials quantized once and the
  door's refusals; the level rises with n (the drain on the old level) and
  decays to exactly 0; the steady states and the three times; a destroyed
  corpse held at 0; the 2^38 refusal; a marine's corpse rises at the
  threshold through the door (alive, zombie team, level 0, integrity full,
  named once); one larva never raises a corpse; a zombie's corpse rises, and a
  risen zombie dies and rises again (three times); a destroyed corpse rises by
  neither trigger; a living unit's level never moves; a death resets both
  columns and neither is hashed at its default; the door's trigger table; same
  seed, same rise; the row registered after the bite row; one writer of the
  columns in `src/`; the integrity's gap (a corpse packet still writes the hp).
- `tests/cuda_swarm_check.py` PART 2d (new): the thermal scenario (continuous
  ruleset) with a marine's, a zombie's and a destroyed corpse under larvae
  piles, CPU / per-call / resident: count plane bytes, every unit's (alive,
  zombie, level, integrity) and the tick digest equal every tick for 400
  ticks; the count plane equals the host scan; both corpses rise; the
  destroyed one never moves.
- `tests/cuda_unit_damage_check.py`: the gas lanes gained four corpses (a
  marine's and a zombie's under larvae, counts per variant; a destroyed one
  under larvae; an infested one with nobody on it); the scene must show
  infestation (marine and zombie), decay and the destroyed corpse held; the
  shadow, batch (N 2 and 8, different counts per env), padding (rows copied
  from live rows, corpses included) and chained legs compare the device 9c3
  with the corpse row. The table check gained the four rows' references.
- `tests/cuda_unit_kernels_check.py` PART CORPSE (new): the box match with
  corpses written between ticks (a marine's corpse rises, dies, rises again;
  a zombie's corpse rises; one just below the threshold does not; a destroyed
  corpse a zombie killed AND infested rises by neither): the device
  conversion equals the host every tick, N = 1 with a twin and N = 2.
- Amended (the design, not bent): `tests/_swarm_testkit.py`,
  `tests/test_swarm_bite.py`, `tests/test_swarm_tick.py` pass the count plane
  to `step_swarm` (a required argument now).

## Breaks (each applied alone, the gate run, the file restored)
All went red (`tests/test_corpse_infestation.py` unless named):
1. the row runs on living units too -- red (a living unit's level);
2. the drain reads the level after the rise -- red;
3. the rise ignores n -- red;
4. the rise rounded twice (the rate quantized, then divided) -- red (55 -> 54);
5. the door ignores the infestation trigger -- red (the marine's rise);
6. the door keeps `not is_zombie` on the infestation trigger -- red (the zombie's rise);
7. the door skips the integrity check -- red (the destroyed corpse);
8. the door does not reset the level -- red;
9. a destroyed corpse's level not held at 0 -- red;
10. the death does not reset the corpse -- red;
11. the row not wired into the conductor -- red;
12. the count plane out of the resident set (no exit download) -- red;
13. a second writer of the level outside `corpse.py` -- red (the scan);
14. the integrity hashed at a default of 0 -- red (a death would hash);
15. `active_on` lets coma count (C++, CPU build) -- red (the count plane);
16. the host step does not zero the count plane (C++) -- red (garbage stays);
17. the resident `k_planes` adds no count (CUDA) -- `cuda_swarm_check` PART 2d red at tick 0;
18. the device 9c3 corpse row moves no level (CUDA) -- the damage gate's gas scene red (tick 1, `infest_q` host 440 device 0);
19. the device conversion ignores the infestation (CUDA) -- PART CORPSE red (tick 16: the marine's corpse stays down on the device);
20. the device conversion skips the integrity (CUDA) -- PART CORPSE red (tick 6: the destroyed corpse rises on the device).

## Findings in the reference (reported, not fixed)
- The same-slot packets on corpses: an attack slot and the zombie slot decide
  against their start state, so a target killed earlier in the slot still
  takes the rest (hp below the killing blow, hit events on a corpse). This is
  the only damage a corpse takes today, and it is invisible in the pins only
  where the corpse converts in the same tick.
- The conversion door renamed every converted unit `Z-<name>`; a zombie's
  corpse rising again would have become `Z-Z-...`: the door now prefixes once
  (the name is reference-only).
- A body that falls in a fresh brood rises in two or three seconds: the fresh
  studio broods (2.4 larvae a tile, P5a's finding) are the harshest place; at
  these numbers any corpse inside a brood rises within seconds.

## Left out
Corpse damage wearing the integrity (above: a re-pin decision); a blast or a
fire reaching corpses (a new reach / a new law); the dot view's ring; larvae
seeking corpses (larvae still do not see units, and the corpse is not food);
any change to the larva law or to physics.
