# Swarm units P5a -- the bite (as built, 2026-11-17)

Arc #63, patch P5a. Branch `63-p5a-bite` off `63-swarm-units` at `36f89a2e`.
The shape is the 2026-11-15 handover on #63 ("The bite, as it must be built")
and Erik's boundary ruling of 2026-11-14: a unit law sees the swarm only
through a per-tile plane the swarm writes, read by a coupling row. Rules:
`.claude/rules/swarm.md` (Bite-dose plane), `.claude/rules/units.md` (Coupling
table), `.claude/rules/combat.md` (Poison).

## Step 0: main merged
`git merge --no-ff origin/main` (`6c5e788c`: #29 P7d the device horde stage,
#195 rung 2c) merged with NO conflict (`51e650ad`). The generated layouts were
current (`gen_unit_layout --write`, `gen_swarm_layout` both "current"); both
modules built; the swarm tests ran green (99) before the bite. The twin ledger
showed one merge debt (`horde`: the swarm's `gamemap.py` / `simulation.py`
meeting main's level), cleared by this patch's `cuda_green` like the rest.

## What was built

### The form of the bite: the expected dose, no draw
Of the two honest forms the brief allowed, (b): the plane carries the EXACT
expected dose. Why: it needs no new `Salt`, no regeneration of the draw layout,
no per-larva draw on ticks when nobody stands near (most ticks), and it keeps
the swarm blind to units -- the plane is written whether or not a body stands
on it, so the swarm reads no unit-side field at all. The poison law's onset
(4 s) smooths a discrete bite into the same curve anyway.

Each biting larva adds `bite_q = bite_dose x bite_rate x dt` (Q16 units of
poison per tick, `swarm.bite_tick_q`: the three quantized ints multiplied and
rounded ONCE, half up) to its tile. The plane is the exact integer sum of those,
and a body's dose is the plane's sum under its footprint, so nothing is rounded
twice and the total handed out is exactly the sum read. At the repo's numbers
`bite_q` = 218 (1 unit x 0.08/s / 24 tps = 218.45).

### The swarm side
- Species rows `bite_rate` (0.08/s) and `bite_dose` (1.0 unit), HOT, with their
  reasons in `config.toml`; the kernel table row `bite_q`
  (`swarm_tables.ROWS`, header regenerated); the shared species rule
  `bite_plane_range` (`max_units x bite_q` fits int32).
- THE PLANE `gmap.swarm_bite_dose`, `(h, w)` int32, allocated with the store,
  a resident synced field (`SWARM_PLANE_NAMES`), NOT in the field digest nor in
  the swarm section (a function of the store). Read accessor `swarm.bite_plane`.
- Who bites: ONE `FP_HD bites_on` in `cpp/src/swarm.h` -- a valid slot that is
  not in coma (dead slots are the all-zero form), on its POST-step tile.
- The host path (`swarm::step_all`): when the step is not dormant, the plane is
  zeroed, and after each present species' law (CPU loop or per-call CUDA) the
  host builder `bite_plane_host` adds that species' biting slots. The
  resident path (`resident_swarm`): `cudaMemsetAsync` of the batch's `(N, h, w)`
  plane, then after each species' core a `k_bites` launch over `(hw, N)`:
  integer `atomicAdd`, outside the law (the law keeps no atomics). The exit
  download carries the plane while a swarm is present (slot 9c3 reads it on
  the host the same tick); no third host sync.

### The unit side
- `exchange.apply_bite_dose`, the `swarm_bite_dose | sum` row of the coupling
  table: the footprint SUM of the plane into `poison.dose(u, sum, -1,
  Cause.LARVA_BITE)`, at 9c3 after the gas poison row and before `poison.tick`.
  Immunity is `exchange.poison_immune` (resist exactly 0), now shared with the
  gas row; the dose door refuses the zombie itself. Two bodies on one tile each
  take it in full.
- `events.Cause.LARVA_BITE = "larva_bite"`, appended (no index moves).
- THE CAUSE OF THE POISON: the law emits its packets later, so the cause has to
  travel with the poison. A new unit column `poison_cause` (an index into
  `CAUSE_NAMES`; default `poison_gas`'s, so an unpoisoned or gas-poisoned unit
  hashes exactly as before) is written by every dose (`poison.dose(..., cause)`),
  carried by the law's packet, and returns to the default when both levels
  reach 0, at death and at conversion. ACCEPTED GAP, the credit's twin: the
  latest dose names the cause of all the poison in the body.
- The device twin of slot 9c3 (`cpp/src/cuda_unit_exchange.cu`, Class T) gained
  the bite row in the same patch (the plane a declared input, `GAS_FIELDS`),
  `reduce_sum` in `cuda_unit_damage.cuh`, and the cause column in the dose, the
  law's packet, the drain, `apply_packet`'s kill clear and the conversion clear.
  Not left in debt. `apply_bite_dose` joined the `9c3 gas rows` label
  (`tools/tick_stage_profile.py`), so the chained harness replaces it too.
- The tile inspector's last line: `bite dose this tick: x units of poison`.
- The dot view: NO bite flash. Form (b) has no discrete bite; a flash would be
  a guess (larvae under a footprint), so it is left out.

## Measured: the larva studio, one marine (`docs/swarm_P5a_measure.py`)
Headless, CPU path, seeds 1..5. The marine crosses the nursery west to east
along rows 16-18 (its 3x3 footprint from the west wall to the east wall, 8 m),
or stands 10 s between the two broods. Counted: the bite doses taken DURING
the exposure x 10 HP a unit (the law's total harm, neutral suit), which arrive
over the following ~30 s. Two moments: the broods fresh (t = 0, 2.4 larvae a
tile in the brood boxes) and after a minute of spreading.

| exposure | time in | larvae under the footprint | poison | HP to come, fresh brood (per seed) | after 60 s of spreading |
|---|---|---|---|---|---|
| walk across (1.4 m/s) | 5.5 s | 17.3 fresh / 8.7 spread | 7.6 / 3.8 units | **76** (80, 78, 82, 78, 63) | **38** (35, 46, 32, 42, 36) |
| sprint across (4.5 m/s) | 1.9 s | 16.4 / 9.1 | 2.5 / 1.4 | **25** (24, 24, 27, 26, 22) | **14** (12, 16, 13, 14, 13) |
| stand 10 s | 10 s | 12.6 / 7.0 | 10.1 / 5.6 | **101** (81, 84, 105, 126, 108) | **56** (55, 79, 44, 41, 59) |

The arithmetic holds (dose = larvae under x 0.08 x seconds: walk fresh 17.3 x
0.08 x 5.5 = 7.6). The GAP against the brief's reasoning is in the inputs, not
the law: the studio's fresh nursery is denser than the assumed brood (17 under
a footprint, not 13; 2.4 a tile in the brood boxes, not 1.5), the marine's
walk is 1.4 m/s, not 2, and the nursery is 8 m, not 6, so a walk takes 5.5 s,
not 3. With the brief's inputs the measured law gives its numbers: 13 x 0.08 x
3 s = 3.1 units, ~31 HP. After a minute of spreading the studio is close to the
reasoning's density: walk 38 HP, sprint 14, 10 s standing 56 (death on a delay
only in a fresh brood: 3 of 5 seeds pass 100). `bite_rate` was not tuned.

## Tests (each names its property and the change that breaks it)
- `tests/test_swarm_bite.py` (new): the plane counts exactly the biting larvae
  per tile (coma, dead and freed slots add nothing; prior garbage gone); the
  bite row is the expected dose rounded once (a sweep of rows); the plane does
  not persist (reclaimed at t, zero at t+1); a swarm-free map never writes it;
  a unit over larvae takes exactly the footprint sum (uncredited, larva_bite)
  and one beside them nothing; the dose grows with count (3:1) and time; two
  bodies on one tile each take it in full; the zombie and the resist-0 unit
  take nothing; a suit's 0.5 resistance halves the harm through the pipeline;
  a kill by bite says larva_bite with no credited unit and clears the cause;
  the latest dose names the cause and a drained body returns to the gas
  default; same seed, same doses; the row is registered after the gas row;
  the plane is resident and not digested.
- `tests/cuda_swarm_check.py` PART 2c (new): the thermal scenario with three
  marines on the CPU, per-call and resident paths: plane bytes, every unit's
  poison and cause, hit / kill events and tick digest equal every tick for 200
  ticks; non-vacuous (plane, dosed marine, a bite kill with cause larva_bite
  and attacker -1, 800 coma-larva-ticks alone on their tile with the plane 0).
- `tests/cuda_batch_invariance_check.py` LEG B: each env's bite plane joins
  its fingerprint (env i of the batch == env i alone), and the planes are per
  env (env 0 non-zero, the swarm-free env 3 zero, env 0 != env 1).
- `tests/cuda_unit_damage_check.py`: the gas lanes scene gained larvae (a pile
  that kills BT by bite, larvae under the zombie, some on WA's walk; counts per
  variant); it must show bite hits and a bite kill; the shadow, batch (N 2 and
  8), padding and chained legs compare the device 9c3 with the bite.
- Amended by the design (reported, not bent): P2's `swarm_scenario_sim`
  larvae now bite the default scenario's marine, so its pre-P2 diff test
  accepts unit lines beside the one presence line per tick
  (`tests/test_swarm_digest.py`); `test_swarm_tick.py` passes the plane to
  `step_swarm` and allows it in the write set; `test_swarm_species.py` has the
  new rule's breaker.

## Breaks (each applied alone, the gate run, the file restored)
All went red; one only after the test was improved:
1. the row takes a max, not the sum -- red (unit over larvae);
2. the row skips the immunity -- red (zombie / immune);
3. the row credits the unit -- red;
4. the row doses with the gas's cause -- red;
5. the row splits a tile among the bodies on it -- red (two units);
6. the law's packet keeps `poison_gas` -- red (bite kill);
7. the drain leaves the cause -- red (latest dose);
8. the dose does not write the cause -- red;
9. `bite_q` floored twice -- GREEN first (at the repo's numbers both round to
   218); the test now sweeps 16 rows -- red;
10. `bite_plane_range` never fires -- red (species rules);
11. the row not wired into the conductor -- red;
12. the plane in the digest set -- red;
13. the exit download without the plane (resident) -- PART 2c red at tick 0;
14. `bites_on` lets coma bite (C++, CPU build) -- red (the plane count);
15. the host step does not zero the plane (C++) -- red (count + persistence);
16. the resident `k_bites` not launched (CUDA) -- PART 2c red at tick 0, and
    LEG B red (all planes zero: the non-vacuity check, since env == alone held);
17. the device 9c3 bite row removed (CUDA) -- the damage gate's gas scene red
    (current_hp host != device at tick 1).

## Findings in the reference (reported, not fixed)
- Adding a `Cause` grows `CAUSE_NAMES`, and `agents/sizes.py` sizes the
  network's `hit_cause` input by it (10 -> 11), and `rl_env`'s `deaths_<cause>`
  counters gain `deaths_larva_bite`. A network saved by rung 2c at the old
  size will be refused by the layout check when this reaches main. Nothing in
  the swarm arc trains; the agent's lead should know before the merge to main.
- The studio's fresh broods are denser than any brood a level would hold for
  long (2.4 larvae a tile): the first seconds of the studio are the harshest
  place to stand.

## Left out
The bite flash in the dot view (above); corpses and infestation, larvae
seeking units, food, bodies, any change to the larva's movement law or to
physics; the temperature checkerboard (#72).
