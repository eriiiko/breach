# Body temperature and stress (#215) -- as built

2026-11-17/18. The design page is `docs/body_temperature_design_2026-11-17.md`
(version 4, AGREED: Erik, "Yes it's great!"). Branch `215-body-temp-stress`,
cut from `215-grenade-bounce` (`07b13c0c`, not merged yet: Erik rules one
re-pin at the end of the stacked #215 patches) with main (`7f7d9e30`) merged
in once (no conflict); the bounce's follow-up (the throw leaves the hand, a
grille stops a grenade) merged in up to `30bd61ca` (one generated-header
conflict, regenerated). The measurements of the pinned matches below are on
that final base; the frag's and the hot air's do not depend on the throw.

**Version 2 (Erik's ruling, 2026-11-18)** on the first version's report: "i
want the 10 MJ back. and protect only the body yes"; "explosions look
absolutely stunning now" (blast temperatures and looks are not touched);
"damage should be prop to body temp"; "we can't let near vacuum tiles at
plasma temperatures heat up a soldier"; the flamethrower "treat it as not
really implemented, it's more a place holder". So `[blast]
heat_mj_per_strength` is 10.0 again, and the body is protected by ONE
env-profile number, `heat_absorption` (below). UNDONE from the first
version: the 0.025 MJ calibration and its comment; the fuel-ignition test's
own 10 MJ row; the bounce's in-game throw test reading the thrower's spawn
(both back to the base's text, both green again); the deposit-calibration
test (replaced by the protection's gates).

## Where each law lives

| Law | Host | Device |
|---|---|---|
| Body temperature (the heat row's energy balance) | `src/simulation/body_temp.py`, called by `exchange.apply_environmental_damage` | `cpp/src/cuda_unit_exchange.cu` `k_heat_damage` over `unit_lib::body_air` / `body_step` (`cpp/src/cuda_unit_damage.cuh`) |
| Stress, its producer door, its tick, the Hill curve | `src/simulation/stress.py`; the producer called from `damage.apply_packet`; the tick at slot 9c3'' (`Simulation.step`, right after `poison.tick`) | the producer and the death clear in the device `apply_packet` (`cpp/src/cuda_unit_damage.cuh`); the tick fused into slot 9c3's `k_gas_rows`; the conversion clear in `cpp/src/cuda_unit_conversion.cu`. The heart rate has no device form (nothing reads it). |
| The clearing step | `src/simulation/clearing.py` (`drain`, `half_life_rate`); `poison.step` calls it | `unit_lib::clearing_drain` (`cpp/src/cuda_unit_lib.cuh`); the poison kernel's step calls it |
| Dials | `[combat] body_radiated_share`, `body_heat_capacity_j_per_k` (fallback), `[env_profiles.*] body_heat_capacity_j_per_k` and `heat_absorption` (v2), `[damage.stress]` | device tables rows `body_a_*`, `body_a_rad_*`, `body_absorb_*` (v2), `body_k_temp_q`, `stress_*` (`src/simulation/unit_tables.py`) |
| Readouts | `renderer/hover_readout.py::body_line` (the tile inspector, F6: a living unit on the hovered tile), `ui/model.py::body_text` (the side panel's selected unit) | -- |

Schema: two group-S rows in the health block, `body_temp_q` (Q16 degrees C,
default the ambient `heat_ambient_ref`, hashed while it differs) and
`stress_q` (stress points x 2^24, default 0, hashed while non-zero). Layouts
regenerated (`tools/gen_unit_layout.py --write`): unit store W 182, tables
W 2832.

## The body law as built

Per tick, every living unit (Q16 degrees; `a` per tick):

    a      = J_per_heat_unit / (heat_flux_to_temp x C)       (exact rational, rounded once)
    a_rad  = round(a x body_radiated_share)
    k_air  = min(round((a - a_rad) x p), ONE - a_rad)
    in     = a (felt - amb) + k_air (T_air - amb)          what reaches the body
    body  += round((round(h x in / ONE) - (k_air + a_rad) (body - amb)) / ONE)
    over   = body - temperature_max  -> the old damage formula

- **The protection** `h` = the env profile's `heat_absorption` (Q16; NEW in
  version 2): the share of the heat reaching the body it takes up. It scales
  BOTH inflows alike (the felt temperature's flow and the air's excess over
  ambient), never the losses: the air term `k_air (T_air - body)` is split at
  ambient into its inflow and its loss, so at `h = 1` the step is the first
  version's bit for bit, and in still ambient air the body settles at
  `ambient + h (felt - ambient)`. The damage law is untouched (proportional
  to the body's excess). No existing profile dial meant this: the species'
  HEAT `resist_mult` (the zombie's x4) scales DAMAGE, not what the body takes
  up, and stays where it is. Every profile (`[env_profiles.default]`, the
  `EnvironmentProfile` default the species baseline and the zombie use)
  carries 0.05, so the zombie's relative heat weakness is unchanged.

- **The loss is derived**: `J_per_heat_unit x tps / heat_flux_to_temp` =
  159.2 W/K at 0.333 m tiles (`body_temp.loss_w_per_k`); with 320 J/K the
  time constant is 2.01 s (`a` = 1359/65536 a tick).
- **The air** is read over the footprint's in-bounds GAS tiles: `p` the mean
  of `max(0, atmosphere)`, `T_air` the pressure-weighted mean temperature
  (`ambient + k_temp x round(sum p T / sum p)`; a vacuum tile adds nothing, a
  solid tile is skipped). The page says "the map's existing reads": the
  reduction vocabulary's `_mean_round` and the planes as they are; the
  pressure weighting is the patch's choice (simplest honest form: a tile of
  near-vacuum at -292 C under a foot would otherwise cool a body as if it
  were air). The engine's ambient air (`T_game = 0`, 293 K) is taken as the
  body model's 20 C.
- **No ceiling**: nothing clips what a body absorbs. `k_air` is held at most
  the whole gap a tick, a stability bound of the explicit step (reached only
  above about 56 atm), not a ceiling on energy.
- **Credit**: a tick's heat damage is credited, as before, to the claimant of
  THIS tick's hurting deposit; the cooling's damage after a burst is the
  environment's (an accepted gap, written in `.claude/rules/units.md`).
- **The time constant scales with the tick rate** (the felt scale is per
  tick, so the derived loss is): 2 s at 24 tps, 4 s at 12. The steady state
  does not move. Accepted gap.
- Device range: |body|, |felt - amb|, |T_air| at most 2^44 Q16 degrees, a
  tile's pressure at most 2^25; past it the row FAULTS.

## The stress law as built

- **Currency (the patch's choice)**: the RATE is a fraction in 2^-24 a tick
  (`clearing.RATE_SHIFT`): 120 s is 4038 counts, 1200 s is 404 (Q16 would
  hold 16 and 2). The LEVEL is in 2^-24 points too, not Q16: the shared step
  truncates, and at a 20-minute half-life a Q16 level of 10 points drains
  about 16 counts a tick, so the truncation alone stretched that half-life by
  4.7 % in the first measurement; at 2^-24 the bias is a part in ten thousand.
- **The shared step serves both without touching poison's arithmetic**:
  `clearing.drain(x, a, shift, flush_q)`; poison calls it at shift 16 with its
  own flush when the depot is empty. Bit for bit against the frozen old step
  on 20 000 random inputs (`tests/test_stress.py`).
- **The producer door**: `stress.produce(u, row, amount_q)`, rows in
  `stress.PRODUCERS` (only `damage`: `stress_per_hp` points per hit point
  actually lost, from `apply_packet` after the hit points are written and
  before the death check). A heal or a hit that took nothing adds nothing.
- **A zombie takes none** (the patch's choice, like poison; Erik's to
  overturn). Death and conversion clear the level (and reset the body to
  ambient).
- **Flush** at 1/16 point, so the level returns to exactly 0.
- **The Hill curve**: an integer table of 8193 entries (64 per half stress, to
  128 halves) built at load in exact rationals (`rest + (max - rest) x i^n /
  (64^n + i^n)`, the stress cancelling into the index), linearly interpolated
  (floored) between entries, the last entry past its reach. The exponent must
  be a whole number in [1, 8] (the door refuses another: an exact table
  needs no roots).
- **The bound**: the level at most 2^38 counts (16 384 points; a production
  past it clipped and counted, host-side diagnostic); a weight is 0 or in
  [1/1024, 16] points per hit point; a hit of 2^44 Q16 points or more fills
  the level whatever the weight (the device product's bound, the same rule on
  the host).

### The half-lives, measured on the law

| Half-life | Rate (2^-24 / tick) | From 10 points | From 100 | From 5000 |
|---|---|---|---|---|
| 120 s | 4038 | 120.00 s (+0.000 %) | 120.00 s | 120.00 s |
| 1200 s | 404 | 1199.58 s (-0.035 %) | 1199.42 s (-0.049 %) | 1199.38 s (-0.052 %) |

The Q16 alternative for comparison: 118.8 s and 981.4 s from 10 points.

## The protection's calibration (version 2)

Candidate values of `heat_absorption` on the marine (every profile alike),
`[blast]` at 10 MJ. Frag: `frag_standard` with the marine's centre tile 1, 2,
3 tiles from the detonation (40 x 40 room, 0.333 m, 6 s, heat HP; the
blast's own 30 at 1 tile in every row). C4: `demolition_c4` at 2 and 4 tiles
(heat HP; blast 112 / 75). Fire: a 3 x 3 furniture pile lit by
`studio_ignite` on each tile and burnt 20 s (its fire peaks at 0.12), then a
marine whose footprint overlaps the pile's east edge ("in") or stands one
tile beyond it ("beside"): HP per second over 10 s / over the last 5 s, and
the body where it settles (`calib.py` in the scratchpad).

| h | frag 1 / 2 / 3 tiles | C4 2 / 4 tiles | fire, in (HP/s) | fire, beside (HP/s) |
|---|---|---|---|---|
| 1.0 (unprotected) | 687.7 / 375.9 / 73.0 | 897.6 / 682.8 | 22.8 / 26.5 (body 546 C) | 15.0 / 17.4 (body 373 C) |
| 0.2 | 132.7 / 70.4 / 8.5 | 174.7 / 131.8 | 3.8 / 4.5 (body 125 C) | 2.2 / 2.7 (body 91 C) |
| 0.1 | 63.4 / 32.0 / 0.5 | 84.4 / 62.9 | 1.35 / 1.75 (body 73 C) | 0 (body 55 C) |
| **0.05 (chosen)** | **28.0 / 12.2 / 0.0** | **39.2 / 28.4** | **0 (body 46 C)** | **0 (body 38 C)** |
| 0.03 | 14.0 / 5.1 / 0.0 | 20.1 / 13.5 | 0 (body 36 C) | 0 (body 31 C) |

**No single value meets both targets** (a frag under 40 heat HP AND a fire
above about 2 HP/s): the flash is a huge flux for a tick, the fire a small
one for long, and one linear protection scales both. As the ruling says, no
second mechanism was added: the best single value for target (a), **0.05**,
is built (a frag on the next tile bills 28 HP of heat on top of its 30:
58 HP, the marine lives -- the old `[blast]` comment's "about 54 HP left at
1 m" was 46 lost). What target (b) became: standing in this furniture fire
no longer hurts (the body settles at 46 C, under the 60 C limit); h = 0.2
gives 2 to 4.5 HP/s in it but a frag then bills 133 HP. Erik's to set.

### The frag before and after (shipped values, the marine 100 HP)

| Distance | Before (base) | After |
|---|---|---|
| 1 tile | 605.9: blast 30 + heat 575.9 -- dead | 58.0: blast 30 + heat 28.0 -- alive (body peak 316 C) |
| 2 tiles | 287.9 heat -- dead | 12.2 heat (body peak 173 C) |
| 3 tiles | 44.8 heat | 0 (body peak 41 C) |

## Near vacuum (measured, the sweep not changed)

A 3 x 3 patch just east of a marine's footprint, its bulk N pinned each tick
to a fraction of ambient and its temperature seeded through
`GameMap.seed_gas_temperature` (`vacuum_rad.py`, a scratch bench); no solid
hot surface. What reaches the body through the felt temperature (the sweep's
`rad_flux_sweep` on its tiles), averaged over the last 40 of 48 ticks (the
pinned pressure is the tick's start; air flows back in within the tick, so
the patch's end-of-tick pressure is shown):

| Patch | Irradiance on the body |
|---|---|
| no smoke, N x 0.005 (p 0.088 atm at tick end), 3000 game degrees | 0.00 kW/m2 |
| no smoke, full density (1.6 atm), 3000 | 0.00 kW/m2 (clear air does not radiate: Kirchhoff) |
| smoke 1.0, N x 0.005 (0.088 atm), 3000 | 0.00 kW/m2 |
| smoke 1.0, full density, 3000 | 0.49 kW/m2 |
| smoke 5.0, N x 0.0005 (0.063 atm), 10 000 | 0.00 kW/m2 |
| smoke 5.0, N x 0.005 (0.80 atm), 10 000 | 0.30 kW/m2 |

Thin hot gas reaches a body through the radiation path with NOTHING
measurable: not a finding in the sweep. The air path already scales with the
pressure (and now with the protection).

## What moved, and why it should

**A match with no heat and no damage is unchanged** (bar the new columns):
the `bots` pinned match's per-tick unit stream (every unit's digest record
without `body_temp_q` / `stress_q`, and the tick's events) is identical on
the base (`30bd61ca`) and this branch (version 2) for all 457 ticks before
its first heat bill, tick 458 (`diverge.py` in the scratchpad, both
checkouts; the same on the first version).

**The first tick each pinned match moves is a heat bill, and nothing else
moves there** (version 2, against `30bd61ca`):
- `bots`: tick 458, the base bills two units a couple of HP of heat (one-tick
  felt temperature); this branch's protected bodies bill nothing.
- `bots_v1`: tick 221, a frag: the blast's 30 HP on unit 8 is in both; the
  base kills units 1, 2, 8, 9 by heat (288 to 2304 HP each); this branch
  bills 0.26, 0.26, 2.25 and 1.03 HP that tick and goes on billing a little
  each tick while the bodies cool.

**The pinned matches** (NOT re-baselined here; the lead re-pins once for both
#215 patches after Erik's word):

| Match | | Winner, tick | Survivors | Heat damage | Total damage (by cause) |
|---|---|---|---|---|---|
On the final base (`215-grenade-bounce` `30bd61ca` + main):

| Match | | Winner, tick | Survivors | Heat damage | Total damage (by cause) |
|---|---|---|---|---|---|
| `bots` (ship_demo, seed 1) | base | team 1 wins, 2121 | team 1 slot 3 | 6486.4 | 7315.5 (melee 204, bullet 518.1, explosion 107, heat 6486.4) |
| | version 1 (0.025 MJ) | team 1 wins, 1871 | team 1 slots 2, 3; zombies 0, 1, 2; 4 marines turned | 47.5 | 1389.6 (melee 748, bullet 487.1, explosion 107, heat 47.5) |
| | **version 2** (10 MJ, h 0.05) | **team 0 wins, 2227** | team 0 slots 1, 2; zombies 2, 3 | 100.0 | 1143.0 (melee 272, bullet 664.0, explosion 107, heat 100.0) |
| `bots_v1` | base | draw, 7200 | team 0 slot 0, team 1 slot 1 | 9102.9 | 9541.9 (melee 136, bullet 183, explosion 120, heat 9102.9) |
| | version 1 (0.025 MJ) | team 0 wins, 267 | team 0 slots 0-3; zombies 0-3 | 0 | 670.9 (melee 136, bullet 397.9, explosion 137) |
| | **version 2** (10 MJ, h 0.05) | **team 1 wins, 2455** | team 1 slot 3; zombie 1; 1 marine turned | 286.0 | 1227.4 (melee 170, bullet 604.5, explosion 167, heat 286.0) |

Hot air's share of the warming in version 2 (the inflows before the
protection, which scales both alike): `bots` 30 %, `bots_v1` 18 %.

(On the first base, `07b13c0c` + main: `bots` team 1 at 2554 -> team 0 at
1986, heat 6463 -> 34; `bots_v1` draw at 7200 -> team 0 at 311, heat 9102 ->
0.) Explosion damage differs in `bots_v1` only after the first heat bill,
where the matches have already parted; per blast it is unchanged (30 at one
tile in both, above). The divergence proof below was taken on the first
base (the follow-up changed where a throw starts, not the heat).

**The field digest**: NO version bump. The two columns are unit-digest rows
hashed while they differ from their default (the poison columns' rule): the
digest's membership (the planes, their dtypes, their currencies) is
untouched, and a unit whose body stays at ambient with no stress hashes as
before. What moves is the unit hash of any match where a body leaves ambient
(ship air is not exactly at ambient, so in practice every match) -- see the
goldens below.

## Tests and breaks

New gates: `tests/test_body_temp.py` (steady state over the literature
table's six rows; the derived loss; the burst; proportionality with no
ceiling; the one-atmosphere identity; vacuum cools by the radiated share
only; thin hot gas barely warms; the air read over gas tiles by pressure; the
frag's deposit calibrated; the frag adjacent no longer kills by heat; one
writer; death resets; a cold body never moves; integer determinism; the
readouts) and `tests/test_stress.py` (the lifted step is poison's bit for
bit; both half-lives within 1 %; back to exactly 0; fed by hit points
actually lost and not by a heal or a zero hit; a zombie and a corpse take
none; conversion clears; the door refuses an unknown producer and bad dials;
the Hill curve's shape; one writer; one door; no reader in `src/`).

Existing tests changed, each because the law it held was deliberately
replaced:
- `tests/test_unit_heat_damage.py`: the per-tick tests now hold the SETTLED
  body (`_settle`, 80 s) to the old per-tick law within a Q16 count; the stub
  map carries one-atmosphere ambient air and the tile size.
- `tests/test_damage_pipeline.py::test_heat_site_end_to_end_bit_identity_marine_and_zombie`:
  the hand-written law now includes the body's first step; the comparison
  with the pre-P2 float chain (the instantaneous law) is dropped.
- `tests/test_poison.py::test_a_level_returns_to_exactly_zero_and_hashes_as_never_poisoned`:
  ONE line -- the never-poisoned twin, which already copies the hit points the
  poison took, copies their stress too. Poison's law and every other poison
  test are untouched.
- `tests/cuda_unit_damage_check.py`: the tables gate reads the new rows; the
  ship chain no longer demands ring events (its 500 ticks hold no hit since
  the body law; the fire and gas lanes carry the events).


More changed tests (found by the not-slow tier):
- `tests/test_hit_direction.py::test_a_heat_hit_carries_no_direction`: its heat
  stub carries the footprint's air and the tile size (the property untouched).
- (version 1 only, UNDONE in version 2: `test_explosion`'s fuel-ignition
  scene brought its own 10 MJ row; with the 10 MJ back it is the base's text
  and green.)
- Version 2: the modules that hold the UNSCALED heat-row law
  (`test_unit_heat_damage.py`, `test_hit_direction.py`, the pipeline's heat
  test, and `test_body_temp.py`'s literature-table, burst and hot-gas tests)
  run it at `heat_absorption` 1 through a monkeypatched `absorption_q`, said
  in each; the shipped value is held by the protection's gates and the frag
  test.
- `tests/test_zombie_control.py::test_the_controller_plays_the_pinned_match_on_a_small_box`:
  the #122 box pin RE-PINNED once (`3a6b2309` -> `4bf0c3eb`): outcome, team,
  ticks and field digest unchanged; with `stress_q` alone stripped from
  every record this tree gives the old pin EXACTLY (`box_sole_cause.py` in
  the scratchpad; the box's bodies stay at ambient). Unchanged by version 2.
- `tests/_xarch_perfield_digest.py` GOLDEN_AGGREGATE re-baselined once
  (`4a13fc72` -> `3246b756` in version 1, -> `cf3c7693` in version 2;
  rationale in the file: every gmap field line and
  every unit sub-hash byte-identical bar `__unit_state__`); its committed
  per-field reference file regenerated. The five CUDA stage tests that assert
  the golden (`test_cuda_p62_sl_advection`, `p66_conduction`, `p68_fire`,
  `p69_combustion`, `s3_water`) and `test_fire_flicker` read it.

### Version 2: what came back, what is an expected failure

- `tests/test_explosion.py::test_a_breach_charge_breaches_a_door_and_spares_its_planter`,
  `::test_wind_at_a_fixed_range_grows_with_strength` and
  `tests/test_reward_v1.py::test_the_books_close_on_the_boxes[charge_ff]`:
  GREEN again by themselves with the 10 MJ back (the wind's heat part is back;
  the C4 kills the pair again; the planter's 0.41 HP was hot air the
  protection now cuts to nothing measurable).
- `tests/test_spray_weapons.py::test_marine_in_flames_loses_hp_via_the_existing_heat_row`:
  still red after the protection (the sprayer loses about 7 HP in 12 ticks:
  its jet's air on its own tiles is about 10 000 C at 6 atm on that test's
  1 m tile, sealed 24 x 24 room), so it is an EXPECTED FAILURE
  (`xfail(strict=False)`) with Erik's ruling of 2026-11-18 in its reason: the
  spray weapon is a placeholder, not tuned for.
- FINDING kept for the lead to file (the gas temperature mirror, not
  changed): in that scene, `tests/test_spray_weapons.py` `_level()` with
  `_shooter("dragon_7")` at (5, 9) firing at (12, 10) and a victim at (8, 9),
  the pressure-weighted air read under the victim's footprint is about -9 400
  game degrees from tick 1 to tick 9 (-9358 at tick 1, -9596 at tick 8; far
  below the -292 floor) on tiles that hold pressure (the read weighs only
  tiles with p > 0): some gas cells there carry a temperature mirror that is
  not physical.

## The break list (each new gate broken once, then restored)

Host (run on a scratch copy of the tree, each break alone):

| Break | Went red |
|---|---|
| the radiated share no longer subtracted from the air's | the six steady-state rows, the one-atmosphere identity, vacuum, and five of test_unit_heat_damage |
| the band reads the felt temperature again | the burst (its harm no longer comes over the cooling) |
| the air term loses its pressure scale | vacuum, thin hot gas |
| a ceiling on the body at 3000 C | more joules, more harm |
| the frag's heat back to 10 MJ (version 1's gate; version 2 ships it) | the calibration, the frag beside a marine |
| v2: the protection scales only the felt temperature's flow (the air's inflow unprotected) | the protection's air row, the frag beside a marine |
| v2: the protection scales the losses too | both protection rows, the frag beside a marine |
| a second writer of the body (the wind row) | one writer |
| the stress rate in Q16 | both half-lives |
| ln 2 off by 1 % | the 1200 s half-life |
| the producer reads the packet's raw amount | fed by hit points actually lost (behind armor) |
| a zombie takes stress | a zombie and a corpse take none |
| the shared step changes poison's flush | test_poison (one dose rises, peaks, decays) |
| the Hill table built off its formula | the Hill curve's shape |
| a bot imports the stress module | no reader |
| death keeps the stress | a zombie and a corpse take none |

(Each first ran GREEN for four breaks -- the band on the felt temperature, a
ceiling, the Q16 rate, the raw amount -- and the gates were sharpened until
red: the harm after a burst over the cooling, bursts of 300 kJ and 3 MJ, the
shipped rate through the door, a marine behind 4 points of armor.)

Device (`cpp/src/cuda_unit_exchange.cu`, one rebuild, both breaks together,
`tests/cuda_unit_damage_check.py` scenes + chained): the heat kernel's band
on the felt temperature -> the fire scene's heat slot vacuous and its hits
missing; the stress tick draining nothing -> `stress_q` host != device in
the ship, fire and gas scenes. Restored and rebuilt.

## Gates

(appended at the final tree)
