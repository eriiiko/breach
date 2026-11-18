# Body temperature and stress (#215) -- as built

2026-11-17/18. The design page is `docs/body_temperature_design_2026-11-17.md`
(version 4, AGREED: Erik, "Yes it's great!"). Branch `215-body-temp-stress`,
cut from `215-grenade-bounce` (`07b13c0c`, not merged yet: it waits for Erik's
word on re-pinning) with main (`7f7d9e30`) merged in once (no conflict).

## Where each law lives

| Law | Host | Device |
|---|---|---|
| Body temperature (the heat row's energy balance) | `src/simulation/body_temp.py`, called by `exchange.apply_environmental_damage` | `cpp/src/cuda_unit_exchange.cu` `k_heat_damage` over `unit_lib::body_air` / `body_step` (`cpp/src/cuda_unit_damage.cuh`) |
| Stress, its producer door, its tick, the Hill curve | `src/simulation/stress.py`; the producer called from `damage.apply_packet`; the tick at slot 9c3'' (`Simulation.step`, right after `poison.tick`) | the producer and the death clear in the device `apply_packet` (`cpp/src/cuda_unit_damage.cuh`); the tick fused into slot 9c3's `k_gas_rows`; the conversion clear in `cpp/src/cuda_unit_conversion.cu`. The heart rate has no device form (nothing reads it). |
| The clearing step | `src/simulation/clearing.py` (`drain`, `half_life_rate`); `poison.step` calls it | `unit_lib::clearing_drain` (`cpp/src/cuda_unit_lib.cuh`); the poison kernel's step calls it |
| Dials | `[combat] body_radiated_share`, `body_heat_capacity_j_per_k` (fallback), `[env_profiles.*] body_heat_capacity_j_per_k`, `[damage.stress]` | device tables rows `body_a_*`, `body_a_rad_*`, `body_k_temp_q`, `stress_*` (`src/simulation/unit_tables.py`) |
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
    body  += round((a (felt - amb) - k_air (body - T_air) - a_rad (body - amb)) / ONE)
    over   = body - temperature_max  -> the old damage formula

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

## The frag

**Where the 77 heat units came from**: neither the payload row nor the
explosive row. `frag_standard` is a GENERIC EXPLOSION (`strength = 1.0`), and
the door refuses `heat_*` on a strength row: its heat is
`[blast] heat_mj_per_strength x strength` = 10 MJ, landed as a fixed total
over the tiles walked within `2 r_s`, linear falloff -- 77 heat units (2.4 MJ)
on the detonation tile, which the felt temperature read as 1.84 MJ absorbed
(576 HP in one tick).

**The one number**: `[blast] heat_mj_per_strength` 10.0 -> **0.025** MJ, its
reason beside it in `config.toml`. A body whose footprint holds the
detonation tile now absorbs about 4.6 kJ from the deposit: its shell jumps
14 C (`tests/test_body_temp.py` holds 10 to 20 C).

**FINDING for Erik (reported, not acted on)**: this dial is SHARED by every
strength row (the frag, `breach_focus` 0.1, `demolition_c4` 12,
`magazine_cookoff` 8, `engine_death` 3), so all of them now heat the air 400
times less: a blast's own heat ignites almost nothing any more (the engine's
death "a fire follows only if its heat reaches fuel"), and
`tests/test_explosion.py::test_fuel_catches_only_when_ordinary_heat_takes_it_to_its_ignition_temp`
had to move to a stronger row (see Tests). If the blasts should keep heating
the world, the body should read only a share of a deposit (a new dial, a
ruling), not the deposit be cut.

**What still burns a marine beside a frag** is not the deposit: with the
deposit set to nothing the blast's divergence source still drives the air on
the detonation tile to about 1700 C at 0.5 atm for a tick or two, and the
sweep's radiation on the body reads up to about 3800 C felt (the blast's
soot, measured with `heat_mj_per_strength` at 1e-6). The body integrates
that: about 14 to 31 HP adjacent.

### Before and after (frag_standard, a 3x3 marine, 100 HP, the room 40 x 40
at 0.333 m; the distance is the marine's centre tile from the detonation tile;
6 s)

| Distance | Before (base) | After |
|---|---|---|
| 1 tile (0.33 m) | 605.9: blast 30 + heat 575.9 -- dead | 61.4: blast 30 + heat 31.4 -- alive (body peak 231 C) |
| 2 tiles | 287.9 heat -- dead | 8.6 heat -- alive (body peak 126 C) |
| 3 tiles | 44.8 heat | 4.5 heat (body peak 85 C) |

## Hot air: how much it adds (measured, not tuned)

The warming a body took, summed over the run, split into the felt
temperature's flow (radiation and deposits) and the air term when the air was
hotter than the body (degree-ticks):

| Scene | From radiation and deposits | From hot air | Hot air's share |
|---|---|---|---|
| ship_demo `bots`, seed 1, every unit, whole match | 136 | 1245 | 90 % |
| ship_demo `bots_v1`, seed 1 | 237 | 49 | 17 % |
| frag, centre 1 tile away (30 x 30 room), 6 s | 155 | 24 | 13 % |
| frag, 2 tiles | 97 | 1 | 1 % |
| frag, 3 tiles | 31 | 5 | 13 % |
| a small furniture fire (`studio_ignite` on a 3 x 3 pile; fire 0.07), 60 s | 711 / 536 / 240 at 0 / 1 / 3 tiles off | 0 | 0 % (the room's air stays at 20 C) |

In the `bots` match hot air is most of what warms a body (blast-heated and
fire-heated rooms the units cross); in a burning room whose air stays cool
the radiation is all of it.

## What moved, and why it should

**A match with no heat and no damage is unchanged** (bar the new columns):
the `bots` pinned match's per-tick unit stream (every unit's digest record
without `body_temp_q` / `stress_q`, and the tick's events) is identical on
the base and this branch for all 457 ticks before its first heat bill, tick
458 (measured: `diverge.py` in the scratchpad, both checkouts).

**The first tick each pinned match moves is a heat bill, and nothing else
moves there**:
- `bots`: tick 458, the base bills units 5 and 6 2.27 / 2.12 HP of heat
  (one-tick felt temperature); this branch's bodies warm and bill nothing.
- `bots_v1`: tick 221, a frag: the blast's 30 HP on unit 8 is in both; the
  base adds heat bills of 288, 288, 2304 and 1152 HP and kills units 1, 2, 8,
  9 by heat; this branch bills no heat.

**The pinned matches** (NOT re-baselined here; the lead re-pins once for both
#215 patches after Erik's word):

| Match | | Winner, tick | Survivors | Heat damage | Total damage (by cause) |
|---|---|---|---|---|---|
| `bots` (ship_demo, seed 1) | base | team 1 wins, 2554 | team 1 slots 2, 3 | 6463.2 | 7325.7 (melee 204, bullet 551.5, explosion 107, heat 6463.2) |
| | this branch | team 0 wins, 1986 | team 0 slots 2, 3 | 34.0 | 1320.0 (melee 272, bullet 907.0, explosion 107, heat 34.0) |
| `bots_v1` | base | draw, 7200 | team 0 slot 0, team 1 slot 1 | 9102.4 | 9541.4 (melee 136, bullet 183, explosion 120, heat 9102.4) |
| | this branch | team 0 wins, 311 | team 0 slots 0-3 (2 converted zombies alive) | 0 | 730.7 (melee 238, bullet 381.7, explosion 111) |

Explosion damage differs in `bots_v1` (120 -> 111) only after tick 221,
where the matches have already parted on heat; per blast it is unchanged
(30 at one tile in both, above).

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
- `tests/test_explosion.py::test_fuel_catches_only_when_ordinary_heat_takes_it_to_its_ignition_temp`:
  the scene brings its own 10 MJ row (the shipped 0.025 MJ ignites nothing:
  "the scenario is moot", as its own docstring foresaw).
- `tests/test_zombie_control.py::test_the_controller_plays_the_pinned_match_on_a_small_box`:
  the #122 box pin RE-PINNED once (`3a6b2309` -> `4bf0c3eb`): outcome, team,
  ticks and field digest unchanged; with `body_temp_q` / `stress_q` stripped
  from every record this tree gives the old pin EXACTLY (`box_sole_cause.py`
  in the scratchpad, run on the base and here).
- `tests/_xarch_perfield_digest.py` GOLDEN_AGGREGATE re-baselined once
  (`4a13fc72` -> `3246b756`, rationale in the file: every gmap field line and
  every unit sub-hash byte-identical bar `__unit_state__`); its committed
  per-field reference file regenerated. The five CUDA stage tests that assert
  the golden (`test_cuda_p62_sl_advection`, `p66_conduction`, `p68_fire`,
  `p69_combustion`, `s3_water`) and `test_fire_flicker` read it.

### Left RED on purpose: findings for Erik (the law or the calibration working as the page says, with a consequence a ruling should own)

- `tests/test_spray_weapons.py::test_marine_in_flames_loses_hp_via_the_existing_heat_row`
  ("the sprayer does not cook"): on that test's 1 m tile, 24 x 24 sealed
  room, the dragon_7's jet heats the air on the SPRAYER's own tiles to about
  10 000 C at 6 atm within 6 ticks; hot air now warms a body (the page's air
  term), so the sprayer cooks in 9 ticks. Before #215 only the `heat` buffer
  and radiation reached a body, and the nozzle rule kept the deposit off the
  sprayer's tiles. Whether a flamethrower's user should feel its hot air (and
  how much: the air term's 0.85 share scaled linearly with pressure) is
  Erik's. The same scene also reads air under the victim at about -9 400 C
  (game degrees far below the -292 floor) on tiles that hold pressure: a
  temperature mirror that is not physical on near-empty cells (a finding
  about the gas mirror, not changed).
- `tests/test_explosion.py::test_a_breach_charge_breaches_a_door_and_spares_its_planter`:
  the planter one clear tile back now takes 0.41 HP (the charge's hot air on
  his footprint). Same cause.
- `tests/test_explosion.py::test_wind_at_a_fixed_range_grows_with_strength`:
  strength 2 now gives 1.07x strength 1's wind at 8 tiles (the test asks
  1.3x): the 400-fold cut of `[blast] heat_mj_per_strength` removed the part
  of the push the heat's expansion made. More strength is still more wind.
- `tests/test_reward_v1.py::test_the_books_close_on_the_boxes[charge_ff]`: the
  scene is vacuous now (no elimination): the C4 (strength 12) killed the pair
  by HEAT before (120 MJ); it now heats 300 kJ, and its blast alone
  (unit_damage 150 within radius 8) does not kill them. The reward books
  themselves close.

All four come from the shared `[blast]` heat dial and the new air term; the
design page says "Left alone: ... every blast number" while its section 2
calibrates "the frag's heat deposit", which IS a blast number shared by every
strength row. The patch followed the brief (change the one number); the
consequence for C4, the engine's death, the breach charge and the blasts'
fires is the first ruling owed.

## The break list (each new gate broken once, then restored)

Host (run on a scratch copy of the tree, each break alone):

| Break | Went red |
|---|---|
| the radiated share no longer subtracted from the air's | the six steady-state rows, the one-atmosphere identity, vacuum, and five of test_unit_heat_damage |
| the band reads the felt temperature again | the burst (its harm no longer comes over the cooling) |
| the air term loses its pressure scale | vacuum, thin hot gas |
| a ceiling on the body at 3000 C | more joules, more harm |
| the frag's heat back to 10 MJ | the calibration, the frag beside a marine |
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
