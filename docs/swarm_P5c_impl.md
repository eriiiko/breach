# Swarm units P5c -- the corpse's body (as built, 2026-11-19)

Arc #63, patch P5c. Branch `63-p5c-corpse-body` off `63-swarm-units` at
`8883ba32`. Erik's ruling (2026-11-19): "Explosions and fire take damage on
corpse: yes." "I'd be also perfectly happy with the same HP reaching minus and
[bodies are destroyed] when health is -max_hp." "In either case fire and
explosion and heat still damage after death. I would prefer if a unit of 100hp
is hit by a single bullet with damage 100 + its body HP, that would also be
enough to destroy the body. No matter where or when the damage is dealt."
Erik approved ONE re-pin for this patch (2026-11-19) for the case where only
corpse data moves. Rules: `.claude/rules/units.md` (Corpse law, Body
temperature), `.claude/rules/rl-agent.md` (Reward profile).

## The law

- ONE POOL. The body is the unit's own hp continuing below zero. A corpse is
  DESTROYED when `hp <= -round(body_factor x max_hp)`: `corpse.body_q(u)` =
  `unit_fixed.mul_q16_round(unit_traits.max_hp_q(u), body_factor_q)` (its own
  class's full health on `is_zombie`, times its HP hook, times the dial,
  rounded once); `corpse.destroyed(u)` is the pure read (a living unit never
  is). Device twin `unit_lib::corpse_destroyed` (`cpp/src/cuda_unit_lib.cuh`).
- `[corpse] body_factor = 1.0` (required, refused when missing, unknown, not a
  number or outside (0, 8]; quantized once; device table row
  `corpse_body_factor_q`). `[corpse] integrity` and the `corpse_integrity_q`
  column are REMOVED (schema, config, resets, device store W 185 -> 184,
  table row, tests, rules).
- The killing blow's overkill counts: nothing clamps the hp at death. A 100 HP
  marine hit for 200 is destroyed at once and never rises (a zombie's kill
  included).
- A DESTROYED corpse takes nothing: `damage.apply_packet` returns 0 at once
  (no hp, no event) on host and device; the blast and the heat row skip it;
  its level is held at 0; the conversion door passes it by.
- A risen corpse gets its full hp back from the conversion door (as before),
  so its body is whole for the next death.

## What reaches a corpse

| reach | before P5c | P5c |
|---|---|---|
| same-slot packets (attack slot, zombie slot) | hp below the kill, UnitHit | unchanged (hp, UnitHit); now they wear the body |
| blast (`exchange.apply_blast_damage`, device payload step 3) | skipped (`alive` gate) | the same falloff packet through `apply_packet`, NO event, NO attacker |
| heat row (`apply_environmental_damage`, device `k_heat_damage`) | skipped | the corpse's body temperature keeps running under the same row; above the band the same HEAT packet, NO event, NO attacker |
| poison, gas doses, bites, sprays, statuses, rounds in flight | stop at death | unchanged (stop at death) |

A corpse takes the hp and nothing else: `apply_packet` already gates stress,
exposure, the second death and the conversion flag on `was_alive`; the new
reaches pass `events=None` and attacker -1 (indirect, so `recent_attackers`
is untouched). No `Cause` row: `hit_cause` stays 11 wide.

THE EVENT DECISION: the new reaches emit NO hit event for a corpse (events
feed the rewards, the sounds and the picture; a burning corpse would bill a
hit a tick). Today's same-slot packets keep their UnitHit exactly as before.
The device carries this as `null_sink()` (an `EventSink` with no records,
`emit` returns at once). Consequence in `reward_v1`: the books' check (events
reproduce every unit's hp and life) now reproduces a LIVING unit's hp and
every unit's life; a corpse's hp is outside the books (its health is 0 either
way). Rule amended (`rl-agent.md`, Reward profile).

THE BODY TEMPERATURE: a death no longer clears `body_temp_q` (host
`damage.apply_packet`, device death branch): the corpse is the same body in
the same air, and the heat row keeps running on it, so a unit burned to death
keeps burning ("heat still damage after death"). The conversion still resets
it to ambient. This is a small extension of `body_temp.py`'s caller (the heat
row's `alive` gate) and its device twin; `body_temp.py` itself is unchanged.

ACCEPTED GAP (found here, not changed): corpses are not stamped
(`stamp_units` flattens the living), so a corpse absorbs no radiation of its
own; fire reaches it through the `heat` deposit on its tiles and the hot air
over them (`body_temp.air_read`). Hence a corpse on a burning pile burns, one
at the pile's edge barely does (measured below). Stamping corpses would move
the physics planes and every field digest: Erik's call.

## Measured (`docs/swarm_P5c_measure.py`, CPU path, continuous ruleset)

FRAGS (`frag_standard`, radius 2 tiles, 60 at the centre), larva studio's
nursery, a corpse killed at exactly its hp (hp 0, whole), a frag on the tile
`d` east of its centre tile each tick until destroyed. The hp off each frag
includes the tick's heat after the frag.

| corpse (max hp; body) | d = 0 | d = 1 | d = 2 (outside the blast; heat only) |
|---|---|---|---|
| marine (100; 100) | 2 frags (61, 60) | 4 frags (31 to 34 each) | 21 frags (0.6, 1.2, 1.7 ... growing) |
| turned zombie (100; 100) | 2 frags (65, 60) | 3 frags (35, 39, 30) | 10 frags (2.2, 4.6, 6.9 ...: x4 heat) |

The brief asked for a 400 HP turned zombie: `[zombie] hp` is 100 on this
branch (Erik's bite balance, 2026-11-12, was 400), so a turned zombie's body
is 100 too. At 400 the same frags would need four times the damage.

FIRE: a 3 x 3 furniture pile lit (`studio_ignite`) and burning 20 s
(`tests/test_body_temp.py`'s scene), then the corpse laid down.

| corpse | footprint on the pile's middle | footprint over the pile's east edge |
|---|---|---|
| marine | destroyed after 64.8 s (body peak +64 C) | not in 180 s (hp -8.2; body peak +42 C) |
| zombie | destroyed after 14.0 s (body peak +64 C) | not in 180 s (hp -32.8) |

## The pins (Erik's one approved re-pin)

All eight pinned-match tests moved; for EVERY one the outcome, the team, the
ticks and the field digest are unchanged and only the per-tick unit stream
and the tick digest move. The first divergent tick (a pytest plugin wrapping
`Simulation.step`: the first dead unit whose body is off ambient -- the old
death cleared it -- or whose hp moves with no hit event) is in every case a
unit dying with a body off ambient, which P5c keeps:

| test | outcome / team / ticks / field digest | moved | first divergence |
|---|---|---|---|
| `test_zombie_control::test_the_controller_plays_the_pinned_ship_demo_match` (bots) | unchanged | unit stream, tick digest | tick 458, zombie 8 `Z-0` dies with its body +431 C |
| `test_bots_v1::test_bots_v1_play_the_pre_g8p_pinned_match` | unchanged | unit stream, tick digest | tick 208, marine 4 `T1-0` dies at +9.3 C |
| `test_exposure::test_with_the_law_dormant_the_pinned_match_is_the_pre_g4p_match` | unchanged | unit stream, tick digest | tick 210, marine 4 `T1-0` dies at +16.6 C |
| `test_grenade_kinds::test_with_no_wind_up_the_pinned_match_is_the_pre_g3p_match` | unchanged | unit stream, tick digest | tick 200, marine 5 `T1-1` dies at +1457 C |
| `test_horde_behaviours::test_with_every_behaviour_off_the_horde_plays_the_stage_1_law` | unchanged | unit stream, tick digest | tick 204, zombie 9 `Z-1` dies at +1744 C |
| `test_turn_rate::test_at_half_a_turn_per_tick_the_pinned_match_is_the_pre_g1p_match` | unchanged | unit stream, tick digest | tick 201, marine 1 `T0-1` dies at +0.49 C |
| `test_turning_law::test_g3_at_unlimited_acceleration_the_pins_are_re_pin_1s` | unchanged | unit stream, tick digest | tick 205, marine 1 `T0-1` dies at +21 C |
| `test_unit_impulse::test_g1_with_every_input_off_nothing_is_pushed_and_the_pins_are_re_pin_1s` | unchanged | unit stream, tick digest | tick 208, marine 1 `T0-1` dies at +21 C |

After the first divergence the corpses keep burning and wear their bodies
(in `bots_v1`: zombie `Z-1`, dead at +1071 C at tick 215, is destroyed at
tick 228; `Z-0`, dead at +384 C, wears its body over the following ticks;
marine `T1-0`'s corpse warms in hot air to +37 C, below the band).

RE-PIN STATUS: (filled at the end)

## Tests (each names its property and the change that breaks it)

- `tests/test_corpse_body.py` (new, 12): the body is the hp against the
  unit's own full health (marine and zombie; a living unit never destroyed);
  the killing blow's overkill alone destroys (max_hp x (1 + factor), one count
  less does not; never rises, a zombie's kill included); a destroyed corpse
  takes nothing and emits nothing; today's same-slot corpse packet keeps its
  UnitHit and gives no stress; a frag beside a corpse lowers its hp, enough
  destroy it, then nothing more, no event naming it; a corpse under blasts
  takes no stress, exposure, poison, impulse, conversion flag or attacker; a
  living unit's blast damage and events unchanged beside a corpse (lockstep of
  the two scenes); a hot corpse in hot air is destroyed in finite time, then
  left alone, a cool corpse never moves, no event; the death keeps the body
  and a hot corpse goes on taking heat; a corpse on a burning pile is
  destroyed within two minutes with no event; a corpse destroyed by frags,
  a zombie's kill and infested, rises by neither trigger and holds level 0; a
  risen corpse is whole again.
- Amended: `tests/test_corpse_infestation.py` (the integrity fixtures become
  hp fixtures; the killing packet is exactly the hp; the P5b gap test is
  replaced by `test_corpse_body.py`; the settings and the door test
  `body_factor`; the column's absence asserted), `tests/test_body_temp.py`
  (the death keeps the body), `tests/test_exposure.py::test_the_block_is_zero_after_conversion`
  and `tests/test_unit_impulse.py::test_g13_...` (their fixtures killed with
  1e6 / 10 000 HP: now exactly the hp, since such an overkill destroys the
  body and the corpse would never rise -- the fixture, not the property).
- Device checks: `tests/cuda_unit_damage_check.py` (the fire scene gained a
  marine's and a zombie's corpse on hot deposits and VA's corpse stays in the
  heat; the scene must show "corpse burned", "corpse destroyed by heat",
  "destroyed corpse held", and never a destroyed corpse burned or a corpse's
  event; the gas scene's corpses killed at their hp, the destroyed one by hp;
  the table row reference), `tests/cuda_unit_payloads_check.py` (the probe's
  sims gained corpses at 0, a third, near and at the body's end; the probe
  must take "a corpse in the walk", "a corpse hit", "a corpse destroyed", "a
  destroyed corpse skipped", and fails if a corpse's packet is handed the
  events), `tests/cuda_swarm_check.py` PART 2d and
  `tests/cuda_unit_kernels_check.py` PART CORPSE (the hp-based fixtures).

## Breaks

(filled at the end)

## Findings (reported, not fixed)

- Corpses are not radiation absorbers (unstamped): a corpse at a fire's edge
  barely burns (above). Erik's call whether corpses join the stamp.
- `[zombie] hp` is 100, not the brief's 400 (above).
- The pinned tests' fixtures that kill with a huge overkill (1e6, 10 000 HP)
  now destroy the corpse: any future test that wants a corpse to rise must
  kill it within its body.
