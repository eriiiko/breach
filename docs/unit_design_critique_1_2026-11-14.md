# Critique 1 of `unit_design_2026-11-14.md` v1 -- determinism and residency fit

**Lens:** determinism and residency fit. Code read at `12a38642` (main
checkout, read-only); the swarm side read from `63-swarm-units`
(`44170261`) and the untracked `breach-swarm/docs/swarm_merge_main_2026-11-14.md`.

**Verdict: SOUND WITH FIXES.** The shape holds: hooks at `ONE` can be made
bit-identical at every site of section 3, the `traits` block fits the schema's
absent-at-default hashing, and dropping the swarm's `N_ENV` fits
`ResidentBatch`. Seven things must change on paper before U1 starts. The
compose step's moment is wrong for the readers that run between ticks. The
compile step cannot run at a conversion once conversion is a device slot. The
rounding rule is not achievable as written, and one placement overflows int64.
The MELEE hook's push sits in a cache keyed by weapon. The stat-draw seed rule
conflicts with the bank's reset. The P6 gates cannot catch a device law that
ignores its hook. The effect-record boundary leaves its cross-tick queue
undefined.

---

## Findings

### 1. The compose moment: readers between ticks see last tick's `eff` -- MAJOR

**Evidence.**
- `simulation.py:1074-1090` `facing_after_turn_q16`: "the pure evaluation of
  exactly slot 1a's step ... Exact when read after the aim is set and before
  `step`". Controllers call it in `handle_frame`, BEFORE `Simulation.step`.
  It reads `unit_turn_rate_q16` / `unit_turn_accel_q16` (lines 1088-1089),
  which the design routes through `eff` (§3 TURN row).
- `obs_spec_v1.py:1439-1450` (`_Aimer.build`), `:1458`
  (`turning.ticks_to_within`) and `:1233` (`omega / turn_rate`): the
  observation reads the same two dials. `EnvRunner.observe` builds it after
  a step, for the next decision.
- The design §2.4 composes "at the head of every tick, before slot 1a".
  `simulation.py:2014-2044` puts that head inside `_step_pre_physics`, which
  runs after the controllers.

**What breaks.** Between tick T and tick T+1 the frozen block is the one
composed at the head of T. Slot 1a of T+1 uses the one composed at the head
of T+1. A status applied in T (2b DoT, 4c) that carries a TURN term then
makes the predictor and `ticks_to_within` disagree with the step they claim
to evaluate. That breaks the Turning law row's contract ("never a copy of
the law", G9) and the bots' trigger discipline. At neutral hooks nothing
differs today, but this is the shape being fixed.

The same staleness makes the claim "`eff` is not hashed: a function of the
synced state at the tick head" only half true. At a checkpoint (end of tick)
`eff` is a function of the PREVIOUS head's state, not of the state the
digest hashes. So a restore (bank reset §4.6, replay seek) hands
between-tick readers whatever block the store held.

**Smallest fix.** Compose at the tick END (after slot 10 / the team picture,
before the checkpoint), and at spawn, at conversion and after a reset.
- Nothing changes the compose inputs between the end of T and the head of
  T+1. Intents are not inputs, and a spawn composes at once. So every law
  of T+1 sees exactly the value it sees under the current text, and the
  ACCEPTED GAP is unchanged.
- The predictor, `ticks_to_within`, the observation, the masks and the panel
  then read the value the coming tick will use.
- `eff` becomes a true function of the hashed end-of-tick state.

Name the end-of-tick slot in the conductor ("slot 12, compose").

### 2. The compile step cannot run at a conversion on the resident path -- MAJOR

**Evidence.**
- Design §2.2 / §3: "the conversion re-runs the compile step on the same
  sheet with the zombie state". D2: the compile step is HOST only.
- Residency design §4.3 slot 10 row: "`convert_marines_to_zombies` every
  tick in id order" is a DEVICE slot (P6). The same section says "There is
  NO host rare-event path ... 'zero host work per tick' holds on every
  tick". This design's P6c also lists "the tick end (conversion ...)" as a
  device patch.
- Today's conversion (`ai_zombie.py:400-434`, called from
  `ruleset.py:280` / `simulation.py:2662`) takes no `sim`.
- The same issue applies to "later at any event that changes the sheet (an
  item picked up or dropped)" (§2.2) and to "an item's effect is a row of
  numbers read by the compile step" (§4). A pickup happens inside the tick.

**What breaks.**
- Once conversion is a kernel, a host compile at conversion forces a host
  round trip per conversion. That is the rare-event path §4.3 forbids, or
  else the device and the reference diverge.
- U1's gate "a zombie keeps its hooks: a unit with non-neutral hooks,
  converted, has the same block" contradicts "re-runs the compile step". In
  U1 every sheet compiles to `ONE`, so a test that injects non-neutral
  traits is reset to `ONE` by a faithful re-compile. Either the gate fails or
  the implementer drops the re-compile.

**Smallest fix.**
- State that a conversion COPIES `traits` unchanged. Everything that
  differs for a zombie is a CLASS value selected on `is_zombie` (`[zombie]
  hp`, `ZOMBIE_STABILITY`, `zombie_hold_mult`, the shamble), which the
  hook multiplies on the device.
- `compile_unit` runs only at spawn and at host boundaries (bank build,
  reset, a sheet change outside a resident tick).
- An in-tick sheet change (an item) acts through a compose term (device),
  or is deferred to a host boundary. Say which in §4's inventory row.

### 3. "Inside the law's existing single rounding" is not achievable as written; one placement overflows int64 -- MAJOR

**Evidence: most section-3 class values are quantized at a door and cached
before the law runs, so there is no single rounding left to put the hook
inside.**
- SPREAD: `combat.py:105-117` `cone_q16_of_deg` quantizes `radians(spread)`
  once and caches it by VALUE in `_CONE_Q_CACHE`. The draw is
  `draws.py:186-191`.
- HANDLING: `weapons.py:549, 559, 599` turn seconds into ticks once per
  weapon row (`throw_windup_ticks`, `plant_ticks`, `reload_ticks`).
  `combat.py:911`, `simulation.py:1418, 1466` read the cached ints.
- HOLD: `unit_impulse.py:597-601` is `quantize_scalar` of a float stability
  at every read.
- TURN: `simulation.py:430-431` quantizes the two dials once per sim.
- MELEE: `combat.py:2022` takes the whole-point damage. The bite's damage is
  a float `CFG.zombie.melee_damage` (`ai_zombie.py:305`), quantized at the
  packet door.

A hook applied "inside" those roundings would need a float product in the
sim path, which `tests/test_p3e_no_float_sites.py` forbids. So every site
gets ONE extra integer rounding: `(x * h) >> 16`, rounded, after the door.
Bit-identity at `ONE` still holds at every site. `(x * ONE) >> 16 == x`, and
`half_up(a * ONE, b * ONE) == half_up(a, b)` for every integer pair (checked
for odd and even `b`). But the rule as worded will send U1's implementer
looking for a rounding that does not exist.

**Headroom.** The bound `[ONE/8, 8 ONE]` is asserted ("the kernels' integer
headroom"), not argued per site, and one natural placement breaks it. In
`movement.move_speed_q16` (`movement.py:241-264`) the numerator `f32 * cf +
t * c+ * (cw - cf)` is about 2^53 already. `f32 <= 2^32`, and `cf ~ 18 *
65536 ~ 2^20` before terrain (x2.5) and effort raise it. Times a hook of up
to 2^19, that is about 2^72, which overflows the device's int64. Placed in
`movement.default_speed`'s `half_up` instead (`movement.py:159`:
`half_up(speed_class * n * 1000 * ONE, total * SPEED)`), it is about 2^49 and
fine. At `ONE` it is identical to today.

**Smallest fix.**
- Reword the rule: "the hook multiplies the law's integer class value once,
  rounded once by a named rule, after the value's door. At `ONE` that is the
  identity."
- Give each row of §3 its exact site and rounding: SPEED in `default_speed`'s
  `half_up` (both costs, and the zombie cadence), not in `move_speed_q16`.
- Add a one-line int64 range argument per row, as the radiation sweep's
  manifest does.

### 4. MELEE's push lives in a cache keyed by weapon row, not in `melee_blow` -- MAJOR

**Evidence.**
- §3 names `combat.melee_blow` as the site of "the damage and the impulse
  of a melee blow". But `melee_blow` (`combat.py:2001-2043`) applies damage
  and a status only.
- The blow's push is `impulse_table.melee_push_q.get(weapon.name)` in
  `process_melee_intents` (`combat.py:2110-2118`). That is a per-MAP table
  built once and keyed by weapon name (`unit_impulse.py:331-335`).
- The shove reads `table.shove_q` (`unit_impulse.py:911`, built at `:346`).
- The bite has no push at all (`ai_zombie.py:304-312`).

**What breaks.** An implementer following the table hooks the damage only.
The push keeps reading the weapon-keyed cache and silently ignores MELEE. A
"hook is live" test that checks the damage ratio passes. This is exactly a
cache keyed by something other than the unit.

**Smallest fix.**
- List three sites: `melee_blow` (damage), `process_melee_intents` (push:
  `j_q` times the hook after the table read) and `unit_impulse.shove`.
- Say the bite has no impulse, so its MELEE term is damage only.
- The live gate asserts damage AND impulse.

### 5. The stat-draw seed rule conflicts with the bank's reset -- MAJOR (latent until #199; the rule is fixed now)

**Evidence.**
- §2.1 / §5: the sheet is "a pure function of the match spec and the
  seed ... keyed by the match seed and the unit id"; "a replay rebuilds it".
- The farm builds a FRESH session per episode with the episode seed:
  `rl_env.py:769-779`, `episode_draw` at `:249-263`. So it draws new
  soldiers every episode.
- The residency design §4.6: the device bank snapshots the post-spawn TICK-0
  state per template, and "Nothing is re-derived at reset: `reset(mask)`
  copies the template's state ... and sets its Philox key". So the device
  keeps the template's soldiers whatever the episode seed. `traits` are
  hashed once non-neutral (group S).
- The training-episode record (§4.5) replays on the reference through
  `spawn_match` with the episode seed. That re-draws the sheet and diverges
  at the first checkpoint.

**What breaks.** Backend equality (§5 item 4) fails, and so does P8c's
replay gate, the first time a stat is drawn.

**Smallest fix.** Pick the seed of the sheet now and write it into §2.1 and
§5. Either:
- (a) the TEMPLATE's (level / bank row) seed, with the farm spawning from
  the same seed and the episode record carrying it; or
- (b) the episode seed, with the draw moved into the reset (which needs a
  device form of the draw, against D2).

(a) fits D2.

### 6. P6's gates cannot fail for a device law that ignores its hook -- MAJOR

**Evidence.**
- P6a's gate is "the bridge is the identity ... the unit digest computed
  from the rows equals the reference's". P6b's is "live lockstep ... tol 0".
  Both run on "the pinned matches", where every hook is `ONE` (D4) and
  `eff` is not hashed (§2.4).
- A compose kernel that writes `ONE` regardless of `traits`, or a slot-1a
  kernel that reads the table dial instead of `eff`, passes both.
- The residency design §5 item 5 (coverage counters: "the pre-flight
  refuses a feature in the run's config with zero exercise") has no row
  for hooks.

**Smallest fix.**
- The device lockstep corpus includes U1's per-hook scenes (one unit at
  2x and 1/2 per hook, `traits` injected by a test door).
- P6a compares the device `eff` to the host compose explicitly, tol 0.
- Add a coverage counter "hook h != ONE exercised" per hook, so the
  pre-flight refuses a run whose bank has non-neutral hooks with no
  exercised twin.

### 7. The effect-record boundary leaves its cross-tick queue undefined -- MAJOR

**Evidence.**
- engine/17 §5 (branch `63-swarm-units`): the swarm step runs at the END of
  slot 7.
- Design §6.4: "larva bites a marine -> applied by the unit side, in the
  attack resolve" (slot 4c). A record written in slot 7 of tick T is
  therefore applied at 4c of T+1, so the queue crosses the tick boundary.
- The residency design §4.6 clears pending per-env queues at reset. The
  digest has no row for this one (compare the noise log, an ACCEPTED GAP
  for exactly this reason in the Noise row).
- engine/17 §7: "never an atomic ticket counter for spawns (the one provably
  order-nondeterministic pattern)". §6.4 says "the record order is defined
  by the sender's order (slot order)" but gives no append mechanism.

**What is undefined.**
- (a) Whether the queue is synced state: hashed while non-empty, in the
  checkpoint, in the bank. Otherwise two runs that differ only in pending
  bites hash equal at a checkpoint.
- (b) The append from a one-thread-per-slot kernel. It must be a
  slot-ordered scan (prefix sum), not an atomic counter.
- (c) Which records drop on overflow (the highest slots, counted, as the
  ring's per-env failure flag).
- (d) Where the larva records sit in `SimultaneousEffects`' record order
  (`simultaneous.py`; shooters by id x records, then sprayers). A POISONED
  on a unit a shot kills in the same resolve depends on it.
- (e) "the unit standing on the tile" when footprints overlap. The tie rule
  must be the lowest id.

**Smallest fix.** One paragraph in §6.4 settling (a)-(e). Or declare (a)
an ACCEPTED GAP like the noise log, with its reason.

### 8. `eff` fits no group of the unit schema -- MINOR

**Evidence.**
- `unit_schema.py:198-248` and `:720-733`: a `Column` is hashed `always` or
  `nondefault`. No rule says "never".
- Group D is "never written after spawn" (`unit_schema.py:572-620`). Group
  R is "never decided on by the continuous match path" (`:626-665`).
- `eff` is written every tick and decided on by every hooked law, so it is
  none of S, D or R.
- `encode_unit` / `decode_into` (the bridge, `:710-719`) iterate only
  `COLUMNS`.

**Fix.** Declare a fourth group ("per-tick derived: recomputed by
`compose_all`, never hashed, carried by the store and the bridge"). Give it
its own list in the schema, so the P3a sort test keeps every attribute in
exactly one group and the bridge carries it on purpose.

### 9. The P6 split omits slots the design's P6 row covers, and reuses a patch name -- MINOR

**Evidence.**
- The residency design §9's P6 row is "slots 1 / 1b / 2b / 3 / 9 / 9b /
  9c-9d / 10".
- §7's P6a-c name 1a, 1c, 2b, 3, the coupling rows (9c), 9 and 10. They
  omit 1b (where HANDLING's `_start_throw`, `_start_plant` and the reload
  verb live), 1b' (orders), 9b (over-pressure walls) and 9d.
- "P6b" collides with the design's existing P6b (entities on the device),
  which the text keeps "after these".

**Fix.**
- Name 1b / 1b' / 9b / 9d in one of the three patches (1b is per unit row
  with a projectile spawn ordinal, so P6b or P6c).
- Rename the three, e.g. P6.1 / P6.2 / P6.3.

### 10. The HP row misstates today's class values -- MINOR

**Evidence.**
- `simulation.py:578` gives EVERY spawned unit, the match horde included
  (`match.py:358-373`), `base_stats.vitality`.
- Only a CONVERTED zombie takes `[zombie] hp` (`ai_zombie.py:417`).
- `stats.effective_vitality` returns the species vitality for a zombie too
  (`stats.py:55`). Both are 100 today (`config.toml:4941`; species mean
  100).

**Fix.** Write the row as it is: spawn = species vitality (any team),
conversion = `[zombie] hp`, the panel / `bots_v1` = vitality. Otherwise
"bit-identical at ONE" holds only while the two numbers happen to be equal.

### 11. U1 touches a Class-P twin's traced file -- MINOR

**Evidence.**
- `tests/twin_manifest.toml:458-483`: the stamp twin (Class P) lists
  `src/simulation/unit.py` among its generated files. U1 will edit
  `unit.py` (the hp at construction, the new slots).
- `unit_schema.py` is `[not_twinned]` "setup only" (`:128`).

**Fix.** Say plainly that U1 needs a CUDA build in its worktree and a
`cuda_green` stamp at HEAD to merge. "Unless a traced file moves" is
already the case.

### 12. Small site gaps -- MINOR

- SPREAD does not say whether it scales the aimed cone, the snap cone
  (`combat.py:1107, 1192, 1270`) and the beam's cone (`combat.py:1440-1470`,
  `Salt.BEAM_CONE`). Name each.
- HANDLING can round a non-zero wind-up to 0 (7 ticks x 1/8). That switches
  `_start_throw` (`simulation.py:1423-1426`) to the same-tick release along
  the intent's vector, a different law. Clamp a scaled duration to >= 1
  when the class value is >= 1.
- TURN: `TURN_TRIALS_MAX` (32, "the device loop's fixed length",
  `turning.py:68-71`) is checked only at the config door
  (`turning.py:279-285`). With per-unit `w'`, `a'` rounded separately the
  check belongs in `compile_unit`'s door too. The shipped dials need 6, so
  there is no risk at `[ONE/8, 8 ONE]` today.

---

## Checked and found correct

- `traits` as a group-S block absent at its default: `digest_record`
  compares the encoded tuple with the encoded default
  (`unit_schema.py:722-733`). A block of `ONE`s hashes nothing, and
  `apply_defaults` seeds it on every construction (`:699-704`).
- Bit-identity at `ONE` is achievable at every §3 site (finding 3 gives the
  wording). Each class value is an integer after its door, and both
  `(x * ONE) >> 16` and `half_up(a * ONE, b * ONE)` are exact identities.
- HOLD has one reader: `exchange._stability_for` is called only by
  `unit_impulse.stability_q16` (grep). `st^2 <= 2^38` at 8x and `effort <=
  2 ONE`, which fits int64.
- TURN: every reader goes through `Simulation.unit_turn_rate_q16` /
  `unit_turn_accel_q16`: `integrate_all` (`unit_impulse.py:744`), `predict`
  (`simulation.py:1088`) and `ObsWorld` (`obs_spec_v1.py:580-597`). `w` is
  capped at `PI_Q16` inside the law. No other cached copy exists.
- HANDLING: each duration is read at exactly one place per verb
  (`combat.py:911`, `simulation.py:1418`, `:1466`), from per-row ints. The
  hook can enter there.
- The verb gates and the action masks read no hooked value
  (`verb_gates.py`; durations and cones are not terms), so masks built
  before the tick are unaffected by D5.
- The WEGO round rewind: compose has no tick-keyed input, so a rewound
  `sim.tick` changes nothing in it.
- Re-running `bind_mass` at a conversion gives the same `mass_g` today
  (`unit_impulse.py:395-405`; the loadout name is kept, so is the species
  mass). Finding 2's copy rule removes the question anyway.
- `ResidentBatch` (`resident_batch.py:55-67, 87-105`) allocates `(n_env,) +
  shape` for each name in the static `GameMap._RESIDENT_FIELD_NAMES`. The
  swarm names are static too (`SWARM_SPECIES` is closed in code,
  `swarm.py` `SWARM_RESIDENT_NAMES`), and `max_units` is a load-time config
  value. So per-map `(max_units,)` arrays (and 0-d `high_water` /
  `next_unit_id`) stack correctly.
- `SWARM_SECT_V1`'s bytes and the recorder's swarm rings read env 0's
  extent `[0, high_water)` only (`swarm.py` `swarm_carrier` /
  `swarm_section_bytes`; `recorder.py` "(T, wmax)"). Dropping the axis does
  not move them, as §6.3 says. Their `[0]` indexing changes with the
  array's rank (implementation detail).
- The swarm kernel's slot-7 placement (engine/17 §5) fits the Batched
  resident engine's `(N, ...)` launch shape with `blockIdx.y` the env.
- The stat draw through `MatchDraws` with its own `Salt` (counter `(0, k,
  salt, unit id)` at spawn) is consistent with `draws.py` and
  `rng_salts.py`. Only the KEY's source is in question (finding 5).
- Credit for swarm hits to `attacker_id` -1 with its own `Cause` row fits
  `events.Cause` and the reward books' uncredited rest.
