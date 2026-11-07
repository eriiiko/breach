# Residency + batching design: critique, lens 2 (systems reuse and scope)

Date 2026-11-07. Critic: Claude (Opus), independent; read §0 to §13 of
`docs/residency_batching_design_2026-11-07.md` as of `7cb73ed`, not §14 and
no other critique. Line numbers below are that file's unless a path is given.

**VERDICT: NOT SOUND as v1.** Under this lens the route holds (farm first,
staged device backend, snapshots with catch-up). What has to change before any
patch is cut, all of it on paper: the device form of the topology seam, an
oracle the device can actually compute, and the patch plan.

---

## BLOCKERS

### B1. On the device backend, topology changes go around the booked topology seam

- **Where:** §4.3 l.238 ("level topology changes ... the host, as edits plus
  mask patches"), l.231 (doors on the device, P6b), l.440 (9b walls on the
  device, P6), §11 l.467.
- **Evidence:** the CLAUDE.md GameMap row says topology changes go "only via
  `destroy_wall`/`seal_tiles`". `destroy_wall` (`src/simulation/gamemap.py:2517`
  ff.) is not a mask write. It carries:
  - the P-M3 mass-books bracket with its named seed channel
    (`n_destruction_seed_sum`);
  - boundary-reservoir evacuation for hull tiles;
  - `gas_energy_retire` and the ambient seed;
  - the membership diff in `on_tile_changed`.

  `seal_tiles` / `unseal_tiles` book the solid energy through
  `_solid_energy_book` (CLAUDE.md closure row: `e_solid_seam_sum`, "the
  identity closes on the tick a slot-9 seam fires"). The callers are not rare:
  - `combat.py:192` (bullet chew, P7: "bullet chew must be able to break a crate");
  - `door_system.py:241-259` (doors and airlocks, every cycle, P6b);
  - `simulation.py:1670/1686` (slots 9 and 9b);
  - `physics.py:124` (`apply_explosion`).

  §4.3's frequency bound covers blasts only. Slot 9 (burn-through) is not
  assigned anywhere in §4.3. Units do not exist on the host on the device
  backend (l.267). The design never says whether a host `GameMap` exists for
  `destroy_wall` to run on. "Edits plus mask patches" names none of the books,
  so the energy-closure identity and the mass books break on the first chewed
  crate or the first airlock cycle.
- **Fix:** pick ONE device form of the topology seam and gate it. Either:
  - (a) `destroy_wall` / `seal_tiles` / `unseal_tiles` as device functions
    that make the same writes and book into the same channels
    (`n_destruction_seed_sum`, the gas energy seam's retire / mint, the
    solid-energy seam). Gate: a device twin of `test_gas_energy_tile_flip.py`
    plus the closure identity on one scene with a burst, a door cycle and a
    chew. Or:
  - (b) a per-env host shadow of the affected window. The real functions run
    on it and every write (planes, masks, books) goes back as edits.

  Then:
  - put the choice in §13(b) with a rule;
  - amend the GameMap row to "... or their device form";
  - assign slot 9;
  - re-file bullet chew and doors as per-tick events, not rare ones.

### B2. The device cannot compute the arc's gameplay oracle (gate 3)

- **Where:** §5.3 l.369 (replay lockstep compares `(field_digest, tick_digest)`),
  §4.4 l.259 (WEGO fields are not columns), §4.5 l.287 (events become counters
  `(N, U, E)`), §5.7 l.382 (units "enter the entity section (a version bump)").
- **Evidence:** `tick_digest` (`tests/field_digest.py:200-239`) folds in the
  `UNIT_DIGEST_KEY` hash from `field_ab_harness._capture_unit_state`
  (`tests/field_ab_harness.py:255-270`). That hash covers:
  - `_unit_record` per unit (l.179-218): it hashes `ap` and `n_orders`
    (WEGO), x / y quantised at 1e-9, `offsets`, the `life_state` string, and
    `statuses` via `status.serialize_statuses`;
  - the ORDERED per-tick list of `_event_record` (kind, unit_id, damage,
    source, killed_by, attacker_id) for UnitHit / UnitKilled, in emission
    order.

  The device as designed keeps neither `ap` / `n_orders` nor an event stream.
  Units are not in the entity section. That section is `ENTITY_SECT_V1`,
  absence-transparent, and "do[es] NOT drive DIGEST_SPEC_VERSION"
  (field_digest.py:204-216). The unit hash has no version at all. So the
  "replay lockstep" gate of P6, P6b and P7 has nothing on the device side to
  compare, and the change procedure in §5.7 is the wrong one.
- **Fix:**
  1. A per-env, per-tick event ring written by the walk-order apply (§5.5)
     with exactly `_event_record`'s fields. The tick digest's event list AND
     the reward counters both derive from it: counters are a reduction of the
     ring, so credited events have one source, not two.
  2. P3's schema commit makes `_unit_record` read the schema's digest flag and
     settles `ap` / `n_orders` explicitly. Either drop them from the record (a
     deliberate re-pin of every pinned tick_digest, ship_demo seed 1's among
     them) or keep them as columns the device holds constant.
  3. Rewrite §5.7 to the unit section's real procedure: a tick_digest re-pin
     with rationale, with GOLDEN_AGGREGATE checked.

---

## MAJORS

### M1. §13(a) misses canonical systems the mechanisms touch

- **Where:** §13(a) l.487-505.
- **Evidence** (each a CLAUDE.md row with a rule the twin or the host path must keep):
  - **Vent / duct system:** plenum ledger R3-counted, measured-delta booked,
    ENTITY_SECT-digested. P6b ports it; §13(a) names only the pump primitives.
  - **Heat source** and **`GameMap.blast_log`:** "THE seam a slot-9e entity
    reads to react to blasts". Blasts are on the host and heat sources on the
    device, so the log must cross.
  - **Timed charge:** the heat trigger compares the temperature mirror at 9e
    (device) and fires `execute_payload` (host), a device-to-host event.
  - **Coupling table, Wind load on a body, Footprint samples builder:** 9c,
    9c2, 9c3 and movement's `default_speed`. The builder's gate scans `src/`
    only, so a kernel twin sits outside it.
  - **Vision / engagements / cover:** `has_los` and `sight_screen`, for
    shooting phase A and zombie activation.
  - **Weapons:** DamagePacket, "nothing writes `current_hp` directly", yet the
    kernels write the hp column.
  - **Starting a fire:** 9d on the device; `IGNITE_HEAT` for the host path.
  - **Generic explosion, Blast reach, Blast air read, Fixed-total tile
    deposit, Divergence source:** these ARE the host rare-event path.
    `div_source` is written in Python ints by `blast_expansion` and consumed
    at the head of the next physics step.
  - **Match controller registry:** the farm's policy and the horde (see M6).
  - **Light accessor** and the sweep's light group (see M4).
  - **Ingress lint, float ratchet, `/fp:strict` list:** new device sim TUs
    join at 0/0/0. The obs gather's Q16-to-float conversion must sit outside a
    scanned sim TU or carry an `ingress-exempt:` reason.
  - **RL-batch habits:** this design supersedes its source doc.
  - **Integer reference:** `sweep_ref_q.py`. P9's sweep reschedule and any
    128-bit narrowing land THERE first. §4.11 says only "the CPU reference".
  - **Extinction planes, Prop tile stamps:** the stamp kernel's fourth and
    fifth outputs (`dyn_heat_atten_q` is digested) and the bank's prop stamps.
  - **Level data layer and LevelSpec `[levelgen]`:** the bank's reader and
    its provenance (see M10).
  - **`unit_fixed.py`:** the one Q16 boundary for synced unit state (§7's
    positions).
- **Fix:** add each to §13(a), with one line saying what the device twin or
  the host path must keep.

### M2. Existing rules and canon chapters the design contradicts are not scheduled for amendment

- **Where:** close row l.446 ("CLAUDE.md rows (§13 b)"). The header says only
  that engine/02 is "revisited".
- **Evidence, CLAUDE.md rows:**
  - **Simulation facade:** "The only tick entry". The device backend is a second one.
  - **Tick conductor:** slot 5 is "the law's ONE driver". The horde kernel is a second driver.
  - **GameMap:** see B1.
  - **FieldEdit:** "The only way to write a continuous field". §13(b).4 should
    amend this row, not add a second row for the same system.
  - **Payload executor:** now runs on the host on a windowed read.
  - **A\*:** "The only pathfinder". The distance field lands in `src/` at P3,
    so the amendment belongs to P3, not to close.
  - **RL-batch habits:** superseded by §13(b).2.
  - **Unit:** "One class".
  - **Weapons:** hp written by kernels.
  - **Match controller registry:** "every unit has ONE driver and it is a controller".
  - **Match recording / replay:** "ONE tap".
  - **Headless match runner:** "never a second loop".
  - **Spawn rooms:** "`spawn_match` is the ONE match spawn". The device reset places units.
  - **Recorder:** see M9.
  - **Sensor accessor:** #25's resident backing gets built.
- **Evidence, canon chapters:**
  - **`engine/02_state_and_ownership.md`:**
    - l.28-33 and l.269: "Entity/logic state → stays CPU";
    - l.15-19: the facade "the only place world state is mutated";
    - l.326 ff.: "a single seeded `numpy.random.Generator` on `sim.rng`".
  - **`ml/01_ml_and_training.md`:**
    - §1: "The environment IS the Simulation facade";
    - l.13-16: "training and gameplay run identical logic. There is no 'AI
      mode' fork of the rules." The device backend IS that fork, accepted by
      Erik's snapshot ruling and policed by gates, and the chapter must say so;
    - §7: batch A\*.
  - **`engine/10_pathfinding.md` §3:** zombies take the nearest marine plus
    A\* every 5 steps; "batch A\* on GPU" is the plan.
  - **`mechanics/02_ai_and_los.md`:** the zombie AI.
  - **`engine/14_determinism_and_number_ingress.md`:** door 4's one stream.
- **Fix:** add a §13(c) "Amendments" listing each row and chapter, each
  assigned to the patch that makes the amendment true.

### M3. The obs gather is a second gather beside the sensor accessor, and the GLOBAL map breaks RL-ready property 3

- **Where:** §4.5 l.279-285.
- **Evidence:**
  - `src/simulation/sensor_accessor.py` header: exactly ONE accessor. Its
    resident backing is the `(n_sites × n_channels)` int32 gather, the channel
    order frozen (`CHANNEL_ORDER`), and `solid` is already channel 7.
  - `rl_ready_scope_2026-10-29.md` §3.3: "A team view is the union of its
    units' sensor-accessor reads; no global observer in the engine".
  - `rl-ready-lead` §4: "Observation (full map vs local crop, line of sight)"
    is agent design.

  The design's crop is "the sensor accessor's seven channels plus material /
  solid / ...". It adds a per-env GLOBAL map "for the team shape", computed by
  the engine's obs kernel. The very next sentence composes the team view from
  crops.
- **Fix:**
  - The crop becomes a method of the accessor (`crop(unit, r, channels)`),
    with the #25 kernel as its device backing: one gather, one channel order,
    new channels appended under the frozen-order rule.
  - Drop the GLOBAL map from v1. A full-map observation is an agent-era
    ruling for Erik.

### M4. A light channel in the observation pre-empts #81, and headless runs never compute light

- **Where:** §4.5 l.281 ("light where requested").
- **Evidence:**
  - CLAUDE.md radiation-sweep row: "`light_q` IS NOT SIM-PURE UNTIL #81 ...
    no rule may read it before then". Its inputs are render state
    (`frame_lights.EMITTER_INPUTS`: cursor lamp, flashlight mode, the W6 queue).
  - Light-accessor row: headless runs never request light.
- **Fix:** the obs table's door refuses a light channel until #81. List it in
  §11 as an `ACCEPTED GAP:`.

### M5. The `[rl.*]` tables overlap match.toml, and they tie every reward experiment to a snapshot

- **Where:** §4.1 l.166 (`BatchedEnv(bank, n_envs, backend, tables)`), §4.5
  l.289-291, §13(a) l.494 (`[rl.*]` in config, "bound once"), §6 l.392
  (`require_snapshot` refuses a dirty tree).
- **Evidence:**
  - `MatchSpec` (`src/simulation/match.py:80-98`) already carries the teams
    (`n`, `loadout`, `env_profile`, `controller`), `seed`, `zombies`,
    `time_limit_s`, `objective` and `zombie_controller`.
  - CLAUDE.md Match config row: "The ONLY way a match is parameterised".
  - The design names no match spec. It derives `truncated` from "the time
    limit", and puts obs and reward rows in the game's config.toml. A reward
    variant therefore costs a commit, a tag and a rebuild. Erik wants to
    "compare ... different rewards" (`rl-ready-lead` §2).
- **Fix:**
  - The bank carries each template's `MatchSpec`, from `LevelData.match`.
  - `[rl.done]` reads MatchRules and restates neither the objective nor the limit.
  - A policy-driven team is a controller name in match.toml (M6).
  - Obs and reward rows live in a run file handed to `BatchedEnv`, refused by
    a door, hashed into the run's record, and kept outside the game snapshot.

### M6. The farm worker is a second match loop, and the policy is not a controller

- **Where:** §4.1 l.178-182 ("run_match's loop without the report", "driven
  over pipes"), §4.5 l.276.
- **Evidence:**
  - `tools/run_match.py:81-138` is one `while not terminal` function.
  - CLAUDE.md rows: "never a second loop" (headless runner); "In a match
    every unit has ONE driver and it is a controller" (registry).
  - `src/control_source.py:6,19,263-265` already reserves `AgentPolicy` as
    the ML track's control element.
  - `rl_ready_scope` §3.1: "every controller is a ControlSource ... and later
    any policy".
- **Fix:**
  - Factor run_match's setup and one tick into a steppable session: the
    constructor, `spawn_match`, `create_match_controllers`, the recorder hook,
    `tick()`. run_match loops it; the worker steps it.
  - Register a `policy` controller that issues the action rows through the
    intent facade, so the tap records them.
  - Reset builds a new session through the same constructor.

### M7. Episodes run on the device backend cannot be recorded, so a trained agent cannot be watched

- **Where:** §4.5 l.276-278 (replayable only through the CPU backend's tap),
  §11 l.470-471.
- **Evidence:**
  - `rl-ready-lead` §3, 2026-10-29: replay "is the only way to judge an agent".
  - CLAUDE.md replay row: one tap, format v1, and `main.py --replay` is the
    only playback path.
  - The device backend has no tap.
- **Fix:**
  - The device backend exports an episode as a `match_replay.Recording`
    (format v1): the action rows per tick become INTENT_KINDS calls, with
    checkpoints from the device's digests (B2) every
    `checkpoint_every_ticks`.
  - Gate: a device episode replays under `run_match --replay` through every
    checkpoint. That is also backend equality on real training data.

### M8. The action buffer is not INTENT_KINDS: it drops the throw fuse and the hold / latch semantics

- **Where:** §4.5 l.272-276.
- **Evidence:**
  - `src/match_replay.py:81-89`: `throw_grenade_intent` is `(int, int, int,
    fhex, int?)`, so `fuse_seconds` is a float.
  - `simulation.py:959-977` latches `ThrowIntent(dx, dy, float fuse, reach)`.
  - `combat.py:129`: `thrown_tick + int(self.fuse_seconds * tps)`.
  - `clear_move_dir` (None) is not a zero vector.
  - Continuous verbs overwrite and edge verbs latch until consumed
    (`simulation.py:900-907`).

  The design's int32 columns omit the fuse, and its verbs live in two tables
  (an "intent schema" plus INTENT_KINDS).
- **Fix:**
  - Derive the buffer's columns from INTENT_KINDS: one table, one door.
  - Add a §7 reference-side change for the fuse (integer ticks or Q16 at the
    facade).
  - State which ticks of a `decimation` step apply the latch verbs.

### M9. Q16 positions: the stated reason does not force the change, and its consumers are not inventoried

- **Where:** §7 l.406, §12.1.
- **Evidence:**
  - `simulation.py:2030-2035`: the only floats in movement are an exact
    power-of-two dequantize and plain IEEE `+ - * /` on `u.x/u.y`,
    "bit-reproducible cross-machine". Ingress-lint door 3 allows exactly that,
    and the CUDA build runs with `--fmad=false` (§8). So float64 columns would
    reproduce the reference on the device. The real case for Q16 is the iron
    rule ("synced sim state is Q16.16 integer only"), which the design does
    not cite.
  - The consumers are not listed: `renderer/` (game_renderer, follow_camera,
    frame_lights, lighting, lit3d), the bots, the 1e-9 quantisation in
    `_unit_record`, and the recorder's frozen `.npz` contract.
    `recorder.py:207` stores `u.x`, and l.265-280 casts it to int32 as
    `unit_fx`/`unit_fy`. Q16 ints there change the meaning of a frozen key,
    yet §13(a) calls the recorder "unchanged".
- **Fix:**
  - Put both options (float64 under door 3, or Q16 under the iron rule) in
    §12.1 for Erik.
  - If Q16 stays: inventory every consumer, route the change through
    `unit_fixed.py`, and treat the recorder key as a contract change.
  - Give it its own patch (M13).

### M10. The level template bank re-invents the bank format levelgen already fixed, and re-derives the level-load initialiser

- **Where:** §4.6 l.293-305, §6 l.392-395.
- **Evidence:**
  - `docs/levelgen_archetype_recipe_2026-10-31.md` §5.3, l.308-320: a bank is
    `levels/banks/<name>.toml` (header: name, generator_version, held_out;
    rows: LevelSpec plus output_digest), with banks named train, test-iid,
    test-ood and probe. "The format is fixed here". l.415: "the banks
    themselves: the RL track".
  - CLAUDE.md LevelSpec row: Banks "NOT BUILT"; `generate_from_spec` refuses a
    stale `recipe_hash`.
  - The Level data layer and Gas energy field rows: `refresh_gas_energy` is
    THE level-load initialiser; the loader seeds through
    `seed_gas_temperature` plus `reseed_gas_energy`.
  - The design instead defines its own file and recomputes the spawn cells.
    Its reset "re-derives ... gas energy ... the mirrors", which makes a
    second initialiser.
- **Fix:** the device bank is a COMPILED §5.3 bank.
  - Each row goes through `generate_from_spec`, `level_loader.load`,
    `Simulation(seed)` and `spawn_match(LevelData.match)`. Snapshot the
    post-load planes, books and unit rows from the result.
  - Reset is copy plus the env's Philox key (after checking that the initial
    state does not depend on the episode seed).
  - Provenance is the §5.3 header plus the compiler's build id.
  - Held-out sizes become separate device banks.

### M11. The rare-event host path has no write inventory and no tick position, and §13(b) is missing it

- **Where:** §4.8 l.318-327, §4.3 l.237, §10 ("a host handler writing
  through the edit queue").
- **Evidence:** `execute_payload` (`payloads.py:365-497`) writes:
  - `apply_explosion`: immediate walls plus `destroy_wall`;
  - a `blast_log` append (l.424-430);
  - `apply_blast_damage(units, ...)` (l.432): packets on units that are
    absent on the device backend;
  - `read_blast_air` and `deposit_over_tiles`;
  - TickCredit claims (l.75-86);
  - an `ExplosionEvent`;
  - for strength rows, `div_source`.

  Detonations happen in slot 2, at 4c and in 9e. Yet §4.2 l.217 says "No
  host sync inside a tick", and §4.10 lets the host read only `terminated` /
  `truncated`. The design names only "deposits and wall damage as edits".
- **Fix:**
  - Inventory every write with its device channel:
    - edits go to the queue;
    - packets go to a device packet queue applied in walk order, with credit;
    - claims go to a device TickCredit plane;
    - `blast_log` goes to device rows;
    - `div_source` goes to its plane;
    - topology goes per B1.
  - State the slot, and how the host learns of the event. Either allow the
    host sync and cost it, or split the tick there.
  - Add a §13(b) rule: "a rare event runs `execute_payload` through these
    channels, never a second payload path".

### M12. The unit schema does not collapse a unit's definitions to one

- **Where:** §4.4 l.252-267.
- **Evidence:** a unit's synced state is defined today in four places:
  - `Unit.__init__` (`unit.py:133-330`);
  - `_unit_record` (tests);
  - `status.serialize_statuses` (`status.py:335`);
  - the recorder's snapshot (`recorder.py:207`).

  The design's column list misses state the match path uses: `has_grenade` /
  `has_explosive`, `killed_by_zombie`, `awakened`, the intent latches
  (`live_move_dir`, `live_aim`, `live_trigger`, `pending_throw`,
  `pending_use`, `_possessed_ids`) and `recent_attackers`. Unless the other
  definitions read the schema, it becomes a fifth.
- **Fix:**
  - `_unit_record`, `serialize_statuses` and the device pack / unpack read the
    schema.
  - A property test: every instance attribute of a spawned `Unit` is a schema
    row or on an explicit non-synced list. A new attribute without a row fails.

### M13. P3 bundles four systems and two HUMAN-TESTs

- **Where:** l.437.
- **Evidence:**
  - The `autonomous-patch-workflow` skill says one system at a time.
  - A held HUMAN-TEST blocks the mechanical commits queued behind it.
  - Four re-baselines in one branch make each re-pin's "sole cause"
    unprovable.
- **Fix:** split P3 into four patches, each its own branch and re-baseline:

  | patch | system | tier | gate |
  |---|---|---|---|
  | P3a | schema (B2, M12) | Sonnet | property gate |
  | P3b | Philox | Sonnet | KAT, layout, re-pins |
  | P3c | positions (M9) | Opus | HUMAN-TEST |
  | P3d | distance field | Opus | HUMAN-TEST; spec in the minors |

### M14. P1 bundles five systems at Sonnet, though it fixes a lasting API

- **Where:** l.435.
- **Evidence:** the oracle ("N workers reproduce run_match's digests") gates
  only the farm. The step / reset contract, the table schemas and the
  snapshot refusal have no oracle and fix lasting names. The skill says: no
  oracle → Opus.
- **Fix:**
  - P1a: BatchedEnv, the farm (M6), the smoke driver, the farm measured. Opus.
  - P1b: snapshot, build id (M19), twin-debt test. Sonnet.
  - The `[rl.*]` tables wait for M5.

### M15. P4 bundles about seven systems, and its gate cannot hold before P7

- **Where:** l.438.
- **Evidence:**
  - Gate "bus counter reads 0". But slots 2-5 and 9-9e stay on host Python
    until P6 / P7 (§4.3; `residency_where_we_stand` §2.1), and they read
    planes every tick: 9b reads atmosphere, 9c / 9d temperature, 9c2 wind,
    9c3 gas, blast damage `wave_p`.
  - The device edit queue is a new canonical system: the gas_energy dual
    write vetoed by the accountable set, the conservative-slice refusal, trace
    saturation, flush-time noise. A tol-0 gate covers only the policies its
    scenes exercise.
- **Fix:**
  - P4a: the resident drivers (combustion, temperature, fire, sweep, water
    tail). Sonnet, tol 0.
  - P4b: the stamp kernel. Sonnet.
  - P4c: the edit queue. Opus, one gate per policy row.
  - The bus gate becomes "zero plane transfers inside slot 7".

### M16. P6 to P8 cannot compare equal games mid-port, and P6b, P7 and P8 are mis-sized

- **Where:** l.440-443, §5.3 l.369-373.
- **Evidence:**
  - The corpus is bot matches with shooting and zombies. Combat and the horde
    only arrive at P7, so P6 and P6b's "replay lockstep" needs a bridge that
    is never specified. R1 "is a state the build passes through" (l.153), but
    its form is undefined.
  - P6b (Sonnet) puts doors on the device (l.231) against l.238.
  - ship_demo holds one duct and two vents (`levels/ship_demo/level.toml`
    l.407-430): no heat source, charge, airlock, sensor or bus. The oracle
    under the Sonnet tier has holes.
  - P7 bundles shooting, sprays, projectiles and grenades, the rare-event path
    and the horde.
  - P8 bundles the bank, reset, obs, reward / done and export.
- **Fix:**
  - Name the passage bridge as a system: a schema-driven Unit ↔ store pack /
    unpack per tick for the slots still on the host. It is also P3a's
    round-trip gate.
  - P6b: per-class scenes (`levels/airlock_demo`, a heat-source plus
    heat-trigger studio), on Opus.
  - Split P7 in four: shooting and sprays; projectiles and grenades; the
    rare-event path; the horde.
  - Split P8 in three: bank and reset; obs, reward and done; export.

### M17. "Bots stay on the host" constrains the agent era but is filed as an accepted gap

- **Where:** §11 l.468-469.
- **Evidence:** `rl-ready-lead` §3, 2026-10-29: the rule-based team AI is
  "the first opponent and the baseline". The design makes the device's
  opponents the horde and self-play.
- **Fix:** move it to §12 as a ruling with options:
  - bots at farm speed only;
  - a host-bot hybrid batch;
  - a later device bot.

### M18. The container env is a second version set

- **Where:** §8 l.417 ("the conda env from `environment.yml`").
- **Evidence:**
  - No `environment.yml` exists in the repo.
  - Master CLAUDE.md and the `ml-stack` skill: ONE version set
    (`ml_stack.txt`) for every machine, never installed by hand, needed
    "before a training batch that needs matching machines".
- **Fix:** the container installs from the ml-stack set through `ml_stack.py`.

### M19. A second git-describe helper

- **Where:** §6 l.390, §13(b).11.
- **Evidence:** CLAUDE.md LevelSpec row: `match_replay.build_id()` is "the one
  git-describe helper". It is used at `tools/gen_ship.py:2777-2780` and
  `tools/run_match.py:126`.
- **Fix:**
  - `bp.build_id()` names the compiled module's provenance only.
    `require_snapshot` checks HEAD and the tree through `match_replay`'s
    helper (or `git describe --exact-match --tags`), plus `bp.build_id()` for
    the `.pyd`.
  - The bank's provenance comes from LevelSpec (M10), not from a third id.

---

## MINORS

### §13(b), the draft rules

- §13(b).13: `tick_stage_profile.py` already exists (`536cfee`) and is also
  listed in §13(a). List it once; its rule should call it the headless sibling
  of `frame_profile.py`, as its own header does.
- §13(b).7, "THE device pathing": after P3 the field is the zombies' pathing
  on BOTH paths. Say so.
- §13(b).12 is narrative. Give it a constraint: the only Linux build entry,
  with digests equal to Windows's.
- §13(b).8 should point at a gate: a scan that no match-path module reads
  `sim.rng`, plus the list of draw sites.
- §13(b) lacks the device tick conductor: where the device slot sequence
  lives, checked against `Simulation.step`'s numbered slots. That check is the
  natural key for `test_twin_debt`.

### Philox

Neither `cpp/src/philox32.h`, nor `src/simulation/philox32.py`, nor a
"chapter 17" exists on main (`engine/` stops at 16). Vendor Philox by
cherry-picking the exact `63-swarm-units` commits, so #63 merges clean.

### §4, architecture

- §4.3 omits slots 1, 1b (where the action buffer lands), 9 and 10.
- §4.5: "material / solid" duplicates the accessor's `solid` channel.
- Store rows should be ordered (team, slot), per `rl_ready` §3.2 and
  `Simulation.roster`, so the policy's slot axis is stable and team blocks
  are contiguous.
- The reward counter axis E should be derived from `events.Cause`, so a new
  cause is a new column rather than a second enumeration.
- In the v1 match the time limit is a DRAW result. Classing it as
  `truncated` is an agent-era call; mark that `[rl.done]` row PLACEHOLDER.
- §4.9, the spec the HUMAN-TEST needs, in writing:
  - target choice: path-nearest, versus today's Euclidean nearest
    (`ai_zombie.py:115-126`);
  - activation and melee: still Euclidean plus `has_los`? (`ai_zombie.py:70-98, 135`);
  - refresh cadence, versus today's repath every 5 steps (engine/10 §3);
  - footprint passability (`ai_zombie.py:177-180`);
  - the law's `math.sqrt` floats (`ai_zombie.py:77, 93, 123`), which sit in
    the twin's path.

### §9, the patch plan

- No patch names its effort level. The skill asks for model tier plus effort.
- P2 contains a human step (Erik runs Nsight) but carries no HUMAN-TEST
  marker. Mark it.
- P9's stop gate compares against a farm measured at P1, before P3's
  re-baselines. Re-measure the farm at P9 on the same tag, so the comparison
  is at equal game.

### §11 and §6

- §11 lacks the `ACCEPTED GAP:` labels and misses two gaps: light in the
  observation (M4) and device episodes that cannot be recorded (until M7).
- §6: `BREACH_BUILD_ID` set at configure time goes stale on a rebuild without
  reconfigure. Plain `git describe` misses a lightweight `train/*` tag.

---

## QUESTIONS (not grounded)

- Does any part of the post-load state depend on the episode seed? If none
  does, M10's reset is a pure copy.
- Where would the freighter recordings of the corpus live? Only `ship_demo`
  of the ship family is in `levels/`.
- Does the device backend keep a host `GameMap` at all? §4.2 and §4.3 do not
  say, and B1 and M11 depend on the answer.
- Is #76 closed? `rl_ready_scope` §4 put batched envs and the device store
  after #76; the line between this arc and the agent-design session decides
  M3 and M5.
