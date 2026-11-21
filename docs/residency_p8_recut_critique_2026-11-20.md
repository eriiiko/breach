# Critique of the P8 re-cut (2026-11-20)

Adversarial review of the paragraph "P8 RE-CUT (2026-11-20" in
`docs/residency_batching_design_2026-11-07.md` (§4.5, before §4.6), read against the code at
`a7f7b924` and the inventory `docs/residency_p8_observation_inventory_2026-11-20.md`. Nothing built
or changed. One-liners ran with `C:/Users/steen/anaconda3/envs/data/python.exe` (numpy 2.5.3,
Python 3.12.14, CPU without AVX-512; `cpu_features`: AVX2 True, FMA3 True, AVX512F False); the two
timing runs used the main checkout's CPU build (`cpp/build/Release`, main `dde25e03`), since this
worktree has no build.

Severity: BLOCKER = the plan as written cannot reach its gate or targets the wrong thing; MAJOR = a
patch will fail, overrun or pass vacuously unless the plan changes; MINOR = a trap to name in the
brief.

---------------------------------------------------------------------------------------------

## Lens 1: decision 1 (floats exact by one shared function)

**1. MAJOR: the list of "four places that call libm" is incomplete. `onto_reach_q` calls `pow`, not
`sqrt`, and its result is SYNCED SIM STATE.**
Evidence: `src/simulation/action_spec_v1.py::onto_reach_q` (line 242) is
`quantize_scalar(float(dx * dx + dy * dy) ** 0.5)`. CPython's float `**` calls the C runtime's
`pow`, which is not correctly rounded. Measured: `float(k) ** 0.5 != math.sqrt(k)` for 8011 of the
integers k in [0, 4e6), the first at k = 343, 525, 550, 971, 1372 (14 below 7200, so inside throw
range). After `quantize_scalar`, 0 of the 4e6 differ. The inventory (F step 3, I table) calls this
"sqrt, IEEE, exact-class", and the plan copies that. The value goes into the throw latch, so it
reaches the digest. The same pattern sits in `control_bots.py:1296` (`_lob`), which is host-only.
Change: add it to decision 1's list. In P8.0b, change the host to `math.sqrt` (or `quantize_scalar`
of the exact integer square root). Show "no quantized reach moves" with an exhaustive sweep over
every d2 the largest batch map allows, so the change needs no re-pin. The device decoder (P8.6) then
uses IEEE `sqrt`.

**2. MAJOR: the decision is bit-reproducible as described, but it buys almost nothing that is ever
served. A counted check in the SERVED dtype is the better default for `log` / `log1p`. Keep only the
cheap half of decision 1 (`hypot` written as `sqrt`).**
Evidence:
- Every log result is used only as a value and then cast to float32 or float16. No flag, mask,
  argmax or sort reads a squashed value: the far argmax reads `load`, which is integer-valued, and
  the masks and sorts are integer. So a 1-ulp float64 difference never propagates. It shows only
  if it moves the cast. Measured: 5e6 random values in (1, 4] (the squash's output range), each
  moved by 1 ulp with `nextafter`. The float16 cast changed 0 times, the float32 cast 0 times. The
  odds per differing value are about 2^-29 for float32 and about 2^-42 for float16.
- On this machine numpy's `np.log` and `math.log` agree on 3e6 inputs, `np.log1p` and
  `math.log1p` on 1e6, and `math.hypot`, `np.hypot` and `sqrt(dx*dx+dy*dy)` on 1e6 integer pairs.
  The real cross-machine hazard is elsewhere. numpy dispatches float64 `log` to its own AVX-512
  (SVML-derived) loop on AVX-512 CPUs. On such a host, `squash` (`math.log`, line 351) and
  `squash_array` (`np.log`, line 363) can disagree with EACH OTHER inside one host observation.
  This does not happen on this CPU. It can on a Linux node.
- A hand-written log costs a reference move (P8.0b, a patch) and three implementations that must
  match op for op: Python scalar (`squash`, `d_c`), numpy vector (`squash_array`) and CUDA. It
  also slows the host builder, which is both the gate's reference and the production farm until
  P8.10.

Proposed form:
- (a) `hypot` becomes `sqrt(dx*dx + dy*dy)` on both sides (`_hit_from` line 1167, `d_c` callers
  1367 / 1398, the wind's `np.hypot` line 938). This is IEEE and correctly rounded. It equals
  CPython 3.12's correctly rounded `math.hypot` for integer inputs below 2^26 (measured: 0
  mismatches), so the host does not move on the `_hit_from` / `d_c` corpus.
- (b) `log` / `log1p`: the card uses CUDA's double `log` / `log1p`. Every gate compares in the
  served dtype, bit for bit, prints any flip and fails on a flip. The flip count is reported, so a
  failure is a finding, never a tolerance.
- (c) Write the shared module only if (b) ever goes red on a corpus, or if Erik wants
  cross-machine identity of the host reference for its own sake.

If decision 1 is kept anyway: state P8.0b's bound as the measured maximum ulp error of the chosen
implementation over a dense sweep, reported in the served dtype. "At most the last bit" is not
true for a fixed-sequence polynomial log, which is typically 1 to 2 ulp from true and up to about
3 ulp from libm. `d_c`'s `log1p(d) / log1p(L)` compounds two such errors, so P8.0b's gate as
worded would go red on its own design.

**3. MAJOR: numpy's summation order is NOT sequential for one of the two reward folds. The plan
lists it as an open item "checked in P8.7"; it is answered here.**
Evidence: `src/rl_env.py::EnvRunner._reward`.
- Line 900 (named profiles): `(counters[:, :, None] * G[None]).sum(axis=1)` reduces over a
  non-last axis. Measured equal to a sequential loop: 0 mismatches over 64 x 24 x 40 random
  float64 values.
- Lines 897-899 (`"counters"`): `(old * w[:k0]).sum(axis=1)` and `(c32[:, k0:] * w[k0:]).sum(axis=1)`
  sum a 2-D float32 array along its LAST, contiguous axis. That is numpy's unrolled pairwise sum
  (8 partial accumulators from 8 terms up). k0 = 7 + `len(CAUSE_NAMES)` = 7 + 11 = 18. Measured on
  float32 against a sequential loop: k = 7 gives 0 mismatches; k = 12, 16 and 20 give 5284, 10830
  and 8678 mismatches of 20000.

Change: in P8.7, do one of two things. Either write both host folds as an explicit loop in a named
order (a host change that moves the `"counters"` reward by float32 ulps; the named profile does not
move), or refuse the `"counters"` profile on the device backend by name. Do not reproduce numpy's
unrolled order on the card.

**4. MINOR: signed zero and NaN must be part of "value for value".**
Evidence: `_hit_from` returns `-dx / n` with `dx = float(hit[1])`, so a hit along y gives -0.0.
`raw["wind_x"] = j * dx` gives -0.0 when j = 0 and dx < 0 (`FieldPlanes.__init__` lines 942-943).
`np.array_equal` treats -0.0 as equal to +0.0, which is what the existing obs tests use
(`tests/test_obs_spec_v1.py`).
Change: the P8 gates compare the served arrays as raw bits (`view(np.uint16 / np.uint32)`), and
each gate counts NaNs (expected 0) separately.

**5. MINOR: double to half on the card.**
Evidence: numpy converts float64 to float16 directly. Going through float32 differs on 275 of 5e6
values in [-4, 4] (measured). Whether `__double2half` is a single correctly rounded step depends on
the architecture and toolkit: native `cvt.rn.f16.f64` on sm_90, a software path on the other four
of `CMAKE_CUDA_ARCHITECTURES` (75 to 89).
Change: one integer-bit-operation `double -> half` function in the observation module, held by an
exhaustive test against numpy over every float16 rounding boundary ±1 ulp of float64. It lands in
P8.0, because P8.0 already writes float16 (`far_f[0]`, exact k/16, so it does not exercise the
problem).

**6. MINOR: the CUDA flags are already right; only the brief needs to say where the new code goes.**
Evidence: `cpp/CMakeLists.txt` (lines 263-266) sets `--fmad=false -prec-div=true -prec-sqrt=true
-ftz=false` on the whole `breach_physics` target, so every new obs TU inherits "contraction off"
with no work. numpy cannot contract: each ufunc rounds on its own.
Change: put the obs double arithmetic in its own `.cu` / `.cuh` files, not in
`cuda_unit_lib.cuh`, so the unit kernels' integer library carries no double code.

---------------------------------------------------------------------------------------------

## Lens 2: the gate and its corpus

**7. BLOCKER: P8.2's "refusal of a level with live cover" refuses corpus (b) and the levels training
is for.**
Evidence: `tools/gen_training_rooms.py` places the game's `cover` entity (`cover_crate`, line 164;
three calls, for example two crates in the room at lines 233-234). Those rooms are the `all_round`
levels decision 2 calls "what training uses". `line_of_fire.solid_in_way` / `solid_in_way_many`
(lines 40, 57) apply EVERY live cover through `blocks_segment`. That is the obs list's col 25
(`lof_solid`) and the full view's lof. `vision.ray_clear` applies only `blocks_los` covers, and a
crate's default is False (`entities/cover.py` line 67). The device SIM is not wrong on these levels:
the continuous path passes `cover=None` to the march (`combat.py` around line 1740: "every shipped
caller"). Only the observation reads cover.
Change: replace the refusal with a per-env cover table (rectangles + `alive`, staged at episode
start) and a device `blocks_segment`. The slab test is IEEE (`1.0 / d`, products, compares:
`cover_system.py` lines 101-120), so it is exact-class; add it to the float list. If a crate's
`alive` can change on the continuous path, that change rides the table. Refuse by name only a cover
with `blocks_los = true`, which feeds `VisionState` (or port that too), and only once a level with
one exists. P8.2 or P8.4 owns this; it must be in a row.

**8. MAJOR: corpora (a) and (b) never exercise the order block, the decoder's columns or the
zombies. Gates that own those columns pass vacuously.**
Evidence: `order_intent` has exactly one caller in the bots, `control_bots.issue_team_order`
(line 1639). It is reached only from `BotCommander` and the door bots. The layered bots of the
pinned matches issue no order, so vector 84-95, list 47-51, `far_f[9]` (route) and `far_u[4]`
(place) are constant on (a). Training rooms have no zombies (`gen_training_rooms` docstring line 8),
so vector 82-83 and list col 33 are constant on (b). `aims` is None on any replayed match
(`observe_rows(..., aims=None)`), so vector 71-76 and list 28 / 66 are constant on (a), (b) and
(d). P8.1 ("vector EXACT on (a), (b)"), P8.4 and P8.5 would be green on columns that never left
their default.
Change: every gate prints, per column / plane it owns, how many decision-rows held a non-default
value, and fails when an owned column is constant over the patch's corpus. That is the design's own
§5.5 rule, "the pre-flight refuses a feature ... with zero exercise", applied to the gates. Corpus
(c) or a (d) scene is named as the owner of the order, route, place and decoder columns.

**9. MAJOR: corpus (c) cannot be a *recorded* episode. It has to be generated live.**
Evidence: a `match_replay.Recording` holds the tap's DECODED intents. Replaying it
(`MatchSession(..., replay=rec)`, the gate shape of `tests/cuda_conductor_check.py::replay_gate`)
puts the `replay` controller in place of `AgentPolicy`. So the replaying reference has no decisions
(N, U, 5), no `list_uid` they answered and no decoder slot state (`aimed_uid`, `prev`,
`throw_aim`). That is exactly what P8.6's gate needs and what the 1h columns read. A stored corpus
also breaks §5.3 ("Nothing stored: a stored corpus stales weekly").
Change: define (c) as an in-process `EnvRunner` (v1 actions, the commander on) inside the CUDA
subprocess, driven by seeded masked-random decisions (`tools/rl_smoke.py`'s drawer). The
reference's tap intents are latched into the conductor every tick, and the decisions are handed to
the device decoder in P8.6. Name the temporary run file the test writes (the run file is
door-checked and must live outside the repo).

**10. MINOR: corpus (a) inside a CUDA subprocess plays its own matches, every patch, and that is
the suite's cost.**
Evidence: the CUDA harness rule (never import the CUDA `.pyd` into pytest) puts the gate in a
subprocess, so the shared `_pinned_matches.watch` cannot serve it. `cuda_conductor_check.
part_pinned` records then replays each match (3 sims). Measured: ship_demo bots run on the CPU
reference at about 22 ms per tick (600 ticks in 13.2 s).
Change: one `tests/cuda_obs_check.py` with a part per patch. It records each pinned match ONCE per
subprocess and runs all of its parts on that one replay, not a new script per patch.

**11. Not a finding (checked): the observation CAN be built on a replayed bots match.**
`spawn_match` starts the picture (`Simulation.start_team_picture`), `ObsContext.build(sim, spec,
n_slots)` needs only the spec, and `observe_rows` takes `aims=None`. The commander (inventory F) is
not needed on (a) / (b), because orders arrive as recorded intents.

---------------------------------------------------------------------------------------------

## Lens 3: order, dependencies, size

**12. MAJOR: columns have no owning patch. P8.1's gate "vector EXACT" cannot be met when P8.1
lands.**
Evidence (inventory A.2): vector col 0 divides by the picture's `bars`, 5-6 read the picture's
`motion`, 66-70 its `last_hit`, and 80-83 its records (all P8.3). 71-76 are the decoder's (P8.6).
84-95 need the room graph and the route (P8.5). 58-65 and 96 are the physics law (P8.5). List col
28 / 66 are the decoder's, and 52-53 the lanes'.
Change: add a column to patch table to the plan, one row per column or plane of A.2 to A.5. Each
gate compares exactly the columns its table rows own, plus everything owned before. "Vector EXACT"
becomes true only in the last patch that owns a vector column.

**13. MAJOR: P8.4, P8.5, P8.9 and P8.10 are each bigger than one Opus run. Split them.**
- P8.4 holds five list blocks x 67 columns, the (d2, id) sort, `aim_of` (a serial
  nearest-clear-candidate loop), `lof_solid` / `lof_mate`, `ticks_to_within` (an iteration),
  `lanes` (a walk plus an atomic rasteriser), objects, doors with the `use_door` lift out of
  `cuda_unit_verbs.cu` (a Class T kernel with its own two-door guard), and the four masks. Split:
  - P8.4a: lanes + `aim_of` + lof. These sight-derived per-unit answers are also needed by
    `near_u[4]` (P8.5) and the full view (P8.8). Give them their own exact gate.
  - P8.4b: the list blocks and entry columns.
  - P8.4c: the masks and the `use_door` geometry lift. `tests/cuda_unit_verbs_check.py`'s
    two-door scene re-runs.
- P8.5: split the route (`opening_states`, seeds, `horde.flood` on the card, `candidates` /
  `field_at`, block-min, plus the vector 84-95 and list 47-51 order columns) into P8.5b, with
  corpus (c) / (d). P8.5a keeps the physics law, air, felt_temp and the two sight planes.
- P8.9: G's table has no per-env form for the store slice, the device tables, the tick row, every
  object table (`DeviceTable.stage_lists` is all-envs) and the horde cache. Split it into P8.9a
  (sim-side per-env restage, gated by §5.2 batch invariance with one env resetting) and P8.9b (the
  obs-side tick-0 state: the picture's first update, decoder slots, books, obs-table rows).
- P8.10: split the env backend (P8.10a, the one API returning CuPy) from `agents/` on device
  tensors (P8.10b, a different region, `.claude/rules/agents.md`, with its own tests).

**14. MAJOR: the terminal observation is in no patch, and it depends on every block.**
Evidence: inventory A.0 ("every key has a `final_` twin") and E.4 (the terminal observation is
built AT the ending tick, while the conductor ticks the ended env on). `rl_env._worker_main`'s
"step" builds `final_obs` / `final_full_view` before the reset. None of the twelve rows mentions
`final_`, the per-env "ended at tick t" latch, or freezing the books and picture of an ended env.
Change: P8.7 takes the ended latch and the frozen books. A new row after P8.8 builds `final_obs` /
`final_full_view` at the ending tick, double-buffered (inventory P8.0 had the buffers), gated on
(a) / (c) where a match ends inside a decision.

**15. MINOR: the order sentence says "P8.4 and P8.7 (both need only P8.3)".**
P8.4 also needs P8.2's pair kernel for `lof_solid` and `aim_of`, and P8.1's obs table for every
denominator. P8.7 needs the obs table's `n_t`. Both are satisfied by the linear order, but the
sentence licenses a wrong parallel start. P8.5 needs P8.4a's lanes (`near_u[4]`).
Change: write the real edges.

**16. MINOR: P8.0 is honestly small and proves the buffers, DLPack, the row map and the in-tick
build. It does not prove the decision cadence under ended envs.**
Change: P8.0's kernel skips an env whose ended latch is set (a stub latch, so P8.7 only fills it).
P8.0's test asserts the CuPy buffer's stream ordering with torch (the blocking tick stream,
inventory H).

---------------------------------------------------------------------------------------------

## Lens 4: decisions 3 and 4 at N = 256

**17. MAJOR: decision 3's "host commander on a 6-tick download" needs a host stand-in for a whole
Simulation that no patch builds, and after resets it costs a sync on every other tick, not every
sixth.**
Evidence: `BotCommander.handle_frame` (control_bots.py lines 1580-1594) and `TeamBrain.update`
(lines 366-430) read `sim.units` (Unit objects: alive, team, zombie, footprint, centres; and through
`fire_lanes(sim, ...)` the mates' trigger, facing and weapon), `sim.team_picture.records`,
`vision.state_for(sim)` (which recomputes `VisionState` on the host from `gmap` when not cached:
B.2's pair rays), `horde.flood(sim.gmap, bots.tile_ok(sim), ...)` (solid + doors),
`room_goal` / `enemy_room_goal` (room graph + units), `sim.tick` and `sim._seed`. On the device
backend no host Simulation is stepped.
Cost: `TeamBrain.update` measured at 0.24 ms mean (200 calls, ship_demo bots, 600 ticks). At
N = 256 with 2 policy teams that is about 512 x 0.24 / 6 ≈ 20 ms of host Python per tick, plus the
mirror's build and download. The plan's 2.6 ms per env per decision puts the batch tick near 166 ms
at N = 256, so this is about 12 %. It does not defeat the purpose by itself.
The phase is the real cost. The commander is due when `tick - updated_at >= 6`, counted from the
env's episode tick 0. Episodes restart at decision boundaries (multiples of decimation 4), so the
envs' phases spread over {0, 2, 4}. After the first resets some env is due on every even tick,
which is a third sync on half the ticks, not one every 6. The orders latch at the head of the tick
whose state they read, so the card waits on host Python each time.
Change: P8.6 names and builds the commander's view (a host object exposing exactly those reads,
filled from one download, with vision taken from P8.2's device `VisionState`, never recomputed).
It measures with staggered resets. It writes the alternative now for Erik: a commander period that
is a multiple of the decimation (4 or 8), which changes what a soldier is shown versus the farm
unless the farm changes too, or porting the team layer.

**18. MINOR: decision 4 (host build + restage) is affordable. Build the next episode ahead of time.**
Evidence: `MatchSession(level, None, seed)` on ship_demo measured 18-21 ms (warm level object; no
ObsContext, ActionTables or level load). Episodes of 300 to 1600 ticks at N = 256 give 0.16 to 0.85
resets per tick, so 3 to 18 ms per tick of host build: a few percent of the batch tick.
`episode_draw` is pure, so env e's next (seed, level) is known an episode ahead.
Change: build env e's next session while its current episode runs (a host thread, or between
ticks), so a reset only restages. Keep the bank as the measured fallback.

---------------------------------------------------------------------------------------------

## Lens 5: what the plan silently drops

**19. MAJOR: the device backend's doors are absent: coverage counters, the pre-flight,
`require_snapshot`'s device branch, the checkpoints, and the twin-manifest rows of the new device
builders.**
Evidence: old P8b (§9 line 3497) carried "the table upload, ... the pre-flight and the coverage
counters". §5.3 / §5.5 require the pre-flight and coverage. §6 says `require_snapshot(backend=
"device")` needs an empty ledger and a passed pre-flight, and the training doors "refuse a feature
without a twin". `.claude/rules/rl-agent.md` says `device` "is refused until P8b", and P8b no
longer exists. The re-cut's P8.10 row names none of these; the checkpoint download appears only
under "open by name".
Change:
- P8.10a lists `require_snapshot`'s device branch (ledger + pre-flight), the checkpoint download at
  the replay cadence (measured at N = 256), and the pre-flight of K random episodes across both
  backends with coverage counters.
- Every patch that lands a device builder adds its Class T row to `tests/twin_manifest.toml` with
  its generated file set (`obs_spec_v1`, `team_picture`, `reward_v1`, `control_policy`,
  `action_spec_v1`, `verb_gates`, ...).

**20. MINOR: stale "P8b" in rules and headers that the re-cut retires.**
`.claude/rules/device-units.md` lines 35, 62, 63, 66, 83; `.claude/rules/hand-verbs.md` line 38;
`.claude/rules/rl-agent.md` line 58; `cpp/src/cuda_unit_damage.cuh` line 8;
`cpp/src/cuda_unit_sight.cuh` line 18; `.claude/rules/sight.md` ("No device kernel").
Change: each P8.x patch that lands the thing amends the line naming it. The design paragraph lists
the mapping P8b → P8.x.

**21. MINOR: smaller drops.**
- `check_actions` / `masked_choices` on the card (inventory D, F step 1).
- `RewardBooksMismatch` as a counted per-env fault (E.1).
- The record capacity U (K12) as a refused overflow.
- The zombie-latch equivalence gate (K8; P8.6 implies it, so state it as a gate).
- The obs-side "per template" rows (G's last row) in P8.9.

---------------------------------------------------------------------------------------------

## Lens 6: rules

**22. MINOR: THE WALL needs a device gate of its own, not only equality.**
Evidence: the rule ("units reach a soldier ONLY through its team's picture") is held on the host by
the counterfactual family (`tests/test_obs_spec_v1.py`). On the card every kernel can read every
store row. Equality catches a leak only when the corpus moves an unseen unit between sightings.
Change: P8.4b ports one counterfactual case. Perturb an unseen enemy's store row (position, hp) on
the device and assert the soldier keys are unchanged, bit for bit.

**23. MINOR: Device tables rule.** The spec's constants (WIN, BLOCK, SQUASH_CAP, E, K, the uid
bases, column indices, the cone threshold `cos_threshold(55)`, `zombie_near_tiles`, lane dials)
must reach the card as generated header constants or obs-table rows, read from `obs_spec_v1`'s own
tables and `ObsContext`. Never a literal in a `.cu`, never a kernel argument.
Change: name the generator (in the pattern of `tools/gen_swarm_layout.py`) in P8.0 and add an index
line for it.

**24. MINOR: floor division.** `team_picture.fire_lanes_of` / `draw_lanes` (lines 393-394, 441:
`(tx - sx) * i // n`, `dx * i // n`) use Python's FLOORING `//` on negative operands. CUDA `/`
truncates (`unit_fixed.div_trunc` docstring).
Change: P8.4a's brief names it, and the lane kernel uses a floor-division helper from the kit.

**25. Not a collision (checked):** Class T is the right class for every obs, decoder, books and
reset kernel (§6 table). Decision 1's host change touches no synced state, except `onto_reach_q`
(finding 1), which is shown not to move any quantized value. "One driver per law per tick" holds
if the device picture stage is the conductor's slot 11 and the played game's host picture is never
also fed by it.

---------------------------------------------------------------------------------------------

Count: 1 BLOCKER (7), 10 MAJOR (1, 2, 3, 8, 9, 12, 13, 14, 17, 19), 12 MINOR (4, 5, 6, 10, 15, 16,
18, 20, 21, 22, 23, 24); 2 checked non-findings (11, 25).
