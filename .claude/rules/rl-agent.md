---
paths:
  - "src/rl_env.py"
  - "src/simulation/obs_spec_v1.py"
  - "src/simulation/action_spec_v1.py"
  - "src/simulation/reward_v1.py"
  - "src/simulation/sensor_accessor.py"
  - "tests/_obs_reference.py"
  - "tests/_1h_forced.py"
  - "tools/rl_smoke.py"
  - "tools/obs_body_imprint.py"
  - "src/simulation/device_obs.py"
  - "cpp/src/cuda_obs.*"
  - "cpp/src/cuda_obs_float.cuh"
  - "cpp/src/cuda_obs_sight.*"
  - "cpp/src/cuda_obs_picture.*"
  - "cpp/src/cuda_obs_list.cu"
  - "cpp/src/cuda_obs_rows.cuh"
  - "src/simulation/device_picture.py"
  - "cpp/src/obs_layout.h"
  - "tools/gen_obs_layout.py"
  - "tests/cuda_obs_check.py"
  - "tests/obs_column_owners.toml"
---

# RL agent: the env, the observation, the action, the reward, snapshots

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say. The
`game-ai-lead` skill is the role's continuity.

## Batched environment
- One API, two backends (`cpu_farm` built; `device` refused until the residency arc delivers it). A trainer never touches a Simulation or a kernel.
- One run file (`run.toml`, OUTSIDE the repo, door-checked by name, its sha256 the manifest's); nothing `[rl.*]` in config.toml.
- Episode k of env e is `episode_draw(run_seed, e, k)`, one Philox block (`Salt.EPISODE`): pure, never a function of N or of other envs' resets. Reset by MASK; rows `(team, slot)`; observations only through the accessor; rewards from the credited events.
- Workers import only the CPU build, nothing from `renderer/`.
- The observation arrays a step returns are the ENV's buffers, valid until the next step or reset: a caller that keeps one copies it, and a new per-step output of that size joins `BatchedEnv._buf`, never a fresh `.copy()` (2026-11-11: a fresh 10 MB array cost the driver 2 to 6 ms a step).
- Farm throughput is measured with `tools/rl_smoke.py` by the design's §9.1 protocol (`docs/residency_benchmark_2026-11-07.md`), at 4 envs per worker.
- `BatchedEnv.team_rows(obs, t)` is the ONE path a trainer feeds a policy: views of team t's rows, nothing of `info`, `reward` or `counters`. An in-process `EnvRunner` on `"v1"` actions must observe after every reset and step. `rl_env.observe_v1` and `rl_env.count_events` are module functions the door bots call: never copies.

## Observation spec
- An observation layout is a named, versioned spec served by the sensor accessor's `observe`; `obs_spec_v1.py`'s tables ARE the spec. A new idea is a new spec, never an edit of one a trained network uses.
- Every feature is a ratio of a table value, an angle's cosine and sine, a flag or a kind, squashed by the one curve: never a game number (`test_no_raw_number` scans the tables).
- Units, grenades and charges reach a soldier ONLY through its team's picture and a teammate's line of sight; the map, doors and fields are unfogged.
- THE WALL: the builders take `(know, u)` and reach the sim ONLY through `TeamKnowledge.build` (with `observe_rows`, the door that hands it the sim) and its `ObsWorld` facade (explicit attributes, no fallback); every cache is keyed by its inputs. An object's id is its team's own counter; a uid is an EQUALITY key only.
- Nothing walks its own lines or floods its own field (sight through Sight for many tiles, the route through the horde's flood). The physics planes are read ONCE per env per decision; the sight plane is cast from RECORDED centres, never live positions.
- ACCEPTED GAP (Erik, 2026-11-12: "on the wind question, its OK! We accept it!"): the physics planes carry the bodies' own aerodynamic and radiative imprint (`stamp_units`), measured by `tools/obs_body_imprint.py`. So the counterfactual family compares every array but the physics planes and the `air` / `felt_temp` columns.
- A length in the observation is `obs_spec_v1.length` (float64 `sqrt(dx*dx + dy*dy)`, its array form doing the same operations) and the decoder's lob reach is `math.sqrt`: never `hypot` or a float `**` (libm algorithms a kernel cannot reproduce bit for bit, so the device observation and decoder could not be held EXACT; #29 P8.0b, `tests/test_portable_host_roots.py`).
- Held by the reconstruction against `tests/_obs_reference.py` and the counterfactual family. Choices beyond the spec are marked `CHOICE` in the module.

## Full view
- What the training-only VALUE network is shown is ONE named spec, `full_v1`, served by the same door into the farm's own buffers under `info["full_view"]`: NEVER a key of `obs` or `info["final_obs"]`; a policy never reads it.
- The OTHER SIDE OF THE WALL: its builders read the sim through ONE `FullView` per env per decision (`FullView.build`, the one door) and keep THEIR OWN caches, never `TeamKnowledge`'s; pair case (f) holds every `obs` array byte-identical with the full view off, built before or after.
- Over its entry cap the farthest doors, then zombies, then objects, are dropped, never a marine. Its balance column is `reward_v1.standing`, the reward's one implementation.
- A new privileged input is a new version of this spec.

## Device observation
- THE device form of observation spec v1, its masks and the full view (`src/simulation/device_obs.py`, `cpp/src/cuda_obs.*`): built on the card in the device conductor's walk (its last call line) into env-owned CuPy buffers with exactly the served keys, shapes and dtypes (and a `final_` set), handed to torch by DLPack. The host builders stay THE reference; never a second observation, never a host copy between the card and the trainer.
- Every column and plane has ONE owning P8 patch in `tests/obs_column_owners.toml`; a block lands with its owner rows and its `device_obs.BUILT` entry in the same patch. Each gate compares what its patch and the earlier ones own, as RAW BITS, at every decision tick, and fails on an owned column constant over its corpus. ONE check script, `tests/cuda_obs_check.py`, a part per patch: never a second script.
- A spec constant reaches a kernel only through the generated `cpp/src/obs_layout.h` (`tools/gen_obs_layout.py`), read from the spec's own tables: never a literal in a kernel, never a kernel argument. The observation's device code is its own TU and headers; it adds to `cpp/src/cuda_unit_lib.cuh` only a read a unit kernel shares (the use verb's reach); it writes no synced buffer and adds no host sync to the tick.
- THE OBS TABLE (`device_obs.table_values` / `team_values`): every value a block reads that is not unit state is ONE row of it, read from the reference's OWN object (the episode's `ObsContext`, the sim's weapon rows, the loadout and env-profile tables, the species registry), its offsets only from the generated header. A float the host reads as a float is carried as its own float64 bit pattern, never a Q16 row of the device tables (inventory K4). A config row that moves a block's width (a loadout, weapon, env profile, species or hand-grenade kind) means `tools/gen_obs_layout.py --write`, commit, rebuild.
- The observation's float arithmetic is `cpp/src/cuda_obs_float.cuh` alone: float64 in the reference's operation order through the never-fused intrinsics, ONE cast per value into the served dtype (float16 only through the kit's hand conversion: never `__double2half`, never through float32). `log` / `log1p` are CUDA's own: a served value they move fails the gate, and is the trigger for the shared log module (the lead's call), never a tolerance.
- The integer reads a block shares with the unit kernels (`mag_status`, `composed_flags`, `handling_ticks`, `scaled`, `wrap_q16`, the hand-verb gates, `reach_key`) are the unit library's, called with the device tables: never a copy in the observation's TU.
- THE AIM ANSWERS (`DeviceObservation.aims`: the fire lanes and each owner's own lane, `aim_of` per team view, the line of fire and `ticks_to_within` per (row, seen unit)) are computed ONCE per decision into their buffers; the list, the lanes plane and every later block read those buffers, never recompute them. Python's flooring `//` on a value that can be negative is `obs_sight::floordiv` (`lane_div` in the lane walk), never C's truncating `/`.
- THE LIST (`DeviceObservation.build_list`, `cpp/src/cuda_obs_list.cu`): each block is taken in (squared centre distance, id) order by successive minima up to its capacity (the host's sort, then `[:cap]`); the order is total only because the id is unique within its block (unit id, object uid, door ordinal), so a new block or key brings such an id or the device order is undefined. The row predicates (`env_due`, `is_mate`, `is_seen`) are `cpp/src/cuda_obs_rows.cuh`'s, the ONE copy every observation TU reads.
- THE MASKS (`DeviceObservation.build_masks`, after the list of the same tick): each hand bit is ONE unit-library gate called at the decision tick's start (`can_melee` at the tick's start); the use bit and the list's `in_reach` are `unit_lib::reach_key` over the live door rows, never a restated ring. A new hand verb brings its device gate to the library and its bit to `k_masks` in the same patch as its `verb_gates` gate; a row that is not valid reads `default_masks`.
- THE WINDOWS (`DeviceObservation.build_windows`, `cpp/src/cuda_obs_windows.cu`, after the list of the same tick): the physics planes read the RESIDENT planes in their own dtype (`device_obs.WINDOW_PLANES`, checked at the build) and dequantize with the host's `*_fixed` currency (generated constants), never a mirror or a host upload; a new plane the observation reads becomes resident, or is derived on the card from resident planes exactly as the host derives it, in the same patch. The far reduce is the block MAX and the wind of the block's FIRST largest load (numpy's argmax, tiles in row-major order); a reduction that adds floats over a block must add in numpy's own order for that expression, never a parallel tree. The felt temperature is the unit library's `felt_temperature_q` over the device tables, whose heat dials the build holds equal to the episode's `ObsContext` (refused by name otherwise). `sees_target` reads the decoder's aimed uid per row (`DeviceObservation.aimed`, -1: none) through the list just built: P8.6a's decoder writes that buffer, never a second aim path. The full view's `full_watched` (P8.8) calls `obs_sight::window_cell` over its own sources as `k_sight_units` does, never a second window walk.
- An owned column the corpus cannot move gets a case in the check's hand match (corpus (d): a hand state at tick 0 given to the reference and the twin alike, hand intents through the tap), never a weakened census.
- DEVICE VISION (`DeviceObservation.vision`: `sees` (N, U, U) per observer / target row, `visible` (N, T + 1, U) per team slot, the zombie team the last) is THE device form of `vision.state_for`: computed EVERY tick at the walk's `11'` line BEFORE the blocks (`stage`), where the reference's slot 11 first asks (after the advance and the conversion); every device block that asks what a team or a unit sees reads these tables, never its own pair loop. The sight rule and its dials are obs-table rows (`device_obs.sight_values`), never a kernel argument; a team value without a slot is refused at the build.
- THE COVER TABLE (`device_obs.cover_values`) is staged ONCE at the build: on the continuous path nothing changes a cover's `alive` (`CoverRuntime.chew` is reached only through a round marched with `cover`, which only OnePhaseWEGO passes). A path that starts chewing cover on the continuous ruleset moves the table onto the card in the same patch. Every line of fire (`line_of_fire.solid_in_way`) on the card is `obs_sight::solid_in_way` over it.
- THE DEVICE TEAM PICTURE (`DeviceObservation.picture`, `src/simulation/device_picture.py`, `cpp/src/cuda_obs_picture.{cuh,cu}`) is updated at the `11'` line between device vision and the blocks, ONCE per tick (a skipped or doubled update is a fault), in `TeamPicture.update`'s order: the per-unit trackers and last hit, then per team the damage window against the records BEFORE the update, the seen deaths, the memory, the sightings, the object tracker. A change to that order or to any tracker lands in the device picture in the same patch, gated every tick by the check's picture comparison. Every later block (the list, the masks' aims, the reward's bars) reads the picture through `cuda_obs_picture.cuh`'s layout and readers, never its own offsets; the damage window's sum is `damage_sum`, in its deque's order.
- Its capacities are refusals, never clamps: records are one per (team, store row), the tracker holds live objects only (both bounded by the law), the damage window is a FIFO of `DMG_CAP` per team; every fault rides the conductor's ONE books read (`picture_faults`), raised by name with the env.
- THE OBJECT SERIAL is a side array of the picture's block that the grenade / charge tables' stable compaction carries (`ObjectTableBuffer.set_side`); never a table column (a column reaches the digest) and never the row index. Its host notion is test-side only: an object's position in `projectiles + planted_charges` (`device_picture.host_form`).
- A per-env reset restages env e's picture through `DevicePicture.restage` (from a host picture or its normal form), never a second conversion.
- An env is built on ITS decision ticks (its episode start and the run's decimation, read on the card) and skipped while its ended latch is set.
- The hand-over's ordering rests on the BLOCKING tick stream and CuPy / torch on the legacy default stream: a change to either is a change to the hand-over, held by the check's handover part.

## Action spec
- Masks come from the game's own gates, never a copy of them. Every number comes from its row through `action_tables`, quantized once: never a literal. A new grenade row APPENDS a hand entry; a new verb is a hand-table row with its gate in the same patch.
- A pointer becomes a game id at the decision (against the `list_uid` the row was chosen on) and is never kept as a position; the aim at a unit follows the team picture's record of it, through the one control-seam quantizer (`aim_vector`).
- The decoder is a per-tick step of the `policy` controller issuing raw rows through `issue_row`, so `VERB_RULES` stays THE hold / latch rule and nothing new reaches the sim or the recording. A dead or turned row issues nothing, ever.
- Masks are built at the START of the decision tick, ONE gate call per hand bit (`build_masks`). A head left out of learning has ONE legal option; no head is ever all-masked. A masked in-range index decodes to its fallback and is COUNTED; an out-of-range index is refused. A masked choice issued by force does nothing (`tests/_1h_forced.py`).
- The flashlight hand entry is RESERVED (always masked, decoded as nothing). The inverse helpers are the nearest-choice translation: never a second table.

## Reward profile
- The env returns ONE number per group for every `in_match` row, dead and turned rows included, and NEVER their sum (the trainer weights the groups and blends own against team). A new term is a new counter in a group, its sign fixed in `_COUNTER_TERMS`.
- Health is counted in BARS (`TeamPicture.bars`), clipped at one. A team's balance is ONE function (`standing` / `balance`), also the full view's column.
- A change not credited to a unit is a team's UNCREDITED REST, built from its sources, never computed as a remainder; a tick whose events do not reproduce a unit's life, or a LIVING unit's hp, raises `RewardBooksMismatch` (a game finding). A corpse's hp is its body (the Corpse law), lowered by the blast and the heat row with no event: outside the books.
- Eliminations, the win and the clock come from `MatchRules` (never a second rule of who won). Under a profile the clock is an ENDING (`terminated`, never `truncated`). Read-side only: nothing synced, no digest moves.

## Training snapshot
- A training run's code is an ANNOTATED tag at HEAD in its own detached checkout (the git-workflow training-checkout row). `require_snapshot` refuses, naming the check, unless the tag is annotated, HEAD is its commit, the tree is clean including untracked files under `src/`, `levels/`, `config.toml`, `cpp/`, `agents/`, `tools/` (`SNAPSHOT_PATHS`: a stray module there would be imported by the trainer, a stray tool would write the rooms; the main checkout's untracked tools are why a kept run starts from its own checkout), both modules come from this checkout, `bp.build_id()` == HEAD, and the run file's sha256 is recorded. `device` also needs an empty twin ledger and is refused until P8b.
- With `[run] out_dir` (refused inside the checkout) `BatchedEnv` writes `run_manifest.json`; a resume refuses any change but the host, by name.
