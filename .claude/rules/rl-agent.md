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
- Held by the reconstruction against `tests/_obs_reference.py` and the counterfactual family. Choices beyond the spec are marked `CHOICE` in the module.

## Full view
- What the training-only VALUE network is shown is ONE named spec, `full_v1`, served by the same door into the farm's own buffers under `info["full_view"]`: NEVER a key of `obs` or `info["final_obs"]`; a policy never reads it.
- The OTHER SIDE OF THE WALL: its builders read the sim through ONE `FullView` per env per decision (`FullView.build`, the one door) and keep THEIR OWN caches, never `TeamKnowledge`'s; pair case (f) holds every `obs` array byte-identical with the full view off, built before or after.
- Over its entry cap the farthest doors, then zombies, then objects, are dropped, never a marine. Its balance column is `reward_v1.standing`, the reward's one implementation.
- A new privileged input is a new version of this spec.

## Action spec
- Masks come from the game's own gates, never a copy of them. Every number comes from its row through `action_tables`, quantized once: never a literal. A new grenade row APPENDS a hand entry; a new verb is a hand-table row with its gate in the same patch.
- A pointer becomes a game id at the decision (against the `list_uid` the row was chosen on) and is never kept as a position; the aim at a unit follows the team picture's record of it, through the one control-seam quantizer (`aim_vector`).
- The decoder is a per-tick step of the `policy` controller issuing raw rows through `issue_row`, so `VERB_RULES` stays THE hold / latch rule and nothing new reaches the sim or the recording. A dead or turned row issues nothing, ever.
- Masks are built at the START of the decision tick, ONE gate call per hand bit (`build_masks`). A head left out of learning has ONE legal option; no head is ever all-masked. A masked in-range index decodes to its fallback and is COUNTED; an out-of-range index is refused. A masked choice issued by force does nothing (`tests/_1h_forced.py`).
- The flashlight hand entry is RESERVED (always masked, decoded as nothing). The inverse helpers are the nearest-choice translation: never a second table.

## Reward profile
- The env returns ONE number per group for every `in_match` row, dead and turned rows included, and NEVER their sum (the trainer weights the groups and blends own against team). A new term is a new counter in a group, its sign fixed in `_COUNTER_TERMS`.
- Health is counted in BARS (`TeamPicture.bars`), clipped at one. A team's balance is ONE function (`standing` / `balance`), also the full view's column.
- A change not credited to a unit is a team's UNCREDITED REST, built from its sources, never computed as a remainder; a tick whose events do not reproduce a unit's hp or life raises `RewardBooksMismatch` (a game finding).
- Eliminations, the win and the clock come from `MatchRules` (never a second rule of who won). Under a profile the clock is an ENDING (`terminated`, never `truncated`). Read-side only: nothing synced, no digest moves.

## Training snapshot
- A training run's code is an ANNOTATED tag at HEAD in its own detached checkout (the git-workflow training-checkout row). `require_snapshot` refuses, naming the check, unless the tag is annotated, HEAD is its commit, the tree is clean including untracked files under `src/`, `levels/`, `config.toml`, `cpp/`, both modules come from this checkout, `bp.build_id()` == HEAD, and the run file's sha256 is recorded. `device` also needs an empty twin ledger and is refused until P8b.
- With `[run] out_dir` (refused inside the checkout) `BatchedEnv` writes `run_manifest.json`; a resume refuses any change but the host, by name.
