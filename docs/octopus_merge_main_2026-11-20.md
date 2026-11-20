# Merging main into the octopus branch (2026-11-20)

Branch `33-octopus-concept` (88 commits off merge base `0b4c4af9`) took
`origin/main` at `7cba57dd` (1191 commits) with `git merge --no-ff`, merge
commit `b8b04d33`, then follow-up commits on the branch. Not pushed. Work
Desktop (DESKTOP-F4DJPJD), the `data` env, the worktree's own CPU
(`cpp/build`, Ninja) and CUDA (`cpp/build_cuda`, `cpp/build_cuda.bat`) builds.

## 1. The conflicts, one line per file (29 files)

| File | Resolution |
|---|---|
| `.gitignore` | both: main's prop-pack line, then the P5z traces line |
| `CLAUDE.md` | main's (the #206 lean index); the octopus's long rows translated afterwards (section 2.1) |
| `config.toml` | both `[render.creature_looks]` / `[render.creature]` and main's `[render.pose]`; `[control.keyboard]` DROPPED with the keyboard scheme (2.2) |
| `cpp/CMakeLists.txt` | both: `arm_chain_solver.cpp` beside main's new TUs in the source list AND on the `/fp:strict` list |
| `cpp/src/bindings.cpp` | both includes (the arm chain + main's horde / sight / philox / swarm / build id) |
| `level_lib.py` | both: the `creature` managed family + main's `levelgen` / `room` / `opening`; the creature row writer beside main's room-graph writers; `creature_controller` and main's `sight` keys in `write_match` |
| `level_loader.py` | both: `LevelData.creatures` + main's `levelgen` / `rooms`; the creature cell checks beside main's parsers |
| `main.py` | `select_play` + main's `changed_between` import; main's camera block (`owns_keys`, the follow camera) instead of the octopus's `pans_camera_with_wasd` |
| `renderer/game_renderer.py` | `compose_world(..., charges, charge_tick, creatures=None)`; the creature filter + `draw_creatures` views kept around main's muzzle-flame and tracer passes; main's `update_camera` |
| `renderer/unit_looks.py` | main's `HandAssigner` / `handedness_table`; `__all__` keeps `creature_looks_table`, `CREATURE_ROLE_PREFIX` |
| `renderer/unit_model_renderer.py` | both imports; both init blocks (creature rig / skin state + main's pose layer state); `draw_units` is main's (the pose-layer prepass) over the octopus's `_frame_setup` split, which already carries main's held-prop materials; both method blocks (creature draw + main's pose prepass) |
| `src/control_gamepad.py` | main's (`GamepadHalf`, `stick_throttle_q`); the octopus's `analog_speed_q` / `first_controlled_unit` dropped |
| `src/control_source.py` | main's helpers (`quantize_point_aim`, `stick_throttle_q`, the controller registry with replay / policy / commander / door bots) + the octopus's `creature` controller row, `play_team`, `select_play`; `_KNOWN_CONTROLS` main's (`keyboard` dropped) |
| `src/simulation/ai_zombie.py` | main's (the G9p horde law); the `can_become_zombie` rule moved into `rises_as_zombie`, the one predicate |
| `src/simulation/exchange.py` | main's wind row; the retired grad-P push and the octopus's creature push row removed; the creature's response folded INTO the one wind row (2.4) |
| `src/simulation/factions.py` | main's; `CREATURE_TEAM` added to `__all__` |
| `src/simulation/gamemap.py` | main's; `topology_serial` dropped, the creature bodies read main's `_tile_serial` |
| `src/simulation/intents.py` | main's `MoveDirIntent(dx_q, dy_q, throttle_q)`; `speed_q` dropped (2.2) |
| `src/simulation/match.py` | both: `creature_controller` / `creatures` / `CreatureSpawn` + main's `sight`; creatures spawned before main's `start_team_picture` |
| `src/simulation/recorder.py` | both: main's swarm rings + the creature ring; `record(..., swarm=None, creatures=None)` |
| `src/simulation/simulation.py` | main's slot comments, throttle `set_move_dir`, turning accessors, swarm carrier; slot 5b added to the docstring; the creature wind through the one wind row; the round-end snap skips a creature through `set_pos_q`; `clear_aim` later dropped |
| `src/simulation/unit.py` | main's (schema-generated slots, Q16 position, turning law); the `_armed` species rule and `hit_shape` kept |
| `tests/field_ab_harness.py` | both carriers (`__creature__`, `__swarm__`) and both diff locators (a lost `return out` restored after the merge) |
| `tests/field_digest.py` | both folds, each presence-gated (`n_creatures > 0`, swarm `present`) |
| `tests/test_exchange_reductions.py` | main's (the table found by field) |
| `tests/test_no_float_in_sim_tu.py` | both: `arm_chain_solver.cpp` and main's swarm TUs at 0/0/0 |
| `tests/test_wave_push.py` | main's |
| `tools/e2e_drive.py` | main's (its `--hold KEY@A-B` supersedes the octopus's `--hold KEY@F0..F1`) |
| `tools/look_shots.toml` | both: main's blast / flashbang / larva shots, then `octopus_tank_rest` |

## 2. Semantic integration

### 2.1 CLAUDE.md and the rules (item 1)
The octopus's long-form CLAUDE.md rows became ONE index line per system in
main's tables (Sim core: Creature table, Chain currency, Activation table,
Chain integer reference, Arm chain solver, Creature bodies, Hit seam, Chain
accessor; Render: Creature render), a row `creatures.md` in the region table,
and a new region rule file `.claude/rules/creatures.md` (`paths:` over the
creature modules, the solver, the reference, the renderer side, the tools and
the tests) holding the constraints. Existing rule files amended: `sim-core.md`
(slot 5b), `units.md` (the creature exception in Unit, Turning law, Coupling
table). The long rows' history (P-numbers, gate numbers) was left out:
rules hold no history. `tests/test_claude_rules_paths.py` passes.

### 2.2 Analog speed onto main's throttle, the keyboard scheme onto main's direct scheme (item 2)
- `MoveDirIntent.speed_q` is gone. A creature reads main's `throttle_q` as its
  analog speed through `creature_primitives.creature_speed_q16`: the throttle
  itself, EXCEPT throttle 0. Main's contract is "0 = the walk" (a MOVE_DIR
  always moves; the stop is `clear_move_dir`), the octopus's was "speed 0
  idles". So a throttle of 0 asks for the creature's walk, taken as the crawl
  at its full tempo (`crawl_until`), which is what the octopus's drivers sent
  for "crawl". OPEN QUESTION Q1.
- `src/control_keyboard.py` (`KeyboardDirect`, `--control keyboard`) and
  `[control.keyboard]` are deleted: main's `DirectControl` ("THE one human
  direct-control scheme; a device is a half") already possesses the first
  living unit of `sim.player_team`, so `--play creature --control direct`
  drives the octopus with main's keyboard half (WASD = throttle 0 = crawl,
  the full-speed key = throttle ONE = the jet train) and pad half (stick
  deflection = `stick_throttle_q` = the analog speed, the same rescale the
  octopus's `analog_speed_q` did). `DirectControl.drives_one_unit = True`
  is what `select_play` checks. The octopus's Shift / Ctrl slow and mid
  speeds have no counterpart on main's keyboard half: OPEN QUESTION Q2.
- `pans_camera_with_wasd` dropped: main's loop never pans while a direct
  scheme owns the keys.
- `Simulation.clear_aim` dropped (only `KeyboardDirect` called it, and it was
  outside main's intent tap). With main's keyboard + mouse half the cursor is
  always an aim, so a played octopus always has a reach goal: Q2.
- `tools/e2e_drive.py` is main's; its `--hold KEY@A-B` covers the octopus's use.

### 2.3 Unit schema, position, facing, device paths (item 3)
- `Unit.hit_shape` is a `REFERENCE_ONLY` schema entry with its reason (DERIVED
  each tick from the chain, never digested). The hit seam asks the species
  FIRST (`Unit._creature_hit_shape`), so a human never reads the attribute
  and main's probe (`test_unit_schema`: REFERENCE_ONLY never read on the
  pinned matches) passes. For a creature it IS read on the match path (the
  march, the laser), which `REFERENCE_ONLY`'s wording does not quite cover:
  OPEN QUESTION Q5.
- `cpp/src/unit_tables_layout.h` regenerated (`tools/gen_unit_layout.py
  --write`): the per-species device rows (stability, mitigation) are two
  species wide with the octopus. `tests/cuda_unit_damage_check.py` drew a
  random block one species wide for a per-species column; it now writes
  species 0's block (its rows are all humans) and leaves the octopus's real.
- Slot 5b writes the creature's anchor through `Unit.set_pos` (the float door,
  quantized once onto the Q16 grid) and its facing through `turning.place`
  from the Q24 heading rounded once to Q16 (`creature_fixed.
  heading_q24_to_facing_q16`, half away from zero). Before the merge the
  unit held the float values exactly; now they are on the Q16 grid, so the
  octopus's tests that compared floats exactly compare the quantized values
  (and the accessor's float heading equals the unit facing within one Q16
  count).
- Slot 1a (`unit_impulse.integrate_all`: impulse integration + the turning
  law) skips a creature, as slot 3, the WEGO plan, the round-end snap and
  the AIM facing did on the branch: the creature's anchor and facing have ONE
  per-tick writer. A push (a shove, a round's push effect) still lands in a
  creature's unit impulse accumulator and is never integrated: OPEN Q4.
- `GameMap.topology_serial` dropped: main's `_tile_serial` is the same counter
  (bumped by `on_tile_changed`, also by the load-time cache rebuild).
- Device: the arm chain stays host-side on every path. The device conductor
  (Class T) now REFUSES a creature by name (`ConductorRefused`, `_refuse`).
  `is_creature` moved beside the species table (`species.is_creature`,
  re-exported by `creature_system`) so the unit laws that skip a creature
  read no creature runtime; the twin manifest was regenerated (species.py
  joins the `files` of six Class-T twins: integrate_turn, wind_impulse,
  movement, verbs, table_appends, doors) and `creature_system.py` is a
  `[not_twinned]` row with its reason.

### 2.4 The wave push became the wind (item 3 / coupling table)
Main deleted the pressure-gradient push (`apply_wave_push`; units.md "Never a
pressure-gradient push ... never a second reader") for the wind row
(`apply_wind_force`, G5p-b). The octopus's creature row read that gradient.
Ported: the ONE wind row now takes `creatures=` and hands a creature unit's
wind to its BODY (`exchange._creature_wind`: THE read `footprint_wind`, THE
price `wind_impulse_q` at the standing drag area, divided by the unit's mass,
added to the crown velocity). One row, one reader; no knockdown; vetoable by
passing `creatures=None`. The PHYSICS (a marine's drag area on an octopus,
the impulse over the unit's whole mass) is a modelling call: OPEN QUESTION Q3.

### 2.5 Species (item 4)
No collision: main's swarm species table is for swarm units (not `Unit`s);
the octopus is a `Unit` species with a creature row. Both work side by side.
`can_become_zombie` now gates main's ONE conversion predicate
(`rises_as_zombie`), so a dead octopus neither rises from a bite nor from
main's larval infestation (the latter is new: main's corpse law lets ANY
corpse rise; an octopus corpse now never does). Part of Q6.

### 2.6 Pose (item 5)
Kept: `creature_pose.py` poses the octopus from its sim chain; it does not
route through main's pose layer (`renderer/pose_layer.py`, clip-driven,
skeleton-agnostic). The collision was real, not only conceptual: main's pose
layer (on by default since 2026-11-07) built a rig for EVERY loaded look,
refused the octopus skeleton (`PoseConfigError: no [render.pose.skeletons]
table fits`) and every window died at frame 0 (all the E2E and window-replay
tests). `_setup_pose` now skips the creature looks. Recorded in
`creatures.md` (Creature render): whether the chain pose becomes a pose-layer
source is Erik's (Q8).

### 2.7 Digest, goldens, recorder (item 6)
`__creature__` and main's `__swarm__` are both presence-gated sections after
the entity sections, each with an always-written carrier; a creature-free,
swarm-free snapshot hashes as before. No golden was re-baselined. Results in
section 3.

### 2.8 Determinism (item 7)
`arm_chain_solver.cpp` is on the `/fp:strict` list and at 0/0/0 on the float
ratchet beside main's swarm TUs. New code at the merge is integer except the
creature wind's float door into the Q24 chain velocity (quantized once by
`CreatureBody.add_velocity_m_s`), the same door-3 shape the branch's push row
had.

### 2.9 The match replay header
Main's replay header (`match_replay.spec_to_dict` / `spec_from_dict`) did not
know the creature fields (`creatures` came back as dicts). They are written
only when present (a creature list, a non-default creature controller), so a
creature-free header is byte for byte main's, and read back as
`CreatureSpawn` rows.

## 3. Gates

All on the Work Desktop, the `data` env, the worktree's own builds (CPU
`cpp/build` via Ninja + MSVC, BUILD_EXIT 0; CUDA `cpp/build_cuda` via
`cpp/build_cuda.bat`, BUILD_EXIT 0). The GPU was otherwise idle (desktop
apps only, 9-23 % utilisation).

| Gate | Result |
|---|---|
| Full suite, first pass (`pytest tests -q`, at `aa215a49`, 42 min) | 6901 passed, 52 failed, 36 skipped, 6 xfailed |
| The 51 failures (all but the stamp test) re-run after the fixes below (at `8ebd525d`) | 46 passed, 5 failed |
| Those 5 after the last two fixes (at `6f85e0aa`) | three pass (98 tests in their files); the two still red are main's own (Finding 1) |
| Octopus tests (creature_*, hit_seam, chain_accessor, octopus_*, activation_table, fixedpoint_twins, bench_octopus, cuda_creature) | all pass; the long gates behind `BREACH_OCTOPUS_LONG_GATES=1` skipped, as on the branch |
| `tests/test_cuda_creature.py` (CPU == `--cuda` == `--resident`) | pass |
| Source lints (match draws, turning law, Q16 positions, impulse, ingress, no-BLAS, traits, footprint builder, float ratchet) | 169 passed |
| `tests/test_claude_rules_paths.py` | 77 passed |
| `tests/test_twin_debt.py` / `tools/twin_debt.py` | pass after the manifest regeneration / "twin debt: none" (ledger level at `6f85e0aa`) |
| `tools/cuda_green.py` + `tests/test_class_p_level.py` | section 3.1 |
| Goldens | NONE re-baselined. `GOLDEN_AGGREGATE`, the ship_demo controller pin and the bots / bots_v1 pins passed in the full suite (no creature in their scenes; every creature path is presence-gated) |
| E2E: `tools/e2e_drive.py --level octopus_tank --play creature --control direct` (330 frames; R held 20-200 = the walk, R+A 220-320 = full speed) | OK, clean exit: the octopus crawls North under the walk and jet-trains under full speed (along its heading, West: the known v1 "a swim does not steer"); pose 1.2 ms + skin 0.45 ms per creature-frame. Shots: `docs/octopus_merge_shots_2026-11-20/` |

What the first pass caught, and the fix (each committed):
- ~33 CUDA tests: the device table header was stale (two species) -> regenerated, CUDA rebuilt.
- 9 window tests (E2E, sound, magazines, throttle, grenades, follow camera, window replay): main's pose layer refused the octopus look -> creature looks skipped; then a stale octopus `wasd` term in `update_camera` -> main's lines restored.
- `test_unit_schema` (2): hit_shape read for humans -> the species gate.
- `test_match_replay`: the replay header (2.9).
- `test_orders` (commander replay) and `test_twin_debt` (2): fixed by the above and the manifest regeneration.

### 3.1 CUDA green

- `tools/cuda_green.py` at `6dd1bd56` (CUDA build id = HEAD): 77 passed, 2
  failed, 69 min -- REFUSED, no stamp. Both failures
  (`test_cuda_unit_payloads`, `test_cuda_unit_shooting`) were
  `numpy ArrayMemoryError: Unable to allocate 866 KiB`: the 16 GB machine
  was out of RAM while ANOTHER session ran its own full `pytest tests` and
  CUDA checks and a `lighting_demo` from the `97-w2-foam` worktree beside
  it (GPU and RAM shared). Re-run alone, both pass (2 passed, 16 min).
- The stamp run at the final HEAD (this document's commit) is reported in
  the hand-back; `.cuda_green` is untracked by design, so it cannot be
  recorded here without moving HEAD past it.

## 4. Findings

1. PRE-EXISTING ON MAIN, this machine: `tests/test_fire_tuning_level.py::test_generator_is_byte_deterministic` and `tests/test_larva_studio.py::test_the_studio_generates_deterministically_and_is_the_checked_in_one` fail on a pristine `origin/main` (`7cba57dd`) here too (checked in a scratch worktree, since removed): the baked `diffuse.png` differs from the committed one, most likely an image-library difference on the Work Desktop. Not touched. The octopus tank's `diffuse.png` was re-baked by its generator for the same reason, and its `level.toml` moved to format v3 (main's loader refuses v2).
2. The device conductor refuses a creature by name; the creature chain has no device form (the branch's recorded debt, unchanged). `--cuda` / `--resident` play still run slot 5b on the host (`test_cuda_creature` green).
3. `is_creature` lives in `species.py` (re-exported by `creature_system`) so the six Class-T twins whose slots now ask it (integrate_turn, wind_impulse, movement, verbs, table_appends, doors) trace the species table, not the creature runtime; `creature_system.py` is a `[not_twinned]` row with its reason.
4. A push other than the wind (a shove, a round's push effect) still lands in a creature's UNIT impulse accumulator, which slot 1a no longer integrates for a creature: never felt, and the accumulator stays non-zero (hashed while non-zero). No golden holds a creature. Q4.
5. Under main's keyboard + mouse half the cursor is ALWAYS an aim (DirectControl holds the last aim; the octopus's `clear_aim` is gone), so a played octopus always has a reach goal toward the cursor. Q2.
6. Main's `cuda_unit_damage_check.py` pinned a one-species device table; it now writes species 0's block (a table designed to grow).
7. The octopus's `test_creature_speed.py` lost two properties with the systems they tested (the octopus keyboard map; "a marine ignores speed_q": the marines and a creature now read ONE throttle) and gained two (the throttle -> creature speed map; the pad's throttle as the creature's speed). The `--play` test drives main's `DirectControl` through its one emit seam. The octopus's position / facing tests compare the Q16-quantized values (main's unit doors).
8. The creature body's xorshift64 predates main's Match draws rule (every random number a keyed Philox block, a `Salt` member). Not redesigned. Q7.
9. The twin ledger: the merge moved the files of every twin (Class P through `fixed_point.h`'s chain kit, Class T through `unit.py` / `simulation.py` / `unit_impulse.py` / `exchange.py` / `ai_zombie.py` / `species.py`); all marked level at `6f85e0aa`, proven by `tools/cuda_green.py` (3.1).

## 5. Open questions for Erik

- **Q1 (control, feel).** Main's throttle says 0 = the WALK (a MOVE_DIR always moves; the stop is no MOVE_DIR); the octopus said speed 0 idles. At the merge a throttle of 0 is the octopus's walk = its crawl at full tempo (`crawl_until`); any throttle above 0 is its speed unchanged, so the stick keeps its slow crawl. Is "the walk is the full crawl" right, or should a creature read the throttle differently?
- **Q2 (control, feel).** The octopus's keyboard scheme had Shift / Ctrl for the slow / mid speeds (the full crawl, the umbrella); main's keyboard half has the walk and the full-speed key only, so on the keyboard the octopus crawls or jets, never the umbrella cruise. And the mouse is always an aim, so the octopus always reaches toward the cursor. A mid-speed key in `[controls.keys]` (a main-wide change), a creature-only binding, or the pad as THE creature device?
- **Q3 (physics).** The creature's wind uses main's read and price (the marine's standing drag area, `[impulse] drag_area_m2`), the impulse divided by the unit's whole mass, added to the crown. The octopus's own drag area and the water it sits in do not enter. Keep, or give creatures their own drag-area row?
- **Q4 (physics).** Shoves and a round's push reach a creature's unit impulse accumulator but are never integrated. Should a push reach the body as the wind does, or be refused for a creature?
- **Q5 (schema).** `hit_shape` is `REFERENCE_ONLY` but is read on the match path for a creature. A fourth group (host-only derived state), or a wording change to group R?
- **Q6 (gameplay).** A dead octopus never rises: not from a zombie's bite (the branch's rule) and now also not from main's larval infestation of a corpse (the `can_become_zombie` gate sits in the one conversion predicate). Intended for the larvae too?
- **Q7 (determinism design).** The creature body's own xorshift64 vs main's keyed Philox draws. Port the creature's draws onto `sim.draws`?
- **Q8 (render).** The octopus is posed from its chain outside main's pose layer. Should the chain pose become a pose-layer source (one compose path), or is a creature legitimately outside it?
