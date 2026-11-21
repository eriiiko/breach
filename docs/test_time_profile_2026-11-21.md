# Where Breach's test time goes -- 2026-11-21

A dated capture, report only: no test was changed. It answers four
questions: where the CPU suite's time goes, where the card gate's time goes
and what arc #217 does to it, which tests pin the same property as another,
and which levers save the most minutes. It follows
`docs/test_suite_timing_2026-11-15.md` (#205 step 1) and uses that report's
instrument (`tools/suite_timing.py`, `tests/_suite_kinds.py`).

Every number names its machine. The Home Desktop, where the #29 session runs
its card gate, was not measured here. Its 72 -> 94 min growth is the #29
session's own figure.

## The data

| id | what | machine | date | how |
|---|---|---|---|---|
| A | full-suite and card-gate wall times from logs (no durations) | Work Desktop | 2026-11-20/21 | five `pytest tests -q` runs and four `tools/cuda_green.py` runs left in session scratch dirs |
| B | the last full run WITH durations (source of `tests/slow_tests.txt`) | Lenovo | 2026-11-15 | serial, 3508 s, `--durations=0` (issue #205) |
| C | the 93 slow non-device tests of the slow list, re-timed at `cdf8f413` | Work Desktop | 2026-11-21 | ONE pytest process, `--durations=0`, CPU module, no CUDA build in the worktree, BLAS at one thread; one other agent's card run on the machine at the same time. 1 failed (`test_class_p_level` against a stale local `main`), 91 passed, 1 xfailed, 1471 s |
| D | the HOST half of the card checks (every reference play), timed alone | Work Desktop | 2026-11-21 | `cuda_obs_check.play` / `cuda_conductor_check.record_match` / `_pinned_matches.pinned_play` on the CPU module, no device twin |
| E | #217's red card runs on its branch (bots runs to 7200 ticks there) | Work Desktop | 2026-11-21 | `C:\tmp\s8_cg{1..4}.txt`, `s8_cuda_green.txt`: the checks' own per-play and per-tick prints |

No full suite and no full card gate was run for this report (the GPU is
shared). Device-side seconds therefore come from E's per-tick prints and are
marked as estimates (est.).

## Headline

1. **`pytest tests -q` runs the card gate inside it.** In a checkout with
   `cpp/build_cuda`, the ~94 card tests are not skipped. The "~100 minute
   CPU suite" is a ~36 min CPU part plus a ~65 min card part, and
   `cuda_green.py` then runs that card part a second time. This is the
   biggest lever by far (Lever 1).
2. The CPU part is still **the matches**: 78 % of the slow tier is tests
   that play a match. The pinned-match sharing of #205 works. Both pinned
   plays together now cost ~46 s per process (Lenovo: ~500 s before).
3. The card gate's long checks spend more time on the HOST reference (the
   host observation is ~32 ms/tick) than on the card twin (~13 ms/tick).
   `bots` is replayed to its end in **3** of the 13 `cuda_obs_check`
   subprocesses, not ten, plus once in `cuda_conductor_check`. At today's
   1349 ticks that is ~3 min of obs. Under #217's 7200 ticks it becomes
   ~15 min of obs and ~14 min of conductor: the card gate grows by ~20 min
   as written, or by ~7-10 min if each match is played once per process.

## 1. The CPU suite

### Serial wall time, with and without the card part (A, Work Desktop)

| run (session scratch) | date | CUDA build in checkout | result | wall |
|---|---|---|---|---:|
| `full_suite2.log` | 2026-11-20 | no (75 skipped) | 6690 passed | 2191 s (36.5 min) |
| `w1/suite.log` | 2026-11-20 | no (75 skipped) | 6711 passed | 2155 s (35.9 min) |
| `arc_close_suite.log` | 2026-11-20 | yes (3 skipped) | 6838 passed, 19 failed | 5991 s (99.9 min) |
| `full_suite.log` (106 worktree) | 2026-11-21 | yes | 6955 passed, 3 failed | 6274 s (104.6 min) |
| `cuda_green.py` alone, 79 card tests | 2026-11-20/21 | yes | green x2, red x1 | 4018 / 4141 / 4337 s (67-72 min) |

With a CUDA build minus without it: 5991-6274 vs 2155-2191 s, so about
**64-68 min**. That is the card gate's own time. The CPU part is ~36 min.

### By kind (C; the slow non-device tier, Work Desktop, 2026-11-21)

| kind | seconds | share of C | tests |
|---|---:|---:|---:|
| plays a match | 1142 | 78 % | 56 |
| other (pure-Python sweeps, small scenes) | 164 | 11 % | 14 |
| physics scene | 119 | 8 % | 14 |
| level generation | 41 | 3 % | 6 |

The whole CPU part on the Work Desktop, put together (est.): the slow
non-device tier is **1466 s**. The window / GL tests are ~70 s (B; not re-run
here, they drive the GPU). The other ~6600 tests plus collection (~30 s) are
the **~630 s** that remain of the ~2170 s CPU part.

### By file (C, over 10 s)

| file | seconds | slow tests | B (Lenovo 11-15) |
|---|---:|---:|---:|
| `test_bots_door.py` | 270.0 | 10 | 228 |
| `test_obs_spec_v1.py` | 230.8 | 4 | 79 |
| `test_reward_v1.py` | 93.7 | 6 | 158 |
| `test_rl_env.py` | 90.9 | 3 | 90 |
| `test_match_draws.py` | 63.5 | 2 | 51 |
| `test_explosion.py` | 63.0 | 1 | 78 |
| `test_unit_schema.py` | 61.4 | 4 | 506 |
| `test_m3_derived_h_bed.py` | 52.5 | 4 | 45 |
| `test_action_spec_v1.py` | 47.5 | 2 | 43 |
| `test_unit_impulse.py` | 46.5 | 2 | 134 |
| `test_fire_o2_invariant.py` | 42.8 | 2 | 21 |
| `test_orders.py` | 41.4 | 3 | 29 |
| `test_turning_law.py` | 37.0 | 2 | 49 |
| `test_wind_doorway_e2e.py` | 30.1 | 4 | 23 |
| `test_unit_store.py` | 27.6 | 1 | 19 |
| `test_trace_resolution.py` | 23.9 | 1 | 13 |
| `test_bake_level_art.py` | 21.1 | 3 | 20 |
| `test_match_replay.py` | 19.8 | 3 | 16 |
| `test_follow_camera.py` | 18.0 | 1 | 14 |
| `test_gen_ship_vents_air.py`, `test_fire_vfx_ref.py`, `test_main_replay.py`, `test_smoke_transport_v2.py`, `test_trace_currency.py`, `test_horde_behaviours.py`, `test_bots.py` | 10.8-13.9 each | | |

What moved since the Lenovo run: `test_unit_schema.py` 506 -> 61 s,
`test_bots_v1.py` 114 -> 8 s and `test_zombie_control.py` 48 -> 3 s, from the
#205 shared pinned plays and the shorter pinned matches (bots 2721 -> 1349
ticks; the bots_v1 variant pins were 7200-tick draws and are now ~270-310
ticks). `test_obs_spec_v1.py` grew 79 -> 231 s (next table).

### The 20 slowest CPU tests (C)

| # | s (C) | s (B) | test | why it is long |
|---:|---:|---:|---|---|
| 1 | 178.1 | 41.2 | `test_obs_spec_v1::test_the_planes_equal_an_independent_reconstruction` | runs on to 1200 ticks of ship_demo (the first fire lane comes at 984) and rebuilds every plane by hand every 24 ticks |
| 2 | 75.1 | 61.7 | `test_rl_env::test_g_a_zero_rows_play_run_match_s_idle_match_on_ship_demo` | 3 farm envs + 3 `run_match`, 480 ticks each |
| 3 | 68.6 | 60.6 | `test_bots_door::test_the_records_are_deterministic_and_change_nothing` | plays the recorded match twice more |
| 4 | 63.0 | 77.6 | `test_explosion::test_a_replay_of_a_match_with_grenades_clears_every_checkpoint` | records + replays ship_demo bots to its end (1349 ticks, cap 2160) |
| 5 | 45.0 | 84.0 | `test_reward_v1::test_the_books_close_on_ship_demo_bots_matches[3]` | whole bots match, seed 3 |
| 6 | 40.0 | 36.0 | `test_action_spec_v1::test_forced_masked_choices_do_nothing` | |
| 7 | 38.9 | 37.7 | `test_bots_door::test_the_capture_is_lossless` | 2 x 600 ticks |
| 8 | 37.5 | 44.9 | `test_unit_impulse::test_g11_every_position_stays_on_the_q16_grid_in_the_pinned_match` | **pays the shared `bots` pinned play** (first watcher in the process) |
| 9 | 36.0 | 33.0 | `test_bots_door::test_capture_only_every_soldier_verb_is_the_decoders` | |
| 10 | 35.7 | 16.8 | `test_fire_o2_invariant::test_o2_gate_heterogeneous_disagreements_are_exact_ties_only` | a sweep |
| 11 | 31.8 | 25.5 | `test_match_draws::test_site_invariance_in_a_played_match` | 2 x 700 ticks |
| 12 | 31.7 | 25.6 | `test_match_draws::test_same_seed_same_match_other_seed_another` | 2 x 700 ticks |
| 13 | 31.4 | 85.7 | `test_unit_schema::test_reference_only_is_not_read_on_the_continuous_match_path[bots]` | own play of bots seed 1 to the end |
| 14 | 30.9 | 26.6 | `test_bots_door::test_the_records_drive_the_game` | |
| 15 | 30.2 | 43.2 | `test_turning_law::test_g9_the_bots_prediction_is_the_facing_the_shot_leaves_along` | own play of bots seed 1 to the end (hooks `handle_frame`) |
| 16 | 29.0 | 22.5 | `test_m3_derived_h_bed::test_the_fleck_damping_stays_inert_through_a_full_corrected_burn` | a full burn |
| 17 | 28.4 | 39.9 | `test_bots_door::test_a_masked_latch_is_never_sent_on_a_whole_match` | |
| 18 | 27.7 | 40.8 | `test_reward_v1::test_the_books_close_on_ship_demo_bots_matches[8]` | |
| 19 | 27.6 | 19.2 | `test_unit_store::test_identity_ship_demo_every_tick` | |
| 20 | 23.9 | 12.7 | `test_trace_resolution::test_v7_the_configured_decay_holds_down_to_visible_densities[smoke]` | |

These 20 tests take 906 s, 62 % of C and ~42 % of the CPU part.

### The pinned match plays (`tests/_pinned_matches.py`)

| item | seconds (Work Desktop) |
|---|---:|
| `pinned_play("bots")`, 1349 ticks, with a deep copy of every tick's view (D) | 35.9 |
| `pinned_play("bots_v1")`, 272 ticks (D) | 7.5 |
| the observers on the shared plays (C): zombie_control 3.0, schema derived[bots_v1] 1.1, positions_q16 and schema capacity < 0.5 each | ~5 |
| **total per pytest process** | **~48 s, 2 % of the CPU part** |

Before #205 these watchers cost ~500 s (Lenovo, B). The views of the bots
play hold ~77 MiB (tracemalloc, D). Under #217 (7200 ticks) the play becomes
~190 s and ~0.4 GB **per process**, so per xdist worker that touches it.

### What #217 does to the CPU part (est., Work Desktop)

The CPU tests that play bots seed 1 to its end scale with 1349 -> 7200 ticks:
the shared `pinned_play("bots")` (36 -> ~190 s),
`test_turning_law::test_g9_...` (30 -> ~160 s) and
`test_unit_schema::test_reference_only_...[bots]` (31 -> ~165 s).
`test_explosion`'s replay grows to its 2160-tick cap (63 -> ~100 s). Together
that is **+~8 min** on the ~36 min CPU part. The other match tests cap their
plays, or play other seeds or bots_v1.

### Serial against xdist

`pytest-xdist` is **not installed** in the Work Desktop `data` env, so the
parallel run has only been measured on the Lenovo (#205, 2026-11-15):
serial 3508 s, `-n 4 --dist loadgroup` 1811 s (1.94x), `-n 8` 1360 s
(2.58x, but `test_wire_tool.py` errors on a shared temp file in the repo
root). Under `loadgroup`, every card and window test is one `device` group on
one worker (`tests/_suite_kinds.py::is_device`). With a CUDA build in the
checkout, that group IS the ~65-94 min card gate, so it bounds the wall
whatever N is. xdist pays only on the CPU part, so only after Lever 1.

## 2. The card gate (`tools/cuda_green.py`)

`cuda_green.py` runs `pytest tests -q -k cuda`: one pytest process, and one
CUDA subprocess per test (`tests/cuda_harness.run_cuda_script`). A check's
plays are cached only inside its subprocess (`cuda_obs_check._PLAYED`), so
two tests of one check share nothing.

### Which check replays which match

Only these checks play a ship_demo match **to its end**. Every other check
caps its ship_demo play: the 16 `cuda_unit_*_check.py` play ship_demo's bots
match for `SHIP_TICKS` = 500 (store 700), `unit_verbs`' done part caps at
2400, and the rest are hand scenes.

| test (one subprocess each) | check part | host plays in that subprocess (ticks at `cdf8f413`) | device twin |
|---|---|---|---|
| `test_cuda_obs::..._on_the_corpus` | `gate` | **bots 1349**, bots_v1 272, range_door 200, hand 320, range_hall 32, hall_cover 26, ship_all_round 480 (cap), door_grenade 150, slow_turn 40, list_hand 24, fire_hand 60, clock_hand 48, corpus (c) 8 policy runs 1297 | every one, N = 1 |
| `test_cuda_obs::test_each_env_of_a_batch_...` | `batch` | bots@11 1664, bots_v1@12 300, bots@13 970, bots_v1@14 275, 4 x range_door ~220 | N = 2 and 4 |
| `test_cuda_obs::test_who_a_team_sees_...` | `wall` | **bots 1349**, ship_all_round 480 | the whole match, for **6** perturbation checks (`done >= 6` then `continue`) |
| `test_cuda_obs::test_the_device_books_...` | `books` | **bots 1349**, range_door 200, clock_hand 48 | profile "range" |
| `test_cuda_obs::test_the_device_decoder_decodes_...` | `dec_gate` | corpus (c), 1297 | gate(c), the same as part `gate`'s (c) |
| `test_cuda_obs::..._refuses_by_name_and_adds_no_sync` | `decoder` | range_door; **all 8 of corpus (c)** to use `corpus_c()[1]` | short |
| `test_cuda_obs::test_a_door_beside_blast_heated_gas_...` | `range_scenes` | 2 policy runs (cap 600) | gate |
| `test_cuda_obs` restage / handover / syncs / ended | | hand 320 / range_door x3 | short |
| `test_cuda_obs` sight / floatkit | | no play | |
| `test_cuda_conductor::..._pinned_matches_to_their_end` | `pinned` | **bots 1349 x2** (record via `run_match`, then the replay controller's reference), bots_v1 272 x2 | conductor, every tick, every plane |
| `test_cuda_conductor::..._a_recorded_match_...` | `match` | bots with a 300-tick clock, x2 | conductor |
| `test_cuda_conductor::..._syncs_only_at_its_named_reads` | `syncs` | bots, 200-tick clock, x2 | conductor |
| `test_cuda_conductor_batch::..._on_recorded_matches` | `matches` | records bots@11, bots_v1@12, bots@13, bots_v1@14 | 4 solo runs to 1664 + batches N = 2, 4 |

### The rates (Work Desktop)

| step | rate | source |
|---|---:|---|
| host play + host observation (`cuda_obs_check.play`), ship_demo bots | 31.7 ms/tick (42.8 s / 1349) | D |
| host `run_match(record=True)`, ship_demo bots | 23.7 ms/tick (32.0 s / 1349) | D |
| obs twin incl. the gate's downloads, N = 1 | 13.0 ms/tick | E (`(a) bots: 7200 ticks, 13.0 ms per tick`) |
| conductor `replay_gate` incl. the reference replay and the every-plane compare | ~85 ms/tick | E (840.6 s part total, minus 202.5 s record, minus bots_v1, over 7200 ticks) |
| host plays of the whole obs corpus (gate part) | 129 s | D |
| corpus (c), the 8 policy runs | 45.3 s | D |
| the batch's four ship plays | 103.6 s | D |

### The #29 belief, checked

> "Most of the growth is `tests/cuda_obs_check.py` replaying `bots` in about
> ten separate subprocesses."

- **Ten subprocesses: refuted.** `cuda_obs_check` runs in 13 subprocesses,
  and `bots` (seed 1) is played to its end in **3** of them (`gate`, `wall`,
  `books`). `batch` plays two other full bots-controller matches (seeds 11,
  13). The conductor plays `bots` to its end in 1 subprocess, twice.
- **Their cost today (est., Work Desktop):** each obs replay is
  42.8 s host + 1349 x 13 ms twin = ~60 s, so 3 x 60 = **~3 min**. The
  conductor's pinned part is 32 + 1621 x 85 ms = **~2.8 min**.
- **Obs is plausibly most of the growth, though: confirmed in kind.** Summed
  from D's host plays plus E's twin rate, the 13 obs tests come to
  **~12-15 min** on the Work Desktop (gate ~3.1, batch ~3.3, wall ~1.4,
  books ~1.1, dec_gate ~1.0, decoder ~0.9, the 7 small parts ~1-2, plus 13
  subprocess starts). That is the size of the Home Desktop's 22-min growth
  (72 -> 94) once the slower card and the conductor tests of P7e are added.
  The mechanism is not `bots` in ten processes. It is every obs test
  replaying its corpus on the host with the host observation (~32 ms/tick,
  2.4x the twin), with `bots` in 3, corpus (c) in 3 (`gate`, `dec_gate`,
  `decoder`) and the batch seeds in 1.

### Once #217 lands (bots to 7200 ticks, bots_v1 ~274; E)

E's own runs on #217's branch show what changes: `played bots@None: 7200
ticks (214.3 s)`, `recorded bots: 7200 ticks (202.5 s)`, the obs twin at
13.0 ms/tick, and the conductor's pinned part at 840.6 s. The batch seeds
SHORTEN there (bots@11 1664 -> 971, bots@13 970 -> 947).

| bots-to-the-end replay | today (1349) | #217, (i) no change | #217, (ii) each match played once per process |
|---|---:|---:|---:|
| obs `gate` (host + twin) | 60 s | 308 s | 308 s |
| obs `wall` | 60 s | 308 s | 94 s (twin only, host play shared with `gate`); ~20 s if capped at its 6th check |
| obs `books` | 60 s | 308 s | 94 s (twin only) |
| conductor `pinned` (record + reference replay + conductor) | 147 s | 815 s | 613 s (one host play: the live reference feeds the conductor) |
| conductor_batch + obs `batch` (seeds 11-14) | ~9 min | ~6 min | ~6 min |
| **sum** | **~15 min** | **~35 min (+20)** | **~25 min (+10)**; with the wall cap and books' twin folded into `gate`'s, ~22 min (+7) |

So, est. on the Work Desktop: **(i)** the card gate grows by ~20 min (~23
min without the batch seeds' shortening). On the Home Desktop that is
roughly **94 -> 115-120 min**. #217's red run on the Work Desktop took
8279 s (2 h 18) under contention, with 12 tests failing early, which fits as
an upper bound. **(ii)** One play per process brings the growth down to
~7-10 min, so ~100-105 min on the Home Desktop.

Things #217 will meet in the card gate, noticed in passing:
`cuda_unit_verbs_check.part_done` requires the bots match (60 s limit) to
end in a WIN, and the obs books' coverage takes "elimination" from bots. Both
assume today's 1349-tick win.

## 3. Duplicate properties (candidates; Erik decides)

Read from the docstrings and asserts. Ranked by the time one could save,
Work Desktop. Nothing was deleted.

| # | candidate to drop | the test that already pins it | evidence | saves |
|---:|---|---|---|---:|
| 1 | `test_cuda_obs::test_the_device_decoder_decodes_as_the_policy_controller_does` (part `dec_gate`) | `test_cuda_obs::..._on_the_corpus` (part `gate`) | `part_dec_gate`'s own docstring: "P8.6a's corpus (c) alone (part GATE runs it too, with the census)". Both run `gate("(c) ...")` on every corpus (c) run, then `judge_decoder(corpus_c())`. The only difference is the try/except naming a refusal | ~1.2 min (45 s host + 1297 twin ticks + a subprocess start) |
| 2 | `test_cuda_conductor::test_the_conductor_replays_a_recorded_match_equal_every_tick` (part `match`) | `..._replays_the_pinned_matches_to_their_end` | the same `record_match` + `replay_gate` on the same level and seed, 300 ticks against the whole match. Today `match` is the conductor's only gate whose match ends on the clock. Under #217 `pinned` bots ends on the clock too, so `match` becomes a strict prefix | ~0.6 min |
| 3 | `test_match_draws::test_same_seed_same_match_other_seed_another`, its seed-3 half | `test_match_draws::test_site_invariance_in_a_played_match` | site invariance asserts that seed 3's draw log and digests with extra foreign draws EQUAL seed 3's plain run, which already implies same-seed determinism. Every pinned digest implies it too. The "seed 4 differs" half is also held by `test_bots::test_a_bots_match_is_identical_twice_and_replays_exactly` (seed 2 differs) | ~0.5 min |
| 4 | the "law dormant -> the pre-law pin" gates: `test_exposure::..._pre_g4p_match` (7.4 s), `test_grenade_kinds::..._pre_g3p_match` (9.0), `test_turn_rate::..._pre_g1p_match` (6.7), `test_horde_behaviours::..._stage_1_law` (10.8), `test_turning_law::test_g3_..._re_pin_1s` (6.8), and the pin half of `test_unit_impulse::test_g1_...` (9.0) | none; flagged against the house standard | each pins the digests of a VARIANT match (a law switched off). Four say in their own words that "the next DELIBERATE re-pin ... by another cause retires this gate's values" (exposure, grenade_kinds, turn_rate, horde_behaviours). Since then they were re-measured at the bite balance, the #183 merge and #215, so they now pin a re-measured snapshot, no longer the cause they were written for. G1's per-tick "impulse state exactly zero" half is a real property | ~0.8 min |
| 5 | `test_unit_impulse::test_g11_every_position_stays_on_the_q16_grid_in_the_pinned_match` | `test_unit_positions_q16::test_every_position_of_a_match_is_its_q16_view` | the same property (every unit's position is on the Q16 grid after every tick); g11 on `bots`, positions_q16 (stricter: ints, `x == x_q/65536`, tiles) on `bots_v1`. Both watch the shared plays | ~1 s (the shared play only moves to the next watcher) |
| 6 | `test_explosion::test_a_replay_of_a_match_with_grenades_clears_every_checkpoint` | `test_match_replay::test_a_recorded_ship_demo_match_replays_to_every_checkpoint` | the same record-and-replay of ship_demo bots, but match_replay stops at 240 ticks, which may hold no blast. NOT a duplicate as is; a cap candidate (Lever 4) | (cap) |

Not duplicates, checked: the obs_spec_v1 reconstruction tests (each checks a
different part of the observation), and the bots_door tests (different
properties; they share the module fixture `recorded` where they can). The
two `_RE_PIN_1_*` pin sets in `test_turning_law.py` and `test_unit_impulse.py`
are different values (231 vs 310 ticks).

### Tests whose docstring names no property or no breaking change

A scan of the 4323 test functions in `tests/test_*.py` (function docstrings;
a module docstring counts for a 1- or 2-test module, which is how the CUDA
wrappers state their property, 54 cases):

- **1260 have no docstring.** The most are in `test_wire_tool.py` (55 of 56),
  `test_onephase_timeline.py` (38/51), `test_air_boundary.py` (33/51),
  `test_entity_selection.py` (32/34), `test_onephase_ui_model.py` (32/40),
  `test_sound_model.py` (27/27), `test_a5_seal_evacuation.py` (25/25),
  `test_onephase_control.py` (25/32), `test_b2_nodes.py` (24/24),
  `test_a6_doors.py` (22/24), `test_onephase_action_registry.py` (22/27),
  `test_b1_signal_bus.py` (21/21). Most are pre-standard (onephase / WEGO is
  legacy).
- **743 have a docstring that names no breaking change** (no "breaks",
  "fails if", "validated" or similar).
- Among the slow tests: `test_match_draws.py` x2 (63 s together; the property
  is stated, the breaking change is not), `test_bake_level_art.py` (19.5 s),
  `test_fire_o2_invariant.py` hetero (35.7 s), `test_wind_doorway_e2e.py` x4
  (30 s; the docstring is just "Property 3."), `test_drag2_venting_gate.py`
  leg 1, `test_quiet_room_drift_smoke.py`, `test_standing_unit_stays_put.py`
  x2 (no docstring), `test_sound_e2e.py` x2 (no docstring).

## 4. Levers, ranked by minutes saved

Minutes per full merge gate, Work Desktop unless named. "Now" is
`cdf8f413`; "#217" is after bots runs to 7200.

| rank | lever | now | #217 | evidence | needs |
|---:|---|---:|---:|---|---|
| 1 | **Run the card tests once.** The CPU gate runs `pytest tests -q -k "not cuda"` wherever `cuda_green.py` is the device gate (#29 already does this; the arc-close run of 2026-11-20 did not), or `pytest tests -q` skips the card tests unless asked | **64-68** | ~85-90 | A: 5991 / 6274 s with a CUDA build vs 2155 / 2191 s without; cuda_green alone 4018-4337 s | a procedure or conftest decision (it touches what "the full suite" means: Erik's call) |
| 2 | **xdist on the CPU part** (`-n 4 --dist loadgroup`, device group serial) | ~17 (-n 4), ~22 (-n 8) | same | B: 1.94x / 2.58x on the Lenovo | `pytest-xdist` in `data` on Work and Home (via `ml-stack`); fix `test_wire_tool`'s `entity_registry.json.tmp` race in the repo root and the s8a payoff ratio under load; pays only after Lever 1 |
| 3 | **One play per process in the card gate**: obs `gate`, `wall`, `books` in one subprocess (or a host-play cache keyed by build id); conductor `pinned` feeds its conductor from the live reference instead of record + replay | ~2 | ~10.5 | §2 table; D; E | a check-side change (#29) |
| 4 | **Caps on replays, keeping each asserted coverage count**: obs `wall` stops after its 6th check; `test_obs_spec_v1` planes (178 s: runs to 1200 ticks); `test_explosion` replay stops a checkpoint after the first blast; conductor `pinned` "to the end" under #217 | ~1 + ~2 + ~0.8 | ~5 + ~2 + ~1.5, and ~11 for conductor `pinned` if "to the end" is relaxed | §2; C | the end-of-match property is #29's / Erik's call |
| 5 | **Delete duplicates** (§3, rows 1-4) | ~3 | ~3 | §3 | Erik decides each |
| 6 | **Fixture sharing (CPU)**: the four `test_obs_spec_v1` reconstruction tests each play the same ship_demo seed-1 match from tick 0 (one play could feed all four cadences, ~0.9 min); `cuda_obs_check.part_decoder` plays all 8 corpus (c) runs to use one (~0.7 min, card side) | ~1.6 | ~1.6 | C; D | a test-side change |
| 7 | **xdist grouping of the pinned watchers** (one `pinned` group, so each match is played once, not once per worker) | 0 serial; up to (N-1) x 44 s CPU at -n N | up to (N-1) x ~190 s CPU and ~0.4 GB RAM per worker | D (35.9 + 7.5 s per play; 77 MiB of views at 1349 ticks) | one `xdist_group` rule in `tests/conftest.py` beside the device group |

Order of play, by return: Lever 1 is a decision, not code, and is worth more
than all the others together. Lever 2 is next and needs Lever 1. Levers 3-4
are the card gate's answer to #217 and belong to the #29 session.
