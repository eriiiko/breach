# P8.4 notes (handover between the three agents of branch `29-p84-obs-list`)

Part c deletes this file before merging, folding what still matters into the
design's dated "P8.4 as-built" paragraph (after P8.3's in
`docs/residency_batching_design_2026-11-07.md`).

## Part a (P8.4a): lanes, aim_of, line of fire, ticks_to_within -- DONE

### What exists (all on the card, every env on its decision tick, ended envs skipped)

`DeviceObservation.aims(store_ptr=None)` (called by `build()` after the
picture vector; one call, five launches on the tick stream, no sync) over
`bp.obs_aims` (`cpp/src/cuda_obs.cu`, declared in `cpp/src/cuda_obs.h`).
Buffers on `DeviceObservation` (U = the store's rows, T = the layout's marine
teams, n_rows = T * S):

| attribute | shape, dtype | meaning |
|---|---|---|
| `lane_grid` | (N, T, GH, GW) int32 | `TeamKnowledge.lanes` of team t: -1 nobody, an owner's uid, -2 several (draw_lanes' codes). `lane_shape` = (GH, GW) = (h // cell + 1, w // cell + 1) |
| `lane_own` | (N, U, GH, GW) uint8 | 1 where store row q's OWN lane is drawn: `TeamKnowledge.lane_of(uid_q)[cell] == uid_q`. So list col 52 `me_in_its_lane` = `lane_own[e, row(mate), cy // cell, cx // cell]`, col 53 `it_in_my_lane` = `lane_own[e, row(me), my // cell, mx // cell]` (cell = obs table `OBS_TAB_LANE_CELL`) |
| `aim` | (N, T, U) int32 | `aim_of(know of team t, unit of row q)`'s uid (`OBS_AIM_NONE` = -1 nobody) for every mate of t and every unit t sees now with a record; `OBS_AIM_NOT_ASKED` (-2) elsewhere. The category is the reader's: 0 if -1, 1 if == me's uid, 2 if a mate of t, else 3 |
| `lof` | (N, n_rows, U) uint8 | for the soldier of row r and every unit q its team sees with a record: `OBS_LOF_ASKED` (1) \| `OBS_LOF_SOLID` (2: `solid_in_way` me -> q's RECORDED centre, col 25) \| `OBS_LOF_MATE` (4: `mate_in_way`, col 26); 0 where not asked |
| `ttw` | (N, n_rows, U) int32 | `turning.ticks_to_within` of `_enemy_geometry` (col 27's ticks, before `squash(ratio(ticks, half_turn))`), -1 where not asked |
| `aim_faults` | (N,) int32 | `OBS_AIM_FAULT_GRID` / `RANGE` / `LAW` (read by the gate; no served level reaches one; NOT on the books read) |

The served plane: `near_u` plane `lanes` (`OBS_NEAR_U_LANES`), written by
`k_lanes_plane` for every row (0 for a row that is not valid).

Device helpers (for the list / full view to reuse, never copy):
`cpp/src/cuda_obs_sight.cuh` (namespace `obs_sight`): `lane_div` (the floor
division, `trunc` = the break), `lane_segment`, `lane_steps`, `lane_cell`,
`lane_claim` (the atomic cell rule), `mate_near_segment`, `AimKey` /
`aim_less` (aim_of's (d2, uid, cx, cy) order). In `cpp/src/cuda_obs.cu`
(anonymous namespace): `is_mate(row, t)`, `is_seen(vis_t, blk, L, U, t, q)`
(TeamKnowledge.seen = visible[t] AND a record), `aim_cone` (`_cone_of`),
`bearing_q16` (atan2_q16(-dy << 16, dx << 16) with the int32 range check),
`ticks_to_within` (iterates `unit_lib::turn_law_q16`). If part b's list
kernel lives in another TU it needs `is_mate` / `is_seen` there: move them
into a header then, do not copy.

Obs table: two new scalar rows `LANE_CELL`, `LANE_HALFWIDTH`
(`device_obs.lane_values()`, read through `TeamKnowledge().lane_dials`).
New generated constants: `OBS_NEAR_U_LANES`, `OBS_AIM_NONE`,
`OBS_AIM_NOT_ASKED`, `OBS_LOF_ASKED/SOLID/MATE`, `OBS_AIM_FAULT_*`,
`OBS_AIM_BREAK_*` (`device_obs.AIM_CODES`, `AIM_FAULT_BITS`,
`AIM_BREAK_CODES`).

### The check (`tests/cuda_obs_check.py`, PATCH = "P8.4a")

- `AimTap`: a test-side tap that wraps, for the duration of one
  `observe_rows`, `obs_spec_v1.build_list` (the soldier), `aim_of`,
  `line_of_fire.solid_in_way` / `mate_in_way`, `_enemy_geometry` +
  `turning.ticks_to_within`, `TeamKnowledge.lanes` / `lane_of`, and restores
  them. Records per decision tick in `Played.aims[tick]`; coverage in
  `Played.aim_cov` (`AIM_CASES`, each > 0 else FAIL, judged by part GATE).
  Part b can keep using it: once the list columns 25-27, 31-35, 41, 52-53
  are owned and compared as served columns the tap stays as the finer gate.
- `aims_equal` (every decision tick of every GATE / BATCH run), `judge_aims`.
- Breaks: `lane_trunc`, `lane_wide` (via `obs.aims_brk`).
- PART WALL extended (`team_aims`): the unseen enemy's row perturbed (trigger
  flipped too) moves none of the team's P8.4a answers.

### Part a's gate evidence (2026-11-21, for the as-built paragraph)

- Short gate `pytest tests/test_cuda_obs.py tests/test_device_obs.py
  tests/test_device_picture.py -q`: 25 passed (385 s) at `3ddf5aa9`.
- Part GATE (N = 1, raw bits, every decision tick; the lanes plane served):
  bots 338, bots_v1 69, range_door 51, hand 81, range_hall 9, hall_cover 7,
  ship_all_round 121, door_grenade 38, slow_turn 11 env-decisions. The
  intermediates equal to the tap's values: bots 676 team lanes / 2126
  aim_of / 1098 lines of fire / 1100 ticks; bots_v1 137 / 850 / 918 / 1168;
  ship_all_round 242 / 1155 / 748 / 748; (2.0 M + 0.5 M + 1.0 M values).
  Coverage: lane_crosses_mate 108, lane_several 272, lof_wall 792, lof_cover
  2, lof_mate 1983, lof_clear 1889, aim_none 3532, aim_mate 786, aim_other
  705, ttw_zero 1033, ttw_between 2094, ttw_cap 11. Census near_u/lanes
  non-constant (22 214 non-default cells).
- Batches: ship_demo N = 2 / 4 (493 / 805 env-decisions), room N = 2 / 4
  (112 / 220). Syncs 2.00 per steady tick at N = 1 and 2.
- WALL extended: cone 7, all_round 6 cases, unchanged.
- Breaks, each red at its first decision: `lane_trunc` (bots tick 420, team
  1's grid cell (5, 15): -1 vs 4; also bots_v1 188, range_door 188,
  ship_all_round 424); `lane_wide` (bots tick 424, cell (5, 16): 4 vs -1;
  bots_v1 188, range_door 188, ship_all_round 424). Both are caught first by
  the team-grid comparison, which runs before the served-plane comparison in
  the same tick.
- New corpus (d) match `slow_turn` (`corpus_turn`): range_door, idle, unit 1
  facing away from unit 0 with hook TURN = ONE/8: the only source of the
  ticks_to_within cap (no played match puts a seen enemy past 4 half turns,
  even with every marine slowed: measured on ship_all_round).

### What part b must know

- The list's sort keys and the record centres: the picture block's REC
  (`obs_pic::rec_of`, `OBS_PIC_REC_CENTRE_X/Y`, `UID`, `PRESENT`).
- `aim` answers are per TEAM VIEW (the hit does not depend on the observing
  soldier); col 31 `facing_on_me` = (hit == me uid); col 32 the category;
  cols 34-35 count the other mates m (m != me) with `aim[e, t, row(m)] ==
  euid` (and their trigger); col 41 `aim_is_unit` = `aim[e, t, row(mate)]
  != -1`.
- The host computes the per-enemy columns only for the first caps[1] (16)
  seen enemies after the sort; the device buffers hold every seen enemy.

## Part b (P8.4b): (to be written by part b)

## Part c (P8.4c): (to be written by part c)
