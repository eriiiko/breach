# P8.6b (part 1) handover -- 2026-11-21 (paused by Erik until Tuesday)

Branch `29-p86b-obs-orders`, stacked on `21a9c97d`.

## Done (commit 05760bb9, BUILT: both modules at that commit, CUDA build clean)
- `src/simulation/device_obs.py`: `BUILT` now covers every served column
  (`ORDER_VECTOR`, `ORDER_LIST`, far_f `route`, far_u `place`); THE ROOM GRAPH
  (device) `room_graph_values(sim, shape)` -> (has_graph, place plane =
  `obs_spec_v1.place_plane`, room grid, openings (ID, X, Y, WIDTH, HEIGHT));
  staged in `__init__`, refused / restaged in `reset_prepare` / `reset_env`
  (opening capacity); `build_orders()` called in `build()` after the windows;
  constants (`OBS_FAR_F_ROUTE`, `OBS_FAR_U_PLACE`, `OBS_ROUTE_*`, `OBS_OPEN_*`,
  `OBS_ORDER_*`), breaks `ORDER_BREAK_CODES` (ord_prev, ord_tie, ord_place) +
  `expiry_brk` (ord_expiry, read by the conductor's 10' line). Header regenerated.
- `cpp/src/cuda_obs_orders.cu` (new, in CMakeLists, bound as `bp.obs_orders`):
  `k_route` (one block per env: walkable tiles OR openable openings' rects,
  distinct (where, fp) keys in row order, per key `horde_dev::flood_block`
  seeded at passable anchors whose block centre (anchor + fp//2) is in the
  place, block minima into `route_blocks`, per row field at anchor (tile_y,
  tile_x) + compass by DESC_DY/DX parity order); `k_order_vector` (cols 84-95);
  `k_order_planes` (route / place planes). Recompute every decision, no cache.
- `cpp/src/cuda_obs_list.cu`: `order_tail` moved above `k_list` and called after
  `mate_entry` (list cols 47-51); `obs_list` takes the place table. Door hull
  test lifted to `obs_rows::door_hull`, `row_unit` lifted to `cuda_obs_rows.cuh`.
- `tests/cuda_obs_check.py` PATCH = "P8.6b"; `tests/test_device_obs.py`: BUILT
  covers the whole spec, room-graph table tests (22 passed on CPU).

## Half done (wip commit, NOT run since): tests/cuda_obs_check.py
- order hand match (`corpus_orders`, `_order_scene`, `order_setup`,
  `order_script`), `count_orders` / `judge_orders` / `ORDER_CASES`, part
  `orders`, gate break hooks + `order_faults` check, part GATE runs it.
- First run of part `orders` (at 05760bb9 + this check): order_hand 41 and
  list_hand 7 env-decisions ALL EQUAL raw bits on the first run. Failing only
  on COVERAGE: list `order_same_place` and vector `compass_dx` constant over the
  two hand matches; `route_1` and `route_changed` never reached.
- WHY (found with a host debug script): the scene's door D is the ONLY opening
  of room `ra` where A and C stand, so after making D a hull door everything
  from ra is unreachable; and A's `use_intent` at tick 41 did NOT open D (door
  state stays 0; probably the use picks no door / A not in reach at anchor
  (7,19) with fp 3, or a hull door). Fix the scene: pick a door whose near room
  has another exit, place A so `verb_gates.use_door` returns D, and give two
  mates the same place (order_same_place) and a soldier moving sideways
  (compass_dx). Corpus (c) (commander on) will add coverage in part GATE.

## Learned (how the host computes it)
- `obs_spec_v1.route`: `horde.flood` over walkable | rects of openable openings
  (`room_graph.opening_states`: not passable AND every unwalkable tile in a
  living CLOSED, unwired, non-hull door span); seeds = passable anchors whose
  centre (anchor + fp//2, NO_PLACE past h-half / w-half) is in the place
  (`place_plane`: room, else lowest-id opening). Cache key = tile serial +
  openable flags, i.e. everything the field reads -> recompute == cache.
- A CLOSED ordinary door is openable, so closing a door never changes a route;
  only a wall destroyed, a hull / wired door, or a door destroyed/opened that
  was not openable does. Compass = `horde.candidates` at (tile_y, tile_x), not
  the centre; place inside / place centre use the centre tile.
- Unreachable: vector 94 = -1, compass 0 0, route plane all -1, place plane
  still drawn.
- Orders are store columns on the card already (written 1b', expired 10');
  nothing of the order rows is host-only. Not on the card: nothing found.

## Left
- Fix the order scene (above), run part `orders`, then the short gate
  (`test_cuda_obs.py` etc., quick CPU tests), the four breaks seen red
  (ord_prev, ord_tie, ord_place, ord_expiry), WALL extension (latch orders via
  the tap at tick 0 in part_wall, add vector 84-95 + `build_orders` to
  team_aims), reset stats (order rows after a reset, old order gone).
- Rules (`.claude/rules/rl-agent.md` Device observation; index line "Room graph
  (device)"), design as-built paragraph, rl-ready-lead skill run-log line, twin
  re-mark, bench (`tools/bench_conductor.py --obs` with orders on every
  soldier), the two commander measurements (NOT taken yet), final rebuild.
