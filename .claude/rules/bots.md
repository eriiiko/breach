---
paths:
  - "src/control_bots.py"
  - "src/control_bots_v1.py"
  - "src/control_bots_door.py"
  - "tools/bot_tournament.py"
  - "tools/bench_bot_matches.py"
---

# Bots

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## Bot layers
- A new bot behaviour is a scored ACTION (an `ACTIONS` row with a `[bots.actions]` weight), a PLAY (a `PLAYS` row with a `[bots.plays]` weight) or a POSITION FEATURE: never a fixed-order rule, a branch in the executor or a second controller.
- The TEAM layer reads the team's records off the sim's team picture (it owns none) and the fire lanes from state (`team_picture.fire_lanes`). A bot COMMITS to its position until its next decision unless beaten by `commit_margin` (#138); it opens a closed non-hull, unwired door on its path with `use_intent` (#142).
- A* runs only where the flood shows a way of at most `plan_max_steps`, one search per team per tick: never a search that cannot succeed (the multi-ms spike).
- Bots never call `reload_intent` (RULED). Deterministic: integer scores, ties by row order and `tie_hash` (a hash, never a drawn stream); host Python, renderer-free. The weights are tuned only by the Bot tournament.
- The TEAM layer doubles as the COMMANDER: `team_order` / `issue_team_order` are module functions that `BotCommander` and the door bots both call, never copies. `PLAY_ORDER` has a row per `PLAYS` row (refused at import when missing).

## Position features
- A tile is judged ONLY through named yes / no features, each scored by a `[bots.features]` weight; a learned scorer reads the same list. A new consideration is a feature row, never a term in the score.
- A feature reads the sim through the existing seams (`vision.ray_clear`, `horde.flood` as a measurement, `EntityFieldAccessor.sample_many`), never a raw plane. Line-of-sight features are cast only for the `los_budget` best candidates.

## Bots through the door
- The layered bots driven through action spec v1, the imitation teacher: an inner `BotTeam` decides UNCHANGED on a capture proxy (nothing reaches the sim), and an inner `AgentPolicy` decoder is the team's ONE issuer.
- Each bot decision is translated to the nearest LEGAL choice through the action spec's helpers, never a second table; a masked or out-of-range choice RAISES `DoorRefused` before it is sent. Latches between decisions wait for the next one.
- A variant is a `DOOR_VARIANTS` row, never a table. The team writes its own brain's orders, so it is not in `DOOR_CONTROLLERS`.
- With a `RecordSink` installed, each door team writes ONE `.npz` per match beside the match's replay; without one no reward book runs.

## Bot tournament
- Through `run_match`, never a second loop. A change is judged on a TUNING seed set and reported on a disjoint HELD-OUT one, never by an opinion. `--set key=value` tries a `[bots]` value without editing config.toml. `--records DIR` is refused inside the checkout.
