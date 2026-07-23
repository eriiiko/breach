# Turn System & Control

**Depends on:** [State & Ownership](../engine/02_state_and_ownership.md)

This chapter defines how time advances in Breach and how a player drives it. It
covers the round/phase clock, the planning-then-execution rhythm, the
real-time-with-pause execution model, order placement and undo, and the path
from a mouse click to a queued order. It also lays out the realised direction:
the deterministic `Simulation` is an **engine**, and a *game* is a turn structure
plus an input scheme layered on top of it — so the same engine hosts both the
tactical two-phase game and a real-time single-character action variant, chosen
at construction. Both are shipped: as of 2026-07-23 the turn structure is a
swappable `Ruleset` and the input scheme a swappable `ControlSource`.

The contract this chapter builds on is the one defined in the state chapter:
`Simulation` owns the world (units, orders, projectiles, RNG, the turn clock) and
exposes exactly `apply_action` / `step` / `get_state`. Everything here is either
inside that facade or sits cleanly above it.

---

## What the system is

Time in Breach is **discrete and tick-based**. The simulation never thinks in
wall-clock seconds; it thinks in ticks. One `Simulation.step()` advances exactly
one tick. Wall-clock time exists only in `main.py`, which accumulates real elapsed
time and converts it into a whole number of `step()` calls. This separation is
deliberate and load-bearing: it is what makes the engine deterministic and
headless. An AI training loop calls `step()` as fast as it can; a human plays at
`ticks_per_second`; both run the identical simulation.

The tick is organised into a fixed hierarchy:

| Unit | Config key | Default | Meaning |
|---|---|---|---|
| Tick | `clock.ticks_per_second` | 24 Hz | The atomic simulation step (41.7 ms of game time). |
| Phase | `clock.phase_duration_seconds` | 5.0 s | A planning sub-unit. `ticks_per_phase = 120` (derived). |
| Round | `clock.phases_per_round` | 2 | A full turn. `ticks_per_round = 240` (derived). |
| AP | `clock.ap_per_phase` | 2 | Action points each unit gets per phase, for AP-costed orders. |

`ticks_per_phase` and `ticks_per_round` are computed once at config load
(`config.py`) and cached on the `Simulation` at construction so a `reset()` never
re-reads `CFG`. The defaults give a round of **2 phases x 120 ticks = 240 ticks =
10 seconds** of game time.

### Why ticks, not seconds

Three properties fall out of a pure tick clock, and all three matter:

- **Determinism.** Given a seed and a sequence of `apply_action` / `step` calls,
  the trajectory is reproducible. There is no dependence on frame rate or
  wall-clock jitter. (The RNG plumbing that backs this lives in the state
  chapter; the turn system simply guarantees the *order* of operations within a
  tick is fixed.)
- **Headless rollouts.** Training never touches the renderer or the clock in
  `main.py`. It constructs a `Simulation`, never calls `set_paused`, and loops
  `step()` until `is_terminal()`.
- **Speed-independence.** Playback speed (slow-mo, fast-forward) is purely a
  presentation concern: `main.py` chooses how many `step()` calls to issue per
  real second. The simulation is bit-identical regardless.

---

## The two-phase round

The default game is tactical and turn-structured. This turn structure is now one
**ruleset** — `TwoPhaseWEGO` (`src/simulation/ruleset.py`), the strategy object the
`Simulation` defers to for round clock, AP, and terminal condition (see *Direction*
below). It ships as the default; the action variant `ContinuousRealtime` is its
sibling. What follows describes `TwoPhaseWEGO`. A round runs in two states:

```
PLANNING  --(Space)-->  EXECUTING  --(240 ticks elapsed)-->  PLANNING (next round)
```

**Planning.** The simulation is paused (`tick` frozen). The player assigns orders
to marines for the round. Orders are tagged with the phase they belong to: the
player plans **both phases up front** — Phase 1 (preparation) and Phase 2
(engagement) — toggling which phase the next order targets with Tab. Movement
orders are free; grenade, fire, and door-explosive orders cost AP, drawn from the
targeted phase's AP pool.

**Execution.** Spacebar resumes. Time flows at `ticks_per_second`. The round plays
through **both phases smoothly in one continuous run** — Phase 1 then Phase 2,
like a movie. There is no forced stop at the phase boundary. The phase split is a
*planning aid* for the player (two windows to think about), not a simulation
interruption. At the end of the round (tick 240) the simulation auto-pauses and
returns to planning for the next round.

This is the key design choice and it is worth stating plainly: **the phase
boundary is a mental model, not a halt.** The simulation crosses tick 120 without
pausing, fires the between-phases door explosives, advances the phase counter, and
keeps going. Earlier designs paused at the phase boundary; the shipped engine does
not, because a continuous 10-second execution reads better and keeps the
two-phase structure as pure planning scaffolding.

### Execution tick order

Each executing `step()` runs a fixed sequence. The ordering is load-bearing — it
determines, for example, that units move *before* shooting resolves and that unit
positions are stamped into the obstacle grid *before* physics runs (so
decompression sees bodies as obstacles):

```
step():                                     # no-op if paused
  0. (tick 0 only) fire DET_START_PHASE1 door explosives;
     reset path offsets; stamp initial unit positions
  1. clear tick_events                      # fresh per-tick signal buffer
  2. update projectiles                     # advance, detonate on fuse
  3. update player movement                 # read precomputed move_path
  4. process shooting                       # fire orders + move&attack auto-fire
  5. zombie AI                              # activation, pathing, melee
  6. stamp units into obstacles             # physics must see new positions
  7. physics step                           # atmosphere, smoke, fire, temperature
  8. burn-through walls                      # fire-destroyed tiles -> events
  9. recorder snapshot
 10. advance tick + real_time
 11. boundary checks:
       tick == ticks_per_phase  -> fire DET_BETWEEN_PHASES; phase = 1 (no pause)
       tick >= ticks_per_round  -> fire DET_END_PHASE2; end-of-round; pause
```

`tick_events` is the one-way channel from sim to renderer for transient signals (a
shot fired, a unit hit, an explosion, a wall destroyed). It is cleared at the top
of every `step()`, so the renderer **must** consume it each frame before the next
step overwrites it. The simulation never tracks fade or decay — that is the
renderer's effect queue. See the events module for the dataclasses.

### End of round

At tick 240 the round tears down (`_end_round`): marines killed by zombies convert
to the zombie team, float positions snap to integer tiles, orders clear, AP
refills to `ap_per_phase` for both phases, undetonated long-fuse projectiles carry
over, the obstacle grid resets to walls-only (so dead bodies stop blocking
physics), the tick rewinds to 0, the phase resets to 0, and the round counter
increments. The simulation then sets `paused = True`, dropping the player back
into planning.

---

## Real-time with pause

Execution is **real-time with scheduled and on-demand pauses**, and the mechanism
is intentionally minimal:

> **Pause = `main.py` skips `step()`.** Nothing more.

`set_paused(True)` makes `step()` an early-returning no-op; the tick counter does
not advance. The renderer keeps drawing the frozen frame. `set_paused(False)`
resumes. There are exactly three ways pause is entered or left:

| Trigger | Who | Effect |
|---|---|---|
| Spacebar | player (input handler) | Toggle pause at any moment during execution. |
| End of round (tick 240) | simulation | Auto-pause; return to planning. |
| Game construction / `reset()` | simulation | Starts paused — the player plans first. |

The phase boundary (tick 120) deliberately does **not** auto-pause in the shipped
build; the round runs straight through.

**AI ignores pause entirely.** A training rollout never calls `set_paused`, so
`step()` is never a no-op for it. Because pause is implemented as "the caller
chooses not to step," there is nothing for the AI path to special-case — it simply
loops `step()`.

**Order editing during pause.** Order placement is gated on `sim.is_paused()` in
the input handler: clicks and mode hotkeys are only honoured while paused. In the
shipping flow the player plans the whole round, then runs it; mid-execution
pause-and-replan is a natural extension of the same gate (the input handler
already allows `apply_action` whenever paused), but the interaction of late edits
with in-flight projectiles and partially-walked paths is not fully specified —
treat anything beyond "pause, then plan the next round" as forward design.

### main.py's clock

The real-time conversion lives entirely in the main loop:

```python
sim_time_per_tick = 1.0 / CFG.clock.ticks_per_second
tick_accum += dt                       # dt = real seconds since last frame
while tick_accum >= sim_time_per_tick and steps < max_catch_up:
    sim.step()
    tick_accum -= sim_time_per_tick
    if sim.is_paused():                # sim may auto-pause mid-batch
        break
```

The catch-up cap (`max_catch_up = 5`) prevents a death spiral if a frame stalls:
at most five ticks are consumed per frame, and the loop breaks immediately if the
simulation auto-pauses partway through the batch. This is the only place
wall-clock time enters the engine.

---

## Orders

An **order** is a queued intent attached to a unit. Orders are plain data, not
behaviour — they are interpreted by the tick loop, not executed by themselves. A
single `Order` class carries a type discriminator plus optional payload, rather
than a subclass hierarchy:

| Field | Meaning |
|---|---|
| `order_type` | One of move-attack / move-cover / sprint / grenade / explosive / fire. |
| `target_fx`, `target_fy` | Target tile (fine grid). |
| `phase` | Which phase the order belongs to (0 or 1). |
| `grenade_fuse` | Seconds; grenade orders only. |
| `det_slot` | Detonation slot; door-explosive orders only. |
| `ap_cost` | Defaults to 1; patched to 0 for movement by the placement code. |

The three movement types (Move & Attack, Move w/ Cover, Sprint) are not cosmetic —
they encode **different speeds**, looked up as ticks-per-tile (sprint fastest,
move-attack slowest). Move & Attack additionally enables auto-fire at the nearest
visible enemy during the walk.

### Placement and validation

Orders enter through `Simulation.apply_action(unit_id, order)`, which validates and
either accepts (returns `True`) or rejects (`False`):

- **Living, player-controlled unit.** Zombies take no player orders; dead units
  take none. `apply_action` is the single chokepoint that enforces "zombies don't
  get to pick orders" — the rule lives in one place, not scattered across order
  types.
- **Movement** validates the target is passable for the unit's full footprint
  (`is_passable_block`), costs no AP, and triggers an immediate path recompute so
  the planning overlay and execution share one trajectory.
- **Grenade / explosive / fire** check AP against the targeted phase's pool (and
  inventory count for grenade/explosive), spend it, and decrement inventory.

A rejected order leaves the unit untouched; the caller (input layer) can surface
the failure as a UI toast. `get_legal_actions(unit_id)` is the planned
AI-facing enumeration of valid orders; today it returns `[]`, and callers use
"did `apply_action` return `True`?" as the legality check.

### Undo

`undo_last_order(unit_id)` pops the most recent order and refunds it: AP returns to
the order's phase pool, grenade/explosive inventory is restored, and a movement
undo recomputes the path. Bound to Backspace. This is a clean inverse of
`apply_action` and the reason placement records `ap_cost` on the order itself —
the refund needs no rule lookup, it just reads the cost back off the order.

### Paths: precomputed and per-tick

Movement is resolved by precomputing each marine's **tick-by-tick trajectory** for
the whole round at order time, then replaying it one position per tick during
execution. The two views are the same data at different granularities:

- **At order time** (`_compute_player_paths`): for each phase, A* over the unit's
  movement waypoints produces a tile path; each tile segment is subdivided into
  `speed` interpolated positions (ticks-per-tile), padded with the final position
  to fill the phase. The result, `unit.move_path`, is a per-tick position list
  spanning both phases.
- **At execution time** (`_update_player_movement`): each tick reads
  `move_path[tick - path_tick_offset]`, sets the unit's position, and faces it
  toward the next step.

Precomputing serves the planning overlay (the renderer draws the planned route);
replaying per-tick serves movement and keeps it deterministic. `orders_for_phase`
exposes per-unit waypoint lists for the overlay, with the first waypoint being the
unit's position at the *start* of that phase (current position for Phase 1, the
planned end of Phase 1 for Phase 2).

---

## Input and control

Input is a thin presentation layer (`WEGOPlanningInput` in `src/input_handler.py` —
the former `InputHandler`, renamed when the `ControlSource` family landed; the old
name is kept as an alias) that translates pyray polling into `apply_action` /
`set_paused` / `undo_last_order` calls. It owns only
UI state — the selected unit, the current order-placement mode, the grenade fuse,
the detonation slot, and the per-unit planning phase — and never tells the
simulation about any of it. Keeping input out of the renderer keeps the renderer
swappable (it draws; it does not interpret intent).

| Binding | Action |
|---|---|
| Left-click unit | Select marine. |
| Left/right-click tile | Place an order of the current mode (selection required). |
| 1 / 2 / 3 | Move & Attack / Move w/ Cover / Sprint mode. |
| F / G / B | Fire / Grenade / Door-explosive mode. |
| Scroll wheel | Grenade fuse (G mode) or detonation slot (B mode). |
| Tab | Toggle the *selected* unit's planning phase (Phase 1 <-> Phase 2). |
| Spacebar | Toggle pause (resume / pause execution). |
| Backspace | Undo last order on the selected unit. |
| Ctrl+R | Reload `config.toml` from disk. |
| Esc | Clear selection, then cancel mode. |
| F8 | Manual physics-recorder dump. |

Two details reflect real decisions in the code. First, the planning phase is
remembered **per unit**: switching between marines preserves where you left off,
so Tab toggles only the currently-selected unit's phase. Second, mode hotkeys and
order-placement clicks are gated on `sim.is_paused()` — they do nothing during
execution — which is why a key like B can mean "explosive mode" while planning and
remain free for a renderer toggle while running. When resuming on the round's first
tick, the input handler materialises any queued grenade orders into in-flight
projectiles (`spawn_projectiles_from_grenade_orders`) before time starts flowing.

Config hot-reload is Ctrl+R, not F5: F5 stays the renderer's normal-map toggle.
This split avoids a key collision between gameplay config reload and a render
debug toggle.

---

## Direction: engine vs. game — swappable rulesets and control modes

The two-phase round and the click-to-plan input scheme described above are **one
game**, not the engine. The engine is the deterministic `Simulation` and its
`apply_action` / `step` / `get_state` contract. A *game* is two things layered on
that contract, and as of 2026-07-23 **both are pluggable strategy objects**, not
baked-in logic:

- A **ruleset** — the turn structure. How long is a round? Are there phases? When
  do orders resolve? When does control return to the player? This is now a
  `Ruleset` strategy object (`src/simulation/ruleset.py`) that the `Simulation`
  owns and defers to, chosen at construction.
- A **control source** — the input scheme. How does intent become orders or
  intents? Plan a whole round up front, or drive a single character directly? This
  is now a `ControlSource` (`src/control_source.py`), a sibling of the facade
  selected at startup.

**The `Ruleset` object.** `Simulation` holds one `Ruleset` and calls it at the
seams that used to be hard-coded: `on_round_start(sim)` / `on_tick_end(sim)` for
clock and phase bookkeeping, `validate_and_cost(sim, unit, order)` / `refund(sim,
unit, order)` for order admission and undo, and `is_terminal(sim)` for the end
condition. Two implementations ship:

- **`TwoPhaseWEGO`** — the shipped two-phase behavior, extracted verbatim: the
  round clock head/tail, per-phase AP check-and-spend, the `_end_round` teardown,
  and the round-complete terminal. It is byte-identical to the pre-refactor engine
  (the digest and golden gates never moved).
- **`ContinuousRealtime`** — the action variant. No phases, no AP
  (`validate_and_cost` checks only *alive + physical preconditions*), no auto-pause,
  no round teardown and no tick-rewind. Zombie conversion is death-triggered — it
  runs each tick and is idempotent, rather than being swept up at an end-of-round
  boundary that no longer exists. `is_terminal` fires when one team is eliminated.
  On corpses: in the current soft-body stamp engine dead units block nothing and
  only walls hard-block, so `ContinuousRealtime` simply omits the WEGO
  end-of-round obstacle reset — no regression, because there is nothing to reset.

**The `ControlSource` family.** `main.py` selects one by a `--control` flag
(default `wego`; values `wego` | `gamepad`) through `create_control_source(name)`:

- **`WEGOPlanningInput`** — the old `InputHandler`, renamed (alias kept), behavior
  unchanged: click-to-plan a whole round while paused.
- **`GamepadDirect`** (`src/control_gamepad.py`) — possesses the first team-0 unit,
  polls the pyray raylib gamepad, and emits **per-tick intents**. It selects
  `ContinuousRealtime` and starts unpaused. Stick floats are quantized to
  fixed-point *at the control seam* (`quantize_stick_direction`) so only
  fixed-point ever crosses into the sim — the determinism law holds at the input
  boundary, not by trusting the driver.

**Per-tick intents** (`src/simulation/intents.py`) are the action-variant's order
analog: they are consumed the tick they are issued (WEGO keeps its queued
tile-orders untouched). The set is `MOVE_DIR` (a Q16.16 unit-vector direction +
speed_mode), `AIM` (a Q16.16 direction → `Unit.facing` via the integer atan2 kit),
`TRIGGER` (held), `THROW` (direction + fuse), and `USE` (edge-triggered). They are
consumed in `Simulation._consume_direct_intents`; `MOVE_DIR` drives a new
`_step_move_dir` branch that reuses `is_passable_block` + the mobility table — the
same predicates A* uses for planned movement. The whole path is **dormant under
WEGO**: with no possessed units, `_consume_direct_intents` short-circuits.

Free-aim directional shooting rides on top of this (details in
mechanics/03 §"Directional (free-aim) fire"): a possessed unit's held `TRIGGER`
fires along its `facing`, pure free-aim, hitting the first thing the march
actually crosses.

The boundary this preserves: **the ruleset and control source live above or beside
the facade, never inside the field state.** World state stays arrays plus a
material table; `step()` stays a single-tick advance; determinism and headless
rollouts survive any control mode, because training is itself just a control source
that never pauses — and the `ControlSource` base plus `create_control_source`
factory is where an AI-driving source will slot in.

---

## Implementation status

**Built and shipping:**

- The full tick / phase / round clock, derived from config
  (`ticks_per_second` 24, `phases_per_round` 2, `phase_duration_seconds` 5.0 →
  120 ticks/phase, 240 ticks/round), cached on the `Simulation`.
- The planning → executing → planning round flow, with the fixed execution tick
  order exactly as listed, in `Simulation.step`.
- **Continuous two-phase execution**: the round plays through both phases without
  pausing at the phase boundary; the phase counter advances and
  DET_BETWEEN_PHASES explosives fire at tick 120, DET_START_PHASE1 at tick 0,
  DET_END_PHASE2 at round end.
- Real-time-with-pause via `set_paused` / `is_paused` (pause = `main.py` skips
  `step()`), the `main.py` real-time accumulator with catch-up cap, and auto-pause
  at round end. Start-paused on construction and `reset`.
- The single `Order` data class with type discriminator + payload; `apply_action`
  with full validation (alive, non-zombie, footprint-passable movement, AP and
  inventory checks); `undo_last_order` with AP/inventory refund.
- Precomputed-and-per-tick movement: `_compute_player_paths` (A* + ticks-per-tile
  interpolation across both phases) and `_update_player_movement`;
  `orders_for_phase` for the overlay. Movement-speed variety across the three move
  modes is wired through `_ticks_per_tile`.
- The `WEGOPlanningInput` control source (formerly `InputHandler`, alias kept)
  with the full binding set above, per-unit planning phase, pause-gated order
  placement, and Ctrl+R config reload (F5 left to the renderer).
- **The ruleset / control-source abstraction** (2026-07-23, human-tested and
  blessed; WEGO byte-identical): the `Ruleset` strategy object owned by
  `Simulation` (`TwoPhaseWEGO` = the shipped behavior extracted verbatim;
  `ContinuousRealtime` = the action variant, no phases/AP/auto-pause/teardown,
  death-triggered zombie conversion, one-team-eliminated terminal), and the
  `ControlSource` family (`src/control_source.py`) selected by `main.py`'s
  `--control` flag via `create_control_source` — `WEGOPlanningInput` (default) and
  `GamepadDirect` (`src/control_gamepad.py`, possesses the first team-0 unit, polls
  the raylib gamepad, selects `ContinuousRealtime`, starts unpaused).
- **Per-tick intents** (`src/simulation/intents.py`): `MOVE_DIR` / `AIM` /
  `TRIGGER` / `THROW` / `USE`, consumed the tick issued in
  `Simulation._consume_direct_intents`; `MOVE_DIR` drives `_step_move_dir` off the
  same `is_passable_block` + mobility table A* uses. Stick floats quantized at the
  control seam (`quantize_stick_direction`) — fixed-point only into the sim.
  Dormant under WEGO (no possessed units).
- **Free-aim directional shooting** (mechanics/03 §"Directional (free-aim) fire"):
  a possessed unit's held `TRIGGER` fires along `facing`, pure free-aim, hitting
  the first thing the march crosses; targeted (WEGO) fire is byte-identical.
- `tick_events` as the per-tick sim→renderer signal channel, cleared each step and
  consumed by the renderer.
- AI-rollout scaffolding present: `reset`, `is_terminal` (round complete or a side
  wiped out), and the headless `step()` loop that ignores pause.

**Designed but not yet built:**

- **`get_legal_actions`** returns `[]` — a stub. Legality today is "did
  `apply_action` return `True`?". The full per-(mode × phase × tile) enumeration
  the AI contract wants is not implemented.
- **`get_reward`** returns `0.0` — a placeholder for the training contract.
- **Mid-execution replanning** is permitted by the input gate (placement works
  whenever paused) but its interaction with in-flight projectiles and
  partially-walked paths is unspecified; the shipped flow is plan-whole-round then
  run.

**Deferred — action-variant follow-ups (designed direction, not built):**

- **Flamer / spray "held continuous stream" feel.** The spray archetype fires a
  fixed burst with aim latched for its duration; under direct control you cannot
  re-aim mid-burst. Intended feel is a continuous stream held as long as ammo
  lasts and swept freely. Its own session (see `docs/TODO.md`).
- **The keyboard + mouse direct control variant** (the P4 source): only
  `GamepadDirect` exists today on the direct side; a mouse-aim + WASD `ControlSource`
  is designed but unbuilt.
- **Grenade button remap** on the gamepad and a dedicated `USE`-button interaction
  pass are polish items on the action variant.
- **The door ↔ unit occupancy engine-fix (rule A + B).** Continuous control makes
  the "dead units block nothing / only walls hard-block" simplification more
  visible; a proper occupancy model for doors-vs-units is future engine work.
- **Melee / block / parry / grab + a stamina system.** The action-combat verbs
  Erik sketched in the *Second Thoughts* note below remain deferred future work
  with their own arc — see the editor's resolution note there.

**Gaps and known issues:**

- **Turn/phase flow has an open regression** flagged at the last playtest before
  the legacy entry point was retired: a not-fully-characterised problem in phase
  transition, pause-release timing, or order resolution at tick boundaries.
  Symptom was not captured precisely; it must be reproduced and diagnosed against
  the legacy reference. Until then, treat the round flow as functional-but-suspect
  at the tick-boundary edges.
- **Playback speed** (the slow-mo / fast-forward control described in the older
  architecture doc) is not wired in the current `main.py`; the loop runs at one
  fixed `ticks_per_second`. Speed scaling is a pure presentation change when added.
- `spawn_projectiles_from_grenade_orders` only fires on the first tick of a round
  (guarded by `sim.tick == 0` in the input handler), so grenades queued during a
  mid-round pause would not materialise on resume — consistent with the
  plan-whole-round flow, but a constraint to revisit if mid-execution replanning
  is formalised.
- Spawn placement, sprite assignment, and zombie variant spawning are
  level/content concerns that intersect the turn flow (units must exist before a
  round runs); these are tracked outside this chapter and were among the items
  reconciled before the legacy entry point was deleted.







> **Editor's resolution note (2026-07-24).** The *modular-control* direction Erik
> reaches for in the note below is now built and shipped. The control scheme became
> modular exactly as he hoped: the swappable `Ruleset` + `ControlSource` split
> (P1–P3) plus free-aim directional shooting merged to `main` on 2026-07-23,
> human-tested and blessed ("i felt joy shooting zombies"), WEGO byte-identical
> throughout — see the as-built *Direction: engine vs. game* section above and the
> archived design docs (`docs/archive/control_modularity_design_2026-07-22.md`,
> `docs/archive/free_aim_shooting_design_2026-07-23.md`). The v0.1 "just control one
> character with a gamepad, move on the left stick, aim on the right, shoot on the
> triggers" idea is the shipped `GamepadDirect` + `ContinuousRealtime`. What remains
> **deferred** is the *action-combat verb set* — melee / block / parry / grab and a
> stamina system (the Rooms-of-Many-Rooms delayed-stamina idea) — which stays future
> work with its own arc (listed under *Deferred* above). Erik's original text is
> preserved verbatim below.

  *** Second Thoughts ***

I'm starting to have second thought aout this 2 phased system.
I'm leaning on either real time with pause, or WEGO with 1 phase per round, but i am also considering simplifying alot as a v0.1 - just control one character,with gamepad or mouse and keys, move with left stick, aimwith right stick.
shoot with thedifferent triggers, we can have normal shot,throwgrenade,melee and some kind of block, which could also serve as a parry mechanic for melee attacks.
Perhaps a grab too. would be amazing, it would be cool if there was some kind of rock paper scissor effect between melee, grab and block, i guess aretheclassics
we alsoneed tofit in shooting-perhaps a shot is same as melee attack in effectiveness
will ofc depend on the units size etc as well.

Now - the reasoning for this is - especially the action oriented control scheme, ithink this is very easy to implement, and be done with, while the more tactical elements might want much more, abilities and what not, we never know when we'redone.

none of this is quite relevant for the machine learning track either, they will need to have their controls - and in the end it will be marines moving in corridors -so their controls can be designed in parallel to the players control scheme (the AI do not care if we pause the game for example).

Anyway - i would love it if the control schema also became modular - we have the game physics, inlcuding in what order everyhing is calculated.

each tick, each unit will recieve orders in one way or another

the way we issue these orders (game pad, mouse +keyboard and pause,) - is free for us to decide anytime, or at least it seems so for me.
One thing i can lock intho, it will be "real time" all the time- im not considering changing to turnbased or anything like that - the physics of the game demands real time i feel like.


I don't know how much time we should puton thisn ow, well , if we could have a modular control scheme, that would be great i thnk.
I wonder if you could help me design it and set it up, also figure out how invasive it would be to the curent codebase.

proposed controls for the controller scheme
xbox controller:
left stick: move
right stick aim
right trigger: Shoot

right buttin -melee
left button block melee
left tirgger = parry

grenade should be X i think, same button u use itemswith in elden ring, forgot ifits X or B honestly.

Weneed a usebutton too.

Blocking could be effective against both melee and ranged perhaps

melee and blocking will probalby need some stamina system too, i wonder if it should be more like the rooms of many rooms, or more like dark souls -- in short, rooms of many rooms has dark souls like mechanics, but fuzed with world of warcraft mana mannagemnet - You need to mind stamina exactrl as in dark souls - but the penalty doesnt come directly, instead it comes after some minutes of innefective play, you are now out of stamina a little earlier that ifoyu had conserved it- basically as a way to introduce delayed rewards. (a too agressive player do not see the downside of depleting staming until it's out - basically, the stamina bar is huge, but dont replenish quickly -so u need to useit economically)-but im not sure this is right for breach. But we can keep it in mind. 