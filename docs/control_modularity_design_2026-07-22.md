# Modular control design (2026-07-22)

**Status: LOCKED 2026-07-22 — Erik approved the split; his clarifications are
folded in below. Opus kickoff in §9. Melee/stamina explicitly deferred.**

Prompted by the *Second Thoughts* section of
`docs/architecture/mechanics/04_turn_and_control.md`: Erik is leaning away from
the two-phase WEGO round. His current read: the shipped game will probably end
up **some form of WEGO or real-time-with-pause — strategy over reactions** —
but he wants **direct action controls** (one possessed marine, gamepad) early
anyway, as a dev tool: run through levels, shoot at will, play out action
scenarios, get a feel for the weapons. One thing is locked: **the game is real
time, always** — no turn-based pivot; the physics demands continuous time.

**The deeper reason (long-term goal):** after Breach, Erik may build an RPG on
the same engine — different physics parameters, different damage rules,
different controls, all different, *same physics engine*. So the target is not
"two control schemes" but **loadable games**: a game =
`Ruleset` + `ControlSource` (+ `AgentPolicy` for the ML track) + config/content
(physics dials and damage tables are already config-driven). This doc's split
is the first concrete step toward that.

---

## 1. The core claim

Erik's instinct in the second-thoughts note is correct and is *already true in
code*: "each tick, each unit receives orders in one way or another — the way
we issue these orders is free to decide anytime."

The engine facade (`apply_action` / `step` / `get_state`) is
structure-agnostic. Pause is "the caller skips `step()`" — the AI path already
runs pause-free with zero special-casing. Orders are plain data through one
validation chokepoint. Gamepad input is no threat to determinism: sampled
stick state becomes per-tick synced orders, exactly like AI actions — a
replay is the same order stream.

What is **not** free is everything phase-shaped that leaked into the
`Simulation` and the order vocabulary. That is the extraction work.

## 2. Where WEGO is baked in today (the coupling inventory)

1. **`Simulation.step()` tail** (`src/simulation/simulation.py` ~1092–1118):
   tick-0 DET_START_PHASE1 firing, tick-60 phase boundary +
   DET_BETWEEN_PHASES, tick-120 DET_END_PHASE2 + `_end_round()` + auto-pause.
2. **`_end_round()`** (~1208–1260): zombie conversion, position snap, order
   clear, AP refill, mag/spray reset, obstacle reset, tick rewind.
3. **`apply_action()`**: reads `order.phase`, checks/spends per-phase AP pools
   (`u.get_ap(phase)` / `u.spend_ap`) — the *cost policy* is inlined at the
   chokepoint. `undo_last_order` is its inverse.
4. **Movement is round-precomputed**: `_compute_player_paths` lays a whole-round
   tick-by-tick trajectory at order time; `_update_player_movement` replays it.
   Entirely WEGO-shaped — direct control needs a per-tick move-intent path
   that doesn't exist yet.
5. **`Order.phase`** on every order; `DET_*` slots are named after phase
   boundaries; `spawn_projectiles_from_grenade_orders` fires only at tick 0
   and is invoked by the input handler on resume.
6. **`is_terminal()`** counts round-complete as an episode boundary.
7. **`InputHandler`** is planning-gated (`sim.is_paused()`) throughout — fine,
   it is one control scheme among several; it just isn't labeled as such yet.

Everything else — combat, physics, zombies, entities, events — never looks at
phase or AP and needs no change.

Note: the clock is **24 Hz** (`config.toml [clock] ticks_per_second = 24`,
i.e. ~41.7 ms per tick, 240 ticks per round). The mechanics/04 chapter's
table still says 12 Hz — stale, fix at the next canon fold.

## 3. Architecture: three swappable pieces (APPROVED)

Follows the interaction/cost-policy split locked in the entity design
(entity_system_design_2026-07-18.md §3d): systems declare *interactions*; a
swappable game-mode policy owns cost + permission.

### 3a. `Ruleset` — the turn structure + cost policy (sim-side strategy object)

Owned by `Simulation`, chosen at construction. Interface (sketch):

```python
class Ruleset:
    def on_round_start(self, sim): ...    # tick-0 work (DET slots, stamps)
    def on_tick_end(self, sim): ...       # phase advance, auto-pause, teardown
    def validate_and_cost(self, sim, unit, order) -> bool: ...  # AP or not
    def refund(self, sim, unit, order): ...                     # undo inverse
    def is_terminal(self, sim) -> bool: ...
```

Two implementations:

- **`TwoPhaseWEGO`** — the current behavior, extracted *verbatim*. Hard gate:
  byte-identical digests/goldens. This is a pure refactor, no re-baseline.
- **`ContinuousRealtime`** — near-trivial: no phases, no AP (validate = alive
  + physical preconditions only), no auto-pause, no round teardown. Design
  item: what replaces `_end_round`'s housekeeping (zombie conversion becomes
  death-triggered/immediate; the obstacle-grid reset becomes per-tick stamp
  semantics — corpses must still stop blocking physics).

`step()` keeps the load-bearing tick order (slots 1–9e) untouched; only the
round-clock head/tail (tick-0 block, boundary checks) route through the
ruleset. AP fields stay on `Unit` but only `TwoPhaseWEGO` ever touches them.

### 3b. `ControlSource` — how intents are produced (above the facade)

A sibling family beside the facade, chosen in `main.py` at startup (a
`--control` launch flag; default = current WEGO planning input):

- **`WEGOPlanningInput`** — the current `InputHandler`, renamed/wrapped.
  Unchanged behavior.
- **`GamepadDirect`** — one possessed unit (default: first team-0 spawn);
  each frame samples sticks/buttons and emits per-tick intents for it.
  **Gamepad comes FIRST** (Erik's call, 2026-07-22: he is traveling with a
  controller and no mouse — trackpad aiming is a non-starter). pyray exposes
  the raylib gamepad API (`is_gamepad_available`, `get_gamepad_axis_movement`,
  `is_gamepad_button_down`); the keyboard+mouse variant (WASD + mouse aim)
  is the same class with a different poll, added later.
- **`AgentPolicy`** — the ML path. Same interface: observe `get_state()`,
  emit per-tick intents. **This is the payoff for the RL project**: the agent
  action space IS the intent vocabulary; designing GamepadDirect designs the
  agent's controls too (agents don't care about pause, exactly as the
  second-thoughts note says).

### 3c. Intents — extend the order vocabulary with per-tick continuous verbs

Keep tile-targeted queued orders (WEGO needs them). Add continuous intents
consumed the tick they're issued, e.g.:

| Intent | Payload | Consumed by |
|---|---|---|
| `MOVE_DIR` | (dx, dy) normalized, speed mode | new per-tick movement branch |
| `AIM` | facing angle (Q16-friendly) | unit facing |
| `TRIGGER` | held / released | `process_shooting` (already per-tick) |
| `THROW` | target or direction + fuse | existing grenade spawn, generalized |
| `USE` | context | entity interactions (doors — already latch-based) |

`_update_player_movement` grows a branch: unit with a live `MOVE_DIR` intent
moves by velocity (mobility-table-scaled, footprint-collision-checked — the
same predicates A* uses); otherwise it replays `move_path` as today.
Determinism: intents are synced inputs; fixed-point the direction vector.

Melee / block / parry / grab / stamina (the rock-paper-scissors triangle and
the Rooms-of-Many-Rooms delayed-stamina idea — huge bar, slow regen, the
penalty for over-spending arrives minutes later as delayed punishment) are
**captured but deferred — Erik agreed 2026-07-22**: that is a new combat
*system*, not control modularity, and it deserves its own design pass. v0.1
direct control = move / aim / shoot / grenade / use with existing weapons.

## 4. Invasiveness assessment

| Piece | Size | Risk | Gate |
|---|---|---|---|
| Ruleset extraction (move ~150 lines of step tail / `_end_round` / AP checks behind the interface) | moderate | low — pure refactor | digests/goldens byte-identical |
| `ContinuousRealtime` ruleset | small | low | new tests |
| `ControlSource` seam in main.py + `--control` flag | small | trivial | — |
| Per-tick MOVE_DIR movement branch + AIM/TRIGGER | medium — the real new code | feel-adjacent | HUMAN-TEST (Erik plays) |
| Gamepad polling (**first** input variant) | small | feel | HUMAN-TEST |
| Keyboard+mouse direct control | small, later | feel | HUMAN-TEST |
| Melee/block/parry/grab/stamina | large | — | separate arc, not this one |

Net: the *modularity* itself is a contained refactor plus one genuinely new
mechanic (per-tick movement). The scary-sounding part of the second-thoughts
list (melee RPS) is severable and deferred.

## 5. Tick rate: resolved as a non-issue for now (Erik, 2026-07-22)

The draft flagged input-at-tick-rate sluggishness as the hard question. Erik's
clarifications dissolve most of it:

- The sim runs at **24 Hz** (not 12); rendering runs at 60 Hz and is not
  locked to the sim tick (and animation updates shouldn't be either).
- **Aim is continuous between ticks**: the stick/cursor state lives in the
  control layer at frame rate; the *visual* aim indicator moves smoothly at
  60 Hz; the sim samples the current aim at each tick boundary. Ticks are
  when physics changes — 24 Hz is plenty for that (many action games run
  20–30 Hz simulation under 60+ Hz presentation), and it could possibly even
  go lower.
- **Fallback if 24 Hz control ever feels coarse** (Erik's sketch): raise the
  unit/control tick to 60 Hz but step the field physics every 3rd tick —
  physics state persists across ticks 0-1-2, updates at 3, etc. Deterministic
  as long as the schedule is fixed. Recorded as the fallback; **not needed
  now, later optimization problem.**

One consequence worth writing down: per-tick dials (weapon cadence, fire
spread, movement ticks-per-tile) are tuned against 24 Hz; any future rate
change moves those dials. The `*_ticks_per_tile` derivation in config already
goes through seconds, which is the right pattern — keep authoring time in
seconds, derive ticks.

### Networking note (Erik raised online play)

The constraint is **not** "latency strictly < 1/24 s". Deterministic lockstep
buffers inputs N ticks ahead: latency budget = N × 41.7 ms. WEGO/RTwP has
seconds of slack (orders are planned, not twitch) — trivially networkable,
another point for the strategy endgame. Direct-action co-op is playable at
2–3 ticks of input delay (~80–125 ms, the classic RTS lockstep feel);
rollback netcode is ruled out — re-simulating GPU field physics for rollback
is not affordable. Good enough for co-op level run-throughs; this engine will
never be a competitive twitch shooter online, and doesn't want to be.

## 6. Sequencing

1. **P1 — Ruleset extraction** (byte-identical, digest-gated). Pure hygiene;
   makes the WEGO question reversible forever.
2. **P2 — ControlSource seam** + rename InputHandler + `--control` flag.
3. **P3 — per-tick intents + `ContinuousRealtime` + `GamepadDirect`** at the
   existing 24 Hz. Gamepad first (§3b). HUMAN-TEST.
4. **P4 — keyboard+mouse direct variant** (when Erik is back at a desk).
   HUMAN-TEST.
5. *(separate arc, later)* melee/block/parry/grab + stamina design.
6. *(only if ever needed)* the 60/3 tick-split fallback from §5.

## 7. Decisions (locked 2026-07-22)

1. Three-piece split (Ruleset / ControlSource / intents): **approved**.
2. v0.1 verb scope move/aim/shoot/grenade/use, melee deferred: **approved**.
3. Tick rate: **24 Hz stands**; aim is frame-rate-continuous, sim samples at
   tick; 60/3 split is the fallback, later-optimization. Not a blocker.
4. Priority: **parallel track** — runs alongside level building/tuning
   (Erik's lane) since Arc C is merged and B2 awaits his human test, neither
   needing further Claude build work. Hand to Opus now (§9).

## 8. Orthogonality vs. in-flight work (checked 2026-07-22)

- **Arc C (editor)**: merged to main (`54cd6cd`). The editor authors levels
  (spawns, zones, entities, wires) and is ruleset-agnostic. Two small
  touchpoints, both forward-compatible: play-from-editor (F5 →
  `_editor_scratch`) launches whatever `--control` defaults to; and direct
  control needs a possessed-unit choice (default: first team-0 spawn — the
  editor's SPAWN T data already provides it). Nothing blocks.
- **B2 smoke-honesty** (`origin/fire-b2-smoke-honesty`, unmerged, awaiting
  Erik's human test): touches **no** control-work core — `simulation.py`,
  `orders.py`, movement are untouched by B2. It does touch `main.py`
  (~62 lines), `input_handler.py` (6 lines), `config.toml`. So: **P1 is fully
  orthogonal**; P2/P3's `main.py` seam has a small merge-conflict surface
  with B2. Mitigation: P1 first; if B2 merges before P2 lands, rebase is
  trivial; if not, the conflict is confined to main.py wiring and resolved
  at merge time. No shared behavior, only shared lines.

## 9. Opus kickoff (fresh session — read this section first)

**Mission:** implement §6 P1→P3 on a new branch `control-modularity` (own
worktree per CLAUDE.md concurrency rules), following the
`autonomous-patch-workflow` skill.

- **P1 (Ruleset extraction).** Move the §2 items 1–3, 5, 6 behind the
  `Ruleset` interface (§3a); `TwoPhaseWEGO` is the current code verbatim.
  GATE: full `pytest tests -q` green + digest/golden byte-identical — this is
  a pure refactor; any golden drift means a bug, never a re-baseline.
  Mechanical + digest-gated: may merge on green **only if Erik pre-authorizes
  in the launch message**; otherwise push and report.
- **P2 (ControlSource seam).** Rename/wrap `InputHandler` as
  `WEGOPlanningInput`; add the `--control` flag (default `wego`); no behavior
  change. Same gate as P1.
- **P3 (direct control v0.1).** `ContinuousRealtime` ruleset + per-tick
  intents (§3c, fixed-point directions) + `GamepadDirect` (gamepad FIRST,
  §3b; pyray gamepad API). Movement branch reuses `is_passable_block` +
  the mobility table. New tests for intents/ruleset; goldens for existing
  WEGO trajectories stay byte-identical (the new code must be dormant under
  `--control wego`). **HUMAN-TEST gate: build, gate, push — Erik plays with
  the controller before any merge. Never auto-merge P3.**
- **Constraints:** determinism iron rules (Q16.16, no floats in sim path);
  no `git add -A`; commit this design doc's branch state before spawning
  worktree agents; escalation triggers per the Fable-designs/Opus-implements
  pattern — stop and ask Erik if (a) the extraction forces any golden
  re-baseline, (b) the intent design collides with the AP/phase fields in a
  way §3 didn't anticipate, (c) gamepad polling behaves nondeterministically
  across machines.
- **Out of scope:** melee/block/parry/grab/stamina, keyboard+mouse variant,
  tick-rate changes, networking. Do not touch B2's files beyond the minimal
  main.py seam (§8).
