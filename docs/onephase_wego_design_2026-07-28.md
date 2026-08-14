# OnePhaseWEGO — the turn-formula redesign (design doc)

**Date:** 2026-07-28 (designed 2026-07-27/28, Erik + Fable)
**Status:** DESIGN LOCKED — Erik and Fable both signed off after three rounds of
discussion; one open refinement (plan invalidation, §14) deliberately left open.
**Supersedes (at canon fold, not before):** the two-phase round in
`architecture/mechanics/04_turn_and_control.md`. `TwoPhaseWEGO` stays shipped and
byte-identical until this ruleset is human-blessed; retirement is a deliberate,
once, golden re-baseline at the canon fold.
**Related:** `archive/control_modularity_design_2026-07-22.md` (the Ruleset /
ControlSource split this builds on) · `archive/free_aim_shooting_design_2026-07-23.md`
· `architecture/engine/16_entity_system.md` (SignalBus, entity items, cover
entities) · Erik's original "Second Thoughts" note preserved at the bottom of
mechanics/04.

---

## 1. Motivation

Two things drove this redesign:

1. **Erik doesn't like the two phases.** The Tab-toggled two-window planning was
   cumbersome, and not every round is a breach round. The phase machinery
   (per-phase AP pools, phase-tagged orders, DET_BETWEEN_PHASES) exists to serve a
   choreography need that shorter rounds + explicit hold/sync verbs serve better.
2. **Why WEGO at all? Multiplayer.** Orders are exchanged only between rounds, so
   deterministic lockstep absorbs all latency — both clients simulate identical
   rounds from identical order sets. WEGO is the multiplayer formula; the
   `ContinuousRealtime` + gamepad ruleset remains the feel-testing/action variant.
   Neither is yet declared "the" way the game ships; both stay pluggable Rulesets.

**ML decoupling (Erik's ruling):** the RL bots do NOT have to use the human
control scheme. They need some way to order units around (A* available to them;
RTS-style control is fine) and their interface is designed in parallel, not
inherited. Noted without obligation: the WEGO cadence — emit an order set every N
ticks — happens to be a natural commander-abstraction action interface for RL.

---

## 2. The round

- **One phase per round.** Phases are gone: no phase tags on orders, no per-phase
  AP pools, no Tab toggling, no DET_BETWEEN_PHASES slot semantics.
- **Round length ~4 s = 96 ticks @ 24 Hz** (`clock.round_duration_seconds`,
  tunable dial; 4 s is the expected neighborhood, not a promise).
- Flow: **PLAN (paused) → EXECUTE (96 ticks, continuous) → PLAN** …
- Short rounds are load-bearing: they shrink the commitment horizon (the real
  complaint behind "I don't like the phases" was committing 10 s of orders
  against a world that burns and floods under you), and they make cross-round
  choreography cheap — get everyone in position this round, strike the next.
- **Round boundaries are invisible seams** (§13): nothing about the world resets
  at the boundary; it is purely "the player may issue orders now."

---

## 3. Time is the only currency

AP is dead. There is no separate action-point economy — **the round's seconds are
the budget**, and actions cost time:

- **Durations**: every action takes ticks (weapon fire time, throw animation,
  tool use…).
- **Cooldowns**: per-action recovery times (registry rows, §5).
- **Global cooldown (GCD)**: short (~**0.5 s or less**, dial). Triggered by
  *actions* — shoot, grenade, tool-use/operate — **never by movement**. Within a
  weapon's own salvo, successive rounds do NOT re-trigger the GCD (an SMG burst is
  one action); the GCD gates *changing* action.
- **Weapon swap cooldown**: **0.75 s** (dial), its own non-global cooldown.
  Swapping is otherwise free — the design intent is *lots* of weapon changes,
  choosing the tool for the task, while preventing instant back-and-forth.
- **All timers persist across round boundaries** (§13) — no cramming actions at
  t=3.9 to reset anything.

The consequence: a unit's plan is a **timeline** — (move to A, arrive 1.4 s) →
(GCD to 1.9 s) → (fire to 3.0 s). The planning UI (§16) is designed around this.

---

## 4. Continuous space (engine ruling)

**Units, bullets, and cover objects live in continuous Q16.16 fixed-point
coordinates — not on the physics tile grid.**

- The physics grid remains what it is (wind, pressure, fire, smoke need it).
  Units/cover are **snap-stamped** into the obstacle grid each tick as an
  approximation for the field solvers — Erik's ruling: "if it's not totally
  correct, who cares a little bit; it's a good approximation." The stamp is
  deterministic rounding, so there is no determinism cost — it is a recorded,
  accepted approximation.
- **The end-of-round integer-tile position snap is REMOVED.** With 4 s rounds it
  would fire 2.5× as often and be visible; nothing needs it.
- Bullets ray-march in continuous space and can hit continuous-space collision
  shapes (units, cover) — this is what makes physical cover (§7) work.
- Pathfinding stays tile-grid A* with continuous per-tick interpolation, as
  today.
- Escape hatch: if continuous ever disappoints, snapping back to grid is always
  available. Build continuous-first.

---

## 5. Orders and the action registry

### The v1 order/action set

| Action | Notes |
|---|---|
| **Move** | THE primary order. No auto-attack while moving. Moves at full speed — the old Sprint is folded in (Sprint and Move-w/-Cover are **removed** as separate orders). Shift-click queues waypoint strings. No GCD. |
| **Shoot** | Secondary-most-important order; default is *stand and shoot*. Targets a **unit**; aim tracks the target during execution. |
| **Move & Shoot** | Exists, with reduced accuracy (wider spread, §7) and reduced speed (§6). |
| **Overwatch** | §9. |
| **Ambush** | §10 — the synced attack. |
| **Hold (until t)** | Wait at the current position until a chosen time; the sequencing verb that makes choreography composable. |
| **Weapon swap** | Primary ↔ secondary, free except the 0.75 s swap CD. |
| **Use item** | Registry rows generated from usable inventory items (grenades, charges, …). |
| **Detonate / plant charge** | Door/wall charges keep their slot mechanics; detonation **time is schedulable anywhere in the round, 0–4 s** (§12). Planting is a *channeled* action. |

**No called shots in v1** (Erik's ruling). The Fallout lesson is recorded: aimed
body-part shots are only a real choice when body parts have functionally
different consequences (crippling, disarming); otherwise one choice dominates
and it's a micro tax. Weapons get **crit/headshot profiles** instead (a sniper
rifle sometimes headshots on its own). A future *skill* may reintroduce called
shots deliberately.

### The action registry (extensibility backbone)

A data-driven table, same pattern as the material table. Each action is a row:

- `name`, `icon`
- `duration` (ticks), `cooldown` (ticks)
- `triggers_gcd` (bool), `gcd_exempt_within_salvo` semantics for weapons
- `interruptible` (bool) — §13; channeled actions (terminal use, planting a
  charge, objective interactions) are `interruptible = false`
- `targeting` (unit / tile / direction / none)
- `start_condition` (immediate / at-time-t / ambush-barrier / *(future)* signal)
- class gating (character-specific actions), item linkage (rows generated from
  inventory items)

Adding a new verb = adding a row (+ its resolution branch). Character-specific
skills, item actions, and future verbs (melee, called-shot skill…) all enter
this way. The hotbar **renders the registry** (§16) — dragging an item to the
hotbar binds a slot to a row.

---

## 6. Movement and aim

Facing/aim is decoupled from movement direction (already true in the engine:
`Unit.facing`, the AIM intent). Speeds are **aim-relative** — the old
three-mode speed table is replaced:

| State | Speed (of full move speed) |
|---|---|
| Move, no engagement | 100 % (this IS sprint now) |
| Move & shoot, target ahead | **60–75 %** (dial) |
| Move & shoot, reversing (aim opposite to movement) | **~25 %** (dial) — backpedaling out of a room with your gun on the door is deliberately slow |

All numbers are config dials. A move-while-aiming order needs a two-part gesture
in the UI (path + aim anchor) — sketched at implementation, not prescribed here.

---

## 7. Accuracy = spread angle; cover = physics

**There is no statistical to-hit model and no XCOM-style modifier stack.** The
whole accuracy system collapses into one dial per situation: **spread angle**.

- Move & shoot → wider spread. Reversing → wider still.
- Overwatch's narrowed cone → slightly tighter spread (§9).
- Spread randomness draws from the sim RNG (deterministic).

**Cover is physical.** Bullets ray-march in continuous space; a cover object
physically eats the rays that clip it. A marine hugging a crate is protected
exactly as much as geometry says — no cover bonus stat anywhere. This also makes
"easiest to hit" (§9's targeting rule) *computable*: the target with the largest
exposed profile.

**Cover objects are entities** (editor-placed, entity-system rows): collision
shape in continuous space, HP (**destructible cover**), optionally a signal on
destruction later.

**v1 fence:** cover is **static-but-destructible**. The full dynamics — cover
and items **pushed by shockwaves** (`wave_p` impulses), **carried/floated by
water** — is a separate continuous-space-dynamics arc with its own design
session (agreed 2026-07-28). This doc only fixes the interface v1 needs: static
destructible collision shapes that block bullets. The dynamics arc slots in
behind it without touching the order system.

---

## 8. Vision (v1 model)

LOS today is only half-real (raycasts exist for lighting and auto-fire; there is
no queryable vision model). Vision becomes a first-class system, because
overwatch acquisition, Ambush, marking, flanking, and enemy visibility all lean
on it:

- Each unit has a **facing vision cone** (half-angle dial) with **UNLIMITED
  range** — walls are the only limit (Erik's ruling: max vision range is
  unrealistic and disliked). Cost note: LOS is a ray-march terminating at the
  first wall, so cost is bounded by map geometry, not a range constant; the
  batched CUDA raycaster already does this shape of work.
- Plus a short **360° awareness radius** (hearing/peripheral, dial).
- **Team vision = union of member cones.**
- **Fog of war = visibility gating only**: enemies without LOS from your team
  simply don't render. No visual fog layer, no last-known-position ghosts in v1
  (unseen = gone; memory-ghosts are a future refinement if blinking feels bad).
- Predicates off the same cones: **discovered** = entering an enemy's vision;
  **flanked** = attacked from outside your own cone.
- Lockstep honesty note: in deterministic lockstep multiplayer the full state
  lives on both machines, so fog is honor-system client-side. Designed as if
  enforced; enforcement is a later server problem.
- **Out of scope (future sessions):** light/flashlight-based stealth, sound
  propagation.

**Flashlights** (feel item, v1): remove the cursor flashlight; marines carry
flashlights (toggleable), rendered as the expression of their facing cone.
During planning the flashlight aims toward the cursor — build BOTH variants
(selected unit only / whole team) behind a toggle and feel it out.
**Render-only in v1**: the moment lights affect gameplay (zombies drawn to
light, vision limited to lit areas) they must cross into the sim — that is a
deliberate stealth-system decision for another session, not drift.

---

## 9. Overwatch

- Marine aims a cone in a chosen direction and engages targets entering it.
- **Adjustable cone width** (player-set per order). Primary purpose = **target
  control**: targets outside the cone are ignored, so narrowing the cone is
  indirect target selection. Secondary: a narrow cone gives a small accuracy
  (spread) bonus. The cost: side blindness — attackers outside the cone aren't
  responded to. If the dial goes unused in play, the fallback default is the
  unit's normal vision-cone width.
- **Target priority:** (1) marked targets (§11); (2) easiest to hit = largest
  exposed profile (computable under physical cover); tie-break (3) closest to
  cone center. Deterministic.
- Continuous engagement (normal weapon behavior while targets are in the cone),
  not one-reaction-shot.
- **Overwatch is a state** — persists across rounds until replaced; buffable
  (pre-aimed bonus) and debuffable (**reduced defense from behind** — which
  generalizes: §Facing).
- Facing/flanking generally: attacked from outside your facing arc = increased
  vulnerability (tunable modifier), for all units, with overwatch simply having
  a wider rear penalty. Facing determines defense.

---

## 10. Ambush (the synced attack)

The group-coordination verb, named **Ambush**.

- Any number of units may queue an `Ambush(target)` order on the same enemy.
- **Readiness = the unit has reached the Ambush order in its queue** — i.e.
  finished everything queued before it (move into position, hold-until-t,
  detonate the door charge, …). No LOS condition, no other semantics — Erik's
  simplification, and it makes timing composable: the breacher's queue
  [move → detonate → Ambush] means everyone else (already waiting on the
  barrier) fires the moment the charge blows.
- **Fire condition:** the instant ALL group members on that target are ready,
  all fire simultaneously. Dead/incapacitated members drop out of the count
  (no deadlock).
- **Ambush sprung:** if ANY group member is fired upon, all *ready* members open
  fire immediately; not-yet-ready members continue their queues (and join when
  they arrive).
- **Timeout backstop:** a group that never becomes ready reverts to idle stance
  at round end — no infinite holds.
- **No SignalBus needed in v1** — Ambush is a per-group readiness counter in the
  sim. The signal start-condition ("fire when signal X") stays on the deferred
  list for when level logic (Arc B wiring) should trigger squad actions; the
  registry's `start_condition` field keeps the slot open.

---

## 11. Target marking

- Mark an enemy **visible to your team** (team vision at mark time required);
  per-team `unit_id → mark` table, consulted by all targeting functions
  (overwatch priority, idle return-fire preference). Deterministic and cheap —
  the real cost is UI.
- **v1: a single "focus" priority level.** Graded priorities 1–5 (focus-fire
  ordering against an incoming zombie group) are a future refinement if play
  demands them.
- Marks persist until the target dies or is unmarked.

---

## 12. Scheduled detonations and breach choreography

- Door/wall charges detonate at a **player-chosen time anywhere in the round
  (0–4 s)** — the old start/between/end det-slots are replaced by a time.
- A charge thrown/planted during planning may detonate at **t=0 of the next
  round** — preserving the cool breach opening: door blows at 0.0, grenades and
  fire follow.
- Cross-round choreography is the intended idiom: position + plant this round;
  Hold / Ambush / detonate at the top of the next.

---

## 13. Round seams, interrupts, idle

- **Invisible seams:** cooldowns, GCD, overwatch state, Ambush groups, in-flight
  projectiles, fires — everything persists across the round boundary. The old
  `_end_round` teardown (position snap, obstacle reset, order clear semantics)
  is rebuilt for this ruleset. (The known tick-boundary regression flagged in
  mechanics/04 lives in exactly that machinery — this rebuild subsumes it.)
- **New orders interrupt by default.** Issuing orders at the planning pause
  replaces the unit's remaining queue immediately. Actions flagged
  `interruptible = false` (channeled: operating a terminal, planting a charge,
  objective interactions) must complete first. Mid-salvo shooting: interruptible.
- **Idle stance:** a unit with no orders (queue finished, or its player
  submitted nothing) **returns fire** at attackers — preferring marked targets —
  but does not free-fire at everything it sees. "Do nothing" reachable by
  config. Philosophy: you're meant to give orders to your whole (small) squad;
  return fire is a floor, not an AI.

## 14. Plan invalidation — OPEN QUESTION (deliberate)

When a planned path becomes blocked mid-round (wall drops, door shuts, rubble):
**v1 starting behavior = halt in place, continue remaining non-move orders, no
auto-repath; replan next round.** Erik has explicitly NOT decided this is right
("I haven't thought about it") — it is an open refinement to revisit after play.
Alternatives on the table: local auto-repath (deterministic but agency-taking),
partial repath within a distance budget, per-order "repath allowed" flag.

---

## 15. Weapons and inventory

- **Loadout: primary + secondary slot** (two weapons, or weapon + item), swapped
  freely for 0 cost except the 0.75 s swap CD. Design intent: frequent,
  tactical weapon choice.
- **Inventory UI: the Dark Souls 3 pattern** (Erik's spec) — Start/menu button →
  [Inventory, Equipment, Character, Options…, Quit]; plus a **quick-item belt**
  (4–5 slots) for seldom-used items. Works beautifully on controller; in WEGO
  the planning pause hosts it naturally; in `ContinuousRealtime` it overlays
  without pausing (exactly like DS). Quick-belt slots are hotbar slots — same
  registry rows, one system wearing two skins.
- **Shared across rulesets:** inventory + hotbar serve both OnePhaseWEGO and
  ContinuousRealtime (in gamepad mode, rows map to buttons).
- **v1 fence: loadout-based inventory, no ground pickup/drop** — world items
  arrive with the entity-item pass later.

## 16. Planning UI and visualization

- **Teal path viz:** ordering a move draws the path line; the **endpoint
  footprint** is highlighted in teal and labeled with the **arrival time**
  ("2.3" = arrives 2.3 s into the round). Shift-click waypoint strings show teal
  footprint markers at each clicked intermediate point, always with the path
  line.
- **Shoot hologram:** a shoot order from a future position shows a teal
  hologram of the marine at the firing tile, indicating its target.
- **The timeline is the concept** to design the UI around (a unit's plan is a
  schedule). v1 ships the timestamps; a **scrubbable preview** — drag a time
  slider, see every marine's ghost at time t — is the natural extension, and
  determinism makes the preview *exact* (dry-run of your own plan). Design for
  it, ship it when it's cheap.
- **Hotbar:** shows orders (move, shoot, move&shoot, overwatch, ambush, …) and
  usable items; items draggable from inventory to hotbar (slot ↔ registry row
  binding).
- **Planning clock (multiplayer):** submit-within-N-seconds timer, simultaneous
  reveal. On timeout with no orders: units idle per §13 (return fire only).
  Single-player planning stays untimed.
- **UI architecture:** a standalone `ui/` package, NOT a framework. Two seams:
  it *reads* sim state + action registry + inventory; it *writes* only orders
  through the ControlSource path. Hotbar, overlays, DS3 menu all live there,
  shared by both rulesets.

## 17. Controls ruling

**In game mode, the control scheme owns every binding.** The OnePhaseWEGO keymap
is designed from scratch for play only. All diagnostic/graphical toggles
scattered today (field overlays, temp/pressure views, 3D-model toggle M,
normal-map F5, …) are removed from game-mode keys and retreat behind a debug
mode (`--debug` overlay or a debug ruleset) — designing that debug home is its
own future evening. Erik designs the final keymap at implementation time.

---

## 18. Engine & implementation plan

- **New ruleset `OnePhaseWEGO` built BESIDE `TwoPhaseWEGO`** (the Ruleset
  abstraction from 2026-07-23 makes this a new strategy class). All existing
  goldens/digests stay untouched during the whole build — zero determinism risk
  while developing. `TwoPhaseWEGO` retires only at the canon fold, with ONE
  deliberate golden re-baseline and written rationale, after Erik's blessing.
- Sim-side pieces: timeline/cooldown bookkeeping, action registry, continuous
  positions + snap-stamp, vision system, overwatch/ambush/marking state,
  physical bullet-vs-shape march, cover entities. All synced state Q16.16; all
  randomness via sim RNG.
- Render/UI-side pieces: teal viz, holograms, hotbar, DS3 menu, flashlights,
  debug-key eviction. Render-layer, no determinism constraints.
- **Feel gate:** this arc is feel-defining — the end gate is a HUMAN-TEST (Erik
  plays it); **no auto-merge**. Mechanical intermediate patches may gate on
  suite-green as usual.
- Implementation session: fresh Opus session with a kickoff doc (to be written
  from this design); design doc committed to main FIRST (this commit) so
  worktree agents can see it.

## 19. v1 scope fence (agreed)

IN: one-phase 4 s rounds · timeline/time-currency (GCD 0.5 s, swap CD 0.75 s) ·
action registry · Move/Shoot/Move&Shoot/Overwatch/Ambush/Hold/swap/use ·
aim-relative speeds · spread-angle accuracy · physical static destructible
cover · continuous-space units/bullets/cover · vision v1 (unlimited-range cones,
visibility gating) · flashlights (render-only) · single-level marking ·
scheduled detonations · primary/secondary loadout · DS3 inventory menu +
hotbar · planning viz (paths, timestamps, holograms) · planning clock ·
game-mode-owns-all-keys.

OUT (deferred, each with a home): cover/item dynamics — shockwave push + water
float (own design arc) · SignalBus start-conditions for squad orders · called
shots (future skill) · locational damage · stealth/light/sound systems ·
last-known-position ghosts · graded marks 1–5 · ground pickup/drop (entity-item
pass) · scrub-preview slider (design-for, ship-later) · rolling-orders /
commitment-latency variant (interesting, explicitly not v1) · melee/block/
parry/grab + stamina (own arc, per mechanics/04) · debug game mode design.

## 20. Open questions / refinement dials

1. **Plan invalidation** (§14) — the one deliberately open semantic.
2. Round length (4 s neighborhood), GCD (≤0.5 s), swap CD (0.75 s), speed
   percentages, spread angles, cone half-angles, awareness radius — all dials
   for the tuning pass.
3. Flashlight aiming: selected-unit vs whole-team (build both, feel decides).
4. Overwatch cone dial: if unused in play, collapse to standard width.
5. Idle stance default: return-fire vs do-nothing (return-fire is the v1
   default).
6. Whether Ambush needs a grace/stagger dial for ragged-volley feel.
