# OnePhaseWEGO — build kickoff (Opus session, 2026-07-28)

**Source of truth:** `docs/onephase_wego_design_2026-07-28.md` (DESIGN LOCKED,
Erik + Fable). This document does not re-decide anything there; it records the
**as-planned architecture**, the **patch sequence**, and the three
implementation-time rulings Erik gave at kickoff.

**Branch:** `onephase-wego` (off `main` @ `4632ac8`), own worktree.
**Baseline before the first patch:** `1680 passed, 22 skipped` (CPU build,
conda env `data`). Every existing golden/digest stays untouched for the whole
arc — the new ruleset is built BESIDE `TwoPhaseWEGO` (design §18).

---

## 1. Erik's kickoff rulings

1. **Keymap (§17) = hotbar + RMB-move.** Numbers `1..0` are hotbar slots that
   RENDER the action registry (items drag in — §16); right-click is always
   *Move*, so the primary order never needs a mode. Full map in §5 below.
2. **Debug keys (§17) = behind `--debug`.** Game mode gets ZERO diagnostic
   keys. `main.py --control onephase --debug` re-arms today's whole debug set
   (I/J/K/U/T/V/P/O/N/M/F5) on the keys it already uses.
3. **Test level = new `levels/wego_test`.** Playground and planetside_demo are
   left exactly as they are. wego_test carries cover entities, a marine squad,
   zombies, and a breachable door.

---

## 2. Architecture (as planned)

### 2.1 The clock — monotonic, not rewound

`TwoPhaseWEGO` rewinds `sim.tick` to 0 at every round boundary; that rewind is
precisely what forces the `last_fire_tick = -999` / `reload_done_tick = -1`
teardown in `_end_round`. OnePhaseWEGO's iron requirement is the opposite
(design §3/§13: *all timers persist across round boundaries*), so:

- `sim.tick` is **free-running and monotonic** under this ruleset.
- `round_index = tick // ticks_per_round`, `round_tick = tick % ticks_per_round`.
- Every cooldown/GCD/duration is an **absolute tick deadline**
  (`gcd_until_tick`, `swap_cd_until_tick`, `busy_until_tick`, …) — so a seam
  crossing is arithmetically invisible, which is what §13 asks for.
- Scheduled detonations (§12) become absolute ticks too: "0–4 s into the
  round" is `round_start_tick + round(t * tps)`.

The round boundary therefore does exactly two things: **pause** (the player may
order now) and **recompile plans**. No snap, no order clear, no AP refill, no
obstacle reset, no tick rewind.

### 2.2 Time is the only currency (§3) — the timeline

A unit's plan compiles, at submit time, into a **schedule**: a list of
`(action, start_tick, end_tick)` plus a per-tick `move_path`. The compiler is
the single place that knows durations, cooldowns, GCD and aim-relative speed —
which makes the §16 UI free (the arrival timestamps the player reads ARE the
compiled schedule) and makes execution a cursor walk.

Per-unit timer state (all absolute ticks, all dormant under `TwoPhaseWEGO`):
`busy_until_tick`, `gcd_until_tick`, `swap_cd_until_tick`,
`action_cd: {action_name: until_tick}`, `plan_cursor`.

GCD rule (§3): triggered by actions, never by movement; successive rounds
within one weapon's salvo do NOT re-trigger it (the GCD gates *changing*
action, not the salvo).

### 2.3 Action registry (§5) — the extensibility backbone

New module `simulation/action_registry.py`, same data-driven pattern as the
material and weapon tables: rows authored in `config.toml` `[actions.*]`,
built into an `ActionTable` at sim construction, plus rows GENERATED from
usable inventory items. Columns: `duration_ticks`, `cooldown_ticks`,
`triggers_gcd`, `interruptible`, `targeting`, `start_condition`, class gate,
item linkage, icon. Adding a verb = adding a row + its resolution branch.

Order vocabulary extends `simulation.orders` with ids ≥ 6 (`ORDER_OVERWATCH`,
`ORDER_AMBUSH`, `ORDER_HOLD`, `ORDER_SWAP`, `ORDER_MOVE_SHOOT`, `ORDER_MARK`,
`ORDER_MOVE`) and optional payload fields on the existing single `Order` class
(`action_name`, `target_unit_id`, `start_tick`, `det_tick`, `cone_deg`,
`aim_anchor`). The old ids keep their exact meaning so `TwoPhaseWEGO` is
untouched. Note `ORDER_MOVE` is NEW and distinct from `ORDER_MOVE_ATTACK`:
v1 Move does **not** auto-attack (§5) and runs at 100 % speed (Sprint folded
in; Sprint and Move-w/-Cover are not offered by this ruleset).

### 2.4 Continuous space (§4)

Units/bullets are already continuous (`Unit.x/y` floats, `BulletInFlight`
marching by exact `n/65536` steps) and `stamp_units` already re-stamps the
obstacle grid every tick — so §4 is mostly a matter of **removing** the
end-of-round integer snap (it simply never runs under this ruleset) and adding
continuous **cover shapes** the march can hit. Pathfinding stays tile A*.

### 2.5 Cover = physics (§7)

New entity class `cover` (`simulation/entities/cover.py`): an editor-placed
row with a continuous AABB (anchor + size in meters, quantized once at load
like a door span), `hp`, destructible. A `CoverRuntime` list is built at
construction beside `_doors`.

The bullet march gains a cover test per step: a round whose position enters a
live cover AABB **stops and chews it**. Under this ruleset the statistical
`cover_exposure_at` / `roll_exposure` path is **not consulted at all** —
design §7 says there is no statistical to-hit model. Accuracy collapses to
spread angle, which already exists on the weapon rows; the ruleset supplies
the situational spread (move&shoot wider, reversing wider still, overwatch
cone tighter).

Cover blocks **bullets, not vision** (a crate is low cover — you see over it).
`blocks_los` is a field, defaulting false, so a full-height barricade is
expressible.

### 2.6 Vision (§8)

New module `simulation/vision.py`, recomputed once per tick and cached:
per-unit facing cone (half-angle dial, **unlimited range**, walls terminate
the ray) + a 360° awareness radius; team vision = union. Exposes
`visible_enemy_ids(team)`, `exposed_profile(shooter, target)` (the "easiest to
hit" metric §9's overwatch priority needs — fraction of silhouette samples
reachable by an unblocked ray, walls + cover), and the `discovered` /
`flanked` predicates. Fog of war is **render gating only**: the renderer draws
no enemy the player's team cannot see.

### 2.7 Overwatch / Ambush / Marking (§9–§11)

- **Overwatch** is persistent unit state (direction + player-set cone
  half-angle), survives round boundaries, engages continuously, priority
  `marked → largest exposed profile → closest to cone centre`.
- **Ambush** is a per-target readiness counter on the sim (no SignalBus, §10):
  readiness = the unit's plan cursor reached its `Ambush` step; all-ready →
  simultaneous fire; any member fired upon → all *ready* members fire; dead
  members drop out; round-end timeout reverts to idle.
- **Marking** is a per-team `{unit_id: mark}` table consulted by every
  targeting function. Single "focus" level in v1.
- **Idle stance** (§13): a unit with an empty plan returns fire at whoever
  last attacked it (preferring marked), and does not free-fire.

### 2.8 UI (§16) — a standalone `ui/` package

Per design §16 the UI is a package, not a framework, with two seams: it READS
sim state + action registry + inventory, and WRITES only orders through the
ControlSource. Hotbar, planning overlays (teal path + endpoint footprint +
arrival timestamp, waypoint markers, shoot holograms), the DS3 menu, the
planning clock and the marks/overwatch overlays live there; `renderer/`
keeps its existing draw primitives and gains nothing ruleset-specific.

Flashlights replace the cursor flashlight: marine-carried, rendered as the
expression of the facing cone, with BOTH aiming variants (selected-unit /
whole-team) behind a toggle key (§20 item 3). Render-only.

---

## 3. Patch sequence (each gates on `pytest tests -q` green)

| # | Patch | Gate |
|---|---|---|
| P1 | `OnePhaseWEGO` ruleset, monotonic clock, config dials | new ruleset tests + suite |
| P2 | action registry + order vocabulary | registry tests + suite |
| P3 | timeline plan compiler + executor | timeline tests + suite |
| P4 | vision v1 | vision tests + suite |
| P5 | cover entities + physical march + spread-only accuracy | cover tests + suite |
| P6 | overwatch + ambush + marking + idle return-fire | behavior tests + suite |
| P7 | loadout/swap, scheduled detonations, inventory model | tests + suite |
| P8 | `ui/` package: hotbar, viz, holograms, DS3 menu, flashlights, clock, fog | headless UI-logic tests + suite |
| P9 | control source + keymap + `--debug` eviction + main.py wiring | suite + launch smoke |
| P10 | `levels/wego_test`, full suite, as-built docs | full suite + HUMAN-TEST |

**Determinism discipline throughout:** synced state stays Q16.16 integer or the
existing float-position discipline (exact `n/65536` steps, no libm — the
`unit_fixed` kit for every angle); all randomness through `sim.rng`. No
existing golden is re-baselined at any point in this arc — `TwoPhaseWEGO`
remains byte-identical, and its retirement is a separate, deliberate event at
the canon fold after Erik blesses this ruleset (design §18).

**Merge discipline:** this arc is feel-defining. The end gate is Erik playing
it. **No auto-merge** (project CLAUDE.md HUMAN-TEST rule).

---

## 4. Out of scope (design §19, restated so no patch drifts into it)

Cover/item dynamics (shockwave push, water float) · SignalBus start-conditions
· called shots · locational damage · stealth/light/sound systems ·
last-known-position ghosts · graded marks 1–5 · ground pickup/drop ·
scrub-preview slider (designed-for, not shipped) · rolling-orders variant ·
melee/block/parry/grab + stamina · the debug-game-mode design (the `--debug`
flag here is the minimum eviction hatch, not that design).

§14 (plan invalidation) ships its stated v1 behavior — halt in place, continue
remaining non-move orders, no auto-repath — and stays flagged open.

---

## 5. The keymap (Erik's ruling 1, as built)

```
MOUSE
  LMB                select marine (click its body)
  RMB                MOVE here — the primary order, no mode needed
  Shift+RMB          queue waypoint (append, don't replace)
  LMB w/ slot armed  apply the armed action at the cursor
  Wheel              zoom

HOTBAR — 1..0, renders the action registry; items drag in
  1 Move   2 Shoot   3 Move&Shoot  4 Overwatch  5 Ambush
  6 Hold   7 Grenade 8 Charge      9,0 quick-belt item slots

VERBS
  Q    weapon swap (primary <-> secondary, 0.75 s CD)
  X    mark target
  Space SUBMIT — execute the round
  Bksp  undo last order       Esc  clear selection / cancel mode
  Tab   cycle marine          I    inventory (DS3 menu)
  L     flashlight variant toggle
  WASD / arrows  pan
```

Everything diagnostic lives behind `--debug` (ruling 2).
