# OnePhaseWEGO — as built (2026-07-28)

**Status:** ✅ **HUMAN-TESTED AND BLESSED by Erik, 2026-07-30 — merged to
main.** Four play sessions, each fixing what it found (see the fix log at the
bottom); Erik's word: *"Now everything seems pretty solid … i think we can
merge this update to main now."*

`TwoPhaseWEGO` is untouched and still the default `--control wego`. Its
retirement + the canon fold are a SEPARATE, deliberate event (design §18) and
did NOT happen here — no golden or digest moved in this merge.

**Design:** `docs/onephase_wego_design_2026-07-28.md` (LOCKED).
**Kickoff + architecture:** `docs/onephase_wego_kickoff_2026-07-28.md`.

**Gate:** `pytest tests -q` → **1961 passed, 22 skipped** (baseline before P1
was 1680 — 281 new tests). No golden or digest was re-baselined at any point.
`TwoPhaseWEGO` remains byte-identical and is still the default.

---

## How to play it

```
python main.py --level wego_test --control onephase
python main.py --level wego_test --control onephase --debug   # dev keys back
```

(Use the conda env `data`'s python, per project CLAUDE.md.)

```
MOUSE  LMB select marine (or apply the armed hotbar slot at the cursor)
       RMB MOVE here — the primary order, no mode needed
       Shift+RMB queue a waypoint      wheel zoom
HOTBAR 1..0 — move, shoot, move&shoot, overwatch, ambush, hold,
       grenade, charge, then your belt items
KEYS   Q swap weapon   X mark   Space SUBMIT (run the round)
       Bksp undo   Esc cancel   Tab cycle marine
       I inventory (DS3 menu)   L flashlight variant   WASD/arrows pan
```

A suggested first session, in the order the design's ideas appear:

1. **Feel the round.** Order a long move and hit Space. Watch the teal path,
   the endpoint footprint, and the arrival timestamp — that number is the
   compiled schedule, not an estimate, and a test pins that the marine arrives
   on exactly that tick.
2. **Feel the seam.** Order a walk longer than 4 s. It keeps going across the
   boundary; nothing resets, nothing snaps.
3. **Feel physical cover.** Put a marine behind a crate and let a zombie shoot
   at him, then step him out. There is no cover stat anywhere — only geometry.
   Shoot a crate long enough and it stops protecting anybody.
4. **Feel the breach.** Breacher plants a charge on the bulkhead door
   (defaults to detonating at t=0 of the next round), everyone else queues
   Ambush on the same target. The volley fires the moment the last man is
   ready.
5. **Feel the cone.** Put someone on Overwatch and narrow the cone — that is
   target control, and its price is a rear that hits harder.

---

## What shipped, by design section

| § | Feature | Where it lives |
|---|---|---|
| 2 | one-phase ~4 s rounds, monotonic clock | `simulation/ruleset.py::OnePhaseWEGO` |
| 3 | time-as-currency: durations, cooldowns, GCD, swap CD | `simulation/timeline.py` |
| 4 | continuous units/bullets/cover; no end-of-round snap | ruleset + `cover_system.py` |
| 5 | the action registry; the v1 verb set | `simulation/action_registry.py` |
| 6 | aim-relative speeds (100 / 70 / 25 %) | `timeline.tile_cadence` |
| 7 | accuracy = spread; physical destructible cover | `timeline.spread_deg_for`, `combat.BulletInFlight.advance` |
| 8 | vision v1: cones, awareness, team union, fog gating | `simulation/vision.py` |
| 9 | overwatch state, cone target control, facing defense | `simulation/engagement.py`, `vision.defense_multiplier` |
| 10 | Ambush readiness barrier | `engagement.update_ambush` |
| 11 | target marking (single focus level) | `timeline.mark_target`, `sim.marks` |
| 12 | scheduled detonations (a moment, not a slot) | `simulation/charges.py` |
| 13 | invisible seams, interrupts, idle stance | ruleset + `timeline` + `engagement.idle_target` |
| 14 | plan invalidation v1 = halt in place | `timeline._advance_movement` |
| 15 | loadout + belt, DS3 menu | `unit.py`, `ui/model.py` |
| 16 | teal viz, timestamps, holograms, hotbar, clock | `ui/` |
| 17 | game mode owns every binding; `--debug` hatch | `control_onephase.py`, `debug_keys.py` |

## Decisions taken during the build (worth knowing before you play)

- **The clock is free-running.** `TwoPhaseWEGO` rewinds `sim.tick` each round,
  which is exactly why it must scrub `last_fire_tick` / `reload_done_tick`.
  §13 demands the opposite, so under this ruleset the tick never rewinds and
  every timer is an absolute deadline. That single choice is what makes
  "invisible seams" free rather than a pile of special cases — and it is what
  lets §12's "detonate at t=0 of the next round" be an ordinary number.
- **The schedule is authoritative for TIME; execution may under-deliver in
  SPACE.** A blocked path costs ground, never the following steps' times. That
  is what makes the displayed arrival times honest.
- **The GCD is charged from a step's start, not its end** — §3 says it gates
  *changing* action, and a weapon's own salvo is one step.
- **A cover LIST (even empty) selects the physical model.** "There is no
  statistical to-hit model" is a property of the ruleset, not of whether a
  level has crates on it.
- **Sustained actions** (shoot, ambush) run until replaced rather than for a
  fixed duration — that is §9's "persists across rounds until replaced" as
  data, and it is why a standing shoot order survives the seam.

## Accepted gaps / where the seams are

- **§14 plan invalidation stays the open question it is in the design.** v1
  ships the stated behaviour (halt in place, continue non-move orders, no
  auto-repath, replan next round) and is deliberately unresolved.
- **Cover art is greybox.** `ui/draw.draw_cover` draws honest rectangles with
  a damage bar; cover is intangible so the tileset cannot draw it. Art pass
  later.
- **The scrub preview is designed-for, not shipped** (§16). Its primitive,
  `ui.position_at`, IS shipped and tested exact against the executor, so the
  slider is a few lines of draw code whenever you want it.
- **Hold-until-t defaults to the end of the round.** Picking an arbitrary
  moment wants the timeline scrubber; the verb and its absolute-tick plumbing
  are complete.
- **No AI opponent behaviour changed.** Zombies still run the shipped
  `ai_zombie` path; this arc is about the player's side of the formula.
- **`TwoPhaseWEGO` is untouched and still the default `--control wego`.** Its
  retirement is a separate, deliberate event at the canon fold, after you
  bless this one (design §18).

## Dials to reach for while playing

All in `config.toml` `[onephase]` (plus `[clock] round_duration_seconds`).
The ones most likely to want moving first, per §20:

`round_duration_seconds` 4.0 · `gcd_seconds` 0.5 · `weapon_swap_seconds` 0.75 ·
`move_shoot_speed_pct` 0.7 · `move_shoot_reverse_speed_pct` 0.25 ·
`spread_move_shoot_mult` 2.0 · `spread_reverse_mult` 3.0 ·
`vision_cone_half_deg` 55 · `awareness_radius_tiles` 4 ·
`overwatch_cone_half_deg` 55 · `flank_damage_mult` 1.5 ·
`overwatch_rear_damage_mult` 1.75 · `ambush_stagger_ticks` 0 ·
`idle_return_fire` true.

Note these are **construction-bound** like the weapon tables: edit, then
relaunch. Ctrl+R alone re-reads config but re-arms nothing.

## If it is blessed

1. Fold as-built into the canon chapters
   (`architecture/mechanics/04_turn_and_control.md` is the one this
   supersedes; `engine/16` for the cover entity).
2. Retire `TwoPhaseWEGO` — ONE deliberate golden re-baseline, with written
   rationale (design §18).
3. Archive the design + kickoff + this doc into `docs/archive/`.

---

## Play-session fix log (what four sessions of Erik playing it found)

Recorded because the pattern is the useful part: every one of these was
invisible to a green suite, and each points at a class of test that was
missing rather than a careless line of code.

| # | What Erik hit | Root cause | Test gap it exposed |
|---|---|---|---|
| 1 | Hard crash on a cover level | `sim.cover` was a PARALLEL list; the bare `EntityInstance` stayed in `sim.entities`, which is what the serializer walks | every cover test ran `enable_recorder=False`, so nothing exercised the serializer |
| 2 | An ordered shot was invisible | §16 covers where a marine will BE, never what an order is aimed AT | the design itself had the hole; no test could have caught it |
| 3 | Orders showed stale times + origin after a round boundary | labels were "seconds since THIS round's start", meaningless for a plan compiled last round; the whole path was drawn, including the walked part | no test read the overlay from mid-round or across a seam |
| 4 | Facing cones looked wrong | flashlights were drawn TWICE — as real light sources AND as translucent sectors | render-layer double-draw is untested by construction |
| 5 | Overwatch cones outlived the posture | the state was never cleared; §9's "until replaced" was unimplemented | tests covered setting overwatch, never ending it |
| 6 | Marines walked through each other | unit-vs-unit collision simply did not exist — `is_passable_block` is terrain only | no test placed two units in one lane |
| 7 | (found while fixing 6) a held unit later teleported THROUGH the body | holding the tick did not hold the path INDEX; the same latent bug applied to knockdown suppression | no test held a unit long enough for the index to run past the obstacle |

Two lessons worth carrying into the next arc:

- **Test the seams the player crosses, not just the units.** Half of these
  (1, 3, 5, 7) are state that is correct at t=0 and wrong later — round
  boundaries, mid-round reads, and long holds are where the bugs live.
- **A "parallel list" of runtime objects is a smell.** #1 happened because
  cover was built beside the entity list instead of into it; doors had the
  right pattern all along.
