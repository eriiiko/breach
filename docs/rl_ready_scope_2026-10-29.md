# RL ready — the scope of milestone #76

**Date:** 2026-10-29 · **Status:** scope set with Erik in session; issues cut
from §5 · **Lead:** Claude (skill `rl-ready-lead`) · **Owner of the
game-design calls:** Erik.

## 1. What "RL ready" means

Erik, 2026-10-29: *"I really meant by RL ready that the game is complete
enough that it makes sense to start building the reinforcement learning."*

So the exit gate is a **match**, not an environment: a seeded ship, two teams
of marines, a result. Agent design (observation, action, reward, per-marine
vs per-team policies, batching, the device unit store) starts AFTER this
milestone and is not pre-empted here. Erik's Berzelius access is bounded by
his PhD (about 3.5 years from today), so the simple version of everything
wins and a nicer version is a follow-up issue.

## 2. The v1 match (decided)

| Item | v1 | Later |
|---|---|---|
| Level | Seeded, simple ship in vacuum: airtight hull, compartments, doors, two mirrored spawn zones, `max_per_team`, vents and ducts auto-placed, zombie and wildlife spawn points | Graph-grammar levelgen (#27) |
| Objective | Eliminate the other team; time limit; draw at the limit | Bomb-site plant and defuse (v1.1) |
| Teams | n per team, a match-config variable within `max_per_team` | — |
| Loadout | Everyone the K5 carbine + 2 frag grenades | One flamethrower per team as a roster variant (after #20) |
| Explosions | The heat-only proxy payload row (#31 open; numbers will move) | #31's design |
| Suits | A per-unit environment-profile row chosen by the match config: suited, unsuited or mixed teams | — |
| Respawn / friendly fire | None / ON | — |
| Opponents | Rule-based team AI (a control source) and zombies (`ai_zombie.py`, a levelgen count); **killed marines convert to zombies** (Erik's ruling 2026-10-29: the legacy conversion stays in) | Trained agents |
| Wildlife | Spawn points only, flag off | Octopus (milestone 8), swarm (#63) |
| Time model | `ContinuousRealtime`, the plain tick loop | — |
| Rendering | Headless: no window, no light requested | Replay viewing (its own milestone) |
| Determinism | One seed drives levelgen, spawn, bots and zombies (Philox, #63 P1 is merged on `63-swarm-units`) | — |

## 3. Built early so both agent shapes fit

Erik intends to compare per-marine policies with shared weights against one
policy per team, and more. Nothing in this milestone may bake in one answer.
Five properties, each gated by a test where it can be:

1. **One control seam.** Every controller is a `ControlSource` emitting
   per-unit intents each tick: the rule-based team AI, the zombie AI, and
   later any policy. Gate: no module outside the seam writes an intent.
2. **Fixed roster slots.** `max_per_team` slots per team, assigned at spawn,
   kept when a unit dies; team size is a mask. Gate: a unit's slot never
   changes within a match; a dead unit's slot is not reused.
3. **Observations per unit and per team through one accessor.** A team view
   is the union of its units' sensor-accessor reads; no global observer in
   the engine.
4. **Credited events.** Every `UnitHitEvent` / `UnitKilledEvent` carries
   attacker (unit or none), victim and cause: bullet, grenade (its owner),
   fire, vacuum, gas, melee. Today `source` is a free string; the audit
   turns it into a typed cause and adds the attacker for environmental
   deaths started by a unit (a grenade's fire, a breach's vacuum). Gate: a
   match in which every death has a cause and, where one exists, an attacker.
5. **Simultaneous actions on one tick.** All units act in the same tick,
   which `ContinuousRealtime` already gives.

## 4. Deliberately out of scope

Observation and action design, reward, episode definition, trainer library,
gym wrapper, batched envs and the device unit store (#29 B1–B4, #77), line
of sight as an observation choice, communication between marines. The swarm
arc (#63: P1 and P2 merged on its branch, P3 designed, P4–P6 not started)
and the octopus (milestone 8). The bomb-site objective. The flamethrower's
feel (#20). The three render issues that sat in this milestone (#6 smoke
graphics, #74 black-body fix, #75 explosion effect) move to the graphics
track: a headless match does not render.

## 5. Work breakdown — the issues

Order is top-down; each is one branch under `autonomous-patch-workflow`.

1. **Match foundation** — a `match` config (teams, n per team, loadout,
   suit profile per team, zombie count, seed, time limit) written and read
   through `level_lib`; fixed roster slots; spawn into the level's zones; a
   headless runner on `ContinuousRealtime` that steps to a result (win, loss,
   draw) and prints it; the seed threaded to every consumer. Gate: two runs
   of one seed give one digest; a run of another seed differs.
2. **Levelgen v1: the ship** — `tools/gen_ship.py` through `level_lib`:
   hull, compartments, doors, mirrored spawn zones, `max_per_team`, zombie
   and wildlife spawn points; passes the airtight lint; A* reachability
   between spawns. Gate: 100 seeds, all airtight, all connected, all mirrored.
3. **Rule-based team AI** — a `ControlSource`: advance toward the last known
   enemy position, shoot on sight, throw a grenade at two or more enemies in
   a room, fall back when alone below half HP. Deterministic from the seed.
   HUMAN-TEST: Erik watches one match in the game window.
4. **Zombies in the match** — spawn from levelgen's count and points, driven
   through the control seam (the zombie AI is wrapped as a `ControlSource`,
   not rewritten). Gate: a match with zombies and no marines firing ends
   with zombie kills credited.
5. **Credited events audit** — §3 item 4. Gate as written there.
6. **Vents auto-placed** — levelgen places one duct loop per compartment
   group with a filter row; depends on #48's residual patches 2 and 3.
   Gate: the generated ship's vents pass the vent system's own load checks
   and ambient airflow is measurable (the vent bench).
7. **Tick budget and map size** — measure ticks per second headless for the
   generated ship at three sizes on both backends; a ruling on the v1 size in
   tiles (at 0.333 m tiles). Instrument: the existing benches; a short table
   on the issue.
8. **Replay milestone** (its own milestone, three issues): the recorder
   grows unit state and credited events alongside the fields (additive to
   `DEFAULT_FIELDS`); a playback mode in `main.py` (a `_parse_*` flag) that
   drives the renderer from a recording; a camera that follows the action.
   HUMAN-TEST.
9. **Exit gate (#76)** — a seeded match, bots vs bots with zombies, runs
   headless to a result, is bit-reproducible, is replayable, and Erik has
   watched one. Then the agent-design session opens.

Prerequisites already on the Roadmap that stay ahead of item 1: the bonfire
tuning job (short), **#73** the thermal books patch, **#116** the standing-
unit stamp drift (a match stamps every marine). **#72**'s measurement batch
and **#31**'s design session stay on the list but do not block the match:
v1 plays on the proxy explosion, knowing its numbers will move.

## 6. Open items for Erik (not blocking)

- Map size at 0.333 m tiles once item 7 has numbers.
- Whether the time limit is wall ticks or sim seconds — I propose sim
  seconds, in the match config.

## 7. Systems

**Existing canonical systems this scope must use** (CLAUDE.md inventory):
Simulation facade and tick conductor (one call line per new system);
Control seams (`ruleset.py` `ContinuousRealtime`, `control_source.py`,
`intents.py`, `action_registry.py`, `timeline.py`); Unit (one class,
`is_zombie` a state); Weapons tables and `DamagePacket`; Level data layer
(`level_lib` writes, `level_loader` reads; the generator precedent
`gen_explosion_studio.py`); Vent / duct system and the filter table; Entity
system and sensor accessor; Recorder (additive `DEFAULT_FIELDS`); Timed
charge (if v1.1's bomb site arrives); A*; `main.py` `_parse_*` flags; the
E2E driver; field digest and GOLDEN_AGGREGATE; the airtight lint; Philox
(`63-swarm-units` P1, to be merged ahead of the match foundation or
vendored the same way).

**New systems this scope creates, with draft one-line rules** (written into
CLAUDE.md at implementation, pointing at the real code):

- *Match config* — `levels/<ship>/match.toml`, written and read through
  `level_lib`: teams, n per team, loadout, suit profile, zombie count, seed,
  time limit. The ONLY way a match is parameterised; never argv scans, never
  per-test dicts.
- *Roster slots* — `max_per_team` fixed slots per team, assigned at spawn,
  never reused in a match; team size is a mask over them.
- *Headless match runner* — `tools/run_match.py`: the ONE way a match runs
  to a result without a window; tests and benches call it, never a second
  loop.
- *Ship generator* — `tools/gen_ship.py` through `level_lib`; every
  generated ship passes the airtight lint and spawn reachability before it
  is written.
- *Bot team AI* — a `ControlSource` subclass; a new behaviour is a rule in
  it, never a second controller path.
- *Credited cause* — a typed `cause` and an `attacker` on every hit and kill
  event; a new damage source adds a cause row.
- *Replay* — the recorder's unit + event rows and `main.py --replay`; the
  ONLY playback path.
