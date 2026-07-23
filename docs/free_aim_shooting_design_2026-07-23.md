# Free-aim directional shooting design (2026-07-23)

**Status: v0.2 — decisions LOCKED with Erik 2026-07-23 (§9). Ready for the
dedicated build session Erik asked for ("its own session to get it right"). No
code yet. Aim model = PURE FREE-AIM (Erik's call).**

Prompted by the first P3 controller human-test (2026-07-23): under direct
gamepad control the possessed marine could move and aim but **could not shoot**.
This doc is the action-variant firing model. WEGO's *targeted* fire is not
touched — the whole point of the modularity split
(`docs/control_modularity_design_2026-07-22.md`) is that the two schemes coexist.

---

## 1. Goal

The action variant needs **free-aim directional fire**: face a direction (right
stick / an enemy in that direction) and shoot along it, hitting the first thing
the shot actually crosses. Today the engine only knows **targeted fire** — an
order names a destination *tile* `(tx, ty)` and the shot is gated on
line-of-sight to that tile. That model is correct for WEGO (you click a target)
and wrong for twin-stick action.

Constraints that shape everything below:

- **WEGO stays byte-identical.** `TwoPhaseWEGO` + its targeted fire + the
  digest/golden gates are frozen. Directional fire is *additive*.
- **Many weapons.** The model must dispatch per weapon *archetype*
  (hitscan / projectile / spray / melee / lobbed / placed) and stay extensible —
  this is why it deserves a design pass, not a patch.
- **Determinism iron rules.** Facing and every march angle live behind the
  Q16.16 integer trig kit (`unit_fixed` → `breach_physics`); no libm on synced
  state (the X-ARCH Ada finding). `facing` is a *hashed* synced field.

## 2. Why P3 shooting fails today (the band-aid to replace)

`Simulation._aim_fire_order` (`simulation.py:640-666`) fabricates a **targeted**
`Order` from the unit's facing: it projects the facing angle out to *exactly*
`weapon.range_tiles`, rounds to the nearest tile, clamps to the grid, and hands
that single point back into the point-target pipeline. Then
`combat.process_shooting` runs its normal **range + LOS pre-gate**
(`combat.py:689-699`): `has_los(shooter, that_endpoint_tile)`.

Consequences (all real, from the engine map):

1. **Silent no-fire.** If anything occludes the *nominal endpoint tile* — a wall
   at 3 tiles when the weapon's range is 60, or the map edge — LOS fails and the
   unit fires at nothing, *even though an enemy is right in front of it*. On the
   playground this is the likely cause of "shooting didn't work."
2. **Only ever aims at max range.** It never "fires along the ray hitting
   whatever's there"; it fires *at one far tile*.
3. **Double quantization.** A continuous, already-exact facing is collapsed to
   `int(round(...))` tile coords, then `fire_burst`/`fire_beam` immediately
   *re-derive* an angle from that tile pair — a round-trip that injects
   quantization noise into an exact direction.
4. **`phase` landmine.** The fabricated order stamps `phase = tick //
   ticks_per_phase`, which under continuous time is an unbounded "5-second
   bucket", not a WEGO 0/1 phase id. Harmless only because the order is never
   stored — a trap for the next reader.

The fix is not to patch this — it's to stop fabricating a tile target and fire
**directionally** from `(origin, facing)`.

## 3. The one lucky fact: the march is already angle-driven

The projectile and hitscan resolvers do **not** actually march toward a tile —
they march along an **angle**, which they currently *derive* from the tile
target at the top of the function:

- `fire_burst` (projectile): `base_angle = atan2_rad(fy2-fy1, fx2-fx1)`
  (`combat.py:835`), then `BulletInFlight.advance` marches by kit-trig step
  vectors, stopping on the first solid / unit / max-range it crosses.
- `fire_beam` (hitscan): same `atan2_rad` derivation (`combat.py:920`), then a
  tile-by-tile march for `range_tiles`, skewering units, stopping on solid.
- `spray` computes its cone bearing from a target tile via `atan2` too
  (`combat.py:1078`), but the cone-membership test is already purely angular.
- Grenades (`_spawn_direct_grenade`, `simulation.py:668`) *already* throw along a
  continuous direction with no tile detour — the existing precedent for a clean
  directional weapon path.

So the tile target is load-bearing in exactly **two** places: (a) the `atan2`
angle derivation at the top of each resolver, and (b) `process_shooting`'s
range+LOS pre-gate. Everything downstream is angle/vector math already.

## 4. The design — a directional fire path parallel to the targeted one

### 4a. Core: an `aim_angle` seam through the existing resolvers

Give `fire_burst` / `fire_beam` (and the spray cone) an optional
**`aim_angle` (Q16.16-exact radians)** parameter:

- `aim_angle is None` → **today's behavior verbatim**: derive the angle from the
  order's tile target. WEGO calls this way → byte-identical, goldens unmoved.
- `aim_angle` provided → skip the tile→angle derivation and march straight from
  `(origin, aim_angle)`. No round-trip quantization; the exact facing is used.

This is the minimal, byte-safe seam. (A shared `fire_ray(gmap, units, shooter,
origin, angle, weapon, ammo, ...)` primitive extracted from the common march
body is a nice-to-have refactor, but only if it comes out byte-identical for the
WEGO path — otherwise keep the optional-param form.)

### 4b. Dispatch: a directional trigger branch in `process_shooting`

A new branch parallel to the targeted one (`combat.py:671-720`) that fires when a
possessed unit holds TRIGGER under `ContinuousRealtime`:

- **Bypasses the range+LOS pre-gate entirely** — a directional shot always fires
  (subject to cadence / mag / status gates), and *range and hit are the march's
  job*: it stops on the first solid/unit or at `range_tiles`. This is the whole
  correctness fix from §2.1.
- Reuses the existing cadence gate (`tick - last_fire_tick < rof_interval_ticks`),
  ammo/mag gate, and status gate unchanged.
- Reads the aim from **`u.facing`** (already set every tick by the `AIM` intent,
  `simulation.py:549` / `intents.py:62`) — **no new intent type needed**. The
  `AIM`/`MOVE_DIR`/`TRIGGER` vocabulary from P3 §3c is sufficient.

### 4c. Per-archetype behavior under free-aim

| Archetype | Free-aim behavior | Effort |
|---|---|---|
| **projectile** (bullets: k5, pdw, lr50…) | march along facing + existing per-bullet cone spread; hit first unit/solid | small — `aim_angle` param |
| **hitscan** (beam: lance-3/5) | instant march along facing, skewer units, stop on solid | small — `aim_angle` param |
| **spray** (dragon, miasma) | cone bearing = `shooter.facing` instead of target-derived | trivial — swap the bearing source |
| **lobbed** (grenade) | already directional (`_spawn_direct_grenade`) — reuse | done |
| **placed** (breach charge, c4) | placed at feet / adjacent via USE-context, not a trigger ray | later, USE path |
| **melee** (knife, baton) | adjacency — **out of scope**, owned by the deferred melee arc | excluded |

Melee is explicitly the separate melee/block/parry/grab/stamina arc (modularity
§3c) — free-aim v1 does not touch it.

### 4d. Aim & trigger feel

- **TRIGGER held** → auto-fire at the weapon's cadence (`rof_interval_ticks`),
  `shots_per_trigger` / burst semantics reused as-is.
- **Spread** is the existing cone (`spread_deg` when aimed) centered on facing;
  per-bullet offset drawn from the seeded RNG exactly as today (deterministic).
- Moving-while-firing accuracy, aim-drift, recoil, etc. are **feel dials**, not
  v1 structure — parked as tuning once it plays.

## 5. Aim model — the pivotal open question (§9 Q1)

Erik's phrase was "target an enemy and shoot in its direction." Two readings, and
this choice shapes 4b/4d:

- **A — pure free-aim.** The shot goes exactly along `u.facing`; you hit whatever
  the ray crosses. Standard twin-stick. Simplest, most honest, most
  agent-friendly (the RL policy's action space is a clean continuous angle).
- **B — soft-lock / aim-assist.** Aiming *near* an enemy biases the shot toward
  it (snap or magnetism). Better console feel with a stick; more code; a
  non-trivial determinism surface (target selection must be deterministic); and
  it complicates the eventual agent action space.

**DECIDED (Erik, 2026-07-23): A — pure free-aim for v1.** The correct, minimal,
deterministic core; cleanest agent action-space. B (aim-assist) stays an optional
later polish layer *on top of* A, not part of this session.

## 6. Determinism

- `u.facing` is already a hashed synced field (`field_ab_harness.py:69`),
  quantized before hashing, and only ever written via the integer trig kit — so
  consuming it for fire introduces no new nondeterminism.
- Cone spread RNG is the sim's single seeded generator; the **lazy-roll rule**
  (a roll that can't matter is never drawn) must be preserved so zero-spread
  weapons don't shift the RNG stream.
- Only `UnitHitEvent` / `UnitKilledEvent` are digest-hashed; `ShotFiredEvent`
  etc. are cosmetic. Any new per-unit aim/fire state must be a *derived*,
  non-hashed field (the `last_fire_tick` pattern) — never independently hashed.
- The math/screen convention split (facing is y-up; march angle is y-down, with
  an explicit sign-flip at each fire site, `combat.py:842` vs `835`) must be
  gotten right in both directions or crit-arc classification silently inverts.

## 7. WEGO coexistence & the gate

- WEGO's targeted path calls the resolvers with `aim_angle=None` → the tile→angle
  derivation runs exactly as today → **digests/goldens byte-identical**. This is
  the hard gate on every patch, same as P1–P3.
- Directional fire is reachable only under `ContinuousRealtime` + a possessed
  unit, so it is dormant under `--control wego` (the P3 dormancy discipline).

## 8. What this replaces / retires

- `Simulation._aim_fire_order` is **deleted** — no more fabricated targeted
  orders. `_consume_direct_intents`'s TRIGGER handling calls the directional fire
  path with `u.facing` instead of building an `Order`.
- The unbounded-`phase` landmine goes away with it.

## 9. Decisions (locked 2026-07-23)

1. **Aim model (§5): PURE FREE-AIM (A).** Erik's call. B (aim-assist) is later
   polish on top, not this session.
2. **Melee:** stays with the deferred melee arc — **out of scope** here.
3. **v1 archetypes:** projectile + hitscan + spray + (existing) grenade. Placed
   charges (breach/c4) via the USE path are **later**, not this session.
4. **Fire-while-moving:** **flat accuracy in v1**; movement/recoil/aim-drift
   penalties are dials tuned later once it plays.
5. **Crash:** still open/empirical — the free-aim rewrite deletes the suspect
   `_aim_fire_order` path, so it may fall out; confirm with a repro in the build
   session. (Tracked in `docs/TODO.md` action-variant items regardless.)

## 10. Build plan (for the dedicated session, after §9 is locked)

Following the arc pattern (design → critique → gated patches → HUMAN-TEST):

- **F1 — `aim_angle` seam** through `fire_burst`/`fire_beam`/spray bearing.
  `aim_angle=None` byte-identical. GATE: goldens byte-identical + new unit tests
  proving `aim_angle` reproduces the tile-derived angle within quantization.
- **F2 — directional trigger branch** in `process_shooting` (bypass range/LOS
  pre-gate; march resolves range/hit) + wire `_consume_direct_intents` to it;
  delete `_aim_fire_order`. GATE: WEGO byte-identical (dormant); new headless E2E
  test — possessed unit facing an enemy at 3 tiles with a 60-range weapon *hits
  it* (the §2.1 regression).
- **F3 — per-archetype pass** (spray bearing, grenade already done, confirm
  hitscan skewer) + the P3 E2E harness extended to assert fire for each
  archetype. HUMAN-TEST: Erik plays — shooting feels right across weapons (N
  cycles weapon).
- Then P3 becomes a complete, mergeable direct-control slice.

**Escalation triggers for the build session:** (a) the `aim_angle=None` path
moves any golden → stop, it's a bug not a re-baseline; (b) directional fire needs
a new synced/hashed field beyond `facing` → stop and check determinism with Erik;
(c) aim-assist (if chosen) makes target selection nondeterministic → stop.
