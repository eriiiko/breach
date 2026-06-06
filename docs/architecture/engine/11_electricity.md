# Electricity & Lightning Arcs

**Depends on:** [Material System](03_material_system.md), Fluid & Water (future).

Electricity in Breach is not a continuous field like atmosphere or heat. It is a
**discrete, event-driven effect**: a single electrical arc that fires once, from one
point to one target, deals damage along its jagged path, and is gone within a few
frames. Damaged electronics, an energy-weapon hit on metal, exposed wiring after a
wall breaches, an electrical creature, or — once fluids exist — a live current
finding a flooded floor: any of these *spawns an arc*. Nothing sustains it and
nothing integrates it over time. This is the right shape for the thing being
modelled: real arcs are violent, brief, and discharge in an instant.

The design deliberately separates the three things an arc *is*:

1. a **target choice** — where the current jumps to (path of least resistance),
2. a **visual path** — the jagged bolt the player sees (recursive midpoint displacement),
3. a **damage footprint** — which units the discharge hurts (tiles on or beside the path).

Keeping these orthogonal is what makes arcs cheap and composable. The target search
reads only the material table; the bolt geometry is pure math on two endpoints; the
damage check is a set membership test. None of them touch the physics solvers.


## 1. Triggers — arcs are spawned, never simulated

An arc is created by a game event calling `spawn_arc`, the same way a grenade
detonation calls `apply_explosion`. There is no per-tick "electricity step." The
intended trigger sources are:

- Ship electronics taking damage (an explosion near a control panel)
- An energy weapon striking metal (discharge on impact)
- Exposed wiring revealed when a wall is destroyed
- An electrical weapon or creature firing deliberately
- An environmental hazard (a damaged power conduit)
- **Future:** a current reaching standing water — the bolt conducts through the
  whole connected pool (see §5).

```
spawn_arc(origin, energy, radius, gmap) -> Arc | None
    origin : (x, y)   tile where the discharge starts (the damage event location)
    energy : float    arc strength — sets damage, brightness, visual thickness
    radius : int      max search distance for a target, in tiles
    gmap   : world    read-only; consulted for the material table only
```

It returns an `Arc` (origin, target, energy, a short frame countdown, and a path
regenerated each frame) or `None` when no target can be found within `radius`.


## 2. Target selection — path of least resistance

Electricity seeks the best conductor it can reach. The search scans every tile
within `radius` of the origin and keeps the **highest-priority, then nearest**
candidate. Priority follows real conductivity, expressed through the material
table rather than hardcoded ids:

| Priority | Candidate | Source of truth |
|---------:|-----------|-----------------|
| 0 (best) | Metal tile (hull, steel, machinery) | `material` whose `conductivity` is high (≥ a `metal_conductivity_threshold`) |
| 1 | Water tile *(future)* | the fluid system's wet-tile predicate |
| 2 | A unit | unit footprint occupancy |
| 3 (fallback) | A random nearby air tile | a wild spark when nothing else qualifies |

Ranking by the table's `conductivity` column (hull = 50, steel = 45, wood = 0.15,
air = 0) — rather than testing material ids by name — means a new conductive
material is ranked correctly the moment it is added to the table, with no edit to
the arc code. This is the same data-driven discipline the rest of the engine
follows.

```
find_arc_target(origin, radius, gmap):
    best = None;  best_priority = INF;  best_dist = radius + 1
    for each tile t within radius of origin (Euclidean):
        if   conductivity[t] >= metal_threshold:   p = 0
        elif is_water(t):                          p = 1     # future
        elif has_unit(t):                          p = 2
        else:                                       continue  # non-conductive: skip
        if (p, dist(origin, t)) < (best_priority, best_dist):
            best = t;  best_priority = p;  best_dist = dist
    if best is None:                                          # wild spark
        best = origin + random_offset(min=2, max=radius/2)
    return best
```

The scan is `O(radius²)` tile reads — a handful of cells for the small radii arcs
use. Because it touches only `conductivity` (a static, table-derived cache) and the
unit list, it is trivially cheap and never blocks the physics step.


## 3. Bolt geometry — recursive midpoint displacement

The visible bolt is drawn with **recursive midpoint displacement** — the classic
jagged-lightning algorithm. Given the two endpoints, the midpoint is offset a random
amount *perpendicular* to the line, then each half is subdivided the same way with
the offset halved. A few levels of recursion produce a convincingly forked,
crackling bolt for almost no cost.

```
generate_bolt_path(p1, p2, depth=5, displacement=None):
    if displacement is None:  displacement = dist(p1, p2) / 4
    if depth == 0:            return [p1, p2]
    mid  = midpoint(p1, p2)
    perp = unit_perpendicular(p2 - p1)
    mid += perp * uniform(-displacement, +displacement)
    left  = generate_bolt_path(p1,  mid, depth-1, displacement/2)
    right = generate_bolt_path(mid, p2,  depth-1, displacement/2)
    return left + right[1:]          # drop the duplicated midpoint
```

At `depth=5` the path is ~33 points; at `depth=6`, ~65. Both are negligible to
generate and to draw.

**Regenerate the path every frame the bolt is alive** (typically 2–3 frames), with
the same endpoints but fresh random offsets. This is the entire trick: a static bolt
looks dead, but one whose jitter is re-rolled each frame *crackles*. The `Arc`
object holds only the endpoints, energy, and a frame counter; its `path` is
recomputed on each `update()` and discarded.


## 4. Damage — along and beside the path

Damage is applied to the **set of tiles the bolt passes through, plus their
4-neighbours** (electricity arcs to things it nearly touches). Any living unit whose
footprint overlaps that set takes electrical damage proportional to the arc's energy.

```
apply_arc_damage(arc, units):
    struck = {}                          # set of (x, y)
    for point in arc.path:
        t = round(point)
        struck.add(t)
        struck.add 4-neighbours of t
    dmg = arc.energy * ELECTRICAL_DAMAGE_MULT
    for u in units where u.alive and u.footprint overlaps struck:
        u.current_hp -= dmg              # could also stun / disable equipment
```

This runs in **serial CPU unit logic**, exactly like every other unit-damage path
in the engine (bullets, blast, the planned heat sampling). The arc never writes the
unit list from inside any kernel; it is computed against the unit list directly. The
damage check is a set lookup over ~60 tiles against the unit footprints — cheap even
with many simultaneous arcs (a cascading electrical failure is no problem).


## 5. Water conduction (future — hooked, not built)

The single most dramatic electrical interaction waits on the fluid system. When the
arc's target (or any tile it crosses) is a water tile, the discharge should not stop
at one point — it should **flood-fill the connected body of water and damage every
unit standing in it**:

```
# inside apply_arc_damage, once fluids exist:
if is_water(arc.target):
    wet = flood_fill_water(arc.target, gmap)        # connected wet tiles
    splash_dmg = arc.energy * WATER_CONDUCT_MULT     # lower per-unit, area effect
    for u in units where u.alive and u.tile in wet:
        u.current_hp -= splash_dmg
```

The hook costs nothing until water exists. When it does, arcs become area hazards:
shoot out the aquarium, water floods the corridor, a damaged conduit arcs into the
pool, and everyone standing in it is hit at once. This is the canonical example of
Breach's "systems, not scripts" principle — the behaviour falls out of three systems
(destruction → fluid → electricity) sharing the world, with no scenario-specific
code. The fluid system owns `is_water` and `flood_fill_water`; electricity only
*calls* them.


## 6. Integration with the other systems

An arc is a guest in the world, not a resident. Its couplings are all one-shot
deposits or reads at spawn time:

- **Lighting (Ray Engine).** A bolt is intensely bright for an instant. The arc
  spawns a **one-frame transient light source** at its origin (a muzzle-flash-class
  emitter — bright, blue-white, omnidirectional) that the ray engine picks up like
  any other source. There is no arc-specific lighting path; the bolt's glow is just
  light. (See [Ray Engine](08_ray_engine.md) — energy-weapon glow is handled the
  same way.)
- **Heat / Temperature.** The bolt deposits a pulse of heat into the `heat` buffer at
  its origin and target tiles, feeding the same conduction/ignition pipeline fire and
  beams use (see [Temperature & Fire](06_temperature_and_fire.md)). Plasma is hot.
- **Fire.** If the path crosses flammable material there is a small chance of
  ignition — the heat deposit above can carry a tile past `ignition_temp`, so this
  is mostly automatic once the temperature consumer lands.
- **Smoke.** A bolt crossing smoke ionises it — a faint glow along the path in smoky
  areas. Purely visual; no state change.

These are the *intended* couplings; none are wired yet (the heat consumer and the
energy-weapon pre-phase are themselves still unbuilt — see those chapters).


## 7. Rendering & the sim/render seam

The bolt is **transient visual state with no persistent simulation footprint** once
its damage is applied — exactly the contract of the engine's tick-event channel
(`src/simulation/events.py`). The clean placement is therefore:

- the **simulation** resolves the arc on the tick it spawns: pick the target, build
  the damage footprint, apply unit damage, deposit heat, and emit the transient light
  source — all deterministic, all on integer tile data and `sim.rng`;
- it emits an **`ArcEvent`** (origin, target, energy) onto `tick_events`, the same
  one-shot channel as `ShotFiredEvent` and `ExplosionEvent`;
- the **renderer** consumes the event and owns the *animation*: it generates and
  regenerates the jagged bolt path for its 2–3-frame life, draws the white core and
  blue glow, and manages the fade. The flicker is a render concern; the sim does not
  store or step the bolt path.

This keeps the simulation pure (the damage and heat are deterministic and
serializable; the cosmetic jitter is not), consistent with how every other transient
effect in Breach is split. Determinism note: the *target choice* and *damage* must
draw any randomness (the wild-spark fallback) from `sim.rng`, never the renderer's
RNG; the *visual* midpoint jitter is render-only and may use any source.


## Implementation status

**Design-only. No electricity or lightning code exists anywhere in the engine.**

A grep of `src/`, `cpp/`, and the renderer for `arc` / `lightning` / `electric` /
`spawn_arc` / `bolt` / `conduct` finds nothing but the brief forward-reference stub
in [Ray Engine](08_ray_engine.md) ("Lightning arcs (design, unbuilt)"), which this
chapter is the proper home for. Concretely:

- **No `spawn_arc`, `find_arc_target`, `generate_bolt_path`, or `apply_arc_damage`.**
- **No `Arc` class and no `ArcEvent`.** The tick-event channel exists
  (`events.py`: `ShotFiredEvent`, `ExplosionEvent`, `UnitHitEvent`, …) and is the
  designed home for the transient bolt, but `ArcEvent` is not defined and the
  renderer has no consumer for it.
- **No bolt rendering** of any kind.

**Plumbing that already exists and the design relies on:**

- **Material `conductivity` column** — shipped. `MaterialTable` (`materials.py`)
  carries the `conductivity` scalar (hull = 50, steel = 45, glass = 1, door = 0.3,
  wood = 0.15, air = 0), and the GameMap projects it to a per-tile
  `gmap.conductivity` cache (`_update_caches`, patched on `on_tile_changed`). Target
  ranking by conductivity is therefore implementable today against existing data.
- **Unit damage shape** — units expose `current_hp`, `tile_x` / `tile_y`,
  `footprint`, `team`, and `alive`; the serial CPU damage pattern in
  `combat.fire_burst` (decrement `current_hp`, emit `UnitHitEvent` / `UnitKilledEvent`
  on death) is exactly the pattern `apply_arc_damage` would follow.
- **Deterministic RNG** — `sim.rng` is threaded through the combat path and is the
  source the wild-spark fallback must use.
- **Transient-light hook** — the ray engine accepts arbitrary one-frame
  `LightSource`s, so the bolt-flash integration needs no new lighting machinery.

**Gaps / blocked-on:**

- **Water conduction (§5) is blocked on the fluid system**, which is itself designed
  but unbuilt (see Fluid & Water). `is_water` and `flood_fill_water` do not exist;
  the hook is written here but cannot be wired until fluids land.
- **Heat / fire / smoke couplings (§6) are blocked on their consumers.** The `heat`
  buffer is deposited into but **read by nobody** (no temperature field, no ignition
  pass — see [Temperature & Fire](06_temperature_and_fire.md)), so an arc's heat
  pulse would currently fall on the floor. Arc → ignition is automatic *only once*
  the temperature consumer exists.
- **`ELECTRICAL_DAMAGE_MULT`, `WATER_CONDUCT_MULT`, the metal-conductivity
  threshold, default `radius`, `energy`, bolt `depth`, and the frame lifetime are
  unspecified tunables.** Per the engine's data-driven rule they belong in
  `config.toml` (a `[weapons.arc]` or `[physics.electricity]` section), not as
  hardcoded constants — there is no such section yet.
- **Energy-weapon → arc trigger** depends on the energy-weapon pre-phase
  ([Ray Engine](08_ray_engine.md)), which is also unbuilt; until then the only
  realistic trigger is an explosion-adjacent event.
