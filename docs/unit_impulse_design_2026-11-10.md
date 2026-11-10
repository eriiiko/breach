# The unit impulse law — design (draft 2, 2026-11-10)

Patch **G5p** of the gameplay pack (#183; pack design
`docs/gameplay_pack_v1_design_2026-11-10.md` §5). Research:
`docs/research/impulse_on_a_person_2026-11-10.md` ("the search"),
`docs/research/rotation_litcheck_2026-11-10.md` ("the check"),
`docs/research/wind_on_a_person_2026-10-31.md`.

**RULED** = Erik's rulings of 2026-11-10; the rest is Claude's design.
Draft 2 replaces draft 1's §5 law after critique lens A (§12): the reaction
delay no longer switches the footing off, an in-place sway absorbs sudden
input, effort reads the demand, the turning law is an exact discrete one.

## 1. Rulings this is built on

- ONE law for everything that pushes a unit: shove, hits, wind, blast.
- Forces act through mass; each unit has its own mass.
- Soft wind never displaces a unit; below a limit a force only slows. A
  sustained force ABOVE the footing limit displaces by its excess.
- Bullets push with their real momentum (a declared multiplier ships at 1).
- Units have angular inertia: for their own turning, and from outside
  forces.
- The wind bands are restated as force over body weight.
- A staggering unit can still shoot. Recoil goes through the law. The
  shover feels no reaction in v1 (ACCEPTED GAP). A shove works on any team.
  Impact damage and bouncing off walls are later. A zombie is a state: a
  copy of the unit it infected, with a hold multiplier that ships at 1.
  Two patches.

## 2. The idea in one paragraph

Everything that pushes a unit hands it an IMPULSE (N·s) at a point on its
body. Once per tick the unit sums what it received and turns the sum into a
velocity change (over its mass) and a spin change (lever arm over its moment
of inertia). Its feet resist up to a limit per tick. A sudden push is first
absorbed IN PLACE, as sway, up to a small capacity; only what overflows the
sway becomes a pushed velocity that moves the unit. How hard the feet are
working slows the walk. Past a fall limit, or when the force is more than
the feet can ever hold, the unit goes down. Wind is the same thing: a force
is an impulse every tick.

## 3. State per unit (all integer)

| state | meaning | zero nearly always |
|---|---|---|
| `imp_x`, `imp_y`, `imp_l` | impulses received since the last integration: linear (N·s) and angular (N·m·s) | yes |
| `swx`, `swy` | the in-place sway: velocity being absorbed without a step | yes |
| `pvx`, `pvy` | pushed velocity (moves the unit) | yes |
| `omega` | angular velocity | when not turning |
| `facing_q` | facing, in the angle kit's own currency (`turning.py`: Q16 radians) | no |
| `effort` | low-passed demand on the footing, Q16, 0 .. 2·ONE | yes |
| `react_ticks` | reaction delay after an overflowing push | yes |

Parameters, per species / env-profile / loadout row (reference marine):

| parameter | value | note |
|---|---|---|
| mass `m` | `Unit.mass` + `[loadouts.*] kit_kg` (80 + 20) | individual mass exists already; the door refuses m ≤ 0 |
| drag area `cd_area` | 0.9 m² standing; 0.15 m² prone (an estimate) | wind note; the check found no prone measurement |
| hold `k_hold` | 0.2248 of body weight, standing | chosen so the fall edge is exactly today's 245 Pa for the reference marine |
| slow start `k_slow` | 0.036 of body weight | = today's wind slow edge (39 Pa) |
| prone hold `mu_slide` | 0.4 of body weight | search §4, an assumption |
| in-place capacity `v_inplace` fwd / side / back | 0.25 / 0.20 / 0.15 m/s | search §1 |
| fall limit `v_fall` fwd / side / back | 1.5 / 1.1 / 0.8 m/s, × `stability` | only fwd is measured |
| reaction delay `t_react` | 0.30 s (7 ticks) | search §1 |
| recovery horizon `n_rec` | 24 ticks | what a unit can still walk off (lens A, B2) |
| yaw inertia `I` | 3.0 kg·m² | the check; the door refuses I ≤ 0 |
| own turning: top speed, acceleration | 360°/s, 30 rad/s² | the check: 90° in 0.46 s, 180° in 0.73 s |
| body radius `r_body` | 0.25 m | every lever arm is clamped to it |
| `bullet_push_mult`, `bullet_twist_mult` | 1.0, 1.0 (RULED: real momentum) | declared experiment dials |
| zombie hold multiplier | 1.0 (RULED) | |

"Forward" is a push along the unit's facing (from behind). The limit for any
direction blends the three values by the cosine of the angle between the
push and the facing, through the kit's cos. The direction is judged each
tick against the current facing (a unit that turns toward its shover braces
worse backward: accepted, stated).

The door refuses a configuration where one tick of hold (`cap`, below) is
not smaller than the smallest `v_inplace` (at 24 Hz: 0.092 against 0.15).

## 4. The one entry

`unit_impulse.add(unit, jx, jy, lever)` is THE way anything pushes a unit: it
adds to `imp_x`, `imp_y` and (lever arm clamped to `r_body`) `imp_l`. Sums of
integers commute, so the result never depends on the order or the ids of
whoever pushed.

| input | impulse | lever arm |
|---|---|---|
| a round that hits | its momentum (`[ammo.*]` gains `bullet_g`, `muzzle_mps`) × `bullet_push_mult`, along its flight | the perpendicular distance from the unit's centre to the bullet's line, from the march's own geometry: no random draw |
| a shove (`shove_intent`) | `[weapons.shove] impulse` (PLACEHOLDER 100 N·s; later scaled by strength), along the shover's facing, on the adjacent unit of any team nearest that facing | the offset of the shover's centre line from the target's centre |
| a melee strike (G6p) | its row's `impulse` (small) | as the shove |
| recoil (RULED) | minus the fired round's momentum, on the shooter | none |
| wind (patch b) | `q · cd_area · dt` along the wind at the footprint's highest-q tile | none in v1 (ACCEPTED GAP: wind does not turn a unit) |
| blast (patch b) | the blast's own wind through the wind row, nothing else, so nothing is counted twice | none |

A hit is integrated in the NEXT tick's movement slot (a one-tick delay,
stated).

## 5. The integration, once per tick (in the movement slot)

`cap = k_hold · g · dt` is one tick of hold, a velocity (0.092 m/s; the same
for every mass). Prone: `cap_prone = mu_slide · g · dt`.

**Standing.**

```
1. dv = (imp_x, imp_y) / m ;  dw = imp_l / I ;  clear the accumulators
2. demand = min(|dv| / cap, 2·ONE)
   effort += (demand − effort) / 8, rounded AWAY from zero (so it reaches its input)
3. sway:  sw += dv
   if |sw| > v_inplace(direction of sw):
        the overflow (sw − v_inplace·ŝw) moves into pv ;  sw keeps v_inplace·ŝw ;  react_ticks = t_react
4. resist, with a budget of cap:
        sw shrinks toward zero by min(|sw|, budget)        EVERY tick, reaction delay or not
        margin = what is left of the budget
        if react_ticks == 0:  pv shrinks by min(|pv|, margin)     else  react_ticks −= 1
   (magnitude form: integer isqrt, components rounded toward zero, exactly zero when it is all used)
5. walking scale from effort: ONE below k_slow/k_hold, linear down to 0.5 at ONE, clamped;
   walking is suppressed while react_ticks > 0 (the unit is catching itself; it can still shoot)
6. KNOCKED_DOWN if any of:
        |pv + sw| >= v_fall(direction) · stability
        effort >= stability²                               (the wind law's integer compare)
        react_ticks == 0 and |pv| > n_rec · margin         (it cannot walk this off)
7. move by pv ONLY (the sway never moves a unit), together with the walk, as ONE displacement
   in substeps of at most half a tile per axis; stop at the first blocked substep;
   zero only pv's component into the wall, and book it
8. turn (§5.1)
```

**Prone (KNOCKED_DOWN).** No sway, no reaction delay: all input meets
`cap_prone` at once and what is left moves the body. The knockdown is
refreshed while the demand the unit WOULD have standing is at least
`stability²`, or while `|pv| > 0`: nobody gets up into a gale or while
sliding. The pushed velocity moves a unit whatever its statuses say about
moving on its own.

**Why soft input can never move a unit.** Only `pv` moves a unit, and `pv`
is fed only by sway overflow. A steady force below the hold adds less than
`cap` to the sway each tick and step 4 removes all of it, every tick, with
no exception for the reaction delay. Any train of pushes whose per-tick
`|dv|` stays within the in-place capacity and whose mean is below the hold
leaves `pv` at exactly zero. This is a clamp, not a tuning, and it is the
same whether five shooters fire on the same tick or on different ones.

**A force above the hold** overflows the sway within ticks, pushes by its
excess (RULED), and floors the unit through condition 2 or 3 within ticks;
prone, the body's smaller area and better hold stop it unless the wind is
far stronger.

**Momentum is booked.** `m · Δ(pv + sw) = Σ impulses − resisted − wall`,
exactly, in integers, per unit per tick.

### 5.1 Turning (an exact discrete law)

Per tick, in the facing's own integer currency, with `a` the unit's own
angular acceleration per tick² and `w` its top speed per tick:

```
omega += dw                                   (outside spin; |omega| clamped below half a turn per tick)
B(x)  = distance covered braking from speed x at a per tick, this tick included
        = (n+1)·x − a·n(n+1)/2,  n = ceil(x / a) − 1
s(e)  = the largest speed in [0, w] with B(speed) <= e        (try n = 0 .. ceil(w/a); exact integers)
e     = wrap(target − facing)
omega = clamp( sign(e)·s(|e|),  omega − a,  omega + a )
facing += omega
```

It reaches a fixed target exactly, with no overshoot and no limit cycle
(lens A, M3, gives the proof: once `|e| <= a` the next step equals `e`).
The top speed bounds the unit's OWN torque only: an outside spin above it
decays at `a` per tick. EVERY facing target goes through this law: the aim,
the travel direction of a unit holding no aim, the zombie's walk, and the
attack resolve's facing snap. Nothing writes the facing directly any more.
G1p's bare cap is this law with unlimited acceleration.

**What a hit does to the aim, stated honestly.** A carbine round at the
body's edge is 0.3 rad/s, below what the unit's own torque cancels within
the tick: the nudge is zero in the game and 0.09° in physics. Only a heavy
round (.50: about 6°) or a shove (tens of degrees) twists an aim under real
momentum. `bullet_twist_mult` (shipped 1.0) is the declared dial if a
visible twist from small arms is wanted; that is Erik's to turn.

## 6. What it gives

Reference marine 100 kg (hold 220 N); zombie 80 kg (hold 176 N), stability
0.9 (falls at effort 0.81). Carbine: 3.6 N·s per round, 12 rounds/s, 43 N.

| input | on the marine | on the zombie |
|---|---|---|
| one carbine round | nothing | nothing |
| one shooter | effort 0.20: 2 % slower | 0.25: about 5 % slower |
| two shooters | 0.39: 13 % slower | 0.49: 19 % slower |
| three shooters | 0.59: 25 % slower | 0.74: 34 % slower |
| four shooters | 0.79: 37 % slower | 0.98: down |
| five shooters | 0.98: half speed, at the edge of falling | down |
| shove from behind (100 N·s) | 1.0 m/s: stumbles about 0.45 m over 18 ticks, cannot walk for 7; off-centre it is spun by tens of degrees | 1.25 m/s: stumbles farther |
| shove from the front | over the 0.8 m/s backward limit: knocked down | down |
| firing the carbine (recoil) | 2 % slower while firing | — |
| a .50 hit | staggers about 0.1 m, twists the aim about 6° | staggers |
| wind (patch b) | today's bands exactly | a lighter body is slowed and floored sooner |
| blast wind (patch b) | flung by the excess, down, slides | same |

In-phase and out-of-phase fire give the same result.

## 7. Two patches

- **G5p-a**: the law, the entry, shove, hits, recoil, the turning law with
  every facing writer on it. Wind keeps its present row. In this patch the
  wind's scale and the effort scale multiply and the two knockdowns do not
  add (a known, temporary inconsistency).
- **G5p-b**: wind and blast move onto the law. The wind row's scale member
  and its knockdown are DELETED in the same commit that wind enters `add`
  (or half a hold of wind would be counted twice): that is a gate. Its gates
  also include #140's own reproduction (the 1.44 atm room must not move the
  squad, with and without a teammate's shove in the doorway) and equal bands
  for the reference marine.

## 8. Gates (properties)

- Any input train whose per-tick `|dv|` stays within the smallest
  `v_inplace` and whose mean is at most the hold: position and `pv` exactly
  unchanged, for any duration, any phasing, with or without an earlier
  stagger. (A steady force is the special case.)
- A steady force above the hold floors a unit of ANY stability within a
  bounded number of ticks.
- The same rounds delivered in phase and out of phase give the same position
  and the same band.
- The momentum identity of §5, every tick, wall term included.
- Permuting the shooters' ids changes nothing.
- Displacement is monotone decreasing in mass; a pushed unit never ends
  inside a wall and never crosses one, at any speed the law can produce.
- A prone unit under sustained fire comes to rest; nobody stands up while
  sliding or in an over-hold wind.
- For a FIXED target and no outside torque the facing reaches it exactly,
  never overshoots, never exceeds the top speed, and then stays; after an
  outside spin it returns and settles exactly.
- No code path writes a unit's facing except the turning law (a scan test,
  like the footprint-samples one).
- With every input switched off and unlimited turning acceleration, all
  pins unchanged.

## 9. Left for Erik's play test (decided here as placeholders)

- Walking is suppressed during the reaction delay (0.3 s) after a push that
  overflows the sway. He ruled only that such a unit can shoot.
- Small arms do not visibly twist an aim at real momentum
  (`bullet_twist_mult` = 1.0).
- The backward and sideways limits are modelled, not measured.

## 10. Not in v1

The shover's reaction; wind turning a unit; impact damage and bouncing off
walls; turning acceleration capped by deck friction; a prone body's drag
depending on how it lies; units colliding with each other (a shoved unit
passes through others, as walking units do today, #142).

## 11. Systems

(a) Uses: Unit (`mass`, `stability`), Speed table and `default_speed` (the
walking scale composes there), Turn-rate cap (`turning.py`: this law
replaces its body and keeps its kit), Wind load on a body + the Footprint
samples builder, status rows (KNOCKED_DOWN), Simultaneous attacks (hits and
the facing snap record at the resolve), the bullet march's geometry,
Fixed-point kits, the collision predicate walking uses, Loadouts, Weapons
tables, Control seams (`shove_intent`: `INTENT_KINDS`, `VERB_RULES`, the
tap, a format bump), Direct control (Space), Config, Field digest.

(b) Creates: **Unit impulse** (`src/simulation/unit_impulse.py`): THE way
anything pushes or turns a unit is `add(unit, jx, jy, lever)` and the one
per-tick integration; never a position or facing write from a force; a new
pusher is a caller, a new response is a term of the integration. **Turning
law** (`turning.py`): THE only writer of a unit's facing; a new reason to
face somewhere is a target handed to it.

## 12. Critique lenses

| lens | critic | verdict on | result |
|---|---|---|---|
| A behaviour and failure modes | Opus, 2026-11-10 | draft 1 | BUILD AFTER THE LISTED CHANGES. 4 blockers: the reaction delay switched all footing off (a .50 hit plus a wind just under the hold slid a marine 23 m: #140 again); the saturated-effort knockdown could never fire at stability 1.0; in-phase and out-of-phase fire gave opposite results; a prone body could be flung without end. 10 majors, among them: the turning step as worded chattered ±1.3° forever; sub-hold bursts made a target creep; lever arms unclamped; fast pushes could tunnel; three other facing writers bypassed the inertia; several §6 numbers did not follow. **All adopted in draft 2** (§5, §5.1, §6, §8), with its corrected law. Its two modelling calls are §9's first two lines. Full text: `docs/research/impulse_critique_A_behaviour_2026-11-10.md` |
| B determinism and integration | Opus, 2026-11-10 | draft 2 | BUILD AFTER THE LISTED CHANGES. 4 blockers: the momentum identity cannot be exact with a rounded division and a float mass; the movement slot never reaches zombies, knocked-down units or units holding no move; the turn placed after the move breaks the Speed table's "this tick's facing" rule and the bots' trigger predictor, and the non-aim facing writers have no stored target; the dormancy gate was false as written. 12 majors on currencies, rounding (Python floors, C truncates), a missing velocity bound, a fifth effect kind, where the effort scale lives, digest and recording rows. **All adopted: §13 makes its resolutions binding.** Full text: `docs/research/impulse_critique_B_integration_2026-11-10.md` |

## 13. Build sheet (binding; from lens B)

Lens B's resolutions and its "Implementer's sheet" are part of this spec.
Where §3 to §8 above and that sheet disagree on representation, placement or
order, THE SHEET WINS; the behaviour of the law (§5, §5.1) is unchanged. The
decisions, in short:

- **Currencies.** Velocities are Q16 tiles per tick, impulses Q16
  gram·tiles per tick, angles Q16 radians (the kit's). Mass is integer
  grams. Every real-unit dial is quantized ONCE in a per-map `ImpulseTable`
  built beside `speed_table`. Division truncates toward zero through one
  helper (`unit_fixed.div_trunc`, equal to C's `/`), and the REMAINDER of
  `J / m` and `J·r / I` stays in the accumulator, so the momentum identity
  closes exactly in int64. Magnitudes use one integer square root
  (`unit_fixed.isqrt`, equal to the kit's `sqrt_q16`). A velocity bound
  `V_MAX` clamps `pv`, booked as its own term of the identity; the door
  bounds the substep and `s(e)` trip counts.
- **Positions** stay float until the residency patch P3c, but every position
  this law (and the walk beside it) writes lies on the Q16 grid, through one
  Q16 move function shared by every path, so P3c is a change of storage
  only.
- **A new conductor slot 1a, "integrate + turn"**, before the intents: for
  every living unit in id order, the integration's steps (accumulators with
  remainder → demand and effort → sway and overflow → resist → the knockdown
  checks → `omega += dw` → the turn). So every impulse is integrated exactly
  one tick after it lands, whichever slot produced it.
- **The displacement runs in slot 3** as ONE loop over every living unit: a
  walker's walk and its `pv` as one displacement; every other unit with
  `pv ≠ 0` (standing, prone, zombie, a legacy WEGO / onephase unit) moves by
  `pv` alone. A zombie displaced this tick drops its path (its next step
  snaps to the tile grid: an accepted gap until the horde field, G9p). WEGO
  and onephase plans overwrite a push (accepted legacy gap).
- **Facing.** `facing_q` (Q16 radians) is the synced representation, with
  `facing` as a read-only float property for the renderer and the pose
  layer. A persistent `facing_target_q` per unit; EVERY writer becomes
  `turning.set_target` (the aim, the travel direction, the zombie walk, the
  attack snap as a recorded effect, WEGO, onephase, spawn). Only
  `turning.step`, called from slot 1a, writes `facing_q`.
  `Simulation.facing_after_turn_q16` becomes the pure evaluation of exactly
  that step, so the bots' trigger rule stays exact. The Speed table's
  direction rule keeps reading the post-turn facing, and its "no aim held
  means θ = 0" shortcut is deleted (a walker holding no aim no longer faces
  its move instantly).
- **`SimultaneousEffects` gains a fifth kind, `push`**, recorded right after
  its `hit` and applied in record order; recoil records through it too; the
  shove at 1b adds directly (sums commute).
- **The effort scale** is a member of `FootprintSamples`
  (`effort_scale_q16`, composed in `default_speed` with the same
  skip-at-ONE guard as wind: wind first, then effort). **The walk ban**
  during the reaction delay is one predicate, `unit_impulse.walk_allowed`,
  read at the three walk sites.
- **Dials for the gates:** `bullet_push_mult`, `bullet_twist_mult` and a
  `recoil_mult` (all 1.0).
- **Digest and recording:** the new state is hashed only when non-zero
  (absence-transparent keys); `shove_intent` gets its `INTENT_KINDS` and
  `VERB_RULES` rows, a `last_shove_tick` cooldown (`[weapons.shove]
  cooldown_seconds`, PLACEHOLDER 1.0) and a format bump; impulse state is
  zeroed at death and at conversion. A render-only `ShoveEvent` /
  `StaggerEvent` for the animation session (#184).
- **Which hits push:** every projectile and pellet that stops on a unit
  pushes with its own row's momentum; beams do not; a payload round (the
  40 mm) pushes with its body's momentum, its blast is the wind's business.
- **Gates:** lens B's G1 to G15 replace §8's dormancy and in-phase gates
  (G5: the final position is exactly equal when no overflow occurs; the
  band is compared as mean effort over a period, not per tick).

**Three implementer runs, each merged on green:**

1. **G5p-a1, turning.** `facing_q` / `facing_target_q`, `turning.set_target`
   / `step` / `predict` (§5.1), every facing writer routed, slot 1a holding
   only the turn. First commit: the representation and the routing with
   unlimited acceleration (`a ≥ 2w`), re-pinned on its own with the cause
   proven; second: the shipped 30 rad/s², re-pinned. Gates G2, G3, G9, G10.
2. **G5p-a2, the linear law.** The table, the accumulators, `add`, the rest
   of slot 1a, the slot-3 displacement loop, `push`, hits, recoil, the shove
   verb and its key (Space), the effort member, the walk ban, the events.
   Gates G1, G4 to G8, G11 to G15.
3. **G5p-b, wind and blast** onto the law (§7).


## As built (G5p-a1, 2026-11-10)

Run 1 of 3, the turning law, built to §5.1 and §13 in two commits of
behaviour. `src/simulation/turning.py` is the only writer of a facing:
`Unit._facing_q` (Q16.16 radians; `Unit.facing` / `facing_q` read-only, no
setter), `facing_target_q`, `omega_q`. `turning.place` is the one
constructor-time door (`Unit.__init__`, north = quantize(pi/2) = 102944; test
fixtures stand units with it); `set_target` is every other writer's voice: the
live aim (stored by slot 1a each tick it is held, so a zero aim holds the last
aimed angle), the travel of a walker holding no aim (slot 3, from the
intent's own vector), the zombie walk and WEGO / onephase steps
(`face_towards`), the onephase overwatch bearing, and the attack snap
(`SimultaneousEffects.face`, now a Q16 target applied in record order).
`settle` (target = facing, omega 0) runs at conversion. Slot 1a is
`turning.step_units(self)`, one call line before 1b. `law_q16` is the §5.1
step; `top_speed_within` is critique B's closed form, at most
`ceil(min(w, pi) / a)` trials (6 at the shipped dials; the door refuses more
than 32). The incoming omega (run 2's `omega += dw` goes in front of the
law unchanged) is clamped to `PI_Q16 - 1`; the own-torque result is not
clamped further, so at a top speed of half a turn a half-turn step still
lands (G2 at `w = PI_Q16`). `Simulation.facing_after_turn_q16` is
`turning.predict`. The direction rule's no-aim shortcut is deleted. The
digest hashes the int facing and, while non-zero, `turn` (wrap(target -
facing)) and `omega`.

Times at the shipped 360 deg/s and 30 rad/s^2: 90 degrees land in 11 ticks
(0.458 s), 180 in 17 (0.708 s; §3's "0.73 s" is the check's rounder figure,
the continuous bang-bang time is 0.709 s).

Re-baselines. (1) Representation + routing at unlimited acceleration
(`turn_accel_rad_per_s2 = 1000`, >= 2w): outcome, ticks and field_digest of
every pinned match unchanged; tick_digest and the unit stream moved;
GOLDEN_AGGREGATE moved in `__unit_facing__` only. Proof: undoing the routing
(instant placement in the writer's slot) and the representation (float pi/2,
q(float) hashing) by monkeypatch gives every previous pin exactly; undoing the
representation alone leaves only the unit stream moved (the zombie walk's
facing lands a tick later, through slot 1a). (2) The shipped 30 rad/s^2:
ship_demo seed 1 is now won by team 0 at tick 265 (was team 1 at 246), the
bots' first walks being held at the sideways factor longer as their facings
come round from the spawn; the box keeps its result (unit stream moved). G3
(`tests/test_turning_law.py`) holds that at a >= 2w both matches are re-pin
1's.

Legacy behaviour changed (Control seams row: one line each): WEGO and
onephase facings (path steps, the attack snap, the overwatch bearing) now
turn at the law instead of jumping, and a snap no longer moves the facing on
the tick it is recorded; a unit still holding a live aim ignores a snap (its
aim is the target).


## As built (G5p-a2, 2026-11-10)

Run 2 of 3, the linear law, built to §5 and the §13 sheet in two commits of
behaviour. `src/simulation/unit_impulse.py` holds the door
(`impulse_settings`, `[impulse]` + `[impulse.shove]`, every key refused by
name), the per-map `ImpulseTable` (`impulse_table(cfg, tile_size_m)`, built
once as `Simulation.impulse_table` beside the speed table and cached on the
map, `table_for(gmap)`, for the combat helpers that take only the map), THE
entry `add(table, unit, jx_q, jy_q, lever, twist)`, the integration
`integrate` / `integrate_all` (slot 1a, then `turning.step(u, a, w, dw)`),
the displacement `move` (slot 3), the walk's readers `effort_scale_q16` and
`walk_allowed`, the shove (`shove_target`, `shove`) and the books
(`books_identity`). The state is integer per unit (`imp_x/imp_y/imp_l`,
`swx/swy`, `pvx/pvy`, `effort`, `react_ticks`, `last_shove_tick`,
`pending_shove`, `mass_g`, and the diagnostic `impulse_books`).

Numbers at the playground's 0.333 m tiles, 24 tps: one tick of hold 753
counts (0.092 m/s), prone 1339, in-place 2048 / 1638 / 1229 (fwd / side /
back), fall 12288 / 9011 / 6554, reaction delay 7 ticks, yaw inertia 27 000
g tile^2, body radius 0.75 tiles; the carbine round 2.95e7 counts (3.6 N s),
the shove 8.19e8 (100 N s). The reference marine is 100 kg (80 + the
rifleman's `kit_kg = 20`), a zombie 80.

Built as the sheet says, with these readings (each the simplest consistent
with §13):

- **The mass** is a loadout column, `[loadouts.*] kit_kg` (optional, 0
  absent); `Unit.mass_g` is bound at construction and again in
  `Simulation.add_unit`, kept at conversion.
- **The momentum of a round** is two `[ammo.*]` columns, `bullet_g` and
  `muzzle_mps` (a pair or neither), on every marching round: 5.56 4.0 g at
  900 m/s (the design's 3.6 N s), 9 mm 9.5 / 300 and 8.0 / 360, .50 42.8 /
  887, a 12-gauge pellet 3.9 / 400, every 40 mm body 230 / 76 (PLACEHOLDER,
  M433-like), the plasma bolts 0 (no slug). The hand grenades are lobbed
  `Projectile`s, not marching rounds: they push nothing.
- **The shove's dials** live in `[impulse.shove]` (`impulse_ns` 100,
  `cooldown_seconds` 1.0, `cone_half_deg` 60, all PLACEHOLDER), not in
  `[weapons.shove]`: a `[weapons.*]` row needs an archetype and would join the
  weapon table (the debug weapon cycle, the bots' rows). The cone is this
  run's addition to critique B's selection rule ("nearest that facing"):
  without it a unit standing BEHIND the shover would be shoved along the
  shover's facing, through the shover. Adjacent = a squared centre distance
  at most `((fp_a + fp_b) / 2 + 1)^2` tiles (B's `(fp + 1)^2` for two
  footprints), in Q16 integers.
- **A push onto a corpse is dropped** (the `push` kind applies only to a
  living unit, as `status` does), and death zeroes the impulse state (as the
  exposure's), so a corpse never carries a hashed push; conversion zeroes it
  again.
- **Prone**: entering it folds the sway into the pushed velocity (momentum
  exact) and clears the delay; the knockdown is refreshed while the effort
  (always on the standing hold, one denominator) is at least stability^2 or
  pv is non-zero after the friction.
- **The torque's sign**: positive = counter-clockwise in the facing's (y-up)
  frame, `lever = ((c - o) x s)` as B writes it. A round passing on the
  shooter's right of the target's centre therefore turns the target
  COUNTER-clockwise, as the physics says (the struck side is driven back);
  B's G6 sentence says "clockwise" for that case and is the one thing built
  against its letter (`tests/test_unit_impulse.py::test_g6_...`).
- **Every signed rounding truncates** (`unit_fixed.div_trunc`, `shr_trunc`;
  the effort step rounds away), so a mirrored scene is mirrored to the count;
  the twins are `fixed_point.h::div_trunc_q`, `shr_trunc_q`,
  `effort_step_q` and the existing `sqrt_q16`, exposed as `fp_*` bindings and
  held equal by `tests/test_unit_impulse_twins.py` (G15).
- **The walk on the Q16 grid** (B's M6, which the sheet put in run 1's
  representation commit and run 1 did not build) lands here:
  `movement.move_speed_q16` is the speed law's rational rounded half up once
  to Q16 tiles per tick, the step `(d_q * v_q) >> 16` toward zero.
- **No `__unit_push__` sub-hash label** (B's M9 asks for one): the impulse
  state is hashed in every unit record, so in `__unit_state__` and the
  pinned unit stream; a new label would re-baseline `GOLDEN_AGGREGATE` and
  its cascade with no behaviour change. Left for P3a's schema.
- **The walk ban** holds the walk for the ticks the reaction delay still
  runs after the overflow's own resist step: 6 slot-3 walks at the shipped 7
  ticks, while pv goes unresisted for 7.

Re-baselines. (1) The law with every input at zero (`bullet_push_mult =
recoil_mult = 0` in that commit's config, no controller shoves) and the walk
on the Q16 grid: ship_demo seed 1 keeps win 0 at 265 and its field_digest;
tick_digest and the unit stream move; the box is unchanged (idle marines
never walk); the instant-turn run's field_digest moves too (ship_demo:
tick f0f3a30a -> 2ca1d895, units 1b3b013f -> ec0dad03). Proof: with the
pre-run-2 float walk step monkeypatched back, this tree gives every previous
pin exactly (ship_demo, the box, and the exposure, wind-up, instant-turn and
unlimited-acceleration runs) -- the integration, the displacement loop, the
effort member, the digest keys, the events and the door move nothing while
every input is off. (2) Hits push and guns recoil (both multipliers 1.0,
RULED): ship_demo seed 1 keeps win 0 at 265 and every kill's tick, unit and
cause; all three digests move (field 65fd9690 -> ab499f29, tick 2ca1d895
-> 2620c396, units ec0dad03 -> badb4a79). Proof: with both at 0 it plays re-pin 1
exactly (G1, permanent, asserting every impulse state 0 on every tick). Per
tick: identical through tick 204; the first round lands on tick 205; no unit
is ever displaced in this match (pv stays 0: a carbine or two is soft input);
the effort (peak 0.32 of the hold) slows the shooters and the struck by a few
per cent. The shove changes no bots match (no bot shoves); it landed in the
first commit with the law (it is dormant until a controller issues it).

E2E (`tools/e2e_drive.py --control direct`, door_test, the mouse east of
Alpha): Space shoves Bravo, one footprint east -- it staggers (StaggerEvent)
and moves 1.09 tiles (0.36 m) over 15 ticks, on its feet; a .50 round from
Alpha staggers Bravo 0.17 tiles and staggers Alpha by its recoil (critique
A, m7: intended); held carbine fire raises the shooter's effort to about
0.2 of the hold (the design's "2 % slower") and never displaces the target:
one carbine is soft input, by the law.

Findings from the gates, left as the law specifies them:

- **§8's soft-input wording is loose.** "Per-tick |dv| within the in-place
  capacity and mean below the hold" admits bursts (two consecutive ticks of
  1.6 holds) that DO overflow the sway. The exact condition is the sway's
  Lindley bound: any k consecutive ticks carry at most the capacity plus
  k - 1 ticks of hold. The gate is written with it.
- **Above stability sqrt(F / hold) a steady over-hold force slides a unit at
  its excess** until its fall limit (v_fall x stability): the effort cannot
  reach stability^2, and the unrecoverable-push fall (condition 3) never
  fires because each tick's overflow re-arms the reaction delay. No shipped
  unit is above 1.0; the gate bounds the time by the law's own climb. G5p-b's
  wind reaches such units; whether a renewed overflow should count as "no
  recovery" is a modelling call.
- **Displacement is monotone in mass within a regime, not across the fall
  boundary**: a 100 N s shove floors a 50 kg body, whose prone friction stops
  it sooner (1.49 tiles) than a 70 kg one that staggers on its feet through
  the reaction delay (1.90). The gate holds the property inside each regime.
- **G5's tolerance**: the effort step B fixes carries a bias of less than 4
  counts (of ~51 000) on the mean of a 2-tick sawtooth (its fixed point is
  D/2 + 4 (d1 - d2), d1, d2 the two roundings); "one count per tick of
  period" is not reachable with it. The gate holds the filter's bound.
