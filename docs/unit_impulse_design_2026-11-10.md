# The unit impulse law — design (draft 1, 2026-11-10)

Patch **G5p** of the gameplay pack (#183; pack design
`docs/gameplay_pack_v1_design_2026-11-10.md` §5). Research:
`docs/research/impulse_on_a_person_2026-11-10.md` (called "the search"
below) and `docs/research/wind_on_a_person_2026-10-31.md`.

Status: DRAFT for the design session with Erik. **RULED** = his rulings of
2026-11-10; everything else is Claude's proposal.

## 1. Rulings this is built on

- ONE law for everything that pushes a unit: shove, hits, wind, blast.
- Forces act through mass; each unit has its own mass.
- Soft wind never displaces a unit; below a limit a force only slows. A
  sustained force ABOVE the footing limit displaces by its excess.
- Bullets push with their real momentum (a declared multiplier ships at 1).
- Units have angular inertia: for their own turning, and from outside
  forces.
- The wind bands are restated as force over body weight.

## 2. The idea in one paragraph

Everything that pushes a unit hands it an IMPULSE (N·s) at a point on its
body. Once per tick the unit sums what it received, turns the sum into a
velocity change (impulse over mass) and a spin change (impulse × lever arm
over moment of inertia), and then RESISTS with its feet and its body, up to
a limit. What it can resist vanishes exactly, so nothing moves; how hard it
had to resist slows its walking. What it cannot resist is left as a pushed
velocity that moves it, and past a fall limit it goes down. Wind is the same
thing: a force is an impulse every tick.

## 3. State per unit (all integer)

| state | meaning | zero nearly always |
|---|---|---|
| `imp_x`, `imp_y`, `imp_l` | impulses received since the last integration: linear (N·s) and angular (N·m·s) | yes |
| `pvx`, `pvy` | pushed velocity, Q16 tiles per tick | yes |
| `omega` | angular velocity, Q16 turns per tick | when not turning |
| `facing_q` | facing, Q16 turns (today a float; P3c's kind of change) | no |
| `effort` | low-passed fraction of the footing limit in use, Q16 in [0, ONE] | yes |
| `react_ticks` | reaction delay after a sudden push | yes |

Parameters, per species / env-profile / loadout row (reference marine; the
search's picks unless marked):

| parameter | value | note |
|---|---|---|
| mass `m` | `Unit.mass` + `[loadouts.*] kit_kg` (80 + 20) | individual mass exists already |
| drag area `cd_area` | 0.9 m² standing; 0.15 m² prone (PLACEHOLDER, unsourced) | wind note |
| hold `k_hold` | 0.22 of body weight, standing | = today's wind fall edge |
| slow start `k_slow` | 0.036 of body weight | = today's wind slow edge |
| prone hold `mu_slide` | 0.4 of body weight | search §4, an assumption |
| sudden-push limit `v_inplace` fwd / side / back | 0.25 / 0.20 / 0.15 m/s | search §1 |
| fall limit `v_fall` fwd / side / back | 1.5 / 1.1 / 0.8 m/s, × `stability` | only fwd is measured |
| reaction delay `t_react` | 0.30 s (7 ticks) | search §1 |
| yaw inertia `I` | 2.5 kg·m² (PLACEHOLDER, from memory, to verify) | body + kit + rifle held forward |
| own turning: top speed, acceleration | 360°/s (G1p's cap), 50 rad/s² (PLACEHOLDER) | the acceleration is new |
| body radius for lever arms | 0.25 m | half a shoulder width |
| `bullet_push_mult` | 1.0 (RULED) | the declared experiment dial |

"Forward" is a push along the unit's facing (from behind). The limit for any
direction blends the three values by the cosine of the angle between the
push and the facing, through the kit's `cos_q16`.

## 4. The one entry

`unit_impulse.add(unit, jx, jy, lever)` is THE way anything pushes a unit: it
adds to `imp_x`, `imp_y` and (through the lever arm) `imp_l`. Sums of
integers commute, so the result never depends on the order or the ids of
whoever pushed (rule 5 of the pack holds by arithmetic).

| input | impulse | lever arm |
|---|---|---|
| a round that hits | its momentum (mass × speed from the `[ammo.*]` row: new columns `bullet_g`, `muzzle_mps`) × `bullet_push_mult`, along its flight | the perpendicular distance from the unit's centre to the bullet's line, from the march's own geometry, clamped to the body radius: no random draw |
| a shove (`shove_intent`) | `[weapons.shove] impulse` (PLACEHOLDER 100 N·s, later scaled by the shover's strength), along the shover's facing | the offset of the shover's centre line from the target's centre |
| a melee strike | its row's `impulse` (small) | as the shove |
| wind | `q · cd_area · dt` along the wind at the footprint's highest-q tile (the #140 read, plus its direction) | none in v1 (ACCEPTED GAP: wind does not turn a unit) |
| blast | v1: the blast's own wind through the wind row, nothing else, so nothing is counted twice | none |
| recoil (PROPOSED) | minus the fired round's momentum, on the shooter | none |

## 5. The integration, once per tick (in the movement slot)

```
1. dv = (imp_x, imp_y) / m ;  dw = imp_l / I ;  clear the accumulators
2. sudden push:  if |dv| >= v_inplace(direction):  react_ticks = t_react
3. pv += dv ;  omega += dw
4. resist (skipped while react_ticks > 0, which then counts down):
       cap = k_hold · g · dt            (prone: mu_slide · g · dt)
       used = min(|pv|, cap) ;  pv shrinks by `used` toward zero   (exact; clamps AT zero)
5. effort += (used / cap − effort) >> 3                            (about 1/3 s)
6. walking scale = ONE below k_slow / k_hold, then linear down to 0.5 at effort = ONE
       (composed in `default_speed` beside the wind scale it replaces)
7. fall:  |pv| >= v_fall(direction) · stability   OR   effort saturated (>= stability²)
       -> KNOCKED_DOWN (the existing status)
8. move by pv through the walking collision predicate; a blocked axis is zeroed
9. turn:  the unit accelerates `omega` toward the speed that brings `facing_q`
       to its aim without overshoot, by at most its own angular acceleration,
       never above its top speed;  facing_q += omega
```

**Why soft wind can never move a unit.** A force below the hold adds less
than `cap` per tick, and step 4 removes all of it, to exactly zero. That is
a clamp, not a tuning: the old vibration and the 35-second slide cannot come
back. A force above the hold leaves its excess, as ruled; `effort` saturates
within about a third of a second and the unit goes down, where its smaller
area and better hold stop it unless the wind is far stronger.

**Momentum is booked.** `m · Δpv = Σ impulses − resisted`, exactly, in
integers, per unit per tick: a property test, in the spirit of the energy
books.

## 6. What it gives (reference marine 100 kg; zombie 80 kg, stability 0.9)

| input | result |
|---|---|
| one carbine round | 0.04 m/s, resisted the same tick; a 0.7° nudge of the aim if it hits off-centre |
| one shooter at 12 rounds/s | 43 N: just inside the slowed band (a zombie about 4 % slower) |
| two / three shooters | 86 / 130 N: slowed about 20 % / 35 % |
| five shooters | 216 N: at the hold; the target is pushed back and goes down |
| shove from behind (100 N·s) | 1 m/s: stumbles about half a metre (1.5 tiles) over 0.75 s; off-centre it is also spun by tens of degrees |
| shove from the front | 1 m/s against a 0.8 m/s limit: knocked down |
| wind at today's slow / fall thresholds | identical bands for the reference marine; a heavier unit holds better, a lighter one worse |
| blast wind | flung by the excess, down, slides; the knockdown and the fling are one law |
| firing the carbine | the shooter's own recoil is 43 N: it walks about 1 % slower while firing (PROPOSED) |

## 7. Two patches

- **G5p-a**: the law, the entry, shove, hits, own-turning inertia (G1p's cap
  becomes its top speed). Wind keeps its present row.
- **G5p-b**: wind and blast move onto the law; the present wind row's scale
  and knockdown retire into it; the CLAUDE.md wind row is rewritten. Its
  gate includes #140's own reproduction (the 1.44 atm room must not move the
  squad) and equal bands for the reference marine.

## 8. Gates (properties)

- Any sustained force below the hold: position and `pv` exactly unchanged,
  for any duration.
- The momentum identity of §5, every tick.
- Permuting the shooters' ids changes nothing.
- Displacement is monotone decreasing in mass; a wall stops a pushed unit.
- With no outside torque the facing reaches the aim exactly, without
  overshoot, never faster than the top speed.
- With every input switched off, all pins unchanged.

## 9. Questions for Erik

1. Can a staggering unit (pushed, not down) still shoot? PROPOSED yes: its
   aim is already disturbed by the spin and its walking is slowed.
2. Recoil on the shooter through the same law: in or out? PROPOSED in.
3. The shover feels no reaction in v1 (it is braced and expects it).
   ACCEPTED GAP?
4. A shove works on any team (push a teammate out of a line of fire).
   PROPOSED yes.
5. A unit pushed into a wall fast takes impact damage: later (the search
   has the injury speeds), not in v1?
6. Zombies: lower hold than a marine (they do not brace), on top of
   stability 0.9? A number for the play test.
7. Two patches as in §7, or one?

**RULED (Erik, 2026-11-10), in the question's order:** 1 yes, a staggering
unit can shoot. 2 yes, recoil goes through the law. 3 accepted gap: the
shover feels no reaction in v1. 4 yes, a shove works on teammates. 5 later,
together with bouncing off walls. 6 a zombie is a STATE: it inherits the
stats of the unit it infected, and in v1 it is a plain copy; a zombie
multiplier on the hold exists as a dial and ships at 1. 7 two patches.

## 10. Not verified

The yaw inertia and the turning acceleration are from memory and want a
short literature check; the backward and sideways limits are modelled, not
measured (the search says so); the prone drag area is a guess. All are
play-test dials.

## 11. Systems

(a) Uses: Unit (`mass`, `stability`), Wind load on a body + the Footprint
samples builder (the scale member), status rows (KNOCKED_DOWN), Simultaneous
attacks (hits record their impulse at the resolve), the bullet march's
geometry, Fixed-point kits (`cos_q16`, `atan2_q16`), the collision predicate
walking uses, Config, Field digest.

(b) Creates: **Unit impulse** (`src/simulation/unit_impulse.py`): THE way
anything pushes or turns a unit is `add(unit, jx, jy, lever)` and the one
per-tick integration; never a position or facing write from a force; a new
pusher is a caller, a new response is a term of the integration.
