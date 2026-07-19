# Damage, Health & Conditions

**Depends on:** [Units](01_units.md),
[Combat & weapons](03_combat_and_weapons.md),
[Physics↔unit exchange](05_physics_unit_exchange.md),
[Determinism & the number-ingress rule](../engine/14_determinism_and_number_ingress.md).

What happens *after* something hits a unit. The founding directive (Erik,
2026-07-04): **"let's not put too much meaning into it yet — let's build a
flexible system."** Concretely: kills are normally just kills; knockdowns are
crowd control, not dying; the final game rules will be iterated until they are
fun, so the machinery must support many rule-sets without rebuilding. The
engine happily affords this — units are ordinary CPU-side records, so rich
state is cheap; the only obligations are determinism (integer durations, fixed
order, doors 1–3) and digest inclusion.

---

## 1. The two-axis state model

The design separates two things the draft `LifeState` enum conflated:

- **LIFE** — is the body functional? **`ALIVE | DEAD`**, minimal on purpose.
  `hp <= 0 → DEAD`. Kills are kills; no mandatory intermediate phase.
  (The enum's unused `DOWNED` value is retired by this design — knockdown was
  never a *life* state. A `DYING`/bleedout life-state is **expressible later**
  if a ruleset wants it — parked, not designed; Erik has not designed for
  bleed.)
- **CONDITIONS** — everything else that is temporarily true of a unit:
  crowd-control (knocked down, immobilized, paralyzed, stunned), afflictions
  (burning, poisoned, suffocating), and buffs (regen, stat modifiers).
  Conditions are **statuses** — one system, many triggers, no meaning baked
  into the state machine.

This is the flexibility requirement satisfied structurally: "many things can
trigger it" is true because a condition is *applied by anyone* (a coupling
row, a weapon, a collision, terrain) and *interpreted uniformly* (behavior
flags, below).

## 2. The DamagePacket pipeline

Every damage source — coupling responses, bullets, melee, DoT statuses —
emits packets; ONE pipeline owns everything after:

```
DamagePacket(amount_q16, dtype, source_id)
   │
   ▼ mitigation      amount' = max(0, amount − armor[dtype]) × resist_mult[dtype]
   ▼ batching (P2)   per unit, per phase — integer sum, order-free (exact)
   ▼ apply           hp −= Σ ;  events carry APPLIED amounts
   ▼ life transition hp <= 0 → DEAD  (+ overkill gib flag, §6)
```

**Damage types (v1):** `KINETIC` (bullets, melee) · `BLAST` (overpressure) ·
`HEAT` (radiant + fire) · `ENERGY` (beams/lasers) · `POISON` (gas dose) ·
`ASPHYX` (O2/water) · `HEAL` (negative-direction, unresisted in v1).
**Reserved, unspec'd:** `ELECTRIC` (engine/11), `PSY` (the Gray / will-stats /
`awakened` — designed when the Gray is designed).

**Proof of shape:** the shipped `zombie.fire_damage_multiplier = 4.0` special
case dissolves into `resist_mult[HEAT] = 4.0` in the zombie species profile —
a vulnerability is just a resistance above 1. No special-case code survives
the migration. Its immunity twin shipped with weapons W3:
`resist_mult[POISON] = 0.0` on the zombie overlay — they don't breathe, and
poison must not become the anti-horde cheese (fire is the answer). A
resist-0 unit draws **no DoT packet at all** from the poison coupling row
(lazy emission — no 0-damage event spam on a horde standing in gas).

## 3. Mitigation: flat armor, then multiplier (DECIDED)

- **`armor[dtype]`** — flat points, integer subtract, floor 0. Mainly
  KINETIC/ENERGY, mainly from equipment. Small arms chip harmlessly off heavy
  plate (a feature); weapons carry **AP** (armor penetration), subtracted from
  flat armor before mitigation.
- **`resist_mult[dtype]`** — Q16.16 multiplier: `0` immune, `1` neutral,
  `>1` vulnerable. Mainly elemental/chemical, mainly from species + statuses.
  The robot: `resist_mult[POISON] = 0`, `breathes = none` — profile data,
  zero code. The fish likewise.
- **Composition:** flat armors ADD (species base + equipment); multipliers
  MULTIPLY (two 50% resists → 0.25× — stacking-safe). All integer/Q16.16
  arithmetic on door-2 constants.

## 4. Statuses & conditions — one system

`StatusEffect(kind, magnitude_q16, remaining_ticks, source_id)` on a per-unit
list; ticked at the top of tick phase 3 in P0 order (unit id, list order);
per-kind stacking rule `refresh | stack | max`; magnitudes/durations are
door-2 config, durations in integer ticks.

**Behavior flags** make conditions CC without special-casing: each status
kind declares what it suppresses/alters — `can_move`, `can_act`, `can_aim`,
`is_prone` (render + hitbox/stamp implications), speed/accuracy modifiers.
Unit logic consults the *composed* flags, never individual statuses.

**The v1 roster:**

| Status | Kind | Effect |
|---|---|---|
| `KNOCKED_DOWN` | CC | prone; no move/act for `remaining_ticks` (the get-up time), then auto-clears. **Trigger:** blast `Δv = J/mass ≥ threshold × stability[profile]` (the wave_p push row, exchange §1 — mass is a live stat; the footprint-summed impulse gives the area/density effect; `stability` is the one non-physical knob, door-2 profile data: a low four-legged robot resists toppling). Future triggers: sprint collisions (pends movement design), ice falls (pends materials) — they just apply the same status |
| `IMMOBILIZED` / `STUNNED` / `PARALYZED` | CC | flag variants of the same machinery (no move / no act / neither) — kinds are config rows, adding one is O(row) |
| `BLINDED` | CC | `can_aim` off, everything else intact — an aimed fire order collapses to the **snap cone** (the cone-selection gate in `process_shooting`, the can_aim consumer). **Trigger:** the teargas coupling row (footprint-max density ≥ `teargas_blind_density`), refresh-stacked per qualifying tick — shipped weapons W3. Adding the kind was exactly O(row), as designed |
| `BURNING` | DoT | emits HEAT packets per tick; later also a FieldEdit smoke/heat emitter (a burning unit *is* a fire) |
| `POISONED` | DoT | dose-driven POISON packets |
| `SUFFOCATING` | DoT | grace timer, then ASPHYX packets (driven by the O2/water coupling rows) |
| `REGEN` | HoT | HEAL packets — also the heal/stabilize mechanism |
| stat modifiers | buff/debuff | magnitude read by movement/aim systems |

**DoTs emit DamagePackets into §2's pipeline** — poison DoT is automatically
reduced by poison resistance, burning by heat resistance. Composition falls
out; it is never coded per-status.

**Knockdown physics note:** because `KNOCKED_DOWN` triggers on *velocity
change* (`Δv = J/mass`, ∇p-derived) while lethality comes from *overpressure
damage*, the knockdown radius of an explosion is naturally **larger** than
the lethal radius — the outer blast zone knocks marines sprawling without
killing them. No tuning hack required; it falls out of the physics, which is
exactly the no-barrier principle paying out again. Division of labor, stated
once: **the resistance table mitigates damage only; force response is
Newtonian (mass × footprint); knockdown susceptibility is `stability` in the
profile.** Three separate knobs, none overloaded.

## 5. Attack resolution (pre-pipeline) — exposure, cover, crits

Whether a packet is emitted at all is the *attack resolution* layer, in front
of the pipeline. Breach uses a **hybrid** (DECIDED 2026-07-04, amending the
earlier pure-no-rolls position the same day it was proposed — the
attack-resolver seam absorbed the change with zero structural rework):

> **Physics decides what is possible; probability models what 2D cannot
> see.**

- **The physical layer stays absolute.** Walls block bullets, period. The
  spread cone, range, and bodies in the way are geometry; LOS is the ray. No
  roll ever hits what physics rules out. "Armor class" still means
  mitigation only (§3) — there is no dodge stat.
- **The exposure roll (cover).** A top-down 2D ray *overstates* exposure: it
  cannot see a marine crouched behind a waist-high crate, because the game
  has no third dimension to duck in. The percentage-to-hit is the
  **compensation for the missing dimension**. When a bullet ray reaches a
  target benefiting from cover, it connects with probability
  `exposure% = f(cover value along the incoming arc, target stance/
  conditions)` — prone (`KNOCKED_DOWN`) and future crouch states plug in
  here. A shot that fails the exposure roll is **absorbed by the cover
  tile**: it deposits its wall damage there — physics keeps its due, missed
  shots chew the crate down until it stops *being* cover.
- **Cover is directional; flanking is geometric.** A cover tile protects
  only against arcs it faces (material/level data — the mission-1 furniture
  layers are the content). Attacking from an uncovered arc = **flanked** =
  full exposure. No flanking flag — it falls out of the shooter→target ray
  direction vs the cover adjacency, deterministic geometry.
- **The crit roll.** On a connecting hit:
  `crit% = f(weapon base, attack arc vs target facing — flank bonus, behind
  bonus; attacker stats later)`. `facing` is synced Q16.16 state (the
  Q2-lift), so arcs are deterministic. A crit multiplies the packet amount
  (Q16.16 multiplier; armor-bypass tags possible later).
  **Facing is universal, arcs are data (DECIDED 2026-07-05):** every unit
  HAS a facing (movement gives one naturally; the digest hashes it); whether
  it *matters* is per-species profile data — a radially symmetric species
  (slime blob) sets vision arc to 2π and back/flank arc widths to zero: no
  behind to stab, no flank to catch, zero special-case code. (Turn rate —
  instant vs rate-limited rotation — is parked for the movement pass.)
  Likewise `stability` (§4) lives on the species profile
  (`unit.environment`), not the Unit class; per-individual stability can
  graduate into the sampled stat vector with the stats redesign if wanted.
- **Melee v1: adjacency auto-hit** (the shipped zombie melee) → a KINETIC
  packet. The same resolver hosts melee to-hit/crit when melee gets its
  design pass — or a future RPG ruleset entirely (the engine/game split:
  same socket, different policy).
- **Determinism.** Every roll draws door-4 from the sim's seeded stream, in
  P0 order; all percentages/multipliers are door-2 config. An entire
  firefight — covers, flanks, crits — replays bit-for-bit from
  `(seed, orders)`.
- **Damage rolls: deliberately UNDECIDED.** The pipeline takes whatever
  amount the source computes; a source that wants variance draws door-4.
  Per-weapon freedom, no global rule needed now.

The exposure/crit *numbers* (base percentages, arc widths, cover values per
material, weapon accuracy) are weapon-framework content — specced in the
item-5 pass, wired with the weapons wave.

## 6. Death, overkill, corpses

- `hp <= 0 → DEAD`. Normal kills are kills — no intermediate phase.
- **Overkill gib (DECIDED):** if one phase's summed packets drive
  `hp <= −max_hp`, the corpse is flagged **gibbed** — destroyed, not
  revivable/lootable, and (consistent with the shipped melee-only conversion
  rule) never zombifiable. A blast dealing 3× your vitality does not leave a
  body. Cosmetic + corpse-rules flag, not a life state.
- Zombie conversion stays **melee-kill-only** (shipped rule; heat/blast/shot
  deaths never convert).
- Parked, expressible later: bleedout/`DYING`, locational damage/hit zones,
  crawling while knocked down, corpse dragging.

## 7. Determinism & the digest

Statuses/conditions and the mitigation tables are **synced unit state**:
integer durations, Q16.16 magnitudes (door 2), P0 ordering, packets batched
per phase (bit-exact integer sums). When statuses land in code they enter the
unit-state digest (a new `__unit_status__` sub-hash) — a legitimate golden
re-baseline. The L2 representation goal (int-backed `hp` as int32 counts)
lands with the same implementation wave.

## 8. Standard values & balance (the build mandate)

Erik, 2026-07-04: once the systems are in place, **wire them all so they
actually work in-game, with standard values to play around with.** Balance is
explicitly a later, iterative problem ("not everything needs to be totally
balanced either") — v1 config ships plausible placeholders: armor/resist
tables per species, status durations, knockdown impulse threshold, DoT rates.
The playground is the A/B scenario + mission-1 slice; feel-tuning is Erik's
gate.

---

## Implementation status (2026-07-05 — statuses live, triggers next)

| Piece | Status |
|---|---|
| Two-layer combat doctrine + rifle/melee/blast/heat damage | ✅ shipped (ch. 03 + combat.py) |
| Q16.16-snapped HP deltas everywhere | ✅ shipped (Q2-lift) |
| DamagePacket pipeline + types + mitigation | ✅ shipped (P2, 2026-07-05 — `simulation/damage.py`; all four sites routed, neutral defaults bit-identical, digest unchanged. Float64-amount form: the integer `amount_q16` packets + per-phase batching are later patches) |
| zombie ×4 → resist_mult[HEAT] dissolution | ✅ shipped (P2 — `species.ZOMBIE_MITIGATION`; since P5 the profile SOURCES its value from `[zombie] fire_damage_multiplier` at species-table import (restart-bound standard value), and the two heat tests read the same key as the expected ratio — sim + tests move together) |
| Status/condition system + behavior flags | ✅ shipped (P3, 2026-07-05 — `simulation/status.py`: kind registry rows + refresh/stack/max stacking + `composed_flags`; ticked at step 2b, the top of the unit-simulation section (ch. 05 §4 phase 3); DoTs emit through `apply_packet` (zombie BURNING = 4× a marine's, proven bitwise). Duration contract: N ticks of suppression AND N emissions (lazy sweep). Consumers wired: movement pauses on `can_move` (path-offset shift), fire orders + auto-fire and zombie melee gate on `can_act`, zombie walk on `can_move`. **Weapons W3 added `BLINDED` (kind 8) + the `can_aim` consumer** (aimed fire → snap cone at the `process_shooting` cone selection) **+ the first in-game applier**: the teargas coupling row (exchange 9c3); zombie POISON immunity (`resist_mult[POISON] = 0`) joined the mitigation overlay |
| KNOCKED_DOWN via blast impulse (+ push row) | ✅ built (P4, 2026-07-05 — `exchange.apply_wave_push` at step 9c2: trigger `dv² ≥ (threshold × stability)²`, refresh-stacked getup timer; `stability` on EnvironmentProfile (1.0) + the `species.ZOMBIE_STABILITY` overlay (0.9); minimal prone visual (sprite rotated 90°). The §4 physics note is now a REGRESSION TEST: measured knockdown ring ~9–10 tiles vs ~4.6-tile meaningful-damage ring (≈2×) for a grenade — `tests/test_wave_push.py`. **HUMAN-TEST pending: Erik's feel check gates the merge** |
| LifeState simplification (retire unused DOWNED value) | ✅ shipped (P3 — `ALIVE | DEAD` only) |
| Digest extension (`__unit_status__`) + golden re-baseline | ✅ shipped (P3 — the synced unit record carries `serialize_statuses`; golden `ae1164ca…` → `6d690fda…`, no field trajectory moved; Lenovo re-attestation owed) |
| Attack resolver: exposure-vs-cover roll + crit roll (flank/behind arcs) | ✅ shipped **with standard values** (weapons W2 `bbfb26a`, 2026-07-05 — `simulation/attack_resolver.py` wired into the unified march at the stop sites: `cover_exposure` materials column (1.0 everywhere, furniture 0.55) rolled ONLY on footprint entry through a concealing tile, failed roll = absorbed + wall-damage chew (crates chew down until they stop being cover; `destroy_wall` handles non-solid destructibles now); crit% = `weapon.crit_chance` × arc mult (`[combat]` ×1/×2/×4) vs the synced facing, arc widths on EnvironmentProfile (120/90; blobs later ship 360/0), crit scales the amount half-away-from-zero pre-mitigation; firing sets facing to the aim bearing. ALL ROLLS LAZY (`float(rng.uniform(0,1)) < p` — the draw form of record): shipped weapons carry crit 0, the digest scenario has no cover on firing lines → golden `07c3f370…` UNCHANGED. HITSCAN skips both rolls in v1 (skewer identity). Numbers are Erik's playground dials |
| Standard-values config + playground | ✅ shipped (P5, 2026-07-05 — every wave value config-visible with feel comments + interesting ranges (`[exchange]`, `[zombie] fire_damage_multiplier`/`stability`); `levels/playground` (one room per system: arena/grenade range, wood fire room, glass gallery, sealed pressure room, steel bunker, pool basin, glass zombie pen, breach bay) + `main.py --level playground` + the experiment guide `docs/playground_guide.md`. Behaviour-preserving: golden unchanged. Erik's feel-tuning is the open loop) |
