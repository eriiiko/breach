# Damage, Health & Conditions

**Depends on:** [Units & entities](01_units_and_entities.md),
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
the migration.

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
| `KNOCKED_DOWN` | CC | prone; no move/act for `remaining_ticks` (the get-up time), then auto-clears. **Triggers:** blast impulse ≥ threshold (the wave_p push row, exchange §1); future: sprint collisions (pends movement design), ice falls (pends materials) — they just apply the same status |
| `IMMOBILIZED` / `STUNNED` / `PARALYZED` | CC | flag variants of the same machinery (no move / no act / neither) — kinds are config rows, adding one is O(row) |
| `BURNING` | DoT | emits HEAT packets per tick; later also a FieldEdit smoke/heat emitter (a burning unit *is* a fire) |
| `POISONED` | DoT | dose-driven POISON packets |
| `SUFFOCATING` | DoT | grace timer, then ASPHYX packets (driven by the O2/water coupling rows) |
| `REGEN` | HoT | HEAL packets — also the heal/stabilize mechanism |
| stat modifiers | buff/debuff | magnitude read by movement/aim systems |

**DoTs emit DamagePackets into §2's pipeline** — poison DoT is automatically
reduced by poison resistance, burning by heat resistance. Composition falls
out; it is never coded per-status.

**Knockdown physics note:** because `KNOCKED_DOWN` triggers on *impulse*
(∇p-derived) while lethality comes from *overpressure damage*, the knockdown
radius of an explosion is naturally **larger** than the lethal radius — the
outer blast zone knocks marines sprawling without killing them. No tuning
hack required; it falls out of the physics, which is exactly the no-barrier
principle paying out again.

## 5. Attack resolution (pre-pipeline) — the RPG seam

Whether a packet is emitted at all is the *attack resolution* layer, in front
of the pipeline:

- **Ranged: physical hits, no evasion rolls (DECIDED).** "Armor class" means
  mitigation only; whether you are hit is physics — the bullet ray actually
  enters your hitbox; misses come from the spread cone, range, cover, and
  bodies in the way. Already how the shipped rifle works.
- **Melee v1: adjacency auto-hit** (the shipped zombie melee), flat config
  damage → a KINETIC packet.
- **The seam:** attack resolution is a **game-layer policy**
  (engine-vs-game split): a future RPG ruleset may install a resolver with
  to-hit and crit rolls — all draws from the seeded stream (door 4), so it
  stays deterministic and replayable. The engine keeps the socket; Breach v1
  doesn't use it.
- **Damage rolls: deliberately UNDECIDED.** The pipeline takes whatever
  amount the source computes; a source that wants variance draws door-4.
  Per-weapon freedom, no global rule needed now.

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

## Implementation status (2026-07-04 — designed, wiring next)

| Piece | Status |
|---|---|
| Two-layer combat doctrine + rifle/melee/blast/heat damage | ✅ shipped (ch. 03 + combat.py) |
| Q16.16-snapped HP deltas everywhere | ✅ shipped (Q2-lift) |
| DamagePacket pipeline + types + mitigation | 📝 designed — wiring patch |
| zombie ×4 → resist_mult[HEAT] dissolution | 📝 with the pipeline patch |
| Status/condition system + behavior flags | 📝 designed — wiring patch |
| KNOCKED_DOWN via blast impulse (+ push row) | 📝 designed — the HUMAN-TEST fun one |
| LifeState simplification (retire unused DOWNED value) | 📝 with the status patch |
| Digest extension (`__unit_status__`) + golden re-baseline | 📝 with the status patch |
| Attack-resolver seam (RPG to-hit/crit socket) | 📝 socket only; no Breach consumer |
| Standard-values config + playground | 📝 the closing wiring patch |
