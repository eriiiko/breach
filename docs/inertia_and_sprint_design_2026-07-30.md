# Momentum, sprint and unit collision — design sketch (2026-07-30)

**Status:** SKETCH, not locked. Captured from Erik's notes at the close of the
OnePhaseWEGO human-test session so nothing is lost; the real design pass is a
session of its own.

**Erik's ruling on sequencing (2026-07-30):** *do not* start with the full
inertial model. Start with the **simple sustained-sprint rule** in §A, which
buys most of the interesting gameplay for a fraction of the risk. §B (real
momentum) stays on the table behind it, and §C (collision) is needed by both.

**Depends on:** `onephase_wego_design_2026-07-28.md` (§4 continuous space, §6
aim-relative speeds, §13 invisible seams) — this is a movement-model change
inside that ruleset.
**Related:** the ML-animation track (`docs/research/ml_animation_litsearch_2026-07-20.md`)
— see §E.

---

## 1. Motivation

Movement is currently memoryless: a marine is at full speed on tick 1 of a move
and stops dead on the last tick, and a 180° reversal costs nothing but the §6
aim-relative fraction. Two things that suggests:

- **The battlefield has no non-linear texture.** Every repositioning decision
  costs exactly its distance; nothing about *how* you moved last round changes
  what this round can do.
- **Multi-round movement has no identity.** With 4 s rounds (§2), crossing a
  map is several rounds of the same thing. A unit that has been running in a
  straight line for three rounds ought to *feel* different from one that just
  set off.

Erik's framing: this "increases complexity of the battlefield, introducing
non-linear relationships in a natural way". The stated risk is equally honest —
it might not be fun. Which is exactly why §A comes first.

---

## A. THE SIMPLE RULE — try this first

> "Perhaps we don't want the full inertial and acceleration model — perhaps we
> want an increased speed and reduced navigation ability after 4 s of
> continuous moving, a simpler rule that gets us where we want anyway."
> — Erik, 2026-07-30

**Sprint returns, but as an EARNED STATE rather than an order.** §5 removed
Sprint as a separate order and folded it into Move; this brings the *idea*
back without bringing the order back — you never pick Sprint, you accumulate
it.

### A.1 The rule

- A unit accumulates `sustained_move_ticks` while it moves without a
  disqualifying event.
- Past a threshold (**~4 s, one round** — deliberately the round length, so
  sprint is inherently a *multi-round* payoff), the unit enters **SPRINT**:
  - **speed up** — a multiplier on the §6 base, stat-derived (§D);
  - **navigation down** — the cost. Concretely: a minimum turn radius, or a
    cap on heading change per tick, so a sprinting unit cannot take the
    tight corners a walking one can.
- Sprint **decays** on: stopping, a sharp turn beyond the cap, taking a
  non-move action, or being knocked down.

### A.2 Why this shape

- **Normal play is unchanged.** Erik's explicit requirement: "gameplay stays
  mostly unchanged from now for normal play, but multi-round sprints can
  utilize the added speed." Inside a single 4 s round almost nothing reaches
  the threshold, so every existing feel-tuned interaction is untouched.
- **It is a state machine, not a physics model.** No velocity vector, no
  integration, no new determinism surface beyond an integer counter — which
  means it can be tried, felt, and thrown away cheaply.
- **It still produces the non-linearity Erik wants.** "Commit to a long
  straight run and you get there faster but can't corner" is a real tactical
  trade, and it is legible to the player in a way a continuous momentum curve
  is not.

### A.3 What it costs the timeline

The compiled plan (timeline §3) must model it, or the arrival timestamps stop
being honest. This is tractable: the compiler already walks the path tile by
tile, so it can carry the sprint counter along the walk and apply the same
rule the executor will. **The plan stays exact.**

The turn-radius cap is the fiddly part — A* produces tile paths that can turn
90° in one step, so either the compiler smooths the path under sprint, or
sprint simply *drops* at such a corner (simpler, and arguably the honest
reading: you slowed to take the corner).

---

## B. The full model — real momentum (deferred)

Kept because Erik wants it eventually and the sketch is worth preserving.

### B.1 State

Erik's instinct is right that the state is small: `velocity` (vx, vy) beside
`x`/`y`. `mass` and the stat block already exist on every unit.

The work is NOT the state — it is three couplings:

1. **Acceleration from stats.** `a = F/m`, force from strength/endurance,
   mass already per-unit. Heavy units are slow to start *and* slow to stop,
   which is the interesting half.
2. **Turning as a vector constraint.** Momentum must be a VECTOR or a 180°
   reversal is free and the whole thing reads as fake. This is the real
   integration work: §6's aim-relative speed table stops being a lookup and
   becomes a constraint on the achievable velocity.
3. **Collisions between moving bodies** — §C.

### B.2 The determinism cost

Velocity is synced state, so it must be Q16.16 integer (project iron rule) —
positions currently ride the documented float discipline (exact `n/65536`
steps). A velocity integrator accumulates, so it wants to be genuinely
fixed-point rather than "floats that happen to be exact". Not hard, but it is
a real piece of work and a real digest-surface change.

---

## C. Collision rules (needed by BOTH A and B)

Bodies became solid on 2026-07-30 (`timeline.occupied_by_unit`) — a unit
simply cannot enter a tile another body occupies, and holds the tick. That is
the floor. What Erik wants is the interesting version:

> "if someone is ran into, and they are standing still, perhaps they have
> higher stability … I'd like to design an 'intricate' system for it depending
> on the unit stats."

### C.1 The hook already exists

`stability` is a real field on the species `EnvironmentProfile`
(`simulation/environment.py`), and `exchange.py` already uses it as the
knockdown threshold for shockwave push:
`KNOCKED_DOWN if |Δv| >= knockdown_dv_threshold * stability`. A body collision
is the *same shape of event* as a blast wave — an impulse that may or may not
put you down — so it should reuse that rule rather than invent a second one.

### C.2 Sketch

On a collision between A (moving) and B:

- **Relative speed** decides severity — two units walking into each other is a
  shuffle; a sprinter into a stationary body is a takedown.
- **Effective stability** scales with: the `stability` profile field, mass, and
  **whether the unit is braced** — Erik's point that a stationary unit resists
  better. A unit that is *moving* has already committed its balance.
- **Outcomes**, in increasing severity: both slow → both stop → the lighter/
  less stable one is displaced → one falls (`KNOCKED_DOWN`, which already
  exists with a get-up timer) → both fall.
- **Friendly vs hostile** probably want different rules — walking into a
  squadmate should be an annoyance, not a tactical disaster.

### C.3 Body-checking as a verb?

Falls out almost free once C.2 exists: an ordered charge into an enemy is just
a collision you *chose*. Worth noting, explicitly out of scope for the first
pass.

---

## D. Speeds from stats

Erik: "speeds (slow and fast) could still depend on the stats, in any way we
see fit."

Now cheap, because speed is authored in **m/s** (`[onephase] move_speed_mps`,
2026-07-30) rather than ticks-per-tile. A per-unit speed is that dial scaled by
a stat-derived factor — agility and endurance for the walk, strength/mass for
the sprint multiplier and for how fast it accumulates.

Constraint: the derivation must be quantize-once (the level's tile cadence is
derived at plan-compile time), and integer/deterministic.

---

## E. The plan-exactness question

Erik's framing, which resolves the objection better than the objection
deserved:

> "the plan not staying exact if you issue orders far from you (certainty
> decreases with distance) … perhaps we can predict exactly for everything
> except collisions — and collisions wouldn't happen if they were predicted
> anyway."

Two ideas worth keeping verbatim:

1. **Certainty decreasing with distance is a FEATURE**, not a defect — and it
   is thematically right for a game about commanding a squad through a burning
   ship. If the UI ever needs to express it, the natural form is the confidence
   of the *displayed* prediction degrading along the path, not the sim becoming
   sloppy.
2. **A predicted collision is self-cancelling.** If the planner can see the
   collision, the player routes around it, so it does not happen; the collisions
   that DO occur are exactly the ones nobody could have predicted (a zombie
   moved, a wall fell). This means "exact except collisions" is a much stronger
   guarantee in practice than it sounds — the residual uncertainty is precisely
   the irreducible kind.

So the timeline invariant survives with a caveat: **the schedule is exact
given the world it was compiled against.** Which is already true today for
§14 plan invalidation.

---

## F. The ML-animation link

Erik: "I hope this can combine well with the ML-animation track."

It does, and in the useful direction. The animation research
(`docs/research/ml_animation_litsearch_2026-07-20.md`) concluded the network is
never the bottleneck and that physics-driven character animation wants a
*velocity-and-contact* signal to drive it. A momentum model produces exactly
that signal as a by-product:

- `velocity` → gait selection and lean, instead of the current facing-only
  8-direction snap;
- collision impulses → stumble/recover/fall transitions, which is precisely
  the class of thing the "physics dolls" line of research is good at;
- `KNOCKED_DOWN` already exists as a state with a timer, so it is a ready-made
  animation hook.

Worth noting the dependency runs THIS way: momentum feeds animation, not the
reverse. Animation stays render-only (the iron rule) and consumes synced
velocity without writing it.

---

## G. Open questions for the design session

1. **§A first, alone** — does earned sprint carry the fun by itself? If yes,
   §B may never be needed, which is the cheapest possible outcome.
2. **Sprint threshold and decay** — 4 s is Erik's opening number. Does sprint
   survive the round seam? (It must, by §13 — but it is worth stating.)
3. **What "reduced navigation" actually is** — turn-rate cap, minimum radius,
   or simply "sprint drops at a sharp corner". The third is much the simplest.
4. **Does sprint need to be visible in the plan viz?** Probably: a path segment
   drawn differently once the compiler knows sprint engages there.
5. **Collision severity curve** and the friendly/hostile split (§C.2).
6. **Do zombies get it?** They run the legacy `ai_zombie` path, which is shared
   with TwoPhaseWEGO — touching it moves goldens. Probably marines first.
7. **Is stability a real STAT** (BaseStats) rather than a species-profile
   constant? Erik referred to it as a stat; today it is a per-species field.

---

## H. Prior art note

Momentum/inertia is rare in tactical games and common in sports games —
NBA 2K's on-ball collision system (including falls) is the closest relative to
§C, and the FIFA/PES momentum models to §B. The nearest tactical relatives are
*Door Kickers* (turn radius, no momentum) and *Arma* (real inertia and stamina,
and famously divisive). So this would be fairly unexplored ground for the
genre — which cuts both ways, and is a reason to build §A cheaply and *feel* it
rather than argue about it.
