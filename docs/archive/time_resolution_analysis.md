# Time Resolution Analysis: How to Discretize a Phase

> **Context**: The two-phase turn system needs a shared time resolution for movement,
> the reservation table (temporal A*), action ordering, and physics. This note compares
> three approaches.
>
> **DECIDED**: Pure grid (Strategy A variant) at **12 ticks/second**. No continuous
> positions. See design_v2_turn_and_combat_overhaul.md for final spec.

---

## The problem

Within one phase (5 real-time seconds), we need to answer:
- How many discrete time steps exist?
- What happens at each step?
- How does the reservation table track tile occupancy over time?
- How do actions (shoot, throw, explode) interleave with movement?

---

## Strategy A: Fine tile = 1 timestep

**Every fine tile traversal is one discrete timestep.**

A unit moving 4 coarse tiles (= 12 fine tiles) takes 12 timesteps per phase.
Sprint (6 coarse = 18 fine tiles) takes 18 timesteps.

| Aspect | Detail |
|--------|--------|
| Reservation table | `reservations[t][fx][fy]` — up to ~18 steps × 120×75 fine tiles |
| Memory | ~18 × 9,000 = 162,000 entries per phase. Trivial. |
| Diagonal cost | Alternating 1-2: first diagonal = 1 timestep, second = 2 timesteps |
| Action ordering | Within a single timestep, resolve in fixed order: (1) door explosives, (2) grenades detonate, (3) shooting resolves, (4) movement advances 1 fine tile |
| Animation | 12–18 steps mapped across 5 seconds. Each step = 0.28–0.42s of real time |

**Pros:**
- Highest fidelity — every tile transition is tracked
- Natural fit for the fine grid physics (atmosphere, smoke already operate at fine resolution)
- Diagonal cost is exact (alternating 1-2 at fine tile scale)
- Reservation table catches even brief overlaps in tight corridors

**Cons:**
- Different movement modes have different numbers of timesteps per phase. Sprint has 18 steps, attack has 12. The reservation table must handle variable-length phases per unit.
- Workaround: pad shorter paths with "wait" steps so all units use the same number of timesteps (= max across all units, i.e., 18). Units that finish early just stand still for the remaining steps. This is clean and consistent.
- More steps = slightly more A* nodes to explore (but still tiny on this grid size)

---

## Strategy B: Fixed N timesteps per phase

**All units share the same number of timesteps per phase (e.g., 6 or 10).**

Movement modes determine how many tiles a unit crosses per timestep. Sprint at 6 steps:
1 coarse tile per step. Attack at 6 steps: 4/6 ≈ 0.67 coarse tiles per step (unit moves
some steps, pauses others).

| Aspect | Detail |
|--------|--------|
| Reservation table | `reservations[t][cx][cy]` — 6–10 steps × 40×25 coarse tiles |
| Memory | ~10 × 1,000 = 10,000 entries. Very small. |
| Diagonal cost | Coarser approximation — 1 coarse diagonal per step |
| Action ordering | Same fixed-order resolution per timestep |
| Animation | Uniform step duration (5s / N steps) |

**Pros:**
- Uniform time axis — every unit has the same number of steps. Reservation table is simple.
- Coarse grid keeps A* fast and memory tiny.
- Easy to reason about during planning ("each unit gets 6 steps")

**Cons:**
- Sub-tile movement is lost. Two units passing in a 1-tile corridor can't be tracked at fine resolution.
- Fractional tile movement per step is awkward (attack mode moves 0.67 tiles/step — need rounding or movement point accumulation)
- Misaligned with physics, which runs at fine tile resolution
- Less tactical precision — the fine grid exists for a reason

---

## Strategy C: Hybrid — coarse tiles, fine time

**Timestep = time to cross 1 coarse tile at the unit's speed. Reservation table at coarse
resolution.**

Different speeds mean different real-time durations per step, but the spatial resolution
is always 1 coarse tile. Sprint crosses 1 tile faster than attack, but both reserve
1 tile per step.

| Aspect | Detail |
|--------|--------|
| Reservation table | `reservations[t][cx][cy]` — variable t per unit |
| Time normalization | Need a common clock. Could use finest granularity (sprint speed) as the base tick. |
| Diagonal cost | Alternating 1-2 at coarse scale |

**Pros:**
- Spatial precision at the scale that matters for unit collisions (units occupy 1 coarse tile)
- Natural "1 step = 1 tile" for movement orders

**Cons:**
- Variable time steps per unit speed — reservation table needs time normalization
- Essentially a more complex version of Strategy A at coarser spatial resolution
- Gains little over Strategy B while being more complex

---

## Recommendation: Strategy A (fine tile = 1 timestep)

**Reasoning:**
1. The fine grid already exists and all physics operate there. Aligning the time system
   with it avoids a spatial resolution mismatch.
2. The reservation table is small enough that memory/performance is not a concern.
3. Variable timestep counts per unit are handled by padding with wait steps — pad to the
   maximum (18 steps for sprint) so all units share the same time axis.
4. Corridor collisions at sub-coarse resolution are important tactically — two marines
   passing in a 3-fine-tile-wide corridor is a real scenario.
5. Action ordering within a timestep is clean: just a fixed priority list.

**The padded timeline:**
```
Phase length = 18 fine-tile-steps (= max movement, sprint 6 coarse × 3 fine/coarse)

Attack unit (12 movement steps):
  Steps 1-12: move 1 fine tile each
  Steps 13-18: wait (standing at destination)

Sprint unit (18 movement steps):
  Steps 1-18: move 1 fine tile each

Cover unit (15 movement steps):
  Steps 1-15: move 1 fine tile each
  Steps 16-18: wait
```

**Within each timestep, resolution order:**
```
1. Door explosives detonate (if scheduled for this timestep)
2. Pathfinding grid updates (walls/doors changed)
3. Grenades detonate (if fuse expires this timestep)
4. Shooting resolves (hitscan damage applied)
5. All units advance 1 fine tile along their path (or wait)
6. Melee damage (zombies adjacent to marines)
7. Death/conversion checks
8. Physics substep (atmosphere, smoke)
```

This gives 18 × 2 = 36 fine-tile-steps per round (both phases), which maps to 10
real-time seconds at the chosen phase duration.

---

## Open sub-questions

1. **Should the reservation table use fine or coarse spatial resolution?** Units occupy
   3×3 fine tiles. Reserving at fine resolution means reserving a 3×3 block per timestep.
   Reserving at coarse resolution (1 tile = the unit) is simpler and probably sufficient
   since units can't share a coarse tile anyway. **Lean toward coarse spatial, fine
   temporal.**

2. **Shooting duration**: A fire order lasts "until end of phase." In Strategy A, that
   means the unit fires every timestep from the order until step 18. Each timestep it
   fires a burst. Total bullets = burst_size × remaining_steps. This might be too many —
   may need a "shots per phase" cap instead.

3. **Grenade travel time**: Currently grenades travel ~0.3T. In the new system, how many
   fine-tile-steps does travel take? If the grenade moves at, say, 3 fine tiles per step,
   a 30-tile throw takes 10 steps. The fuse timer starts on throw — a 0-second fuse
   grenade would detonate when it arrives (travel time is the minimum effective fuse).
