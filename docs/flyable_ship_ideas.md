# Flyable Ships — Idea Notes

**Status:** IDEA COLLECTION ONLY — not a design document, not final.
Captures brainstorming for a future direction. No implementation or reuse
decisions are made or implied here. A proper design session comes later,
once the picture is complete and the base physics engine is done.

**Date:** 2026-06-05 · **Origin:** Erik + Claude brainstorm

---

## Framing & ground rules

- These notes introduce a concept that is **entirely new to breach**: a second,
  *exterior* (space) state space alongside the existing *interior* one.
- The current codebase is **interior-only**. Nothing in it is aware of an
  exterior, and none of it was written with one in mind (`coords.py`, the
  physics systems, etc. all describe the interior). Anything that *might* be
  reusable for the exterior is to be **evaluated in a later design session,
  never assumed here.**
- Non-destructive: this is an additive idea log. It proposes nothing about how
  existing systems should change.
- Lots must happen first — notably completing the base physics engine. These
  are future ideas, parked deliberately.

---

## 1. Dual state space — interior + exterior

- Two largely independent simulations:
  - **Interior** — the existing tile-grid world (the insides of ships).
  - **Exterior** — a new **continuum** space domain. A ship is a single point
    with position, velocity, orientation, angular velocity.
- Each is unaware of the other's internals. Any interaction is a **separate,
  thin coupling layer**, not shared state.
- Cardinality: **1 exterior + N interiors**, with N expected to be small.
- Open: what exactly crosses the coupling layer, and in which direction.

## 2. Flyable ships

- Capture a ship and fly it from the exterior domain.
- Open: how piloting/control maps from interior state to exterior motion.
- Open: rotation in space — does the interior have a notion of "down" at all
  when adrift (artificial gravity?), or only near a planet? (New question; the
  existing fluid `tilt` is unrelated — that serves aquariums splashing and
  ships landing/sinking on an ocean, not space rotation.)
- Open: whether hull venting could impart thrust/torque — flagged as an idea
  to explore, with no assumption about how it would be computed.

## 3. Scale transitions (space ↔ planet ↔ station)

- Acceptable baseline: **loading-screen** transitions between scales.
- Ambitious dream: **seamless zoom**, Google-Earth-style — space → planet →
  station, continuous. "Never done in 2D."
- Pragmatic cheats welcome to sell the *feeling* before earning the tech:
  generate-on-the-fly, always descend to the same authored spot, etc.
- Open: coordinate / precision strategy across vastly different scales (later).

## 4. Orbital bombardment → surface fire

- Bombardment from the exterior creates **forest fires** on a surface.
- Idea: the *fire concept* at a different scale — much slower spread, coarser
  cells, different parameters. Whether this literally reuses the interior fire
  system or is a same-shaped sibling with different couplings is **open** (the
  interior fire is currently tied to the hull/O₂ model; a surface fire isn't).
- Note: this is a cross-scale *event* — an action in the exterior producing an
  effect in a surface domain.

---

## Open questions parked for the design session

- Exact contents and direction of the interior↔exterior coupling layer.
- Whether inactive interiors are *frozen* (state preserved, no ticks) or
  *abstracted* (a cheap proxy advancing coarse outcomes).
- Rotation & "down" semantics in the exterior.
- Coordinate / precision strategy for seamless zoom.
- Reuse vs. rebuild for fire at surface scale.

## Dependencies / sequencing

- **Prerequisite:** base physics engine completed first.
- Then: a **design session** — what systems are needed, how they interact —
  *before* any implementation. Design first.

## Related

- Doors (a related new system / interactable) are tracked separately in
  `docs/TODO.md`, not here.
