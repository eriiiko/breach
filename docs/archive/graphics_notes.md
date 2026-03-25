# Graphics & Raycaster Notes

_Created: 2026-03-23_

---

## Current State

We use raylib (C++) for all rendering. The game already has a unique visual style
that we're happy with.

---

## Raycaster — Understanding the Costs

_Goal: understand exactly what costs in the raycaster, so we can use it creatively
and scale to many lights confidently._

### What We Want to Know

1. **Where is the cost?** Ray casting itself (origin + direction, DDA traversal)
   vs. tile sampling per ray step (cache misses if tile data is scattered) vs.
   shadow accumulation per light (scales with light count x rays x tile depth)

2. **Is it parallelised?** Raycasters are embarrassingly parallel per-ray. If our
   implementation isn't already threaded, that's the biggest free lunch available.

3. **How does cost scale with light count?** One light runs fine at 60fps. The
   question is whether cost scales linearly (good) or worse (bad) — that tells us
   if it's the ray math or the compositing.

### Cost Estimates (from implementation_plan_radiation_temperature.md)

- One flashlight: <1ms (Python), ~us (C++)
- 20 sources worst case: ~8ms (Python), <1ms (C++)
- ~60 points per bolt path, trivial cost
- These are theoretical — need profiling to confirm

### Tweak Candidates (to investigate)

- **Max ray length** — hard cutoff, biggest single knob
- **Angular resolution** — fewer rays per light
- **Ray count scaling with distance** from camera (LOD for rays)
- **Tile early-exit** on opaque hit
- **Dirty-flagging** — only recast rays for lights whose environment changed
- **Light budgets per frame** — prioritise visible, nearby lights
- **Baked ambient + dynamic for key lights only**
- **Clustered lighting** — group lights by screen region

### The Many-Lights Problem

This is the classic real-time lighting bottleneck. Our C++ implementation gives us
full control. The profiling session should:

1. Time just the ray casting vs the rest of the render
2. Scale light count from 1 → 2 → 4 → 8 and check if linear or worse
3. Check if tile map is contiguous in memory (struct-of-arrays vs array-of-structs)

### TODO

- [ ] Profile the raycaster with varying light counts
- [ ] Check if SIMD/threading is being used in the ray loop
- [ ] Measure memory access patterns (cache friendliness of tile data)
- [ ] Document findings and decide which optimizations to pursue

---

## Visual Style

_Notes on what we like about the current look and where we want to go._

- Raylib gives us a unique aesthetic — low-fidelity but atmospherically rich
- The raycaster lighting is central to the feel (dynamic shadows, emergency lighting)
- We want to lean into this, not fight it

### Future Graphics Work

- [ ] Unit sprites — distinct silhouettes per unit type (marine, zombie, gray, femme fatale)
- [ ] Normal maps for tiles (see `graphics_lighting_design.md` Phase 2)
- [ ] Fire/explosion visual effects tied to the physics (particle + light source)
- [ ] Smoke rendering (opacity tied to atmosphere sim concentration)
