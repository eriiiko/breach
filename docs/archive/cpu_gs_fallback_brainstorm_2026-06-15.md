# Hybrid CPU/GPU determinism — keep the hard-to-determinize ops on CPU

_Status: SUPERSEDED, archived 2026-11-07. Brainstorm of 2026-06-15, written to avoid a Q16.16 migration of every solver. That migration was then done, and fixed point is confirmed bit-identical across GPU architectures at full-engine scale (CLAUDE.md, "Determinism is a hard requirement"). Kept for the reasoning; nothing here is open work._

---

## The idea

The resolution proposal frames Q16.16 fixed-point everywhere as the unconditional path
to cross-GPU determinism. There's a simpler alternative: **partition the sim so the
two genuinely hard-to-determinize ops stay on CPU, and everything else runs on GPU.**

The two hard ops:

1. **Per-cell division in the GS atmosphere step** —
   `atmosphere[i] = (rhs[i] + μ·nb) / (1 + μ·wsum)` ([atmosphere_solver.cpp:196](../cpp/src/atmosphere_solver.cpp#L196)).
   Fixed-point division is the nastiest op to make bit-exact AND fast on GPU.
2. **The global `mean_wp = sum/count` reduction** —
   structurally different summation orders across CPU-scalar, CPU-SIMD,
   and CUDA warp-shuffle, all required to agree bit-for-bit.

Both are textbook deterministic on CPU (same source, same compile flags,
pinned summation order → bit-exact across any x86 machine).

## The proposed split

**CPU (deterministic by construction):**
- Gauss-Seidel atmosphere solve
- Global float reductions (`mean_wp`, etc.)

**GPU (per-cell stencils + DDA — straightforward to determinize):**
- Wave update (`w_new = 2w − w_old + c²·Δt²·∇²w` — pure multiply-add, no division)
- Smoke advection (semi-Lagrangian, but boundary handling must be pinned)
- Smoke diffusion (explicit Euler stencil)
- Temperature, fire (per-cell)
- Raycaster (DDA, all integer-ish increments)

**Per-tick cost:** one CPU↔GPU memcpy of the pressure field per tick.
At 50×120 floats that's 24 KB; at 100×240 it's 96 KB. Sub-millisecond either way.

## Why this might be better than full-GPU fixed-point

- Sidesteps the deepest determinism trap (parallel reduction).
- Sidesteps the second-deepest (fixed-point division).
- GS on CPU at our grid size is probably 1–2 ms — likely not the bottleneck.
- Q16.16 migration for the *other* solvers is still required, but easier:
  stencils + multiply-add are trivially bit-exact in fixed-point.
- Engineering surface is smaller than full-GPU fixed-point.

## Open questions before committing

- **GS-on-CPU cost at scale.** Single game is fine; what about a farm of 100+
  parallel instances on one machine? Does CPU GS become the bottleneck?
  Profile before committing.
- **CPU↔GPU sync stalls.** Is the per-tick memcpy actually sub-ms in practice,
  or does PCIe latency hide a bigger cost? Measure.
- **Does the GS step have to run before the wave step?** If yes, the wave can't
  start on GPU until GS finishes on CPU — serializes the pipeline.
  If no (e.g. GS reads tick `t`, wave reads tick `t`, both write `t+1`),
  they can overlap.
- **Does the SAME tradeoff apply to the smoke sink-pull?** That op also has
  a non-trivial determinism story. Worth checking whether it joins the CPU side.

## Relation to the resolution proposal

This is a refinement, not a rejection. The proposal's analysis is right:
fixed-point everywhere on GPU IS the deterministic path. The point here is that
**the same determinism guarantee is buyable more cheaply with a hybrid split**, and
that's worth measuring before committing to a multi-week Q16.16 migration of
every solver including GS.

Worth folding into a v2 of the resolution proposal if research confirms the
costs come out as expected.
