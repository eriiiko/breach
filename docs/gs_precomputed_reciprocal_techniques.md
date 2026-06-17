# Precomputed reciprocal techniques for the GS atmosphere divide

_Research note, 2026-06-17. Captured from a multi-agent research pass on
"how do we make the GS divide GPU-fixed-point-friendly." Not a commitment
to any technique — kept here so the home-PC session can pick up the thread
without re-deriving._

---

## The divide in question

[cpp/src/atmosphere_solver.cpp:196](../cpp/src/atmosphere_solver.cpp#L196):

```cpp
atmosphere[i] = (rhs[i] + mu * nb) / (1.0f + mu * wsum);
```

Where:
- `rhs[i]` = pre-sweep snapshot of the atmosphere (the IMEX `u*`).
- `nb` = `w_up*atm_up + w_down*atm_down + w_left*atm_left + w_right*atm_right`.
- `wsum` = `w_up + w_down + w_left + w_right`.
- Each `w_n` = `min(perm[self], perm[neighbor])`, zeroed at wall/obstacle faces.
- `mu = d_atm * dt` — a **global scalar per substep**, NOT per cell or per material.

At shipped config (`d_atm=200`, `gs_iters=8`, red-black, `~6 substeps/tick`), each
non-skipped cell receives **~48 divides per tick**.

---

## Headline reframe (do not lose this)

The divide is not the cross-architecture determinism villain. IEEE-754 `div.rn.f32`
is correctly-rounded on every target (x86, ARM, CUDA). The actual cross-arch
hazards in this loop are:

1. **FMA contraction** in `mu * nb + rhs[i]` (compilers fuse inconsistently).
2. **4-term sum order** in `nb` under SIMD or CUDA threads.
3. **Red-black GS intra-sweep read-after-write ordering.**
4. The `mean_wp` float reduction at step 3 — separate, deepest trap.

`/fp:fast` / `-ffast-math` are currently on at
[cpp/CMakeLists.txt:14,16](../cpp/CMakeLists.txt#L14-L16). With those flags,
the compiler is already substituting `RCPSS`-class reciprocal approximations
for the divide AND is free to reorder the 4-term sum AND fuse FMA
inconsistently across MSVC/GCC/clang. **Removing fast-math and pinning
`-ffp-contract=off` probably moves the needle on determinism more than any
divide-elimination would.**

So: hoist the divide out of the hot loop for code-shape and SIMD/CUDA
vectorization reasons, not because the divide itself is "fragile." With that
established, here are the techniques.

---

## Technique 1: Precomputed inverse-diagonal field `D[i]`

The cleanest move. Same solver, same residual, same physics — just hoist the
one divide per cell out of the 16-sweep inner loop into a precompute pass.

```cpp
// Once per tick (or when permeability / mu changes):
for (int i = 0; i < N; ++i) {
    if (skip(i)) continue;                              // explicit skip — NOT D[i]=0
    D[i] = 1.0f / (1.0f + mu * wsum[i]);
}

// Hot loop, every sweep:
atmosphere[i] = (rhs[i] + mu * nb) * D[i];              // multiply, no divide
```

**Divide cost: ~48/cell/tick → 1/cell/tick.** Reduction in arithmetic; no model change.

### Three real caveats from the adversarial pass

1. **Algebraically identical but NOT bit-identical to today's output.** Every
   recorded golden trace shifts by ~1 ULP per cell. Must coordinate with the
   in-flight `recorder.py` work (regenerate goldens in the same commit).
2. **Skipped-cell handling: explicit `continue`, NOT a `D[i] = 0` sentinel.**
   If the skip predicate ever drifts (e.g. a future flooded-tile rule), a
   `D=0` shortcut multiplies the GS update by zero and destroys mass at that
   cell. Use `continue` in both the precompute and the sweep, keyed on the
   same skip mask.
3. **Rebuild trigger must include more than permeability.** Key it on
   `(mu | obstacles | is_wall | is_vacuum | permeability)` — not just
   permeability. Hull-breach ticks change `is_vacuum` and adjacency; a
   permeability-only trigger leaves stale `D` at the freed tile and its four
   neighbours.

### Pairs naturally with delta-tracked stamp_units

This technique only pays off if `D[i]` doesn't rebuild every tick. Which means
it pairs naturally with edit-triggered `stamp_units` (the delta-tracking
discipline), not with a per-tick full-field rebuild. The two work together:
when `stamp_units` only touches the moved-unit footprints, `D[i]` only
rebuilds those same cells + 1-cell neighbourhoods.

---

## Technique 2: Integer-bucket LUT (refinement of #1)

Quantize permeability to a small integer alphabet (current code already uses
`{0, 0.5, 1.0}` — that's a 3-symbol alphabet, giving a 9-entry `wsum` bucket
table at Q=2). The per-cell precompute becomes a 9-entry table build per tick;
the hot loop is one FMA + one MUL with no per-cell divide and no per-cell
reciprocal.

```cpp
// At tick start:
for (int k = 0; k <= 8; ++k) {                          // 9 buckets at Q=2
    Dlut[k] = 1.0f / (1.0f + mu * (k * 0.5f));
}

// Per-cell, in the sweep:
int bucket = wsum_int[i];                               // already integer
atmosphere[i] = (rhs[i] + mu * nb) * Dlut[bucket];
```

**Adds over #1**:
- Per-cell precompute → ~9-entry table build (free).
- The integer `wsum`-bucket key is **reduction-order-independent across
  SIMD/CUDA** in a way the float `wsum` is not.
- Natural stepping stone to the strict-power-of-two Q16.16 variant if you
  later go all-fixed-point.

### Caveats

- Requires a config-load-time invariant: **all permeabilities are integer
  multiples of `perm_quantum`**, asserted at `materials.py` and at the
  per-unit permeability hook (`gamemap.py:557`). Without that assert, a
  future material at `perm=0.37` silently snaps to the nearest bucket and
  changes diffusion rates by tens of percent with no warning.
- Making the bucket bit-exact requires `permeability` to become a `uint8`
  Q-quantized field, which propagates into the wave Laplacian, wind
  gradient, and smoke advection. **Cross-cutting change, not a localized
  solver patch.**
- `/fp:fast` must be audited at the LUT-build site (the build itself is a
  divide-and-multiply that fast-math can reorder differently across
  compilers). Lift the LUT-build into a strict-fp translation unit, or
  pre-compile the table once at startup if `mu` is per-tick-constant.

**When to use:** AFTER #1 has landed and the determinism harness exists.

---

## Technique 3 (endgame): Strict power-of-2 `mu` pin + Q16.16 shifts

Pick `d_atm` so that the open-interior denominator `1 + mu * 4` is a power
of two. The hot loop then becomes a fixed-point right-shift — bit-identical
on every target by definition.

```
1 + mu * 4 = 2^k    →    mu = (2^k - 1) / 4
```

For `k = 4`: `mu = 15/4 = 3.75` → `d_atm = mu / dt ≈ 36` at shipped `dt`.
That's a 5.6× change from the current `d_atm = 200`. Real physics change;
explosions settle differently; needs visual-harness A/B.

**Open-interior cells** hit the fast path (shift). **Fringe cells** (wsum < 4)
fall back to the LUT path. So this is really "shift for the bulk, LUT for the
edges" — modest extra speedup over #2 alone.

**Why this is the endgame, not the next step:**
- Requires the full Q16.16 migration of atmosphere (currently float32
  everywhere — touched by `smoke_dynamics.cpp`, `fire_simulation.cpp`, wind
  gradient, `recorder.py`).
- Requires the `d_atm` retune ratified as a deliberate physics change.
- Best framed as where you land once the LUT (#2) has proven itself and the
  determinism harness exists.

---

## What got rejected (don't pursue)

- **Newton-Raphson / Goldschmidt reciprocal iteration with a tabled seed.**
  Misidentifies the determinism problem (divide is already correctly-rounded;
  FMA + reduction order are the real hazards). On modern GPUs, `ptxas`
  already lowers `f32 div` to `rcp.approx + 2 NR steps` internally — so
  "replace divide with NR" on GPU is replacing hardware NR with software NR.
  Slower, not faster. Adversarial pass marked it `fragile`.

---

## Prerequisites before implementing any of these

From the synthesis, in order:

1. **Two-seeded determinism harness** (engine/ml/01 — specified, not written).
   Until it exists, you cannot tell whether the divide is even contributing
   to cross-arch divergence. Without the harness, every reformulation is
   guesswork.
2. **Audit `/fp:fast` and `-ffast-math`** at
   [cpp/CMakeLists.txt:14,16](../cpp/CMakeLists.txt#L14-L16). Replace blanket
   fast-math with explicit per-op intent: `-ffp-contract=off` and
   `#pragma fp_contract(off)` in the solver TU. Opt into specific
   approximations where you actually want them.
3. **Then land Technique #1.** Smallest, lowest-risk, no model change.
4. With the harness in place, **measure** what's causing cross-arch divergence
   in your current build. If it's FMA contraction (likely), step 2 already
   fixed it. If it's the divide (unlikely), #2 is the next move.

---

## Open questions worth pinning before adoption

- **Determinism contract: same-machine vs any-machine.** D-C, still
  unresolved across sessions. The answer determines whether Q16.16 #3 is the
  eventual destination or whether float32 + Technique #1 is the long-term
  answer.
- **Is `d_atm = 200` tuned or default?** Affects whether the `mu` pin for #3
  is on the table without a re-tune campaign.
- **Will permeability ever leave the `{0, 0.5, 1.0}` alphabet?** Continuous
  per-material perm, damage decay, soot occlusion would force #2's LUT key
  to grow a continuous fallback OR enforce quantization at config-load. The
  answer drives how much to invest in the integer-bucket machinery.
- **Is the recorder.py work writing or reading atmosphere bytes?**
  Technique #1's ~1 ULP per-cell shift needs to land before any recorded
  format freezes — or after with a versioning story.

---

_See also: [cpu_gs_fallback_brainstorm.md](cpu_gs_fallback_brainstorm.md)
(adversarially marked **fragile** — do not adopt without revisiting); the
unification work for the coherent dt policy
(`resolution_architecture_proposal.md`)._
