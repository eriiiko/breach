# Determinism, Fixed-Point & GPU-Residency for Physics/Learned Animation

**Agent 5 of 5 — ML-animation literature search, 2026-07-20.**
Theme: *can a physics-based or learned animation controller be made bit-identically
deterministic across machines, and at what cost — so it can drive gameplay (falls,
collisions, prone monsters) inside Breach's synced Q16.16 sim?*

Grounded in `docs/procedural_animation_brainstorm.md` (esp. §0.2 the determinism split,
§2 rungs (e)/(g)). This digest **extends** that brief — it does not restate the technique
stack. Where the brainstorm said "do a determinism cost evaluation before designing a
world-interacting body," this is that evaluation, in general form.

---

## 1. The determinism problem, restated for animation

Breach's iron rule exists because floats are not portable. The literature is unanimous on
*why*, and the causes fall into two families that map cleanly onto Breach's two animation
concerns — **rigid-body/IK math** and **neural-net inference**.

**Family A — arithmetic non-associativity + platform variance (the physics/IK side).**
The classic Fiedler / Gaffer-on-Games survey lists the concrete offenders, and they are
exactly the operations a pose solver uses:
- **x87 vs SSE**, 80-bit vs 64-bit intermediate precision, and rounding-mode drift. Gas
  Powered Games (Supreme Commander) had to *force* precision-control to 64-bit and
  round-nearest at startup and **re-assert every tick** because loaded DLLs silently
  changed the FPU control word.
- **Transcendentals** (`sin`, `cos`, `tan`, `atan2`) differ between Intel and AMD, and
  between platforms' libm — there is no IEEE mandate on their last bits. Box2D's author,
  making a *float* engine deterministic in 2024, still had to replace `sinf/cosf/atan2f`
  with his own approximations because `atan2f` "gives different answers on different
  platforms."
- **Fused multiply-add (FMA)**: a PowerPC/ARM `fma(a,b,c)` rounds once; an Intel
  `mul`+`add` rounds twice. Box2D had to compile with `-ffp-contract=off` to stop the
  compiler fusing.
- **Compiler optimization / fast-math**: algebraic reassociation reorders operations and
  changes the last bits; debug≠release.

**Family B — parallel reduction order (the NN-inference side).** Even with identical
per-op rounding, a *sum computed in a different order* gives a different result because
FP addition is non-associative. GPUs reduce in whatever order threads happen to commit,
so:
- **Atomic reductions** (a shared accumulator hit by many threads) are the canonical
  desync source — commit order varies run-to-run, so `Σ` varies.
- **Vendor kernels** (cuDNN, cuBLAS) legitimately pick different tiling / split-K /
  reduction trees per GPU arch, per input shape, per driver. The TAO verification paper
  (arXiv 2510.16028) frames this precisely: "vendor kernels legitimately reorder
  reductions and fuse operators; thread scheduling and atomics introduce nondeterministic
  execution traces," so "the same neural network on different hardware, or even repeated
  executions on the same GPU, can yield slightly different outputs."

Breach has already been *burned by Family B* on the CPU side: the X-ARCH finding
([[xarch-ada-beatb-finding]]) traced a cross-machine desync to **BLAS/LAPACK inside a
spawn-stat RNG path**, not the GPU. That is the same disease — a library silently choosing
a different reduction/codepath — and it is the single most important prior for this digest:
*the danger is not "the GPU," it is any float library call that leaks into the synced path.*

**Consequence for animation:** an animation controller becomes a determinism hazard the
moment its output crosses back into synced sim state. A pose that only draws pixels is
Family-free. A pose that says "this tile is now blocked / this marine is prone" inherits
the full problem.

---

## 2. Fixed-point rigid-body & IK physics — prior art and cost

**Fixed-point is the RTS/fighting-game gold standard for a reason.** StarCraft, and the
lockstep-RTS canon generally, run the *simulation* in fixed-point integer math and use
float only for rendering — because "different CPUs produce slightly different FP results."
The Q16.16 convention Breach uses (`fixed_point.h`) is the same convention the TS
deterministic-physics community reaches for ("values as integers scaled by a fixed factor,
default Q16.16 … the exact same bit-pattern everywhere," dev.to/shaisrc). This is exactly
Breach's proven path.

**The good news for animation math specifically.** The brainstorm's claim that "2-bone IK
and FABRIK are mostly mul/add/sqrt, so integer IK is plausible" is correct and the
literature supports it:
- **Analytic 2-bone IK** is a law-of-cosines evaluation: a couple of dot products, one
  `sqrt`, one `acos`/`atan2`. Q16.16 already ships `atan2_q16/sin_q16/cos_q16`; a Q16.16
  `acos` and `sqrt` complete the kit. No reductions, no libm — fully in-family-A and
  fully controllable.
- **FABRIK** (Aristidou & Lasenby) is iterative point-projection: each step is
  `normalize(child−parent)*bonelen` — mul/add + one `sqrt` per joint per iteration. It is
  *heap-of-scalars* math with no parallel reduction, so fixed-point FABRIK is
  bit-reproducible by construction once the scalar ops are.
- **Verlet / PBD secondary motion** (Müller et al.) is `x' = 2x − x_prev + a·dt²` plus
  constraint projection — again mul/add/sqrt, the friendliest possible case for Q16.16.

**The genuinely hard part is contact, not kinematics.** A *ragdoll* (marines falling into
each other, a monster collapsing prone with gameplay effect) is a constrained rigid-body
solver — iterative LCP / sequential-impulse contact resolution. Costs the literature flags:
- **Overflow / dynamic range.** Box2D's author's explicit argument against fixed-point:
  it would "reduce the feasible world size, have overflow, and likely more problems."
  Q16.16 gives ±32768 units of range and ~1.5e-5 resolution; velocities and accumulated
  impulses in a contact solver can blow the high bits. Mitigations (wider Q32.32
  intermediates, careful scaling) exist but are the "not free" the brainstorm warned of.
- **`sqrt`/normalize everywhere** in collision normals and constraint directions — each
  needs a deterministic Q16.16 integer sqrt.
- **Solver iteration count and ordering** must be fixed and identical (already true for
  any lockstep engine; Breach's sim already lives here).

**The under-appreciated alternative: deterministic *float*.** Box2D 3.1 (Aug 2024) achieved
**cross-platform determinism across x64+ARM and MSVC/GCC/Clang without fixed-point**, by
(1) no fast-math, (2) `-ffp-contract=off` to kill FMA, (3) custom `sinf/cosf/atan2f`, (4)
controlled SIMD. Their claim: "no noticeable performance impact," and fixed-point "would
likely be much slower, difficult to code…". **This does not change Breach's rule** — Breach
is *GPU*-resident and multi-vendor, a far harsher environment than Box2D's CPU test matrix,
and Breach has already paid for and proven the Q16.16 path. But it is the honest
counter-data point: for a *world-interacting body*, "deterministic rig" does not
*necessarily* mean fixed-point; it means "no uncontrolled float." On Breach's GPU sim,
Q16.16 is still the cheaper way to *guarantee* that than auditing every kernel's FP contract
per-arch. Box2D even flags its own residual risk — "I don't trust this result will hold" for
trig across untested platforms — which is precisely the fragility the iron rule buys out.

---

## 3. Deterministic NN inference — quantize, or keep the net outside the fence

This is the (g) "learned motion" rung's determinism question. Two routes; Breach's own
architecture already prefers the second.

**Route 1 — integer-only quantized inference.** Post-training quantization / QAT to int8
(or int16) makes the *entire* matmul+conv pipeline integer arithmetic. TFLite's
"integer-only" path is the reference implementation (weights+activations int8; Jacob et al.
"Integer Quantization for Deep Learning Inference," arXiv 2004.09602). Because integer
add *is* associative and there are no atomics-on-floats, an integer-only kernel with a
**fixed reduction order** is bit-reproducible cross-platform — this is exactly why the
embedded/FPGA community reaches for it ("integer-only inference pipeline that guarantees
deterministic computation for cross-platform consistency"; "Integer Intelligence,"
Electronics 15(5):1117). Accuracy cost is small: int8 "matches or is within 1% of
floating-point accuracy." **Caveat:** integer *arithmetic* is necessary but not sufficient
— you still must fix the accumulation order (no split-K atomics), or two GPUs summing the
same int8 products in different order can still differ if intermediate saturation/rounding
is order-sensitive. Integer removes Family A; you must *also* pin Family B.

**Route 2 — keep the net entirely outside the synced fence (Breach's native pattern).**
The cheapest deterministic NN is one whose floats never touch synced state. The physics
literature's framing: let the net run wherever (render thread, async, even a different GPU
in float), and let **only an integer "order" cross the fence** — a discrete action, a
quantized target pose, an integer footprint. This mirrors:
- **DeepMimic / AMP** (Peng et al. 2018/2021): the policy net outputs *joint target
  orientations*; a separate physics sim consumes them. The net is a *controller emitting
  targets*, not the authority on state. Breach can keep that split literally: PFNN/MANN or
  a PPO policy proposes a pose or an action; if anything gameplay-authoritative results, it
  is a **quantized integer target** the deterministic sim then solves.
- Breach's existing convention already assumes NN inference is render-side or produces an
  integer order — consistent with the training architecture (PPO + AlphaStar-style,
  determinism-as-superpower) captured in [[rl-litsearch-2026-07]].

**Recommendation collapses to one rule:** *the neural net never writes floats into synced
state.* Either it lives outside the fence and hands across an integer (Route 2, default,
zero fixed-point work), or — only if a learned controller must run *inside* the synced sim —
it is int-quantized **and** its reductions are order-pinned (Route 1, real work). For
32-px top-down Breach, Route 2 dominates; Route 1 is a parked contingency, not a plan.

---

## 4. Split-spine vs deterministic-rig — fleshed out

The brainstorm's two routes for world-interacting bodies (§0.2). Filling in the tradeoff:

**(a) Split spine — integer footprint carries gameplay; visual rig only decorates.**
- *What's authoritative:* the sim-side articulated-occupancy segment chain
  (`breach_unit_class_design.md`) — integer, synced, already the design for snakes/worms.
  It owns which tiles are blocked, where the grab/hit originates, whether a body fell prone.
- *What's cosmetic:* the FABRIK/verlet visual rig runs render-side in float, *reads* the
  integer segment positions, and drapes bones over them for beauty.
- *Fidelity lost:* the gameplay resolution is the segment granularity, not the visual
  joint granularity. A tentacle that *looks* like it curls around a corner only *blocks*
  the tiles its coarse segment chain occupies. Fine sub-segment contact (a fingertip
  grab) is visual-only — it cannot have mechanical effect finer than the integer chain.
  For top-down 48-px tiles this mismatch is nearly invisible; for a marine "falling over,"
  the fall's gameplay effect (prone flag, one blocked tile) is trivially representable as
  an integer footprint change while the *visual* topple stays a render-side ragdoll-lite.
- *Cost:* **near zero new determinism work.** Reuses the proven integer sim; the rig is in
  the render-exempt layer. This is the brainstorm's "cheap default," and it is correct.

**(b) Deterministic rig — the pose/physics solver itself is fixed-point.**
- *When needed:* only when a creature's *visual pose* must be gameplay-authoritative at
  joint resolution — e.g. an octopus arm whose exact curl determines exactly which enemy
  it grabs, beyond what a coarse segment chain can express.
- *What it costs (the "not free"):*
  - Q16.16 ports of any missing scalar ops (`sqrt`, `acos`) — small, one-time.
  - **Goldens + digest gates** for every solver, re-baselined only deliberately (Breach's
    established discipline; the tol-0 gate from the S8a residency arc is the template).
  - **Per-arch verification** — the actual recurring tax. Every target GPU/CPU must
    produce the bit-identical pose; this is the cross-machine test matrix Breach already
    runs for the sim ([[xarch-ada-beatb-finding]] proved it *can* pass).
  - For a *ragdoll* solver (contact, not just IK): overflow/scaling engineering per §2.
- *Fidelity gained:* joint-resolution gameplay authority. Rarely worth it top-down.

**The dividing question is not "IK or ragdoll," it is "at what spatial resolution must the
pose be mechanically authoritative?"** If tile-resolution suffices → (a). If joint-
resolution is required → (b). The brainstorm's instinct (a-by-default, b-only-if-must) is
the right ordering; this digest adds the resolution test as the concrete decision rule.

---

## 5. GPU-residency angle

Breach just moved the sim to GPU residency (CuPy/CUDA, S8a Path B, [[physics-closeout-status]]).
If per-monster animation physics runs *inside* the synced sim, it should be a batched GPU
fixed-point kernel like the rest — and the residency literature says that is achievable but
has one sharp edge.

- **Batched per-creature solvers fit the GPU model well.** N monsters × K bones × a fixed
  iteration count is embarrassingly parallel with no cross-creature coupling — one thread
  block per creature, integer math in registers/shared memory. This is the ideal case for
  determinism because each creature's reduction is *local* and can be warp-synchronous.
- **The edge: never reduce with float atomics; pin the order.** NVIDIA's own CCCL
  determinism guidance is explicit — atomics give "unordered execution across threads …
  a different order of operations between runs"; the fix is a "fixed, hierarchical tree
  rather than atomics" (warp shuffles + shared memory + predetermined kernel sequence) for
  run-to-run determinism, and RFA (fixed exponent-range binning) for *cross-GPU*
  determinism at a **20–30% perf cost**. The MICRO "Deterministic Atomic Buffering" work
  and the DPD-on-GPU literature reach the same conclusion: **structure reductions as an
  explicitly ordered tree; avoid float atomics.**
- **Fixed-point sidesteps most of this for free.** Integer add *is* associative, so an
  integer reduction is order-independent for the *value* (though saturation can still be
  order-sensitive — keep accumulators wide, Q32.32, and avoid mid-sum clamping). This is
  the deep reason Breach's Q16.16 choice pays off on GPU specifically: it collapses the
  expensive cross-GPU-RFA problem into "just don't overflow." Any animation-physics kernel
  that stays integer inherits the sim's already-proven cross-GPU bit-identity.
- **Practical rule for a residency kernel:** one block per creature, Q16.16/Q32.32
  integer, no float atomics anywhere, fixed iteration count, contact/constraint ordering
  deterministic (e.g. by stable creature/bone index, never by thread-arrival order).

---

## 6. The pragmatic determinism ladder for Breach

Four rungs, cheapest→dearest. Each monster/unit is placed on exactly one.

| Rung | What it is | Determinism work | Use for |
|---|---|---|---|
| **0. Render-only** | pose is pure decoration; sim state flows in, never back | **none** (render-exempt) | marines, humanoids, anything whose pose has no gameplay effect |
| **1. Split-spine cosmetic** | integer articulated-occupancy chain is authoritative; float visual rig decorates it | **≈none new** — reuses proven integer sim; rig is render-side | most world-interacting monsters (snakes, worms, tentacled things); marines' *fall* as an integer prone/footprint flag under a render-side topple |
| **2. Deterministic fixed-point IK/kinematics** | the pose *solver* is Q16.16 (2-bone IK / FABRIK / verlet), no contact | Q16.16 `sqrt`/`acos`; goldens + gates; per-arch verify | a creature whose *reach/curl* must be gameplay-authoritative at joint resolution (rare) |
| **3. Deterministic fixed-point ragdoll** | full fixed-point constrained rigid-body contact solver | rung-2 work **plus** overflow/scaling engineering, contact-order pinning, GPU-residency kernel | only if collapse/collision dynamics must be mechanically exact and synced (marines physically piling up as authoritative geometry) |

The ladder's shape is the whole recommendation: **effort roughly decuples per rung, and the
gameplay need almost never climbs past rung 1.**

---

## 7. Compute + engineering cost reality check

**Rung 0 — render-only.** Cost: normal graphics work, *zero* determinism engineering. The
brainstorm's P0 (marines) lives here forever.

**Rung 1 — split-spine.** Cost: the integer footprint model is *already designed*
(articulated occupancy). The only new work is the render-side rig reading it — and that's
in the exempt layer. **This is the sweet spot** and covers essentially all of Erik's stated
world-interacting cases: a tentacle that occupies tiles, a fall that knocks-prone/blocks-a-
tile. The fall example specifically: represent "prone" as an integer stance + footprint
delta (deterministic, tiny), and render a float ragdoll-lite topple over it (§2 verlet,
zero determinism stakes). You get the gameplay effect *and* the visual with no fixed-point.

**Rung 2 — deterministic IK.** One-time: ~a few Q16.16 scalar ops. Recurring: goldens per
solver + the per-arch cross-machine gate (Breach already runs this machinery; the *marginal*
cost is adding the solver to the existing golden/digest harness). Runtime: negligible —
IK/FABRIK is dozens of scalars per creature. The tax is **process** (gates, verification),
not FLOPs. Note it is exactly the tax the S8a residency arc paid (tol-0 gate, bit-identity
check) — so the team has done this dance and knows its real weight.

**Rung 3 — deterministic ragdoll.** The only genuinely expensive rung: fixed-point contact
solving with overflow discipline, deterministic constraint ordering, and a batched GPU
kernel if resident. This is a *project*, not a feature — comparable to a physics arc. Box2D's
author's warning (world-size, overflow, coding difficulty) applies in full. **Do not enter
rung 3 without a concrete gameplay reason that rungs 0–2 cannot serve.**

---

## 8. Breach-fit recommendation — where the fence sits for animation

**Put the determinism fence exactly where the iron rule already puts it, and make animation
respect it rather than perforate it.** Concretely:

1. **Default every creature to rung 0 or 1.** Marines → rung 0 (render-only), full stop.
   Any creature whose body "interacts with the world" → rung 1: its *gameplay footprint*
   is the existing integer articulated-occupancy chain; its *visual rig* is a render-side
   float decoration that reads the chain and never writes back. This is the brainstorm's
   split-spine, and it is the answer for ~all of the menagerie.

2. **Falls/knockdowns are a stance+footprint integer event, not a physics simulation.**
   "Marines sprinting into each other and falling over" → the *gameplay* effect (prone
   flag, blocked tile, maybe an integer knockback impulse resolved in the existing sim) is
   a handful of integers, already deterministic. The *tumble* is a render-side ragdoll-lite.
   Never let the visual ragdoll's floats decide the gameplay outcome.

3. **The NN never writes floats across the fence.** Learned motion (rung (g)), if it ever
   ships, stays outside the synced path and hands across an *integer action or quantized
   pose target* — DeepMimic's controller/sim split, matching Breach's existing NN pattern
   and the hard lesson of the BLAS-in-RNG desync. Only if a learned controller must run
   *inside* the sim does int-quantization-with-pinned-reductions (§3 Route 1) come off the
   shelf — and that is a parked contingency.

4. **Reserve fixed-point rig work (rungs 2–3) for a specific, argued need** — a single
   creature whose *joint-resolution pose* must be mechanically authoritative. When that day
   comes, the math is friendly (mul/add/sqrt, existing Q16.16 trig), the *contact* is the
   cost, and the process tax is the S8a-style golden/per-arch gate the team already runs.
   If it lives in the sim, it's a batched Q16.16 GPU kernel with **no float atomics, fixed
   iteration count, index-ordered constraints** (§5).

5. **On GPU residency:** because the sim is Q16.16-integer, any animation-physics that
   graduates into the synced sim inherits cross-GPU bit-identity *for free* (integer adds
   are associative) — provided it never introduces a float reduction. That single
   discipline ("no float in the synced kernel, ever, including inside animation") is the
   entire cross-GPU determinism story; it is cheaper to hold than to audit float FP
   contracts per-arch the way Box2D must.

**One-line fence:** *animation may consume synced integer state and may propose integer
events; it may never let a float — IK, ragdoll, or neural — become synced state.* That is
the iron rule applied to bones, and it makes rungs 0–1 (which is where Breach should live)
determinism-free by construction.

---

## 9. If you read three things

1. **Box2D 3.1 determinism post** (box2d.org/posts/2024/08/determinism/) — the honest
   modern counter-view: cross-platform determinism *without* fixed-point, plus the clearest
   short argument for *why fixed-point is still often the right call* (overflow, world-size).
   Read it to know exactly what deterministic float would cost Breach if it ever tempted you.
2. **NVIDIA CCCL floating-point-determinism blog**
   (developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/) — the
   GPU-residency playbook: atomics-free hierarchical reductions for run-to-run, RFA for
   cross-GPU (20–30% cost). Tells you precisely what to avoid in an animation kernel.
3. **Gaffer-on-Games, "Floating Point Determinism"**
   (gafferongames.com/post/floating_point_determinism/) — the canonical enumeration of
   *every* way floats desync (x87/SSE, transcendentals, FMA, fast-math, FPU control word)
   and why lockstep games either discipline them brutally or go fixed-point. The "why the
   iron rule exists" document.

---

## 10. Flagged citations

**High-confidence, directly load-bearing:**
- Fiedler, "Floating Point Determinism" & "Deterministic Lockstep," Gaffer On Games —
  x87/SSE, transcendental, FMA, FPU-control-word causes; Supreme Commander's re-assert-
  every-tick discipline. https://gafferongames.com/post/floating_point_determinism/
- Box2D 3.1 determinism (Aug 2024) — deterministic float via no-fast-math + `-ffp-contract=off`
  + custom `sinf/cosf/atan2f`; tested x64/ARM × MSVC/GCC/Clang; explicit fixed-point
  cost argument. https://box2d.org/posts/2024/08/determinism/
- NVIDIA CCCL FP-determinism blog — not-guaranteed vs run-to-run (hierarchical tree) vs
  GPU-to-GPU (RFA, 20–30% cost); atomics as the desync source.
  https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/
- Jacob et al., "Integer Quantization for Deep Learning Inference," arXiv 2004.09602 —
  int8 integer-only inference, within-1% accuracy; the TFLite integer path.

**Solid, corroborating:**
- TAO, "Tolerance-Aware Optimistic Verification for FP Neural Networks," arXiv 2510.16028 —
  crisp modern statement of *why* NN inference is non-reproducible cross-hardware (reorders,
  fusion, atomics).
- "Integer Intelligence: A Reproducible Path from Training to FPGA," Electronics 15(5):1117
  — integer-only pipeline for cross-platform determinism (embedded framing).
- Peng et al., DeepMimic (arXiv 1804.02717) & AMP (2021) — net-emits-targets / sim-is-
  authority split; the template for keeping learned motion outside the fence.
- StarCraft/RTS lockstep folklore (SnapNet, socratopia, dev.to/shaisrc) — fixed-point Q16.16
  sim + float render as the established RTS determinism pattern.
- MICRO 2020 "Deterministic Atomic Buffering" (research.cs.wisc.edu/hal) — hardware-level
  confirmation that atomic commit order is the reduction-determinism problem.

**Flagged uncertain / caveats:**
- Box2D's own hedge — "I don't trust this result will hold" for trig on *untested*
  platforms — is the reason deterministic-float is riskier than it looks; treat any
  "float determinism works" claim as *tested-matrix-specific*, not universal. Breach's
  multi-vendor GPU target is harsher than any of these CPU test matrices, which is the
  standing justification for staying Q16.16.
- Integer-only NN determinism requires **both** integer arithmetic **and** pinned reduction
  order — several sources state the first and imply the second; do not assume int8 alone
  buys cross-GPU bit-identity without controlling accumulation order/saturation.
- NetherRealm's "8 Frames in 16ms" (GDC 2018) confirms fighting-game rollback needs strict
  determinism and cost ~8 man-years to retrofit, but the public materials don't disclose
  fixed-point vs disciplined-float internals — cited as a *cost* data point only.
