# Cross-Machine Determinism Without Fixed-Point? A Research Report for Breach

> **STATUS: RESEARCH REPORT — not canon.** This is a literature- and theory-backed analysis to inform a design decision. It does not modify any architecture chapter. Where the underlying research synthesis was checked and found overstated, the prose below uses the corrected version; see *Corrections folded in* at the end.

---

## 1. The verdict, with the math up front

**Coarsening gameplay decisions to a coarser resolution of the underlying float does NOT achieve cross-machine / cross-architecture determinism.** Your stated worry — that the tiny difference *reintroduces itself via "superposition of many such numbers"* — is correct, and it is the rigorous crux. There is no free lunch between "render-only float" and "fixed-point": coarsening-as-a-decision-filter on a float you keep iterating fails cross-machine, and coarsening-as-storage (snap the *whole iterated state* to the grid every tick so no sub-bin residual survives) is just fixed-point under another name.

The decisive framing: **the right question is never "how many decimals carry information."** In a chaotic system the last decimal *does* carry information about the future once it is fed back. The only question that matters is **"does the raw STATE evolve bit-identically?"** — which coarsening never achieves, because the difference already exists in the value being quantized.

Two compounding mechanisms make this rigorous.

### Mechanism 1 — Boundary-straddling: rounding relocates a threshold, it does not remove it

The coarsening map is the uniform scalar quantizer

```
R(x) = q · round(x / q)
```

a step function with decision boundaries at every midpoint **(k + ½)·q**. Suppose machines A and B already differ by ε > 0 in the pre-quantization float (from FMA contraction, x87-vs-SSE intermediates, MSVC-vs-nvcc transcendentals, or reduction order). Then `R(x_A) ≠ R(x_B)` precisely when the true value lies within ε of a boundary. Per-comparison disagreement probability ≈ **ε/q** — small but *nonzero*, and a single flip is a full-bin discrete difference (ignite vs not; a lethal HP step vs not).

You do not eliminate the knife-edge; you replace one threshold with a periodic comb of them spaced `q` apart, and a value computed even 1 ULP differently on the two machines flips across the nearest `(k+½)q` to a different bin. This is the documented cross-machine hazard for lockstep floats: a value near a boundary on machine A lands just across it on machine B, and that one discrete difference then feeds back.

*(Correction, see §6: the scalar quantizer `R` is in fact **idempotent** — `R(R(x)) = R(x)` — so the "non-idempotent snap-rounding drift" pathology from computational geometry [Halperin–Packer Iterated Snap Rounding; Belussi Snap-Rounding-with-Restore] does **not** apply here; that is a 2D line-segment-arrangement phenomenon, not scalar number rounding. And coarser bins are **sparser**, not "a denser lattice." Neither of those changes the operative conclusion: the boundary-tie hazard is real because the float feeding the bin is not bit-identical.)*

### Mechanism 2 — Chaotic amplification with logarithmic delay (the "superposition" term)

Breach's atmosphere/wind/smoke advection–diffusion plus the fire → heat → buoyancy → pressure → wind feedback loop is chaotic, with a positive largest Lyapunov exponent λ. Two trajectories differing by δ separate as **δ·e^{λt}**. The time until the gap reaches a coarse bin Q (forcing a decision flip) is

```
t*  ≈  (1/λ) · ln(Q / δ)
```

— **logarithmic** in both the coarseness Q and in 1/δ. The consequences:

- Making bins **1000× coarser** buys only `ln(1000)/λ ≈ 6.9/λ` extra ticks.
- To **double** the agreement horizon you must **square** the precision ratio `Q/δ`.
- With double precision δ ≈ 2⁻⁵³ and a 3-significant-figure quantum, `t* ≈ ln(10⁴…10⁶)/λ ≈ 9–14` Lyapunov times. For λ of order 0.1–1 per tick that is tens-to-hundreds of ticks — **seconds at 60 Hz, not a match.**

This formula is standard Lyapunov theory in the *infinitesimal linear regime* (valid below the saturation scale ≈ the attractor diameter; beyond it no precision buys time). Liao's *Clean Numerical Simulation* is the right **motivation** (correct digits decay linearly, reliable horizon grows only linearly in digits = logarithmically in 1/δ) but does not state this closed form — cite it as motivation, not as the formula's source.

**Your "superposition of many numbers" worry, made precise**, is the *upscale error cascade* of CFD/weather. Sub-quantum residuals in many *coupled* cells are not averaging noise; the nonlinearity organizes them into a grid-scale, decision-level difference in finite time, independent of how small they start. Weather's hard ~2-week predictability ceiling — *regardless of how small the initial error is* (Zhang et al., *J. Atmos. Sci.* 2019) — is the macroscopic existence proof.

### Why bit-identity is the only escape

Chaos amplifies a **zero** seed to zero: `0·e^{λt} = 0`. If ε_machine is *exactly* 0 across heterogeneous ISAs, there is nothing to amplify, for an unbounded horizon. Every float scheme gives only a *bounded* horizon `t ≈ (1/λ)·ln(tol/ε)`. That asymmetry is the entire justification for Breach's apparatus.

**The shadowing lemma does not rescue you.** It guarantees a numerical pseudo-orbit tracks *some* true orbit — not the *same* orbit on both machines. Two players each see a plausible-but-different game = still a desync. Worse, the classical lemma requires uniform hyperbolicity; fluid sims violate this (unstable dimension variability), so even that weak comfort expires after a finite shadowing time ("there does not exist a trajectory that shadows the truth for an arbitrarily long time" — Lorenz-96 study, arXiv:1106.0084). Note the subtlety: for Anosov systems a bi-infinite pseudo-orbit *is* shadowed by a *unique* true orbit — but the two machines feed the lemma *two different* pseudo-orbits, so they get two different (each unique) shadowing orbits. Uniqueness does not collapse them.

---

## 2. The menu of real alternatives

The genuine design axis is **"correctly-rounded float vs fixed-point,"** *not* "fixed-point vs coarsening" — coarsening is disqualified as a determinism mechanism by §1.

| Approach | Guarantee | Cost vs fixed-point | Fit for Breach |
|---|---|---|---|
| **Full fixed-point (Q16.16)** — current choice | **HARD** cross-machine, cross-arch (CPU≡CUDA, MSVC≡nvcc, A≡B), **unbounded** horizon. ε_machine = exactly 0 forever; chaos can't amplify zero. | Baseline (this *is* fixed-point). Cost = per-field conversion, range/overflow discipline, transcendental host-baking. Near-minimal for the guarantee. | **Best fit** for the stated cross-arch tol-0 requirement. Already shipped the hardest artifact: GPU heat == CPU heat at tol 0 (merge `90ad719`). Lever is SCOPE, not removal. |
| **Reproducible-float** (correctly-rounded transcendentals + reproducible reductions + /fp:strict + --fmad=false + force SSE) | **HARD** cross-arch *for the restricted op-set {+,−,*,/,√}* that IEEE-754 mandates correctly-rounded — *iff* every op is provably in that set and transcendentals use one shared lib. Unbounded horizon, but needs a **permanent empirical bit-gate** per arch (no by-construction guarantee). | Comparable engineering, **weaker** guarantee (least trustworthy exactly across the CPU/GPU boundary). RFA reductions +20–30% and *still* no CPU==GPU. | The real alternative to Q16.16. Breach already does ~all of it. Fixed-point's surviving edge: exact *integer* threshold compares + already-host-baked transcendentals. |
| **Software IEEE-754 float** (Berkeley SoftFloat) | **HARD** cross-arch bit-identity on any conforming compiler, unbounded, by construction. | **10–50× runtime** in the per-cell kernel; similar/worse engineering. Still needs host-baked transcendentals. | Poor as the CUDA inner loop (GPU-hostile). A fallback only for a value-path that can't be proven correctly-rounded. Strictly dominated by fixed-point for the hot loop. |
| **Robust-predicates / detect-then-exact** (EGC filter; Shewchuk/CGAL + Simulation-of-Simplicity) | **HARD for the DECISION** (sign/threshold) — *iff* the cheap float pre-pass is itself reproducible on both machines and the error bound holds on both. Reduces to fixed-point/soft-float at the decision point. | Adds a cheap float fast-path; the reproducible fallback is still integer/soft-float. Potential perf win, same correctness dependency. | An **optimization layered on** fixed-point, not a replacement: lets the integer kernel run only *near* thresholds. Validates Breach's render-float exemption as textbook EGC. Worth prototyping. |
| **Rollback + checksums** (GGPO model: detect, don't prove) | **NOT a guarantee — a MONITOR.** Converts "prove identical forever" → "detect divergence fast + recover." Still needs the core to actually *be* deterministic. | Cheap, additive. Does not remove the need for determinism in the authoritative core. | **Highest value-per-effort cheap step.** Extend the existing field-digest harness to a per-N-tick checksum; tells you exactly which field/horizon drifts so you harden only that one. |
| **Server / host-authoritative streaming** | **Eliminates the cross-machine-identity requirement entirely** — only one box's bits are authoritative; the fluid can be unconstrained float. Collapses to single-machine determinism (free). | Zero fixed-point engineering; cost paid in **bandwidth + input latency**. | Strong fit for co-op/PvE (zombies, few human agents). ML training trivially satisfies it (trainer = the single authority). The "nuclear" avoid-fixed-point option for heterogeneous MP. |
| **Single-machine-only** (same binary, same ISA) + pin reduction order & RNG | Float is **already deterministic** here. **HARD same-machine**; **NO** cross-machine/cross-arch guarantee. | Near-free (order + RNG pinning). | Fully covers **headless ML-training replay** in pure float. Covers lockstep only among identical-hardware clients. Does NOT cover CPU-client-vs-GPU-client or A-vs-B. |

**Two reductions deserve emphasis.** Two's-complement **wrapping** integer addition is exact and associative on every conforming CPU/GPU, so an integer `atomicAdd` reduction is *simultaneously* order-independent **and** cross-arch bit-identical — a property no float reduction delivers without pinning every per-element op. NVIDIA's own RFA reproducible-float reduction costs +20–30% on large inputs and guarantees only **GPU-to-GPU** identity, *not* CPU-vs-GPU. So integer atomics are strictly stronger *and* cheaper than every float-reduction alternative.

*(Two caveats from verification, §6: (i) the order-independence guarantee holds for **wrapping** integer add; **saturating** add is **not** associative — e.g. clamp [−100,100]: (120+10)−10 = 117 but 120+(10−10) = 120 — so a *saturating* atomic reduction is order-independent only if the true sum can never reach the clamp under any ordering. Confirm Breach's per-cell heat deposits never saturate mid-reduction. (ii) "IEEE-identical" is a misnomer for integers; the correct statement is "exact in two's-complement modular arithmetic, identical across conforming hardware.")*

---

## 3. Single-machine vs cross-machine — and what Breach can defer

This distinction is what the whole question turns on.

**Float arithmetic is ALREADY deterministic when you run the identical binary on the identical instruction set**, modulo a short, non-FP list: (a) parallel atomic/reduction **order** (float add is non-associative → nondeterministic even on one GPU) and (b) **RNG seeding**. Both are cheaply pinnable *without* fixed-point — integer/saturating atomics (Breach already uses these for heat) plus frame-indexed RNG. Bruce Dawson: same-binary-same-CPU determinism "just works" absent those.

Note the field-proven nuance: float lockstep is **not** strictly impossible cross-ISA — Factorio ships cross-architecture deterministic float-lockstep (PC x86 ↔ Switch ARM; x86_64 ↔ Apple-Silicon arm64, validated by per-tick state-CRC over 2,417 tests). But they paid for it by *replacing* the divergent operations (custom transcendentals) and hunting undefined behaviour (out-of-range `double→int` casts), and that case is CPU-vs-CPU. **Breach's MSVC-CPU vs nvcc-CUDA is harder**, because the GPU adds massively-parallel reduction-order nondeterminism *and* a separate compiler/math-library on top of cross-ISA differences. So the honest justification for the apparatus is "fixed-point is the robust/pragmatic choice for a chaotic cross-arch GPU sim," **not** "float lockstep is literally impossible cross-ISA."

**Consequently Breach's two needs split cleanly:**

1. **Headless ML-training replay** is almost always same-binary + same-box → needs ONLY order + RNG pinning, **not fixed-point.** Spend no fixed-point effort here.
2. **Lockstep multiplayer** needs cross-machine identity *only if* clients have different CPUs/builds/GPUs. If the canonical sim is the GPU path run identically on every client's GPU (with SM-count-independent reductions pinned), you again need only order + RNG.

The **expensive** goal — CPU-vs-CUDA bit-identity, machine-A-vs-B across MSVC/nvcc, different vendor GPUs — is the *only* thing the full Q16.16 apparatus is strictly required for.

**What Breach needs NOW vs can DEFER:**

- **NOW** — (i) the CPU-vs-CUDA *dev-validation gate* (already shipped: GPU heat == CPU heat tol 0, merge `90ad719`), which lets the GPU port be trusted against the CPU golden; (ii) same-box ML replay (effectively free).
- **DEFER** — true cross-*machine*, cross-*vendor* heterogeneous lockstep (CPU client vs GPU client; A's MSVC build vs B's). For a co-op/PvE game this is a nice-to-have, not a launch requirement — and it is the case that, *if* pursued, genuinely forces fixed-point on the authoritative chaotic fields.

Honest framing: **Breach has already paid for the hardest single artifact (CPU≡CUDA tol-0).** The open axis is not "fixed-point vs coarsening" (coarsening is disqualified) but **"how much of the authoritative surface must stay fixed-point vs can revert to constrained-float + checksums once the chaotic fields are made render-only."**

---

## 4. When coarsening *is* the right tool

Coarsening is genuinely useful — for a *different* problem than cross-machine bit-identity. It can make an already-identical value robust at a threshold, or render a non-shared value irrelevant. Good-enough regimes:

1. **Render-only / non-authoritative fields** crossing no gameplay threshold and never feeding back — Breach's `light_rgb`, `smoke_glow`, smoke density. Nondeterminism never becomes a state difference, so coarsening isn't even needed; leaving them float is correct (textbook EGC: protect only values feeding a discrete branch).
2. **Single-machine / same-binary** determinism (same-box ML replay, identical-GPU lockstep) — float is already deterministic once order + RNG are pinned; coarsening adds nothing but is harmless.
3. **Server-authoritative** — only one box's bits matter; the chaotic fluid can run in unconstrained float.
4. **Rare-flip-tolerable single-player sandbox** where occasional divergence has no correctness consequence.
5. **Drift-control once arithmetic is already integer** — snapping stored state each tick bounds residual to q/2 — but that is just fixed-point restated.

Where coarsening is **NOT** good enough — the hard cases Breach actually has: lockstep across heterogeneous hardware where a full-state hash demands *exact* equality every tick (a hash is maximally sensitive — one differing bucket flips it), and any case requiring an authoritative *chaotic* field reproduced bit-for-bit over a long horizon. There, coarsening only delays desync logarithmically and the checksum gate makes "close enough" meaningless.

---

## 5. Recommendation (opinionated)

**Reject coarsening as a determinism mechanism.** Keep it only as (a) a free-but-redundant snap *on top of already-bit-identical integer values*, and (b) the existing render-only float exemption.

Lightest path meeting Breach's real needs, ordered:

1. **Do NOT rip out the fixed-point core.** It is the EGC "exact stage" specialized for determinism, the field-correct choice for the cross-arch (MSVC-vs-nvcc) requirement Breach explicitly has, and it already bought the tol-0 CPU≡CUDA heat gate — the single hardest artifact, done. Integer `atomicAdd` is strictly stronger *and* cheaper than every float-reduction alternative. This is near-minimal cost, not gymnastics.

2. **The real lever is SCOPE: shrink the deterministic surface.** The fluid is the largest and *only* chaotic part of state. Push the render-only carve-out as far as design tolerates — make authoritative *only* the few scalar crossing-quantities the fluid hands to gameplay (ignition flag, `water_depth` heat-sink, shockwave impulse, HP). These are few, non-chaotic once extracted, and integer-friendly. Let full pressure/wind/smoke stay float render-only per client (shipped practice: Company of Heroes ran nondeterministic explosions over a synced sim).

3. **Add a runtime authoritative-state checksum** — extend the field-digest harness to a per-N-tick xxHash/CRC over the integer gameplay core. Change the goal from *prove-forever* to *detect-fast* (GGPO ladder: hash → rollback → resync). Highest value-per-effort; ship and harden empirically only the proven-divergent field.

4. **ML replay:** pin RNG (frame-indexed) + reduction order → pure-float on one box. Spend no fixed-point effort there.

5. **Keep host-precomputed transcendentals + --fmad=false + /fp:strict regardless.** IEEE-754 leaves transcendentals *unspecified* (recommended, never required, to be correctly rounded — because of the Table-Maker's Dilemma); no flag fixes them. These measures are cheap and float-compatible.

6. **Gated escape hatch:** if future features need *device* sin/cos/exp (dynamic light dirs, heat-shield materials), link ONE shared correctly-rounded libm (RLIBM-32 / CORE-MATH) or SLEEF-1ULP called identically host+device behind a bit-identity gate, rather than abandoning baked dirs.

7. **Nuclear deferral for heterogeneous PvP** if it ever matters: host-authoritative streaming eliminates cross-machine identity at a bandwidth/latency cost.

**Net: do not add coarsening; do scope-down + checksum.** Convert "fixed-point everything that crosses a threshold" into "fixed-point only the proven-divergent authoritative chaotic crossing-fields."

---

## 6. Corrections folded in (where the research synthesis was overstated)

For transparency, three claims in the underlying synthesis were checked and trimmed; the prose above already uses the corrected forms:

- **Scalar quantizer idempotence.** `R(x) = q·round(x/q)` **is** idempotent (`R(R(x)) = R(x)`); the "non-idempotent snap-rounding relocation/drift" pathology and the Iterated-Snap-Rounding / Snap-Rounding-with-Restore remedies belong to *line-segment-arrangement* rounding in computational geometry, **not** scalar number quantization. "Coarsening → denser lattice of thresholds" is backwards — coarser bins are *sparser*. The operative hazard (boundary ties on a non-bit-identical float) stands; the geometry citations and arXiv:2403.09603 (which is about verifiable GPU training, not boundary-straddling) were misattributed and are dropped.
- **Saturating atomics.** Order-independence + cross-arch identity hold for **wrapping** integer add; **saturating** add is *not* associative, so a saturating atomic reduction is safe only if the true sum can never reach the clamp under any ordering. ("IEEE-identical" for integers → "exact in two's-complement modular arithmetic.")
- **"Float lockstep only within a single ISA + single compiler."** Overstated — Factorio ships cross-ISA deterministic float-lockstep (x86 ↔ ARM ↔ Apple Silicon). The corrected justification for Breach's fixed-point is *pragmatic robustness for a chaotic cross-arch GPU sim* (MSVC-CPU vs nvcc-CUDA adds GPU reduction-order nondeterminism + a second math library), **not** an impossibility claim. The cures in shipped titles were "remove OR replace with a pinned deterministic implementation" — never "round it away."
- **Liao CNS as the formula source.** The closed form `t* ≈ (1/λ)·ln(Q/δ)` is standard Lyapunov theory; Liao's Clean Numerical Simulation papers (arXiv:1609.09344, arXiv:1707.04720) support the *tradeoff* empirically and should be cited as motivation, not as the formula's origin. The weather ceiling (Zhang et al. 2019) and shadowing/UDV claims (arXiv:1106.0084) verified clean.

The IEEE-754 / transcendentals claim, the chaotic-divergence formula, the EGC split, and the shadowing-lemma argument all verified **CONFIRMED**.

---

## 7. References

**IEEE-754 & GPU floating point**
- NVIDIA, *Floating Point and IEEE 754 Compliance for NVIDIA GPUs* — https://docs.nvidia.com/cuda/floating-point/index.html ("same inputs give same results for individual IEEE 754 operations… on the CPU and GPU"; host vs device math libs "often differ slightly").
- CUDA C Programming Guide, *Mathematical Functions* appendix (ULP bounds, e.g. double `sin` within 2 ULP).
- IEEE-754-2008/2019: correctly-rounded elementary functions are *recommended*, not *required* (only +, −, *, /, √, FMA, remainder mandated).
- J.-M. Lefèvre & J.-M. Muller, *The Table Maker's Dilemma* / worst cases for correct rounding — https://perso.ens-lyon.fr/jean-michel.muller/Intro-to-TMD.htm.
- Correctly-rounded libms: RLIBM (arXiv:2111.12852), CORE-MATH, CR-libm.

**Chaos, predictability, shadowing**
- Liao, *Clean Numerical Simulation* — arXiv:1609.09344, arXiv:1707.04720 (motivation: correct digits decay ~linearly).
- Zhang et al., *What Is the Predictability Limit of Midlatitude Weather*, *J. Atmos. Sci.* 76(4), 2019 (intrinsic ~2-week ceiling).
- *Aggressive shadowing of a low-dimensional model of atmospheric dynamics* (Lorenz-96) — arXiv:1106.0084 ("no trajectory shadows the truth for an arbitrarily long time"; UDV as cause).
- Unstable dimension variability obstructing shadowing — arXiv:chao-dyn/9911027; Lorenz-96 heterogeneity arXiv:2306.04336; *On the risks of using double precision in spatio-temporal chaos* arXiv:1910.11976.
- Anosov (1967), Bowen (1970/1975) shadowing lemma; Scholarpedia "Shadowing"; PlanetMath "Shadowing lemma".

**Reductions / reproducibility**
- Singh et al., *Impacts of floating-point non-associativity on reproducibility for HPC and DL* — arXiv:2408.05148.
- NVIDIA, *Controlling Floating-Point Determinism in NVIDIA CCCL* (RFA: GPU-to-GPU only, +20–30%) — https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/.
- ProofWiki, *Integer Addition is Associative*; Wikipedia, *Saturation arithmetic* (non-associativity example); Wikipedia, *Quantization (signal processing)*.

**Robust geometry / EGC**
- C. Yap, Exact Geometric Computation paradigm; J. Shewchuk, *Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates* (1997); CGAL exact-computation paradigm.
- Halperin & Packer, *Iterated Snap Rounding* (Comp. Geom. T&A, 2002); Belussi et al., *Snap Rounding with Restore* (ACM TSAS) — *for context on why these do NOT apply to scalar quantization*.

**Game-dev determinism (post-mortems & practice)**
- B. Dawson, *Floating-Point Determinism* — https://randomascii.wordpress.com/2013/07/16/floating-point-determinism/.
- G. Fiedler, *Floating Point Determinism* — https://gafferongames.com/post/floating_point_determinism/.
- Terrano & Bettner, *1500 Archers on a 28.8* (Age of Empires lockstep) — https://zoo.cs.yale.edu/classes/cs538/readings/papers/terrano_1500arch.pdf.
- Factorio FFF-370 / FFF-371 (cross-ISA deterministic lockstep; out-of-range cast UB) — https://factorio.com/blog/post/fff-370 , https://factorio.com/blog/post/fff-371.
- Klotho (FP64 32.32 fixed-point, cross-platform) — https://github.com/xpTURN/Klotho; bepuphysics1int — https://github.com/sam-vdp/bepuphysics1int.
- *Cross-platform RTS synchronization and floating point indeterminism* (gamedeveloper.com); shaderfun, *Understanding Determinism Part 1*; christian-seiler.de/projekte/fpmath (software-FP route); Berkeley SoftFloat.