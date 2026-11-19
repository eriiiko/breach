# cudaunit literature brief — GPU-resident agents under Q16.16 determinism

> Web survey, 2026-09-06, for the "cudaunit" design (thousands-to-millions of
> simple GPU-resident units; pilot species: larvae with a 3-state
> thermal/foraging AI) inside breach's deterministic integer sim.
> Constraints taken as given: Q16.16 integer sim state, bit-identical replays,
> CPU-vs-GPU and Ampere-vs-Ada bit-identity gates, integer atomicAdd is
> order-free, no device __int128.
> Local copy of the FLAME GPU 2 paper: `richmond2023_flamegpu2.pdf` (this
> scratchpad) — move under `docs/papers/` when the design doc lands.

---

## 1. Agent-based simulation frameworks on GPU

**Key citation:** P. Richmond, R. Chisholm, M. Leach, M. K. Heywood,
J. Pianini — *"FLAME GPU 2: A framework for flexible and performant agent
based simulation on GPUs"*, Software: Practice and Experience 53(8):
1659–1680, 2023.
Open-access PDF (White Rose):
<https://eprints.whiterose.ac.uk/199416/> (direct PDF linked from that page;
already downloaded to this scratchpad as `richmond2023_flamegpu2.pdf`).
Docs: <https://docs.flamegpu.com/> · Repo: <https://github.com/FLAMEGPU/FLAMEGPU2>

What it teaches us:

- **Agents are finite state machines; an "agent function" runs as one CUDA
  kernel over every agent currently in a given state.** This is exactly our
  shape: a `larva_step` kernel (or one kernel per AI state) over all live
  larvae. With only 3 states and trivial per-state work, a single kernel with
  a branch on the state variable is fine at our scale — FLAME's per-state
  kernel split exists to fight divergence at millions of complex agents.
- **Data is stored SoA** (one device array per agent variable), which is what
  makes the per-agent kernels coalesce. Confirms the layout choice (thread 3).
- **All agent-to-agent communication is indirect, via message lists**
  (brute-force, spatial-2D/3D, bucket, array variants): agents *output*
  messages in one function and *read* them in a later function, which
  structurally prevents read-write races. For us this is the important
  architectural idea even though v1 needs no messaging: agents read a
  *snapshot* of fields written by earlier tick slots, and their writes land
  via order-free integer atomics — same double-buffered read/write discipline,
  achieved with our existing field seams instead of message lists.
- **Birth and death are opt-in per agent function** (off by default to avoid
  overhead — `setAllowAgentDeath()` etc.). Death = a per-agent flag processed
  after the step; birth = new agents buffered during the step and appended.
  Under the hood both use prefix-sum scan + scatter (see thread 3). The
  opt-in framing is a good default for us too: the lifecycle machinery should
  cost nothing on ticks where nobody spawns or dies.
- **RNG:** FLAME GPU 2 uses cuRAND seeded from one simulation seed; the
  default cuRAND engine was later switched from XORWOW to **Philox**
  (changelog, `FLAMEGPU_CURAND_ENGINE` option: XORWOW/PHILOX/MRG) — i.e. the
  flagship GPU ABM framework independently converged on a counter-based RNG.
  Runs are reproducible for a fixed seed *on fixed hardware/build*; FLAME GPU
  does **not** claim cross-platform bit-identity (float agent variables,
  float RNG outputs, cuRAND dependency). That last mile — cross-arch,
  CPU-vs-GPU bit-identity — is ours to build, and integer Q16.16 state plus
  an integer counter-based RNG is precisely what closes it.

Newer frameworks (2023–2026), briefly:

- **ABMax** — J. et al., *"ABMax: A JAX-based Agent-based Modeling
  Framework"*, arXiv:2508.16508 (2025). <https://arxiv.org/abs/2508.16508>.
  JAX forbids changing array shapes at runtime, so ABMax handles agent
  add/remove with JIT-compilable **in-place update algorithms over
  fixed-capacity arrays** — independent validation of the "stable slots +
  mask, never realloc" lifecycle we want (thread 3). Also demonstrates
  vectorizing *many whole simulations* in parallel — the same `(N, h, w)`
  batch shape our RL-batch habits rule already mandates.
- **AMBER** — *"AMBER: A Columnar Architecture for High-Performance
  Agent-Based Modeling in Python"*, arXiv:2601.16292 (2026).
  <https://arxiv.org/pdf/2601.16292>. Columnar (=SoA) storage as the central
  performance idea, again.
- Survey (older but the standard reference): *"A Survey on Agent-based
  Simulation using Hardware Accelerators"*, arXiv:1807.01014 (ACM CSUR).
  <https://arxiv.org/pdf/1807.01014>.

**Design guidance:** adopt FLAME's three structural ideas — SoA arrays,
state-as-data with per-state (or state-branched) kernels, and strictly
staged read-snapshot/write-atomic phases — but *not* its float RNG or its
compacting birth/death machinery (reordering; see thread 3). cudaunit slots
into the tick conductor as one numbered slot like any other solver.

---

## 2. Counter-based / parallel RNG

**Key citation:** J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw —
*"Parallel random numbers: as easy as 1, 2, 3"*, SC'11 (Proc. Supercomputing
2011), ACM. Best Paper. DOI 10.1145/2063384.2063405.
Free PDF: <https://www.thesalmons.org/john/random123/papers/random123sc11.pdf>
Library: Random123, <https://www.thesalmons.org/john/random123/>

- The core idea is exactly our requirement stated as a thesis: a good RNG can
  be a **stateless keyed bijection applied to a counter** —
  `out = f_key(counter)` — so any thread can compute the k-th random value
  for any stream directly, with **no per-agent RNG state stored, no
  sequential dependence, no warm-up**. Keying by `(unit_id)` and countering
  by `(tick, draw_index)` gives every larva an independent, replayable
  stream indexed by time. This is the paper to archive.
- **Philox** (their multiplication-based family, e.g. Philox-4x32-10):
  integer-only — 32×32→64-bit multiplies (needs only `__umulhi`/64-bit
  product, *not* __int128), XORs, and a Weyl key schedule. Passes
  BigCrush/Crush; fastest of the three families on GPUs (they measured on
  Fermi; still true). **Threefry**: add/rotate/XOR only (no multiplier),
  slightly slower on GPU, trivially portable. **ARS** needs AES-NI — CPU
  only, irrelevant to us.
- Ecosystem convergence is strong evidence of soundness: cuRAND ships Philox
  (and FLAME GPU 2 made it the default engine), JAX's RNG is Threefry-2x32,
  PyTorch GPU uses Philox. Counter-based is the modern default for parallel
  RNG, not an exotic choice.

**Squares:** B. Widynski — *"Squares: A Fast Counter-Based RNG"*,
arXiv:2004.06278 (2020, rev. 2022). <https://arxiv.org/abs/2004.06278> ·
PDF <https://arxiv.org/pdf/2004.06278>

- A counter-based middle-square: `x = ctr * key`, then 4 rounds of
  square-and-rotate in plain 64-bit arithmetic (mod 2^64 — no wide multiply
  at all) → 32-bit output; a 5-round 64-bit variant exists. Passes BigCrush
  and PractRand per the paper. Fewer instructions than Philox-4x32-10 and a
  one-screen implementation.
- Caveats: single-author, far less independent scrutiny than Philox (which
  has 15 years of HPC deployment and third-party test batteries); the key
  must be chosen from the paper's recommended key-generation procedure (keys
  with poor digit structure weaken it), which is an extra footgun for a
  scheme keyed by raw `unit_id`. Philox accepts arbitrary keys by design.

**Design guidance / pick:** **Philox-4x32-10**, integer path only. Key =
`(unit_id, stream_salt)` where `stream_salt` names the purpose (turn-angle
vs tumble-check vs spawn), counter = `(tick, draw_index, 0, 0)`. Identical
code compiles for CPU (`uint64_t` products) and CUDA (`__umulhi` or 64-bit
mul) — same pattern as our existing `mul128_shr_signed` kit, so CPU-vs-GPU
bit-identity is a compile test, not a research problem. Map outputs to
Q16.16 by shifting the 32-bit word — never through float. Squares is the
fallback if profiling ever shows RNG on the critical path (unlikely: 3-state
larvae draw ~1–2 words/tick).

---

## 3. Data layout + lifecycle at scale

**Citations:**

- SoA-for-coalescing is stated as a design principle in the FLAME GPU 2
  paper (thread 1) and is the whole thesis of AMBER (arXiv:2601.16292);
  NVIDIA's CUDA best-practices guide says the same. No dedicated paper
  needed — cite Richmond 2023 for it.
- M. Harris, S. Sengupta, J. D. Owens — *"Parallel Prefix Sum (Scan) with
  CUDA"*, GPU Gems 3, ch. 39 (2007).
  <https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda>
  — scan + scatter is the canonical stream-compaction recipe (used by FLAME
  for death compaction and birth append).
- M. Safari et al. — *"Formal verification of parallel prefix sum and stream
  compaction algorithms in CUDA"*, Theoretical Computer Science 912 (2022).
  <https://www.sciencedirect.com/science/article/pii/S0304397522001232> —
  machine-checked proof that prefix-sum compaction computes a *function of
  its input* (order-preserving, no data races): i.e. scan-based compaction
  is **deterministic**; atomic-counter-based compaction is the variant that
  is *not* (output order depends on atomic timing).

What the literature implies for us:

- **SoA, fixed capacity.** One int32 Q16.16 (or narrower int) device array
  per attribute (`pos_x`, `pos_y`, `heading`, `state`, `energy`, `body_T`,
  …), allocated once at `N_max`, batch-shaped `(N_env, N_max)` per the
  RL-batch rule. AoS would break coalescing for exactly the per-field kernels
  we'll write; AoSoA tiling is a micro-optimization we don't need at 10^4–10^6
  simple agents.
- **Lifecycle: stable slots + alive mask, no compaction in v1.** Compaction
  via prefix sum *is* deterministic (proved above), but it **reorders and
  renumbers** — which would (a) change every agent's array slot between a
  run and a replay-with-different-camera, breaking per-slot digest
  comparison, (b) decouple `unit_id` from slot so RNG keys and recorder
  columns need an indirection anyway, and (c) add a sync point per tick.
  Dead agents at our simplicity cost ~nothing to skip with a branch
  (`state == DEAD → return`). ABMax (thread 1) reaches the same
  fixed-capacity-plus-in-place-update answer from JAX's static-shape
  constraint.
- **Spawning must not use an atomic ticket counter** (`slot =
  atomicAdd(&n,1)`) — that's the order-nondeterministic pattern. Deterministic
  alternative that stays parallel: prefix-sum over the dead mask to assign
  the k-th spawn request to the k-th free slot (both sides in a canonical
  order: requests by spawner id, free slots ascending). Spawn requests in v1
  will mostly come from the host/level layer anyway (nests laying eggs on a
  cadence), where a simple sequential fill is fine.
- **IDs:** give each unit a persistent `unit_id` (assigned monotonically at
  birth, int32/int64) distinct from its slot, or tag slots with a
  `generation` counter bumped on reuse — either keeps RNG streams and
  recorder identity stable across death/reuse. Digest the SoA arrays
  directly (slot-stable, so trivially comparable).

---

## 4. Spatial queries

**Key citation:** S. Green — *"Particle Simulation using CUDA"*, NVIDIA SDK
whitepaper (2007–2012 revisions).
PDF: <https://developer.download.nvidia.com/compute/DevZone/C/html_x64/5_Simulations/particles/doc/particles.pdf>

- The canonical uniform-grid recipe: hash each particle to its cell, **radix
  sort (cell, particle-id) pairs**, record per-cell start/end indices, then
  each particle iterates neighbors cell-by-cell. Because radix sort on the
  (cell, id) key is a pure function of the input (and ties broken by id make
  it order-independent even without stability), the neighbor **iteration
  order is fully deterministic** — this, not atomics-into-cell-lists, is the
  determinism-safe binning pattern. Atomic-append cell lists are faster to
  build but their order is timing-dependent; if ever used, the consumer must
  be an order-free reduction.
- **v1: skip binning entirely.** Our larvae interact with *fields*
  (temperature, walls, food density), not each other. That makes tick step =
  pure per-agent gather (read Q16.16 field cells at own tile — read-only
  snapshot, embarrassingly parallel, trivially deterministic) plus scatter
  via integer `atomicAdd` (eat food = subtract N from a food field cell,
  deposit heat through the gas-energy seam) — order-free by our own proven
  doctrine. There is no neighbor query to make deterministic. This is a real
  and legitimate simplification the frameworks can't assume; take it.
- When agent-agent interaction arrives (larva crowding, predation, the
  queen): the world is already a tile grid, so "binning" degenerates to
  sort-by-(tile, unit_id) — Green's recipe with the hash already computed.
  Alternatively, keep interactions **commutative integer reductions per
  tile** (e.g. each larva atomicAdds its mass into a per-tile crowding
  counter; next tick everyone reads it) and even sorting stays unnecessary.
  Prefer that form; it's the message-list idea (thread 1) collapsed onto our
  existing field machinery.

---

## 5. Determinism in parallel / distributed agent sims

**Citations:**

- P. Bettner, M. Terrano — *"1500 Archers on a 28.8: Network Programming in
  Age of Empires and Beyond"*, GDC 2001 / Gamasutra.
  <https://www.gamedeveloper.com/programming/1500-archers-on-a-28-8-network-programming-in-age-of-empires-and-beyond>
  The lockstep-RTS foundation: ship inputs, not state; the sim must be a
  deterministic function of (state, inputs). Our replay/digest architecture
  is this tradition; cudaunits must live inside it (unit AI consumes only
  synced state + the keyed RNG — no wall-clock, no frame-rate, no render
  reads).
- G. Fiedler — *"Floating Point Determinism"*, gafferongames.com.
  <https://gafferongames.com/post/floating_point_determinism/> — the
  practitioner's catalogue of why cross-machine float determinism is
  fragile (compiler flags, FMA contraction, x87 vs SSE, libm); the standing
  argument for our Q16.16 rule. (StarCraft used fixed point; AoE2 HD pinned
  compiler flags — both approaches are attested in the RTS literature.)
- S. Collange, D. Defour, S. Graillat, R. Iakymchuk — *"Numerical
  reproducibility for the parallel reduction on multi- and many-core
  architectures"*, Parallel Computing 49 (2015).
  <https://www.sciencedirect.com/science/article/abs/pii/S0167819115001155>
  — reproducible *float* summation is a whole research program
  (superaccumulators, ExBLAS). We sidestep it entirely: int64 atomicAdd is
  associative, so our reductions are reproducible by construction. Cite it
  as the "what we avoid by being integer" reference.
- *"Impacts of floating-point non-associativity on reproducibility for HPC
  and deep learning applications"*, arXiv:2408.05148 (2024).
  <https://arxiv.org/pdf/2408.05148> — recent measurement study; documents
  that float `atomicAdd`-style reductions on GPUs vary run-to-run, i.e. the
  precise failure mode our integer rule forbids.
- PDES literature (conservative Chandy–Misra / optimistic Time Warp): on
  inspection, reproducibility across schedules is "left to the
  implementation" in mainstream PDES systems; nothing there beats our
  synchronous fixed-tick data-parallel model, which is the easy case. No
  load-bearing citation needed; do not import PDES machinery.

**What the thread adds for us specifically:** the literature contains *no*
off-the-shelf framework claiming cross-vendor/cross-arch GPU bit-identity —
every published route to it is exactly the discipline we already run
(integer/fixed-point state, order-free integer reductions, no
timing-dependent ordering, pinned toolchains). Two cudaunit-specific rules
fall out: (1) **kernel launch geometry must never leak into results** — no
per-block partial results combined in an order that depends on block count;
per-agent independent work + integer atomics already guarantees this, so
keep the guarantee when tempted by shared-memory staging; (2) the CPU twin
of every cudaunit kernel is a straight sequential loop over slots — with
integer math and counter-based RNG the twin is bit-identical by
construction, which is what makes the existing A/B lockstep harness (and the
Ampere-vs-Ada gate) directly reusable.

---

## 6. Biology anchors for the pilot AI

- **H. C. Berg, D. A. Brown — "Chemotaxis in Escherichia coli analysed by
  three-dimensional tracking", Nature 239:500–504 (1972).** The origin of
  the run-and-tumble model: motion = straight "runs" ended by random
  reorientations ("tumbles"); the organism climbs gradients not by steering
  but by *modulating run length* — runs lengthen when conditions improve
  (temporal gradient sensing). One threshold comparison + one RNG draw per
  tick implements it. Successive headings are also forward-biased (~63°
  mean turn), a one-parameter touch that makes trails look organic.
- **L. Luo, M. Gershow, M. Rosenzweig, K. Kang, C. Fang-Yen, P. A. Garrity,
  A. D. T. Samuel — "Navigational Decision Making in Drosophila
  Thermotaxis", J. Neuroscience 30(12):4261–4272 (2010).** Open access:
  <https://www.jneurosci.org/content/30/12/4261>. Crawling larvae navigate
  temperature gradients with runs + head-sweep turns; they bias *when to
  turn* (postpone reorientation while temperature changes favourably) and
  *which way to turn* (accept/reject head sweeps by the temperature change
  they sample). I.e. the run-and-tumble kernel plus a turn-direction bias —
  a second RNG draw weighted by the local field gradient sign. Companion
  with richer dynamics: M. Klein et al., *"Sensory determinants of
  behavioral dynamics in Drosophila thermotaxis"*, PNAS 112(2):E220–E229
  (2015), <https://www.pnas.org/doi/10.1073/pnas.1416212112> (larvae on
  15–23 °C gradients move toward preferred temperature).
- **H. A. MacMillan, B. J. Sinclair — "Mechanisms underlying insect
  chill-coma", J. Insect Physiology 57(1):12–20 (2011).**
  <https://pubmed.ncbi.nlm.nih.gov/20969872/>. Below a critical thermal
  minimum (CTmin) insects enter *chill coma*: reversible paralysis — no
  movement, no feeding — recovered on rewarming; measured CTmin values span
  −16 °C to +21 °C across species, so a per-species Q16.16 threshold is
  faithful. Distinct (lower) lethal temperature kills. This is our third AI
  state for free, and it couples the larvae to the temperature field and
  the vent/fire systems in a legible, biologically-grounded way.

**Behavior section skeleton this supports:** states RUN / TURN / COMA.
RUN: move along heading; draw u ~ Philox(unit_id, tick); tumble iff
u < p(ΔT_favourable) (lower p when warming toward preference — Berg/Luo).
TURN: new heading = old ± biased draw (gradient-sign bias — Luo). COMA:
entered when tile T < CTmin, exit on rewarming (MacMillan/Sinclair); below
T_lethal → dead. All thresholds Q16.16 table rows in `materials.py` style
(a species is a table row, never an if-chain).

---

## Recommendations

- **RNG pick: Philox-4x32-10** (Salmon et al., SC'11), integer-only,
  stateless, key = (unit_id, stream_salt), counter = (tick, draw_index).
  15 years of scrutiny, arbitrary keys safe, needs only 32×32→64 multiplies
  (CUDA `__umulhi` / CPU `uint64_t` — no __int128), identical source both
  sides. Squares (Widynski) is the documented fallback for instruction
  count; not chosen because of its key-quality footgun and thinner
  independent validation.
- **Layout pick: SoA, fixed capacity `(N_env, N_max)`, one Q16.16/int array
  per attribute** (Richmond 2023; AMBER; ABMax's static-shape convergence).
- **Lifecycle pick: stable slots + alive/state mask + monotone unit_id (or
  slot generation), no compaction in v1**; spawning fills free slots via a
  deterministic prefix-sum pairing (never an atomic ticket counter — that is
  the one known-nondeterministic compaction pattern, per Safari 2022's
  taxonomy). Digests stay per-slot comparable; RNG keys stay stable.
- **Spatial pick: no binning in v1** — field-gather + integer-atomic scatter
  covers larvae completely; when agent-agent lands, prefer commutative
  per-tile integer reductions, else radix-sort (tile, unit_id) à la Green.
- **Three most load-bearing papers to archive under `docs/papers/`:**
  1. Salmon et al., *Parallel Random Numbers: As Easy as 1, 2, 3*, SC'11 —
     <https://www.thesalmons.org/john/random123/papers/random123sc11.pdf>
  2. Richmond et al., *FLAME GPU 2*, Softw. Pract. Exper. 2023 — OA PDF via
     <https://eprints.whiterose.ac.uk/199416/> (already fetched:
     `richmond2023_flamegpu2.pdf` in this scratchpad)
  3. Widynski, *Squares: A Fast Counter-Based RNG*, arXiv:2004.06278 —
     <https://arxiv.org/pdf/2004.06278>
  (Plus, for the behavior section's credit rule: Luo et al. 2010, open
  access at jneurosci.org.)
