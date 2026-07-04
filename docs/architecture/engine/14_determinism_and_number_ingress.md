# Determinism & the Number-Ingress Rule

**Depends on:** [State & ownership (GameMap)](02_state_and_ownership.md),
[Config & hot-reload](12_config_and_hot_reload.md),
[FieldEdit (write primitive)](13_field_edit.md).

Cross-machine determinism is a **hard requirement** of this engine: lockstep
multiplayer exchanges *orders*, not state; distributed RL training generates
trajectories on many boxes that must be one reproducible world; replays are
`(seed, order stream)` and nothing else; and the property itself is a
portfolio-grade engineering result — proven 2026-07-04 by the `cuda-breached`
attestation (the same 30-tick trajectory digest, `ae1164ca…`, bit-identical on
an Ampere desktop and an Ada laptop, across MSVC 14.50/14.44, py3.11/3.12,
different CPUs and different GPUs).

That property is not a test you pass once; it is an invariant you can lose
with one line of code. This chapter is the law that keeps it:

> **Every number that enters synced simulation state must arrive through one
> of four audited doors. Everything else is banned from the synced path.**

The rule was not designed up front — every clause below was paid for by a real
cross-machine divergence in this repo (§8, the case log).

---

## 1. The boundary: synced vs local

**Synced state** is everything the digest hashes — everything that must be
bit-identical on two machines running the same seed and orders:

- the gameplay **fields** (`atmosphere`, `smoke`, `fire`, `temperature`,
  `heat`, `water_depth`, `wave_p`, `gas`, … — all int32 Q16.16),
- **unit state** (hp, position, facing, AP, life state, orders),
- derived **integer tables** consumed by synced solvers (material shift
  tables, occlusion caches feeding the heat path),
- the **RNG stream position**.

**Local state** is everything else: render buffers (`light_rgb`, smoke glow,
ripples-as-visuals), UI, audio, debug overlays. Local state may be float, may
be nondeterministic, may differ per machine — nobody ever hashes it. The
S2 raycaster is the canonical example of the split: one ray march, where the
`heat` channel (synced) is decoupled by construction from the RGB/`exp` light
path (local) — see [Ray engine](08_ray_engine.md).

**Geography enforces the boundary.** `src/simulation/` is the gated territory
(the lint, §6, scans exactly this tree). `prototypes/`, `tools/`, notebooks
and the renderer are free territory — use scipy, `np.random.normal`, whatever
iterates fastest. *The transition ritual is the move*: when experimental code
enters `src/simulation/`, the gate applies automatically. This preserves the
project's prototype-in-Python speed while making "we chose to really implement
it" a mechanical, lintable event rather than a judgment call.

## 2. Why this is achievable at all (the floating-point facts)

IEEE-754 **requires** `+ − × ÷ √` to be *correctly rounded*: for the same
inputs, same precision, same rounding mode, every conforming machine produces
the same bits. Pure algebraic float chains are therefore cross-machine exact —
*if* nothing rewrites them.

What breaks it, concretely:

| Breaker | Mechanism |
|---|---|
| **libm transcendentals** (`sin`, `cos`, `exp`, `log`, `atan2`, `pow`, …) | not required correctly rounded; implementations differ at the last ULP across CRT/UCRT versions, Python builds, OSes. An ULP is invisible — until `×65536` quantization flips an integer count |
| **BLAS/LAPACK** (`np.linalg`, `multivariate_normal`, scipy) | kernels are **dispatched by CPU microarchitecture at runtime**; different machines run different code. With degenerate eigen/singular subspaces the results differ by O(1), not ULPs |
| **FMA contraction / fast-math** | the compiler fuses `a*b+c` into one differently-rounded op, or reassociates sums; varies by compiler version and flags |
| **SIMD-dispatched vectorized math** | numpy's `np.exp` etc. pick AVX2/AVX-512 variants per CPU |
| **OS entropy / wall clock** | unseeded RNGs, `time.time()` in sim logic |

The mental model (transcendental *functions* : algebraic *functions* ::
transcendental *numbers* : algebraic *numbers*): the five algebraic operations
are exactly pinned by the standard; everything computed by iteration or
polynomial approximation is implementation-defined. So the engine is built on
integers plus the five pinned ops — and where it genuinely needs a
transcendental on the synced path, we ship our own integer implementation
(`atan2_q16`/`sin_q16`/`cos_q16` in `cpp/src/fixed_point.h`: Q.30 internals,
one final round, pinned 9.0e-6 accuracy, exact symmetries, `static_assert`ed
constants).

## 3. THE FOUR DOORS

Numbers enter synced state **only** through these. Every write site should be
attributable to a door in one sentence; if it can't, it's a bug.

### Door 1 — Integer / fixed-point arithmetic
Q16.16 (or plain int) values produced by integer operations from integer
inputs — including the deterministic kit: `mul_q16`, `div`/`recip_mul`,
`sqrt_q16`, `atan2/sin/cos_q16`, `quantize`/`dequantize`. Exact by
construction; the bulk of the engine (all seven solvers) lives here.
*Example:* the integer semi-Lagrangian smoke advection; the fire logistic.

### Door 2 — Constants quantized once at the boundary
Config/table floats (`config.toml`, species tables, material tables) snap onto
the Q16.16 grid **once, at load** — after which they are integers (or exact
dyadic floats `n/65536`). Authoring stays human-friendly decimal; the sim
never sees the unsnapped value.
*Example:* `generation.predefined_unit_attributes` — spawn stats are the
species means, `_q16_snap`ped; spawn hp is exactly `100.0`.
*Corollary:* changing a config value changes the golden — that is legitimate
re-baselining, not a determinism failure.

### Door 3 — Audited algebraic float bridges
Float chains **restricted to `+ − × ÷ √`** on deterministic inputs, compiled
under the pinned floor (`/fp:strict` on MSVC sim TUs, `--fmad=false
-prec-div -prec-sqrt -ftz=false` on CUDA — no contraction, no reassociation),
**quantized at the write boundary**. Correct rounding makes these
cross-machine exact; the audit is confirming no banned op hides in the chain.
*Example:* the raycaster heat deposit — float32 survival products →
`heat_quantize` → saturating integer add. The environmental-damage chain
(`peak_raw/65536 → linear float64 → quantize_hp_delta`) is another: verified
clean during the §8 case-2 investigation precisely because it is door-3-shaped.
Door 3 is a *concession*, not a preference — prefer door 1; every bridge is
one audit away from a leak.

### Door 4 — The raw RNG stream
Randomness enters **only** as the seeded PCG64 bitstream, which is
cross-machine exact. Allowed draws are the pure integer/dyadic transforms:
`.integers()`, `.random()` (= `uint64 >> 11` × 2⁻⁵³ — exact), `.choice()`,
`.shuffle()`, `.permutation()`, `.bytes()`. **Any transform of the stream into
game values must itself pass doors 1–3.** Distribution methods (`.normal()`,
`.multivariate_normal()`, `.gamma()`, …) are banned — they hide libm and
LAPACK inside (§8, case 2). When the game needs a distribution, we build it:
inverse-CDF via a fixed rational-polynomial approximation (pure `+−×÷` —
deterministic), correlation via a Cholesky factor computed in pure-Python
`+−×√` (algebraic → deterministic) or precomputed and checked in as door-2
constants, output snapped to Q16.16. That is the spec for the units-redesign
deterministic stat sampler.

## 4. The banned list

Banned **in `src/simulation/`**, enforced by the lint (§6). Each entry names
its replacement.

| Banned | Why | Use instead |
|---|---|---|
| `math.sin/cos/tan/atan2/exp/log/pow/hypot/…` (all libm transcendentals) | last-ULP CRT variance, amplified by quantization | the kit via `unit_fixed` (trig); design the need away; future kit extensions (exp/log) if chemistry demands |
| `math.sqrt` | — **not banned** | correctly rounded, door 3 |
| `np.linalg.*`, scipy anything | BLAS/LAPACK, CPU-dispatched | pure-Python algebraic routines at load time; precomputed door-2 constants |
| RNG distribution methods (`.normal`, `.multivariate_normal`, `.standard_normal`, `.gamma`, `.exponential`, `.poisson`, …) | libm/LAPACK inside a black box | door-4 raw draws + doors 1–3 transforms |
| `random.gauss/normalvariate/expovariate/…` | libm inside | same |
| unseeded `np.random.default_rng()` | OS entropy | thread the sim's seeded rng (or a fixed-seed child stream) |
| vectorized `np.exp/np.sin/…` over synced arrays | SIMD dispatch varies per CPU | integer solvers; door-3 scalar bridges |
| float `**` | CPython routes float pow through libm | `x*x` for squares; integer exponents unrolled; (known lint gap, §7) |
| wall clock / `time` in sim logic | obvious | tick counters |

## 5. The enforcement ladder

"Follow the rules" is not a mechanism. Three independent layers make
violations **fail loudly at the earliest layer that can see them** — the
Python-typing answer to "is there an abstract-class-like way to guarantee
this?":

**L1 — the lint (static, at commit time).** `tests/test_ingress_lint.py`
AST-scans `src/simulation/` for the banned list — imports, attribute calls,
unseeded RNGs. AST means comments/docstrings can't false-positive. Deliberate
audited uses carry an inline **`ingress-exempt: <why safe>`** pragma (on the
line or ≤6 lines above); an exemption without a written justification is a
review reject. First real catch: `materials.py`'s config-time `math.log2`
shift tables (exempted with rationale + an integer-`bit_length` TODO). Run
retroactively, L1 catches both historical incidents in §8 at commit time.

**L2 — representation (structural).** The strongest guarantee: **synced state
is stored as integers, so a nondeterministic float physically cannot live in
it** — it must pass an explicit `quantize()` to fit, and those call sites are
the entire audit surface. Shipped instances: every gameplay field is int32
Q16.16; [FieldEdit](13_field_edit.md) is the choke-point write API that
re-quantizes everything passing through. Owed to the units redesign: unit
synced attributes move to int-backed storage (hp as int32 counts; positions as
Q16.16 pairs per the locked position decision) behind properties that
snap-or-raise. Enforce at boundaries only — inner loops and free territory
stay unwrapped; a checked type on every temporary would strangle iteration
speed for nothing.

**L3 — the empirical backstop.** The per-field, per-tick digest
(`tests/_xarch_perfield_digest.py`) names the exact (field, tick) — and unit
sub-field — of any divergence; the golden aggregate is pinned in every CUDA
gate; cross-machine attestation re-runs the digest on other hardware
(`docs/cuda_xarch_ada_runbook.md`). Golden lineage is append-only provenance:
`542931c7…` → `60bd331f…` (S2 re-tune) → `453829a6…` (Q2-lift) → `ae1164ca…`
(spawn pin) → `6d690fda…` (P3 statuses — the unit record grew the
`__unit_status__` list, no field trajectory moved; current). L3 catches
whatever L1/L2 cannot see — including C++ regressions and the unknown
unknowns. It is the layer that caught everything in §8.

## 6. The C++/CUDA floor

The same rule, compiled: sim TUs build `/fp:strict` (no contraction, no
reassociation — the per-file list in `cpp/CMakeLists.txt`); CUDA kernels build
`--fmad=false -prec-div -prec-sqrt -ftz=false`; device integer helpers live in
`cpp/src/cuda_fixedpoint_device.cuh` (no `__int128` on MSVC-host nvcc →
`mul128_shr_signed`); order-free accumulation patterns (int64 `atomicAdd`
reductions, integer scatter) keep parallel execution deterministic. Every
solver is gated GPU==CPU at tol 0. Details: the CUDA arc docs; the contract:
**C++ is not exempt from the doors — it is where doors 1 and 3 are
implemented.**

## 7. Known gaps (accepted, documented)

- **float `**`** is not lint-detectable without false positives (v1 skips
  `BinOp Pow`); banned by convention, caught in review and by L3.
- **materials.py log2 exemption** — safe on current config (exact
  power-of-two ratios; integer output empirically stable on Ada); TODO:
  integer log2 via `bit_length`.
- **Unit records quantize floats at 1e-9** in the digest harness — fine while
  unit attrs are floats; dissolves when L2 lands int-backed attrs.
- **No C++ ingress lint** — the compile-flag floor + L3 gates carry that side.
- **exp/log have no kit implementation yet** — first needed by combustion
  chemistry / EOS work; extend the kit then (same Q.30 recipe as trig).

## 8. Case log — what each door was paid for with

1. **Unit facing / `math.atan2` (the Q2 fence, found June 29 → fixed by the
   Q2-lift).** Facing is synced; py3.11 vs py3.12 libm differed at the last
   ULP → whole-trajectory digest divergence on Ada. Fix: the integer trig kit
   (door 1) + facing/bullet-trig/HP-snap wiring. *Lesson → doors 1/3 and the
   transcendental ban.*
2. **Spawn stats / `rng.multivariate_normal` (found + fixed 2026-07-04).**
   The seeded *stream* was exact, but numpy's MVN factorizes the covariance
   through LAPACK SVD — CPU-dispatch-dependent, and with repeated variances
   the singular basis is non-unique: a forced kernel flip
   (`OPENBLAS_CORETYPE=NEHALEM`) moved stats by whole σ-fractions (mass
   96→64). `current_hp = vitality` → tick-0 `__unit_hp__` divergence, every
   field identical. Diagnosed to bit-for-bit repro in one session
   (`tests/_xarch_liveheat_dump.py`); fixed by the spawn pin (door 2).
   *Lesson → door 4's stream/transform split and the BLAS ban.*
3. **`materials.py` `math.log2` (near-miss, caught by the new lint the day it
   landed).** Config-time transcendental feeding synced integer shift tables —
   currently exact on power-of-two inputs, but rule-violating in principle.
   Exempted with written rationale + TODO. *Lesson → the pragma protocol:
   exemptions exist, but they are loud, justified, and tracked.*

The meta-lesson, twice over: **the culprit was never where the first
hypothesis pointed** (case 1's prime suspect was the raycaster's cos/sin —
falsified by experiment; case 2's was residual float in the raycaster heat
deposit — exonerated by reading the actual chain). The per-field digest plus
honest falsification found the truth both times. Trust the instrument, not
the hunch.

---

## Implementation status (2026-07-04)

| Piece | Status |
|---|---|
| Doors 1–3 across all seven solvers + fields (int32 Q16.16, /fp:strict floor) | ✅ shipped (the fixed-point + CUDA arcs) |
| Deterministic trig kit (`atan2/sin/cos_q16`) | ✅ shipped (Q2-lift `4b2d0d7`) |
| Door 2 at spawn (`predefined_unit_attributes`) | ✅ shipped (spawn pin `bb52368`) |
| L1 lint (`tests/test_ingress_lint.py`) | ✅ shipped (`bb52368`) |
| L3 digest + cross-machine attestation | ✅ operational — **`cuda-breached` tag: Ampere ↔ Ada bit-identical, golden `ae1164ca…`** |
| L2 for unit attributes (int-backed hp/pos, snap-or-raise properties) | 📝 owed to the units/stats redesign |
| Deterministic stat sampler (door-4 spec, §3) | 📝 owed to the stats redesign |
| Kit exp/log | 📝 when chemistry/EOS needs them |
| materials integer log2 | 📝 TODO noted in the exemption |
