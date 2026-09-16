# P3a-1 — the live radiation planes widen to int64

Issue #12, ray engine v2, patch P3a-1. Branch `12-p3a1-widen-live-rad-planes`
off `fire-12`, worktree `C:\Users\steen\projects\breach-p3a1`.

**Written incrementally, section by section, committed as it is written.** If
this report ends mid-sentence, the work it describes above that point is on the
branch and the rest was never started.

---

## 0. What this patch is, in one paragraph

`gmap.rad_net`, `gmap.rad_amb` and `gmap.rad_flux` — the three LIVE radiation
planes, the ones the old `cast_fire_heat` writes every tick and the temperature
fold reads — go `np.int32 -> np.int64`, and every C++ signature, binding and
kernel along their path widens with them **in the same commit**. Nothing else
changes. The old cast still writes the same integers; the fold still reads
whatever it reads today. **No golden moves.** The flip of the fold onto the
sweep's shadow planes is P3a-2 and is feel-gated.

The patch exists because a widening that is *incomplete* is worse than no
widening at all: pybind11's default `forcecast` would let a missed binding
silently hand C++ a truncated int32 **copy** of an int64 plane — fire heat
vanishing with no error anywhere. So the whole path moves at once, and every
plane argument of every binding touched gets `.noconvert()` (design row 36) so
a stale caller raises `TypeError` instead.

## 1. Status

- [x] Report skeleton committed (this file), before any code change.
- [x] Surface inventory verified against the tree (section 2).
- [x] D1 — the widening (section 3).
- [x] D2 — the loudness test (section 4).
- [x] Gate: CPU build + suite (section 5).
- [x] Gate: CUDA build + suite (section 5).
- [x] Findings (section 6).

## 2. The inventory — every 32-bit surface on the live planes

Verified against the tree by grep over `cpp/`, `src/`, `tests/`, `tools/`,
`renderer/`, not trusted from the brief. 26 surfaces in 11 files, plus 5 test
callers. The brief's list named 8 of them; the ones it did not name are marked
**[+]** and are the reason this had to be inventoried rather than patched
from a list.

### Storage (Python)

| # | Where | What |
|---|---|---|
| 1-3 | `src/simulation/gamemap.py:517/537/552` | `rad_net`, `rad_amb`, `rad_flux` allocations `np.int32 -> np.int64` |
| — | `gamemap.py:142`, `:452`, `:507`, `:528` | the comments that say "int32" / "stay int32 through P2" |

The live planes are **not** resident (`_RESIDENT_FIELDS` names only the four
shadow planes), so there is no CuPy device buffer to widen — checked.

### CPU C++

| # | Where | What |
|---|---|---|
| 4 | `raycaster.h:382/391/407` | the three `RadCtx` accumulator members |
| 5 | `raycaster.h:712-714` | `cast_from_fire_plane`'s three plane parameters |
| 6 **[+]** | `raycaster.h:309` `rad_signed_add` | the plain wrapping signed add — `int32_t*` -> `int64_t*`, `uint32_t` -> `uint64_t` |
| 7 **[+]** | `raycaster.h` (new) `rad_flux_saturating_add` | `heat_saturating_add` is int32-only and is what `rad_flux` accumulates through; it needs an int64 twin |
| 8 | `raycaster.cpp:989-991` | the definition's parameters |
| 9 **[+]** | `raycaster.cpp:542`, `:617` | the two `(int32_t)` NARROWING CASTS on the clamped term (`s32`, `x32`) |
| 10 **[+]** | `raycaster.cpp:645` | the `rad_flux` saturating add |
| 11 | `temperature_solver.h:658` | the fold's `const int32_t* rad_net` |
| 12 | `temperature_solver.cpp:60` | the definition's parameter |
| 13 **[+]** | `temperature_solver.cpp:251/258/263` | **the fold's arithmetic**: `shr_round0` -> `shr_round0_i64`, `sat_add_q16` -> its i64-delta twin |
| 14 | `physics_engine.h:166` + `physics_engine.cpp:77` | `step_tail`'s pass-through |
| 15 **[+]** | `fixed_point.h` (new) `sat_add_q16_i64` | one `FP_HD` definition of the symmetric saturating add with an int64 delta, so CPU and CUDA folds cannot drift (P1's `shr_round0_i64` precedent, immediately above it) |

### CUDA

| # | Where | What |
|---|---|---|
| 16 | `cuda_raycaster.h:65` (comment), `:115-121` | the emission cast's three plane parameters + the comment that claims a signed int32 atomic |
| 17 | `cuda_raycaster.cu:193` | `march_radiation_kernel`'s parameters |
| 18 **[+]** | `cuda_raycaster.cu:229/230/276/277` | **the four `atomicAdd(int32_t*)` calls** — the interesting part, section 3.2 |
| 19 **[+]** | `cuda_raycaster.cu:247`, `:274` | the two `(int32_t)` narrowing casts (`s32`, `x32`), the device twins of #9 |
| 20 **[+]** | `cuda_raycaster.cu:44` + `:284` | `heat_atomic_sat_add`'s int64 twin for `rad_flux` |
| 21 **[+]** | `cuda_raycaster.cu:376-440` | the host wrapper: parameters, `upload_opt`, and the **three `n * sizeof(int32_t)` D2H byte counts** |
| 22 | `cuda_temperature.h:110` | the temperature twin's `const int32_t* rad_net` |
| 23 **[+]** | `cuda_temperature.cu:202/220/224/225` | the device fold's parameter and its `shr_round0`/`sat_add_q16` pair |
| 24 **[+]** | `cuda_temperature.cu:487/568/583` | the host wrapper's parameter, `cudaMalloc` and **the `nb` byte count** |

### Bindings — all four, each made loud

| # | Where | Loud by |
|---|---|---|
| 25 | `bindings.cpp:613-615` (`cast_fire_plane_cuda`) | `py::array_t<int64_t, py::array::c_style>` + `.noconvert()` on all three |
| 26 | `bindings.cpp:2469-2471` (`Raycaster.cast_from_fire_plane`) | same |
| 27 **[+]** | `bindings.cpp:2101` (`TemperatureSolver.step`) | this arg is a **`py::object`** (`None` -> no fold), so `.noconvert()` is a no-op on it — loudness is an explicit `py::isinstance<py::array_t<int64_t>>` check + `py::type_error`, the pattern `rad_fluence` already uses 18 lines below in the same lambda |
| 28 **[+]** | `bindings.cpp:3457` (`PhysicsEngine.step_tail`) | same `py::object` idiom, same explicit check |

**Finding (the brief's #4 and the trap's real shape).** The brief listed three
bindings; there are four, and **two of them cannot be fixed with
`.noconvert()` at all**. `TemperatureSolver.step` and `PhysicsEngine.step_tail`
take `rad_net` as a nullable `py::object` and do `obj.cast<py::array_t<int32_t>>()`
inside the lambda. `py::array_t<T>`'s default `ExtraFlags` is `forcecast`, and
`.cast<>()` on a `py::object` runs that caster directly — **there is no overload
pass to annotate**, so `.noconvert()` on the `py::arg` is silently inert. A
stale int32 caller there would have been converted to a widened temporary with
no diagnostic whatsoever. The explicit `isinstance` + `type_error` is the only
loud form for this idiom, and it is already the house pattern (`rad_fluence`,
P1).

### Test callers that construct the planes

`tests/test_fire_heat_source.py:85/88/89`, `tests/test_pf1a_radiation_books.py:100-102`,
`tests/test_pr1_fire_plane_cast.py:146-148` (three fake-GameMaps),
`tests/test_radiation_sweep_gates.py:229/409` (the P1 int64->int32 narrowing
for the fold, and gate 5's `INT32_MAX` deposit), and
`tests/test_radiation_sweep_shadow_wiring.py` (D2). `tests/cuda_s2b_raycaster_live_check.py:307`
uses `np.zeros_like`, so it follows the widening for free.

## 3. D1 — the widening

Commit `0c7915b`, one commit, all 28 surfaces.

### 3.1 What made it behaviour-neutral, surface by surface

The patch has exactly three kinds of change, and each one is neutral for its
own reason.

**(a) Pure retyping — most of the list.** A pointer, a member, a numpy dtype, a
`sizeof`, a `cudaMalloc` size. Nothing arithmetic happens; the same integers
land in wider boxes.

**(b) Two deleted narrowing casts** (`raycaster.cpp:542/617` and their CUDA
twins). These looked like the risky part and are in fact the safest, because
`raycaster.h` already carried the proof that they never truncated. After the
flux limiter clamps a term,

    |x| <= (|dT| << his) >> RAD_LIM_SHIFT
        <= (T_MAX_PHYS * 65536 << 5) >> 4        (his <= 5, steel)
         = 16000 * 65536 * 2 = 2.097e9  <  2^31 - 1 = 2.147e9

so every clamped term — and the halved rule-2 term, smaller still — already fit
int32 **exactly**. The narrow was total. Deleting a total narrow is a no-op on
every value the march can produce, and I rewrote that comment block to say so:
the bound that justified the cast is now the proof that removing it changed
nothing.

**(c) Three arithmetic functions that got int64 twins.** This is the only place
a value could in principle differ, and in each case the twin agrees with the
narrow form on the whole int32 range:

| Narrow form | Twin | Differs only when |
|---|---|---|
| `rad_signed_add(int32_t*)`, plain wrapping | `rad_signed_add(int64_t*)`, same modular add on `uint64` | the per-cell SUM crosses 2^31 — the ceiling this patch exists to remove; P1 measured the live max at 7 % of it |
| `shr_round0(q16)` + `sat_add_q16` in the fold | `shr_round0_i64` + `sat_add_q16_i64` | `rad_net` leaves int32 range. **`shr_round0_i64` is P1's, added for exactly this day** ("what keeps the fold byte-identical on the day its plane widens (P3)"), and `tests/test_fixed_point_i64_twins.py` gates the agreement |
| `heat_saturating_add`, ceiling `INT32_MAX` | `rad_flux_saturating_add`, ceiling `INT64_MAX` | the sensor's per-cell sum crosses 2^31 |

Two deliberate NON-changes belong here too, because each was a place the patch
could have quietly grown:

* **`rad_quantize_signed` is untouched.** `rad_flux`'s per-term quantize still
  saturates at `INT32_MAX`. That is a *rounding boundary*, not storage; moving
  it would be a behaviour change in a place this patch does not own. Only the
  accumulator widened.
* **The fold still reads the old cast's plane.** Not one line of the flip is
  here. `rad_fluence`, the clamp, `dyn_heat_atten_q`, the digest spec and every
  `[materials.*]` key are exactly as P2b left them.

### 3.2 The CUDA atomic — why it is still exact and still order-free

The one genuinely interesting surface. `rad_net`/`rad_amb` were scattered with
a **plain signed `atomicAdd(int32_t*, int32_t)`**, and the header's claim rested
on two properties: CUDA's int atomicAdd wraps on overflow exactly as the CPU's
`rad_signed_add` does (so the backends agree even out of band), and integer
addition is associative and commutative (so the result does not depend on the
order the hardware interleaves the atomics).

CUDA has **no signed 64-bit `atomicAdd`**. The device overloads are
`unsigned long long`, so the widened form is the tree's standard idiom
(design §8.2; `cuda_bulk_transport.cu:313`, `cuda_combustion.cu`,
`cuda_eos_resident.cu:940`):

```cuda
__device__ __forceinline__ void rad_atomic_signed_add(int64_t* addr, int64_t delta) {
    atomicAdd((unsigned long long*)addr, (unsigned long long)delta);
}
```

**Both properties survive, and neither needs a caveat:**

1. **Exactness.** Unsigned addition modulo 2^64 *is* two's-complement signed
   64-bit addition — the same bit operation, read through a different type. So
   the int64 read back is precisely what the CPU's `rad_signed_add` computes.
   That is not an approximation that holds "while values are small": the CPU
   twin is *itself written through `uint64`* (`*cell = (int64_t)((uint64_t)*cell
   + (uint64_t)delta)`) for the identical reason — to make the out-of-band wrap
   deterministic instead of C++ UB. The two functions are the same function.
2. **Order-freedom.** Modular addition is associative and commutative. Any
   interleaving of the atomics produces the same residue, hence the same bits.
   This is the *same* argument that held at 32 bits, one width up — nothing
   about it depended on the width.

And the reason the widening does not move a number today: every term is still
clamped below 2^31 (§3.1(b)), so on all in-range accumulations the int64 result
and the old int32 result are equal integers. The widening only changes what
happens *past* 2^31, which is the point.

The D3 sensor's atomic is the other half. It is **saturating**, not plain —
order-free for non-negative deltas because saturation composes with a monotone
non-negative stream in any order. Its int64 twin is a CAS loop on
`unsigned long long` clamping at `INT64_MAX`, keeping that argument intact:

```cuda
__device__ __forceinline__ void rad_flux_atomic_sat_add(int64_t* addr, int32_t delta) {
    if (delta <= 0) return;
    unsigned long long* uaddr = (unsigned long long*)addr;
    unsigned long long old = *uaddr, assumed;
    do {
        assumed = old;
        const int64_t cur = (int64_t)assumed;
        const int64_t sum = (cur > INT64_MAX - (int64_t)delta) ? INT64_MAX
                                                               : (cur + (int64_t)delta);
        old = atomicCAS(uaddr, assumed, (unsigned long long)sum);
    } while (assumed != old);
}
```

The delta stays int32 — it is still `rad_quantize_signed`'s output, deliberately.

The real proof that the device side is exact is not this argument but the
existing **tol-0 CPU/CUDA lockstep gates**: the CUDA temperature twin and the
emission cast are held bit-identical to the CPU march, and they run green
against the widened path (§5).

### 3.3 The byte counts, which are the quiet trap

Three `n * sizeof(int32_t)` D2H copies in `cuda_raycaster.cu` and one shared
`nb` in `cuda_temperature.cu`. The latter is the nastier of the two: `nb` is
`n * sizeof(int32_t)` and is used by **eight** other planes that are still
int32, so it could not simply be changed — `rad_net` needed its own `nb64`. A
missed byte count here is not a type error; it is half a plane copied back.

## 4. D2 — the test that proves it is loud

Commit `e6266d6`, extending `tests/test_radiation_sweep_shadow_wiring.py`
(where P1 put the same gate for the four shadow planes).

Two tests, because the four bindings need two different mechanisms:

* `test_the_three_live_planes_are_int64_and_the_engine_refuses_a_narrow_one` —
  the planes' dtype, and `PhysicsEngine.step_tail` refusing an int32 `rad_net`.
* `test_the_casts_and_the_fold_all_refuse_a_narrow_live_plane` —
  `Raycaster.cast_from_fire_plane`, `cuda_raycaster_cast_from_fire_plane` and
  `TemperatureSolver.step`, each refusing each of the three planes narrow.

**Property**: no caller of the radiation path can hand C++ a narrow plane and
be silently given a truncated copy to write into.
**Breaks if**: a plane is allocated narrow again, or a binding loses its
`.noconvert()` / its dtype check.

**Non-vacuity is built in, not checked once.** Every refusal is paired with the
identical call carrying int64 planes, which must not raise. Confirmed by hand
as well, and the messages say which guard fired:

```
OK      TemperatureSolver.step accepts int64 rad_net
OK      Raycaster.cast_from_fire_plane accepts the int64 planes
OK      PhysicsEngine.step_tail accepts the int64 rad_net
RAISES  TemperatureSolver.step    TemperatureSolver.step: rad_net must be an int64 numpy array ...
RAISES  step_tail                 PhysicsEngine.step_tail: rad_net must be an int64 numpy array ...
RAISES  cast_from_fire_plane      incompatible function arguments. The following argument types are supported:
```

That third message is worth reading twice: *overload resolution refused it*.
That is what `.noconvert()` buys, and it is exactly what dropping `forcecast`
alone would **not** have produced — pybind11's second (convert) pass would have
performed the safe int32 → int64 cast into a discarded temporary and the call
would have succeeded, writing nothing the caller could see (design row 36).

## 5. The gate

Both builds rebuilt from this branch on DESKTOP-0E98HUV (RTX 3070, sm_86,
CUDA 12.4): `cpp\build_cpu_home.bat` then `cpp\build_cuda.bat`, both
`BUILD_EXIT=0`, no narrowing warnings.

| Gate | Result |
|---|---|
| **Goldens unmoved** | **Yes.** No golden, no spec toml, no `DIGEST_SPEC_VERSION`, nothing in `tests/field_digest.py` or `tests/_xarch_perfield_digest.py` is touched on this branch — `git diff --stat <base>..HEAD` over those paths is empty. `GOLDEN_AGGREGATE` and every digest test pass unchanged. **Nothing was re-baselined.** |
| **Full suite, CPU build** | `2532 passed, 29 skipped, 4 xfailed, 0 failed` (1 m 57 s) |
| **Full suite, CUDA build present** | `2557 passed, 4 skipped, 4 xfailed, 0 failed` (3 m 24 s) — the 25 CUDA gates that skip on a CPU-only box all run and all pass |
| Float ratchet (`test_no_float_in_sim_tu.py`) | green; counts only went down (it reports pre-existing slack in five TUs I did not create and did not tighten) |
| Ingress lint (`test_ingress_lint.py`) | green |
| `git status` | clean; every commit staged by explicit path, never `git add -A` |

### The CUDA lockstep gate is the real proof of the device widening

`test_cuda_pr1_fire_plane.py` holds the emission cast CPU↔CUDA at **tol 0** on
all three planes. Fully exercised on this branch:

* 600-emitter synthetic firestorm — bit-identical, tol 0;
* equal-T lattices at 180 / 443 / 1000 / 2500 game — bit-identical, tol 0;
* the hot/cold pair with the **flux limiter engaged** (raw per-ray net
  2 934 280 704 against a per-pair budget of 98 304 000) — bit-identical, tol 0;
* a **12-tick live evolving burn**, where a single divergent count would
  compound into a different emitter set the next tick — bit-identical on all
  three planes every tick, `rad_net + rad_amb` summing to **exactly 0**, and
  temperature bit-identical across the whole trajectory;
* the 600-emitter cost budget: 1.994 ms against a 3.0 ms budget (unchanged).

`PR4_RADIATION_RESULT: PASS`.

Two of those scenarios report `flux max = 2147483647` — `RAD_FLUX_CEILING`
engaging, with the CPU saturating add and the CUDA CAS loop agreeing on it to
the count. That is the preserved ceiling (§6 finding 3) proving itself on both
backends at once.

And gate 4/5 in `test_radiation_sweep_gates.py` compare the **widened C++ fold**
to the integer reference tick for tick and still match, which is the fold's own
byte-identity check.

### What this patch did NOT do

The fold still reads the old cast's plane; the flip is P3a-2. Untouched: the
maximum-principle clamp, `rad_fluence`, `dyn_heat_atten_q`, the digest spec,
every golden, `[materials.*]`, `cool_shift`, `thermal_mass`, every calibration
key, and the old cast and all its dials (P3c). Nothing was merged, pushed, or
deleted.

**Note for the orchestrator**: `fire-12` has advanced to `9b65b29` since this
branch was cut at `34634b7`. I did not rebase or merge — the branch is exactly
its eight commits on the original base.

## 6. Findings

### Finding 1 — there are FOUR bindings on this path, and two of them cannot be made loud with `.noconvert()`

The brief named three bindings. There are four:
`cuda_raycaster_cast_from_fire_plane`, `Raycaster.cast_from_fire_plane`,
`TemperatureSolver.step` and `PhysicsEngine.step_tail`.

The two the brief did not name are the two that matter, because they take
`rad_net` as a **nullable `py::object`** (the `None` → no-fold idiom) and do
`obj.cast<py::array_t<int32_t>>()` inside the lambda. `py::array_t<T>`'s default
`ExtraFlags` is `forcecast`, and `.cast<>()` on a `py::object` runs that caster
**directly** — there is no overload pass, so `.noconvert()` on the `py::arg` is
silently inert. Had I applied the brief's recipe mechanically, those two would
have looked hardened and been exactly as silent as before.

The loud form for this idiom is an explicit
`py::isinstance<py::array_t<int64_t>>` check raising `py::type_error`, which is
already the house pattern — P1 used it for `rad_fluence` eighteen lines below
one of the two sites. Both now use it, and D2 has a test per binding.

**The general shape**: "drop forcecast / add noconvert" is advice about *typed*
arguments. Any argument that must be nullable in this codebase is a
`py::object`, and every one of those is a forcecast hole that no annotation can
close.

### Finding 2 — `rad_net` and `rad_amb` never approach their ceiling in real play; the widening is pure headroom

Measured on the shipped playground under the **full conductor**, one wood tile
lit the canonical way (heat + fire), 120 ticks:

| plane | peak | of its old ceiling | ticks over it |
|---|---|---|---|
| `rad_net` | 556 375 847 | **25.9 %** of 2³¹ | **0 / 120** |
| `rad_amb` | 0 (nothing escapes this enclosed level) | — | 0 / 120 |

Tick 0 reproduces P1's number exactly (`max|rad_net| = 150 148 108`), which is a
useful check that the harness measures the same thing P1 did.

So for these two the widening changes no number on any reachable trajectory; it
buys the room the sweep's values will need at P3a-2. (Driven as a *bare*
`PhysicsRunner.step` loop with no conductor the scene runs away to the
T_MAX_PHYS rail and `rad_net` does cross 2³¹ by tick 12 — but that is an
un-conducted runaway, not a trajectory the game can produce, and under the old
code it was the documented out-of-band wrap regime.)

### Finding 3 — THE ONE THAT MATTERS: `rad_flux`'s ceiling is a unit-damage cap, it binds in ordinary play, and I did NOT lift it

**This is the finding to take to Erik.**

My first cut of the widening did the obvious thing: the accumulator goes int64,
so its saturating add now clamps at `INT64_MAX` instead of `INT32_MAX`. Every
golden stayed put and the whole suite was green, which is precisely why it is
worth stating how nearly that shipped.

Then I measured it. Same scene, same 120 conducted ticks:

| | |
|---|---|
| ticks where `rad_flux` exceeded `INT32_MAX` | **38 of 120** |
| most cells over it in a single tick | **249** |
| first tick over it | **12** — with the level only at **2740 game** |
| peak, un-capped | 9 861 538 651 = **459 %** of the old ceiling |
| the ceiling, in game units | 32 768 |

`INT32_MAX` on this plane was **never an overflow guard**. `rad_flux` is the D3
radiant-flux SENSOR: it is outside the energy ledger, no solver reads it, and
its one consumer is **unit heat damage** —
`exchange.apply_environmental_damage` takes the per-tile peak as
`phi = raw / HEAT_SCALE`. So that number is a cap on **how hard a fire can burn
a marine**, and it engages in ordinary play from an ordinary fire — not in some
pathological regime.

Widening it would therefore have made fires meaningfully more lethal in hot
rooms. That is a **feel change** (CLAUDE.md: *feel-adjacent changes never
auto-merge*, HUMAN-TEST gate) smuggled inside a patch whose defining property is
behaviour-neutrality — and no golden would ever have caught it, because
`rad_flux` is deliberately in neither `DIGEST_FIELDS` nor `SIM_FIELDS`.

**So the ceiling stays exactly where the int32 plane put it**, now as an
explicit named constant (`raycaster.h::RAD_FLUX_CEILING`) read by both the CPU
add and the CUDA CAS loop, with the measurement written where the next reader
will meet it. P3a-1 is thereby behaviour-neutral on every trajectory I can
measure, not merely on every gated one.

`tests/test_radiation_sweep_shadow_wiring.py::test_the_flux_sensor_ceiling_did_not_move_with_the_width`
is the tripwire, and it is non-vacuous — the scene reaches the cap (157 capped
cell-ticks in 24 ticks), so the test exercises the clamp rather than an absence
of flux.

**THE OPEN QUESTION, for Erik, not for this patch**: *should* that cap exist?
It is an artefact of a storage width nobody chose for this purpose, it is
currently invisible to every gate, and the plane it lives on can now carry
4.5 billion times more. Lifting it is a one-constant change with a HUMAN-TEST
gate. Leaving it is also defensible — it has been the shipped feel throughout
the fire-tuning arc, and #12's calibration was measured with it in place. What
is not defensible is letting a widening decide it.

### Finding 4 — `report_p1.md` does not exist

The brief says to read `docs/ray_engine_v2_scheme_study_2026-09-13/report_p1.md`
and follow its pattern. There is no such file; the study directory has
`report.md`, `report_p0.md`, `report_p2a.md`, `report_p2b.md`. P1's findings
were folded straight into design v3 (rows 36–38, commit `13bdbb7`) instead of
getting a report of their own. I reconstructed the pattern from the design rows
and from P1's own commits (`b22182a`, `d5ec711`) plus the shadow-plane binding
and its wiring test, which carry it in the code. Worth correcting in whatever
index the orchestrator is working from.

### Finding 5 — three small things the patch had to get right that the list did not mention

* **`cuda_temperature.cu`'s `nb`.** One shared `const size_t nb = n * sizeof(int32_t)` serves nine planes there. `rad_net` needed its own `nb64`; reusing `nb` is not a compile error, it copies half a plane.
* **The two `(int32_t)` narrowing casts** in each march (CPU and CUDA). Deleting them is what makes the widening real, and `raycaster.h` already carried the proof they never truncated — so the deletion is provably value-preserving, not merely believed to be.
* **`shr_round0_i64` was already there, waiting.** P1 added it with the comment "what keeps the fold byte-identical on the day its plane widens (P3)", and `tests/test_fixed_point_i64_twins.py` already gated its agreement with the narrow form. Its sibling for the landing (`sat_add_q16_i64`) did not exist; I added it beside `sat_add_q16` as one `FP_HD` definition, so the CPU and CUDA folds cannot drift.

