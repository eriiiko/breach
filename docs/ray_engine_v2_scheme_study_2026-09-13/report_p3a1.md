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
- [ ] Gate: CPU build + suite (section 5).
- [ ] Gate: CUDA build + suite (section 5).
- [ ] Findings (section 6).

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

(filled with the measured counts.)

## 6. Findings

(filled as they are found.)
