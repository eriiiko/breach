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
- [ ] D1 — the widening (section 3).
- [ ] D2 — the loudness test (section 4).
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

(filled as the change lands.)

## 4. D2 — the test that proves it is loud

(filled as the test lands.)

## 5. The gate

(filled with the measured counts.)

## 6. Findings

(filled as they are found.)
