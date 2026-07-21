# S8c item 1 — `cast_fire_heat` batched device cast (impl design, 2026-07-21)

**Status:** DESIGN v1 (for adversarial critique before build). **Arc:** S8c
(kickoff `docs/s8c_kickoff_2026-07-21.md`). **Workflow:**
autonomous-patch-workflow, design-pass-first (sim-affecting). **Machine:**
Lenovo/Ada (build + gate here).
**Gate:** raycaster/S2 digest (`tests/cuda_s2_check.py`,
`tests/cuda_s2b_raycaster_live_check.py`) + S8a full-engine `heat` A/B tol 0
(`tests/cuda_s8a_check.py`, `heat` ∈ `_FIELDS`) + full `pytest tests -q` +
a many-fires payoff bench. **NO re-baseline.**

---

## 0. The problem (measured: 3 fps, 2026-07-20 B5 feel-test)

`PhysicsRunner.cast_fire_heat` (`physics_runner.py:824`) enumerates every
burning tile (`fire > 0`, row-major) and — on `--cuda` — casts **one source
per call** through `bp.cuda_raycaster_cast`. Each call
(`raycaster_cast_directional`, `cuda_raycaster.cu:166`) does a full
`cudaMalloc` + H2D of **all** inputs (gas planes, light_atten, heat_atten, the
running `heat` plane + render scratch), one march, then D2H of every output and
`cudaFree`. With hundreds of burning tiles that is hundreds of whole-plane
round-trips per tick. That transfer tax — not the march — is the 3 fps.

## 1. The observation that makes this trivial and safe

`Raycaster::build_ray_list(src)` (`raycaster.cpp:536`) folds a source into a
`std::vector<RayHD>` where **each ray carries its own origin** (`sx=src.x,
sy=src.y`) **and precomputed direction** (`dx,dy`), plus its pre-normalised
per-channel energy and `heat_emit`. A `RayHD` is a self-contained POD; the
march kernel is **one thread per ray** (`cuda_raycaster.cu:52`) with **no
cross-ray or cross-source state** — the only shared mutable outputs are:

- `heat` — deposited via **saturating integer atomic add of non-negative
  deltas** (`heat_atomic_sat_add`, `cuda_raycaster.cu:41`). Order-free:
  saturating add under a monotone clamp is associative + commutative, so the
  final `heat` is independent of ray/source order (the proven CUDA-S2
  argument; `heat` is the only sim-affecting output).
- `light_rgb / light_dx / light_dy / smoke_glow` — float atomics + device
  `expf`. **Render-only, determinism-EXEMPT** (`cuda_raycaster.h:13`), and in
  `cast_fire_heat` they are **discarded** (throwaway scratch; `smoke_glow` is
  `None`).

Therefore **concatenating every source's `build_ray_list` output into one ray
array and marching it in a single cast is byte-identical, on `heat`, to the
per-source loop.** No march arithmetic changes. No `(x*7+y*13) % ray_count`
phase change (the phase is still computed per source in Python, into
`src.angle_center`, unchanged). No golden/digest re-baseline. Fits inside all
three S8c escalation triggers.

★ RNG note: `build_ray_list` seeds a per-source `std::mt19937` but only draws
from it when `src.jitter > 0`. `cast_fire_heat` sets `src.jitter = 0.0`
(heat is sim-affecting — no dither), so the RNG is **never drawn**; there is no
cross-source RNG-sequence coupling to preserve. (Guard this with an assert /
comment so a future non-zero-jitter heat source can't silently desync.)

## 2. Scope

**In:** one new pybind entry `cuda_raycaster_cast_batch(raycaster, sources,
light_rgb, light_dx, light_dy, gas, gas_absorption, gas_scatter, light_atten,
heat, smoke_glow, heat_atten)` that loops `build_ray_list` over `sources` in
the **/fp:strict TU** (same place the per-call entry builds its one list),
concatenates into a single `std::vector<RayHD>`, and calls the existing
`raycaster_cast_directional` **once**. Python `cast_fire_heat` builds the
`LightSource` list in its existing row-major loop (phase formula, dequant,
params **unchanged**) and, on the CUDA path, issues **one** batch call after
the loop instead of one call per source.

**Out (unchanged):** the march kernel and `raycaster_cast_directional`
(untouched — the batch entry is a pure host-side wrapper); the CPU path (stays
the per-source `cast_source_directional` loop — no round-trip tax to remove);
the phase formula; the source-build math; `heat` consumers (tail temperature
pass + unit damage still read the mirror `heat` exactly as today — the batch
cast lands `heat` back on the host buffer, the simplest tick-contract-
preserving option per the kickoff). No device-resident `heat` plane, no moving
consumers device-side (kickoff: "bigger — probably out of scope"). No
`coarse_cluster` use (that helper is for a C++-side source build; keeping the
build in Python is what makes byte-identity free).

## 3. C++ surface (`bindings.cpp`, inside the existing `#ifdef BREACH_HAS_CUDA`)

New lambda modelled field-for-field on `cuda_raycaster_cast`
(`bindings.cpp:284`), differing only in taking a **sequence of sources** and
concatenating ray lists:

```cpp
m.def("cuda_raycaster_cast_batch",
  [](const Raycaster& self,
     const std::vector<LightSource>& sources,   // pybind converts a py list
     py::array_t<float> light_rgb, light_dx, light_dy,
     py::array_t<float> gas, gas_absorption, gas_scatter, light_atten,
     py::object heat, py::object smoke_glow, py::object heat_atten) {
      // ... identical array unpacking to cuda_raycaster_cast ...
      std::vector<breach_cuda::RayHD> rays;
      for (const auto& src : sources) {
          std::vector<breach_cuda::RayHD> r = self.build_ray_list(src);
          rays.insert(rays.end(), r.begin(), r.end());   // preserve source order
      }
      if (rays.empty()) return;                          // nothing burning path
      breach_cuda::raycaster_cast_directional(
          rays.data(), (int)rays.size(),
          lrgb, ldx, ldy, heat_ptr, glow_ptr,
          gas_field, gabs, gsca, n_gases, atten, hatten,
          self.smoke_absorb_scale, self.light_cull, self.heat_cull, h, w);
  }, ...);
```

Notes:
- `std::vector<LightSource>` as the arg type lets pybind auto-convert a Python
  `list[LightSource]` (LightSource is already a bound copyable class). If the
  auto-convert is awkward, fall back to `py::iterable` + `.cast<LightSource>()`
  per element — identical result.
- Source **order** is preserved in the concatenation (row-major, matching the
  per-source loop) — not required for `heat` correctness (order-free), but
  keeps the render scratch and any future ordered output aligned.
- `n_rays` is `int`; a burning-tile count × 8 rays stays well within int range
  for any playable map (256²×8 ≈ 5e5 worst case if every tile burns). Document
  the ceiling.
- Non-CUDA build: the entry lives inside the CUDA `#ifdef` exactly like
  `cuda_raycaster_cast`, so CPU-only builds never see it (Python guards on
  `use_cuda_ray`).

## 4. Python surface (`physics_runner.py::cast_fire_heat`)

Split the existing per-tile loop body: **build** the `LightSource` (all the
current field assignments — `x,y,max_range,ray_count,angle_spread,
angle_center` (the phase), `intensity,heat,jitter=0,color` — **verbatim**),
then:

- **CUDA path:** append the source to a `sources` list; after the loop, one
  `bp.cuda_raycaster_cast_batch(self.raycaster, sources, rgb, dx, dy, gas_f,
  absorption, scatter_albedo, dyn_light_atten, gmap.heat, None,
  gmap.heat_atten)`. (Skip the call if `sources` is empty — already guarded by
  the `burning.any()` early-out.)
- **CPU path:** unchanged — cast each source immediately with
  `self.raycaster.cast_source_directional(...)`.

The scratch-buffer zeroing (`_fire_scratch_*`) and the gas dequant
(`_fire_gas_f`) stay exactly as today (allocated/zeroed once per pass, before
the loop). The `heat` buffer is `gmap.heat` in both paths — the batch entry
uploads its current contents, saturating-adds every source, downloads once.

## 5. The gate

1. **Correctness (mechanical oracle, tol 0):**
   - `tests/cuda_s2_check.py` + `tests/cuda_s2b_raycaster_live_check.py` — the
     raycaster/live-cast digest gates (the per-source path stays live and is
     still exercised where those tests drive it).
   - `tests/cuda_s8a_check.py` PART 1 — the full-engine A/B, `heat` ∈ `_FIELDS`
     compared tol 0. This now exercises the **batched** cast in the live tick
     (the runner's CUDA fire path). CPU vs CUDA `heat` must stay bit-identical.
   - A **direct batch-vs-per-source A/B** (new, cheap): on a multi-fire state,
     run the old per-source loop into `heat_A` and the batch entry into
     `heat_B`; assert `heat_A == heat_B` byte-for-byte. Localised proof that
     concatenation changed nothing. (Both are the CUDA path — proves batching
     alone.)
   - Full `pytest tests -q` green.
2. **Payoff bench (the point of the patch):** a many-fires scenario
   (dozens–hundreds of burning tiles). Measure per-source-loop tick time vs
   batched tick time on `--cuda`; assert (a) batched clearly wins, (b) an
   **absolute playable floor** (e.g. tick well under the 3-fps ≈ 333 ms/frame;
   target comfortably interactive). Laptop power-state noise → assert the win +
   an absolute floor, not a ratio (S8a bench discipline).

## 6. Build order

1. Add the batch binding (`bindings.cpp`); build CUDA (`cpp/build_cuda_lenovo.bat`)
   + CPU (`cpp/build_cpu_data.bat`); rebuild main-tree `.pyd`s.
2. Rework `cast_fire_heat` Python (CUDA path → batch call).
3. Gate: §5.1 correctness (incl. the new batch-vs-per-source A/B) →
   §5.2 payoff bench.
4. Auto-merge on green (Erik pre-authorized this session: tol-0 heat A/B +
   bench floor) — canon fold (engine/06 fire, engine/08 ray) + ledger note.

## 7. Accepted-risk register

- The per-source CUDA entry (`cuda_raycaster_cast`) stays bound and live (other
  callers / the S2 gate use it). We add a path, we don't remove one.
- Render scratch float-atomic order differs between batched and per-source
  (all rays interleave in one launch vs. per-source launches) — **exempt**
  (render-only) **and discarded** in `cast_fire_heat` (`smoke_glow=None`,
  rgb/dir thrown away). No consumer sees it. If a future caller needs the
  render channels from the batch entry, that exemption must be re-stated for
  that caller (documented in the binding's docstring).
- `int` ray-count ceiling documented (§3).
- Jitter-must-stay-zero invariant for heat sources — asserted/commented (§1).
