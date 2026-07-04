# CUDA cross-architecture attestation — runbook (Ada / Lenovo laptop)

**For:** a Claude Code session on the **Lenovo laptop** (Ada GPU, compute capability 8.9).
**Hand-off:** Erik points this Claude at this file. Work through the phases top-to-bottom.
**Goal:** empirically confirm the breach GPU kernels are **bit-identical on Ada** as on the
Ampere RTX 3070 where they were developed — closing the only open item in the CUDA arc's
determinism story.

---

## Background (read first, ~60 seconds)

The breach CUDA arc GPU-ported all 7 physics solvers (S1 temperature, S2 raycaster, S3
water, S4 smoke+sink, S5 wave, S6 fire, S7 atmosphere), each **bit-identical to the CPU
(tol 0)** and adversarially reviewed. All merged to `main` (chain `08f4cd8`…`6b7397f`).

Every gate so far ran on **one** GPU — the Ampere 3070. Cross-architecture determinism
(that the *same* kernels give the *same* integer results on a *different* GPU arch) is
**argued** (the kernels are pure-integer, or IEEE-double with `--fmad=false` / `-prec-div`
/ `-prec-sqrt` / `-ftz=false` — all architecture-invariant) **but not yet empirically
shown on Ada.** This runbook shows it.

**The CPU integer golden is `ae1164ca163b4bf49a86694ba78ea5319f86cfff46301c6aa59190207e6c1a12`**
(field_digest spec v1, 30-step canonical A/B scenario; lineage `542931c7…` → S2
re-baseline `60bd331f…` → Q2-lift `453829a6…` → **spawn-stat pin `ae1164ca…`**,
2026-07-04: spawn stats moved off `rng.multivariate_normal` (LAPACK — the Ada tick-0
`__unit_hp__` divergence, lenovo_dev_setup.md §8b) onto Q16.16-quantized species
means). On Ampere, the GPU path reproduces
it. **The attestation succeeds iff the GPU path reproduces this same golden on Ada.**

**The bar, concretely:** the per-solver CUDA gates (`tests/test_cuda_s*.py`) each run the
engine with that solver's GPU backend ON for 30 ticks and assert (a) GPU == CPU bit-for-bit
and (b) the CPU path reproduces `ae1164ca…`. **If all the gates go green on the Ada box,
cross-arch determinism is proven** (Ada-GPU == CPU-golden == Ampere-GPU). Running the gates
on Ada *is* the attestation.

Full arc context: the project memory `project_cuda_migration` (its tail) + `docs/dev_setup.md`.

---

## Phase 0 — orient

1. Confirm the machine: this should be the **Lenovo** (Git Bash; **miniconda** python at
   `C:\Users\steen\miniconda3\python.exe` — NOT anaconda). `python --version` should be a
   cp3xx that has `numpy` + `pybind11` (and ideally `cupy`).
2. Get the code: `git fetch origin && git status` then `git pull` so `main` has all of S1–S7
   (HEAD should be at/after `6b7397f` "Merge CUDA-S7"). `git log --oneline -1`.
3. Check the GPU + toolchain:
   - `nvidia-smi` — confirm the Ada GPU is visible + note the **driver version**.
   - `nvcc --version` — is the **CUDA toolkit** installed? (It may NOT be on the Lenovo yet —
     see Phase 1.) Note the version.
   - `where cl` (inside a VS dev shell) — is the MSVC C++ toolchain present?
4. Read `project_cuda_migration` memory tail for the build/interpreter/cudart facts, and skim
   `cpp/build_cuda.bat` (the build script — its paths are the Work Desktop's; you adapt them).

---

## Phase 1 — environment (only what's missing)

The CPU build + game already run on the Lenovo. You only need to stand up the **CUDA build**:
- **CUDA toolkit** — if `nvcc` is absent, install CUDA **12.x** (12.4+ is fine; the driver
  from `nvidia-smi` must support it). This needs admin. The build reads `CUDA_PATH` from the
  environment (the installer sets it).
- **MSVC C++** — VS2022 (any edition) or Build Tools with the "Desktop development with C++"
  workload, for `cl.exe` + `vcvars64.bat`.
- **Python deps** — the build's interpreter needs `pybind11` (for the cmake config) and the
  runtime needs `numpy` (+ `cupy` for the broader suite). Match the `.pyd` ABI to the python
  you build with (e.g. cp311).

---

## Phase 2 — build the CUDA target (adapt the script)

`cpp/build_cuda.bat` is hard-coded for the **Work Desktop** (VS2022 Community, anaconda 3.11,
CUDA 12.4). Its own header says to adjust for the Lenovo. **Copy it (don't clobber the
committed one) or edit these for the Lenovo:**
- `vcvars64.bat` path → the Lenovo's VS edition (Community / BuildTools / Professional).
- `CMAKE` / `NINJA` paths → the Lenovo's VS CMake+Ninja (or system cmake/ninja if installed).
- `CUDA_PATH` → leave it to the environment if the installer set it, else point at the
  installed version.
- `PYEXE` / `PYBIND` → **miniconda** (`C:/Users/steen/miniconda3/...`), not anaconda.
- `CMAKE_CUDA_ARCHITECTURES="75;86;89"` → **leave as-is.** 89 = Ada, so the Lenovo is already
  covered (75 Turing / 86 Ampere / 89 Ada — one fatbin for all three boxes).

Run it. Expect `CONFIGURE_EXIT=0` then `BUILD_EXIT=0`, producing
`cpp/build_cuda/breach_physics.cp3xx-win_amd64.pyd`. The CPU build (`cpp/build/Release`) is
untouched — the game keeps running on CPU regardless.

> If the VS generator complains "No CUDA toolset found", that's expected — the script uses
> **Ninja + direct nvcc** precisely to avoid VS CUDA MSBuild integration. Keep the Ninja path.

---

## Phase 3 — THE ATTESTATION (run the gates on Ada)

1. Sanity: confirm the CUDA build is importable + sees the device. The gates use
   `tests/cuda_harness.py`, which runs each check in an isolated subprocess with the CUDA
   `.pyd` on the path and `os.add_dll_directory(<CUDA>/bin)` for `cudart`. **Check
   `cuda_harness.py` for hard-coded paths** (it may point at the anaconda python and/or
   `cpp/build_cuda`) — adapt the interpreter to **miniconda** if needed.
2. Run the CUDA gates (the real attestation):
   ```
   C:/Users/steen/miniconda3/python.exe -m pytest tests/test_cuda_s0_hello.py \
       tests/test_cuda_s1_temperature.py tests/test_cuda_s2_raycaster.py \
       tests/test_cuda_s3_water.py tests/test_cuda_s4a_smoke.py \
       tests/test_cuda_s4b_smoke_sink.py tests/test_cuda_s5_wave.py \
       tests/test_cuda_s6_fire.py tests/test_cuda_s7_diffuse.py -v
   ```
   (Or just run the whole suite: `… -m pytest tests/ --ignore=tests/test_main_smoke.py
   --ignore=tests/test_renderer_smoke.py`.) Each gate's PART 1 proves Ada-CPU == Ada-GPU on
   rich synthetic inputs; PART 2 runs the engine with that backend ON for 30 ticks and
   reproduces the golden `ae1164ca…`.
   - **ALL GREEN on Ada ⇒ ATTESTED.** Every GPU kernel is bit-identical Ada-GPU == CPU-golden
     == Ampere-GPU. Cross-architecture determinism is empirically confirmed.
   - **ANY GATE RED ⇒ a real cross-arch determinism bug** (some op is NOT arch-invariant —
     likely an FMA contraction, a rounding mode, or an intrinsic that differs Ampere↔Ada).
     This is the exact thing the arc's `--fmad=false`/`/fp:strict` discipline guards against.
     **Do not paper over it.** Localize: which gate, then which field + which tick first
     diverges (the digest harness / `diff_trajectories` pinpoints it), then which kernel/op.
     Capture the inputs and **surface to Erik** — it's a significant finding.
3. (Optional, holistic) A single combined digest with ALL backends on:
   - Note: `tests/xarch_digest.py` is an **S0 seed** — it captures the trajectory + digests it
     but currently only READS `PHYSICS_BACKEND` as a *label*; it does **not** itself flip the
     `set_*_backend(True)` flags. To get a true all-GPU digest, either (a) add a few lines to
     enable all 7 backends when `PHYSICS_BACKEND=cuda` (call `bp.set_temperature_backend(True)`,
     `set_water_backend`, `set_smoke_backend`, `set_wave_backend`, `set_fire_backend`,
     `set_atmos_backend` — confirm the exact names in `bindings.cpp`), run it against the CUDA
     `.pyd`, and confirm the printed digest == `ae1164ca…`; or (b) just rely on the per-solver
     gates above, which already flip the backends and check the golden. Run with `--write` to
     drop `tests/digest_<host>_cuda_gpu.txt` for the record.

---

## Phase 4 — record + close

1. Record the result in `tests/XARCH_PENDING.md` (currently empty) — a line per box:
   `<host>  <gpu model>  <driver>  <cuda version>  <gates: PASS/FAIL>  <digest>  <date>`.
   And/or commit the `digest_<host>_cuda_gpu.txt` from Phase 3.3.
2. `git add` those + commit (subject e.g. `xarch: attest Ada (cc 8.9) — GPU == CPU golden`)
   and push, so the Work Desktop sees the attestation.
3. Update the project memory: mark the **"cross-machine xarch re-attestation"** owed item as
   **done for Ada** in `project_cuda_migration` (and `MEMORY.md`'s pointer). Note the driver +
   CUDA version used.
4. If both **Ampere + Ada** are attested, the cross-arch leg is closed. (Bonus: the **RTX 2060
   laptop** is Turing cc 7.5 — a more distant arch, already in the `75;86;89` fatbin; the same
   runbook attests it if/when available, and Turing is the strongest cross-arch evidence.)

---

## Gotchas (Lenovo-specific)

- **Interpreter = miniconda**, not anaconda — `C:\Users\steen\miniconda3\python.exe`. The
  `.pyd` ABI must match (cp3xx). Several scripts/`cuda_harness.py` may hard-code anaconda —
  grep + adapt.
- **cudart DLL**: the CUDA `.pyd` needs `os.add_dll_directory(<CUDA>/bin)` before import
  (PATH isn't searched for extension-module deps since py3.8). `cuda_harness.py` does this —
  just make sure its `<CUDA>` path matches the Lenovo's install.
- **Don't run bare `pytest`** — it scans the whole repo (prototypes open a window; tools/ has
  hundreds of tests) and hangs. Always `pytest tests/…` with explicit targets or the two
  `--ignore`s.
- The **game is unaffected** throughout — it uses the CPU build (`cpp/build/Release`,
  `BREACH_CUDA=OFF`), so nothing here changes how the demo runs.
- If `cupy` is missing/mismatched on the Lenovo, the per-solver gates may still run (they use
  the C++ `.pyd`, not cupy) — but install/repair cupy if the broader suite needs it. A
  dependency-resolver warning is not a blocker (test empirically before reverting).

---

## TL;DR for the impatient

```
git pull                                   # get S1–S7
# (Phase 1: ensure CUDA toolkit + VS C++ + miniconda deps)
# edit a copy of cpp/build_cuda.bat for the Lenovo (VS path, miniconda, CUDA_PATH)
<lenovo-build-cuda>.bat                     # -> cpp/build_cuda/...pyd, BUILD_EXIT=0
C:/Users/steen/miniconda3/python.exe -m pytest tests/test_cuda_s*.py -v
#   ALL GREEN -> Ada attested (GPU == CPU golden ae1164ca). Record + push.
#   ANY RED   -> real cross-arch bug. Localize the field/tick/op. Surface to Erik.
```
