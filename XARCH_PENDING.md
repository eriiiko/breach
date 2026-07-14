# X-ARCH pending — verify the EOS engine is bit-identical on the Ampere desktop

> **STATUS: PENDING.** For **Claude Code on the Home Desktop** (NVIDIA RTX 3070,
> Ampere sm_86, machine `DESKTOP-0E98HUV`). Written on the Lenovo/Ada 2026-07-12
> and pushed to `main`. Erik: on the home PC, `git pull`, then point Claude Code
> at this file and say "run the X-ARCH sweep."

## What & why (30 seconds)

The whole physics engine was rewritten (the **EOS refactor** — compressible
`P = C·N·T`, Kwatra semi-implicit solver, unified temperature, combustion on real
O2) **and** fully ported to CUDA (patches P6.0–P6.9 + a water determinism fix).
Every CUDA kernel and the CPU solver were proven **bit-identical on the Ada laptop
(sm_89)**. The one proof still owed: that the **same integer bits** come out on a
**different GPU generation** — this desktop's **Ampere RTX 3070 (sm_86)**.
Cross-GPU determinism is the foundation of the RL-training plan, so it must be
verified, not assumed.

**Expectation: this should PASS clean.** The bedrock is already proven — Q16.16
*integer* arithmetic was verified bit-identical Ampere↔Ada at the spike-0 level
(egregore `concept:q16-fixed-point-cross-arch-determinism`: identical `raw_int64`
on both GPUs). The entire EOS engine rides that same integer arithmetic. **If
anything DOES diverge, the historical culprit was CPU-side** (numpy RNG
spawn-stat via BLAS/LAPACK), **never the GPU integer kernels** — that was found
and fixed 2026-07-04 (tag `cuda-breached`). Keep that in mind.

## Before you start — verify THIS machine (its setup is under-documented)

`environment.md` lists the Home Desktop's conda/CUDA as "TBD", so confirm first:

1. `nvidia-smi` — RTX 3070 present + driver version (must support the CUDA
   toolkit you build against).
2. `nvcc --version` — CUDA toolkit installed (need 12.x). **If nvcc is NOT
   present, Step 2 (CUDA) is blocked** — do Step 1 (the primary determinism
   proof, CPU-only) and report the CUDA leg as "blocked: no CUDA toolkit on the
   Home Desktop yet."
3. Python — `C:\Users\steen\anaconda3\python.exe` (base). `conda env list` to see
   options; any env with **numpy + pytest** works. The `.pyd` you build must
   match whatever python you run the digest with (ABI).
4. Locale is **Swedish** — the Drive mount may be `G:\Min enhet\`, not
   `G:\My Drive\`. Shell quirk on this box: bash *built-ins* (`echo`, `whoami`)
   exit 1, but external exes (`git`, `python`, `nvcc`) work fine — use the file
   tools and call exes directly.

## Step 0 — pull post-EOS main

```
cd C:\Users\steen\projects\breach
git fetch origin && git status
git pull --ff-only origin main
git log -1 --oneline      # expect the recorder fix f601455 or later = post-EOS
```

## Step 1 — CPU cross-arch determinism (the PRIMARY proof, no GPU needed)

1. Build the clean CPU module (configure `cpp/build` first if absent — see
   `docs/dev_setup.md`; the `.pyd` lands at `cpp/build/Release/`):
   ```
   cmake --build cpp/build --config Release
   ```
2. Run the per-field digest — **no args, identical script on every machine**:
   ```
   C:\Users\steen\anaconda3\python.exe tests\_xarch_perfield_digest.py
   ```
   It runs 30 ticks of the canonical A/B scenario, writes
   `tests/_xarch_perfield_DESKTOP-0E98HUV.txt`, and — because the committed Ada
   baseline `tests/_xarch_perfield_erik_lenovo.txt` is in the tree — **auto-diffs
   against it** and prints the first diverging `(tick, field)`, or nothing.
   (It writes a *desktop-named* file; it does NOT touch the Ada baseline.)
3. **SUCCESS** = stdout reports **no divergence** AND the printed
   `aggregate_trajectory_digest` equals the committed Ada aggregate:
   ```
   98d3dd7eaf3d574d6e562513cd95f3b5ac077b7c69b1d0b024db931261735473
   ```
   → the post-EOS **CPU** sim is bit-identical Ada↔Ampere. 🎉
4. **If it diverges:** the first `(tick, field)` line names the responsible
   solver (`atmosphere` → EOS pressure solve, `temperature` → TemperatureSolver,
   `water_depth` → WaterSolver, `__unit_*__` → unit state). Copy that exact line
   + the aggregate hash and report to Erik. Prime suspect is CPU-side RNG/BLAS
   (see the `cuda-breached` note above), not the GPU.

## Step 2 — CUDA cross-arch (the GPU kernels on Ampere) — only if nvcc exists

1. Build the CUDA module (**the desktop build script — NOT the `_lenovo` one**):
   ```
   cpp\build_cuda.bat
   ```
   Produces `cpp/build_cuda/breach_physics*.pyd` (arches 75;86;89).
2. Run the full suite — the `cuda_*` gates activate automatically when a CUDA
   build is present:
   ```
   C:\Users\steen\anaconda3\python.exe -m pytest tests -q
   ```
   Expect **~781 passed, a few skipped, ZERO failed**. Each `cuda_*` gate asserts
   `digest(CUDA) == digest(CPU)` in-process, `tol=0.0`, on THIS machine's 3070.
3. **SUCCESS** = every `cuda_*` gate PASSES on the 3070. Combined with Step 1
   (CPU bit-identical Ada↔Ampere), this transitively proves the GPU kernels are
   bit-identical across the two GPU generations:
   `Ampere-GPU == Ampere-CPU == Ada-CPU == Ada-GPU`.
4. **If a `cuda_*` gate FAILS** (while it passes on Ada): that would be the first
   genuine cross-arch GPU divergence. Name the gate + its output and report — do
   not paper over it.

## On full success — record the proof

```
git add tests/_xarch_perfield_DESKTOP-0E98HUV.txt
git commit -m "xarch: Ampere RTX 3070 (sm_86) per-field digest — bit-identical to Ada post-EOS"
git push origin main
```
Then flip this file's STATUS to **RESOLVED** (date + the aggregate-hash match),
and tell Erik: *"cross-GPU bit-identity confirmed Ada↔Ampere after the EOS
update — the RL-training determinism foundation holds across GPU generations."*

---

_History: the pre-EOS per-kernel cross-arch table (S0/S1…, proven Ampere-first,
awaiting Ada) is superseded — the EOS migration re-proved the entire surface on
Ada, so this doc is now the **Ampere-verification** runbook. Bedrock integer
cross-arch proof: egregore `concept:q16-fixed-point-cross-arch-determinism`;
the CPU-side RNG divergence that was fixed: tag `cuda-breached`, 2026-07-04._
