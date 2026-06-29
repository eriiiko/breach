# X-ARCH Beat-B (Ada) findings — 2026-06-29

First-ever cross-machine (Beat-B) run of the CUDA port, on the **Ada Lenovo**.
Honest record; **nothing here is "passed and crossed off"** — the on-machine
proof is green, the **cross-machine** proof is **red**, and that red is the
interesting result. We park here; resume tomorrow/Wednesday.

## Machine + toolchain (this run)
- Host: **erik_lenovo** (LENOVO 21KV001RMX, Intel Core Ultra 7 155H, 31.5 GB, Win 11 Pro 10.0.26200.8655)
- GPU: **NVIDIA RTX 1000 Ada Generation Laptop GPU**, sm_89, 6 GB, driver **596.47** (CUDA 13.2-capable)
- Build env: miniconda **`data`** env, **Python 3.12.11** (cp312) — has torch (future RL loop)
- Compiler: **MSVC 14.44.35207** (VS 2022 Build Tools 17.14)
- CUDA toolkit: **12.9** (V12.9.86) — chosen over the Work Desktop's 12.4 because 12.4's nvcc rejects MSVC 14.44; CMakeLists already passes `-allow-unsupported-compiler`
- Builds produced: `cpp/build/Release/breach_physics.cp312-win_amd64.pyd` (CPU) and `cpp/build_cuda/breach_physics.cp312-win_amd64.pyd` (CUDA, archs 75;86;89)

## What is GREEN (the genuine positive) ✅
**On this Ada machine, every GPU kernel is bit-identical to the CPU, tol 0.**
All 10 in-process checks confirm `GPU == CPU`:
- S0 hello (mul_q16), S2 raycaster heat — **pass outright**.
- S1, S2b(raycaster-live), S3, S4a, S4b, S5, S6, S7 — their **PART 1 (isolated GPU-vs-CPU)** and **PART 2 (full-engine backend-switch, 30 ticks)** are **bit-identical** (e.g. raycaster-live: "ALL 23 synced fields bit-identical over 30 ticks"; S5 reduction "GPU sum == CPU sum to the LSB"; S7 "200 configs bit-identical on 6 fields").
- A search for any `GPU != CPU` divergence returned **nothing**.

So the CUDA port itself is faithful on Ada. The game **runs** on this hardware (CPU and `--cuda`).

## What is RED (the finding) ❌
The **full-engine 30-tick trajectory digest differs across machines**:

| | digest |
|---|---|
| **This Lenovo (Ada)** — CPU **and** GPU, identical to each other | `fe8edddafbb1966d95e5c5a76a59f019a7f4fda8d379099a44e9d13ba9cfa5e3` |
| **Work Desktop (Ampere)** golden — confirmed **current** by Erik 2026-06-29 | `60bd331faccc0b08c11e1ccad3ca75fa6f2aa26232b0b04c1a070b6c65c86ba1` |

**Decisive clue:** the Lenovo's **CPU** digest equals its **GPU** digest (`fe8eddda`).
So the divergence from Ampere is **host/CPU-side — present with no GPU involved at all.**
It is **not** a GPU-arch issue and **not** a GPU-port issue.

## Q1 — why did 2 pass but 8 fail?
Pure bookkeeping, not "2 kernels OK / 8 broken." The pass/fail split is exactly
**"does the test compare against the cross-machine golden?"**
- The **2 passes** (S0, S2 non-live) assert **only on-machine `GPU==CPU`** — no 30-tick golden. `GPU==CPU` holds → pass.
- The **8 fails** each *also* compare PART 2's full-engine trajectory against the hardcoded Ampere **GOLDEN** (`60bd331f`, in every `cuda_s*_check.py`). Their `GPU==CPU` half is green; only the **golden half** is red.

i.e. **every test that checks cross-machine fails the same single way; every test that doesn't, passes.** One divergence, surfaced 8 times.

## Q2 — "the CPU integer digest differs across machines — isn't that odd? floats?"
Not odd, and yes — it's the floats. The digest hashes **integer** (Q16.16) fields,
but those integers are not all produced by integer-only math:

- Integer **arithmetic** (add/mul/shift) IS bit-identical across machines — spike0 proved it cross-machine (Ampere == Ada: `-1514247643326`, `0xAB27B2370160FFF4`), and every PART-1 here is integer-clean.
- BUT the full engine still has **float precompute / "bridge" steps** (the checks name them: *sink float bridge, head bridge, tilt poly, logistic*). They run in float, then **quantize to Q16.16**.
- A different **MSVC/CRT version** can produce a **last-ULP** difference in those floats (e.g. `sinf/expf/powf/sqrtf`, which `/fp:strict` does **not** standardize across compiler versions — it only kills reordering/FMA *within* one compiler). Quantizing ×65536+round **amplifies that ULP into a whole integer count**, and from that tick on the integer fields — and their digest — diverge.

**So the integers faithfully and identically propagate whatever they're seeded with; they were just seeded from floats that differ by a ULP across compilers.** "It's an integer digest" ≠ "it was computed by integers only." Eliminating the last float bridges (the migration's actual endgame) is what should close this.

## Honesty / what is NOT yet established
- This is a **strong, well-supported hypothesis**, **not a localized proof**. We have not yet pinned the exact field/tick/operation that first diverges.
- Leading suspect: **CRT transcendentals in the `/fp:strict` sim TUs**. Not yet ruled out: AVX2 float codegen differences, a stray float in a non-strict TU, or (low-probability) something in the Python harness. The compiler (MSVC version) is the prime variable, since CUDA version is irrelevant here (the CPU-only build already diverges).

## Next steps (tomorrow/Wednesday)
1. **Localize**: add per-field, per-tick digest dumping to `field_ab_harness`; diff Lenovo vs Ampere to find the **first** diverging (field, tick) → names the responsible solver.
2. **Inspect** that solver's float bridge; grep the `/fp:strict` sim TUs for `sinf/cosf/expf/powf/sqrtf/logf`.
3. **Clean experiment to confirm the compiler-float hypothesis**: rebuild the *same source* on the Lenovo with an **older MSVC toolset (14.3x)** matching the Work Desktop. If the digest shifts toward `60bd331f`, the cross-machine cause is confirmed to be compiler-float (not arch). (We can add the 14.3x toolset to the existing VS Build Tools install.)
4. Decide policy: integerize the offending bridge (preferred, the migration's goal) vs pin the toolchain.

## Uncommitted working-tree artifacts (on `main`, NOT pushed)
- `docs/xarch_ada_beatB_findings_2026-06-29.md` (this file)
- `cpp/build_cuda_lenovo.bat` (new — Lenovo build recipe: BuildTools + data env + CUDA 12.9)
- `tests/cuda_harness.py` (edit — `CUDA_PYTHON` now env-overridable via `BREACH_CUDA_PYTHON`; default unchanged so Work Desktop is unaffected)
- `tests/digest_erik_lenovo_cpu_cpu.txt` (the Lenovo CPU digest artifact)
- Build outputs in `cpp/build/` and `cpp/build_cuda/` (gitignored)
