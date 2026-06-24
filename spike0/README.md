# Spike-0 — GPU determinism de-risk (integer vs float)

**Status:** throwaway / experimental. NOT part of the main Breach build. Lives on
branch `spike0-gpu-derisk`, do **not** merge into `main`.

## What this proves

The upcoming fixed-point physics migration rests on one load-bearing claim:

> **Integer arithmetic is bit-identical across GPU runs and architectures,
> while floating-point is not.**

### THE headline (read this first)

**The proof of cross-GPU determinism is method 3 (integer): it produces the
IDENTICAL digest on Turing, Ampere, and Ada — bit-for-bit — every run.** That
single fact (the integer columns being identical across all three machine rows
below) is the whole result. Everything else is supporting cast:

- **method 1 (float atomicAdd)** is **nondeterministic on EVERY machine** — it
  varies run-to-run *within one machine*. This is the actual production hazard
  (`mean_wp`'s global sum): unspecified atomic ordering ⇒ different rounded bits.
- **method 2 (float fixed-order tree)** is a **CONTROL, not a cross-arch proof.**
  It is reproducible here and is *expected to MATCH across architectures too*,
  because it contains no fused multiply-add to contract. It demonstrates "not all
  float is the problem — the problem is *unspecified order* + *FMA choices*," and
  nothing more. **Do not read a differing tree column as the cross-arch result;
  the integer column is the result.**
- **method 4 (FMA contraction)** is the genuine cross-arch float hazard, isolated:
  the **same** fixed-order sum-of-products gives **different bits** depending on
  whether `w*x+acc` is *fused* (one rounding) or *separate* (two roundings) — a
  choice the compiler/arch makes, and different arches make differently. Integer
  has no fused-vs-separate choice, so it cannot diverge this way.

Spike-0 demonstrates the claim with two tiny CUDA programs that mirror the two
scary GPU operations in the real solver:

| Spike | Mimics | Shows |
|-------|--------|-------|
| **0a** | the `mean_wp` global sum (a big reduction) | float-atomic jitters run-to-run on every machine; float depends on fused-vs-separate (method 4); **integer is bit-identical across run and arch** |
| **0b** | the Red-Black Gauss-Seidel diffusion step with a **precomputed reciprocal multiply** | the integer GS-reciprocal pattern is deterministic AND matches a CPU integer reference bit-for-bit, including over NEGATIVE values (signed arithmetic shift) |

### Spike-0a — the reduction

Sums a 32M-element deterministic array three ways, 20 repeats each, in one
process, plus a fourth fixed-order demo, reporting whether the result **bits**
are stable or vary:

1. **float, `atomicAdd`** (the hazard) — every thread atomically adds its float
   into one accumulator. atomicAdd-float ordering is nondeterministic, so the
   rounded result **jitters run-to-run even on one machine**. This is the
   `mean_wp` reduction in the real solver. (Printed as raw hex bits so a 1-ULP
   difference is visible.) **Expected: VARIES, on every machine.**
2. **float, fixed tree reduction** (a CONTROL) — a fixed-order shared-memory
   tree plus a fixed-order final pass. Stable per-run, and — because it has no
   FMA to contract — **expected to MATCH across architectures too**. It is *not*
   the cross-arch proof; it is here to show that fixing the *order* removes the
   run-to-run jitter, so the remaining float hazard is specifically FMA (method
   4), not "all float."
3. **integer Q16.16 → int64** (THE proof) — each value is quantized to int32
   Q16.16 and summed via integer `atomicAdd` into an int64 accumulator. Integer
   addition is associative, so the result is **bit-identical regardless of
   order, run, or architecture**. (Printed as exact decimal.) **This is the
   headline: identical on Turing, Ampere, and Ada.**
4. **FMA contraction: fused vs separate** (the genuine float portability hazard,
   isolated) — the **same** fixed-order sum-of-products `acc += w_k * x_k`,
   computed (a) with `fmaf(w,x,acc)` (fused: one rounding) and (b) with
   `__fmul_rn` then `__fadd_rn` (separate: two roundings, the product forced
   through a `volatile` so the compiler can't re-fuse it). **Same math, same
   order — different bits.** Arches/compilers choose fused-vs-separate
   differently, so float results aren't portable; integer has no such choice to
   make.

The array is seeded from a **fixed closed-form formula** (a SplitMix64 hash of
the element index) — no wall-clock or random seeding, byte-reproducible on
every machine. The value distribution is deliberately **wide-dynamic-range and
mixed-sign**, the regime where float reordering genuinely changes the rounded
sum. (A benign all-positive, near-equal-magnitude array can sum
deterministically by luck and make the experiment vacuous — we avoid that on
purpose.)

The Q16.16 quantizer (`to_q16_16`) is **exact by construction** (×2^16 is an
exact exponent bump; the ±0.5 round and the truncation are exact for our value
range, all well below 2^52). The generator caps magnitudes ~0.0002% under the
Q16.16 clip boundary, so no clamping happens. See the in-source comment — do not
"fix" any step into something that rounds.

### Spike-0b — the GS reciprocal

A fixed-point Red-Black Gauss-Seidel diffusion step on a 128×128 grid, in
integer Q16.16, using a **precomputed reciprocal multiply, not a per-cell
divide** — the exact pattern the real solver will use:

```
num   = b + ((alpha * neighbour_sum) >> 16)          // Q16.16
x_new = (num * RECIP) >> RECIP_SHIFT                  // RECIP precomputed once on host
```

All products are widened to int64 and the shift is an arithmetic right shift,
which behaves identically on every CUDA architecture and on the CPU. Red-Black
ordering means each colour reads only the other colour, so there is no
read-write race and the result is independent of thread scheduling. A **CPU
integer reference** runs the identical arithmetic; the program hashes both
final grids (FNV-1a) and **asserts they match bit-for-bit** (non-zero exit code
on mismatch).

Three things make the test non-vacuous (an earlier version was weaker on all
three):

- **Non-degenerate reciprocal.** `alpha = 0.2` (not 0.25), so the divisor
  `1 + 4·alpha = 1.8` is **not a power of two**. The precomputed reciprocal
  therefore genuinely **rounds** (`RECIP·denom ≠ 2^(RS+QS)`, gap = −1780), so the
  widened-multiply + truncation does real work. (At alpha = 0.25 the divisor is
  exactly 2, the reciprocal is exact, and the whole thing collapses to `num>>1`
  — testing nothing.) The program **prints and asserts this non-degeneracy**.
- **Negative values.** The source term `b` is a smooth bilinear ramp:
  **positive** in one corner, sloping through zero to a **negative** sink in the
  opposite corner. About half the field goes negative, so the one genuinely
  implementation-defined op the scheme relies on — an **arithmetic right shift of
  a negative operand** — is actually executed (on GPU and CPU) and must still
  bit-match.
- **No saturation.** The ramp settles to a smooth gradient of ~1400 **distinct**
  values spanning roughly ±0.95 in Q16.16 (well inside the ±32767 range): not a
  clamped plateau, not trivially zero. The program verifies the spread (distinct
  count, min<0<max, biggest single-value plateau < 25%) and asserts it.

## How to run

### Fastest: the committed portable binaries (no toolkit needed)

`bin/` contains pre-built **portable** executables that run on Turing / Ampere /
Ada with **only an NVIDIA driver** — no CUDA toolkit, no Visual Studio, no
`vcvars`. Just `git pull` and run:

```bash
# Git Bash or cmd or PowerShell, from spike0/:
bin/spike0a_reduction.exe
bin/spike0b_gs.exe
```

They are built multi-arch with a PTX fallback and the **static** CUDA runtime:

```
nvcc -O2 \
  -gencode arch=compute_75,code=sm_75 \      # Turing  (native SASS)
  -gencode arch=compute_86,code=sm_86 \      # Ampere  (native SASS)
  -gencode arch=compute_89,code=sm_89 \      # Ada     (native SASS)
  -gencode arch=compute_89,code=compute_89 \ # PTX fallback (JIT for newer/unknown)
  -cudart static -o bin/<spike>.exe <spike>.cu
```

so the only DLL imports are `KERNEL32.dll` and `nvcuda.dll` (the driver). The
embedded code is `sm_75 + sm_86 + sm_89` SASS plus `sm_89` PTX, so Turing/Ampere/
Ada run native machine code and any newer arch JITs from PTX.

**Driver requirement:** a CUDA-12.4 binary needs a reasonably recent driver
(~**r550+**; this Ampere box is on **560.94**). The **OLD Turing laptop** is the
one to check — if its GeForce driver is ancient, either update the driver, or
(if you'd rather) install the CUDA 12.4 toolkit there and recompile via
`run.bat` / `run.sh`. If a binary ever fails to start with a driver/version
error, that is the signal to update the driver or recompile.

### Recompile-from-source fallback

Run the **same** script unchanged on each machine (Ampere / Turing / Ada):

```bash
# Git Bash:
./run.sh

# or cmd.exe / double-click:
run.bat
```

The script finds the MSVC host compiler, compiles both spikes with
`nvcc -O2 -arch=native`, runs them, and prints a greppable digest block at the
end (the `RESULT ...` / `DIGEST ...` lines) that you copy into the table below.

### Notes / gotchas

- **`-arch=native`** (CUDA 12.x) auto-targets the local GPU, so the script needs
  no per-machine edits. If a machine has an **older CUDA toolkit** that rejects
  `-arch=native`, run with an explicit gencode instead:
  `ARCH=sm_75 ./run.sh` — **Turing RTX 20xx = `sm_75`**, **Ampere RTX 30xx =
  `sm_86`**, **Ada RTX 40xx = `sm_89`**. The source itself hardcodes nothing
  arch-specific.
- **MSVC version:** CUDA 12.4 only supports Visual Studio 2017–2022. `run.bat`
  uses `vswhere` with a `[17.0,18.0)` version constraint so a too-new VS install
  (e.g. a preview "VS 18") is **not** picked — that one makes nvcc fail with
  *"unsupported Microsoft Visual Studio version"*. If you only have a newer VS,
  install the VS 2022 Build Tools (Desktop development with C++).
- You may see one harmless line `'C:\Program' is not recognized ...` near the
  top — that leaks out of Microsoft's own `vcvars64.bat`/`vswhere` and does not
  affect the build or results. Ignore it.

## Results table

Fill one row per machine. **The result is the two INTEGER columns being
identical across every row** — that is the whole point, and it is what you are
checking. The float-atomic column "VARIES" within every run by design; the
float-tree column is a control that is *expected to match* across rows (don't
read a difference there as the proof); the FMA column shows fused ≠ separate.

| machine | GPU | arch | **0a integer** (raw int64) — *must be identical* | **0b integer** (gpu_hash, == cpu_hash) — *must be identical* | 0a float-atomic (the hazard) | 0a FMA fused / separate | 0a float-tree (control) |
|---------|-----|------|---|---|---|---|---|
| Work Desktop (DESKTOP-0E98HUV) | RTX 3070 | sm_86 (native) | **`-1514247643326`** | **`0xAB27B2370160FFF4`** (PASS) | VARIES run-to-run, 20/20 distinct | `0xCEF16263` ≠ `0xCEF16261` (differ, 2 ULP) | STABLE `0xCBB04814` |
| _(Turing)_ | RTX 20xx | sm_75 | | | | | |
| Lenovo Laptop (erik_lenovo) | RTX 1000 Ada Generation Laptop | sm_89 (native) | **`-1514247643326`** | **`0xAB27B2370160FFF4`** (PASS) | VARIES run-to-run, 20/20 distinct | `0xCEF16263` ≠ `0xCEF16261` (differ, 2 ULP) | STABLE `0xCBB04814` |

### Expected conclusion

- **0a integer** (`raw int64` = `-1514247643326`) and **0b integer**
  (`gpu_hash` = `0xAB27B2370160FFF4`, and `gpu_hash == cpu_hash` on each box):
  **IDENTICAL on all three machines.** *This is the proof.* If either integer
  digest differs between Turing/Ampere/Ada, the migration premise is wrong —
  stop and investigate. The 0b run also self-checks, on every machine, that the
  reciprocal is non-degenerate, that the field has negative values and a real
  spread, and that GPU integer == CPU integer bit-for-bit.
- **0a float-atomic**: **VARIES within every single run, on every machine.**
  This is the production hazard (`mean_wp`) and the experiment's negative control
  — it shows the harness can actually detect nondeterminism. (Cross-machine
  values will differ too, but the within-run jitter is already the point.)
- **0a FMA fused vs separate** (`0xCEF16263` vs `0xCEF16261` here): the two
  differ — same math, same order, fused-vs-separate rounding ⇒ different bits.
  This is the genuine float portability hazard; a different arch/compiler can
  make the *fused* path produce yet another value. Integer cannot do this.
- **0a float-tree** (`0xCBB04814` here): a CONTROL. It is stable within a machine
  and, having no FMA to contract, **is expected to MATCH on Turing and Ada too.**
  A match here is the *expected* outcome and is NOT a failure of the experiment —
  it demonstrates that the float problem is specifically *unspecified order*
  (method 1) and *FMA choice* (method 4), not "all floating-point." **Do not
  treat this column differing as the cross-arch result; the integer columns are
  the result.**

If the two integer columns hold identical across all three rows (and within each
run method 1 jitters while method 4's fused≠separate), the migration premise is
**confirmed** and the multi-week fixed-point migration is de-risked.

## Files

- `spike0a_reduction.cu` — the reduction experiment (4 methods: float-atomic, float-tree, integer, FMA-contraction).
- `spike0b_gs.cu` — integer Red-Black GS with non-degenerate precomputed reciprocal, signed (negative) values, CPU reference + bit-exact assert + field-spread assert.
- `run.sh` / `run.bat` — compile from source + run both, print the digest block (recompile fallback).
- `bin/spike0a_reduction.exe`, `bin/spike0b_gs.exe` — committed **portable** binaries (multi-arch SASS + PTX fallback + static runtime; driver-only).
- `.gitignore` — ignores build artifacts (`*.exe`, `*.lib`, `*.exp`) **except** `bin/*.exe`.

## Validation on this machine (Ampere RTX 3070, sm_86, driver 560.94) — 2026-06-24

Recompiled from source AND re-run from the committed portable `bin/*.exe`
(clean invocation, no `vcvars`); both produce identical digests.

```
# --- METHOD 1: float atomicAdd (order nondeterministic) ---  [the hazard]
repeat 00  float_atomic  bits=0xCBB0496D  approx=-2.310626600e+07
repeat 01  float_atomic  bits=0xCBB04A30  approx=-2.310665600e+07
repeat 02  float_atomic  bits=0xCBB0499D  approx=-2.310636200e+07
   ... all 20 repeats distinct (20/20 distinct bit patterns) ...
RESULT method1 float_atomic : VARIES (jitter observed -- as expected) across 20 repeats

# --- METHOD 2: float fixed tree reduction (fixed order) ---  [control]
   ... all 20 repeats bits=0xCBB04814 ...
RESULT method2 float_tree   : STABLE (per-arch; may still differ ACROSS arch) across 20 repeats

# --- METHOD 3: integer Q16.16 -> int64 atomicAdd (associative) ---  [THE proof]
   ... all 20 repeats raw=-1514247643326 (1 distinct value) ...
RESULT method3 int_q16_16   : IDENTICAL (bit-exact every repeat -- as expected) across 20 repeats
DIGEST 0a_integer raw_int64 = -1514247643326

# --- METHOD 4: FMA contraction (fused fmaf vs separate mul/add) ---  [real float hazard]
  fused    fmaf(w,x,acc)            bits=0xCEF16263  approx=-2.024878464e+09
  separate __fmul_rn then __fadd_rn bits=0xCEF16261  approx=-2.024878208e+09
RESULT method4 fma_contract : fused != separate  (DIFFER -- same math, fused vs separate -> different bits)
DIGEST 0a_fma fused=0xCEF16263 separate=0xCEF16261

# Spike-0b
# recip check   : r*denom_q16 = 70368744175884 , 2^(RS+QS) = 70368744177664 , gap = -1780   (non-degenerate)
DIGEST 0b_integer cpu_hash = 0xAB27B2370160FFF4
DIGEST 0b_integer gpu_hash = 0xAB27B2370160FFF4
# field spread  : min=-62048 (-0.9468)  max=62045 (0.9467)
# field spread  : distinct=1444  pos=7875  neg=8001  zero=508  biggest_plateau=3.1%
RESULT 0b_field : NEGATIVE values present (signed >> exercised), smooth NON-saturated gradient (1444 distinct, plateau 3.1%)  (PASS)
RESULT 0b : GPU-integer == CPU-integer  BIT-FOR-BIT  (PASS)
```

**Read:** on Ampere, float-atomic genuinely jitters run-to-run (20/20 distinct),
the FMA fused/separate bits differ (genuine float portability hazard), the
fixed-order float tree is stable (control), and **both** the integer reduction
(`-1514247643326`) and the integer GS-reciprocal (`0xAB27B2370160FFF4`,
gpu==cpu) are bit-exact — the latter over a non-degenerate reciprocal and a
field that is half-negative and non-saturated. The premise holds on this
machine; **the two integer digests are the cross-arch witnesses** to fill on the
Turing and Ada rows.

## Validation on this machine (Ada RTX 1000, sm_89, driver 596.47) — 2026-06-24

Lenovo Laptop `erik_lenovo` (LENOVO 21KV001RMX, Intel Core Ultra 7 155H, 31.5 GB,
Windows 11 Pro build 10.0.26200.8655), GPU **NVIDIA RTX 1000 Ada Generation
Laptop GPU** (6141 MiB, compute_cap **8.9**, driver **596.47**, CUDA driver 13.2).
Full hardware in `hardware_log.md`.

Run **from the committed portable `bin/*.exe` only** — no CUDA toolkit, no MSVC,
no `vcvars` on this box (driver-only, native sm_89 SASS). Both spikes exit 0.

```
# Spike-0a
RESULT method1 float_atomic : VARIES (jitter observed -- as expected) across 20 repeats  (20/20 distinct)
RESULT method2 float_tree   : STABLE 0xCBB04814 (per-arch control)
DIGEST 0a_integer raw_int64 = -1514247643326          # == Ampere, bit-identical
DIGEST 0a_fma fused=0xCEF16263 separate=0xCEF16261     # fused != separate (differ)

# Spike-0b
# recip check   : gap = -1780   (non-degenerate)
DIGEST 0b_integer cpu_hash = 0xAB27B2370160FFF4
DIGEST 0b_integer gpu_hash = 0xAB27B2370160FFF4        # == Ampere, gpu==cpu (PASS)
# field spread  : distinct=1444  pos=7875  neg=8001  zero=508  biggest_plateau=3.1%
RESULT 0b : GPU-integer == CPU-integer  BIT-FOR-BIT  (PASS)
```

**Read:** the two integer digests (`-1514247643326` and `0xAB27B2370160FFF4`) are
**bit-identical to the Ampere reference** — the cross-arch premise now holds on a
**second** architecture (Ampere → Ada). The float controls behave as predicted:
atomic jitters 20/20, fused≠separate, fixed tree stable. Only the **Turing**
(RTX 2060, sm_75) row remains to complete the three-generation proof.
