# Spike-0 — GPU determinism de-risk (integer vs float)

**Status:** throwaway / experimental. NOT part of the main Breach build. Lives on
branch `spike0-gpu-derisk`, do **not** merge into `main`.

## What this proves

The upcoming fixed-point physics migration rests on one load-bearing claim:

> **Integer arithmetic is bit-identical across GPU runs and architectures,
> while floating-point is not.**

Spike-0 demonstrates that claim empirically with two tiny CUDA programs that
mirror the two scary GPU operations in the real solver:

| Spike | Mimics | Shows |
|-------|--------|-------|
| **0a** | the `mean_wp` global sum (a big reduction) | float reductions can differ bit-for-bit run-to-run; integer cannot |
| **0b** | the Red-Black Gauss-Seidel diffusion step with a **precomputed reciprocal multiply** | the integer GS-reciprocal pattern is deterministic AND matches a CPU integer reference bit-for-bit |

### Spike-0a — the reduction

Sums a 32M-element deterministic array three ways, 20 repeats each, in one
process, reporting whether the result **bits** are stable or vary:

1. **float, `atomicAdd`** — every thread atomically adds its float into one
   accumulator. atomicAdd-float ordering is nondeterministic, so the rounded
   result should **jitter run-to-run even on one machine**. (Printed as raw
   hex bits so a 1-ULP difference is visible.)
2. **float, fixed tree reduction** — a fixed-order shared-memory tree plus a
   fixed-order final pass. Stable per-architecture, but the bits **may differ
   across architectures** (FMA contraction / rounding). Included so the
   cross-arch comparison can reveal that.
3. **integer Q16.16 → int64** — each value is quantized to int32 Q16.16 and
   summed via integer `atomicAdd` into an int64 accumulator. Integer addition
   is associative, so the result is **bit-identical regardless of order, run,
   or architecture**. (Printed as exact decimal.)

The array is seeded from a **fixed closed-form formula** (a SplitMix64 hash of
the element index) — no wall-clock or random seeding, byte-reproducible on
every machine. The value distribution is deliberately **wide-dynamic-range and
mixed-sign**, the regime where float reordering genuinely changes the rounded
sum. (A benign all-positive, near-equal-magnitude array can sum
deterministically by luck and make the experiment vacuous — we avoid that on
purpose.)

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

## How to run

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

Fill one row per machine. The **integer columns must be identical across every
row** (that is the whole point). The **float columns may differ across rows**
(and the float-atomic column "varies" *within* every single run by design).

| machine | GPU | arch | 0a float-atomic | 0a float-tree | 0a integer (raw int64) | 0b integer (gpu_hash == cpu_hash) |
|---------|-----|------|-----------------|---------------|------------------------|-----------------------------------|
| Work Desktop (DESKTOP-0E98HUV) | RTX 3070 | sm_86 (native) | VARIES run-to-run, e.g. `0xCBB047B5`…`0xCBB04C19` | STABLE `0xCBB04814` | `-1514247643326` | `0x1BF2045B60AB8E59` (PASS) |
| _(Turing)_ | RTX 20xx | sm_75 | | | | |
| _(Ada)_ | RTX 40xx | sm_89 | | | | |

### Expected conclusion

- **0a integer** (`raw int64`): **identical** on all three machines
  (`-1514247643326`). If it differs, the premise is wrong — investigate.
- **0b integer** (`gpu_hash`): **identical** on all three machines
  (`0x1BF2045B60AB8E59`), and on each machine `gpu_hash == cpu_hash`.
- **0a float-atomic**: **varies within every run** and almost certainly differs
  between machines. This is the negative control — it shows the experiment can
  actually detect nondeterminism.
- **0a float-tree**: stable within a machine; **may differ between machines**.
  If the Turing/Ada bits differ from Ampere's `0xCBB04814`, that is the
  cross-arch float divergence the migration is meant to eliminate.

If integer holds identical and float diverges as above, the migration premise
is **confirmed** and the multi-week fixed-point migration is de-risked.

## Files

- `spike0a_reduction.cu` — the three-way reduction experiment.
- `spike0b_gs.cu` — integer Red-Black GS with precomputed reciprocal + CPU reference + bit-exact assert.
- `run.sh` / `run.bat` — compile + run both, print the digest block.
- `.gitignore` — keeps build artifacts (`*.exe`, `*.lib`, `*.exp`) out of git.

## Validation on this machine (Ampere RTX 3070, sm_86) — 2026-06-24

```
# --- METHOD 1: float atomicAdd (order nondeterministic) ---
repeat 00  float_atomic  bits=0xCBB047B5  approx=-2.310538600e+07
repeat 03  float_atomic  bits=0xCBB04B2A  approx=-2.310715600e+07
repeat 06  float_atomic  bits=0xCBB04C19  approx=-2.310763400e+07
   ... (all 20 repeats differ in the low bits) ...
RESULT method1 float_atomic : VARIES (jitter observed -- as expected) across 20 repeats

# --- METHOD 2: float fixed tree reduction (fixed order) ---
   ... all 20 repeats bits=0xCBB04814 ...
RESULT method2 float_tree   : STABLE (per-arch; may still differ ACROSS arch) across 20 repeats

# --- METHOD 3: integer Q16.16 -> int64 atomicAdd (associative) ---
   ... all 20 repeats raw=-1514247643326 ...
RESULT method3 int_q16_16   : IDENTICAL (bit-exact every repeat -- as expected) across 20 repeats
DIGEST 0a_integer raw_int64 = -1514247643326

# Spike-0b
DIGEST 0b_integer cpu_hash = 0x1BF2045B60AB8E59
DIGEST 0b_integer gpu_hash = 0x1BF2045B60AB8E59
RESULT 0b : GPU-integer == CPU-integer  BIT-FOR-BIT  (PASS)
```

**Read:** on Ampere, float-atomic genuinely jitters run-to-run (hundreds of ULP
spread), the fixed-order float tree is stable, and **both** the integer
reduction and the integer GS-reciprocal are bit-exact. The premise holds on
this machine; the cross-arch rows remain to be filled on Turing and Ada.
