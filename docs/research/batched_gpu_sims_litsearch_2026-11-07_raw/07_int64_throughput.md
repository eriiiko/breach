# Q7. 64-bit and 128-bit integer arithmetic on NVIDIA GPUs (Ampere consumer vs A100)

Raw research note, 2026-11-07, for #29. Briefer pass.

## Sources
- CUDA C++ Programming Guide 12.4, "Arithmetic Instructions" (table "Throughput of Native
  Arithmetic Instructions", downloaded and text-extracted):
  https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#arithmetic-instructions
- NVIDIA forum threads on 64-bit multiply cost (secondary, not authoritative):
  https://forums.developer.nvidia.com/t/long-integer-multiplication-mul-wide-u64-and-mul-wide-u128/51520 ,
  https://forums.developer.nvidia.com/t/how-much-speed-of-64bit-integer-algebra-in-the-latest-gpus/32982 ,
  https://forums.developer.nvidia.com/t/squeezing-the-last-17-5-out-of-a-compute-bound-256-bit-modular-arithmetic-kernel-sm-89-82-5-sm-throughput/367788
- `__umul64hi` / `__mul64hi` libdevice: https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umul64hi.html
- Specs (secondary sources): Tom's Hardware Ampere deep dive
  https://www.tomshardware.com/features/nvidia-ampere-architecture-deep-dive ; Wikipedia Ampere
  https://en.wikipedia.org/wiki/Ampere_(microarchitecture) ; RTX 3070 review
  https://geeks3d.com/?p=10095

## Findings
- Programming guide table (results per clock per SM): 32-bit integer multiply, multiply-add and
  extended-precision multiply-add = 64 on cc 7.x, 8.0, 8.6, 8.9, 9.0; 32-bit integer add and
  extended-precision add = 64 on the same; shifts and compare/min/max 64. FP32 add/mul/fma: 64
  on 8.0, 128 on 8.6. FP64: 32 on 8.0 (A100), 2 on 8.6 (RTX 30xx). There is NO 64-bit integer
  row: "Other instructions and functions are implemented on top of the native instructions."
  So int64 add is two 32-bit adds with carry, and int64 / int128 multiplies are sequences of
  32-bit IMADs (IMAD.WIDE.U32 produces a 64-bit product of two 32-bit operands in one
  instruction).
- Consequence: per SM per clock, the integer pipe is the SAME on the RTX 3070 (sm_86) and the
  A100 (sm_80). The A100's FP64 advantage (16x per SM) is irrelevant to integer fixed point.
- Cost model from the forum (secondary, architecture unstated): a 64 x 64 -> 64 multiply ~ 4
  32-bit multiplies; the high half (`__umul64hi`) roughly twice that; 64-bit division ~ 40
  multiplies plus as much other arithmetic. One poster measured `__umul64hi` at ~10x a 32-bit
  multiply on Fermi (old). A 2024 sm_89 post: ptxas turns ordinary 64-bit C++ multiplies into
  the same IMAD.WIDE sequence as hand-written 32-bit code, so there is little to gain from PTX.
- Device totals (secondary sources): RTX 3070 46 SMs, ~1.73 GHz boost, 448 GB/s, 4 MB L2,
  8 GB; A100 108 SMs, 1.41 GHz boost, 40 MB L2, 1,555 GB/s (40 GB) / ~2 TB/s (80 GB). Integer
  throughput A100 / 3070 ~ (108 x 1.41) / (46 x 1.73) ~ 1.9x; bandwidth 3.5-4.5x; L2 10x.

## Reading for Breach
- Breach's MG cell-pass does 1 + <= 4 `mul128_shr_signed` (plus one in the smoother), each a
  `__mul64hi` + low-half combine (cuda_mg_solve.cu header). At ~8-10 IMAD-class instructions per
  128-bit product that is tens of integer instructions per cell-pass, against 64 IMAD/clk/SM.
  Whether it is compute- or bandwidth-bound per kernel is a profiler question (Nsight Compute),
  not answerable from the literature.
- Tricks that apply: keep products 32 x 32 -> 64 (one IMAD.WIDE) where the operand ranges allow
  (a Q16 coefficient times a Q16 value whose magnitude is known to fit 32 bits); hoist 128-bit
  staging out of inner loops where a bound proves int64 suffices; never divide in the device path
  (Breach already: "No division, no reciprocal, no float anywhere on the device"). Each such
  narrowing is an arithmetic change and must land in the CPU reference first (the twin rule).
- For planning: expect an A100 to be ~2x a 3070 on integer-bound kernels and up to ~4x on
  bandwidth-bound ones. Rough size (estimate, not measured): a small ship's synced planes are
  ~2,400 cells x ~100 B ~ 0.25 MB, so the A100's 40 MB L2 holds on the order of 100 envs' worth
  of one tick's hot planes, the 3070's 4 MB about 15.
