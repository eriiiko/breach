# Spike-0 — hardware log (exact machines the digests were produced on)

This file records the **exact hardware** each row of the Spike-0 results table in
`README.md` was produced on. The goal of Spike-0 is to confirm the integer
(Q16.16 fixed-point) digests are bit-identical across **three GPU generations**:
Turing (sm_75) → Ampere (sm_86) → Ada (sm_89). One entry per physical machine.

| gen | arch | machine (hostname) | GPU | driver / CUDA | how run | integer digests match? |
|-----|------|--------------------|-----|---------------|---------|------------------------|
| Turing | sm_75 | _(RTX 2060 box — TBD)_ | RTX 2060 | TBD | TBD | _pending_ |
| Ampere | sm_86 | Work Desktop (DESKTOP-0E98HUV) | RTX 3070 | 560.94 | recompiled + portable bin | ✅ (reference row) |
| **Ada** | **sm_89** | **Lenovo Laptop (erik_lenovo)** | **RTX 1000 Ada Generation Laptop GPU** | **596.47 / 13.2** | **portable bin (no toolkit)** | **✅ matches Ampere** |

---

## Ada — Lenovo Laptop (`erik_lenovo`) — 2026-06-24

- **Machine:** LENOVO 21KV001RMX
- **CPU:** Intel(R) Core(TM) Ultra 7 155H
- **RAM:** 31.5 GB
- **OS:** Windows 11 Pro (build 10.0.26200.8655)
- **GPU:** NVIDIA RTX 1000 Ada Generation Laptop GPU
  - VRAM: 6141 MiB
  - compute capability: **8.9** (sm_89, Ada)
  - PCI bus id: `00000000:01:00.0`
- **NVIDIA driver:** 596.47
- **CUDA driver API:** 13.2
- **Toolchain:** none installed (no `nvcc`, no MSVC `cl.exe`, no `g++`)
- **How run:** committed portable binaries `bin/spike0a_reduction.exe` /
  `bin/spike0b_gs.exe` — driver-only, no CUDA toolkit needed. Native sm_89 SASS
  was embedded in the multi-arch binary, so this Ada GPU ran native machine code
  (not PTX-JIT).
- **Result:** both integer digests bit-identical to the Ampere reference; both
  spikes exit 0.
  - `0a_integer raw_int64 = -1514247643326`  (identical to Ampere)
  - `0b_integer gpu_hash = 0xAB27B2370160FFF4` and `gpu_hash == cpu_hash` (PASS)
  - 0a float-atomic VARIES 20/20 distinct; 0a FMA fused `0xCEF16263` ≠ separate
    `0xCEF16261`; 0a float-tree control STABLE `0xCBB04814`.

## Ampere — Work Desktop (`DESKTOP-0E98HUV`) — 2026-06-24

- **GPU:** NVIDIA GeForce RTX 3070, compute capability 8.6 (sm_86, Ampere)
- **NVIDIA driver:** 560.94
- **How run:** recompiled from source via `run.bat`/`run.sh` AND re-run from the
  committed portable `bin/*.exe`; both produced identical digests.
- **Result (reference row):**
  - `0a_integer raw_int64 = -1514247643326`
  - `0b_integer gpu_hash = 0xAB27B2370160FFF4` (`gpu_hash == cpu_hash`, PASS)
- _(Other specs — CPU/RAM/OS build — not yet captured on this box; fill in when next at that machine.)_

## Turing — RTX 2060 box — _pending_

- **GPU:** NVIDIA GeForce RTX 2060, compute capability 7.5 (sm_75, Turing)
- **Status:** not yet run. This is the third generation needed to complete the
  cross-arch proof. Run `bin/spike0a_reduction.exe` and `bin/spike0b_gs.exe`
  there (or recompile via `run.sh`/`run.bat` if the driver is too old for the
  portable binary), then fill exact hardware + digests here and in `README.md`.
