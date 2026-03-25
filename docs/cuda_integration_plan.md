# CUDA Integration Plan — Breach

_Created: 2026-03-25_

> **Goal**: Move performance-critical systems to the GPU via CUDA.
> Every kernel must be optimized — if it runs a million times per second,
> every instruction matters. Research before implementing.

---

## 1. What Is CUDA (For This Project)

CUDA lets us write C/C++ "kernels" that run on all 3584 cores of the RTX 3060
simultaneously. Instead of processing one ray, one tile, or one pixel at a time,
we process thousands in parallel.

**Key constraint**: GPU cores are simple. No branching (expensive), no random
memory access (cache misses kill performance). The ideal kernel does the same
math on contiguous memory for every thread. Design data layouts accordingly.

**Memory hierarchy** (fastest to slowest):
1. **Registers** — per-thread, ~255 per thread. Fastest.
2. **Shared memory** — per-block (32-64 KB), shared between ~256 threads. Fast.
3. **L1/L2 cache** — automatic, helps with repeated access patterns.
4. **Global memory (VRAM)** — 12 GB on RTX 3060. Slow (~400 GB/s bandwidth).
   Every kernel design should minimize global memory reads.

**Occupancy**: The GPU can hide memory latency by switching between thousands of
threads. More threads = better latency hiding. Design kernels to use few
registers and little shared memory so more threads can run simultaneously.

---

## 2. Setup

**Requirements**:
- CUDA Toolkit (includes nvcc compiler, cuBLAS, cuFFT)
- MSVC Build Tools (already have)
- CMake with CUDA language support (add `enable_language(CUDA)` to CMakeLists.txt)

**Integration with existing code**:
- CUDA kernels live in `.cu` files
- Host code (CPU) calls kernels via `<<<blocks, threads>>>` syntax
- Data transfer: `cudaMemcpy` between CPU and GPU memory
- Can mix .cpp and .cu files in the same CMake project

**Critical optimization: minimize CPU↔GPU transfers.** The transfer bus is the
bottleneck, not the computation. Ideal: upload tile map once, run all physics
on GPU, download only the final render buffer.

---

## 3. Target 1: Raycaster

### Why First
- Embarrassingly parallel (each ray independent)
- Already implemented in C++ (algorithm understood)
- Immediately visible result (lighting on screen)
- Current cost estimate: 20 lights × ~60 rays each = 1200 rays per frame
  On GPU: 1200 rays is nothing — could do 100,000+ rays per frame

### Current Algorithm (CPU)
```
for each light source:
    for each ray (angular sweep):
        march through tiles (DDA algorithm)
        accumulate shadow/light contribution
        write to light buffer
```

### CUDA Design

**Kernel**: One thread per ray. Each thread:
1. Reads light position and angle from constant memory
2. Marches through tile map (read-only, in texture memory for cache)
3. Writes light contribution to output buffer (atomicAdd for overlapping lights)

**Data layout**:
- Tile map: 2D texture (CUDA texture memory has spatial cache — perfect for
  DDA traversal where adjacent rays read adjacent tiles)
- Light sources: constant memory (small, read by all threads)
- Output light buffer: global memory, one float per tile (or per pixel)

**Optimization considerations**:
- DDA ray marching is inherently branchy (different rays traverse different
  numbers of tiles). This hurts GPU occupancy. Mitigation: set a max ray length
  so all threads do similar work.
- Tile map as CUDA texture: hardware-accelerated 2D interpolation and caching.
  This is specifically designed for spatial access patterns like ray traversal.
- `atomicAdd` for compositing multiple lights into one buffer. Alternative:
  one buffer per light, sum on CPU (trades memory for avoiding atomics).
- **Warp divergence**: 32 threads in a warp must execute the same instruction.
  If rays in the same warp take different paths (one hits a wall, one doesn't),
  both paths execute sequentially. Group rays by direction to minimize this.

**Research before implementing**:
- [ ] Study existing CUDA 2D raycaster implementations
- [ ] Compare DDA on GPU vs distance-field ray marching (SDF might be better on GPU)
- [ ] Benchmark texture memory vs global memory for tile map reads
- [ ] Profile atomicAdd contention vs per-light buffers

### Expected Speedup
CPU (single-threaded): ~1200 rays × ~50 tile steps = 60k operations, sequential
GPU (3584 cores): 1200 rays in parallel, each doing 50 steps = 50 operations wall time
**Theoretical: ~1000x speedup.** In practice: 50-200x (memory latency, launch overhead).

This means we could go from 20 lights to **2000+ lights** at the same frame budget.
Every spark, every muzzle flash, every flame tip could be its own point light.

---

## 4. Target 2: Diffusion Solver (Atmosphere, Temperature)

### Why Second
- Currently bottlenecked by CFL stability (dt must be tiny for large kappa)
- The 2D stencil operation (Laplacian) is a textbook GPU kernel
- Higher resolution atmosphere = more realistic smoke, decompression, fire spread

### Current Algorithm
```
for each tile (i, j):
    laplacian = field[i+1,j] + field[i-1,j] + field[i,j+1] + field[i,j-1] - 4*field[i,j]
    field_new[i,j] = field[i,j] + dt * kappa * laplacian
```

### CUDA Design

**Kernel**: One thread per tile. Each thread reads 5 values (center + 4 neighbors),
computes the update, writes one value.

**This is the "hello world" of GPU computing** — 2D stencil. Extensively studied.

**Optimization considerations**:
- **Shared memory tiling**: Each thread block loads a tile of the field into shared
  memory (including halo cells for neighbor access), then computes. This reduces
  global memory reads from 5 per thread to ~1 per thread.
- **Double buffering**: Read from buffer A, write to buffer B, swap. Avoids
  read-write hazards without explicit synchronization.
- **Multiple timesteps per kernel launch**: If CFL requires dt=0.01s and we need
  1s of simulation, we can run 100 iterations inside one kernel (ping-pong between
  two shared memory buffers) without returning to CPU. This eliminates launch
  overhead for sub-stepping.
- **Square grid (3×3 fine tiles per unit tile)**: Standard 4-neighbor Laplacian
  stencil — the textbook GPU case. No hex complications.

**Research before implementing**:
- [ ] Standard 2D stencil implementations on GPU (abundant literature)
- [ ] Optimal shared memory tile size for our grid dimensions
- [ ] Multi-step kernel (how many iterations can we fit in shared memory?)
- [ ] Compare explicit (Euler) vs implicit (Jacobi iteration on GPU) for stability

### Expected Speedup
CPU: 24×48 fine grid × 100 sub-steps = 115k operations, sequential
GPU: 24×48 tiles in parallel, 100 sub-steps = 100 operations wall time
Could increase resolution to 240×480 (100x more tiles) at the same cost.

---

## 5. Target 3: Wave Equation (Explosions)

### Why
- Same stencil pattern as diffusion but second-order (needs velocity field too)
- Higher resolution = sharper blast waves, more realistic reflections off walls
- Currently limited by same CFL constraint

### CUDA Design
Same as diffusion but with two fields (pressure + velocity), updated together.
The kernel is slightly larger but the parallelism is identical.

**Research**:
- [ ] FDTD (Finite-Difference Time-Domain) wave solvers on GPU — standard in
  acoustics and electromagnetics, lots of literature
- [ ] Perfectly Matched Layer (PML) absorbing boundaries on GPU
- [ ] Can we use the same kernel for both diffusion and wave equation?
  (Same stencil, different update rule — template or flag parameter)

---

## 6. Target 4: Smoke Advection

### Why
Smoke needs both diffusion (spreading) and advection (carried by wind/pressure
gradients). Advection is trickier on GPU because it involves non-grid-aligned
movement — a particle at (3.7, 5.2) reads from interpolated grid positions.

### CUDA Design
- Semi-Lagrangian advection: for each grid cell, trace backward along the
  velocity field, interpolate the source value. This is naturally parallel
  (one thread per cell) and uses the GPU's texture interpolation hardware.
- Combine with diffusion in one kernel pass (advection-diffusion operator).

**Research**:
- [ ] Semi-Lagrangian advection on GPU (Jos Stam's Stable Fluids is the classic)
- [ ] GPU texture interpolation for advection (bilinear built into hardware)

---

## 7. Data Flow Architecture

The ideal: **upload once, compute everything on GPU, download once per frame.**

```
CPU side (per frame):
    1. Upload changes only (destroyed tiles, new fire sources, moved lights)
    2. Launch kernels:
       a. Diffusion/wave (multiple sub-steps inside one launch)
       b. Smoke advection
       c. Temperature update
       d. Raycaster (all lights)
    3. Download: light buffer + smoke density buffer (for rendering)

GPU memory (persistent between frames):
    - Tile map (updated only when tiles change)
    - Atmosphere field(s)
    - Temperature field
    - Velocity field (for smoke)
    - Light buffer (output)
```

**Key principle**: The GPU should own the simulation state. The CPU only sends
deltas (tile destroyed, light added) and reads the final result for rendering.

---

## 8. Carmack's Fast Inverse Square Root & Related

The famous `0x5f3f759df` trick is obsolete on modern GPUs — they have hardware
`rsqrt()` that's faster and exact. But the thinking applies:

**GPU has specialized hardware for**:
- `rsqrtf()` — reciprocal square root (1 cycle)
- `__sinf()`, `__cosf()` — fast trig (1 cycle, slightly less accurate)
- `__expf()`, `__logf()` — fast exp/log
- `fmaf(a, b, c)` — fused multiply-add (a*b+c in 1 cycle)
- Texture fetch with interpolation (hardware bilinear, essentially free)

**Use these intrinsics everywhere.** The compiler often finds them but explicit
use guarantees it. For the raycaster: DDA step calculation uses no sqrt at all
(it's integer grid stepping). For normal map lighting: the dot product is just
multiply-adds, no sqrt needed. For distance calculations: `rsqrtf` if needed.

**Where sqrt actually appears in our code**:
- Distance calculations (Euclidean needs sqrt, but can use rsqrtf)
- Normal map lighting (normalize light direction — needs rsqrt)
- Wave equation (wave speed = sqrt(tension/density) — precompute, not per-frame)

---

## 9. For Civulator

Not the priority, but worth noting:

**State encoding**: Building the 25-channel tensor from game state could be a
CUDA kernel — one thread per (channel, row, col). Currently 0.85s in numpy.

**Multi-game simulation**: Run 8+ games in parallel on GPU for faster training.
Each game is independent — perfect for GPU. This is what AlphaZero does.

**Pathfinding**: Batch A* for all units simultaneously. Each unit's A* is
independent. On GPU, 50 units pathfinding simultaneously = same time as 1.

---

## 10. Implementation Order

1. **Setup**: Install CUDA toolkit, add CUDA to CMakeLists.txt, compile a
   "hello world" kernel to verify the toolchain works.
2. **Raycaster**: Port existing DDA to CUDA kernel. Benchmark vs CPU.
3. **Diffusion**: 2D stencil kernel with shared memory tiling. Benchmark.
4. **Wave equation**: Extend diffusion kernel. Benchmark.
5. **Smoke**: Semi-Lagrangian advection + diffusion. Benchmark.
6. **Architecture**: Persistent GPU state, upload deltas, download results.

**At every step**: profile, measure, compare. No premature optimization —
but when we optimize, do it thoroughly.

---

## 11. Resources to Study

- [ ] NVIDIA CUDA Programming Guide (official, comprehensive)
- [ ] "GPU Gems" chapters on fluid simulation and ray casting
- [ ] Jos Stam, "Stable Fluids" (1999) — the foundation for real-time fluids
- [ ] CUDA stencil examples in NVIDIA samples (ships with toolkit)
- [ ] GPU-accelerated 2D lighting tutorials (shadertoy.com has many examples)

---

_This document is a research plan, not an implementation spec. Every section
has "research before implementing" items. We build understanding first,
then implement with confidence._
