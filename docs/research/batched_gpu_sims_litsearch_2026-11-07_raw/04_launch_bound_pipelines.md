# Q4. Launch-bound pipelines of many small kernels: CUDA graphs, persistent kernels, fusion

Raw research note, 2026-11-07, for #29.

## Sources
- Alan Gray, "Getting Started with CUDA Graphs", NVIDIA blog 2019:
  https://developer.nvidia.com/blog/cuda-graphs/
- "Constant Time Launch for Straight-Line CUDA Graphs and Other Performance Enhancements",
  NVIDIA blog 2024:
  https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements
- "Dynamic Control Flow in CUDA Graphs with Conditional Nodes", NVIDIA blog:
  https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/
- NVIDIA DL CUDA-graph guide, quantitative benefits:
  https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/quantitative-benefits.html
- "A Guide to CUDA Graphs in GROMACS 2023", NVIDIA blog:
  https://developer.nvidia.com/blog/a-guide-to-cuda-graphs-in-gromacs-2023/
- Zhang, Wahib, Zhang, Matsuoka, "A Study of Single and Multi-device Synchronization Methods in
  Nvidia GPUs", IPDPS 2020: https://arxiv.org/abs/2004.05371 (read via ar5iv)
- Zhang et al., "PERKS: a Locality-Optimized Execution Model for Iterative Memory-bound GPU
  Applications", ICS 2023: https://arxiv.org/abs/2204.02064
- Madrona megakernel + PGO: note 01.

## Measured numbers
| source | setting | number |
|---|---|---|
| Gray 2019 | V100, CUDA 10.1, 20 kernels of 2.9 us each per step, 1,000 steps | per kernel: 9.6 us with a sync after each launch; 3.8 us with one sync per step; 3.4 us as a graph. Instantiation ~400 us once; first graph launch ~33% slower |
| NVIDIA 2024 | RTX 3060 (Ampere), CUDA 11.8 vs 12.6, straight-line graphs | repeat-launch CPU cost 2 us + 200 ns/node (11.8) -> ~2.5 us + ~1 ns/node (12.6, >= 10 nodes); 100-node graph: instantiation 127 us, first launch 15 us, repeat-launch DEVICE runtime 53 us; 1,025 nodes: 567 us device. So the device floor is ~0.55 us per (tiny) kernel node even in a graph |
| NVIDIA DL guide | rule of thumb, "observed in practice" | many small kernels 1.5-3x; medium 1.3-2x; large 1.1-1.4x; already >90% utilised 1.05-1.15x; assume ~1-5 us per launch of driver + hardware overhead; Python frameworks have higher per-launch overhead |
| GROMACS 2023 | water boxes, 1 and 4 GPUs | graph benefit largest at small systems (single GPU: benefit up to 24K atoms, growing to ~100K and plateauing); graphs must be re-captured every 100-400 steps (neighbour list) |
| Zhang 2020 | V100/P100 null kernels | launch overhead ~1.08 us (traditional), ~1.06 us (cooperative), ~1.26 us (multi-device); grid.sync vs implicit launch barrier differ by at most ~2.5 us; on a reduction the launch barrier was slightly FASTER (865 vs 856 GB/s on V100) |
| PERKS 2023 | V100 + A100, persistent kernel, time loop inside the kernel, grid sync between steps, part of the domain cached in registers / shared memory | 2D stencils 2.12x, 3D 1.24x (geomean); CG 4.86x on small SpMV inputs (L2-resident), 1.43x on large. The gain is from on-chip reuse across steps, not from the barrier |
| Madrona 2023 | ~200-node task graph as one megakernel | cheap nodes vanish from the profile; but ONE register allocation for all systems cost 35%, recovered by splitting into 5 sub-kernels by register need |

## Mechanisms worth knowing
- Graph conditional nodes: IF and WHILE since CUDA 12.4, SWITCH and IF/ELSE since 12.8; the
  condition is set from device code (`cudaGraphSetConditional`) by an upstream kernel, so a loop
  count decided on the device (e.g. a substep count) needs no host round trip. Bodies may hold
  kernels, memcpy/memset, child graphs and nested conditionals, but NOT cooperative kernels.
- GROMACS and the 2019 blog both show graphs pay off only when the graph is re-launched many
  times unchanged; capture and instantiation cost hundreds of us.
- A persistent kernel with grid sync must fit all blocks co-resident (cooperative launch); it
  wins when it keeps data on chip across steps (PERKS), not merely by replacing launches
  (Zhang 2020).

## Arithmetic for Breach (from residency_benchmark_2026-11-07.md, not literature)
- The batched sweep issues 63 launches at 40 x 60 and costs 0.395 ms at N = 1: ~6 us per launch
  as issued today. A graph's device floor (~0.55 us per node, RTX 3060 numbers) would put 63
  nodes near 35 us. At N = 128 the same sweep costs 15.6 ms, so 63 launches are ~2-3% of it.
- A tick of ~150 small kernels: ~80 us of device floor per batch tick as a graph, against a
  batch tick of tens of ms at N >= 64. Launch overhead matters at small N (and for the host CPU's
  time), not at the batch sizes training will use. The host floor (~400 ns per cell) and the
  bracket stages are the larger costs (benchmark "Reading").
- `n_sub` per env (where-we-stand §7 item 3): a WHILE conditional node driven by a device-side
  max over envs, with masked work for envs that finished, keeps it graph-able.
