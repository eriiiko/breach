# Q5. Wavefront (KBA / hyperplane) sweeps on GPUs for discrete-ordinates transport

Raw research note, 2026-11-07, for #29.

## Sources
- W. Joubert, "Porting the Denovo Radiation Transport Code to Titan: Lessons Learned", OLCF Titan
  Workshop 2012 (slides, text extracted):
  https://www.olcf.ornl.gov/wp-content/uploads/2012/01/TitanWorkshop2012_Day3_Joubert.pdf
- Kripke (LLNL) sweep execution policy, `src/Kripke/Arch/SweepSubdomains.h` (read):
  https://raw.githubusercontent.com/LLNL/Kripke/develop/src/Kripke/Arch/SweepSubdomains.h ;
  repo https://github.com/LLNL/Kripke
- T. Deakin et al., "Reviewing the Computational Performance of Deterministic Sn Transport Sweeps
  on Many-core Architectures", 2019 slides (text extracted):
  https://uob-hpc.github.io/transport2019/slides/many_core_sweeps.pdf
- Deakin, McIntosh-Smith, Gaudin, "Many-Core Acceleration of a Discrete Ordinates Transport
  Mini-App at Extreme Scale", ISC 2016:
  https://research-information.bris.ac.uk/en/publications/many-core-acceleration-of-a-discrete-ordinates-transport-mini-app/
  (full text https://core.ac.uk/download/83928769.pdf, not read)
- Deakin, McIntosh-Smith, Martineau, Gaudin, "An improved parallelism scheme for deterministic
  discrete ordinates transport", IJHPCA 2016 (abstract via search; full text not read)
- Minisweep (ORNL, Joubert): https://github.com/wdj/minisweep (README only; no scheme details)
- Gong et al., "Accelerating the Sweep3D for a Graphic Processor Unit", JIPS 7(1) 2011, and
  "GPU accelerated simulations of 3D deterministic particle transport using discrete ordinates
  method", JCP 230(15) 2011 (abstracts only, via search):
  https://koreascience.kr/article/JAKO201113753748127.view

## Findings
- Denovo on Fermi (Joubert 2012): the dimensions are split by how independent they are.
  Octants and energy groups are "fully decoupled" and go to independent thread blocks (8x from
  octants, 256 groups); space is the KBA recursion done as block wavefronts with a "sync between
  block wavefronts"; moments/angles go to threads in a warp (small dense mat-vecs, matrices in
  L1/shared). NVIDIA's advice then: 4K-8K threads to cover latency. A first scheme ran at ~1% of
  peak until data reuse moved into shared memory. Result: 32 x 32 x 128 cells, 16 groups,
  256 angles: kernel 3.2 s on a C2050 vs 171 s on one Istanbul core (54x; 40x with the PCIe face
  transfers).
- Kripke's CUDA sweep (source): ONE kernel (`CudaKernelAsync`) per subdomain sweep; direction ->
  `cuda_block_x`, group -> `cuda_block_y`; the spatial (j, i) plane -> threads; the k axis is a
  sequential `Hyperplane` loop INSIDE the block with `CudaSyncThreads` between hyperplanes. So
  independent directions and groups are blocks, and the wavefront order is a block-local loop
  with __syncthreads, not one launch per wavefront.
- Deakin (SNAP on K20X, P100, V100): GPU concurrency comes from angles in an octant x all energy
  groups (Jacobi in groups, after Baker) x cells on a wavefront; the scheme is device-memory-
  bandwidth bound; GPU sweep time is modelled better by the NUMBER OF LOCAL WAVEFRONTS than by
  grind time x work ("#angles/groups somewhat unrelated"). UnSNAP's CUDA port launches a kernel
  per t-level of the combined sweep schedule, one block per (element, group, angle), and merges
  angle sets because one angle's schedule has too little work.
- Gong et al. 2011: the wavefront dependency limits concurrent GPU threads; reported speedups
  2.25-4.38x (CPU-GPU hybrid vs CPU) and 2.56-8.14x for an M2050 vs one CPU chip (abstract
  numbers, baselines differ; not verified in full text).

## Reading for Breach
- Breach's batched sweep (`radiation_sweep_launch_resident`) already puts envs on the grid's z
  axis, so K stays max(h, w) and launches stay 63 (40 x 60) / 103 (70 x 100) at any N
  (benchmark). Envs are a "fully decoupled" axis exactly like Denovo's octants and groups.
- The Kripke pattern removes the per-wavefront launch: one block per (env, ordinate), the
  wavefront loop inside the block with __syncthreads. Under the shear transport a wavefront is a
  full column or row (40-100 cells here), so a block of 64-128 threads holds one wavefront per
  step, and the previous wavefront's outflow can live in shared memory. 16 ordinates x N envs
  blocks: 1,024 blocks at N = 64. The books' unsigned atomicAdds stay order-free.
- What Deakin's model predicts for the current design: cost ~ number of wavefronts until the
  device fills, which is what the benchmark shows (0.395 / 0.401 / 0.517 ms at N = 1 / 2 / 4 at
  40 x 60, then linear in N from N = 8).
