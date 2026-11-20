# Q6. Batched multigrid / many small linear solves on one GPU

Raw research note, 2026-11-07, for #29. Briefer pass (priority list in the brief).

## Sources
- Aggarwal, Kashi, Nayak, Balos, Woodward, Anzt, "Batched Sparse Iterative Solvers for
  Computational Chemistry Simulations on GPUs", SC21 workshop (ScalA):
  https://impact.ornl.gov/en/publications/batched-sparse-iterative-solvers-for-computational-chemistry-simu/
  (abstract only)
- Nayak et al., "Porting Batched Iterative Solvers onto Intel GPUs with SYCL", 2023 (read, text
  extracted): https://arxiv.org/abs/2308.08417
- P. Nayak, IPDPS 2023 slides on Ginkgo batched solvers (via search):
  https://pratikvn.github.io/about/cv/docs/doctorate/slides/ipdps23.pdf
- Kripke / Denovo block-local wavefront loops (note 05) and Madrona's one-invocation-per-world
  idle SMs (note 01) as analogues.

## Findings
- Ginkgo's batched iterative solvers (CG, BiCGStab, GMRES with Jacobi / ILU preconditioners) map
  ONE work-group (CUDA thread block) to ONE linear system and run the WHOLE solver as a single
  kernel, keeping vectors in shared memory where they fit ("packing all functionality into a
  single kernel" to cut main-memory traffic). Target sizes: "of the order of 10 to 2,000 rows"
  per system; batch sizes in the hundreds to thousands. The SC21 abstract says they beat the
  vendor's batched solutions (no number in the abstract). The SYCL port reports 1.5-2.0x from
  one to two PVC stacks; block size is chosen from the row count (rounded up to the sub-group
  size).
- I found no published "batched geometric multigrid, one V-cycle per env" for thousands of small
  2D grids with numbers. AmgX and cuSPARSE are single-system libraries in what I read (cuSPARSE's
  batched tridiagonal solvers exist but I did not verify their API details here: unverified).
- The general pattern from notes 01 and 05: when one item per env is too little work for a
  launch (Madrona's per-world systems left SMs idle; Kripke's hyperplanes; Ginkgo's small
  systems), put the env (or system) in a block and loop inside the block.

## Reading for Breach (reasoning, not literature)
- Fine levels: Breach's grids are 2.4K-17.6K cells per env, above Ginkgo's per-block sweet spot;
  these levels stay as wide grid-stride kernels over (N, h, w), as the sweep core already is.
- Coarse levels: Breach ALREADY does the Ginkgo/Kripke thing for one grid. `cuda_mg_solve.cu`'s
  `mg_fused_tail` runs every level with <= 1,024 cells (`MG_TAIL_MAX_CELLS`) as one sub-V-cycle
  in ONE kernel, ONE block, with __syncthreads at every CPU pass boundary (bit-identical by
  construction), removing "~238 of ~304 solve launches/tick" (file header; the schedule is
  `mg_cycles = 8` V-cycles per tick, `mg_nu1 = mg_nu2 = 2`, `mg_coarsest_sweeps = 32`,
  config.toml). The batched form is the same kernel with one block per env (gridDim = N): the
  tail stays launch-free and batch-invariant, and N blocks fill the device that one block
  cannot.
- One tall grid with separators (the stacked-grid probe) shares ONE hierarchy, so a coarse cell
  can straddle two copies and couple them unless every env's extent is padded to a multiple of
  2^L and the separators align with the coarsest cells. The (N, h, w) layout with a per-env
  hierarchy avoids that, and is what batch invariance (where-we-stand §7 item 4) needs anyway.
- Per-env convergence: the fixed V-cycle count (8 per tick today) keeps envs in lockstep; an
  adaptive count would need masks or a graph WHILE node (note 04).
