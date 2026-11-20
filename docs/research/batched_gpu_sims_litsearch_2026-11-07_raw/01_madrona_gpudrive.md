# Q1. Madrona internals (+ GPUDrive)

Raw research note, 2026-11-07, for #29. Read in full: the SIGGRAPH 2023 paper (13 pages, text
extracted locally), the madrona README, and the madrona_escape_room README plus `src/sim.cpp`.
GPUDrive read through its arXiv HTML.

## Sources
- Shacklett, Rosenzweig, Xie, Sarkar, Szot, Wijmans, Koltun, Batra, Fatahalian, "An Extensible,
  Data-Oriented Architecture for High-Performance, Many-World Simulation", ACM TOG 42(4), 2023,
  doi 10.1145/3592427. PDF: https://madrona-engine.github.io/shacklett_siggraph23.pdf
- Project site: https://madrona-engine.github.io/
- Engine: https://github.com/shacklettbp/madrona
- Example env: https://github.com/shacklettbp/madrona_escape_room (src/sim.cpp:
  https://raw.githubusercontent.com/shacklettbp/madrona_escape_room/main/src/sim.cpp)
- GPUDrive, Kazemkhani et al. 2024: https://arxiv.org/abs/2408.01584 (HTML https://arxiv.org/html/2408.01584)

## Entity storage (paper §5.1)
- ONE table per archetype holding that archetype's entities of ALL worlds, a column store
  (component data contiguous), plus an implicit `EnvID` column in every table. Not per-world
  tables, and no per-world fixed capacity in the engine's storage.
- Why one table, not per-world tables: (1) neighbouring threads stay coherent when few entities
  per world match a query (with 2 agents per world, per-world tables would give 2 coherent
  threads per warp); (2) over-allocation is amortised across worlds (per-world tables fragment;
  their ECS-GPU baseline ran out of memory at ~8K worlds); (3) a component column IS a batched
  tensor, so it is exported zero-copy to the learner.
- Growth: tables reserve a large virtual range at start-up and page physical memory in on demand
  (the virtual-texturing feature), so storage stays contiguous without a custom GPU allocator.
  The README says this unified-memory feature is why the GPU backend does not run on Windows.
- Deletion: set `EnvID = -1` (no synchronisation between deleting threads); entry functions skip
  rows with -1. Reclamation: parallel radix sort of the table by EnvID pushes -1 rows to the end,
  then truncate. They sort at the start of every step (only for tables that had adds/removes),
  because sorted-by-world rows also make per-world state accesses coherent.

## Execution (paper §5.2-5.4, §6.6)
- Each system is a per-entity C++ function; one GPU thread per matching table row by default,
  optionally a warp (32) or block (256) per invocation (used for narrow-phase collision and
  LIDAR rays, with shared memory).
- MEGAKERNEL: every system of the step's task graph is compiled into one CUDA kernel, launched
  once per batch step. Reasons stated: (1) many cheap systems => launch overhead would dominate;
  (2) systems whose thread count is only known after the previous system (e.g. NarrowPhase over
  pairs created by FindOverlaps) would need a CPU readback before each launch.
- Persistent-threads design: warps loop fetching work at warp granularity; all threads work on
  the same graph node, and the warp finishing the last invocation advances the node. HideSeek's
  graph is ~200 nodes; the cheap nodes (query counts, sorts, action application) "essentially
  disappear from the profile".
- Cost of one megakernel: one register allocation for all systems. With PGO they compile
  variants and split the graph into 5 sub-kernels (255 down to 64 registers per thread):
  1.4M -> 1.9M steps/s (+35%), compute utilisation 22% -> 33%, L2 bandwidth 46% -> 69%.
- Profile of one HideSeek step at 16K worlds: 7.9 ms on the RTX 4090's 128 SMs. Systems with one
  invocation per world (constraint solve, velocity filter) leave SMs idle (16,384 invocations vs
  30,000+ threads).

## Numbers (RTX 4090 + i9-13900K, 32 threads; Table 1-2, Fig. 3)
| env | batch (worlds) | BATCH-ECS-GPU steps/s | ECS-CPU (32 thr) | ECS-GPU (1 world/thread) | reference CPU |
|---|---|---|---|---|---|
| HideSeek (3D physics + rays, 5 agents) | 32K | 1.9e6 | 1.6e5 | 4.5e4 | 6.9e3 |
| Overcooked | 64K | 4.0e7 | 1.2e6 | 1.1e6 | 9.7e3 |
| Hanabi | 128K | 2.1e7 | 4.1e6 | 4.1e6 | 1.3e4 |
| Cartpole | 1024K | 3.4e9 | 2.8e7 | 1.4e8 | 1.4e5 |
- ECS-GPU (each world and its ECS in one CUDA thread) reaches 1.9% of compute and memory
  speed-of-light; batched reaches 33% / 69%.
- HideSeek keeps >75% of peak down to 8K worlds; the GPU overtakes ECS-CPU at ~500 worlds
  (HideSeek) and ~200 (Overcooked). Hanabi is memory-bandwidth bound (>90% of peak BW, 6%
  compute).
- Per-agent scaling (Fig. 4): per-agent throughput rises up to ~10 agents per world, then falls
  where pairwise observations (quadratic) dominate.

## Resets, RNG, determinism, CPU parity
- Reset (escape room `sim.cpp`): a per-world `WorldReset` singleton exported as a tensor; the
  `resetSystem` (a node in the graph) reads it, or sets it itself when `autoReset` and an agent
  is done; `cleanupWorld` destroys the level's entities, `generateWorld` builds a new level; on
  the GPU a `RecycleEntitiesNode` reclaims deleted ids. So partial resets are ordinary per-world
  work inside the same step, selected by a per-world flag; no host round trip.
- RNG: per world, `RNG(rand::split_i(initRandKey, curWorldEpisode++, worldID))`, a counter-based
  split keyed by (episode, world). The draws do not depend on thread scheduling.
- Determinism: the paper makes NO determinism claim (searched the full text). Floats throughout.
- CPU backend: "Simulators can execute on GPU or CPU with no code changes" (README;
  `TaskGraphExecutor` vs `MWCudaExecutor`). The escape room records actions and replays them in a
  CPU viewer. The paper's ECS-CPU baseline runs one ECS per world per thread (EnvPool-like).
- Export: nanobind + DLPack, PyTorch entry points only (project site).

## GPUDrive (built on Madrona)
- Peak 2.3M agent steps/s ("over a million steps per second" on consumer and datacenter GPUs);
  Nocturne (CPU) max 15,000; Waymax (JAX) OOM above 16 parallel envs.
- Memory: allocates per actual agent count, but a world's footprint is "proportional to the
  maximum number of agents across all worlds" (so a batch is sized by its worst world).
- PPO end-to-end: 200K-500K controlled-agent steps/s; training config 50 worlds x 92 rollout.
- No determinism discussion; no megakernel discussion.
