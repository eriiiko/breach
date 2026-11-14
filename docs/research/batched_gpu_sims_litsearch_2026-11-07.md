# Batched GPU simulators, kernel throughput and GPU pathfinding: a targeted literature pass (#29)

2026-11-07. One agent, one pass, for the residency + batching arc (issue #29). Raw notes with
every number and quote: `batched_gpu_sims_litsearch_2026-11-07_raw/01..11`.

**Scope.** July's infrastructure digest (`rl_litsearch_2026-07-20_raw/05_gpu_parallel_infra_cadence.md`)
already covers Isaac Gym / Lab, Brax, Madrona / GPUDrive, WarpDrive, JaxMARL / SMAX, EnvPool,
PufferLib, SEED, Anakin, PQL and frame-skip at survey level; that is not repeated. This pass goes
one level down on mechanisms and numbers, read against `docs/residency_where_we_stand_2026-11-07.md`
§2-3 and `docs/residency_benchmark_2026-11-07.md`: at N = 1 the tick is launch-, transfer- and
host-bound; batched, the device EOS chain reaches ~70 ns per cell per tick and the batched sweep
~3 ns per cell-update with 63 launches per tick; a host floor of ~400 ns per cell remains; one CPU
core runs the small ship at ~196 env-ticks/s.

## 1. Madrona internals

Madrona keeps ONE column-store table per entity archetype holding that archetype's rows of ALL
worlds, plus an implicit `EnvID` column; so a component column is already the batched tensor the
learner reads, zero-copy [Madrona §5.1]. Tables reserve virtual memory and page in physical memory
on demand (no custom allocator; the reason the GPU backend needs Linux [madrona README]).
Deletion writes `EnvID = -1`; a radix sort by EnvID at the start of every step compacts the table
and keeps each world's rows adjacent. Systems run one thread per row (optionally a warp or block
per invocation). The whole ~200-node task graph runs as ONE persistent-threads megakernel per
step, chosen because many systems are cheap and because some systems' thread counts are known only
after the previous one ran (which would need a CPU readback per launch) [§5.3]. The price: one
register allocation for everything; splitting into 5 sub-kernels by register need gave +35%
(1.4M -> 1.9M steps/s) [§5.4, Table 2]. Resets are a per-world `WorldReset` flag tensor read by a
reset system inside the step (auto-reset on done), and each world's RNG is a counter-based split of
(seed, episode, world id) [escape-room `sim.cpp`]. The paper makes no determinism claim. CPU and
GPU backends share the code [README].

| (RTX 4090, i9-13900K 32 thr) | worlds | batched GPU | 32-thread CPU | one world per GPU thread |
|---|---|---|---|---|
| HideSeek (3D physics, rays, 5 agents) | 32K | 1.9M steps/s | 160K | 45K (1.9% of the GPU) |
| Overcooked (2D grid) | 64K | 40M | 1.2M | 1.1M |

HideSeek keeps >75% of peak down to 8K worlds and passes the CPU at ~500 worlds [Fig. 3]. GPUDrive
on Madrona: 2.3M agent steps/s peak; a world's memory is sized by the largest world in the batch
[GPUDrive].

**What applies to Breach.** Lay the unit store out as Madrona does: SoA columns across all envs
with an env-id column, or its fixed-capacity special case `(N, max_units)` plus a `valid` mask, so
each column is the tensor torch reads in place. Reset through a per-env flag tensor read inside
the step, and key each env's RNG on (seed, episode, env). Madrona's megakernel solves two problems
Breach mostly does not have (dynamic per-step work counts, ~200 nodes of entity logic); its CPU
backend is not a determinism claim, so Breach's batch-invariance gate is its own to build.

## 2. CuLE: one emulator per thread

CuLE runs one Atari console per CUDA thread, split into a game-code kernel and a render kernel
because their register needs and divergence differ (and the render kernel is skipped on unobserved
frames) [CuLE §3]. The mapping is "not the most computationally efficient", kept for debuggability
and a shared CPU build. Measured: the CPU wins below ~128 envs; at 4,096 envs CuLE gives 3.1-3.6x a
12-core CPU; warp divergence costs up to 30% once envs decorrelate (complete divergence would leave
1/32); throughput spread across games is 3.78x on the GPU vs 1.93x on the CPU; FPS per env is
lower than a CPU core's, which forced a different batching strategy for learning [§4]. Resets copy
one of 30 cached start states instead of running up to 94 frames. WarpDrive instead puts one env
per thread block and one agent per thread (A100: 9.8M env steps/s at 2,000 envs x 5 agents)
[WarpDrive]; Madrona's one-world-per-thread baseline ran at 1.9% of the GPU, 42x slower than its
batched layout (§1).

**What applies to Breach.** Do not run gameplay as one thread per env: Breach's envs hold a few
dozen units in different fights, the case where Madrona measured 1.9% utilisation. Map each
gameplay slot to one thread per unit row across all envs, or one block per env when a slot needs
in-env cooperation (shared-memory LOS); keep stages with different register and divergence
profiles in separate kernels (CuLE's split). Reset from a bank of pre-built level states copied in
by a masked kernel (CuLE's cache).

## 3. Isaac Gym / Isaac Lab tensor API

Isaac Gym: global tensors (all actors of all envs) refreshed explicitly once per step, never
copied to the host on the GPU pipeline; partial resets via `set_*_tensor_indexed` with an int32
index list, one setter call per step [Isaac Gym docs]. Isaac Lab's `DirectRLEnv.step()` (source):
pre-physics action -> `decimation` x (apply action, sim step) -> episode counter -> `terminated`,
`time_outs` -> rewards -> same-step reset of done envs (`_reset_idx`, which also runs the "reset"
randomisation events) -> "interval" events -> observations -> return
`(obs, rew, terminated, truncated, extras)`; the terminal observation can be kept before reset.
The default reset path calls `reset_buf.nonzero()`, which the source documents as a host sync,
and invites tasks with "backend-native mask reset support" to avoid it [direct_rl_env.py].
Randomisation is per-env device data sampled at start-up, reset or intervals; the reproducibility
page advises fixing physics-material randomisation at set-up because mid-run changes reorder float
work [Isaac Lab].

**What applies to Breach.** Design the step contract on Isaac Lab's: `step(actions)` runs
`decimation` sim ticks (the action repeat), returns separate terminated / truncated bool tensors,
resets done envs in the same step and returns post-reset observations (terminal observation
optional). Reset and randomise by MASK, never by an env-id list: that is the one host sync Isaac
Lab still has. Per-env parameters (seed, template id, loadouts) are length-N device arrays written
at reset, as the RL-batch habits already say.

## 4. Launch-bound pipelines

| source | number |
|---|---|
| CUDA graphs, V100, 20 x 2.9 us kernels [Gray 2019] | 9.6 us/kernel with a sync after each; 3.8 us with one sync per step; 3.4 us as a graph; instantiation ~400 us |
| CUDA 12.6 vs 11.8, RTX 3060 [NVIDIA 2024] | repeat-launch CPU cost 2 us + 200 ns/node -> ~2.5 us + ~1 ns/node; 100-node straight-line graph: 53 us DEVICE time (~0.55 us/node floor), instantiation 127 us |
| NVIDIA rule of thumb [DL graph guide] | many small kernels 1.5-3x; large kernels 1.1-1.4x; ~1-5 us per launch |
| GROMACS 2023 | largest graph gains at small systems; re-capture every 100-400 steps |
| grid sync vs launch, V100/P100 [Zhang 2020] | launch ~1.1 us; grid sync no faster than a launch barrier (reduction 856 vs 865 GB/s) |
| PERKS (persistent kernel + grid sync + on-chip caching), V100/A100 | 2D stencils 2.12x, 3D 1.24x, small CG 4.86x; the gain is reuse across steps, not the barrier |

Graph conditional nodes (IF / WHILE since CUDA 12.4, SWITCH since 12.8) let a device-computed
value drive a loop count without the host; cooperative kernels are not allowed in their bodies.

**What applies to Breach.** At training batch sizes launches are a few percent: ~150 kernels as a
graph cost ~80 us of device floor per batch tick against tens of ms; the host floor (~400 ns per
cell) and the brackets are the target. Capture the tick as a graph anyway (#24's spike is already
bit-identical): it removes the per-launch Python and driver cost, and a WHILE node over a device
max of `n_sub`, with finished envs masked, keeps the variable substep count off the host. Do not
build a megakernel or a cooperative persistent kernel for the physics; fuse only where data stays
on chip across passes (as the MG tail already does).

## 5. Wavefront sweeps

Production discrete-ordinates codes fill a GPU with the independent axes and keep the wavefront
order local. Denovo on Fermi: octants and energy groups ("fully decoupled") go to thread blocks,
space is a block-wavefront KBA with syncs, angles and moments to threads in a warp; 54x one CPU
core on the kernel [Joubert 2012]. Kripke's CUDA sweep is ONE kernel per subdomain: direction ->
block x, group -> block y, the (j, i) plane -> threads, and the hyperplane loop runs INSIDE the
block with `CudaSyncThreads` between hyperplanes [Kripke source]. Deakin et al. (SNAP on
K20X / P100 / V100): concurrency from angles x groups x wavefront cells; bandwidth-bound; GPU sweep
time is modelled better by the number of local wavefronts than by work; UnSNAP launches a kernel
per level and merges angle sets for work [Deakin 2019].

**What applies to Breach.** Replace one launch per wavefront with Kripke's shape: one block per
(env, ordinate), the wavefront loop inside the block with `__syncthreads`, the previous
wavefront's outflow in shared memory; 63 launches per tick become one, and each step is one 40-100
cell column or row under the shear transport. Envs are a fully decoupled axis like Denovo's octants
and groups, which is how production sweeps fill a GPU. Only the schedule changes: the per-cell
arithmetic and the books' unsigned atomics stay, so gate 0 (bit for bit against the reference)
still holds.

## 6. Batched multigrid / many small solves

Ginkgo's batched iterative solvers map one thread block to one system and run the whole solver
in a single kernel with vectors in shared memory, for systems of ~10-2,000 rows; they beat the
vendor batched solvers [Aggarwal SC21; Nayak 2023]. I found no published batched geometric
multigrid (one V-cycle per env over thousands of small 2D grids) with numbers. Breach already
contains the pattern for one grid: `mg_fused_tail` runs every level of <= 1,024 cells as one
sub-V-cycle in one block, bit-identical by construction, removing ~238 of ~304 solve launches per
tick (`cuda_mg_solve.cu`), at `mg_cycles = 8`, `mg_nu1 = mg_nu2 = 2`, `mg_coarsest_sweeps = 32`.

**What applies to Breach.** Keep the fine levels as wide `(N, h, w)` kernels and launch the fused
coarse tail with one block per env. Use a per-env hierarchy on the `(N, h, w)` layout, not one
tall grid: a shared hierarchy couples copies at coarse levels unless each env is padded to a
multiple of 2^L, which breaks batch invariance. The fixed 8 V-cycles keep envs in lockstep.

## 7. 64-bit integer throughput

The CUDA guide's throughput table gives 32-bit integer multiply / multiply-add at 64 results per
clock per SM on cc 7.x, 8.0, 8.6, 8.9 and 9.0, and has no 64-bit integer row: int64 is built from
32-bit instructions (IMAD.WIDE gives 32 x 32 -> 64 in one) [CUDA guide 12.4]. Forum estimates:
64 x 64 -> 64 ~ 4 multiplies, `__umul64hi` ~ twice that; ptxas emits the same IMAD.WIDE sequence
from plain C++ as from hand-written PTX (forum, secondary). FP64 is 32/clk/SM on the A100 and 2 on
the RTX 3070, irrelevant here. Device totals: 3070 46 SMs, 448 GB/s, 4 MB L2; A100 108 SMs,
1,555-2,039 GB/s, 40 MB L2 (secondary sources).

**What applies to Breach.** The integer pipe per SM per clock is the same on the 3070 and the
A100; expect ~2x from an A100 on integer-bound kernels and ~3.5-4.5x on bandwidth-bound ones.
The 128-bit staging (`__mul64hi`) is the costly primitive; narrowing a product to 32 x 32 -> 64
where ranges allow is an arithmetic change that lands in the CPU reference first, and only after
Nsight Compute says the kernel is compute-bound.

## 8. GPU pathfinding for many agents

Shared per-goal fields are the standard answer: Continuum Crowds computes one potential per group
with a common goal, "only once, deriving optimal paths for all group members simultaneously"
(fast marching; >10,000 people at several fps in 2006) [Treuille 2006]; Supreme Commander 2's
flow-field tiles build an integration field per goal over sectors on the high-level path
[Emerson 2013]. On a GPU the field is a wavefront relaxation whose pass count "scales with the
grid's largest path length" [NativeFlowField]; level-synchronous GPU BFS is weak on high-diameter
graphs [Merrill 2012], and a grid is one. The parallel fast sweeping method orders cells by
diagonals, the same structure as the radiation sweep [Detrixhe 2013]. An integer shortest-path
field is the unique fixed point of a min-relaxation, so any parallel order reaches the same field.
I found no controlled, published per-field timing on 2K-20K-cell grids.

**What applies to Breach.** For the horde, one multi-source integer distance field per env from the
living marines; for bots, a field per goal; each field computed in ONE block per (env, field) with
the frontier loop inside the block (int16 distances for 17,600 cells are 35 KB of shared memory),
never one launch per BFS level. Tie-break the descent by a fixed neighbour order; refresh every K
ticks. A field descent changes paths against today's A* (the parity test pins A*), so it is a
behaviour change with a golden and a HUMAN-TEST.

## 9. Deterministic parallel game logic

Deterministic lockstep (Age of Empires): same commands everywhere, commands executed two turns
later, checksums of every system to catch divergence, a synchronised RNG, sim-external randomness
isolated [Terrano 2001]. On GPUs, agent-based conflict resolution where agents "indicate their
interest" through atomics outperforms the alternatives [Yang 2018]; it is deterministic only if
the winner is chosen by an order-free reduction on a deterministic key. Madrona and WarpDrive say
nothing about determinism or update order; Isaac Lab's is per hardware and version, floats.

**What applies to Breach.** Breach's two-phase `SimultaneousEffects` (#169) is the device shape:
decide in parallel against start-of-tick state, resolve conflicts by integer reductions
(`atomicMin` on (key, slot), integer `atomicAdd`), then apply. Its serial walk-order apply needs a
device form (a per-env loop, or a sort by (env, walk index) and a segmented scan), and every draw a
counter-based RNG keyed by (seed, env, tick, slot, purpose). No surveyed system gates batch
invariance; Breach has to build that gate itself.

## 10. The CPU-farm yardstick

| | single thread | full machine | scaling |
|---|---|---|---|
| EnvPool Atari, DGX-A100 (2 x EPYC 7742, 128C/256T) [EnvPool T1] | 4,640 FPS | 1.07M FPS | 230x, ~1.8x per physical core |
| EnvPool MuJoCo, same | 11,569 | 3.13M | 271x |
| PufferLib Neural MMO, i9-14900K 24 cores [Puffer T1-2] | 2,400 SPS | 13-19K | 5.4-7.9x |
| PufferLib NetHack, same | 29K | 96K | 3.3x |
| Sample Factory APPO training, 36 cores, share of pure-sim rate [SF T1] | | 45-85% | |

One EPYC 7742 thread runs Atari at 0.59x a Ryzen 5950X thread. PufferLib reports Gymnasium's and
SB3's multiprocessing degrading above ~1,000 synchronisations per second per core. Berzelius: 8 x A100 + 2 x EPYC 7742
(128 cores) per node, i.e. 16 cores per A100 [NSC].

**What applies to Breach.** Per A100 the CPU farm yardstick is ~2-3K env-ticks/s for the small
ship (16 EPYC cores at 0.6-0.8x a 3700X core, SMT 1.0-1.3x: assumptions), less for medium; the GPU
route projects 5-14K (benchmark arithmetic x the A100 factor), 1.5-7x, before the policy shares the
GPU. Measure the farm rather than assume it: an 8-process run here and one Berzelius node. At ~200
env-ticks/s per core Breach is far below that degradation point, so a plain multiprocess farm
should scale, and the 16 cores can run CPU envs beside the GPU ones.

## 11. Zero-copy CuPy -> torch

DLPack and the CUDA Array Interface give zero-copy exchange; lifetime is the user's job
[CuPy interop]. DLPack: the consumer passes its stream to `__dlpack__`, and the producer waits or
synchronises [DLPack spec]. CAI v3: `stream` names where the producer may still have work;
`None` means no synchronisation is needed, `0` is disallowed [CAI spec]. CuPy 14 recommends
`cupy.cuda.Stream.from_external()` (the old `ExternalStream` pattern is deprecated); pool and
stream sharing with torch via pytorch-pfn-extras. PyTorch's caching allocator can reuse freed
memory under another stream's pending work unless synchronised or `record_stream` is used
[PyTorch CUDA notes].

**What applies to Breach.** Export observations through DLPack from persistent, preallocated
device buffers, with the sim and the policy on one shared stream or an explicit event wait;
never a raw pointer without the stream field. Run the sim on torch's stream through
`Stream.from_external()`, and share or cap the two memory pools on the 8 GB card.

## Open questions the design must still decide

- Unit store: kind tables with an env-id column (Madrona) or fixed `(N, max_units)` with a valid
  mask (simpler, batch-invariant, wastes slots); which columns are synced and digested.
- The device form of `resolve()`'s walk-order apply, and whether any effect type truly needs
  serial order.
- Order of the boundary move: whether gameplay stays on the host in a first batched milestone
  (bounded by the ~400 ns/cell host floor) or moves with the brackets.
- Sweep block shape: one block per (env, ordinate) or per (env, ordinate group); shared-memory
  budget; how the shear transport's upstream read maps onto the block-local loop for x-major and
  y-major ordinates.
- EOS pre-stage on the device: a WHILE graph node over max `n_sub` with masked envs, or per-env
  loops; how the per-env reductions are done order-free.
- Pathfinding: refresh cadence K; one horde field vs per-target fields; whether bots move to
  fields in v1 or keep A* on the host.
- Graph capture granularity (whole tick or per stage) and how resets keep the captured shapes
  static.
- Policy placement on Berzelius: sim and policy on one A100 (Madrona: environment ~54% of a
  training run) or split across a node's GPUs; Linux build with CUDA >= 12.4 for conditional
  nodes.
- The measured CPU-farm yardstick (8 cores here, one Berzelius node), replacing the assumptions
  in §10.

## References

- [Madrona] Shacklett et al., ACM TOG 42(4) 2023. https://madrona-engine.github.io/shacklett_siggraph23.pdf ; https://madrona-engine.github.io/ ; README https://github.com/shacklettbp/madrona ; escape room https://github.com/shacklettbp/madrona_escape_room (src/sim.cpp)
- [GPUDrive] Kazemkhani et al. 2024. https://arxiv.org/abs/2408.01584
- [CuLE] Dalton, Frosio, Garland, NeurIPS 2020. https://arxiv.org/abs/1907.08467 ; https://github.com/NVlabs/cule
- [WarpDrive] Lan et al., JMLR 2022. https://arxiv.org/html/2108.13976v3
- [Isaac Gym docs] https://lequn-f.github.io/isaacgym/programming/tensors.html (mirror of the Preview docs)
- [direct_rl_env.py] https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/direct_rl_env.py
- [Isaac Lab] envs API https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html ; events https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/envs/mdp/events.html ; reproducibility https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html
- [Gray 2019] https://developer.nvidia.com/blog/cuda-graphs/
- [NVIDIA 2024] https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements
- [Conditional nodes] https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/
- [DL graph guide] https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/quantitative-benefits.html
- [GROMACS 2023] https://developer.nvidia.com/blog/a-guide-to-cuda-graphs-in-gromacs-2023/
- [Zhang 2020] Zhang, Wahib, Zhang, Matsuoka, IPDPS 2020. https://arxiv.org/abs/2004.05371
- [PERKS] Zhang et al., ICS 2023. https://arxiv.org/abs/2204.02064
- [Joubert 2012] https://www.olcf.ornl.gov/wp-content/uploads/2012/01/TitanWorkshop2012_Day3_Joubert.pdf
- [Kripke source] https://raw.githubusercontent.com/LLNL/Kripke/develop/src/Kripke/Arch/SweepSubdomains.h ; https://github.com/LLNL/Kripke
- [Deakin 2019] https://uob-hpc.github.io/transport2019/slides/many_core_sweeps.pdf ; ISC 2016 https://research-information.bris.ac.uk/en/publications/many-core-acceleration-of-a-discrete-ordinates-transport-mini-app/
- [Gong 2011] https://koreascience.kr/article/JAKO201113753748127.view (abstract only)
- [Minisweep] https://github.com/wdj/minisweep
- [Aggarwal SC21] https://impact.ornl.gov/en/publications/batched-sparse-iterative-solvers-for-computational-chemistry-simu/
- [Nayak 2023] https://arxiv.org/abs/2308.08417
- [CUDA guide 12.4] https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#arithmetic-instructions
- [64-bit multiply, forum] https://forums.developer.nvidia.com/t/long-integer-multiplication-mul-wide-u64-and-mul-wide-u128/51520 ; https://forums.developer.nvidia.com/t/squeezing-the-last-17-5-out-of-a-compute-bound-256-bit-modular-arithmetic-kernel-sm-89-82-5-sm-throughput/367788
- [GPU specs] https://www.tomshardware.com/features/nvidia-ampere-architecture-deep-dive ; https://en.wikipedia.org/wiki/Ampere_(microarchitecture) ; https://geeks3d.com/?p=10095
- [Treuille 2006] https://grail.cs.washington.edu/projects/continuum-crowds
- [Emerson 2013] https://www.taylorfrancis.com/books/9780429100277/chapters/10.1201/b16725-29
- [NativeFlowField] https://discussions.unity.com/t/nativeflowfield-gpu-powered-flow-field-generation/1676407
- [Merrill 2012] https://research.nvidia.com/publication/2012-02_scalable-gpu-graph-traversal
- [Detrixhe 2013] https://me.ucsb.edu/sites/me.ucsb.edu/files/publications/Gibou/FG_Pub49.pdf
- [Terrano 2001] https://www.gamedeveloper.com/programming/1500-archers-on-a-28-8-network-programming-in-age-of-empires-and-beyond
- [Yang 2018] https://portal.fis.tum.de/en/publications/evaluation-of-conflict-resolution-methods-for-agent-based-simulat/
- [EnvPool] https://arxiv.org/abs/2206.10558
- [Puffer] https://arxiv.org/abs/2406.12905
- [SF] Sample Factory, https://arxiv.org/abs/2006.11751
- [NSC] https://nsc.liu.se/systems/berzelius ; https://supr.naiss.se/resource/berzelius-compute/
- [CuPy interop] https://docs.cupy.dev/en/stable/user_guide/interoperability.html
- [CAI spec] https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
- [DLPack spec] https://dmlc.github.io/dlpack/latest/python_spec.html
- [PyTorch CUDA notes] https://docs.pytorch.org/docs/2.15/notes/cuda.html

**Not verified in this pass:** the Gong et al. speedups (abstracts only); Deakin's ISC 2016 full
text; Minisweep's GPU scheme; cuSPARSE batched-solver APIs; the 64-bit multiply instruction counts
(forum level); any published per-field GPU pathfinding timing on 2K-20K-cell grids (none found);
the Isaac Gym docs are read from a mirror; the EPYC per-core and SMT factors in §10 are
assumptions; CuPy's `CUPY_CUDA_ARRAY_INTERFACE_SYNC` semantics are not documented on the page read.
