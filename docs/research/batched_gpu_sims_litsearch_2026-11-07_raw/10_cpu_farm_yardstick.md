# Q10. The CPU-farm yardstick, cited

Raw research note, 2026-11-07, for #29. Read in full: EnvPool, PufferLib 2.0 and Sample Factory
papers (text extracted). Berzelius spec from NSC.

## Sources
- Weng et al., "EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine",
  NeurIPS 2022 D&B: https://arxiv.org/abs/2206.10558
- Suarez, "PufferLib 2.0: Reinforcement Learning at 1M steps/s", RLC 2025:
  https://arxiv.org/abs/2406.12905
- Petrenko et al., "Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with
  Asynchronous Reinforcement Learning", ICML 2020: https://arxiv.org/abs/2006.11751
- Madrona ECS-CPU baseline (note 01).
- Berzelius: https://nsc.liu.se/systems/berzelius ; https://supr.naiss.se/resource/berzelius-compute/

## EnvPool (Table 1; Atari FPS counts frames at frame-skip 4; MuJoCo counts physics sub-steps, 5 per step)
| machine | for-loop, 1 thread | Subprocess | Sample Factory | EnvPool async | EnvPool numa+async |
|---|---|---|---|---|---|
| laptop, i7-8750H, "12 CPU cores" (6C/12T) | Atari 4,893 / MuJoCo 12,861 | 15,863 / 36,586 | 28,216 / 62,510 | 49,439 / 105,126 | - |
| workstation, Ryzen 9 5950X, "32 CPU cores" (16C/32T) | 7,914 / 20,298 | 47,699 / 105,432 | 138,847 / 309,264 | 200,428 / 582,446 | - |
| DGX-A100, 2x EPYC 7742, "256 CPU cores" (128C/256T, 8 NUMA nodes) | 4,640 / 11,569 | 71,943 / 163,656 | 707,494 / 1,573,262 | 891,286 / 2,363,864 | 1,069,922 / 3,134,287 |
- The "CPU cores" EnvPool counts are hardware threads (the EPYC 7742 is 64 cores / 128 threads).
- DGX-A100 scaling: 230x the single-thread for-loop on Atari, 271x on MuJoCo, over 128 physical
  cores: ~1.8-2.1x per physical core (SMT included) for these small-working-set envs. Plain
  Python `Subprocess` reaches only 15.5x (Atari).
- One EPYC 7742 thread (4,640 Atari FPS) is ~0.59x one 5950X thread (7,914): the DGX's cores
  are slow per thread.

## PufferLib 2.0 (desktop i9-14900K, 24 cores = 8 P + 16 E; laptop i7-10750H, 6 cores)
| env | 1 core SPS | desktop vectorised | desktop with Puffer Pool | scaling on 24 cores |
|---|---|---|---|---|
| Neural MMO (complex multi-agent) | 2,400 | 13k | 19k | 5.4x / 7.9x |
| NetHack | 29k | 96k | - | 3.3x |
| MiniHack | 11k | 55k | - | 5x |
| Pokemon Red | 700 | 5k | 7.2k | 7x / 10x |
| Crafter | 320 | 360 | 2.8k | 1.1x / 8.8x |
| Procgen Bigfish | 25k | 33k | 150k | 1.3x / 6x |
- "The Gymnasium and SB3 multiprocessing implementations experience significant scaling
  degradation above 1000 synchronizations per second per core"; PufferLib runs several envs per
  process to avoid it. Breach at ~200 env-ticks/s per core is far below that threshold.
- Complex envs scale sub-linearly on a hybrid-core desktop (E-cores are slower).

## Sample Factory (System #2: 36-core server CPU + one RTX 2080 Ti)
| | Atari | VizDoom | DMLab |
|---|---|---|---|
| pure simulation, FPS | 181,740 | 322,907 | 49,679 |
| APPO training, FPS (share of pure) | 135,893 (74.8%) | 146,551 (45.4%) | 42,149 (84.8%) |
| SeedRL V-trace | 39,726 (21.9%) | 34,428 (10.7%) | 34,773 (70.0%) |
- Even the best single-machine trainer keeps 45-85% of the CPUs' pure simulation rate; the rest
  goes to sampling, inference traffic and learning on the same box.

## Madrona's CPU baseline
- ECS-CPU (one ECS per world per thread, EnvPool-like, i9-13900K 32 threads): HideSeek 1.6e5
  steps/s vs the GPU's 1.9e6 (11x); the GPU overtakes it at ~500 worlds.

## Berzelius
- 94 DGX-A100 nodes (60 thin: A100 40 GB, 1 TB RAM; 34 fat: A100 80 GB, 2 TB RAM), each 8 x A100
  + 2 x EPYC 7742 = 128 cores (256 threads). So 16 cores / 32 threads per A100.

## Yardstick for Breach (arithmetic with stated assumptions)
- Measured (benchmark, Ryzen 7 3700X, one core): small ship physics-only 196 env-ticks/s; the
  headless 4 v 4 + 4 zombies match: small 5.9x real time = 142 env-ticks/s, medium 2.0x = 48,
  large 0.7x = 17 (24 ticks per second).
- Assumptions: an EPYC 7742 core runs Breach at 0.6-0.8x a 3700X core (EnvPool's thread ratio
  vs a 5950X; Zen 2 at 3.4 GHz boost vs 4.4 GHz); SMT adds 1.0-1.3x for Breach (unmeasured;
  EnvPool's ~1.8-2x per core is for cache-resident Atari); near-linear process scaling at ~200
  syncs/s per core (the Gymnasium / SB3 degradation PufferLib reports starts at 1,000/s).
- 16 cores per A100: small physics-only ~1.9-3.3K env-ticks/s; small match ~1.4-2.4K; medium
  match ~0.5-0.8K. Training would keep 45-85% of that (Sample Factory) unless sampling is free.
- GPU route (benchmark arithmetic, host stages gone, RTX 3070): small 2.4-3.4K, medium 0.9-1.2K
  env-ticks/s. On an A100 (note 07: ~2x integer, ~3.5-4.5x bandwidth): small ~5-14K, medium
  ~2-5K, before the policy shares the GPU (Madrona: environment = 54% of an end-to-end
  training run on HideSeek).
- So per A100, the GPU route must beat roughly 2-3K env-ticks/s of CPU farm on the small ship;
  the projected range is 1.5-7x (5-14K against 1.9-3.3K), and the CPU cores remain available
  alongside it. Both ends rest on unmeasured assumptions (EPYC per-core speed, SMT, the A100
  factor, the stages with no resident core yet).
