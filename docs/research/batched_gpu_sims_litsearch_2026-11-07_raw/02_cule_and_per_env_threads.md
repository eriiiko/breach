# Q2. CuLE: one emulator per thread for branchy serial game logic (+ WarpDrive's mapping)

Raw research note, 2026-11-07, for #29. Read in full: the CuLE arXiv paper (text extracted).
WarpDrive read through its arXiv HTML. Madrona's ECS-GPU baseline is in note 01.

## Sources
- Dalton, Frosio, Garland, "Accelerating Reinforcement Learning through GPU Atari Emulation",
  NeurIPS 2020. https://arxiv.org/abs/1907.08467 ; code https://github.com/NVlabs/cule
- Lan et al., "WarpDrive: Extremely Fast End-to-End Deep Multi-Agent RL on a GPU", JMLR 2022.
  https://arxiv.org/html/2108.13976v3
- Madrona paper (note 01), Table 1-2.

## CuLE mechanism
- One CUDA thread emulates one Atari console (the 6502 CPU + the TIA video chip); state in
  global memory, ROM in constant memory.
- Split into TWO kernels: the CPU kernel runs the game code and writes the TIA instructions to a
  buffer; the TIA kernel replays that buffer and renders the 160 x 210 frame. Reasons: different
  register needs and different divergence of the two tasks; and the TIA kernel is skipped on
  frames that are not observed (frame-skip 4 needs only 2 of 4 frames).
- The one-thread-per-env mapping is "not the most computationally efficient", chosen for ease of
  debugging and so that the same code runs on the CPU (CuLE_CPU).
- Resets: a CPU emulator resets with up to 94 frames (64 start-up + up to 30 random no-ops),
  which would diverge thousands of SIMD emulators; CuLE caches 30 random initial states at start
  and a terminal env copies one in.

## CuLE measurements (System I: 12-core i7-5930K + Titan V)
- Below ~128 envs the CPU is faster (low GPU occupancy). At 4,096 envs, emulation only: CuLE
  median 64K FPS vs OpenAI Gym's CPU peak 18K FPS (3.56x); inference path: 56K vs 18K (3.11x).
- Across games 41K (UpNDown) to 155K (Riverraid) FPS, a 3.78x spread; the same spread on the
  CPU is 1.93x. GPU throughput is more sensitive to how branchy the code is.
- WARP DIVERGENCE, measured: with 512 envs starting from one state, FPS is highest at first and
  falls as random actions decorrelate the envs, settling up to 30% lower in the worst game.
  Complete divergence would give 1/32 ~ 3% of peak; real games stay far above that because the
  envs share most of their code path.
- FPS per env is LOWER on the GPU than on a CPU core; flat up to 512 envs, then falling. That
  lowered A2C's updates per second, so they needed a different batching strategy (A2C + V-trace,
  update every SPU steps) to turn throughput into faster convergence.
- Memory: they kept training below 5K envs (frames + model + rollout storage).
- Headline: up to 155M frames/hour on one GPU; training 26-68K FPS on one GPU, 187K on four.

## WarpDrive mapping (for contrast)
- "each block is designed to contain an environment replica and each thread simulates an
  agent"; a reset kernel resets individual env blocks from copies of the initial arrays made at
  the first reset; one host-to-device copy, then everything in place; PyTorch reads the same
  buffers.
- A100: discrete Tag 9.8M env steps/s at 2,000 envs x 5 agents; 2.9M at 2,000 envs x 1,000
  agents; end-to-end training 1.3M / 0.58M iterations/s. CPU baseline a 16-CPU GCP N1 node.
- No discussion of determinism, update order or races (e.g. two taggers on one runner); a
  CPU-vs-GPU consistency checker (`EnvironmentCPUvsGPU`) compares obs/actions/rewards/dones step
  by step.

## Madrona's verdict on one-world-per-thread
- Madrona's ECS-GPU baseline (each world and its ECS in one thread) was 42x slower than the
  batched ECS on HideSeek and 3x slower than the 32-thread CPU, at 1.9% of compute and bandwidth
  (warp divergence + incoherent access + per-world allocation limiting it to ~8K worlds).

## Reading for Breach (also in the synthesis)
- One thread per env works for SMALL, mostly shared code (Atari: 128 B of RAM). Breach's
  gameplay is a few dozen units per env and each slot is a separate pass over units, so the
  coherent mapping is Madrona's: a unit table across envs, one thread per unit row per slot,
  per-env scalars looked up by env id. A per-env thread pays for the longest unit loop in its
  warp, and Breach's envs are not lockstep-similar (different fights, different unit counts).
- CuLE's two-kernel split is the precedent for keeping stages with different register and
  divergence profiles in different kernels.
- CuLE's reset cache is the precedent for a template bank of pre-built level states copied in on
  reset (where-we-stand §7 item 5).
