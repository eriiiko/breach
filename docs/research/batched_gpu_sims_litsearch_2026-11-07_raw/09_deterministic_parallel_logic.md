# Q9. Deterministic parallel game logic: simultaneous actions without order dependence

Raw research note, 2026-11-07, for #29. Briefer pass.

## Sources
- Terrano & Bettner, "1500 Archers on a 28.8: Network Programming in Age of Empires and
  Beyond", Gamasutra 2001:
  https://www.gamedeveloper.com/programming/1500-archers-on-a-28-8-network-programming-in-age-of-empires-and-beyond
- Yang, Andelfinger, Cai, Knoll, "Evaluation of Conflict Resolution Methods for Agent-Based
  Simulations on the GPU", SIGSIM-PADS 2018, doi 10.1145/3200921.3200940 (abstract):
  https://portal.fis.tum.de/en/publications/evaluation-of-conflict-resolution-methods-for-agent-based-simulat/
- Richmond et al., "FLAME GPU 2", 2023 (abstract via search): https://eprints.whiterose.ac.uk/199416/
- Madrona paper + escape-room source (note 01); WarpDrive (note 02); Isaac Lab reproducibility
  (note 03).
- Breach: `docs/cuda_migration_plan.md` (integer-atomic contract), CLAUDE.md "Simultaneous
  attacks" row (#169).

## Findings
- Deterministic lockstep (Age of Empires): every machine runs the same simulation from the same
  command stream; commands issued in turn n execute in turn n + 2; turns ~200 ms; checksums of
  "the world, the objects, the pathfinding, targeting and every other system" detect a
  divergence and stop the game; the RNG is seeded and synchronised; randomness outside the
  simulation (sound) saves and restores the generator. Lesson: "The code path taken on all
  machines must match"; divergences surface minutes later and needed 50 MB traces to find.
  (This is serial determinism across machines, not parallel order-independence; the analogue
  for Breach is the field digest + replay checkpoints.)
- GPU agent-based conflict resolution (Yang et al. 2018): methods where agents "actively
  indicate their interest in a resource" with atomics generally outperform the alternatives
  despite the atomic cost. The abstract makes no determinism claim; interest-by-atomics is
  deterministic only if the winner is decided by an order-free reduction on a deterministic key
  (e.g. integer `atomicMin` on (priority, id)), not by who wrote first.
- FLAME GPU 2 resolves competing moves (Sugarscape) with a recursive sub-model (abstract; the
  mechanism not read).
- Madrona, WarpDrive: neither paper discusses determinism or update order. Madrona's per-world
  RNG is a counter-based split of (seed, episode, world id), independent of thread scheduling;
  its entity creation order across threads is not discussed (the radix sort by EnvID is not a
  stable order guarantee within a world, as far as the paper says).
- Isaac Lab / PhysX: determinism only for same hardware, same version, same set-up; float GPU
  scheduling changes bits when parameters change mid-run.

## The standard recipes (from the above + Breach's own contract)
1. Two-phase decide / apply: every actor decides against the tick's START state and records its
   effect; effects apply after a barrier. Breach has this for attacks since #169
   (`SimultaneousEffects`), applied in walk order.
2. Order-free reductions: integer add / min / max (and atomics thereof) give the same result in
   any order; float atomics do not. Breach's CUDA contract already uses integer `atomicAdd` for
   books.
3. Conflicts (two units into one cell, two hits killing one unit): resolve by an order-free
   rule on a deterministic key: min over (key, unit slot), or a per-(env, tick, slot) hash, never
   by arrival order.
4. Sequential-in-a-fixed-order application (Breach's "walk order" in `resolve()`) is deterministic
   but serial; on the GPU it becomes a per-env loop (one thread per env) or a sort by
   (env, walk index) followed by a segmented scan, whichever the effect types allow.
5. RNG per (seed, env, tick, unit slot, purpose), counter-based (Philox-style), so draws do not
   depend on which thread runs first.
