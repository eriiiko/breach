# Q8. GPU pathfinding for many agents: per-goal distance / flow fields

Raw research note, 2026-11-07, for #29. The literature has the mechanisms; I found no
controlled, published per-field timing on 2K-20K-cell grids. Numbers for Breach are arithmetic,
marked as such.

## Sources
- E. Emerson, "Crowd Pathfinding and Steering Using Flow Field Tiles", Game AI Pro (2013),
  ch. 23, pp. 307-316 (record only): https://www.taylorfrancis.com/books/9780429100277/chapters/10.1201/b16725-29 ;
  context https://howtorts.github.io/2014/01/04/basic-flow-fields.html
- Treuille, Cooper, Popovic, "Continuum Crowds", SIGGRAPH 2006 (read, text extracted):
  https://grail.cs.washington.edu/projects/continuum-crowds (PDF copy:
  https://ics-websites.science.uu.nl/docs/vakken/mcrws/papers_new/Treuille%20et%20al%20-%202006%20-%20Continuum%20crowds.pdf)
- NativeFlowField (Unity, GPU compute-shader flow fields), developer post:
  https://discussions.unity.com/t/nativeflowfield-gpu-powered-flow-field-generation/1676407 ;
  https://github.com/nathanrun1/NativeFlowField
- Detrixhe, Gibou, Min, "A parallel fast sweeping method for the Eikonal equation", JCP 237
  (2013) 46-55: https://me.ucsb.edu/sites/me.ucsb.edu/files/publications/Gibou/FG_Pub49.pdf
  (abstract via search)
- Merrill, Garland, Grimshaw, "Scalable GPU Graph Traversal", PPoPP 2012:
  https://research.nvidia.com/publication/2012-02_scalable-gpu-graph-traversal
- Fischer et al., "GPU accelerated path-planning for multi-agents in virtual environments", 2009
  (read): https://ics-websites.science.uu.nl/docs/vakken/mcrws/papers_new/Fischer%20et%20al%20-%202009%20-%20GPU-accelerated%20path%20planning%20for%20multi-agents%20in%20virtual%20environments.pdf

## Findings
- Shared per-goal fields: Continuum Crowds computes ONE potential per group with a common goal,
  "only once, deriving optimal paths for all group members simultaneously", with fast marching
  on a regular grid; >10,000 people at "several frames a second" on coarse grids, simulation
  updates at 2-5 fps (2006 CPU). Emerson's flow-field tiles (Supreme Commander 2): an
  integration (cost-to-goal) field per goal, a flow (direction) field derived from it, tiled
  into sectors so only sectors on the high-level path are built; aimed at hundreds to thousands
  of agents (record + secondary sources).
- GPU construction: a distance field on a grid is a wavefront relaxation; NativeFlowField runs it
  as compute-shader passes that propagate from the target, "the number of passes scales with
  the grid's largest path length", with ~1-2 frames of readback latency (developer's post; no
  benchmark, the author says a fair one is hard to build).
- Level-synchronous BFS on GPUs is weak on high-diameter graphs (Merrill et al.: earlier GPU BFS
  "perform poorly on graphs with non-trivial diameter"); a grid is high-diameter, so one launch
  per BFS level is the wrong shape for 100-300-level fields.
- Eikonal / Euclidean alternatives: the parallel fast sweeping method orders points by
  Cuthill-McKee levels (diagonals) and updates each level in parallel; this is the same
  wavefront structure as Breach's radiation sweep. (Jump flooding gives Euclidean distance
  transforms in log2(n) passes but ignores walls; not a pathfinder.)
- Fischer 2009 uses Laplace potential fields relaxed by Gauss-Seidel per agent group (up to 56x
  vs CPU; harmonic, not shortest path).

## Determinism and tie-breaking
- An integer BFS / Dijkstra distance on a grid is a unique fixed point of
  d(c) = min over neighbours (d(n) + cost) with d(goal) = 0; computing it by min-relaxation is
  order-independent (min is associative and commutative on integers), so any parallel schedule
  reaches the same field. Integer `atomicMin` is likewise order-free.
- Ties appear only when a unit chooses among neighbours of equal distance; a fixed neighbour
  order (or a hash of (unit slot, tick)) makes the step choice deterministic.

## Arithmetic for Breach (not literature)
- Grids: 2,400 / 7,000 / 17,600 cells (small / medium / large); worst path lengths of order the
  perimeter, ~100-300 cells.
- Shape that fits: one thread block per (env, goal field), the whole BFS inside the block with a
  frontier and __syncthreads per level (the Kripke / MG-tail pattern). The cost field and the
  distance field (int16) fit shared memory up to ~20K cells (17,600 x 2 B = 35 KB for distances;
  the 3070 has up to 100 KB of shared memory per SM). Work is each cell once plus one barrier per
  level; no launch per level.
- Zombies need a field from all living marines (multi-source BFS, one field per env); bots need
  a field per goal they path to (few per team). Refresh cadence can be every K ticks with the
  same field reused between refreshes (Continuum Crowds refreshes per frame; Emerson builds on
  demand); the K and the behaviour change are design questions.
- A field descent does not reproduce today's A* paths bit for bit (`tests/test_pathfinding_parity.py`
  pins A*), so it is a behaviour change: golden + HUMAN-TEST, not a refactor.
