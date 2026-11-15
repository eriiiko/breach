---
paths:
  - "src/simulation/horde.py"
  - "src/simulation/noise.py"
  - "src/simulation/ai_zombie.py"
  - "src/control_zombies.py"
  - "cpp/src/horde_field.cpp"
  - "cpp/src/horde_field.h"
  - "pathfinding.py"
---

# The horde, noise and A*

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## Horde field
- The horde paths ONLY by the distance field; a new zombie behaviour is a MASK, a GOAL SET or a RULE ON THE DESCENT, never a per-zombie search, and no A* call may return to the zombie law.
- Every field and the mask drop at a tick `% field_refresh_ticks == 0` and whenever `GameMap._tile_serial` moved; flooded the first tick a step needs it; never synced, never digested.
- A zombie steps to its lowest 4-neighbour STRICTLY below its own cell that it can enter now, ties by the cell's parity order (`DESCENT_ORDER`), or waits.
- The flood is C++ wherever an engine is bound; `flood_reference` is the same law for a bare map: an arithmetic change lands in both.
- Behaviours are `[zombie]` switches; all off == the stage-1 law, a permanent gate. WANDER uses a Philox block as a seedless HASH of (id, tick) (`Salt.WANDER`), never a stream. A door with a span tile bordering vacuum, the ambient ring or the map edge is NEVER bashed.
- Per-zombie state is synced, fixed-width, hashed only while set, cleared at death and conversion (`horde.clear`).

## Noise
- Hearing reads noises, never the sources, by squared distance alone (walls ignored, no sound propagation). A new loud thing calls `noise.emit`, never a hearing check of its own.
- `loudness` is METRES of hearing radius, quantized once per map to whole tiles; REQUIRED on a weapon row that fires, optional (0 = silent) on a payload.
- The log is drained once per tick at the zombie slot and never digested; it crosses into the next tick only for a slot-9e detonation or a slot-5 bash (ACCEPTED GAP for a save / load in between).

## A*
- The horde does NOT path by it. A caller that holds a footprint grid passes it as `passable=` instead of a callback; either way the path is the pre-#143 one bit for bit (`tests/test_pathfinding_parity.py` against the frozen body in `tests/_reference_scans.py`): a faster search, never a different one.
