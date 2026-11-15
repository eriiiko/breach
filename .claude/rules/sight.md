---
paths:
  - "src/simulation/vision.py"
  - "src/simulation/engagement.py"
  - "src/simulation/cover_system.py"
  - "src/simulation/line_of_fire.py"
  - "cpp/src/sight_lines.cpp"
  - "cpp/src/sight_lines.h"
---

# Sight: who sees whom, lines in bulk, lines of fire

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## Vision / engagements / cover
- What stops a line of SIGHT is a `solid` tile or `GameMap.sight_screen` (`has_los`): the gas-open tiles of a material whose row says `blocks_sight` (#134). Sight and air flow are separate columns: a sight read goes through `has_los`, never `solid` alone. `has_los(physical=True)` asks whether what travels with the air gets through. A bullet's march stops on `solid` (penetration per material is Erik's open session).
- WHO sees whom is the match's sight rule, read by `can_see` from the sim (`Simulation.sight`, never a caller's argument; #195 1a): `cone` = the onephase cone + awareness radius; `all_round` = every living non-zombie sees every living unit of another team along a clear `vision.ray_clear`, and a zombie observes nothing (its waking is `ai_zombie`'s own law).
- `src/simulation/line_of_fire.py` judges a ROUND's path, never sight: `solid_in_way` = `rays_clear(physical=True)` plus every live cover's `blocks_segment`; `mate_in_way` is the bots' `_teammate_on_line` geometry verbatim (held equal by a test).

## Sight for many tiles
- A line is `GameMap.has_los`'s own walk step for step (int64, no float), through two entries over the one walk (many rays, a window). Cover that blocks sight stays in Python, in the `vision.py` wrappers, so `rays_clear` / `window_clear` equal `ray_clear` on every map.
- A bare map with no engine takes `sight_lines_reference`, the same walk in Python: an arithmetic change lands in `has_los` first, then the reference and the C++.
- A new reader of "what can be seen from here" calls these: nothing walks its own lines; a window is the shape for a soldier's near map, never a whole-map fill from a viewpoint. It holds no state and is not cached. No device kernel: the CUDA module runs the same host TU.
