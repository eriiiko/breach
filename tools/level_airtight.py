"""Airtight check for a Breach level.

Flood-fills the interior air through every non-wall cell and flags any cell
that reaches a vacuum tile or the grid edge -- i.e. a hole the atmosphere
vents through. Uses the real GameMap decoding (CSV codes -> materials ->
solid/vacuum masks), so it can't disagree with what the sim actually does.

Usage:
    python tools/level_airtight.py [level_name]      # default: unhcr_vessel
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import deque

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    sys.path.insert(0, str(_p))

import numpy as np
from level_loader import load as load_level
from simulation.gamemap import GameMap


def check(level_name: str) -> bool:
    g = GameMap(load_level(level_name))
    solid = np.asarray(g.solid, dtype=bool)
    is_vac = np.asarray(g.is_vacuum, dtype=bool)
    h, w = solid.shape
    interior = (~solid) & (~is_vac)          # air / floor the crew breathes

    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    for y, x in np.argwhere(interior):
        visited[y, x] = True
        q.append((int(y), int(x)))

    vac_leak: set = set()
    edge_leak: set = set()
    while q:
        y, x = q.popleft()
        if is_vac[y, x]:
            vac_leak.add((y, x))
        if y == 0 or y == h - 1 or x == 0 or x == w - 1:
            edge_leak.add((y, x))           # interior air sitting on the grid boundary
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and not solid[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    print(f"Level '{level_name}': grid {h}x{w}, interior reaches {int(visited.sum())} cells")
    print(f"  vacuum-connected leaks : {len(vac_leak)}")
    print(f"  grid-edge leaks        : {len(edge_leak)}")
    for leaks, kind in ((vac_leak, "vacuum"), (edge_leak, "edge")):
        if not leaks:
            continue
        by_row: dict = {}
        for y, x in leaks:
            by_row.setdefault(y, []).append(x)
        for r in sorted(by_row):
            cs = sorted(by_row[r])
            print(f"    {kind} leak  row {r}: cols {cs[0]}-{cs[-1]} ({len(cs)} cells)")
    airtight = not vac_leak and not edge_leak
    print("  AIRTIGHT" if airtight else "  *** LEAKING ***")
    return airtight


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "unhcr_vessel"
    sys.exit(0 if check(name) else 1)
