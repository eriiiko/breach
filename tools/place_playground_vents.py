"""#48 patch-2: place the proposed vent loops into playground via level_lib
(the ONE writer — tools/light_entity_port.py's managed-block pattern), then
prove the circulation is alive with a headless real-physics run."""
import sys
from pathlib import Path

ROOT = Path(r"c:\Users\steen\projects\breach.worktrees\velocity-clamp")
for p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    sys.path.insert(0, str(p))

import numpy as np
import level_lib
from level_loader import load as load_level, EntityInstance
from simulation import Simulation
from simulation.entities import REGISTRY

lvl = load_level("playground", levels_dir=str(ROOT / "levels"))
sim = Simulation(lvl, seed=1, breach_physics=None, enable_recorder=False)
solid = np.asarray(sim.gmap.obstacles, dtype=bool)


def open_tile(x, y):
    if not solid[y, x]:
        return x, y
    for r in range(1, 3):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if not solid[y + dy, x + dx]:
                    return x + dx, y + dy
    raise SystemExit(f"no open tile near ({x},{y})")


def inst(cls_name, eid, ordinal, authored):
    cls = REGISTRY[cls_name]
    fields = {f.name: f.default for f in cls.FIELDS}
    fields.update(authored)
    return EntityInstance(id=eid, class_name=cls_name, ordinal=ordinal,
                          tags=(), fields=fields,
                          authored_keys=tuple(authored))


placements = [
    ("vent_loop_supply",   68, 39, "supply", "duct_loop",   0.65),
    ("vent_loop_return",   68, 23, "return", "duct_loop",   0.65),
    ("vent_sealed_supply", 85, 51, "supply", "duct_sealed", 0.6),
    ("vent_sealed_return", 93, 63, "return", "duct_sealed", 0.6),
]
existing_ids = {e.id for e in lvl.entities}
for eid, *_ in placements:
    assert eid not in existing_ids, f"id collision: {eid}"

base = len(lvl.entities)
new = [
    inst("duct", "duct_loop", base, {"filter": "hepa_basic"}),
    inst("duct", "duct_sealed", base + 1, {"filter": "derelict"}),
]
for i, (eid, x, y, role, duct, q) in enumerate(placements):
    x2, y2 = open_tile(x, y)
    if (x2, y2) != (x, y):
        print(f"nudged {eid}: ({x},{y}) -> ({x2},{y2})")
    new.append(inst("vent", eid, base + 2 + i,
                    {"x": x2, "y": y2, "role": role, "duct": duct,
                     "q_circ": q}))

merged = list(lvl.entities) + new
toml_path = ROOT / "levels" / "playground" / "level.toml"
bak = level_lib.write_managed_blocks(
    toml_path,
    {"entity": lambda nl: level_lib.format_entity_lines(merged, nl)})
print(f"wrote {toml_path.name}; backup: {bak}")

# ---- verify: reload, build on REAL physics, 10 s, wind near each vent ----
lvl2 = load_level("playground", levels_dir=str(ROOT / "levels"))
vd = [e for e in lvl2.entities if e.class_name in ("vent", "duct")]
print(f"entities: {len(lvl2.entities)} total, {len(vd)} vent/duct")

import breach_physics as bp
sim2 = Simulation(lvl2, seed=1, breach_physics=bp, enable_recorder=False)
for _ in range(240):                       # 10 s at 24 tps
    sim2.set_paused(False)
    sim2.step()
wx, wy = sim2.gmap.wind_x, sim2.gmap.wind_y
for (eid, x, y, role, duct, q) in placements:
    box = np.s_[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3]
    m = np.sqrt((wx[box] / 65536.0) ** 2 + (wy[box] / 65536.0) ** 2).max()
    print(f"{eid:>20} ({role:6}): max |u| within 2 tiles = {m:.4f} m/s")
print("headless 10 s: OK")
