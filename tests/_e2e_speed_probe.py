"""G1p-c E2E probe (a harness, not a test): ``tools/e2e_drive.py`` -- the
real ``main.py`` in a hidden window -- with a recorder on
``Simulation.step`` that logs the possessed marine at every tick: its
position, facing and live move intent. Run as
``python tests/_e2e_speed_probe.py <e2e_drive args>``; prints one
``SPEEDPROBE <json>`` line: ``{"tile_size_m", "tps", "table", "log"}`` with
``log`` a list of ``[tick, x0, y0, x1, y1, facing, dx_q, dy_q, throttle_q]``
(``x0, y0`` before the step, ``x1, y1`` after; ``dx_q`` None when the unit
held no move)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tools", ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT):
    sys.path.insert(0, str(_p))

import e2e_drive  # noqa: E402
import breach_physics  # noqa: E402,F401
from simulation import simulation as S  # noqa: E402

LOG: list = []
META: dict = {}
_orig = S.Simulation.step


def _step(self, *a, **k):
    u = next((u for u in self.units
              if u.team == self.player_team and u.alive), None)
    if u is None:
        return _orig(self, *a, **k)
    t, x0, y0 = self.tick, u.x, u.y
    md = getattr(u, "live_move_dir", None)
    r = _orig(self, *a, **k)
    if not META:
        tab = self.speed_table
        META.update(tile_size_m=self.gmap.tile_size_m,
                    tps=tab.ticks_per_second,
                    table=dict(walk_cost_q=tab.walk_cost_q,
                               full_cost_q=tab.full_cost_q,
                               side_q=tab.side_q, back_q=tab.back_q))
    LOG.append([t, x0, y0, u.x, u.y, u.facing,
                None if md is None else md.dx_q,
                None if md is None else md.dy_q,
                None if md is None else md.throttle_q])
    return r


S.Simulation.step = _step
rc = e2e_drive.main(sys.argv[1:])
print("SPEEDPROBE " + json.dumps(dict(META, log=LOG)))
sys.exit(rc)
